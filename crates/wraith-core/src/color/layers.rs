//! Per-region coverage fields with trapping underlays (roadmap M1) built by
//! a single scatter pass (roadmap M2).
//!
//! M1 semantics, from the tested prototype:
//! * `cov`   = own coverage of the region (mixture weights whose ranks land
//!   in this region),
//! * `later` = coverage of everything painted *after* this region,
//! * `F′ = cov + later · dilate_r(cov)` — own region plus an `r`-px underlay
//!   into later layers ("trapping"). Where region *i* meets a later region
//!   *j*, *i* extends underneath *j* (no hairline seam even at 8× zoom);
//!   where *i* meets an earlier region, the edge is the exact 0.5 iso-line of
//!   *i*'s own coverage.
//!
//! M2 changes versus the prototype: one scatter pass pushes every pixel's
//! contributions into per-rank accumulators (own-pixel lists + tile-bucketed
//! "later" lists), so layer work becomes proportional to the layer's area
//! plus its trap band instead of K full-image passes; bounding boxes come
//! free from the same pass; the `nxt` chain table and the field buffers are
//! reused across layers through a thread-local scratch pool.

use std::cell::RefCell;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::color::contour::{self, TileSpan};
use crate::color::fit::{self, Seg};
use crate::color::gradients::{RegionKind, Regions};
use crate::color::model::{Mixture, dominant_rank};
use crate::color::palette::{Uniques, V3, oklab_to_rgb};
use crate::color::primitives;
use crate::color::refine;
use crate::color::{ColorLayer, Paint, PathSeg, Shape};
use crate::settings::ColorSettings;
use crate::stats::{Stage, Stopwatch};

/// Tile edge length for the "later" scatter buckets.
pub(crate) const SCATTER_TILE: usize = contour::TILE;

/// Scatter accumulator for one paint rank.
pub(crate) struct RankAcc {
    /// `(pixel index, weight)` contributions to this rank's own coverage.
    pub cov: Vec<(u32, f32)>,
    /// Tile-bucketed `(pixel index, weight)` contributions to this rank's
    /// `later` field (pixels of ranks painted after this one).
    pub later: Vec<Vec<(u32, f32)>>,
    /// Exact bounding box of own coverage: `(x0, y0, x1, y1)` with `x1`/`y1`
    /// exclusive.
    pub bbox: (usize, usize, usize, usize),
}

impl RankAcc {
    pub(crate) fn new(tiles: usize) -> Self {
        Self {
            cov: Vec::new(),
            later: vec![Vec::new(); tiles],
            bbox: (usize::MAX, usize::MAX, 0, 0),
        }
    }
}

/// Reusable per-thread scratch buffers (M2 item 4): field arrays and the
/// marching-squares chain table grow monotonically and are never re-allocated
/// for layers of the same or smaller size.
#[derive(Default)]
struct Scratch {
    cov: Vec<f32>,
    later: Vec<f32>,
    maxf: Vec<f32>,
    tmp: Vec<f32>,
    field: Vec<f32>,
    nxt: Vec<u32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// The per-rank coverage field produced by the scatter pass.
pub(crate) struct RankField {
    /// Padded field values (`spw × sph`); lattice `(0,0)` is 1 px outside the
    /// bbox.
    pub field: Vec<f32>,
    /// Padded field width.
    pub spw: usize,
    /// Padded field height.
    pub sph: usize,
    /// Image-space offset of the bbox (lattice (1,1) is the bbox corner).
    pub ox: f64,
    /// Image-space offset of the bbox.
    pub oy: f64,
}

/// Runs the layer stage over the whole trace: scatter → per-rank fields →
/// contours → fit → refine → primitives. Returns layers in paint order.
pub(crate) fn build_layers(
    uniques: &Uniques,
    pixels: &[u8],
    pal: &[V3],
    mix: &Mixture,
    regions: &Regions,
    w: usize,
    h: usize,
    settings: &ColorSettings,
    background_rect: bool,
    sw: &mut Stopwatch,
) -> Vec<ColorLayer> {
    let n_px = w * h;
    let tiles_x = w.div_ceil(SCATTER_TILE);
    let tiles = tiles_x * h.div_ceil(SCATTER_TILE);

    // ---- paint units: connected components of each region ----------------
    // One region may own several disconnected areas that need *different*
    // paint order (the classic case: a background and an enclosed lighter
    // shape of the same colour). Every 8-connected component becomes its own
    // paint unit, ranked by pixel count, so an enclosed component always
    // paints after the region that contains it.
    let (lab, comp, units) = paint_units(uniques, mix, regions, pixels, w, h);
    let unit_count = units.len();
    let mut order: Vec<usize> = (0..unit_count).collect();
    order.sort_by(|&a, &b| units[b].px.cmp(&units[a].px));
    let mut ranks = vec![0_u16; unit_count];
    for (r, &unit) in order.iter().enumerate() {
        ranks[unit] = r as u16;
    }
    let unit_of_rank: Vec<usize> = order.clone();
    // Fallback target for mixture weights pointing at a region from a pixel
    // that does not touch any of its components (rare: distant AA).
    let largest_of_region: Vec<Option<u16>> = {
        let mut best: Vec<Option<(u32, u16)>> = vec![None; regions.pixel_count.len()];
        for (unit, u) in units.iter().enumerate() {
            let e = &mut best[u.region as usize];
            match *e {
                Some((px, _)) if u.px <= px => {}
                _ => *e = Some((u.px, unit as u16)),
            }
        }
        best.into_iter().map(|o| o.map(|(_, u)| u)).collect()
    };

    // The background rect is only legal for a *solid*, border-connected
    // component — an enclosed same-colour shape must stay real geometry.
    let rank0_unit = unit_of_rank[0];
    let rank0_is_flat = units[rank0_unit].border
        && regions.kinds[units[rank0_unit].region as usize] == RegionKind::Flat;

    // ---- scatter pass: one walk over the pixels --------------------------
    let mut accs: Vec<RankAcc> = (0..unit_count).map(|_| RankAcc::new(tiles)).collect();
    for i in 0..n_px {
        let alpha = f32::from(pixels[i * 4 + 3]) / 255.0;
        if alpha == 0.0 {
            continue;
        }
        let uid = uniques.id[i] as usize;
        let pa = regions.band_to_region[mix.a[uid] as usize] as usize;
        let pb = regions.band_to_region[mix.b[uid] as usize] as usize;
        // Resolve each mixture side to a concrete component of its region:
        // the pixel's own component when the side is dominant, otherwise the
        // nearest neighbouring component of that region.
        let ua = unit_for_region(pa, i, &comp, &lab, w, h, &largest_of_region);
        let ub = unit_for_region(pb, i, &comp, &lab, w, h, &largest_of_region);
        let t = mix.t[uid];
        let wa = (1.0 - t) * alpha;
        let wb = t * alpha;
        // Own coverage: *both* mixture sides always contribute their
        // fractional weight. Dropping the partner's weight (the 0.1.x-scatter
        // bug) left 1-px steps in the covered-under field wherever the
        // dominant side painted later, which shattered the iso-0.5 contours
        // along every shape-on-shape edge. When both sides resolve to the
        // same unit (gradient-region interiors, pure colours) the two pushes
        // sum to exactly `alpha`.
        let (ra, rb) = match (ua, ub) {
            (Some(ua), Some(ub)) => (ranks[ua as usize] as usize, ranks[ub as usize] as usize),
            (Some(ua), None) | (None, Some(ua)) => (ranks[ua as usize] as usize, usize::MAX),
            (None, None) => continue,
        };
        accs[ra].cov.push((i as u32, wa));
        if rb != usize::MAX {
            accs[rb].cov.push((i as u32, wb));
        }
        let (lo, hi, whi) = match ra.cmp(&rb) {
            std::cmp::Ordering::Less => (ra, rb, wb),
            std::cmp::Ordering::Greater => (rb, ra, wa),
            std::cmp::Ordering::Equal => (ra, ra, 0.0),
        };
        // bbox from own coverage
        let (x, y) = (i % w, i / w);
        for r in [ra, rb] {
            if r == usize::MAX {
                continue;
            }
            let b = &mut accs[r].bbox;
            b.0 = b.0.min(x);
            b.1 = b.1.min(y);
            b.2 = b.2.max(x + 1);
            b.3 = b.3.max(y + 1);
        }
        // later contributions for every rank painted before the top rank:
        // r < lo sees both weights (= alpha); lo ≤ r < hi sees the top rank.
        let tile = (y / SCATTER_TILE) * tiles_x + x / SCATTER_TILE;
        for (r, acc) in accs.iter_mut().enumerate().take(hi) {
            let wgt = if r < lo { alpha } else { whi };
            acc.later[tile].push((i as u32, wgt));
        }
    }

    // ---- per-unit field build + trace (parallel over units) ---------------
    let trap_r = settings.trap_radius();
    let tol = f64::from(settings.tolerance_px);
    let corner_rad = settings.corner_angle_rad();
    let min_area = f64::from(settings.min_area_px2);
    let smooth = usize::from(settings.smooth_iters);
    let skip_rank0 = background_rect && rank0_is_flat;

    let results: Vec<Option<ColorLayer>> = (0..unit_count)
        .into_par_iter()
        .map(|rank| {
            let acc = &accs[rank];
            if acc.cov.is_empty() || (rank == 0 && skip_rank0) {
                return None;
            }
            let unit_idx = unit_of_rank[rank];
            let unit_px = f64::from(units[unit_idx].px);
            if unit_px < min_area * 2.5 {
                let reg = units[unit_idx].region as usize;
                if regions.kinds[reg] == RegionKind::Flat {
                    let c_self = pal[regions.flat_rank[reg]];
                    let (bx0, by0, bx1, by1) = acc.bbox;
                    let mut max_oklab_d2 = 0.0_f32;
                    let mut has_opaque_nb = false;
                    let y0 = by0.saturating_sub(1);
                    let y1 = (by1 + 1).min(h);
                    let x0 = bx0.saturating_sub(1);
                    let x1 = (bx1 + 1).min(w);
                    for yy in y0..y1 {
                        for xx in x0..x1 {
                            let nreg = lab[yy * w + xx] as usize;
                            if nreg < regions.kinds.len() && nreg != reg && regions.kinds[nreg] == RegionKind::Flat {
                                has_opaque_nb = true;
                                let c_nb = pal[regions.flat_rank[nreg]];
                                let d0 = c_self[0] - c_nb[0];
                                let d1 = c_self[1] - c_nb[1];
                                let d2 = c_self[2] - c_nb[2];
                                max_oklab_d2 = max_oklab_d2.max(d0 * d0 + d1 * d1 + d2 * d2);
                            }
                        }
                    }
                    if has_opaque_nb && max_oklab_d2 < 0.02 {
                        return None;
                    }
                }
            }
            let unit_trap = if unit_px >= min_area { trap_r } else { 0 };
            let field = build_rank_field(acc, w, unit_trap);
            let shapes = trace_rank_field(
                &field,
                acc,
                w,
                h,
                unit_trap,
                tol,
                corner_rad,
                min_area,
                smooth,
                settings.refine_iters,
                settings.primitives,
            );
            (!shapes.is_empty()).then(|| ColorLayer {
                paint: paint_of_rank(rank, &unit_of_rank, &units, regions, pal),
                shapes,
            })
        })
        .collect();

    sw.lap(Stage::Trace);
    let mut layers: Vec<ColorLayer> = results.into_iter().flatten().collect();
    // The background rect layer (rank 0, solid, full bleed) paints first.
    if skip_rank0 {
        layers.insert(
            0,
            ColorLayer {
                paint: paint_of_rank(0, &unit_of_rank, &units, regions, pal),
                shapes: vec![Shape::Rect {
                    x: 0.0,
                    y: 0.0,
                    w: w as f64,
                    h: h as f64,
                    r: 0.0,
                }],
            },
        );
    }
    layers
}

/// One paint unit: an 8-connected component of a region's dominant pixels.
struct PaintUnit {
    /// Owning region id.
    region: u16,
    /// Dominant-label pixel count.
    px: u32,
    /// Whether any pixel touches the image border (rect-shortcut eligible).
    border: bool,
}

/// Labels pixels with their dominant region and splits every region into
/// 8-connected components. Returns the per-pixel region label, the per-pixel
/// component id (`u32::MAX` where absent) and the component table.
fn paint_units(
    uniques: &Uniques,
    mix: &Mixture,
    regions: &Regions,
    pixels: &[u8],
    w: usize,
    h: usize,
) -> (Vec<u16>, Vec<u32>, Vec<PaintUnit>) {
    let n = w * h;
    let none = u16::MAX;
    let lab: Vec<u16> = uniques
        .id
        .iter()
        .enumerate()
        .map(|(i, &uid)| {
            if pixels[i * 4 + 3] > 0 {
                regions.band_to_region[dominant_rank(uid as usize, mix)]
            } else {
                none
            }
        })
        .collect();
    let mut comp = vec![u32::MAX; n];
    let mut units: Vec<PaintUnit> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if lab[start] == none || comp[start] != u32::MAX {
            continue;
        }
        let region = lab[start];
        let unit = units.len() as u32;
        units.push(PaintUnit {
            region,
            px: 0,
            border: false,
        });
        comp[start] = unit;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let u = &mut units[unit as usize];
            u.px += 1;
            let (x, y) = (p % w, p / w);
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                u.border = true;
            }
            let x0 = x.saturating_sub(1);
            let x1 = (x + 2).min(w);
            let y0 = y.saturating_sub(1);
            let y1 = (y + 2).min(h);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let q = yy * w + xx;
                    if lab[q] == region && comp[q] == u32::MAX {
                        comp[q] = unit;
                        stack.push(q);
                    }
                }
            }
        }
    }
    (lab, comp, units)
}

/// Resolves which paint unit of `region` receives pixel `i`'s coverage
/// weight: the pixel's own component when it belongs to the region, else the
/// nearest neighbouring component, else the region's largest component.
/// `None` when the region owns no component at all (a zero-pixel band).
fn unit_for_region(
    region: usize,
    i: usize,
    comp: &[u32],
    lab: &[u16],
    w: usize,
    h: usize,
    largest_of_region: &[Option<u16>],
) -> Option<u16> {
    if lab[i] == region as u16 && comp[i] != u32::MAX {
        return Some(comp[i] as u16);
    }
    // Neighbour scan: any 4-neighbour already assigned to the region.
    let (x, y) = (i % w, i / w);
    for &(dx, dy) in &[(1_usize, 0_usize), (usize::MAX, 0), (0, 1), (0, usize::MAX)] {
        let (nx, ny) = (x.wrapping_add(dx), y.wrapping_add(dy));
        if nx >= w || ny >= h {
            continue;
        }
        let q = ny * w + nx;
        if lab[q] == region as u16 && comp[q] != u32::MAX {
            return Some(comp[q] as u16);
        }
    }
    largest_of_region[region]
}

/// Paint of a unit, resolved through its paint rank.
fn paint_of_rank(
    rank: usize,
    unit_of_rank: &[usize],
    units: &[PaintUnit],
    regions: &Regions,
    pal: &[V3],
) -> Paint {
    let unit = unit_of_rank[rank];
    let region = units[unit].region as usize;
    match regions.kinds[region] {
        RegionKind::Gradient(idx) => Paint::Gradient(idx),
        RegionKind::Flat => Paint::Solid(oklab_to_rgb(pal[regions.flat_rank[region]])),
    }
}

/// Builds the padded coverage field `F′` for one rank from its scatter
/// accumulators. Work is proportional to the layer's bbox, not to K·W·H.
pub(crate) fn build_rank_field(acc: &RankAcc, w: usize, trap_r: usize) -> RankField {
    let (bx0, by0, bx1, by1) = acc.bbox;
    debug_assert!(bx1 > bx0 && by1 > by0, "rank has own coverage");
    // Expand the bbox by the trap radius + lattice margin; the dilated
    // coverage never reaches further than that.
    let x0 = bx0.saturating_sub(trap_r + 2);
    let y0 = by0.saturating_sub(trap_r + 2);
    let x1 = (bx1 + trap_r + 2).min(w);
    let y1 = by1 + trap_r + 2; // clamped via field size below
    let bw = x1 - x0;
    let bh = y1 - y0;
    let (spw, sph) = (bw + 2, bh + 2);
    let tiles_x = w.div_ceil(SCATTER_TILE);

    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        s.cov.clear();
        s.later.clear();
        s.maxf.clear();
        s.cov.resize(bw * bh, 0.0);
        s.later.resize(bw * bh, 0.0);
        s.maxf.resize(bw * bh, 0.0);

        // Drain own coverage into the local buffer.
        for &(px, weight) in &acc.cov {
            let p = px as usize;
            let (x, y) = (p % w, p / w);
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                s.cov[(y - y0) * bw + (x - x0)] += weight;
            }
        }
        // Drain later contributions from overlapping scatter tiles.
        let tx0 = x0 / SCATTER_TILE;
        let ty0 = y0 / SCATTER_TILE;
        let tx1 = x1.div_ceil(SCATTER_TILE).min(tiles_x);
        let tiles_y = acc.later.len().div_ceil(tiles_x);
        let ty1 = y1.div_ceil(SCATTER_TILE).min(tiles_y);
        for ty in ty0..ty1.min(sph) {
            for tx in tx0..tx1 {
                for &(px, weight) in &acc.later[ty * tiles_x + tx] {
                    let p = px as usize;
                    let (x, y) = (p % w, p / w);
                    if x >= x0 && x < x1 && y >= y0 && y < y1 {
                        s.later[(y - y0) * bw + (x - x0)] += weight;
                    }
                }
            }
        }

        // Separable max filter of cov with radius `trap_r` (0 → identity).
        if trap_r == 0 {
            s.maxf.copy_from_slice(&s.cov);
        } else {
            s.tmp.clear();
            s.tmp.resize(bw * bh, 0.0);
            for y in 0..bh {
                for x in 0..bw {
                    let mut m = 0.0_f32;
                    let lo = x.saturating_sub(trap_r);
                    let hi = (x + trap_r + 1).min(bw);
                    for xx in lo..hi {
                        m = m.max(s.cov[y * bw + xx]);
                    }
                    s.tmp[y * bw + x] = m;
                }
            }
            for y in 0..bh {
                for x in 0..bw {
                    let mut m = 0.0_f32;
                    let lo = y.saturating_sub(trap_r);
                    let hi = (y + trap_r + 1).min(bh);
                    for yy in lo..hi {
                        m = m.max(s.tmp[yy * bw + x]);
                    }
                    s.maxf[y * bw + x] = m;
                }
            }
        }

        // Padded field: F′ = cov + later·max(cov), clamped to 1.
        s.field.clear();
        s.field.resize(spw * sph, 0.0);
        for y in 0..bh {
            for x in 0..bw {
                let j = y * bw + x;
                let v = (s.cov[j] + s.later[j] * s.maxf[j]).min(1.0);
                s.field[(y + 1) * spw + x + 1] = v;
            }
        }
        RankField {
            field: std::mem::take(&mut s.field),
            spw,
            sph,
            ox: x0 as f64,
            oy: y0 as f64,
        }
    })
}

/// Turns one rank's field into fitted shapes: contours → corners → smooth →
/// fit → refine (M5) → primitives (M4).
fn trace_rank_field(
    field: &RankField,
    acc: &RankAcc,
    w: usize,
    h: usize,
    trap_r: usize,
    tol: f64,
    corner_rad: f64,
    min_area: f64,
    smooth: usize,
    refine_iters: u8,
    use_primitives: bool,
) -> Vec<Shape> {
    let _ = (trap_r, acc);
    let full = SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        // Rings come out in padded-lattice coordinates (bbox origin at
        // lattice (1,1), pixel centres at lattice − 0.5): translate back to
        // image space and clamp, exactly like the prototype.
        contour::contours_with(
            &field.field,
            field.spw,
            field.sph,
            &mut s.nxt,
            &[TileSpan {
                x0: 0,
                x1: field.spw.div_ceil(contour::TILE),
                y0: 0,
                y1: field.sph.div_ceil(contour::TILE),
            }],
        )
        .into_iter()
        .map(|ring| {
            ring.into_iter()
                .map(|p| {
                    [
                        (p[0] + field.ox).clamp(0.0, w as f64),
                        (p[1] + field.oy).clamp(0.0, h as f64),
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
    });
    let fitted = fit::fit_rings_parallel(&full, tol, corner_rad, smooth);
    // Primitive emission must not mix native elements with ring paths: a
    // `<circle>` cannot carry the hole subpaths of the same layer, so any
    // hole ring (opposite winding from the body ring) disables primitives
    // for the whole unit and every ring goes into the compound path.
    let use_prim = use_primitives && !has_hole_rings(&full);
    let mut shapes: Vec<Shape> = Vec::new();
    let mut path_segs: Vec<([f64; 2], Vec<PathSeg>)> = Vec::new();
    for (ri, (start, segs)) in fitted.iter().enumerate() {
        let dense = dedupe_ring(&full[ri]);
        if dense.len() < 3 || fit::area(&dense).abs() < min_area {
            continue;
        }
        let mut segs = segs.clone();
        if refine_iters > 0 {
            let _ = refine::refine_ring(
                &field.field,
                field.spw,
                field.sph,
                field.ox,
                field.oy,
                *start,
                &mut segs,
                refine_iters,
            );
        }
        if use_prim {
            let segs_arc = primitives::arcs_for_ring(*start, &segs, tol);
            if let Some(shape) = primitives::detect_ring(&dense, *start, &segs_arc, tol) {
                shapes.push(shape);
                continue;
            }
            path_segs.push((*start, to_path_segs(&segs_arc)));
        } else {
            path_segs.push((*start, to_path_segs(&segs)));
        }
    }
    // One Shape::Path per ring; the writer joins ring shapes of a layer into
    // a single compound <path> so hole rings keep subtracting (nonzero).
    for (start, segs) in path_segs {
        shapes.push(Shape::Path { start, segs });
    }
    shapes
}

/// Whether the ring set contains a hole: a ring whose winding opposes the
/// largest-|area| (body) ring. Bodies and holes always wind oppositely in
/// the marching-squares output (inside-on-left orientation).
fn has_hole_rings(rings: &[Vec<crate::color::contour::P>]) -> bool {
    let areas: Vec<f64> = rings.iter().map(|r| fit::area(&dedupe_ring(r))).collect();
    let body = areas
        .iter()
        .copied()
        .max_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0.0);
    if body == 0.0 {
        return false;
    }
    areas
        .iter()
        .any(|a| a.abs() > 1.0 && a.signum() != body.signum())
}

/// Removes consecutive duplicates and trailing closure point.
fn dedupe_ring(ring: &[crate::color::contour::P]) -> Vec<crate::color::contour::P> {
    let mut dd: Vec<crate::color::contour::P> = Vec::with_capacity(ring.len());
    for &p in ring {
        if dd.last().is_none_or(|q| fit::len(fit::sub(p, *q)) > 1e-4) {
            dd.push(p);
        }
    }
    while dd.len() > 1 && fit::len(fit::sub(dd[0], dd[dd.len() - 1])) <= 1e-4 {
        dd.pop();
    }
    dd
}

/// Converts fitted segments into writer segments.
fn to_path_segs(segs: &[Seg]) -> Vec<PathSeg> {
    segs.iter()
        .map(|s| match s {
            Seg::Line(p) => PathSeg::Line { p: *p },
            Seg::Cubic(c1, c2, p) => PathSeg::Cubic {
                c1: *c1,
                c2: *c2,
                p: *p,
            },
            Seg::Arc {
                rx,
                ry,
                rot_deg,
                large_arc,
                sweep,
                p,
            } => PathSeg::Arc {
                rx: *rx,
                ry: *ry,
                rot_deg: *rot_deg,
                large_arc: *large_arc,
                sweep: *sweep,
                p: *p,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::color::gradients::flat_regions;
    use crate::color::model::build_model;
    use crate::color::palette::{build_palette, unique_colors};
    use crate::settings::{Background, ColorSettings};

    use super::*;

    /// A 3-rank synthetic image (white bg, red square, blue square on top of
    /// the red one): the field of the white rank must be 1 everywhere except
    /// a trap underlay under the shapes; the blue rank's field must be its
    /// own square; the red rank's field must include a 1-px band under the
    /// blue square (trapping).
    #[test]
    fn scatter_fields_match_trapping_semantics() {
        let (w, h) = (12usize, 12usize);
        let mut px = vec![255_u8; w * h * 4]; // white bg
        for y in 2..8 {
            for x in 2..8 {
                let i = (y * w + x) * 4;
                px[i] = 200;
                px[i + 1] = 30;
                px[i + 2] = 30;
            }
        }
        for y in 2..5 {
            for x in 5..8 {
                let i = (y * w + x) * 4;
                px[i] = 30;
                px[i + 1] = 30;
                px[i + 2] = 200;
            }
        }
        let u = unique_colors(&px, w);
        let pal = build_palette(&u, w, h, &ColorSettings::default(), false);
        let mix = build_model(&u, &pal);
        let counts = crate::color::gradients::rank_pixel_counts(&u, &px, &mix);
        let regions = flat_regions(pal.len(), &counts);
        let ranks = regions.paint_ranks();
        // Scatter accumulators (mirror of build_layers' scatter loop; this
        // synthetic image has one component per region, so unit ranks equal
        // region ranks).
        let region_count = regions.pixel_count.len();
        let tiles = w.div_ceil(SCATTER_TILE) * h.div_ceil(SCATTER_TILE);
        let mut accs: Vec<RankAcc> = (0..region_count).map(|_| RankAcc::new(tiles)).collect();
        for i in 0..w * h {
            let alpha = f32::from(px[i * 4 + 3]) / 255.0;
            if alpha == 0.0 {
                continue;
            }
            let uid = u.id[i] as usize;
            let pa = regions.band_to_region[mix.a[uid] as usize] as usize;
            let pb = regions.band_to_region[mix.b[uid] as usize] as usize;
            let ra = ranks[pa] as usize;
            let rb = ranks[pb] as usize;
            let t = mix.t[uid];
            let (wa, wb) = ((1.0 - t) * alpha, t * alpha);
            accs[ra].cov.push((i as u32, wa));
            accs[rb].cov.push((i as u32, wb));
            let (lo, hi, whi) = match ra.cmp(&rb) {
                std::cmp::Ordering::Less => (ra, rb, wb),
                std::cmp::Ordering::Greater => (rb, ra, wa),
                std::cmp::Ordering::Equal => (ra, ra, 0.0),
            };
            let (x, y) = (i % w, i / w);
            for r in [ra, rb] {
                let b = &mut accs[r].bbox;
                b.0 = b.0.min(x);
                b.1 = b.1.min(y);
                b.2 = b.2.max(x + 1);
                b.3 = b.3.max(y + 1);
            }
            let tile = (y / SCATTER_TILE) * w.div_ceil(SCATTER_TILE) + x / SCATTER_TILE;
            for (r, acc) in accs.iter_mut().enumerate().take(hi) {
                let wgt = if r < lo { alpha } else { whi };
                acc.later[tile].push((i as u32, wgt));
            }
        }
        // Find ranks: white = rank 0, red = next, blue = top.
        let white_rank = 0;
        let red_region = regions
            .flat_rank
            .iter()
            .position(|&r| {
                pal[r]
                    .iter()
                    .zip(rgb_oklab([200, 30, 30]))
                    .all(|(a, b)| (*a - b).abs() < 0.05)
            })
            .expect("red region");
        let red_rank = ranks[red_region] as usize;
        let blue_region = regions
            .flat_rank
            .iter()
            .position(|&r| {
                pal[r]
                    .iter()
                    .zip(rgb_oklab([30, 30, 200]))
                    .all(|(a, b)| (*a - b).abs() < 0.05)
            })
            .expect("blue region");
        let blue_rank = ranks[blue_region] as usize;

        let bg = build_rank_field(&accs[white_rank], w, 1);
        // Background field: 1 everywhere except a 1-px trap band under the
        // shapes (both squares are painted after white).
        let at = |f: &RankField, x: usize, y: usize| {
            f.field[(y + 1 - f.oy as usize) * f.spw + (x + 1 - f.ox as usize)]
        };
        assert!((at(&bg, 0, 0) - 1.0).abs() < 1e-6); // white pixel: own coverage
        // Trap underlay: the red edge pixel (2,2) has white at (1,2) within
        // the 3×3 window, so the background extends 1 px under it.
        assert!((at(&bg, 2, 2) - 1.0).abs() < 1e-6);
        // Deep inside the shapes (>1 px from any white pixel) the background
        // field is exactly 0.
        assert!(at(&bg, 5, 3) < 1e-6);
        assert!(at(&bg, 6, 6) < 1.0); // red square interior: not background
        let blue = build_rank_field(&accs[blue_rank], w, 1);
        assert!((at(&blue, 6, 3) - 1.0).abs() < 1e-6); // own square
        assert!(at(&blue, 2, 2) < 1e-6); // red area only: not blue
        let red = build_rank_field(&accs[red_rank], w, 1);
        assert!((at(&red, 3, 3) - 1.0).abs() < 1e-6); // own square
        // Trap band under the blue square: blue pixel (5,3) sits next to red
        // (4,3), so the red field underlays exactly there. Deeper blue pixels
        // like (6,3) have no red within the 3×3 window → field stays 0.
        assert!((at(&red, 5, 3) - 1.0).abs() < 1e-6);
        assert!(at(&red, 6, 3) < 1e-6);
        // (6,5) is red-owned (blue occupies y∈[2,5)) → plain coverage.
        assert!((at(&red, 6, 5) - 1.0).abs() < 1e-6);
    }

    fn rgb_oklab(c: [u8; 3]) -> [f32; 3] {
        crate::color::palette::rgb_to_oklab(c)
    }

    /// End-to-end layer build: red square on white → two layers, background
    /// rect promoted, shape present, paint order correct.
    #[test]
    fn build_layers_produces_ordered_layers() {
        let (w, h) = (16usize, 16usize);
        let mut px = vec![255_u8; w * h * 4];
        for y in 4..12 {
            for x in 4..12 {
                let i = (y * w + x) * 4;
                px[i] = 220;
                px[i + 1] = 40;
                px[i + 2] = 40;
            }
        }
        let u = unique_colors(&px, w);
        let pal = build_palette(&u, w, h, &ColorSettings::default(), false);
        let mix = build_model(&u, &pal);
        let counts = crate::color::gradients::rank_pixel_counts(&u, &px, &mix);
        let regions = flat_regions(pal.len(), &counts);
        let mut sw = Stopwatch::default();
        let layers = build_layers(
            &u,
            &px,
            &pal,
            &mix,
            &regions,
            w,
            h,
            &ColorSettings::default(),
            true,
            &mut sw,
        );
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].paint, Paint::Solid([255, 255, 255]));
        assert!(matches!(layers[0].shapes[0], Shape::Rect { .. }));
        assert_eq!(layers[1].paint, Paint::Solid([220, 40, 40]));
        assert!(
            layers[1]
                .shapes
                .iter()
                .any(|s| matches!(s, Shape::Path { .. }))
        );
        assert_eq!(sw.laps().last().map(|(s, _)| *s), Some(Stage::Trace));
    }

    /// `Background::Alpha` on an opaque raster: no rect promotion — every
    /// region is traced as geometry.
    #[test]
    fn background_alpha_never_promotes_rect() {
        let (w, h) = (16usize, 16usize);
        let mut px = vec![255_u8; w * h * 4];
        for y in 4..12 {
            for x in 4..12 {
                let i = (y * w + x) * 4;
                px[i] = 220;
                px[i + 1] = 40;
                px[i + 2] = 40;
            }
        }
        let u = unique_colors(&px, w);
        let pal = build_palette(&u, w, h, &ColorSettings::default(), false);
        let mix = build_model(&u, &pal);
        let counts = crate::color::gradients::rank_pixel_counts(&u, &px, &mix);
        let regions = flat_regions(pal.len(), &counts);
        let mut sw = Stopwatch::default();
        let layers = build_layers(
            &u,
            &px,
            &pal,
            &mix,
            &regions,
            w,
            h,
            &ColorSettings::default(),
            false,
            &mut sw,
        );
        assert!(
            !layers
                .iter()
                .any(|l| l.shapes.iter().any(|s| matches!(s, Shape::Rect { .. })))
        );
        let _ = Background::Alpha;
    }

    /// Trap radius 0 disables the underlay: the red field stops exactly at
    /// its own pixels.
    #[test]
    fn trap_radius_zero_disables_underlay() {
        let (w, h) = (12usize, 12usize);
        let mut px = vec![255_u8; w * h * 4];
        for y in 2..8 {
            for x in 2..8 {
                let i = (y * w + x) * 4;
                px[i] = 200;
                px[i + 1] = 30;
                px[i + 2] = 30;
            }
        }
        for y in 2..5 {
            for x in 5..8 {
                let i = (y * w + x) * 4;
                px[i] = 30;
                px[i + 1] = 30;
                px[i + 2] = 200;
            }
        }
        let u = unique_colors(&px, w);
        let cfg = ColorSettings {
            trap_px: 0.0,
            ..ColorSettings::default()
        };
        let pal = build_palette(&u, w, h, &cfg, false);
        let mix = build_model(&u, &pal);
        let counts = crate::color::gradients::rank_pixel_counts(&u, &px, &mix);
        let regions = flat_regions(pal.len(), &counts);
        let ranks = regions.paint_ranks();
        // Red rank field: no trap band under the blue square.
        let red_region = regions
            .flat_rank
            .iter()
            .position(|&r| {
                pal[r]
                    .iter()
                    .zip(rgb_oklab([200, 30, 30]))
                    .all(|(a, b)| (*a - b).abs() < 0.05)
            })
            .expect("red region");
        // Recreate scatter for the red rank only.
        let tiles = w.div_ceil(SCATTER_TILE) * h.div_ceil(SCATTER_TILE);
        let mut accs: Vec<RankAcc> = (0..regions.pixel_count.len())
            .map(|_| RankAcc::new(tiles))
            .collect();
        for i in 0..w * h {
            let alpha = f32::from(px[i * 4 + 3]) / 255.0;
            if alpha == 0.0 {
                continue;
            }
            let uid = u.id[i] as usize;
            let ra = ranks[regions.band_to_region[mix.a[uid] as usize] as usize] as usize;
            let rb = ranks[regions.band_to_region[mix.b[uid] as usize] as usize] as usize;
            let t = mix.t[uid];
            let (wa, wb) = ((1.0 - t) * alpha, t * alpha);
            accs[ra].cov.push((i as u32, wa));
            if rb != ra {
                accs[rb].cov.push((i as u32, wb));
            }
            let (x, y) = (i % w, i / w);
            for r in [ra, rb] {
                let b = &mut accs[r].bbox;
                b.0 = b.0.min(x);
                b.1 = b.1.min(y);
                b.2 = b.2.max(x + 1);
                b.3 = b.3.max(y + 1);
            }
        }
        let red_rank = ranks[red_region] as usize;
        let red = build_rank_field(&accs[red_rank], w, 0);
        let at = |f: &RankField, x: usize, y: usize| {
            f.field[(y + 1 - f.oy as usize) * f.spw + (x + 1 - f.ox as usize)]
        };
        assert!((at(&red, 6, 5) - 1.0).abs() < 1e-6); // own pixel (blue starts at y=5? no: blue y∈[2,5))
        // (6,5) is red-owned: blue occupies y∈[2,5). ✓
        assert!(at(&red, 6, 3) < 1e-6); // blue square interior: no underlay
    }
    /// Blue disc r=40 with a white disc r=12 of the *same palette colour* as
    /// the background: the white interior must be painted as its own layer on
    /// top of the blue disc (containment), and the blue layer must keep the
    /// hole as a subpath of its compound path (no separate blue-filled hole).
    #[test]
    fn enclosed_same_colour_shape_paints_above_its_container() {
        let (w, h) = (96usize, 96usize);
        let mut px = vec![255_u8; w * h * 4];
        let (cx, cy) = (47.5_f64, 47.5_f64);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let d = (x as f64 - cx).hypot(y as f64 - cy);
                if (12.0..40.0).contains(&d) {
                    px[i] = 30;
                    px[i + 1] = 90;
                    px[i + 2] = 200;
                }
                px[i + 3] = 255;
            }
        }
        let u = unique_colors(&px, w);
        let pal = build_palette(&u, w, h, &ColorSettings::default(), false);
        let mix = build_model(&u, &pal);
        let counts = crate::color::gradients::rank_pixel_counts(&u, &px, &mix);
        let regions = flat_regions(pal.len(), &counts);
        let layers = build_layers(
            &u,
            &px,
            &pal,
            &mix,
            &regions,
            w,
            h,
            &ColorSettings {
                gradients: crate::settings::Gradients::Off,
                ..ColorSettings::default()
            },
            false,
            &mut Stopwatch::default(),
        );
        // The white unit (enclosed disc) must exist as its own layer with real
        // geometry — not swallowed by the background/rect shortcut.
        let white = layers
            .iter()
            .find(|l| l.paint == Paint::Solid([255, 255, 255]))
            .expect("white interior layer");
        assert_ne!(white.shapes.len(), 0);
        // The blue layer carries the body ring; its compound path must contain
        // the hole subpath (two rings), not fill the hole area.
        let blue = layers
            .iter()
            .find(|l| l.paint == Paint::Solid([30, 90, 200]))
            .expect("blue layer");
        assert_ne!(blue.shapes.len(), 0);
        // Render-back (feature `metrics`): the pixel at the centre is white
        // (enclosed disc), the ring midway is blue, the corner is background.
        #[cfg(feature = "metrics")]
        {
            let svg = crate::color::write::write_svg(&crate::color::ColorTrace {
                width: w,
                height: h,
                layers,
                gradients: Vec::new(),
                palette_size: pal.len(),
            });
            let tree =
                resvg::usvg::Tree::from_data(svg.as_bytes(), &resvg::usvg::Options::default())
                    .expect("parses");
            let mut pm = resvg::tiny_skia::Pixmap::new(w as u32, h as u32).expect("pixmap");
            resvg::render(
                &tree,
                resvg::tiny_skia::Transform::identity(),
                &mut pm.as_mut(),
            );
            let px_at = |x: usize, y: usize| {
                let c = pm.pixel(x as u32, y as u32).expect("in bounds");
                [c.red(), c.green(), c.blue()]
            };
            let centre = px_at(48, 48);
            assert!(
                centre.iter().all(|&c| c > 240),
                "centre must be white: {centre:?}"
            );
            let ring = px_at(48, 15);
            let blue_px = [30, 90, 200];
            assert!(
                ring.iter()
                    .zip(blue_px)
                    .all(|(a, b)| i16::from(*a) - i16::from(u8::try_from(b).unwrap_or(255)) < 40),
                "ring must be blue: {ring:?}"
            );
        }
        #[cfg(not(feature = "metrics"))]
        {
            let _ = (w, h);
        }
    }
}
