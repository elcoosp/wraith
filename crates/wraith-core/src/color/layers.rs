//! Per-region coverage fields with trapping underlays (roadmap M1) built by
//! a single scatter pass (roadmap M2).

use std::cell::RefCell;
use std::collections::HashMap;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::color::contour::{self, P, TileSpan};
use crate::color::fit::{self, Seg};
use crate::color::gradients::{RegionKind, Regions};
use crate::color::model::{Mixture, dominant_rank};
use crate::color::palette::{Uniques, V3, oklab_to_rgb, rgb_to_oklab};
use crate::color::primitives;
use crate::color::refine;
use crate::color::{ColorLayer, Paint, PathSeg, Shape};
use crate::settings::ColorSettings;
use crate::stats::{Stage, Stopwatch};

#[cfg(test)]
pub(crate) const SCATTER_TILE: usize = contour::TILE;

#[cfg(test)]
pub(crate) struct RankAcc {
    pub cov: Vec<(u32, f32)>,
    pub later: Vec<Vec<(u32, f32)>>,
    pub bbox: (usize, usize, usize, usize),
}

#[cfg(test)]
impl RankAcc {
    pub(crate) fn new(tiles: usize) -> Self {
        Self {
            cov: Vec::new(),
            later: vec![Vec::new(); tiles],
            bbox: (usize::MAX, usize::MAX, 0, 0),
        }
    }
}

struct UnitAcc {
    cov: Vec<(u32, f32)>,
    bbox: (usize, usize, usize, usize),
}

impl UnitAcc {
    fn new() -> Self {
        Self {
            cov: Vec::new(),
            bbox: (usize::MAX, usize::MAX, 0, 0),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct PxLater {
    lo: u32,
    hi: u32,
    whi: f32,
    wlo: f32,
    alpha: f32,
}

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

pub(crate) struct RankField {
    pub field: Vec<f32>,
    pub spw: usize,
    pub sph: usize,
    pub ox: f64,
    pub oy: f64,
}

/// Spatially constrained two-colour decomposition of an edge pixel.
fn local_mixture(
    i: usize,
    lab: &[u16],
    reg_rgb: &[Option<V3>],
    pixels: &[u8],
    w: usize,
    h: usize,
) -> Option<(usize, usize, f32)> {
    let none = u16::MAX;
    let (x, y) = (i % w, i / w);
    let mut votes: [(u16, u8); 6] = [(none, 0); 6];
    let mut nv = 0_usize;
    for yy in y.saturating_sub(2)..(y + 3).min(h) {
        for xx in x.saturating_sub(2)..(x + 3).min(w) {
            let l = lab[yy * w + xx];
            if l == none {
                continue;
            }
            if let Some(v) = votes[..nv].iter_mut().find(|v| v.0 == l) {
                v.1 = v.1.saturating_add(1);
            } else if nv < votes.len() {
                votes[nv] = (l, 1);
                nv += 1;
            }
        }
    }
    let own = lab[i];
    let mut cands: [u16; 6] = [none; 6];
    let mut nc = 0;
    for &(l, c) in &votes[..nv] {
        if c >= 2 || l == own {
            cands[nc] = l;
            nc += 1;
        }
    }
    if nc == 1 {
        return Some((own as usize, own as usize, 0.0));
    }
    let p = [
        f32::from(pixels[i * 4]),
        f32::from(pixels[i * 4 + 1]),
        f32::from(pixels[i * 4 + 2]),
    ];
    let mut best: Option<(f32, usize, usize, f32)> = None;
    for a in 0..nc {
        let ca = reg_rgb[cands[a] as usize]?;
        for b in a + 1..nc {
            let cb = reg_rgb[cands[b] as usize]?;
            let (d, t) = crate::color::model::seg_fit(p, ca, cb);
            if best.is_none_or(|bst| d < bst.0) {
                best = Some((d, cands[a] as usize, cands[b] as usize, t));
            }
        }
    }
    let (d, a, b, t) = best?;
    (d < 28.0 * 28.0).then_some((a, b, t))
}

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
) -> std::vec::Vec<ColorLayer> {
    let n_px = w * h;
    let min_px = (settings.min_area_px2.ceil() as u32).max(4);
    let (lab, comp, units) = paint_units(uniques, mix, regions, pal, pixels, w, h, min_px);
    let unit_count = units.len();
    if unit_count == 0 {
        sw.lap(Stage::Trace);
        return std::vec::Vec::new();
    }

    let mut order: std::vec::Vec<usize> = (0..unit_count).collect();
    order.sort_by(|&a, &b| units[b].px.cmp(&units[a].px));
    let mut ranks = vec![0_u32; unit_count];
    for (r, &unit) in order.iter().enumerate() {
        ranks[unit] = r as u32;
    }
    let unit_of_rank = order;

    let rank0_unit = unit_of_rank[0];
    let rank0_is_flat = units[rank0_unit].border
        && regions.kinds[units[rank0_unit].region as usize] == RegionKind::Flat;

    let mut accs = (0..unit_count)
        .map(|_| UnitAcc::new())
        .collect::<std::vec::Vec<_>>();
    let mut px_later = vec![PxLater::default(); n_px];
    // Pixels sitting inside a wide anti-aliased transition between two units
    // (mixture weight 0.25..=0.75), accumulated per rank pair for bridge-band
    // emission (see [`build_edge_band_layers`]).
    let mut band_acc: HashMap<(u32, u32), Vec<(u32, f32)>> = HashMap::new();
    let edge_bands_on = settings.edge_bands;

    let local_mix = settings.local_mixture;
    let reg_rgb: std::vec::Vec<Option<V3>> = (0..regions.kinds.len())
        .map(|r| match regions.kinds[r] {
            RegionKind::Flat => {
                let [r0, g0, b0] = oklab_to_rgb(pal[regions.flat_rank[r]]);
                Some([f32::from(r0), f32::from(g0), f32::from(b0)])
            }
            _ => None,
        })
        .collect();
    for i in 0..n_px {
        let alpha = f32::from(pixels[i * 4 + 3]) / 255.0;
        if alpha == 0.0 || lab[i] == u16::MAX {
            continue;
        }
        let uid = uniques.id[i] as usize;
        let dom0 = regions.band_to_region[dominant_rank(uid, mix)];
        let (pa, pb, t) = if lab[i] != dom0 {
            (lab[i] as usize, lab[i] as usize, 0.0)
        } else if !local_mix {
            (
                regions.band_to_region[mix.a[uid] as usize] as usize,
                regions.band_to_region[mix.b[uid] as usize] as usize,
                mix.t[uid],
            )
        } else if let Some(m) = local_mixture(i, &lab, &reg_rgb, pixels, w, h) {
            m
        } else {
            (
                regions.band_to_region[mix.a[uid] as usize] as usize,
                regions.band_to_region[mix.b[uid] as usize] as usize,
                mix.t[uid],
            )
        };

        let ua = unit_for_region(pa, i, &comp, &lab, w, h);
        let ub = if pb == pa {
            ua
        } else {
            unit_for_region(pb, i, &comp, &lab, w, h)
        };
        let wa = (1.0 - t) * alpha;
        let wb = t * alpha;

        let (ra, rb, eff_wa, eff_wb) = match (ua, ub) {
            (Some(ua), Some(ub)) => (
                ranks[ua as usize] as usize,
                ranks[ub as usize] as usize,
                wa,
                wb,
            ),
            (Some(ua), None) | (None, Some(ua)) => {
                (ranks[ua as usize] as usize, usize::MAX, alpha, 0.0)
            }
            (None, None) => continue,
        };

        accs[ra].cov.push((i as u32, eff_wa));
        if rb != usize::MAX && eff_wb > 0.0 {
            accs[rb].cov.push((i as u32, eff_wb));
        }
        if edge_bands_on {
            collect_edge_band(
                &mut band_acc,
                pa,
                pb,
                t,
                ra,
                rb,
                eff_wa,
                eff_wb,
                uid,
                mix,
                &regions.band_to_region,
                &reg_rgb,
                &lab,
                &comp,
                &ranks,
                pixels,
                w,
                h,
                i,
            );
        }
        let (lo, hi, whi, wlo) = match ra.cmp(&rb) {
            std::cmp::Ordering::Less => (ra as u32, rb as u32, eff_wb, eff_wa),
            std::cmp::Ordering::Greater => (rb as u32, ra as u32, eff_wa, eff_wb),
            std::cmp::Ordering::Equal => (ra as u32, ra as u32, 0.0, eff_wa),
        };
        px_later[i] = PxLater {
            lo,
            hi,
            whi,
            wlo,
            alpha,
        };

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
    }

    prune_floating_coverage(&mut accs, &px_later, w, h);

    let trap_r = settings.trap_radius();
    let tol = f64::from(settings.tolerance_px);
    let corner_rad = settings.corner_angle_rad();
    let min_area = f64::from(settings.min_area_px2);
    let smooth = usize::from(settings.smooth_iters);
    let skip_rank0 = background_rect && rank0_is_flat;

    let results = (0..unit_count)
        .into_par_iter()
        .map(|rank| {
            let acc = &accs[rank];
            if acc.cov.is_empty() || (rank == 0 && skip_rank0) {
                return None;
            }
            let unit_idx = unit_of_rank[rank];
            let unit_px = f64::from(units[unit_idx].px);
            let unit_trap = if unit_px >= min_area { trap_r } else { 0 };
            let blur = if acc.cov.len() >= 96 {
                settings.field_blur as usize
            } else {
                0
            };
            let field = build_rank_field_table(acc, &px_later, rank as u32, w, h, unit_trap, blur);
            let shapes = trace_rank_field(
                &field,
                w,
                h,
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
        .collect::<std::vec::Vec<_>>();

    sw.lap(Stage::Trace);
    let bridges = build_edge_band_layers(
        &band_acc,
        &unit_of_rank,
        &units,
        regions,
        pal,
        pixels,
        w,
        h,
        settings,
    );

    let mut layers = Vec::new();
    if let Some(underlay) =
        (!background_rect)
            .then(|| build_silhouette_underlay(pixels, pal, regions, w, h, tol, corner_rad, smooth))
            .flatten()
    {
        layers.push(underlay);
    }
    for (rank, layer) in results.into_iter().enumerate() {
        if rank == 0 && skip_rank0 {
            layers.push(ColorLayer {
                paint: paint_of_rank(0, &unit_of_rank, &units, regions, pal),
                shapes: vec![Shape::Rect {
                    x: 0.0,
                    y: 0.0,
                    w: w as f64,
                    h: h as f64,
                    r: 0.0,
                }],
            });
        } else if let Some(layer) = layer {
            layers.push(layer);
        }
        layers.extend(
            bridges
                .iter()
                .filter(|(r, _)| *r == rank)
                .map(|(_, l)| l.clone()),
        );
    }
    layers
}

/// Coverage weight below which a pixel is a *floating-scrap* candidate. A
/// real 1-px feature still holds at least this much coverage at its core
/// (a centred 1-px line lands at ≥ 0.5), while mixture misassignments at
/// 3-way junctions produce isolated scraps well below it.
const FLOATING_MIN_W: f32 = 0.45;

/// Drops isolated low-coverage scraps from every rank's coverage list.
///
/// At 3-way junctions (ink stroke meeting two fills) the local-mixture
/// decomposition can assign a stray partial weight to an unrelated rank a
/// pixel or two beyond its region. Each stray weight becomes a 1–2 px
/// floating coverage island, which marching squares turns into a closed
/// sliver ring — the measured source of the black specks at junctions and of
/// 100–150 degenerate micro-`<ellipse>` primitives per mascot image. A scrap
/// survives only when a same-rank neighbour within a 5×5 window carries a
/// solid (≥ [`FLOATING_MIN_W`]) weight, which real anti-aliased features
/// always have and artifacts never do.
fn prune_floating_coverage(accs: &mut [UnitAcc], px_later: &[PxLater], w: usize, h: usize) {
    for (rank, acc) in accs.iter_mut().enumerate() {
        let rank = rank as u32;
        acc.cov.retain(|&(px, weight)| {
            if weight >= FLOATING_MIN_W {
                return true;
            }
            let i = px as usize;
            let (x, y) = (i % w, i / w);
            for yy in y.saturating_sub(2)..(y + 3).min(h) {
                for xx in x.saturating_sub(2)..(x + 3).min(w) {
                    let pl = &px_later[yy * w + xx];
                    let anchored = (pl.lo == rank && pl.wlo >= FLOATING_MIN_W)
                        || (pl.hi == rank && pl.whi >= FLOATING_MIN_W);
                    if anchored {
                        return true;
                    }
                }
            }
            false
        });
    }
}

/// Max squared sRGB residual of a decomposition accepted for band collection
/// (mirrors the [`local_mixture`] acceptance gate).
const BAND_RESIDUAL2: f32 = 28.0 * 28.0;

/// Collects one pixel into the per-rank-pair band accumulator when it sits
/// inside a wide anti-aliased transition (mixture weight 0.25..=0.75 between
/// two distinct units).
///
/// Two sources feed the accumulator: the pixel's own (local-mixture) cross
/// decomposition, and — when that collapses to a single unit (the outer edge
/// of a ramp often lacks ≥2 same-side votes in its 5×5 window) — the global
/// unique-colour decomposition, gated by the same residual bound and by the
/// pixel carrying one of the two sides' labels.
#[allow(clippy::too_many_arguments)]
fn collect_edge_band(
    band_acc: &mut HashMap<(u32, u32), Vec<(u32, f32)>>,
    pa: usize,
    pb: usize,
    _t: f32,
    ra: usize,
    rb: usize,
    eff_wa: f32,
    eff_wb: f32,
    uid: usize,
    mix: &Mixture,
    band_to_region: &[u16],
    reg_rgb: &[Option<V3>],
    lab: &[u16],
    comp: &[u32],
    ranks: &[u32],
    pixels: &[u8],
    w: usize,
    h: usize,
    i: usize,
) {
    // Local cross decomposition: the pixel sits between its own two units.
    if pa != pb && eff_wa > 0.0 && eff_wb > 0.0 && ra != usize::MAX && rb != usize::MAX {
        let t_eff = eff_wb / (eff_wa + eff_wb);
        if (0.25..=0.75).contains(&t_eff) {
            let key = (ra.min(rb) as u32, ra.max(rb) as u32);
            band_acc.entry(key).or_default().push((i as u32, t_eff));
        }
        return;
    }
    // Global-decomposition fallback: the local window collapsed to one unit
    // (the outer edge of a ramp often lacks ≥2 same-side votes in its 5×5
    // window, so the scatter's own radius-1 unit lookup missed), but the
    // unique colour itself decomposes cleanly between two flat regions and
    // the pixel carries one of their labels. Units are re-resolved here with
    // a radius-2 window, which reaches over the 1–2 px the local votes miss.
    let (ga, gb, gt) = (mix.a[uid], mix.b[uid], mix.t[uid]);
    if !(0.25..=0.75).contains(&gt) {
        return;
    }
    let ra2 = usize::from(band_to_region[ga as usize]);
    let rb2 = usize::from(band_to_region[gb as usize]);
    if ra2 == rb2 || ra2 >= reg_rgb.len() || rb2 >= reg_rgb.len() {
        return;
    }
    let own = lab[i];
    if own != ra2 as u16 && own != rb2 as u16 {
        return;
    }
    let (Some(ca), Some(cb)) = (reg_rgb[ra2], reg_rgb[rb2]) else {
        return;
    };
    let p = [
        f32::from(pixels[i * 4]),
        f32::from(pixels[i * 4 + 1]),
        f32::from(pixels[i * 4 + 2]),
    ];
    if crate::color::model::seg_fit(p, ca, cb).0 >= BAND_RESIDUAL2 {
        return;
    }
    let ua = unit_for_region_within(ra2, i, comp, lab, w, h, 2);
    let ub = unit_for_region_within(rb2, i, comp, lab, w, h, 2);
    if let (Some(ua), Some(ub)) = (ua, ub) {
        let (ru, rv) = (ranks[ua as usize] as usize, ranks[ub as usize] as usize);
        if ru != rv && ru != usize::MAX && rv != usize::MAX {
            let key = (ru.min(rv) as u32, ru.max(rv) as u32);
            band_acc.entry(key).or_default().push((i as u32, gt));
        }
    }
}

/// Mean sRGB colour of a paint (gradient stops average out).
fn paint_mean_rgb(paint: &Paint, gradient_defs: &[crate::color::GradientDef]) -> [u8; 3] {
    match paint {
        Paint::Solid(c) => *c,
        Paint::Gradient(i) => {
            let stops = &gradient_defs[*i].stops;
            let n = stops.len().max(1);
            let mut acc = [0.0_f64; 3];
            for (_, c) in stops {
                for ch in 0..3 {
                    acc[ch] += f64::from(c[ch]);
                }
            }
            let [r, g, b] = acc;
            [
                (r / n as f64).round().clamp(0.0, 255.0) as u8,
                (g / n as f64).round().clamp(0.0, 255.0) as u8,
                (b / n as f64).round().clamp(0.0, 255.0) as u8,
            ]
        }
    }
}

/// Builds midtone **bridge bands** along wide anti-aliased transitions.
///
/// A sharp vector boundary placed at the 50 % coverage line reproduces a
/// soft (2–3 px) raster ramp with a ±25 % colour error on the quarter-step
/// pixels on both sides — the dominant error class on soft-edged art. For
/// every unit pair with (a) high contrast (ΔOklab ≥ 0.08), (b) enough band pixels
/// pixels and (c) a strip at least [`BAND_MIN_WIDTH`] px wide on average,
/// this emits midtone layers filled with the mean source colour of their
/// t-range, ordered right after the later of the two parents.
///
/// The per-pixel mixture weight `t` forms a continuous field across the
/// transition; marching squares over it at iso 0.25 / 0.75 gives the strip's
/// outer rings, and — for wide ramps — one or two further iso pairs split it
/// into 2–3 sub-bands whose fills quantise the ramp. The strip shape is the
/// iso-0.25 region minus the nested iso-0.75 region (inner rings reversed,
/// so the nonzero fill rule subtracts them). Crisp 1-px AA edges fail the
/// width gate and stay sharp.
fn build_edge_band_layers(
    band_acc: &HashMap<(u32, u32), Vec<(u32, f32)>>,
    unit_of_rank: &[usize],
    units: &[PaintUnit],
    regions: &Regions,
    pal: &[V3],
    pixels: &[u8],
    w: usize,
    h: usize,
    settings: &ColorSettings,
) -> Vec<(usize, ColorLayer)> {
    /// Minimum Oklab distance between the two parent paints.
    const CONTRAST_MIN: f32 = 0.08;
    /// Minimum mean strip width (px) — a true soft ramp, not a 1-px AA line.
    const BAND_MIN_WIDTH: f64 = 1.2;
    /// Minimum strip area (px²) — kills speckle-level fragments.
    const BAND_MIN_AREA: f64 = 6.0;
    /// Minimum band pixels before a pair is considered at all.
    const BAND_MIN_PX: usize = 12;
    /// Strip widths (px) that trigger 2 and 3 sub-bands.
    const WIDE2: f64 = 3.2;
    const WIDE3: f64 = 6.0;
    /// Mixture window kept in the field: pixels outside it clamp to 0 so the
    /// iso rings hug the transition instead of wrapping the whole parent.
    const T_LO: f32 = 0.125;
    const T_HI: f32 = 0.875;

    if band_acc.is_empty() || w < 3 || h < 3 {
        return Vec::new();
    }
    let tol = f64::from(settings.tolerance_px);
    let corner_rad = settings.corner_angle_rad();
    let smooth = usize::from(settings.smooth_iters);
    let min_speck = 1.0_f64;
    let blur = usize::from(settings.field_blur);
    let mut out = Vec::new();
    for (&(ra, rb), pts) in band_acc {
        if pts.len() < BAND_MIN_PX {
            continue;
        }
        let pa = paint_of_rank(ra as usize, unit_of_rank, units, regions, pal);
        let pb = paint_of_rank(rb as usize, unit_of_rank, units, regions, pal);
        let ca = rgb_to_oklab(paint_mean_rgb(&pa, &regions.gradient_defs));
        let cb = rgb_to_oklab(paint_mean_rgb(&pb, &regions.gradient_defs));
        if crate::color::palette::d2(ca, cb).sqrt() < CONTRAST_MIN {
            continue;
        }
        let (mut minx, mut miny) = (w, h);
        let (mut maxx, mut maxy) = (0_usize, 0_usize);
        for &(i, t) in pts {
            if t < T_LO || t > T_HI {
                continue;
            }
            let (x, y) = (i as usize % w, i as usize / w);
            minx = minx.min(x);
            miny = miny.min(y);
            maxx = maxx.max(x + 1);
            maxy = maxy.max(y + 1);
        }
        if minx >= maxx || miny >= maxy {
            continue;
        }
        let x0 = minx.saturating_sub(2);
        let y0 = miny.saturating_sub(2);
        let x1 = (maxx + 2).min(w);
        let y1 = (maxy + 2).min(h);
        let bw = x1 - x0;
        let bh = y1 - y0;
        let (spw, sph) = (bw + 2, bh + 2);
        let mut field = vec![0.0_f32; spw * sph];
        for &(i, t) in pts {
            if t < T_LO || t > T_HI {
                continue;
            }
            let (x, y) = (i as usize % w, i as usize / w);
            field[(y - y0 + 1) * spw + (x - x0 + 1)] = t;
        }
        // The width/area gate measures the *raw* mixture field: the binomial
        // blur below smears t sideways, inflating the strip and hiding real
        // ramps — measured on the mascot corpus, a 2-px silhouette ramp read
        // ≈ 1.4 px blurred but ≈ 1.05 px raw, enough to fall under the gate.
        // Band *geometry* is still traced from the blurred field (smoother
        // rings, better fits).
        let raw = field.clone();
        blur_field(&mut field, spw, sph, blur);
        let span = [TileSpan {
            x0: 0,
            x1: spw.div_ceil(contour::TILE),
            y0: 0,
            y1: sph.div_ceil(contour::TILE),
        }];
        let mut nxt = Vec::new();
        let trace_iso = |field: &[f32], iso: f32, nxt: &mut Vec<u32>| -> Vec<Vec<P>> {
            contour::contours_with_iso(field, spw, sph, nxt, &span, iso)
                .into_iter()
                .map(|r| {
                    r.into_iter()
                        .map(|[px, py]| {
                            [
                                (px + x0 as f64).clamp(0.0, w as f64),
                                (py + y0 as f64).clamp(0.0, h as f64),
                            ]
                        })
                        .collect::<Vec<P>>()
                })
                .filter(|r| {
                    dedupe_ring(r).len() >= 3 && fit::area(&dedupe_ring(r)).abs() >= min_speck
                })
                .collect::<Vec<_>>()
        };
        let outer = trace_iso(&field, 0.25, &mut nxt);
        let inner = trace_iso(&field, 0.75, &mut nxt);
        // Gate measurement on the raw field (see above); the blurred rings
        // remain the emitted geometry.
        let raw_outer = trace_iso(&raw, 0.25, &mut nxt);
        let raw_inner = trace_iso(&raw, 0.75, &mut nxt);
        let strip_area = raw_outer
            .iter()
            .map(|r| fit::area(&dedupe_ring(r)).abs())
            .sum::<f64>()
            - raw_inner
                .iter()
                .map(|r| fit::area(&dedupe_ring(r)).abs())
                .sum::<f64>();
        // Width from the strip's outer boundary only: the nested iso-0.75
        // sliver ring roughly doubles the perimeter and would halve the
        // measured width of every band.
        let outer_perim = raw_outer
            .iter()
            .map(|r| ring_perim(&dedupe_ring(r)))
            .sum::<f64>();
        let width = strip_area / outer_perim.max(1.0) * 2.0;
        if strip_area < BAND_MIN_AREA || width < BAND_MIN_WIDTH {
            continue;
        }
        // Sub-band count from measured width; each sub-band is one layer.
        let n_sub = if width >= WIDE3 {
            3
        } else if width >= WIDE2 {
            2
        } else {
            1
        };
        let isos: Vec<f32> = match n_sub {
            3 => vec![0.25, 5.0 / 12.0, 7.0 / 12.0, 0.75],
            2 => vec![0.25, 0.5, 0.75],
            _ => vec![0.25, 0.75],
        };
        for k in 0..isos.len() - 1 {
            let (lo_iso, hi_iso) = (isos[k], isos[k + 1]);
            let lo = if k == 0 {
                outer.clone()
            } else {
                trace_iso(&field, lo_iso, &mut nxt)
            };
            let hi = if k + 2 == isos.len() {
                inner.clone()
            } else {
                trace_iso(&field, hi_iso, &mut nxt)
            };
            // Mean source colour of the sub-range: the optimal fill for the
            // strip (captures non-linear ramps too).
            let (mut sr, mut sg, mut sb) = (0.0_f64, 0.0_f64, 0.0_f64);
            let mut n = 0.0_f64;
            for &(i, t) in pts {
                if t < lo_iso || t > hi_iso {
                    continue;
                }
                sr += f64::from(pixels[i as usize * 4]);
                sg += f64::from(pixels[i as usize * 4 + 1]);
                sb += f64::from(pixels[i as usize * 4 + 2]);
                n += 1.0;
            }
            if n <= 0.0 {
                continue;
            }
            let fill = [
                (sr / n).round().clamp(0.0, 255.0) as u8,
                (sg / n).round().clamp(0.0, 255.0) as u8,
                (sb / n).round().clamp(0.0, 255.0) as u8,
            ];
            // Strip = lower-iso region minus the nested higher-iso region.
            let mut rings: Vec<Vec<P>> = lo
                .iter()
                .map(|r| dedupe_ring(r))
                .filter(|r| fit::area(r).abs() >= min_speck)
                .collect();
            rings.extend(
                hi.iter()
                    .map(|r| {
                        let mut d = dedupe_ring(r);
                        d.reverse();
                        d
                    })
                    .filter(|r| fit::area(r).abs() >= min_speck),
            );
            let fitted = fit::fit_rings_parallel(&rings, tol, corner_rad, smooth);
            let shapes = fitted
                .into_iter()
                .filter(|(_, segs)| !segs.is_empty())
                .map(|(start, segs)| Shape::Path {
                    start,
                    segs: to_path_segs(&segs),
                })
                .collect::<Vec<_>>();
            if shapes.is_empty() {
                continue;
            }
            out.push((
                rb as usize,
                ColorLayer {
                    paint: Paint::Solid(fill),
                    shapes,
                },
            ));
        }
    }
    out
}

/// Closed-path length of a dense ring (perimeter estimate).
fn ring_perim(ring: &[P]) -> f64 {
    ring.windows(2)
        .map(|w| fit::len(fit::sub(w[1], w[0])))
        .sum::<f64>()
        + fit::len(fit::sub(ring[0], ring[ring.len() - 1]))
}

fn build_silhouette_underlay(
    pixels: &[u8],
    pal: &[V3],
    regions: &Regions,
    w: usize,
    h: usize,
    tol: f64,
    corner_rad: f64,
    smooth: usize,
) -> std::option::Option<ColorLayer> {
    if w < 3 || h < 3 || pal.is_empty() {
        return None;
    }
    let n = w * h;
    let mut exterior = vec![false; n];
    let mut stack = std::vec::Vec::new();
    for x in 0..w {
        for y in [0, h - 1] {
            let idx = y * w + x;
            if pixels[idx * 4 + 3] == 0 && !exterior[idx] {
                exterior[idx] = true;
                stack.push(idx);
            }
        }
    }
    for y in 0..h {
        for x in [0, w - 1] {
            let idx = y * w + x;
            if pixels[idx * 4 + 3] == 0 && !exterior[idx] {
                exterior[idx] = true;
                stack.push(idx);
            }
        }
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let nbs = [
            x.checked_sub(1).map(|nx| y * w + nx),
            (x + 1 < w).then(|| y * w + x + 1),
            y.checked_sub(1).map(|ny| ny * w + x),
            (y + 1 < h).then(|| (y + 1) * w + x),
        ];
        for nb in nbs.into_iter().flatten() {
            if pixels[nb * 4 + 3] == 0 && !exterior[nb] {
                exterior[nb] = true;
                stack.push(nb);
            }
        }
    }

    let (spw, sph) = (w + 2, h + 2);
    let mut field = vec![0.0_f32; spw * sph];
    let mut any_fg = false;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !exterior[idx] {
                let a = f32::from(pixels[idx * 4 + 3]) / 255.0;
                let v = if pixels[idx * 4 + 3] == 0 || a >= 0.95 {
                    1.0
                } else {
                    a
                };
                if v >= 0.5 {
                    any_fg = true;
                }
                field[(y + 1) * spw + x + 1] = v;
            }
        }
    }
    if !any_fg {
        return None;
    }

    let mut nxt = std::vec::Vec::new();
    let rings = contour::contours_with(
        &field,
        spw,
        sph,
        &mut nxt,
        &[TileSpan {
            x0: 0,
            x1: spw.div_ceil(contour::TILE),
            y0: 0,
            y1: sph.div_ceil(contour::TILE),
        }],
    )
    .into_iter()
    .map(|r| {
        r.into_iter()
            .map(|[px, py]| [px.clamp(0.0, w as f64), py.clamp(0.0, h as f64)])
            .collect::<std::vec::Vec<_>>()
    })
    .filter(|r| fit::area(&dedupe_ring(r)) > 4.0)
    .collect::<std::vec::Vec<_>>();

    if rings.is_empty() {
        return None;
    }

    let fitted = fit::fit_rings_parallel(&rings, tol, corner_rad, smooth);
    let mut shapes = Vec::new();
    for (start, segs) in fitted {
        if !segs.is_empty() {
            shapes.push(Shape::Path {
                start,
                segs: to_path_segs(&segs),
            });
        }
    }
    if shapes.is_empty() {
        return None;
    }

    let mut best_pal = 0_usize;
    let mut best_l = f32::MAX;
    for (reg, &kind) in regions.kinds.iter().enumerate() {
        if kind == RegionKind::Flat && regions.pixel_count[reg] >= 16 {
            let p_idx = regions.flat_rank[reg];
            let [l, _, _] = pal[p_idx];
            if l < best_l {
                best_l = l;
                best_pal = p_idx;
            }
        }
    }
    Some(ColorLayer {
        paint: Paint::Solid(oklab_to_rgb(pal[best_pal])),
        shapes,
    })
}
struct PaintUnit {
    region: u16,
    px: u32,
    border: bool,
}
fn absorb_small_components(
    lab: &mut [u16],
    reg_lab: &[Option<V3>],
    w: usize,
    h: usize,
    min_px: u32,
) {
    const MAX_TEXTURE_D2: f32 = 0.075 * 0.075;
    let n = w * h;
    let none = u16::MAX;
    let texture_max_px = min_px.max(((n / 650) as u32).max(min_px.saturating_mul(8)));

    let mut visited = vec![false; n];
    let mut stack = std::vec::Vec::new();
    let mut members = std::vec::Vec::new();
    let mut border_counts: std::vec::Vec<(u16, u32)> = std::vec::Vec::new();

    for start in 0..n {
        let reg = lab[start];
        if reg == none || visited[start] {
            continue;
        }
        visited[start] = true;
        stack.push(start);
        members.clear();
        while let Some(p) = stack.pop() {
            members.push(p);
            let (x, y) = (p % w, p / w);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 2).min(w);
            let y0 = y.saturating_sub(1);
            let y1 = (y + 2).min(h);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let q = yy * w + xx;
                    if !visited[q] && lab[q] == reg {
                        visited[q] = true;
                        stack.push(q);
                    }
                }
            }
        }

        let area = members.len() as u32;
        if area >= texture_max_px {
            continue;
        }

        border_counts.clear();
        let mut total_border = 0_u32;
        for &p in &members {
            let (x, y) = (p % w, p / w);
            for &(dx, dy) in &[(1_usize, 0_usize), (usize::MAX, 0), (0, 1), (0, usize::MAX)] {
                let (nx, ny) = (x.wrapping_add(dx), y.wrapping_add(dy));
                if nx >= w || ny >= h {
                    continue;
                }
                let nreg = lab[ny * w + nx];
                if nreg != none && nreg != reg {
                    total_border += 1;
                    if let Some((_, c)) = border_counts.iter_mut().find(|(r, _)| *r == nreg) {
                        *c += 1;
                    } else {
                        border_counts.push((nreg, 1));
                    }
                }
            }
        }

        if total_border == 0 {
            continue;
        }

        if area < min_px {
            if let Some(&(best_reg, _)) = border_counts.iter().max_by_key(|item| item.1) {
                for &p in &members {
                    lab[p] = best_reg;
                }
            }
            continue;
        }

        // For larger regions up to `texture_max_px`, check how much of the perimeter
        // is shared with perceptually close neighbours (internal mottling/texture).
        let c1 = reg_lab.get(reg as usize).copied().flatten();
        let mut low_contrast_border = 0_u32;
        let mut best_low_reg: Option<(u16, u32)> = None;

        if let Some(c1) = c1 {
            for &(nreg, cnt) in &border_counts {
                if let Some(c2) = reg_lab.get(nreg as usize).copied().flatten() {
                    if crate::color::palette::d2(c1, c2) <= MAX_TEXTURE_D2 {
                        low_contrast_border += cnt;
                        if best_low_reg.is_none_or(|(_, bc)| cnt > bc) {
                            best_low_reg = Some((nreg, cnt));
                        }
                    }
                }
            }
        }

        if let Some((target_reg, _)) = best_low_reg {
            if low_contrast_border.saturating_mul(100) >= total_border.saturating_mul(75) {
                for &p in &members {
                    lab[p] = target_reg;
                }
            }
        }
    }
}

fn paint_units(
    uniques: &Uniques,
    mix: &Mixture,
    regions: &Regions,
    pal: &[V3],
    pixels: &[u8],
    w: usize,
    h: usize,
    min_px: u32,
) -> (Vec<u16>, Vec<u32>, Vec<PaintUnit>) {
    let n = w * h;
    let none = u16::MAX;
    let mut lab = uniques
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
        .collect::<std::vec::Vec<_>>();
    let reg_lab: std::vec::Vec<Option<V3>> = (0..regions.kinds.len())
        .map(|r| match regions.kinds[r] {
            RegionKind::Flat => Some(pal[regions.flat_rank[r]]),
            RegionKind::Gradient(_) => None,
        })
        .collect();
    absorb_small_components(&mut lab, &reg_lab, w, h, min_px);
    let mut comp = vec![u32::MAX; n];
    let mut units = std::vec::Vec::new();
    let mut stack = std::vec::Vec::new();
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
fn unit_for_region(
    region: usize,
    i: usize,
    comp: &[u32],
    lab: &[u16],
    w: usize,
    h: usize,
) -> Option<u32> {
    unit_for_region_within(region, i, comp, lab, w, h, 1)
}
fn unit_for_region_within(
    region: usize,
    i: usize,
    comp: &[u32],
    lab: &[u16],
    w: usize,
    h: usize,
    radius: usize,
) -> Option<u32> {
    if lab[i] == region as u16 && comp[i] != u32::MAX {
        return Some(comp[i]);
    }
    let (x, y) = (i % w, i / w);
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(w);
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(h);
    for yy in y0..y1 {
        for xx in x0..x1 {
            let q = yy * w + xx;
            if lab[q] == region as u16 && comp[q] != u32::MAX {
                return Some(comp[q]);
            }
        }
    }
    None
}
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
fn build_rank_field_table(
    acc: &UnitAcc,
    px_later: &[PxLater],
    rank: u32,
    w: usize,
    h: usize,
    trap_r: usize,
    blur_passes: usize,
) -> RankField {
    let (bx0, by0, bx1, by1) = acc.bbox;
    debug_assert!(bx1 > bx0 && by1 > by0, "rank has own coverage");
    let x0 = bx0.saturating_sub(trap_r + 2);
    let y0 = by0.saturating_sub(trap_r + 2);
    let x1 = (bx1 + trap_r + 2).min(w);
    let y1 = (by1 + trap_r + 2).min(h);
    let bw = x1 - x0;
    let bh = y1 - y0;
    let (spw, sph) = (bw + 2, bh + 2);
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        s.cov.clear();
        s.later.clear();
        s.maxf.clear();
        s.cov.resize(bw * bh, 0.0);
        s.later.resize(bw * bh, 0.0);
        s.maxf.resize(bw * bh, 0.0);
        for &(px, weight) in &acc.cov {
            let p = px as usize;
            let (x, y) = (p % w, p / w);
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                s.cov[(y - y0) * bw + (x - x0)] += weight;
            }
        }
        for y in y0..y1 {
            let row = y * w;
            let brow = (y - y0) * bw;
            for x in x0..x1 {
                let pl = px_later[row + x];
                if rank < pl.hi {
                    s.later[brow + (x - x0)] = if rank < pl.lo { pl.alpha } else { pl.whi };
                }
            }
        }
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
        s.field.clear();
        s.field.resize(spw * sph, 0.0);
        for y in 0..bh {
            for x in 0..bw {
                let j = y * bw + x;
                let v = (s.cov[j] + s.later[j] * s.maxf[j]).min(1.0);
                s.field[(y + 1) * spw + x + 1] = v;
            }
        }
        blur_field(&mut s.field, spw, sph, blur_passes);
        RankField {
            field: std::mem::take(&mut s.field),
            spw,
            sph,
            ox: x0 as f64,
            oy: y0 as f64,
        }
    })
}
#[cfg(test)]
pub(crate) fn build_rank_field(acc: &RankAcc, w: usize, trap_r: usize) -> RankField {
    let (bx0, by0, bx1, by1) = acc.bbox;
    let x0 = bx0.saturating_sub(trap_r + 2);
    let y0 = by0.saturating_sub(trap_r + 2);
    let x1 = (bx1 + trap_r + 2).min(w);
    let y1 = by1 + trap_r + 2;
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
        for &(px, weight) in &acc.cov {
            let p = px as usize;
            let (x, y) = (p % w, p / w);
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                s.cov[(y - y0) * bw + (x - x0)] += weight;
            }
        }
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
        if trap_r == 0 {
            s.maxf.copy_from_slice(&s.cov);
        } else {
            s.tmp.clear();
            s.tmp.resize(bw * bh, 0.0);
            for y in 0..bh {
                for x in 0..bw {
                    let mut m = 0.0_f32;
                    for xx in x.saturating_sub(trap_r)..(x + trap_r + 1).min(bw) {
                        m = m.max(s.cov[y * bw + xx]);
                    }
                    s.tmp[y * bw + x] = m;
                }
            }
            for y in 0..bh {
                for x in 0..bw {
                    let mut m = 0.0_f32;
                    for yy in y.saturating_sub(trap_r)..(y + trap_r + 1).min(bh) {
                        m = m.max(s.tmp[yy * bw + x]);
                    }
                    s.maxf[y * bw + x] = m;
                }
            }
        }
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
fn trace_rank_field(
    field: &RankField,
    w: usize,
    h: usize,
    tol: f64,
    corner_rad: f64,
    min_area: f64,
    smooth: usize,
    refine_iters: u8,
    use_primitives: bool,
) -> Vec<Shape> {
    let full = SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
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
                .map(|[px, py]| {
                    [
                        (px + field.ox).clamp(0.0, w as f64),
                        (py + field.oy).clamp(0.0, h as f64),
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
    });
    let fitted = fit::fit_rings_parallel(&full, tol, corner_rad, smooth);
    let use_prim = use_primitives && !has_hole_rings(&full);
    // A primitive claim is a *standalone shape* claim: the total ring area of
    // the layer gates whether any single ring may be rewritten (see
    // `primitives::detect_ring`).
    let layer_area = full
        .iter()
        .map(|r| fit::area(&dedupe_ring(r)).abs())
        .sum::<f64>();
    let mut shapes = std::vec::Vec::new();
    let mut path_segs: Vec<(P, Vec<PathSeg>)> = Vec::new();
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
            if let Some(shape) = primitives::detect_ring(&dense, *start, &segs_arc, tol, layer_area)
            {
                shapes.push(shape);
                continue;
            }
            path_segs.push((*start, to_path_segs(&segs_arc)));
        } else {
            path_segs.push((*start, to_path_segs(&segs)));
        }
    }
    for (start, segs) in path_segs {
        shapes.push(Shape::Path { start, segs });
    }
    shapes
}
fn has_hole_rings(rings: &[Vec<P>]) -> bool {
    let areas = rings
        .iter()
        .map(|r| fit::area(&dedupe_ring(r)))
        .collect::<std::vec::Vec<_>>();
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
fn dedupe_ring(ring: &[P]) -> Vec<P> {
    let mut dd = std::vec::Vec::with_capacity(ring.len());
    for &p in ring {
        if dd.last().is_none_or(|q| fit::len(fit::sub(p, *q)) > 1e-4) {
            dd.push(p);
        }
    }
    while dd.len() > 1 && fit::len(fit::sub(*dd.first().unwrap(), *dd.last().unwrap())) <= 1e-4 {
        dd.pop();
    }
    dd
}
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
    use super::*;
    use crate::color::gradients::flat_regions;
    use crate::color::model::build_model;
    use crate::color::palette::{build_palette, unique_colors};
    use crate::settings::{Background, ColorSettings};
    fn rgb_oklab(c: [u8; 3]) -> V3 {
        crate::color::palette::rgb_to_oklab(c)
    }
    #[test]
    fn scatter_fields_match_trapping_semantics() {
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
        let pal = build_palette(&u, w, h, &ColorSettings::default(), false);
        let mix = build_model(&u, &pal);
        let counts = crate::color::gradients::rank_pixel_counts(&u, &px, &mix);
        let regions = flat_regions(pal.len(), &counts);
        let ranks = regions.paint_ranks();
        let region_count = regions.pixel_count.len();
        let tiles = w.div_ceil(SCATTER_TILE) * h.div_ceil(SCATTER_TILE);
        let mut accs = (0..region_count)
            .map(|_| RankAcc::new(tiles))
            .collect::<Vec<_>>();
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
        assert!(bg.field[2 * bg.spw + 2] > 0.9);
        assert!(bg.field[5 * bg.spw + 5] < 0.1);
        let red = build_rank_field(&accs[red_rank], w, 1);
        assert!(red.field[4 * red.spw + 6] > 0.9);
        let blue = build_rank_field(&accs[blue_rank], w, 1);
        assert!(blue.field[4 * blue.spw + 6] > 0.9);
        assert!(blue.field[6 * blue.spw + 4] < 0.1);
    }
    #[test]
    fn background_alpha_never_promotes_rect() {
        let mut px = vec![0_u8; 8 * 8 * 4];
        for y in 2..6 {
            for x in 2..6 {
                let i = (y * 8 + x) * 4;
                px[i] = 20;
                px[i + 1] = 160;
                px[i + 2] = 90;
                px[i + 3] = 255;
            }
        }
        let img = vtracer::ColorImage {
            width: 8,
            height: 8,
            pixels: px,
        };
        let mut sw = Stopwatch::default();
        let out = crate::color::trace_color(
            &img,
            &ColorSettings::default(),
            Background::Color {
                rgb: [u8::MAX; 3],
                tolerance: 0,
            },
            &mut sw,
        );
        assert!(
            !out.layers
                .iter()
                .any(|l| matches!(l.shapes.first(), Some(Shape::Rect { .. })))
        );
    }
    #[test]
    fn enclosed_same_colour_shape_paints_above_its_container() {
        let mut px = vec![255_u8; 16 * 16 * 4];
        for y in 2..14 {
            for x in 2..14 {
                let i = (y * 16 + x) * 4;
                px[i] = 20;
                px[i + 1] = 30;
                px[i + 2] = 40;
            }
        }
        for y in 6..10 {
            for x in 6..10 {
                let i = (y * 16 + x) * 4;
                px[i] = 255;
                px[i + 1] = 255;
                px[i + 2] = 255;
            }
        }
        let img = vtracer::ColorImage {
            width: 16,
            height: 16,
            pixels: px,
        };
        let mut sw = Stopwatch::default();
        let out =
            crate::color::trace_color(&img, &ColorSettings::default(), Background::None, &mut sw);
        assert!(out.layers.len() >= 3);
    }

    /// A wide (3 px) soft transition between two high-contrast flats must be
    /// bridged by a midtone band layer painted after both parents; without it
    /// the quarter-step pixels on either side of the sharp boundary carry a
    /// ±25 % colour error (the dominant defect class on soft-edged art).
    #[test]
    fn wide_soft_edge_emits_midtone_bridge_band() {
        let (w, h) = (32usize, 16usize);
        let mut px = vec![0_u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                // 2-px transparent frame, then dark | 3-px ramp | light.
                let (r, g, b, a) = match x {
                    0 | 1 | 30 | 31 => (0, 0, 0, 0_u8),
                    2..=13 => (10, 10, 10, 255),
                    14 => (74, 74, 74, 255),
                    15 => (128, 128, 128, 255),
                    16 => (181, 181, 181, 255),
                    _ => (240, 240, 240, 255),
                };
                px[i] = r;
                px[i + 1] = g;
                px[i + 2] = b;
                px[i + 3] = a;
            }
        }
        let img = vtracer::ColorImage {
            width: w,
            height: h,
            pixels: px,
        };
        let mut sw = Stopwatch::default();
        let out =
            crate::color::trace_color(&img, &ColorSettings::default(), Background::Alpha, &mut sw);
        let mid = out
      .layers
      .iter()
      .position(|l| matches!(l.paint, Paint::Solid(c) if c.iter().all(|&v| (i32::from(v) - 128).abs() <= 6)));
        let mid = mid.expect("no midtone bridge band layer");
        let after_dark = out.layers[..mid]
            .iter()
            .any(|l| matches!(l.paint, Paint::Solid(c) if c.iter().all(|&v| v <= 40)));
        let after_light = out.layers[..mid]
            .iter()
            .any(|l| matches!(l.paint, Paint::Solid(c) if c.iter().all(|&v| v >= 200)));
        assert!(
            after_dark && after_light,
            "bridge must paint after both parents"
        );
    }

    /// A crisp 1-px boundary (no ramp) must not gain a bridge band: the width
    /// gate keeps hairline AA bands sharp instead of blurring them.
    #[test]
    fn crisp_edge_emits_no_bridge_band() {
        let (w, h) = (32usize, 16usize);
        let mut px = vec![0_u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let (r, a) = match x {
                    0 | 1 | 30 | 31 => (0_u8, 0_u8),
                    2..=15 => (10, 255),
                    _ => (240, 255),
                };
                px[i] = r;
                px[i + 1] = r;
                px[i + 2] = r;
                px[i + 3] = a;
            }
        }
        let img = vtracer::ColorImage {
            width: w,
            height: h,
            pixels: px,
        };
        let mut sw = Stopwatch::default();
        let out =
            crate::color::trace_color(&img, &ColorSettings::default(), Background::Alpha, &mut sw);
        assert!(
            !out.layers.iter().any(|l| matches!(l.paint, Paint::Solid(c)
        if c.iter().all(|&v| (i32::from(v) - 128).abs() <= 20))),
            "crisp edge must not gain a midtone bridge"
        );
    }

    /// `edge_bands = false` disables bridge emission entirely (escape hatch).
    #[test]
    fn edge_bands_off_disables_bridge() {
        let (w, h) = (32usize, 16usize);
        let mut px = vec![0_u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let (v, a) = match x {
                    0 | 1 | 30 | 31 => (0_u8, 0_u8),
                    2..=13 => (10, 255),
                    14 => (74, 255),
                    15 => (128, 255),
                    16 => (181, 255),
                    _ => (240, 255),
                };
                px[i] = v;
                px[i + 1] = v;
                px[i + 2] = v;
                px[i + 3] = a;
            }
        }
        let img = vtracer::ColorImage {
            width: w,
            height: h,
            pixels: px,
        };
        let cfg = ColorSettings {
            edge_bands: false,
            ..ColorSettings::default()
        };
        let mut sw = Stopwatch::default();
        let out = crate::color::trace_color(&img, &cfg, Background::Alpha, &mut sw);
        assert!(
            !out.layers.iter().any(|l| matches!(l.paint, Paint::Solid(c)
        if c.iter().all(|&v| (i32::from(v) - 128).abs() <= 20))),
            "edge_bands=false must not emit bridges"
        );
    }
}

/// Separable [1 2 1]/4 binomial blur (σ≈0.71 px per pass), edges clamped.
fn blur_field(f: &mut [f32], w: usize, h: usize, passes: usize) {
    if passes == 0 || w < 3 || h < 3 {
        return;
    }
    let mut tmp = vec![0.0_f32; f.len()];
    for _ in 0..passes {
        for y in 0..h {
            let r = y * w;
            for x in 0..w {
                let l = f[r + x.saturating_sub(1)];
                let rr = f[r + (x + 1).min(w - 1)];
                tmp[r + x] = 0.25 * l + 0.5 * f[r + x] + 0.25 * rr;
            }
        }
        for y in 0..h {
            let (u, d) = (y.saturating_sub(1) * w, (y + 1).min(h - 1) * w);
            for x in 0..w {
                f[y * w + x] = 0.25 * tmp[u + x] + 0.5 * tmp[y * w + x] + 0.25 * tmp[d + x];
            }
        }
    }
}
