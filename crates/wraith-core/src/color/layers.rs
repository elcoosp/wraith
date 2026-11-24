//! Per-region coverage fields with trapping underlays (roadmap M1) built by
//! a single scatter pass (roadmap M2).

use std::cell::RefCell;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::color::contour::{self, P, TileSpan};
use crate::color::fit::{self, Seg};
use crate::color::gradients::{RegionKind, Regions};
use crate::color::model::{Mixture, dominant_rank};
use crate::color::palette::{Uniques, V3, oklab_to_rgb};
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
    let (lab, comp, units) = paint_units(uniques, mix, regions, pixels, w, h, min_px);
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
        let (lo, hi, whi) = match ra.cmp(&rb) {
            std::cmp::Ordering::Less => (ra as u32, rb as u32, eff_wb),
            std::cmp::Ordering::Greater => (rb as u32, ra as u32, eff_wa),
            std::cmp::Ordering::Equal => (ra as u32, ra as u32, 0.0),
        };
        px_later[i] = PxLater { lo, hi, whi, alpha };

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
            let blur = if acc.cov.len() >= 96 { settings.field_blur as usize } else { 0 };
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
    let mut layers = results.into_iter().flatten().collect::<std::vec::Vec<_>>();

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
    } else if !background_rect {
        if let Some(underlay) =
            build_silhouette_underlay(pixels, pal, regions, w, h, tol, corner_rad, smooth)
        {
            layers.insert(0, underlay);
        }
    }
    layers
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
fn absorb_small_components(lab: &mut [u16], w: usize, h: usize, min_px: u32) {
  let n = w * h;
  let none = u16::MAX;
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
    if (members.len() as u32) >= min_px {
      continue;
    }
    border_counts.clear();
    for &p in &members {
      let (x, y) = (p % w, p / w);
      for &(dx, dy) in &[(1_usize, 0_usize), (usize::MAX, 0), (0, 1), (0, usize::MAX)] {
        let (nx, ny) = (x.wrapping_add(dx), y.wrapping_add(dy));
        if nx >= w || ny >= h {
          continue;
        }
        let nreg = lab[ny * w + nx];
        if nreg != none && nreg != reg {
          if let Some((_, c)) = border_counts.iter_mut().find(|(r, _)| *r == nreg) {
            *c += 1;
          } else {
            border_counts.push((nreg, 1));
          }
        }
      }
    }
    if let Some(best) = border_counts.iter().max_by_key(|item| item.1) {
      let best_reg = best.0;
      for &p in &members {
        lab[p] = best_reg;
      }
    }
  }
}
fn paint_units(
  uniques: &Uniques,
  mix: &Mixture,
  regions: &Regions,
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
  absorb_small_components(&mut lab, w, h, min_px);
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
  if lab[i] == region as u16 && comp[i] != u32::MAX {
    return Some(comp[i]);
  }
  let (x, y) = (i % w, i / w);
  let x0 = x.saturating_sub(1);
  let x1 = (x + 2).min(w);
  let y0 = y.saturating_sub(1);
  let y1 = (y + 2).min(h);
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
      if let Some(shape) = primitives::detect_ring(&dense, *start, &segs_arc, tol) {
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
  use crate::color::gradients::flat_regions;
  use crate::color::model::build_model;
  use crate::color::palette::{build_palette, unique_colors};
  use crate::settings::{Background, ColorSettings};
  use super::*;
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
      Background::Color { rgb: [u8::MAX; 3], tolerance: 0 },
      &mut sw,
    );
    assert!(!out.layers.iter().any(|l| matches!(l.shapes.first(), Some(Shape::Rect { .. }))));
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
    let out = crate::color::trace_color(
      &img,
      &ColorSettings::default(),
      Background::None,
      &mut sw,
    );
    assert!(out.layers.len() >= 3);
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
