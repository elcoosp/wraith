//! Per-region coverage fields with trapping underlays (roadmap M1) built by
//! a single scatter pass (roadmap M2).

use std::cell::RefCell;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::color::contour::{self, P, TileSpan};
use crate::color::fit::{self, Seg};
use crate::color::gradients::{RegionKind, Regions};
use crate::color::model::{Mixture, dominant_rank};
use crate::color::palette::{Uniques, V3, oklab_to_rgb, rgb_to_oklab};
use crate::color::primitives;
use crate::color::refine;
use crate::color::{ColorLayer, Paint, PathSeg, Shape};
use crate::settings::{ColorSettings, DenoiseSettings, LineworkPolicy};
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

/// Debug hook for the out-of-tree `diag` driver (enabled by the `metrics`
/// feature): runs the same label pass as [`build_layers`] and returns the
/// per-pixel region map plus each region's median fill, without tracing.
#[cfg(feature = "metrics")]
pub fn debug_labels(
    uniques: &Uniques,
    pixels: &[u8],
    pal: &[V3],
    mix: &Mixture,
    regions: &Regions,
    w: usize,
    h: usize,
    settings: &ColorSettings,
) -> (std::vec::Vec<u16>, std::vec::Vec<[u8; 3]>) {
    let min_px = (settings.min_area_px2.ceil() as u32).max(4);
    let (lab, _comp, _units, fills) =
        paint_units(uniques, mix, regions, pal, pixels, w, h, min_px, settings.denoise);
    (lab, fills)
}

/// Label-map quantisation error of a palette variant: RMS colour distance
/// between every pixel and its region's median fill, returned as
/// `(full, interior)`. The interior figure counts only pixels whose
/// 4-neighbours share the label — boundary pixels are excluded there,
/// because their error is dominated by anti-aliasing, which both palette
/// variants carry, and would drown out the signal.
///
/// Used as the variant-quality proxy for the coarse-fallback decision in
/// `mod.rs`. Measured on the 18-image mascot corpus: the full error tracks
/// flat-band quality on smooth-ramp images (mascot −3.6 %, gsx −8 %), while
/// the interior error is what exposes the merged-facet failure mode
/// (mascot-danger +8 % where the unguarded flip cost 6.2 RMSE) — so the
/// decision requires a clear full-error win AND no meaningful interior
/// loss.
pub(crate) fn label_quant_error(
    uniques: &Uniques,
    pixels: &[u8],
    pal: &[V3],
    mix: &Mixture,
    regions: &Regions,
    w: usize,
    h: usize,
    settings: &ColorSettings,
) -> (f64, f64) {
    let min_px = (settings.min_area_px2.ceil() as u32).max(4);
    let (lab, _comp, _units, fills) =
        paint_units(uniques, mix, regions, pal, pixels, w, h, min_px, settings.denoise);
    let mut acc = 0.0_f64;
    let mut iacc = 0.0_f64;
    let mut n = 0.0_f64;
    let mut inn = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let l = lab[i];
            if l == u16::MAX {
                continue;
            }
            let f = fills[l as usize];
            let p = i * 4;
            let mut e2 = 0.0_f64;
            for ch in 0..3 {
                let d = f32::from(pixels[p + ch]) - f32::from(f[ch]);
                e2 += f64::from(d * d);
            }
            acc += e2;
            n += 1.0;
            let interior = (x > 0 && lab[i - 1] == l)
                && (x + 1 < w && lab[i + 1] == l)
                && (y > 0 && lab[i - w] == l)
                && (y + 1 < h && lab[i + w] == l);
            if interior {
                iacc += e2;
                inn += 1.0;
            }
        }
    }
    (
        (acc / (n * 3.0).max(1.0)).sqrt(),
        (iacc / (inn * 3.0).max(1.0)).sqrt(),
    )
}

fn build_layers_supersampled(_lab: &[u16], regions: &Regions, region_fills: &[[u8; 3]], pixels: &[u8], w: usize, h: usize, settings: &ColorSettings, background_rect: bool, sw: &mut Stopwatch) -> std::vec::Vec<ColorLayer> {
    let scale = usize::from(settings.supersample.max(2)); let sw_w = w * scale; let sw_h = h * scale; let none = u16::MAX; let n_reg = regions.kinds.len();
    let reg_oklab: std::vec::Vec<Option<[f32; 3]>> = (0..n_reg).map(|r| match regions.kinds[r] { RegionKind::Flat => Some(rgb_to_oklab(region_fills[r])), RegionKind::Gradient(_) => None }).collect();
    let is_top_ink: std::vec::Vec<bool> = (0..n_reg).map(|r| settings.linework == LineworkPolicy::TopInkOverlay && reg_oklab[r].is_some_and(|[l, _, _]| l < 0.24)).collect();
    let mut sub_lab = vec![none; sw_w * sw_h];
    for sy in 0..sw_h {
        let fy = (sy as f32 + 0.5) / (scale as f32) - 0.5; let y0 = (fy.floor().max(0.0) as usize).min(h - 1); let y1 = (y0 + 1).min(h - 1); let wy = (fy - (y0 as f32)).clamp(0.0, 1.0);
        for sx in 0..sw_w {
            let fx = (sx as f32 + 0.5) / (scale as f32) - 0.5; let x0 = (fx.floor().max(0.0) as usize).min(w - 1); let x1 = (x0 + 1).min(w - 1); let wx = (fx - (x0 as f32)).clamp(0.0, 1.0);
            let (i00, i10, i01, i11) = ((y0 * w + x0) * 4, (y0 * w + x1) * 4, (y1 * w + x0) * 4, (y1 * w + x1) * 4);
            let (w00, w10, w01, w11) = ((1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy);
            let a = f32::from(pixels[i00 + 3]) * w00 + f32::from(pixels[i10 + 3]) * w10 + f32::from(pixels[i01 + 3]) * w01 + f32::from(pixels[i11 + 3]) * w11;
            if a < 128.0 { continue; }
            let r = (f32::from(pixels[i00]) * w00 + f32::from(pixels[i10]) * w10 + f32::from(pixels[i01]) * w01 + f32::from(pixels[i11]) * w11).round() as u8;
            let g = (f32::from(pixels[i00 + 1]) * w00 + f32::from(pixels[i10 + 1]) * w10 + f32::from(pixels[i01 + 1]) * w01 + f32::from(pixels[i11 + 1]) * w11).round() as u8;
            let b = (f32::from(pixels[i00 + 2]) * w00 + f32::from(pixels[i10 + 2]) * w10 + f32::from(pixels[i01 + 2]) * w01 + f32::from(pixels[i11 + 2]) * w11).round() as u8;
            let okl = rgb_to_oklab([r, g, b]);
            let cands = [_lab[y0 * w + x0], _lab[y0 * w + x1], _lab[y1 * w + x0], _lab[y1 * w + x1]];
            let mut best_r = 0_u16; let mut best_d = f32::MAX;
            for ri in 0..n_reg { if !is_top_ink[ri] && !cands.contains(&(ri as u16)) { continue; } if let Some(rc) = reg_oklab[ri] { let d = crate::color::palette::d2(okl, rc); if d < best_d { best_d = d; best_r = ri as u16; } } }
            sub_lab[sy * sw_w + sx] = best_r;
        }
    }
    sever_label_bridges(&mut sub_lab, &reg_oklab, sw_w, sw_h, settings.denoise.sever_noise_bridges);
    let min_px_sub = ((settings.min_area_px2.ceil() as u32).max(4)) * ((scale * scale) as u32);
    absorb_small_components(&mut sub_lab, &reg_oklab, sw_w, sw_h, min_px_sub, settings.denoise);
    if settings.linework == LineworkPolicy::TopInkOverlay { let mut vis = vec![false; sw_w * sw_h]; let mut st = std::vec::Vec::new(); let mut mem = std::vec::Vec::new(); for start in 0..(sw_w * sw_h) { let reg = sub_lab[start]; if reg == none || (reg as usize) >= n_reg || is_top_ink[reg as usize] || vis[start] { continue; } vis[start] = true; st.push(start); mem.clear(); let mut ink_border = 0_u32; let mut best_nb = none; let mut best_cnt = 0_u32; let mut nb_counts: std::vec::Vec<(u16, u32)> = std::vec::Vec::new(); while let Some(p) = st.pop() { mem.push(p); let (px, py) = (p % sw_w, p / sw_w); for &(dx, dy) in &[(1_usize,0),(usize::MAX,0),(0,1),(0,usize::MAX)] { let (nx, ny) = (px.wrapping_add(dx), py.wrapping_add(dy)); if nx < sw_w && ny < sw_h { let nreg = sub_lab[ny * sw_w + nx]; if nreg == reg { if !vis[ny * sw_w + nx] { vis[ny * sw_w + nx] = true; st.push(ny * sw_w + nx); } } else if nreg != none && (nreg as usize) < n_reg { if is_top_ink[nreg as usize] { ink_border += 1; } else { if let Some((_, c)) = nb_counts.iter_mut().find(|(r, _)| *r == nreg) { *c += 1; } else { nb_counts.push((nreg, 1)); } } } } } } for &(nr, cnt) in &nb_counts { if cnt > best_cnt { best_cnt = cnt; best_nb = nr; } } let total_b = ink_border + best_cnt; let is_light = reg_oklab[reg as usize].is_some_and(|[l, _, _]| l > 0.88); if ink_border > 0 && best_nb == none && mem.len() < 360 && is_light { for &p in &mem { sub_lab[p] = none; } } else if ink_border > 0 && best_nb != none && (mem.len() < 240 || (mem.len() < 460 && (ink_border * 100 >= total_b * 35 || (total_b as usize * total_b as usize) > mem.len() * 22))) { for &p in &mem { sub_lab[p] = best_nb; } } } }
    let mut reg_px = vec![0_u32; n_reg]; for &l in &sub_lab { if l != none && (l as usize) < n_reg { reg_px[l as usize] += 1; } }
    let mut reg_order: std::vec::Vec<usize> = (0..n_reg).filter(|&r| reg_px[r] > 0).collect(); reg_order.sort_by(|&a, &b| is_top_ink[a].cmp(&is_top_ink[b]).then_with(|| reg_px[b].cmp(&reg_px[a])));
    let mut fill_lab = sub_lab.clone(); if settings.linework == LineworkPolicy::TopInkOverlay { for _ in 0..5 { let prev = fill_lab.clone(); for sy in 1..(sw_h - 1) { let row = sy * sw_w; for sx in 1..(sw_w - 1) { let idx = row + sx; let cur = prev[idx]; if cur != none && (cur as usize) < n_reg && is_top_ink[cur as usize] { let mut best_nb = none; let mut best_sz = 0_u32; for &(dx, dy) in &[(-1_isize,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)] { let nb = prev[((sy as isize + dy) as usize) * sw_w + ((sx as isize + dx) as usize)]; if nb != none && (nb as usize) < n_reg && !is_top_ink[nb as usize] && reg_px[nb as usize] > best_sz { best_sz = reg_px[nb as usize]; best_nb = nb; } } if best_nb != none { fill_lab[idx] = best_nb; } } } } } }
    let mut bboxes = vec![(usize::MAX, usize::MAX, 0_usize, 0_usize); n_reg]; for sy in 0..sw_h { let row = sy * sw_w; for sx in 0..sw_w { for &l in &[sub_lab[row + sx], fill_lab[row + sx]] { if l != none && (l as usize) < n_reg { let b = &mut bboxes[l as usize]; b.0 = b.0.min(sx); b.1 = b.1.min(sy); b.2 = b.2.max(sx + 1); b.3 = b.3.max(sy + 1); } } } }
    let tol = f64::from(settings.tolerance_px); let corner_rad = settings.corner_angle_rad(); let min_area = f64::from(settings.min_area_px2); let smooth = usize::from(settings.smooth_iters); let inv_scale = 1.0 / (scale as f64);
    let traced_layers: std::vec::Vec<Option<ColorLayer>> = reg_order.into_par_iter().enumerate().map(|(order_idx, reg)| {
        let (bx0, by0, bx1, by1) = bboxes[reg]; if bx1 <= bx0 || by1 <= by0 { return None; }
        if order_idx == 0 && background_rect && regions.kinds[reg] == RegionKind::Flat { return Some(ColorLayer { paint: Paint::Solid(region_fills[reg]), shapes: vec![Shape::Rect { x: 0.0, y: 0.0, w: w as f64, h: h as f64, r: 0.0 }] }); }
        let is_white = reg_oklab[reg].is_some_and(|[l, a, b]| l > 0.92 && (a * a + b * b) < 0.0025);
        let bleed = if is_top_ink[reg] || is_white || settings.trap_px <= 0.0 { 0_usize } else { 2_usize };
        let pad = bleed + 2; let x0 = bx0.saturating_sub(pad); let y0 = by0.saturating_sub(pad); let x1 = (bx1 + pad).min(sw_w); let y1 = (by1 + pad).min(sw_h); let bw = x1 - x0; let bh = y1 - y0; let (spw, sph) = (bw + 2, bh + 2);
        let target = reg as u16; let src_grid = if is_top_ink[reg] { &sub_lab } else { &fill_lab }; let mut mask = vec![0.0_f32; bw * bh];
        for y in 0..bh { let srow = (y + y0) * sw_w; let drow = y * bw; for x in 0..bw {
            if is_top_ink[reg] {
                if is_top_ink[reg] {
                    let fx = ((x + x0) as f32 + 0.5) / (scale as f32) - 0.5; let fy = ((y + y0) as f32 + 0.5) / (scale as f32) - 0.5;
                    let ix = (fx.floor().max(0.0) as usize).min(w - 1); let iy = (fy.floor().max(0.0) as usize).min(h - 1);
                    let ix1 = (ix + 1).min(w - 1); let iy1 = (iy + 1).min(h - 1);
                    let wx = (fx - ix as f32).clamp(0.0, 1.0); let wy = (fy - iy as f32).clamp(0.0, 1.0);
                    let lum = |px: usize, py: usize| { let i = (py * w + px) * 4; (0.2126 * f32::from(pixels[i]) + 0.7152 * f32::from(pixels[i+1]) + 0.0722 * f32::from(pixels[i+2])) / 255.0 };
                    let l = lum(ix, iy)*(1.0-wx)*(1.0-wy) + lum(ix1, iy)*wx*(1.0-wy) + lum(ix, iy1)*(1.0-wx)*wy + lum(ix1, iy1)*wx*wy;
                    mask[drow + x] = ((0.31 - l) / 0.12).clamp(0.0, 1.0);
                }
            } else if src_grid[srow + x + x0] == target { mask[drow + x] = 1.0; }
        } }
        for _ in 0..bleed { let mut dilated = mask.clone(); for y in 1..(bh - 1) { let row = y * bw; for x in 1..(bw - 1) { if mask[row + x] > 0.5 || mask[row + x - 1] > 0.5 || mask[row + x + 1] > 0.5 || mask[row - bw + x] > 0.5 || mask[row + bw + x] > 0.5 { dilated[row + x] = 1.0; } } } mask = dilated; }
        if is_white && bw >= 3 && bh >= 3 { let mut vis = vec![false; bw * bh]; let mut st = std::vec::Vec::new(); let mut mem = std::vec::Vec::new(); for idx in 0..(bw * bh) { if mask[idx] < 0.5 || vis[idx] { continue; } vis[idx] = true; st.push(idx); mem.clear(); let mut perim = 0_usize; while let Some(p) = st.pop() { mem.push(p); let (px, py) = (p % bw, p / bw); for &(dx, dy) in &[(1_usize,0),(usize::MAX,0),(0,1),(0,usize::MAX)] { let (nx, ny) = (px.wrapping_add(dx), py.wrapping_add(dy)); if nx >= bw || ny >= bh || mask[ny*bw+nx] < 0.5 { perim += 1; } else if !vis[ny*bw+nx] { vis[ny*bw+nx] = true; st.push(ny*bw+nx); } } } if mem.len() < 80 || (perim * perim) > mem.len() * 28 { for &p in &mem { mask[p] = 0.0; } } } }
        if is_top_ink[reg] && bw >= 5 && bh >= 5 {
            let mut er = vec![0.0_f32; bw * bh];
            for y in 1..(bh - 1) { let r = y * bw; for x in 1..(bw - 1) {
                er[r+x] = mask[r+x].min(mask[r+x-1]).min(mask[r+x+1]).min(mask[r-bw+x]).min(mask[r+bw+x]);
            } }
            for y in 1..(bh - 1) { let r = y * bw; for x in 1..(bw - 1) {
                mask[r+x] = er[r+x].max(er[r+x-1]).max(er[r+x+1]).max(er[r-bw+x]).max(er[r+bw+x]);
            } }
        }
        if is_top_ink[reg] && bw >= 5 && bh >= 5 {
            let mut er = vec![0.0_f32; bw * bh];
            for y in 1..(bh - 1) { let r = y * bw; for x in 1..(bw - 1) {
                er[r+x] = mask[r+x].min(mask[r+x-1]).min(mask[r+x+1]).min(mask[r-bw+x]).min(mask[r+bw+x]);
            } }
            for y in 1..(bh - 1) { let r = y * bw; for x in 1..(bw - 1) {
                mask[r+x] = er[r+x].max(er[r+x-1]).max(er[r+x+1]).max(er[r-bw+x]).max(er[r+bw+x]);
            } }
        }
        let mut field = vec![0.0_f32; spw * sph]; for y in 0..bh { let drow = (y + 1) * spw + 1; let srow = y * bw; field[drow..(drow + bw)].copy_from_slice(&mask[srow..(srow + bw)]); } blur_field(&mut field, spw, sph, if is_top_ink[reg] { 5 } else { 3 });
        let mut nxt = std::vec::Vec::new();
let raw_rings = contour::contours_with(&field, spw, sph, &mut nxt, &[TileSpan { x0: 0, x1: spw.div_ceil(contour::TILE), y0: 0, y1: sph.div_ceil(contour::TILE) }]).into_iter().map(|ring| ring.into_iter().map(|[px, py]| [((px + x0 as f64) * inv_scale).clamp(0.0, w as f64), ((py + y0 as f64) * inv_scale).clamp(0.0, h as f64)]).collect::<std::vec::Vec<_>>()).filter(|r| fit::area(&dedupe_ring(r)).abs() >= min_area).collect::<std::vec::Vec<_>>();
if raw_rings.is_empty() { return None; }
let fitted = fit::fit_rings_parallel(&raw_rings, tol, corner_rad, smooth);
let shapes: std::vec::Vec<Shape> = fitted.into_iter().filter(|(_, segs)| !segs.is_empty()).map(|(start, segs)| Shape::Path { start, segs: to_path_segs(&segs) }).collect();
(!shapes.is_empty()).then(|| ColorLayer { paint: match regions.kinds[reg] { RegionKind::Gradient(idx) => Paint::Gradient(idx), RegionKind::Flat => Paint::Solid(region_fills[reg]) }, shapes })
}).collect();
sw.lap(Stage::Trace); let mut out = std::vec::Vec::new();
if background_rect { if let Some(first) = traced_layers.first().and_then(|l| l.clone()) { out.push(first); } }
if let Some(mut underlay) = build_silhouette_underlay(pixels, regions, region_fills, w, h, tol, corner_rad, smooth) { if let Some(ink_r) = (0..n_reg).find(|&r| is_top_ink[r]) { underlay.paint = Paint::Solid(region_fills[ink_r]); } out.push(underlay); }
for layer in traced_layers.into_iter().skip(usize::from(background_rect)).flatten() { out.push(layer); } out
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
) -> std::vec::Vec<ColorLayer> {    let n_px = w * h;
    let min_px = (settings.min_area_px2.ceil() as u32).max(4);
    let (lab, comp, units, region_fills) =
        paint_units(uniques, mix, regions, pal, pixels, w, h, min_px, settings.denoise);
    let unit_count = units.len();
    if settings.supersample >= 2 { return build_layers_supersampled(&lab, regions, &region_fills, pixels, w, h, settings, background_rect, sw); }
    if settings.supersample >= 2 { return build_layers_supersampled(&lab, regions, &region_fills, pixels, w, h, settings, background_rect, sw); }
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
                let [r0, g0, b0] = region_fills[r];
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
                paint: paint_of_rank(rank, &unit_of_rank, &units, regions, &region_fills),
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
        &region_fills,
        pixels,
        w,
        h,
        settings,
    );

    let mut layers = Vec::new();
    if let Some(underlay) = (!background_rect)
        .then(|| {
            build_silhouette_underlay(
                pixels,
                regions,
                &region_fills,
                w,
                h,
                tol,
                corner_rad,
                smooth,
            )
        })
        .flatten()
    {
        layers.push(underlay);
    }
    for (rank, layer) in results.into_iter().enumerate() {
        if rank == 0 && skip_rank0 {
            layers.push(ColorLayer {
                paint: paint_of_rank(0, &unit_of_rank, &units, regions, &region_fills),
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

/// Mixture window kept in the band field: pixels outside it clamp to 0 so
/// the iso rings hug the transition instead of wrapping the whole parent.
const BAND_T_LO: f32 = 0.125;
const BAND_T_HI: f32 = 0.875;

/// P90−P10 spread of the collected mixture weights under which a transition
/// is a *constant-t* band rather than a spatial ramp.
///
/// A wide soft strip whose every pixel decomposes to the same blend ratio
/// (the palette gave it no rank of its own, so each pixel sits on the same
/// segment between two flat ranks — measured on the mascot corpus: a ~10 px
/// shading strip at t≈0.24 between the yellow facet and the dark band)
/// renders worst as the dominant parent's opaque fill (ΔRGB ≈ half the
/// contrast across the whole strip) and best as one flat band filled with
/// the strip's own colour. A genuine ramp sweeps the window, so its spread
/// is far above this floor.
const CONST_T_SPREAD: f32 = 0.2;

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
        if (BAND_T_LO..=BAND_T_HI).contains(&t_eff) {
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
    if !(BAND_T_LO..=BAND_T_HI).contains(&gt) {
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
    // The other side's unit may sit well beyond the strip when the palette
    // gave the strip no rank of its own: its pixels all label the dominant
    // region (measured: a ~10 px shading strip at t≈0.25 whose blend-partner
    // labels start 8–40 px away), so the partner lookup needs a radius that
    // reaches across the strip. The seg_fit gate above already pins the
    // pixel to the two regions' colour segment, so a distant unit can only
    // widen the pair's point set, never mislabel a non-blend pixel.
    let ub = unit_for_region_within(rb2, i, comp, lab, w, h, 12);
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
///
/// A *constant-t* transition (every collected pixel holds the same blend
/// ratio — the palette gave the strip no rank of its own, see
/// [`CONST_T_SPREAD`]) has no t-field iso crossing, so it renders as one
/// flat band traced from the collected coverage at iso 0.5 and filled with
/// the strip's own mean colour. Without it, opaque parent fills paint the
/// whole strip in the dominant parent's colour — the measured source of the
/// yellow fringe/dots along soft shading edges on the mascot corpus.
fn build_edge_band_layers(
    band_acc: &HashMap<(u32, u32), Vec<(u32, f32)>>,
    unit_of_rank: &[usize],
    units: &[PaintUnit],
    regions: &Regions,
    region_fills: &[[u8; 3]],
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
        let pa = paint_of_rank(ra as usize, unit_of_rank, units, regions, region_fills);
        let pb = paint_of_rank(rb as usize, unit_of_rank, units, regions, region_fills);
        let ca = rgb_to_oklab(paint_mean_rgb(&pa, &regions.gradient_defs));
        let cb = rgb_to_oklab(paint_mean_rgb(&pb, &regions.gradient_defs));
        if crate::color::palette::d2(ca, cb).sqrt() < CONTRAST_MIN {
            continue;
        }
        let (mut minx, mut miny) = (w, h);
        let (mut maxx, mut maxy) = (0_usize, 0_usize);
        for &(i, t) in pts {
            if !(BAND_T_LO..=BAND_T_HI).contains(&t) {
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
        // Mixture-weight spread of the collected pixels: a spatial ramp
        // sweeps the window, while a constant-t strip holds one blend ratio
        // (see [`CONST_T_SPREAD`]). Ramps trace the t-field; constant-t
        // strips trace the collected coverage and render as one flat band —
        // a constant t-field has no iso crossing to trace.
        let mut ts: Vec<f32> = pts
            .iter()
            .map(|&(_, t)| t)
            .filter(|t| (BAND_T_LO..=BAND_T_HI).contains(t))
            .collect();
        if ts.len() < BAND_MIN_PX {
            continue;
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let constant_t = ts[ts.len() * 9 / 10] - ts[ts.len() / 10] < CONST_T_SPREAD;

        let mut field = vec![0.0_f32; spw * sph];
        for &(i, t) in pts {
            if !(BAND_T_LO..=BAND_T_HI).contains(&t) {
                continue;
            }
            if !constant_t && !(0.25..=0.75).contains(&t) {
                continue;
            }
            let (x, y) = (i as usize % w, i as usize / w);
            field[(y - y0 + 1) * spw + (x - x0 + 1)] = if constant_t { 1.0 } else { t };
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
        let (outer, inner, gate_iso) = if constant_t {
            (trace_iso(&field, 0.5, &mut nxt), Vec::new(), 0.5_f32)
        } else {
            (
                trace_iso(&field, 0.25, &mut nxt),
                trace_iso(&field, 0.75, &mut nxt),
                0.25_f32,
            )
        };
        // Gate measurement on the raw field (see above); the blurred rings
        // remain the emitted geometry.
        let raw_outer = trace_iso(&raw, gate_iso, &mut nxt);
        let raw_inner = if constant_t {
            Vec::new()
        } else {
            trace_iso(&raw, 0.75, &mut nxt)
        };
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
        // A constant-t strip is always one flat band (its fill is the
        // strip's own mean colour, so quantising further would only add
        // seams).
        let n_sub = if constant_t {
            1
        } else if width >= WIDE3 {
            3
        } else if width >= WIDE2 {
            2
        } else {
            1
        };
        let isos: Vec<f32> = if constant_t {
            vec![BAND_T_LO, BAND_T_HI]
        } else {
            match n_sub {
                3 => vec![0.25, 5.0 / 12.0, 7.0 / 12.0, 0.75],
                2 => vec![0.25, 0.5, 0.75],
                _ => vec![0.25, 0.75],
            }
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
    regions: &Regions,
    region_fills: &[[u8; 3]],
    w: usize,
    h: usize,
    tol: f64,
    corner_rad: f64,
    smooth: usize,
) -> std::option::Option<ColorLayer> {
    if w < 3 || h < 3 || region_fills.is_empty() {
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

    let mut best_reg = 0_usize;
    let mut best_l = f32::MAX;
    for (reg, &kind) in regions.kinds.iter().enumerate() {
        if kind == RegionKind::Flat && regions.pixel_count[reg] >= 16 {
            let [l, _, _] = rgb_to_oklab(region_fills[reg]);
            if l < best_l {
                best_l = l;
                best_reg = reg;
            }
        }
    }
    Some(ColorLayer {
        paint: Paint::Solid(region_fills[best_reg]),
        shapes,
    })
}
struct PaintUnit {
    region: u16,
    px: u32,
    border: bool,
}
fn sever_label_bridges(lab: &mut [u16], reg_lab: &[Option<[f32; 3]>], w: usize, h: usize, radius: u8) {
    if radius == 0 || w < 5 || h < 5 { return; } let n = w * h; let none = u16::MAX; let mut eroded = vec![false; n]; let mut opened = vec![false; n]; let r = usize::from(radius);
    let mut reg_counts = vec![0_u32; reg_lab.len()]; for &l in lab.iter() { if (l as usize) < reg_counts.len() { reg_counts[l as usize] += 1; } }
    let eligible: std::vec::Vec<bool> = reg_lab.iter().enumerate().map(|(idx, opt)| reg_counts[idx] >= 1500 && opt.is_some_and(|[l, _, _]| (0.24..=0.88).contains(&l))).collect();
    for (reg_idx, &is_elig) in eligible.iter().enumerate() { if !is_elig { continue; } let target = reg_idx as u16; eroded.fill(false); for y in r..(h - r) { let row = y * w; for x in r..(w - r) { let idx = row + x; if lab[idx] != target { continue; } let mut all_match = true; 'win: for dy in 0..=(2 * r) { let nrow = (y + dy - r) * w; for dx in 0..=(2 * r) { if lab[nrow + x + dx - r] != target { all_match = false; break 'win; } } } if all_match { eroded[idx] = true; } } } opened.fill(false); for y in r..(h - r) { let row = y * w; for x in r..(w - r) { if !eroded[row + x] { continue; } for dy in 0..=(2 * r) { let nrow = (y + dy - r) * w; for dx in 0..=(2 * r) { opened[nrow + x + dx - r] = true; } } } } for y in r..(h - r) { let row = y * w; for x in r..(w - r) { let idx = row + x; if lab[idx] == target && !opened[idx] { let mut best_nb = none; let mut best_cnt = 0_u8; let nbs = [lab[idx - 1], lab[idx + 1], lab[idx - w], lab[idx + w]]; for &nb in &nbs { if nb != none && nb != target && reg_lab.get(nb as usize).copied().flatten().is_some_and(|[nl, _, _]| nl >= 0.24) { let c = nbs.iter().filter(|&&v| v == nb).count() as u8; if c > best_cnt { best_cnt = c; best_nb = nb; } } } if best_nb != none { lab[idx] = best_nb; } } } } }
}
fn absorb_small_components(lab: &mut [u16], reg_lab: &[Option<[f32; 3]>], w: usize, h: usize, min_px: u32, denoise: DenoiseSettings) {
    let n = w * h; let none = u16::MAX; let fg_px = lab.iter().filter(|&&l| l != none).count() as f32; let texture_max_px = min_px.max((fg_px * denoise.max_blotch_fraction).round() as u32); let max_d2 = denoise.blotch_oklab_threshold * denoise.blotch_oklab_threshold;
    let mut visited = vec![false; n]; let mut stack = std::vec::Vec::new(); let mut members = std::vec::Vec::new(); let mut border_counts: std::vec::Vec<(u16, u32)> = std::vec::Vec::new();
    for start in 0..n { let reg = lab[start]; if reg == none || visited[start] { continue; } visited[start] = true; stack.push(start); members.clear(); while let Some(p) = stack.pop() { members.push(p); let (x, y) = (p % w, p / w); for yy in y.saturating_sub(1)..(y + 2).min(h) { for xx in x.saturating_sub(1)..(x + 2).min(w) { let q = yy * w + xx; if !visited[q] && lab[q] == reg { visited[q] = true; stack.push(q); } } } } let area = members.len() as u32; if area >= texture_max_px { continue; } border_counts.clear(); let mut total_border = 0_u32; for &p in &members { let (x, y) = (p % w, p / w); for &(dx, dy) in &[(1_usize, 0_usize), (usize::MAX, 0), (0, 1), (0, usize::MAX)] { let (nx, ny) = (x.wrapping_add(dx), y.wrapping_add(dy)); if nx < w && ny < h { let nreg = lab[ny * w + nx]; if nreg != none && nreg != reg { total_border += 1; if let Some((_, c)) = border_counts.iter_mut().find(|(r, _)| *r == nreg) { *c += 1; } else { border_counts.push((nreg, 1)); } } } } } if total_border == 0 { continue; } if area < min_px { if let Some(&(best_reg, _)) = border_counts.iter().max_by_key(|item| item.1) { for &p in &members { lab[p] = best_reg; } } continue; } let c1 = reg_lab.get(reg as usize).copied().flatten(); if let Some([l1, _, _]) = c1 && l1 < 0.24 { continue; } if let Some(&(target_reg, best_c)) = border_counts.iter().max_by_key(|item| item.1) && area < min_px * 22 && best_c.saturating_mul(100) >= total_border.saturating_mul(85) && c1.is_some_and(|[l1, _, _]| l1 <= 0.90) { if let (Some(c1), Some(c2)) = (c1, reg_lab.get(target_reg as usize).copied().flatten()) { if c2[0] >= 0.24 && (area < min_px * 6 || (c1[0] < c2[0] && crate::color::palette::d2(c1, c2) <= max_d2 * 3.2)) { for &p in &members { lab[p] = target_reg; } continue; } } } let mut non_ink_border = 0_u32; let mut low_contrast_border = 0_u32; let mut best_low_reg: Option<(u16, u32)> = None; if let Some(c1) = c1 { for &(nreg, cnt) in &border_counts { if let Some([l2, a2, b2]) = reg_lab.get(nreg as usize).copied().flatten() { if l2 < 0.24 { continue; } non_ink_border += cnt; if crate::color::palette::d2(c1, [l2, a2, b2]) <= max_d2 { low_contrast_border += cnt; if best_low_reg.is_none_or(|(_, bc)| cnt > bc) { best_low_reg = Some((nreg, cnt)); } } } } } if let Some((target_reg, _)) = best_low_reg && non_ink_border > 0 && low_contrast_border.saturating_mul(100) >= non_ink_border.saturating_mul(55) { for &p in &members { lab[p] = target_reg; } } }
}
fn paint_units(uniques: &Uniques, mix: &Mixture, regions: &Regions, pal: &[[f32; 3]], pixels: &[u8], w: usize, h: usize, min_px: u32, denoise: DenoiseSettings) -> (std::vec::Vec<u16>, std::vec::Vec<u32>, std::vec::Vec<PaintUnit>, std::vec::Vec<[u8; 3]>) {
    let n = w * h; let none = u16::MAX; let mut lab = uniques.id.iter().enumerate().map(|(i, &uid)| if pixels[i * 4 + 3] > 0 { regions.band_to_region[dominant_rank(uid as usize, mix)] } else { none }).collect::<std::vec::Vec<u16>>();
    let region_fills = region_median_fills(&lab, pixels, regions, pal); let reg_lab: std::vec::Vec<Option<[f32; 3]>> = (0..regions.kinds.len()).map(|r| match regions.kinds[r] { RegionKind::Flat => Some(rgb_to_oklab(region_fills[r])), RegionKind::Gradient(_) => None }).collect();
    sever_label_bridges(&mut lab, &reg_lab, w, h, denoise.sever_noise_bridges); absorb_small_components(&mut lab, &reg_lab, w, h, min_px, denoise);
    let mut comp = vec![u32::MAX; n]; let mut units = std::vec::Vec::new(); let mut stack = std::vec::Vec::new(); for start in 0..n { if lab[start] == none || comp[start] != u32::MAX { continue; } let region = lab[start]; let unit = units.len() as u32; units.push(PaintUnit { region, px: 0, border: false }); comp[start] = unit; stack.push(start); while let Some(p) = stack.pop() { let u = &mut units[unit as usize]; u.px += 1; let (x, y) = (p % w, p / w); if x == 0 || y == 0 || x + 1 == w || y + 1 == h { u.border = true; } for yy in y.saturating_sub(1)..(y + 2).min(h) { for xx in x.saturating_sub(1)..(x + 2).min(w) { let q = yy * w + xx; if lab[q] == region && comp[q] == u32::MAX { comp[q] = unit; stack.push(q); } } } } } (lab, comp, units, region_fills)
}

const REGION_FILL_SAMPLE_CAP: usize = 8192;

/// Median sRGB of the pixels labelled with each region (strided sample,
/// capped at [`REGION_FILL_SAMPLE_CAP`] px per region). Regions without
/// labelled pixels fall back to their palette rank's centroid.
///
/// The k-means centroids are means over *unique colours* weighted by
/// frequency — when the palette is over-segmented (texture shades split into
/// many ranks) the centroids drift off the true fill and the whole region
/// renders too dark/light. The median of the actual pixels is robust to that
/// drift, so flat regions are painted from it instead.
fn region_median_fills(lab: &[u16], pixels: &[u8], regions: &Regions, pal: &[V3]) -> Vec<[u8; 3]> {
    let mut buckets: Vec<Vec<[u8; 3]>> = (0..regions.kinds.len()).map(|_| Vec::new()).collect();
    for (i, &reg) in lab.iter().enumerate() {
        if reg != u16::MAX {
            let p = i * 4;
            buckets[reg as usize].push([pixels[p], pixels[p + 1], pixels[p + 2]]);
        }
    }
    buckets
        .into_iter()
        .enumerate()
        .map(|(reg, px)| {
            if px.len() < 16 {
                return match regions.flat_rank[reg] {
                    r if r != usize::MAX => oklab_to_rgb(pal[r]),
                    _ => [0, 0, 0],
                };
            }
            let [pl, pa, pb] = pal[reg];
            if pl > 0.96 && (pa * pa + pb * pb) < 0.0004 { return [255_u8, 255, 255]; }
            let [pl, pa, pb] = pal[reg];
            if pl > 0.96 && (pa * pa + pb * pb) < 0.0004 { return [255_u8, 255, 255]; }
            let [pl, pa, pb] = pal[reg];
            if pl > 0.96 && (pa * pa + pb * pb) < 0.0004 { return [255_u8, 255, 255]; }
            let step = px.len().div_ceil(REGION_FILL_SAMPLE_CAP);
            let mut chan: [Vec<u8>; 3] = std::array::from_fn(|_| {
                Vec::with_capacity(px.len() / step + 1)
            });
            for c in px.into_iter().step_by(step) {
                for ch in 0..3 {
                    chan[ch].push(c[ch]);
                }
            }
            std::array::from_fn(|ch| {
                chan[ch].sort_unstable();
                chan[ch][chan[ch].len() / 2]
            })
        })
        .collect()
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
    region_fills: &[[u8; 3]],
) -> Paint {
    let unit = unit_of_rank[rank];
    let region = units[unit].region as usize;
    match regions.kinds[region] {
        RegionKind::Gradient(idx) => Paint::Gradient(idx),
        RegionKind::Flat => Paint::Solid(region_fills[region]),
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
