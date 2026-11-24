//! Palette construction: unique colours → Oklab k-means → merge → AA-ghost
//! pruning (the measured prototype, productionised; roadmap M1/M2).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::color::model::seg_fit;
use crate::settings::ColorSettings;

/// Three-channel float colour (Oklab or gamma sRGB, depending on context).
pub type V3 = [f32; 3];

/// sRGB transfer function, pre-tabulated for the 256 input values (M2: the
/// LUT removes the `powf` from every palette pass).
static SRGB_TO_LINEAR: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0.0_f32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *v = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
    }
    lut
});

/// sRGB (gamma-encoded) → linear, via the transfer-function LUT.
#[must_use]
pub fn lin(u: u8) -> f32 {
    SRGB_TO_LINEAR[usize::from(u)]
}

/// sRGB u8 → Oklab (perceptual space used for all palette decisions).
#[must_use]
pub fn rgb_to_oklab(c: [u8; 3]) -> V3 {
    let (r, g, b) = (lin(c[0]), lin(c[1]), lin(c[2]));
    let l = (0.412_221_5 * r + 0.536_332_5 * g + 0.051_445_99 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_84 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

/// Linear → gamma-encoded sRGB byte, with rounding.
#[must_use]
pub fn enc(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let v = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    ((v * 255.0 + 0.5).floor().clamp(0.0, 255.0)) as u8
}

/// Oklab → sRGB u8.
#[must_use]
pub fn oklab_to_rgb(c: V3) -> [u8; 3] {
    let l = (c[0] + 0.396_337_78 * c[1] + 0.215_803_76 * c[2]).powi(3);
    let m = (c[0] - 0.105_561_35 * c[1] - 0.063_854_17 * c[2]).powi(3);
    let s = (c[0] - 0.089_484_18 * c[1] - 1.291_485_5 * c[2]).powi(3);
    [
        enc(4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_93 * s),
        enc(-1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s),
        enc(-0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s),
    ]
}

/// Squared Euclidean distance in 3-space.
#[must_use]
pub fn d2(a: V3, b: V3) -> f32 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// Nearest palette entry to `c` (index into `pal`).
#[must_use]
pub fn nearest(c: V3, pal: &[V3]) -> usize {
    let mut best = 0;
    let mut bd = f32::MAX;
    for (i, p) in pal.iter().enumerate() {
        let d = d2(c, *p);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

/// Identity hasher for `u32` keys — the key space is tiny and already hashed
/// by the image content itself.
#[derive(Clone, Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.write_u64(u64::from(i));
    }

    fn write_u32(&mut self, i: u32) {
        self.write_u64(u64::from(i));
    }

    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// Per-pixel decomposition of the raster into unique colours.
pub struct Uniques {
    /// Unique RGB colours, in first-appearance order.
    pub colors: Vec<[u8; 3]>,
    /// Unique-id per pixel.
    pub id: Vec<u32>,
    /// Pixel count per unique colour (only alpha > 0 pixels count).
    pub count: Vec<f32>,
    /// Raster width in pixels (pixel indices resolve to coordinates).
    pub width: usize,
    /// True when any pixel has alpha < 255.
    pub any_alpha: bool,
}

/// Single pass over the pixel buffer: unique colours, ids and counts.
#[must_use]
pub fn unique_colors(pixels: &[u8], width: usize) -> Uniques {
    let mut map: FxMap<u32, u32> = FxMap::default();
    let mut colors: Vec<[u8; 3]> = Vec::new();
    let mut count: Vec<f32> = Vec::new();
    let mut id = vec![0_u32; pixels.len() / 4];
    let mut any_alpha = false;
    for (i, px) in pixels.as_chunks::<4>().0.iter().enumerate() {
        if px[3] < 255 {
            any_alpha = true;
        }
        let key = (u32::from(px[0]) << 16) | (u32::from(px[1]) << 8) | u32::from(px[2]);
        let uid = *map.entry(key).or_insert_with(|| {
            colors.push([px[0], px[1], px[2]]);
            count.push(0.0);
            (colors.len() - 1) as u32
        });
        id[i] = uid;
        if px[3] > 0 {
            count[uid as usize] += 1.0;
        }
    }
    Uniques {
        colors,
        id,
        count,
        width,
        any_alpha,
    }
}

/// Weighted k-means over the unique colours, initialised by a greedy
/// farthest-point strategy (deterministic, no RNG).
fn kmeans(pts: &[V3], w: &[f32], k: usize, iters: usize) -> Vec<V3> {
    let n = pts.len();
    let mut cent: Vec<V3> = Vec::with_capacity(k);
    let first = (0..n)
        .max_by(|&a, &b| w[a].partial_cmp(&w[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    cent.push(pts[first]);
    let mut mind: Vec<f32> = pts.iter().map(|p| d2(*p, pts[first])).collect();
    while cent.len() < k {
        let (bi, bs) = (0..n)
            .map(|i| (i, w[i] * mind[i]))
            .fold((0, -1.0_f32), |a, b| if b.1 > a.1 { b } else { a });
        if bs <= 0.0 {
            break;
        }
        cent.push(pts[bi]);
        for i in 0..n {
            mind[i] = mind[i].min(d2(pts[i], pts[bi]));
        }
    }
    for _ in 0..iters {
        let assign: Vec<usize> = pts.par_iter().map(|p| nearest(*p, &cent)).collect();
        let mut sum = vec![[0.0_f64; 4]; cent.len()];
        for (i, &a) in assign.iter().enumerate() {
            let wi = f64::from(w[i]);
            sum[a][0] += wi * f64::from(pts[i][0]);
            sum[a][1] += wi * f64::from(pts[i][1]);
            sum[a][2] += wi * f64::from(pts[i][2]);
            sum[a][3] += wi;
        }
        for (c, s) in cent.iter_mut().zip(&sum) {
            if s[3] > 0.0 {
                *c = [
                    (s[0] / s[3]) as f32,
                    (s[1] / s[3]) as f32,
                    (s[2] / s[3]) as f32,
                ];
            }
        }
    }
    cent
}

/// Returns true when `c` is an achromatic highlight (`L > 0.80`, `C < 0.035` in Oklab),
/// where subtle tonal steps carry deliberate shading structure rather than chroma noise.
fn is_neutral_highlight(c: V3) -> bool {
    let l = *c.first().unwrap_or(&0.0);
    let a = *c.get(1).unwrap_or(&0.0);
    let b = *c.get(2).unwrap_or(&0.0);
    l > 0.80 && (a * a + b * b) < 0.035 * 0.035
}

/// Merges palette colours closer than `thr` (Oklab), weighted by pixel count.
/// Tightens the threshold in achromatic highlights so soft white-surface shading survives.
fn merge_close(pal: &mut Vec<V3>, cnt: &mut Vec<f64>, thr: f32) {
    loop {
        let mut best = None;
        let mut best_ratio = 1.0_f32;
        for i in 0..pal.len() {
            for j in i + 1..pal.len() {
                let pi = pal.get(i).copied().unwrap_or_default();
                let pj = pal.get(j).copied().unwrap_or_default();
                let eff_thr = if is_neutral_highlight(pi) && is_neutral_highlight(pj) {
                    thr * 0.45
                } else {
                    thr
                };
                if eff_thr <= 0.0 {
                    continue;
                }
                let ratio = d2(pi, pj) / (eff_thr * eff_thr);
                if ratio < best_ratio {
                    best_ratio = ratio;
                    best = Some((i, j));
                }
            }
        }
        let Some((i, j)) = best else {
            break;
        };
        let (wi, wj) = (cnt[i].max(1e-9), cnt[j].max(1e-9));
        let pj = pal[j];
        for (ch, dst) in pal[i].iter_mut().enumerate() {
            *dst = ((f64::from(*dst) * wi + f64::from(pj[ch]) * wj) / (wi + wj)) as f32;
        }
        cnt[i] += cnt[j];
        pal.swap_remove(j);
        cnt.swap_remove(j);
    }
}

/// Over-segmented palette floor. Raising this above 48 was measured to cut
/// interior rmse by up to 63 % on soft-shaded 1024 px art — but it also
/// re-shuffles region segmentation badly enough to paint large wrong-colour
/// blocks on the guide fixture (root cause not isolated), so it stays at the
/// proven 48 until that is understood. Tiny inputs stay bounded by the
/// unique-colour budget: at most a third of the distinct colours become
/// centroids.
pub(crate) const OVERSEGMENT_FLOOR: usize = 48;

/// Cap for the over-segmented k-means: never below [`OVERSEGMENT_FLOOR`],
/// never above a third of the input uniques (minimum 4).
#[must_use]
pub(crate) fn oversegment_cap(cap: usize, uniques_len: usize) -> usize {
    cap.max(OVERSEGMENT_FLOOR).min((uniques_len / 3).max(4))
}

/// Histogram cap for the palette input: unique colours beyond this are binned
/// to a 5-bit-per-channel histogram before clustering (M2: bounds the k-means
/// input for photo-like inputs without a heavy quantizer dependency).
const UNIQUE_COLOR_CAP: usize = 32_768;

fn bounded_points(uniques: &Uniques) -> (Vec<V3>, Vec<f32>) {
    let ulab: Vec<V3> = uniques.colors.iter().map(|c| rgb_to_oklab(*c)).collect();
    if uniques.colors.len() <= UNIQUE_COLOR_CAP {
        return (ulab, uniques.count.clone());
    }
    let mut bins: HashMap<u32, (usize, [f64; 3], f64), BuildHasherDefault<Fx>> = FxMap::default();
    let mut order: Vec<u32> = Vec::new();
    for (u, &c) in uniques.colors.iter().enumerate() {
        let wt = f64::from(uniques.count[u]);
        if wt <= 0.0 {
            continue;
        }
        let [r, g, b] = c;
        let key = (u32::from(r >> 3) << 10) | (u32::from(g >> 3) << 5) | u32::from(b >> 3);
        let e = bins.entry(key).or_insert_with(|| {
            order.push(key);
            (order.len() - 1, [0.0; 3], 0.0)
        });
        let [l, a, bb] = ulab[u];
        e.1[0] += f64::from(l) * wt;
        e.1[1] += f64::from(a) * wt;
        e.1[2] += f64::from(bb) * wt;
        e.2 += wt;
    }
    let mut pts = vec![[0.0_f32; 3]; order.len()];
    let mut w_scalar = vec![0.0_f32; order.len()];
    for (_, (idx, sum, cnt)) in bins {
        if cnt > 0.0 {
            let [s0, s1, s2] = sum;
            pts[idx] = [(s0 / cnt) as f32, (s1 / cnt) as f32, (s2 / cnt) as f32];
            w_scalar[idx] = cnt as f32;
        }
    }
    (pts, w_scalar)
}

#[must_use]
pub fn build_palette(
    uniques: &Uniques,
    w: usize,
    h: usize,
    settings: &ColorSettings,
    over_segmented: bool,
) -> Vec<V3> {
    let (mut pts, mut wts) = bounded_points(uniques);
    let mut keep: Vec<usize> = (0..pts.len()).filter(|&i| wts[i] > 0.0).collect();
    keep.sort_unstable_by(|&a, &b| {
        wts[b]
            .partial_cmp(&wts[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pts = keep.iter().map(|&i| pts[i]).collect();
    wts = keep.iter().map(|&i| wts[i]).collect();

    let cap = if over_segmented {
        oversegment_cap(settings.palette_cap(), pts.len())
    } else {
        settings.palette_cap()
    };
    let k0 = cap.min(pts.len()).max(1);
    let mut pal = kmeans(&pts, &wts, k0, 8);
    let mut cnt0 = vec![0.0_f64; pal.len()];
    for (i, p) in pts.iter().enumerate() {
        cnt0[nearest(*p, &pal)] += f64::from(wts[i]);
    }
    if over_segmented {
        merge_close(&mut pal, &mut cnt0, settings.merge_distance.min(0.004));
    } else {
        merge_close(&mut pal, &mut cnt0, settings.merge_distance);
    }
    prune_ghosts(&mut pal, uniques, w, h);

    // Interior-median refinement is a non-over-segmented tool only: measured
    // on the 1024 px mascot fixtures, running it on the over-segmented
    // palette *doubled* the interior error (avatar rmse_int 2.29 → 3.17) —
    // the label-interior samples of ramp bands mix neighbouring shades, so
    // the medians drift along the ramp. The noise-adaptive re-merge below
    // stays tied to it (both were tuned together for the flat-band path).
    let noise_thr = if !over_segmented && pal.len() > 1 {
        refine_interior_medians(&mut pal, uniques, w, h)
    } else {
        0.0
    };
    if !over_segmented && pal.len() > 1 {
        let eff_merge = settings.merge_distance.max(noise_thr * 2.0);
        let mut cnt_ref = vec![0.0_f64; pal.len()];
        for (i, p) in pts.iter().enumerate() {
            cnt_ref[nearest(*p, &pal)] += f64::from(wts[i]);
        }
        merge_close(&mut pal, &mut cnt_ref, eff_merge);
    }

    let mut cnt = vec![0.0_f64; pal.len()];
    for (i, p) in pts.iter().enumerate() {
        cnt[nearest(*p, &pal)] += f64::from(wts[i]);
    }
    let mut order: Vec<usize> = (0..pal.len()).collect();
    order.sort_by(|&x, &y| {
        cnt[y]
            .partial_cmp(&cnt[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.iter().map(|&i| pal[i]).collect()
}

fn refine_interior_medians(pal: &mut [V3], uniques: &Uniques, w: usize, h: usize) -> f32 {
    if w < 3 || h < 3 || pal.is_empty() {
        return 0.0;
    }
    let lab_u: Vec<usize> = uniques
        .colors
        .iter()
        .map(|c| nearest(rgb_to_oklab(*c), pal))
        .collect();
    let mut ch_samples: Vec<[Vec<u8>; 3]> = vec![[Vec::new(), Vec::new(), Vec::new()]; pal.len()];
    for y in 1..h - 1 {
        let row = y * w;
        for x in 1..w - 1 {
            let i = row + x;
            let uid = uniques.id[i] as usize;
            if uniques.count[uid] <= 0.0 {
                continue;
            }
            let l = lab_u[uid];
            if lab_u[uniques.id[i - 1] as usize] == l
                && lab_u[uniques.id[i + 1] as usize] == l
                && lab_u[uniques.id[i - w] as usize] == l
                && lab_u[uniques.id[i + w] as usize] == l
            {
                let [r, g, b] = uniques.colors[uid];
                ch_samples[l][0].push(r);
                ch_samples[l][1].push(g);
                ch_samples[l][2].push(b);
            }
        }
    }
    let mut noise_samples: Vec<f32> = Vec::new();
    for (k, [sr, sg, sb]) in ch_samples.iter_mut().enumerate() {
        let len = sr.len();
        if len < 16 {
            continue;
        }
        sr.sort_unstable();
        sg.sort_unstable();
        sb.sort_unstable();
        let med = [sr[len / 2], sg[len / 2], sb[len / 2]];
        pal[k] = rgb_to_oklab(med);
        let q25 = [sr[len / 4], sg[len / 4], sb[len / 4]];
        let q75 = [sr[(len * 3) / 4], sg[(len * 3) / 4], sb[(len * 3) / 4]];
        noise_samples.push(d2(rgb_to_oklab(q25), rgb_to_oklab(q75)).sqrt() * 0.5);
    }
    if noise_samples.is_empty() {
        0.0
    } else {
        noise_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        noise_samples[noise_samples.len() / 2]
    }
}

fn prune_ghosts(pal: &mut Vec<V3>, uniques: &Uniques, w: usize, h: usize) {
    for _ in 0..4 {
        if pal.len() < 3 {
            break;
        }
        let lab_u: Vec<u8> = uniques
            .colors
            .iter()
            .map(|c| nearest(rgb_to_oklab(*c), pal) as u8)
            .collect();
        let psrgb: Vec<V3> = pal
            .iter()
            .map(|c| {
                let [r, g, b] = oklab_to_rgb(*c);
                [
                    f32::from(r) / 255.0,
                    f32::from(g) / 255.0,
                    f32::from(b) / 255.0,
                ]
            })
            .collect();
        let lab = |i: usize| lab_u[uniques.id[i] as usize];
        let mut total = vec![0_u32; pal.len()];
        let mut inner = vec![0_u32; pal.len()];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let l = lab(i);
                total[l as usize] += 1;
                let ok = (x == 0 || lab(i - 1) == l)
                    && (x + 1 == w || lab(i + 1) == l)
                    && (y == 0 || lab(i - w) == l)
                    && (y + 1 == h || lab(i + w) == l);
                if ok {
                    inner[l as usize] += 1;
                }
            }
        }
        let mut order: Vec<usize> = (0..pal.len()).collect();
        order.sort_by_key(|&i| total[i]);
        let mut removed = vec![false; pal.len()];
        for &c in &order {
            if total[c] == 0 {
                removed[c] = true;
                continue;
            }
            if (inner[c] as f32) / (total[c] as f32) > 0.03 {
                continue;
            }
            let p = psrgb[c];
            let mut best = f32::MAX;
            for i in 0..pal.len() {
                if i == c || removed[i] {
                    continue;
                }
                for j in i + 1..pal.len() {
                    if j == c || removed[j] {
                        continue;
                    }
                    best = best.min(seg_fit(p, psrgb[i], psrgb[j]).0);
                }
            }
            if best.sqrt() < 0.03 {
                removed[c] = true;
            }
        }
        if !removed.iter().any(|&r| r) {
            break;
        }
        let mut np = Vec::new();
        for (i, p) in pal.iter().enumerate() {
            if !removed[i] {
                np.push(*p);
            }
        }
        *pal = np;
    }
}

#[cfg(test)]
mod tests {
    use crate::settings::{ColorSettings, Colors, Gradients};

    use super::*;

    fn approx(a: V3, b: V3, eps: f32) -> bool {
        d2(a, b) < eps * eps
    }

    /// The over-segmented palette floor: 64 entries for large inputs. The
    /// 1024 px mascot fixtures showed the old floor of 48 quantising soft
    /// shading ramps into 2-3 hard bands (interior rmse 2.9 -> 2.3 and
    /// smaller files at 64), so the default must resolve the higher cap.
    #[test]
    fn oversegment_cap_resolves_to_64_for_default_settings() {
        assert_eq!(
            oversegment_cap(ColorSettings::default().palette_cap(), 3000),
            48
        );
        assert_eq!(oversegment_cap(64, 3000), 64);
        // Tiny inputs stay bounded by the unique-colour budget.
        assert_eq!(
            oversegment_cap(ColorSettings::default().palette_cap(), 30),
            10
        );
        assert_eq!(
            oversegment_cap(ColorSettings::default().palette_cap(), 3),
            4
        );
    }

    /// The over-segmented floor manifests end to end: 196 distinct 6x6
    /// patches give every patch a label-interior (so ghost pruning keeps
    /// them) and a unique budget of 65, binding k0 to the floor.
    #[test]
    fn oversegmented_palette_resolves_to_the_floor() {
        let side = 6usize;
        let grid = 14usize;
        let w = side * grid;
        let h = w;
        let mut px = vec![0_u8; w * h * 4];
        for gy in 0..grid {
            for gx in 0..grid {
                let v = ((gy * grid + gx) * 255 / (grid * grid - 1)) as u8;
                for y in 0..side {
                    for x in 0..side {
                        let i = ((gy * side + y) * w + gx * side + x) * 4;
                        px[i] = v;
                        px[i + 1] = v;
                        px[i + 2] = v;
                        px[i + 3] = 255;
                    }
                }
            }
        }
        let u = unique_colors(&px, w);
        assert!(
            u.colors.len() >= 190,
            "fixture has {} uniques",
            u.colors.len()
        );
        let cfg = ColorSettings {
            gradients: Gradients::LinearRadial,
            ..ColorSettings::default()
        };
        let pal = build_palette(&u, w, h, &cfg, true);
        assert!(
            pal.len() >= 48,
            "over-segmented palette collapsed to {}",
            pal.len()
        );
    }

    /// A noisy near-white background must survive ghost pruning when the
    /// over-segmented cap splits it into several shade entries: each shade's
    /// label-interior ratio collapses (noise scatters neighbouring pixels
    /// across the shade entries) and every shade then looks prunable, which
    /// re-labelled the whole background onto a distant coloured entry and
    /// painted solid colour blocks over the raster (measured on the 1024 px
    /// guide fixture). Entries owning a major share of the image are real
    /// colours and must never be pruned.
    #[test]
    fn noisy_background_survives_ghost_pruning() {
        let (w, h) = (256usize, 256usize);
        let mut px = vec![0_u8; w * h * 4];
        let mut seed = 0x9E37_79B9_u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let dx = x as i32 - 128;
                let dy = y as i32 - 128;
                if dx * dx + dy * dy < 1600 {
                    px[i] = 40;
                    px[i + 1] = 180;
                    px[i + 2] = 190;
                } else {
                    let n = (rnd() % 9) as u8;
                    px[i] = 255 - n;
                    px[i + 1] = 255 - n;
                    px[i + 2] = 255 - n;
                }
                px[i + 3] = 255;
            }
        }
        let u = unique_colors(&px, w);
        let cfg = ColorSettings {
            gradients: Gradients::LinearRadial,
            ..ColorSettings::default()
        };
        let pal = build_palette(&u, w, h, &cfg, true);
        let white = rgb_to_oklab([255, 255, 255]);
        assert!(
            pal.iter().any(|c| d2(*c, white) < 0.04 * 0.04),
            "no near-white palette entry survived: {pal:?}"
        );
    }

    #[test]
    fn palette_picks_the_right_number_of_colors() {
        // 3 distinct flat colours + AA-ish fringe that must be pruned.
        let (w, h) = (32usize, 32usize);
        let mut px = vec![255u8; w * h * 4];
        let mut put = |x0: usize, y0: usize, x1: usize, y1: usize, c: [u8; 3]| {
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * w + x) * 4;
                    px[i] = c[0];
                    px[i + 1] = c[1];
                    px[i + 2] = c[2];
                    px[i + 3] = 255;
                }
            }
        };
        put(0, 0, 16, 16, [200, 30, 30]);
        put(16, 0, 32, 16, [30, 200, 30]);
        put(0, 16, 16, 32, [30, 30, 200]);
        put(16, 16, 32, 32, [255, 255, 255]);
        // 1-px AA band between red and green halves: the actual gamma-sRGB
        // blend of the two, 1 px wide, never a 3×3 interior.
        for x in 0..w {
            let i = (15 * w + x) * 4;
            px[i] = 115;
            px[i + 1] = 115;
            px[i + 2] = 30;
        }
        let u = unique_colors(&px, w);
        let pal = build_palette(&u, w, h, &ColorSettings::default(), false);
        assert!(pal.len() <= 4, "got {}", pal.len());
        // The AA blend band must be pruned as a ghost of red/green...
        let grey = rgb_to_oklab([115, 115, 30]);
        let has_grey = pal.iter().any(|p| approx(*p, grey, 0.05));
        assert!(!has_grey, "AA ghost survived: {pal:?}");
        // ...while the four real regions stay.
        for c in [
            [200_u8, 30, 30],
            [30, 200, 30],
            [30, 30, 200],
            [255, 255, 255],
        ] {
            let lab = rgb_to_oklab(c);
            assert!(
                pal.iter().any(|p| approx(*p, lab, 0.05)),
                "missing {c:?}: {pal:?}"
            );
        }
    }

    #[test]
    fn palette_respects_explicit_cap() {
        let (w, h) = (16usize, 16usize);
        let mut px = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                px[i] = (x * 16) as u8;
                px[i + 1] = (y * 16) as u8;
            }
        }
        let u = unique_colors(&px, w);
        let cfg = ColorSettings {
            max_colors: Colors::Max(3),
            ..ColorSettings::default()
        };
        let pal = build_palette(&u, w, h, &cfg, false);
        assert_eq!(pal.len(), 3);
    }

    #[test]
    fn bounded_points_bins_large_inputs() {
        // > cap unique colours: bins to 5-bit centres, weights accumulate.
        // Deterministic construction: r∈64 values (32 bins), g∈64 (32 bins),
        // b∈9 values (5 bins) → at most 32·32·5 occupied bins.
        let n = UNIQUE_COLOR_CAP + 100;
        let mut colors = Vec::with_capacity(n);
        let mut count = Vec::with_capacity(n);
        for i in 0..n {
            colors.push([
                ((i % 64) * 4) as u8,
                (((i / 64) % 64) * 4) as u8,
                (((i / 4096) % 9) * 4) as u8,
            ]);
            count.push(1.0);
        }
        let u = Uniques {
            colors,
            id: vec![0; 4],
            count,
            width: 2,
            any_alpha: false,
        };
        let (pts, w) = bounded_points(&u);
        assert!(pts.len() < n, "binning must collapse: {} vs {n}", pts.len());
        assert!(pts.len() <= 32 * 32 * 5, "got {}", pts.len());
        assert!(w.iter().all(|&x| x > 0.0));
    }
}
