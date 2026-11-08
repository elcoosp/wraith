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

/// Merges palette colours closer than `thr` (Oklab), weighted by pixel count.
fn merge_close(pal: &mut Vec<V3>, cnt: &mut Vec<f64>, thr: f32) {
    loop {
        let mut best = None;
        let mut bd = thr * thr;
        for i in 0..pal.len() {
            for j in i + 1..pal.len() {
                let d = d2(pal[i], pal[j]);
                if d < bd {
                    bd = d;
                    best = Some((i, j));
                }
            }
        }
        let Some((i, j)) = best else {
            break;
        };
        let (wi, wj) = (cnt[i].max(1e-9), cnt[j].max(1e-9));
        for c in 0..3 {
            pal[i][c] =
                ((f64::from(pal[i][c]) * wi + f64::from(pal[j][c]) * wj) / (wi + wj)) as f32;
        }
        cnt[i] += cnt[j];
        pal.swap_remove(j);
        cnt.swap_remove(j);
    }
}

/// Histogram cap for the palette input: unique colours beyond this are binned
/// to a 5-bit-per-channel histogram before clustering (M2: bounds the k-means
/// input for photo-like inputs without a heavy quantizer dependency).
const UNIQUE_COLOR_CAP: usize = 32_768;

/// Bins the unique colours to 5-bit RGB when their number exceeds the cap.
/// Returns cluster points and their accumulated pixel counts.
fn bounded_points(uniques: &Uniques) -> (Vec<V3>, Vec<f32>) {
    let ulab: Vec<V3> = uniques.colors.iter().map(|c| rgb_to_oklab(*c)).collect();
    if uniques.colors.len() <= UNIQUE_COLOR_CAP {
        return (ulab, uniques.count.clone());
    }
    let mut bins: HashMap<u32, (usize, f32), BuildHasherDefault<Fx>> = FxMap::default();
    let mut order: Vec<u32> = Vec::new();
    for (u, c) in uniques.colors.iter().enumerate() {
        let key = (u32::from(c[0] >> 3) << 10) | (u32::from(c[1] >> 3) << 5) | u32::from(c[2] >> 3);
        let e = bins.entry(key).or_insert_with(|| {
            order.push(key);
            (order.len() - 1, 0.0)
        });
        e.1 += uniques.count[u];
    }
    let mut pts = vec![[0.0_f32; 3]; order.len()];
    let mut w = vec![0.0_f32; order.len()];
    for (key, (idx, cnt)) in bins {
        let c = [
            (((key >> 10) & 0x1f) << 3) as u8 + 4,
            (((key >> 5) & 0x1f) << 3) as u8 + 4,
            ((key & 0x1f) << 3) as u8 + 4,
        ];
        pts[idx] = rgb_to_oklab(c);
        w[idx] = cnt;
    }
    (pts, w)
}

/// Builds the paint palette: k-means → merge → AA-ghost pruning → rank by
/// pixel count (rank 0 = largest).
///
/// Ghost pruning (prototype §4.2): a palette colour is dropped when (a) fewer
/// than 3% of its pixels form a 3×3 same-label interior — i.e. it is almost
/// exclusively an anti-aliasing fringe — and (b) it lies within `ΔOklab` 0.03
/// of the segment between two other palette colours, i.e. it is explained as
/// a blend of its neighbours. A real flat 3×3 region survives; a 1-px AA band
/// does not. The rule is safe in *over-segmented* mode too: ramp bands are
/// wide enough to hold 3×3 interiors, so only true AA ghosts die and the
/// ramp survives as a chain of adjacent bands.
///
/// When `over_segmented` is true (gradient detection on, roadmap M3 step 1),
/// the cap is raised to ≥ 48 and the merge threshold is tightened so ramp
/// steps survive; the merge/prune stages would otherwise collapse them.
#[must_use]
pub fn build_palette(
    uniques: &Uniques,
    w: usize,
    h: usize,
    settings: &ColorSettings,
    over_segmented: bool,
) -> Vec<V3> {
    let (mut pts, mut wts) = bounded_points(uniques);
    // Drop zero-weight points (fully transparent-only colours) so they cannot
    // become seeds.
    let mut keep: Vec<usize> = (0..pts.len()).filter(|&i| wts[i] > 0.0).collect();
    keep.sort_unstable_by(|&a, &b| {
        wts[b]
            .partial_cmp(&wts[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let pts_kept: Vec<V3> = keep.iter().map(|&i| pts[i]).collect();
    let wts_kept: Vec<f32> = keep.iter().map(|&i| wts[i]).collect();
    pts = pts_kept;
    wts = wts_kept;

    // Over-segmentation raises the cap, but never beyond a third of the
    // distinct colours: one centroid per shade would make every band 1 px
    // wide (no 3×3 interior), and ghost pruning would then — correctly, by
    // its own rule — treat the whole ramp as anti-aliasing. Real fixtures:
    // 932 uniques → cap stays 48 (bands of ~19 shades); a 64 px icon with
    // 100 uniques gets 33 bands of ~3 shades instead of a degenerate ramp.
    let cap = if over_segmented {
        settings.palette_cap().max(48).min((pts.len() / 3).max(4))
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
        // Collapse near-identical colours (duplicate centroids) without
        // eating ramp steps: the k-means spread over a quantised ramp puts
        // some centroids within ΔOklab 0.004 of each other, while real
        // neighbouring ramp steps sit ≥ 0.008 apart.
        merge_close(&mut pal, &mut cnt0, settings.merge_distance.min(0.004));
    } else {
        merge_close(&mut pal, &mut cnt0, settings.merge_distance);
    }
    // Anti-aliasing ghost pruning rounds (both modes; see doc comment).
    if uniques.colors.len() <= UNIQUE_COLOR_CAP {
        prune_ghosts(&mut pal, uniques, w, h);
    }

    // Rank by pixel count (descending, stable).
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

/// Rounds of AA-ghost pruning over the full-resolution labelling.
///
/// Rule (b) — "explained as a mixture of two other colours" — is evaluated in
/// **gamma sRGB**, because that is the space rasterisers blend in: an sRGB
/// blend lies on the straight segment between its endpoints there, while in
/// Oklab a mid-blend deviates by up to ~0.1 (the same insight that places the
/// mixture model in sRGB).
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
        // Normalised to 0..1 so the blend tolerance below is scale-free.
        let psrgb: Vec<V3> = pal
            .iter()
            .map(|c| {
                let r = oklab_to_rgb(*c);
                [
                    f32::from(r[0]) / 255.0,
                    f32::from(r[1]) / 255.0,
                    f32::from(r[2]) / 255.0,
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
            if total[c] == 0 || (inner[c] as f32) / (total[c] as f32) > 0.03 {
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
    use crate::settings::{ColorSettings, Colors};

    use super::*;

    fn approx(a: V3, b: V3, eps: f32) -> bool {
        d2(a, b) < eps * eps
    }

    #[test]
    fn oklab_round_trip_stays_within_one_bit() {
        // Round-tripping every channel extreme and a mid ramp through
        // Oklab must not drift more than 1/255 per channel.
        for r in [0_u8, 64, 128, 200, 255] {
            for g in [0_u8, 37, 128, 240, 255] {
                for b in [0_u8, 90, 128, 255] {
                    let back = oklab_to_rgb(rgb_to_oklab([r, g, b]));
                    assert!(
                        (i16::from(back[0]) - i16::from(r)).abs() <= 1
                            && (i16::from(back[1]) - i16::from(g)).abs() <= 1
                            && (i16::from(back[2]) - i16::from(b)).abs() <= 1,
                        "{r} {g} {b} -> {back:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn oklab_known_reference_values() {
        // White → L = 1.0, a = b = 0; black → all zero.
        let white = rgb_to_oklab([255, 255, 255]);
        assert!((white[0] - 1.0).abs() < 1e-4);
        assert!(white[1].abs() < 1e-4 && white[2].abs() < 1e-4);
        let black = rgb_to_oklab([0, 0, 0]);
        assert!(approx(black, [0.0, 0.0, 0.0], 1e-6));
        // Pure red has positive a* (redness) in Oklab.
        let red = rgb_to_oklab([255, 0, 0]);
        assert!(red[1] > 0.15);
    }

    #[test]
    fn unique_colors_counts_and_ids() {
        let px = vec![
            1, 2, 3, 255, 1, 2, 3, 255, 9, 9, 9, 255, 1, 2, 3, 0, // alpha-0: no count
        ];
        let u = unique_colors(&px, 2);
        assert_eq!(u.colors.len(), 2);
        assert!((u.count[0] - 2.0).abs() < 1e-9); // two opaque pixels of colour 0
        assert!((u.count[1] - 1.0).abs() < 1e-9);
        assert_eq!(u.id, vec![0, 0, 1, 0]);
        assert!(u.any_alpha);
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

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn dbg_ghost_pruning() {
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
        for x in 0..w {
            let i = (15 * w + x) * 4;
            px[i] = 115;
            px[i + 1] = 115;
            px[i + 2] = 30;
        }
        let u = unique_colors(&px, w);
        println!("uniques: {}", u.colors.len());
        for (i, c) in u.colors.iter().enumerate() {
            println!("  u{i}: {c:?} count={}", u.count[i]);
        }
        let pal = build_palette(&u, w, h, &ColorSettings::default(), false);
        println!("palette len: {}", pal.len());
        for c in pal {
            println!("  {:?}", oklab_to_rgb(c));
        }
    }
}
