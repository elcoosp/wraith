//! Mixture model: per-unique-colour `(a, b, t)` decomposition.
//!
//! The model lives in **gamma-encoded sRGB**, not Oklab: 2-D rasterisers
//! blend anti-aliased edge pixels in gamma-encoded sRGB, so
//! `pixel ≈ (1−t)·A + t·B` is a straight line there and `t` equals the true
//! pixel coverage. Fitting `t` in Oklab distorted edge positions (measured in
//! the prototype: visible wobble; roadmap §4.1).

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::color::palette::{Uniques, V3, d2, oklab_to_rgb, rgb_to_oklab};

/// Mixture decomposition for every unique colour of the raster.
pub struct Mixture {
    /// Dominant palette rank per unique colour.
    pub a: Vec<u8>,
    /// Partner palette rank per unique colour (== `a` for pure colours).
    pub b: Vec<u8>,
    /// Mixing weight of the partner (0 for pure colours).
    pub t: Vec<f32>,
}

/// Squared distance from `p` to the segment `a–b`, plus the projection
/// parameter in 0..=1. Also used by palette ghost pruning.
#[must_use]
pub fn seg_fit(p: V3, a: V3, b: V3) -> (f32, f32) {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let den = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if den < 1e-12 {
        0.0
    } else {
        ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / den).clamp(0.0, 1.0)
    };
    let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    (d2(p, q), t)
}

/// Builds the mixture model with **best-pair** selection (prototype §4.3).
///
/// Evaluates the best (dominant, partner) pair among the four nearest
/// dominant candidates rather than greedy nearest-first matching.
/// Nearest-first alone produced a red fringe on yellow/blue edges because a
/// mixed pixel's nearest centroid can be an unrelated colour.
///
/// Computed per unique colour, not per pixel: cost is O(U·K²) with U small
/// for graphic art (and histogram-capped by the palette stage for photos).
#[must_use]
pub fn build_model(uniques: &Uniques, pal: &[V3]) -> Mixture {
    // Palette colours in the blending space (gamma sRGB).
    let psrgb: Vec<V3> = pal
        .iter()
        .map(|c| {
            let r = oklab_to_rgb(*c);
            [f32::from(r[0]), f32::from(r[1]), f32::from(r[2])]
        })
        .collect();
    let usrgb: Vec<V3> = uniques
        .colors
        .iter()
        .map(|c| [f32::from(c[0]), f32::from(c[1]), f32::from(c[2])])
        .collect();

    let model_fn = |p: V3| -> (u8, u8, f32) {
        let mut near: Vec<(f32, usize)> = psrgb
            .iter()
            .enumerate()
            .map(|(i, q)| (d2(p, *q), i))
            .collect();
        near.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        let na = near[0].1;
        let (mut ba, mut bj, mut bd, mut bt) = (na, na, near[0].0, 0.0_f32);
        for &(_, ca) in near.iter().take(4) {
            // 4 nearest as "dominant" candidates
            for j in 0..psrgb.len() {
                if j == ca {
                    continue;
                }
                let (d, tt) = seg_fit(p, psrgb[ca], psrgb[j]);
                if d + 1e-3 < bd {
                    bd = d;
                    ba = ca;
                    bj = j;
                    bt = tt;
                }
            }
        }
        (ba as u8, bj as u8, if bj == ba { 0.0 } else { bt })
    };

    let fits: Vec<(u8, u8, f32)> = usrgb.par_iter().map(|p| model_fn(*p)).collect();
    let (a, b, t) = fits.into_iter().fold(
        (
            Vec::with_capacity(uniques.colors.len()),
            Vec::new(),
            Vec::new(),
        ),
        |(mut a, mut b, mut t), (aa, bb, tt)| {
            a.push(aa);
            b.push(bb);
            t.push(tt);
            (a, b, t)
        },
    );
    Mixture { a, b, t }
}

/// Labelling of a pixel's dominant palette rank (used by gradient detection
/// and ghost pruning): pure pixels carry their dominant rank, mixed pixels
/// the rank whose weight exceeds 0.5.
#[must_use]
pub fn dominant_rank(u: usize, mix: &Mixture) -> usize {
    if mix.t[u] > 0.5 {
        mix.b[u] as usize
    } else {
        mix.a[u] as usize
    }
}

/// Converts unique colours to Oklab once, for region work.
#[must_use]
pub fn unique_oklab(uniques: &Uniques) -> Vec<V3> {
    uniques.colors.iter().map(|c| rgb_to_oklab(*c)).collect()
}

#[cfg(test)]
mod tests {
    use crate::color::palette::{build_palette, unique_colors};

    use super::*;

    /// A vertical ramp between two flat colours: mixed pixels must decompose
    /// as A↔B with monotonically increasing t across the edge.
    #[test]
    fn mixture_decomposes_edge_pixels_monotonically() {
        // 6 columns: 2 red, 2 blended, 2 blue. A 2-colour palette forces the
        // blend colours onto the red↔blue segment, exactly like AA fringes.
        let (w, h) = (6usize, 4usize);
        let mut px = vec![0u8; w * h * 4];
        let red = [200_u8, 30, 30];
        let blue = [30, 30, 200];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let t = match x {
                    0 | 1 => 0.0_f32,
                    2 => 0.25,
                    3 => 0.5,
                    4 => 0.75,
                    _ => 1.0,
                };
                for c in 0..3 {
                    px[i + c] = (f32::from(red[c]) * (1.0 - t) + f32::from(blue[c]) * t) as u8;
                }
                px[i + 3] = 255;
            }
        }
        let u = unique_colors(&px, w);
        // Default palette: k-means puts every unique colour (including the
        // blends) in the palette, then ghost pruning removes the blends —
        // they lie on the red↔blue segment in gamma sRGB — so the mixture
        // must decompose them via best-pair, exactly like AA fringes.
        let pal = build_palette(&u, w, h, &crate::settings::ColorSettings::default(), false);
        let mix = build_model(&u, &pal);
        // Find the red-ish and blue-ish ranks.
        let red_lab = rgb_to_oklab(red);
        let blue_lab = rgb_to_oklab(blue);
        let mut r_rank = usize::MAX;
        let mut b_rank = usize::MAX;
        for (i, p) in pal.iter().enumerate() {
            if d2(*p, red_lab) < d2(*p, blue_lab) {
                r_rank = r_rank.min(i);
            } else {
                b_rank = b_rank.min(i);
            }
        }
        assert!(r_rank != usize::MAX && b_rank != usize::MAX);
        // Blend columns must be explained as red↔blue pairs with the right t.
        for (x, want_t) in [(2_usize, 0.25_f32), (3, 0.5), (4, 0.75)] {
            let uid = u.id[x] as usize;
            let (a, b, t) = (mix.a[uid], mix.b[uid], mix.t[uid]);
            let pair = [(a as usize, b as usize), (b as usize, a as usize)];
            assert!(
                pair.contains(&(r_rank, b_rank)) || pair.contains(&(b_rank, r_rank)),
                "x={x}: a={a} b={b}"
            );
            let flipped = (a as usize) == b_rank;
            let tt = if flipped { 1.0 - t } else { t };
            assert!((tt - want_t).abs() < 0.15, "x={x}: t={tt} want {want_t}");
        }
    }

    #[test]
    fn seg_fit_distance_and_projection() {
        // Point on the segment → distance 0, t = 0.5.
        let (d, t) = seg_fit([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(d < 1e-12 && (t - 0.5).abs() < 1e-6);
        // Point beyond the end clamps to t = 1.
        let (_, t2) = seg_fit([2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((t2 - 1.0).abs() < 1e-6);
        // Degenerate segment behaves as point distance.
        let (d3, t3) = seg_fit([0.0, 3.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!((t3 - 0.0).abs() < 1e-6 && (d3 - 10.0).abs() < 1e-6); // (1−0)² + (0−3)²
    }

    #[test]
    fn pure_colors_map_to_themselves() {
        let (w, h) = (4usize, 4usize);
        let mut px = vec![255u8; w * h * 4];
        for y in 1..3 {
            for x in 1..3 {
                let i = (y * w + x) * 4;
                px[i] = 10;
                px[i + 1] = 180;
                px[i + 2] = 90;
            }
        }
        let u = unique_colors(&px, w);
        let pal = build_palette(&u, w, h, &crate::settings::ColorSettings::default(), false);
        let mix = build_model(&u, &pal);
        let white_uid = 0; // first-appearance: white corner pixel
        let green_uid = u.id[2 * w + 2] as usize;
        assert_eq!(mix.a[white_uid], mix.b[white_uid]);
        assert!(mix.t[white_uid].abs() < 1e-9);
        assert_eq!(mix.a[green_uid], mix.b[green_uid]);
        assert!(mix.t[green_uid].abs() < 1e-9);
    }
}
