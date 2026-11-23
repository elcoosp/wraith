//! Render-back quality metrics (roadmap M0/M7; feature `metrics`).
//!
//! Rasterises an SVG with resvg and scores it against a reference raster:
//! RMSE, the share of "bad" pixels (any channel off by more than 40), ΔE in
//! Oklab (mean and max), a lightweight SSIM, and a *seam score* that counts
//! thin 1-px runs of high error along shared layer edges — the artefact
//! trapping exists to prevent.

use crate::error::{Result, WraithError};

/// Quality metrics of a rendered SVG against a reference raster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMetrics {
    /// Root-mean-square error over RGB, composited on white.
    pub rmse: f64,
    /// Share of pixels with any channel off by more than 40, in 0..=1.
    pub bad_frac: f64,
    /// Mean ΔE (Oklab Euclidean) over all pixels.
    pub delta_e_mean: f64,
    /// Maximum ΔE over all pixels.
    pub delta_e_max: f64,
    /// Global SSIM over luma (8×8 windows, flat weights).
    pub ssim: f64,
    /// Share of pixels that sit in a thin (≤2 px) run of high error — the
    /// seam artefact signature. 0 means no seams detected.
    pub seam_frac: f64,
}

/// Rasterises an SVG string at the requested size, composited on white,
/// returning tightly packed RGB bytes.
///
/// # Errors
///
/// Returns [`crate::error::WraithError::Render`] when the SVG cannot be
/// parsed or the pixmap cannot be allocated.
pub fn render_svg(svg: &str, width: u32, height: u32) -> Result<Vec<u8>> {
    let opt = usvg_options();
    let tree =
        resvg::usvg::Tree::from_str(svg, &opt).map_err(|e| WraithError::Render(e.to_string()))?;
    let Some(mut pm) = resvg::tiny_skia::Pixmap::new(width, height) else {
        return Err(WraithError::Render(String::from(
            "pixmap allocation failed",
        )));
    };
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(
            f32::from(width as u16) / tree.size().width(),
            f32::from(height as u16) / tree.size().height(),
        ),
        &mut pm.as_mut(),
    );
    Ok(composite_on_white_rgb(pm.data()))
}

/// Composites premultiplied RGBA over white and returns packed RGB.
#[must_use]
pub fn composite_on_white_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            let inv = 255 - u32::from(p[3]);
            [
                (u32::from(p[0]) + inv).min(255) as u8,
                (u32::from(p[1]) + inv).min(255) as u8,
                (u32::from(p[2]) + inv).min(255) as u8,
            ]
        })
        .collect()
}

/// resvg options: default, explicit for testability.
fn usvg_options() -> resvg::usvg::Options<'static> {
    resvg::usvg::Options::default()
}

/// Compares a rendered output (RGB) against a reference raster (RGBA or RGB,
/// tightly packed) of the same dimensions.
///
/// # Panics
///
/// Panics if the two buffers do not hold the same number of pixels.
#[must_use]
pub fn compare(reference: &[u8], output_rgb: &[u8], channels: usize) -> QualityMetrics {
    assert_eq!(reference.len(), output_rgb.len(), "pixel count mismatch");
    let n = reference.len() / channels;
    let (mut sum2, mut bad) = (0.0_f64, 0_u64);
    let (mut de_sum, mut de_max) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let (r0, r1) = (
            &reference[i * channels..i * channels + 3],
            &output_rgb[i * 3..i * 3 + 3],
        );
        let mut any_bad = false;
        for c in 0..3 {
            let d = f64::from(i16::from(r0[c]) - i16::from(r1[c]));
            sum2 += d * d;
            if d.abs() > 40.0 {
                any_bad = true;
            }
        }
        if any_bad {
            bad += 1;
        }
        let de = delta_e(r0, r1);
        de_sum += de;
        de_max = de_max.max(de);
    }
    let rmse = (sum2 / (3.0 * n as f64)).sqrt();
    let bad_frac = bad as f64 / n as f64;
    let ssim = ssim_luma(reference, output_rgb, channels);
    let seam_frac = seam_score(reference, output_rgb, channels);
    QualityMetrics {
        rmse,
        bad_frac,
        delta_e_mean: de_sum / n as f64,
        delta_e_max: de_max,
        ssim,
        seam_frac,
    }
}

/// sRGB u8 → Oklab lightness-weighted distance (ΔE in Oklab units).
fn delta_e(a: &[u8], b: &[u8]) -> f64 {
    let la = crate::color::palette::rgb_to_oklab([a[0], a[1], a[2]]);
    let lb = crate::color::palette::rgb_to_oklab([b[0], b[1], b[2]]);
    let (dx, dy, dz) = (
        f64::from(la[0] - lb[0]),
        f64::from(la[1] - lb[1]),
        f64::from(la[2] - lb[2]),
    );
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Rec. 601 luma in 0..=255.
fn luma(px: &[u8]) -> f64 {
    0.299 * f64::from(px[0]) + 0.587 * f64::from(px[1]) + 0.114 * f64::from(px[2])
}

/// Flat-window SSIM over luma (8×8 windows, C1/C2 from the standard paper).
fn ssim_luma(a: &[u8], b: &[u8], channels: usize) -> f64 {
    let n = a.len() / channels;
    let w = (n as f64).sqrt().floor() as usize;
    let win = 8_usize;
    if w < win {
        return 1.0;
    }
    let c1 = (0.01 * 255.0) * (0.01 * 255.0);
    let c2 = (0.03 * 255.0) * (0.03 * 255.0);
    let mut total = 0.0_f64;
    let mut count = 0.0_f64;
    for wy in (0..=(w - win)).step_by(win / 2) {
        for wx in (0..=(w - win)).step_by(win / 2) {
            let (mut ma, mut mb, mut maa, mut mbb, mut mab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let mut k = 0.0_f64;
            for y in wy..wy + win {
                for x in wx..wx + win {
                    let i = y * w + x;
                    let la = luma(&a[i * channels..i * channels + 3]);
                    let lb = luma(&b[i * 3..i * 3 + 3]);
                    ma += la;
                    mb += lb;
                    maa += la * la;
                    mbb += lb * lb;
                    mab += la * lb;
                    k += 1.0;
                }
            }
            let (va, vb, cov) = (
                maa / k - (ma / k) * (ma / k),
                mbb / k - (mb / k) * (mb / k),
                mab / k - (ma / k) * (mb / k),
            );
            let s = ((2.0 * (ma / k) * (mb / k) + c1) * (2.0 * cov + c2))
                / (((ma / k) * (ma / k) + (mb / k) * (mb / k) + c1) * (va + vb + c2));
            total += s;
            count += 1.0;
        }
    }
    total / count.max(1.0)
}

/// Seam score: pixels with high error that sit in a thin (≤2 px) run *and*
/// are a colour spike against both clean neighbours — the signature of a
/// hairline gap leaking the under-layer along a shared edge. A boundary-
/// placement offset also produces thin high-error runs, but its band colour
/// equals one neighbour's (the band *is* a shifted copy of that side), so
/// the spike test keeps the metric specific to real gaps.
fn seam_score(a: &[u8], b: &[u8], channels: usize) -> f64 {
    let n = a.len() / channels;
    let w = (n as f64).sqrt().round() as usize;
    if w == 0 {
        return 0.0;
    }
    let h = n / w;
    let err = |i: usize| -> bool {
        (0..3).any(|c| {
            i16::from(a[i * channels + c]) - i16::from(b[i * 3 + c]) > 40
                || i16::from(b[i * 3 + c]) - i16::from(a[i * channels + c]) > 40
        })
    };
    let mut seam_pixels = 0_u64;
    let mut bad_pixels = 0_u64;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !err(i) {
                continue;
            }
            bad_pixels += 1;
            // Thin run: high error here but clean above AND below (or left
            // and right) within 3 px — a line, not a region.
            let v_clean = (0..4).all(|k| {
                let (yu, yd) = (y.checked_sub(k + 1), y + k + 1);
                (yu.is_none_or(|yy| !err(yy * w + x))) && (yd >= h || !err(yd * w + x))
            });
            let h_clean = (0..4).all(|k| {
                let (xl, xr) = (x.checked_sub(k + 1), x + k + 1);
                (xl.is_none_or(|xx| !err(y * w + xx))) && (xr >= w || !err(y * w + xr))
            });
            if !(v_clean || h_clean) {
                continue;
            }
            // Spike test on the clean axis: a trapping gap leaks the
            // *under-layer* colour, which matches neither clean neighbour;
            // a boundary-placement offset blends into one side, so its band
            // colour equals that neighbour's and is not counted.
            let c = [
                f64::from(b[i * 3]),
                f64::from(b[i * 3 + 1]),
                f64::from(b[i * 3 + 2]),
            ];
            let dist = |q: [f64; 3]| -> f64 {
                (0..3).map(|k| (c[k] - q[k]).abs()).fold(0.0_f64, f64::max)
            };
            let nb = |q: Option<usize>| -> Option<[f64; 3]> {
                q.map(|j| {
                    [
                        f64::from(b[j * 3]),
                        f64::from(b[j * 3 + 1]),
                        f64::from(b[j * 3 + 2]),
                    ]
                })
            };
            let (d1, d2) = if v_clean {
                let (u, d) = (
                    y.checked_sub(1).map(|yy| yy * w + x),
                    (y + 1 < h).then(|| (y + 1) * w + x),
                );
                match (nb(u), nb(d)) {
                    (Some(u), Some(d)) => (dist(u), dist(d)),
                    (Some(u), None) => (dist(u), f64::MAX),
                    (None, Some(d)) => (dist(d), f64::MAX),
                    (None, None) => (f64::MAX, f64::MAX),
                }
            } else {
                let (l, r) = (
                    x.checked_sub(1).map(|xx| y * w + xx),
                    (x + 1 < w).then(|| y * w + x + 1),
                );
                match (nb(l), nb(r)) {
                    (Some(l), Some(r)) => (dist(l), dist(r)),
                    (Some(l), None) => (dist(l), f64::MAX),
                    (None, Some(r)) => (dist(r), f64::MAX),
                    (None, None) => (f64::MAX, f64::MAX),
                }
            };
            if d1 > 15.0 && d2 > 15.0 {
                seam_pixels += 1;
            }
        }
    }
    if bad_pixels == 0 {
        return 0.0;
    }
    seam_pixels as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_over_white_matches_hand_computation() {
        // Premultiplied black at 50% alpha over white → 127/128 grey.
        let rgba = [0, 0, 0, 128];
        let rgb = composite_on_white_rgb(&rgba);
        assert!(rgb.iter().all(|&c| (127..=128).contains(&c)));
        // Fully opaque passes through.
        let rgb2 = composite_on_white_rgb(&[10, 20, 30, 255]);
        assert_eq!(rgb2, vec![10, 20, 30]);
    }

    #[test]
    fn identical_rasters_score_perfectly() {
        let a: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 251) as u8).collect();
        let m = compare(&a, &a, 3);
        assert!(m.rmse < 1e-9);
        assert!(m.bad_frac < 1e-9);
        assert!(m.delta_e_mean < 1e-9);
        assert!(m.delta_e_max < 1e-9);
        assert!((m.ssim - 1.0).abs() < 1e-9);
        assert!(m.seam_frac < 1e-9);
    }

    #[test]
    fn seam_score_detects_a_thin_line_not_a_region() {
        // 16×16 reference all white; output has a 1-px red horizontal line.
        let (w, h) = (16usize, 16usize);
        let reference = vec![255_u8; w * h * 3];
        let mut output = reference.clone();
        for x in 0..w {
            let i = (8 * w + x) * 3;
            output[i] = 200;
        }
        let m = compare(&reference, &output, 3);
        assert!(m.seam_frac > 0.0, "thin line must register as a seam");
        // A solid red half-plane is a region, not a seam: score lower.
        let mut output2 = reference.clone();
        for y in 0..h / 2 {
            for x in 0..w {
                let i = (y * w + x) * 3;
                output2[i] = 200;
            }
        }
        let m2 = compare(&reference, &output2, 3);
        assert!(m2.seam_frac < m.seam_frac, "regions are not seams");
    }

    #[test]
    fn metrics_react_to_magnitude_of_error() {
        let a = vec![128_u8; 32 * 32 * 3];
        let b_close = vec![120_u8; 32 * 32 * 3];
        let b_far = vec![250_u8; 32 * 32 * 3];
        let m_close = compare(&a, &b_close, 3);
        let m_far = compare(&a, &b_far, 3);
        assert!(m_close.rmse < m_far.rmse);
        assert!(m_close.bad_frac < f64::EPSILON && (m_far.bad_frac - 1.0).abs() < f64::EPSILON);
        assert!(m_close.delta_e_mean < m_far.delta_e_mean);
    }
}
