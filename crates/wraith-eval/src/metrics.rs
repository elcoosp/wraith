//! Metrics of the eval harness: render-back scoring against ground truth.
//!
//! RMSE, bad-pixel share, ΔE in Oklab (mean/max), a flat-window SSIM, and the
//! seam score (thin 1-px runs of high error along shared layer edges — the
//! artefact trapping exists to prevent). Most heavy lifting is shared with
//! `wraith_core::metrics`; this module adds the ground-truth plumbing.

use wraith_core::metrics::{compare, composite_on_white_rgb};

/// Full metric set for one engine on one fixture.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Scores {
    /// Wall-clock time of the trace stage, milliseconds (min of N runs).
    pub ms: f64,
    /// Number of geometry elements.
    pub paths: usize,
    /// Unminified SVG bytes.
    pub bytes_raw: usize,
    /// Bytes after the OXVG default pass.
    pub bytes_min: usize,
    /// RMSE at 1× against ground truth.
    pub rmse_1x: f64,
    /// RMSE at 4× against ground truth (sub-pixel fidelity).
    pub rmse_4x: f64,
    /// Bad-pixel share at 4× (any channel off by > 40).
    pub bad_4x: f64,
    /// Mean ΔE (Oklab) at 4×.
    pub delta_e_mean_4x: f64,
    /// Max ΔE (Oklab) at 4×.
    pub delta_e_max_4x: f64,
    /// SSIM at 1×.
    pub ssim_1x: f64,
    /// Seam score at 8× (thin high-error runs along shared edges).
    pub seam_8x: f64,
}

/// Renders `svg` at `size × scale` and returns packed RGB composited on white.
///
/// # Errors
///
/// Propagates [`wraith_core::WraithError::Render`].
pub fn render_rgb(svg: &str, size: u32, scale: u32) -> wraith_core::Result<Vec<u8>> {
    let rgba_pm = render_rgba_premultiplied(svg, size * scale)?;
    Ok(composite_on_white_rgb(&rgba_pm))
}

/// Renders an SVG to a premultiplied RGBA pixmap of exactly `dim × dim`.
///
/// # Errors
///
/// Propagates render errors from [`wraith_core::metrics::render_svg`] — the
/// pixel dimensions are enforced by re-rendering at the exact size.
pub fn render_rgba_premultiplied(svg: &str, dim: u32) -> wraith_core::Result<Vec<u8>> {
    // render_svg returns composited RGB; we need the RGBA buffer, so render
    // through resvg directly with the same options.
    let opt = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_str(svg, &opt)
        .map_err(|e| wraith_core::WraithError::Render(e.to_string()))?;
    let Some(mut pm) = resvg::tiny_skia::Pixmap::new(dim, dim) else {
        return Err(wraith_core::WraithError::Render(String::from(
            "pixmap alloc failed",
        )));
    };
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(
            f32::from(dim as u16) / tree.size().width(),
            f32::from(dim as u16) / tree.size().height(),
        ),
        &mut pm.as_mut(),
    );
    Ok(pm.data().to_vec())
}

/// Scores one output SVG against its ground-truth SVG at the working size.
///
/// # Errors
///
/// Propagates render errors.
pub fn score(
    gt_svg: &str,
    out_svg: &str,
    size: u32,
    ms: f64,
    paths: usize,
    bytes_raw: usize,
    bytes_min: usize,
) -> wraith_core::Result<Scores> {
    let gt1 = render_rgb(gt_svg, size, 1)?;
    let gt4 = render_rgb(gt_svg, size, 4)?;
    let out1 = render_rgb(out_svg, size, 1)?;
    let out4 = render_rgb(out_svg, size, 4)?;
    let out8 = render_rgb(out_svg, size, 8)?;
    let gt8 = render_rgb(gt_svg, size, 8)?;
    let m1 = compare(&gt1, &out1, 3);
    let m4 = compare(&gt4, &out4, 3);
    let m8 = compare(&gt8, &out8, 3);
    Ok(Scores {
        ms,
        paths,
        bytes_raw,
        bytes_min,
        rmse_1x: m1.rmse,
        rmse_4x: m4.rmse,
        bad_4x: m4.bad_frac * 100.0,
        delta_e_mean_4x: m4.delta_e_mean,
        delta_e_max_4x: m4.delta_e_max,
        ssim_1x: m1.ssim,
        seam_8x: m8.seam_frac * 100.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_svgs_score_perfectly() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><rect width="32" height="32" fill="#f00"/></svg>"##;
        let s = score(svg, svg, 32, 1.0, 1, 100, 90).expect("scores");
        assert!(s.rmse_1x < 1e-9 && s.rmse_4x < 1e-9);
        assert!(s.seam_8x < 1e-9);
        assert!((s.ssim_1x - 1.0).abs() < 1e-9);
        assert_eq!(s.paths, 1);
    }

    #[test]
    fn render_rgb_is_composited_and_sized() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" viewBox="0 0 8 8"><rect width="8" height="8" fill="#00f"/></svg>"##;
        let rgb = render_rgb(svg, 8, 2).expect("renders");
        assert_eq!(rgb.len(), 16 * 16 * 3);
        // Blue channel dominates everywhere.
        assert!(rgb.chunks_exact(3).all(|p| p[2] > 200));
    }

    #[test]
    fn broken_svg_is_a_render_error() {
        let err = render_rgb("<svg xmlns=", 8, 1);
        assert!(err.is_err());
    }
}
