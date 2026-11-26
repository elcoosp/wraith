//! # wraith-core
//!
//! The conversion pipeline behind [`wraith`](https://github.com/wraith-rs/wraith):
//! PNG bytes in, optimised SVG out, stage-by-stage timings along the way.
//!
//! ```no_run
//! use wraith_core::{convert, Settings};
//!
//! let png: Vec<u8> = std::fs::read("input.png").expect("read png");
//! let out = convert(&png, &Settings::new()).expect("convert");
//! std::fs::write("output.svg", out.svg).expect("write svg");
//! println!("saved {:.1}%", out.stats.reduction() * 100.0);
//! ```
//!
//! Pipeline stages, in order:
//! 1. [`preprocess`] — decode PNG, background removal, optional luminance
//!    threshold,
//! 2. [`trace`] (legacy vtracer) or [`color`] (geometric sub-pixel engine),
//! 3. [`svgout`] — canonical serialization with fill hoisting,
//! 4. [`minify`] — OXVG job set.
//!
//! With the `metrics` feature, [`metrics`] adds render-back quality scoring
//! (RMSE, ΔE Oklab, SSIM, seam detection) used by `wraith-eval` and the CLI.

pub mod color;
pub mod error;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod minify;
pub mod pipeline;
pub mod preprocess;
pub mod settings;
pub mod stats;
pub mod svgout;
pub mod trace;

pub use error::{Result, WraithError};
pub use pipeline::{Output, convert};
pub use settings::{
    Background, ColorSettings, Colors, ContentPreset, DenoiseSettings, Engine, Gradients,
    LegacyColorSettings, LineworkPolicy,
    LegacyHierarchy, Minify, Mode, Quality, Settings,
};
pub use stats::{ConvertStats, Stage, Stopwatch};

#[cfg(feature = "metrics")]
pub use metrics::{QualityMetrics, compare, render_svg};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_api_reexports_resolve() {
        let s = Settings::new();
        let _ = Engine::Geometric;
        let _ = Background::None;
        let _ = Colors::Auto;
        let _ = Gradients::LinearRadial;
        let _ = ContentPreset::Logo;
        let _ = LegacyColorSettings::default();
        let _ = LegacyHierarchy::Stacked;
        let _ = ColorSettings::default();
        assert_eq!(s.precision, 2);
    }
}
