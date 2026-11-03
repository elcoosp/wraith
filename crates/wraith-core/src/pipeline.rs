//! The pipeline: decode → preprocess → trace → serialize → minify.

use crate::error::Result;
use crate::preprocess::prepare;
use crate::settings::Settings;
use crate::stats::{ConvertStats, Stopwatch};
use crate::svgout::serialize;
use crate::trace::trace;

/// A finished conversion: final SVG plus accounting.
#[derive(Debug, Clone)]
pub struct Output {
    /// Final SVG document (minification level applied as configured).
    pub svg: String,
    /// Byte counts and stage timings.
    pub stats: ConvertStats,
}

/// Converts encoded PNG bytes into an optimised SVG document.
///
/// # Errors
///
/// Propagates [`crate::error::WraithError`] from any failing stage; see the
/// stage modules for specifics.
pub fn convert(png: &[u8], settings: &Settings) -> Result<Output> {
    let mut sw = Stopwatch::default();
    let prepared = prepare(png, settings, &mut sw)?;
    let traced = trace(prepared.image, settings, &mut sw)?;
    let canonical = serialize(
        &traced,
        settings.strip_white,
        Some(u32::from(settings.precision)),
        &mut sw,
    );
    let svg = match settings.minify {
        crate::settings::Minify::Off => canonical,
        _ => crate::minify::minify(&canonical, settings.minify, &mut sw)?,
    };

    let (width, height) = (traced.width, traced.height);
    let stats = ConvertStats {
        width: u32::try_from(width).unwrap_or(u32::MAX),
        height: u32::try_from(height).unwrap_or(u32::MAX),
        input_bytes: png.len() as u64,
        output_bytes: svg.len() as u64,
        timings: sw.laps().to_vec(),
    };
    Ok(Output { svg, stats })
}

#[cfg(test)]
mod tests {
    use crate::preprocess::fixture_png;
    use crate::settings::{Minify, Mode, Quality, Settings};

    use super::*;

    #[test]
    fn end_to_end_produces_svg_with_stats() {
        let png = fixture_png(64);
        let out = convert(&png, &Settings::new()).expect("converts");
        assert!(out.svg.starts_with("<svg"));
        assert!(out.svg.contains("</svg>"));
        assert_eq!(out.stats.width, 64);
        assert_eq!(out.stats.height, 64);
        assert_eq!(out.stats.input_bytes, png.len() as u64);
        assert_eq!(out.stats.output_bytes, out.svg.len() as u64);
        // Five timed stages: decode, preprocess, trace, serialize, minify.
        assert_eq!(out.stats.timings.len(), 5);
        assert!(out.stats.total().as_nanos() > 0);
    }

    #[test]
    fn minify_off_keeps_canonical_output() {
        let png = fixture_png(64);
        let cfg = Settings {
            minify: Minify::Off,
            ..Settings::new()
        };
        let out = convert(&png, &cfg).expect("converts");
        assert!(out.svg.starts_with("<svg"));
        assert_eq!(out.stats.timings.len(), 4);
    }

    #[test]
    fn fast_quality_produces_valid_smaller_or_equal_output_than_quality() {
        let png = fixture_png(128);
        let fast = convert(
            &png,
            &Settings {
                quality: Quality::Fast,
                ..Settings::new()
            },
        )
        .expect("converts");
        let hi = convert(
            &png,
            &Settings {
                quality: Quality::Quality,
                ..Settings::new()
            },
        )
        .expect("converts");
        assert!(fast.svg.contains("<svg"));
        assert!(hi.svg.contains("<svg"));
        assert!(fast.svg.len() <= hi.svg.len());
    }

    #[test]
    fn color_mode_with_strip_white_removes_background() {
        let png = fixture_png(64);
        let cfg = Settings {
            mode: Mode::Color,
            strip_white: true,
            ..Settings::new()
        };
        let out = convert(&png, &cfg).expect("converts");
        assert!(!out.svg.contains("fill=\"#FFFFFF\""));
    }

    #[test]
    fn threshold_flows_through_to_trace() {
        let png = fixture_png(64);
        let cfg = Settings {
            threshold: Some(200),
            ..Settings::new()
        };
        let out = convert(&png, &cfg).expect("converts");
        // Geometry survives; fill may be rewritten by the minifier (#000000 →
        // #000), so only assert on the shape of the document.
        assert!(out.svg.contains("<path"));
        assert_eq!(out.stats.timings.len(), 5);
    }

    #[test]
    fn invalid_png_propagates_decode_error() {
        let err = convert(b"junk", &Settings::new());
        assert!(matches!(err, Err(crate::error::WraithError::Decode(_))));
    }
}
