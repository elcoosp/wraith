//! The pipeline: decode → preprocess → trace → serialize → minify.
//!
//! Engine dispatch (roadmap M7): `Mode::BlackWhite` always uses the legacy
//! vtracer tracer; `Mode::Color` defaults to the geometric engine
//! ([`crate::color`]) with the legacy path available via [`Engine::Legacy`].

use crate::color;
use crate::error::Result;
use crate::preprocess::prepare;
use crate::settings::{Engine, Mode, Settings};
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
    let use_legacy = settings.mode == Mode::BlackWhite || settings.engine == Engine::Legacy;
    let (dims, canonical, palette_size, path_count) = if use_legacy {
        let traced = trace(prepared.image, settings, &mut sw)?;
        let dims = (traced.width, traced.height);
        let canonical = serialize(&traced, Some(u32::from(settings.precision)), &mut sw);
        let paths = canonical.matches("<path").count();
        (dims, canonical, 0, paths)
    } else {
        let traced = color::trace_color(
            &prepared.image,
            &settings.color,
            settings.background,
            &mut sw,
        );
        let dims = (traced.width, traced.height);
        let canonical = color::write::write_svg(&traced);
        let paths: usize = traced.layers.iter().map(|l| l.shapes.len()).sum();
        (dims, canonical, traced.palette_size, paths)
    };
    let svg = match settings.minify {
        crate::settings::Minify::Off => canonical.clone(),
        _ => crate::minify::minify(&canonical, settings.minify, &mut sw)?,
    };

    let stats = ConvertStats {
        width: u32::try_from(dims.0).unwrap_or(u32::MAX),
        height: u32::try_from(dims.1).unwrap_or(u32::MAX),
        input_bytes: png.len() as u64,
        output_bytes: svg.len() as u64,
        timings: sw.laps().to_vec(),
        palette_size,
        path_count,
    };
    Ok(Output { svg, stats })
}

#[cfg(test)]
mod tests {
    use crate::preprocess::fixture_png;
    use crate::settings::{Engine, Gradients, Minify, Mode, Quality, Settings};

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
        // Timed stages: decode, preprocess, trace, serialize, minify.
        assert_eq!(out.stats.timings.len(), 5);
        assert!(out.stats.total().as_nanos() > 0);
        assert_eq!(out.stats.palette_size, 0); // legacy engine
        assert!(out.stats.path_count > 0);
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
    fn color_mode_geometric_engine_is_the_default() {
        let png = fixture_png(64);
        let cfg = Settings {
            mode: Mode::Color,
            ..Settings::new()
        };
        let out = convert(&png, &cfg).expect("converts");
        assert!(out.svg.contains("<svg"));
        assert!(out.stats.palette_size > 0);
        assert!(out.stats.path_count > 0);
        // Stage list includes Palette.
        assert!(
            out.stats
                .timings
                .iter()
                .any(|(s, _)| *s == crate::stats::Stage::Palette)
        );
    }

    #[test]
    fn color_mode_legacy_engine_still_available() {
        let png = fixture_png(64);
        let cfg = Settings {
            mode: Mode::Color,
            engine: Engine::Legacy,
            ..Settings::new()
        };
        let out = convert(&png, &cfg).expect("converts");
        assert!(out.svg.contains("<svg"));
        assert_eq!(out.stats.palette_size, 0);
        assert!(
            !out.stats
                .timings
                .iter()
                .any(|(s, _)| *s == crate::stats::Stage::Palette)
        );
    }

    #[test]
    fn geometric_color_mode_renders_shapes_with_gradients_capable_defaults() {
        let png = fixture_png(64);
        let cfg = Settings {
            mode: Mode::Color,
            color: crate::settings::ColorSettings {
                gradients: Gradients::LinearRadial,
                ..crate::settings::ColorSettings::default()
            },
            ..Settings::new()
        };
        let out = convert(&png, &cfg).expect("converts");
        assert!(out.svg.contains("<path") || out.svg.contains("<rect"));
        assert!(out.svg.contains("</svg>"));
    }

    #[test]
    fn threshold_flows_through_to_trace() {
        let png = fixture_png(64);
        let cfg = Settings {
            threshold: Some(200),
            ..Settings::new()
        };
        let out = convert(&png, &cfg).expect("converts");
        assert!(out.svg.contains("<path"));
        assert_eq!(out.stats.timings.len(), 5);
    }

    #[test]
    fn invalid_png_propagates_decode_error() {
        let err = convert(b"junk", &Settings::new());
        assert!(matches!(err, Err(crate::error::WraithError::Decode(_))));
    }
}
