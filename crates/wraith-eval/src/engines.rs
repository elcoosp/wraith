//! Engines under evaluation: the legacy vtracer path and the geometric
//! colour engine, both driven through `wraith_core`'s public API. Timing
//! covers the *trace stage only* (decode/minify excluded), matching the
//! prototype's measurement protocol (min of N runs).

use std::time::Instant;

use wraith_core::settings::{
    Engine, LegacyColorSettings, LegacyHierarchy, Minify, Mode, Quality, Settings,
};
use wraith_core::{Stopwatch, color};

/// One engine configuration under test.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Report key.
    pub key: &'static str,
    /// Settings used for the run.
    pub settings: Settings,
}

/// The evaluation matrix (prototype §5): the two legacy vtracer baselines and
/// the geometric engine (default knobs, and once without gradients for the
/// M3 comparison).
#[must_use]
pub fn engines() -> Vec<EngineConfig> {
    let balanced = Settings {
        mode: Mode::Color,
        engine: Engine::Legacy,
        legacy_color: LegacyColorSettings::default(),
        ..Settings::new()
    };
    let quality = Settings {
        mode: Mode::Color,
        engine: Engine::Legacy,
        quality: Quality::Quality,
        legacy_color: LegacyColorSettings {
            color_precision: 8,
            layer_difference: 8,
            hierarchy: LegacyHierarchy::Stacked,
        },
        ..Settings::new()
    };
    let geometric = Settings {
        mode: Mode::Color,
        engine: Engine::Geometric,
        ..Settings::new()
    };
    let geometric_no_grad = Settings {
        mode: Mode::Color,
        engine: Engine::Geometric,
        color: wraith_core::ColorSettings {
            gradients: wraith_core::Gradients::Off,
            ..wraith_core::ColorSettings::default()
        },
        ..Settings::new()
    };
    vec![
        EngineConfig {
            key: "legacy_balanced",
            settings: balanced,
        },
        EngineConfig {
            key: "legacy_quality",
            settings: quality,
        },
        EngineConfig {
            key: "geometric",
            settings: geometric,
        },
        EngineConfig {
            key: "geometric_nograd",
            settings: geometric_nograd_key(&geometric_no_grad),
        },
    ]
}

fn geometric_nograd_key(s: &Settings) -> Settings {
    *s
}

/// Run result of one engine over one input raster.
pub struct RunOutcome {
    /// Final SVG (minified according to the config).
    pub svg: String,
    /// Trace-stage wall time in milliseconds.
    pub ms: f64,
    /// Geometry element count.
    pub paths: usize,
    /// Palette size (0 for the legacy engine).
    pub palette_size: usize,
}

/// Runs one engine config over RGBA pixels.
///
/// # Errors
///
/// Propagates pipeline errors.
pub fn run_engine(
    cfg: &EngineConfig,
    rgba: &[u8],
    w: usize,
    h: usize,
) -> wraith_core::Result<RunOutcome> {
    let img = vtracer::ColorImage {
        pixels: rgba.to_vec(),
        width: w,
        height: h,
    };
    let uses_geometric =
        cfg.settings.mode == Mode::Color && cfg.settings.engine == Engine::Geometric;
    let mut sw = Stopwatch::default();
    let t0 = Instant::now();
    let (canonical, paths, palette_size) = if uses_geometric {
        let traced =
            color::trace_color(&img, &cfg.settings.color, cfg.settings.background, &mut sw);
        let svg = color::write::write_svg(&traced);
        let paths: usize = traced.layers.iter().map(|l| l.shapes.len()).sum();
        (svg, paths, traced.palette_size)
    } else {
        let traced = wraith_core::trace::trace(img, &cfg.settings, &mut sw)?;
        let paths = traced.paths.len();
        let svg = wraith_core::svgout::serialize(
            &traced,
            Some(u32::from(cfg.settings.precision)),
            &mut sw,
        );
        (svg, paths, 0)
    };
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let svg = match cfg.settings.minify {
        Minify::Off => canonical,
        _ => {
            wraith_core::minify::minify(&canonical, cfg.settings.minify, &mut Stopwatch::default())?
        }
    };
    Ok(RunOutcome {
        svg,
        ms,
        paths,
        palette_size,
    })
}

/// Encodes RGBA bytes as a PNG (fixtures are persisted as PNG inputs).
///
/// # Panics
///
/// Panics if `rgba` does not hold exactly `w × h` pixels (caller invariant).
///
/// # Errors
///
/// Propagates encode errors (cannot happen for in-memory buffers).
pub fn encode_png(rgba: &[u8], w: u32, h: u32) -> image::ImageResult<Vec<u8>> {
    let mut buf = Vec::new();
    let img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(w, h, rgba.to_vec()).expect("buffer matches dimensions");
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(buf)
}
