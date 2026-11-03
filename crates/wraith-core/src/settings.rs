//! Conversion settings handed to every pipeline stage.

use visioncortex::PathSimplifyMode;

/// Colour handling strategy for the tracer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Threshold the raster to pure black/white and trace a single fill colour.
    /// The default: ideal for art on a plain white background, and the fastest
    /// path — the white background never becomes geometry.
    #[default]
    BlackWhite,
    /// Trace every colour cluster, keeping the full palette.
    Color,
}

/// Speed/quality trade-off applied to the tracer's parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Aggressive simplification: fewest nodes, fastest trace, smallest file.
    Fast,
    /// Balanced defaults, tuned for line art and logos.
    #[default]
    Balanced,
    /// Maximum fidelity: more nodes, slower trace, larger file.
    Quality,
}

/// How hard the minifier (OXVG) works on the traced SVG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Minify {
    /// Skip minification entirely; the serializer's output is final.
    Off,
    /// Correctness-first job set that is unlikely to alter rendering.
    Safe,
    /// The full OXVG default job set (collapses groups, moves attributes,
    /// cleans numbers, …). The default and the best size reduction.
    #[default]
    Default,
}

/// Complete configuration for one conversion.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// Colour handling strategy.
    pub mode: Mode,
    /// Speed/quality trade-off for tracing.
    pub quality: Quality,
    /// Optional luminance threshold (0–255) applied before tracing in
    /// [`Mode::BlackWhite`]. `None` defers to the tracer's built-in rule.
    pub threshold: Option<u8>,
    /// In [`Mode::Color`], drop every pure-white path (i.e. wipe the white
    /// background from the geometry). No effect in [`Mode::BlackWhite`],
    /// where white never becomes geometry in the first place.
    pub strip_white: bool,
    /// Minifier aggressiveness.
    pub minify: Minify,
    /// Decimal digits kept for path coordinates (1–8).
    pub precision: u8,
}

impl Settings {
    /// Clamp guard used by both the library and the CLI.
    pub const MIN_PRECISION: u8 = 1;
    /// Upper bound for [`Settings::precision`].
    pub const MAX_PRECISION: u8 = 8;

    /// Settings with sane defaults for white-background line art.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validates and clamps precision into the supported range.
    #[must_use]
    pub fn with_precision(mut self, precision: u8) -> Self {
        self.precision = precision.clamp(Self::MIN_PRECISION, Self::MAX_PRECISION);
        self
    }

    /// Maps this profile onto the tracer's native configuration.
    #[must_use]
    pub fn tracer_config(&self) -> vtracer::Config {
        let mut cfg = match self.quality {
            Quality::Fast => vtracer::Config {
                filter_speckle: 16,
                color_precision: 5,
                layer_difference: 48,
                mode: PathSimplifyMode::Polygon,
                corner_threshold: 90,
                length_threshold: 8.0,
                max_iterations: 4,
                path_precision: Some(1),
                ..vtracer::Config::from_preset(vtracer::Preset::Bw)
            },
            Quality::Balanced => vtracer::Config::from_preset(vtracer::Preset::Bw),
            Quality::Quality => vtracer::Config {
                filter_speckle: 2,
                color_precision: 8,
                layer_difference: 8,
                mode: PathSimplifyMode::Spline,
                corner_threshold: 45,
                length_threshold: 2.0,
                max_iterations: 20,
                path_precision: Some(3),
                ..vtracer::Config::from_preset(vtracer::Preset::Bw)
            },
        };
        cfg.color_mode = match self.mode {
            Mode::BlackWhite => vtracer::ColorMode::Binary,
            Mode::Color => vtracer::ColorMode::Color,
        };
        cfg
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            mode: Mode::default(),
            quality: Quality::default(),
            threshold: None,
            strip_white: false,
            minify: Minify::default(),
            precision: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use vtracer::ColorMode;

    use super::*;

    #[test]
    fn defaults_target_black_and_white_line_art() {
        let s = Settings::new();
        assert_eq!(s.mode, Mode::BlackWhite);
        assert_eq!(s.quality, Quality::Balanced);
        assert_eq!(s.minify, Minify::Default);
        assert_eq!(s.threshold, None);
        assert!(!s.strip_white);
        assert_eq!(s.precision, 2);
    }

    #[test]
    fn precision_is_clamped_into_range() {
        assert_eq!(Settings::new().with_precision(0).precision, 1);
        assert_eq!(Settings::new().with_precision(9).precision, 8);
        assert_eq!(Settings::new().with_precision(5).precision, 5);
    }

    #[test]
    fn tracer_config_follows_mode() {
        assert!(matches!(
            Settings::new().tracer_config().color_mode,
            ColorMode::Binary
        ));
        assert!(matches!(
            Settings {
                mode: Mode::Color,
                ..Settings::new()
            }
            .tracer_config()
            .color_mode,
            ColorMode::Color
        ));
    }

    #[test]
    fn quality_presets_tune_simplification() {
        let fast_cfg = Settings {
            quality: Quality::Fast,
            ..Settings::new()
        }
        .tracer_config();
        let bal_cfg = Settings {
            quality: Quality::Balanced,
            ..Settings::new()
        }
        .tracer_config();
        let hi_cfg = Settings {
            quality: Quality::Quality,
            ..Settings::new()
        }
        .tracer_config();
        assert!(fast_cfg.filter_speckle > bal_cfg.filter_speckle);
        assert!(hi_cfg.filter_speckle < bal_cfg.filter_speckle);
        assert!(fast_cfg.max_iterations < hi_cfg.max_iterations);
        assert_eq!(fast_cfg.path_precision, Some(1));
    }
}
