//! Conversion settings handed to every pipeline stage.
//!
//! v0.2.0 redesign (roadmap M7): the colour path now has two engines —
//! [`Engine::Legacy`] (the vtracer cluster tracer) and [`Engine::Geometric`]
//! (the sub-pixel coverage engine, [`crate::color`]). `Mode::Color` defaults
//! to the geometric engine; black-and-white mode always uses the legacy
//! tracer because thresholded line art is its exact specialty.
//!
//! The breaking changes versus 0.1.x:
//! * [`Settings::strip_white`] is gone — [`Background`] replaces the
//!   fill-colour-equality hack with alpha-based removal (roadmap §2.2.2).
//! * Colour quality knobs are no longer bolted onto the black-and-white
//!   preset: [`Settings::legacy_color`] carries them explicitly (M0).
//! * The geometric engine's knobs live in [`ColorSettings`].

use visioncortex::PathSimplifyMode;

/// Colour handling strategy for the tracer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Threshold the raster to pure black/white and trace a single fill colour.
    /// The default: ideal for art on a plain white background, and the fastest
    /// path — the white background never becomes geometry.
    #[default]
    BlackWhite,
    /// Trace every colour region, keeping the full palette.
    Color,
}

/// Tracing engine. `Mode::Color` defaults to [`Engine::Geometric`]; the
/// legacy vtracer path stays available for byte-compatible 0.1.x output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Engine {
    /// The vtracer cluster tracer (0.1.x behaviour for both modes).
    Legacy,
    /// Sub-pixel geometric colour engine (default for [`Mode::Color`]).
    #[default]
    Geometric,
}

/// Speed/quality trade-off applied to the *legacy* tracer's parameters.
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

/// Cluster stacking mode of the *legacy* colour tracer, decoupled from the
/// black-and-white preset (roadmap M0: colour settings must be independently
/// exposed rather than inherited from `Preset::Bw`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LegacyHierarchy {
    /// Later clusters paint over earlier ones (0.1.x behaviour).
    #[default]
    Stacked,
    /// Cutout: overlapping cluster geometry is subtracted from earlier ones.
    Cutout,
}

/// Explicit colour-mode knobs for [`Engine::Legacy`]. Defaults reproduce the
/// 0.1.x `Quality::Balanced` colour behaviour byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegacyColorSettings {
    /// Bits of RGB kept when clustering (1–8): higher keeps more colours.
    pub color_precision: i32,
    /// Raw-RGB L1 distance below which clusters are merged (0–256).
    pub layer_difference: i32,
    /// Cluster stacking mode.
    pub hierarchy: LegacyHierarchy,
}

impl Default for LegacyColorSettings {
    fn default() -> Self {
        Self {
            color_precision: 6,
            layer_difference: 16,
            hierarchy: LegacyHierarchy::Stacked,
        }
    }
}

/// Palette size for the geometric engine. [`Colors::Auto`] keeps the tuned
/// 32-colour default; [`Colors::Max`] caps the palette explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Colors {
    /// Heuristic default (32 colours) — right for logos, icons and flat art.
    #[default]
    Auto,
    /// Hard palette cap. Values are clamped to 1..=256.
    Max(u8),
}

/// Gradient detection/fitting for the geometric engine (roadmap M3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gradients {
    /// Flat bands only; smooth ramps stay as stacked colour layers.
    Off,
    /// Detect and fit linear gradients.
    Linear,
    /// Detect and fit linear and radial gradients (default: radial costs one
    /// extra Gauss–Newton pass only on regions that survived the linear fit).
    #[default]
    LinearRadial,
}

/// Background removal policy, applied before tracing (roadmap M0/M7).
///
/// This replaces 0.1.x's `strip_white`, which dropped paths whose fill was
/// *exactly* `#FFFFFF` — a colour-equality hack that deleted legitimate white
/// shapes and kept near-white AA clusters. Removal now happens on the raster:
/// only border-connected regions are candidates, and matching is by alpha or
/// by colour tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Background {
    /// Keep everything. An image without alpha gets its dominant colour
    /// rendered as a full-bleed `<rect>` (byte-cheap and rendering-identical).
    #[default]
    None,
    /// Treat the PNG alpha channel as authoritative: fully transparent pixels
    /// never become geometry and no synthetic background rect is emitted.
    Alpha,
    /// Flood-remove border-connected regions within `tolerance` (per-channel
    /// max distance) of `rgb`, turning them transparent, then behave like
    /// [`Background::Alpha`].
    Color {
        /// Background colour to remove.
        rgb: [u8; 3],
        /// Per-channel tolerance (0–255).
        tolerance: u8,
    },
}

/// Content preset for the geometric engine (CLI `--preset`). Each preset is a
/// thin override layer on top of [`ColorSettings::default`]; explicit flags
/// win over preset values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentPreset {
    /// Logos and word marks: tuned defaults, gradients on, primitives on.
    #[default]
    Logo,
    /// Small icons: slightly looser tolerance, smaller speck floor.
    Icon,
    /// Dense illustration: larger palette for detail-heavy art.
    Illustration,
}

/// Knobs of the geometric colour engine (roadmap M7 sketch). Every value has
/// a documented default; [`Settings::with_color_preset`] applies a
/// [`ContentPreset`] on top.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorSettings {
    /// Palette size cap.
    pub max_colors: Colors,
    /// Oklab merge threshold: palette colours closer than this are merged.
    pub merge_distance: f32,
    /// Curve-fit tolerance in pixels (Schneider error bound).
    pub tolerance_px: f32,
    /// Minimum turn (degrees) recognised as a corner.
    pub corner_angle_deg: f32,
    /// Trapping underlay width in pixels (0 disables trapping).
    pub trap_px: f32,
    /// Gradient detection/fitting level.
    pub gradients: Gradients,
    /// Emit `<circle>`/`<ellipse>`/`<rect>`/arc primitives when they fit.
    pub primitives: bool,
    /// Field-driven refinement iterations (0 = off, roadmap M5).
    pub refine_iters: u8,
    /// Binomial contour-smoothing rounds applied before curve fitting (0 = off, default 2).
    pub smooth_iters: u8,
    /// Speck filter: rings below this signed area (px²) are dropped.
    pub min_area_px2: f32,
    /// Run the 2-pass bilateral denoise before palette construction.
    /// Default off: it blends low-contrast facet edges and makes traced
    /// edges 2-3x wobblier than the source (measured). Palette refinement and
    /// small-component absorption already handle noise.
    pub prefilter: bool,
    /// Binomial blur passes on each coverage field before contouring
    /// (0 = off, default 1). Removes sub-pixel contour noise without moving
    /// straight edges.
    pub field_blur: u8,
    /// Decompose edge pixels using only colours present in their 5x5
    /// neighbourhood (default on) instead of the global best-pair search.
    pub local_mixture: bool,
    /// Emit midtone bridge bands along wide (≥ ~1.3 px) anti-aliased
    /// transitions between high-contrast regions (default on). A sharp vector
    /// boundary loses the quarter-step pixels on both sides of a soft ramp;
    /// the bridge repaints the transition with the ramp's mean colour,
    /// roughly halving the boundary error. Crisp 1-px edges are skipped by
    /// the width gate.
    pub edge_bands: bool,
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            max_colors: Colors::Auto,
            merge_distance: 0.02,
            tolerance_px: 0.3,
            corner_angle_deg: 40.0,
            trap_px: 1.0,
            gradients: Gradients::LinearRadial,
            primitives: true,
            refine_iters: 0,
            smooth_iters: 2,
            min_area_px2: 2.0,
            prefilter: false,
            field_blur: 1,
            local_mixture: true,
            edge_bands: true,
        }
    }
}

impl ColorSettings {
    /// Applies a [`ContentPreset`] over the current values; fields the preset
    /// does not care about are left untouched.
    #[must_use]
    pub const fn with_preset(mut self, preset: ContentPreset) -> Self {
        match preset {
            ContentPreset::Logo => {}
            ContentPreset::Icon => {
                self.tolerance_px = 0.35;
                self.min_area_px2 = 4.0;
                self.gradients = Gradients::Linear;
            }
            ContentPreset::Illustration => {
                self.max_colors = Colors::Max(64);
            }
        }
        self
    }

    /// Clamps every knob into its documented range. Library and CLI both call
    /// this, so downstream code can rely on invariants instead of re-checking.
    #[must_use]
    pub fn clamped(mut self) -> Self {
        self.merge_distance = self.merge_distance.clamp(0.0, 0.2);
        self.tolerance_px = self.tolerance_px.clamp(0.05, 2.0);
        self.corner_angle_deg = self.corner_angle_deg.clamp(5.0, 85.0);
        self.trap_px = self.trap_px.clamp(0.0, 3.0);
        self.min_area_px2 = self.min_area_px2.clamp(0.0, 1_000.0);
        self.field_blur = self.field_blur.min(3);
        self.smooth_iters = self.smooth_iters.min(64);
        if let Colors::Max(n) = &mut self.max_colors {
            *n = (*n).clamp(1, 64);
        }
        self
    }

    /// Effective palette cap: [`Colors::Auto`] resolves to 32.
    #[must_use]
    pub fn palette_cap(&self) -> usize {
        match self.max_colors {
            Colors::Auto => 32,
            Colors::Max(n) => usize::from(n),
        }
    }

    /// Trapping dilation radius in pixels (0 when trapping is disabled).
    #[must_use]
    pub fn trap_radius(&self) -> usize {
        if self.trap_px <= 0.0 {
            0
        } else {
            f32::round(self.trap_px) as usize
        }
    }

    /// Corner threshold in radians, derived from [`Self::corner_angle_deg`].
    #[must_use]
    pub fn corner_angle_rad(&self) -> f64 {
        f64::from(self.corner_angle_deg).to_radians()
    }
}

/// Complete configuration for one conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Colour handling strategy.
    pub mode: Mode,
    /// Tracing engine ([`Engine::Geometric`] is the default for colour mode).
    pub engine: Engine,
    /// Geometric colour engine knobs (ignored by [`Engine::Legacy`]).
    pub color: ColorSettings,
    /// Speed/quality trade-off for the legacy tracer.
    pub quality: Quality,
    /// Optional luminance threshold (0–255) applied before tracing in
    /// [`Mode::BlackWhite`]. `None` defers to the tracer's built-in rule.
    pub threshold: Option<u8>,
    /// Explicit legacy colour knobs (used by [`Engine::Legacy`] in
    /// [`Mode::Color`]; black-and-white mode keeps the preset-derived values).
    pub legacy_color: LegacyColorSettings,
    /// Background removal policy.
    pub background: Background,
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

    /// Applies a [`ContentPreset`] to the geometric colour knobs and clamps
    /// everything into range.
    #[must_use]
    pub fn with_color_preset(mut self, preset: ContentPreset) -> Self {
        self.color = self.color.with_preset(preset).clamped();
        self
    }

    /// Clamps every setting into its documented range.
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            color: self.color.clamped(),
            precision: self
                .precision
                .clamp(Self::MIN_PRECISION, Self::MAX_PRECISION),
            ..self
        }
    }

    /// Maps this profile onto the *legacy* tracer's native configuration.
    ///
    /// Black-and-white mode keeps the preset-derived 0.1.x values. Colour
    /// mode reads [`Settings::legacy_color`] instead of inheriting from the
    /// black-and-white preset (roadmap M0), with the quality tier supplying
    /// the simplification params exactly as 0.1.x did.
    #[must_use]
    pub fn tracer_config(&self) -> vtracer::Config {
        let mut cfg = match self.mode {
            Mode::BlackWhite => self.bw_config(),
            Mode::Color => self.color_config(),
        };
        cfg.color_mode = match self.mode {
            Mode::BlackWhite => vtracer::ColorMode::Binary,
            Mode::Color => vtracer::ColorMode::Color,
        };
        cfg
    }

    /// Black-and-white tracer config: the 0.1.x quality presets, unchanged.
    fn bw_config(&self) -> vtracer::Config {
        match self.quality {
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
        }
    }

    /// Colour tracer config for [`Engine::Legacy`].
    ///
    /// Values reproduce 0.1.x behaviour exactly (asserted by tests): the
    /// quality tier overrides the simplification knobs, and the remaining
    /// colour fields come from [`Settings::legacy_color`] rather than from
    /// the black-and-white preset.
    const fn color_config(&self) -> vtracer::Config {
        let (mode, corner, length, iters, precision) = match self.quality {
            Quality::Fast => (
                PathSimplifyMode::Polygon,
                90_i32,
                8.0_f64,
                4_usize,
                Some(1_u32),
            ),
            Quality::Balanced => (
                PathSimplifyMode::Spline,
                60_i32,
                4.0_f64,
                10_usize,
                Some(2_u32),
            ),
            Quality::Quality => (
                PathSimplifyMode::Spline,
                45_i32,
                2.0_f64,
                20_usize,
                Some(3_u32),
            ),
        };
        // Allow explicit --corner-angle to tune legacy corner detection while
        // preserving exact 0.1.x quality defaults when corner_angle_deg == 40.0.
        let corner_thresh = if (self.color.corner_angle_deg - 40.0).abs() > f32::EPSILON {
            self.color.corner_angle_deg.round().clamp(1.0, 180.0) as i32
        } else {
            corner
        };
        vtracer::Config {
            color_mode: vtracer::ColorMode::Color,
            hierarchical: match self.legacy_color.hierarchy {
                LegacyHierarchy::Stacked => vtracer::Hierarchical::Stacked,
                LegacyHierarchy::Cutout => vtracer::Hierarchical::Cutout,
            },
            filter_speckle: 4,
            color_precision: self.legacy_color.color_precision,
            layer_difference: self.legacy_color.layer_difference,
            mode,
            corner_threshold: corner_thresh,
            length_threshold: length,
            max_iterations: iters,
            splice_threshold: 45,
            path_precision: precision,
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            mode: Mode::default(),
            engine: Engine::default(),
            color: ColorSettings::default(),
            quality: Quality::default(),
            threshold: None,
            legacy_color: LegacyColorSettings::default(),
            background: Background::default(),
            minify: Minify::default(),
            precision: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use visioncortex::PathSimplifyMode;
    use vtracer::{ColorMode, Hierarchical};

    use super::*;

    #[test]
    fn defaults_target_black_and_white_line_art() {
        let s = Settings::new();
        assert_eq!(s.mode, Mode::BlackWhite);
        assert_eq!(s.engine, Engine::Geometric);
        assert_eq!(s.quality, Quality::Balanced);
        assert_eq!(s.minify, Minify::Default);
        assert_eq!(s.threshold, None);
        assert_eq!(s.background, Background::None);
        assert_eq!(s.precision, 2);
        assert_eq!(s.color, ColorSettings::default());
    }

    #[test]
    fn geometric_color_settings_have_documented_defaults() {
        let c = ColorSettings::default();
        assert_eq!(c.max_colors, Colors::Auto);
        assert!((c.merge_distance - 0.02).abs() < 1e-6);
        assert!((c.tolerance_px - 0.3).abs() < 1e-6);
        assert!((c.corner_angle_deg - 40.0).abs() < 1e-6);
        assert!((c.trap_px - 1.0).abs() < 1e-6);
        assert_eq!(c.gradients, Gradients::LinearRadial);
        assert!(c.primitives);
        assert_eq!(c.refine_iters, 0);
        assert!((c.min_area_px2 - 2.0).abs() < 1e-6);
        assert_eq!(c.palette_cap(), 32);
        assert_eq!(c.trap_radius(), 1);
    }

    #[test]
    fn precision_is_clamped_into_range() {
        assert_eq!(Settings::new().with_precision(0).precision, 1);
        assert_eq!(Settings::new().with_precision(9).precision, 8);
        assert_eq!(Settings::new().with_precision(5).precision, 5);
    }

    #[test]
    fn color_settings_clamp_into_documented_ranges() {
        let c = ColorSettings {
            merge_distance: -1.0,
            tolerance_px: 99.0,
            corner_angle_deg: 0.0,
            trap_px: 100.0,
            min_area_px2: -5.0,
            max_colors: Colors::Max(255),
            ..ColorSettings::default()
        }
        .clamped();
        assert!((c.merge_distance - 0.0).abs() < 1e-6);
        assert!((c.tolerance_px - 2.0).abs() < 1e-6);
        assert!((c.corner_angle_deg - 5.0).abs() < 1e-6);
        assert_eq!(c.trap_radius(), 3);
        assert!((c.min_area_px2 - 0.0).abs() < 1e-6);
        assert_eq!(c.palette_cap(), 64);
    }

    #[test]
    fn content_presets_override_the_expected_knobs() {
        let icon = ColorSettings::default().with_preset(ContentPreset::Icon);
        assert!((icon.tolerance_px - 0.35).abs() < 1e-6);
        assert!((icon.min_area_px2 - 4.0).abs() < 1e-6);
        assert_eq!(icon.gradients, Gradients::Linear);
        let illus = ColorSettings::default().with_preset(ContentPreset::Illustration);
        assert_eq!(illus.max_colors, Colors::Max(64));
        let logo = ColorSettings::default().with_preset(ContentPreset::Logo);
        assert_eq!(logo, ColorSettings::default());
    }

    #[test]
    fn tracer_config_follows_mode() {
        assert!(matches!(
            Settings::new().tracer_config().color_mode,
            ColorMode::Binary
        ));
        let color_cfg = Settings {
            mode: Mode::Color,
            ..Settings::new()
        }
        .tracer_config();
        assert!(matches!(color_cfg.color_mode, ColorMode::Color));
    }

    /// M0 gate: colour mode must no longer inherit from the BW preset. The
    /// explicit defaults must reproduce the 0.1.x *balanced* colour output
    /// byte-for-byte (this pins the decoupling against silent drift).
    #[test]
    fn legacy_color_mode_is_decoupled_but_behaviour_identical_to_0_1_x() {
        let cfg = Settings {
            mode: Mode::Color,
            ..Settings::new()
        }
        .tracer_config();
        // 0.1.x: Balanced derived every field from `Preset::Bw` and only set
        // `color_mode`. The Bw preset values are asserted here so the
        // explicit `color_config()` copy cannot drift from them.
        let bw = vtracer::Config::from_preset(vtracer::Preset::Bw);
        assert_eq!(cfg.filter_speckle, bw.filter_speckle);
        assert_eq!(cfg.color_precision, bw.color_precision);
        assert_eq!(cfg.layer_difference, bw.layer_difference);
        assert!(matches!(cfg.mode, PathSimplifyMode::Spline));
        assert_eq!(cfg.corner_threshold, bw.corner_threshold);
        assert!((cfg.length_threshold - bw.length_threshold).abs() < 1e-9);
        assert_eq!(cfg.max_iterations, bw.max_iterations);
        assert_eq!(cfg.splice_threshold, bw.splice_threshold);
        assert_eq!(cfg.path_precision, bw.path_precision);
        assert!(matches!(cfg.hierarchical, Hierarchical::Stacked));
    }

    #[test]
    fn legacy_color_knobs_flow_through() {
        let cfg = Settings {
            mode: Mode::Color,
            legacy_color: LegacyColorSettings {
                color_precision: 8,
                layer_difference: 32,
                hierarchy: LegacyHierarchy::Cutout,
            },
            ..Settings::new()
        }
        .tracer_config();
        assert_eq!(cfg.color_precision, 8);
        assert_eq!(cfg.layer_difference, 32);
        assert!(matches!(cfg.hierarchical, Hierarchical::Cutout));
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

    #[test]
    fn bw_mode_keeps_0_1_x_preset_values() {
        let cfg = Settings::new().tracer_config();
        let bw = vtracer::Config::from_preset(vtracer::Preset::Bw);
        assert_eq!(cfg.filter_speckle, bw.filter_speckle);
        assert_eq!(cfg.corner_threshold, bw.corner_threshold);
        assert_eq!(cfg.splice_threshold, bw.splice_threshold);
    }
}
