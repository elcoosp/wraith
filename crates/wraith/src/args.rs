//! Command-line argument model.
//!
//! The v0.2.0 surface (roadmap M7): engine selection, the geometric colour
//! knobs, gradient/refinement switches and alpha-based background removal.
//! `--strip-white` is gone — `--background` replaces it (roadmap §2.2.2).

use std::str::FromStr;

use clap::{Parser, Subcommand, ValueEnum};

/// Shell targets for generated completions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Shell {
    /// POSIX bash.
    Bash,
    /// Z shell.
    Zsh,
    /// Friendly interactive shell.
    Fish,
    // Elvish and PowerShell are supported by clap_complete too; exposing the
    // four most common keeps the help table tight.
}

/// Colour handling mirrored from [`wraith_core::Mode`] for clap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum ModeArg {
    /// Threshold to pure black/white; white background never becomes
    /// geometry. Best for art on a plain white background.
    #[default]
    Bw,
    /// Trace every colour region and keep the palette (geometric engine by
    /// default).
    Color,
}

/// Tracing engine mirrored from [`wraith_core::Engine`] for clap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum EngineArg {
    /// Sub-pixel geometric colour engine (default for `--mode color`).
    #[default]
    Geometric,
    /// The vtracer cluster tracer: 0.1.x behaviour, byte-compatible output.
    Legacy,
}

/// Speed/quality trade-off mirrored from [`wraith_core::Quality`] for clap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum QualityArg {
    /// Fewest nodes, fastest trace, smallest file.
    Fast,
    /// Balanced defaults, tuned for line art and logos.
    #[default]
    Balanced,
    /// Maximum fidelity, slower and larger.
    Quality,
}

/// Minifier aggressiveness mirrored from [`wraith_core::Minify`] for clap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum MinifyArg {
    /// Full OXVG default job set (best reduction).
    #[default]
    Default,
    /// Correctness-first job set, unlikely to alter rendering.
    Safe,
    /// Skip minification.
    Off,
}

/// Content preset mirrored from [`wraith_core::ContentPreset`] for clap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum PresetArg {
    /// Logos and word marks: tuned defaults, gradients on, primitives on.
    #[default]
    Logo,
    /// Small icons: slightly looser tolerance, smaller speck floor.
    Icon,
    /// Dense illustration: larger palette for detail-heavy art.
    Illustration,
}

/// Gradient fitting level mirrored from [`wraith_core::Gradients`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum GradientsArg {
    /// Flat bands only; smooth ramps stay as stacked colour layers.
    Off,
    /// Detect and fit linear gradients.
    Linear,
    /// Detect and fit linear and radial gradients (default).
    #[default]
    LinearRadial,
}

/// Legacy cluster stacking mode mirrored from
/// [`wraith_core::LegacyHierarchy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum HierarchyArg {
    /// Later clusters paint over earlier ones (0.1.x behaviour).
    #[default]
    Stacked,
    /// Cutout: overlapping cluster geometry is subtracted from earlier ones.
    Cutout,
}

/// Palette cap: `auto` (heuristic default) or an explicit count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorsArg {
    /// Heuristic default (32 colours).
    #[default]
    Auto,
    /// Hard palette cap (clamped to 1–64 downstream).
    Max(u8),
}

impl FromStr for ColorsArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let n: u8 = s
            .parse()
            .map_err(|_| format!("invalid palette size `{s}`: expected `auto` or 1–64"))?;
        if n == 0 {
            return Err(format!(
                "invalid palette size `{s}`: expected `auto` or 1–64"
            ));
        }
        Ok(Self::Max(n))
    }
}

/// Background removal policy: `none`, `alpha`, or an RGB hex colour such as
/// `#ffffff` / `ffffff` (border-connected matching regions become
/// transparent, with `--bg-tolerance` slack).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackgroundArg {
    /// Keep everything.
    #[default]
    None,
    /// Alpha channel is authoritative.
    Alpha,
    /// Remove a border-connected colour background.
    Color([u8; 3]),
}

impl FromStr for BackgroundArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "none" => return Ok(Self::None),
            "alpha" => return Ok(Self::Alpha),
            _ => {}
        }
        let hex = lower.strip_prefix('#').unwrap_or(&lower);
        if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "invalid background `{s}`: expected `none`, `alpha` or #RRGGBB"
            ));
        }
        let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0);
        Ok(Self::Color([byte(0), byte(2), byte(4)]))
    }
}

/// Optional subcommands.
#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Generate shell completions on stdout.
    Completions {
        /// Target shell.
        #[arg(value_enum)]
        shell: Shell,
    },
}

/// Raster in. Wraith out.
///
/// Blazing-fast PNG → optimised SVG with a sub-pixel geometric colour engine.
// Booleans are the natural type for independent CLI flags; grouping them
// would only hurt the help output (workspace lint: `struct_excessive_bools`).
#[derive(Debug, Clone, Parser)]
#[command(
    name = "wraith",
    version,
    about,
    disable_help_subcommand = true,
    infer_subcommands = true,
    after_help = concat!(
        "Exit codes: 0 success, 1 conversion failure, 2 usage error\n\n",
        "Examples:\n",
        "  wraith logo.png              logo.svg next to it\n",
        "  wraith -m color logo.png     sub-pixel geometric colour trace\n",
        "  wraith -m color --engine legacy a.png   0.1.x vtracer output\n",
        "  wraith -m color --preset icon *.png     icon-tuned colour trace\n",
        "  wraith --background alpha a.png         respect the alpha channel\n",
        "  wraith *.png -o dist/        every PNG into dist/\n",
        "  cat logo.png | wraith -o -   SVG on stdout\n",
        "  wraith a.png --json          machine-readable stats on stderr"
    )
)]
pub struct Cli {
    /// PNG file(s), directory (scanned for *.png), or `-` for stdin.
    /// Reads stdin when omitted.
    #[arg(value_name = "INPUT")]
    pub inputs: Vec<String>,

    /// Output file (one input), directory (many inputs), or `-` for stdout.
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,

    /// Colour handling for the tracer.
    #[arg(short = 'm', long, value_enum, default_value_t)]
    pub mode: ModeArg,

    /// Shorthand for `-m color` (roadmap M7).
    #[arg(long)]
    pub color: bool,

    /// Tracing engine for colour mode: `geometric` (default, sub-pixel
    /// coverage fields) or `legacy` (vtracer, 0.1.x-compatible).
    #[arg(long, value_enum, default_value_t)]
    pub engine: EngineArg,

    /// Content preset for the geometric engine: `logo`, `icon` or
    /// `illustration`. Explicit knob flags override preset values.
    #[arg(long, value_enum, default_value_t)]
    pub preset: PresetArg,

    /// Tracer speed/quality trade-off (legacy engine knobs).
    #[arg(short = 'Q', long, value_enum, default_value_t)]
    pub quality: QualityArg,

    /// Luminance threshold (0–255) applied before tracing in `--mode bw`.
    #[arg(short = 't', long, value_parser = clap::value_parser!(u8))]
    pub threshold: Option<u8>,

    /// Geometric engine palette cap: `auto` or a count 1–64.
    #[arg(long, value_name = "N|auto", default_value = "auto")]
    pub colors: ColorsArg,

    /// Geometric engine curve-fit tolerance in pixels (default 0.3).
    #[arg(long, value_name = "PX")]
    pub tolerance: Option<f32>,

    /// Geometric engine corner turn threshold in degrees (default 40).
    #[arg(long, value_name = "DEG")]
    pub corner_angle: Option<f32>,

    /// Trapping underlay width in pixels; 0 disables trapping (default 1).
    #[arg(long, value_name = "PX")]
    pub trap: Option<f32>,

    /// Oklab palette merge threshold (default 0.02).
    #[arg(long, value_name = "D")]
    pub merge: Option<f32>,

    /// Speck filter: minimum ring area in px² (default 2).
    #[arg(long, value_name = "PX2")]
    pub min_area: Option<f32>,

    /// Enable the bilateral pre-denoise (off by default; softens edges).
    #[arg(long)]
    pub prefilter: bool,

    /// Coverage-field blur passes before contouring, 0-3 (default 1).
    #[arg(long, value_name = "N")]
    pub field_blur: Option<u8>,

    /// Use the global best-pair mixture instead of the 5x5-local one.
    #[arg(long)]
    pub no_local_mixture: bool,

    /// Gradient detection/fitting: `off`, `linear` or `linear-radial`.
    #[arg(long, value_enum)]
    pub gradients: Option<GradientsArg>,

    /// Disable `<circle>`/`<ellipse>`/`<rect>` primitive emission.
    #[arg(long)]
    pub no_primitives: bool,

    /// Field-driven refinement iterations for the geometric engine
    /// (0 = off, default 0).
    #[arg(
        long,
        default_value_t = 0,
        value_parser = clap::value_parser!(u8).range(..=30)
    )]
    pub refine: u8,

    /// Binomial contour-smoothing iterations before curve fitting (0 = off, default 2).
    #[arg(
        long,
        default_value_t = 2,
        value_parser = clap::value_parser!(u8).range(..=64)
    )]
    pub smooth: u8,

    /// Background removal: `none`, `alpha`, or a hex colour `#RRGGBB`.
    /// Hex backgrounds flood-remove border-connected matching regions
    /// (replaces 0.1.x `--strip-white`).
    #[arg(long, value_name = "none|alpha|#RRGGBB", default_value = "none")]
    pub background: BackgroundArg,

    /// Per-channel tolerance for `--background #RRGGBB` (0–255, default 8).
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u8))]
    pub bg_tolerance: u8,

    /// Legacy engine: colour precision bits kept when clustering (1–8).
    #[arg(long)]
    pub color_precision: Option<i32>,

    /// Legacy engine: raw-RGB cluster merge distance (0–256).
    #[arg(long)]
    pub layer_difference: Option<i32>,

    /// Legacy engine: cluster stacking mode.
    #[arg(long, value_enum)]
    pub hierarchy: Option<HierarchyArg>,

    /// Minifier job set.
    #[arg(long, value_enum, default_value_t)]
    pub minify: MinifyArg,

    /// Decimal digits kept for path coordinates (1–8).
    #[arg(
        short = 'p',
        long,
        default_value_t = 2,
        value_parser = clap::value_parser!(u8).range(1..=8)
    )]
    pub precision: u8,

    /// Parallel conversions for batch input (0 = all cores).
    #[arg(short = 'j', long, default_value_t = 0, value_name = "N")]
    pub jobs: usize,

    /// Print per-file byte counts and stage timings to stderr.
    #[arg(long)]
    pub stats: bool,

    /// Print machine-readable JSON stats to stderr (implies `--stats`).
    #[arg(long)]
    pub json: bool,

    /// Print nothing but errors.
    #[arg(short, long)]
    pub quiet: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

impl Cli {
    /// Converts CLI-facing enums into pipeline settings: the preset is
    /// applied first, then every explicitly passed knob flag wins over it.
    #[must_use]
    pub fn settings(&self) -> wraith_core::Settings {
        let mut s = wraith_core::Settings {
            mode: match self.mode {
                ModeArg::Bw => wraith_core::Mode::BlackWhite,
                ModeArg::Color => wraith_core::Mode::Color,
            },
            engine: match self.engine {
                EngineArg::Geometric => wraith_core::Engine::Geometric,
                EngineArg::Legacy => wraith_core::Engine::Legacy,
            },
            quality: match self.quality {
                QualityArg::Fast => wraith_core::Quality::Fast,
                QualityArg::Balanced => wraith_core::Quality::Balanced,
                QualityArg::Quality => wraith_core::Quality::Quality,
            },
            threshold: self.threshold,
            minify: match self.minify {
                MinifyArg::Default => wraith_core::Minify::Default,
                MinifyArg::Safe => wraith_core::Minify::Safe,
                MinifyArg::Off => wraith_core::Minify::Off,
            },
            precision: self.precision,
            ..wraith_core::Settings::default()
        };
        if self.color {
            s.mode = wraith_core::Mode::Color;
        }
        // Preset first, explicit flags second.
        s = s.with_color_preset(match self.preset {
            PresetArg::Logo => wraith_core::ContentPreset::Logo,
            PresetArg::Icon => wraith_core::ContentPreset::Icon,
            PresetArg::Illustration => wraith_core::ContentPreset::Illustration,
        });
        s.color.max_colors = match self.colors {
            ColorsArg::Auto => wraith_core::Colors::Auto,
            ColorsArg::Max(n) => wraith_core::Colors::Max(n),
        };
        if let Some(v) = self.tolerance {
            s.color.tolerance_px = v;
        }
        if let Some(v) = self.corner_angle {
            s.color.corner_angle_deg = v;
        }
        if let Some(v) = self.trap {
            s.color.trap_px = v;
        }
        if let Some(v) = self.merge {
            s.color.merge_distance = v;
        }
        if let Some(v) = self.min_area {
            s.color.min_area_px2 = v;
        }
        s.color.prefilter = self.prefilter;
        if let Some(v) = self.field_blur {
            s.color.field_blur = v;
        }
        if self.no_local_mixture {
            s.color.local_mixture = false;
        }
        if let Some(g) = self.gradients {
            s.color.gradients = match g {
                GradientsArg::Off => wraith_core::Gradients::Off,
                GradientsArg::Linear => wraith_core::Gradients::Linear,
                GradientsArg::LinearRadial => wraith_core::Gradients::LinearRadial,
            };
        }
        s.color.primitives = !self.no_primitives;
        s.color.refine_iters = self.refine;
        s.color.smooth_iters = self.smooth;
        s.legacy_color = wraith_core::LegacyColorSettings {
            color_precision: self
                .color_precision
                .unwrap_or(wraith_core::LegacyColorSettings::default().color_precision),
            layer_difference: self
                .layer_difference
                .unwrap_or(wraith_core::LegacyColorSettings::default().layer_difference),
            hierarchy: match self.hierarchy.unwrap_or_default() {
                HierarchyArg::Stacked => wraith_core::LegacyHierarchy::Stacked,
                HierarchyArg::Cutout => wraith_core::LegacyHierarchy::Cutout,
            },
        };
        s.background = match self.background {
            BackgroundArg::None => wraith_core::Background::None,
            BackgroundArg::Alpha => wraith_core::Background::Alpha,
            BackgroundArg::Color(rgb) => wraith_core::Background::Color {
                rgb,
                tolerance: self.bg_tolerance,
            },
        };
        s.clamped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("wraith").chain(args.iter().copied())).expect("parses")
    }

    #[test]
    fn defaults_are_black_white_geometric_balanced_minified() {
        let cli = parse(&["a.png"]);
        assert_eq!(cli.mode, ModeArg::Bw);
        assert_eq!(cli.engine, EngineArg::Geometric);
        assert_eq!(cli.preset, PresetArg::Logo);
        assert_eq!(cli.quality, QualityArg::Balanced);
        assert_eq!(cli.minify, MinifyArg::Default);
        assert_eq!(cli.precision, 2);
        assert_eq!(cli.jobs, 0);
        assert_eq!(cli.refine, 0);
        assert!(!cli.no_primitives);
        assert!(!cli.stats && !cli.json && !cli.quiet);
        assert!(cli.threshold.is_none());
        assert_eq!(cli.colors, ColorsArg::Auto);
        assert_eq!(cli.background, BackgroundArg::None);
    }

    #[test]
    fn settings_map_onto_core() {
        let cli = parse(&[
            "a.png",
            "-m",
            "color",
            "--engine",
            "legacy",
            "-Q",
            "fast",
            "--minify",
            "safe",
            "-t",
            "200",
            "--color-precision",
            "8",
            "--layer-difference",
            "32",
            "--hierarchy",
            "cutout",
            "-p",
            "5",
        ]);
        let s = cli.settings();
        assert_eq!(s.mode, wraith_core::Mode::Color);
        assert_eq!(s.engine, wraith_core::Engine::Legacy);
        assert_eq!(s.quality, wraith_core::Quality::Fast);
        assert_eq!(s.minify, wraith_core::Minify::Safe);
        assert_eq!(s.threshold, Some(200));
        assert_eq!(s.precision, 5);
        assert_eq!(s.legacy_color.color_precision, 8);
        assert_eq!(s.legacy_color.layer_difference, 32);
        assert_eq!(
            s.legacy_color.hierarchy,
            wraith_core::LegacyHierarchy::Cutout
        );
    }

    #[test]
    fn color_shorthand_sets_color_mode() {
        let cli = parse(&["a.png", "--color"]);
        assert_eq!(cli.mode, ModeArg::Bw); // flag itself does not flip
        assert_eq!(cli.settings().mode, wraith_core::Mode::Color);
    }

    #[test]
    fn geometric_knobs_flow_and_clamp() {
        let cli = parse(&[
            "a.png",
            "--color",
            "--colors",
            "16",
            "--tolerance",
            "0.45",
            "--corner-angle",
            "35",
            "--trap",
            "0",
            "--merge",
            "0.03",
            "--min-area",
            "6",
            "--gradients",
            "linear",
            "--no-primitives",
            "--refine",
            "4",
        ]);
        let c = cli.settings().color;
        assert_eq!(c.max_colors, wraith_core::Colors::Max(16));
        assert!((c.tolerance_px - 0.45).abs() < 1e-6);
        assert!((c.corner_angle_deg - 35.0).abs() < 1e-6);
        assert!((c.trap_px - 0.0).abs() < 1e-6);
        assert!((c.merge_distance - 0.03).abs() < 1e-6);
        assert!((c.min_area_px2 - 6.0).abs() < 1e-6);
        assert_eq!(c.gradients, wraith_core::Gradients::Linear);
        assert!(!c.primitives);
        assert_eq!(c.refine_iters, 4);
        // Out-of-range values are clamped, not rejected (values above 255 do
        // not parse as u8 at the CLI boundary at all).
        let wild = parse(&["a.png", "--tolerance", "99", "--colors", "200"])
            .settings()
            .color;
        assert!((wild.tolerance_px - 2.0).abs() < 1e-6);
        assert_eq!(wild.palette_cap(), 64);
        assert!(Cli::try_parse_from(["wraith", "a.png", "--colors", "999"]).is_err());
    }

    #[test]
    fn background_flag_parses_all_forms() {
        let none = parse(&["a.png"]).settings().background;
        assert_eq!(none, wraith_core::Background::None);
        let alpha = parse(&["a.png", "--background", "alpha"])
            .settings()
            .background;
        assert_eq!(alpha, wraith_core::Background::Alpha);
        let white = parse(&["a.png", "--background", "#ffffff"])
            .settings()
            .background;
        assert_eq!(
            white,
            wraith_core::Background::Color {
                rgb: [255, 255, 255],
                tolerance: 8
            }
        );
        let bare = parse(&["a.png", "--background", "ff0000", "--bg-tolerance", "20"])
            .settings()
            .background;
        assert_eq!(
            bare,
            wraith_core::Background::Color {
                rgb: [255, 0, 0],
                tolerance: 20
            }
        );
        assert!(Cli::try_parse_from(["wraith", "a.png", "--background", "zzz"]).is_err());
    }

    #[test]
    fn preset_applies_and_explicit_flags_win() {
        // Icon preset alone.
        let icon = parse(&["a.png", "--color", "--preset", "icon"])
            .settings()
            .color;
        assert!((icon.tolerance_px - 0.35).abs() < 1e-6);
        assert_eq!(icon.gradients, wraith_core::Gradients::Linear);
        // Explicit tolerance overrides the preset value.
        let over = parse(&["a.png", "--color", "--preset", "icon", "--tolerance", "0.5"])
            .settings()
            .color;
        assert!((over.tolerance_px - 0.5).abs() < 1e-6);
        // Preset gradients stay.
        assert_eq!(over.gradients, wraith_core::Gradients::Linear);
    }

    #[test]
    fn colors_arg_rejects_zero_and_garbage() {
        assert!(Cli::try_parse_from(["wraith", "a.png", "--colors", "0"]).is_err());
        assert!(Cli::try_parse_from(["wraith", "a.png", "--colors", "lots"]).is_err());
        assert_eq!(
            "auto".parse::<ColorsArg>(),
            Ok(ColorsArg::Auto),
            "`auto` parses"
        );
    }

    #[test]
    fn precision_range_is_enforced_by_clap() {
        assert!(Cli::try_parse_from(["wraith", "a.png", "-p", "0"]).is_err());
        assert!(Cli::try_parse_from(["wraith", "a.png", "-p", "9"]).is_err());
        assert!(Cli::try_parse_from(["wraith", "a.png", "-p", "8"]).is_ok());
    }

    #[test]
    fn refine_range_is_enforced_by_clap() {
        assert!(Cli::try_parse_from(["wraith", "a.png", "--refine", "31"]).is_err());
        assert!(Cli::try_parse_from(["wraith", "a.png", "--refine", "30"]).is_ok());
    }

    #[test]
    fn completions_subcommand_parses() {
        let cli = parse(&["completions", "zsh"]);
        assert!(matches!(
            cli.command,
            Some(Command::Completions { shell: Shell::Zsh })
        ));
    }
}
