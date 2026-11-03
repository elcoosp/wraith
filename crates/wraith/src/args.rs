//! Command-line argument model.

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
    /// Trace every colour cluster and keep the palette.
    Color,
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
/// Blazing-fast white-background PNG → optimised SVG.
// Booleans are the natural type for independent CLI flags; grouping them
// would only hurt the help output.
#[allow(clippy::struct_excessive_bools)]
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
        "  wraith *.png -o dist/        every PNG into dist/\n",
        "  cat logo.png | wraith -o -   SVG on stdout\n",
        "  wraith a.png --stats         per-file timings on stderr"
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

    /// Tracer speed/quality trade-off.
    #[arg(short = 'Q', long, value_enum, default_value_t)]
    pub quality: QualityArg,

    /// Luminance threshold (0–255) applied before tracing in `--mode bw`.
    #[arg(short = 't', long, value_parser = clap::value_parser!(u8))]
    pub threshold: Option<u8>,

    /// In `--mode color`, drop every pure-white path (wipes white
    /// backgrounds from the geometry).
    #[arg(long)]
    pub strip_white: bool,

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
    /// Converts CLI-facing enums into pipeline settings.
    #[must_use]
    pub fn settings(&self) -> wraith_core::Settings {
        wraith_core::Settings {
            mode: match self.mode {
                ModeArg::Bw => wraith_core::Mode::BlackWhite,
                ModeArg::Color => wraith_core::Mode::Color,
            },
            quality: match self.quality {
                QualityArg::Fast => wraith_core::Quality::Fast,
                QualityArg::Balanced => wraith_core::Quality::Balanced,
                QualityArg::Quality => wraith_core::Quality::Quality,
            },
            threshold: self.threshold,
            strip_white: self.strip_white,
            minify: match self.minify {
                MinifyArg::Default => wraith_core::Minify::Default,
                MinifyArg::Safe => wraith_core::Minify::Safe,
                MinifyArg::Off => wraith_core::Minify::Off,
            },
            precision: self.precision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("wraith").chain(args.iter().copied())).expect("parses")
    }

    #[test]
    fn defaults_are_black_white_balanced_minified() {
        let cli = parse(&["a.png"]);
        assert_eq!(cli.mode, ModeArg::Bw);
        assert_eq!(cli.quality, QualityArg::Balanced);
        assert_eq!(cli.minify, MinifyArg::Default);
        assert_eq!(cli.precision, 2);
        assert_eq!(cli.jobs, 0);
        assert!(!cli.stats && !cli.json && !cli.quiet);
        assert!(cli.threshold.is_none());
    }

    #[test]
    fn settings_map_onto_core() {
        let cli = parse(&[
            "a.png",
            "-m",
            "color",
            "-Q",
            "fast",
            "--minify",
            "safe",
            "-t",
            "200",
            "--strip-white",
            "-p",
            "5",
        ]);
        let s = cli.settings();
        assert_eq!(s.mode, wraith_core::Mode::Color);
        assert_eq!(s.quality, wraith_core::Quality::Fast);
        assert_eq!(s.minify, wraith_core::Minify::Safe);
        assert_eq!(s.threshold, Some(200));
        assert!(s.strip_white);
        assert_eq!(s.precision, 5);
    }

    #[test]
    fn precision_range_is_enforced_by_clap() {
        assert!(Cli::try_parse_from(["wraith", "a.png", "-p", "0"]).is_err());
        assert!(Cli::try_parse_from(["wraith", "a.png", "-p", "9"]).is_err());
        assert!(Cli::try_parse_from(["wraith", "a.png", "-p", "8"]).is_ok());
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
