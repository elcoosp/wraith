//! Per-stage timing and byte accounting.

use std::time::Duration;

/// Pipeline stages that are timed individually.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// PNG decode.
    Decode,
    /// Pre-thresholding, background removal and conversion to the tracer's
    /// image format.
    Preprocess,
    /// Palette extraction (geometric engine only).
    Palette,
    /// Cluster / layer tracing.
    Trace,
    /// Canonical serialization of the traced SVG.
    Serialize,
    /// OXVG minification.
    Minify,
}

impl Stage {
    /// Human-readable stage name, used by the CLI report.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Preprocess => "preprocess",
            Self::Palette => "palette",
            Self::Trace => "trace",
            Self::Serialize => "serialize",
            Self::Minify => "minify",
        }
    }
}

/// Measures wall-clock time per pipeline stage.
///
/// Laps chain: after the first `start()`, every `lap()` records the elapsed
/// time since the previous lap's end when no explicit `start()` intervened.
/// Stage code that forgets to restart the clock therefore still gets a
/// truthful timing instead of a silent `0.00 ms`.
#[derive(Debug, Default)]
pub struct Stopwatch {
    laps: Vec<(Stage, Duration)>,
    started: Option<std::time::Instant>,
    last_end: Option<std::time::Instant>,
}

impl Stopwatch {
    /// Starts timing the next stage.
    pub fn start(&mut self) {
        self.started = Some(std::time::Instant::now());
    }

    /// Stops the current stage and records its duration.
    pub fn lap(&mut self, stage: Stage) {
        let dur = if let Some(t) = self.started {
            t.elapsed()
        } else if let Some(t) = self.last_end {
            t.elapsed()
        } else {
            Duration::ZERO
        };
        self.laps.push((stage, dur));
        self.started = None;
        self.last_end = Some(std::time::Instant::now());
    }

    /// Recorded stage durations, in execution order.
    #[must_use]
    pub fn laps(&self) -> &[(Stage, Duration)] {
        &self.laps
    }
}

/// Byte counts and timings for one conversion.
#[derive(Debug, Clone)]
pub struct ConvertStats {
    /// Raster dimensions in pixels.
    pub width: u32,
    /// Raster dimensions in pixels.
    pub height: u32,
    /// Size of the encoded PNG input.
    pub input_bytes: u64,
    /// Size of the final SVG output.
    pub output_bytes: u64,
    /// Stage timings, in execution order.
    pub timings: Vec<(Stage, Duration)>,
    /// Palette entries used by the geometric engine (0 for the legacy
    /// engine, which does not build an explicit palette).
    pub palette_size: usize,
    /// Number of output geometry elements (paths and primitive shapes).
    pub path_count: usize,
}

impl ConvertStats {
    /// Sum of all recorded stage durations.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.timings.iter().map(|(_, d)| *d).sum()
    }

    /// Fraction of input bytes removed, in `0.0..=1.0`.
    /// Negative when the SVG is larger than the PNG.
    #[must_use]
    pub fn reduction(&self) -> f64 {
        if self.input_bytes == 0 {
            return 0.0;
        }
        let saved = self.input_bytes.saturating_sub(self.output_bytes);
        f64::from(u32::try_from(saved).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(self.input_bytes).unwrap_or(u32::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_names_are_stable() {
        assert_eq!(Stage::Decode.as_str(), "decode");
        assert_eq!(Stage::Preprocess.as_str(), "preprocess");
        assert_eq!(Stage::Palette.as_str(), "palette");
        assert_eq!(Stage::Trace.as_str(), "trace");
        assert_eq!(Stage::Serialize.as_str(), "serialize");
        assert_eq!(Stage::Minify.as_str(), "minify");
    }

    #[test]
    fn stopwatch_records_laps_in_order() {
        let mut sw = Stopwatch::default();
        sw.start();
        std::thread::sleep(Duration::from_micros(50));
        sw.lap(Stage::Decode);
        sw.start();
        sw.lap(Stage::Trace);
        let laps = sw.laps();
        assert_eq!(laps.len(), 2);
        assert_eq!(laps[0].0, Stage::Decode);
        assert_eq!(laps[1].0, Stage::Trace);
    }

    #[test]
    fn lap_without_start_records_zero() {
        let mut sw = Stopwatch::default();
        sw.lap(Stage::Minify);
        assert_eq!(sw.laps()[0].1, Duration::ZERO);
    }

    /// Consecutive stages may lap without an explicit `start()`: the clock
    /// keeps running from the previous lap's end. Without this, every stage
    /// after a `lap()` that forgot to restart the clock (the geometric
    /// engine's palette/trace stages did exactly that) reported 0.00 ms and
    /// the `--stats` output lied about where time went.
    #[test]
    fn consecutive_laps_without_start_stay_timed() {
        let mut sw = Stopwatch::default();
        sw.start();
        std::thread::sleep(Duration::from_micros(40));
        sw.lap(Stage::Decode);
        std::thread::sleep(Duration::from_micros(40));
        sw.lap(Stage::Preprocess);
        let laps = sw.laps();
        assert_eq!(laps.len(), 2);
        assert!(laps[0].1 > Duration::ZERO);
        assert!(laps[1].1 >= Duration::from_micros(30), "{:?}", laps[1]);
    }

    #[test]
    fn totals_and_reduction_math() {
        let stats = ConvertStats {
            width: 10,
            height: 10,
            input_bytes: 1_000,
            output_bytes: 250,
            timings: vec![
                (Stage::Decode, Duration::from_millis(2)),
                (Stage::Trace, Duration::from_millis(3)),
            ],
            palette_size: 4,
            path_count: 5,
        };
        assert_eq!(stats.total(), Duration::from_millis(5));
        assert!((stats.reduction() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn reduction_handles_growing_output_and_zero_input() {
        let grown = ConvertStats {
            width: 1,
            height: 1,
            input_bytes: 100,
            output_bytes: 400,
            timings: vec![],
            palette_size: 0,
            path_count: 0,
        };
        assert!((grown.reduction() - 0.0).abs() < 1e-9);
        let empty = ConvertStats {
            width: 0,
            height: 0,
            input_bytes: 0,
            output_bytes: 0,
            timings: vec![],
            palette_size: 0,
            path_count: 0,
        };
        assert!((empty.reduction() - 0.0).abs() < 1e-9);
    }
}
