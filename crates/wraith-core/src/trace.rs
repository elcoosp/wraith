//! Stage 2: cluster tracing via vtracer.

use crate::error::{Result, WraithError};
use crate::settings::Settings;
use crate::stats::{Stage, Stopwatch};

/// Runs the tracer over a prepared image.
///
/// # Errors
///
/// Returns [`crate::error::WraithError::Trace`] when the tracer rejects the
/// image (practically only for pathological colour-mode inputs).
pub fn trace(
    image: vtracer::ColorImage,
    settings: &Settings,
    sw: &mut Stopwatch,
) -> Result<vtracer::SvgFile> {
    sw.start();
    let svg = vtracer::convert(image, settings.tracer_config()).map_err(WraithError::Trace);
    sw.lap(Stage::Trace);
    svg
}

#[cfg(test)]
mod tests {
    use crate::preprocess::{fixture_png, prepare};
    use crate::stats::Stopwatch;

    use super::*;

    #[test]
    fn trace_produces_paths_for_fixture() {
        let png = fixture_png(32);
        let mut sw = Stopwatch::default();
        let settings = Settings::new();
        let prepared = prepare(&png, &settings, &mut sw).expect("prepares");
        let svg = trace(prepared.image, &settings, &mut sw).expect("traces");
        assert!(!svg.paths.is_empty());
        assert_eq!(svg.width, 32);
        assert_eq!(svg.height, 32);
    }

    #[test]
    fn trace_runs_in_the_trace_lap() {
        let png = fixture_png(32);
        let mut sw = Stopwatch::default();
        let settings = Settings::new();
        let prepared = prepare(&png, &settings, &mut sw).expect("prepares");
        let _ = trace(prepared.image, &settings, &mut sw).expect("traces");
        assert_eq!(sw.laps().last().map(|(s, _)| *s), Some(Stage::Trace));
    }
}
