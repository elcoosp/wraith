//! Stage 4: OXVG minification.

use oxvg_ast::{parse::roxmltree::parse, serialize::Node as _, visitor::Info};
use oxvg_optimiser::Jobs;

use crate::error::Result;
use crate::settings::Minify;
use crate::stats::{Stage, Stopwatch};

/// Runs the configured minifier job set over a canonical SVG string.
///
/// # Errors
///
/// Returns [`crate::error::WraithError::SvgParse`] when `svg` is not
/// well-formed XML, [`crate::error::WraithError::Minify`] when a job aborts,
/// and [`crate::error::WraithError::SvgWrite`] when serialization fails.
pub fn minify(svg: &str, level: Minify, sw: &mut Stopwatch) -> Result<String> {
    let jobs = match level {
        Minify::Off => return Ok(svg.to_owned()),
        // wraith's `safe` preset enables the thorough pre-clean checks that
        // upstream's `Jobs::safe` leaves off, so potentially unsafe documents
        // fail fast instead of being silently mangled.
        Minify::Safe => oxvg_optimiser::Jobs {
            precheck: Some(oxvg_optimiser::Precheck {
                fail_fast: true,
                preclean_checks: true,
            }),
            ..Jobs::safe()
        },
        Minify::Default => Jobs::default(),
    };
    sw.start();
    let out = parse(svg, |dom, allocator| {
        let info = Info::new(allocator);
        jobs.run(dom, &info)
            .map_err(|e| crate::error::WraithError::Minify(e.to_string()))?;
        dom.serialize().map_err(crate::error::WraithError::SvgWrite)
    })
    .map_err(|e| crate::error::WraithError::SvgParse(e.to_string()))?;
    sw.lap(Stage::Minify);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical() -> String {
        String::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\" \
             viewBox=\"0 0 8 8\"><g fill=\"#000000\"><path d=\"M0,0 L4,0 L4,4 Z\"/>\
             <path d=\"M4,4 L8,4 L8,8 Z\"/></g></svg>",
        )
    }

    #[test]
    fn off_level_returns_input_unchanged() {
        let mut sw = Stopwatch::default();
        let svg = canonical();
        let out = minify(&svg, Minify::Off, &mut sw).expect("passthrough");
        assert_eq!(out, svg);
        assert_eq!(sw.laps().len(), 0);
    }

    #[test]
    fn default_jobs_shorten_a_verbose_document() {
        let verbose = String::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\" \
             viewBox=\"0 0 8 8\"><g fill=\"#000000\"><path fill=\"#000000\" \
             d=\"M0,0 L4,0 L4,4 Z\"/><path fill=\"#000000\" \
             d=\"M4,4 L8,4 L8,8 Z\"/></g></svg>",
        );
        let mut sw = Stopwatch::default();
        let out = minify(&verbose, Minify::Default, &mut sw).expect("minifies");
        assert!(out.len() < verbose.len());
        assert_eq!(sw.laps().last().map(|(s, _)| *s), Some(Stage::Minify));
    }

    #[test]
    fn safe_jobs_run_and_preserve_geometry() {
        let svg = canonical();
        let mut sw = Stopwatch::default();
        let out = minify(&svg, Minify::Safe, &mut sw).expect("minifies");
        assert!(out.contains("<svg"));
        // Both subpath starts survive (possibly merged into one compound
        // path by the minifier).
        assert!(out.contains("M0"));
        assert!(out.contains("M4"));
        assert_eq!(sw.laps().last().map(|(s, _)| *s), Some(Stage::Minify));
    }

    #[test]
    fn malformed_xml_is_a_parse_error() {
        let mut sw = Stopwatch::default();
        let err = minify("<svg xmlns=\"oops\"", Minify::Default, &mut sw);
        assert!(matches!(err, Err(crate::error::WraithError::SvgParse(_))));
    }

    #[test]
    fn empty_document_survives() {
        let mut sw = Stopwatch::default();
        let out = minify(
            "<svg xmlns=\"http://www.w3.org/2000/svg\"/>",
            Minify::Default,
            &mut sw,
        )
        .expect("minifies");
        assert!(out.contains("svg"));
    }

    #[test]
    fn scripted_document_is_a_minify_error() {
        // The Safe preset runs a precheck that fail-fasts on animation and
        // scripting elements.
        let svg = String::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\">\
             <animate attributeName=\"x\"/>\
             <path d=\"M0,0 L4,0 L4,4 Z\"/></svg>",
        );
        let mut sw = Stopwatch::default();
        let err = minify(&svg, Minify::Safe, &mut sw);
        assert!(matches!(err, Err(crate::error::WraithError::Minify(_))));
    }
}
