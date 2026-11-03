//! Human and JSON reporting for conversion results.

use std::fmt::Write as _;
use std::time::Duration;

/// One finished (or failed) conversion, ready to report.
#[derive(Debug)]
pub struct FileResult {
    /// Display name of the source (path or `<stdin>`).
    pub name: String,
    /// Stats on success.
    pub stats: Option<wraith_core::ConvertStats>,
    /// Error message on failure.
    pub error: Option<String>,
}

/// Renders one human-readable per-file line.
#[must_use]
pub fn file_line(r: &FileResult, color: bool) -> String {
    let Some(stats) = &r.stats else {
        return paint(
            &format!("✗ {} {}", r.name, r.error.as_deref().unwrap_or("failed")),
            "31",
            color,
        );
    };
    let base = format!(
        "✓ {}  {} → {} B  (-{:.1}%)  {:.1}ms",
        r.name,
        stats.input_bytes,
        stats.output_bytes,
        stats.reduction() * 100.0,
        stats.total().as_secs_f64() * 1000.0
    );
    paint(&base, "32", color)
}

/// Renders per-file stage timings.
#[must_use]
pub fn timing_line(stats: &wraith_core::ConvertStats) -> String {
    let parts: Vec<String> = stats
        .timings
        .iter()
        .map(|(stage, dur)| format!("{} {:.2}ms", stage.as_str(), dur.as_secs_f64() * 1000.0))
        .collect();
    parts.join(", ")
}

/// Renders the summary line for a batch.
#[must_use]
pub fn summary_line(ok: usize, failed: usize, total: Duration, color: bool) -> String {
    let head = if failed == 0 {
        paint(&format!("✓ {ok}/{ok} converted"), "32", color)
    } else {
        let all = ok + failed;
        paint(&format!("⚠ {ok}/{all} converted"), "33", color)
    };
    format!("{head} in {:.1}ms", total.as_secs_f64() * 1000.0)
}

/// Renders the whole batch as one JSON document.
#[must_use]
pub fn json_report(results: &[FileResult]) -> String {
    let mut out = String::with_capacity(64 + results.len() * 128);
    out.push_str("{\"files\":[");
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        push_json_escaped(&mut out, &r.name);
        out.push('"');
        if let Some(stats) = &r.stats {
            let _ = write!(
                out,
                ",\"width\":{},\"height\":{},\"input_bytes\":{},\"output_bytes\":{},\"reduction\":{:.4},\"total_us\":{},\"timings_us\":{{",
                stats.width,
                stats.height,
                stats.input_bytes,
                stats.output_bytes,
                stats.reduction(),
                stats.total().as_micros()
            );
            for (j, (stage, dur)) in stats.timings.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                let _ = write!(out, "\"{}\":{}", stage.as_str(), dur.as_micros());
            }
            out.push_str("},\"error\":null");
        } else {
            out.push_str(",\"error\":\"");
            push_json_escaped(&mut out, r.error.as_deref().unwrap_or("failed"));
            out.push('"');
        }
        out.push('}');
    }
    out.push_str("]}");
    out
}

/// Minimal JSON string escaping for names flowing into reports.
fn push_json_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

fn paint(s: &str, code: &str, enabled: bool) -> String {
    if enabled {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn stats() -> wraith_core::ConvertStats {
        wraith_core::ConvertStats {
            width: 16,
            height: 16,
            input_bytes: 1000,
            output_bytes: 250,
            timings: vec![
                (wraith_core::Stage::Decode, Duration::from_micros(10)),
                (wraith_core::Stage::Minify, Duration::from_micros(20)),
            ],
        }
    }

    #[test]
    fn file_line_reports_success_and_failure() {
        let ok = FileResult {
            name: String::from("a.png"),
            stats: Some(stats()),
            error: None,
        };
        let plain = file_line(&ok, false);
        assert!(plain.contains("✓ a.png"));
        assert!(plain.contains("1000 → 250 B"));
        assert!(plain.contains("-75.0%"));
        assert!(!plain.contains("\x1b["));

        let colored = file_line(&ok, true);
        assert!(colored.contains("\x1b[32m"));

        let bad = FileResult {
            name: String::from("b.png"),
            stats: None,
            error: Some(String::from("boom")),
        };
        let line = file_line(&bad, false);
        assert!(line.contains("✗ b.png boom"));
        assert!(file_line(&bad, true).contains("\x1b[31m"));
    }

    #[test]
    fn timing_line_lists_stages_in_order() {
        let line = timing_line(&stats());
        assert_eq!(line, "decode 0.01ms, minify 0.02ms");
    }

    #[test]
    fn summary_line_handles_partial_failure() {
        let all = summary_line(3, 0, Duration::from_millis(12), false);
        assert!(all.contains("✓ 3/3 converted"));
        assert!(all.contains("in 12.0ms"));
        let partial = summary_line(2, 1, Duration::ZERO, false);
        assert!(partial.contains("⚠ 2/3 converted"));
        assert!(summary_line(2, 1, Duration::ZERO, true).contains("\x1b[33m"));
    }

    #[test]
    fn json_report_encodes_success_and_error() {
        let results = vec![
            FileResult {
                name: String::from("a.png"),
                stats: Some(stats()),
                error: None,
            },
            FileResult {
                name: String::from("b\"x.png"),
                stats: None,
                error: Some(String::from("no\tdice")),
            },
        ];
        let json = json_report(&results);
        assert!(json.starts_with("{\"files\":["));
        assert!(json.contains("\"name\":\"a.png\""));
        assert!(json.contains("\"input_bytes\":1000"));
        assert!(json.contains("\"reduction\":0.7500"));
        assert!(json.contains("\"decode\":10"));
        assert!(json.contains("\"name\":\"b\\\"x.png\""));
        assert!(json.contains("\"error\":\"no\\tdice\""));
        assert!(json.ends_with("]}"));
    }

    #[test]
    fn json_escaping_covers_control_chars() {
        let mut out = String::new();
        push_json_escaped(&mut out, "a\\b\"c\nd\re\u{1}f");
        assert_eq!(out, "a\\\\b\\\"c\\nd\\re\\u0001f");
    }
}
