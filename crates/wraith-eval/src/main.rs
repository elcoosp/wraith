//! wraith-eval: quality evaluation harness for wraith's tracing engines
//! (roadmap M0 — "truth first").
//!
//! ```text
//! wraith-eval run  [--out DIR] [--scale N]   # print + write report.json
//! wraith-eval check baseline.json [--tolerance 0.05]
//! ```
//!
//! `run` rasterises the ground-truth fixtures, runs every engine, scores the
//! output against the ground truth at 1×/4×/8× and prints a §5-style table.
//! `check` re-runs the harness and compares quality/size metrics against a
//! committed baseline; CI fails on a >5 % regression (roadmap M0 gate).

mod engines;
mod fixtures;
mod metrics;

use std::fmt::Write as _;
use std::path::PathBuf;

use wraith_core::settings::Minify;

use engines::{EngineConfig, engines, run_engine};
use fixtures::fixtures;
use metrics::{Scores, render_rgba_premultiplied, score};
use serde::{Deserialize, Serialize};

/// One row of the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    /// Fixture name.
    pub fixture: String,
    /// Engine key.
    pub engine: String,
    /// Palette size (0 for the legacy engine).
    pub palette: usize,
    /// Scores (flat for easy diffing).
    #[serde(flatten)]
    pub scores: Scores,
}

/// Full report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Rows in fixture × engine order.
    pub rows: Vec<Row>,
}

/// Per-metric tolerance for the CI gate.
#[derive(Debug, Clone, Copy, Deserialize)]
struct Tolerances {
    /// Relative tolerance on `rmse_4x` / `bad_4x` / `delta_e` (default 0.05).
    quality: f64,
    /// Relative tolerance on bytes (default 0.05).
    bytes: f64,
    /// Absolute tolerance on the seam score at 8× (fraction, not %).
    seam: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            quality: 0.05,
            bytes: 0.05,
            seam: 0.002,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("check") => args.get(1).map_or_else(
            || {
                eprintln!("usage: wraith-eval check <baseline.json> [--tolerance 0.05]");
                2
            },
            |path| cmd_check(path, &args[2..]),
        ),
        _ => {
            eprintln!("usage: wraith-eval <run|check> [args]");
            eprintln!(
                "  run [--out DIR]                  run all engines, print table, write report"
            );
            eprintln!(
                "  check <baseline.json> [--tolerance F]  compare against baseline, exit 1 on regression"
            );
            2
        }
    };
    std::process::exit(code);
}

fn cmd_run(args: &[String]) -> i32 {
    let mut out_dir: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_dir = args.get(i).map(PathBuf::from);
            }
            other => {
                eprintln!("unknown arg {other}");
                return 2;
            }
        }
        i += 1;
    }
    let size = 512_u32;
    let reps = 5_usize;
    let mut rows = Vec::new();
    let mut table = String::new();
    let _ = writeln!(
        table,
        "{:<18} {:<16} {:>8} {:>6} {:>9} {:>9} {:>9} {:>8} {:>9} {:>8}",
        "fixture",
        "engine",
        "ms",
        "paths",
        "bytes",
        "rmse@1x",
        "rmse@4x",
        "bad%@4x",
        "dE mean",
        "seam%8x"
    );
    for fixture in fixtures() {
        let gt_rgba = {
            let rgba_pm = render_rgba_premultiplied(&fixture.svg, size).expect("gt renders");
            demultiply(&rgba_pm)
        };
        for cfg in &engines() {
            // min-of-N for a stable time figure
            let mut best_ms = f64::MAX;
            let mut last: Option<engines::RunOutcome> = None;
            for _ in 0..reps {
                let outcome =
                    run_engine(cfg, &gt_rgba, size as usize, size as usize).expect("engine runs");
                best_ms = best_ms.min(outcome.ms);
                last = Some(outcome);
            }
            let outcome = last.expect("at least one run");
            let bytes_min = minified_len(&outcome.svg, cfg);
            let s = score(
                &fixture.svg,
                &outcome.svg,
                size,
                best_ms,
                outcome.paths,
                outcome.svg.len(),
                bytes_min,
            )
            .expect("scores");
            let _ = writeln!(
                table,
                "{:<18} {:<16} {:>8.2} {:>6} {:>9} {:>9.2} {:>9.2} {:>8.3} {:>9.3} {:>8.3}",
                fixture.name,
                cfg.key,
                s.ms,
                s.paths,
                s.bytes_raw,
                s.rmse_1x,
                s.rmse_4x,
                s.bad_4x,
                s.delta_e_mean_4x,
                s.seam_8x
            );
            rows.push(Row {
                fixture: fixture.name.to_owned(),
                engine: cfg.key.to_owned(),
                palette: outcome.palette_size,
                scores: s.clone(),
            });
            if let Some(dir) = &out_dir {
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(
                    dir.join(format!("{0}_{1}.svg", fixture.name, cfg.key)),
                    &outcome.svg,
                );
                if cfg.key == engines()[0].key {
                    let png = engines::encode_png(&gt_rgba, size, size).expect("png encodes");
                    let _ = std::fs::write(dir.join(format!("{0}_in.png", fixture.name)), png);
                }
            }
        }
    }
    print!("{table}");
    let report = Report { rows };
    match out_dir {
        Some(dir) => {
            let path = dir.join("report.json");
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&report).expect("serialises"),
            )
            .expect("writes report");
            println!("report written to {}", path.display());
        }
        None => {
            println!("{}", serde_json::to_string(&report).expect("serialises"));
        }
    }
    0
}

fn cmd_check(baseline_path: &str, args: &[String]) -> i32 {
    let mut tol = Tolerances::default();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--tolerance" {
            i += 1;
            if let Some(t) = args.get(i).and_then(|v| v.parse::<f64>().ok()) {
                tol.quality = t;
                tol.bytes = t;
            } else {
                eprintln!("--tolerance needs a number");
                return 2;
            }
        }
        i += 1;
    }
    let Ok(baseline_json) = std::fs::read_to_string(baseline_path) else {
        eprintln!("cannot read baseline {baseline_path}");
        return 2;
    };
    let Ok(baseline) = serde_json::from_str::<Report>(&baseline_json) else {
        eprintln!("baseline is not a valid report");
        return 2;
    };
    // Re-run the harness.
    let size = 512_u32;
    let mut failures = Vec::new();
    let mut rows = Vec::new();
    for fixture in fixtures() {
        let gt_rgba = {
            let rgba_pm = render_rgba_premultiplied(&fixture.svg, size).expect("gt renders");
            demultiply(&rgba_pm)
        };
        for cfg in &engines() {
            let outcome =
                run_engine(cfg, &gt_rgba, size as usize, size as usize).expect("engine runs");
            let bytes_min = minified_len(&outcome.svg, cfg);
            let s = score(
                &fixture.svg,
                &outcome.svg,
                size,
                outcome.ms,
                outcome.paths,
                outcome.svg.len(),
                bytes_min,
            )
            .expect("scores");
            rows.push(Row {
                fixture: fixture.name.to_owned(),
                engine: cfg.key.to_owned(),
                palette: outcome.palette_size,
                scores: s.clone(),
            });
            let Some(base) = baseline
                .rows
                .iter()
                .find(|r| r.fixture == fixture.name && r.engine == cfg.key)
            else {
                failures.push(format!(
                    "{}/{}: missing from baseline",
                    fixture.name, cfg.key
                ));
                continue;
            };
            for (name, now, base, rel) in [
                ("rmse_4x", s.rmse_4x, base.scores.rmse_4x, tol.quality),
                ("bad_4x", s.bad_4x, base.scores.bad_4x, tol.quality),
                (
                    "delta_e_mean_4x",
                    s.delta_e_mean_4x,
                    base.scores.delta_e_mean_4x,
                    tol.quality,
                ),
                (
                    "bytes_raw",
                    s.bytes_raw as f64,
                    base.scores.bytes_raw as f64,
                    tol.bytes,
                ),
                (
                    "bytes_min",
                    s.bytes_min as f64,
                    base.scores.bytes_min as f64,
                    tol.bytes,
                ),
            ] {
                let limit = base * (1.0 + rel) + 1e-9;
                if now > limit {
                    failures.push(format!(
                        "{}/{}: {name} regressed: {now:.4} > {base:.4} × (1+{rel})",
                        fixture.name, cfg.key
                    ));
                }
            }
            if s.seam_8x / 100.0 > tol.seam {
                failures.push(format!(
                    "{}/{}: seam score {}‰ exceeds {:?}",
                    fixture.name, cfg.key, s.seam_8x, tol.seam
                ));
            }
        }
    }
    if failures.is_empty() {
        println!("check OK: {} rows within tolerance", rows.len());
        0
    } else {
        eprintln!("check FAILED ({} regressions):", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        1
    }
}

/// OXVG-minified byte count for a config's output.
fn minified_len(svg: &str, cfg: &EngineConfig) -> usize {
    match cfg.settings.minify {
        Minify::Off => svg.len(),
        _ => wraith_core::minify::minify(
            svg,
            cfg.settings.minify,
            &mut wraith_core::Stopwatch::default(),
        )
        .map_or(svg.len(), |m| m.len()),
    }
}

/// resvg pixmaps are premultiplied; the pipeline wants straight alpha.
fn demultiply(rgba_pm: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgba_pm.len());
    for p in rgba_pm.as_chunks::<4>().0 {
        let a = u32::from(p[3]);
        out.push(demul(p[0], a));
        out.push(demul(p[1], a));
        out.push(demul(p[2], a));
        out.push(p[3]);
    }
    out
}

fn demul(c: u8, a: u32) -> u8 {
    if a == 0 || a == 255 {
        return c;
    }
    u8::try_from((u32::from(c) * 255 + a / 2) / a).unwrap_or(255)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demultiply_round_trip() {
        // 50% premultiplied white → straight white at 50% alpha.
        let pm = [128_u8, 128, 128, 128];
        let straight = demultiply(&pm);
        assert_eq!(straight, vec![255, 255, 255, 128]);
        // Fully opaque passes through.
        let pm2 = [10_u8, 20, 30, 255];
        assert_eq!(demultiply(&pm2), vec![10, 20, 30, 255]);
        // Fully transparent stays zero.
        assert_eq!(demultiply(&[0, 0, 0, 0]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn tolerances_default_to_the_m0_gate() {
        let t = Tolerances::default();
        assert!((t.quality - 0.05).abs() < 1e-9);
        assert!((t.bytes - 0.05).abs() < 1e-9);
    }

    #[test]
    fn engines_matrix_is_stable() {
        let keys: Vec<&str> = engines().iter().map(|e| e.key).collect();
        assert_eq!(
            keys,
            vec![
                "legacy_balanced",
                "legacy_quality",
                "geometric",
                "geometric_nograd"
            ]
        );
    }
}
