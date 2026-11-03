//! # wraith (CLI)
//!
//! Thin, testable shell around [`wraith_core`]: argument parsing, input
//! resolution, parallel batch execution and reporting. `main.rs` is a
//! two-liner that wires real process I/O into [`run`].

pub mod args;
pub mod report;

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use anyhow::Context as _;
use clap::{CommandFactory, Parser as _};
use wraith_core::Settings;

use crate::args::{Cli, Command, Shell};
use crate::report::FileResult;

/// Exit code for a fully successful run.
pub const EXIT_OK: i32 = 0;
/// Exit code when at least one conversion failed.
pub const EXIT_FAILURE: i32 = 1;
/// Exit code for usage errors (bad flags, bad argument combinations).
pub const EXIT_USAGE: i32 = 2;

/// A batch job writing to a concrete file path.
#[derive(Debug)]
struct FileJob {
    name: String,
    source: Source,
    path: PathBuf,
}

/// The single job allowed to write to stdout (exactly one input).
#[derive(Debug)]
struct StdoutJob {
    name: String,
    source: Source,
}

/// Where the PNG bytes come from.
#[derive(Debug)]
enum Source {
    /// A file on disk, read inside the worker.
    Path(PathBuf),
    /// Bytes already read from stdin.
    Stdin(Vec<u8>),
}

/// Runs the CLI end-to-end against the supplied streams and returns the
/// process exit code.
///
/// `argv` includes the program name. `color` enables ANSI styling in the
/// report; callers should pass `stderr.is_terminal() && NO_COLOR is unset`.
pub fn run(
    argv: &[OsString],
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    color: bool,
) -> i32 {
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(e) => {
            let rendered = e.render();
            return if e.use_stderr() {
                let _ = write!(stderr, "{rendered}");
                EXIT_USAGE
            } else {
                let _ = write!(stdout, "{rendered}");
                EXIT_OK
            };
        }
    };

    if let Some(Command::Completions { shell }) = cli.command {
        let mut cmd = Cli::command();
        let target = match shell {
            Shell::Bash => clap_complete::aot::Shell::Bash,
            Shell::Zsh => clap_complete::aot::Shell::Zsh,
            Shell::Fish => clap_complete::aot::Shell::Fish,
        };
        clap_complete::aot::generate(target, &mut cmd, "wraith", stdout);
        return EXIT_OK;
    }

    let started = Instant::now();
    let settings = cli.settings();

    let (file_jobs, stdout_job) = match resolve_jobs(&cli, stdin) {
        Ok(resolved) => resolved,
        Err(msg) => {
            let _ = writeln!(stderr, "wraith: {msg}");
            return EXIT_USAGE;
        }
    };

    let mut results = execute(&file_jobs, settings, cli.jobs);
    if let Some(job) = &stdout_job {
        results.push(run_stdout_job(job, settings, stdout));
    }
    report_results(&cli, &results, started, stderr, color);

    if results.iter().any(|r| r.stats.is_none()) {
        EXIT_FAILURE
    } else {
        EXIT_OK
    }
}

/// Expands CLI inputs into concrete jobs with resolved destinations.
/// Returns file-bound jobs plus the optional single stdout job.
fn resolve_jobs(
    cli: &Cli,
    stdin: &mut dyn Read,
) -> Result<(Vec<FileJob>, Option<StdoutJob>), String> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut use_stdin = false;

    if cli.inputs.is_empty() {
        use_stdin = true;
    }
    for input in &cli.inputs {
        if input == "-" {
            use_stdin = true;
            continue;
        }
        let path = PathBuf::from(input);
        if path.is_dir() {
            scan_pngs(&path, &mut files).map_err(|e| format!("cannot scan `{input}`: {e}"))?;
        } else if path.is_file() {
            files.push(path);
        } else {
            return Err(format!("`{input}` is not a file or directory"));
        }
    }
    if use_stdin && !files.is_empty() {
        return Err("stdin input `-` cannot be combined with file inputs".to_owned());
    }

    if use_stdin {
        let mut bytes = Vec::new();
        stdin
            .read_to_end(&mut bytes)
            .map_err(|e| format!("cannot read stdin: {e}"))?;
        let source = Source::Stdin(bytes);
        return match &cli.output {
            None => Ok((
                Vec::new(),
                Some(StdoutJob {
                    name: String::from("<stdin>"),
                    source,
                }),
            )),
            Some(o) if o == "-" => Ok((
                Vec::new(),
                Some(StdoutJob {
                    name: String::from("<stdin>"),
                    source,
                }),
            )),
            Some(o) => Ok((
                vec![FileJob {
                    name: String::from("<stdin>"),
                    source,
                    path: PathBuf::from(o),
                }],
                None,
            )),
        };
    }

    let many = files.len() > 1;
    if many && cli.output.as_deref() == Some("-") {
        return Err("stdout output `-` requires exactly one input".to_owned());
    }
    if many && let Some(o) = &cli.output {
        // A shared output directory for a batch is created eagerly, so a
        // typo fails before any conversion work happens.
        std::fs::create_dir_all(o).map_err(|e| format!("cannot create output dir `{o}`: {e}"))?;
    }

    let mut jobs = Vec::with_capacity(files.len());
    for path in &files {
        let dest = match &cli.output {
            None => sibling_svg(path),
            Some(o) if many => PathBuf::from(o).join(file_name_svg(path)),
            Some(o) => PathBuf::from(o),
        };
        jobs.push(FileJob {
            name: path.display().to_string(),
            source: Source::Path(path.clone()),
            path: dest,
        });
    }
    Ok((jobs, None))
}

/// `dir/name.png` → `dir/name.svg`.
fn sibling_svg(path: &Path) -> PathBuf {
    path.with_extension("svg")
}

/// Final file name for a converted source path.
fn file_name_svg(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map_or_else(|| "out".to_owned(), |s| s.to_string_lossy().into_owned());
    format!("{stem}.svg")
}

/// Recursively collects `*.png` files under `dir`, sorted for determinism.
fn scan_pngs(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        // file_type() never follows symlinks, so symlinked directories cannot
        // create cycles.
        if entry.file_type()?.is_dir() {
            scan_pngs(&entry.path(), out)?;
        } else if entry
            .path()
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png"))
        {
            out.push(entry.path());
        }
    }
    Ok(())
}

/// Executes all file jobs (parallel when requested) and returns results in
/// input order.
fn execute(jobs: &[FileJob], settings: Settings, threads: usize) -> Vec<FileResult> {
    let convert = |job: &FileJob| -> FileResult {
        let png = read_source(&job.source);
        let png = match png {
            Ok(png) => png,
            Err(msg) => {
                return FileResult {
                    name: job.name.clone(),
                    stats: None,
                    error: Some(msg),
                };
            }
        };
        match wraith_core::convert(&png, &settings) {
            Ok(out) => match write_file(&job.path, &out.svg) {
                Ok(()) => FileResult {
                    name: job.name.clone(),
                    stats: Some(out.stats),
                    error: None,
                },
                Err(msg) => FileResult {
                    name: job.name.clone(),
                    stats: None,
                    error: Some(msg),
                },
            },
            Err(e) => FileResult {
                name: job.name.clone(),
                stats: None,
                error: Some(e.to_string()),
            },
        }
    };

    match jobs.len() {
        0 => Vec::new(),
        1 => vec![convert(&jobs[0])],
        n => {
            let threads = if threads == 0 {
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
            } else {
                threads
            };
            if threads <= 1 {
                return jobs.iter().map(convert).collect();
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.min(n))
                .build()
                .expect("thread pool builds");
            pool.install(|| jobs.par_iter().map(convert).collect::<Vec<_>>())
        }
    }
}

/// Reads PNG bytes from a job source.
fn read_source(source: &Source) -> Result<Vec<u8>, String> {
    match source {
        Source::Stdin(bytes) => Ok(bytes.clone()),
        Source::Path(p) => std::fs::read(p)
            .with_context(|| format!("cannot read `{}`", p.display()))
            .map_err(|e| format!("{e:#}")),
    }
}

/// Converts the single stdin/stdout job inline.
fn run_stdout_job(job: &StdoutJob, settings: Settings, stdout: &mut dyn Write) -> FileResult {
    let png = match read_source(&job.source) {
        Ok(png) => png,
        Err(msg) => {
            return FileResult {
                name: job.name.clone(),
                stats: None,
                error: Some(msg),
            };
        }
    };
    match wraith_core::convert(&png, &settings) {
        Ok(out) => {
            let written = stdout
                .write_all(out.svg.as_bytes())
                .and_then(|()| stdout.flush())
                .map_err(|e| format!("cannot write to stdout: {e}"));
            match written {
                Ok(()) => FileResult {
                    name: job.name.clone(),
                    stats: Some(out.stats),
                    error: None,
                },
                Err(msg) => FileResult {
                    name: job.name.clone(),
                    stats: None,
                    error: Some(msg),
                },
            }
        }
        Err(e) => FileResult {
            name: job.name.clone(),
            stats: None,
            error: Some(e.to_string()),
        },
    }
}

/// Writes finished SVG bytes to a file, creating parent directories.
fn write_file(path: &Path, svg: &str) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create `{}`: {e}", parent.display()))?;
    }
    std::fs::write(path, svg).map_err(|e| format!("cannot write `{}`: {e}", path.display()))
}

/// Emits summary, per-file stats and JSON to stderr as configured.
fn report_results(
    cli: &Cli,
    results: &[FileResult],
    started: Instant,
    stderr: &mut dyn Write,
    color: bool,
) {
    let ok = results.iter().filter(|r| r.stats.is_some()).count();
    let failed = results.len() - ok;

    if cli.json {
        let _ = writeln!(stderr, "{}", report::json_report(results));
    }
    if !cli.quiet {
        let _ = writeln!(
            stderr,
            "{}",
            report::summary_line(ok, failed, started.elapsed(), color)
        );
    }
    if !cli.quiet && cli.stats {
        for r in results {
            match &r.stats {
                Some(stats) => {
                    let _ = writeln!(
                        stderr,
                        "  {} [{}]",
                        report::file_line(r, color),
                        report::timing_line(stats)
                    );
                }
                None => {
                    let _ = writeln!(stderr, "  {}", report::file_line(r, color));
                }
            }
        }
    } else {
        // Failures are never swallowed — not even under `--quiet`.
        for r in results.iter().filter(|r| r.stats.is_none()) {
            let _ = writeln!(stderr, "{}", report::file_line(r, color));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{MinifyArg, ModeArg, QualityArg};

    fn cli(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("wraith").chain(args.iter().copied())).expect("parses")
    }

    fn png_bytes(square: bool) -> Vec<u8> {
        let mut img = image::ImageBuffer::from_pixel(16, 16, image::Rgba([255, 255, 255, 255]));
        if square {
            for y in 4..12 {
                for x in 4..12 {
                    img.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
                }
            }
        }
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .expect("encodes");
        buf
    }

    struct Streams {
        stdin: std::io::Cursor<Vec<u8>>,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    }

    /// Single-instantiation wrapper so `run` sees one concrete argv type.
    fn run_args(
        args: &[&str],
        stdin: &mut dyn Read,
        stdout: &mut dyn Write,
        stderr: &mut dyn Write,
        color: bool,
    ) -> i32 {
        let os_argv: Vec<OsString> = args.iter().map(OsString::from).collect();
        run(&os_argv, stdin, stdout, stderr, color)
    }

    fn streams(png: Vec<u8>) -> Streams {
        Streams {
            stdin: std::io::Cursor::new(png),
            stdout: Vec::new(),
            stderr: Vec::new(),
        }
    }

    /// A writer that accepts bytes but fails to flush, for exercising sink
    /// error paths.
    struct BrokenWriter;
    impl Write for BrokenWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("broken pipe"))
        }
    }

    #[test]
    fn completions_cover_every_shell() {
        for shell in ["bash", "zsh", "fish"] {
            let mut s = streams(Vec::new());
            let code = run_args(
                &["wraith", "completions", shell],
                &mut s.stdin,
                &mut s.stdout,
                &mut s.stderr,
                false,
            );
            assert_eq!(code, EXIT_OK, "shell {shell}");
            assert!(s.stdout.len() > 100);
        }
    }

    #[test]
    fn blank_image_still_converts_without_paths() {
        let mut s = streams(png_bytes(false));
        let code = run_args(
            &["wraith", "-o", "-"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        let svg = String::from_utf8_lossy(&s.stdout);
        assert!(svg.starts_with("<svg"));
        assert!(!svg.contains("<path"));
    }

    #[test]
    fn stdout_job_converts_even_without_paths_and_survives_broken_stdout() {
        let settings = Settings::default();

        // A conversion failure flows through the error arm.
        let job = StdoutJob {
            name: String::from("<stdin>"),
            source: Source::Stdin(b"junk".to_vec()),
        };
        let mut out = Vec::new();
        let failed = run_stdout_job(&job, settings, &mut out);
        assert!(failed.stats.is_none());

        // A broken stdout surfaces as a file result error.
        let job = StdoutJob {
            name: String::from("<stdin>"),
            source: Source::Stdin(png_bytes(true)),
        };
        let mut broken = BrokenWriter;
        let failed = run_stdout_job(&job, settings, &mut broken);
        assert!(
            failed
                .error
                .as_deref()
                .unwrap_or("")
                .contains("cannot write to stdout")
        );

        // An unreadable source file surfaces as a read error.
        let job = StdoutJob {
            name: String::from("ghost.png"),
            source: Source::Path(PathBuf::from("/definitely/not/here.png")),
        };
        let failed = run_stdout_job(&job, settings, &mut Vec::new());
        assert!(
            failed
                .error
                .as_deref()
                .unwrap_or("")
                .contains("cannot read")
        );
    }

    #[test]
    fn unwritable_output_path_fails_that_file() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"i am a file").expect("write blocker");
        let job = FileJob {
            name: String::from("x.png"),
            source: Source::Stdin(png_bytes(true)),
            // Parent exists but is a regular file → create_dir_all fails.
            path: blocker.join("out.svg"),
        };
        let results = execute(std::slice::from_ref(&job), Settings::default(), 0);
        assert!(results[0].stats.is_none());
        assert!(
            results[0]
                .error
                .as_deref()
                .unwrap_or("")
                .contains("cannot create")
        );

        // Destination itself is a directory → the final write fails.
        let dir_dest = tempfile::tempdir().expect("tmpdir");
        let job = FileJob {
            name: String::from("x.png"),
            source: Source::Stdin(png_bytes(true)),
            path: dir_dest.path().to_path_buf(),
        };
        let results = execute(std::slice::from_ref(&job), Settings::default(), 0);
        assert!(results[0].stats.is_none());
        assert!(
            results[0]
                .error
                .as_deref()
                .unwrap_or("")
                .contains("cannot write")
        );
    }

    #[test]
    fn relative_filename_output_skips_parent_creation() {
        // A bare filename has an empty parent; no directory must be created.
        let name = "wraith-relative-test.svg";
        let result = write_file(Path::new(name), "<svg/>");
        let _ = std::fs::remove_file(name);
        assert!(result.is_ok());
    }

    /// A reader that always fails, for exercising stdin error paths.
    struct BrokenReader;
    impl Read for BrokenReader {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("broken stdin"))
        }
    }

    #[test]
    fn stdin_read_error_is_a_usage_error() {
        let cli = cli(&["-o", "-"]);
        let err = resolve_jobs(&cli, &mut BrokenReader).expect_err("stdin fails");
        assert!(err.contains("cannot read stdin"));
    }

    #[test]
    fn output_dir_creation_failure_is_a_usage_error() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let a = tmp.path().join("a.png");
        let b = tmp.path().join("b.png");
        std::fs::write(&a, png_bytes(true)).expect("png");
        std::fs::write(&b, png_bytes(true)).expect("png");
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"i am a file").expect("blocker");
        let out = blocker.join("out");
        let cli = cli(&[
            a.to_str().expect("utf8"),
            b.to_str().expect("utf8"),
            "-o",
            out.to_str().expect("utf8"),
        ]);
        let mut empty = std::io::empty();
        let err = resolve_jobs(&cli, &mut empty).expect_err("mkdir fails");
        assert!(err.contains("cannot create output dir"));
    }

    #[test]
    fn unreadable_dir_reports_scan_error() {
        // Exhaust file descriptors so `read_dir` fails with EMFILE; this is
        // the only portable way to make directory reading fail as root.
        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut fds = Vec::new();
        for _ in 0..2048 {
            match std::fs::File::open("/dev/null") {
                Ok(f) => fds.push(f),
                Err(_) => break,
            }
        }
        let cli = cli(&[tmp.path().to_str().expect("utf8")]);
        let mut empty = std::io::empty();
        let err = resolve_jobs(&cli, &mut empty);
        fds.clear();
        // The scan must fail under fd exhaustion (soft limit 1024 on Linux);
        // on platforms where exhaustion is impossible this degrades to a
        // no-op instead of failing.
        let _ = err.is_err_and(|msg| {
            assert!(msg.contains("cannot scan"), "unexpected error: {msg}");
            true
        });
    }

    #[test]
    fn stats_mode_lists_failures_without_timings() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        std::fs::write(tmp.path().join("bad.png"), b"not a png").expect("bad");
        let dir_arg = tmp.path().display().to_string().leak();
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", dir_arg, "--stats"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_FAILURE);
        let err = String::from_utf8_lossy(&s.stderr);
        assert!(err.contains("✗"));
        assert!(!err.contains("[decode"));
    }

    #[test]
    fn usage_error_returns_two_and_prints_message() {
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", "-p", "not-a-number"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_USAGE);
        assert_ne!(s.stderr.len(), 0);
    }

    #[test]
    fn help_request_returns_zero_on_stdout() {
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", "--help"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        assert!(String::from_utf8_lossy(&s.stdout).contains("Usage:"));
    }

    #[test]
    fn completions_write_bash_script() {
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", "completions", "bash"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        let script = String::from_utf8_lossy(&s.stdout);
        assert!(script.contains("wraith"));
        assert!(script.contains("COMPREPLY") || script.contains("compdef") || script.len() > 100);
    }

    #[test]
    fn stdin_to_stdout_end_to_end() {
        let mut s = streams(png_bytes(true));
        let code = run_args(
            &["wraith", "-o", "-"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        let svg = String::from_utf8_lossy(&s.stdout);
        assert!(svg.starts_with("<svg"));
        assert!(String::from_utf8_lossy(&s.stderr).contains("converted"));
    }

    #[test]
    fn stdin_defaults_to_stdout_without_output_flag() {
        let mut s = streams(png_bytes(true));
        let code = run_args(
            &["wraith"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        assert!(String::from_utf8_lossy(&s.stdout).starts_with("<svg"));
    }

    #[test]
    fn stdin_json_reports_to_stderr() {
        let mut s = streams(png_bytes(true));
        let code = run_args(
            &["wraith", "--json"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        let err = String::from_utf8_lossy(&s.stderr);
        assert!(err.contains("\"files\":["));
        assert!(err.contains("\"name\":\"<stdin>\""));
    }

    #[test]
    fn single_file_writes_sibling_svg() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let src = tmp.path().join("logo.png");
        std::fs::write(&src, png_bytes(true)).expect("write fixture");
        let mut s = streams(Vec::new());
        let src_arg = src.display().to_string();
        let code = run_args(
            &["wraith", &src_arg],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        let out = tmp.path().join("logo.svg");
        let svg = std::fs::read_to_string(&out).expect("svg exists");
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn single_file_to_explicit_output_and_nested_parents_are_created() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let src = tmp.path().join("logo.png");
        std::fs::write(&src, png_bytes(true)).expect("write fixture");
        let dst = tmp.path().join("deep/dir/out.svg");
        let mut s = streams(Vec::new());
        let args = [
            "wraith",
            src.to_str().expect("utf8").to_owned().leak(),
            "-o",
            dst.to_str().expect("utf8"),
        ];
        let code = run_args(&args, &mut s.stdin, &mut s.stdout, &mut s.stderr, false);
        assert_eq!(code, EXIT_OK);
        assert!(dst.exists());
    }

    #[test]
    fn batch_writes_into_output_directory() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        for name in ["a.png", "b.png"] {
            std::fs::write(tmp.path().join(name), png_bytes(true)).expect("write fixture");
        }
        let outdir = tmp.path().join("svg-out");
        let dir_arg = tmp.path().display().to_string().leak();
        let out_arg = outdir.display().to_string().leak();
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", dir_arg, "-o", out_arg],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        assert!(outdir.join("a.svg").exists());
        assert!(outdir.join("b.svg").exists());
    }

    #[test]
    fn directory_input_recurses_and_skips_non_png() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        std::fs::create_dir_all(tmp.path().join("nested")).expect("mkdir");
        std::fs::write(tmp.path().join("a.png"), png_bytes(true)).expect("png");
        std::fs::write(tmp.path().join("nested/b.PNG"), png_bytes(true)).expect("png upper");
        std::fs::write(tmp.path().join("notes.txt"), b"nope").expect("txt");
        let outdir = tmp.path().join("out");
        let dir_arg = tmp.path().display().to_string().leak();
        let out_arg = outdir.display().to_string().leak();
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", dir_arg, "-o", out_arg, "--stats"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        assert!(outdir.join("a.svg").exists());
        assert!(outdir.join("b.svg").exists());
        assert!(!outdir.join("notes.svg").exists());
        let err = String::from_utf8_lossy(&s.stderr);
        assert!(err.contains("decode") && err.contains("trace"));
    }

    #[test]
    fn mixed_batch_reports_failure_and_continues() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        std::fs::write(tmp.path().join("good.png"), png_bytes(true)).expect("png");
        std::fs::write(tmp.path().join("bad.png"), b"not a png").expect("bad");
        let outdir = tmp.path().join("out");
        let dir_arg = tmp.path().display().to_string().leak();
        let out_arg = outdir.display().to_string().leak();
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", dir_arg, "-o", out_arg],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_FAILURE);
        assert!(outdir.join("good.svg").exists());
        let err = String::from_utf8_lossy(&s.stderr);
        assert!(err.contains("bad.png"));
        assert!(err.contains("⚠ 1/2 converted"));
    }

    #[test]
    fn quiet_suppresses_summary_but_keeps_errors() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        std::fs::write(tmp.path().join("bad.png"), b"not a png").expect("bad");
        let dir_arg = tmp.path().display().to_string().leak();
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", dir_arg, "-q"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_FAILURE);
        let err = String::from_utf8_lossy(&s.stderr);
        assert!(!err.contains("converted"));
        assert!(err.contains("✗"));
    }

    #[test]
    fn missing_input_is_a_usage_error() {
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", "/definitely/not/here.png"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_USAGE);
        assert!(String::from_utf8_lossy(&s.stderr).contains("not a file or directory"));
    }

    #[test]
    fn stdin_cannot_be_combined_with_files() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let src = tmp.path().join("a.png");
        std::fs::write(&src, png_bytes(true)).expect("png");
        let src_arg = src.display().to_string();
        let mut s = streams(png_bytes(true));
        let code = run_args(
            &["wraith", "-", &src_arg],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_USAGE);
        assert!(String::from_utf8_lossy(&s.stderr).contains("cannot be combined"));
    }

    #[test]
    fn stdout_output_rejects_multiple_inputs() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        std::fs::write(tmp.path().join("a.png"), png_bytes(true)).expect("png");
        std::fs::write(tmp.path().join("b.png"), png_bytes(true)).expect("png");
        let dir_arg = tmp.path().display().to_string().leak();
        let mut s = streams(Vec::new());
        let code = run_args(
            &["wraith", dir_arg, "-o", "-"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_USAGE);
        assert!(String::from_utf8_lossy(&s.stderr).contains("exactly one input"));
    }

    #[test]
    fn unreadable_file_fails_gracefully() {
        let job = FileJob {
            name: String::from("ghost.png"),
            source: Source::Path(PathBuf::from("/definitely/not/here.png")),
            path: PathBuf::from("/definitely/not/here.svg"),
        };
        let results = execute(std::slice::from_ref(&job), Settings::default(), 0);
        assert_eq!(results.len(), 1);
        assert!(results[0].stats.is_none());
        assert!(
            results[0]
                .error
                .as_deref()
                .unwrap_or("")
                .contains("cannot read")
        );
    }

    #[test]
    fn sequential_and_parallel_batches_agree() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        std::fs::write(tmp.path().join("a.png"), png_bytes(true)).expect("png");
        std::fs::write(tmp.path().join("b.png"), png_bytes(true)).expect("png");
        let dir_arg = tmp.path().display().to_string().leak();

        let run_batch = |threads: &str| {
            let outdir = tempfile::tempdir().expect("tmpdir");
            let out_arg = outdir.path().display().to_string().leak();
            let mut s = streams(Vec::new());
            let t_arg = threads.to_owned().leak();
            let code = run_args(
                &["wraith", dir_arg, "-o", out_arg, "-j", t_arg],
                &mut s.stdin,
                &mut s.stdout,
                &mut s.stderr,
                false,
            );
            let a = std::fs::read(outdir.path().join("a.svg")).expect("a.svg");
            let b = std::fs::read(outdir.path().join("b.svg")).expect("b.svg");
            (code, a, b)
        };
        let (c1, a1, b1) = run_batch("1");
        let (c2, a2, b2) = run_batch("4");
        assert_eq!(c1, EXIT_OK);
        assert_eq!(c2, EXIT_OK);
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
    }

    #[test]
    fn quiet_success_is_silent_on_stderr() {
        let mut s = streams(png_bytes(true));
        let code = run_args(
            &["wraith", "-q", "-o", "-"],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        assert_eq!(s.stderr.len(), 0);
    }

    #[test]
    fn stdin_writes_to_explicit_file_output() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let dst = tmp.path().join("from-stdin.svg");
        let mut s = streams(png_bytes(true));
        let code = run_args(
            &["wraith", "-o", dst.to_str().expect("utf8")],
            &mut s.stdin,
            &mut s.stdout,
            &mut s.stderr,
            false,
        );
        assert_eq!(code, EXIT_OK);
        assert!(dst.exists());
        assert_eq!(s.stdout.len(), 0);
    }

    #[test]
    fn cli_settings_bridge_covers_all_variants() {
        let c = cli(&["x.png", "-m", "color", "-Q", "quality", "--minify", "off"]);
        let s = c.settings();
        assert_eq!(s.mode, wraith_core::Mode::Color);
        assert_eq!(s.quality, wraith_core::Quality::Quality);
        assert_eq!(s.minify, wraith_core::Minify::Off);
        assert_eq!(ModeArg::default(), ModeArg::Bw);
        assert_eq!(QualityArg::default(), QualityArg::Balanced);
        assert_eq!(MinifyArg::default(), MinifyArg::Default);
    }

    #[test]
    fn sibling_and_file_names_follow_the_source() {
        assert_eq!(
            sibling_svg(Path::new("/x/y/logo.png")),
            PathBuf::from("/x/y/logo.svg")
        );
        assert_eq!(file_name_svg(Path::new("/x/y/logo.png")), "logo.svg");
        // Hidden files keep their whole name as the stem, per
        // `std::path::Path::file_stem` semantics.
        assert_eq!(file_name_svg(Path::new("/x/y/.png")), ".png.svg");
        assert_eq!(file_name_svg(Path::new("")), "out.svg");
    }
}
