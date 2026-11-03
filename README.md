<div align="center">
  <img src="docs/logo.png" alt="Wraith Logo" width="300"/>
  <p>
    <strong>Raster in. Wraith out.</strong><br/>
    Blazing-fast white-background PNG → optimised SVG. Erases the white and keeps only the shape, with per-stage timings.
  </p>
  <p>
    <img src="https://img.shields.io/badge/Rust-1.98.1%2B%20%7C%202024-000000?style=flat-square&logo=rust" alt="Rust"/>
    <img src="https://img.shields.io/badge/License-MIT%2FApache--2.0-blue?style=flat-square" alt="License MIT/Apache-2.0"/>
    <img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square&logo=githubactions" alt="Build"/>
    <img src="https://img.shields.io/badge/Coverage-100%25-brightgreen?style=flat-square" alt="Coverage"/>
    <img src="https://img.shields.io/badge/Pipeline-In--Memory-FF4500?style=flat-square" alt="In-Memory Pipeline"/>
    <img src="https://img.shields.io/badge/Tracer-vtracer-6A0DAD?style=flat-square" alt="vtracer"/>
    <img src="https://img.shields.io/badge/Minifier-OXVG-007ACC?style=flat-square" alt="OXVG"/>
    <img src="https://img.shields.io/badge/Parallel-Rayon-228B22?style=flat-square" alt="Rayon"/>
  </p>
</div>

---

## Overview

`wraith` erases the white and keeps only the shape: PNG bytes go in, a minimal, fully optimised SVG comes out — with per-stage timings if you want them.

```text
PNG ──▶ decode ──▶ preprocess ──▶ trace ──▶ serialize ──▶ minify ──▶ SVG
              (threshold)    (vtracer) (canonical)    (OXVG)
```

```console
$ wraith logo.png --stats
✓ 1/1 converted in 6.4ms
  ✓ logo.png  3289 → 765 B  (-76.7%)  6.3ms [decode 0.33ms, preprocess 0.71ms, trace 5.32ms, serialize 0.05ms, minify 0.14ms]
```

## Features

- **In-memory pipeline end to end:** The PNG is decoded once, the RGBA buffer is *moved* (not copied) into the tracer, and the traced SVG is serialized straight into one pre-sized `String`. Zero intermediate files, zero redundant allocations.
- **`vtracer` for tracing:** O(n) cluster tracing, Rust-native. In black-and-white mode, the white background *never becomes geometry*.
- **Purpose-built canonical serializer:** Replaces vtracer's own writer: hoists shared fills onto one `<g>`, drops `transform` when it's the origin, skips empty paths, and controls coordinate precision.
- **OXVG for minification:** Native SVGO replacement. Measured cost on a 512×512 trace: ~0.14 ms for another ~25–30% byte reduction. Default on.
- **Rayon-parallel batches:** `wraith dir/ -o out/` converts every PNG on all cores, with results reported in input order.
- **Tuned release profile:** Fat LTO, one codegen unit, `panic=abort`, stripped binary: **24% faster** than a stock release build and 3× smaller (30.8 MB → 10.0 MB), with byte-identical output.

## Benchmarks

Sandbox: 2-core Linux x86_64, release build, generated fixtures (solid shapes on white), criterion 0.8.

| Benchmark | Time |
| --- | --- |
| Decode PNG 512×512 | 0.32 ms |
| Preprocess + luminance threshold | 0.71 ms |
| Trace BW 512×512 | 5.8–6.1 ms |
| Trace color 512×512 | 33.6 ms |
| Serialize 1024×1024 (wraith) | 70 µs |
| Minify BW 512×512 (OXVG default) | 0.14 ms |
| **Full pipeline BW 512×512** | **6.4 ms** |
| **Full pipeline BW 1024×1024** | **26.9 ms** |
| **Full pipeline color 512×512 + strip-white** | **32.7 ms** |
| Batch 8×512 sequential | 50.1 ms |
| Batch 8×512 rayon (2 cores) | 33.8 ms — **1.47×** |
| Release profile: vanilla vs tuned (8×256 batch, real CLI) | 15.3 ms → **11.6 ms** |

> [!TIP]
> Reproduce with `cargo bench -p wraith-core`. Per-file, per-stage timings are available in the CLI itself via `--stats`.

## Installation

```sh
cargo install --path crates/wraith   # from a checkout
# or grab a prebuilt binary from GitHub Releases
```

## Usage

```sh
wraith logo.png                    # → logo.svg next to it
wraith logo.png -o out.svg         # explicit output
wraith assets/*.png -o dist/       # batch into a directory (parallel)
cat logo.png | wraith -o -         # stdin → stdout
wraith dir/ --stats                # recursive scan, per-stage timings
wraith dir/ --json -o out/         # machine-readable stats on stderr
wraith completions bash >> ~/.bashrc
```

> [!NOTE]
> The pipeline assumes art on a plain white background. Defaults are tuned for line art and logos.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --mode <bw\|color>` | `bw` | Threshold to pure black/white, or trace every colour cluster |
| `-Q, --quality <fast\|balanced\|quality>` | `balanced` | Tracer simplification trade-off |
| `-t, --threshold <0-255>` | tracer's rule | Luminance threshold applied before tracing (bw) |
| `--strip-white` | off | In color mode, drop every pure-white path |
| `--minify <default\|safe\|off>` | `default` | OXVG job set (`safe` fail-fasts on scripts/animations) |
| `-p, --precision <1-8>` | `2` | Decimal digits kept for path coordinates |
| `-j, --jobs <N>` | all cores | Parallel conversions for batch input |
| `--stats` / `--json` | off | Per-file byte counts and stage timings on **stderr** |
| `-q, --quiet` | off | Suppress the summary; failures are always printed |

> [!IMPORTANT]
> Exit codes: `0` success · `1` conversion failure · `2` usage error.

## Library

Everything the CLI does is available as a library call:

```rust
use wraith_core::{convert, Settings};

let png = std::fs::read("input.png")?;
let out = convert(&png, &Settings::new())?;
std::fs::write("output.svg", out.svg)?;
println!("saved {:.1}%", out.stats.reduction() * 100.0);
```

`wraith-core` is organised as four stages (`preprocess`, `trace`, `svgout`, `minify`), each independently callable and independently timed through `Stopwatch`.

## Workspace Layout

```text
crates/
├── wraith-core/     # Pipeline library: decode → trace → serialize → minify
└── wraith/          # CLI: clap args, job resolution, rayon batch, reports
```

## Quality Gates

- **Clippy:** `cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic` (zero warnings)
- **Coverage:** `cargo llvm-cov --workspace --fail-under-lines 100` (**100% line and function coverage**, excluding the two-line binary shim in `crates/wraith/src/main.rs`)
- **Benchmarks:** Criterion benchmarks run in CI compile check
- **Commits:** Conventional commits enforced for a clean, automated history
