<div align="center">
  <img src="docs/logo.png" alt="Wraith Logo" width="300"/>
  <p>
    <strong>Raster in. Wraith out.</strong><br/>
    Blazing-fast PNG → optimised SVG. Sub-pixel geometric colour engine, gradient fitting and alpha-safe backgrounds — with per-stage timings.
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

`wraith` turns PNG bytes into a minimal, fully optimised SVG — with per-stage timings if you want them. Two colour engines:

```text
PNG ──▶ decode ──▶ preprocess ──▶ ┌─ Mode::BlackWhite ─▶ vtracer (threshold) ─┐
                                 └─ Mode::Color ──────▶ geometric engine      ├─▶ serialize ─▶ minify ─▶ SVG
                                                       (sub-pixel coverage)  ┘   (canonical)  (OXVG)
```

```console
$ wraith logo.png --stats
✓ 1/1 converted in 6.4ms
  ✓ logo.png  3289 → 765 B  (-76.7%)  6.3ms [decode 0.33ms, preprocess 0.71ms, trace 5.32ms, serialize 0.05ms, minify 0.14ms]
$ wraith -m color art.png --stats
✓ 1/1 converted in 31.9ms
  ✓ art.png  15.2 kB → 4.3 kB  10 paths, palette 6  31.8ms [decode 0.4ms, palette 2.1ms, trace 29.2ms, …]
```

## Features

- **In-memory pipeline end to end:** The PNG is decoded once, the RGBA buffer is *moved* (not copied) into the tracer, and the traced SVG is serialized straight into one pre-sized `String`. Zero intermediate files, zero redundant allocations.
- **Sub-pixel geometric colour engine (default for `-m color`):** recovers anti-aliased coverage from the raster, places boundaries at sub-pixel precision, fits error-bounded Bézier curves with corner reconstruction, and composes layers with trapping underlays — measured at **2.2–3.5× lower error** than the legacy tracer at 4× zoom on flat art, with **30–45% smaller** files (`docs/multicolor-roadmap.md`).
- **Linear + radial gradient fitting:** over-segmented palette bands are merged into gradient regions when an affine/radial colour model explains them within a 2/255 residual — the 512² gradient fixture fits as **one `<linearGradient>`, one `<radialGradient>` and 19 paths at 18.8 kB**, where flat banding needs 1,575 paths and 226 kB.
- **Primitives:** circles, ellipses and rounded rects are detected on rings and emitted as `<circle>`/`<ellipse>`/`<rect rx>` (≈ 40 bytes vs 150–400 for Béziers).
- **`vtracer` stays for black-and-white** (and `--engine legacy` for byte-compatible 0.1.x colour output). In BW mode the white background *never becomes geometry*.
- **Alpha-first backgrounds:** `--background alpha` treats the PNG alpha channel as authoritative; `--background #ffffff` flood-removes border-connected colour backgrounds — replacing 0.1.x's `--strip-white` colour-equality hack.
- **Purpose-built canonical serializer:** hoists shared fills onto one `<g>`, drops `transform` when it's the origin, skips empty paths, and controls coordinate precision.
- **OXVG for minification:** Native SVGO replacement. Measured cost on a 512×512 trace: ~0.14 ms for another ~25–30% byte reduction. Default on.
- **Rayon-parallel batches:** `wraith dir/ -o out/` converts every PNG on all cores, with results reported in input order.
- **Tuned release profile:** Fat LTO, one codegen unit, `panic=abort`, stripped binary.

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
wraith -m color logo.png           # sub-pixel geometric colour trace
wraith --color --preset icon i.png # icon-tuned knobs (shorthand: --color)
wraith -m color --engine legacy a.png   # byte-compatible 0.1.x vtracer output
wraith --background alpha a.png    # alpha channel is authoritative
wraith --background '#fafafa' a.png # flood-remove a colour background
wraith logo.png -o out.svg         # explicit output
wraith assets/*.png -o dist/       # batch into a directory (parallel)
cat logo.png | wraith -o -         # stdin → stdout
wraith dir/ --stats                # recursive scan, per-stage timings
wraith dir/ --json -o out/         # machine-readable stats on stderr
wraith completions bash >> ~/.bashrc
```

> [!NOTE]
> `-m bw` (the default) assumes art on a plain white background; `-m color` handles any raster, including transparency.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m, --mode <bw\|color>`, `--color` | `bw` | Threshold to pure black/white, or trace every colour region |
| `--engine <geometric\|legacy>` | `geometric` | Colour-mode engine (`legacy` = 0.1.x vtracer) |
| `--preset <logo\|icon\|illustration>` | `logo` | Content-tuned knob layer; explicit flags win |
| `--colors <auto\|1-64>` | `auto` | Geometric engine palette cap |
| `--tolerance <px>` | `0.3` | Curve-fit error bound (Schneider tolerance) |
| `--corner-angle <deg>` | `40` | Minimum turn recognised as a corner |
| `--trap <px>` | `1` | Trapping underlay width (0 disables) |
| `--merge <dOklab>` | `0.02` | Palette merge threshold |
| `--min-area <px²>` | `2` | Speck filter floor |
| `--gradients <off\|linear\|linear-radial>` | `linear-radial` | Gradient detection/fitting |
| `--no-primitives` | off | Disable circle/ellipse/rect emission |
| `--refine <0-30>` | `0` | Field-driven refinement iterations |
| `--background <none\|alpha\|#RRGGBB>` | `none` | Background removal policy (replaces 0.1.x `--strip-white`) |
| `--bg-tolerance <0-255>` | `8` | Per-channel tolerance for `#RRGGBB` backgrounds |
| `--color-precision` / `--layer-difference` / `--hierarchy` | 6 / 16 / stacked | `--engine legacy` colour knobs |
| `-Q, --quality <fast\|balanced\|quality>` | `balanced` | Legacy tracer simplification trade-off |
| `-t, --threshold <0-255>` | tracer's rule | Luminance threshold applied before tracing (bw) |
| `--minify <default\|safe\|off>` | `default` | OXVG job set (`safe` fail-fasts on scripts/animations) |
| `-p, --precision <1-8>` | `2` | Decimal digits kept for path coordinates |
| `-j, --jobs <N>` | all cores | Parallel conversions for batch input |
| `--stats` / `--json` | off | Byte counts, stage timings, palette size and path count on **stderr** |
| `-q, --quiet` | off | Suppress the summary; failures are always printed |

> [!IMPORTANT]
> Exit codes: `0` success · `1` conversion failure · `2` usage error.

## Library

Everything the CLI does is available as a library call:

```rust
use wraith_core::{convert, Mode, Settings};

let png = std::fs::read("input.png")?;
let settings = Settings {
    mode: Mode::Color,
    ..Settings::new()
};
let out = convert(&png, &settings)?;
std::fs::write("output.svg", out.svg)?;
println!(
    "{} paths, palette {}",
    out.stats.path_count, out.stats.palette_size
);
```

The geometric engine is organised as `palette → model → gradients → layers → contour → fit → refine → primitives → write`, each stage independently callable and timed through `Stopwatch`. Colour knobs live in `wraith_core::ColorSettings` (roadmap M7 redesign; 0.1.x's `Settings::strip_white` is gone — `Background` replaces it).

## Evaluation Harness

`wraith-eval` rasterises ground-truth SVG fixtures with resvg, runs every engine, and scores RMSE (1×/4×), the share of large-error pixels, mean ΔE(Oklab), SSIM, bytes and per-stage time:

```sh
cargo run --release -p wraith-eval -- run            # print the score table
cargo run --release -p wraith-eval -- check docs/baseline.json   # CI gate
```

CI compares against the committed `docs/baseline.json`; a >5 % RMSE or byte regression fails the check (roadmap M0: truth first).

## Workspace Layout

```text
crates/
├── wraith-core/     # Pipeline library: palette → model → gradients → layers → contour → fit → write
├── wraith/          # CLI: clap args, engine dispatch, rayon batch, reports
└── wraith-eval/     # Ground-truth evaluation harness (M0 truth first)
```

## Quality Gates

- **Clippy:** `cargo clippy --workspace --all-targets --all-features -- -D warnings` with `[workspace.lints.clippy] pedantic = warn` and every exception documented in `Cargo.toml` (zero inline `allow`s)
- **Coverage:** `cargo llvm-cov --workspace --fail-under-lines 100` (**100% line and function coverage**, excluding the two-line binary shim in `crates/wraith/src/main.rs`)
- **Quality:** `wraith-eval check docs/baseline.json` in CI fails on RMSE/byte regressions
- **Benchmarks:** Criterion benchmarks run in CI compile check
- **Commits:** Conventional commits enforced for a clean, automated history
