# wraith

> **Raster in. Wraith out.**
> Blazing-fast white-background PNG → optimised SVG.

`wraith` erases the white and keeps only the shape: PNG bytes go in, a
minimal, fully optimised SVG comes out — with per-stage timings if you want
them.

```text
  PNG ──▶ decode ──▶ preprocess ──▶ trace ──▶ serialize ──▶ minify ──▶ SVG
                    (threshold)    (vtracer) (canonical)    (OXVG)
```

```console
$ wraith logo.png --stats
✓ 1/1 converted in 6.4ms
  ✓ logo.png  3289 → 765 B  (-76.7%)  6.3ms [decode 0.33ms, preprocess 0.71ms, trace 5.32ms, serialize 0.05ms, minify 0.14ms]
```

## Why it's fast

* **In-memory pipeline end to end.** The PNG is decoded once, the RGBA buffer
  is *moved* (not copied) into the tracer, and the traced SVG is serialized
  straight into one pre-sized `String`. Zero intermediate files, zero
  redundant allocations.
* **`vtracer` for tracing** — O(n) cluster tracing, Rust-native. In
  black-and-white mode the white background *never becomes geometry*.
* **A purpose-built canonical serializer** replaces vtracer's own writer:
  hoists shared fills onto one `<g>`, drops `transform` when it's the origin,
  skips empty paths, controls coordinate precision.
* **OXVG (native SVGO replacement) for minification** — measured cost on a
  512×512 trace: ~0.14 ms for another ~25–30 % byte reduction. Default on.
* **usvg normalisation was evaluated and cut.** Traced SVGs are already
  canonical; a second parse/normalise/serialize round-trip added latency for
  zero size win. OXVG's `applyTransforms` covers the transform baking.
* **Rayon-parallel batches.** `wraith dir/ -o out/` converts every PNG on all
  cores, results reported in input order.
* **Tuned release profile** — fat LTO, one codegen unit, panic=abort, stripped
  binary: **24 % faster** than a stock release build and 3× smaller
  (30.8 MB → 10.0 MB), byte-identical output.

## Benchmarks

Sandbox: 2-core Linux x86_64, release build, generated fixtures (solid shapes
on white), criterion 0.8.

| benchmark | time |
| --- | --- |
| decode PNG 512×512 | 0.32 ms |
| preprocess + luminance threshold | 0.71 ms |
| trace BW 512×512 | 5.8–6.1 ms |
| trace color 512×512 | 33.6 ms |
| serialize 1024×1024 (wraith) | 70 µs |
| minify BW 512×512 (OXVG default) | 0.14 ms |
| **full pipeline BW 512×512** | **6.4 ms** |
| **full pipeline BW 1024×1024** | **26.9 ms** |
| **full pipeline color 512×512 + strip-white** | **32.7 ms** |
| batch 8×512 sequential | 50.1 ms |
| batch 8×512 rayon (2 cores) | 33.8 ms — **1.47×** |
| release profile: vanilla vs tuned (8×256 batch, real CLI) | 15.3 ms → **11.6 ms** |

Reproduce with `cargo bench -p wraith-core`. Per-file, per-stage timings are
available in the CLI itself via `--stats`.

## Install

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

The pipeline assumes art on a plain white background. Defaults are tuned for
line art and logos:

| flag | default | meaning |
| --- | --- | --- |
| `-m, --mode <bw\|color>` | `bw` | threshold to pure black/white, or trace every colour cluster |
| `-Q, --quality <fast\|balanced\|quality>` | `balanced` | tracer simplification trade-off |
| `-t, --threshold <0-255>` | tracer's own rule | luminance threshold applied before tracing (bw) |
| `--strip-white` | off | in color mode, drop every pure-white path |
| `--minify <default\|safe\|off>` | `default` | OXVG job set (`safe` fail-fasts on scripts/animations) |
| `-p, --precision <1-8>` | `2` | decimal digits kept for path coordinates |
| `-j, --jobs <N>` | all cores | parallel conversions for batch input |
| `--stats` / `--json` | off | per-file byte counts and stage timings on **stderr** |
| `-q, --quiet` | off | suppress the summary; failures are always printed |

Exit codes: `0` success · `1` conversion failure · `2` usage error.

## Library

Everything the CLI does is a library call:

```rust
use wraith_core::{convert, Settings};

let png = std::fs::read("input.png")?;
let out = convert(&png, &Settings::new())?;
std::fs::write("output.svg", out.svg)?;
println!("saved {:.1}%", out.stats.reduction() * 100.0);
```

`wraith-core` is organised as four stages (`preprocess`, `trace`, `svgout`,
`minify`), each independently callable and independently timed through
`Stopwatch`.

## Workspace layout

```text
crates/
├── wraith-core/     # pipeline library: decode → trace → serialize → minify
└── wraith/          # CLI: clap args, job resolution, rayon batch, reports
```

## Quality gates

* `cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic` — zero warnings
* `cargo llvm-cov --workspace --fail-under-lines 100` — **100 % line and function coverage**
  (the two-line binary shim in `crates/wraith/src/main.rs` is excluded)
* criterion benchmarks run in CI compile check
* conventional commits enforced through [git-cliff](https://git-cliff.org) for the changelog

## License

MIT OR Apache-2.0
