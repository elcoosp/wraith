//! End-to-end and per-stage benchmarks of the wraith pipeline.
//!
//! Fixtures are generated in-process (white background, solid shapes), so
//! numbers are reproducible without binary artefacts. Run with:
//!
//! ```sh
//! cargo bench -p wraith-core
//! ```

use std::hint::black_box;
use std::sync::LazyLock;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use wraith_core::settings::{Minify, Mode, Quality, Settings};
use wraith_core::{Stopwatch, convert, svgout};

/// 512×512 white background with three solid colour shapes.
static IMG_512: LazyLock<Vec<u8>> = LazyLock::new(|| fixture_png(512, false));
/// 512×512 black-and-white line-art style fixture.
static IMG_512_BW: LazyLock<Vec<u8>> = LazyLock::new(|| fixture_png(512, true));
/// 1024×1024 for a heavier trace.
static IMG_1024_BW: LazyLock<Vec<u8>> = LazyLock::new(|| fixture_png(1024, true));
/// Eight 512×512 images for batch-scaling benches.
static BATCH_8: LazyLock<Vec<Vec<u8>>> = LazyLock::new(|| {
    (0..8)
        .map(|i| fixture_png(512, i % 2 == 0))
        .collect::<Vec<_>>()
});

/// White background; `bw` adds a black ring + square, colour adds three
/// coloured shapes.
fn fixture_png(size: u32, bw: bool) -> Vec<u8> {
    let mut img = image::ImageBuffer::from_pixel(size, size, image::Rgba([255, 255, 255, 255]));
    let s = f64::from(size);
    if bw {
        let r = s * 0.35;
        let cx = s / 2.0;
        let cy = s / 2.0;
        for y in 0..size {
            for x in 0..size {
                let d = ((f64::from(x) - cx).powi(2) + (f64::from(y) - cy).powi(2)).sqrt();
                if (d < r && d > r * 0.8)
                    || (f64::from(x) > s * 0.1
                        && f64::from(x) < s * 0.3
                        && f64::from(y) > s * 0.1
                        && f64::from(y) < s * 0.3)
                {
                    img.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
                }
            }
        }
    } else {
        for y in 0..size {
            for x in 0..size {
                let px = if (f64::from(x) - s / 2.0).powi(2) + (f64::from(y) - s / 2.0).powi(2)
                    < (s * 0.4).powi(2)
                {
                    image::Rgba([30, 90, 200, 255])
                } else if f64::from(x) > s * 0.2
                    && f64::from(x) < s * 0.5
                    && f64::from(y) > s * 0.2
                    && f64::from(y) < s * 0.5
                {
                    image::Rgba([220, 40, 40, 255])
                } else if f64::from(x) < s / 3.0 && f64::from(y) < s / 3.0 {
                    image::Rgba([250, 210, 60, 255])
                } else {
                    image::Rgba([255, 255, 255, 255])
                };
                img.put_pixel(x, y, px);
            }
        }
    }
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("fixture encodes");
    buf
}

fn bench_decode(c: &mut Criterion) {
    let png = &*IMG_512_BW;
    c.bench_function("decode/512bw", |b| {
        b.iter(|| image::load_from_memory(black_box(png)).expect("decodes"));
    });
}

fn bench_preprocess(c: &mut Criterion) {
    let png = &*IMG_512_BW;
    let settings = Settings {
        threshold: Some(128),
        ..Settings::new()
    };
    c.bench_function("preprocess/512bw+threshold", |b| {
        b.iter(|| {
            let mut sw = Stopwatch::default();
            let prepared = wraith_core::preprocess::prepare(black_box(png), &settings, &mut sw)
                .expect("prepares");
            black_box(prepared.image.pixels.len());
        });
    });
}

fn bench_trace_bw(c: &mut Criterion) {
    let settings = Settings::new();
    let mut sw = Stopwatch::default();
    let prepared =
        wraith_core::preprocess::prepare(&IMG_512_BW, &settings, &mut sw).expect("prepares");
    c.bench_function("trace/bw-512", |b| {
        b.iter(|| {
            let mut sw = Stopwatch::default();
            let svg = wraith_core::trace::trace(prepared.image.clone(), &settings, &mut sw)
                .expect("traces");
            black_box(svg.paths.len());
        });
    });
}

fn bench_trace_color(c: &mut Criterion) {
    let settings = Settings {
        mode: Mode::Color,
        ..Settings::new()
    };
    let mut sw = Stopwatch::default();
    let prepared =
        wraith_core::preprocess::prepare(&IMG_512, &settings, &mut sw).expect("prepares");
    c.bench_function("trace/color-512", |b| {
        b.iter(|| {
            let mut sw = Stopwatch::default();
            let svg = wraith_core::trace::trace(prepared.image.clone(), &settings, &mut sw)
                .expect("traces");
            black_box(svg.paths.len());
        });
    });
}

fn bench_serialize_vs_vtracer_display(c: &mut Criterion) {
    let settings = Settings::new();
    let mut sw = Stopwatch::default();
    let prepared =
        wraith_core::preprocess::prepare(&IMG_1024_BW, &settings, &mut sw).expect("prepares");
    let traced = wraith_core::trace::trace(prepared.image, &settings, &mut sw).expect("traces");

    let mut group = c.benchmark_group("serialize/1024bw");
    group.bench_function("vtracer_display", |b| {
        b.iter(|| black_box(format!("{}", black_box(&traced))));
    });
    group.bench_function("wraith_serialize", |b| {
        b.iter(|| {
            let mut sw = Stopwatch::default();
            black_box(svgout::serialize(
                black_box(&traced),
                false,
                Some(2),
                &mut sw,
            ))
        });
    });
    group.finish();
}

fn bench_minify_variants(c: &mut Criterion) {
    let canonical = convert(
        &IMG_512_BW,
        &Settings {
            minify: Minify::Off,
            ..Settings::new()
        },
    )
    .expect("converts")
    .svg;

    let mut group = c.benchmark_group("minify/bw-512");
    group.bench_function("off", |b| {
        b.iter(|| black_box(&canonical).len());
    });
    group.bench_function("safe", |b| {
        b.iter(|| {
            let mut sw = Stopwatch::default();
            black_box(
                wraith_core::minify::minify(black_box(&canonical), Minify::Safe, &mut sw)
                    .expect("minifies")
                    .len(),
            )
        });
    });
    group.bench_function("default", |b| {
        b.iter(|| {
            let mut sw = Stopwatch::default();
            black_box(
                wraith_core::minify::minify(black_box(&canonical), Minify::Default, &mut sw)
                    .expect("minifies")
                    .len(),
            )
        });
    });
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");
    group.bench_function("bw-512-balanced", |b| {
        b.iter(|| black_box(convert(black_box(&IMG_512_BW), &Settings::new()).expect("converts")));
    });
    group.bench_function("bw-1024-balanced", |b| {
        b.iter(|| black_box(convert(black_box(&IMG_1024_BW), &Settings::new()).expect("converts")));
    });
    group.bench_function("bw-512-fast", |b| {
        b.iter(|| {
            black_box(
                convert(
                    black_box(&IMG_512_BW),
                    &Settings {
                        quality: Quality::Fast,
                        ..Settings::new()
                    },
                )
                .expect("converts"),
            )
        });
    });
    group.bench_function("color-512-balanced+strip", |b| {
        b.iter(|| {
            black_box(
                convert(
                    black_box(&IMG_512),
                    &Settings {
                        mode: Mode::Color,
                        strip_white: true,
                        ..Settings::new()
                    },
                )
                .expect("converts"),
            )
        });
    });
    group.finish();
}

fn bench_batch_scaling(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("batch/8x512");
    group.sample_size(10);
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let settings = Settings::new();
            for png in black_box(&*BATCH_8) {
                black_box(convert(png, &settings).expect("converts"));
            }
        });
    });
    group.bench_function("rayon-threads", |b| {
        b.iter(|| {
            let settings = Settings::new();
            black_box(&*BATCH_8)
                .par_iter()
                .map(|png| convert(png, &settings).expect("converts"))
                .count()
        });
    });
    group.finish();
}

fn bench_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision/bw-512");
    for p in [1u8, 2, 4] {
        group.bench_with_input(criterion::BenchmarkId::from_parameter(p), &p, |b, &p| {
            let settings = Settings::new().with_precision(p);
            b.iter(|| black_box(convert(black_box(&IMG_512_BW), &settings).expect("converts")));
        });
    }
    group.finish();
}

#[allow(clippy::too_many_lines)]
fn benches() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_millis(500))
}

criterion_group!(
    name = benches_all;
    config = benches();
    targets =
        bench_decode,
        bench_preprocess,
        bench_trace_bw,
        bench_trace_color,
        bench_serialize_vs_vtracer_display,
        bench_minify_variants,
        bench_full_pipeline,
        bench_batch_scaling,
        bench_precision
);
criterion_main!(benches_all);
