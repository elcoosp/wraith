//! Geometric colour engine (roadmap M1–M5).
//!
//! PNG pixels in, layered sub-pixel SVG out. The pipeline, in order:
//!
//! 1. [`palette`] — unique colours → Oklab k-means palette → merge →
//!    anti-aliasing ghost pruning (the tested prototype, productionised);
//! 2. [`model`] — per-unique-colour `(a, b, t)` mixture in gamma-encoded
//!    sRGB, so `t` equals the true pixel coverage an ordinary 2-D rasteriser
//!    blended;
//! 3. [`gradients`] — over-segmented bands merged into linear/radial gradient
//!    regions by an MDL-style greedy merge (M3);
//! 4. [`layers`] — per-region soft coverage fields with trapping underlays
//!    and bounding boxes, built by a single scatter pass (M2);
//! 5. [`contour`] — marching squares at iso 0.5 with linear interpolation →
//!    sub-pixel rings, scanned tile-actively (M2);
//! 6. [`fit`] — corner reconstruction, TLS line snapping, error-bounded
//!    Schneider cubic fitting with G1 joins;
//! 7. [`refine`] — optional field-driven control-point refinement (M5);
//! 8. [`primitives`] — circle/ellipse/rounded-rect/arc detection (M4);
//! 9. [`write`] — relative-coordinate writer at integer hundredths.

pub mod contour;
pub mod fit;
pub mod gradients;
pub mod layers;
pub mod model;
pub mod palette;
pub mod primitives;
pub mod refine;
pub mod write;

use crate::settings::{Background, ColorSettings, Gradients};
use crate::stats::{Stage, Stopwatch};

/// One fitted segment of a ring, in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathSeg {
    /// Straight line to `p`.
    Line {
        /// Segment end point.
        p: [f64; 2],
    },
    /// Cubic Bézier to `p` with interior control points `c1`, `c2`.
    Cubic {
        /// First control point.
        c1: [f64; 2],
        /// Second control point.
        c2: [f64; 2],
        /// Segment end point.
        p: [f64; 2],
    },
    /// Elliptical arc to `p` (SVG `A`/`a` command).
    Arc {
        /// x radius.
        rx: f64,
        /// y radius.
        ry: f64,
        /// x-axis rotation in degrees.
        rot_deg: f64,
        /// Large-arc flag.
        large_arc: bool,
        /// Sweep flag.
        sweep: bool,
        /// Segment end point.
        p: [f64; 2],
    },
}

/// A closed region's geometry: either a compound path or a detected
/// primitive (roadmap M4).
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// One closed subpath (absolute coordinates; the writer relativises).
    Path {
        /// Ring start point.
        start: [f64; 2],
        /// Segments following `start`; the writer closes with `z`.
        segs: Vec<PathSeg>,
    },
    /// Axis-aligned circle.
    Circle {
        /// Centre x.
        cx: f64,
        /// Centre y.
        cy: f64,
        /// Radius.
        r: f64,
    },
    /// Axis-aligned ellipse.
    Ellipse {
        /// Centre x.
        cx: f64,
        /// Centre y.
        cy: f64,
        /// x radius.
        rx: f64,
        /// y radius.
        ry: f64,
    },
    /// Axis-aligned (optionally rounded) rectangle.
    Rect {
        /// Left edge.
        x: f64,
        /// Top edge.
        y: f64,
        /// Width.
        w: f64,
        /// Height.
        h: f64,
        /// Corner radius (x and y).
        r: f64,
    },
}

/// Fill of one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Paint {
    /// Opaque sRGB colour.
    Solid([u8; 3]),
    /// Reference into [`ColorTrace::gradients`].
    Gradient(usize),
}

/// A fitted paint gradient (roadmap M3).
#[derive(Debug, Clone, PartialEq)]
pub struct GradientDef {
    /// Geometry of the gradient axis.
    pub kind: GradientKind,
    /// Stops as `(offset 0..=1, sRGB colour)`, sorted by offset.
    pub stops: Vec<(f32, [u8; 3])>,
}

/// Geometry of a gradient axis in user space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientKind {
    /// Linear gradient from `(x1, y1)` (offset 0) to `(x2, y2)` (offset 1).
    Linear {
        /// Start point.
        x1: f64,
        /// Start point.
        y1: f64,
        /// End point.
        x2: f64,
        /// End point.
        y2: f64,
    },
    /// Radial gradient centred at `(cx, cy)` with radius `r`.
    Radial {
        /// Centre x.
        cx: f64,
        /// Centre y.
        cy: f64,
        /// Radius.
        r: f64,
    },
}

/// One painted layer of the geometric output, in bottom-to-top order.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorLayer {
    /// Fill paint (solid colour or gradient reference).
    pub paint: Paint,
    /// Geometry painted with [`ColorLayer::paint`].
    pub shapes: Vec<Shape>,
}

/// The full geometric trace of one raster: layers plus gradient defs.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorTrace {
    /// Raster width in pixels.
    pub width: usize,
    /// Raster height in pixels.
    pub height: usize,
    /// Layers in paint order (bottom first).
    pub layers: Vec<ColorLayer>,
    /// Gradient definitions referenced by [`Paint::Gradient`].
    pub gradients: Vec<GradientDef>,
    /// Number of palette entries the engine settled on.
    pub palette_size: usize,
}

/// Runs the geometric colour engine over a prepared RGBA raster.
///
/// # Panics
///
/// Only if the raster dimensions disagree with the pixel buffer length, which
/// `prepare` guarantees they cannot.
#[must_use]
pub fn trace_color(
    image: &vtracer::ColorImage,
    settings: &ColorSettings,
    background: Background,
    sw: &mut Stopwatch,
) -> ColorTrace {
    let (w, h) = (image.width, image.height);
    let over_segmented = settings.gradients != Gradients::Off;

    let filtered_buf;
    let work_pixels: &[u8] = if settings.prefilter && w >= 3 && h >= 3 {
        let pass1 = conditional_median_rgba(&image.pixels, w, h);
        filtered_buf = conditional_median_rgba(&pass1, w, h);
        &filtered_buf
    } else {
        &image.pixels
    };

    // 1) unique colours + palette (over-segmented when gradient detection is
    // on, so ramps become chains of bands — roadmap M3 step 1)
    let uniques = palette::unique_colors(work_pixels, w);
    let pal = palette::build_palette(&uniques, w, h, settings, over_segmented);
    sw.lap(Stage::Palette);

    // 2) mixture model per unique colour
    let mix = model::build_model(&uniques, &pal);

    // 3) gradient region detection (M3)
    let regions = match settings.gradients {
        Gradients::Off => {
            let counts = gradients::rank_pixel_counts(&uniques, work_pixels, &mix);
            gradients::flat_regions(pal.len(), &counts)
        }
        Gradients::Linear | Gradients::LinearRadial => {
            gradients::detect_regions(&uniques, work_pixels, w, h, &pal, &mix, settings)
        }
    };

    // 5) per-unit fields → contours → fit → refine → primitives (paint
    // units and their order are derived inside `build_layers`)
    let background_rect = background == Background::None && !uniques.any_alpha;
    let layers = layers::build_layers(
        &uniques,
        work_pixels,
        &pal,
        &mix,
        &regions,
        w,
        h,
        settings,
        background_rect,
        sw,
    );
    let palette_size = pal.len();
    ColorTrace {
        width: w,
        height: h,
        layers,
        gradients: regions.gradient_defs,
        palette_size,
    }
}

/// Edge-preserving conditional median filter (3×3 in Oklab).
fn conditional_median_rgba(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const MAX_D2: f32 = 0.045 * 0.045;
    let mut dst = src.to_vec();
    let lab: Vec<palette::V3> = src
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&[r, g, b, _]| palette::rgb_to_oklab([r, g, b]))
        .collect();
    let (mut r_buf, mut g_buf, mut b_buf) = ([0_u8; 9], [0_u8; 9], [0_u8; 9]);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            if src[i * 4 + 3] != 255 {
                continue;
            }
            let c0 = lab[i];
            let mut count = 0_usize;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let ni = y.wrapping_add_signed(dy) * w + x.wrapping_add_signed(dx);
                    if src[ni * 4 + 3] == 255 && palette::d2(c0, lab[ni]) <= MAX_D2 {
                        r_buf[count] = src[ni * 4];
                        g_buf[count] = src[ni * 4 + 1];
                        b_buf[count] = src[ni * 4 + 2];
                        count += 1;
                    }
                }
            }
            if count >= 5 {
                r_buf[..count].sort_unstable();
                g_buf[..count].sort_unstable();
                b_buf[..count].sort_unstable();
                let mid = count / 2;
                dst[i * 4] = r_buf[mid];
                dst[i * 4 + 1] = g_buf[mid];
                dst[i * 4 + 2] = b_buf[mid];
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use crate::settings::{Background, ColorSettings};

    use super::*;

    /// Solid red square on white: engine must produce a background layer plus
    /// a red shape layer, with every colour's geometry present.
    #[test]
    fn trace_color_two_flat_layers() {
        let (w, h) = (8usize, 8usize);
        let mut px = vec![255u8; w * h * 4];
        for y in 2..6 {
            for x in 2..6 {
                let i = (y * w + x) * 4;
                px[i] = 200;
                px[i + 1] = 30;
                px[i + 2] = 30;
            }
        }
        let img = vtracer::ColorImage {
            pixels: px,
            width: w,
            height: h,
        };
        let mut sw = Stopwatch::default();
        let trace = trace_color(&img, &ColorSettings::default(), Background::None, &mut sw);
        assert_eq!(trace.width, 8);
        assert_eq!(trace.height, 8);
        assert!(trace.palette_size >= 2);
        // Bottom layer = white background (full-bleed rect), top = red shape.
        let bg = &trace.layers[0];
        assert_eq!(bg.paint, Paint::Solid([255, 255, 255]));
        let red = trace
            .layers
            .iter()
            .find(|l| l.paint == Paint::Solid([200, 30, 30]))
            .expect("red layer");
        assert_ne!(red.shapes.len(), 0);
    }

    #[test]
    fn trace_color_alpha_keeps_transparency() {
        // Transparent background, opaque blue dot: Background::Alpha must not
        // emit a full-bleed rect and the dot must survive.
        let (w, h) = (8usize, 8usize);
        let mut px = vec![0u8; w * h * 4];
        for y in 3..5 {
            for x in 3..5 {
                let i = (y * w + x) * 4;
                px[i] = 20;
                px[i + 1] = 60;
                px[i + 2] = 200;
                px[i + 3] = 255;
            }
        }
        let img = vtracer::ColorImage {
            pixels: px,
            width: w,
            height: h,
        };
        let mut sw = Stopwatch::default();
        let trace = trace_color(&img, &ColorSettings::default(), Background::Alpha, &mut sw);
        let svg = write::write_svg(&trace);
        assert!(!svg.contains("<rect"));
        assert!(svg.contains("<path"));
    }
}
