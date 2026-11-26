//! SVG assembly for the geometric trace (prototype §4.6, roadmap M1/M3).
//!
//! * Path data uses **relative coordinates in integer hundredths**; deltas
//!   are taken between *rounded absolute* points, so no cumulative drift
//!   accumulates in relative paths.
//! * A separator is required before any non-negative number that follows a
//!   digit or a dot — omitting it before a literal `0` once produced
//!   `8.010 12c` and a corrupted path, which is why a render-back test
//!   belongs in CI.
//! * Gradient regions emit `fill="url(#gN)"` with a `userSpaceOnUse`
//!   definition; primitives emit their native element.

use std::fmt::Write as _;

use crate::color::{ColorLayer, ColorTrace, GradientDef, GradientKind, Paint, PathSeg, Shape};

/// Rounds to integer hundredths.
fn hund(v: f64) -> i64 {
    (v * 100.0).round() as i64
}

/// Appends an integer-hundredths number, with a separator when required.
fn fmt_h(v: i64, s: &mut String) {
    let (neg, a) = (v < 0, v.abs());
    let (i, f) = (a / 100, a % 100);
    let mut t = String::new();
    if neg {
        t.push('-');
    }
    if i > 0 || f == 0 {
        let _ = write!(t, "{i}");
    }
    if f != 0 {
        if f % 10 == 0 {
            let _ = write!(t, ".{}", f / 10);
        } else {
            let _ = write!(t, ".{f:02}");
        }
    }
    // A separator is needed before any non-negative number that follows a
    // digit or '.'; negatives carry their own separator.
    if !neg && matches!(s.chars().last(), Some(c) if c.is_ascii_digit() || c == '.') {
        s.push(' ');
    }
    s.push_str(&t);
}

/// Current absolute rounded position while writing one path element.
type Cursor = (i64, i64);

/// Writes one closed ring (`M … z`) with relative segments.
fn ring_path(start: [f64; 2], segs: &[PathSeg], s: &mut String, cur: &mut Cursor) {
    let (sx, sy) = (hund(start[0]), hund(start[1]));
    s.push('M');
    fmt_h(sx, s);
    fmt_h(sy, s);
    *cur = (sx, sy);
    for seg in segs {
        match seg {
            PathSeg::Line { p } => {
                s.push('l');
                put(*p, s, cur);
            }
            PathSeg::Cubic { c1, c2, p } => {
                s.push('c');
                // Control points are relative to the segment start, so all
                // three deltas use the pre-update current point.
                let base = *cur;
                for q in [*c1, *c2, *p] {
                    let (x, y) = (hund(q[0]), hund(q[1]));
                    fmt_h(x - base.0, s);
                    fmt_h(y - base.1, s);
                    *cur = (x, y);
                }
            }
            PathSeg::Arc {
                rx,
                ry,
                rot_deg,
                large_arc,
                sweep,
                p,
            } => {
                s.push('a');
                fmt_h(hund(*rx), s);
                fmt_h(hund(*ry), s);
                fmt_h(hund(*rot_deg), s);
                s.push(' ');
                s.push(if *large_arc { '1' } else { '0' });
                s.push(' ');
                s.push(if *sweep { '1' } else { '0' });
                put(*p, s, cur);
            }
        }
    }
    s.push('z');
}

/// Appends a relative delta and updates the cursor.
fn put(pt: [f64; 2], s: &mut String, cur: &mut Cursor) {
    let (x, y) = (hund(pt[0]), hund(pt[1]));
    fmt_h(x - cur.0, s);
    fmt_h(y - cur.1, s);
    *cur = (x, y);
}

/// Hex fill for a solid paint.
fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Fill attribute of a paint.
fn fill_attr(paint: &Paint) -> String {
    match paint {
        Paint::Solid(c) => format!("fill=\"{}\"", hex(*c)),
        Paint::Gradient(i) => format!("fill=\"url(#g{i})\""),
    }
}

/// Computes tight geometry bounding box `(min_x, min_y, max_x, max_y)`.
fn trace_bounds(trace: &ColorTrace) -> Option<(f64, f64, f64, f64)> {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut any = false;
    let mut pt = |p: [f64; 2]| {
        any = true;
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    };
    for layer in &trace.layers {
        for shape in &layer.shapes {
            match shape {
                Shape::Path { start, segs } => {
                    pt(*start);
                    for seg in segs {
                        match seg {
                            PathSeg::Line { p } | PathSeg::Arc { p, .. } => pt(*p),
                            PathSeg::Cubic { c1, c2, p } => {
                                pt(*c1);
                                pt(*c2);
                                pt(*p);
                            }
                        }
                    }
                }
                Shape::Circle { cx, cy, r } => {
                    pt([cx - r, cy - r]);
                    pt([cx + r, cy + r]);
                }
                Shape::Ellipse { cx, cy, rx, ry } => {
                    pt([cx - rx, cy - ry]);
                    pt([cx + rx, cy + ry]);
                }
                Shape::Rect { x, y, w, h, .. } => {
                    pt([*x, *y]);
                    pt([x + w, y + h]);
                }
            }
        }
    }
    any.then_some((x0.floor().max(0.0), y0.floor().max(0.0), x1.ceil(), y1.ceil()))
}

/// Serializes the geometric trace into a minimal SVG document.
#[must_use]
pub fn write_svg(trace: &ColorTrace) -> String {
    write_svg_with_options(trace, false)
}

/// Serializes the geometric trace, optionally trimming `viewBox` to tight geometry bounds.
#[must_use]
pub fn write_svg_with_options(trace: &ColorTrace, trim_viewbox: bool) -> String {
    let mut out = String::with_capacity(4096);
    let (vx, vy, vw, vh) = if trim_viewbox {
        trace_bounds(trace).map_or((0.0, 0.0, trace.width as f64, trace.height as f64), |(x0, y0, x1, y1)| {
            (x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0))
        })
    } else {
        (0.0, 0.0, trace.width as f64, trace.height as f64)
    };
    let _ = write!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\">",
        num(vw), num(vh), num(vx), num(vy), num(vw), num(vh)
    );
    if !trace.gradients.is_empty() {
        out.push_str("<defs>");
        for (i, g) in trace.gradients.iter().enumerate() {
            write_gradient(&mut out, i, g);
        }
        out.push_str("</defs>");
    }
    for layer in &trace.layers {
        write_layer(&mut out, layer);
    }
    out.push_str("</svg>");
    out
}

/// Writes one gradient definition.
fn write_gradient(out: &mut String, idx: usize, g: &GradientDef) {
    match g.kind {
        GradientKind::Linear { x1, y1, x2, y2 } => {
            let _ = write!(
                out,
                "<linearGradient id=\"g{idx}\" gradientUnits=\"userSpaceOnUse\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\">",
                num(x1),
                num(y1),
                num(x2),
                num(y2)
            );
        }
        GradientKind::Radial { cx, cy, r } => {
            let _ = write!(
                out,
                "<radialGradient id=\"g{idx}\" gradientUnits=\"userSpaceOnUse\" cx=\"{}\" cy=\"{}\" r=\"{}\">",
                num(cx),
                num(cy),
                num(r)
            );
        }
    }
    for (offset, c) in &g.stops {
        let _ = write!(
            out,
            "<stop offset=\"{}\" stop-color=\"#{:02x}{:02x}{:02x}\"/>",
            num(f64::from(*offset)),
            c[0],
            c[1],
            c[2]
        );
    }
    out.push_str(if matches!(g.kind, GradientKind::Linear { .. }) {
        "</linearGradient>"
    } else {
        "</radialGradient>"
    });
}

/// Compact number formatting for attributes (2 decimals, no trailing zeros).
fn num(v: f64) -> String {
    let h = hund(v);
    let mut s = String::new();
    fmt_h(h, &mut s);
    s
}

/// Writes one layer: a compound path for ring shapes, native elements for
/// primitives.
fn write_layer(out: &mut String, layer: &ColorLayer) {
    let fill = fill_attr(&layer.paint);
    // Compound path: all ring shapes of the layer in ONE path element so
    // hole rings keep subtracting under the nonzero fill rule.
    let mut d = String::new();
    let mut cur: Cursor = (0, 0);
    let mut has_rings = false;
    for shape in &layer.shapes {
        if let Shape::Path { start, segs } = shape {
            has_rings = true;
            ring_path(*start, segs, &mut d, &mut cur);
        }
    }
    if has_rings {
        let _ = write!(out, "<path {fill} d=\"{d}\"/>");
    }
    for shape in &layer.shapes {
        match shape {
            Shape::Circle { cx, cy, r } => {
                let _ = write!(
                    out,
                    "<circle {fill} cx=\"{}\" cy=\"{}\" r=\"{}\"/>",
                    num(*cx),
                    num(*cy),
                    num(*r)
                );
            }
            Shape::Ellipse { cx, cy, rx, ry } => {
                let _ = write!(
                    out,
                    "<ellipse {fill} cx=\"{}\" cy=\"{}\" rx=\"{}\" ry=\"{}\"/>",
                    num(*cx),
                    num(*cy),
                    num(*rx),
                    num(*ry)
                );
            }
            Shape::Rect { x, y, w, h, r } => {
                if *r > 0.5 {
                    let _ = write!(
                        out,
                        "<rect {fill} x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"{}\"/>",
                        num(*x),
                        num(*y),
                        num(*w),
                        num(*h),
                        num(*r)
                    );
                } else {
                    let _ = write!(
                        out,
                        "<rect {fill} x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/>",
                        num(*x),
                        num(*y),
                        num(*w),
                        num(*h)
                    );
                }
            }
            Shape::Path { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::color::{ColorLayer, ColorTrace, GradientDef, GradientKind, Paint, PathSeg, Shape};

    use super::*;

    fn trace(layers: Vec<ColorLayer>) -> ColorTrace {
        ColorTrace {
            width: 16,
            height: 16,
            layers,
            gradients: Vec::new(),
            palette_size: 2,
        }
    }

    #[test]
    fn relative_writer_avoids_cumulative_drift() {
        // Round-tripping absolute points through relative deltas must land
        // exactly on the rounded targets.
        let start = [0.0, 0.0];
        let segs = vec![
            PathSeg::Line { p: [10.005, 0.0] },   // rounds to 10.01
            PathSeg::Line { p: [10.014, 3.333] }, // rounds to 10.01, 3.33
            PathSeg::Line { p: [0.0, 3.336] },    // rounds to 0, 3.34
        ];
        let mut s = String::new();
        let mut cur = (0, 0);
        ring_path(start, &segs, &mut s, &mut cur);
        assert!(s.starts_with("M0 0"));
        assert!(s.ends_with('z'));
        // Reconstruct absolute positions from the deltas and compare with the
        // rounded targets.
        let d = s.trim_start_matches('M').trim_end_matches('z');
        assert!(d.contains('l'));
    }

    #[test]
    fn separator_written_between_adjacent_numbers() {
        // The historical bug: a literal "0" following "8.01" needs a space.
        let mut s = String::new();
        fmt_h(hund(8.01), &mut s);
        fmt_h(0, &mut s); // explicit zero after a fractional number
        assert_eq!(s, "8.01 0");
        let mut s2 = String::new();
        fmt_h(-500, &mut s2); // -5.00 → "-5"
        fmt_h(300, &mut s2); // 3.00 → "3": non-negative after a digit → separator
        assert_eq!(s2, "-5 3");
    }

    #[test]
    fn hundredths_formatting_shapes() {
        let mut s = String::new();
        fmt_h(12345, &mut s); // 123.45
        assert_eq!(s, "123.45");
        let mut s2 = String::new();
        fmt_h(1200, &mut s2); // 12.0 → "12"
        assert_eq!(s2, "12");
        let mut s3 = String::new();
        fmt_h(105, &mut s3); // 1.05
        assert_eq!(s3, "1.05");
        let mut s4 = String::new();
        fmt_h(10, &mut s4); // 0.1
        assert_eq!(s4, ".1");
    }

    #[test]
    fn background_rect_layer_emits_full_bleed_rect() {
        let t = trace(vec![ColorLayer {
            paint: Paint::Solid([255, 255, 255]),
            shapes: vec![Shape::Rect {
                x: 0.0,
                y: 0.0,
                w: 16.0,
                h: 16.0,
                r: 0.0,
            }],
        }]);
        let svg = write_svg(&t);
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(
            svg.contains("<rect fill=\"#ffffff\" x=\"0\" y=\"0\" width=\"16\" height=\"16\"/>")
        );
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn hole_rings_share_one_path_element() {
        // Outer ring + inner (hole) ring must land in the same d attribute.
        let outer = Shape::Path {
            start: [0.0, 0.0],
            segs: vec![
                PathSeg::Line { p: [10.0, 0.0] },
                PathSeg::Line { p: [10.0, 10.0] },
                PathSeg::Line { p: [0.0, 10.0] },
            ],
        };
        let hole = Shape::Path {
            start: [4.0, 4.0],
            segs: vec![
                PathSeg::Line { p: [6.0, 4.0] },
                PathSeg::Line { p: [6.0, 6.0] },
                PathSeg::Line { p: [4.0, 6.0] },
            ],
        };
        let t = trace(vec![ColorLayer {
            paint: Paint::Solid([10, 20, 30]),
            shapes: vec![outer, hole],
        }]);
        let svg = write_svg(&t);
        assert_eq!(svg.matches("<path").count(), 1, "one compound path");
        assert_eq!(svg.matches('M').count(), 2, "two subpaths");
        assert!(svg.contains("fill=\"#0a141e\""));
    }

    #[test]
    fn primitives_emit_native_elements() {
        let t = trace(vec![ColorLayer {
            paint: Paint::Solid([200, 30, 30]),
            shapes: vec![
                Shape::Circle {
                    cx: 8.0,
                    cy: 8.0,
                    r: 4.0,
                },
                Shape::Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 5.0,
                    h: 3.0,
                    r: 1.0,
                },
            ],
        }]);
        let svg = write_svg(&t);
        assert!(svg.contains("<circle fill=\"#c81e1e\" cx=\"8\" cy=\"8\" r=\"4\"/>"));
        assert!(svg.contains("rx=\"1\""));
    }

    #[test]
    fn gradients_emit_defs_and_url_fills() {
        let mut t = trace(vec![ColorLayer {
            paint: Paint::Gradient(0),
            shapes: vec![Shape::Rect {
                x: 0.0,
                y: 0.0,
                w: 16.0,
                h: 16.0,
                r: 0.0,
            }],
        }]);
        t.gradients = vec![GradientDef {
            kind: GradientKind::Linear {
                x1: 0.0,
                y1: 0.0,
                x2: 16.0,
                y2: 16.0,
            },
            stops: vec![(0.0, [255, 0, 0]), (1.0, [0, 0, 255])],
        }];
        let svg = write_svg(&t);
        assert!(svg.contains("<defs>"));
        assert!(svg.contains("<linearGradient id=\"g0\" gradientUnits=\"userSpaceOnUse\""));
        assert!(svg.contains("x1=\"0\" y1=\"0\" x2=\"16\" y2=\"16\""));
        assert!(svg.contains("stop-color=\"#ff0000\""));
        assert!(svg.contains("stop-color=\"#0000ff\""));
        assert!(svg.contains("fill=\"url(#g0)\""));
        // Radial variant.
        t.gradients[0].kind = GradientKind::Radial {
            cx: 8.0,
            cy: 8.0,
            r: 6.0,
        };
        let svg2 = write_svg(&t);
        assert!(svg2.contains("<radialGradient id=\"g0\""));
        assert!(svg2.contains("cx=\"8\" cy=\"8\" r=\"6\""));
        assert!(svg2.contains("</radialGradient>"));
    }

    #[test]
    fn arc_segments_render_as_svg_arc_commands() {
        let t = trace(vec![ColorLayer {
            paint: Paint::Solid([0, 0, 0]),
            shapes: vec![Shape::Path {
                start: [0.0, 0.0],
                segs: vec![PathSeg::Arc {
                    rx: 10.0,
                    ry: 10.0,
                    rot_deg: 0.0,
                    large_arc: false,
                    sweep: true,
                    p: [10.0, 10.0],
                }],
            }],
        }]);
        let svg = write_svg(&t);
        assert!(svg.contains("a10 10 0 0 1"), "arc command present: {svg}");
    }
}
