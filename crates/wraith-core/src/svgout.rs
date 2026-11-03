//! Stage 3: canonical serialization of the traced SVG.
//!
//! Replaces vtracer's `Display` writer with a purpose-built one that:
//! * hoists a shared fill colour onto a single `<g>` (huge byte win for BW
//!   output, where every path is black),
//! * omits `transform` when the offset is the origin,
//! * skips empty paths and (optionally) pure-white background paths,
//! * writes straight into a pre-sized `String` — one allocation, no
//!   intermediate buffers, no XML declaration boilerplate.

use std::fmt::Write as _;

use visioncortex::{Color, PointF64};

use crate::stats::{Stage, Stopwatch};

/// Pure white in the tracer's colour space.
const WHITE: Color = Color {
    r: 255,
    g: 255,
    b: 255,
    a: 255,
};

/// Serializes a traced [`vtracer::SvgFile`] into a minimal, canonical SVG.
#[must_use]
pub fn serialize(
    svg: &vtracer::SvgFile,
    strip_white: bool,
    precision: Option<u32>,
    sw: &mut Stopwatch,
) -> String {
    sw.start();
    let out = write_svg(svg, strip_white, precision);
    sw.lap(Stage::Serialize);
    out
}

fn write_svg(svg: &vtracer::SvgFile, strip_white: bool, precision: Option<u32>) -> String {
    // Render path data first so fill-hoisting only happens when there is
    // actually geometry to wrap (empty and stripped paths vanish here).
    let paths: Vec<Rendered> = svg
        .paths
        .iter()
        .filter(|p| !(strip_white && p.color == WHITE))
        .filter_map(|p| {
            let (d, offset) = p.path.to_svg_string(true, PointF64::default(), precision);
            let d = d.trim_end();
            (!d.is_empty()).then(|| Rendered {
                d: d.to_owned(),
                offset,
                color: p.color,
            })
        })
        .collect();

    // BW traces share one fill by construction; hoist it onto a group and
    // drop it from every path.
    let shared_fill = if paths.len() > 1 {
        let first = paths[0].color;
        paths
            .iter()
            .all(|p| p.color == first)
            .then(|| first.to_hex_string())
    } else {
        None
    };

    // One pre-sized allocation for the whole document.
    let mut out = String::with_capacity(96 + svg.paths.len() * 72);
    let _ = write!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
        svg.width, svg.height, svg.width, svg.height
    );
    if let Some(fill) = &shared_fill {
        let _ = write!(out, "<g fill=\"{fill}\">");
    }
    for path in &paths {
        write_path(&mut out, path, shared_fill.is_none());
    }
    if shared_fill.is_some() {
        out.push_str("</g>");
    }
    out.push_str("</svg>");
    out
}

/// A path whose data has already been rendered and rebased.
struct Rendered {
    d: String,
    offset: PointF64,
    color: Color,
}

fn write_path(out: &mut String, path: &Rendered, with_fill: bool) {
    out.push_str("<path d=\"");
    out.push_str(&path.d);
    out.push('"');
    if with_fill {
        let _ = write!(out, " fill=\"{}\"", path.color.to_hex_string());
    }
    let offset = path.offset;
    if offset.x.abs() > f64::EPSILON || offset.y.abs() > f64::EPSILON {
        let _ = write!(out, " transform=\"translate({},{})\"", offset.x, offset.y);
    }
    out.push_str("/>");
}

#[cfg(test)]
mod tests {
    use visioncortex::{Color, ColorName, CompoundPath, PathI32, Point2};

    use super::*;
    use crate::preprocess::{fixture_png, prepare};
    use crate::settings::Settings;

    fn traced(size: u32) -> vtracer::SvgFile {
        let png = fixture_png(size);
        let mut sw = Stopwatch::default();
        let settings = Settings::new();
        let prepared = prepare(&png, &settings, &mut sw).expect("prepares");
        crate::trace::trace(prepared.image, &settings, &mut sw).expect("traces")
    }

    fn compound(points: &[(i32, i32)]) -> CompoundPath {
        let mut cp = CompoundPath::default();
        let path: Vec<Point2<i32>> = points.iter().map(|(x, y)| Point2::new(*x, *y)).collect();
        cp.add_path_i32(PathI32 { path });
        cp
    }

    fn black() -> Color {
        Color::color(&ColorName::Black)
    }

    #[test]
    fn bw_output_hoists_single_black_fill_into_group() {
        // Two black clusters: exactly what a BW trace of multi-shape art
        // produces — every path shares one fill, so it must hoist to a group.
        let mut svg = vtracer::SvgFile::new(16, 16, Some(2));
        svg.add_path(compound(&[(0, 0), (4, 0), (4, 4)]), black());
        svg.add_path(compound(&[(8, 8), (12, 8), (12, 12)]), black());
        let mut sw = Stopwatch::default();
        let out = serialize(&svg, false, Some(2), &mut sw);
        assert!(out.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(out.contains("<g fill=\"#000000\">"));
        assert!(out.ends_with("</g></svg>"));
        assert_eq!(out.matches("<path d=\"").count(), 2);
        assert_eq!(sw.laps().last().map(|(s, _)| *s), Some(Stage::Serialize));
    }

    #[test]
    fn traced_fixture_serializes_to_geometry() {
        let traced = traced(32);
        let mut sw = Stopwatch::default();
        let out = serialize(&traced, false, Some(2), &mut sw);
        assert!(out.contains("<svg"));
        assert!(out.contains("<path"));
        assert!(out.ends_with("</svg>"));
        assert_eq!(sw.laps().last().map(|(s, _)| *s), Some(Stage::Serialize));
    }

    #[test]
    fn mixed_fills_use_per_path_attributes() {
        let mut svg = vtracer::SvgFile::new(10, 10, Some(2));
        svg.add_path(compound(&[(0, 0), (4, 0), (4, 4)]), black());
        svg.add_path(
            compound(&[(5, 5), (9, 5), (9, 9)]),
            Color::color(&ColorName::Red),
        );
        let out = serialize(&svg, false, Some(2), &mut Stopwatch::default());
        assert!(!out.contains("<g "));
        assert!(out.contains("fill=\"#000000\""));
        assert!(out.contains("fill=\"#FF0000\""));
    }

    #[test]
    fn white_paths_are_stripped_only_when_asked() {
        let mut svg = vtracer::SvgFile::new(10, 10, Some(2));
        svg.add_path(compound(&[(0, 0), (4, 0), (4, 4)]), WHITE);
        svg.add_path(compound(&[(5, 5), (9, 5), (9, 9)]), black());
        let kept = serialize(&svg, false, Some(2), &mut Stopwatch::default());
        assert!(kept.contains("fill=\"#FFFFFF\""));
        let stripped = serialize(&svg, true, Some(2), &mut Stopwatch::default());
        assert!(!stripped.contains("fill=\"#FFFFFF\""));
        assert!(stripped.contains("fill=\"#000000\""));
    }

    #[test]
    fn all_white_paths_leave_no_group_and_no_paths() {
        let mut svg = vtracer::SvgFile::new(10, 10, Some(2));
        svg.add_path(compound(&[(0, 0), (4, 0), (4, 4)]), WHITE);
        let out = serialize(&svg, true, Some(2), &mut Stopwatch::default());
        assert!(!out.contains("<path"));
        assert!(!out.contains("<g "));
    }

    #[test]
    fn empty_path_data_is_skipped() {
        let mut svg = vtracer::SvgFile::new(4, 4, Some(2));
        svg.add_path(CompoundPath::default(), black());
        svg.add_path(CompoundPath::default(), black());
        let out = serialize(&svg, false, Some(2), &mut Stopwatch::default());
        assert!(!out.contains("<path"));
        // Exactly two paths sharing a fill would hoist a group; zero emitted
        // paths must not leave a dangling one.
        assert!(!out.contains("<g "));
    }

    #[test]
    fn zero_offset_omits_transform() {
        let mut svg = vtracer::SvgFile::new(10, 10, Some(2));
        svg.add_path(compound(&[(0, 0), (4, 0), (4, 4)]), black());
        let out = serialize(&svg, false, Some(2), &mut Stopwatch::default());
        assert!(!out.contains("transform="));
        assert!(out.contains("d=\"M0,0"));
    }

    #[test]
    fn nonzero_offset_becomes_translate() {
        let mut svg = vtracer::SvgFile::new(10, 10, Some(2));
        svg.add_path(compound(&[(2, 3), (6, 3), (6, 7)]), black());
        let out = serialize(&svg, false, Some(2), &mut Stopwatch::default());
        assert!(out.contains("transform=\"translate(2,3)\""));
    }

    #[test]
    fn higher_precision_produces_longer_path_data() {
        let traced = traced(64);
        let coarse = serialize(&traced, false, Some(1), &mut Stopwatch::default());
        let fine = serialize(&traced, false, Some(6), &mut Stopwatch::default());
        assert!(fine.len() > coarse.len());
    }
}
