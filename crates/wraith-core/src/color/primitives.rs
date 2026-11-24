//! Primitive detection on rings (roadmap M4).
//!
//! Motivated by the rounded-rectangle failure in the prototype's §5.2 and by
//! Vectorizer.AI's published feature set:
//! * **circle/ellipse** — direct least-squares conic fit; when the max radial
//!   residual is below tolerance and the ring has no straight pieces, emit
//!   `<ellipse>`/`<circle>` (≈ 60 bytes vs 150–400 for Béziers);
//! * **rounded rect** — 4 axis-aligned lines + 4 congruent arcs → `<rect rx>`;
//! * **arcs** — a fitted cubic that a circle explains within tolerance is
//!   rewritten as an SVG `A` command (keeps G1 at joins since endpoints are
//!   shared with the neighbours).

use crate::color::Shape;
use crate::color::contour::P;
use crate::color::fit::{self, Seg};

/// Maximum allowed conic-fit deviation for primitive acceptance.
const PRIMITIVE_TOL: f64 = 0.25;

/// Minimum ring area (px²) before a circle/ellipse hypothesis is even
/// considered. Primitive detection targets *designed* full shapes; anything
/// smaller is speck-scale geometry (leak slivers at 3-way junctions render as
/// 5–15 px² rings) that must stay a path — measured on the mascot corpus,
/// 100 % of false-positive ellipses sat below this floor while real circles
/// (fixtures, icons) sit two orders of magnitude above it.
const PRIMITIVE_MIN_AREA: f64 = 40.0;

/// Minimum semi-axis (px) of an accepted circle/ellipse.
const PRIMITIVE_MIN_RADIUS: f64 = 2.5;

/// Maximum axis ratio of an accepted ellipse. Degenerate sliver rings
/// (a 17×1 px leak contour) fit "ellipses" with ratio ≈ 17: real designed
/// ellipses in graphic art stay far below this.
const PRIMITIVE_MAX_ASPECT: f64 = 6.0;

/// Fraction of the layer's own coverage a ring must enclose to be considered
/// for primitive detection. Primitives are a *standalone shape* claim: a ring
/// that is one component among many (specks, trapped slivers, holes) is not
/// the layer's geometry, and rewriting it silently repaints what lies under
/// it with the layer's fill.
const PRIMITIVE_MIN_DOMINANCE: f64 = 0.9;

/// Circumcircle through three well-separated points: the exact circle
/// through `a`, `b`, `c`, or `None` when they are (near-)collinear. Unlike an
/// algebraic least-squares fit this cannot mirror the centre across the
/// chord on nearly-straight runs, which is what made rewritten arcs take the
/// wrong (canvas-spanning) side of their chord.
fn circumcircle(a: P, b: P, c: P) -> Option<(f64, f64, f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let a2 = a[0] * a[0] + a[1] * a[1];
    let b2 = b[0] * b[0] + b[1] * b[1];
    let c2 = c[0] * c[0] + c[1] * c[1];
    let cx = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
    let cy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
    let r = (a[0] - cx).hypot(a[1] - cy);
    if !r.is_finite() || r < 1e-9 {
        return None;
    }
    Some((cx, cy, r))
}

/// Max deviation of the sample points from the fitted circle.
fn circle_dev(pts: &[P], cx: f64, cy: f64, r: f64) -> f64 {
    pts.iter()
        .map(|p| {
            let d = (p[0] - cx).hypot(p[1] - cy);
            (d - r).abs()
        })
        .fold(0.0_f64, f64::max)
}

/// Direct least-squares ellipse fit via the ring's second moments: the axes
/// are the eigenvectors of the covariance, the radii are `sqrt(2·λ)`. Returns
/// `(cx, cy, rx, ry, theta_deg)` with `rx ≥ ry`.
fn ellipse_fit(pts: &[P]) -> Option<(f64, f64, f64, f64, f64)> {
    let n = pts.len() as f64;
    if n < 12.0 {
        return None;
    }
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut syy = 0.0_f64;
    for p in pts {
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    sxx /= n;
    sxy /= n;
    syy /= n;
    let tr = sxx + syy;
    let disc = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let (l1, l2) = (f64::midpoint(tr, disc), (tr - disc) * 0.5);
    if l2 <= 1e-9 {
        return None;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    Some((
        cx,
        cy,
        (2.0 * l1).sqrt(),
        (2.0 * l2).sqrt(),
        theta.to_degrees(),
    ))
}

/// Max deviation of ring points from the parametric ellipse (nearest of 256
/// sampled boundary points).
fn ellipse_dev(pts: &[P], cx: f64, cy: f64, rx: f64, ry: f64, theta_deg: f64) -> f64 {
    let th = theta_deg.to_radians();
    let (ct, st) = (th.cos(), th.sin());
    let mut max_dev = 0.0_f64;
    for p in pts {
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        // Rotate into ellipse frame; normalise to a unit circle; the residual
        // is the radial distance of the normalised point from 1.
        let (ex, ey) = (dx * ct + dy * st, -dx * st + dy * ct);
        let norm = (ex / rx).hypot(ey / ry);
        let dev = (norm - 1.0) * rx.min(ry);
        max_dev = max_dev.max(dev.abs());
    }
    max_dev
}

/// Detects a standalone primitive for a single-ring layer, ring-aware:
/// `start` is the ring's start point preceding `segs`. Ellipse/circle first
/// (needs zero straight pieces), then axis-aligned rounded rect.
///
/// `layer_area` is the total absolute ring area of the owning layer; the
/// candidate ring must enclose at least [`PRIMITIVE_MIN_DOMINANCE`] of it
/// before any primitive is accepted, so fragments and slivers of multi-ring
/// layers can never be promoted to filled shapes.
#[must_use]
pub fn detect_ring(dense: &[P], start: P, segs: &[Seg], tol: f64, layer_area: f64) -> Option<Shape> {
    let prim_tol = PRIMITIVE_TOL.max(tol * 0.8);
    let has_lines = segs.iter().any(|s| matches!(s, Seg::Line(..)));
    if !has_lines {
        let ring_area = fit::area(dense).abs();
        let dominant = ring_area >= PRIMITIVE_MIN_DOMINANCE * layer_area.max(1.0);
        if dominant && ring_area >= PRIMITIVE_MIN_AREA {
            if let Some((cx, cy, rx, ry, theta)) = ellipse_fit(dense) {
                // Reject degenerate sliver fits up front: an ellipse whose
                // axes are a few px long or wildly unequal is a contour leak,
                // not a designed shape.
                if rx >= PRIMITIVE_MIN_RADIUS
                    && ry >= PRIMITIVE_MIN_RADIUS
                    && rx / ry <= PRIMITIVE_MAX_ASPECT
                {
                    if (rx - ry).abs() <= 0.02 * rx.max(ry) {
                        let r = f64::midpoint(rx, ry);
                        if ellipse_dev(dense, cx, cy, r, r, 0.0) < prim_tol {
                            return Some(Shape::Circle { cx, cy, r });
                        }
                    }
                    if ellipse_dev(dense, cx, cy, rx, ry, theta) < prim_tol {
                        return Some(Shape::Ellipse { cx, cy, rx, ry });
                    }
                }
            }
        }
        return None;
    }
    // Rounded rect walk: every straight piece axis-aligned, no bare cubics,
    // exactly 4 congruent arcs.
    let mut cur = start;
    let mut radii = Vec::new();
    for seg in segs {
        match seg {
            Seg::Line(p) => {
                let (dx, dy) = (p[0] - cur[0], p[1] - cur[1]);
                let l = dx.hypot(dy);
                if l < 1e-9 {
                    return None;
                }
                let (ux, uy) = (dx / l, dy / l);
                if ux.abs() > 0.02 && uy.abs() > 0.02 {
                    return None;
                }
                cur = *p;
            }
            Seg::Cubic(..) => return None, // arcs must have been rewritten
            Seg::Arc { rx, p, .. } => {
                radii.push(*rx);
                cur = *p;
            }
        }
    }
    if radii.len() == 4
        && segs.len() == 8
        && radii.iter().all(|r| (*r - radii[0]).abs() < 0.5)
        && radii[0] > 0.5
    {
        let xs = dense.iter().map(|p| p[0]);
        let ys = dense.iter().map(|p| p[1]);
        let (x0, x1) = (
            xs.clone().fold(f64::MAX, f64::min),
            xs.fold(f64::MIN, f64::max),
        );
        let (y0, y1) = (
            ys.clone().fold(f64::MAX, f64::min),
            ys.fold(f64::MIN, f64::max),
        );
        return Some(Shape::Rect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
            r: radii[0],
        });
    }
    None
}

/// Ring-aware arc rewriting for circular cubic segments.
///
/// A fitted cubic that a circle explains within tolerance becomes an SVG `A`
/// segment. Endpoints stay untouched so G1 continuity at joins is preserved.
#[must_use]
pub fn arcs_for_ring(start: P, segs: &[Seg], tol: f64) -> Vec<Seg> {
    let arc_tol = tol.max(0.2);
    let mut out = Vec::with_capacity(segs.len());
    let mut cur = start;
    for seg in segs {
        match seg {
            Seg::Line(p) => {
                out.push(Seg::Line(*p));
                cur = *p;
            }
            Seg::Cubic(c1, c2, p3) => {
                let samples: Vec<P> = (0..=16)
                    .map(|k| {
                        let t = f64::from(k) / 16.0;
                        let u = 1.0 - t;
                        let (b0, b1, b2, b3) =
                            (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                        [
                            b0 * cur[0] + b1 * c1[0] + b2 * c2[0] + b3 * p3[0],
                            b0 * cur[1] + b1 * c1[1] + b2 * c2[1] + b3 * p3[1],
                        ]
                    })
                    .collect();
                let as_arc = circumcircle(
                    samples[0],
                    samples[8.min(samples.len() - 1)],
                    samples[samples.len() - 1],
                )
                .and_then(|(cx, cy, r)| {
                    (circle_dev(&samples, cx, cy, r) < arc_tol && r < 1e5).then(|| {
                        let s = samples[0];
                        let e = samples[samples.len() - 1];
                        let mid = samples[8.min(samples.len() - 1)];
                        // Screen coords are y-down: a positive-angle (SVG
                        // sweep=1) rotation from s to e yields cross < 0.
                        let cross =
                            (e[0] - s[0]) * (mid[1] - s[1]) - (e[1] - s[1]) * (mid[0] - s[0]);
                        let sweep = cross < 0.0;
                        let a0 = (s[1] - cy).atan2(s[0] - cx);
                        let a1 = (e[1] - cy).atan2(e[0] - cx);
                        let span = if sweep {
                            (a1 - a0).rem_euclid(std::f64::consts::TAU)
                        } else {
                            (a0 - a1).rem_euclid(std::f64::consts::TAU)
                        };
                        let large = span > std::f64::consts::PI;
                        Seg::Arc {
                            rx: r,
                            ry: r,
                            rot_deg: 0.0,
                            large_arc: large,
                            sweep,
                            p: *p3,
                        }
                    })
                });
                match as_arc {
                    Some(a) => out.push(a),
                    None => out.push(Seg::Cubic(*c1, *c2, *p3)),
                }
                cur = *p3;
            }
            Seg::Arc { p, .. } => {
                out.push(*seg);
                cur = *p;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense ring of a circle (radius `r`, centre `(cx, cy)`).
    fn circle_ring(r: f64, cx: f64, cy: f64) -> Vec<P> {
        (0..256)
            .map(|i| {
                let t = std::f64::consts::TAU * f64::from(i) / 256.0;
                [cx + r * t.cos(), cy + r * t.sin()]
            })
            .collect()
    }

    #[test]
    fn circumcircle_recovers_center_and_radius() {
        let pts = circle_ring(20.0, 50.0, 50.0);
        let (cx, cy, r) = circumcircle(pts[0], pts[64], pts[128]).expect("fits");
        assert!((cx - 50.0).abs() < 1e-6 && (cy - 50.0).abs() < 1e-6 && (r - 20.0).abs() < 1e-6);
        assert!(circle_dev(&pts, cx, cy, r) < 1e-6);
    }

    #[test]
    fn circumcircle_stays_on_the_bulge_side_for_nearly_straight_runs() {
        // A shallow arc: centre must sit on the opposite side of the chord
        // from the mid sample, and the reconstructed arc must pass through
        // all three points — the mirrored-centre failure mode of a
        // least-squares fit is impossible here.
        let a = [0.0, 0.0];
        let e = [16.0, 0.0];
        let mid = [8.0, 0.11]; // bulges downward (+y)
        let (cx, cy, r) = circumcircle(a, mid, e).expect("fits");
        // The centre lies on the side *opposite* the bulge (bulge points
        // +y, centre at negative y) and the radius matches the sagitta.
        assert!(cy < 0.0, "centre opposite the bulge: {cy}");
        assert!(((r - 290.9).abs() < 2.0), "sagitta-derived radius: {r}");
        assert!(circle_dev(&[a, mid, e], cx, cy, r) < 1e-9);
        // Collinear points are rejected.
        assert!(circumcircle([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]).is_none());
    }

    #[test]
    fn ellipse_fit_recovers_axes() {
        // Axis-aligned ellipse rx=30, ry=10.
        let pts: Vec<P> = (0..256)
            .map(|i| {
                let t = std::f64::consts::TAU * f64::from(i) / 256.0;
                [100.0 + 30.0 * t.cos(), 40.0 + 10.0 * t.sin()]
            })
            .collect();
        let (cx, cy, rx, ry, theta) = ellipse_fit(&pts).expect("fits");
        assert!((cx - 100.0).abs() < 1e-3 && (cy - 40.0).abs() < 1e-3);
        assert!(
            (rx - 30.0).abs() < 0.1 && (ry - 10.0).abs() < 0.1,
            "{rx} {ry}"
        );
        assert!(theta.abs() < 0.5 || (theta.abs() - 90.0).abs() < 0.5);
        assert!(ellipse_dev(&pts, cx, cy, rx, ry, theta) < 0.2);
    }

    #[test]
    fn detect_ring_emits_circle_for_perfect_circle() {
        // A fitted ring of a circle is all cubics, no lines.
        let ring = circle_ring(25.0, 60.0, 60.0);
        let segs = vec![
            Seg::Cubic([60.0, 35.0], [85.0, 35.0], [85.0, 60.0]),
            Seg::Cubic([85.0, 85.0], [60.0, 85.0], [60.0, 60.0]),
        ];
        let layer_area = fit::area(&ring).abs();
        let shape = detect_ring(&ring, ring[0], &segs, 0.3, layer_area);
        assert!(matches!(shape, Some(Shape::Circle { .. })), "got {shape:?}");
    }

    #[test]
    fn detect_ring_emits_ellipse_for_ellipse() {
        let ring: Vec<P> = (0..256)
            .map(|i| {
                let t = std::f64::consts::TAU * f64::from(i) / 256.0;
                [60.0 + 40.0 * t.cos(), 40.0 + 12.0 * t.sin()]
            })
            .collect();
        let segs = vec![
            Seg::Cubic([100.0, 28.0], [100.0, 52.0], [60.0, 52.0]),
            Seg::Cubic([20.0, 52.0], [20.0, 28.0], [60.0, 28.0]),
        ];
        let layer_area = fit::area(&ring).abs();
        let shape = detect_ring(&ring, ring[0], &segs, 0.3, layer_area);
        assert!(
            matches!(shape, Some(Shape::Ellipse { .. })),
            "got {shape:?}"
        );
    }

    #[test]
    fn arcs_for_ring_rewrites_circular_cubic_as_arc() {
        // A quarter circle approximated by ONE cubic (the classic kappa
        // approximation has ~0.027% mid error) — well within tolerance.
        let r = 50.0;
        let k = r * 0.5523;
        let start = [100.0, 50.0];
        let segs = vec![Seg::Cubic(
            [100.0 + k, 50.0],
            [150.0, 100.0 - k],
            [150.0, 100.0],
        )];
        let out = arcs_for_ring(start, &segs, 0.3);
        assert!(
            matches!(out[0], Seg::Arc { .. }),
            "quarter-circle cubic becomes an arc: {out:?}"
        );
        // Endpoint preserved.
        if let Seg::Arc { p, rx, .. } = out[0] {
            assert!((p[0] - 150.0).abs() < 1e-9 && (p[1] - 100.0).abs() < 1e-9);
            assert!((rx - 50.0).abs() < 1.0, "r={rx}");
        }
        // A wiggly cubic must NOT become an arc.
        let wiggle = vec![Seg::Cubic([110.0, 30.0], [90.0, 70.0], [150.0, 100.0])];
        let out2 = arcs_for_ring(start, &wiggle, 0.3);
        assert!(matches!(out2[0], Seg::Cubic(..)));
    }

    /// A degenerate sliver ring — the leak-contour signature measured on the
    /// mascot corpus (100–150 tiny `<ellipse>` fills per image before the
    /// gate, every one of them a junction leak, every one measurably wrong) —
    /// must never be promoted to a filled primitive: it is below the area
    /// floor, below the minimum radius, above the aspect bound, and it is a
    /// fragment of a multi-component layer rather than a standalone shape.
    #[test]
    fn detect_ring_rejects_micro_sliver_ellipses() {
        // A 17×1 px sliver: the exact shape of the worst offender in the
        // baseline (`rx=8.66 ry=.5`).
        let ring: Vec<P> = (0..64)
            .map(|i| {
                let t = std::f64::consts::TAU * f64::from(i) / 64.0;
                [180.0 + 8.66 * t.cos(), 647.5 + 0.5 * t.sin()]
            })
            .collect();
        let segs = vec![
            Seg::Cubic([188.0, 647.0], [188.0, 648.0], [180.0, 648.0]),
            Seg::Cubic([172.0, 648.0], [172.0, 647.0], [180.0, 647.0]),
        ];
        // As its own tiny layer (dominant but below the area floor)...
        assert!(detect_ring(&ring, ring[0], &segs, 0.3, fit::area(&ring).abs()).is_none());
        // ...and as a fragment of a large layer (a big sibling exists).
        assert!(detect_ring(&ring, ring[0], &segs, 0.3, 5_000.0).is_none());
        // A 4:1 ellipse of real size is still accepted when standalone.
        let big: Vec<P> = (0..256)
            .map(|i| {
                let t = std::f64::consts::TAU * f64::from(i) / 256.0;
                [100.0 + 40.0 * t.cos(), 60.0 + 10.0 * t.sin()]
            })
            .collect();
        let big_segs = vec![
            Seg::Cubic([140.0, 50.0], [140.0, 70.0], [100.0, 70.0]),
            Seg::Cubic([60.0, 70.0], [60.0, 50.0], [100.0, 50.0]),
        ];
        let layer_area = fit::area(&big).abs();
        assert!(
            matches!(
                detect_ring(&big, big[0], &big_segs, 0.3, layer_area),
                Some(Shape::Ellipse { .. })
            ),
            "real ellipse must survive the gates"
        );
        // ...but the same ellipse as a fragment (layer has a big sibling)
        // must be rejected: not a standalone shape.
        assert!(detect_ring(&big, big[0], &big_segs, 0.3, layer_area * 3.0).is_none());
    }

    #[test]
    fn detect_ring_emits_rect_for_rounded_rect_rings() {
        // Rounded rect 100×80, r=10: 4 lines + 4 arcs.
        let (x0, y0, x1, y1, rr) = (0.0_f64, 0.0_f64, 100.0_f64, 80.0_f64, 10.0_f64);
        let start = [x0 + rr, y0];
        let segs = vec![
            Seg::Line([x1 - rr, y0]),
            Seg::Arc {
                rx: rr,
                ry: rr,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [x1, y0 + rr],
            },
            Seg::Line([x1, y1 - rr]),
            Seg::Arc {
                rx: rr,
                ry: rr,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [x1 - rr, y1],
            },
            Seg::Line([x0 + rr, y1]),
            Seg::Arc {
                rx: rr,
                ry: rr,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [x0, y1 - rr],
            },
            Seg::Line([x0, y0 + rr]),
            Seg::Arc {
                rx: rr,
                ry: rr,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [x0 + rr, y0],
            },
        ];
        let mut dense = vec![start];
        for s in &segs {
            match s {
                Seg::Line(p) | Seg::Arc { p, .. } => dense.push(*p),
                Seg::Cubic(..) => {}
            }
        }
        let shape = detect_ring(&dense, start, &segs, 0.3, 1_000_000);
        match shape {
            Some(Shape::Rect { x, y, w, h, r }) => {
                assert!((x - x0).abs() < 1e-6 && (y - y0).abs() < 1e-6);
                assert!((w - 100.0).abs() < 1e-6 && (h - 80.0).abs() < 1e-6);
                assert!((r - rr).abs() < 0.5);
            }
            other => panic!("expected rect, got {other:?}"),
        }
    }

    #[test]
    fn detect_ring_rejects_non_axis_aligned_lines() {
        let start = [0.0, 0.0];
        let segs = vec![
            Seg::Line([50.0, 50.0]),
            Seg::Arc {
                rx: 5.0,
                ry: 5.0,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [55.0, 55.0],
            },
            Seg::Line([5.0, 55.0]),
            Seg::Arc {
                rx: 5.0,
                ry: 5.0,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [0.0, 50.0],
            },
            Seg::Line([10.0, 10.0]),
            Seg::Arc {
                rx: 5.0,
                ry: 5.0,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [15.0, 15.0],
            },
            Seg::Line([65.0, 65.0]),
            Seg::Arc {
                rx: 5.0,
                ry: 5.0,
                rot_deg: 0.0,
                large_arc: false,
                sweep: true,
                p: [70.0, 70.0],
            },
        ];
        let dense: Vec<P> = vec![[0.0, 0.0], [50.0, 50.0], [55.0, 55.0]];
        assert!(detect_ring(&dense, start, &segs, 0.3, 1_000_000).is_none());
    }
}
