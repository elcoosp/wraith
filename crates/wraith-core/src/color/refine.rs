//! Field-driven refinement of fitted curves (roadmap M5).
//!
//! Because the pipeline already has the target coverage field `F′` for every
//! layer, refinement needs no differentiable renderer: minimise
//!
//! `E = Σₛ (F′(pₛ) − 0.5)²`
//!
//! over sampled points `pₛ` on each fitted curve, by gradient descent on the
//! interior control points using the bilinear gradient `∇F′`. This removes
//! the interpolation bias marching squares introduced, runs only on boundary
//! samples, and keeps corners pinned (endpoints are never touched). The
//! result is accepted only when the squared residual actually improves.

use crate::color::contour::P;
use crate::color::fit::Seg;

/// Learning rate for one control-point step (field values are 0..1, so the
/// raw step magnitude is `lr · |r| · |∇F′| · w` ≈ 0.35 · 0.5 · 1 · 1).
const LR: f64 = 0.35;

/// Bernstein weight of interior control point 1 at parameter t.
fn w1(t: f64) -> f64 {
    let u = 1.0 - t;
    3.0 * u * u * t
}

/// Bernstein weight of interior control point 2 at parameter t.
fn w2(t: f64) -> f64 {
    let u = 1.0 - t;
    3.0 * u * t * t
}

/// Bilinear value and gradient of the field at an image point.
fn bilinear(field: &[f32], spw: usize, sph: usize, lx: f64, ly: f64) -> (f64, [f64; 2]) {
    let lx = lx.clamp(0.0, (spw - 2) as f64);
    let ly = ly.clamp(0.0, (sph - 2) as f64);
    let x0 = lx.floor() as usize;
    let y0 = ly.floor() as usize;
    let fx = lx - x0 as f64;
    let fy = ly - y0 as f64;
    let at = |x: usize, y: usize| f64::from(field[y * spw + x]);
    let (v00, v10) = (at(x0, y0), at(x0 + 1, y0));
    let (v01, v11) = (at(x0, y0 + 1), at(x0 + 1, y0 + 1));
    let v = v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy;
    let gx = (v10 - v00) * (1.0 - fy) + (v11 - v01) * fy;
    let gy = (v01 - v00) * (1.0 - fx) + (v11 - v10) * fx;
    (v, [gx, gy])
}

/// Cubic point at parameter t given start `p0` and the triple `(c1, c2, p3)`.
fn cubic_at(p0: P, c1: P, c2: P, p3: P, t: f64) -> P {
    let u = 1.0 - t;
    let (b0, b1, b2, b3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    [
        b0 * p0[0] + b1 * c1[0] + b2 * c2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * c1[1] + b2 * c2[1] + b3 * p3[1],
    ]
}

/// Squared field residual over 8 samples per cubic segment.
fn residual(
    field: &[f32],
    spw: usize,
    sph: usize,
    ox: f64,
    oy: f64,
    start: P,
    segs: &[Seg],
) -> f64 {
    let mut acc = 0.0_f64;
    let mut cur = start;
    for seg in segs {
        if let Seg::Cubic(c1, c2, p3) = seg {
            for k in 1..=8 {
                let t = f64::from(k) / 8.0;
                let p = cubic_at(cur, *c1, *c2, *p3, t);
                let lx = p[0] - ox + 0.5;
                let ly = p[1] - oy + 0.5;
                let (v, _) = bilinear(field, spw, sph, lx, ly);
                let r = v - 0.5;
                acc += r * r;
            }
        }
        cur = match seg {
            Seg::Line(p) | Seg::Cubic(_, _, p) | Seg::Arc { p, .. } => *p,
        };
    }
    acc
}

/// Refines interior control points of cubic segments against the coverage field.
///
/// Returns `true` when the refined version was accepted (lower residual),
/// or `false` when the original segments were restored.
pub fn refine_ring(
    field: &[f32],
    spw: usize,
    sph: usize,
    ox: f64,
    oy: f64,
    start: P,
    segs: &mut [Seg],
    iters: u8,
) -> bool {
    if iters == 0 || !segs.iter().any(|s| matches!(s, Seg::Cubic(..))) {
        return false;
    }
    let before = residual(field, spw, sph, ox, oy, start, segs);
    let original = segs.to_vec();
    for _ in 0..iters {
        let mut p_prev = start;
        for seg in segs.iter_mut() {
            let (mut c1, mut c2, p3) = match seg {
                Seg::Cubic(c1, c2, p3) => (*c1, *c2, *p3),
                Seg::Line(p) | Seg::Arc { p, .. } => {
                    p_prev = *p;
                    continue;
                }
            };
            for k in 1..=8 {
                let t = f64::from(k) / 8.0;
                let p = cubic_at(p_prev, c1, c2, p3, t);
                let lx = p[0] - ox + 0.5;
                let ly = p[1] - oy + 0.5;
                let (v, g) = bilinear(field, spw, sph, lx, ly);
                let r = v - 0.5;
                c1[0] -= LR * r * g[0] * w1(t);
                c1[1] -= LR * r * g[1] * w1(t);
                c2[0] -= LR * r * g[0] * w2(t);
                c2[1] -= LR * r * g[1] * w2(t);
            }
            *seg = Seg::Cubic(c1, c2, p3);
            p_prev = p3;
        }
    }
    let after = residual(field, spw, sph, ox, oy, start, segs);
    if after < before {
        true
    } else {
        segs.copy_from_slice(&original);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field with a linear ramp across the iso-line at image x = 7.5 (a soft
    /// edge, as real coverage fields are); a cubic biased inward of the
    /// iso-line must be pulled back toward it with a lower residual.
    #[test]
    fn refinement_pulls_biased_curve_toward_the_iso_line() {
        let (spw, sph) = (16_usize, 16_usize);
        let mut field = vec![0.0_f32; spw * sph];
        for y in 0..sph {
            for x in 0..spw {
                // ramp from lattice 7 (0.0) to lattice 9 (1.0)
                field[y * spw + x] = (((x as f32) - 7.0) / 2.0).clamp(0.0, 1.0);
            }
        }
        // ox = oy = 0 → image point x maps to lattice x + 0.5; the iso-line
        // (field 0.5) sits at lattice 8 → image x = 7.5.
        let start = [7.5, 4.0];
        let mut segs = vec![Seg::Cubic([7.1, 6.0], [7.1, 8.0], [7.5, 10.0])];
        let before = residual(&field, spw, sph, 0.0, 0.0, start, &segs);
        assert!(before > 0.01, "biased curve must start with residual");
        let accepted = refine_ring(&field, spw, sph, 0.0, 0.0, start, &mut segs, 12);
        assert!(accepted, "refinement must improve the residual");
        let after = residual(&field, spw, sph, 0.0, 0.0, start, &segs);
        assert!(after < before * 0.5, "residual {before:.4} → {after:.4}");
        // The mid-curve point must now sit close to the iso-line.
        if let Seg::Cubic(c1, c2, p3) = &segs[0] {
            let mid = cubic_at(start, *c1, *c2, *p3, 0.5);
            assert!((mid[0] - 7.5).abs() < 0.25, "midpoint at {mid:?}");
        }
    }

    /// Refinement must refuse to make things worse: an already-optimal
    /// straight-edge cubic is left alone (endpoints pinned).
    #[test]
    fn refinement_is_rejected_when_it_would_worsen() {
        let (spw, sph) = (16_usize, 16_usize);
        let mut field = vec![0.0_f32; spw * sph];
        for y in 0..sph {
            for x in 0..spw {
                field[y * spw + x] = if x >= 8 { 1.0 } else { 0.0 };
            }
        }
        let start = [7.5, 4.0];
        let orig = vec![Seg::Cubic([7.5, 6.0], [7.5, 8.0], [7.5, 10.0])];
        let mut segs = orig.clone();
        // The curve already follows the iso-line: any change is rejected.
        refine_ring(&field, spw, sph, 0.0, 0.0, start, &mut segs, 6);
        assert_eq!(segs, orig, "original segments restored");
    }

    #[test]
    fn lines_and_cubics_mix_is_untouched_outside_cubics() {
        let (spw, sph) = (16_usize, 16_usize);
        let field = vec![0.5_f32; spw * sph];
        let start = [1.0, 1.0];
        let line = Seg::Line([2.0, 2.0]);
        let mut segs = vec![line, Seg::Cubic([3.0, 3.0], [4.0, 4.0], [5.0, 5.0])];
        // Field is uniformly 0.5 → r = 0 everywhere → no movement, accepted
        // as a (weak) improvement or equal; either way the line stays.
        let before = residual(&field, spw, sph, 0.0, 0.0, start, &segs);
        assert!(before.abs() < 1e-9);
        refine_ring(&field, spw, sph, 0.0, 0.0, start, &mut segs, 3);
        assert_eq!(segs[0], line);
    }

    #[test]
    fn bilinear_value_and_gradient_are_consistent() {
        let (spw, sph) = (8_usize, 8_usize);
        let mut field = vec![0.0_f32; spw * sph];
        for y in 0..sph {
            for x in 0..spw {
                field[y * spw + x] = x as f32; // v = lx exactly
            }
        }
        let (v, g) = bilinear(&field, spw, sph, 3.25, 4.0);
        assert!((v - 3.25).abs() < 1e-6, "v={v}");
        assert!((g[0] - 1.0).abs() < 1e-6, "gx={g:?}");
        assert!(g[1].abs() < 1e-6, "gy={g:?}");
    }
}
