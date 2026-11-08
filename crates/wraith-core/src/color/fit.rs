//! Ring fitting: corner reconstruction, TLS line snapping and error-bounded
//! Schneider cubic fitting with G1 continuity at smooth joins (prototype
//! §4.5, roadmap M1/M4).
//!
//! Non-obvious part — **chamfer reconstruction**: the iso-contour of a
//! bilinear field chamfers sharp corners. A "chamfer" is 1–3 short RDP
//! segments between two long ones with a large total turn: the rounded
//! footprint of a sharp corner. The chain is replaced by the intersection of
//! the two long segments' lines. Straight pieces (TLS deviation below
//! tolerance, chord ≥ 4 px) become exact line segments, and a corner between
//! two snapped lines moves to their intersection — restoring sharp, exact
//! corners.

use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::color::contour::P;

/// A fitted ring segment: line, cubic, or elliptical arc to the end point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Seg {
    /// Straight line to the end point.
    Line(P),
    /// Cubic Bézier (interior controls + end point).
    Cubic(P, P, P),
    /// Elliptical arc (SVG `A` command, produced by primitive detection).
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
        /// End point.
        p: P,
    },
}

pub(crate) fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}

pub(crate) fn add(a: P, b: P) -> P {
    [a[0] + b[0], a[1] + b[1]]
}

pub(crate) fn mul(a: P, s: f64) -> P {
    [a[0] * s, a[1] * s]
}

pub(crate) fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

pub(crate) fn len(a: P) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn unit(a: P) -> P {
    let l = len(a);
    if l < 1e-12 {
        [0.0, 0.0]
    } else {
        mul(a, 1.0 / l)
    }
}

/// Perpendicular distance from `p` to the segment `a–b`.
#[must_use]
pub fn seg_dist(p: P, a: P, b: P) -> f64 {
    let ab = sub(b, a);
    let den = dot(ab, ab);
    let t = if den < 1e-12 {
        0.0
    } else {
        (dot(sub(p, a), ab) / den).clamp(0.0, 1.0)
    };
    len(sub(p, add(a, mul(ab, t))))
}

/// Signed area of a closed ring (shoelace).
#[must_use]
pub fn area(r: &[P]) -> f64 {
    let mut s = 0.0;
    for i in 0..r.len() {
        let (a, b) = (r[i], r[(i + 1) % r.len()]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    s * 0.5
}

/// Intersection of the lines through `a–b` and `c–d`, if any.
#[must_use]
pub fn line_isect(a: P, b: P, c: P, d: P) -> Option<P> {
    let (r, s) = (sub(b, a), sub(d, c));
    let den = r[0] * s[1] - r[1] * s[0];
    if den.abs() < 1e-9 {
        return None;
    }
    let t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den;
    Some(add(a, mul(r, t)))
}

/// Turn angle (radians) at `b` between segments `a–b` and `b–c`.
fn turn(a: P, b: P, c: P) -> f64 {
    let (u, v) = (unit(sub(b, a)), unit(sub(c, b)));
    dot(u, v).clamp(-1.0, 1.0).acos()
}

/// Ring-adapted Ramer–Douglas–Peucker: returns kept indices.
fn rdp_ring(pts: &[P], eps: f64) -> Vec<usize> {
    let n = pts.len();
    if n < 4 {
        return (0..n).collect();
    }
    let far = (1..n)
        .max_by(|&a, &b| {
            len(sub(pts[a], pts[0]))
                .partial_cmp(&len(sub(pts[b], pts[0])))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    for (a, b) in [(0_usize, far), (far, n)] {
        let mut st = vec![(a, b)];
        while let Some((a, b)) = st.pop() {
            if b <= a + 1 {
                continue;
            }
            let (pa, pb) = (pts[a % n], pts[b % n]);
            let (mut md, mut mi) = (0.0_f64, a);
            for i in a + 1..b {
                let d = seg_dist(pts[i % n], pa, pb);
                if d > md {
                    md = d;
                    mi = i;
                }
            }
            if md > eps {
                keep[mi % n] = true;
                st.push((a, mi));
                st.push((mi, b));
            }
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// Returns the ring with chamfered corners rebuilt as sharp corners, plus a
/// per-point corner flag. `corner_rad` is the minimum turn (radians) that
/// counts as a corner.
#[must_use]
pub fn find_corners(dense: &[P], corner_rad: f64) -> (Vec<P>, Vec<bool>) {
    let chamfer_rad = corner_rad.max(0.05);
    let flag_rad = chamfer_rad + 0.1;
    let n = dense.len();
    let idx = rdp_ring(dense, 0.2);
    let m = idx.len();
    if m < 4 {
        return (dense.to_vec(), vec![false; n]);
    }
    let v: Vec<P> = idx.iter().map(|&i| dense[i]).collect();
    let l = |a: usize, b: usize| len(sub(v[a % m], v[b % m]));
    // Chamfer detection: chain verts j..j+e (each < 2 px, total < 4.5 px)
    // followed by a long segment (≥ 3 px), with a large total turn.
    let mut chamfer: Vec<Option<(P, usize)>> = vec![None; m]; // (corner, chain length in verts)
    let mut used = vec![false; m];
    for j in 0..m {
        if used[j] {
            continue;
        }
        let pj = (j + m - 1) % m;
        if used[pj] || l(pj, j) < 3.0 {
            continue;
        }
        for e in 1..=3_usize {
            let k = (j + e) % m;
            if (0..e).any(|q| l(j + q, j + q + 1) >= 2.0) {
                break;
            }
            if (0..e).map(|q| l(j + q, j + q + 1)).sum::<f64>() > 4.5 {
                break;
            }
            if (0..=e).any(|q| used[(j + q) % m]) {
                break;
            }
            if l(k, k + 1) < 3.0 {
                continue;
            }
            let (a, b, c, d) = (v[pj], v[j], v[k], v[(k + 1) % m]);
            let ang = dot(unit(sub(b, a)), unit(sub(d, c)))
                .clamp(-1.0, 1.0)
                .acos();
            if ang < chamfer_rad {
                continue;
            }
            if let Some(x) = line_isect(a, b, c, d) {
                let mid = mul(add(b, c), 0.5);
                if len(sub(x, mid)) < 2.5 {
                    chamfer[j] = Some((x, e + 1));
                    for q in 0..=e {
                        used[(j + q) % m] = true;
                    }
                    break;
                }
            }
        }
    }
    let mut skip_first = vec![false; m];
    let mut skip_chunk = vec![false; m];
    for j in 0..m {
        if let Some((_, nv)) = chamfer[j] {
            // chunks j..j+nv-2 are dropped; the first point of chunk j+nv-1 too
            for q in 0..nv - 1 {
                skip_chunk[(j + q) % m] = true;
            }
            skip_first[(j + nv - 1) % m] = true;
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut cor = Vec::with_capacity(n);
    for j in 0..m {
        let (s0, e0) = (idx[j], idx[(j + 1) % m]);
        let chunk: Vec<P> = if j + 1 < m {
            dense[s0..e0].to_vec()
        } else {
            dense[s0..]
                .iter()
                .chain(dense[..e0].iter())
                .copied()
                .collect()
        };
        if let Some((x, _)) = chamfer[j] {
            out.push(x);
            cor.push(true);
        }
        if skip_chunk[j] {
            continue;
        }
        let start = usize::from(skip_first[j]);
        let is_corner = !used[j] && {
            let (a, b, c) = (v[(j + m - 1) % m], v[j], v[(j + 1) % m]);
            len(sub(b, a)) >= 2.0 && len(sub(c, b)) >= 2.0 && turn(a, b, c) >= flag_rad
        };
        for (k, p) in chunk.iter().enumerate().skip(start) {
            out.push(*p);
            cor.push(k == 0 && is_corner);
        }
    }
    (out, cor)
}

/// Binomial smoothing of the dense ring; corners stay pinned.
pub(crate) fn smooth_ring(r: &mut [P], cor: &[bool], iters: usize) {
    let n = r.len();
    if n < 5 {
        return;
    }
    for _ in 0..iters {
        let old = r.to_vec();
        for i in 0..n {
            if cor[i] {
                continue;
            }
            let (a, b) = (old[(i + n - 1) % n], old[(i + 1) % n]);
            r[i] = [
                (a[0] + 2.0 * old[i][0] + b[0]) * 0.25,
                (a[1] + 2.0 * old[i][1] + b[1]) * 0.25,
            ];
        }
    }
}

// ---- Schneider fitting -----------------------------------------------------

type Cubic = [P; 4];

fn bez(c: &Cubic, t: f64) -> P {
    let u = 1.0 - t;
    let (b0, b1, b2, b3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    [
        b0 * c[0][0] + b1 * c[1][0] + b2 * c[2][0] + b3 * c[3][0],
        b0 * c[0][1] + b1 * c[1][1] + b2 * c[2][1] + b3 * c[3][1],
    ]
}

fn bez1(c: &Cubic, t: f64) -> P {
    let u = 1.0 - t;
    let d = [sub(c[1], c[0]), sub(c[2], c[1]), sub(c[3], c[2])];
    let (a, b, cc) = (3.0 * u * u, 6.0 * u * t, 3.0 * t * t);
    [
        a * d[0][0] + b * d[1][0] + cc * d[2][0],
        a * d[0][1] + b * d[1][1] + cc * d[2][1],
    ]
}

fn bez2(c: &Cubic, t: f64) -> P {
    let u = 1.0 - t;
    let d = [sub(c[1], c[0]), sub(c[2], c[1]), sub(c[3], c[2])];
    let e = [sub(d[1], d[0]), sub(d[2], d[1])];
    [
        6.0 * (u * e[0][0] + t * e[1][0]),
        6.0 * (u * e[0][1] + t * e[1][1]),
    ]
}

fn chord_param(d: &[P]) -> Vec<f64> {
    let mut u = vec![0.0; d.len()];
    for i in 1..d.len() {
        u[i] = u[i - 1] + len(sub(d[i], d[i - 1]));
    }
    let tot = u[d.len() - 1].max(1e-12);
    for x in &mut u {
        *x /= tot;
    }
    u
}

fn gen_bezier(d: &[P], u: &[f64], t1: P, t2: P) -> Cubic {
    let n = d.len();
    let (p0, p3) = (d[0], d[n - 1]);
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let t = u[i];
        let s = 1.0 - t;
        let (b0, b1, b2, b3) = (s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t);
        let a1 = mul(t1, b1);
        let a2 = mul(t2, b2);
        c00 += dot(a1, a1);
        c01 += dot(a1, a2);
        c11 += dot(a2, a2);
        let tmp = sub(d[i], add(mul(p0, b0 + b1), mul(p3, b2 + b3)));
        x0 += dot(a1, tmp);
        x1 += dot(a2, tmp);
    }
    let det = c00 * c11 - c01 * c01;
    let (mut al, mut ar) = if det.abs() > 1e-12 {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (0.0, 0.0)
    };
    let seg = len(sub(p3, p0));
    if al < 1e-6 * seg || ar < 1e-6 * seg {
        al = seg / 3.0;
        ar = seg / 3.0;
    }
    [p0, add(p0, mul(t1, al)), add(p3, mul(t2, ar)), p3]
}

fn max_err(d: &[P], c: &Cubic, u: &[f64]) -> (f64, usize) {
    let (mut m, mut mi) = (0.0, d.len() / 2);
    for i in 1..d.len() - 1 {
        let e = dot(sub(bez(c, u[i]), d[i]), sub(bez(c, u[i]), d[i]));
        if e > m {
            m = e;
            mi = i;
        }
    }
    (m, mi)
}

fn reparam(d: &[P], u: &[f64], c: &Cubic) -> Vec<f64> {
    d.iter()
        .zip(u)
        .map(|(p, &t)| {
            let q = sub(bez(c, t), *p);
            let (q1, q2) = (bez1(c, t), bez2(c, t));
            let num = dot(q, q1);
            let den = dot(q1, q1) + dot(q, q2);
            if den.abs() < 1e-12 {
                t
            } else {
                (t - num / den).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Schneider fit of one open piece, splitting recursively until the max
/// squared error is within `e2` (the error bound is the whole point: the
/// smoothness/fidelity trade-off is a *guarantee*, not a heuristic).
fn fit_cubic(d: &[P], t1: P, t2: P, e2: f64, out: &mut Vec<Cubic>) {
    let n = d.len();
    if n < 2 {
        return;
    }
    if n == 2 {
        let dist = len(sub(d[1], d[0])) / 3.0;
        out.push([
            d[0],
            add(d[0], mul(t1, dist)),
            add(d[1], mul(t2, dist)),
            d[1],
        ]);
        return;
    }
    let mut u = chord_param(d);
    let mut c = gen_bezier(d, &u, t1, t2);
    let (mut me, mut split) = max_err(d, &c, &u);
    if me < e2 {
        out.push(c);
        return;
    }
    if me < e2 * 16.0 {
        for _ in 0..5 {
            u = reparam(d, &u, &c);
            c = gen_bezier(d, &u, t1, t2);
            let r = max_err(d, &c, &u);
            me = r.0;
            split = r.1;
            if me < e2 {
                out.push(c);
                return;
            }
        }
    }
    let split = split.clamp(1, n - 2);
    let ct = unit(sub(d[split - 1], d[split + 1]));
    fit_cubic(&d[..=split], t1, ct, e2, out);
    fit_cubic(&d[split..], mul(ct, -1.0), t2, e2, out);
}

/// Point `dist` pixels away from `ring[i]` along `dir` (±1), wrapping.
fn arc_pt(ring: &[P], i: usize, dir: isize, dist: f64) -> P {
    let n = ring.len() as isize;
    let mut acc = 0.0;
    let mut j = i as isize;
    for _ in 0..n {
        let nj = (j + dir).rem_euclid(n);
        acc += len(sub(ring[nj as usize], ring[j as usize]));
        j = nj;
        if acc >= dist {
            break;
        }
    }
    ring[j as usize]
}

/// Total-least-squares line through `pts`: (centroid, unit direction, max
/// deviation).
#[must_use]
pub fn tls_line(pts: &[P]) -> (P, P, f64) {
    let n = pts.len() as f64;
    let c = [
        pts.iter().map(|p| p[0]).sum::<f64>() / n,
        pts.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let d = [th.cos(), th.sin()];
    let dev = pts
        .iter()
        .map(|p| {
            let v = sub(*p, c);
            (v[0] * d[1] - v[1] * d[0]).abs()
        })
        .fold(0.0, f64::max);
    (c, d, dev)
}

/// Fits one closed ring: cuts at corners (or at two synthetic cuts when the
/// ring is corner-free), snaps straight pieces to their TLS line, restores
/// corners as line intersections, and Schneider-fits the curved remainder.
#[must_use]
pub fn fit_ring(ring: &[P], corner: &[bool], tol: f64) -> (P, Vec<Seg>) {
    let n = ring.len();
    let mut cuts: Vec<usize> = (0..n).filter(|&i| corner[i]).collect();
    let smooth_cut = cuts.is_empty();
    if smooth_cut {
        cuts = vec![0, n / 2];
    }
    let nc = cuts.len();
    // Pieces between consecutive cuts.
    let mut pieces: Vec<Vec<P>> = Vec::with_capacity(nc);
    for ci in 0..nc {
        let (s, e) = (cuts[ci], cuts[(ci + 1) % nc]);
        let mut piece = vec![ring[s]];
        let mut i = s;
        while piece.len() <= n {
            i = (i + 1) % n;
            piece.push(ring[i]);
            if i == e {
                break;
            }
        }
        pieces.push(piece);
    }
    // Straight pieces snap to their TLS line; corners between two snapped
    // lines become the lines' intersection.
    let lines: Vec<Option<(P, P)>> = pieces
        .iter()
        .map(|pc| {
            if smooth_cut || pc.len() < 4 {
                return None;
            }
            let (c, d, dev) = tls_line(pc);
            let chord = len(sub(pc[pc.len() - 1], pc[0]));
            (chord >= 4.0 && dev < tol).then_some((c, d))
        })
        .collect();
    let mut cpos: Vec<P> = cuts.iter().map(|&i| ring[i]).collect();
    if !smooth_cut {
        for ci in 0..nc {
            let (prev, next) = (lines[(ci + nc - 1) % nc], lines[ci]);
            if let (Some((c1, d1)), Some((c2, d2))) = (prev, next)
                && let Some(x) = line_isect(c1, add(c1, d1), c2, add(c2, d2))
                && len(sub(x, cpos[ci])) < 1.5
            {
                cpos[ci] = x;
            }
        }
    }
    let mut segs = Vec::new();
    for ci in 0..nc {
        let piece = &mut pieces[ci];
        let last = piece.len() - 1;
        piece[0] = cpos[ci];
        piece[last] = cpos[(ci + 1) % nc];
        let (p0, p1) = (piece[0], piece[last]);
        if piece.len() == 2 || lines[ci].is_some() {
            segs.push(Seg::Line(p1));
            continue;
        }
        let dev = piece
            .iter()
            .map(|p| seg_dist(*p, p0, p1))
            .fold(0.0, f64::max);
        if dev < 0.12 {
            segs.push(Seg::Line(p1));
            continue;
        }
        let (s, e) = (cuts[ci], cuts[(ci + 1) % nc]);
        let t1 = if smooth_cut {
            unit(sub(arc_pt(ring, s, 1, 1.5), arc_pt(ring, s, -1, 1.5)))
        } else {
            unit(sub(arc_pt(ring, s, 1, 1.5), ring[s]))
        };
        let t2 = if smooth_cut {
            unit(sub(arc_pt(ring, e, -1, 1.5), arc_pt(ring, e, 1, 1.5)))
        } else {
            unit(sub(arc_pt(ring, e, -1, 1.5), ring[e]))
        };
        let mut cs = Vec::new();
        fit_cubic(piece, t1, t2, tol * tol, &mut cs);
        for c in cs {
            segs.push(Seg::Cubic(c[1], c[2], c[3]));
        }
    }
    (cpos[0], segs)
}

/// Fits many rings in parallel (M2: rings are independent).
pub(crate) fn fit_rings_parallel(
    rings: &[Vec<P>],
    tol: f64,
    corner_rad: f64,
    smooth: usize,
) -> Vec<(P, Vec<Seg>)> {
    rings
        .par_iter()
        .map(|ring| {
            let (mut ring2, cor) = find_corners(ring, corner_rad);
            smooth_ring(&mut ring2, &cor, smooth);
            fit_ring(&ring2, &cor, tol)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense ring of a unit square (CW or CCW), sampled at 0.25 px.
    fn square_ring(side: f64, origin: P) -> Vec<P> {
        let mut ring = Vec::new();
        let n = (side / 0.25).round() as usize;
        for i in 0..n {
            let t = side * i as f64 / n as f64;
            ring.push([origin[0] + t, origin[1]]);
        }
        for i in 0..n {
            let t = side * i as f64 / n as f64;
            ring.push([origin[0] + side, origin[1] + t]);
        }
        for i in 0..n {
            let t = side * i as f64 / n as f64;
            ring.push([origin[0] + side - t, origin[1] + side]);
        }
        for i in 0..n {
            let t = side * i as f64 / n as f64;
            ring.push([origin[0], origin[1] + side - t]);
        }
        ring
    }

    #[test]
    fn corners_of_a_square_are_found() {
        let ring = square_ring(10.0, [0.0, 0.0]);
        let (pts, cor) = find_corners(&ring, 40.0_f64.to_radians());
        let n_corners = cor.iter().filter(|&&c| c).count();
        assert_eq!(n_corners, 4, "square must have exactly 4 corners");
        // The corner points must sit at (0,0), (10,0), (10,10), (0,10).
        let mut found = 0;
        for (&c, p) in cor.iter().zip(&pts) {
            if c {
                found += 1;
                assert!(
                    (p[0] < 0.5 || (p[0] - 10.0).abs() < 0.5)
                        && (p[1] < 0.5 || (p[1] - 10.0).abs() < 0.5),
                    "corner at {p:?}"
                );
            }
        }
        assert_eq!(found, 4);
    }

    #[test]
    fn schneider_fit_respects_error_bound() {
        // A quarter circle sampled densely must be fitted within tolerance.
        let arc: Vec<P> = (0..=64)
            .map(|i| {
                let t = std::f64::consts::FRAC_PI_2 * f64::from(i) / 64.0;
                [t.cos() * 50.0, t.sin() * 50.0]
            })
            .collect();
        let t1 = [0.0, 1.0];
        let t2 = [1.0, 0.0];
        let mut out = Vec::new();
        fit_cubic(&arc, t1, t2, 0.3_f64 * 0.3, &mut out);
        assert!(!out.is_empty());
        // Max deviation of the union of cubics vs the source arc.
        for c in &out {
            for i in 0..=32 {
                let t = f64::from(i) / 32.0;
                let p = bez(c, t);
                // Nearest source sample (arc is densely sampled).
                let md = arc.iter().map(|q| len(sub(*q, p))).fold(f64::MAX, f64::min);
                assert!(md <= 0.35, "deviation {md:.3} exceeds bound");
            }
        }
    }

    #[test]
    fn fit_ring_snaps_straight_pieces_to_lines() {
        let ring = square_ring(20.0, [3.0, 3.0]);
        let (pts, cor) = find_corners(&ring, 40.0_f64.to_radians());
        let (start, segs) = fit_ring(&pts, &cor, 0.3);
        // A perfect square: 4 segments, all lines.
        assert_eq!(segs.len(), 4);
        assert!(segs.iter().all(|s| matches!(s, Seg::Line(_))));
        // Corners restored to exact intersections.
        let ends = [start]
            .into_iter()
            .chain(segs.iter().map(|s| match s {
                Seg::Line(p) | Seg::Cubic(_, _, p) | Seg::Arc { p, .. } => *p,
            }))
            .collect::<Vec<_>>();
        for p in &ends {
            assert!(
                (p[0] - 3.0).abs() < 0.1
                    || (p[0] - 23.0).abs() < 0.1
                    || (p[1] - 3.0).abs() < 0.1
                    || (p[1] - 23.0).abs() < 0.1,
                "end {p:?} not on square edge"
            );
        }
    }

    #[test]
    fn tls_line_recovers_orientation() {
        let pts: Vec<P> = (0..20)
            .map(|i| [f64::from(i) * 1.0, f64::from(i) * 0.5 + 0.01])
            .collect();
        let (c, d, dev) = tls_line(&pts);
        assert!(dev < 0.05, "deviation {dev}");
        // Direction ~ (1, 0.5) normalised.
        let u = unit([1.0, 0.5]);
        assert!((dot(d, u) - 1.0).abs() < 1e-6 || (dot(d, u) + 1.0).abs() < 1e-6);
        assert!((c[1] - 4.75).abs() < 0.1);
    }

    #[test]
    fn chamfer_reconstruction_sharpens_a_cut_corner() {
        // A square whose top-right corner (10,10) carries a 45° chamfer
        // chain: two 1.31-px segments between two ≥3-px segments. The chain
        // must be replaced by the sharp corner at the lines' intersection.
        let ring: Vec<P> = vec![
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [10.0, 8.0],
            [9.07, 9.07],
            [8.0, 10.0],
            [4.0, 10.0],
            [0.0, 10.0],
            [0.0, 5.0],
        ];
        let (pts, cor) = find_corners(&ring, 40.0_f64.to_radians());
        // The chamfered corner should reconstruct near (10, 10).
        let sharp = pts
            .iter()
            .zip(&cor)
            .filter(|(_, c)| **c)
            .map(|(p, _)| *p)
            .find(|p| p[0] > 7.5 && p[1] > 7.5);
        assert!(sharp.is_some(), "chamfer not reconstructed: {cor:?}");
        let p = sharp.unwrap_or([0.0, 0.0]);
        assert!(
            (p[0] - 10.0).abs() < 0.6 && (p[1] - 10.0).abs() < 0.6,
            "{p:?}"
        );
    }

    #[test]
    fn area_sign_tracks_orientation() {
        let ring = square_ring(4.0, [0.0, 0.0]);
        assert!((area(&ring).abs() - 16.0).abs() < 1e-6);
    }
}
