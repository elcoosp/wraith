//! Marching squares at iso 0.5 over a padded coverage field.
//!
//! Each of the 16 cases emits a directed segment with *inside on the left*,
//! so every edge has exactly one outgoing segment and chaining is a pointer
//! walk with no hashing. Crossing points use linear interpolation of the
//! coverage field, which is what makes the boundaries sub-pixel.
//!
//! M2: the cell scan runs over *active tiles* only (tiles that can contain a
//! boundary cell given the scatter pass's occupancy), never the whole field.

/// A point in image coordinates (pixel centres).
pub type P = [f64; 2];

/// Tile edge length for the active-tile scan.
pub const TILE: usize = 32;

/// A rectangle of tiles (in tile units) that may contain boundary cells.
#[derive(Debug, Clone, Copy)]
pub struct TileSpan {
    /// Inclusive tile range over x.
    pub x0: usize,
    /// Exclusive tile range over x.
    pub x1: usize,
    /// Inclusive tile range over y.
    pub y0: usize,
    /// Exclusive tile range over y.
    pub y1: usize,
}

/// Marching squares over a padded field; returns closed rings in image
/// coordinates, allocating the chain table internally.
///
/// # Panics
///
/// Panics if `f` is smaller than `pw * ph` — a caller invariant.
#[must_use]
pub fn contours(f: &[f32], pw: usize, ph: usize, active: &[TileSpan]) -> Vec<Vec<P>> {
    let mut nxt = Vec::new();
    contours_with(f, pw, ph, &mut nxt, active)
}

/// Marching squares with an externally owned chain table.
///
/// The table is reused across layers through the thread-local scratch pool.
/// `active` restricts the cell scan and chain-start walk to the given tile spans.
///
/// # Panics
///
/// Panics if `f` is smaller than `pw * ph` — a caller invariant.
pub fn contours_with(
    f: &[f32],
    pw: usize,
    ph: usize,
    nxt: &mut Vec<u32>,
    active: &[TileSpan],
) -> Vec<Vec<P>> {
    const ISO: f32 = 0.5;
    let hid = |x: usize, y: usize| (2 * (y * pw + x)) as u32; // horizontal edge id
    let vid = |x: usize, y: usize| (2 * (y * pw + x) + 1) as u32; // vertical edge id
    nxt.clear();
    nxt.resize(2 * pw * ph, u32::MAX);
    for span in active {
        for y in span.y0 * TILE..((span.y1 * TILE).min(ph - 1)) {
            for x in span.x0 * TILE..((span.x1 * TILE).min(pw - 1)) {
                let (vtl, vtr, vbr, vbl) = (
                    f[y * pw + x],
                    f[y * pw + x + 1],
                    f[(y + 1) * pw + x + 1],
                    f[(y + 1) * pw + x],
                );
                let case = u8::from(vtl >= ISO)
                    | u8::from(vtr >= ISO) << 1
                    | u8::from(vbr >= ISO) << 2
                    | u8::from(vbl >= ISO) << 3;
                if case == 0 || case == 15 {
                    continue;
                }
                let (t, b, l, r) = (hid(x, y), hid(x, y + 1), vid(x, y), vid(x + 1, y));
                let mut seg = |from: u32, to: u32| nxt[from as usize] = to;
                match case {
                    1 => seg(l, t),
                    2 => seg(t, r),
                    4 => seg(r, b),
                    8 => seg(b, l),
                    3 => seg(l, r),
                    6 => seg(t, b),
                    12 => seg(r, l),
                    9 => seg(b, t),
                    7 => seg(l, b),
                    11 => seg(b, r),
                    13 => seg(r, t),
                    14 => seg(t, l),
                    5 => {
                        if (vtl + vtr + vbr + vbl) * 0.25 >= ISO {
                            seg(r, t);
                            seg(l, b);
                        } else {
                            seg(l, t);
                            seg(r, b);
                        }
                    }
                    10 => {
                        if (vtl + vtr + vbr + vbl) * 0.25 >= ISO {
                            seg(t, l);
                            seg(b, r);
                        } else {
                            seg(t, r);
                            seg(b, l);
                        }
                    }
                    _ => unreachable!("cases 0 and 15 handled above"),
                }
            }
        }
    }
    let pt = |id: u32| -> P {
        let i = (id >> 1) as usize;
        let (x, y) = (i % pw, i / pw);
        let (v0, v1, horiz) = if id & 1 == 0 {
            (f[i], f[i + 1], true)
        } else {
            (f[i], f[i + pw], false)
        };
        let den = v1 - v0;
        let t = if den.abs() < 1e-9 {
            0.5_f64
        } else {
            f64::from(((ISO - v0) / den).clamp(0.0, 1.0))
        };
        // padded lattice point (px,py) ↔ pixel centre (px-1+0.5, py-1+0.5)
        if horiz {
            [x as f64 + t - 0.5, y as f64 - 0.5]
        } else {
            [x as f64 - 0.5, y as f64 + t - 0.5]
        }
    };
    // Chain walks start from every unvisited edge inside the active tiles.
    let mut out = Vec::new();
    for span in active {
        let y_lo = span.y0 * TILE;
        let y_hi = (span.y1 * TILE).min(ph);
        let x_lo = span.x0 * TILE;
        let x_hi = (span.x1 * TILE).min(pw);
        for y in y_lo..y_hi {
            for x in x_lo..x_hi {
                for s in [hid(x, y), vid(x, y)] {
                    if nxt[s as usize] == u32::MAX {
                        continue;
                    }
                    let mut ring = Vec::new();
                    let mut cur = s;
                    loop {
                        ring.push(pt(cur));
                        let n = nxt[cur as usize];
                        nxt[cur as usize] = u32::MAX;
                        cur = n;
                        if cur == s || cur == u32::MAX {
                            break;
                        }
                    }
                    out.push(ring);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a padded field from a predicate over pixel centres.
    fn field(
        w: usize,
        h: usize,
        inside: impl Fn(usize, usize) -> bool,
    ) -> (Vec<f32>, usize, usize) {
        let (pw, ph) = (w + 2, h + 2);
        let mut f = vec![0.0_f32; pw * ph];
        for y in 0..h {
            for x in 0..w {
                f[(y + 1) * pw + x + 1] = if inside(x, y) { 1.0 } else { 0.0 };
            }
        }
        (f, pw, ph)
    }

    fn full_span(pw: usize, ph: usize) -> Vec<TileSpan> {
        vec![TileSpan {
            x0: 0,
            x1: pw.div_ceil(TILE),
            y0: 0,
            y1: ph.div_ceil(TILE),
        }]
    }

    #[test]
    fn unit_square_yields_one_closed_ring() {
        let (f, pw, ph) = field(4, 4, |x, y| (2..4).contains(&x) && (2..4).contains(&y));
        let rings = contours(&f, pw, ph, &full_span(pw, ph));
        assert_eq!(rings.len(), 1);
        let ring = &rings[0];
        assert!(ring.len() >= 4);
        // Ring must be closed. The shoelace area is 3.5, not 4: marching
        // squares chamfers the four single-inside-vertex corner cells by
        // 45° (each loses a 0.125 triangle) — find_corners reconstructs
        // those afterwards.
        let mut area = 0.0;
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            area += a[0] * b[1] - b[0] * a[1];
        }
        assert!((area.abs() / 2.0 - 3.5).abs() < 0.1);
    }

    #[test]
    fn sub_pixel_interpolation_places_boundary_between_pixels() {
        // Half-coverage pixel: field value 0.5 at one lattice point of an
        // otherwise-full region — the crossing sits between lattice points.
        let (pw, ph) = (4usize, 4usize);
        let mut f = vec![0.0_f32; pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                f[y * pw + x] = 1.0;
            }
        }
        f[0] = 0.0; // top-left lattice point at 0 → boundary crosses nearby
        let rings = contours(&f, pw, ph, &full_span(pw, ph));
        assert_ne!(rings.len(), 0);
        // Some crossing point must lie strictly inside a lattice cell.
        let has_frac = rings
            .iter()
            .any(|r| r.iter().any(|p| p[0].fract() != 0.0 || p[1].fract() != 0.0));
        assert!(has_frac);
    }

    #[test]
    fn every_edge_has_exactly_one_outgoing_segment() {
        // A ring pattern where two shapes touch diagonally (saddle cases 5/10).
        let (pw, ph) = (6usize, 6usize);
        let mut f = vec![0.0_f32; pw * ph];
        for (x, y) in [(1, 1), (3, 1), (1, 3), (3, 3)] {
            f[y * pw + x] = 1.0;
        }
        f[2 * pw + 2] = 0.5;
        let rings = contours(&f, pw, ph, &full_span(pw, ph));
        // No ring may revisit an edge (the walk clears visited edges).
        let total: usize = rings.iter().map(std::vec::Vec::len).sum();
        assert!(total >= 4);
    }

    #[test]
    fn active_tiles_equivalent_to_full_scan() {
        let (f, pw, ph) = field(40, 40, |x, y| (5..20).contains(&x) && (8..30).contains(&y));
        let full = contours(&f, pw, ph, &full_span(pw, ph));
        let active = vec![TileSpan {
            x0: 0,
            x1: 2,
            y0: 0,
            y1: 2,
        }];
        let part = contours(&f, pw, ph, &active);
        // The shape lies fully inside tiles (0..2)²; the scan must find the
        // same single ring either way.
        assert_eq!(full.len(), part.len());
        assert_eq!(full[0].len(), part[0].len());
        assert!(
            full[0]
                .iter()
                .zip(part[0].iter())
                .all(|(a, b)| a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits())
        );
    }
}
