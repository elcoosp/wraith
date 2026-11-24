//! Gradient detection and fitting (roadmap M3).
//!
//! Region-first, MDL-style band merging. The pipeline treats smooth ramps as
//! many flat bands; the fix detects gradients *before* they are split into
//! layers:
//!
//! 1. the palette is over-segmented (K = 32–64), so a ramp becomes a chain of
//!    adjacent bands;
//! 2. a rank adjacency graph is built (edge weight = shared boundary length);
//! 3. neighbouring ranks are merged greedily when the union's best-fit
//!    affine colour model `c(x,y) = c0 + gx·x + gy·y` (per channel, gamma
//!    sRGB) has RMS residual below δ ≈ 2/255 *and* the merge reduces a
//!    description-length cost (path bytes saved − gradient-definition bytes);
//! 4. a merged group of ≥ 3 ranks becomes a gradient region; the gradient is
//!    fitted along the dominant gradient axis (linear) or by Gauss–Newton on
//!    the centre (radial), whichever has the lower residual, and stops are
//!    reduced to 2–6 by RDP in colour space;
//! 5. the region outline is traced with the *same* soft-field machinery, so
//!    gradient regions interoperate with trapping and flat layers.
//!
//! A gradient region is never created from a ≤ 2-rank chain (the M3 gate).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::color::model::{Mixture, dominant_rank};
use crate::color::palette::{Uniques, V3, rgb_to_oklab};
use crate::color::{GradientDef, GradientKind};
use crate::settings::{ColorSettings, Gradients};

/// Identity hasher for small integer keys.
#[derive(Clone, Default)]
struct Idx(u64);

impl Hasher for Idx {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.write_u64(u64::from(i));
    }

    fn write_u32(&mut self, i: u32) {
        self.write_u64(u64::from(i));
    }

    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }
}

type IdxMap<K, V> = HashMap<K, V, BuildHasherDefault<Idx>>;

/// Affine linear fit of a colour ramp: `c(x,y) = c0 + gx·x + gy·y` per
/// channel, in normalised gamma sRGB.
struct LinearFit {
    c0: [f64; 3],
    gx: [f64; 3],
    gy: [f64; 3],
    rmse: f64,
}

/// Radial fit: colour as a function of distance from a Gauss–Newton-fitted
/// centre; stops along the radius.
struct RadialFit {
    cx: f64,
    cy: f64,
    r: f64,
    /// Per-channel `c0` and gradient `g` of the `c0 + g·r` model.
    c0: [f64; 3],
    g: [f64; 3],
    rmse: f64,
}

/// Per-region classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Single palette rank, solid palette colour.
    Flat,
    /// Gradient region; index into [`Regions::gradient_defs`].
    Gradient(usize),
}

/// Region assignment over palette ranks.
pub struct Regions {
    /// Palette rank → region id.
    pub band_to_region: Vec<u16>,
    /// Kind per region id.
    pub kinds: Vec<RegionKind>,
    /// Palette rank of each single-rank flat region (its fill colour).
    pub flat_rank: Vec<usize>,
    /// Pixel count per region id (dominant-rank pixels).
    pub pixel_count: Vec<u32>,
    /// Fitted gradient definitions.
    pub gradient_defs: Vec<GradientDef>,
}

impl Regions {
    /// Paint rank (0 = largest pixel count) per region id.
    #[must_use]
    pub fn paint_ranks(&self) -> Vec<u16> {
        let mut order: Vec<usize> = (0..self.pixel_count.len()).collect();
        order.sort_by(|&a, &b| {
            self.pixel_count[b]
                .partial_cmp(&self.pixel_count[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut ranks = vec![0_u16; self.pixel_count.len()];
        for (r, &region) in order.iter().enumerate() {
            ranks[region] = r as u16;
        }
        ranks
    }
}

/// Identity regions: one region per palette rank (gradient detection off).
#[must_use]
pub fn flat_regions(pal_len: usize, rank_pixel_counts: &[u32]) -> Regions {
    Regions {
        band_to_region: (0..pal_len).map(|i| i as u16).collect(),
        kinds: vec![RegionKind::Flat; pal_len],
        flat_rank: (0..pal_len).collect(),
        pixel_count: rank_pixel_counts.to_vec(),
        gradient_defs: Vec::new(),
    }
}

/// One sample point of a ramp: coordinates + normalised sRGB colour.
struct Sample {
    x: f64,
    y: f64,
    c: [f32; 3],
}

/// Full gradient detection: over-segmented ranks → adjacency → MDL merges →
/// fits. Falls back to flat regions when nothing qualifies.
#[must_use]
pub fn detect_regions(
    uniques: &Uniques,
    pixels: &[u8],
    w: usize,
    h: usize,
    pal: &[V3],
    mix: &Mixture,
    settings: &ColorSettings,
) -> Regions {
    let n = pal.len();
    if n < 3 {
        let counts = rank_pixel_counts(uniques, pixels, mix);
        return flat_regions(n, &counts);
    }
    let opaque = |i: usize| pixels.get(i * 4 + 3).is_some_and(|&a| a > 0);
    let lab: Vec<u8> = uniques
        .id
        .iter()
        .enumerate()
        .map(|(i, &uid)| {
            if opaque(i) {
                dominant_rank(uid as usize, mix) as u8
            } else {
                u8::MAX
            }
        })
        .collect();

    // Rank adjacency: boundary length and mean Oklab difference per rank
    // pair (right + down links). The mean ΔOklab is the roadmap's edge
    // weight: smooth ramps have small steps, flat boundaries large ones.
    let ulab: Vec<V3> = uniques.colors.iter().map(|c| rgb_to_oklab(*c)).collect();
    let mut adj: IdxMap<u64, (u32, f64)> = IdxMap::default();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let la = lab[i];
            if la == u8::MAX {
                continue;
            }
            let la_u = uniques.id[i] as usize;
            for (j, n) in [(i + 1, x + 1 < w), (i + w, y + 1 < h)] {
                if !n {
                    continue;
                }
                let lb = lab[j];
                if lb == u8::MAX || lb == la {
                    continue;
                }
                let lb_u = uniques.id[j] as usize;
                let d = oklab_dist(ulab[la_u], ulab[lb_u]);
                let e = adj.entry(key(la, lb)).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += d;
            }
        }
    }

    // Pixel indices per rank.
    let mut rank_pixels: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, &l) in lab.iter().enumerate() {
        if l != u8::MAX {
            rank_pixels[l as usize].push(i as u32);
        }
    }

    // Per-rank perimeter estimate: sum of boundary links (each rank's outline
    // length in pixel steps, image-border pixels counted too).
    let mut perim: Vec<f64> = vec![0.0; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let l = lab[i];
            if l == u8::MAX {
                continue;
            }
            let mut edge = 0_u32;
            if x == 0 || lab[y * w + x - 1] != l {
                edge += 1;
            }
            if x + 1 == w || lab[y * w + x + 1] != l {
                edge += 1;
            }
            if y == 0 || lab[(y - 1) * w + x] != l {
                edge += 1;
            }
            if y + 1 == h || lab[(y + 1) * w + x] != l {
                edge += 1;
            }
            perim[l as usize] += f64::from(edge);
        }
    }

    // Union-find; candidate pairs ordered by boundary length (descending).
    let mut parent: Vec<usize> = (0..n).collect();
    let mut pixels_per_root: Vec<Vec<u32>> = rank_pixels.clone();
    let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    };
    let perim0: Vec<f64> = perim.clone();
    let mut pairs: Vec<(u32, f64, u8, u8)> = adj
        .iter()
        .map(|(&k, &(cnt, dsum))| {
            let (a, b) = unkey(k);
            (cnt, dsum / f64::from(cnt).max(1.0), a, b)
        })
        .collect();
    // Greedy order: most-similar neighbours first (roadmap M3 step 3) — ramp
    // chains form before any big flat region can contaminate them.
    pairs.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));

    let radial_on = matches!(settings.gradients, Gradients::LinearRadial);
    let mut merged: Vec<Vec<usize>> = Vec::new(); // accepted groups (rank lists)
    for &(boundary, _d_mean, a, b) in &pairs {
        if boundary < 4 {
            continue; // noise-level contact cannot carry a ramp
        }
        let (ra, rb) = (
            find(&mut parent, usize::from(a)),
            find(&mut parent, usize::from(b)),
        );
        if ra == rb {
            continue;
        }
        // Balanced sample: neither rank may dominate the fit (a 100k-px
        // background would drown a 3k-px stripe in a stride sample).
        let union_px = balanced_union(&pixels_per_root[ra], &pixels_per_root[rb]);
        // Pairwise gate: the union must be consistent under the affine model
        // *or* — when enabled — the radial model (a curved ramp is affine
        // only while the merged ring stays thin). The residual is per-pixel
        // RMS against the model the renderer will evaluate — the roadmap M3
        // definition of "consistent". MDL, width and rank-count gates apply
        // to the *final* group.
        // Ordering gate: the two sides must be distinct segments of the
        // projection axis, not interleaved texture shades (see
        // [`ranks_ordered_along`]).
        let ordered = |fit: Option<LinearFit>, radial: Option<RadialFit>| -> bool {
            if let Some(f) = &fit {
                return ranks_ordered_along(linear_proj(f), uniques, &union_px, &lab);
            }
            if let Some(r) = &radial {
                return ranks_ordered_along(radial_proj(r), uniques, &union_px, &lab);
            }
            false
        };
        let lin_fit = fit_linear_sample(uniques, &union_px);
        let rad_fit = radial_on.then(|| fit_radial(uniques, &union_px)).flatten();
        let residual = union_residual(uniques, &union_px, radial_on);
        if residual < DELTA_ABS && ordered(lin_fit, rad_fit) {
            parent[rb] = ra;
            pixels_per_root[ra] = concat_all(&pixels_per_root[ra], &pixels_per_root[rb]);
            pixels_per_root[rb].clear();
        }
    }
    // Collect surviving groups and drop those with fewer than 3 ranks.
    {
        let mut by_root: IdxMap<usize, Vec<usize>> = IdxMap::default();
        for r in 0..n {
            by_root.entry(find(&mut parent, r)).or_default().push(r);
        }
        for (_, mut ranks) in by_root {
            ranks.sort_unstable();
            if ranks.len() >= 3 {
                merged.push(ranks);
            }
        }
    }

    // Group gates: each candidate must (a) still fit a gradient on its own,
    // (b) hold ≥ 3 ranks (never a ≤ 2-band chain — the M3 gate), (c) be wide
    // enough to be a *region* rather than a 1-px AA fringe, and (d) pass the
    // MDL test: one ring + a gradient definition must be cheaper than the
    // separate rings (isoperimetric estimate: perimeter ≈ 4·√area, ≈ 4 bytes
    // per perimeter pixel).
    let radial_enabled = matches!(settings.gradients, Gradients::LinearRadial);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut in_gradient = vec![false; n];
    for ranks in &merged {
        if ranks.len() < 3 {
            continue;
        }
        // Population-representative sample: the fit and its residual must
        // describe the region's real pixel distribution, not a per-rank
        // balanced mix (see [`population_sample`]).
        let px = population_sample(&rank_pixels, ranks, 8192);
        let fit = fit_linear_sample(uniques, &px);
        let linear = fit.as_ref().map_or(f64::MAX, |f| {
            bin_profile_residual(uniques, &px, linear_proj(f), f.rmse)
        });
        let radial_fit = if radial_enabled {
            fit_radial(uniques, &px)
        } else {
            None
        };
        let radial = radial_fit.as_ref().map_or(f64::MAX, |r| {
            bin_profile_residual(uniques, &px, radial_proj(r), r.rmse)
        });
        // Final gates: the rendered gradient (bin-profile residual) must
        // beat the flat-band fallback's per-pixel error
        // ([`gradient_bound`]), and the group's ranks must be ordered along
        // the winning model's axis (see [`ranks_ordered_along`]).
        let flat_pp = flat_baseline_rmse(uniques, &px, &lab);
        if linear.min(radial) >= gradient_bound(flat_pp) {
            continue;
        }
        let ordered = if radial < linear {
            ranks_ordered_along(radial_proj(radial_fit.as_ref().expect("radial won")), uniques, &px, &lab)
        } else {
            fit.as_ref()
                .is_some_and(|f| ranks_ordered_along(linear_proj(f), uniques, &px, &lab))
        };
        if !ordered {
            continue;
        }
        // True pixel count (the fit sample is capped per rank and would
        // skew the isoperimetric ratio); perimeter from the full-resolution
        // estimate. The ratio is half the characteristic width: a 1-px AA
        // fringe sits at ≈ 0.5. Real over-segmented *rings* of ~2 px radial
        // width sit at ≈ 1.0 — the palette's ghost pruning already removed
        // the true fringes, so the gate only needs to separate those two
        // regimes.
        let area: f64 = ranks.iter().map(|&r| rank_pixels[r].len() as f64).sum();
        let mut group_perim = 0.0;
        let mut group_boundary = 0_u32;
        for (a_i, &ra) in ranks.iter().enumerate() {
            group_perim += perim0[ra];
            for &rb in &ranks[a_i + 1..] {
                group_boundary += adj.get(&key(ra as u8, rb as u8)).map_or(0, |&(c, _)| c);
            }
        }
        group_perim -= 2.0 * f64::from(group_boundary);
        let width_ratio = area / group_perim.max(1.0);
        if width_ratio < 0.9 {
            continue;
        }
        let est_members: f64 = ranks
            .iter()
            .map(|&r| 16.0 * (rank_pixels[r].len() as f64).sqrt())
            .sum();
        let est_union =
            16.0 * (ranks.iter().map(|&r| rank_pixels[r].len()).sum::<usize>() as f64).sqrt();
        if est_union + 300.0 >= est_members + 100.0 {
            continue;
        }
        for &r in ranks {
            in_gradient[r] = true;
        }
        groups.push(ranks.clone());
    }

    // Build the region table: gradient groups first, then leftover singles.
    let mut band_to_region = vec![0_u16; n];
    let mut kinds: Vec<RegionKind> = Vec::new();
    let mut flat_rank: Vec<usize> = Vec::new();
    let mut pixel_count: Vec<u32> = Vec::new();
    let mut gradient_defs: Vec<GradientDef> = Vec::new();
    for ranks in &groups {
        let px_all: Vec<u32> = ranks
            .iter()
            .flat_map(|&r| rank_pixels[r].iter().copied())
            .collect();
        let px = sample_cap(&px_all, 8192);
        let fit = fit_linear_sample(uniques, &px).unwrap_or(LinearFit {
            c0: [0.0; 3],
            gx: [0.0; 3],
            gy: [0.0; 3],
            rmse: 1.0,
        });
        let region = kinds.len();
        for &r in ranks {
            band_to_region[r] = region as u16;
        }
        let def = if matches!(settings.gradients, Gradients::LinearRadial) {
            match fit_radial(uniques, &px) {
                Some(rf) if rf.rmse < fit.rmse => radial_def(&rf, uniques, &px),
                _ => linear_def(&fit, uniques, &px),
            }
        } else {
            linear_def(&fit, uniques, &px)
        };
        let idx = gradient_defs.len();
        gradient_defs.push(def);
        kinds.push(RegionKind::Gradient(idx));
        flat_rank.push(usize::MAX);
        pixel_count.push(px_all.len() as u32);
    }
    for r in 0..n {
        if !in_gradient[r] {
            let region = kinds.len();
            band_to_region[r] = region as u16;
            kinds.push(RegionKind::Flat);
            flat_rank.push(r);
            pixel_count.push(rank_pixels[r].len() as u32);
        }
    }
    Regions {
        band_to_region,
        kinds,
        flat_rank,
        pixel_count,
        gradient_defs,
    }
}

/// Dominant-rank pixel counts per palette rank (for identity regions).
#[must_use]
pub fn rank_pixel_counts(uniques: &Uniques, pixels: &[u8], mix: &Mixture) -> Vec<u32> {
    let mut counts = vec![0_u32; mix.a.len()];
    for (i, &uid) in uniques.id.iter().enumerate() {
        if pixels.get(i * 4 + 3).is_some_and(|&a| a > 0) {
            counts[dominant_rank(uid as usize, mix)] += 1;
        }
    }
    counts
}

fn key(a: u8, b: u8) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    (u64::from(lo) << 8) | u64::from(hi)
}

fn unkey(k: u64) -> (u8, u8) {
    (
        u8::try_from(k >> 8).unwrap_or(0),
        u8::try_from(k & 0xff).unwrap_or(0),
    )
}

/// Caps a pixel list to ≤ `cap` evenly strided samples (successor of the
/// older 4096-only `sample`).
fn sample_cap(px: &[u32], cap: usize) -> Vec<u32> {
    if px.len() <= cap {
        return px.to_vec();
    }
    let stride = px.len().div_ceil(cap).max(1);
    (0..px.len()).step_by(stride).map(|i| px[i]).collect()
}

/// Population-representative sample of a *group* of ranks: every rank keeps
/// its true share of the sample (total capped at `cap`), so the fitted model
/// and its residual describe the region as the renderer sees it.
///
/// The pairwise merge gate deliberately samples each side balanced
/// ([`balanced_union`]) so a huge background cannot drown a small stripe in a
/// merge *decision*; reusing that balance for the final group fit was
/// measured to tilt gradients toward minority bands — on the mascot corpus a
/// bright chest area merged with darker hip shades came out ΔRGB 23–31 too
/// dark across ~47 000 px because the chest's single dominant rank was
/// capped to 512 samples while four dark shade ranks contributed 4 × 512.
fn population_sample(rank_pixels: &[Vec<u32>], ranks: &[usize], cap: usize) -> Vec<u32> {
    let total: usize = ranks.iter().map(|&r| rank_pixels[r].len()).sum();
    if total <= cap {
        return ranks.iter().flat_map(|&r| rank_pixels[r].iter().copied()).collect();
    }
    let mut out = Vec::with_capacity(cap);
    for (k, &r) in ranks.iter().enumerate() {
        // Last rank absorbs the rounding remainder so `out` never exceeds cap.
        let share = if k + 1 == ranks.len() {
            cap - out.len()
        } else {
            (u64::from(rank_pixels[r].len() as u32) * cap as u64 / total as u64).max(1) as usize
        };
        out.extend(sample_cap(&rank_pixels[r], share));
    }
    out
}

/// Balanced union of two pixel lists: each side is capped independently so a
/// huge rank cannot drown a small one in the fit sample.
fn balanced_union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = sample_cap(a, 4096);
    out.extend_from_slice(&sample_cap(b, 4096));
    out
}

/// Full concatenation (the merged root keeps every pixel for later stages).
fn concat_all(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

/// Euclidean distance in Oklab.
fn oklab_dist(a: V3, b: V3) -> f64 {
    let (dx, dy, dz) = (
        f64::from(a[0] - b[0]),
        f64::from(a[1] - b[1]),
        f64::from(a[2] - b[2]),
    );
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Number of projection bins for the piecewise-profile residual.
const RESIDUAL_BINS: usize = 16;

/// Flat-band parity factor of the final acceptance bound.
///
/// The final gate must compare the two representations the engine actually
/// chooses between: rendering the group as one gradient (per-pixel error =
/// [`bin_profile_residual`], the bin profile the emitted stops interpolate)
/// or as flat bands (per-pixel error = deviation of each pixel from its own
/// rank's mean colour — the palette centroid it would be filled with).
/// A gradient is accepted when it beats that flat baseline by
/// [`FLAT_PARITY`], or meets [`DELTA_ABS`] outright.
///
/// This is what lets deliberately banded shading (band steps of several
/// 1/255, where *any* smooth ramp leaves ≈ step/2) keep merging — the flat
/// fallback leaves more, because the global palette's centroids drift along
/// ramps — while a spatially mixed group (a bright facet merged with darker
/// shades sharing the same projection range) leaves ≈ half the mix gap on
/// every pixel, far above its flat baseline, and is rejected.
const FLAT_PARITY: f64 = 1.1;

/// Per-pixel RMS of the sample against the flat-band representation: each
/// pixel compared to the mean colour of its dominant rank within the sample
/// (≈ the palette centroid that region would render with). Normalised to
/// 0..1, matching the fit residuals.
fn flat_baseline_rmse(uniques: &Uniques, px: &[u32], lab: &[u8]) -> f64 {
    let mut sum = [[0.0_f64; 4]; 256];
    for &i in px {
        let i = i as usize;
        let l = lab[i] as usize;
        if l >= 256 {
            continue;
        }
        for ch in 0..3 {
            sum[l][ch] += f64::from(uniques.colors[uniques.id[i] as usize][ch]) / 255.0;
        }
        sum[l][3] += 1.0;
    }
    let mut acc = 0.0_f64;
    let mut n = 0.0_f64;
    for &i in px {
        let i = i as usize;
        let l = lab[i] as usize;
        if l >= 256 || sum[l][3] <= 0.0 {
            continue;
        }
        let c = uniques.colors[uniques.id[i] as usize];
        for ch in 0..3 {
            let d = f64::from(c[ch]) / 255.0 - sum[l][ch] / sum[l][3];
            acc += d * d;
        }
        n += 1.0;
    }
    (acc / (n * 3.0).max(1.0)).sqrt()
}

/// Adaptive acceptance bound for a fitted gradient model: meet the absolute
/// smoothness floor outright, or beat the flat-band baseline by parity.
fn gradient_bound(flat_pp: f64) -> f64 {
    DELTA_ABS.max(flat_pp * FLAT_PARITY)
}

/// Maximum core-range overlap fraction between consecutive ranks of a
/// gradient group along the fitted axis.
///
/// A genuine over-segmented ramp is a chain of *ordered* bands: each rank
/// occupies a distinct segment of the projection axis (interquartile ranges
/// barely overlap — only anti-aliasing straddles the boundaries). Texture
/// shades of a painterly facet are the opposite: every rank's pixels spread
/// across the whole projection range, so consecutive core ranges overlap
/// almost completely. Ordering is what separates the two regimes — and only
/// the first kind renders correctly as one gradient; the second renders its
/// mixed bin means across whole sub-regions (the measured chest failure:
/// ≈ 47 000 px ΔRGB 23–31 off the source).
const RANK_OVERLAP_MAX: f64 = 0.6;

/// Core (interquartile) overlap of two populations along a projection axis
/// as a fraction of the shorter core range. Both slices are sorted in place.
fn core_overlap(a: &mut [f64], b: &mut [f64]) -> f64 {
    let quart = |v: &mut [f64]| -> (f64, f64) {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        (v[n / 4], v[(3 * n) / 4])
    };
    let (a_lo, a_hi) = quart(a);
    let (b_lo, b_hi) = quart(b);
    let overlap = (a_hi.min(b_hi) - a_lo.max(b_lo)).max(0.0);
    let minlen = (a_hi - a_lo).min(b_hi - b_lo).max(1e-9);
    overlap / minlen
}

/// Returns `true` when, along `proj_of`, no pair of the given per-rank
/// projection samples has core overlap above [`RANK_OVERLAP_MAX`] with its
/// median-neighbour. `projs` maps a rank id to its sample projections.
fn ranks_ordered_along(proj_of: impl Fn(f64, f64) -> f64, uniques: &Uniques, px: &[u32], lab: &[u8]) -> bool {
    let mut per_rank: std::collections::BTreeMap<u8, Vec<f64>> = std::collections::BTreeMap::new();
    for &i in px {
        let i = i as usize;
        let t = proj_of((i % uniques.width) as f64, (i / uniques.width) as f64);
        per_rank.entry(lab[i]).or_default().push(t);
    }
    // Ranks with too few samples carry no range information; skip them.
    let mut ranges: Vec<(f64, Vec<f64>)> = per_rank
        .into_iter()
        .filter(|(_, v)| v.len() >= 8)
        .map(|(_, mut v)| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            (v[v.len() / 2], v)
        })
        .collect();
    ranges.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    for pair in ranges.windows(2) {
        let mut a = pair[0].1.clone();
        let mut b = pair[1].1.clone();
        if core_overlap(&mut a, &mut b) > RANK_OVERLAP_MAX {
            return false;
        }
    }
    true
}

/// Per-pixel smoothness floor: a region may only become a gradient when the
/// rendered gradient predicts every sampled pixel to within this RMS
/// (roadmap M3 spec: "RMS residual < δ ≈ 2/255").
const DELTA_ABS: f64 = 2.0 / 255.0;

/// Residual of the *rendered* gradient over a sample: every pixel compared
/// against the mean colour of its projection bin, where the projection is
/// the gradient's own coordinate (axis position for linear, radius for
/// radial). The emitted SVG gradient interpolates exactly this bin profile
/// (`linear_def`/`radial_def` build their stops from the same bins), so this
/// residual bounds the rendering error pixel-for-pixel.
///
/// Two properties make it the right gate. Against the global affine residual
/// it adds freedom: a ramp whose colour profile is not a straight line in
/// space (curved falloff, vignette) still fits — its bins are pure and each
/// pixel sits on its bin mean. Against binned *means* it adds honesty: a
/// spatially mixed union (a bright facet merged with darker shades sharing
/// the same projection range) puts both populations into the same bins, so
/// every pixel sits ≈ half the mix gap off the bin mean and the merge is
/// rejected — the failure mode that painted a ≈ 47 000 px chest area
/// ΔRGB 23–31 too dark on the mascot corpus.
///
/// Samples below `RESIDUAL_BINS * 4` cannot bin meaningfully; the caller's
/// per-pixel affine residual is used instead.
fn bin_profile_residual(
    uniques: &Uniques,
    px: &[u32],
    proj_of: impl Fn(f64, f64) -> f64,
    fallback: f64,
) -> f64 {
    let mut s = samples_of(uniques, px, uniques.width);
    if s.len() < RESIDUAL_BINS * 4 {
        return fallback;
    }
    let mut proj: Vec<(f64, [f64; 3])> = s
        .iter_mut()
        .map(|p| (proj_of(p.x, p.y), [f64::from(p.c[0]), f64::from(p.c[1]), f64::from(p.c[2])]))
        .collect();
    proj.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let per = proj.len() / RESIDUAL_BINS;
    let mut acc = 0.0_f64;
    let mut n = 0.0_f64;
    for bi in 0..RESIDUAL_BINS {
        let seg = &mut proj[bi * per..(bi + 1) * per];
        let mut mean = [0.0_f64; 3];
        for (_, c) in seg.iter() {
            for ch in 0..3 {
                mean[ch] += c[ch];
            }
        }
        for m in &mut mean {
            *m /= per as f64;
        }
        for (_, c) in seg.iter() {
            for ch in 0..3 {
                let d = mean[ch] - c[ch];
                acc += d * d;
            }
        }
        n += per as f64;
    }
    (acc / (n * 3.0).max(1.0)).sqrt()
}

/// Residual of the best colour model for a candidate *pair merge*: affine,
/// or — when enabled — radial, whichever is lower, evaluated on
/// projection-binned mean colours.
///
/// The merge cascade is deliberately permissive: over-segmented shade chains
/// carry per-pixel staircase floors of several 1/255 per band step (each
/// band is a real flat step in the source), so a per-pixel bound here would
/// split every shaded facet into flat bands — measurably *more* interior
/// boundaries and *worse* edge error on the mascot corpus. Bin means align
/// with the model for any ordered band chain, which is exactly what a merge
/// candidate must be. Mixed-population groups (the failure this permissiveness
/// admits) are rejected at the final gate by [`bin_profile_residual`], where
/// the emitted gradient is judged against the pixels it renders.
fn union_residual(uniques: &Uniques, px: &[u32], radial_enabled: bool) -> f64 {
    let affine = fit_linear_sample(uniques, px)
        .map_or(f64::MAX, |f| binned_linear_residual(&f, uniques, px));
    if !radial_enabled {
        return affine;
    }
    affine.min(
        fit_radial(uniques, px)
            .map_or(f64::MAX, |r| binned_radial_residual(&r, uniques, px)),
    )
}

/// Residual of the affine model on projection-binned mean colours (merge
/// cascade gate; see [`union_residual`]).
fn binned_linear_residual(fit: &LinearFit, uniques: &Uniques, px: &[u32]) -> f64 {
    let w = uniques.width;
    let s = samples_of(uniques, px, w);
    if s.len() < RESIDUAL_BINS * 4 {
        return fit.rmse;
    }
    let axis = linear_proj(fit);
    let mut proj: Vec<(f64, &Sample)> =
        s.iter().map(|p| (axis(p.x, p.y), p)).collect();
    proj.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let per = proj.len() / RESIDUAL_BINS;
    let mut acc = 0.0_f64;
    let mut n = 0.0_f64;
    for bi in 0..RESIDUAL_BINS {
        let seg = &proj[bi * per..(bi + 1) * per];
        let mx = seg.iter().map(|p| p.1.x).sum::<f64>() / per as f64;
        let my = seg.iter().map(|p| p.1.y).sum::<f64>() / per as f64;
        for ch in 0..3 {
            let mc = seg.iter().map(|p| f64::from(p.1.c[ch])).sum::<f64>() / per as f64;
            let pred = fit.c0[ch] + fit.gx[ch] * mx + fit.gy[ch] * my;
            let d = pred - mc;
            acc += d * d;
            n += 1.0;
        }
    }
    (acc / n.max(1.0)).sqrt()
}

/// Residual of the radial model on radius-binned mean colours (merge
/// cascade gate; see [`union_residual`]).
fn binned_radial_residual(rf: &RadialFit, uniques: &Uniques, px: &[u32]) -> f64 {
    let w = uniques.width;
    let s = samples_of(uniques, px, w);
    if s.len() < RESIDUAL_BINS * 4 {
        return rf.rmse;
    }
    let radius = radial_proj(rf);
    let mut proj: Vec<(f64, &Sample)> =
        s.iter().map(|p| (radius(p.x, p.y), p)).collect();
    proj.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let per = proj.len() / RESIDUAL_BINS;
    let mut acc = 0.0_f64;
    let mut n = 0.0_f64;
    for bi in 0..RESIDUAL_BINS {
        let seg = &proj[bi * per..(bi + 1) * per];
        let mr = seg.iter().map(|p| p.0).sum::<f64>() / per as f64;
        for ch in 0..3 {
            let mc = seg.iter().map(|p| f64::from(p.1.c[ch])).sum::<f64>() / per as f64;
            let pred = rf.c0[ch] + rf.g[ch] * mr;
            let d = pred - mc;
            acc += d * d;
            n += 1.0;
        }
    }
    (acc / n.max(1.0)).sqrt()
}


/// Gradient coordinate of the affine model: projection on the dominant axis.
fn linear_proj(f: &LinearFit) -> impl Fn(f64, f64) -> f64 {
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let mut c = 0.0_f64;
    for ch in 0..3 {
        a += f.gx[ch] * f.gx[ch];
        b += f.gx[ch] * f.gy[ch];
        c += f.gy[ch] * f.gy[ch];
    }
    let axis = dominant_eigen2(a, b, c);
    move |x, y| axis[0] * x + axis[1] * y
}

/// Gradient coordinate of the radial model: distance from the fitted centre.
fn radial_proj(r: &RadialFit) -> impl Fn(f64, f64) -> f64 {
    let (cx, cy) = (r.cx, r.cy);
    move |x, y| (x - cx).hypot(y - cy)
}

const fn coords(i: u32, w: usize) -> (f64, f64) {
    let i = i as usize;
    ((i % w) as f64, (i / w) as f64)
}

fn colour_of(uniques: &Uniques, i: u32) -> [f32; 3] {
    let c = uniques.colors[uniques.id[i as usize] as usize];
    [
        f32::from(c[0]) / 255.0,
        f32::from(c[1]) / 255.0,
        f32::from(c[2]) / 255.0,
    ]
}

fn samples_of(uniques: &Uniques, px: &[u32], w: usize) -> Vec<Sample> {
    sample_cap(px, 4096)
        .into_iter()
        .map(|i| {
            let (x, y) = coords(i, w);
            Sample {
                x,
                y,
                c: colour_of(uniques, i),
            }
        })
        .collect()
}

/// Least-squares affine colour fit over a capped sample of pixels.
fn fit_linear_sample(uniques: &Uniques, px: &[u32]) -> Option<LinearFit> {
    if px.len() < 12 {
        return None;
    }
    let s = samples_of(uniques, px, uniques.width);
    if s.len() < 12 {
        return None;
    }
    // Shared geometric moments.
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut n = 0.0_f64;
    let mut scx = [0.0_f64; 3];
    let mut scy = [0.0_f64; 3];
    let mut scc = [0.0_f64; 3];
    for p in &s {
        sxx += p.x * p.x;
        sxy += p.x * p.y;
        syy += p.y * p.y;
        sx += p.x;
        sy += p.y;
        n += 1.0;
        for ch in 0..3 {
            let v = f64::from(p.c[ch]);
            scx[ch] += v * p.x;
            scy[ch] += v * p.y;
            scc[ch] += v;
        }
    }
    let m = [sxx, sxy, sx, sxy, syy, sy, sx, sy, n];
    let mut c0 = [0.0_f64; 3];
    let mut gx = [0.0_f64; 3];
    let mut gy = [0.0_f64; 3];
    for ch in 0..3 {
        let (a, b, c) = solve3(&m, &[scx[ch], scy[ch], scc[ch]]);
        gx[ch] = a;
        gy[ch] = b;
        c0[ch] = c;
    }
    let mut acc = 0.0_f64;
    for p in &s {
        for ch in 0..3 {
            let pred = c0[ch] + gx[ch] * p.x + gy[ch] * p.y;
            let d = pred - f64::from(p.c[ch]);
            acc += d * d;
        }
    }
    Some(LinearFit {
        c0,
        gx,
        gy,
        rmse: (acc / (s.len() as f64 * 3.0)).sqrt(),
    })
}

/// Solves the 3×3 system `m·x = rhs` by Cramer's rule (m row-major).
fn solve3(m: &[f64; 9], rhs: &[f64; 3]) -> (f64, f64, f64) {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return (0.0, 0.0, rhs[2]);
    }
    let dx = (rhs[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (rhs[1] * m[8] - m[5] * rhs[2])
        + m[2] * (rhs[1] * m[7] - m[4] * rhs[2]))
        / det;
    let dy = (m[0] * (rhs[1] * m[8] - m[5] * rhs[2]) - rhs[0] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * rhs[2] - rhs[1] * m[6]))
        / det;
    let dz = (m[0] * (m[4] * rhs[2] - rhs[1] * m[7]) - m[1] * (m[3] * rhs[2] - rhs[1] * m[6])
        + rhs[0] * (m[3] * m[7] - m[4] * m[6]))
        / det;
    (dx, dy, dz)
}

/// Top eigenvector of the 2×2 symmetric matrix [[a, b], [b, c]], unit length.
fn dominant_eigen2(a: f64, b: f64, c: f64) -> [f64; 2] {
    let tr = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let lambda = f64::midpoint(tr, disc);
    let v1 = [b, lambda - a];
    let v2 = [lambda - c, b];
    let v = if v1[0] * v1[0] + v1[1] * v1[1] >= v2[0] * v2[0] + v2[1] * v2[1] {
        v1
    } else {
        v2
    };
    let n = v[0].hypot(v[1]);
    if n < 1e-12 {
        // Degenerate (diagonal) matrix: axes are the coordinate axes.
        return if a >= c { [1.0, 0.0] } else { [0.0, 1.0] };
    }
    [v[0] / n, v[1] / n]
}

/// RDP simplification of a colour sequence (3-D points) with a max colour
/// deviation tolerance; keeps at least the two endpoints and at most 6 stops.
fn rdp_stops(pts: &[[f64; 3]], eps: f64) -> Vec<usize> {
    let n = pts.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0_usize, n - 1_usize)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 || keep.iter().filter(|&&k| k).count() >= 6 {
            continue;
        }
        let (pa, pb) = (pts[a], pts[b]);
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let abn = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]).sqrt();
        let (mut md, mut mi) = (0.0_f64, a);
        for (k, p) in pts[a + 1..b].iter().enumerate() {
            let i = a + 1 + k;
            let ap = [p[0] - pa[0], p[1] - pa[1], p[2] - pa[2]];
            let cross = [
                ap[1] * ab[2] - ap[2] * ab[1],
                ap[2] * ab[0] - ap[0] * ab[2],
                ap[0] * ab[1] - ap[1] * ab[0],
            ];
            let d = if abn < 1e-9 {
                (ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]).sqrt()
            } else {
                (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt() / abn
            };
            if d > md {
                md = d;
                mi = i;
            }
        }
        if md > eps {
            keep[mi] = true;
            stack.push((a, mi));
            stack.push((mi, b));
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// Bins `(offset, colour)` pairs into `nbins` mean-colour samples.
fn bin_ramp(offs: &[f64], cs: &[[f32; 3]], nbins: usize) -> (Vec<f64>, Vec<[f64; 3]>) {
    let lo = offs.iter().copied().fold(f64::MAX, f64::min);
    let hi = offs.iter().copied().fold(f64::MIN, f64::max);
    let span = (hi - lo).max(1e-6);
    let mut bins = vec![0.0_f64; nbins * 3];
    let mut counts = vec![0.0_f64; nbins];
    for (t, c) in offs.iter().zip(cs) {
        let b = (((t - lo) / span) * ((nbins - 1) as f64))
            .floor()
            .clamp(0.0, (nbins - 1) as f64) as usize;
        counts[b] += 1.0;
        for ch in 0..3 {
            bins[b * 3 + ch] += f64::from(c[ch]);
        }
    }
    let mut pts = Vec::new();
    let mut out_offs = Vec::new();
    for b in 0..nbins {
        if counts[b] > 0.0 {
            out_offs.push(b as f64 / (nbins - 1) as f64);
            pts.push([
                bins[b * 3] / counts[b],
                bins[b * 3 + 1] / counts[b],
                bins[b * 3 + 2] / counts[b],
            ]);
        }
    }
    (out_offs, pts)
}

fn stops_from(offs: &[f64], pts: &[[f64; 3]]) -> Vec<(f32, [u8; 3])> {
    let kept = rdp_stops(pts, 2.0 / 255.0);
    kept.iter()
        .map(|&i| {
            let c = pts[i];
            (
                offs[i] as f32,
                [
                    (c[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (c[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                    (c[2] * 255.0).round().clamp(0.0, 255.0) as u8,
                ],
            )
        })
        .collect()
}

/// Converts a linear fit + region pixels into an SVG gradient definition:
/// axis = dominant singular direction of the 3×2 gradient matrix, stops =
/// binned projected colours reduced by RDP.
fn linear_def(fit: &LinearFit, uniques: &Uniques, px: &[u32]) -> GradientDef {
    let w = uniques.width;
    // MᵀM for the 3×2 matrix of (gx, gy) rows.
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let mut c = 0.0_f64;
    for ch in 0..3 {
        a += fit.gx[ch] * fit.gx[ch];
        b += fit.gx[ch] * fit.gy[ch];
        c += fit.gy[ch] * fit.gy[ch];
    }
    let axis = dominant_eigen2(a, b, c);
    // Project sample pixels onto the axis.
    let mut ts: Vec<f64> = Vec::new();
    let mut cs: Vec<[f32; 3]> = Vec::new();
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut n = 0.0_f64;
    for s in samples_of(uniques, px, w) {
        ts.push(axis[0] * s.x + axis[1] * s.y);
        cs.push(s.c);
        cx += s.x;
        cy += s.y;
        n += 1.0;
    }
    cx /= n.max(1.0);
    cy /= n.max(1.0);
    let tmin = ts.iter().copied().fold(f64::MAX, f64::min);
    let tmax = ts.iter().copied().fold(f64::MIN, f64::max);
    let (offs, pts) = bin_ramp(&ts, &cs, 24);
    let stops = stops_from(&offs, &pts);
    // Gradient line endpoints: the projections t are absolute
    // (`t = axis·p`), so a point with projection t on the line through the
    // centroid is `centroid + axis·(t − axis·centroid)` — the earlier
    // `centre ± half-span` form silently shifted clipped ramps' stops.
    let base = axis[0] * cx + axis[1] * cy;
    let p0 = [cx + axis[0] * (tmin - base), cy + axis[1] * (tmin - base)];
    let p1 = [cx + axis[0] * (tmax - base), cy + axis[1] * (tmax - base)];
    GradientDef {
        kind: GradientKind::Linear {
            x1: p0[0],
            y1: p0[1],
            x2: p1[0],
            y2: p1[1],
        },
        stops,
    }
}

/// Per-pixel RMS residual of a fitted radial model over a sample. Used to
/// compare candidate centres on the pixels they will actually render (the
/// binned-mean comparison this replaces could prefer a centre whose bin
/// means fit while its pixels do not).
fn radial_pixel_rmse(rf: &RadialFit, s: &[Sample]) -> f64 {
    if s.is_empty() {
        return f64::MAX;
    }
    let mut acc = 0.0_f64;
    for p in s {
        let r = (p.x - rf.cx).hypot(p.y - rf.cy);
        for ch in 0..3 {
            let res = rf.c0[ch] + rf.g[ch] * r - f64::from(p.c[ch]);
            acc += res * res;
        }
    }
    (acc / (s.len() as f64 * 3.0)).sqrt()
}

/// Gauss–Newton fit of a radial ramp: colour ≈ c0 + g·r around a fitted
/// centre. Returns the centre, radius span and residual.
fn fit_radial(uniques: &Uniques, px: &[u32]) -> Option<RadialFit> {    let w = uniques.width;
    let s = samples_of(uniques, px, w);
    if s.len() < 12 {
        return None;
    }
    let mut cx = s.iter().map(|p| p.x).sum::<f64>() / s.len() as f64;
    let mut cy = s.iter().map(|p| p.y).sum::<f64>() / s.len() as f64;
    let mut best = f64::MAX;
    let mut best_c = (cx, cy);
    let mut best_span = 0.0_f64;
    let mut best_cg = ([0.0_f64; 3], [0.0_f64; 3]);
    for _ in 0..6 {
        let radii: Vec<f64> = s.iter().map(|p| (p.x - cx).hypot(p.y - cy)).collect();
        let rmean = radii.iter().sum::<f64>() / radii.len() as f64;
        let srr: f64 = radii.iter().map(|r| (r - rmean) * (r - rmean)).sum();
        let var_r = srr / radii.len() as f64;
        if var_r < 1e-9 {
            break;
        }
        let mut c0 = [0.0_f64; 3];
        let mut g = [0.0_f64; 3];
        for ch in 0..3 {
            let cr = s.iter().map(|p| f64::from(p.c[ch])).sum::<f64>() / s.len() as f64;
            let crl = s
                .iter()
                .zip(&radii)
                .map(|(p, r)| f64::from(p.c[ch]) * (r - rmean))
                .sum::<f64>();
            g[ch] = crl / srr;
            c0[ch] = cr - g[ch] * rmean;
        }
        let mut jtj = [0.0_f64; 4];
        let mut jtr = [0.0_f64; 2];
        let mut acc = 0.0_f64;
        for (k, r) in radii.iter().enumerate() {
            let (dx, dy) = (s[k].x - cx, s[k].y - cy);
            let scale = if *r < 1e-6 { 0.0 } else { 1.0 / r };
            for ch in 0..3 {
                let res = c0[ch] + g[ch] * r - f64::from(s[k].c[ch]);
                acc += res * res;
                let jx = g[ch] * dx * scale;
                let jy = g[ch] * dy * scale;
                jtj[0] += jx * jx;
                jtj[1] += jx * jy;
                jtj[3] += jy * jy;
                jtr[0] += jx * res;
                jtr[1] += jy * res;
            }
        }
        jtj[2] = jtj[1];
        best = (acc / (s.len() as f64 * 3.0)).sqrt();
        best_cg = (c0, g);
        let det = jtj[0] * jtj[3] - jtj[1] * jtj[2];
        if det.abs() < 1e-12 {
            break;
        }
        let dcx = (jtj[3] * jtr[0] - jtj[1] * jtr[1]) / det;
        let dcy = (jtj[0] * jtr[1] - jtj[2] * jtr[0]) / det;
        cx -= dcx.clamp(-5.0, 5.0);
        cy -= dcy.clamp(-5.0, 5.0);
        best_c = (cx, cy);
        best_span = radii.iter().copied().fold(0.0_f64, f64::max);
    }
    if !best.is_finite() || best >= f64::MAX {
        return None;
    }
    // The Gauss–Newton walk drifts on a quantised staircase (each ring is a
    // flat colour step, so the centre is weakly determined); a few px of
    // centre error inflates the residual the merge gates use. Evaluate the
    // plain centroid as well — for (near-)complete rings the centroid *is*
    // the centre — and keep whichever explains the sampled pixels better.
    let (mut best_c, mut best_cg) = (best_c, best_cg);
    let (ccx, ccy) = (
        s.iter().map(|p| p.x).sum::<f64>() / s.len() as f64,
        s.iter().map(|p| p.y).sum::<f64>() / s.len() as f64,
    );
    if (ccx - best_c.0).abs() > 0.5 || (ccy - best_c.1).abs() > 0.5 {
        let radii: Vec<f64> = s.iter().map(|p| (p.x - ccx).hypot(p.y - ccy)).collect();
        let rmean = radii.iter().sum::<f64>() / radii.len() as f64;
        let srr: f64 = radii.iter().map(|r| (r - rmean) * (r - rmean)).sum();
        if srr > 1e-9 {
            let mut c0 = [0.0_f64; 3];
            let mut g = [0.0_f64; 3];
            for ch in 0..3 {
                let cr = s.iter().map(|p| f64::from(p.c[ch])).sum::<f64>() / s.len() as f64;
                let crl = s
                    .iter()
                    .zip(&radii)
                    .map(|(p, r)| f64::from(p.c[ch]) * (r - rmean))
                    .sum::<f64>();
                g[ch] = crl / srr;
                c0[ch] = cr - g[ch] * rmean;
            }
            let cand = RadialFit {
                cx: ccx,
                cy: ccy,
                r: radii.iter().copied().fold(0.0_f64, f64::max),
                c0,
                g,
                rmse: best,
            };
            if radial_pixel_rmse(&cand, &s) < radial_pixel_rmse(&RadialFit {
                cx: best_c.0,
                cy: best_c.1,
                r: best_span,
                c0: best_cg.0,
                g: best_cg.1,
                rmse: best,
            }, &s)
            {
                best_c = (ccx, ccy);
                best_cg = (c0, g);
                best_span = cand.r;
            }
        }
    }
    Some(RadialFit {
        cx: best_c.0,
        cy: best_c.1,
        r: best_span,
        c0: best_cg.0,
        g: best_cg.1,
        rmse: best,
    })
}

/// Builds the radial gradient definition: stops binned by radius.
///
/// The bin range is the *linear extent* of the ramp, not the pixel extent:
/// SVG radial ramps are clamped beyond the gradient radius, so the outer
/// pixels of a region (e.g. a rounded-rect's corners beyond the 50 %-bbox
/// radius) hold a constant plateau colour. Normalising over the plateau
/// compressed the true ramp into the first 70 % of the gradient (measured:
/// mid-range colours 20/255 too dark). Bins whose mean colour matches the
/// outermost bin within `3/255` are treated as plateau; the gradient radius
/// ends where the plateau begins and the plateau colour becomes the last
/// stop at offset 1.
fn radial_def(rf: &RadialFit, uniques: &Uniques, px: &[u32]) -> GradientDef {
    let w = uniques.width;
    let s = samples_of(uniques, px, w);
    let radius = |p: &Sample| (p.x - rf.cx).hypot(p.y - rf.cy);
    let nb = 24_usize;
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for p in &s {
        let r = radius(p);
        lo = lo.min(r);
        hi = hi.max(r);
    }
    let span = (hi - lo).max(1e-6);
    let mut cnt = vec![0.0_f64; nb];
    let mut sr = vec![0.0_f64; nb];
    let mut sc = vec![[0.0_f64; 3]; nb];
    for p in &s {
        let r = radius(p);
        let b = (((r - lo) / span) * (nb as f64 - 1.0))
            .round()
            .clamp(0.0, (nb - 1) as f64) as usize;
        cnt[b] += 1.0;
        sr[b] += r;
        for (dst, &src) in sc[b].iter_mut().zip(&p.c) {
            *dst += f64::from(src);
        }
    }
    let mut bin_r: Vec<f64> = Vec::new();
    let mut bin_c: Vec<[f64; 3]> = Vec::new();
    for b in 0..nb {
        if cnt[b] > 0.0 {
            bin_r.push(sr[b] / cnt[b]);
            bin_c.push([sc[b][0] / cnt[b], sc[b][1] / cnt[b], sc[b][2] / cnt[b]]);
        }
    }
    if bin_r.len() < 2 {
        return GradientDef {
            kind: GradientKind::Radial {
                cx: rf.cx,
                cy: rf.cy,
                r: rf.r.max(1.0),
            },
            stops: Vec::new(),
        };
    }
    // Outer plateau walk: bins matching the outermost colour are the clamp.
    let plateau_tol = 3.0 / 255.0;
    let c_outer = *bin_c.last().expect("non-empty");
    let mut last = bin_c.len() - 1;
    while last > 0 {
        let d = (0..3)
            .map(|ch| (bin_c[last - 1][ch] - c_outer[ch]).abs())
            .fold(0.0_f64, f64::max);
        if d >= plateau_tol {
            break;
        }
        last -= 1;
    }
    let has_plateau = last + 1 < bin_c.len();
    // Gradient radius: the outer edge of the last *changing* bin.
    let r_lin = if has_plateau && last + 1 < bin_r.len() {
        bin_r[last + 1]
    } else {
        hi
    };
    let r_lin = r_lin.max(1.0);
    // Stop points over [0, r_lin]: offset = radius / r_lin.
    let mut pts: Vec<[f64; 3]> = Vec::new();
    let mut offs: Vec<f64> = Vec::new();
    for k in 0..=last {
        offs.push((bin_r[k] / r_lin).clamp(0.0, 1.0));
        pts.push(bin_c[k]);
    }
    if has_plateau {
        // The plateau colour closes the ramp at offset 1.
        offs.push(1.0);
        pts.push(c_outer);
    }
    let stops = stops_from(&offs, &pts);
    GradientDef {
        kind: GradientKind::Radial {
            cx: rf.cx,
            cy: rf.cy,
            r: r_lin,
        },
        stops,
    }
}

#[cfg(test)]
mod tests {
    use crate::color::model::build_model;
    use crate::color::palette::{build_palette, unique_colors};
    use crate::settings::{ColorSettings, Colors, Gradients};

    use super::*;

    fn ramp_settings() -> ColorSettings {
        // Same K as the real config (`Auto` → 32, raised to ≥ 48 in
        // over-segmented mode); `build_palette`'s uniques/3 guard keeps the
        // band geometry realistic on these mini fixtures.
        ColorSettings {
            gradients: Gradients::LinearRadial,
            max_colors: Colors::Max(48),
            ..ColorSettings::default()
        }
    }

    fn run(px: &[u8], w: usize, h: usize) -> Regions {
        let u = unique_colors(px, w);
        let pal = build_palette(&u, w, h, &ramp_settings(), true);
        let mix = build_model(&u, &pal);
        detect_regions(&u, px, w, h, &pal, &mix, &ramp_settings())
    }

    /// Horizontal ramp red→orange across the left half; flat white right
    /// half. The ramp must become ONE gradient region; white stays flat.
    #[test]
    fn linear_ramp_is_detected_as_one_gradient_region() {
        let (w, h) = (64usize, 32usize);
        let mut px = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                if x < w / 2 {
                    // Continuous ramp quantised to 8 bits by the pixel
                    // format: the natural regime of the real fixtures (each
                    // band spans several shades and keeps 3×3 interiors).
                    let t = x as f32 / ((w / 2 - 1) as f32);
                    px[i] = (255.0 - t * 140.0) as u8;
                    px[i + 1] = (80.0 + t * 100.0) as u8;
                    px[i + 2] = 40;
                }
                px[i + 3] = 255;
            }
        }
        let regions = run(&px, w, h);
        assert_eq!(regions.gradient_defs.len(), 1, "one gradient region");
        let grad = regions
            .kinds
            .iter()
            .position(|k| matches!(k, RegionKind::Gradient(_)))
            .expect("gradient region present");
        assert!(regions.pixel_count[grad] > 0);
        let def = &regions.gradient_defs[0];
        assert!(matches!(def.kind, GradientKind::Linear { .. }));
        assert!(def.stops.len() >= 2 && def.stops.len() <= 6);
        // Stops run bright red → orange.
        let first = def.stops[0].1;
        let last = def.stops[def.stops.len() - 1].1;
        assert!(first[0] > last[0] - 30, "red decreases: {first:?} {last:?}");
    }

    /// Two flat colours sharing one edge: must NOT create a gradient region.
    #[test]
    fn flat_boundary_never_becomes_a_gradient() {
        let (w, h) = (32usize, 32usize);
        let mut px = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                if x < w / 2 {
                    px[i] = 200;
                    px[i + 1] = 30;
                    px[i + 2] = 30;
                } else {
                    px[i] = 30;
                    px[i + 1] = 30;
                    px[i + 2] = 200;
                }
                px[i + 3] = 255;
            }
        }
        let regions = run(&px, w, h);
        assert!(regions.gradient_defs.is_empty(), "no gradient regions");
        assert!(regions.kinds.iter().all(|k| *k == RegionKind::Flat));
    }

    /// Concentric colour rings: radial fit must win when enabled.
    #[test]
    fn radial_ramp_prefers_radial_fit() {
        // A quantised radial ramp (the natural 8-bit regime of a resvg
        // render): rings of ~8 px spanning ~8 shades each. The unit under
        // test is the *fit*, not the full merge pipeline: on ring data the
        // radial model must explain the pixels strictly better than any
        // affine model (whose level sets are lines, not circles), and its
        // binned residual must sit far below the merge threshold.
        let (w, h) = (192usize, 192usize);
        let mut px = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let r = (((x as f64 - 95.5).hypot(y as f64 - 95.5) / 90.0).clamp(0.0, 1.0) * 32.0)
                    .floor()
                    / 32.0;
                px[i] = (30.0 + r * 200.0) as u8;
                px[i + 1] = (120.0 + r * 40.0) as u8;
                px[i + 2] = (200.0 - r * 160.0) as u8;
                px[i + 3] = 255;
            }
        }
        let uniques = unique_colors(&px, w);
        // Sample the disc's pixels through the same capped sampler the
        // merge gates use.
        let disc: Vec<u32> = (0..w * h)
            .filter(|&p| {
                let dx = (p % w) as f64 - 95.5;
                let dy = (p / w) as f64 - 95.5;
                dx.hypot(dy) < 90.0
            })
            .map(|p| p as u32)
            .collect();
        let px_sample = sample_cap(&disc, 4096);
        let affine = fit_linear_sample(&uniques, &px_sample).expect("affine fit");
        let radial = fit_radial(&uniques, &px_sample).expect("radial fit");
        assert!(
            radial.rmse < affine.rmse * 0.5,
            "radial fit must dominate affine on ring data: radial={} affine={}",
            radial.rmse,
            affine.rmse
        );
        // The fitted centre must land near the true one (95.5, 95.5). The
        // Gauss–Newton walk is clamped to 5 px per step over 6 iterations,
        // so a few px of drift on a quantised staircase is expected; what
        // matters for the merge gates is the binned residual below.
        assert!(
            (radial.cx - 95.5).abs() < 8.0 && (radial.cy - 95.5).abs() < 8.0,
            "radial centre recovered: ({}, {})",
            radial.cx,
            radial.cy
        );
        // And the per-pixel residual must be far below the merge threshold:
        // the 32-step quantisation puts ≈ step/√12 ≈ 1.8/255 on every pixel,
        // which the flat-band parity floor absorbs (see [`gradient_bound`]).
        assert!(
            radial.rmse < 2.0 / 255.0,
            "per-pixel radial residual under delta: {:.5}",
            radial.rmse
        );
    }
    #[test]
    fn rdp_stops_keeps_endpoints_and_bounds_count() {
        // Linear ramp in colour space: RDP keeps only the endpoints.
        let pts: Vec<[f64; 3]> = (0..24)
            .map(|i| {
                let t = f64::from(i) / 23.0;
                [t, t * 0.5, 1.0 - t]
            })
            .collect();
        let kept = rdp_stops(&pts, 2.0 / 255.0);
        assert_eq!(kept.first(), Some(&0));
        assert_eq!(kept.last(), Some(&23));
        // Wiggly ramp needs more stops but never more than 6.
        let pts2: Vec<[f64; 3]> = (0..24)
            .map(|i| {
                let t = f64::from(i) / 23.0;
                [t, (t * 12.0).sin().abs(), 1.0 - t]
            })
            .collect();
        let kept2 = rdp_stops(&pts2, 2.0 / 255.0);
        assert!(kept2.len() <= 6, "got {}", kept2.len());
        assert!(kept2.len() >= 2);
    }

    #[test]
    fn dominant_eigen2_axis_matches_gradient_direction() {
        // gx=(1,0,0), gy=(0,0,0) → axis = x.
        let axis = dominant_eigen2(1.0, 0.0, 0.0);
        assert!((axis[0].abs() - 1.0).abs() < 1e-9);
        // Equal eigenvalues → any axis is fine; must stay unit length.
        let axis2 = dominant_eigen2(1.0, 0.0, 1.0);
        assert!(((axis2[0] * axis2[0] + axis2[1] * axis2[1]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn solve3_recovers_known_system() {
        let (x, y, z) = solve3(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[1.0, 2.0, 3.0],
        );
        assert!((x - 1.0).abs() < 1e-9 && (y - 2.0).abs() < 1e-9 && (z - 3.0).abs() < 1e-9);
        // c(x,y) = 0.2 + 0.5·x + 0.25·y over the 4×4 grid x,y ∈ 0..4:
        // Σx² = 56, Σxy = 36, Σx = 24, Σy² = 56, Σy = 24, n = 16;
        // rhs = [Σc·x, Σc·y, Σc] = [41.8, 36.8, 21.2].
        let m = [56.0, 36.0, 24.0, 36.0, 56.0, 24.0, 24.0, 24.0, 16.0];
        let (gx, gy, c0) = solve3(&m, &[41.8, 36.8, 21.2]);
        assert!((gx - 0.5).abs() < 1e-6, "gx={gx}");
        assert!((gy - 0.25).abs() < 1e-6, "gy={gy}");
        assert!((c0 - 0.2).abs() < 1e-6, "c0={c0}");
    }

    #[test]
    fn paint_ranks_order_by_pixel_count() {
        let regions = flat_regions(3, &[10, 500, 100]);
        assert_eq!(regions.paint_ranks(), vec![2, 0, 1]);
    }

    /// The final group fit must sample the region's real population: a
    /// 90/10 rank split sampled per-rank-balanced would tilt the fit toward
    /// the minority (the measured chest-darkening bug); the population
    /// sample keeps the 90/10 shares.
    #[test]
    fn population_sample_keeps_population_shares() {
        let big: Vec<u32> = (0..9000).collect();
        let small: Vec<u32> = (9000..10_000).collect();
        let rank_pixels = vec![big, small];
        let s = population_sample(&rank_pixels, &[0, 1], 1000);
        let from_big = s.iter().filter(|&&p| p < 9000).count();
        assert_eq!(s.len(), 1000);
        let share = f64::from(from_big as u32) / 1000.0;
        assert!(
            (share - 0.9).abs() < 0.05,
            "big rank share {share} must stay ≈ 0.9"
        );
        // Small inputs are returned verbatim (no artificial cap).
        let s2 = population_sample(&rank_pixels, &[1], 8192);
        assert_eq!(s2.len(), 1000);
    }

    /// The final group fit must sample the region's real population: a
    /// 90/10 rank split sampled per-rank-balanced would tilt the fit toward
    /// the minority (the measured chest-darkening bug); the population
    /// sample keeps the 90/10 shares.
    #[test]
    fn population_sample_keeps_population_shares() {
        let big: Vec<u32> = (0..9000).collect();
        let small: Vec<u32> = (9000..10_000).collect();
        let rank_pixels = vec![big, small];
        let s = population_sample(&rank_pixels, &[0, 1], 1000);
        let from_big = s.iter().filter(|&&p| p < 9000).count();
        assert_eq!(s.len(), 1000);
        let share = f64::from(from_big as u32) / 1000.0;
        assert!(
            (share - 0.9).abs() < 0.05,
            "big rank share {share} must stay ≈ 0.9"
        );
        // Small inputs are returned verbatim (no artificial cap).
        let s2 = population_sample(&rank_pixels, &[1], 8192);
        assert_eq!(s2.len(), 1000);
    }

}
