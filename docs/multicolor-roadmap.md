# Wraith → best-in-class multicolor PNG→SVG

**Scope:** `elcoosp/wraith` @ `8962674`. I read the whole code base plus the `vtracer 0.6.5` / `visioncortex 0.8.8` sources it depends on, researched current techniques, and built and **measured** a prototype of the proposed engine against wraith's current color path.

---

## 0. TL;DR

1. **The draft is right about *what exists* and wrong about *what to do*.** `Mode::Color` exists, but adding "a quantization stage in front of vtracer" and "grouping by color" would not get you to best quality. Grouping is actively unsafe (§2.3). The quality ceiling comes from how vtracer *places boundaries* (whole-pixel tracing, no use of anti-aliasing information, no error bound on curve fitting), not from the number of colors.
2. **The best architecture for "quality at high speed" is a classical geometric pipeline, not a neural or differentiable-rendering one.** It should:
   - recover **sub-pixel coverage** from anti-aliased edges;
   - extract **sub-pixel contours** from that coverage;
   - fit **corner-aware, error-bounded** curves;
   - compose layers with **trapping** (underlay) so there are no seams and shared edges aren't duplicated.

   Optimization-based vectorizers (LIVE, DiffVG family, SAMVG, Bézier Splatting) produce impressive results on photos/complex art but cost seconds-to-minutes. They belong behind an optional flag (§3).
3. **Measured result (prototype, 4 ground-truth fixtures, 1 vCPU):** against wraith's exact current Color-mode settings, the prototype has **2.6–3.3× lower error vs. ground truth at 4× zoom** on flat art (3.0 on the transparent fixture), **10–55% smaller files**, and is **~1.0–2.5× faster**. It **fails badly on gradients** (307 KB vs 3.5 KB). That is the single biggest remaining problem, and it is milestone M3. Details and every failure are in §5.
4. **Breaking the API is justified.** The `Settings` struct has no way to express palette size, tolerance, or gradients. A proposed redesign is in M7.

> **How to read the evidence in this document.** Code marked ✅ *tested* was compiled and run in the prototype crate (in the zip). Code marked ⚠️ *sketch* was **not** compiled or run. I did not build wraith itself: it declares Rust 1.90 and the sandbox toolchain was 1.85, so the prototype is a standalone crate. The vtracer baseline uses the exact config values from `settings.rs::tracer_config()`.

---

## 1. What was verified (and how)

| Claim | Evidence |
|---|---|
| wraith has a color mode | `settings.rs`: `Mode::Color` → `vtracer::ColorMode::Color` |
| Color mode never thresholds | `preprocess.rs`: `threshold` forced to `None` unless `Mode::BlackWhite` |
| Serializer only hoists a fill if *all* paths share it | `svgout.rs::write_svg` |
| vtracer has **no parallelism** | grep for `rayon`/`thread` in both crates: no hits |
| vtracer's "palette" is **not** k-means/median-cut | `visioncortex/color_clusters/runner.rs`: hierarchical region growing; "same color" = RGB values compared after `>> (8 - color_precision)`; layering decided by an **L1 distance in raw RGB** |
| Curves are not error-bounded | `path/smooth.rs` + `spline.rs`: staircase removal → 4-point subdivision smoothing → cuts at angle changes → `bezier::Curve::fit_from_points(points, 10.0)`, **only `curves[0]` is used**, then handle retraction. There is no max-distance guarantee. |
| Boundaries are whole-pixel | `BinaryCluster::image_to_compound_path` traces pixel outlines, so anti-aliasing information is thrown away at clustering time |

---

## 2. Audit of wraith + corrections to the draft

### 2.1 Draft claims vs reality

| Draft claim | Verdict |
|---|---|
| "`Mode::Color` exists and is wired to vtracer" | ✅ True |
| "Preprocessor preserves color" | ✅ True |
| "Serializer keeps per-path fill in color mode" | ✅ True |
| "No color quantization stage → add median-cut/k-means before `trace`" | ⚠️ **Misdiagnosis.** vtracer already merges by color distance, so pre-quantizing mostly duplicates it. The measured error is concentrated on **edges**, not palette choice. *(I did not test "quantize then vtracer"; this is reasoned from the source plus the §5 edge crops.)* |
| "Group paths by fill color to shrink the SVG" | ❌ **Unsafe as stated.** vtracer's output is in *stacked* paint order. Moving same-color paths together changes z-order and breaks rendering. Only merge paths of equal fill that are consecutive in z-order. In the proposed engine each layer is already one `<path>`. |
| "Quality presets are tuned for BW" | ⚠️ Partly. Presets derive from `Preset::Bw` but override most fields. The real problem: `hierarchical` (always Stacked) and `splice_threshold` are fixed, and the only color knobs are `color_precision`/`layer_difference`, which are not independently exposed. |
| "Gradients unsupported" | ✅ Directionally true (each cluster gets one `residue_color`). |
| Priorities P1–P5 | Missing: sub-pixel boundaries, error-bounded fitting, alpha handling, any quality measurement. |

### 2.2 Bugs and sharp edges found in the code

1. **BW mode ignores alpha.** vtracer's binary path does `to_binary_image(|x| x.r < 128)` (`converter.rs`). wraith's threshold pass preserves alpha, but vtracer never reads it. A PNG with transparent pixels stored as RGB(0,0,0,0) would trace as a solid black block. *(From source; I did not run it.)*
2. **`strip_white` is a color-equality hack.** It drops paths whose fill is exactly `#FFFFFF`. Near-white anti-aliasing clusters survive, and legitimately white interior shapes (eye whites, a white ring) are deleted. The correct tool is alpha, not color.
3. **Color-mode alpha is only keyed when ≥20% of pixels on five scanlines are transparent** (`should_key_image`). Semi-transparent edges are otherwise ignored.
4. **`panic = "abort"` + a dependency with `assert!`s** (e.g. `assert!(is_same_color_a < 8)`) means any panic inside a rayon batch kills the whole process. Fine today because `Settings` can't produce the bad value, but it is a fragile contract once colors/precision become user-controlled.
5. **Benchmarks measure speed only.** The fixtures are flat shapes with no anti-aliasing noise, and there is no quality metric, so nothing in CI can detect a quality regression. This is why M0 exists.

### 2.3 Why vtracer's color output is limited (root causes)

- **No sub-pixel information.** Anti-aliased edge pixels become either slivers or get absorbed by a neighboring cluster; the boundary is then a pixel staircase that is smoothed afterwards.
- **No error bound** on curve fitting (see §1), so the smoothness/fidelity trade-off is a set of heuristic angle thresholds.
- **Stacked clusters** duplicate each parent's area under each child, and each cluster's fill is just an average color.
- **RGB-space decisions**, not perceptual.

---

## 3. State of the art: what is relevant to a fast Rust CLI

| Technique | What it contributes | Verdict for wraith |
|---|---|---|
| **vtracer / visioncortex** (baseline) | Fast O(n) clustering + spline fit | Replace for color mode; keep for `--engine legacy` |
| **Vectorizer.AI** (commercial leader) | Per its own site: deep nets + classical geometry; **full-shape fitting** (circles, ellipses, rounded rects, stars), **lines, circular/elliptical arcs, quadratic & cubic Béziers**; **sub-pixel precision using anti-aliasing values**; corner cleaning, tangent matching, curve fairing | **Best public target.** We copy the *ideas*: sub-pixel (done in prototype), corner cleaning (done), primitives (M4) |
| **Levien's optimal Bézier fitting / `kurbo`** | Error-bounded fitting. `fit_to_bezpath` is fast; `fit_to_bezpath_opt` is slower but finds better subdivisions. The `simplify` module is experimental and doesn't report corners. | Bake-off against my Schneider fitter in M4 *(not tested here)* |
| **`quantette` / `okolors`** | Oklab quantization; Wu (fast) and k-means (accurate); SIMD, rayon. Its README shows k-means-Oklab DSSIM roughly half that of Wu-sRGB at 16 and 64 colors | Use for palette extraction on large images (M2) |
| **LIVE** (CVPR'22) | Layer-wise: add closed Bézier paths progressively, optimize with a differentiable rasterizer; compact, topology-aware | Philosophy adopted (layers); optimizer too slow for default |
| **O&R / SAMVG / SuperSVG** | Top-down optimize-and-prune; SAM masks as initialization then differentiable refinement; superpixel-based model | Optional future `--photo` track (M6), GPU/ML dependency |
| **Segmentation-guided gradient fills** (ECCV'24) | Radial gradient fills with gradient-aware segmentation | Informs M3 (region-first gradient detection) |
| **Bézier Splatting** (NeurIPS'25) | Gaussians sampled along Bézier curves; reported 30×/150× faster forward/backward rasterization than DiffVG for open curves; SVG export | Watch; relevant only if M5 grows into a differentiable refiner |
| **COVec** (2025) | Albedo/shade/light layer decomposition for real-world images | Out of scope; photographic |

**Decision:** *classical geometric front-end + optional local refinement*. Rationale: for logos, icons and flat illustration (wraith's stated domain) the information needed is in the anti-aliased edges, and a geometric pipeline can exploit it at tens of milliseconds per 512² image. Differentiable renderers solve a harder problem (photos) at 100–10,000× the cost.

Sources: kurbo docs (`doc.servo.org/kurbo/fit`), Levien's *Fitting cubic Bézier curves* (raphlinus.github.io/curves/2021/03/11/bezier-fitting.html), `lib.rs/crates/quantette`, `vectorizer.ai`, arXiv 2206.04655 (LIVE), 2311.05276 (SAMVG), 2408.15741 (gradient fills), 2511.20034 (COVec), NeurIPS 2025 poster 117178 (Bézier Splatting).

---

## 4. Proposed architecture

```text
PNG ─ decode ─▶ unique colors ─▶ Oklab palette ─▶ AA-ghost pruning ─▶ mixture model (a,b,t) per color
                                                                              │ sub-pixel coverage
        ┌─────────────────────────────────────────────────────────────────────┘
        ▼
   per-layer soft field  F′ᵢ = covᵢ + laterᵢ · dilate₁(covᵢ)         ← "trapping"
        ▼
   marching squares @ 0.5 (linear interpolation → sub-pixel rings)
        ▼
   smooth → corner reconstruction (chamfer → line intersection) → line snapping (TLS)
        ▼
   Schneider cubic fit, error-bounded, G1 at smooth joins
        ▼
   one <path> per layer, relative coords, integer-hundredths  ─▶ (OXVG) ─▶ SVG
```

Key ideas, each of which I implemented and measured:

1. **Palette in Oklab, anti-aliasing model in gamma-sRGB.** Cluster perceptually; but a rasterizer blends edge pixels in gamma-encoded sRGB, so the mixture `pixel ≈ (1−t)·A + t·B` is only a straight line *there*. Fitting `t` in Oklab distorted edge positions; moving the mixture model to sRGB removed visible wobble.
2. **AA-ghost pruning.** Edge pixels create phantom palette entries. Rule: drop a color if (a) < 3% of its pixels have all four neighbors with the same label, and (b) it lies within ΔOklab 0.03 of the segment between two other palette colors. A real flat 3×3 region survives; a 1-px anti-aliasing band does not.
3. **Best-pair mixture selection** (not nearest-color-then-partner). With nearest-first I got a red fringe on a yellow/blue edge, because a mixed pixel's nearest centroid was an unrelated color.
4. **Trapping instead of cumulative stacking.** Each layer = its own region plus a 1-px underlay into layers painted later. No seams, and shared edges appear about twice instead of once per intervening layer. Measured in §5.
5. **Corner reconstruction.** The iso-contour of a bilinear field chamfers sharp corners. Detect short RDP chains between long segments and replace them with the intersection of the long segments' lines; snap near-straight pieces to their total-least-squares line.
6. **Error-bounded Schneider fit** (tolerance 0.3 px default) with G1 continuity at non-corner joins.

---

## 5. Measured results ✅

**Setup.** 4 fixtures defined as SVG (ground truth), rasterized to 512² PNG by `resvg`. Each engine's output SVG is rendered back at 1× and **4×** and compared with the *ground-truth SVG rendered at the same scale*. The 4× figure measures true vectorization fidelity (sub-pixel placement), not just reproduction of the input pixels. 1 vCPU, release build (fat LTO), min of 5 runs. Sizes are **unminified** for every engine (wraith's OXVG pass would shrink all of them). Error = RMSE over RGB composited on white. "Bad" = share of pixels with any channel off by more than 40.

| Fixture | Engine | ms | paths | bytes | RMSE 1× | RMSE 4× | bad% 4× |
|---|---|---:|---:|---:|---:|---:|---:|
| flat_logo | vtracer *balanced* | 30.8 | 7 | 9,730 | 9.54 | 11.90 | 0.84 |
| | vtracer *quality* | 30.0 | 7 | 12,846 | 9.88 | 12.22 | 0.89 |
| | **proposed** | **12.5** | 5 | **4,569** | **2.28** | **4.10** | **0.28** |
| illustration | vtracer *balanced* | 29.1 | 15 | 11,893 | 5.33 | 5.94 | 0.48 |
| | vtracer *quality* | 30.2 | 17 | 15,537 | 5.23 | 5.81 | 0.46 |
| | **proposed** | **22.2** | 13 | **8,597** | **1.25** | **2.29** | **0.15** |
| transparent_logo | vtracer *balanced* | 15.9 | 4 | 6,321 | 13.36 | 16.71 | 1.07 |
| | vtracer *quality* | 16.2 | 4 | 7,952 | 13.51 | 16.83 | 1.08 |
| | **proposed** | **14.7** | 4 | **3,587** | **2.78** | **5.11** | **0.31** |
| **gradient_logo** | vtracer *balanced* | 28.8 | 4 | 3,542 | 17.71 | 18.11 | 9.48 |
| | vtracer *quality* | 78.8 | 75 | 124,900 | 8.99 | 9.67 | 0.47 |
| | **proposed** ❌ | 72.3 | 21 | **307,073** | 5.05 | 5.70 | 0.13 |

The vtracer figures (~30 ms for 512² color on one core) are consistent with the README's 33.6 ms, so the baseline is a fair reproduction.

### 5.1 What each step bought (sequential log, not isolated ablations)

| Step | Illustration bytes | Illustration RMSE 4× | Note |
|---|---:|---:|---|
| First working version (cumulative stacking, raw contour fit, tol 0.2) | 92,093 | 4.24 | fitter shatters edges into ~1 px segments: contour noise > tolerance |
| + ring smoothing, tol 0.3 | 51,138 | 4.25 | confirms noise hypothesis |
| + **trapping** (replaces cumulative) | 23,791 | 3.48 | per-layer cost profile showed the 5 largest layers at 6–8 KB each |
| + best-pair mixture, chain-chamfer, TLS line snapping | 12,483 | 2.75 | red fringe gone, sharp corners |
| + sRGB mixture model, `nonzero` fill | **8,597** | **2.29** | wobble and white specks gone |
| + per-layer bounding boxes | 8,597 *(identical)* | 2.29 | **47.5 → 22.2 ms** |

### 5.2 Visual check (crops in `figures/`)

- **Star / circle edge:** the proposed output is visually indistinguishable from ground truth at 4×: sharp concave corner, straight edge, accurate arc. vtracer rounds the corner, truncates the tip and wobbles the arc.
- **Rounded rectangle (24-px radius):** both engines blob the arc transition, and mine kinks slightly where the arc meets the straight edge. Bézier curves cannot represent line + circular arc exactly → motivates primitives (M4).
- **Thin window frame (3 px white):** mine merges the frame with the near-white cloud color (merge threshold ΔOk 0.02) and leaves a faint ghost ring inside. vtracer also fails here, differently (grey rounded frame).
- **Door corner** shows a small bump (corner not detected).

### 5.3 What the prototype does **not** show

- **Gradients: fails.** 307 KB, and 21 layers of banding rings. The mixture model explains smooth ramps as noisy mixtures of far-apart bands. Needs M3.
- **1 vCPU only.** Rayon parallelism over layers is implemented but its scaling is **not measured**. The README reports 1.47× on 2 cores for batch parallelism; I make no claim beyond that.
- **Only 4 synthetic fixtures**, rendered by one rasterizer (`resvg`: exact area coverage, blended in sRGB). Real PNGs differ (other renderers, resampled/upscaled art, JPEG-origin artifacts, gamma-correct blending). The sRGB-mixture assumption must be validated on a real corpus (M0/M1 gate). A per-image blend-gamma estimate is the planned fallback.
- No text, photos, or hairline-heavy line art tested.

---

## 6. Milestones

Effort figures are my rough estimates for one engineer familiar with the code base, not measurements.

| # | Milestone | Effort | Gate (must pass to merge) |
|---|---|---|---|
| **M0** | Truth first: eval harness, corpus, bug fixes | 2–3 d | Baseline numbers recorded; CI fails on quality regression |
| **M1** | Geometric color engine v1 (the tested prototype, productionized) | 5–8 d | On corpus: RMSE@4× ≤ 0.6× vtracer-balanced; bytes (post-OXVG) ≤ vtracer; single-thread time ≤ vtracer; **no seams at 8×** |
| **M2** | Performance pass | 4–6 d | 512² flat ≤ 8 ms/core (target); ≥1.7× on 2 cores for single large image; no per-layer full-image passes |
| **M3** | Gradients (linear + radial) | 8–12 d | gradient fixture: ≤ 20 KB and RMSE@4× ≤ 3.5; no regression on flat |
| **M4** | Fitting upgrade: kurbo bake-off, lines/arcs, circle/ellipse/rounded-rect primitives | 6–10 d | rounded-rect crop error ↓; bytes ↓ ≥15% on icon set |
| **M5** | Field-driven refinement (`--refine`) | 5–8 d | RMSE@4× ↓ ≥10% on corpus at ≤ +30% time |
| **M6** | Photo / complex mode (research track) | open | Decide go/no-go after M5 |
| **M7** | API & CLI redesign (breaking) | 2–3 d | Docs, `--json` stats incl. palette + path counts, semver bump |

---

### M0: Truth first

**Why first:** every claim above depends on numbers; wraith currently cannot detect a quality regression.

**Deliverables**

1. New crate `wraith-eval` (dev tool + CI job):
   - Corpus: SVG ground truths (icon sets such as Twemoji/Noto/Font-Awesome/simple-icons; plus your own logos) → rasterize with `resvg` at 512²/1024² → run engine → rasterize output at 1× and 4× → metrics.
   - Metrics: RMSE, share of pixels with large error, **ΔE in Oklab**, an SSIM-family metric (e.g. the `ssimulacra2` crate), path count, bytes **after OXVG**, time.
   - Output a JSON report; CI compares to a committed baseline with tolerances.
2. Fix §2.2 bugs. **⚠️ sketch** (preprocess alpha handling):

```rust
// preprocess.rs — treat alpha as first-class, never as a color.
// BW mode: composite over white *before* thresholding so transparent == background.
fn composite_on_white(px: &mut [u8; 4]) {
    let a = px[3] as u32;
    for c in 0..3 { px[c] = ((px[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8; }
    px[3] = 255;
}
```

   - `strip_white` becomes `strip_background: Option<Background>` where `Background::{Alpha, Color(rgb, tolerance)}`: remove by *alpha* or by *tolerance*, and only for regions touching the image border.
   - Decouple color settings from `Preset::Bw` in `tracer_config` (expose `color_precision`, `layer_difference`, `hierarchical`).

**Acceptance:** `cargo run -p wraith-eval` prints the §5-style table for the legacy engine; CI job fails when RMSE@4× regresses >5% or bytes >5%.

---

### M1: Geometric color engine v1 ✅ (tested prototype)

New module layout in `wraith-core`:

```text
src/color/
  mod.rs        // pub fn trace_color(img, &ColorSettings) -> ColorTrace
  palette.rs    // unique colors, k-means/Wu, merge, AA-ghost pruning
  model.rs      // mixture model (a,b,t) per unique color
  layers.rs     // coverage fields, trapping, bbox
  contour.rs    // marching squares
  fit.rs        // corners, line snapping, Schneider
  write.rs      // relative-coordinate path writer
```

Everything below is quoted from the **tested** prototype (`wraith-color-proto/src/pipeline.rs`).

#### 1) Mixture model + best-pair selection ✅

```rust
// Mixture model lives in gamma-encoded sRGB: that is the space in which
// 2D rasterisers blend anti-aliased edges, so t == true pixel coverage.
let psrgb: Vec<V3> = pal.iter().map(|c| { let r = oklab_to_rgb(*c); [r[0] as f32, r[1] as f32, r[2] as f32] }).collect();
for (u, p) in usrgb.iter().enumerate() {
    let mut near: Vec<(f32, usize)> = psrgb.iter().enumerate().map(|(i, q)| (d2(*p, *q), i)).collect();
    near.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let na = near[0].1;
    let (mut ba, mut bj, mut bd, mut bt) = (na, na, near[0].0, 0.0f32);
    for &(_, ca) in near.iter().take(4) {              // 4 nearest as "dominant" candidates
        for j in 0..psrgb.len() {
            if j == ca { continue; }
            let (d, tt) = seg_fit(*p, psrgb[ca], psrgb[j]);   // distance to segment ca–j + projection t
            if d + 1e-3 < bd { bd = d; ba = ca; bj = j; bt = tt; }
        }
    }
    a[u] = ba as u8; b[u] = bj as u8; t[u] = if bj == ba { 0.0 } else { bt };
}
```

Computed **per unique color, not per pixel**, so cost is O(U·K²) with U small for graphic art.

#### 2) AA-ghost pruning ✅

```rust
// Drop palette colours that (1) almost never form a 3x3 interior and
// (2) are explained as a mixture of two other colours.
for _ in 0..4 {
    let lab_u: Vec<u8> = ulab.iter().map(|p| nearest(*p, &pal) as u8).collect();
    // total[c] = pixels labelled c; inner[c] = those whose 4 neighbours share the label
    ...
    for &c in &order_by_total_ascending {
        if inner[c] as f32 / total[c] as f32 > 0.03 { continue; }       // real flat region
        let best = min over pairs (i,j) != c of seg_fit(pal[c], pal[i], pal[j]).0;
        if best.sqrt() < 0.03 { removed[c] = true; }                    // explained as a blend
    }
    if !removed.iter().any(|&r| r) { break; }
    pal.retain(not removed);
}
```

#### 3) Marching squares with sub-pixel interpolation ✅

The field is padded by one cell of zeros. Each of the 16 cases emits a directed segment with *inside on the left*, so every edge has exactly one outgoing segment and chaining is a pointer walk with no hashing.

```rust
fn contours(f: &[f32], pw: usize, ph: usize) -> Vec<Vec<P>> {
    const ISO: f32 = 0.5;
    let hid = |x: usize, y: usize| (2 * (y * pw + x)) as u32;      // horizontal edge id
    let vid = |x: usize, y: usize| (2 * (y * pw + x) + 1) as u32;  // vertical edge id
    let mut nxt = vec![u32::MAX; 2 * pw * ph];                     // edge → next edge
    for y in 0..ph - 1 { for x in 0..pw - 1 {
        let (vtl, vtr, vbr, vbl) = (f[y*pw+x], f[y*pw+x+1], f[(y+1)*pw+x+1], f[(y+1)*pw+x]);
        let case = (vtl >= ISO) as u8 | ((vtr >= ISO) as u8) << 1 | ((vbr >= ISO) as u8) << 2 | ((vbl >= ISO) as u8) << 3;
        if case == 0 || case == 15 { continue; }
        let (t, b, l, r) = (hid(x, y), hid(x, y + 1), vid(x, y), vid(x + 1, y));
        let mut seg = |from: u32, to: u32| nxt[from as usize] = to;
        match case {
            1 => seg(l, t), 2 => seg(t, r), 4 => seg(r, b), 8 => seg(b, l),
            3 => seg(l, r), 6 => seg(t, b), 12 => seg(r, l), 9 => seg(b, t),
            7 => seg(l, b), 11 => seg(b, r), 13 => seg(r, t), 14 => seg(t, l),
            5  => if (vtl+vtr+vbr+vbl) * 0.25 >= ISO { seg(r, t); seg(l, b) } else { seg(l, t); seg(r, b) },
            10 => if (vtl+vtr+vbr+vbl) * 0.25 >= ISO { seg(t, l); seg(b, r) } else { seg(t, r); seg(b, l) },
            _ => unreachable!(),
        }
    } }
    // crossing point on an edge = linear interpolation of the coverage field to ISO
    let pt = |id: u32| -> P { /* t = (ISO - v0) / (v1 - v0); returns sub-pixel (x, y) */ };
    // chain: start at any unvisited edge, follow nxt[] until we return
    ...
}
```

#### 4) Layers with trapping + bounding boxes ✅

```rust
// cov   = own coverage of layer li            (labels == li)
// later = coverage of everything painted after (labels  > li)
// F′    = cov + later · max3x3(cov)           → own region + 1px underlay into later layers
let mut f = vec![0f32; spw * sph];                       // padded, bbox-sized
for y in 0..bh { for x in 0..bw {
    let i = (y + y0) * w + x + x0;
    let m = max3x3(cov, i);                              // separable 3×3 max filter (2 passes)
    f[(y + 1) * spw + x + 1] = if alpha[i] > 0 { (cov[i] + later[i] * m).min(1.0) } else { 0.0 };
} }
```

Why it works: where layer *i* meets a **later** layer *j*, *i* extends ~1 px underneath *j* (no hairline gap even at 8× zoom). Where *i* meets an **earlier** layer *k*, `later = 0`, so the edge is the exact 0.5 iso-line of *i*'s own coverage. Each shared boundary is thus drawn about twice (underlay + visible edge) instead of once per intervening layer (§5.1: 51 KB → 24 KB at that step).

Use `fill-rule="nonzero"` (default). Contours have consistent orientation, so tiny fitter loops stay filled instead of becoming `evenodd` holes (this fixed the white specks).

#### 5) Corners, line snapping, Schneider ✅

Full code is in the zip (`find_corners`, `tls_line`, `fit_ring`, `fit_cubic`). The non-obvious part is **chamfer reconstruction**:

```rust
// A "chamfer" = 1–3 short RDP segments between two long ones with a large total turn:
// the rounded footprint of a sharp corner. Replace it by the intersection of the two long segments' lines.
for e in 1..=3usize {
    if (0..e).any(|q| l(j + q, j + q + 1) >= 2.0) { break; }                  // chain must be short
    if (0..e).map(|q| l(j + q, j + q + 1)).sum::<f64>() > 4.5 { break; }
    if l(k, k + 1) < 3.0 { continue; }                                         // followed by a long segment
    let ang = dot(unit(sub(b, a)), unit(sub(d, c))).clamp(-1.0, 1.0).acos();
    if ang < 0.7 { continue; }                                                 // ≥ ~40° turn
    if let Some(x) = line_isect(a, b, c, d) {
        if len(sub(x, mid(b, c))) < 2.5 { /* accept corner at x */ }
    }
}
```

Straight pieces (`dev` from total-least-squares line < tolerance and chord ≥ 4 px) become exact `l` segments, and a corner between two snapped lines moves to their intersection.

#### 6) Writer: relative coordinates in integer hundredths ✅

```rust
fn hund(v: f64) -> i64 { (v * 100.0).round() as i64 }
// Deltas are taken between *rounded absolute* points → no cumulative drift in relative paths.
fn fmt_h(v: i64, s: &mut String) {
    let (neg, a) = (v < 0, v.abs());
    let (i, f) = (a / 100, a % 100);
    let mut t = String::new();
    if neg { t.push('-'); }
    if i > 0 || f == 0 { let _ = write!(t, "{i}"); }
    if f != 0 { if f % 10 == 0 { let _ = write!(t, ".{}", f / 10); } else { let _ = write!(t, ".{f:02}"); } }
    // a separator is needed before any non-negative number that follows a digit or '.'
    if !neg && matches!(s.chars().last(), Some(c) if c.is_ascii_digit() || c == '.') { s.push(' '); }
    s.push_str(&t);
}
```

*(My first version omitted the separator before a literal `0`, producing `8.010 12c` and a corrupted path. This is why a render-back test belongs in CI.)*

**Dependencies:** `rayon` (already present); optional `quantette` for M2. No new heavy deps.

**M1 gate:** §5 table on the full M0 corpus; plus a **seam test**: render at 8× and assert no pixel along any shared edge differs from ground truth by more than a threshold.

---

### M2: Performance

Measured profile of the prototype before bounding boxes (512², 1 core): palette ≈ 3 ms; per-layer work (field + filter + contour scan + fit) = 80–95% of runtime, because every layer did several full-image passes. Bounding boxes alone cut flat_logo 18.7 → 12.5 ms and illustration 47.5 → 22.2 ms with identical output.

Plan, in order of expected payoff:

1. **Single scatter pass instead of K full-image passes.** Walk pixels once; push each pixel's `(a,b,t,alpha)` contribution into per-layer sparse tile lists (e.g. 32×32 tiles). Layer work becomes proportional to the layer's area plus its trap band, not to K·W·H.
2. **Skip homogeneous tiles in marching squares.** Tiles where all field values are 0 or all ≥ 0.5 cannot contain boundary cells. Scan only boundary tiles.
3. **Parallelism.** Layers are independent (`par_iter` over layers is already in the prototype; not measured here). For one huge layer, split by tile bands and stitch rings at tile borders. Rings fit independently (`par_iter` over rings).
4. **Reuse scratch buffers** (thread-local `Vec`s). The `nxt` table is `2·pw·ph` `u32`s, which is allocated per layer today.
5. **Palette on big inputs.** K-means over all unique colors is fine for graphic art but not for 4 MP+ images. Switch to `quantette`'s Wu quantizer on a 5–6-bit histogram (SIMD) and optionally k-means refine; keep the ghost-pruning stage.
6. **SIMD** for sRGB→Oklab (LUT for the transfer function + `wide`) and nearest-palette search.

**Gate (targets, not yet achieved):** 512² flat ≤ 8 ms per core; ≥1.7× on 2 cores for a single large image; peak memory ≤ ~3× decoded image.

---

### M3: Gradients (linear + radial)

**Problem shown in §5.3.** The pipeline treats smooth ramps as many bands. The fix has to detect gradients *before* they are split into layers.

**Approach: region-first, MDL-style band merging** (informed by gradient-aware segmentation in the ECCV'24 paper; the algorithm below is my design and is untested).

1. **Over-segment** with a larger palette (K = 32–64) so a ramp becomes a *chain* of adjacent bands.
2. Build a **band adjacency graph**; edge weight = mean Oklab difference across the shared boundary.
3. **Greedily merge** neighboring bands when (a) the union's best-fit *affine color model* `c(x,y) = c₀ + gₓ·x + g_y·y` (per channel, in gamma sRGB) has RMS residual < δ (≈ 2/255), and (b) the merge reduces a description-length cost (path bytes saved − gradient-definition bytes). A merged region with ≥ 3 original bands becomes a **gradient region**.
4. Fit the gradient: axis = dominant singular direction of the 3×2 gradient matrix; project pixels on the axis; bin to 16–32 samples; reduce to 2–6 stops by RDP in color space (tolerance ≈ 2/255). For radial: Gauss–Newton on center and a radius-to-color curve; accept whichever model has the lower residual; fall back to flat bands when neither beats δ.
5. Trace the region outline with the **same** soft-field/trapping machinery (union of its bands' coverage). Emit `fill="url(#gN)"`.

⚠️ **Sketch** (not compiled):

```rust
pub struct LinearFit { pub p0: [f32; 2], pub p1: [f32; 2], pub stops: Vec<(f32, [u8; 3])>, pub rmse: f32 }

/// c(x,y) = c0 + gx·x + gy·y per channel, by least squares on centred coordinates.
pub fn fit_linear(px: &[(f32, f32, [f32; 3])]) -> Option<LinearFit> {
    let n = px.len() as f32;
    let (cx, cy) = (px.iter().map(|p| p.0).sum::<f32>() / n, px.iter().map(|p| p.1).sum::<f32>() / n);
    // normal equations for [x y] (2x2) per channel; solve → g[c] = (gx, gy)
    // M = 3x2 matrix of g; principal axis = top eigenvector of MᵀM (closed form for 2x2)
    // t = ((x-cx, y-cy) · axis − tmin) / (tmax − tmin); bin mean colours; RDP → stops
    // rmse over all pixels of the *reconstructed* gradient; return None if > threshold
    todo!()
}
```

**Gate:** gradient fixture ≤ 20 KB and RMSE@4× ≤ 3.5; flat fixtures unchanged (a gradient region must never be created from a ≤2-band chain).

---

### M4: Fitting upgrade and primitives

1. **Bake-off** of three fitters on the same rings, scored by max distance, segment count and bytes: current Schneider vs `kurbo::fit_to_bezpath` vs `kurbo::fit_to_bezpath_opt`. kurbo's API takes a `ParamCurveFit` source, so a polyline needs a small adapter, and its simplify module doesn't report corners. Keep wraith's own corner pass in front. *(Not tested here.)* Choose by data.
2. **Primitive detection on rings** (⚠️ sketch; motivated by the rounded-rectangle failure in §5.2 and by Vectorizer.AI's published feature set):
   - **Circle/ellipse:** direct least-squares ellipse fit; if max radial residual < 0.25 px and no corners, emit `<ellipse>` (≈ 40 bytes vs 150–400 for Béziers).
   - **Rounded rect:** 4 snapped lines + 4 arcs with equal radii → `<rect rx>`.
   - **Arcs:** replace Bézier runs by SVG `A` commands when the circle-fit residual is below tolerance.
3. **Tangent matching at primitive/curve joins** to keep G1.

**Gate:** rounded-rect crop error ↓; bytes ↓ ≥ 15% on an icon set; zero regressions elsewhere.

---

### M5: Field-driven refinement (`--refine`)

Idea (my design; untested). Because the pipeline already has a **target coverage field** F′ for every layer, refinement needs no differentiable renderer: minimize

`E = Σₛ (F′(pₛ) − 0.5)²`

over sampled points `pₛ` on each fitted curve, by gradient descent on control points using the bilinear gradient `∇F′`. It removes interpolation bias that marching squares introduced, runs only on boundary pixels, and can be limited to paths whose residual exceeds a threshold.

```rust
// ⚠️ sketch
for _ in 0..iters {
    for seg in layer.segments_mut() {
        for (u, w) in sample_uniform(seg, 8) {                       // points + Bernstein weights
            let (v, g) = bilinear_value_and_grad(&field, seg.at(u)); // F′ and ∇F′
            let r = v - 0.5;
            for (k, ctrl) in seg.interior_controls_mut().enumerate() {
                *ctrl -= lr * r * g * w[k];                           // chain rule: ∂p/∂ctrl_k = w_k
            }
        }
    }
}
```

Corners stay pinned. Refit and re-measure error afterwards (accept only if the max distance improves). **Gate:** RMSE@4× ↓ ≥ 10% at ≤ +30% time.

---

### M6: Photo / complex-image mode (research, go/no-go after M5)

The techniques in §3 (LIVE, SAMVG, O&R, Bézier Splatting) target photographs and need many path updates and a rasterizer in the loop. If a `--photo` track is wanted: palette up to 128–256, edge-preserving pre-smoothing, hierarchical region merging, M3 gradients as the workhorse, and optional rendering-in-the-loop refinement. Treat it as a separate product decision, not part of the "best quality at the most performant level" core.

---

### M7: API and CLI redesign (breaking) ⚠️ sketch

```rust
pub enum Engine { Legacy, Geometric }            // Geometric default for Mode::Color

pub struct ColorSettings {
    pub max_colors: Colors,                      // Auto | N
    pub merge_distance: f32,                     // ΔOklab, default 0.02
    pub tolerance_px: f32,                       // curve fit, default 0.3
    pub corner_angle_deg: f32,                   // default ~40
    pub trap_px: f32,                            // underlay width, default 1.0
    pub gradients: Gradients,                    // Off | Linear | LinearRadial
    pub primitives: bool,                        // M4
    pub refine_iters: u8,                        // M5, 0 = off
    pub background: Background,                  // None | Alpha | Color{rgb, tol}
    pub min_area_px2: f32,                       // speck filter
}
```

CLI additions: `--engine`, `--colors <N|auto>`, `--tolerance`, `--gradients`, `--refine <N>`, `--preset <logo|icon|illustration>`. `--json` stats should include palette size, path count, per-stage time and (when a reference is available) quality metrics. The `-m color` shorthand `--color` from the draft is fine, but secondary.

---

## 7. Risks and open questions

1. **Blend-space assumption.** The sRGB mixture model holds for the renderer that produced my fixtures. If a real PNG was blended gamma-correctly (or resampled), `t` is biased. Mitigation: estimate a per-image blend exponent from edge pixels; validate on real exports from several tools in M0/M1.
2. **Palette merge vs. thin details.** The window-frame result (§5.2) shows `merge_distance` can swallow a thin, near-white feature. Needs a feature-size-aware merge (don't merge colors whose combined region would delete a ≥ 1-px structure).
3. **Many-color art / screenshots.** K² mixture search and K-layer output grow with palette size. M2's scatter pass and M3's region merging are required before this is acceptable.
4. **JPEG-origin or noisy PNGs.** Mixture residuals are large; the ghost-pruning heuristics may over- or under-prune. Add a denoise pre-pass or a noise-adaptive threshold.
5. **Hierarchy and editability.** Trapped layers are flat and ordered by pixel count. For designer-friendly output, order by adjacency/containment.
6. **Real parallel scaling** is unmeasured on my side (1 vCPU).

---

## 8. Reproducing

```sh
unzip wraith-color-proto.zip && cd wraith-color-proto
cargo run --release                         # table in §5; writes out/*.svg and 4x PNGs
TOL=0.25 SMOOTH=2 cargo run --release       # tolerance sweep
TRAP=0 cargo run --release                  # cumulative stacking instead of trapping
PROFILE=1 cargo run --release               # palette vs layer time
```

Needs Rust ≥ 1.85 (pins `image = 0.25.6` for older compilers; the prototype itself uses edition 2021).

**Files in the zip:** `src/pipeline.rs` (the engine, ~740 lines), `src/main.rs` (fixtures, vtracer baseline, resvg-based metrics), `figures/` (inputs, SVG outputs, and 4× crops: ground truth | vtracer | proposed).

---

## 9. Suggested order of work

**M0 → M1 → M2** gets a measurably better, faster color mode on flat art and a safety net against regressions. **M3** is the biggest remaining quality gap and should follow immediately. **M4/M5** push from "very good" to "indistinguishable at 8×". M6 is optional, and M7 should land together with M1 so the public API changes once.
