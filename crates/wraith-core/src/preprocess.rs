//! Stage 1: PNG bytes → tracer-ready RGBA image.
//!
//! Decode, background removal (roadmap M0: alpha is first-class, never a
//! colour) and (optionally) luminance-threshold happen in as few passes over
//! the pixel buffer as possible, so there is never a redundant allocation for
//! the tracer's input.

use image::DynamicImage;

#[cfg(test)]
use image::{ImageBuffer, Rgba};

use crate::error::Result;
use crate::settings::{Background, Mode, Settings};
use crate::stats::{Stage, Stopwatch};

/// A prepared raster in the tracer's native [`vtracer::ColorImage`] format.
pub struct Prepared {
    /// Tracer-ready RGBA buffer.
    pub image: vtracer::ColorImage,
}

/// Decodes PNG bytes and prepares the tracer input.
///
/// # Errors
///
/// Returns [`crate::error::WraithError::Decode`] when `png` is not a valid
/// PNG image.
pub fn prepare(png: &[u8], settings: &Settings, sw: &mut Stopwatch) -> Result<Prepared> {
    sw.start();
    let decoded: DynamicImage = image::load_from_memory(png)?;
    sw.lap(Stage::Decode);

    sw.start();
    // Take ownership of the RGBA buffer when it already is one; convert only
    // when the source has fewer channels. Dimensions are captured before the
    // buffer is moved.
    let (mut rgba, width, height) = match decoded {
        DynamicImage::ImageRgba8(buf) => {
            let (w, h) = (buf.width() as usize, buf.height() as usize);
            (buf.into_raw(), w, h)
        }
        other => {
            let buf = other.to_rgba8();
            let (w, h) = (buf.width() as usize, buf.height() as usize);
            (buf.into_raw(), w, h)
        }
    };

    // Background removal precedes everything else so both engines (and the
    // threshold pass) see the same "no background" raster.
    if let Background::Color { rgb, tolerance } = settings.background {
        flood_remove_background(&mut rgba, width, height, rgb, tolerance);
    }

    let threshold = match settings.mode {
        // BW mode composites over white *first* (transparent == background),
        // then optionally thresholds (roadmap §2.2.1).
        Mode::BlackWhite => {
            composite_on_white(&mut rgba);
            settings.threshold
        }
        Mode::Color => None,
    };
    let image = vtracer::ColorImage {
        pixels: threshold_pixels(rgba, threshold),
        width,
        height,
    };
    sw.lap(Stage::Preprocess);
    Ok(Prepared { image })
}

/// Rec. 601 luma, integer-only.
///
/// The intermediate sum is bounded to `0..=261_632`, so the shift result is
/// always within `0..=255` and both casts below are lossless.
fn luminance(px: &[u8]) -> u8 {
    // 0.299R + 0.587G + 0.114B, scaled by 1024 with a single rounding shift.
    let (r, g, b) = (i32::from(px[0]), i32::from(px[1]), i32::from(px[2]));
    let luma = (306 * r + 601 * g + 117 * b + 512) >> 10;
    u8::try_from(luma).unwrap_or(255)
}

/// Composites every pixel over white in place, forcing alpha to 255.
///
/// `pixel' = pixel·a + white·(1−a)` in gamma-encoded sRGB — exactly what a
/// renderer does before the pixels hit a black-and-white threshold, so a
/// transparent pixel can never become black geometry (roadmap §2.2.1).
fn composite_on_white(rgba: &mut [u8]) {
    for px in rgba.as_chunks_mut::<4>().0 {
        let a = u32::from(px[3]);
        if a == 255 {
            continue;
        }
        let inv = 255 - a;
        for c in &mut px[..3] {
            let v = (u32::from(*c) * a + 255 * inv + 127) / 255;
            *c = u8::try_from(v).unwrap_or(255);
        }
        px[3] = 255;
    }
}

/// Flood-removes border-connected regions within `tolerance` of `rgb`.
///
/// Four-connected scan from every border pixel whose per-channel distance to
/// `rgb` is within tolerance; matched pixels become fully transparent. Only
/// regions touching the image border are removed, so an interior shape with
/// the same colour survives (the failure mode of fill-colour stripping).
fn flood_remove_background(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    rgb: [u8; 3],
    tolerance: u8,
) {
    let tol = i32::from(tolerance);
    let matches = |px: &[u8]| {
        (0..3).all(|c| {
            i32::from(px[c]) - i32::from(rgb[c]) <= tol
                && i32::from(rgb[c]) - i32::from(px[c]) <= tol
        })
    };
    let mut visited = vec![false; width * height];
    let mut stack: Vec<usize> = Vec::new();
    let border = {
        let mut b = Vec::with_capacity(2 * (width + height));
        for x in 0..width {
            b.push(x);
            b.push((height - 1) * width + x);
        }
        for y in 0..height {
            b.push(y * width);
            b.push(y * width + width - 1);
        }
        b
    };
    for i in border {
        if !visited[i] && matches(&rgba[i * 4..i * 4 + 4]) {
            visited[i] = true;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        rgba[i * 4 + 3] = 0;
        let (x, y) = (i % width, i / width);
        let neighbours = [
            x.checked_sub(1).map(|nx| y * width + nx),
            (x + 1 < width).then(|| y * width + x + 1),
            y.checked_sub(1).map(|ny| ny * width + x),
            (y + 1 < height).then(|| (y + 1) * width + x),
        ];
        for n in neighbours.into_iter().flatten() {
            if !visited[n] && matches(&rgba[n * 4..n * 4 + 4]) {
                visited[n] = true;
                stack.push(n);
            }
        }
    }
}

/// Hands the decoded buffer to the tracer, snapping pixels to pure black/white
/// when a luminance threshold is supplied.
///
/// Without a threshold the decoded buffer is passed through **unmoved** — the
/// zero-copy path.
fn threshold_pixels(src: Vec<u8>, threshold: Option<u8>) -> Vec<u8> {
    let Some(t) = threshold else {
        return src;
    };
    // Pre-sized buffer + direct per-pixel writes: no call overhead in the
    // loop, no reallocation.
    let mut out = vec![0u8; src.len()];
    let (dst_px, _) = out.as_chunks_mut::<4>();
    let (src_px, _) = src.as_chunks::<4>();
    for (dst, px) in dst_px.iter_mut().zip(src_px.iter()) {
        let v = if luminance(px) < t { 0_u8 } else { 255_u8 };
        dst[0] = v;
        dst[1] = v;
        dst[2] = v;
        dst[3] = px[3];
    }
    out
}

/// Builds a small RGBA test fixture: white background with a black square.
#[cfg(test)]
pub(crate) fn fixture_rgba(size: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut img = ImageBuffer::from_pixel(size, size, Rgba([255, 255, 255, 255]));
    for y in size / 4..size * 3 / 4 {
        for x in size / 4..size * 3 / 4 {
            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
    }
    img
}

/// Encodes [`fixture_rgba`] as PNG bytes.
#[cfg(test)]
pub(crate) fn fixture_png(size: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgba8(fixture_rgba(size))
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("fixture encodes");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(threshold: Option<u8>) -> Settings {
        Settings {
            threshold,
            ..Settings::new()
        }
    }

    #[test]
    fn prepare_decodes_and_reports_dimensions() {
        let png = fixture_png(8);
        let mut sw = Stopwatch::default();
        let prepared = prepare(&png, &settings(None), &mut sw).expect("prepares");
        assert_eq!(prepared.image.width, 8);
        assert_eq!(prepared.image.height, 8);
        assert_eq!(prepared.image.pixels.len(), 8 * 8 * 4);
        let laps = sw.laps();
        assert_eq!(laps[0].0, Stage::Decode);
        assert_eq!(laps.last().map(|(s, _)| *s), Some(Stage::Preprocess));
    }

    #[test]
    fn prepare_without_threshold_passes_pixels_through() {
        let png = fixture_png(8);
        let mut sw = Stopwatch::default();
        let prepared = prepare(&png, &settings(None), &mut sw).expect("prepares");
        // White corner pixel survives untouched.
        assert_eq!(&prepared.image.pixels[0..4], &[255, 255, 255, 255]);
    }

    #[test]
    fn threshold_snaps_gray_and_white() {
        // Hand-built RGBA: mid-gray, white, near-black.
        let px = vec![128, 128, 128, 255, 255, 255, 255, 255, 10, 10, 10, 255];
        let out = threshold_pixels(px.clone(), Some(128));
        assert_eq!(&out[0..4], &[255, 255, 255, 255]); // 187 ≥ 128 → white
        assert_eq!(&out[4..8], &[255, 255, 255, 255]);
        assert_eq!(&out[8..12], &[0, 0, 0, 255]); // 10 < 128 → black
    }

    #[test]
    fn luminance_matches_rec601_midpoints() {
        assert_eq!(luminance(&[255, 255, 255, 255]), 255);
        assert_eq!(luminance(&[0, 0, 0, 255]), 0);
        // (306+601+117) * 127 = 1024 * 127, +512 rounds to 127 after shift.
        assert_eq!(luminance(&[127, 127, 127, 255]), 127);
    }

    #[test]
    fn invalid_png_is_a_decode_error() {
        let mut sw = Stopwatch::default();
        let err = prepare(b"definitely not a png", &settings(None), &mut sw);
        assert!(err.is_err());
    }

    #[test]
    fn non_rgba_pngs_are_converted_to_rgba() {
        // RGB8 source takes the `to_rgba8` conversion branch.
        let mut img = image::ImageBuffer::from_pixel(4, 4, image::Rgb([255, 255, 255]));
        img.put_pixel(1, 1, image::Rgb([0, 0, 0]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encodes");
        let mut sw = Stopwatch::default();
        let prepared = prepare(&png, &settings(None), &mut sw).expect("prepares");
        assert_eq!(prepared.image.pixels.len(), 4 * 4 * 4);
        // The dark pixel keeps its channels after conversion.
        assert_eq!(&prepared.image.pixels[20..24], &[0, 0, 0, 255]);
    }

    #[test]
    fn fixture_has_black_centre_and_white_corners() {
        let img = fixture_rgba(8);
        assert_eq!(*img.get_pixel(0, 0), Rgba([255, 255, 255, 255]));
        assert_eq!(*img.get_pixel(4, 4), Rgba([0, 0, 0, 255]));
        assert_ne!(fixture_png(8).len(), 0);
    }

    // ---- M0: alpha is first-class --------------------------------------

    #[test]
    fn bw_mode_composites_transparent_pixels_over_white() {
        // §2.2.1: transparent pixels stored as RGB(0,0,0,0) must never trace
        // as black geometry; compositing over white turns them white.
        let mut rgba = vec![
            0, 0, 0, 0, // transparent "black" pixel
            255, 0, 0, 128, // semi-transparent red
        ];
        composite_on_white(&mut rgba);
        assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
        // 50% red over white = pink, with the rounding form.
        assert_eq!(&rgba[4..8], &[255, 127, 127, 255]);
    }

    #[test]
    fn bw_mode_keeps_opaque_pixels_untouched() {
        let mut rgba = vec![12, 34, 56, 255, 250, 251, 252, 255];
        composite_on_white(&mut rgba);
        assert_eq!(&rgba[0..8], &[12, 34, 56, 255, 250, 251, 252, 255]);
    }

    #[test]
    fn bw_pipeline_traces_transparent_input_as_empty_geometry() {
        // Fully transparent PNG: with the 0.1.x path this traced as a solid
        // black block; now it must produce (almost) no geometry.
        let mut img = image::ImageBuffer::from_pixel(16, 16, image::Rgba([0, 0, 0, 0]));
        img.put_pixel(8, 8, image::Rgba([0, 0, 0, 255]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encodes");
        let mut sw = Stopwatch::default();
        let prepared = prepare(&png, &settings(None), &mut sw).expect("prepares");
        // Composited: every pixel white except the opaque black dot.
        assert_eq!(&prepared.image.pixels[0..4], &[255, 255, 255, 255]);
        assert_eq!(
            &prepared.image.pixels[(8 * 16 + 8) * 4..(8 * 16 + 8) * 4 + 4],
            &[0, 0, 0, 255]
        );
    }

    // ---- M0/M7: background removal --------------------------------------

    #[test]
    fn background_colour_flood_removes_only_border_connected_regions() {
        // 8×6: white border, light-grey 1-px interior island at (4,2) —
        // outside tolerance so only connectivity matters; the island must
        // survive (not border-connected), the border must go transparent.
        let (w, h) = (8usize, 6usize);
        let mut rgba = vec![255u8; w * h * 4];
        let at = |x: usize, y: usize| (y * w + x) * 4;
        rgba[at(4, 2)..at(4, 2) + 3].copy_from_slice(&[240, 240, 240]);
        flood_remove_background(&mut rgba, w, h, [255, 255, 255], 8);
        assert_eq!(rgba[at(0, 0) + 3], 0); // border removed
        assert_eq!(rgba[at(7, 5) + 3], 0); // border removed
        assert_eq!(rgba[at(4, 2) + 3], 255); // interior island survives
    }

    #[test]
    fn background_flood_respects_tolerance() {
        let (w, h) = (4usize, 4usize);
        let mut rgba = [250u8, 249, 248, 255].repeat(w * h);
        flood_remove_background(&mut rgba, w, h, [255, 255, 255], 8);
        assert_eq!(rgba[3], 0); // |255-250| = 5 ≤ 8 → removed
        let mut rgba2 = [240u8, 240, 240, 255].repeat(w * h);
        flood_remove_background(&mut rgba2, w, h, [255, 255, 255], 8);
        assert_eq!(rgba2[3], 255); // |255-240| = 15 > 8 → kept
    }

    #[test]
    fn background_colour_pipeline_strips_border_in_color_mode() {
        let png = fixture_png(16); // white bg + black square
        let cfg = Settings {
            mode: Mode::Color,
            background: Background::Color {
                rgb: [255, 255, 255],
                tolerance: 8,
            },
            ..Settings::new()
        };
        let mut sw = Stopwatch::default();
        let prepared = prepare(&png, &cfg, &mut sw).expect("prepares");
        // The white background is now transparent; the black square is intact.
        assert_eq!(prepared.image.pixels[3], 0);
        assert_eq!(prepared.image.pixels[(8 * 16 + 8) * 4 + 3], 255);
    }
}
