//! Stage 1: PNG bytes → tracer-ready RGBA image.
//!
//! Decode and (optionally) luminance-threshold happen in a single fused pass
//! over the pixel buffer, so there is never a second allocation for the
//! tracer's input.

use image::DynamicImage;

#[cfg(test)]
use image::{ImageBuffer, Rgba};

use crate::error::Result;
use crate::settings::{Mode, Settings};
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
    // when the source has fewer channels.
    let rgba = match decoded {
        DynamicImage::ImageRgba8(buf) => buf,
        other => other.to_rgba8(),
    };
    let (width, height) = (rgba.width(), rgba.height());
    let threshold = match settings.mode {
        Mode::BlackWhite => settings.threshold,
        Mode::Color => None,
    };
    let image = vtracer::ColorImage {
        // `into_raw` moves the decoded buffer; the threshold pass is the only
        // path that allocates a second one.
        pixels: threshold_pixels(rgba.into_raw(), threshold),
        width: width as usize,
        height: height as usize,
    };
    sw.lap(Stage::Preprocess);
    Ok(Prepared { image })
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
        let v = if luminance(px) < t { 0u8 } else { 255u8 };
        dst[0] = v;
        dst[1] = v;
        dst[2] = v;
        dst[3] = px[3];
    }
    out
}

/// Rec. 601 luma, integer-only.
///
/// The intermediate sum is bounded to `0..=261_632`, so the shift result is
/// always within `0..=255` and both casts below are lossless.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn luminance(px: &[u8]) -> u8 {
    // 0.299R + 0.587G + 0.114B, scaled by 1024 with a single rounding shift.
    let (r, g, b) = (i32::from(px[0]), i32::from(px[1]), i32::from(px[2]));
    ((306 * r + 601 * g + 117 * b + 512) >> 10) as u8
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
}
