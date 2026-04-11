use std::{fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result, bail};
use tiff::{
    decoder::{Decoder, DecodingResult},
    tags::Tag,
};

use crate::tiff_decode::decoding_result_to_rgba;

#[derive(Debug)]
pub(crate) struct Volume {
    pub(crate) buffer: Vec<u8>,
    pub(crate) page_count: usize,
    pub(crate) base_w: usize,
    pub(crate) base_h: usize,
}

pub(crate) fn extension_of(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext.to_ascii_lowercase()))
        .unwrap_or_default()
}

fn is_tiff_path(path: &Path) -> bool {
    matches!(extension_of(path).as_str(), ".tif" | ".tiff")
}

pub(crate) fn load_volume(path: &Path) -> Result<Volume> {
    if is_tiff_path(path) {
        load_tiff_volume(path)
    } else {
        load_raster_volume(path)
    }
}

fn load_raster_volume(path: &Path) -> Result<Volume> {
    let image =
        image::open(path).with_context(|| format!("imat: failed to decode {}", path.display()))?;
    let rgba = image.to_rgba8();
    Ok(Volume {
        buffer: rgba.into_raw(),
        page_count: 1,
        base_w: image.width() as usize,
        base_h: image.height() as usize,
    })
}

fn load_tiff_volume(path: &Path) -> Result<Volume> {
    let file =
        File::open(path).with_context(|| format!("imat: failed to open {}", path.display()))?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .with_context(|| format!("imat: failed to read TIFF headers from {}", path.display()))?;

    let mut data = DecodingResult::I8(Vec::new());
    let mut base_dims: Option<(u32, u32)> = None;
    let mut logical_pages = 0usize;
    let mut volume = Vec::new();

    loop {
        let color_type = decoder
            .colortype()
            .context("imat: failed to read TIFF color type")?;
        let (width, height) = decoder
            .dimensions()
            .context("imat: failed to read TIFF dimensions")?;
        let samples_per_pixel = decoder
            .find_tag_unsigned::<u16>(Tag::SamplesPerPixel)
            .context("imat: failed to read TIFF SamplesPerPixel")?
            .unwrap_or(color_type.num_samples()) as usize;

        if !(1..=4).contains(&samples_per_pixel) {
            bail!("imat: TIFF Samples/Pixel {samples_per_pixel} > 4 not supported");
        }

        if let Some((expected_w, expected_h)) = base_dims {
            if width != expected_w || height != expected_h {
                bail!(
                    "imat: TIFF page is {width}x{height}px; page 0 is {expected_w}x{expected_h}px (sizes must match)"
                );
            }
        } else {
            base_dims = Some((width, height));
        }

        let layout = decoder
            .read_image_to_buffer(&mut data)
            .context("imat: failed to decode TIFF page")?;
        if data.as_buffer(0).byte_len() < layout.complete_len {
            bail!("imat: TIFF planar data did not fit into the decoder buffer");
        }

        let rgba = decoding_result_to_rgba(&mut data, &layout, color_type, width, height)?;
        let page_bytes = (width as usize) * (height as usize) * 4;
        if rgba.len() != page_bytes {
            bail!(
                "imat: TIFF page decoded to {} bytes, expected {}",
                rgba.len(),
                page_bytes
            );
        }

        if samples_per_pixel == 1 {
            volume.extend_from_slice(&rgba);
            logical_pages += 1;
        } else {
            let pixels = (width as usize) * (height as usize);
            for channel in 0..samples_per_pixel {
                for pixel in 0..pixels {
                    let value = rgba[pixel * 4 + channel];
                    volume.extend_from_slice(&[value, value, value, 255]);
                }
                logical_pages += 1;
            }
        }

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .context("imat: failed to advance to next TIFF page")?;
    }

    let Some((base_w, base_h)) = base_dims else {
        bail!("imat: no page found in TIFF");
    };

    Ok(Volume {
        buffer: volume,
        page_count: logical_pages,
        base_w: base_w as usize,
        base_h: base_h as usize,
    })
}
