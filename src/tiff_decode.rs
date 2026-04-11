use anyhow::{Result, anyhow, bail};
use tiff::{
    ColorType,
    decoder::{BufferLayoutPreference, DecodingBuffer, DecodingResult},
};

#[derive(Clone, Copy)]
enum Normalization {
    Fixed { lo: f64, hi: f64 },
    Dynamic,
}

pub(crate) fn decoding_result_to_rgba(
    data: &mut DecodingResult,
    layout: &BufferLayoutPreference,
    color_type: ColorType,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let pixel_count = (width as usize) * (height as usize);
    let samples = color_type.num_samples() as usize;
    let plane_stride_elems = if layout.planes > 1 {
        let plane_stride = layout
            .plane_stride
            .ok_or_else(|| anyhow!("imat: missing TIFF plane stride"))?
            .get();
        Some(plane_stride)
    } else {
        None
    };

    let normalization = normalization_for(color_type);
    let channels = match data.as_buffer(0) {
        DecodingBuffer::U8(buf) => {
            if color_type.bit_depth() != 8 {
                bail!(
                    "imat: TIFF bit depth {} is not supported yet",
                    color_type.bit_depth()
                );
            }
            normalize_channels(
                buf,
                pixel_count,
                samples,
                plane_stride_elems.map(|stride| stride / std::mem::size_of::<u8>()),
                normalization,
                |value| value as f64,
            )?
        }
        DecodingBuffer::U16(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<u16>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::U32(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<u32>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::U64(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<u64>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::I8(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<i8>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::I16(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<i16>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::I32(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<i32>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::I64(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<i64>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::F16(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of_val(&buf[0])),
            normalization,
            |value| f32::from(value) as f64,
        )?,
        DecodingBuffer::F32(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<f32>()),
            normalization,
            |value| value as f64,
        )?,
        DecodingBuffer::F64(buf) => normalize_channels(
            buf,
            pixel_count,
            samples,
            plane_stride_elems.map(|stride| stride / std::mem::size_of::<f64>()),
            normalization,
            |value| value,
        )?,
    };

    pack_rgba(&channels, color_type, pixel_count)
}

fn normalization_for(color_type: ColorType) -> Normalization {
    match color_type {
        ColorType::Gray(8) | ColorType::GrayA(8) | ColorType::RGB(8) | ColorType::RGBA(8) => {
            Normalization::Fixed { lo: 0.0, hi: 255.0 }
        }
        ColorType::CMYK(8) | ColorType::CMYKA(8) | ColorType::YCbCr(8) | ColorType::Palette(8) => {
            Normalization::Fixed { lo: 0.0, hi: 255.0 }
        }
        // High bit-depth RGB family: fixed full-scale maps scientific stacks (values in a small
        // subrange of 16-bit) to black; match typical viewer behavior (min–max per plane).
        ColorType::RGB(bits)
        | ColorType::RGBA(bits)
        | ColorType::CMYK(bits)
        | ColorType::CMYKA(bits)
            if bits > 8 && bits <= 32 =>
        {
            Normalization::Dynamic
        }
        ColorType::RGB(bits)
        | ColorType::RGBA(bits)
        | ColorType::CMYK(bits)
        | ColorType::CMYKA(bits)
            if bits > 0 && bits <= 8 =>
        {
            Normalization::Fixed {
                lo: 0.0,
                hi: ((1u128 << bits.min(32)) - 1) as f64,
            }
        }
        _ => Normalization::Dynamic,
    }
}

fn normalize_channels<T, F>(
    values: &[T],
    pixel_count: usize,
    samples: usize,
    plane_stride_elems: Option<usize>,
    normalization: Normalization,
    to_f64: F,
) -> Result<Vec<Vec<u8>>>
where
    T: Copy,
    F: Fn(T) -> f64 + Copy,
{
    let mut min_values = vec![f64::INFINITY; samples];
    let mut max_values = vec![f64::NEG_INFINITY; samples];

    if let Normalization::Dynamic = normalization {
        for pixel in 0..pixel_count {
            for sample in 0..samples {
                let value = sample_value(
                    values,
                    pixel_count,
                    samples,
                    plane_stride_elems,
                    pixel,
                    sample,
                    to_f64,
                )?;
                min_values[sample] = min_values[sample].min(value);
                max_values[sample] = max_values[sample].max(value);
            }
        }
    } else if let Normalization::Fixed { lo, hi } = normalization {
        for sample in 0..samples {
            min_values[sample] = lo;
            max_values[sample] = hi;
        }
    }

    let mut channels = vec![vec![0u8; pixel_count]; samples];
    for pixel in 0..pixel_count {
        for sample in 0..samples {
            let value = sample_value(
                values,
                pixel_count,
                samples,
                plane_stride_elems,
                pixel,
                sample,
                to_f64,
            )?;
            channels[sample][pixel] = scale_to_u8(value, min_values[sample], max_values[sample]);
        }
    }

    Ok(channels)
}

fn sample_value<T, F>(
    values: &[T],
    _pixel_count: usize,
    samples: usize,
    plane_stride_elems: Option<usize>,
    pixel: usize,
    sample: usize,
    to_f64: F,
) -> Result<f64>
where
    T: Copy,
    F: Fn(T) -> f64,
{
    let index = if let Some(stride) = plane_stride_elems {
        sample
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(pixel))
            .ok_or_else(|| anyhow!("imat: TIFF plane index overflow"))?
    } else {
        pixel
            .checked_mul(samples)
            .and_then(|offset| offset.checked_add(sample))
            .ok_or_else(|| anyhow!("imat: TIFF sample index overflow"))?
    };
    let value = values
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("imat: decoded TIFF buffer was smaller than expected"))?;
    Ok(to_f64(value))
}

fn scale_to_u8(value: f64, min: f64, max: f64) -> u8 {
    if !value.is_finite() || !min.is_finite() || !max.is_finite() {
        return 0;
    }
    if (max - min).abs() < f64::EPSILON {
        return if value > 0.0 { 255 } else { 0 };
    }
    let scaled = ((value - min) / (max - min) * 255.0).round();
    scaled.clamp(0.0, 255.0) as u8
}

fn pack_rgba(channels: &[Vec<u8>], color_type: ColorType, pixel_count: usize) -> Result<Vec<u8>> {
    let mut rgba = vec![0u8; pixel_count * 4];
    for pixel in 0..pixel_count {
        let out = &mut rgba[pixel * 4..pixel * 4 + 4];
        match color_type {
            ColorType::Gray(_) | ColorType::Palette(_) => {
                let gray = channels[0][pixel];
                out.copy_from_slice(&[gray, gray, gray, 255]);
            }
            ColorType::GrayA(_) => {
                let gray = channels[0][pixel];
                let alpha = channels.get(1).map(|channel| channel[pixel]).unwrap_or(255);
                out.copy_from_slice(&[gray, gray, gray, alpha]);
            }
            ColorType::RGB(_) => {
                out.copy_from_slice(&[
                    channels[0][pixel],
                    channels[1][pixel],
                    channels[2][pixel],
                    255,
                ]);
            }
            ColorType::RGBA(_) => {
                out.copy_from_slice(&[
                    channels[0][pixel],
                    channels[1][pixel],
                    channels[2][pixel],
                    channels[3][pixel],
                ]);
            }
            ColorType::CMYK(_) => {
                let [r, g, b] = cmyk_to_rgb(
                    channels[0][pixel],
                    channels[1][pixel],
                    channels[2][pixel],
                    channels[3][pixel],
                );
                out.copy_from_slice(&[r, g, b, 255]);
            }
            ColorType::CMYKA(_) => {
                let [r, g, b] = cmyk_to_rgb(
                    channels[0][pixel],
                    channels[1][pixel],
                    channels[2][pixel],
                    channels[3][pixel],
                );
                let alpha = channels.get(4).map(|channel| channel[pixel]).unwrap_or(255);
                out.copy_from_slice(&[r, g, b, alpha]);
            }
            ColorType::YCbCr(_) => {
                let [r, g, b] =
                    ycbcr_to_rgb(channels[0][pixel], channels[1][pixel], channels[2][pixel]);
                out.copy_from_slice(&[r, g, b, 255]);
            }
            ColorType::Multiband { .. } | ColorType::Lab(_) => {
                let r = channels[0][pixel];
                let g = channels.get(1).map(|channel| channel[pixel]).unwrap_or(r);
                let b = channels.get(2).map(|channel| channel[pixel]).unwrap_or(r);
                let a = channels.get(3).map(|channel| channel[pixel]).unwrap_or(255);
                out.copy_from_slice(&[r, g, b, a]);
            }
            _ => bail!("imat: unsupported TIFF color type {color_type:?}"),
        }
    }
    Ok(rgba)
}

fn cmyk_to_rgb(c: u8, m: u8, y: u8, k: u8) -> [u8; 3] {
    let c = c as f32 / 255.0;
    let m = m as f32 / 255.0;
    let y = y as f32 / 255.0;
    let k = k as f32 / 255.0;
    let r = 255.0 * (1.0 - c) * (1.0 - k);
    let g = 255.0 * (1.0 - m) * (1.0 - k);
    let b = 255.0 * (1.0 - y) * (1.0 - k);
    [r.round() as u8, g.round() as u8, b.round() as u8]
}

fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = y as f32;
    let cb = cb as f32 - 128.0;
    let cr = cr as f32 - 128.0;
    let r = (y + 1.402 * cr).clamp(0.0, 255.0);
    let g = (y - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 255.0);
    let b = (y + 1.772 * cb).clamp(0.0, 255.0);
    [r.round() as u8, g.round() as u8, b.round() as u8]
}
