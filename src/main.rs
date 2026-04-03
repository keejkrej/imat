use std::{
    fs::File,
    io::{self, BufReader, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage, imageops::FilterType};
use ratatui::{
    Frame, Terminal, backend::CrosstermBackend, buffer::Buffer, layout::Rect, style::Color,
};
use tiff::{
    ColorType,
    decoder::{BufferLayoutPreference, Decoder, DecodingBuffer, DecodingResult},
    tags::Tag,
};

const ALLOWED_EXTENSIONS: &[&str] = &[".png", ".jpg", ".jpeg", ".tif", ".tiff"];
const HEADER_ROWS: u16 = 2;
const SLIDER_LABEL_COLS: usize = 4;
const BACKGROUND: [u8; 4] = [12, 12, 18, 255];
const HEADER_BG: Color = Color::Rgb(28, 28, 40);
const HEADER_FG: Color = Color::Rgb(200, 200, 220);
const HEADER_DIM: Color = Color::Rgb(120, 120, 150);
const HEADER_ERR: Color = Color::Rgb(255, 140, 140);
const SLIDER_ACTIVE: Color = Color::Rgb(220, 220, 255);
const SLIDER_THUMB: Color = Color::Rgb(255, 200, 120);

fn main() {
    if let Err(error) = try_main() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let cli = parse_cli_args(std::env::args().skip(1).collect())?;
    let volume = load_volume(&cli.path)?;
    let pixel_count = volume.buffer.len() / 4;
    if pixel_count == 0 {
        bail!("imat: decoded image is empty");
    }

    let cli_shape_locked = cli.shape_spec.is_some();
    let view_shape = if let Some(shape_spec) = &cli.shape_spec {
        resolve_cli_shape(shape_spec, pixel_count, volume.base_h, volume.base_w)?
    } else {
        vec![volume.base_h, volume.base_w]
    };

    let mut app = App::new(cli.path, volume, view_shape, cli_shape_locked);
    run_app(&mut app)
}

fn usage() {
    eprintln!("usage: imat [--shape D0,D1,...,H,W] <image.png|jpg|jpeg|tif|tiff>");
    eprintln!("  --shape: optional full row-major shape (product must equal RGBA pixel count)");
    eprintln!(
        "  multipage TIFF: one HxW plane per page (multi-sample pages split into consecutive planes); r reshape"
    );
    eprintln!(
        "  reshape grid: factors with x or ,; product <= pages; use 0 once to infer one dimension (e.g. 3x0)"
    );
    eprintln!("  up/down active axis  left/right step  r reshape  Esc flat  q quit");
}

#[derive(Debug)]
struct CliArgs {
    path: PathBuf,
    shape_spec: Option<String>,
}

fn parse_cli_args(args: Vec<String>) -> Result<CliArgs> {
    let mut shape_spec = None;
    let mut positionals = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--shape" => {
                let Some(next) = args.get(i + 1) else {
                    usage();
                    bail!("imat: --shape requires a comma-separated list of positive integers");
                };
                if next.starts_with('-') {
                    usage();
                    bail!("imat: --shape requires a comma-separated list of positive integers");
                }
                shape_spec = Some(next.clone());
                i += 2;
            }
            option if option.starts_with('-') => {
                usage();
                bail!("imat: unknown option: {option}");
            }
            value => {
                positionals.push(value.to_owned());
                i += 1;
            }
        }
    }

    if positionals.len() > 1 {
        usage();
        bail!("imat: expected one image path");
    }

    let Some(path) = positionals.into_iter().next() else {
        usage();
        bail!("imat: missing image path");
    };

    let ext = extension_of(Path::new(&path));
    if !ALLOWED_EXTENSIONS.contains(&ext.as_str()) {
        usage();
        bail!(
            "imat: unsupported extension {}",
            if ext.is_empty() { "(none)" } else { &ext }
        );
    }

    let path = PathBuf::from(path);
    if !path.exists() {
        bail!("imat: file not found: {}", path.display());
    }

    Ok(CliArgs { path, shape_spec })
}

fn extension_of(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext.to_ascii_lowercase()))
        .unwrap_or_default()
}

fn is_tiff_path(path: &Path) -> bool {
    matches!(extension_of(path).as_str(), ".tif" | ".tiff")
}

#[derive(Debug)]
struct Volume {
    buffer: Vec<u8>,
    page_count: usize,
    base_w: usize,
    base_h: usize,
}

fn load_volume(path: &Path) -> Result<Volume> {
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

#[derive(Clone, Copy)]
enum Normalization {
    Fixed { lo: f64, hi: f64 },
    Dynamic,
}

fn decoding_result_to_rgba(
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
        ColorType::RGB(bits)
        | ColorType::RGBA(bits)
        | ColorType::CMYK(bits)
        | ColorType::CMYKA(bits)
            if bits > 0 && bits <= 32 =>
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

fn parse_shape_spec(spec: &str) -> Result<Vec<usize>> {
    let normalized = spec.replace('×', "x");
    let parts = normalized.split([',', 'x']).filter_map(|part| {
        let trimmed = part.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    });
    let mut shape = Vec::new();
    for part in parts {
        let value = part.parse::<usize>().with_context(|| {
            format!("imat: shape components must be positive integers: {spec:?}")
        })?;
        if value == 0 {
            bail!("imat: shape components must be positive integers: {spec:?}");
        }
        shape.push(value);
    }
    if shape.len() < 2 {
        bail!("imat: shape must have at least two dimensions (..., H, W)");
    }
    Ok(shape)
}

fn parse_page_grid_spec(spec: &str) -> Result<Vec<usize>> {
    let normalized = spec.replace('×', "x");
    let parts = normalized.split([',', 'x']).filter_map(|part| {
        let trimmed = part.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    });
    let mut dims = Vec::new();
    let mut inferred = 0usize;
    for part in parts {
        let value = part
            .parse::<usize>()
            .with_context(|| format!("imat: invalid page grid factor: {spec:?}"))?;
        if value == 0 {
            inferred += 1;
            if inferred > 1 {
                bail!("imat: at most one 0 is allowed (infer one dimension)");
            }
        }
        dims.push(value);
    }
    if dims.is_empty() {
        bail!("imat: enter at least one dimension (e.g. 5 or 5x4x2)");
    }
    Ok(dims)
}

fn resolve_page_grid_infer(dims: &[usize], page_count: usize) -> Result<Vec<usize>> {
    if !dims.contains(&0) {
        return Ok(dims.to_vec());
    }

    let known_product = dims
        .iter()
        .copied()
        .filter(|dim| *dim != 0)
        .try_fold(1usize, checked_mul)?;

    if known_product == 0 {
        bail!("imat: invalid known product for infer");
    }

    let inferred = page_count / known_product;
    if inferred == 0 {
        bail!(
            "imat: cannot infer 0: known product {known_product} is larger than {page_count} pages"
        );
    }

    Ok(dims
        .iter()
        .map(|dim| if *dim == 0 { inferred } else { *dim })
        .collect())
}

fn resolve_cli_shape(
    spec: &str,
    pixel_count: usize,
    base_h: usize,
    base_w: usize,
) -> Result<Vec<usize>> {
    let shape = parse_shape_spec(spec)?;
    let product = shape_product(&shape)?;
    if product != pixel_count {
        bail!(
            "imat: shape product {product} does not match image pixels {pixel_count} (buffer has {pixel_count} RGBA samples)"
        );
    }

    let h = shape[shape.len() - 2];
    let w = shape[shape.len() - 1];
    if h != base_h || w != base_w {
        bail!("imat: --shape ends with {h}x{w} but image pages are {base_h}x{base_w}");
    }

    Ok(shape)
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| anyhow!("imat: shape is too large"))
}

fn shape_product(shape: &[usize]) -> Result<usize> {
    shape.iter().copied().try_fold(1usize, checked_mul)
}

fn lead_strides(shape: &[usize]) -> Vec<usize> {
    let lead_dims = shape.len().saturating_sub(2);
    let mut strides = vec![0usize; lead_dims];
    for axis in 0..lead_dims {
        let mut product = 1usize;
        for dim in &shape[axis + 1..] {
            product *= *dim;
        }
        strides[axis] = product;
    }
    strides
}

fn linear_pixel_offset(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides)
        .map(|(index, stride)| index * stride)
        .sum()
}

struct App {
    path: PathBuf,
    volume: Vec<u8>,
    tiff_page_count: usize,
    base_w: usize,
    base_h: usize,
    cli_shape_locked: bool,
    view_shape: Vec<usize>,
    strides: Vec<usize>,
    indices: Vec<usize>,
    active_axis: usize,
    flat_page_index: usize,
    reshape_entry_active: bool,
    reshape_draft: String,
    reshape_error: Option<String>,
}

impl App {
    fn new(path: PathBuf, volume: Volume, view_shape: Vec<usize>, cli_shape_locked: bool) -> Self {
        Self {
            path,
            volume: volume.buffer,
            tiff_page_count: volume.page_count,
            base_w: volume.base_w,
            base_h: volume.base_h,
            cli_shape_locked,
            strides: lead_strides(&view_shape),
            indices: vec![0; view_shape.len().saturating_sub(2)],
            view_shape,
            active_axis: 0,
            flat_page_index: 0,
            reshape_entry_active: false,
            reshape_draft: String::new(),
            reshape_error: None,
        }
    }

    fn page_pixels(&self) -> usize {
        self.base_w * self.base_h
    }

    fn is_flat_multipage(&self) -> bool {
        self.tiff_page_count > 1 && self.view_shape.len() == 2 && !self.cli_shape_locked
    }

    fn slider_row_count(&self) -> usize {
        if self.reshape_entry_active {
            0
        } else if self.is_flat_multipage() {
            1
        } else {
            self.indices.len()
        }
    }

    fn current_offset_pixels(&self) -> Result<usize> {
        if self.is_flat_multipage() {
            self.flat_page_index
                .checked_mul(self.page_pixels())
                .ok_or_else(|| anyhow!("imat: page offset overflow"))
        } else {
            Ok(linear_pixel_offset(&self.indices, &self.strides))
        }
    }

    fn grid_page_linear(&self) -> usize {
        if self.page_pixels() == 0 {
            0
        } else {
            linear_pixel_offset(&self.indices, &self.strides) / self.page_pixels()
        }
    }

    fn rebuild_slice(&self) -> Result<RgbaImage> {
        let height = self.view_shape[self.view_shape.len() - 2];
        let width = self.view_shape[self.view_shape.len() - 1];
        let offset_pixels = self.current_offset_pixels()?;
        let slice_len = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| anyhow!("imat: slice length overflow"))?;
        let offset_bytes = offset_pixels
            .checked_mul(4)
            .ok_or_else(|| anyhow!("imat: slice byte offset overflow"))?;
        let slice = self
            .volume
            .get(offset_bytes..offset_bytes + slice_len)
            .ok_or_else(|| anyhow!("imat: slice out of bounds"))?;
        ImageBuffer::<Rgba<u8>, _>::from_raw(width as u32, height as u32, slice.to_vec())
            .ok_or_else(|| anyhow!("imat: failed to build image slice"))
    }

    fn reset_to_flat(&mut self) {
        let page = self
            .grid_page_linear()
            .min(self.tiff_page_count.saturating_sub(1));
        self.view_shape = vec![self.base_h, self.base_w];
        self.strides = lead_strides(&self.view_shape);
        self.indices.clear();
        self.active_axis = 0;
        self.flat_page_index = page;
    }

    fn apply_page_grid(&mut self) -> Result<()> {
        let dims = resolve_page_grid_infer(
            &parse_page_grid_spec(self.reshape_draft.trim())?,
            self.tiff_page_count,
        )?;
        let product = shape_product(&dims)?;
        if product > self.tiff_page_count {
            bail!("imat: product {product} > {} pages", self.tiff_page_count);
        }
        self.view_shape = dims;
        self.view_shape.push(self.base_h);
        self.view_shape.push(self.base_w);
        self.strides = lead_strides(&self.view_shape);
        self.indices = vec![0; self.view_shape.len().saturating_sub(2)];
        self.active_axis = 0;
        self.reshape_entry_active = false;
        self.reshape_draft.clear();
        self.reshape_error = None;
        Ok(())
    }

    fn on_key(&mut self, key: KeyEvent) -> Result<bool> {
        if key.kind != KeyEventKind::Press {
            return Ok(false);
        }

        if key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER)
        {
            return Ok(false);
        }

        if self.reshape_entry_active {
            self.on_reshape_key(key)?;
            return Ok(false);
        }

        match key.code {
            KeyCode::Char('q') => Ok(true),
            KeyCode::Esc
                if self.tiff_page_count > 1
                    && !self.cli_shape_locked
                    && self.view_shape.len() > 2 =>
            {
                self.reset_to_flat();
                Ok(false)
            }
            KeyCode::Char('r') if self.tiff_page_count > 1 && !self.cli_shape_locked => {
                self.reshape_entry_active = true;
                self.reshape_draft.clear();
                self.reshape_error = None;
                Ok(false)
            }
            KeyCode::Up if self.indices.len() > 1 => {
                self.active_axis = (self.active_axis + self.indices.len() - 1) % self.indices.len();
                Ok(false)
            }
            KeyCode::Down if self.indices.len() > 1 => {
                self.active_axis = (self.active_axis + 1) % self.indices.len();
                Ok(false)
            }
            KeyCode::Left => {
                if self.is_flat_multipage() {
                    self.flat_page_index =
                        (self.flat_page_index + self.tiff_page_count - 1) % self.tiff_page_count;
                } else if !self.indices.is_empty() {
                    let size = self.view_shape[self.active_axis];
                    self.indices[self.active_axis] =
                        (self.indices[self.active_axis] + size - 1) % size;
                }
                Ok(false)
            }
            KeyCode::Right => {
                if self.is_flat_multipage() {
                    self.flat_page_index = (self.flat_page_index + 1) % self.tiff_page_count;
                } else if !self.indices.is_empty() {
                    let size = self.view_shape[self.active_axis];
                    self.indices[self.active_axis] = (self.indices[self.active_axis] + 1) % size;
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    fn on_reshape_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Esc => {
                self.reshape_entry_active = false;
                self.reshape_draft.clear();
                self.reshape_error = None;
            }
            KeyCode::Enter => match self.apply_page_grid() {
                Ok(()) => {}
                Err(error) => {
                    self.reshape_error = Some(error.to_string().replace("imat: ", ""));
                }
            },
            KeyCode::Backspace => {
                self.reshape_draft.pop();
                self.reshape_error = None;
            }
            KeyCode::Char(',') => {
                self.reshape_draft.push(',');
                self.reshape_error = None;
            }
            KeyCode::Char('x') | KeyCode::Char('X') => {
                self.reshape_draft.push('x');
                self.reshape_error = None;
            }
            KeyCode::Char(char) if char.is_ascii_digit() => {
                self.reshape_draft.push(char);
                self.reshape_error = None;
            }
            _ => {}
        }
        Ok(())
    }
}

fn run_app(app: &mut App) -> Result<()> {
    enable_raw_mode().context("imat: failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("imat: failed to switch to alternate screen")?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("imat: failed to initialize terminal")?;
    terminal
        .hide_cursor()
        .context("imat: failed to hide cursor")?;

    let result = run_event_loop(&mut terminal, app);

    disable_raw_mode().ok();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    let _ = terminal.backend_mut().flush();

    result
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        terminal.draw(|frame| render(frame, app))?;

        if !event::poll(Duration::from_millis(250))
            .context("imat: failed while waiting for input")?
        {
            continue;
        }

        match event::read().context("imat: failed to read input event")? {
            Event::Key(key) => {
                if app.on_key(key)? {
                    break;
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    Ok(())
}

fn render(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let buffer = frame.buffer_mut();
    fill_area(
        buffer,
        area,
        ' ',
        Color::Reset,
        Color::Rgb(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]),
    );

    let slider_rows = app.slider_row_count() as u16;
    let top_rows = HEADER_ROWS + slider_rows;
    let header_height = top_rows.min(area.height);
    if header_height > 0 {
        let header_area = Rect::new(area.x, area.y, area.width, header_height);
        fill_area(buffer, header_area, ' ', HEADER_FG, HEADER_BG);
    }

    if area.height > 0 {
        let (line0, line1, line0_color) = header_lines(app);
        set_string(
            buffer,
            area.x,
            area.y,
            area.width,
            &line0,
            line0_color,
            HEADER_BG,
        );
        if area.height > 1 {
            set_string(
                buffer,
                area.x,
                area.y + 1,
                area.width,
                &line1,
                HEADER_DIM,
                HEADER_BG,
            );
        }
    }

    if !app.reshape_entry_active {
        if app.is_flat_multipage() && area.height > HEADER_ROWS {
            draw_slider_row(
                buffer,
                area.x,
                area.y + HEADER_ROWS,
                area.width,
                "p",
                app.flat_page_index,
                app.tiff_page_count,
                true,
            );
        } else if !app.indices.is_empty() {
            for (axis, index) in app.indices.iter().enumerate() {
                let row = area.y + HEADER_ROWS + axis as u16;
                if row >= area.y + area.height {
                    break;
                }
                draw_slider_row(
                    buffer,
                    area.x,
                    row,
                    area.width,
                    &axis.to_string(),
                    *index,
                    app.view_shape[axis],
                    axis == app.active_axis,
                );
            }
        }
    }

    if area.height <= top_rows {
        return;
    }
    let image_area = Rect::new(
        area.x,
        area.y + top_rows,
        area.width,
        area.height - top_rows,
    );
    let _ = render_image(buffer, image_area, app);
}

fn header_lines(app: &App) -> (String, String, Color) {
    let height = app.view_shape[app.view_shape.len() - 2];
    let width = app.view_shape[app.view_shape.len() - 1];

    if app.reshape_entry_active {
        let mut line0 = format!(
            "[reshape pages i x j x ... <= {}  (0 = infer)]  {}_",
            app.tiff_page_count, app.reshape_draft
        );
        if let Some(error) = &app.reshape_error {
            line0.push_str("  ");
            line0.push_str(error);
        }
        return (
            line0,
            "Enter apply  Esc cancel  backspace".to_owned(),
            if app.reshape_error.is_some() {
                HEADER_ERR
            } else {
                HEADER_FG
            },
        );
    }

    let line0 = if app.is_flat_multipage() {
        format!(
            "{}  flat {}p  page {}/{}  {}x{}px",
            app.path.display(),
            app.tiff_page_count,
            app.flat_page_index + 1,
            app.tiff_page_count,
            width,
            height
        )
    } else {
        let mut line = format!(
            "{}  {}  {}x{}px",
            app.path.display(),
            join_shape(&app.view_shape),
            width,
            height
        );
        if !app.indices.is_empty() {
            line.push_str(&format!(
                "  [{}]  dim{}  map->p{}",
                app.indices
                    .iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                app.active_axis,
                app.grid_page_linear()
            ));
        }
        line
    };

    let mut help = vec!["q quit".to_owned()];
    if app.tiff_page_count > 1 && !app.cli_shape_locked {
        help.push("r reshape".to_owned());
        if app.view_shape.len() > 2 {
            help.push("Esc flat".to_owned());
        }
    }
    if app.indices.len() > 1 {
        help.push("up/down dim".to_owned());
    }
    if !app.indices.is_empty() || app.is_flat_multipage() {
        help.push("left/right".to_owned());
    }

    (line0, help.join("  "), HEADER_FG)
}

fn join_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn fill_area(buffer: &mut Buffer, area: Rect, ch: char, fg: Color, bg: Color) {
    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_char(ch).set_fg(fg).set_bg(bg);
            }
        }
    }
}

fn set_string(buffer: &mut Buffer, x: u16, y: u16, width: u16, text: &str, fg: Color, bg: Color) {
    let truncated = truncate(text, width as usize);
    for (offset, ch) in truncated.chars().enumerate() {
        if let Some(cell) = buffer.cell_mut((x + offset as u16, y)) {
            cell.set_char(ch).set_fg(fg).set_bg(bg);
        }
    }
}

fn truncate(text: &str, max_cells: usize) -> String {
    let count = text.chars().count();
    if count <= max_cells {
        return text.to_owned();
    }
    if max_cells == 0 {
        return String::new();
    }
    if max_cells == 1 {
        return ".".to_owned();
    }
    let mut out = text.chars().take(max_cells - 1).collect::<String>();
    out.push('.');
    out
}

fn draw_slider_row(
    buffer: &mut Buffer,
    x: u16,
    y: u16,
    width: u16,
    label: &str,
    value_index: usize,
    value_count: usize,
    is_active: bool,
) {
    if width == 0 {
        return;
    }

    let label_color = if is_active { SLIDER_ACTIVE } else { HEADER_DIM };
    let mut label_text = label.to_owned();
    while label_text.chars().count() < SLIDER_LABEL_COLS {
        label_text.push(' ');
    }
    label_text = label_text.chars().take(SLIDER_LABEL_COLS).collect();
    set_string(
        buffer,
        x,
        y,
        width.min(SLIDER_LABEL_COLS as u16),
        &label_text,
        label_color,
        HEADER_BG,
    );

    let track_width = width.saturating_sub(SLIDER_LABEL_COLS as u16) as usize;
    if track_width == 0 {
        return;
    }

    let thumb = if value_count <= 1 {
        track_width / 2
    } else {
        value_index.saturating_mul(track_width.saturating_sub(1)) / (value_count - 1)
    };

    for offset in 0..track_width {
        let ch = if offset == thumb {
            'o'
        } else if offset < thumb {
            '-'
        } else {
            '.'
        };
        if let Some(cell) = buffer.cell_mut((x + SLIDER_LABEL_COLS as u16 + offset as u16, y)) {
            cell.set_char(ch).set_fg(SLIDER_THUMB).set_bg(HEADER_BG);
        }
    }
}

fn render_image(buffer: &mut Buffer, area: Rect, app: &App) -> Result<()> {
    if area.width == 0 || area.height == 0 {
        return Ok(());
    }

    let available_w = area.width as u32;
    let available_h = (area.height as u32).saturating_mul(2);
    if available_w == 0 || available_h == 0 {
        return Ok(());
    }

    let slice = app.rebuild_slice()?;
    let scaled = DynamicImage::ImageRgba8(slice)
        .resize(available_w, available_h, FilterType::Triangle)
        .to_rgba8();

    let mut canvas = vec![BACKGROUND; (available_w as usize) * (available_h as usize)];
    let offset_x = ((available_w - scaled.width()) / 2) as usize;
    let offset_y = ((available_h - scaled.height()) / 2) as usize;

    for y in 0..scaled.height() as usize {
        for x in 0..scaled.width() as usize {
            let pixel = scaled.get_pixel(x as u32, y as u32).0;
            let dest_index = (offset_y + y) * available_w as usize + offset_x + x;
            canvas[dest_index] = blend_over(pixel, canvas[dest_index]);
        }
    }

    for cell_y in 0..area.height as usize {
        for cell_x in 0..area.width as usize {
            let top = canvas[(cell_y * 2) * available_w as usize + cell_x];
            let bottom = canvas[(cell_y * 2 + 1) * available_w as usize + cell_x];
            if let Some(cell) = buffer.cell_mut((area.x + cell_x as u16, area.y + cell_y as u16)) {
                cell.set_char('▀')
                    .set_fg(Color::Rgb(top[0], top[1], top[2]))
                    .set_bg(Color::Rgb(bottom[0], bottom[1], bottom[2]));
            }
        }
    }

    Ok(())
}

fn blend_over(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let alpha = src[3] as u16;
    if alpha == 255 {
        return src;
    }
    if alpha == 0 {
        return dst;
    }

    let inv_alpha = 255u16 - alpha;
    let blend = |src_channel: u8, dst_channel: u8| -> u8 {
        (((src_channel as u16 * alpha) + (dst_channel as u16 * inv_alpha)) / 255) as u8
    };

    [
        blend(src[0], dst[0]),
        blend(src[1], dst[1]),
        blend(src[2], dst[2]),
        255,
    ]
}
