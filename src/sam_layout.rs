use image::{DynamicImage, RgbaImage, imageops::FilterType};
use ratatui::layout::Rect;

pub(crate) struct HalfBlockLayout {
    pub(crate) scaled_w: u32,
    pub(crate) scaled_h: u32,
    pub(crate) off_x: u32,
    pub(crate) off_y: u32,
    pub(crate) slice_w: u32,
    pub(crate) slice_h: u32,
}

pub(crate) fn halfblock_layout_for_slice(slice: &RgbaImage, area: Rect) -> Option<HalfBlockLayout> {
    if area.width == 0 || area.height == 0 {
        return None;
    }
    let avail_w = area.width as u32;
    let avail_h = area.height.saturating_mul(2) as u32;
    let slice_w = slice.width();
    let slice_h = slice.height();
    let scaled = DynamicImage::ImageRgba8(slice.clone())
        .resize(avail_w, avail_h, FilterType::Triangle)
        .to_rgba8();
    let off_x = avail_w.saturating_sub(scaled.width()) / 2;
    let off_y = avail_h.saturating_sub(scaled.height()) / 2;
    Some(HalfBlockLayout {
        scaled_w: scaled.width(),
        scaled_h: scaled.height(),
        off_x,
        off_y,
        slice_w,
        slice_h,
    })
}

pub(crate) fn canvas_px_to_slice(layout: &HalfBlockLayout, px: u32, py: u32) -> (usize, usize) {
    let sx = px
        .saturating_sub(layout.off_x)
        .min(layout.scaled_w.saturating_sub(1));
    let sy = py
        .saturating_sub(layout.off_y)
        .min(layout.scaled_h.saturating_sub(1));
    let sw = layout.scaled_w.max(1);
    let sh = layout.scaled_h.max(1);
    let ox = (sx as u64 * layout.slice_w as u64 / sw as u64) as usize;
    let oy = (sy as u64 * layout.slice_h as u64 / sh as u64) as usize;
    (
        ox.min(layout.slice_w.saturating_sub(1) as usize),
        oy.min(layout.slice_h.saturating_sub(1) as usize),
    )
}

pub(crate) fn tint_rgb_channel(rgb: [u8; 3], tint: [u8; 3], alpha: f32) -> [u8; 3] {
    let a = alpha.clamp(0.0, 1.0);
    let blend = |c: u8, t: u8| -> u8 { ((c as f32 * (1.0 - a)) + (t as f32 * a)).round() as u8 };
    [
        blend(rgb[0], tint[0]),
        blend(rgb[1], tint[1]),
        blend(rgb[2], tint[2]),
    ]
}
