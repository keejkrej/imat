use anyhow::Result;
use image::{DynamicImage, imageops::FilterType};
#[cfg(feature = "sam")]
use ratatui::style::Modifier;
use ratatui::{Frame, buffer::Buffer, layout::Rect, style::Color};

use crate::app::App;
#[cfg(feature = "sam")]
use crate::sam_layout::{canvas_px_to_slice, halfblock_layout_for_slice, tint_rgb_channel};
use crate::shape::join_shape;
use crate::theme::{HEADER_ROWS, SLIDER_LABEL_COLS};

/// Letterbox behind scaled image (half-block compositing needs RGBA).
const CANVAS_LETTERBOX: [u8; 4] = [0, 0, 0, 255];
#[cfg(feature = "sam")]
const SAM_MASK_TINT: [u8; 3] = [0, 128, 0];

pub(crate) fn render(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    let buffer = frame.buffer_mut();
    fill_area(buffer, area, ' ', Color::Reset, Color::Reset);

    let slider_rows = app.slider_row_count() as u16;
    let top_rows = HEADER_ROWS + slider_rows;
    let header_height = top_rows.min(area.height);
    if header_height > 0 {
        let header_area = Rect::new(area.x, area.y, area.width, header_height);
        fill_area(buffer, header_area, ' ', Color::Reset, Color::Reset);
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
            Color::Reset,
        );
        if area.height > 1 {
            set_string(
                buffer,
                area.x,
                area.y + 1,
                area.width,
                &line1,
                Color::DarkGray,
                Color::Reset,
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
    #[cfg(feature = "sam")]
    {
        app.last_image_area = Some(image_area);
        if app.seg_mode {
            app.clamp_seg_cursor(image_area);
        }
    }
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
                Color::Red
            } else {
                Color::Reset
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

    #[cfg(feature = "sam")]
    let mut line0 = line0;
    #[cfg(feature = "sam")]
    if let Some(error) = &app.sam_last_error {
        line0.push_str("  ");
        line0.push_str(error);
    }

    let mut help = vec!["q quit".to_owned()];
    #[cfg(feature = "sam")]
    if app.sam.is_some() {
        help.push("s SAM".to_owned());
        if app.seg_mode {
            help.push("hjkl point Enter".to_owned());
            help.push("Esc exit SAM".to_owned());
        }
    }
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

    #[cfg(not(feature = "sam"))]
    let line0_color = Color::Reset;
    #[cfg(feature = "sam")]
    let line0_color = if app.sam_last_error.is_some() {
        Color::Red
    } else {
        Color::Reset
    };

    (line0, help.join("  "), line0_color)
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

/// Crosshair over the SAM cursor. Fewer vertical than horizontal **cells**, so the cross reads
/// ~square on screen (cells are taller than wide in pixels; previously equal cell counts looked tall).
#[cfg(feature = "sam")]
fn draw_sam_crosshair(buffer: &mut Buffer, area: Rect, cx: u16, cy: u16) {
    const HORIZONTAL_ARM: i32 = 2;
    const VERTICAL_ARM: i32 = 1;

    let w = area.width as i32;
    let h = area.height as i32;
    let cx = cx as i32;
    let cy = cy as i32;

    let plot = |buffer: &mut Buffer, x: i32, y: i32, ch: char| {
        if x < 0 || y < 0 || x >= w || y >= h {
            return;
        }
        let ax = area.x + x as u16;
        let ay = area.y + y as u16;
        if let Some(cell) = buffer.cell_mut((ax, ay)) {
            cell.set_char(ch).set_fg(Color::Red).set_bg(Color::Reset);
            cell.modifier = Modifier::BOLD;
        }
    };

    for dx in -HORIZONTAL_ARM..=HORIZONTAL_ARM {
        let ch = if dx == 0 { '┼' } else { '─' };
        plot(buffer, cx + dx, cy, ch);
    }
    for dy in -VERTICAL_ARM..=VERTICAL_ARM {
        if dy == 0 {
            continue;
        }
        plot(buffer, cx, cy + dy, '│');
    }
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

    let label_color = if is_active {
        Color::Reset
    } else {
        Color::DarkGray
    };
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
        Color::Reset,
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
            cell.set_char(ch).set_fg(Color::Yellow).set_bg(Color::Reset);
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
    let scaled = DynamicImage::ImageRgba8(slice.clone())
        .resize(available_w, available_h, FilterType::Triangle)
        .to_rgba8();

    let mut canvas = vec![CANVAS_LETTERBOX; (available_w as usize) * (available_h as usize)];
    let offset_x = ((available_w - scaled.width()) / 2) as usize;
    let offset_y = ((available_h - scaled.height()) / 2) as usize;

    for y in 0..scaled.height() as usize {
        for x in 0..scaled.width() as usize {
            let pixel = scaled.get_pixel(x as u32, y as u32).0;
            let dest_index = (offset_y + y) * available_w as usize + offset_x + x;
            canvas[dest_index] = blend_over(pixel, canvas[dest_index]);
        }
    }

    #[cfg(feature = "sam")]
    let layout = halfblock_layout_for_slice(&slice, area);

    for cell_y in 0..area.height as usize {
        for cell_x in 0..area.width as usize {
            let top = canvas[(cell_y * 2) * available_w as usize + cell_x];
            let bottom = canvas[(cell_y * 2 + 1) * available_w as usize + cell_x];

            #[cfg(feature = "sam")]
            let (top, bottom) = {
                let mut t = top;
                let mut b = bottom;
                if let (Some(layout), Some(mask)) = (layout.as_ref(), app.sam_mask.as_ref()) {
                    let slice_w = layout.slice_w as usize;
                    let (tx, ty) = canvas_px_to_slice(layout, cell_x as u32, (cell_y * 2) as u32);
                    if mask.get(ty * slice_w + tx).copied().unwrap_or(0) > 0 {
                        let rgb = tint_rgb_channel([t[0], t[1], t[2]], SAM_MASK_TINT, 0.38);
                        t = [rgb[0], rgb[1], rgb[2], 255];
                    }
                    let (bx, by) =
                        canvas_px_to_slice(layout, cell_x as u32, (cell_y * 2 + 1) as u32);
                    if mask.get(by * slice_w + bx).copied().unwrap_or(0) > 0 {
                        let rgb = tint_rgb_channel([b[0], b[1], b[2]], SAM_MASK_TINT, 0.38);
                        b = [rgb[0], rgb[1], rgb[2], 255];
                    }
                }
                (t, b)
            };

            let fg = Color::Rgb(top[0], top[1], top[2]);
            let bg = Color::Rgb(bottom[0], bottom[1], bottom[2]);
            if let Some(cell) = buffer.cell_mut((area.x + cell_x as u16, area.y + cell_y as u16)) {
                cell.set_char('▀').set_fg(fg).set_bg(bg);
            }
        }
    }

    #[cfg(feature = "sam")]
    if app.seg_mode {
        draw_sam_crosshair(buffer, area, app.seg_cursor_x, app.seg_cursor_y);
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
