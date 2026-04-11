use std::path::PathBuf;

#[cfg(feature = "sam")]
use anyhow::Context;
use anyhow::{Result, anyhow, bail};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use image::{ImageBuffer, Rgba, RgbaImage};
#[cfg(feature = "sam")]
use ratatui::layout::Rect;
#[cfg(feature = "sam")]
use sam_rs::Sam1Session;

#[cfg(feature = "sam")]
use crate::sam_layout::{canvas_px_to_slice, halfblock_layout_for_slice};
use crate::shape::{
    lead_strides, linear_pixel_offset, parse_page_grid_spec, resolve_page_grid_infer, shape_product,
};
use crate::volume::Volume;

pub(crate) struct App {
    pub(crate) path: PathBuf,
    pub(crate) volume: Vec<u8>,
    pub(crate) tiff_page_count: usize,
    pub(crate) base_w: usize,
    pub(crate) base_h: usize,
    pub(crate) cli_shape_locked: bool,
    pub(crate) view_shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) indices: Vec<usize>,
    pub(crate) active_axis: usize,
    pub(crate) flat_page_index: usize,
    pub(crate) reshape_entry_active: bool,
    pub(crate) reshape_draft: String,
    pub(crate) reshape_error: Option<String>,
    #[cfg(feature = "sam")]
    pub(crate) sam: Option<Sam1Session>,
    #[cfg(feature = "sam")]
    pub(crate) seg_mode: bool,
    #[cfg(feature = "sam")]
    pub(crate) seg_cursor_x: u16,
    #[cfg(feature = "sam")]
    pub(crate) seg_cursor_y: u16,
    #[cfg(feature = "sam")]
    pub(crate) last_image_area: Option<Rect>,
    #[cfg(feature = "sam")]
    sam_cache_key: Option<(usize, usize, usize)>,
    #[cfg(feature = "sam")]
    sam_embedding: Option<sam_rs::SamImageEmbedding>,
    #[cfg(feature = "sam")]
    pub(crate) sam_mask: Option<Vec<u8>>,
    #[cfg(feature = "sam")]
    pub(crate) sam_last_error: Option<String>,
}

impl App {
    #[cfg(feature = "sam")]
    pub(crate) fn new(
        path: PathBuf,
        volume: Volume,
        view_shape: Vec<usize>,
        cli_shape_locked: bool,
        sam: Option<Sam1Session>,
    ) -> Self {
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
            sam,
            seg_mode: false,
            seg_cursor_x: 0,
            seg_cursor_y: 0,
            last_image_area: None,
            sam_cache_key: None,
            sam_embedding: None,
            sam_mask: None,
            sam_last_error: None,
        }
    }

    #[cfg(not(feature = "sam"))]
    pub(crate) fn new(
        path: PathBuf,
        volume: Volume,
        view_shape: Vec<usize>,
        cli_shape_locked: bool,
    ) -> Self {
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

    #[cfg(feature = "sam")]
    fn invalidate_sam_cache(&mut self) {
        self.sam_embedding = None;
        self.sam_cache_key = None;
        self.sam_mask = None;
    }

    #[cfg(feature = "sam")]
    pub(crate) fn clamp_seg_cursor(&mut self, area: Rect) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        self.seg_cursor_x = self.seg_cursor_x.min(area.width.saturating_sub(1));
        self.seg_cursor_y = self.seg_cursor_y.min(area.height.saturating_sub(1));
    }

    #[cfg(feature = "sam")]
    pub(crate) fn ensure_sam_embedding(
        &mut self,
        rgba: &[u8],
        w: usize,
        h: usize,
        offset: usize,
    ) -> Result<()> {
        let Some(ref mut sam) = self.sam else {
            return Ok(());
        };
        let key = (offset, w, h);
        if self.sam_cache_key == Some(key) && self.sam_embedding.is_some() {
            return Ok(());
        }
        let emb = sam
            .encode_rgba(rgba, h, w)
            .map_err(|e| anyhow!("imat: SAM encode failed: {e}"))?;
        self.sam_embedding = Some(emb);
        self.sam_cache_key = Some(key);
        Ok(())
    }

    #[cfg(feature = "sam")]
    fn submit_sam_point(&mut self) -> Result<()> {
        self.sam_last_error = None;
        let Some(area) = self.last_image_area else {
            self.sam_last_error = Some("resize terminal to refresh image area".to_owned());
            return Ok(());
        };
        if self.sam.is_none() {
            return Ok(());
        }
        let slice = self.rebuild_slice()?;
        let slice_h = slice.height() as usize;
        let slice_w = slice.width() as usize;
        let offset = self.current_offset_pixels()?;
        let rgba = slice.as_raw().to_vec();
        self.ensure_sam_embedding(&rgba, slice_w, slice_h, offset)?;
        let emb = self
            .sam_embedding
            .as_ref()
            .context("imat: SAM embedding missing after encode")?;
        let layout = halfblock_layout_for_slice(&slice, area)
            .context("imat: could not compute layout for SAM prompt")?;
        let px = self.seg_cursor_x as u32;
        let py = self.seg_cursor_y as u32 * 2;
        let (msx, msy) = canvas_px_to_slice(&layout, px, py);
        let Some(mut sam) = self.sam.take() else {
            return Ok(());
        };
        let decode = sam.decode_foreground_point(emb, msx as f32 + 0.5, msy as f32 + 0.5);
        self.sam = Some(sam);
        let mask = decode.map_err(|e| anyhow!("imat: SAM decode failed: {e}"))?;
        self.sam_mask = Some(mask);
        Ok(())
    }

    fn page_pixels(&self) -> usize {
        self.base_w * self.base_h
    }

    pub(crate) fn is_flat_multipage(&self) -> bool {
        self.tiff_page_count > 1 && self.view_shape.len() == 2 && !self.cli_shape_locked
    }

    pub(crate) fn slider_row_count(&self) -> usize {
        if self.reshape_entry_active {
            0
        } else if self.is_flat_multipage() {
            1
        } else {
            self.indices.len()
        }
    }

    pub(crate) fn current_offset_pixels(&self) -> Result<usize> {
        if self.is_flat_multipage() {
            self.flat_page_index
                .checked_mul(self.page_pixels())
                .ok_or_else(|| anyhow!("imat: page offset overflow"))
        } else {
            Ok(linear_pixel_offset(&self.indices, &self.strides))
        }
    }

    pub(crate) fn grid_page_linear(&self) -> usize {
        if self.page_pixels() == 0 {
            0
        } else {
            linear_pixel_offset(&self.indices, &self.strides) / self.page_pixels()
        }
    }

    pub(crate) fn rebuild_slice(&self) -> Result<RgbaImage> {
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
        #[cfg(feature = "sam")]
        self.invalidate_sam_cache();
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
        #[cfg(feature = "sam")]
        self.invalidate_sam_cache();
        Ok(())
    }

    pub(crate) fn on_key(&mut self, key: KeyEvent) -> Result<bool> {
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

        #[cfg(feature = "sam")]
        if self.seg_mode {
            if self.sam.is_none() {
                self.seg_mode = false;
                return Ok(false);
            }
            match key.code {
                KeyCode::Esc => {
                    self.seg_mode = false;
                    self.sam_last_error = None;
                    return Ok(false);
                }
                KeyCode::Enter => {
                    if let Err(error) = self.submit_sam_point() {
                        self.sam_last_error = Some(error.to_string().replace("imat: ", ""));
                    }
                    return Ok(false);
                }
                KeyCode::Char('s') => {
                    self.seg_mode = false;
                    self.sam_last_error = None;
                    return Ok(false);
                }
                KeyCode::Char('h') => {
                    self.seg_cursor_x = self.seg_cursor_x.saturating_sub(1);
                    if let Some(a) = self.last_image_area {
                        self.clamp_seg_cursor(a);
                    }
                    return Ok(false);
                }
                KeyCode::Char('l') => {
                    self.seg_cursor_x = self.seg_cursor_x.saturating_add(1);
                    if let Some(a) = self.last_image_area {
                        self.clamp_seg_cursor(a);
                    }
                    return Ok(false);
                }
                KeyCode::Char('k') => {
                    self.seg_cursor_y = self.seg_cursor_y.saturating_sub(1);
                    if let Some(a) = self.last_image_area {
                        self.clamp_seg_cursor(a);
                    }
                    return Ok(false);
                }
                KeyCode::Char('j') => {
                    self.seg_cursor_y = self.seg_cursor_y.saturating_add(1);
                    if let Some(a) = self.last_image_area {
                        self.clamp_seg_cursor(a);
                    }
                    return Ok(false);
                }
                KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down => {
                    return Ok(false);
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Char('q') => Ok(true),
            #[cfg(feature = "sam")]
            KeyCode::Char('s') => {
                if self.sam.is_some() {
                    self.seg_mode = true;
                    self.sam_last_error = None;
                    if let Some(a) = self.last_image_area {
                        self.seg_cursor_x = a.width / 2;
                        self.seg_cursor_y = a.height / 2;
                    }
                } else {
                    self.sam_last_error = Some(
                        "SAM models not loaded (use --sam-encoder and --sam-decoder)".to_owned(),
                    );
                }
                Ok(false)
            }
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
                #[cfg(feature = "sam")]
                self.invalidate_sam_cache();
                Ok(false)
            }
            KeyCode::Right => {
                if self.is_flat_multipage() {
                    self.flat_page_index = (self.flat_page_index + 1) % self.tiff_page_count;
                } else if !self.indices.is_empty() {
                    let size = self.view_shape[self.active_axis];
                    self.indices[self.active_axis] = (self.indices[self.active_axis] + 1) % size;
                }
                #[cfg(feature = "sam")]
                self.invalidate_sam_cache();
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
