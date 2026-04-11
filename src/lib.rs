//! Terminal image viewer (PNG/JPEG/TIFF) using ratatui.

mod app;
mod cli;
mod render;
#[cfg(feature = "sam")]
mod sam_layout;
mod shape;
mod theme;
mod tiff_decode;
mod tui;
mod volume;

#[cfg(feature = "sam")]
use anyhow::Context;
use anyhow::{Result, bail};
#[cfg(feature = "sam")]
use sam_rs::Sam1Session;

use crate::app::App;
use crate::cli::parse_cli_args;
use crate::shape::resolve_cli_shape;
use crate::volume::load_volume;

pub fn run() -> Result<()> {
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

    #[cfg(feature = "sam")]
    let sam_session = match (&cli.sam_encoder, &cli.sam_decoder) {
        (None, None) => None,
        (Some(enc), Some(dec)) => Some(
            Sam1Session::new(enc.as_path(), dec.as_path(), false)
                .with_context(|| "imat: failed to load SAM ONNX sessions")?,
        ),
        _ => {
            bail!("imat: --sam-encoder and --sam-decoder must be given together");
        }
    };

    #[cfg(feature = "sam")]
    let mut app = App::new(cli.path, volume, view_shape, cli_shape_locked, sam_session);
    #[cfg(not(feature = "sam"))]
    let mut app = App::new(cli.path, volume, view_shape, cli_shape_locked);
    tui::run_app(&mut app)
}
