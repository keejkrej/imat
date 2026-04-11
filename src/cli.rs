use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::theme::ALLOWED_EXTENSIONS;
use crate::volume::extension_of;

#[derive(Debug)]
pub(crate) struct CliArgs {
    pub(crate) path: PathBuf,
    pub(crate) shape_spec: Option<String>,
    #[cfg(feature = "sam")]
    pub(crate) sam_encoder: Option<PathBuf>,
    #[cfg(feature = "sam")]
    pub(crate) sam_decoder: Option<PathBuf>,
}

pub(crate) fn parse_cli_args(args: Vec<String>) -> Result<CliArgs> {
    let mut shape_spec = None;
    #[cfg(feature = "sam")]
    let mut sam_encoder = None;
    #[cfg(feature = "sam")]
    let mut sam_decoder = None;
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
            #[cfg(feature = "sam")]
            "--sam-encoder" => {
                let Some(next) = args.get(i + 1) else {
                    usage();
                    bail!("imat: --sam-encoder requires a path");
                };
                if next.starts_with('-') {
                    usage();
                    bail!("imat: --sam-encoder requires a path");
                }
                sam_encoder = Some(PathBuf::from(next));
                i += 2;
            }
            #[cfg(feature = "sam")]
            "--sam-decoder" => {
                let Some(next) = args.get(i + 1) else {
                    usage();
                    bail!("imat: --sam-decoder requires a path");
                };
                if next.starts_with('-') {
                    usage();
                    bail!("imat: --sam-decoder requires a path");
                }
                sam_decoder = Some(PathBuf::from(next));
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

    Ok(CliArgs {
        path,
        shape_spec,
        #[cfg(feature = "sam")]
        sam_encoder,
        #[cfg(feature = "sam")]
        sam_decoder,
    })
}

fn usage() {
    #[cfg(feature = "sam")]
    eprintln!(
        "usage: imat [--shape D0,D1,...,H,W] [--sam-encoder P --sam-decoder P] <image.png|jpg|jpeg|tif|tiff>"
    );
    #[cfg(not(feature = "sam"))]
    eprintln!("usage: imat [--shape D0,D1,...,H,W] <image.png|jpg|jpeg|tif|tiff>");
    eprintln!("  --shape: optional full row-major shape (product must equal RGBA pixel count)");
    #[cfg(feature = "sam")]
    eprintln!(
        "  --sam-encoder / --sam-decoder: SAM1 ONNX models (requires `cargo build --features sam`)"
    );
    eprintln!(
        "  multipage TIFF: one HxW plane per page (multi-sample pages split into consecutive planes); r reshape"
    );
    eprintln!(
        "  reshape grid: factors with x or ,; product <= pages; use 0 once to infer one dimension (e.g. 3x0)"
    );
    eprintln!("  up/down active axis  left/right step  r reshape  Esc flat  q quit");
    #[cfg(feature = "sam")]
    eprintln!("  s SAM mode  hjkl cursor  Enter point  Esc exit SAM");
}
