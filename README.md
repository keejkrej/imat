# imat-rs

`imat-rs` is a Rust + `ratatui` port of the sibling `../imat` terminal image viewer.

It opens PNG, JPEG, and TIFF images directly in the terminal, including multipage TIFF stacks with reshape and axis navigation.

## Run

```bash
cargo run -- <path/to/image.png>
cargo run -- <path/to/image.tif>
```

## SAM segmentation (optional)

Build with the `sam` feature and pass ONNX models exported from [`keejkrej/sam-rs`](https://github.com/keejkrej/sam-rs) (`image_encoder.onnx` + `mask_decoder.onnx`, Meta SAM1-style, `return_single_mask=true` on the decoder).

```bash
cargo run --features sam -- \
  --sam-encoder /path/to/image_encoder.onnx \
  --sam-decoder /path/to/mask_decoder.onnx \
  image.png
```

In the viewer: `s` toggles SAM mode, `hjkl` moves the cursor (vim directions), `Enter` runs a foreground point prompt, `Esc` exits SAM mode. Page navigation with arrow keys is disabled while SAM mode is on. The mask is tinted green on the half-block image.

## Controls

```text
q            quit
r            reshape multipage TIFF pages
Esc          return to flat page mode
Left/Right   move within the current page axis or page list
Up/Down      change the active reshape axis
```

With `--features sam` and both `--sam-encoder` / `--sam-decoder`: `s` SAM mode, `hjkl` cursor, `Enter` prompt, `Esc` exit SAM.

## Shape

```bash
cargo run -- --shape 3,8,512,512 image.tif
```

`--shape` expects a full row-major shape ending in `H,W`, and the product must equal the total RGBA pixel count.
