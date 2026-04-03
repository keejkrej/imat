# imat-rs

`imat-rs` is a Rust + `ratatui` port of the sibling `../imat` terminal image viewer.

It opens PNG, JPEG, and TIFF images directly in the terminal, including multipage TIFF stacks with reshape and axis navigation.

## Run

```bash
cargo run -- <path/to/image.png>
cargo run -- <path/to/image.tif>
```

## Controls

```text
q            quit
r            reshape multipage TIFF pages
Esc          return to flat page mode
Left/Right   move within the current page axis or page list
Up/Down      change the active reshape axis
```

## Shape

```bash
cargo run -- --shape 3,8,512,512 image.tif
```

`--shape` expects a full row-major shape ending in `H,W`, and the product must equal the total RGBA pixel count.
