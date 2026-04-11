use anyhow::{Context, Result, anyhow, bail};

pub(crate) fn parse_shape_spec(spec: &str) -> Result<Vec<usize>> {
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

pub(crate) fn parse_page_grid_spec(spec: &str) -> Result<Vec<usize>> {
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

pub(crate) fn resolve_page_grid_infer(dims: &[usize], page_count: usize) -> Result<Vec<usize>> {
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

pub(crate) fn resolve_cli_shape(
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

pub(crate) fn shape_product(shape: &[usize]) -> Result<usize> {
    shape.iter().copied().try_fold(1usize, checked_mul)
}

pub(crate) fn lead_strides(shape: &[usize]) -> Vec<usize> {
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

pub(crate) fn linear_pixel_offset(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides)
        .map(|(index, stride)| index * stride)
        .sum()
}

pub(crate) fn join_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("x")
}
