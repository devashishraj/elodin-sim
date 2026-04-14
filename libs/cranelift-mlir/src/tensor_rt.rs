use std::slice;

unsafe fn f64_slices<'a>(
    dst: *mut f64,
    a: *const f64,
    b: *const f64,
    n: usize,
) -> (&'a mut [f64], &'a [f64], &'a [f64]) {
    (
        unsafe { slice::from_raw_parts_mut(dst, n) },
        unsafe { slice::from_raw_parts(a, n) },
        unsafe { slice::from_raw_parts(b, n) },
    )
}

unsafe fn f64_unary<'a>(dst: *mut f64, a: *const f64, n: usize) -> (&'a mut [f64], &'a [f64]) {
    (unsafe { slice::from_raw_parts_mut(dst, n) }, unsafe {
        slice::from_raw_parts(a, n)
    })
}

// ---------------------------------------------------------------------------
// Elementwise binary operations (f64)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_add_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = a[i] + b[i];
    }
}

pub extern "C" fn tensor_sub_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = a[i] - b[i];
    }
}

pub extern "C" fn tensor_mul_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = a[i] * b[i];
    }
}

pub extern "C" fn tensor_div_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = a[i] / b[i];
    }
}

pub extern "C" fn tensor_max_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = if a[i] > b[i] { a[i] } else { b[i] };
    }
}

pub extern "C" fn tensor_min_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = if a[i] < b[i] { a[i] } else { b[i] };
    }
}

pub extern "C" fn tensor_pow_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = a[i].powf(b[i]);
    }
}

pub extern "C" fn tensor_rem_f64(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
    let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
    for i in 0..n {
        dst[i] = a[i] % b[i];
    }
}

// ---------------------------------------------------------------------------
// Elementwise binary operations (i64)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_add_i64(dst: *mut i64, a: *const i64, b: *const i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = a[i].wrapping_add(b[i]);
    }
}

pub extern "C" fn tensor_sub_i64(dst: *mut i64, a: *const i64, b: *const i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = a[i].wrapping_sub(b[i]);
    }
}

pub extern "C" fn tensor_mul_i64(dst: *mut i64, a: *const i64, b: *const i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = a[i].wrapping_mul(b[i]);
    }
}

// ---------------------------------------------------------------------------
// Elementwise binary operations (i32)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_select_i32(
    dst: *mut i32,
    cond: *const u8,
    on_true: *const i32,
    on_false: *const i32,
    n: usize,
) {
    let cond = unsafe { slice::from_raw_parts(cond, n) };
    let t = unsafe { slice::from_raw_parts(on_true, n) };
    let f = unsafe { slice::from_raw_parts(on_false, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = if cond[i] != 0 { t[i] } else { f[i] };
    }
}

pub extern "C" fn tensor_widen_i32_to_i64(dst: *mut i64, src: *const i32, n: usize) {
    let src = unsafe { slice::from_raw_parts(src, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = src[i] as i64;
    }
}

pub extern "C" fn tensor_broadcast_i32(dst: *mut i32, val: i32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = val;
    }
}

pub extern "C" fn tensor_add_i32(dst: *mut i32, a: *const i32, b: *const i32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = a[i].wrapping_add(b[i]);
    }
}

pub extern "C" fn tensor_sub_i32(dst: *mut i32, a: *const i32, b: *const i32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = a[i].wrapping_sub(b[i]);
    }
}

pub extern "C" fn tensor_div_i32(dst: *mut i32, a: *const i32, b: *const i32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] / b[i] } else { 0 };
    }
}

pub extern "C" fn tensor_div_ui32(dst: *mut u32, a: *const u32, b: *const u32, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] / b[i] } else { 0 };
    }
}

pub extern "C" fn tensor_cmp_lt_i32(dst: *mut u8, a: *const i32, b: *const i32, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] < b[i]) as u8;
    }
}

// ---------------------------------------------------------------------------
// Elementwise unary operations (f64)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_neg_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = -a[i];
    }
}

pub extern "C" fn tensor_sqrt_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].sqrt();
    }
}

pub extern "C" fn tensor_abs_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].abs();
    }
}

pub extern "C" fn tensor_floor_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].floor();
    }
}

pub extern "C" fn tensor_sin_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].sin();
    }
}

pub extern "C" fn tensor_cos_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].cos();
    }
}

pub extern "C" fn tensor_exp_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].exp();
    }
}

pub extern "C" fn tensor_log_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].ln();
    }
}

pub extern "C" fn tensor_sign_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = if a[i] > 0.0 {
            1.0
        } else if a[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

pub extern "C" fn tensor_tan_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].tan();
    }
}

pub extern "C" fn tensor_tanh_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = a[i].tanh();
    }
}

// ---------------------------------------------------------------------------
// Comparison: writes i8 (0 or 1) results
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_cmp_eq_f64(dst: *mut u8, a: *const f64, b: *const f64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] == b[i]) as u8;
    }
}

pub extern "C" fn tensor_cmp_lt_f64(dst: *mut u8, a: *const f64, b: *const f64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] < b[i]) as u8;
    }
}

pub extern "C" fn tensor_cmp_le_f64(dst: *mut u8, a: *const f64, b: *const f64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] <= b[i]) as u8;
    }
}

pub extern "C" fn tensor_cmp_gt_f64(dst: *mut u8, a: *const f64, b: *const f64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] > b[i]) as u8;
    }
}

pub extern "C" fn tensor_cmp_ge_f64(dst: *mut u8, a: *const f64, b: *const f64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] >= b[i]) as u8;
    }
}

pub extern "C" fn tensor_cmp_ne_f64(dst: *mut u8, a: *const f64, b: *const f64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] != b[i]) as u8;
    }
}

// ---------------------------------------------------------------------------
// Integer comparison (i64)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_cmp_eq_i64(dst: *mut u8, a: *const i64, b: *const i64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] == b[i]) as u8;
    }
}

pub extern "C" fn tensor_cmp_lt_i64(dst: *mut u8, a: *const i64, b: *const i64, n: usize) {
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = (a[i] < b[i]) as u8;
    }
}

// ---------------------------------------------------------------------------
// Select: dst[i] = cond[i] ? on_true[i] : on_false[i]
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_select_f64(
    dst: *mut f64,
    cond: *const u8,
    on_true: *const f64,
    on_false: *const f64,
    n: usize,
) {
    let cond = unsafe { slice::from_raw_parts(cond, n) };
    let t = unsafe { slice::from_raw_parts(on_true, n) };
    let f = unsafe { slice::from_raw_parts(on_false, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = if cond[i] != 0 { t[i] } else { f[i] };
    }
}

pub extern "C" fn tensor_select_i64(
    dst: *mut i64,
    cond: *const u8,
    on_true: *const i64,
    on_false: *const i64,
    n: usize,
) {
    let cond = unsafe { slice::from_raw_parts(cond, n) };
    let t = unsafe { slice::from_raw_parts(on_true, n) };
    let f = unsafe { slice::from_raw_parts(on_false, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = if cond[i] != 0 { t[i] } else { f[i] };
    }
}

// ---------------------------------------------------------------------------
// Type conversion
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_convert_i64_to_f64(dst: *mut f64, a: *const i64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as f64;
    }
}

pub extern "C" fn tensor_convert_f64_to_i64(dst: *mut i64, a: *const f64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as i64;
    }
}

pub extern "C" fn tensor_convert_i1_to_f64(dst: *mut f64, a: *const u8, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = if a[i] != 0 { 1.0 } else { 0.0 };
    }
}

pub extern "C" fn tensor_convert_f64_to_i32(dst: *mut i32, a: *const f64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as i32;
    }
}

pub extern "C" fn tensor_convert_i1_to_i32(dst: *mut i32, a: *const u8, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as i32;
    }
}

pub extern "C" fn tensor_convert_i64_to_i32(dst: *mut i32, a: *const i64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as i32;
    }
}

pub extern "C" fn tensor_convert_i32_to_f64(dst: *mut f64, a: *const i32, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as f64;
    }
}

pub extern "C" fn tensor_convert_f64_to_f32(dst: *mut f32, a: *const f64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as f32;
    }
}

pub extern "C" fn tensor_convert_f32_to_f64(dst: *mut f64, a: *const f32, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i] as f64;
    }
}

// ---------------------------------------------------------------------------
// Iota: fill with index values
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_iota_i64(dst: *mut i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = i as i64;
    }
}

pub extern "C" fn tensor_iota_f64(dst: *mut f64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = i as f64;
    }
}

// ---------------------------------------------------------------------------
// Broadcast: fill dst with a single scalar value
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_broadcast_f64(dst: *mut f64, val: f64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = val;
    }
}

pub extern "C" fn tensor_broadcast_i64(dst: *mut i64, val: i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = val;
    }
}

// ---------------------------------------------------------------------------
// Memcpy (typed for f64 element size)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_memcpy(dst: *mut u8, src: *const u8, n_bytes: usize) {
    unsafe { std::ptr::copy_nonoverlapping(src, dst, n_bytes) };
}

// ---------------------------------------------------------------------------
// Transpose (2D, row-major): dst[j*rows + i] = src[i*cols + j]
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_transpose_f64(dst: *mut f64, src: *const f64, rows: usize, cols: usize) {
    let src = unsafe { slice::from_raw_parts(src, rows * cols) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, rows * cols) };
    for i in 0..rows {
        for j in 0..cols {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce sum along last axis: dst[i] = sum(src[i*cols .. i*cols+cols])
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_reduce_sum_f64(
    dst: *mut f64,
    src: *const f64,
    outer: usize,
    inner: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, outer * inner) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
    for i in 0..outer {
        let mut acc = 0.0f64;
        for j in 0..inner {
            acc += src[i * inner + j];
        }
        dst[i] = acc;
    }
}

pub extern "C" fn tensor_reduce_max_f64(
    dst: *mut f64,
    src: *const f64,
    outer: usize,
    inner: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, outer * inner) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
    for i in 0..outer {
        let mut acc = f64::NEG_INFINITY;
        for j in 0..inner {
            let v = src[i * inner + j];
            if v > acc {
                acc = v;
            }
        }
        dst[i] = acc;
    }
}

pub extern "C" fn tensor_reduce_min_f64(
    dst: *mut f64,
    src: *const f64,
    outer: usize,
    inner: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, outer * inner) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
    for i in 0..outer {
        let mut acc = f64::INFINITY;
        for j in 0..inner {
            let v = src[i * inner + j];
            if v < acc {
                acc = v;
            }
        }
        dst[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// Scatter: operand[indices[i]] = updates[i]
// (simple 1D index-set pattern matching StableHLO scatter)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_scatter_f64(
    dst: *mut f64,
    src: *const f64,
    n_src: usize,
    indices: *const i64,
    updates: *const f64,
    n_updates: usize,
    inner_size: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_src) };
    dst.copy_from_slice(src);
    let indices = unsafe { slice::from_raw_parts(indices, n_updates) };
    let updates = unsafe { slice::from_raw_parts(updates, n_updates * inner_size) };
    for i in 0..n_updates {
        let idx = indices[i] as usize;
        let base = idx * inner_size;
        for j in 0..inner_size {
            if base + j < n_src {
                dst[base + j] = updates[i * inner_size + j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gather: dst[i] = src[indices[i]]
// (simple row-select pattern)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_gather_f64(
    dst: *mut f64,
    src: *const f64,
    n_src: usize,
    indices: *const i64,
    n_indices: usize,
    row_size: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_indices * row_size) };
    let indices = unsafe { slice::from_raw_parts(indices, n_indices) };
    for i in 0..n_indices {
        let idx = indices[i] as usize;
        let src_off = idx * row_size;
        let dst_off = i * row_size;
        if src_off + row_size <= n_src {
            dst[dst_off..dst_off + row_size].copy_from_slice(&src[src_off..src_off + row_size]);
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix multiply: C = A * B (row-major, naive triple loop)
// Will be replaced with faer in Phase C.
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_matmul_f64(
    dst: *mut f64,
    a: *const f64,
    b: *const f64,
    m: usize,
    k: usize,
    n: usize,
) {
    let a = unsafe { slice::from_raw_parts(a, m * k) };
    let b = unsafe { slice::from_raw_parts(b, k * n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, m * n) };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            dst[i * n + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// N-dimensional broadcast: replicate src according to broadcast_dims mapping
// src_shape mapped via broadcast_dims into dst_shape
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_broadcast_nd_f64(
    dst: *mut f64,
    src: *const f64,
    n_dst: usize,
    n_src: usize,
    dst_shape: *const i64,
    dst_rank: usize,
    src_shape: *const i64,
    src_rank: usize,
    broadcast_dims: *const i64,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst_shape = unsafe { slice::from_raw_parts(dst_shape, dst_rank) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, src_rank) };
    let broadcast_dims = unsafe { slice::from_raw_parts(broadcast_dims, src_rank) };

    let mut dst_strides = vec![1usize; dst_rank];
    for i in (0..dst_rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1] as usize;
    }

    let mut src_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    for flat_dst in 0..n_dst {
        let mut src_idx = 0usize;
        let mut remaining = flat_dst;
        for d in 0..dst_rank {
            let coord = remaining / dst_strides[d];
            remaining %= dst_strides[d];

            for (s, &bd) in broadcast_dims.iter().enumerate() {
                if bd as usize == d {
                    let src_coord = if src_shape[s] == 1 { 0 } else { coord };
                    src_idx += src_coord * src_strides[s];
                }
            }
        }
        dst[flat_dst] = src[src_idx.min(n_src.saturating_sub(1))];
    }
}

// ---------------------------------------------------------------------------
// Slice: copy a rectangular slice from src into dst
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_slice_f64(
    dst: *mut f64,
    src: *const f64,
    n_dst: usize,
    n_src: usize,
    shape: *const i64,
    rank: usize,
    starts: *const i64,
    limits: *const i64,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let starts = unsafe { slice::from_raw_parts(starts, rank) };
    let limits = unsafe { slice::from_raw_parts(limits, rank) };

    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1] as usize;
    }

    let mut dst_shape = Vec::with_capacity(rank);
    for d in 0..rank {
        dst_shape.push((limits[d] - starts[d]) as usize);
    }
    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];
    }

    for flat_dst in 0..n_dst {
        let mut src_flat = 0usize;
        let mut remaining = flat_dst;
        for d in 0..rank {
            let coord = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            src_flat += (starts[d] as usize + coord) * src_strides[d];
        }
        dst[flat_dst] = src[src_flat];
    }
}

// ---------------------------------------------------------------------------
// Concatenate along a dimension
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_concat_f64(
    dst: *mut f64,
    src_ptrs: *const *const f64,
    src_lens: *const usize,
    n_srcs: usize,
    n_dst: usize,
) {
    let src_ptrs = unsafe { slice::from_raw_parts(src_ptrs, n_srcs) };
    let src_lens = unsafe { slice::from_raw_parts(src_lens, n_srcs) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let mut off = 0usize;
    for i in 0..n_srcs {
        let src = unsafe { slice::from_raw_parts(src_ptrs[i], src_lens[i]) };
        dst[off..off + src_lens[i]].copy_from_slice(src);
        off += src_lens[i];
    }
}

pub extern "C" fn tensor_concat_nd_f64(
    dst: *mut f64,
    n_dst: usize,
    src_a: *const f64,
    n_a: usize,
    src_b: *const f64,
    n_b: usize,
    dst_shape: *const i64,
    a_shape: *const i64,
    rank: usize,
    dim: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let a = unsafe { slice::from_raw_parts(src_a, n_a) };
    let b = unsafe { slice::from_raw_parts(src_b, n_b) };
    let dst_shape = unsafe { slice::from_raw_parts(dst_shape, rank) };
    let a_shape = unsafe { slice::from_raw_parts(a_shape, rank) };

    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1] as usize;
    }

    let a_dim_size = a_shape[dim] as usize;

    for flat_dst in 0..n_dst {
        let mut remaining = flat_dst;
        let mut coords = vec![0usize; rank];
        for d in 0..rank {
            coords[d] = remaining / dst_strides[d];
            remaining %= dst_strides[d];
        }

        if coords[dim] < a_dim_size {
            let mut a_strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                a_strides[i] = a_strides[i + 1] * a_shape[i + 1] as usize;
            }
            let mut src_flat = 0usize;
            for d in 0..rank {
                src_flat += coords[d] * a_strides[d];
            }
            dst[flat_dst] = a[src_flat.min(n_a.saturating_sub(1))];
        } else {
            let mut b_shape = vec![0i64; rank];
            for d in 0..rank {
                b_shape[d] = if d == dim {
                    dst_shape[d] - a_shape[d]
                } else {
                    dst_shape[d]
                };
            }
            let mut b_strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                b_strides[i] = b_strides[i + 1] * b_shape[i + 1] as usize;
            }
            let mut b_coords = coords.clone();
            b_coords[dim] -= a_dim_size;
            let mut src_flat = 0usize;
            for d in 0..rank {
                src_flat += b_coords[d] * b_strides[d];
            }
            dst[flat_dst] = b[src_flat.min(n_b.saturating_sub(1))];
        }
    }
}

// ---------------------------------------------------------------------------
// Pad: zero-pad a tensor
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_pad_f64(
    dst: *mut f64,
    src: *const f64,
    n_dst: usize,
    n_src: usize,
    pad_value: f64,
    dst_shape: *const i64,
    src_shape: *const i64,
    rank: usize,
    low: *const i64,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst_shape = unsafe { slice::from_raw_parts(dst_shape, rank) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, rank) };
    let low = unsafe { slice::from_raw_parts(low, rank) };

    for v in dst.iter_mut() {
        *v = pad_value;
    }

    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1] as usize;
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    for flat_src in 0..n_src {
        let mut remaining = flat_src;
        let mut flat_dst = 0usize;
        for d in 0..rank {
            let coord = remaining / src_strides[d];
            remaining %= src_strides[d];
            flat_dst += (low[d] as usize + coord) * dst_strides[d];
        }
        if flat_dst < n_dst {
            dst[flat_dst] = src[flat_src];
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic slice / dynamic update slice
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_dynamic_slice_f64(
    dst: *mut f64,
    src: *const f64,
    n_dst: usize,
    n_src: usize,
    shape: *const i64,
    rank: usize,
    start_indices: *const i64,
    slice_sizes: *const i64,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let start_indices = unsafe { slice::from_raw_parts(start_indices, rank) };
    let slice_sizes = unsafe { slice::from_raw_parts(slice_sizes, rank) };

    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1] as usize;
    }
    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * slice_sizes[i + 1] as usize;
    }

    for flat_dst in 0..n_dst {
        let mut src_flat = 0usize;
        let mut remaining = flat_dst;
        for d in 0..rank {
            let coord = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            let start =
                (start_indices[d] as usize).min(shape[d] as usize - slice_sizes[d] as usize);
            src_flat += (start + coord) * src_strides[d];
        }
        dst[flat_dst] = src[src_flat.min(n_src.saturating_sub(1))];
    }
}

pub extern "C" fn tensor_dynamic_update_slice_f64(
    dst: *mut f64,
    src: *const f64,
    update: *const f64,
    n_src: usize,
    n_update: usize,
    shape: *const i64,
    rank: usize,
    start_indices: *const i64,
    update_shape: *const i64,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_src) };
    let update = unsafe { slice::from_raw_parts(update, n_update) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let start_indices = unsafe { slice::from_raw_parts(start_indices, rank) };
    let update_shape = unsafe { slice::from_raw_parts(update_shape, rank) };

    dst.copy_from_slice(src);

    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1] as usize;
    }
    let mut upd_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        upd_strides[i] = upd_strides[i + 1] * update_shape[i + 1] as usize;
    }

    for flat_upd in 0..n_update {
        let mut dst_flat = 0usize;
        let mut remaining = flat_upd;
        for d in 0..rank {
            let coord = remaining / upd_strides[d];
            remaining %= upd_strides[d];
            let start =
                (start_indices[d] as usize).min((shape[d] - update_shape[d]).max(0) as usize);
            dst_flat += (start + coord) * src_strides[d];
        }
        if dst_flat < n_src {
            dst[dst_flat] = update[flat_upd];
        }
    }
}

// ---------------------------------------------------------------------------
// Iota N-dimensional: fills a tensor where element = coordinate along `dimension`
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_iota_nd_i64(
    dst: *mut i64,
    n: usize,
    shape: *const i64,
    rank: usize,
    dimension: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };

    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    for flat in 0..n {
        let coord = (flat / strides[dimension]) % shape[dimension] as usize;
        dst[flat] = coord as i64;
    }
}

pub extern "C" fn tensor_transpose_nd_f64(
    dst: *mut f64,
    src: *const f64,
    n: usize,
    src_shape: *const i64,
    perm: *const i64,
    rank: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, rank) };
    let perm = unsafe { slice::from_raw_parts(perm, rank) };

    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    let mut dst_shape = vec![0i64; rank];
    for i in 0..rank {
        dst_shape[i] = src_shape[perm[i] as usize];
    }
    let mut dst_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1] as usize;
    }

    for flat_dst in 0..n {
        let mut remaining = flat_dst;
        let mut src_flat = 0usize;
        for d in 0..rank {
            let coord = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            src_flat += coord * src_strides[perm[d] as usize];
        }
        dst[flat_dst] = src[src_flat];
    }
}

pub extern "C" fn tensor_iota_nd_f64(
    dst: *mut f64,
    n: usize,
    shape: *const i64,
    rank: usize,
    dimension: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };

    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    for flat in 0..n {
        let coord = (flat / strides[dimension]) % shape[dimension] as usize;
        dst[flat] = coord as f64;
    }
}
