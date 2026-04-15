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
// Macro-generated elementwise operations
// ---------------------------------------------------------------------------

macro_rules! binary_f64_op {
    ($name:ident, $op:expr) => {
        pub extern "C" fn $name(dst: *mut f64, a: *const f64, b: *const f64, n: usize) {
            let (dst, a, b) = unsafe { f64_slices(dst, a, b, n) };
            for i in 0..n {
                dst[i] = ($op)(a[i], b[i]);
            }
        }
    };
}

binary_f64_op!(tensor_add_f64, |a: f64, b: f64| a + b);
binary_f64_op!(tensor_sub_f64, |a: f64, b: f64| a - b);
binary_f64_op!(tensor_mul_f64, |a: f64, b: f64| a * b);
binary_f64_op!(tensor_div_f64, |a: f64, b: f64| a / b);
binary_f64_op!(tensor_max_f64, |a: f64, b: f64| if a > b { a } else { b });
binary_f64_op!(tensor_min_f64, |a: f64, b: f64| if a < b { a } else { b });
binary_f64_op!(tensor_pow_f64, |a: f64, b: f64| a.powf(b));
binary_f64_op!(tensor_rem_f64, |a: f64, b: f64| a % b);
binary_f64_op!(tensor_atan2_f64, |a: f64, b: f64| a.atan2(b));

macro_rules! binary_int_op {
    ($name:ident, $ty:ty, $op:expr) => {
        pub extern "C" fn $name(dst: *mut $ty, a: *const $ty, b: *const $ty, n: usize) {
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
            for i in 0..n {
                dst[i] = ($op)(a[i], b[i]);
            }
        }
    };
}

binary_int_op!(tensor_add_i64, i64, |a: i64, b: i64| a.wrapping_add(b));
binary_int_op!(tensor_sub_i64, i64, |a: i64, b: i64| a.wrapping_sub(b));
binary_int_op!(tensor_mul_i64, i64, |a: i64, b: i64| a.wrapping_mul(b));
binary_int_op!(tensor_add_i32, i32, |a: i32, b: i32| a.wrapping_add(b));
binary_int_op!(tensor_sub_i32, i32, |a: i32, b: i32| a.wrapping_sub(b));
binary_int_op!(tensor_mul_i32, i32, |a: i32, b: i32| a.wrapping_mul(b));
binary_int_op!(tensor_sshr_i64, i64, |a: i64, b: i64| a.wrapping_shr(b as u32));
binary_int_op!(tensor_sshr_i32, i32, |a: i32, b: i32| a.wrapping_shr(b as u32));
binary_int_op!(tensor_max_i64, i64, |a: i64, b: i64| a.max(b));
binary_int_op!(tensor_min_i64, i64, |a: i64, b: i64| a.min(b));
binary_int_op!(tensor_max_i32, i32, |a: i32, b: i32| a.max(b));
binary_int_op!(tensor_min_i32, i32, |a: i32, b: i32| a.min(b));

pub extern "C" fn tensor_div_i64(dst: *mut i64, a: *const i64, b: *const i64, n: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    for i in 0..n {
        dst[i] = if b[i] != 0 { a[i] / b[i] } else { 0 };
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

// ---------------------------------------------------------------------------
// Elementwise unary operations (f64)
// ---------------------------------------------------------------------------

macro_rules! unary_f64_op {
    ($name:ident, $op:expr) => {
        pub extern "C" fn $name(dst: *mut f64, a: *const f64, n: usize) {
            let (dst, a) = unsafe { f64_unary(dst, a, n) };
            for i in 0..n {
                dst[i] = ($op)(a[i]);
            }
        }
    };
}

unary_f64_op!(tensor_neg_f64, |x: f64| -x);

macro_rules! unary_int_op {
    ($name:ident, $ty:ty, $op:expr) => {
        pub extern "C" fn $name(dst: *mut $ty, a: *const $ty, n: usize) {
            let a = unsafe { slice::from_raw_parts(a, n) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n { dst[i] = ($op)(a[i]); }
        }
    };
}

unary_int_op!(tensor_neg_i64, i64, |x: i64| x.wrapping_neg());
unary_int_op!(tensor_neg_i32, i32, |x: i32| x.wrapping_neg());
unary_int_op!(tensor_abs_i64, i64, |x: i64| x.wrapping_abs());
unary_int_op!(tensor_abs_i32, i32, |x: i32| x.wrapping_abs());
unary_f64_op!(tensor_sqrt_f64, f64::sqrt);
unary_f64_op!(tensor_abs_f64, f64::abs);
unary_f64_op!(tensor_floor_f64, f64::floor);
unary_f64_op!(tensor_sin_f64, f64::sin);
unary_f64_op!(tensor_cos_f64, f64::cos);
unary_f64_op!(tensor_exp_f64, f64::exp);
unary_f64_op!(tensor_log_f64, f64::ln);
unary_f64_op!(tensor_tan_f64, f64::tan);
unary_f64_op!(tensor_tanh_f64, f64::tanh);
unary_f64_op!(tensor_acos_f64, f64::acos);
unary_f64_op!(tensor_rsqrt_f64, |x: f64| 1.0 / x.sqrt());
unary_f64_op!(tensor_log1p_f64, f64::ln_1p);
unary_f64_op!(tensor_ceil_f64, f64::ceil);

pub extern "C" fn tensor_is_finite_f64(dst: *mut u8, a: *const f64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = a[i].is_finite() as u8;
    }
}

pub extern "C" fn tensor_not_i64(dst: *mut i64, a: *const i64, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n { dst[i] = !a[i]; }
}

pub extern "C" fn tensor_not_i32(dst: *mut i32, a: *const i32, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n { dst[i] = !a[i]; }
}

pub extern "C" fn tensor_not_i1(dst: *mut u8, a: *const u8, n: usize) {
    let a = unsafe { slice::from_raw_parts(a, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n { dst[i] = if a[i] == 0 { 1 } else { 0 }; }
}

fn round_ties_even(x: f64) -> f64 {
    let r = x.round();
    if (x - r).abs() == 0.5 {
        let t = r / 2.0;
        if t.floor() == t { r } else { r - x.signum() }
    } else {
        r
    }
}

unary_f64_op!(tensor_round_f64, round_ties_even);

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

pub extern "C" fn tensor_erf_inv_f64(dst: *mut f64, a: *const f64, n: usize) {
    let (dst, a) = unsafe { f64_unary(dst, a, n) };
    for i in 0..n {
        dst[i] = erf_inv_impl(a[i]);
    }
}

fn erf_inv_impl(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    ndtri_impl((x + 1.0) * 0.5) * std::f64::consts::FRAC_1_SQRT_2
}

#[allow(clippy::excessive_precision)]
fn ndtri_impl(y0: f64) -> f64 {
    const P0: [f64; 5] = [
        -5.99633501014107895267E1, 9.80010754185999661536E1,
        -5.66762857469070293439E1, 1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ];
    const Q0: [f64; 8] = [
        1.95448858338141759834E0, 4.67627912898881538453E0,
        8.63602421390890590575E1, -2.25462687854119370527E2,
        2.00260212380060660359E2, -8.20372256168333339912E1,
        1.59056225126211695515E1, -1.18331621121330003142E0,
    ];
    const P1: [f64; 9] = [
        4.05544892305962419923E0, 3.15251094599893866154E1,
        5.71628192246421288162E1, 4.40805073893200834700E1,
        1.46849561928858024014E1, 2.18663306850790267539E0,
        -1.40256079171354495875E-1, -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ];
    const Q1: [f64; 8] = [
        1.57799883256466749731E1, 4.53907635128879210584E1,
        4.13172038254672030440E1, 1.50425385692907503408E1,
        2.50464946208309415979E0, -1.42182922854787788574E-1,
        -3.80806407691578277194E-2, -9.33259480895457427372E-4,
    ];
    const P2: [f64; 9] = [
        3.23774891776946035970E0, 6.91522889068984211695E0,
        3.93881025292474443415E0, 1.33303460815807542389E0,
        2.01485389549179081538E-1, 1.23716634817820021358E-2,
        3.01581553508235416007E-4, 2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ];
    const Q2: [f64; 8] = [
        6.02427039364742014255E0, 3.67983563856160859403E0,
        1.37702099489081330271E0, 2.16236993594496635890E-1,
        1.34204006088543189037E-2, 3.28014464682127739104E-4,
        2.89247864745380683936E-6, 6.79019408009981274425E-9,
    ];
    if y0 <= 0.0 { return f64::NEG_INFINITY; }
    if y0 >= 1.0 { return f64::INFINITY; }
    if y0 == 0.5 { return 0.0; }
    let s2pi: f64 = 2.50662827463100050242;
    let (code, mut y) = if y0 > 0.86466471676338730811 { (0i32, 1.0 - y0) } else { (1i32, y0) };
    if y > 0.13533528323661269189 {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * poly_eval(y2, &P0) / poly_eval_1(y2, &Q0));
        return x * s2pi;
    }
    let mut x = (-2.0 * y.ln()).sqrt();
    let x0 = x - (2.0f64 * std::f64::consts::PI).ln() / (2.0 * x);
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * poly_eval(z, &P1) / poly_eval_1(z, &Q1)
    } else {
        z * poly_eval(z, &P2) / poly_eval_1(z, &Q2)
    };
    x = x0 - x1;
    if code != 0 { x = -x; }
    x
}

fn poly_eval(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

fn poly_eval_1(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().fold(1.0, |acc, &c| acc * x + c)
}

pub extern "C" fn tensor_clamp_f64(
    dst: *mut f64,
    src: *const f64,
    min: *const f64,
    max: *const f64,
    n: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n) };
    let min = unsafe { slice::from_raw_parts(min, n) };
    let max = unsafe { slice::from_raw_parts(max, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    for i in 0..n {
        dst[i] = if src[i] < min[i] {
            min[i]
        } else if src[i] > max[i] {
            max[i]
        } else {
            src[i]
        };
    }
}

pub extern "C" fn tensor_reverse_f64(
    dst: *mut f64,
    src: *const f64,
    n: usize,
    shape: *const i64,
    rank: usize,
    dims: *const i64,
    n_dims: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let rev_dims = unsafe { slice::from_raw_parts(dims, n_dims) };

    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    for flat in 0..n {
        let mut src_flat = 0usize;
        let mut remaining = flat;
        for d in 0..rank {
            let coord = remaining / strides[d];
            remaining %= strides[d];
            let c = if rev_dims.contains(&(d as i64)) {
                shape[d] as usize - 1 - coord
            } else {
                coord
            };
            src_flat += c * strides[d];
        }
        dst[flat] = src[src_flat];
    }
}

// ---------------------------------------------------------------------------
// Comparison operations
// ---------------------------------------------------------------------------

macro_rules! cmp_op {
    ($name:ident, $ty:ty, $op:expr) => {
        pub extern "C" fn $name(dst: *mut u8, a: *const $ty, b: *const $ty, n: usize) {
            let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = ($op)(a[i], b[i]) as u8;
            }
        }
    };
}

cmp_op!(tensor_cmp_eq_f64, f64, |a: f64, b: f64| a == b);
cmp_op!(tensor_cmp_lt_f64, f64, |a: f64, b: f64| a < b);
cmp_op!(tensor_cmp_le_f64, f64, |a: f64, b: f64| a <= b);
cmp_op!(tensor_cmp_gt_f64, f64, |a: f64, b: f64| a > b);
cmp_op!(tensor_cmp_ge_f64, f64, |a: f64, b: f64| a >= b);
cmp_op!(tensor_cmp_ne_f64, f64, |a: f64, b: f64| a != b);
cmp_op!(tensor_cmp_eq_i64, i64, |a: i64, b: i64| a == b);
cmp_op!(tensor_cmp_ne_i64, i64, |a: i64, b: i64| a != b);
cmp_op!(tensor_cmp_lt_i64, i64, |a: i64, b: i64| a < b);
cmp_op!(tensor_cmp_le_i64, i64, |a: i64, b: i64| a <= b);
cmp_op!(tensor_cmp_gt_i64, i64, |a: i64, b: i64| a > b);
cmp_op!(tensor_cmp_ge_i64, i64, |a: i64, b: i64| a >= b);
cmp_op!(tensor_cmp_eq_i32, i32, |a: i32, b: i32| a == b);
cmp_op!(tensor_cmp_ne_i32, i32, |a: i32, b: i32| a != b);
cmp_op!(tensor_cmp_lt_i32, i32, |a: i32, b: i32| a < b);
cmp_op!(tensor_cmp_le_i32, i32, |a: i32, b: i32| a <= b);
cmp_op!(tensor_cmp_gt_i32, i32, |a: i32, b: i32| a > b);
cmp_op!(tensor_cmp_ge_i32, i32, |a: i32, b: i32| a >= b);

// ---------------------------------------------------------------------------
// Select: dst[i] = cond[i] ? on_true[i] : on_false[i]
// ---------------------------------------------------------------------------

macro_rules! select_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(
            dst: *mut $ty,
            cond: *const u8,
            on_true: *const $ty,
            on_false: *const $ty,
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
    };
}

select_op!(tensor_select_i32, i32);
select_op!(tensor_select_f64, f64);
select_op!(tensor_select_i64, i64);

// ---------------------------------------------------------------------------
// Type conversion
// ---------------------------------------------------------------------------

macro_rules! convert_op {
    ($name:ident, $src_ty:ty, $dst_ty:ty, $conv:expr) => {
        pub extern "C" fn $name(dst: *mut $dst_ty, a: *const $src_ty, n: usize) {
            let a = unsafe { slice::from_raw_parts(a, n) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = ($conv)(a[i]);
            }
        }
    };
}

convert_op!(tensor_widen_i32_to_i64, i32, i64, |x: i32| x as i64);
convert_op!(tensor_convert_i64_to_f64, i64, f64, |x: i64| x as f64);
convert_op!(tensor_convert_f64_to_i64, f64, i64, |x: f64| x as i64);
convert_op!(tensor_convert_i1_to_f64, u8, f64, |x: u8| if x != 0 {
    1.0
} else {
    0.0
});
convert_op!(tensor_convert_f64_to_i32, f64, i32, |x: f64| x as i32);
convert_op!(tensor_convert_i1_to_i32, u8, i32, |x: u8| x as i32);
convert_op!(tensor_convert_i64_to_i32, i64, i32, |x: i64| x as i32);
convert_op!(tensor_convert_i32_to_f64, i32, f64, |x: i32| x as f64);
convert_op!(tensor_convert_f64_to_f32, f64, f32, |x: f64| x as f32);
convert_op!(tensor_convert_f32_to_f64, f32, f64, |x: f32| x as f64);
convert_op!(tensor_convert_ui32_to_i64, u32, i64, |x: u32| x as i64);
convert_op!(tensor_convert_ui32_to_f64, u32, f64, |x: u32| x as f64);
convert_op!(tensor_convert_f64_to_i1, f64, u8, |x: f64| (x != 0.0) as u8);
convert_op!(tensor_convert_i64_to_i1, i64, u8, |x: i64| (x != 0) as u8);
convert_op!(tensor_convert_ui64_to_f64, u64, f64, |x: u64| x as f64);
convert_op!(tensor_convert_i32_to_f32, i32, f32, |x: i32| x as f32);
convert_op!(tensor_convert_f32_to_i32, f32, i32, |x: f32| x as i32);

// ---------------------------------------------------------------------------
// Broadcast: fill dst with a single scalar value
// ---------------------------------------------------------------------------

macro_rules! broadcast_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(dst: *mut $ty, val: $ty, n: usize) {
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = val;
            }
        }
    };
}

broadcast_op!(tensor_broadcast_f64, f64);
broadcast_op!(tensor_broadcast_i64, i64);
broadcast_op!(tensor_broadcast_i32, i32);

// ---------------------------------------------------------------------------
// Iota: fill with index values
// ---------------------------------------------------------------------------

macro_rules! iota_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(dst: *mut $ty, n: usize) {
            let dst = unsafe { slice::from_raw_parts_mut(dst, n) };
            for i in 0..n {
                dst[i] = i as $ty;
            }
        }
    };
}

iota_op!(tensor_iota_i64, i64);
iota_op!(tensor_iota_f64, f64);

// ---------------------------------------------------------------------------
// Memcpy
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
// Reduce along last axis
// ---------------------------------------------------------------------------

macro_rules! reduce_op {
    ($name:ident, $init:expr, $update:expr) => {
        pub extern "C" fn $name(
            dst: *mut f64,
            src: *const f64,
            outer: usize,
            inner: usize,
        ) {
            let src = unsafe { slice::from_raw_parts(src, outer * inner) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
            for i in 0..outer {
                let mut acc: f64 = $init;
                for j in 0..inner {
                    acc = ($update)(acc, src[i * inner + j]);
                }
                dst[i] = acc;
            }
        }
    };
}

reduce_op!(tensor_reduce_sum_f64, 0.0, |acc: f64, v: f64| acc + v);
reduce_op!(tensor_reduce_max_f64, f64::NEG_INFINITY, |acc: f64, v: f64| if v > acc { v } else { acc });
reduce_op!(tensor_reduce_min_f64, f64::INFINITY, |acc: f64, v: f64| if v < acc { v } else { acc });

macro_rules! reduce_int_op {
    ($name:ident, $ty:ty, $init:expr, $update:expr) => {
        pub extern "C" fn $name(dst: *mut $ty, src: *const $ty, outer: usize, inner: usize) {
            let src = unsafe { slice::from_raw_parts(src, outer * inner) };
            let dst = unsafe { slice::from_raw_parts_mut(dst, outer) };
            for i in 0..outer {
                let mut acc: $ty = $init;
                for j in 0..inner { acc = ($update)(acc, src[i * inner + j]); }
                dst[i] = acc;
            }
        }
    };
}

reduce_int_op!(tensor_reduce_sum_i64, i64, 0, |acc: i64, v: i64| acc.wrapping_add(v));
reduce_int_op!(tensor_reduce_max_i64, i64, i64::MIN, |acc: i64, v: i64| acc.max(v));
reduce_int_op!(tensor_reduce_min_i64, i64, i64::MAX, |acc: i64, v: i64| acc.min(v));

// ---------------------------------------------------------------------------
// Scatter: operand[indices[i]] = updates[i]
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
// Gather: simple row-select pattern
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
// N-D indexed gather
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_gather_nd_f64(
    dst: *mut f64,
    src: *const f64,
    n_src: usize,
    indices: *const i64,
    n_batch: usize,
    n_index_dims: usize,
    src_shape: *const i64,
    src_rank: usize,
    start_index_map: *const i64,
    slice_sizes: *const i64,
    n_dst: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst) };
    let indices = unsafe { slice::from_raw_parts(indices, n_batch * n_index_dims) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, src_rank) };
    let start_index_map = unsafe { slice::from_raw_parts(start_index_map, n_index_dims) };
    let slice_sizes = unsafe { slice::from_raw_parts(slice_sizes, src_rank) };

    let mut src_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    let slice_elems: usize = slice_sizes.iter().map(|&s| s as usize).product();

    let mut slice_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        slice_strides[i] = slice_strides[i + 1] * slice_sizes[i + 1] as usize;
    }

    let mut dst_off = 0usize;
    for b in 0..n_batch {
        let mut base_flat = 0usize;
        for j in 0..n_index_dims {
            let idx = indices[b * n_index_dims + j] as usize;
            let dim = start_index_map[j] as usize;
            let clamped = idx.min((src_shape[dim] as usize).saturating_sub(slice_sizes[dim] as usize));
            base_flat += clamped * src_strides[dim];
        }

        for s in 0..slice_elems {
            let mut src_flat = base_flat;
            let mut rem = s;
            for d in 0..src_rank {
                let coord = rem / slice_strides[d];
                rem %= slice_strides[d];
                src_flat += coord * src_strides[d];
            }
            if dst_off < n_dst && src_flat < n_src {
                dst[dst_off] = src[src_flat];
            }
            dst_off += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-generic gather: row-select with explicit element size
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_gather_generic(
    dst: *mut u8,
    src: *const u8,
    n_src: usize,
    indices: *const i64,
    n_indices: usize,
    row_size: usize,
    elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_indices * row_size * elem_sz) };
    let indices = unsafe { slice::from_raw_parts(indices, n_indices) };
    for i in 0..n_indices {
        let idx = indices[i] as usize;
        let src_off = idx * row_size * elem_sz;
        let dst_off = i * row_size * elem_sz;
        let row_bytes = row_size * elem_sz;
        if src_off + row_bytes <= src.len() {
            dst[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..src_off + row_bytes]);
        }
    }
}

pub extern "C" fn tensor_gather_nd_generic(
    dst: *mut u8,
    src: *const u8,
    n_src: usize,
    indices: *const i64,
    n_batch: usize,
    n_index_dims: usize,
    src_shape: *const i64,
    src_rank: usize,
    start_index_map: *const i64,
    slice_sizes: *const i64,
    n_dst: usize,
    elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let indices = unsafe { slice::from_raw_parts(indices, n_batch * n_index_dims) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, src_rank) };
    let start_index_map = unsafe { slice::from_raw_parts(start_index_map, n_index_dims) };
    let slice_sizes = unsafe { slice::from_raw_parts(slice_sizes, src_rank) };

    let mut src_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }
    let slice_elems: usize = slice_sizes.iter().map(|&s| s as usize).product();
    let mut slice_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        slice_strides[i] = slice_strides[i + 1] * slice_sizes[i + 1] as usize;
    }

    let mut dst_off = 0usize;
    for b in 0..n_batch {
        let mut base_flat = 0usize;
        for j in 0..n_index_dims {
            let idx = indices[b * n_index_dims + j] as usize;
            let dim = start_index_map[j] as usize;
            let clamped = idx.min((src_shape[dim] as usize).saturating_sub(slice_sizes[dim] as usize));
            base_flat += clamped * src_strides[dim];
        }
        for s in 0..slice_elems {
            let mut src_flat = base_flat;
            let mut rem = s;
            for d in 0..src_rank {
                let coord = rem / slice_strides[d];
                rem %= slice_strides[d];
                src_flat += coord * src_strides[d];
            }
            if dst_off < n_dst && src_flat < n_src {
                let sb = src_flat * elem_sz;
                let db = dst_off * elem_sz;
                dst[db..db + elem_sz].copy_from_slice(&src[sb..sb + elem_sz]);
            }
            dst_off += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Byte-generic scatter with explicit element size
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_scatter_generic(
    dst: *mut u8,
    src: *const u8,
    n_src: usize,
    indices: *const i64,
    updates: *const u8,
    n_updates: usize,
    inner_size: usize,
    elem_sz: usize,
) {
    let total_bytes = n_src * elem_sz;
    let src = unsafe { slice::from_raw_parts(src, total_bytes) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, total_bytes) };
    dst.copy_from_slice(src);
    let indices = unsafe { slice::from_raw_parts(indices, n_updates) };
    let updates = unsafe { slice::from_raw_parts(updates, n_updates * inner_size * elem_sz) };
    for i in 0..n_updates {
        let idx = indices[i] as usize;
        let base = idx * inner_size;
        for j in 0..inner_size {
            if base + j < n_src {
                let db = (base + j) * elem_sz;
                let ub = (i * inner_size + j) * elem_sz;
                dst[db..db + elem_sz].copy_from_slice(&updates[ub..ub + elem_sz]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix multiply: C = A * B (row-major, naive triple loop)
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
// N-dimensional broadcast
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
// Slice
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
    dst: *mut u8,
    n_dst: usize,
    src_a: *const u8,
    n_a: usize,
    src_b: *const u8,
    n_b: usize,
    dst_shape: *const i64,
    a_shape: *const i64,
    rank: usize,
    dim: usize,
    elem_sz: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let a = unsafe { slice::from_raw_parts(src_a, n_a * elem_sz) };
    let b = unsafe { slice::from_raw_parts(src_b, n_b * elem_sz) };
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
            let idx = src_flat.min(n_a.saturating_sub(1));
            dst[flat_dst * elem_sz..(flat_dst + 1) * elem_sz]
                .copy_from_slice(&a[idx * elem_sz..(idx + 1) * elem_sz]);
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
            let idx = src_flat.min(n_b.saturating_sub(1));
            dst[flat_dst * elem_sz..(flat_dst + 1) * elem_sz]
                .copy_from_slice(&b[idx * elem_sz..(idx + 1) * elem_sz]);
        }
    }
}

// ---------------------------------------------------------------------------
// Pad
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
// Byte-generic layout operations (for non-f64 element types)
// ---------------------------------------------------------------------------

pub extern "C" fn tensor_broadcast_nd_generic(
    dst: *mut u8, src: *const u8, n_dst: usize, n_src: usize,
    dst_shape: *const i64, dst_rank: usize,
    src_shape: *const i64, src_rank: usize,
    broadcast_dims: *const i64, elem_sz: usize,
) {
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst_shape = unsafe { slice::from_raw_parts(dst_shape, dst_rank) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, src_rank) };
    let broadcast_dims = unsafe { slice::from_raw_parts(broadcast_dims, src_rank) };
    let mut ds = vec![1usize; dst_rank];
    for i in (0..dst_rank.saturating_sub(1)).rev() { ds[i] = ds[i+1] * dst_shape[i+1] as usize; }
    let mut ss = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() { ss[i] = ss[i+1] * src_shape[i+1] as usize; }
    for fd in 0..n_dst {
        let mut si = 0usize;
        let mut rem = fd;
        for d in 0..dst_rank {
            let coord = rem / ds[d]; rem %= ds[d];
            for (s, &bd) in broadcast_dims.iter().enumerate() {
                if bd as usize == d {
                    si += (if src_shape[s] == 1 { 0 } else { coord }) * ss[s];
                }
            }
        }
        si = si.min(n_src.saturating_sub(1));
        let db = fd * elem_sz;
        let sb = si * elem_sz;
        dst[db..db+elem_sz].copy_from_slice(&src[sb..sb+elem_sz]);
    }
}

pub extern "C" fn tensor_slice_generic(
    dst: *mut u8, src: *const u8, n_dst: usize, n_src: usize,
    shape: *const i64, rank: usize, starts: *const i64, limits: *const i64, elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let starts = unsafe { slice::from_raw_parts(starts, rank) };
    let limits = unsafe { slice::from_raw_parts(limits, rank) };
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { src_s[i] = src_s[i+1] * shape[i+1] as usize; }
    let mut dst_shape: Vec<usize> = (0..rank).map(|d| (limits[d]-starts[d]) as usize).collect();
    let mut dst_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { dst_s[i] = dst_s[i+1] * dst_shape[i+1]; }
    for fd in 0..n_dst {
        let mut sf = 0usize; let mut rem = fd;
        for d in 0..rank { let c = rem/dst_s[d]; rem %= dst_s[d]; sf += (starts[d] as usize+c)*src_s[d]; }
        let db = fd*elem_sz; let sb = sf*elem_sz;
        dst[db..db+elem_sz].copy_from_slice(&src[sb..sb+elem_sz]);
    }
}

pub extern "C" fn tensor_transpose_nd_generic(
    dst: *mut u8, src: *const u8, n: usize,
    src_shape: *const i64, perm: *const i64, rank: usize, elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n * elem_sz) };
    let src_shape = unsafe { slice::from_raw_parts(src_shape, rank) };
    let perm = unsafe { slice::from_raw_parts(perm, rank) };
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { src_s[i] = src_s[i+1] * src_shape[i+1] as usize; }
    let mut dst_shape: Vec<i64> = (0..rank).map(|i| src_shape[perm[i] as usize]).collect();
    let mut dst_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { dst_s[i] = dst_s[i+1] * dst_shape[i+1] as usize; }
    for fd in 0..n {
        let mut rem = fd; let mut sf = 0usize;
        for d in 0..rank { let c = rem/dst_s[d]; rem %= dst_s[d]; sf += c*src_s[perm[d] as usize]; }
        let db = fd*elem_sz; let sb = sf*elem_sz;
        dst[db..db+elem_sz].copy_from_slice(&src[sb..sb+elem_sz]);
    }
}

pub extern "C" fn tensor_dynamic_slice_generic(
    dst: *mut u8, src: *const u8, n_dst: usize, n_src: usize,
    shape: *const i64, rank: usize, start_indices: *const i64, slice_sizes: *const i64, elem_sz: usize,
) {
    let src = unsafe { slice::from_raw_parts(src, n_src * elem_sz) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, n_dst * elem_sz) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let start_indices = unsafe { slice::from_raw_parts(start_indices, rank) };
    let slice_sizes = unsafe { slice::from_raw_parts(slice_sizes, rank) };
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { src_s[i] = src_s[i+1] * shape[i+1] as usize; }
    let mut dst_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { dst_s[i] = dst_s[i+1] * slice_sizes[i+1] as usize; }
    for fd in 0..n_dst {
        let mut sf = 0usize; let mut rem = fd;
        for d in 0..rank {
            let c = rem/dst_s[d]; rem %= dst_s[d];
            let start = (start_indices[d] as usize).min(shape[d] as usize - slice_sizes[d] as usize);
            sf += (start+c)*src_s[d];
        }
        sf = sf.min(n_src.saturating_sub(1));
        let db = fd*elem_sz; let sb = sf*elem_sz;
        dst[db..db+elem_sz].copy_from_slice(&src[sb..sb+elem_sz]);
    }
}

pub extern "C" fn tensor_dynamic_update_slice_generic(
    dst: *mut u8, src: *const u8, update: *const u8,
    n_src: usize, n_update: usize,
    shape: *const i64, rank: usize, start_indices: *const i64, update_shape: *const i64, elem_sz: usize,
) {
    let total = n_src * elem_sz;
    let src = unsafe { slice::from_raw_parts(src, total) };
    let dst = unsafe { slice::from_raw_parts_mut(dst, total) };
    let update = unsafe { slice::from_raw_parts(update, n_update * elem_sz) };
    let shape = unsafe { slice::from_raw_parts(shape, rank) };
    let start_indices = unsafe { slice::from_raw_parts(start_indices, rank) };
    let update_shape = unsafe { slice::from_raw_parts(update_shape, rank) };
    dst.copy_from_slice(src);
    let mut src_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { src_s[i] = src_s[i+1] * shape[i+1] as usize; }
    let mut upd_s = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() { upd_s[i] = upd_s[i+1] * update_shape[i+1] as usize; }
    for fu in 0..n_update {
        let mut df = 0usize; let mut rem = fu;
        for d in 0..rank {
            let c = rem/upd_s[d]; rem %= upd_s[d];
            let start = (start_indices[d] as usize).min((shape[d]-update_shape[d]).max(0) as usize);
            df += (start+c)*src_s[d];
        }
        if df < n_src {
            let db = df*elem_sz; let ub = fu*elem_sz;
            dst[db..db+elem_sz].copy_from_slice(&update[ub..ub+elem_sz]);
        }
    }
}

// ---------------------------------------------------------------------------
// Iota N-dimensional
// ---------------------------------------------------------------------------

macro_rules! iota_nd_op {
    ($name:ident, $ty:ty) => {
        pub extern "C" fn $name(
            dst: *mut $ty,
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
                dst[flat] = coord as $ty;
            }
        }
    };
}

iota_nd_op!(tensor_iota_nd_i64, i64);
iota_nd_op!(tensor_iota_nd_f64, f64);

// ---------------------------------------------------------------------------
// N-D transpose
// ---------------------------------------------------------------------------

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
