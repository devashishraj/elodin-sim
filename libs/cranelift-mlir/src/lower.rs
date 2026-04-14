use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::instructions::BlockArg;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{
    AbiParam, Block, InstBuilder, MemFlags, Signature, StackSlotData, StackSlotKind, Type, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};

use crate::ir::*;
use crate::tensor_rt;

pub struct CompiledModule {
    module: JITModule,
    main_fn_id: FuncId,
}

impl CompiledModule {
    pub fn get_main_fn(&self) -> *const u8 {
        self.module.get_finalized_function(self.main_fn_id)
    }
}

#[derive(Default, Clone, Copy)]
pub struct CompileConfig {
    pub force_pointer_abi_main: bool,
}

type TensorVals = Vec<Value>;

const MAX_REG_RETURNS: usize = 8;
const LARGE_TENSOR_THRESHOLD: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FuncAbi {
    Scalar,
    Pointer,
}

fn classify_function(func_def: &FuncDef) -> FuncAbi {
    if func_def.name == "main" {
        return FuncAbi::Scalar;
    }
    let param_max = func_def
        .params
        .iter()
        .map(|(_, t)| t.num_elements())
        .max()
        .unwrap_or(0);
    let ret_max = func_def
        .result_types
        .iter()
        .map(|t| t.num_elements())
        .max()
        .unwrap_or(0);
    let body_max = scan_body_max_elements(&func_def.body);
    if param_max.max(ret_max).max(body_max) > LARGE_TENSOR_THRESHOLD {
        FuncAbi::Pointer
    } else {
        FuncAbi::Scalar
    }
}

fn scan_body_max_elements(body: &[InstrResult]) -> usize {
    let mut max_elem = 0usize;
    for ir in body {
        for (_, ty) in &ir.values {
            max_elem = max_elem.max(ty.num_elements());
        }
        match &ir.instr {
            Instruction::While {
                cond_body,
                loop_body,
                ..
            } => {
                max_elem = max_elem.max(scan_body_max_elements(cond_body));
                max_elem = max_elem.max(scan_body_max_elements(loop_body));
            }
            Instruction::Case { branches, .. } => {
                for branch in branches {
                    max_elem = max_elem.max(scan_body_max_elements(branch));
                }
            }
            _ => {}
        }
    }
    max_elem
}

fn classify_all_functions(ir_module: &crate::ir::Module) -> HashMap<String, FuncAbi> {
    ir_module
        .functions
        .iter()
        .map(|f| (f.name.clone(), classify_function(f)))
        .collect()
}

fn cranelift_type_for(et: ElementType) -> Type {
    match et {
        ElementType::F64 => types::F64,
        ElementType::F32 => types::F32,
        ElementType::I1 => types::I8,
        ElementType::I32 => types::I32,
        ElementType::I64 => types::I64,
        ElementType::UI32 => types::I32,
        ElementType::UI64 => types::I64,
    }
}

fn ptr_type() -> Type {
    types::I64
}

fn is_float(et: ElementType) -> bool {
    matches!(et, ElementType::F64 | ElementType::F32)
}

fn is_unsigned(et: ElementType) -> bool {
    matches!(et, ElementType::UI32 | ElementType::UI64)
}

fn total_return_elements(result_types: &[TensorType]) -> usize {
    result_types.iter().map(|t| t.num_elements()).sum()
}

fn needs_sret(result_types: &[TensorType]) -> bool {
    total_return_elements(result_types) > MAX_REG_RETURNS
}

fn add_tensor_params(sig: &mut Signature, ty: &TensorType) {
    let ct = cranelift_type_for(ty.element_type);
    for _ in 0..ty.num_elements() {
        sig.params.push(AbiParam::new(ct));
    }
}

fn add_tensor_returns(sig: &mut Signature, ty: &TensorType) {
    let ct = cranelift_type_for(ty.element_type);
    for _ in 0..ty.num_elements() {
        sig.returns.push(AbiParam::new(ct));
    }
}

// ---------------------------------------------------------------------------
// Extern "C" shims for libm functions
// ---------------------------------------------------------------------------

extern "C" fn libc_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn libc_cos(x: f64) -> f64 {
    x.cos()
}
extern "C" fn libc_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}
extern "C" fn libc_sqrt(x: f64) -> f64 {
    x.sqrt()
}
extern "C" fn libc_fabs(x: f64) -> f64 {
    x.abs()
}
extern "C" fn libc_fmod(x: f64, y: f64) -> f64 {
    x % y
}
extern "C" fn libc_acos(x: f64) -> f64 {
    x.acos()
}
extern "C" fn libc_log(x: f64) -> f64 {
    x.ln()
}
extern "C" fn libc_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn libc_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}
extern "C" fn libc_tanh(x: f64) -> f64 {
    x.tanh()
}
extern "C" fn libc_tan(x: f64) -> f64 {
    x.tan()
}

/// erfinv via the Cephes ndtri rational approximation.
/// erfinv(x) = ndtri((x + 1) / 2) / sqrt(2)
extern "C" fn erf_inv_scalar(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    ndtri((x + 1.0) * 0.5) * std::f64::consts::FRAC_1_SQRT_2
}

/// Cephes ndtri: inverse of the standard normal CDF.
/// Ported from scipy/special/cephes/ndtri.c (BSD licensed).
#[allow(clippy::excessive_precision)]
fn ndtri(y0: f64) -> f64 {
    const P0: [f64; 5] = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ];
    const Q0: [f64; 8] = [
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ];
    const P1: [f64; 9] = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ];
    const Q1: [f64; 8] = [
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ];
    const P2: [f64; 9] = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ];
    const Q2: [f64; 8] = [
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ];

    if y0 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 >= 1.0 {
        return f64::INFINITY;
    }
    if y0 == 0.5 {
        return 0.0;
    }

    let s2pi: f64 = 2.50662827463100050242;

    let (code, mut y) = if y0 > (1.0 - 0.13533528323661269189) {
        (0i32, 1.0 - y0)
    } else {
        (1i32, y0)
    };

    if y > 0.13533528323661269189 {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polevl(y2, &P0) / p1evl(y2, &Q0));
        return x * s2pi;
    }

    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &P1) / p1evl(z, &Q1)
    } else {
        z * polevl(z, &P2) / p1evl(z, &Q2)
    };
    let x = x0 - x1;

    if code != 0 { -x } else { x }
}

fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

// ---------------------------------------------------------------------------
// Libm function IDs in the JIT module
// ---------------------------------------------------------------------------

struct LibmIds {
    sin: FuncId,
    cos: FuncId,
    atan2: FuncId,
    sqrt: FuncId,
    fabs: FuncId,
    fmod: FuncId,
    acos: FuncId,
    erf_inv: FuncId,
    log: FuncId,
    exp: FuncId,
    pow: FuncId,
    tanh: FuncId,
    tan: FuncId,
}

fn declare_libm_functions(
    jit_module: &mut JITModule,
    call_conv: CallConv,
) -> Result<LibmIds, String> {
    let mut mk = |name: &str, n_params: usize, n_rets: usize| -> Result<FuncId, String> {
        let mut sig = jit_module.make_signature();
        sig.call_conv = call_conv;
        for _ in 0..n_params {
            sig.params.push(AbiParam::new(types::F64));
        }
        for _ in 0..n_rets {
            sig.returns.push(AbiParam::new(types::F64));
        }
        jit_module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| format!("declare libm {name}: {e}"))
    };

    Ok(LibmIds {
        sin: mk("sin", 1, 1)?,
        cos: mk("cos", 1, 1)?,
        atan2: mk("atan2", 2, 1)?,
        sqrt: mk("sqrt", 1, 1)?,
        fabs: mk("fabs", 1, 1)?,
        fmod: mk("fmod", 2, 1)?,
        acos: mk("acos", 1, 1)?,
        erf_inv: mk("erf_inv_impl", 1, 1)?,
        log: mk("log", 1, 1)?,
        exp: mk("exp", 1, 1)?,
        pow: mk("pow", 2, 1)?,
        tanh: mk("tanh", 1, 1)?,
        tan: mk("tan", 1, 1)?,
    })
}

// ---------------------------------------------------------------------------
// Tensor runtime function IDs
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct TensorRtIds {
    add_f64: FuncId,
    sub_f64: FuncId,
    mul_f64: FuncId,
    div_f64: FuncId,
    max_f64: FuncId,
    min_f64: FuncId,
    pow_f64: FuncId,
    rem_f64: FuncId,
    neg_f64: FuncId,
    sqrt_f64: FuncId,
    abs_f64: FuncId,
    floor_f64: FuncId,
    sin_f64: FuncId,
    cos_f64: FuncId,
    exp_f64: FuncId,
    log_f64: FuncId,
    tanh_f64: FuncId,
    sign_f64: FuncId,
    tan_f64: FuncId,
    transpose_nd_f64: FuncId,
    cmp_eq_f64: FuncId,
    cmp_lt_f64: FuncId,
    cmp_le_f64: FuncId,
    cmp_gt_f64: FuncId,
    cmp_ge_f64: FuncId,
    cmp_ne_f64: FuncId,
    cmp_eq_i64: FuncId,
    cmp_lt_i64: FuncId,
    select_f64: FuncId,
    select_i64: FuncId,
    convert_i64_to_f64: FuncId,
    convert_f64_to_i64: FuncId,
    convert_i1_to_f64: FuncId,
    broadcast_f64: FuncId,
    broadcast_i64: FuncId,
    broadcast_nd_f64: FuncId,
    memcpy: FuncId,
    transpose_f64: FuncId,
    reduce_sum_f64: FuncId,
    reduce_max_f64: FuncId,
    reduce_min_f64: FuncId,
    scatter_f64: FuncId,
    gather_f64: FuncId,
    gather_nd_f64: FuncId,
    matmul_f64: FuncId,
    slice_f64: FuncId,
    pad_f64: FuncId,
    concat_nd_f64: FuncId,
    dynamic_slice_f64: FuncId,
    dynamic_update_slice_f64: FuncId,
    iota_nd_i64: FuncId,
    iota_nd_f64: FuncId,
    add_i64: FuncId,
    sub_i64: FuncId,
    mul_i64: FuncId,
    widen_i32_to_i64: FuncId,
    convert_i64_to_i32: FuncId,
    select_i32: FuncId,
    broadcast_i32: FuncId,
    add_i32: FuncId,
    sub_i32: FuncId,
    div_i32: FuncId,
    div_ui32: FuncId,
    cmp_lt_i32: FuncId,
    convert_f64_to_i32: FuncId,
    convert_i1_to_i32: FuncId,
    convert_i32_to_f64: FuncId,
    convert_f64_to_f32: FuncId,
    convert_f32_to_f64: FuncId,
}

fn register_tensor_rt_symbols(jit_builder: &mut JITBuilder) {
    jit_builder.symbol("__trt_add_f64", tensor_rt::tensor_add_f64 as *const u8);
    jit_builder.symbol("__trt_sub_f64", tensor_rt::tensor_sub_f64 as *const u8);
    jit_builder.symbol("__trt_mul_f64", tensor_rt::tensor_mul_f64 as *const u8);
    jit_builder.symbol("__trt_div_f64", tensor_rt::tensor_div_f64 as *const u8);
    jit_builder.symbol("__trt_max_f64", tensor_rt::tensor_max_f64 as *const u8);
    jit_builder.symbol("__trt_min_f64", tensor_rt::tensor_min_f64 as *const u8);
    jit_builder.symbol("__trt_pow_f64", tensor_rt::tensor_pow_f64 as *const u8);
    jit_builder.symbol("__trt_rem_f64", tensor_rt::tensor_rem_f64 as *const u8);
    jit_builder.symbol("__trt_neg_f64", tensor_rt::tensor_neg_f64 as *const u8);
    jit_builder.symbol("__trt_sqrt_f64", tensor_rt::tensor_sqrt_f64 as *const u8);
    jit_builder.symbol("__trt_abs_f64", tensor_rt::tensor_abs_f64 as *const u8);
    jit_builder.symbol("__trt_floor_f64", tensor_rt::tensor_floor_f64 as *const u8);
    jit_builder.symbol("__trt_sin_f64", tensor_rt::tensor_sin_f64 as *const u8);
    jit_builder.symbol("__trt_cos_f64", tensor_rt::tensor_cos_f64 as *const u8);
    jit_builder.symbol("__trt_exp_f64", tensor_rt::tensor_exp_f64 as *const u8);
    jit_builder.symbol("__trt_log_f64", tensor_rt::tensor_log_f64 as *const u8);
    jit_builder.symbol("__trt_sign_f64", tensor_rt::tensor_sign_f64 as *const u8);
    jit_builder.symbol("__trt_tan_f64", tensor_rt::tensor_tan_f64 as *const u8);
    jit_builder.symbol(
        "__trt_transpose_nd_f64",
        tensor_rt::tensor_transpose_nd_f64 as *const u8,
    );
    jit_builder.symbol("__trt_tanh_f64", tensor_rt::tensor_tanh_f64 as *const u8);
    jit_builder.symbol(
        "__trt_cmp_eq_f64",
        tensor_rt::tensor_cmp_eq_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cmp_lt_f64",
        tensor_rt::tensor_cmp_lt_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cmp_le_f64",
        tensor_rt::tensor_cmp_le_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cmp_gt_f64",
        tensor_rt::tensor_cmp_gt_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cmp_ge_f64",
        tensor_rt::tensor_cmp_ge_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cmp_ne_f64",
        tensor_rt::tensor_cmp_ne_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cmp_eq_i64",
        tensor_rt::tensor_cmp_eq_i64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cmp_lt_i64",
        tensor_rt::tensor_cmp_lt_i64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_select_f64",
        tensor_rt::tensor_select_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_select_i64",
        tensor_rt::tensor_select_i64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_i64_f64",
        tensor_rt::tensor_convert_i64_to_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_f64_i64",
        tensor_rt::tensor_convert_f64_to_i64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_i1_f64",
        tensor_rt::tensor_convert_i1_to_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_bcast_f64",
        tensor_rt::tensor_broadcast_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_bcast_i64",
        tensor_rt::tensor_broadcast_i64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_bcast_nd_f64",
        tensor_rt::tensor_broadcast_nd_f64 as *const u8,
    );
    jit_builder.symbol("__trt_memcpy", tensor_rt::tensor_memcpy as *const u8);
    jit_builder.symbol(
        "__trt_transpose_f64",
        tensor_rt::tensor_transpose_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_reduce_sum_f64",
        tensor_rt::tensor_reduce_sum_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_reduce_max_f64",
        tensor_rt::tensor_reduce_max_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_reduce_min_f64",
        tensor_rt::tensor_reduce_min_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_scatter_f64",
        tensor_rt::tensor_scatter_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_gather_f64",
        tensor_rt::tensor_gather_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_gather_nd_f64",
        tensor_rt::tensor_gather_nd_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_matmul_f64",
        tensor_rt::tensor_matmul_f64 as *const u8,
    );
    jit_builder.symbol("__trt_slice_f64", tensor_rt::tensor_slice_f64 as *const u8);
    jit_builder.symbol("__trt_pad_f64", tensor_rt::tensor_pad_f64 as *const u8);
    jit_builder.symbol(
        "__trt_concat_nd_f64",
        tensor_rt::tensor_concat_nd_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_dyn_slice_f64",
        tensor_rt::tensor_dynamic_slice_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_dyn_upd_slice_f64",
        tensor_rt::tensor_dynamic_update_slice_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_iota_nd_i64",
        tensor_rt::tensor_iota_nd_i64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_iota_nd_f64",
        tensor_rt::tensor_iota_nd_f64 as *const u8,
    );
    jit_builder.symbol("__trt_add_i64", tensor_rt::tensor_add_i64 as *const u8);
    jit_builder.symbol("__trt_sub_i64", tensor_rt::tensor_sub_i64 as *const u8);
    jit_builder.symbol("__trt_mul_i64", tensor_rt::tensor_mul_i64 as *const u8);
    jit_builder.symbol(
        "__trt_select_i32",
        tensor_rt::tensor_select_i32 as *const u8,
    );
    jit_builder.symbol(
        "__trt_bcast_i32",
        tensor_rt::tensor_broadcast_i32 as *const u8,
    );
    jit_builder.symbol("__trt_add_i32", tensor_rt::tensor_add_i32 as *const u8);
    jit_builder.symbol(
        "__trt_widen_i32_i64",
        tensor_rt::tensor_widen_i32_to_i64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_i64_i32",
        tensor_rt::tensor_convert_i64_to_i32 as *const u8,
    );
    jit_builder.symbol("__trt_sub_i32", tensor_rt::tensor_sub_i32 as *const u8);
    jit_builder.symbol("__trt_div_i32", tensor_rt::tensor_div_i32 as *const u8);
    jit_builder.symbol("__trt_div_ui32", tensor_rt::tensor_div_ui32 as *const u8);
    jit_builder.symbol(
        "__trt_cmp_lt_i32",
        tensor_rt::tensor_cmp_lt_i32 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_f64_i32",
        tensor_rt::tensor_convert_f64_to_i32 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_i1_i32",
        tensor_rt::tensor_convert_i1_to_i32 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_i32_f64",
        tensor_rt::tensor_convert_i32_to_f64 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_f64_f32",
        tensor_rt::tensor_convert_f64_to_f32 as *const u8,
    );
    jit_builder.symbol(
        "__trt_cvt_f32_f64",
        tensor_rt::tensor_convert_f32_to_f64 as *const u8,
    );
}

fn declare_tensor_rt_functions(
    jit_module: &mut JITModule,
    call_conv: CallConv,
) -> Result<TensorRtIds, String> {
    let pt = ptr_type();
    let i64t = types::I64;
    let f64t = types::F64;
    let decl =
        |m: &mut JITModule, name: &str, params: &[Type], rets: &[Type]| -> Result<FuncId, String> {
            let mut sig = m.make_signature();
            sig.call_conv = call_conv;
            for &t in params {
                sig.params.push(AbiParam::new(t));
            }
            for &t in rets {
                sig.returns.push(AbiParam::new(t));
            }
            m.declare_function(name, Linkage::Import, &sig)
                .map_err(|e| format!("declare trt {name}: {e}"))
        };
    let binop = |m: &mut JITModule, n: &str| decl(m, n, &[pt, pt, pt, i64t], &[]);
    let unop = |m: &mut JITModule, n: &str| decl(m, n, &[pt, pt, i64t], &[]);
    let cmp = |m: &mut JITModule, n: &str| decl(m, n, &[pt, pt, pt, i64t], &[]);
    Ok(TensorRtIds {
        add_f64: binop(jit_module, "__trt_add_f64")?,
        sub_f64: binop(jit_module, "__trt_sub_f64")?,
        mul_f64: binop(jit_module, "__trt_mul_f64")?,
        div_f64: binop(jit_module, "__trt_div_f64")?,
        max_f64: binop(jit_module, "__trt_max_f64")?,
        min_f64: binop(jit_module, "__trt_min_f64")?,
        pow_f64: binop(jit_module, "__trt_pow_f64")?,
        rem_f64: binop(jit_module, "__trt_rem_f64")?,
        neg_f64: unop(jit_module, "__trt_neg_f64")?,
        sqrt_f64: unop(jit_module, "__trt_sqrt_f64")?,
        abs_f64: unop(jit_module, "__trt_abs_f64")?,
        floor_f64: unop(jit_module, "__trt_floor_f64")?,
        sin_f64: unop(jit_module, "__trt_sin_f64")?,
        cos_f64: unop(jit_module, "__trt_cos_f64")?,
        exp_f64: unop(jit_module, "__trt_exp_f64")?,
        log_f64: unop(jit_module, "__trt_log_f64")?,
        tanh_f64: unop(jit_module, "__trt_tanh_f64")?,
        sign_f64: unop(jit_module, "__trt_sign_f64")?,
        tan_f64: unop(jit_module, "__trt_tan_f64")?,
        transpose_nd_f64: decl(
            jit_module,
            "__trt_transpose_nd_f64",
            &[pt, pt, i64t, pt, pt, i64t],
            &[],
        )?,
        cmp_eq_f64: cmp(jit_module, "__trt_cmp_eq_f64")?,
        cmp_lt_f64: cmp(jit_module, "__trt_cmp_lt_f64")?,
        cmp_le_f64: cmp(jit_module, "__trt_cmp_le_f64")?,
        cmp_gt_f64: cmp(jit_module, "__trt_cmp_gt_f64")?,
        cmp_ge_f64: cmp(jit_module, "__trt_cmp_ge_f64")?,
        cmp_ne_f64: cmp(jit_module, "__trt_cmp_ne_f64")?,
        cmp_eq_i64: cmp(jit_module, "__trt_cmp_eq_i64")?,
        cmp_lt_i64: cmp(jit_module, "__trt_cmp_lt_i64")?,
        select_f64: decl(jit_module, "__trt_select_f64", &[pt, pt, pt, pt, i64t], &[])?,
        select_i64: decl(jit_module, "__trt_select_i64", &[pt, pt, pt, pt, i64t], &[])?,
        convert_i64_to_f64: unop(jit_module, "__trt_cvt_i64_f64")?,
        convert_f64_to_i64: unop(jit_module, "__trt_cvt_f64_i64")?,
        convert_i1_to_f64: unop(jit_module, "__trt_cvt_i1_f64")?,
        broadcast_f64: decl(jit_module, "__trt_bcast_f64", &[pt, f64t, i64t], &[])?,
        broadcast_i64: decl(jit_module, "__trt_bcast_i64", &[pt, i64t, i64t], &[])?,
        broadcast_nd_f64: decl(
            jit_module,
            "__trt_bcast_nd_f64",
            &[pt, pt, i64t, i64t, pt, i64t, pt, i64t, pt],
            &[],
        )?,
        memcpy: decl(jit_module, "__trt_memcpy", &[pt, pt, i64t], &[])?,
        transpose_f64: decl(
            jit_module,
            "__trt_transpose_f64",
            &[pt, pt, i64t, i64t],
            &[],
        )?,
        reduce_sum_f64: decl(
            jit_module,
            "__trt_reduce_sum_f64",
            &[pt, pt, i64t, i64t],
            &[],
        )?,
        reduce_max_f64: decl(
            jit_module,
            "__trt_reduce_max_f64",
            &[pt, pt, i64t, i64t],
            &[],
        )?,
        reduce_min_f64: decl(
            jit_module,
            "__trt_reduce_min_f64",
            &[pt, pt, i64t, i64t],
            &[],
        )?,
        scatter_f64: decl(
            jit_module,
            "__trt_scatter_f64",
            &[pt, pt, i64t, pt, pt, i64t, i64t],
            &[],
        )?,
        gather_f64: decl(
            jit_module,
            "__trt_gather_f64",
            &[pt, pt, i64t, pt, i64t, i64t],
            &[],
        )?,
        gather_nd_f64: decl(
            jit_module,
            "__trt_gather_nd_f64",
            &[pt, pt, i64t, pt, i64t, i64t, pt, i64t, pt, pt, i64t],
            &[],
        )?,
        matmul_f64: decl(
            jit_module,
            "__trt_matmul_f64",
            &[pt, pt, pt, i64t, i64t, i64t],
            &[],
        )?,
        slice_f64: decl(
            jit_module,
            "__trt_slice_f64",
            &[pt, pt, i64t, i64t, pt, i64t, pt, pt],
            &[],
        )?,
        pad_f64: decl(
            jit_module,
            "__trt_pad_f64",
            &[pt, pt, i64t, i64t, f64t, pt, pt, i64t, pt],
            &[],
        )?,
        concat_nd_f64: decl(
            jit_module,
            "__trt_concat_nd_f64",
            &[pt, i64t, pt, i64t, pt, i64t, pt, pt, i64t, i64t, i64t],
            &[],
        )?,
        dynamic_slice_f64: decl(
            jit_module,
            "__trt_dyn_slice_f64",
            &[pt, pt, i64t, i64t, pt, i64t, pt, pt],
            &[],
        )?,
        dynamic_update_slice_f64: decl(
            jit_module,
            "__trt_dyn_upd_slice_f64",
            &[pt, pt, pt, i64t, i64t, pt, i64t, pt, pt],
            &[],
        )?,
        iota_nd_i64: decl(
            jit_module,
            "__trt_iota_nd_i64",
            &[pt, i64t, pt, i64t, i64t],
            &[],
        )?,
        iota_nd_f64: decl(
            jit_module,
            "__trt_iota_nd_f64",
            &[pt, i64t, pt, i64t, i64t],
            &[],
        )?,
        add_i64: binop(jit_module, "__trt_add_i64")?,
        sub_i64: binop(jit_module, "__trt_sub_i64")?,
        mul_i64: binop(jit_module, "__trt_mul_i64")?,
        widen_i32_to_i64: unop(jit_module, "__trt_widen_i32_i64")?,
        convert_i64_to_i32: unop(jit_module, "__trt_cvt_i64_i32")?,
        select_i32: decl(jit_module, "__trt_select_i32", &[pt, pt, pt, pt, i64t], &[])?,
        broadcast_i32: decl(jit_module, "__trt_bcast_i32", &[pt, types::I32, i64t], &[])?,
        add_i32: binop(jit_module, "__trt_add_i32")?,
        sub_i32: binop(jit_module, "__trt_sub_i32")?,
        div_i32: binop(jit_module, "__trt_div_i32")?,
        div_ui32: binop(jit_module, "__trt_div_ui32")?,
        cmp_lt_i32: cmp(jit_module, "__trt_cmp_lt_i32")?,
        convert_f64_to_i32: unop(jit_module, "__trt_cvt_f64_i32")?,
        convert_i1_to_i32: unop(jit_module, "__trt_cvt_i1_i32")?,
        convert_i32_to_f64: unop(jit_module, "__trt_cvt_i32_f64")?,
        convert_f64_to_f32: unop(jit_module, "__trt_cvt_f64_f32")?,
        convert_f32_to_f64: unop(jit_module, "__trt_cvt_f32_f64")?,
    })
}

// ---------------------------------------------------------------------------
// Module compilation entry point
// ---------------------------------------------------------------------------

pub fn compile_module(ir_module: &crate::ir::Module) -> Result<CompiledModule, String> {
    compile_module_with_config(ir_module, CompileConfig::default())
}

pub fn compile_module_with_config(
    ir_module: &crate::ir::Module,
    config: CompileConfig,
) -> Result<CompiledModule, String> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder = cranelift_native::builder().map_err(|e| format!("native ISA: {e}"))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| format!("ISA finish: {e}"))?;

    let call_conv = isa.default_call_conv();
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    jit_builder.symbol("sin", libc_sin as *const u8);
    jit_builder.symbol("cos", libc_cos as *const u8);
    jit_builder.symbol("atan2", libc_atan2 as *const u8);
    jit_builder.symbol("sqrt", libc_sqrt as *const u8);
    jit_builder.symbol("fabs", libc_fabs as *const u8);
    jit_builder.symbol("fmod", libc_fmod as *const u8);
    jit_builder.symbol("acos", libc_acos as *const u8);
    jit_builder.symbol("log", libc_log as *const u8);
    jit_builder.symbol("exp", libc_exp as *const u8);
    jit_builder.symbol("pow", libc_pow as *const u8);
    jit_builder.symbol("tanh", libc_tanh as *const u8);
    jit_builder.symbol("tan", libc_tan as *const u8);
    jit_builder.symbol("erf_inv_impl", erf_inv_scalar as *const u8);
    jit_builder.symbol("__cranelift_svd", cranelift_svd as *const u8);
    jit_builder.symbol("__cranelift_lu", cranelift_lu as *const u8);
    jit_builder.symbol("__cranelift_trsm", cranelift_trsm as *const u8);
    jit_builder.symbol("__cranelift_cholesky", cranelift_cholesky as *const u8);
    jit_builder.symbol("__cranelift_qr", cranelift_qr as *const u8);
    jit_builder.symbol("__cranelift_orgqr", cranelift_orgqr as *const u8);
    jit_builder.symbol("__cranelift_syevd", cranelift_syevd as *const u8);
    register_tensor_rt_symbols(&mut jit_builder);

    let mut jit_module = JITModule::new(jit_builder);
    let func_abis = classify_all_functions(ir_module);
    let func_ids = declare_all_functions(ir_module, &mut jit_module, call_conv, &func_abis)?;
    let libm_ids = declare_libm_functions(&mut jit_module, call_conv)?;
    let trt_ids = declare_tensor_rt_functions(&mut jit_module, call_conv)?;

    for func_def in &ir_module.functions {
        let fid = func_ids[&func_def.name];
        let abi = func_abis
            .get(&func_def.name)
            .copied()
            .unwrap_or(FuncAbi::Scalar);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            define_function(
                &mut jit_module,
                func_def,
                ir_module,
                &func_ids,
                &func_abis,
                &libm_ids,
                &trt_ids,
                abi,
                fid,
                config,
            )
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(format!("define {}: {e}", func_def.name)),
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                return Err(format!("panic compiling {}: {msg}", func_def.name));
            }
        }
    }

    jit_module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {e}"))?;

    let main_fn_id = *func_ids.get("main").ok_or("no main function")?;

    Ok(CompiledModule {
        module: jit_module,
        main_fn_id,
    })
}

fn declare_all_functions(
    ir_module: &crate::ir::Module,
    jit_module: &mut JITModule,
    call_conv: CallConv,
    func_abis: &HashMap<String, FuncAbi>,
) -> Result<HashMap<String, FuncId>, String> {
    let mut ids = HashMap::new();
    for func_def in &ir_module.functions {
        let mut sig = jit_module.make_signature();
        sig.call_conv = call_conv;
        let abi = func_abis
            .get(&func_def.name)
            .copied()
            .unwrap_or(FuncAbi::Scalar);

        if func_def.name == "main" {
            sig.params.push(AbiParam::new(ptr_type()));
            sig.params.push(AbiParam::new(ptr_type()));
        } else if abi == FuncAbi::Pointer {
            for _ in &func_def.params {
                sig.params.push(AbiParam::new(ptr_type()));
            }
            sig.params.push(AbiParam::new(ptr_type()));
        } else {
            for (_vid, ty) in &func_def.params {
                add_tensor_params(&mut sig, ty);
            }
            if needs_sret(&func_def.result_types) {
                sig.params.push(AbiParam::new(ptr_type()));
            } else {
                for ty in &func_def.result_types {
                    add_tensor_returns(&mut sig, ty);
                }
            }
        }

        let linkage = if func_def.is_public {
            Linkage::Export
        } else {
            Linkage::Local
        };
        let fid = jit_module
            .declare_function(&func_def.name, linkage, &sig)
            .map_err(|e| format!("declare {}: {e}", func_def.name))?;
        ids.insert(func_def.name.clone(), fid);
    }
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Function definition (body lowering + ABI handling)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn define_function(
    jit_module: &mut JITModule,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    func_abis: &HashMap<String, FuncAbi>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    abi: FuncAbi,
    fid: FuncId,
    config: CompileConfig,
) -> Result<(), String> {
    let mut ctx = jit_module.make_context();
    let mut func_ctx = FunctionBuilderContext::new();

    ctx.func.signature = jit_module
        .declarations()
        .get_function_decl(fid)
        .signature
        .clone();
    ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, fid.as_u32());

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let block_params: Vec<Value> = builder.block_params(entry_block).to_vec();
        let mut value_map: HashMap<ValueId, TensorVals> = HashMap::new();
        let mut type_map: HashMap<ValueId, TensorType> = HashMap::new();

        if func_def.name == "main" && config.force_pointer_abi_main {
            lower_main_body_mem(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        } else if func_def.name == "main" {
            lower_main_body(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        } else if abi == FuncAbi::Pointer {
            lower_pointer_body(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        } else {
            lower_callee_body(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &block_params,
                &mut value_map,
                &mut type_map,
            )?;
        }

        builder.finalize();
    }

    jit_module
        .define_function(fid, &mut ctx)
        .map_err(|e| format!("define {}: {:?}", func_def.name, e))?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_main_body(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    let inputs_ptr = block_params[0];
    let outputs_ptr = block_params[1];

    for (i, (vid, ty)) in func_def.params.iter().enumerate() {
        let n = ty.num_elements();
        let ct = cranelift_type_for(ty.element_type);
        let buf_ptr =
            builder
                .ins()
                .load(ptr_type(), MemFlags::trusted(), inputs_ptr, (i * 8) as i32);
        let mut vals = Vec::with_capacity(n);
        for j in 0..n {
            let offset = (j * ty.element_type.byte_size()) as i32;
            let v = builder.ins().load(ct, MemFlags::trusted(), buf_ptr, offset);
            vals.push(v);
        }
        value_map.insert(*vid, vals);
        type_map.insert(*vid, ty.clone());
    }

    lower_body(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last()
        && let Instruction::Return { operands } = &ret_instr.instr
    {
        for (i, (vid, ty)) in operands
            .iter()
            .zip(func_def.result_types.iter())
            .enumerate()
        {
            let vals = get_vals(value_map, vid)?;
            let buf_ptr =
                builder
                    .ins()
                    .load(ptr_type(), MemFlags::trusted(), outputs_ptr, (i * 8) as i32);
            for (j, &v) in vals.iter().enumerate() {
                let offset = (j * ty.element_type.byte_size()) as i32;
                builder.ins().store(MemFlags::trusted(), v, buf_ptr, offset);
            }
        }
    }
    builder.ins().return_(&[]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory-backed (pointer ABI) body lowering
// All values in value_map are vec![ptr] -- a single i64 pointer to a stack buffer.
// All ops dispatch to tensor_rt functions.
// ---------------------------------------------------------------------------

fn alloc_slot(builder: &mut FunctionBuilder, byte_size: usize) -> Value {
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        byte_size as u32,
        3,
    ));
    builder.ins().stack_addr(ptr_type(), ss, 0)
}

fn store_i64_array(builder: &mut FunctionBuilder, vals: &[i64]) -> Value {
    let ptr = alloc_slot(builder, vals.len() * 8);
    for (i, &v) in vals.iter().enumerate() {
        let cv = builder.ins().iconst(types::I64, v);
        builder
            .ins()
            .store(MemFlags::trusted(), cv, ptr, (i * 8) as i32);
    }
    ptr
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn lower_pointer_body(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    for (i, (vid, ty)) in func_def.params.iter().enumerate() {
        let param_ptr = block_params[i];
        value_map.insert(*vid, vec![param_ptr]);
        type_map.insert(*vid, ty.clone());
    }

    let out_ptr = block_params[func_def.params.len()];

    lower_body_mem(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last()
        && let Instruction::Return { operands } = &ret_instr.instr
    {
        let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
        let mut byte_offset = 0i64;
        for (vid, ty) in operands.iter().zip(func_def.result_types.iter()) {
            let vals = value_map
                .get(vid)
                .ok_or_else(|| format!("ptr body: missing return {:?}", vid))?;
            let dst = builder.ins().iadd_imm(out_ptr, byte_offset);
            let nb = builder.ins().iconst(types::I64, ty.byte_size() as i64);
            builder.ins().call(memcpy_ref, &[dst, vals[0], nb]);
            byte_offset += ty.byte_size() as i64;
        }
    }
    builder.ins().return_(&[]);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_main_body_mem(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    let inputs_ptr = block_params[0];
    let outputs_ptr = block_params[1];

    for (i, (vid, ty)) in func_def.params.iter().enumerate() {
        let buf_ptr =
            builder
                .ins()
                .load(ptr_type(), MemFlags::trusted(), inputs_ptr, (i * 8) as i32);
        value_map.insert(*vid, vec![buf_ptr]);
        type_map.insert(*vid, ty.clone());
    }

    lower_body_mem(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last()
        && let Instruction::Return { operands } = &ret_instr.instr
    {
        let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
        for (i, (vid, ty)) in operands
            .iter()
            .zip(func_def.result_types.iter())
            .enumerate()
        {
            let vals = value_map
                .get(vid)
                .ok_or_else(|| format!("main_mem: missing return {:?}", vid))?;
            let buf_ptr =
                builder
                    .ins()
                    .load(ptr_type(), MemFlags::trusted(), outputs_ptr, (i * 8) as i32);
            let nb = builder.ins().iconst(types::I64, ty.byte_size() as i64);
            builder.ins().call(memcpy_ref, &[buf_ptr, vals[0], nb]);
        }
    }
    builder.ins().return_(&[]);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_body_mem(
    builder: &mut FunctionBuilder,
    body: &[InstrResult],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    for ir in body {
        if matches!(ir.instr, Instruction::Return { .. }) {
            break;
        }
        let result_types: Vec<TensorType> = ir.values.iter().map(|(_, t)| t.clone()).collect();
        let result_vals = lower_instruction_mem(
            builder,
            &ir.instr,
            &result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
        )?;
        for (i, (vid, ty)) in ir.values.iter().enumerate() {
            if i < result_vals.len() {
                value_map.insert(*vid, result_vals[i].clone());
                type_map.insert(*vid, ty.clone());
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_instruction_mem(
    builder: &mut FunctionBuilder,
    instr: &Instruction,
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<Vec<TensorVals>, String> {
    let rt = result_types
        .first()
        .cloned()
        .unwrap_or(TensorType::scalar(ElementType::F64));
    let n = rt.num_elements();
    let elem_sz = rt.element_type.byte_size();

    let get = |vid: &ValueId| -> Result<Value, String> {
        value_map
            .get(vid)
            .and_then(|v| v.first().copied())
            .ok_or_else(|| format!("mem: missing value {:?}", vid))
    };

    let trt_call = |builder: &mut FunctionBuilder,
                    jit_module: &mut JITModule,
                    func_id: FuncId,
                    args: &[Value]|
     -> Value {
        let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
        builder.ins().call(func_ref, args);
        args[0]
    };

    match instr {
        Instruction::Constant { value } => {
            let dst = alloc_slot(builder, n * elem_sz);
            match value {
                ConstantValue::DenseScalar(sv) | ConstantValue::DenseSplat(sv, _) => {
                    let scalar = scalar_to_cranelift(builder, sv, rt.element_type);
                    let fid = match rt.element_type {
                        ElementType::F64 | ElementType::F32 => trt_ids.broadcast_f64,
                        ElementType::I32 | ElementType::UI32 => trt_ids.broadcast_i32,
                        _ => trt_ids.broadcast_i64,
                    };
                    let n_val = builder.ins().iconst(types::I64, n as i64);
                    trt_call(builder, jit_module, fid, &[dst, scalar, n_val]);
                }
                ConstantValue::DenseArray(arr) => {
                    let mut bytes = Vec::with_capacity(n * elem_sz);
                    for sv in arr {
                        match rt.element_type {
                            ElementType::F64 => bytes.extend_from_slice(&sv.as_f64().to_ne_bytes()),
                            ElementType::F32 => {
                                bytes.extend_from_slice(&(sv.as_f64() as f32).to_ne_bytes())
                            }
                            ElementType::I64 | ElementType::UI64 => {
                                bytes.extend_from_slice(&sv.as_i64().to_ne_bytes())
                            }
                            ElementType::I32 | ElementType::UI32 => {
                                bytes.extend_from_slice(&(sv.as_i64() as i32).to_ne_bytes())
                            }
                            ElementType::I1 => bytes.push(if sv.as_i64() != 0 { 1 } else { 0 }),
                        }
                    }
                    let data_id = jit_module
                        .declare_anonymous_data(false, false)
                        .map_err(|e| format!("declare data: {e}"))?;
                    let mut desc = DataDescription::new();
                    desc.define(bytes.into_boxed_slice());
                    desc.set_align(8);
                    jit_module
                        .define_data(data_id, &desc)
                        .map_err(|e| format!("define data: {e}"))?;
                    let gv = jit_module.declare_data_in_func(data_id, builder.func);
                    let src = builder.ins().global_value(ptr_type(), gv);
                    let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                    let nb = builder.ins().iconst(types::I64, (n * elem_sz) as i64);
                    builder.ins().call(memcpy_ref, &[dst, src, nb]);
                }
            }
            Ok(vec![vec![dst]])
        }

        Instruction::Add { lhs, rhs } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.add_f64,
                ElementType::I32 | ElementType::UI32 => trt_ids.add_i32,
                _ => trt_ids.add_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Subtract { lhs, rhs } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.sub_f64,
                ElementType::I32 | ElementType::UI32 => trt_ids.sub_i32,
                _ => trt_ids.sub_i64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Multiply { lhs, rhs } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = if is_float(rt.element_type) {
                trt_ids.mul_f64
            } else {
                trt_ids.mul_i64
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Divide { lhs, rhs } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.div_f64,
                ElementType::UI32 => trt_ids.div_ui32,
                ElementType::I32 => trt_ids.div_i32,
                _ => trt_ids.div_f64,
            };
            trt_call(
                builder,
                jit_module,
                fid,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Maximum { lhs, rhs } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.max_f64,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Minimum { lhs, rhs } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.min_f64,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Power { lhs, rhs } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.pow_f64,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Remainder { lhs, rhs } if is_float(rt.element_type) => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.rem_f64,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::Negate { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.neg_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Sqrt { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.sqrt_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Abs { operand } if is_float(rt.element_type) => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.abs_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Floor { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.floor_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Sine { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.sin_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Cosine { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.cos_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Exponential { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.exp_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Log { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.log_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Tanh { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.tanh_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::Reshape { operand } => Ok(vec![vec![get(operand)?]]),

        Instruction::Compare {
            lhs,
            rhs,
            direction,
            compare_type,
        } => {
            let l_ty = type_map
                .get(lhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let use_float = is_float(l_ty.element_type)
                || matches!(compare_type, CompareType::Float | CompareType::TotalOrder);
            let func_id = if use_float {
                match direction {
                    CompareDirection::Eq => trt_ids.cmp_eq_f64,
                    CompareDirection::Ne => trt_ids.cmp_ne_f64,
                    CompareDirection::Lt => trt_ids.cmp_lt_f64,
                    CompareDirection::Le => trt_ids.cmp_le_f64,
                    CompareDirection::Gt => trt_ids.cmp_gt_f64,
                    CompareDirection::Ge => trt_ids.cmp_ge_f64,
                }
            } else if matches!(l_ty.element_type, ElementType::I32 | ElementType::UI32) {
                match direction {
                    CompareDirection::Lt => trt_ids.cmp_lt_i32,
                    _ => trt_ids.cmp_eq_i64,
                }
            } else {
                match direction {
                    CompareDirection::Eq => trt_ids.cmp_eq_i64,
                    CompareDirection::Lt => trt_ids.cmp_lt_i64,
                    _ => trt_ids.cmp_eq_i64,
                }
            };
            let dst = alloc_slot(builder, n);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
            builder
                .ins()
                .call(func_ref, &[dst, get(lhs)?, get(rhs)?, n_val]);
            Ok(vec![vec![dst]])
        }

        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let fid = match rt.element_type {
                ElementType::F64 | ElementType::F32 => trt_ids.select_f64,
                ElementType::I32 | ElementType::UI32 => trt_ids.select_i32,
                _ => trt_ids.select_i64,
            };
            let func_ref = jit_module.declare_func_in_func(fid, builder.func);
            builder.ins().call(
                func_ref,
                &[dst, get(cond)?, get(on_true)?, get(on_false)?, n_val],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::Convert { operand } => {
            let src_ty = type_map
                .get(operand)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            if src_ty.element_type == rt.element_type
                || (matches!(src_ty.element_type, ElementType::I32 | ElementType::UI32)
                    && matches!(rt.element_type, ElementType::I32 | ElementType::UI32))
                || (matches!(src_ty.element_type, ElementType::I64 | ElementType::UI64)
                    && matches!(rt.element_type, ElementType::I64 | ElementType::UI64))
            {
                return Ok(vec![vec![get(operand)?]]);
            }
            let func_id = match (src_ty.element_type, rt.element_type) {
                (ElementType::I64 | ElementType::UI64, ElementType::F64) => {
                    trt_ids.convert_i64_to_f64
                }
                (ElementType::F64, ElementType::I64) => trt_ids.convert_f64_to_i64,
                (ElementType::I1, ElementType::F64) => trt_ids.convert_i1_to_f64,
                (ElementType::F64, ElementType::I32 | ElementType::UI32) => {
                    trt_ids.convert_f64_to_i32
                }
                (ElementType::I64, ElementType::I32 | ElementType::UI32) => {
                    trt_ids.convert_i64_to_i32
                }
                (ElementType::I1, ElementType::I32 | ElementType::UI32) => {
                    trt_ids.convert_i1_to_i32
                }
                (ElementType::I32 | ElementType::UI32, ElementType::F64) => {
                    trt_ids.convert_i32_to_f64
                }
                (ElementType::F64, ElementType::F32) => trt_ids.convert_f64_to_f32,
                (ElementType::F32, ElementType::F64) => trt_ids.convert_f32_to_f64,
                _ => {
                    return Err(format!(
                        "mem: unsupported convert {:?} -> {:?}",
                        src_ty.element_type, rt.element_type
                    ));
                }
            };
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
            builder.ins().call(func_ref, &[dst, get(operand)?, n_val]);
            Ok(vec![vec![dst]])
        }

        Instruction::BroadcastInDim {
            operand,
            broadcast_dims,
        } => {
            let src_ty = type_map
                .get(operand)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let src_n = src_ty.num_elements();
            let dst = alloc_slot(builder, n * elem_sz);

            if src_n == 1 {
                let src_ptr = get(operand)?;
                let ct = cranelift_type_for(src_ty.element_type);
                let scalar = builder.ins().load(ct, MemFlags::trusted(), src_ptr, 0);
                let fid = match rt.element_type {
                    ElementType::F64 | ElementType::F32 => trt_ids.broadcast_f64,
                    ElementType::I32 | ElementType::UI32 => trt_ids.broadcast_i32,
                    _ => trt_ids.broadcast_i64,
                };
                let func_ref = jit_module.declare_func_in_func(fid, builder.func);
                let n_val = builder.ins().iconst(types::I64, n as i64);
                builder.ins().call(func_ref, &[dst, scalar, n_val]);
            } else if src_n == n {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let nb = builder.ins().iconst(types::I64, (n * elem_sz) as i64);
                builder.ins().call(memcpy_ref, &[dst, get(operand)?, nb]);
            } else {
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.broadcast_nd_f64, builder.func);
                let n_dst_v = builder.ins().iconst(types::I64, n as i64);
                let n_src_v = builder.ins().iconst(types::I64, src_n as i64);
                let dst_shape_ptr = store_i64_array(builder, &rt.shape);
                let dst_rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
                let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
                let src_rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
                let bd_ptr = store_i64_array(builder, broadcast_dims);
                builder.ins().call(
                    func_ref,
                    &[
                        dst,
                        get(operand)?,
                        n_dst_v,
                        n_src_v,
                        dst_shape_ptr,
                        dst_rank_v,
                        src_shape_ptr,
                        src_rank_v,
                        bd_ptr,
                    ],
                );
            }
            Ok(vec![vec![dst]])
        }

        Instruction::Transpose {
            operand,
            permutation,
        } if is_float(rt.element_type) => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let dst = alloc_slot(builder, n * elem_sz);
            if rt.rank() == 2 {
                let func_ref = jit_module.declare_func_in_func(trt_ids.transpose_f64, builder.func);
                let rows_v = builder.ins().iconst(types::I64, src_ty.shape[0]);
                let cols_v = builder.ins().iconst(types::I64, src_ty.shape[1]);
                builder
                    .ins()
                    .call(func_ref, &[dst, get(operand)?, rows_v, cols_v]);
            } else {
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.transpose_nd_f64, builder.func);
                let n_val = builder.ins().iconst(types::I64, n as i64);
                let shape_ptr = store_i64_array(builder, &src_ty.shape);
                let perm_ptr = store_i64_array(builder, permutation);
                let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
                builder.ins().call(
                    func_ref,
                    &[dst, get(operand)?, n_val, shape_ptr, perm_ptr, rank_v],
                );
            }
            Ok(vec![vec![dst]])
        }

        Instruction::Slice {
            operand,
            start_indices,
            limit_indices,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let dst = alloc_slot(builder, n * elem_sz);
            let func_ref = jit_module.declare_func_in_func(trt_ids.slice_f64, builder.func);
            let n_dst_v = builder.ins().iconst(types::I64, n as i64);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let starts_ptr = store_i64_array(builder, start_indices);
            let limits_ptr = store_i64_array(builder, limit_indices);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_dst_v,
                    n_src_v,
                    shape_ptr,
                    rank_v,
                    starts_ptr,
                    limits_ptr,
                ],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::Concatenate {
            operands,
            dimension,
        } => {
            let dim = *dimension as usize;
            let dst = alloc_slot(builder, n * elem_sz);

            if dim == 0 || rt.rank() <= 1 {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let mut byte_off = 0i64;
                for vid in operands {
                    let src_ty = type_map.get(vid).cloned().unwrap_or(rt.clone());
                    let src_bytes = src_ty.byte_size();
                    let d = builder.ins().iadd_imm(dst, byte_off);
                    let nb = builder.ins().iconst(types::I64, src_bytes as i64);
                    builder.ins().call(memcpy_ref, &[d, get(vid)?, nb]);
                    byte_off += src_bytes as i64;
                }
            } else if operands.len() == 2 {
                let a_ty = type_map.get(&operands[0]).cloned().unwrap_or(rt.clone());
                let b_ty = type_map.get(&operands[1]).cloned().unwrap_or(rt.clone());
                let func_ref = jit_module.declare_func_in_func(trt_ids.concat_nd_f64, builder.func);
                let n_dst_v = builder.ins().iconst(types::I64, n as i64);
                let n_a_v = builder.ins().iconst(types::I64, a_ty.num_elements() as i64);
                let n_b_v = builder.ins().iconst(types::I64, b_ty.num_elements() as i64);
                let dst_shape_ptr = store_i64_array(builder, &rt.shape);
                let a_shape_ptr = store_i64_array(builder, &a_ty.shape);
                let rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
                let dim_v = builder.ins().iconst(types::I64, dim as i64);
                let esz_v = builder.ins().iconst(types::I64, elem_sz as i64);
                builder.ins().call(
                    func_ref,
                    &[
                        dst,
                        n_dst_v,
                        get(&operands[0])?,
                        n_a_v,
                        get(&operands[1])?,
                        n_b_v,
                        dst_shape_ptr,
                        a_shape_ptr,
                        rank_v,
                        dim_v,
                        esz_v,
                    ],
                );
            } else {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let mut byte_off = 0i64;
                for vid in operands {
                    let src_ty = type_map.get(vid).cloned().unwrap_or(rt.clone());
                    let src_bytes = src_ty.byte_size();
                    let d = builder.ins().iadd_imm(dst, byte_off);
                    let nb = builder.ins().iconst(types::I64, src_bytes as i64);
                    builder.ins().call(memcpy_ref, &[d, get(vid)?, nb]);
                    byte_off += src_bytes as i64;
                }
            }
            Ok(vec![vec![dst]])
        }

        Instruction::Pad {
            operand,
            padding_value,
            low,
            ..
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let dst = alloc_slot(builder, n * elem_sz);
            let pad_ptr = get(padding_value)?;
            let pad_scalar = builder
                .ins()
                .load(types::F64, MemFlags::trusted(), pad_ptr, 0);
            let func_ref = jit_module.declare_func_in_func(trt_ids.pad_f64, builder.func);
            let n_dst_v = builder.ins().iconst(types::I64, n as i64);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let dst_shape_ptr = store_i64_array(builder, &rt.shape);
            let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
            let low_ptr = store_i64_array(builder, low);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_dst_v,
                    n_src_v,
                    pad_scalar,
                    dst_shape_ptr,
                    src_shape_ptr,
                    rank_v,
                    low_ptr,
                ],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let dst = alloc_slot(builder, n * elem_sz);
            let func_ref = jit_module.declare_func_in_func(trt_ids.dynamic_slice_f64, builder.func);
            let n_dst_v = builder.ins().iconst(types::I64, n as i64);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let sizes_ptr = store_i64_array(builder, slice_sizes);

            let starts_ss = alloc_slot(builder, start_indices.len() * 8);
            for (i, idx_vid) in start_indices.iter().enumerate() {
                let idx_ptr = get(idx_vid)?;
                let idx_et = type_map
                    .get(idx_vid)
                    .map(|t| t.element_type)
                    .unwrap_or(ElementType::I64);
                let ct = cranelift_type_for(idx_et);
                let raw = builder.ins().load(ct, MemFlags::trusted(), idx_ptr, 0);
                let idx_val = if ct == types::I32 {
                    builder.ins().sextend(types::I64, raw)
                } else {
                    raw
                };
                builder
                    .ins()
                    .store(MemFlags::trusted(), idx_val, starts_ss, (i * 8) as i32);
            }

            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_dst_v,
                    n_src_v,
                    shape_ptr,
                    rank_v,
                    starts_ss,
                    sizes_ptr,
                ],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let upd_ty = type_map.get(update).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let n_upd = upd_ty.num_elements();
            let dst = alloc_slot(builder, n * elem_sz);
            let func_ref =
                jit_module.declare_func_in_func(trt_ids.dynamic_update_slice_f64, builder.func);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let n_upd_v = builder.ins().iconst(types::I64, n_upd as i64);
            let shape_ptr = store_i64_array(builder, &src_ty.shape);
            let rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
            let upd_shape_ptr = store_i64_array(builder, &upd_ty.shape);

            let starts_ss = alloc_slot(builder, start_indices.len() * 8);
            for (i, idx_vid) in start_indices.iter().enumerate() {
                let idx_ptr = get(idx_vid)?;
                let idx_et = type_map
                    .get(idx_vid)
                    .map(|t| t.element_type)
                    .unwrap_or(ElementType::I64);
                let ct = cranelift_type_for(idx_et);
                let raw = builder.ins().load(ct, MemFlags::trusted(), idx_ptr, 0);
                let idx_val = if ct == types::I32 {
                    builder.ins().sextend(types::I64, raw)
                } else {
                    raw
                };
                builder
                    .ins()
                    .store(MemFlags::trusted(), idx_val, starts_ss, (i * 8) as i32);
            }

            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    get(update)?,
                    n_src_v,
                    n_upd_v,
                    shape_ptr,
                    rank_v,
                    starts_ss,
                    upd_shape_ptr,
                ],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::Gather {
            operand,
            indices,
            dims,
            slice_sizes,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_ty = type_map
                .get(indices)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::I64));

            let ivd = dims.index_vector_dim as usize;
            let idx_rank = idx_ty.rank();
            let n_index_dims = if ivd < idx_rank {
                idx_ty.shape[ivd] as usize
            } else {
                1
            };
            let use_nd = n_index_dims > 1 && !dims.start_index_map.is_empty();

            let n_total_idx = idx_ty.num_elements();
            let idx_ptr = get(indices)?;
            let widened_idx = if matches!(idx_ty.element_type, ElementType::I32 | ElementType::UI32)
            {
                let wide_buf = alloc_slot(builder, n_total_idx * 8);
                let widen_ref =
                    jit_module.declare_func_in_func(trt_ids.widen_i32_to_i64, builder.func);
                let n_v = builder.ins().iconst(types::I64, n_total_idx as i64);
                builder.ins().call(widen_ref, &[wide_buf, idx_ptr, n_v]);
                wide_buf
            } else {
                idx_ptr
            };

            let dst = alloc_slot(builder, n * elem_sz);

            if use_nd {
                let n_batch = if idx_rank > 1 {
                    n_total_idx / n_index_dims
                } else {
                    1
                };
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.gather_nd_f64, builder.func);
                let n_src_v = builder
                    .ins()
                    .iconst(types::I64, src_ty.num_elements() as i64);
                let n_batch_v = builder.ins().iconst(types::I64, n_batch as i64);
                let n_idx_dims_v = builder.ins().iconst(types::I64, n_index_dims as i64);
                let src_shape_ptr = store_i64_array(builder, &src_ty.shape);
                let src_rank_v = builder.ins().iconst(types::I64, src_ty.rank() as i64);
                let sim_ptr = store_i64_array(builder, &dims.start_index_map);
                let ss_ptr = store_i64_array(builder, slice_sizes);
                let n_dst_v = builder.ins().iconst(types::I64, n as i64);
                builder.ins().call(
                    func_ref,
                    &[
                        dst,
                        get(operand)?,
                        n_src_v,
                        widened_idx,
                        n_batch_v,
                        n_idx_dims_v,
                        src_shape_ptr,
                        src_rank_v,
                        sim_ptr,
                        ss_ptr,
                        n_dst_v,
                    ],
                );
            } else {
                let n_idx = if !dims.collapsed_slice_dims.is_empty() {
                    idx_ty.shape.first().copied().unwrap_or(1) as usize
                } else {
                    n_total_idx
                };
                let row_size = if n_idx > 0 { n / n_idx } else { 1 };
                let func_ref =
                    jit_module.declare_func_in_func(trt_ids.gather_f64, builder.func);
                let n_src_v = builder
                    .ins()
                    .iconst(types::I64, src_ty.num_elements() as i64);
                let n_idx_v = builder.ins().iconst(types::I64, n_idx as i64);
                let row_v = builder.ins().iconst(types::I64, row_size as i64);
                builder.ins().call(
                    func_ref,
                    &[dst, get(operand)?, n_src_v, widened_idx, n_idx_v, row_v],
                );
            }
            Ok(vec![vec![dst]])
        }

        Instruction::Scatter {
            operand,
            indices,
            updates,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_ty = type_map
                .get(indices)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::I64));
            let upd_ty = type_map.get(updates).cloned().unwrap_or(rt.clone());
            let n_src = src_ty.num_elements();
            let n_updates = idx_ty.num_elements();
            let inner_size = if n_updates > 0 {
                upd_ty.num_elements() / n_updates
            } else {
                1
            };
            let idx_ptr = get(indices)?;
            let widened_idx = if matches!(idx_ty.element_type, ElementType::I32 | ElementType::UI32)
            {
                let wide_buf = alloc_slot(builder, n_updates * 8);
                let widen_ref =
                    jit_module.declare_func_in_func(trt_ids.widen_i32_to_i64, builder.func);
                let n_upd_v2 = builder.ins().iconst(types::I64, n_updates as i64);
                builder
                    .ins()
                    .call(widen_ref, &[wide_buf, idx_ptr, n_upd_v2]);
                wide_buf
            } else {
                idx_ptr
            };
            let dst = alloc_slot(builder, n * elem_sz);
            let func_ref = jit_module.declare_func_in_func(trt_ids.scatter_f64, builder.func);
            let n_src_v = builder.ins().iconst(types::I64, n_src as i64);
            let n_upd_v = builder.ins().iconst(types::I64, n_updates as i64);
            let inner_v = builder.ins().iconst(types::I64, inner_size as i64);
            builder.ins().call(
                func_ref,
                &[
                    dst,
                    get(operand)?,
                    n_src_v,
                    widened_idx,
                    get(updates)?,
                    n_upd_v,
                    inner_v,
                ],
            );
            Ok(vec![vec![dst]])
        }

        Instruction::DotGeneral { lhs, rhs, dims } => {
            let l_ty = type_map.get(lhs).cloned().unwrap_or(rt.clone());
            let r_ty = type_map.get(rhs).cloned().unwrap_or(rt.clone());
            if l_ty.rank() < 2 || r_ty.rank() < 2 {
                return Err(format!(
                    "mem: DotGeneral needs rank>=2 operands, got {:?} and {:?}",
                    l_ty, r_ty
                ));
            }
            let m = l_ty.shape[0] as usize;
            let nn = r_ty.shape[r_ty.rank() - 1] as usize;
            let k = if !dims.lhs_contracting.is_empty() {
                l_ty.shape[dims.lhs_contracting[0] as usize] as usize
            } else {
                1
            };
            let out_size = m * nn * elem_sz;
            let dst = alloc_slot(builder, out_size);
            let func_ref = jit_module.declare_func_in_func(trt_ids.matmul_f64, builder.func);
            let m_v = builder.ins().iconst(types::I64, m as i64);
            let k_v = builder.ins().iconst(types::I64, k as i64);
            let n_v = builder.ins().iconst(types::I64, nn as i64);
            builder
                .ins()
                .call(func_ref, &[dst, get(lhs)?, get(rhs)?, m_v, k_v, n_v]);
            Ok(vec![vec![dst]])
        }

        Instruction::Reduce {
            operand,
            init: _,
            op,
            dimensions: _,
        } => {
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let func_id = match op {
                ReduceOp::Add => trt_ids.reduce_sum_f64,
                ReduceOp::Maximum => trt_ids.reduce_max_f64,
                ReduceOp::Minimum => trt_ids.reduce_min_f64,
                ReduceOp::And | ReduceOp::Or => {
                    return Err(format!("mem: reduce {:?} not yet supported", op));
                }
            };
            let n_in = src_ty.num_elements();
            let n_out = n;
            let inner = if n_out > 0 { n_in / n_out } else { n_in };
            let dst = alloc_slot(builder, n_out * elem_sz);
            let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
            let outer_v = builder.ins().iconst(types::I64, n_out as i64);
            let inner_v = builder.ins().iconst(types::I64, inner as i64);
            builder
                .ins()
                .call(func_ref, &[dst, get(operand)?, outer_v, inner_v]);
            Ok(vec![vec![dst]])
        }

        Instruction::While {
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids,
        } => {
            let mut slots: Vec<(cranelift_codegen::ir::StackSlot, TensorType)> = Vec::new();
            for vid in init_values {
                let ty = type_map
                    .get(vid)
                    .cloned()
                    .unwrap_or(TensorType::scalar(ElementType::F64));
                let byte_sz = ty.byte_size();
                let ss = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    byte_sz as u32,
                    3,
                ));
                let addr = builder.ins().stack_addr(ptr_type(), ss, 0);
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                let nb = builder.ins().iconst(types::I64, byte_sz as i64);
                builder.ins().call(memcpy_ref, &[addr, get(vid)?, nb]);
                slots.push((ss, ty));
            }

            let header = builder.create_block();
            let body_blk = builder.create_block();
            let exit = builder.create_block();
            builder.ins().jump(header, &[]);

            builder.switch_to_block(header);
            let mut cond_vm = value_map.clone();
            let mut cond_tm = type_map.clone();
            for (i, (ss, ty)) in slots.iter().enumerate() {
                let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
                cond_vm.insert(vid, vec![addr]);
                cond_tm.insert(vid, ty.clone());
            }
            lower_body_mem(
                builder,
                cond_body,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &mut cond_vm,
                &mut cond_tm,
            )?;

            let cond_val = {
                let mut cv = None;
                for ir in cond_body.iter().rev() {
                    if let Instruction::Return { operands } = &ir.instr {
                        if let Some(vid) = operands.first() {
                            let ptr = cond_vm.get(vid).and_then(|v| v.first().copied());
                            if let Some(p) = ptr {
                                cv = Some(builder.ins().load(types::I8, MemFlags::trusted(), p, 0));
                            }
                        }
                        break;
                    }
                }
                cv.ok_or("while: no condition value")?
            };
            builder.ins().brif(cond_val, body_blk, &[], exit, &[]);

            builder.switch_to_block(body_blk);
            builder.seal_block(body_blk);
            let mut body_vm = value_map.clone();
            let mut body_tm = type_map.clone();
            for (i, (ss, ty)) in slots.iter().enumerate() {
                let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
                body_vm.insert(vid, vec![addr]);
                body_tm.insert(vid, ty.clone());
            }
            lower_body_mem(
                builder,
                loop_body,
                ir_module,
                func_ids,
                libm_ids,
                trt_ids,
                func_abis,
                jit_module,
                &mut body_vm,
                &mut body_tm,
            )?;

            if let Some(ret_ir) = loop_body
                .iter()
                .rev()
                .find(|ir| matches!(ir.instr, Instruction::Return { .. }))
                && let Instruction::Return { operands } = &ret_ir.instr
            {
                let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
                for (vid, (ss, ty)) in operands.iter().zip(slots.iter()) {
                    if let Some(vals) = body_vm.get(vid) {
                        let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                        let nb = builder.ins().iconst(types::I64, ty.byte_size() as i64);
                        builder.ins().call(memcpy_ref, &[addr, vals[0], nb]);
                    }
                }
            }
            builder.ins().jump(header, &[]);
            builder.seal_block(header);

            builder.switch_to_block(exit);
            builder.seal_block(exit);

            let mut result_groups = Vec::new();
            for (i, rty) in result_types.iter().enumerate() {
                if i < slots.len() {
                    let (ss, _) = &slots[i];
                    let addr = builder.ins().stack_addr(ptr_type(), *ss, 0);
                    result_groups.push(vec![addr]);
                }
            }
            Ok(result_groups)
        }

        Instruction::Iota { dimension } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let fid = if is_float(rt.element_type) {
                trt_ids.iota_nd_f64
            } else {
                trt_ids.iota_nd_i64
            };
            let func_ref = jit_module.declare_func_in_func(fid, builder.func);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            let shape_ptr = store_i64_array(builder, &rt.shape);
            let rank_v = builder.ins().iconst(types::I64, rt.rank() as i64);
            let dim_v = builder.ins().iconst(types::I64, *dimension);
            builder
                .ins()
                .call(func_ref, &[dst, n_val, shape_ptr, rank_v, dim_v]);
            Ok(vec![vec![dst]])
        }

        Instruction::Call { callee, args } => {
            let fid = func_ids
                .get(callee)
                .ok_or_else(|| format!("mem: unknown callee: {callee}"))?;
            let callee_def = ir_module
                .get_func(callee)
                .ok_or_else(|| format!("mem: no func def for {callee}"))?;
            let callee_abi = func_abis.get(callee).copied().unwrap_or(FuncAbi::Scalar);
            let func_ref = jit_module.declare_func_in_func(*fid, builder.func);

            if callee_abi == FuncAbi::Pointer {
                let mut call_args: Vec<Value> = args.iter().map(get).collect::<Result<_, _>>()?;
                let total_ret_bytes: usize =
                    callee_def.result_types.iter().map(|t| t.byte_size()).sum();
                let ret_buf = alloc_slot(builder, total_ret_bytes.max(8));
                call_args.push(ret_buf);
                builder.ins().call(func_ref, &call_args);

                let mut result_groups = Vec::new();
                let mut off = 0i64;
                for rty in &callee_def.result_types {
                    let addr = builder.ins().iadd_imm(ret_buf, off);
                    result_groups.push(vec![addr]);
                    off += rty.byte_size() as i64;
                }
                Ok(result_groups)
            } else {
                let callee_sret = needs_sret(&callee_def.result_types);
                let mut call_args = Vec::new();
                for (vid, (_pv, pty)) in args.iter().zip(callee_def.params.iter()) {
                    let ptr = get(vid)?;
                    let n_elem = pty.num_elements();
                    let ct = cranelift_type_for(pty.element_type);
                    let esz = pty.element_type.byte_size();
                    for j in 0..n_elem {
                        let v = builder
                            .ins()
                            .load(ct, MemFlags::trusted(), ptr, (j * esz) as i32);
                        call_args.push(v);
                    }
                }

                if callee_sret {
                    let total_bytes: usize =
                        callee_def.result_types.iter().map(|t| t.byte_size()).sum();
                    let ret_buf = alloc_slot(builder, total_bytes);
                    call_args.push(ret_buf);
                    builder.ins().call(func_ref, &call_args);

                    let mut result_groups = Vec::new();
                    let mut byte_off = 0i64;
                    for rty in &callee_def.result_types {
                        let addr = builder.ins().iadd_imm(ret_buf, byte_off);
                        result_groups.push(vec![addr]);
                        byte_off += rty.byte_size() as i64;
                    }
                    Ok(result_groups)
                } else {
                    let call = builder.ins().call(func_ref, &call_args);
                    let results: Vec<Value> = builder.inst_results(call).to_vec();

                    let mut result_groups = Vec::new();
                    let mut off = 0;
                    for rty in &callee_def.result_types {
                        let n_elem = rty.num_elements();
                        let esz = rty.element_type.byte_size();
                        let buf = alloc_slot(builder, n_elem * esz);
                        for j in 0..n_elem {
                            if off + j < results.len() {
                                builder.ins().store(
                                    MemFlags::trusted(),
                                    results[off + j],
                                    buf,
                                    (j * esz) as i32,
                                );
                            }
                        }
                        off += n_elem;
                        result_groups.push(vec![buf]);
                    }
                    Ok(result_groups)
                }
            }
        }

        Instruction::Atan2 { .. } => Err("mem: atan2 not yet supported".into()),
        Instruction::Acos { .. } => Err("mem: acos not yet supported".into()),
        Instruction::ErfInv { .. } => Err("mem: erf_inv not yet supported".into()),
        Instruction::Tan { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.tan_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Sign { operand } => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.sign_f64,
                &[dst, get(operand)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::Remainder { lhs, rhs } if is_float(rt.element_type) => {
            let dst = alloc_slot(builder, n * elem_sz);
            let n_val = builder.ins().iconst(types::I64, n as i64);
            trt_call(
                builder,
                jit_module,
                trt_ids.rem_f64,
                &[dst, get(lhs)?, get(rhs)?, n_val],
            );
            Ok(vec![vec![dst]])
        }
        Instruction::BitcastConvert { operand } => Ok(vec![vec![get(operand)?]]),
        Instruction::RoundNearestEven { operand } => {
            Err("mem: round_nearest_even not yet supported".into())
        }
        Instruction::Reverse { .. } => Err("mem: reverse not yet supported".into()),
        Instruction::Clamp { .. } => Err("mem: clamp not yet supported".into()),
        Instruction::Case { index, branches } => {
            let idx_ptr = get(index)?;
            let idx = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), idx_ptr, 0);

            let result_slots: Vec<_> = result_types
                .iter()
                .map(|rty| alloc_slot(builder, rty.byte_size()))
                .collect();

            let branch_blocks: Vec<Block> = (0..branches.len())
                .map(|_| builder.create_block())
                .collect();
            let merge_block = builder.create_block();

            let idx_ty = builder.func.dfg.value_type(idx);
            let empty_args: &[BlockArg] = &[];
            if branches.len() == 1 {
                builder.ins().jump(branch_blocks[0], empty_args);
            } else if branches.len() == 2 {
                let zero = builder.ins().iconst(idx_ty, 0);
                let cmp = builder.ins().icmp(IntCC::Equal, idx, zero);
                builder.ins().brif(
                    cmp,
                    branch_blocks[0],
                    empty_args,
                    branch_blocks[1],
                    empty_args,
                );
            } else {
                for i in 0..branches.len() - 1 {
                    let cmp_val = builder.ins().iconst(idx_ty, i as i64);
                    let cmp = builder.ins().icmp(IntCC::Equal, idx, cmp_val);
                    let next = if i == branches.len() - 2 {
                        branch_blocks[branches.len() - 1]
                    } else {
                        builder.create_block()
                    };
                    builder
                        .ins()
                        .brif(cmp, branch_blocks[i], empty_args, next, empty_args);
                    if i < branches.len() - 2 {
                        builder.switch_to_block(next);
                        builder.seal_block(next);
                    }
                }
            }

            let memcpy_ref = jit_module.declare_func_in_func(trt_ids.memcpy, builder.func);
            for (bi, branch) in branches.iter().enumerate() {
                builder.switch_to_block(branch_blocks[bi]);
                builder.seal_block(branch_blocks[bi]);

                let mut br_vm = value_map.clone();
                let mut br_tm = type_map.clone();
                lower_body_mem(
                    builder, branch, ir_module, func_ids, libm_ids, trt_ids, func_abis, jit_module,
                    &mut br_vm, &mut br_tm,
                )?;

                if let Some(ret_ir) = branch
                    .iter()
                    .rev()
                    .find(|ir| matches!(ir.instr, Instruction::Return { .. }))
                    && let Instruction::Return { operands } = &ret_ir.instr
                {
                    for (i, vid) in operands.iter().enumerate() {
                        if let (Some(vals), Some(rty)) = (br_vm.get(vid), result_types.get(i)) {
                            let nb = builder.ins().iconst(types::I64, rty.byte_size() as i64);
                            builder
                                .ins()
                                .call(memcpy_ref, &[result_slots[i], vals[0], nb]);
                        }
                    }
                }
                builder.ins().jump(merge_block, empty_args);
            }

            builder.switch_to_block(merge_block);
            builder.seal_block(merge_block);

            Ok(result_slots.into_iter().map(|s| vec![s]).collect())
        }
        Instruction::CustomCall {
            call_target,
            operands,
            ..
        } => Err(format!("mem: custom_call not yet supported: {call_target}")),

        Instruction::Return { .. } => Ok(vec![]),

        other => Err(format!("mem: unsupported instruction: {other:?}")),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_callee_body(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    block_params: &[Value],
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    let use_sret = needs_sret(&func_def.result_types);
    let mut param_offset = 0;

    for (vid, ty) in &func_def.params {
        let n = ty.num_elements();
        let vals: Vec<Value> = block_params[param_offset..param_offset + n].to_vec();
        value_map.insert(*vid, vals);
        type_map.insert(*vid, ty.clone());
        param_offset += n;
    }

    let sret_ptr = if use_sret {
        Some(block_params[param_offset])
    } else {
        None
    };

    lower_body(
        builder,
        &func_def.body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        value_map,
        type_map,
    )?;

    if let Some(ret_instr) = func_def.body.last() {
        if let Instruction::Return { operands } = &ret_instr.instr {
            if let Some(out_ptr) = sret_ptr {
                let mut offset = 0i32;
                for vid in operands {
                    if let Some(vals) = value_map.get(vid) {
                        for &v in vals {
                            let sz = builder.func.dfg.value_type(v).bytes() as i32;
                            builder.ins().store(MemFlags::trusted(), v, out_ptr, offset);
                            offset += sz;
                        }
                    }
                }
                builder.ins().return_(&[]);
            } else {
                let mut ret_vals = Vec::new();
                for vid in operands {
                    if let Some(vals) = value_map.get(vid) {
                        ret_vals.extend_from_slice(vals);
                    }
                }
                builder.ins().return_(&ret_vals);
            }
        } else {
            builder.ins().return_(&[]);
        }
    } else {
        builder.ins().return_(&[]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Body and instruction lowering
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn lower_body(
    builder: &mut FunctionBuilder,
    body: &[InstrResult],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<(), String> {
    for ir in body {
        if matches!(ir.instr, Instruction::Return { .. }) {
            break;
        }

        let result_types: Vec<TensorType> = ir.values.iter().map(|(_, t)| t.clone()).collect();

        let result_vals = lower_instruction(
            builder,
            &ir.instr,
            &result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
        )?;

        for (i, (vid, ty)) in ir.values.iter().enumerate() {
            if i < result_vals.len() {
                value_map.insert(*vid, result_vals[i].clone());
                type_map.insert(*vid, ty.clone());
            }
        }
    }
    Ok(())
}

fn get_vals<'a>(
    value_map: &'a HashMap<ValueId, TensorVals>,
    vid: &ValueId,
) -> Result<&'a TensorVals, String> {
    value_map
        .get(vid)
        .ok_or_else(|| format!("missing value {:?}", vid))
}

fn to_block_args(vals: &[Value]) -> Vec<BlockArg> {
    vals.iter().map(|&v| BlockArg::Value(v)).collect()
}

fn make_zero(builder: &mut FunctionBuilder, et: ElementType) -> Value {
    if is_float(et) {
        builder.ins().f64const(0.0)
    } else {
        builder.ins().iconst(cranelift_type_for(et), 0)
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_instruction(
    builder: &mut FunctionBuilder,
    instr: &Instruction,
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<Vec<TensorVals>, String> {
    let rt = result_types
        .first()
        .cloned()
        .unwrap_or(TensorType::scalar(ElementType::F64));

    match instr {
        // ----- Constants -----
        Instruction::Constant { value } => Ok(vec![lower_constant(builder, value, &rt)]),

        // ----- Arithmetic -----
        Instruction::Add { lhs, rhs } => {
            let l = get_vals(value_map, lhs)?;
            let r = get_vals(value_map, rhs)?;
            let et = rt.element_type;
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fadd(a, c)
                } else {
                    b.ins().iadd(a, c)
                }
            });
            Ok(vec![out])
        }

        Instruction::Subtract { lhs, rhs } => {
            let l = get_vals(value_map, lhs)?;
            let r = get_vals(value_map, rhs)?;
            let et = rt.element_type;
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fsub(a, c)
                } else {
                    b.ins().isub(a, c)
                }
            });
            Ok(vec![out])
        }

        Instruction::Multiply { lhs, rhs } => {
            let l = get_vals(value_map, lhs)?;
            let r = get_vals(value_map, rhs)?;
            let et = rt.element_type;
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fmul(a, c)
                } else {
                    b.ins().imul(a, c)
                }
            });
            Ok(vec![out])
        }

        Instruction::Divide { lhs, rhs } => {
            let l = get_vals(value_map, lhs)?;
            let r = get_vals(value_map, rhs)?;
            let et = rt.element_type;
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    b.ins().fdiv(a, c)
                } else if is_unsigned(et) {
                    b.ins().udiv(a, c)
                } else {
                    b.ins().sdiv(a, c)
                }
            });
            Ok(vec![out])
        }

        Instruction::Maximum { lhs, rhs } => {
            let l = get_vals(value_map, lhs)?;
            let r = get_vals(value_map, rhs)?;
            let et = rt.element_type;
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    let cmp = b.ins().fcmp(FloatCC::GreaterThan, a, c);
                    b.ins().select(cmp, a, c)
                } else {
                    let cmp = b.ins().icmp(IntCC::SignedGreaterThan, a, c);
                    b.ins().select(cmp, a, c)
                }
            });
            Ok(vec![out])
        }

        // ----- Unary -----
        Instruction::Negate { operand } => {
            let vals = get_vals(value_map, operand)?;
            let et = rt.element_type;
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| {
                    if is_float(et) {
                        builder.ins().fneg(v)
                    } else {
                        let zero = builder.ins().iconst(cranelift_type_for(et), 0);
                        builder.ins().isub(zero, v)
                    }
                })
                .collect();
            Ok(vec![out])
        }

        Instruction::Sqrt { operand } => {
            let vals = get_vals(value_map, operand)?;
            let out: Vec<Value> = vals.iter().map(|&v| builder.ins().sqrt(v)).collect();
            Ok(vec![out])
        }

        // ----- Transcendental (libm calls) -----
        Instruction::Sine { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.sin, jit_module)
        }
        Instruction::Cosine { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.cos, jit_module)
        }
        Instruction::Atan2 { lhs, rhs } => {
            lower_libm_binary(builder, value_map, lhs, rhs, libm_ids.atan2, jit_module)
        }
        Instruction::Acos { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.acos, jit_module)
        }
        Instruction::Exponential { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.exp, jit_module)
        }
        Instruction::Log { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.log, jit_module)
        }
        Instruction::Power { lhs, rhs } => {
            lower_libm_binary(builder, value_map, lhs, rhs, libm_ids.pow, jit_module)
        }
        Instruction::Tan { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.tan, jit_module)
        }
        Instruction::Tanh { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.tanh, jit_module)
        }
        Instruction::ErfInv { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.erf_inv, jit_module)
        }

        Instruction::Abs { operand } => {
            let vals = get_vals(value_map, operand)?;
            let et = rt.element_type;
            let out: Vec<Value> = if is_float(et) {
                let func_ref = jit_module.declare_func_in_func(libm_ids.fabs, builder.func);
                vals.iter()
                    .map(|&v| {
                        let call = builder.ins().call(func_ref, &[v]);
                        builder.inst_results(call)[0]
                    })
                    .collect()
            } else {
                vals.iter()
                    .map(|&v| {
                        let zero = builder.ins().iconst(cranelift_type_for(et), 0);
                        let neg = builder.ins().isub(zero, v);
                        let is_neg = builder.ins().icmp(IntCC::SignedLessThan, v, zero);
                        builder.ins().select(is_neg, neg, v)
                    })
                    .collect()
            };
            Ok(vec![out])
        }

        Instruction::Sign { operand } => {
            let vals = get_vals(value_map, operand)?;
            let et = rt.element_type;
            let out: Vec<Value> = if is_float(et) {
                vals.iter()
                    .map(|&v| {
                        let (zero, one, neg_one) = match et {
                            ElementType::F32 => (
                                builder.ins().f32const(0.0),
                                builder.ins().f32const(1.0),
                                builder.ins().f32const(-1.0),
                            ),
                            _ => (
                                builder.ins().f64const(0.0),
                                builder.ins().f64const(1.0),
                                builder.ins().f64const(-1.0),
                            ),
                        };
                        let is_pos = builder.ins().fcmp(FloatCC::GreaterThan, v, zero);
                        let is_neg = builder.ins().fcmp(FloatCC::LessThan, v, zero);
                        let step1 = builder.ins().select(is_pos, one, zero);
                        builder.ins().select(is_neg, neg_one, step1)
                    })
                    .collect()
            } else {
                let ct = cranelift_type_for(et);
                vals.iter()
                    .map(|&v| {
                        let zero = builder.ins().iconst(ct, 0);
                        let one = builder.ins().iconst(ct, 1);
                        let neg_one = builder.ins().iconst(ct, -1i64);
                        let is_pos = builder.ins().icmp(IntCC::SignedGreaterThan, v, zero);
                        let is_neg = builder.ins().icmp(IntCC::SignedLessThan, v, zero);
                        let step1 = builder.ins().select(is_pos, one, zero);
                        builder.ins().select(is_neg, neg_one, step1)
                    })
                    .collect()
            };
            Ok(vec![out])
        }

        Instruction::Minimum { lhs, rhs } => {
            let l = get_vals(value_map, lhs)?;
            let r = get_vals(value_map, rhs)?;
            let et = rt.element_type;
            let out = elementwise_binop(builder, l, r, |b, a, c| {
                if is_float(et) {
                    let cmp = b.ins().fcmp(FloatCC::LessThan, a, c);
                    b.ins().select(cmp, a, c)
                } else {
                    let cmp = b.ins().icmp(IntCC::SignedLessThan, a, c);
                    b.ins().select(cmp, a, c)
                }
            });
            Ok(vec![out])
        }

        Instruction::Remainder { lhs, rhs } => {
            if is_float(rt.element_type) {
                lower_libm_binary(builder, value_map, lhs, rhs, libm_ids.fmod, jit_module)
            } else {
                let l = get_vals(value_map, lhs)?;
                let r = get_vals(value_map, rhs)?;
                let n = l.len().max(r.len());
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let lv = if i < l.len() { l[i] } else { l[0] };
                        let rv = if i < r.len() { r[i] } else { r[0] };
                        builder.ins().srem(lv, rv)
                    })
                    .collect();
                Ok(vec![out])
            }
        }

        Instruction::Clamp { operand, min, max } => {
            let vals = get_vals(value_map, operand)?;
            let mins = get_vals(value_map, min)?;
            let maxs = get_vals(value_map, max)?;
            let et = rt.element_type;
            let n = vals.len();
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    let v = vals[i];
                    let lo = if i < mins.len() { mins[i] } else { mins[0] };
                    let hi = if i < maxs.len() { maxs[i] } else { maxs[0] };
                    if is_float(et) {
                        let clamped_lo = builder.ins().fmax(v, lo);
                        builder.ins().fmin(clamped_lo, hi)
                    } else {
                        let gt_lo = builder.ins().icmp(IntCC::SignedGreaterThan, v, lo);
                        let step1 = builder.ins().select(gt_lo, v, lo);
                        let lt_hi = builder.ins().icmp(IntCC::SignedLessThan, step1, hi);
                        builder.ins().select(lt_hi, step1, hi)
                    }
                })
                .collect();
            Ok(vec![out])
        }

        Instruction::Reverse {
            operand,
            dimensions,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let shape = &src_ty.shape;
            let n = vals.len();

            if shape.is_empty() || n <= 1 {
                return Ok(vec![vals]);
            }

            let mut result = vals.clone();
            let rank = shape.len();
            let strides: Vec<usize> = {
                let mut s = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    s[i] = s[i + 1] * shape[i + 1] as usize;
                }
                s
            };

            for &dim in dimensions {
                let d = dim as usize;
                let dim_size = shape[d] as usize;
                if dim_size <= 1 {
                    continue;
                }
                let mut next = result.clone();
                for (flat_idx, slot) in next.iter_mut().enumerate().take(n) {
                    let coord_d = (flat_idx / strides[d]) % dim_size;
                    let reversed_coord = dim_size - 1 - coord_d;
                    let src_idx = flat_idx - coord_d * strides[d] + reversed_coord * strides[d];
                    *slot = result[src_idx];
                }
                result = next;
            }
            Ok(vec![result])
        }

        Instruction::Floor { operand } => {
            let vals = get_vals(value_map, operand)?;
            let out: Vec<Value> = vals.iter().map(|&v| builder.ins().floor(v)).collect();
            Ok(vec![out])
        }

        Instruction::RoundNearestEven { operand } => {
            let vals = get_vals(value_map, operand)?;
            let out: Vec<Value> = vals.iter().map(|&v| builder.ins().nearest(v)).collect();
            Ok(vec![out])
        }

        // ----- Comparison and select -----
        Instruction::Compare {
            lhs,
            rhs,
            direction,
            compare_type,
        } => {
            let l = get_vals(value_map, lhs)?;
            let r = get_vals(value_map, rhs)?;
            let l_ty = type_map
                .get(lhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let use_float = is_float(l_ty.element_type)
                || matches!(compare_type, CompareType::Float | CompareType::TotalOrder);
            let out: Vec<Value> = l
                .iter()
                .zip(r.iter())
                .map(|(&a, &b)| {
                    if use_float {
                        let cc = match direction {
                            CompareDirection::Eq => FloatCC::Equal,
                            CompareDirection::Ne => FloatCC::NotEqual,
                            CompareDirection::Lt => FloatCC::LessThan,
                            CompareDirection::Le => FloatCC::LessThanOrEqual,
                            CompareDirection::Gt => FloatCC::GreaterThan,
                            CompareDirection::Ge => FloatCC::GreaterThanOrEqual,
                        };
                        builder.ins().fcmp(cc, a, b)
                    } else {
                        let uns = matches!(compare_type, CompareType::Unsigned)
                            || is_unsigned(l_ty.element_type);
                        let cc = match (direction, uns) {
                            (CompareDirection::Eq, _) => IntCC::Equal,
                            (CompareDirection::Ne, _) => IntCC::NotEqual,
                            (CompareDirection::Lt, false) => IntCC::SignedLessThan,
                            (CompareDirection::Le, false) => IntCC::SignedLessThanOrEqual,
                            (CompareDirection::Gt, false) => IntCC::SignedGreaterThan,
                            (CompareDirection::Ge, false) => IntCC::SignedGreaterThanOrEqual,
                            (CompareDirection::Lt, true) => IntCC::UnsignedLessThan,
                            (CompareDirection::Le, true) => IntCC::UnsignedLessThanOrEqual,
                            (CompareDirection::Gt, true) => IntCC::UnsignedGreaterThan,
                            (CompareDirection::Ge, true) => IntCC::UnsignedGreaterThanOrEqual,
                        };
                        builder.ins().icmp(cc, a, b)
                    }
                })
                .collect();
            Ok(vec![out])
        }

        Instruction::Select {
            cond,
            on_true,
            on_false,
        } => {
            let c = get_vals(value_map, cond)?;
            let t = get_vals(value_map, on_true)?;
            let f = get_vals(value_map, on_false)?;
            let out: Vec<Value> = t
                .iter()
                .zip(f.iter())
                .enumerate()
                .map(|(i, (&tv, &fv))| {
                    let cv = if i < c.len() { c[i] } else { c[0] };
                    builder.ins().select(cv, tv, fv)
                })
                .collect();
            Ok(vec![out])
        }

        // ----- Shape ops -----
        Instruction::Reshape { operand } => {
            let vals = get_vals(value_map, operand)?.clone();
            let n = rt.num_elements();
            let out = if vals.len() == n {
                vals
            } else if vals.len() > n {
                vals[..n].to_vec()
            } else {
                let mut out = vals;
                let ct = cranelift_type_for(rt.element_type);
                while out.len() < n {
                    out.push(builder.ins().iconst(ct, 0));
                }
                out
            };
            Ok(vec![out])
        }

        Instruction::BroadcastInDim {
            operand,
            broadcast_dims,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let n = rt.num_elements();
            if vals.len() == 1 {
                Ok(vec![vec![vals[0]; n]])
            } else {
                Ok(vec![broadcast_values(
                    &vals,
                    &rt.shape,
                    broadcast_dims,
                    &src_ty.shape,
                )])
            }
        }

        Instruction::Slice {
            operand,
            start_indices,
            limit_indices,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            Ok(vec![slice_tensor(
                &vals,
                &src_ty.shape,
                start_indices,
                limit_indices,
            )])
        }

        Instruction::Concatenate {
            operands,
            dimension,
        } => {
            let parts: Vec<(&TensorVals, TensorType)> = operands
                .iter()
                .map(|vid| {
                    let v = get_vals(value_map, vid)?;
                    let ty = type_map
                        .get(vid)
                        .cloned()
                        .unwrap_or(TensorType::scalar(ElementType::F64));
                    Ok((v, ty))
                })
                .collect::<Result<_, String>>()?;

            let dim = *dimension as usize;
            if dim == 0 || parts.iter().all(|(_, ty)| ty.rank() <= 1) {
                let mut all_vals = Vec::new();
                for (v, _) in &parts {
                    all_vals.extend_from_slice(v);
                }
                Ok(vec![all_vals])
            } else {
                Ok(vec![lower_concatenate_nd(&parts, dim, &rt)])
            }
        }

        // ----- Type conversions -----
        Instruction::Convert { operand } => {
            let vals = get_vals(value_map, operand)?.clone();
            let src_ty = type_map
                .get(operand)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| convert_value(builder, v, &src_ty.element_type, &rt.element_type))
                .collect();
            Ok(vec![out])
        }

        Instruction::BitcastConvert { operand } => {
            let vals = get_vals(value_map, operand)?.clone();
            let dst_ct = cranelift_type_for(rt.element_type);
            let out: Vec<Value> = vals
                .iter()
                .map(|&v| builder.ins().bitcast(dst_ct, MemFlags::new(), v))
                .collect();
            Ok(vec![out])
        }

        Instruction::Iota { dimension } => {
            let n = rt.num_elements();
            let ct = cranelift_type_for(rt.element_type);
            let dim = *dimension as usize;
            let shape: Vec<usize> = rt.shape.iter().map(|&d| d as usize).collect();
            let rank = shape.len();
            let mut strides = vec![1usize; rank];
            for d in (0..rank.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let out: Vec<Value> = (0..n)
                .map(|flat| {
                    let idx_along_dim = (flat / strides[dim]) % shape[dim];
                    if is_float(rt.element_type) {
                        builder.ins().f64const(idx_along_dim as f64)
                    } else {
                        builder.ins().iconst(ct, idx_along_dim as i64)
                    }
                })
                .collect();
            Ok(vec![out])
        }

        // ----- Integer bitwise ops -----
        Instruction::Xor { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().bxor(a, c))
        }
        Instruction::Or { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().bor(a, c))
        }
        Instruction::And { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().band(a, c))
        }
        Instruction::ShiftLeft { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().ishl(a, c))
        }
        Instruction::ShiftRightLogical { lhs, rhs } => {
            lower_int_binop(builder, value_map, lhs, rhs, |b, a, c| b.ins().ushr(a, c))
        }

        // ----- Dot product / matmul -----
        Instruction::DotGeneral { lhs, rhs, dims } => {
            let l = get_vals(value_map, lhs)?.clone();
            let r = get_vals(value_map, rhs)?.clone();
            let l_ty = type_map
                .get(lhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            let r_ty = type_map
                .get(rhs)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::F64));
            Ok(vec![lower_dot_general(
                builder, &l, &r, &l_ty, &r_ty, &rt, dims,
            )])
        }

        // ----- Reduce -----
        Instruction::Reduce {
            operand,
            init,
            op,
            dimensions,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let init_vals = get_vals(value_map, init)?;
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            Ok(vec![lower_reduce(
                builder, &vals, init_vals, &src_ty, &rt, op, dimensions,
            )])
        }

        // ----- Gather -----
        Instruction::Gather {
            operand,
            indices,
            dims,
            slice_sizes,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let idx_vals = get_vals(value_map, indices)?.clone();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_ty = type_map
                .get(indices)
                .cloned()
                .unwrap_or(TensorType::scalar(ElementType::I32));
            Ok(vec![lower_gather(
                builder,
                &vals,
                &idx_vals,
                &src_ty,
                &idx_ty,
                &rt,
                dims,
                slice_sizes,
            )])
        }

        // ----- Function call -----
        Instruction::Call { callee, args } => lower_call(
            builder, callee, args, ir_module, func_ids, func_abis, trt_ids, jit_module, value_map,
            type_map,
        ),

        // ----- While loop (real Cranelift loop blocks) -----
        Instruction::While {
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids,
        } => lower_while(
            builder,
            cond_body,
            loop_body,
            init_values,
            iter_arg_ids,
            result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
        ),

        // ----- Case / conditional branching -----
        Instruction::Case { index, branches } => lower_case(
            builder,
            index,
            branches,
            result_types,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            value_map,
            type_map,
        ),

        // ----- Transpose -----
        Instruction::Transpose {
            operand,
            permutation,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            Ok(vec![lower_transpose(&vals, &src_ty, &rt, permutation)])
        }

        // ----- Dynamic slice -----
        Instruction::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let idx_vals: Vec<Value> = start_indices
                .iter()
                .map(|v| get_vals(value_map, v).map(|vs| vs[0]))
                .collect::<Result<_, _>>()?;
            Ok(vec![lower_dynamic_slice(
                builder,
                &vals,
                &idx_vals,
                &src_ty,
                slice_sizes,
            )])
        }

        // ----- Dynamic update slice -----
        Instruction::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let upd = get_vals(value_map, update)?.clone();
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());
            let upd_ty = type_map.get(update).cloned().unwrap_or(rt.clone());
            let idx_vals: Vec<Value> = start_indices
                .iter()
                .map(|v| get_vals(value_map, v).map(|vs| vs[0]))
                .collect::<Result<_, _>>()?;
            Ok(vec![lower_dynamic_update_slice(
                builder, &vals, &upd, &idx_vals, &src_ty, &upd_ty,
            )])
        }

        Instruction::Pad {
            operand,
            padding_value,
            low,
            high,
            interior,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let pad_val = get_vals(value_map, padding_value)?[0];
            let src_ty = type_map.get(operand).cloned().unwrap_or(rt.clone());

            let is_noop = low.iter().all(|&x| x == 0)
                && high.iter().all(|&x| x == 0)
                && interior.iter().all(|&x| x == 0);
            if is_noop {
                return Ok(vec![vals]);
            }

            let n = rt.num_elements();
            let rank = src_ty.shape.len();
            let src_shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
            let out_shape: Vec<usize> = rt.shape.iter().map(|&d| d as usize).collect();
            let mut src_strides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
            }
            let mut out_strides = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }

            let mut result = vec![pad_val; n];
            for (src_flat, &src_val) in vals.iter().enumerate() {
                let mut valid = true;
                let mut out_flat = 0;
                let mut remaining = src_flat;
                for d in 0..rank {
                    let coord = remaining / src_strides[d];
                    remaining %= src_strides[d];
                    let int_step = interior.get(d).copied().unwrap_or(0) as usize;
                    let out_coord =
                        low.get(d).copied().unwrap_or(0) as usize + coord * (1 + int_step);
                    if out_coord >= out_shape[d] {
                        valid = false;
                        break;
                    }
                    out_flat += out_coord * out_strides[d];
                }
                if valid && out_flat < n {
                    result[out_flat] = src_val;
                }
            }
            Ok(vec![result])
        }

        Instruction::Scatter {
            operand,
            indices,
            updates,
        } => {
            let vals = get_vals(value_map, operand)?.clone();
            let idx_vals = get_vals(value_map, indices)?;
            let upd_vals = get_vals(value_map, updates)?;
            let et = rt.element_type;
            let elem_sz = et.byte_size();

            let total_bytes = vals.len() * elem_sz;
            let ss = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                total_bytes as u32,
                3,
            ));
            let base = builder.ins().stack_addr(ptr_type(), ss, 0);
            for (i, &v) in vals.iter().enumerate() {
                builder
                    .ins()
                    .store(MemFlags::trusted(), v, base, (i * elem_sz) as i32);
            }

            for (u_idx, &upd_v) in upd_vals.iter().enumerate() {
                let raw_idx = if u_idx < idx_vals.len() {
                    idx_vals[u_idx]
                } else {
                    idx_vals[0]
                };
                let idx_i64 = if builder.func.dfg.value_type(raw_idx) == types::I64 {
                    raw_idx
                } else if builder.func.dfg.value_type(raw_idx).bytes() < 8 {
                    builder.ins().sextend(types::I64, raw_idx)
                } else {
                    raw_idx
                };
                let byte_offset = builder.ins().imul_imm(idx_i64, elem_sz as i64);
                let addr = builder.ins().iadd(base, byte_offset);
                builder.ins().store(MemFlags::trusted(), upd_v, addr, 0);
            }

            let ct = cranelift_type_for(et);
            let mut result = Vec::with_capacity(vals.len());
            for i in 0..vals.len() {
                let v = builder
                    .ins()
                    .load(ct, MemFlags::trusted(), base, (i * elem_sz) as i32);
                result.push(v);
            }
            Ok(vec![result])
        }

        Instruction::CustomCall {
            call_target,
            operands,
            backend_config,
        } => lower_custom_call(
            builder,
            call_target,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
            libm_ids,
            backend_config,
        ),

        Instruction::Return { .. } => Ok(vec![]),
    }
}

// ---------------------------------------------------------------------------
// While loop — real Cranelift loop with header/body/exit blocks
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn lower_while(
    builder: &mut FunctionBuilder,
    cond_body: &[InstrResult],
    loop_body: &[InstrResult],
    init_values: &[ValueId],
    iter_arg_ids: &[ValueId],
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<Vec<TensorVals>, String> {
    let mut init_tensors: Vec<(TensorVals, TensorType)> = Vec::new();
    for vid in init_values {
        let vals = get_vals(value_map, vid)?.clone();
        let ty = type_map
            .get(vid)
            .cloned()
            .unwrap_or(TensorType::scalar(ElementType::F64));
        init_tensors.push((vals, ty));
    }

    let all_init_flat: Vec<Value> = init_tensors
        .iter()
        .flat_map(|(vals, _)| vals.iter().copied())
        .collect();
    let all_types: Vec<Type> = init_tensors
        .iter()
        .flat_map(|(vals, ty)| std::iter::repeat_n(cranelift_type_for(ty.element_type), vals.len()))
        .collect();

    let header_block = builder.create_block();
    let body_block = builder.create_block();
    let exit_block = builder.create_block();

    for &ct in &all_types {
        builder.append_block_param(header_block, ct);
        builder.append_block_param(body_block, ct);
        builder.append_block_param(exit_block, ct);
    }

    let init_args = to_block_args(&all_init_flat);
    builder.ins().jump(header_block, &init_args);

    // --- Header: evaluate condition ---
    builder.switch_to_block(header_block);
    let header_params = builder.block_params(header_block).to_vec();

    let mut cond_vmap: HashMap<ValueId, TensorVals> = HashMap::new();
    let mut cond_tmap: HashMap<ValueId, TensorType> = HashMap::new();
    let mut offset = 0;
    for (i, (vals, ty)) in init_tensors.iter().enumerate() {
        let n = vals.len();
        let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
        cond_vmap.insert(vid, header_params[offset..offset + n].to_vec());
        cond_tmap.insert(vid, ty.clone());
        offset += n;
    }

    lower_body(
        builder,
        cond_body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        &mut cond_vmap,
        &mut cond_tmap,
    )?;

    let cond_val = extract_return_predicate(cond_body, &cond_vmap)?;

    let header_args = to_block_args(&header_params);
    builder
        .ins()
        .brif(cond_val, body_block, &header_args, exit_block, &header_args);

    // --- Body: execute loop iteration ---
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);

    let body_params = builder.block_params(body_block).to_vec();
    let mut body_vmap: HashMap<ValueId, TensorVals> = HashMap::new();
    let mut body_tmap: HashMap<ValueId, TensorType> = HashMap::new();
    offset = 0;
    for (i, (vals, ty)) in init_tensors.iter().enumerate() {
        let n = vals.len();
        let vid = iter_arg_ids.get(i).copied().unwrap_or(ValueId(i as u32));
        body_vmap.insert(vid, body_params[offset..offset + n].to_vec());
        body_tmap.insert(vid, ty.clone());
        offset += n;
    }

    lower_body(
        builder,
        loop_body,
        ir_module,
        func_ids,
        libm_ids,
        trt_ids,
        func_abis,
        jit_module,
        &mut body_vmap,
        &mut body_tmap,
    )?;

    let new_vals = extract_return_values(loop_body, &body_vmap)?;
    let new_args = to_block_args(&new_vals);
    builder.ins().jump(header_block, &new_args);
    builder.seal_block(header_block);

    // --- Exit: collect results ---
    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);

    let exit_params = builder.block_params(exit_block).to_vec();
    let mut result_groups = Vec::new();
    offset = 0;
    for rty in result_types {
        let n = rty.num_elements();
        let end = (offset + n).min(exit_params.len());
        result_groups.push(exit_params[offset..end].to_vec());
        offset = end;
    }
    if result_groups.is_empty() {
        result_groups.push(exit_params);
    }

    Ok(result_groups)
}

fn extract_return_predicate(
    body: &[InstrResult],
    value_map: &HashMap<ValueId, TensorVals>,
) -> Result<Value, String> {
    for ir in body.iter().rev() {
        if let Instruction::Return { operands } = &ir.instr
            && let Some(vid) = operands.first()
        {
            let vals = value_map
                .get(vid)
                .ok_or_else(|| format!("cond return value {:?} not found", vid))?;
            return Ok(vals[0]);
        }
    }
    Err("no return instruction in condition body".to_string())
}

fn extract_return_values(
    body: &[InstrResult],
    value_map: &HashMap<ValueId, TensorVals>,
) -> Result<Vec<Value>, String> {
    for ir in body.iter().rev() {
        if let Instruction::Return { operands } = &ir.instr {
            let mut vals = Vec::new();
            for vid in operands {
                let v = value_map
                    .get(vid)
                    .ok_or_else(|| format!("loop body return value {:?} not found", vid))?;
                vals.extend_from_slice(v);
            }
            return Ok(vals);
        }
    }
    Ok(vec![])
}

// ---------------------------------------------------------------------------
// Case — real branching with dispatch chain and merge block
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn lower_case(
    builder: &mut FunctionBuilder,
    index: &ValueId,
    branches: &[Vec<InstrResult>],
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    trt_ids: &TensorRtIds,
    func_abis: &HashMap<String, FuncAbi>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &mut HashMap<ValueId, TensorType>,
) -> Result<Vec<TensorVals>, String> {
    let idx_vals = get_vals(value_map, index)?;
    let idx = idx_vals[0];
    let n_branches = branches.len();

    let merge_block = builder.create_block();
    for rty in result_types {
        let ct = cranelift_type_for(rty.element_type);
        for _ in 0..rty.num_elements() {
            builder.append_block_param(merge_block, ct);
        }
    }

    let branch_blocks: Vec<Block> = (0..n_branches).map(|_| builder.create_block()).collect();

    let idx_ty = builder.func.dfg.value_type(idx);
    let empty_args: &[BlockArg] = &[];
    if n_branches == 1 {
        builder.ins().jump(branch_blocks[0], empty_args);
    } else if n_branches == 2 {
        let zero = builder.ins().iconst(idx_ty, 0);
        let cmp = builder.ins().icmp(IntCC::Equal, idx, zero);
        builder.ins().brif(
            cmp,
            branch_blocks[0],
            empty_args,
            branch_blocks[1],
            empty_args,
        );
    } else {
        for i in 0..n_branches - 1 {
            let cmp_val = builder.ins().iconst(idx_ty, i as i64);
            let cmp = builder.ins().icmp(IntCC::Equal, idx, cmp_val);
            if i == n_branches - 2 {
                builder.ins().brif(
                    cmp,
                    branch_blocks[i],
                    empty_args,
                    branch_blocks[n_branches - 1],
                    empty_args,
                );
            } else {
                let next_dispatch = builder.create_block();
                builder
                    .ins()
                    .brif(cmp, branch_blocks[i], empty_args, next_dispatch, empty_args);
                builder.switch_to_block(next_dispatch);
                builder.seal_block(next_dispatch);
            }
        }
    }

    for (i, branch) in branches.iter().enumerate() {
        builder.switch_to_block(branch_blocks[i]);
        builder.seal_block(branch_blocks[i]);

        // Branch bodies share the parent's value space. Clone the parent maps
        // so that captures of outer-scope values resolve correctly when the
        // parser preserves parent ValueIds for captured names.
        let mut br_vmap = value_map.clone();
        let mut br_tmap = type_map.clone();

        lower_body(
            builder,
            branch,
            ir_module,
            func_ids,
            libm_ids,
            trt_ids,
            func_abis,
            jit_module,
            &mut br_vmap,
            &mut br_tmap,
        )?;

        let ret_vals = extract_return_values(branch, &br_vmap)?;
        let ret_args = to_block_args(&ret_vals);
        builder.ins().jump(merge_block, &ret_args);
    }

    builder.switch_to_block(merge_block);
    builder.seal_block(merge_block);

    let merge_params = builder.block_params(merge_block).to_vec();
    let mut result_groups = Vec::new();
    let mut offset = 0;
    for rty in result_types {
        let n = rty.num_elements();
        let end = (offset + n).min(merge_params.len());
        result_groups.push(merge_params[offset..end].to_vec());
        offset = end;
    }
    if result_groups.is_empty() && !merge_params.is_empty() {
        result_groups.push(merge_params);
    }

    Ok(result_groups)
}

// ---------------------------------------------------------------------------
// Function call lowering
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn lower_call(
    builder: &mut FunctionBuilder,
    callee: &str,
    args: &[ValueId],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    func_abis: &HashMap<String, FuncAbi>,
    trt_ids: &TensorRtIds,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
    type_map: &HashMap<ValueId, TensorType>,
) -> Result<Vec<TensorVals>, String> {
    let fid = func_ids
        .get(callee)
        .ok_or_else(|| format!("unknown callee: {callee}"))?;
    let callee_def = ir_module
        .get_func(callee)
        .ok_or_else(|| format!("no func def for {callee}"))?;
    let callee_abi = func_abis.get(callee).copied().unwrap_or(FuncAbi::Scalar);

    let func_ref = jit_module.declare_func_in_func(*fid, builder.func);

    if callee_abi == FuncAbi::Pointer {
        let mut call_args = Vec::new();
        for (vid, (_param_vid, param_ty)) in args.iter().zip(callee_def.params.iter()) {
            let v = get_vals(value_map, vid)?;
            let byte_sz = param_ty.byte_size();
            let ss = builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                byte_sz as u32,
                3,
            ));
            let addr = builder.ins().stack_addr(ptr_type(), ss, 0);
            let elem_sz = param_ty.element_type.byte_size();
            for (j, &val) in v.iter().enumerate() {
                builder
                    .ins()
                    .store(MemFlags::trusted(), val, addr, (j * elem_sz) as i32);
            }
            call_args.push(addr);
        }

        let total_ret_bytes: usize = callee_def.result_types.iter().map(|t| t.byte_size()).sum();
        let ret_ss = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            total_ret_bytes.max(8) as u32,
            3,
        ));
        let ret_addr = builder.ins().stack_addr(ptr_type(), ret_ss, 0);
        call_args.push(ret_addr);

        let _call = builder.ins().call(func_ref, &call_args);

        let mut result_groups = Vec::new();
        let mut byte_offset = 0i32;
        for ret_ty in &callee_def.result_types {
            let n = ret_ty.num_elements();
            let ct = cranelift_type_for(ret_ty.element_type);
            let elem_sz = ret_ty.element_type.byte_size() as i32;
            let mut group = Vec::new();
            for j in 0..n {
                let v = builder.ins().load(
                    ct,
                    MemFlags::trusted(),
                    ret_addr,
                    byte_offset + (j as i32 * elem_sz),
                );
                group.push(v);
            }
            byte_offset += (n as i32) * elem_sz;
            result_groups.push(group);
        }
        return Ok(result_groups);
    }

    let callee_sret = needs_sret(&callee_def.result_types);

    let mut call_args = Vec::new();
    for vid in args {
        let v = get_vals(value_map, vid)?;
        call_args.extend_from_slice(v);
    }

    if callee_sret {
        let total_bytes: usize = callee_def.result_types.iter().map(|t| t.byte_size()).sum();
        let ss = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            total_bytes as u32,
            3,
        ));
        let ss_addr = builder.ins().stack_addr(ptr_type(), ss, 0);
        call_args.push(ss_addr);

        let _call = builder.ins().call(func_ref, &call_args);

        let mut result_groups = Vec::new();
        let mut byte_offset = 0i32;
        for ret_ty in &callee_def.result_types {
            let n = ret_ty.num_elements();
            let ct = cranelift_type_for(ret_ty.element_type);
            let elem_sz = ret_ty.element_type.byte_size() as i32;
            let mut group = Vec::new();
            for j in 0..n {
                let v = builder.ins().load(
                    ct,
                    MemFlags::trusted(),
                    ss_addr,
                    byte_offset + (j as i32 * elem_sz),
                );
                group.push(v);
            }
            byte_offset += (n as i32) * elem_sz;
            result_groups.push(group);
        }
        Ok(result_groups)
    } else {
        let call = builder.ins().call(func_ref, &call_args);
        let results: Vec<Value> = builder.inst_results(call).to_vec();

        let mut result_groups = Vec::new();
        let mut off = 0;
        for ret_ty in &callee_def.result_types {
            let n = ret_ty.num_elements();
            if off + n <= results.len() {
                result_groups.push(results[off..off + n].to_vec());
                off += n;
            }
        }
        if result_groups.is_empty() && !results.is_empty() {
            result_groups.push(results);
        }
        Ok(result_groups)
    }
}

// ---------------------------------------------------------------------------
// Helpers: constants, conversions, elementwise ops
// ---------------------------------------------------------------------------

fn lower_constant(
    builder: &mut FunctionBuilder,
    value: &ConstantValue,
    ty: &TensorType,
) -> TensorVals {
    let n = ty.num_elements();
    match value {
        ConstantValue::DenseScalar(sv) => {
            let v = scalar_to_cranelift(builder, sv, ty.element_type);
            if n == 1 { vec![v] } else { vec![v; n] }
        }
        ConstantValue::DenseArray(arr) => arr
            .iter()
            .map(|sv| scalar_to_cranelift(builder, sv, ty.element_type))
            .collect(),
        ConstantValue::DenseSplat(sv, _) => {
            let v = scalar_to_cranelift(builder, sv, ty.element_type);
            vec![v; n]
        }
    }
}

fn scalar_to_cranelift(builder: &mut FunctionBuilder, sv: &ScalarValue, et: ElementType) -> Value {
    match et {
        ElementType::F64 => builder.ins().f64const(sv.as_f64()),
        ElementType::F32 => builder.ins().f32const(sv.as_f64() as f32),
        ElementType::I64 | ElementType::UI64 => builder.ins().iconst(types::I64, sv.as_i64()),
        ElementType::I32 | ElementType::UI32 => builder.ins().iconst(types::I32, sv.as_i64()),
        ElementType::I1 => builder.ins().iconst(types::I8, sv.as_i64()),
    }
}

fn elementwise_binop(
    builder: &mut FunctionBuilder,
    l: &[Value],
    r: &[Value],
    f: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
) -> TensorVals {
    let n = l.len().max(r.len());
    (0..n)
        .map(|i| {
            let lv = if i < l.len() { l[i] } else { l[0] };
            let rv = if i < r.len() { r[i] } else { r[0] };
            f(builder, lv, rv)
        })
        .collect()
}

fn lower_int_binop(
    builder: &mut FunctionBuilder,
    value_map: &HashMap<ValueId, TensorVals>,
    lhs: &ValueId,
    rhs: &ValueId,
    f: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
) -> Result<Vec<TensorVals>, String> {
    let l = get_vals(value_map, lhs)?;
    let r = get_vals(value_map, rhs)?;
    Ok(vec![elementwise_binop(builder, l, r, f)])
}

fn lower_libm_unary(
    builder: &mut FunctionBuilder,
    value_map: &HashMap<ValueId, TensorVals>,
    operand: &ValueId,
    func_id: FuncId,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let vals = get_vals(value_map, operand)?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
    let out: Vec<Value> = vals
        .iter()
        .map(|&v| {
            let call = builder.ins().call(func_ref, &[v]);
            builder.inst_results(call)[0]
        })
        .collect();
    Ok(vec![out])
}

fn lower_libm_binary(
    builder: &mut FunctionBuilder,
    value_map: &HashMap<ValueId, TensorVals>,
    lhs: &ValueId,
    rhs: &ValueId,
    func_id: FuncId,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let l = get_vals(value_map, lhs)?;
    let r = get_vals(value_map, rhs)?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);
    let n = l.len().max(r.len());
    let out: Vec<Value> = (0..n)
        .map(|i| {
            let lv = if i < l.len() { l[i] } else { l[0] };
            let rv = if i < r.len() { r[i] } else { r[0] };
            let call = builder.ins().call(func_ref, &[lv, rv]);
            builder.inst_results(call)[0]
        })
        .collect();
    Ok(vec![out])
}

fn convert_value(
    builder: &mut FunctionBuilder,
    v: Value,
    src: &ElementType,
    dst: &ElementType,
) -> Value {
    match (src, dst) {
        _ if src == dst => v,
        (ElementType::F64, ElementType::I32) => builder.ins().fcvt_to_sint(types::I32, v),
        (ElementType::F64, ElementType::I64) => builder.ins().fcvt_to_sint(types::I64, v),
        (ElementType::F64, ElementType::UI32) => builder.ins().fcvt_to_uint(types::I32, v),
        (ElementType::F64, ElementType::F32) => builder.ins().fdemote(types::F32, v),
        (ElementType::F32, ElementType::F64) => builder.ins().fpromote(types::F64, v),
        (ElementType::I64, ElementType::F64) => builder.ins().fcvt_from_sint(types::F64, v),
        (ElementType::I64, ElementType::F32) => {
            let f = builder.ins().fcvt_from_sint(types::F64, v);
            builder.ins().fdemote(types::F32, f)
        }
        (ElementType::I32, ElementType::F64) => {
            let ext = builder.ins().sextend(types::I64, v);
            builder.ins().fcvt_from_sint(types::F64, ext)
        }
        (ElementType::UI32, ElementType::F64) => {
            let ext = builder.ins().uextend(types::I64, v);
            builder.ins().fcvt_from_uint(types::F64, ext)
        }
        (ElementType::I32, ElementType::I64) => builder.ins().sextend(types::I64, v),
        (ElementType::I32, ElementType::UI32) | (ElementType::UI32, ElementType::I32) => v,
        (ElementType::I64, ElementType::UI64) | (ElementType::UI64, ElementType::I64) => v,
        (ElementType::I64, ElementType::I32) => builder.ins().ireduce(types::I32, v),
        (ElementType::I64, ElementType::UI32)
        | (ElementType::UI64, ElementType::UI32)
        | (ElementType::UI64, ElementType::I32) => builder.ins().ireduce(types::I32, v),
        (ElementType::UI32, ElementType::I64) | (ElementType::UI32, ElementType::UI64) => {
            builder.ins().uextend(types::I64, v)
        }
        (ElementType::I1, ElementType::I32) => builder.ins().uextend(types::I32, v),
        (ElementType::I1, ElementType::I64) => builder.ins().uextend(types::I64, v),
        (ElementType::I1, ElementType::F64) => {
            let ext = builder.ins().uextend(types::I64, v);
            builder.ins().fcvt_from_uint(types::F64, ext)
        }
        (ElementType::I1, ElementType::F32) => {
            let ext = builder.ins().uextend(types::I32, v);
            builder.ins().fcvt_from_uint(types::F32, ext)
        }
        (ElementType::I32, ElementType::I1) => builder.ins().ireduce(types::I8, v),
        (ElementType::I64, ElementType::I1) => builder.ins().ireduce(types::I8, v),
        (ElementType::F64, ElementType::I1) => {
            let i = builder.ins().fcvt_to_sint(types::I32, v);
            builder.ins().ireduce(types::I8, i)
        }
        (ElementType::UI32, ElementType::UI64) => builder.ins().uextend(types::I64, v),
        (ElementType::UI64, ElementType::F64) => builder.ins().fcvt_from_uint(types::F64, v),
        (ElementType::F64, ElementType::UI64) => builder.ins().fcvt_to_uint(types::I64, v),
        (ElementType::I32, ElementType::F32) => builder.ins().fcvt_from_sint(types::F32, v),
        _ => v,
    }
}

// ---------------------------------------------------------------------------
// Shape operation helpers
// ---------------------------------------------------------------------------

fn broadcast_values(
    vals: &[Value],
    target_shape: &[i64],
    broadcast_dims: &[i64],
    src_shape: &[i64],
) -> Vec<Value> {
    let n: usize = target_shape.iter().product::<i64>() as usize;
    if vals.len() == 1 {
        return vec![vals[0]; n];
    }
    if vals.len() == n && broadcast_dims.is_empty() {
        return vals.to_vec();
    }

    let out_rank = target_shape.len();
    let mut out_strides = vec![1usize; out_rank];
    for i in (0..out_rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1] as usize;
    }

    let src_rank = src_shape.len();
    let mut src_strides = vec![1usize; src_rank.max(1)];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    let mut out = Vec::with_capacity(n);
    for flat_out in 0..n {
        let mut remaining = flat_out;
        let mut src_flat = 0;
        for (d, &stride) in out_strides.iter().enumerate() {
            let idx = remaining / stride;
            remaining %= stride;
            if let Some(pos) = broadcast_dims.iter().position(|&bd| bd as usize == d)
                && pos < src_rank
                && src_shape[pos] > 1
            {
                src_flat += idx * src_strides[pos];
            }
        }
        out.push(vals[src_flat.min(vals.len() - 1)]);
    }
    out
}

fn lower_concatenate_nd(
    parts: &[(&TensorVals, TensorType)],
    dim: usize,
    out_ty: &TensorType,
) -> TensorVals {
    let out_shape: Vec<usize> = out_ty.shape.iter().map(|&d| d as usize).collect();
    let rank = out_shape.len();
    let n = out_ty.num_elements();

    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let part_infos: Vec<(Vec<usize>, Vec<usize>)> = parts
        .iter()
        .map(|(_, ty)| {
            let shape: Vec<usize> = ty.shape.iter().map(|&d| d as usize).collect();
            let mut strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            (shape, strides)
        })
        .collect();

    let mut result = Vec::with_capacity(n);
    for flat_out in 0..n {
        let mut remaining = flat_out;
        let mut out_indices = vec![0usize; rank];
        for d in 0..rank {
            out_indices[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }

        let concat_idx = out_indices[dim];
        let mut part_idx = 0;
        let mut offset_in_dim = 0;
        for (i, (shape, _)) in part_infos.iter().enumerate() {
            if concat_idx < offset_in_dim + shape[dim] {
                part_idx = i;
                break;
            }
            offset_in_dim += shape[dim];
        }

        let local_dim_idx = concat_idx - offset_in_dim;
        let (ref _shape, ref strides) = part_infos[part_idx];
        let mut src_flat = 0;
        for d in 0..rank {
            let idx = if d == dim {
                local_dim_idx
            } else {
                out_indices[d]
            };
            src_flat += idx * strides[d];
        }

        result.push(parts[part_idx].0[src_flat]);
    }
    result
}

fn slice_tensor(vals: &[Value], src_shape: &[i64], starts: &[i64], limits: &[i64]) -> Vec<Value> {
    if src_shape.is_empty() {
        return vals.to_vec();
    }
    let rank = src_shape.len();
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as usize;
    }

    let slice_shape: Vec<usize> = (0..rank)
        .map(|d| {
            let s = starts.get(d).copied().unwrap_or(0) as usize;
            let l = limits.get(d).copied().unwrap_or(src_shape[d]) as usize;
            l - s
        })
        .collect();
    let n_out: usize = slice_shape.iter().product();
    let mut out = Vec::with_capacity(n_out);

    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * slice_shape[i + 1];
    }

    for flat_out in 0..n_out {
        let mut src_flat = 0;
        let mut remaining = flat_out;
        for d in 0..rank {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            let src_coord = coord + starts.get(d).copied().unwrap_or(0) as usize;
            src_flat += src_coord * src_strides[d];
        }
        if src_flat < vals.len() {
            out.push(vals[src_flat]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dot product / matrix operations
// ---------------------------------------------------------------------------

fn lower_dot_general(
    builder: &mut FunctionBuilder,
    l: &[Value],
    r: &[Value],
    l_ty: &TensorType,
    r_ty: &TensorType,
    out_ty: &TensorType,
    dims: &DotDims,
) -> TensorVals {
    let n = out_ty.num_elements();

    // Batched dot product: batching_dims=[0]x[0], contracting_dims=[1]x[1]
    // tensor<BxK> . tensor<BxK> -> tensor<B>
    if !dims.lhs_batch.is_empty()
        && l_ty.rank() == 2
        && r_ty.rank() == 2
        && dims.lhs_batch == [0]
        && dims.rhs_batch == [0]
    {
        let batch = l_ty.shape[0] as usize;
        let k = l_ty.shape[1] as usize;
        let mut out = Vec::new();
        for b in 0..batch {
            let zero = builder.ins().f64const(0.0);
            let mut acc = zero;
            for i in 0..k {
                let lv = l[b * k + i];
                let rv = r[b * k + i];
                let prod = builder.ins().fmul(lv, rv);
                acc = builder.ins().fadd(acc, prod);
            }
            out.push(acc);
        }
        return out;
    }

    if l_ty.is_scalar() && r_ty.is_scalar() {
        return vec![builder.ins().fmul(l[0], r[0])];
    }

    if l_ty.rank() == 1 && r_ty.rank() == 1 {
        let zero = builder.ins().f64const(0.0);
        let mut acc = zero;
        for i in 0..l.len().min(r.len()) {
            let prod = builder.ins().fmul(l[i], r[i]);
            acc = builder.ins().fadd(acc, prod);
        }
        return vec![acc];
    }

    if l_ty.rank() == 1 && r_ty.rank() == 2 {
        let rhs_contract_dim = dims.rhs_contracting[0] as usize;
        let k = l_ty.shape[0] as usize;
        let rhs_rows = r_ty.shape[0] as usize;
        let rhs_cols = r_ty.shape[1] as usize;
        let out_size = if rhs_contract_dim == 1 {
            rhs_rows
        } else {
            rhs_cols
        };
        let mut out = Vec::new();
        for i in 0..out_size {
            let zero = builder.ins().f64const(0.0);
            let mut acc = zero;
            for j in 0..k {
                let lv = l[j];
                let rv = if rhs_contract_dim == 1 {
                    r[i * rhs_cols + j]
                } else {
                    r[j * rhs_cols + i]
                };
                let prod = builder.ins().fmul(lv, rv);
                acc = builder.ins().fadd(acc, prod);
            }
            out.push(acc);
        }
        return out;
    }

    if l_ty.rank() == 2 && r_ty.rank() == 1 {
        let rows = l_ty.shape[0] as usize;
        let cols = l_ty.shape[1] as usize;
        let mut out = Vec::new();
        for row in 0..rows {
            let zero = builder.ins().f64const(0.0);
            let mut acc = zero;
            for col in 0..cols {
                let lv = l[row * cols + col];
                let rv = r[col];
                let prod = builder.ins().fmul(lv, rv);
                acc = builder.ins().fadd(acc, prod);
            }
            out.push(acc);
        }
        return out;
    }

    if l_ty.rank() == 2 && r_ty.rank() == 2 {
        let m = l_ty.shape[0] as usize;
        let k = l_ty.shape[1] as usize;
        let n_cols = r_ty.shape[1] as usize;
        let mut out = Vec::new();
        for row in 0..m {
            for col in 0..n_cols {
                let zero = builder.ins().f64const(0.0);
                let mut acc = zero;
                for i in 0..k {
                    let lv = l[row * k + i];
                    let rv = r[i * n_cols + col];
                    let prod = builder.ins().fmul(lv, rv);
                    acc = builder.ins().fadd(acc, prod);
                }
                out.push(acc);
            }
        }
        return out;
    }

    let zero = builder.ins().f64const(0.0);
    vec![zero; n]
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

fn lower_reduce(
    builder: &mut FunctionBuilder,
    vals: &[Value],
    init_vals: &[Value],
    src_ty: &TensorType,
    out_ty: &TensorType,
    op: &ReduceOp,
    dimensions: &[i64],
) -> TensorVals {
    let n_out = out_ty.num_elements();

    if src_ty.rank() == 1 && dimensions == [0] {
        let mut acc = init_vals[0];
        for &v in vals {
            acc = apply_reduce_op(builder, acc, v, op);
        }
        return vec![acc];
    }

    if src_ty.rank() == 2 && dimensions == [0] {
        let rows = src_ty.shape[0] as usize;
        let cols = src_ty.shape[1] as usize;
        let mut out = Vec::new();
        for c in 0..cols {
            let mut acc = if c < init_vals.len() {
                init_vals[c]
            } else {
                init_vals[0]
            };
            for r in 0..rows {
                let idx = r * cols + c;
                if idx < vals.len() {
                    acc = apply_reduce_op(builder, acc, vals[idx], op);
                }
            }
            out.push(acc);
        }
        return out;
    }

    if src_ty.rank() == 2 && dimensions == [1] {
        let rows = src_ty.shape[0] as usize;
        let cols = src_ty.shape[1] as usize;
        let mut out = Vec::new();
        for r in 0..rows {
            let mut acc = init_vals[0];
            for c in 0..cols {
                let idx = r * cols + c;
                if idx < vals.len() {
                    acc = apply_reduce_op(builder, acc, vals[idx], op);
                }
            }
            out.push(acc);
        }
        return out;
    }

    init_vals[..n_out.min(init_vals.len())].to_vec()
}

fn apply_reduce_op(builder: &mut FunctionBuilder, acc: Value, v: Value, op: &ReduceOp) -> Value {
    let val_type = builder.func.dfg.value_type(acc);
    match op {
        ReduceOp::Add => {
            if val_type.is_float() {
                builder.ins().fadd(acc, v)
            } else {
                builder.ins().iadd(acc, v)
            }
        }
        ReduceOp::Minimum => {
            if val_type.is_float() {
                let cmp = builder.ins().fcmp(FloatCC::LessThan, acc, v);
                builder.ins().select(cmp, acc, v)
            } else {
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, acc, v);
                builder.ins().select(cmp, acc, v)
            }
        }
        ReduceOp::Maximum => {
            if val_type.is_float() {
                let cmp = builder.ins().fcmp(FloatCC::GreaterThan, acc, v);
                builder.ins().select(cmp, acc, v)
            } else {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, acc, v);
                builder.ins().select(cmp, acc, v)
            }
        }
        ReduceOp::And => builder.ins().band(acc, v),
        ReduceOp::Or => builder.ins().bor(acc, v),
    }
}

// ---------------------------------------------------------------------------
// Gather — stack-slot based element lookup for small tensors
// ---------------------------------------------------------------------------

fn lower_gather(
    builder: &mut FunctionBuilder,
    operand: &[Value],
    indices: &[Value],
    src_ty: &TensorType,
    idx_ty: &TensorType,
    out_ty: &TensorType,
    dims: &GatherDims,
    slice_sizes: &[i64],
) -> TensorVals {
    let n = out_ty.num_elements();
    let et = out_ty.element_type;
    let ct = cranelift_type_for(et);
    let elem_sz = et.byte_size();

    if operand.is_empty() || indices.is_empty() {
        return vec![make_zero(builder, et); n];
    }

    let src_shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
    let src_rank = src_shape.len();
    let out_shape: Vec<usize> = out_ty.shape.iter().map(|&d| d as usize).collect();
    let out_rank = out_shape.len();
    let idx_shape: Vec<usize> = idx_ty.shape.iter().map(|&d| d as usize).collect();
    let index_vector_dim = dims.index_vector_dim as usize;

    let mut src_strides = vec![1usize; src_rank];
    for i in (0..src_rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    let mut idx_strides = vec![1usize; idx_shape.len()];
    for i in (0..idx_shape.len().saturating_sub(1)).rev() {
        idx_strides[i] = idx_strides[i + 1] * idx_shape[i + 1];
    }

    let total_bytes = operand.len() * elem_sz;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);
    for (i, &v) in operand.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base, (i * elem_sz) as i32);
    }

    // Batch dims of the indices tensor = all dims except index_vector_dim
    let idx_rank = idx_shape.len();
    let batch_dims: Vec<usize> = (0..idx_rank).filter(|&d| d != index_vector_dim).collect();
    let index_depth = if index_vector_dim < idx_rank {
        idx_shape[index_vector_dim]
    } else {
        1
    };

    // Batch shape = indices shape with index_vector_dim removed
    let batch_shape: Vec<usize> = batch_dims.iter().map(|&d| idx_shape[d]).collect();
    let n_batch: usize = batch_shape.iter().product::<usize>().max(1);
    let batch_rank = batch_shape.len();

    // Offset shape = slice_sizes with collapsed dims removed
    let offset_shape: Vec<usize> = (0..src_rank)
        .filter(|d| !dims.collapsed_slice_dims.contains(&(*d as i64)))
        .map(|d| slice_sizes[d] as usize)
        .collect();
    let offset_rank = offset_shape.len();
    let n_offset: usize = offset_shape.iter().product::<usize>().max(1);

    let mut results = Vec::with_capacity(n);

    for flat_out in 0..n {
        // Decompose flat output index into multi-index
        let mut out_indices = vec![0usize; out_rank];
        {
            let mut rem = flat_out;
            for d in (0..out_rank).rev() {
                if out_shape[d] > 0 {
                    out_indices[d] = rem % out_shape[d];
                    rem /= out_shape[d];
                }
            }
        }

        // Split output indices into batch part and offset part
        let mut batch_idx = vec![0usize; batch_rank];
        let mut offset_idx = vec![0usize; offset_rank];
        let offset_dims = &dims.offset_dims;
        let mut bi = 0;
        let mut oi = 0;
        for d in 0..out_rank {
            if offset_dims.contains(&(d as i64)) {
                if oi < offset_rank {
                    offset_idx[oi] = out_indices[d];
                    oi += 1;
                }
            } else {
                if bi < batch_rank {
                    batch_idx[bi] = out_indices[d];
                    bi += 1;
                }
            }
        }

        // Look up start indices from the indices tensor
        let mut start_index = vec![0usize; index_depth];
        for k in 0..index_depth {
            // Build the multi-index into the indices tensor
            let mut idx_multi = vec![0usize; idx_rank];
            let mut b = 0;
            for d in 0..idx_rank {
                if d == index_vector_dim {
                    idx_multi[d] = k;
                } else {
                    if b < batch_rank {
                        idx_multi[d] = batch_idx[b];
                    }
                    b += 1;
                }
            }
            let mut flat_idx = 0;
            for d in 0..idx_rank {
                flat_idx += idx_multi[d] * idx_strides[d];
            }
            // Read the index value at compile time from the SSA value
            // (will be resolved at runtime via stack load)
            start_index[k] = flat_idx;
        }

        // Build source multi-index
        let mut src_idx = vec![0usize; src_rank];

        // Place start indices via start_index_map (runtime values)
        // Place offset indices into non-collapsed dims
        let mut oi2 = 0;
        for d in 0..src_rank {
            if dims.collapsed_slice_dims.contains(&(d as i64)) {
                // This dim is collapsed; its source index comes from start_index_map
            } else {
                if oi2 < offset_rank {
                    src_idx[d] = offset_idx[oi2];
                    oi2 += 1;
                }
            }
        }

        // The start_index values are RUNTIME (SSA) values from the indices tensor.
        // We need to compute the source address dynamically.
        // Static part: offset contribution to flat index
        let mut static_offset = 0usize;
        for d in 0..src_rank {
            if !dims.collapsed_slice_dims.contains(&(d as i64)) {
                static_offset += src_idx[d] * src_strides[d];
            }
        }

        // Dynamic part: start_index contributions
        // For each k in start_index_map, add indices[start_index[k]] * src_strides[start_index_map[k]]
        let static_byte_off = (static_offset * elem_sz) as i32;
        let mut addr = builder.ins().iadd_imm(base, static_byte_off as i64);

        for (k, &mapped_dim) in dims.start_index_map.iter().enumerate() {
            if k >= index_depth {
                break;
            }
            let flat_idx = start_index[k];
            if flat_idx >= indices.len() {
                continue;
            }
            let raw_idx = indices[flat_idx];
            let idx_i64 = if builder.func.dfg.value_type(raw_idx) == types::I64 {
                raw_idx
            } else {
                builder.ins().sextend(types::I64, raw_idx)
            };
            let stride_bytes = (src_strides[mapped_dim as usize] * elem_sz) as i64;
            let byte_off = builder.ins().imul_imm(idx_i64, stride_bytes);
            addr = builder.ins().iadd(addr, byte_off);
        }

        let v = builder.ins().load(ct, MemFlags::trusted(), addr, 0);
        results.push(v);
    }

    results
}

fn lower_transpose(
    vals: &[Value],
    src_ty: &TensorType,
    out_ty: &TensorType,
    permutation: &[i64],
) -> TensorVals {
    let n = out_ty.num_elements();
    if vals.len() != n || src_ty.shape.len() != permutation.len() {
        return vals.to_vec();
    }

    let rank = src_ty.shape.len();
    let src_shape: Vec<usize> = src_ty.shape.iter().map(|&d| d as usize).collect();
    let out_shape: Vec<usize> = out_ty.shape.iter().map(|&d| d as usize).collect();

    let mut src_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let mut result = vec![vals[0]; n];
    for (flat_out, slot) in result.iter_mut().enumerate() {
        let mut remaining = flat_out;
        let mut out_indices = vec![0usize; rank];
        for d in 0..rank {
            out_indices[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }
        let mut src_flat = 0;
        for d in 0..rank {
            let src_dim = permutation[d] as usize;
            src_flat += out_indices[d] * src_strides[src_dim];
        }
        *slot = vals[src_flat];
    }
    result
}

fn lower_dynamic_slice(
    builder: &mut FunctionBuilder,
    vals: &[Value],
    start_indices: &[Value],
    src_ty: &TensorType,
    slice_sizes: &[i64],
) -> TensorVals {
    let et = src_ty.element_type;
    let ct = cranelift_type_for(et);
    let elem_sz = et.byte_size();

    let total_bytes = vals.len() * elem_sz;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);
    for (i, &v) in vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base, (i * elem_sz) as i32);
    }

    let rank = src_ty.shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * src_ty.shape[i + 1] as usize;
    }

    // Compute the flat byte offset from dynamic start indices.
    // Per StableHLO spec, indices are clamped to [0, dim_size - slice_size].
    let mut flat_offset = builder.ins().iconst(types::I64, 0);
    for d in 0..rank {
        if d < start_indices.len() {
            let idx = start_indices[d];
            let idx_i64 = if builder.func.dfg.value_type(idx) == types::I64 {
                idx
            } else {
                builder.ins().sextend(types::I64, idx)
            };
            let max_idx = src_ty.shape[d] - slice_sizes.get(d).copied().unwrap_or(1);
            let max_val = builder.ins().iconst(types::I64, max_idx);
            let zero = builder.ins().iconst(types::I64, 0);
            let clamped_lo = {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, idx_i64, zero);
                builder.ins().select(cmp, idx_i64, zero)
            };
            let clamped = {
                let cmp = builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, clamped_lo, max_val);
                builder.ins().select(cmp, clamped_lo, max_val)
            };
            let stride_bytes = (strides[d] * elem_sz) as i64;
            let contrib = builder.ins().imul_imm(clamped, stride_bytes);
            flat_offset = builder.ins().iadd(flat_offset, contrib);
        }
    }
    let slice_base = builder.ins().iadd(base, flat_offset);

    let out_n: usize = slice_sizes.iter().product::<i64>() as usize;
    let mut results = Vec::with_capacity(out_n);
    for i in 0..out_n {
        let v = builder
            .ins()
            .load(ct, MemFlags::trusted(), slice_base, (i * elem_sz) as i32);
        results.push(v);
    }
    results
}

fn lower_dynamic_update_slice(
    builder: &mut FunctionBuilder,
    base_vals: &[Value],
    update_vals: &[Value],
    start_indices: &[Value],
    src_ty: &TensorType,
    _upd_ty: &TensorType,
) -> TensorVals {
    let et = src_ty.element_type;
    let ct = cranelift_type_for(et);
    let elem_sz = et.byte_size();

    let total_bytes = base_vals.len() * elem_sz;
    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total_bytes as u32,
        3,
    ));
    let base_addr = builder.ins().stack_addr(ptr_type(), ss, 0);
    for (i, &v) in base_vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base_addr, (i * elem_sz) as i32);
    }

    let rank = src_ty.shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * src_ty.shape[i + 1] as usize;
    }

    let upd_shape = &_upd_ty.shape;
    let mut flat_offset = builder.ins().iconst(types::I64, 0);
    for d in 0..rank.min(start_indices.len()) {
        let idx = start_indices[d];
        let idx_i64 = if builder.func.dfg.value_type(idx) == types::I64 {
            idx
        } else {
            builder.ins().sextend(types::I64, idx)
        };
        let upd_dim = upd_shape.get(d).copied().unwrap_or(1);
        let max_idx = src_ty.shape[d] - upd_dim;
        let max_val = builder.ins().iconst(types::I64, max_idx);
        let zero = builder.ins().iconst(types::I64, 0);
        let clamped_lo = {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, idx_i64, zero);
            builder.ins().select(cmp, idx_i64, zero)
        };
        let clamped = {
            let cmp = builder
                .ins()
                .icmp(IntCC::SignedLessThan, clamped_lo, max_val);
            builder.ins().select(cmp, clamped_lo, max_val)
        };
        let stride_bytes = (strides[d] * elem_sz) as i64;
        let contrib = builder.ins().imul_imm(clamped, stride_bytes);
        flat_offset = builder.ins().iadd(flat_offset, contrib);
    }
    let update_addr = builder.ins().iadd(base_addr, flat_offset);

    for (i, &v) in update_vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, update_addr, (i * elem_sz) as i32);
    }

    let mut results = Vec::with_capacity(base_vals.len());
    for i in 0..base_vals.len() {
        let v = builder
            .ins()
            .load(ct, MemFlags::trusted(), base_addr, (i * elem_sz) as i32);
        results.push(v);
    }
    results
}

#[allow(clippy::too_many_arguments)]
fn lower_custom_call(
    builder: &mut FunctionBuilder,
    call_target: &str,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
    libm_ids: &LibmIds,
    backend_config: &HashMap<String, i64>,
) -> Result<Vec<TensorVals>, String> {
    if call_target.starts_with("lapack_dgesdd") {
        return lower_svd_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dgetrf") {
        return lower_lu_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dtrsm") {
        return lower_trsm_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
            backend_config,
        );
    }
    if call_target.starts_with("lapack_dpotrf") {
        return lower_cholesky_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dgeqrf") {
        return lower_qr_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dorgqr") {
        return lower_orgqr_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    if call_target.starts_with("lapack_dsyevd") {
        return lower_syevd_custom_call(
            builder,
            operands,
            result_types,
            value_map,
            type_map,
            jit_module,
        );
    }
    Err(format!("unsupported custom_call target: {call_target}"))
}

// ---------------------------------------------------------------------------
// Host LAPACK functions backed by faer
// ---------------------------------------------------------------------------

fn row_major_to_col_major(row: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut col = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            col[j * m + i] = row[i * n + j];
        }
    }
    col
}

fn col_major_to_row_major(col: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut row = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            row[i * n + j] = col[j * m + i];
        }
    }
    row
}

extern "C" fn cranelift_svd(
    a_ptr: *const f64,
    n: usize,
    u_ptr: *mut f64,
    s_ptr: *mut f64,
    vt_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    use faer::prelude::*;
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let u_out = unsafe { std::slice::from_raw_parts_mut(u_ptr, n * n) };
    let s_out = unsafe { std::slice::from_raw_parts_mut(s_ptr, n) };
    let vt_out = unsafe { std::slice::from_raw_parts_mut(vt_ptr, n * n) };

    // Input is row-major from the IR; faer needs column-major
    let col_data = row_major_to_col_major(a, n, n);
    let mat = faer::mat::from_column_major_slice(&col_data, n, n);
    let svd = mat.thin_svd();

    // U output: convert from faer column-major back to row-major for the IR
    let u_col = svd.u();
    for i in 0..n {
        for j in 0..n {
            u_out[i * n + j] = u_col.read(i, j);
        }
    }
    let s_diag = svd.s_diagonal();
    for (i, val) in s_out.iter_mut().enumerate() {
        *val = s_diag.read(i);
    }
    // VT output: row i of VT = column i of V transposed
    let v = svd.v();
    for i in 0..n {
        for j in 0..n {
            vt_out[i * n + j] = v.read(j, i);
        }
    }

    unsafe { *info_ptr = 0 };
}

extern "C" fn cranelift_lu(
    a_ptr: *const f64,
    m: usize,
    n: usize,
    lu_ptr: *mut f64,
    ipiv_ptr: *mut i32,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let lu_out = unsafe { std::slice::from_raw_parts_mut(lu_ptr, m * n) };
    let ipiv_out = unsafe { std::slice::from_raw_parts_mut(ipiv_ptr, m.min(n)) };

    let min_mn = m.min(n);

    // Manual LU with partial pivoting producing LAPACK-compatible ipiv (1-indexed)
    let mut mat = vec![0.0f64; m * n];
    mat.copy_from_slice(a);
    let idx = |r: usize, c: usize| r * n + c;

    for k in 0..min_mn {
        // Find pivot
        let mut max_val = mat[idx(k, k)].abs();
        let mut max_row = k;
        for i in (k + 1)..m {
            let v = mat[idx(i, k)].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        // LAPACK 1-indexed pivot
        ipiv_out[k] = max_row as i32 + 1;

        // Swap rows k and max_row
        if max_row != k {
            for j in 0..n {
                mat.swap(idx(k, j), idx(max_row, j));
            }
        }

        let pivot = mat[idx(k, k)];
        if pivot.abs() < 1e-300 {
            continue;
        }

        for i in (k + 1)..m {
            mat[idx(i, k)] /= pivot;
            let factor = mat[idx(i, k)];
            for j in (k + 1)..n {
                mat[idx(i, j)] -= factor * mat[idx(k, j)];
            }
        }
    }

    lu_out.copy_from_slice(&mat);
    unsafe { *info_ptr = 0 };
}

extern "C" fn cranelift_trsm(
    a_ptr: *const f64,
    b_ptr: *const f64,
    m: usize,
    n: usize,
    nrhs: usize,
    uplo: u8,
    diag: u8,
    out_ptr: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, m * nrhs) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, m * nrhs) };

    let a_col = row_major_to_col_major(a, m, n);
    let a_mat = faer::mat::from_column_major_slice(&a_col, m, n);

    let mut x_col = row_major_to_col_major(b, m, nrhs);
    let x_mat = faer::mat::from_column_major_slice_mut(&mut x_col, m, nrhs);

    let is_lower = uplo == b'L';
    let is_unit = diag == b'U';

    if is_lower {
        if is_unit {
            faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                a_mat,
                x_mat,
                faer::Parallelism::None,
            );
        } else {
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                a_mat,
                x_mat,
                faer::Parallelism::None,
            );
        }
    } else if is_unit {
        faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
            a_mat,
            x_mat,
            faer::Parallelism::None,
        );
    } else {
        faer::linalg::triangular_solve::solve_upper_triangular_in_place(
            a_mat,
            x_mat,
            faer::Parallelism::None,
        );
    }

    let result = col_major_to_row_major(&x_col, m, nrhs);
    out.copy_from_slice(&result);
}

extern "C" fn cranelift_cholesky(a_ptr: *const f64, n: usize, l_ptr: *mut f64, info_ptr: *mut i32) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let l_out = unsafe { std::slice::from_raw_parts_mut(l_ptr, n * n) };

    let mut col_data = row_major_to_col_major(a, n, n);
    let mat = faer::mat::from_column_major_slice_mut(&mut col_data, n, n);

    let req = faer::linalg::cholesky::llt::compute::cholesky_in_place_req::<f64>(
        n,
        faer::Parallelism::None,
        Default::default(),
    )
    .unwrap();
    let mut work = vec![0u8; req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::PodStack::new(&mut work);

    let result = faer::linalg::cholesky::llt::compute::cholesky_in_place(
        mat,
        Default::default(),
        faer::Parallelism::None,
        stack,
        Default::default(),
    );

    // Zero the strict upper triangle in column-major: elements where row < col
    for col in 0..n {
        for row in 0..col {
            col_data[col * n + row] = 0.0;
        }
    }

    let row_data = col_major_to_row_major(&col_data, n, n);
    l_out.copy_from_slice(&row_data);

    unsafe { *info_ptr = if result.is_ok() { 0 } else { 1 } };
}

extern "C" fn cranelift_qr(
    a_ptr: *const f64,
    m: usize,
    n: usize,
    qr_ptr: *mut f64,
    tau_ptr: *mut f64,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, m * n) };
    let qr_out = unsafe { std::slice::from_raw_parts_mut(qr_ptr, m * n) };
    let tau_out = unsafe { std::slice::from_raw_parts_mut(tau_ptr, m.min(n)) };

    let mut col_data = row_major_to_col_major(a, m, n);
    let mut factors = faer::mat::from_column_major_slice_mut(&mut col_data, m, n);

    let min_mn = m.min(n);
    let blocksize = faer::linalg::qr::no_pivoting::compute::recommended_blocksize::<f64>(m, n);
    let mut h_data = vec![0.0f64; blocksize * min_mn];
    let mut householder = faer::mat::from_column_major_slice_mut(&mut h_data, blocksize, min_mn);

    let params = Default::default();
    let parallelism = faer::Parallelism::None;
    let req = faer::linalg::qr::no_pivoting::compute::qr_in_place_req::<f64>(
        m,
        n,
        blocksize,
        parallelism,
        params,
    )
    .unwrap();
    let mut work = vec![0u8; req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::PodStack::new(&mut work);

    faer::linalg::qr::no_pivoting::compute::qr_in_place(
        factors.as_mut(),
        householder.as_mut(),
        parallelism,
        stack,
        params,
    );

    let row_data = col_major_to_row_major(&col_data, m, n);
    qr_out.copy_from_slice(&row_data);

    // Store first row of householder block as tau (the reflector coefficients)
    for i in 0..min_mn {
        tau_out[i] = h_data[i * blocksize];
    }
}

extern "C" fn cranelift_orgqr(
    qr_ptr: *const f64,
    tau_ptr: *const f64,
    m: usize,
    n: usize,
    q_ptr: *mut f64,
) {
    let qr_data = unsafe { std::slice::from_raw_parts(qr_ptr, m * n) };
    let tau = unsafe { std::slice::from_raw_parts(tau_ptr, m.min(n)) };
    let q_out = unsafe { std::slice::from_raw_parts_mut(q_ptr, m * n) };

    let min_mn = m.min(n);
    let blocksize = faer::linalg::qr::no_pivoting::compute::recommended_blocksize::<f64>(m, n);

    // Rebuild the full block householder factor matrix from the scalar tau values
    // The first row of each block column contains the tau value
    let mut h_data = vec![0.0f64; blocksize * min_mn];
    for i in 0..min_mn {
        h_data[i * blocksize] = tau[i];
    }
    let householder = faer::mat::from_column_major_slice(&h_data, blocksize, min_mn);

    let qr_col = row_major_to_col_major(qr_data, m, n);
    let factors = faer::mat::from_column_major_slice(&qr_col, m, n);

    let mut q = faer::Mat::<f64>::zeros(m, m);
    q.as_mut().diagonal_mut().column_vector_mut().fill(1.0);

    let req = faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_req::<f64>(
        m, blocksize, m,
    ).unwrap();
    let mut work = vec![0u8; req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::PodStack::new(&mut work);

    faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        factors,
        householder,
        faer::Conj::No,
        q.as_mut(),
        faer::Parallelism::None,
        stack,
    );

    for i in 0..m {
        for j in 0..n {
            q_out[i * n + j] = q.read(i, j);
        }
    }
}

extern "C" fn cranelift_syevd(
    a_ptr: *const f64,
    n: usize,
    eigvecs_ptr: *mut f64,
    eigvals_ptr: *mut f64,
    info_ptr: *mut i32,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, n * n) };
    let eigvecs_out = unsafe { std::slice::from_raw_parts_mut(eigvecs_ptr, n * n) };
    let eigvals_out = unsafe { std::slice::from_raw_parts_mut(eigvals_ptr, n) };

    let col_data = row_major_to_col_major(a, n, n);
    let mat = faer::mat::from_column_major_slice(&col_data, n, n);
    let eigen = faer::linalg::solvers::SelfAdjointEigendecomposition::new(mat, faer::Side::Lower);

    let u = eigen.u();
    for i in 0..n {
        for j in 0..n {
            eigvecs_out[i * n + j] = u.read(i, j);
        }
    }
    let s = eigen.s();
    for (i, val) in eigvals_out.iter_mut().enumerate() {
        *val = s.column_vector().read(i);
    }

    unsafe { *info_ptr = 0 };
}

// ---------------------------------------------------------------------------
// Cranelift IR lowering for each LAPACK custom_call
// ---------------------------------------------------------------------------

fn store_f64_vals(
    builder: &mut FunctionBuilder,
    vals: &[cranelift_codegen::ir::Value],
    base: cranelift_codegen::ir::Value,
    offset: i32,
) {
    for (i, &v) in vals.iter().enumerate() {
        builder
            .ins()
            .store(MemFlags::trusted(), v, base, offset + (i * 8) as i32);
    }
}

fn load_f64_vals(
    builder: &mut FunctionBuilder,
    count: usize,
    base: cranelift_codegen::ir::Value,
    offset: i32,
) -> Vec<cranelift_codegen::ir::Value> {
    (0..count)
        .map(|i| {
            builder.ins().load(
                types::F64,
                MemFlags::trusted(),
                base,
                offset + (i * 8) as i32,
            )
        })
        .collect()
}

fn load_i32_vals(
    builder: &mut FunctionBuilder,
    count: usize,
    base: cranelift_codegen::ir::Value,
    offset: i32,
) -> Vec<cranelift_codegen::ir::Value> {
    (0..count)
        .map(|i| {
            builder.ins().load(
                types::I32,
                MemFlags::trusted(),
                base,
                offset + (i * 4) as i32,
            )
        })
        .collect()
}

fn lower_svd_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let a_vals = get_vals(value_map, &operands[0])?;
    let n = (a_vals.len() as f64).sqrt() as usize;
    if n * n != a_vals.len() {
        return Err(format!(
            "SVD: non-square matrix ({} elements)",
            a_vals.len()
        ));
    }

    let a_bytes = n * n * 8;
    let u_bytes = n * n * 8;
    let s_bytes = n * 8;
    let vt_bytes = n * n * 8;
    let info_bytes = 4;
    let total = a_bytes + u_bytes + s_bytes + vt_bytes + info_bytes;

    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);

    store_f64_vals(builder, a_vals, base, 0);

    let u_off = a_bytes as i32;
    let s_off = (a_bytes + u_bytes) as i32;
    let vt_off = (a_bytes + u_bytes + s_bytes) as i32;
    let info_off = (a_bytes + u_bytes + s_bytes + vt_bytes) as i32;

    let u_ptr = builder.ins().iadd_imm(base, u_off as i64);
    let s_ptr = builder.ins().iadd_imm(base, s_off as i64);
    let vt_ptr = builder.ins().iadd_imm(base, vt_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);

    // fn(a_ptr, n, u_ptr, s_ptr, vt_ptr, info_ptr)
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(types::I64));
    for _ in 0..4 {
        sig.params.push(AbiParam::new(ptr_type()));
    }
    let func_id = jit_module
        .declare_function("__cranelift_svd", Linkage::Import, &sig)
        .map_err(|e| format!("declare svd: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);

    let n_val = builder.ins().iconst(types::I64, n as i64);
    builder
        .ins()
        .call(func_ref, &[base, n_val, u_ptr, s_ptr, vt_ptr, info_ptr]);

    // XLA dgesdd_ffi convention: (A_overwritten, sigma, U, VT, info)
    // With JOBZ='S', A is overwritten with U columns, so result[0] = U.
    let mut result_groups = Vec::new();
    result_groups.push(load_f64_vals(builder, n * n, base, u_off)); // [0] A overwritten = U
    result_groups.push(load_f64_vals(builder, n, base, s_off)); // [1] sigma
    result_groups.push(load_f64_vals(builder, n * n, base, u_off)); // [2] U
    if result_types.len() > 3 {
        result_groups.push(load_f64_vals(builder, n * n, base, vt_off)); // [3] VT
    }
    if result_types.len() > 4 {
        let info_val = builder
            .ins()
            .load(types::I32, MemFlags::trusted(), base, info_off);
        result_groups.push(vec![info_val]); // [4] info
    }

    Ok(result_groups)
}

fn lower_lu_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let a_vals = get_vals(value_map, &operands[0])?;
    let n = (a_vals.len() as f64).sqrt() as usize;
    if n * n != a_vals.len() {
        return Err(format!("LU: non-square matrix ({} elements)", a_vals.len()));
    }
    let m = n;

    let lu_bytes = m * n * 8;
    let ipiv_bytes = m.min(n) * 4;
    let info_bytes = 4;
    let total = lu_bytes + lu_bytes + ipiv_bytes + info_bytes;

    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);

    store_f64_vals(builder, a_vals, base, 0);

    let lu_off = lu_bytes as i32;
    let ipiv_off = (lu_bytes + lu_bytes) as i32;
    let info_off = (lu_bytes + lu_bytes + ipiv_bytes) as i32;

    let lu_ptr = builder.ins().iadd_imm(base, lu_off as i64);
    let ipiv_ptr = builder.ins().iadd_imm(base, ipiv_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);

    // fn(a_ptr, m, n, lu_ptr, ipiv_ptr, info_ptr) = 1 ptr + 2 i64 + 3 ptr
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(ptr_type()));
    let func_id = jit_module
        .declare_function("__cranelift_lu", Linkage::Import, &sig)
        .map_err(|e| format!("declare lu: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);

    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    builder
        .ins()
        .call(func_ref, &[base, m_val, n_val, lu_ptr, ipiv_ptr, info_ptr]);

    let mut result_groups = Vec::new();
    result_groups.push(load_f64_vals(builder, m * n, base, lu_off));
    result_groups.push(load_i32_vals(builder, m.min(n), base, ipiv_off));
    let info_val = builder
        .ins()
        .load(types::I32, MemFlags::trusted(), base, info_off);
    result_groups.push(vec![info_val]);

    Ok(result_groups)
}

#[allow(clippy::too_many_arguments)]
fn lower_trsm_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
    backend_config: &HashMap<String, i64>,
) -> Result<Vec<TensorVals>, String> {
    let a_vals = get_vals(value_map, &operands[0])?;
    let b_vals = get_vals(value_map, &operands[1])?;

    let rt = &result_types[0];
    let (m, nrhs) = if rt.shape.len() == 2 {
        (rt.shape[0] as usize, rt.shape[1] as usize)
    } else {
        return Err("TRSM: expected 2D result".to_string());
    };
    let n = (a_vals.len() as f64).sqrt() as usize;

    let a_bytes = a_vals.len() * 8;
    let b_bytes = b_vals.len() * 8;
    let out_bytes = m * nrhs * 8;
    let total = a_bytes + b_bytes + out_bytes;

    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);

    store_f64_vals(builder, a_vals, base, 0);
    let b_off = a_bytes as i32;
    store_f64_vals(builder, b_vals, base, b_off);
    let out_off = (a_bytes + b_bytes) as i32;
    let b_ptr = builder.ins().iadd_imm(base, b_off as i64);
    let out_ptr = builder.ins().iadd_imm(base, out_off as i64);

    let uplo = *backend_config.get("uplo").unwrap_or(&(b'L' as i64)) as u8;
    let diag = *backend_config.get("diag").unwrap_or(&(b'N' as i64)) as u8;

    // fn(a_ptr, b_ptr, m, n, nrhs, uplo, diag, out_ptr)
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    for _ in 0..2 {
        sig.params.push(AbiParam::new(ptr_type()));
    }
    for _ in 0..3 {
        sig.params.push(AbiParam::new(types::I64));
    }
    for _ in 0..2 {
        sig.params.push(AbiParam::new(types::I8));
    }
    sig.params.push(AbiParam::new(ptr_type()));
    let func_id = jit_module
        .declare_function("__cranelift_trsm", Linkage::Import, &sig)
        .map_err(|e| format!("declare trsm: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);

    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_imm = builder.ins().iconst(types::I64, n as i64);
    let nrhs_val = builder.ins().iconst(types::I64, nrhs as i64);
    let uplo_val = builder.ins().iconst(types::I8, uplo as i64);
    let diag_val = builder.ins().iconst(types::I8, diag as i64);
    builder.ins().call(
        func_ref,
        &[
            base, b_ptr, m_val, n_imm, nrhs_val, uplo_val, diag_val, out_ptr,
        ],
    );

    Ok(vec![load_f64_vals(builder, m * nrhs, base, out_off)])
}

fn lower_cholesky_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let a_vals = get_vals(value_map, &operands[0])?;
    let n = (a_vals.len() as f64).sqrt() as usize;
    if n * n != a_vals.len() {
        return Err(format!(
            "Cholesky: non-square matrix ({} elements)",
            a_vals.len()
        ));
    }

    let a_bytes = n * n * 8;
    let l_bytes = n * n * 8;
    let info_bytes = 4;
    let total = a_bytes + l_bytes + info_bytes;

    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);

    store_f64_vals(builder, a_vals, base, 0);

    let l_off = a_bytes as i32;
    let info_off = (a_bytes + l_bytes) as i32;
    let l_ptr = builder.ins().iadd_imm(base, l_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);

    // fn(a_ptr, n, l_ptr, info_ptr)
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(ptr_type()));
    let func_id = jit_module
        .declare_function("__cranelift_cholesky", Linkage::Import, &sig)
        .map_err(|e| format!("declare cholesky: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);

    let n_val = builder.ins().iconst(types::I64, n as i64);
    builder
        .ins()
        .call(func_ref, &[base, n_val, l_ptr, info_ptr]);

    let mut result_groups = Vec::new();
    result_groups.push(load_f64_vals(builder, n * n, base, l_off));
    let info_val = builder
        .ins()
        .load(types::I32, MemFlags::trusted(), base, info_off);
    result_groups.push(vec![info_val]);

    Ok(result_groups)
}

fn lower_qr_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let a_vals = get_vals(value_map, &operands[0])?;
    let rt = &result_types[0];
    let (m, n) = if rt.shape.len() == 2 {
        (rt.shape[0] as usize, rt.shape[1] as usize)
    } else {
        return Err("QR: expected 2D result".to_string());
    };

    let a_bytes = m * n * 8;
    let qr_bytes = m * n * 8;
    let tau_bytes = m.min(n) * 8;
    let total = a_bytes + qr_bytes + tau_bytes;

    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);

    store_f64_vals(builder, a_vals, base, 0);

    let qr_off = a_bytes as i32;
    let tau_off = (a_bytes + qr_bytes) as i32;
    let qr_ptr = builder.ins().iadd_imm(base, qr_off as i64);
    let tau_ptr = builder.ins().iadd_imm(base, tau_off as i64);

    // fn(a_ptr, m, n, qr_ptr, tau_ptr)
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(ptr_type()));
    let func_id = jit_module
        .declare_function("__cranelift_qr", Linkage::Import, &sig)
        .map_err(|e| format!("declare qr: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);

    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    builder
        .ins()
        .call(func_ref, &[base, m_val, n_val, qr_ptr, tau_ptr]);

    let result_groups = vec![
        load_f64_vals(builder, m * n, base, qr_off),
        load_f64_vals(builder, m.min(n), base, tau_off),
    ];

    Ok(result_groups)
}

fn lower_orgqr_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let qr_vals = get_vals(value_map, &operands[0])?;
    let tau_vals = get_vals(value_map, &operands[1])?;

    let rt = &result_types[0];
    let (m, n) = if rt.shape.len() == 2 {
        (rt.shape[0] as usize, rt.shape[1] as usize)
    } else {
        return Err("ORGQR: expected 2D result".to_string());
    };

    let qr_bytes = qr_vals.len() * 8;
    let tau_bytes = tau_vals.len() * 8;
    let q_bytes = m * n * 8;
    let total = qr_bytes + tau_bytes + q_bytes;

    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);

    store_f64_vals(builder, qr_vals, base, 0);
    let tau_off = qr_bytes as i32;
    store_f64_vals(builder, tau_vals, base, tau_off);
    let q_off = (qr_bytes + tau_bytes) as i32;

    let tau_ptr = builder.ins().iadd_imm(base, tau_off as i64);
    let q_ptr = builder.ins().iadd_imm(base, q_off as i64);

    // fn(qr_ptr, tau_ptr, m, n, q_ptr)
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(ptr_type()));
    let func_id = jit_module
        .declare_function("__cranelift_orgqr", Linkage::Import, &sig)
        .map_err(|e| format!("declare orgqr: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);

    let m_val = builder.ins().iconst(types::I64, m as i64);
    let n_val = builder.ins().iconst(types::I64, n as i64);
    builder
        .ins()
        .call(func_ref, &[base, tau_ptr, m_val, n_val, q_ptr]);

    Ok(vec![load_f64_vals(builder, m * n, base, q_off)])
}

fn lower_syevd_custom_call(
    builder: &mut FunctionBuilder,
    operands: &[ValueId],
    result_types: &[TensorType],
    value_map: &HashMap<ValueId, TensorVals>,
    _type_map: &HashMap<ValueId, TensorType>,
    jit_module: &mut JITModule,
) -> Result<Vec<TensorVals>, String> {
    let a_vals = get_vals(value_map, &operands[0])?;
    let n = (a_vals.len() as f64).sqrt() as usize;
    if n * n != a_vals.len() {
        return Err(format!(
            "SYEVD: non-square matrix ({} elements)",
            a_vals.len()
        ));
    }

    let a_bytes = n * n * 8;
    let eigvecs_bytes = n * n * 8;
    let eigvals_bytes = n * 8;
    let info_bytes = 4;
    let total = a_bytes + eigvecs_bytes + eigvals_bytes + info_bytes;

    let ss = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        total as u32,
        3,
    ));
    let base = builder.ins().stack_addr(ptr_type(), ss, 0);

    store_f64_vals(builder, a_vals, base, 0);

    let eigvecs_off = a_bytes as i32;
    let eigvals_off = (a_bytes + eigvecs_bytes) as i32;
    let info_off = (a_bytes + eigvecs_bytes + eigvals_bytes) as i32;

    let eigvecs_ptr = builder.ins().iadd_imm(base, eigvecs_off as i64);
    let eigvals_ptr = builder.ins().iadd_imm(base, eigvals_off as i64);
    let info_ptr = builder.ins().iadd_imm(base, info_off as i64);

    // fn(a_ptr, n, eigvecs_ptr, eigvals_ptr, info_ptr)
    let mut sig = jit_module.make_signature();
    sig.call_conv = jit_module.isa().default_call_conv();
    sig.params.push(AbiParam::new(ptr_type()));
    sig.params.push(AbiParam::new(types::I64));
    for _ in 0..3 {
        sig.params.push(AbiParam::new(ptr_type()));
    }
    let func_id = jit_module
        .declare_function("__cranelift_syevd", Linkage::Import, &sig)
        .map_err(|e| format!("declare syevd: {e}"))?;
    let func_ref = jit_module.declare_func_in_func(func_id, builder.func);

    let n_val = builder.ins().iconst(types::I64, n as i64);
    builder
        .ins()
        .call(func_ref, &[base, n_val, eigvecs_ptr, eigvals_ptr, info_ptr]);

    let mut result_groups = Vec::new();
    result_groups.push(load_f64_vals(builder, n * n, base, eigvecs_off));
    result_groups.push(load_f64_vals(builder, n, base, eigvals_off));
    let info_val = builder
        .ins()
        .load(types::I32, MemFlags::trusted(), base, info_off);
    result_groups.push(vec![info_val]);

    Ok(result_groups)
}
