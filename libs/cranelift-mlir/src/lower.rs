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
use cranelift_module::{FuncId, Linkage, Module};

use crate::ir::*;

pub struct CompiledModule {
    module: JITModule,
    main_fn_id: FuncId,
}

impl CompiledModule {
    pub fn get_main_fn(&self) -> *const u8 {
        self.module.get_finalized_function(self.main_fn_id)
    }
}

type TensorVals = Vec<Value>;

const MAX_REG_RETURNS: usize = 8;

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
    })
}

// ---------------------------------------------------------------------------
// Module compilation entry point
// ---------------------------------------------------------------------------

pub fn compile_module(ir_module: &crate::ir::Module) -> Result<CompiledModule, String> {
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
    jit_builder.symbol("erf_inv_impl", erf_inv_scalar as *const u8);

    let mut jit_module = JITModule::new(jit_builder);
    let func_ids = declare_all_functions(ir_module, &mut jit_module, call_conv)?;
    let libm_ids = declare_libm_functions(&mut jit_module, call_conv)?;

    for func_def in &ir_module.functions {
        let fid = func_ids[&func_def.name];
        define_function(
            &mut jit_module,
            func_def,
            ir_module,
            &func_ids,
            &libm_ids,
            fid,
        )?;
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
) -> Result<HashMap<String, FuncId>, String> {
    let mut ids = HashMap::new();
    for func_def in &ir_module.functions {
        let mut sig = jit_module.make_signature();
        sig.call_conv = call_conv;

        if func_def.name == "main" {
            sig.params.push(AbiParam::new(ptr_type()));
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

fn define_function(
    jit_module: &mut JITModule,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
    fid: FuncId,
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

        if func_def.name == "main" {
            lower_main_body(
                &mut builder,
                func_def,
                ir_module,
                func_ids,
                libm_ids,
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

#[allow(clippy::too_many_arguments)]
fn lower_callee_body(
    builder: &mut FunctionBuilder,
    func_def: &crate::ir::FuncDef,
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
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
        Instruction::ErfInv { operand } => {
            lower_libm_unary(builder, value_map, operand, libm_ids.erf_inv, jit_module)
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
            let n = rt.num_elements();
            if vals.len() == 1 {
                Ok(vec![vec![vals[0]; n]])
            } else if vals.len() == n {
                Ok(vec![vals])
            } else {
                Ok(vec![broadcast_values(&vals, &rt.shape, broadcast_dims)])
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
            dimension: _,
        } => {
            let mut all_vals = Vec::new();
            for vid in operands {
                let v = get_vals(value_map, vid)?;
                all_vals.extend_from_slice(v);
            }
            Ok(vec![all_vals])
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

        Instruction::Iota { dimension: _ } => {
            let n = rt.num_elements();
            let ct = cranelift_type_for(rt.element_type);
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    if is_float(rt.element_type) {
                        builder.ins().f64const(i as f64)
                    } else {
                        builder.ins().iconst(ct, i as i64)
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
            Ok(vec![lower_gather(
                builder,
                &vals,
                &idx_vals,
                &src_ty,
                &rt,
                dims,
                slice_sizes,
            )])
        }

        // ----- Function call -----
        Instruction::Call { callee, args } => lower_call(
            builder, callee, args, ir_module, func_ids, jit_module, value_map,
        ),

        // ----- While loop (real Cranelift loop blocks) -----
        Instruction::While {
            cond_body,
            loop_body,
            init_values,
        } => lower_while(
            builder,
            cond_body,
            loop_body,
            init_values,
            result_types,
            ir_module,
            func_ids,
            libm_ids,
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
            jit_module,
            value_map,
            type_map,
        ),

        Instruction::Return { .. } => Ok(vec![]),
    }
}

// ---------------------------------------------------------------------------
// While loop — real Cranelift loop with header/body/exit blocks
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn lower_while(
    builder: &mut FunctionBuilder,
    cond_body: &[InstrResult],
    loop_body: &[InstrResult],
    init_values: &[ValueId],
    result_types: &[TensorType],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    libm_ids: &LibmIds,
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
        cond_vmap.insert(
            ValueId(i as u32),
            header_params[offset..offset + n].to_vec(),
        );
        cond_tmap.insert(ValueId(i as u32), ty.clone());
        offset += n;
    }

    lower_body(
        builder,
        cond_body,
        ir_module,
        func_ids,
        libm_ids,
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
        body_vmap.insert(ValueId(i as u32), body_params[offset..offset + n].to_vec());
        body_tmap.insert(ValueId(i as u32), ty.clone());
        offset += n;
    }

    lower_body(
        builder,
        loop_body,
        ir_module,
        func_ids,
        libm_ids,
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

fn lower_call(
    builder: &mut FunctionBuilder,
    callee: &str,
    args: &[ValueId],
    ir_module: &crate::ir::Module,
    func_ids: &HashMap<String, FuncId>,
    jit_module: &mut JITModule,
    value_map: &mut HashMap<ValueId, TensorVals>,
) -> Result<Vec<TensorVals>, String> {
    let fid = func_ids
        .get(callee)
        .ok_or_else(|| format!("unknown callee: {callee}"))?;
    let callee_def = ir_module
        .get_func(callee)
        .ok_or_else(|| format!("no func def for {callee}"))?;

    let func_ref = jit_module.declare_func_in_func(*fid, builder.func);
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
        (ElementType::I32, ElementType::I1) => builder.ins().ireduce(types::I8, v),
        _ => v,
    }
}

// ---------------------------------------------------------------------------
// Shape operation helpers
// ---------------------------------------------------------------------------

fn broadcast_values(vals: &[Value], target_shape: &[i64], _broadcast_dims: &[i64]) -> Vec<Value> {
    let n: usize = target_shape.iter().product::<i64>() as usize;
    if vals.len() == 1 {
        return vec![vals[0]; n];
    }
    if vals.len() == n {
        return vals.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(vals[i % vals.len()]);
    }
    out
}

fn slice_tensor(vals: &[Value], src_shape: &[i64], starts: &[i64], limits: &[i64]) -> Vec<Value> {
    if src_shape.is_empty() {
        return vals.to_vec();
    }
    if src_shape.len() == 1 {
        let s = starts[0] as usize;
        let e = limits[0] as usize;
        return vals[s..e.min(vals.len())].to_vec();
    }
    if src_shape.len() == 2 {
        let cols = src_shape[1] as usize;
        let r0 = starts[0] as usize;
        let r1 = limits[0] as usize;
        let c0 = starts.get(1).copied().unwrap_or(0) as usize;
        let c1 = limits.get(1).copied().unwrap_or(cols as i64) as usize;
        let mut out = Vec::new();
        for r in r0..r1 {
            for c in c0..c1 {
                let idx = r * cols + c;
                if idx < vals.len() {
                    out.push(vals[idx]);
                }
            }
        }
        return out;
    }
    let s = starts[0] as usize;
    let e = limits[0] as usize;
    vals[s..e.min(vals.len())].to_vec()
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
    _dims: &DotDims,
) -> TensorVals {
    let n = out_ty.num_elements();

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
    match op {
        ReduceOp::Add => builder.ins().fadd(acc, v),
        ReduceOp::Minimum => {
            let cmp = builder.ins().fcmp(FloatCC::LessThan, acc, v);
            builder.ins().select(cmp, acc, v)
        }
        ReduceOp::Maximum => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, acc, v);
            builder.ins().select(cmp, acc, v)
        }
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
    out_ty: &TensorType,
    _dims: &GatherDims,
    _slice_sizes: &[i64],
) -> TensorVals {
    let n = out_ty.num_elements();
    let et = out_ty.element_type;
    let ct = cranelift_type_for(et);
    let elem_sz = et.byte_size();

    if operand.is_empty() || indices.is_empty() {
        return vec![make_zero(builder, et); n];
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

    let mut results = Vec::with_capacity(n);
    for &raw_idx in indices.iter().take(n) {
        let idx_i64 = if builder.func.dfg.value_type(raw_idx) == types::I64 {
            raw_idx
        } else {
            builder.ins().sextend(types::I64, raw_idx)
        };
        let byte_offset = builder.ins().imul_imm(idx_i64, elem_sz as i64);
        let addr = builder.ins().iadd(base, byte_offset);
        let v = builder.ins().load(ct, MemFlags::trusted(), addr, 0);
        results.push(v);
    }

    while results.len() < n {
        results.push(make_zero(builder, et));
    }

    results
}
