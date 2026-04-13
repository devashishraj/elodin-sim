use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

fn u32_buf(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_u32s(buf: &[u8]) -> Vec<u32> {
    buf.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i64_buf(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn run_mlir(mlir: &str, inputs: &[&[u8]], output_sizes: &[usize]) -> Vec<Vec<u8>> {
    let module = parse_module(mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let fn_ptr = compiled.get_main_fn();
    let tick_fn: unsafe extern "C" fn(*const *const u8, *mut *mut u8) =
        unsafe { std::mem::transmute(fn_ptr) };

    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
    let mut output_bufs: Vec<Vec<u8>> = output_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe {
        tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr());
    }

    output_bufs
}

#[test]
fn test_threefry_round() {
    let ball_mlir = include_str!("../testdata/ball.stablehlo.mlir");
    let module = parse_module(ball_mlir).expect("parse failed");

    let func = module.get_func("closed_call").expect("closed_call not found");
    eprintln!(
        "closed_call params: {}",
        func.params
            .iter()
            .map(|(_, t)| format!("{t}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "closed_call result_types: {}",
        func.result_types
            .iter()
            .map(|t| format!("{t}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
fn test_inner_prng() {
    let mlir = include_str!("../testdata/ball.stablehlo.mlir");

    let module = parse_module(mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");

    let inner_func = module.get_func("inner").expect("inner not found");
    eprintln!(
        "inner params: {}",
        inner_func
            .params
            .iter()
            .map(|(_, t)| format!("{t}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "inner result_types: {}",
        inner_func
            .result_types
            .iter()
            .map(|t| format!("{t}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let main_func = module.main_func().unwrap();
    eprintln!(
        "main params: {}",
        main_func
            .params
            .iter()
            .map(|(_, t)| format!("{t}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
fn test_simple_shift_right_logical_ui32() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<ui32> {
    %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<ui32>
    return %0 : tensor<ui32>
  }
}
"#;
    let out = run_mlir(mlir, &[&u32_buf(&[0xDEADBEEF]), &u32_buf(&[4])], &[4]);
    let result = read_u32s(&out[0]);
    let expected = 0xDEADBEEFu32 >> 4;
    assert_eq!(result[0], expected, "got {:#010X} expected {:#010X}", result[0], expected);
}

#[test]
fn test_ui32_add_overflow() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<ui32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<ui32>
    return %0 : tensor<ui32>
  }
}
"#;
    let out = run_mlir(
        mlir,
        &[&u32_buf(&[0xFFFFFFFF]), &u32_buf(&[1])],
        &[4],
    );
    let result = read_u32s(&out[0]);
    assert_eq!(result[0], 0u32, "overflow should wrap");
}

#[test]
fn test_i64_to_ui32_convert() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>) -> tensor<ui32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    return %0 : tensor<ui32>
  }
}
"#;
    let out = run_mlir(mlir, &[&i64_buf(&[4294967295])], &[4]);
    let result = read_u32s(&out[0]);
    assert_eq!(result[0], 0xFFFFFFFF, "should be all ones");
}

#[test]
fn test_ui64_shift_right_logical() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
    %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<ui64>
    return %0 : tensor<ui64>
  }
}
"#;
    let val: u64 = 0x0000000100000002;
    let out = run_mlir(
        mlir,
        &[&val.to_le_bytes().to_vec(), &32u64.to_le_bytes().to_vec()],
        &[8],
    );
    let result = u64::from_le_bytes(out[0][..8].try_into().unwrap());
    assert_eq!(result, 1u64, "high word should shift to position 0");
}

#[test]
fn test_bitcast_convert_ui64_to_f64() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<ui64>) -> tensor<f64> {
    %0 = stablehlo.bitcast_convert %arg0 : (tensor<ui64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let val: u64 = 0x3FF0000000000000; // 1.0 as f64 bits
    let out = run_mlir(mlir, &[&val.to_le_bytes().to_vec()], &[8]);
    let result = f64::from_le_bytes(out[0][..8].try_into().unwrap());
    assert!((result - 1.0).abs() < 1e-15, "should be 1.0, got {result}");
}
