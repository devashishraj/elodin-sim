use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

fn f64_buf(vals: &[f64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn read_f64s(buf: &[u8]) -> Vec<f64> {
    buf.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
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
    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };
    output_bufs
}

#[test]
fn test_sret_call_3x6_return() {
    // Test a function that returns tensor<3x6xf64> (18 elements) via sret
    let mlir = r#"
module @module {
  func.func private @add_tensors(%arg0: tensor<3x6xf64>, %arg1: tensor<3x6xf64>) -> tensor<3x6xf64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3x6xf64>
    return %0 : tensor<3x6xf64>
  }
  func.func public @main(%arg0: tensor<3x6xf64>, %arg1: tensor<3x6xf64>) -> tensor<3x6xf64> {
    %0 = call @add_tensors(%arg0, %arg1) : (tensor<3x6xf64>, tensor<3x6xf64>) -> tensor<3x6xf64>
    return %0 : tensor<3x6xf64>
  }
}
"#;
    let a: Vec<f64> = (1..=18).map(|i| i as f64).collect();
    let b: Vec<f64> = (1..=18).map(|i| (i * 10) as f64).collect();
    let in0 = f64_buf(&a);
    let in1 = f64_buf(&b);
    let out = run_mlir(mlir, &[&in0, &in1], &[144]);
    let result = read_f64s(&out[0]);
    let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    assert_eq!(result, expected);
}

#[test]
fn test_sret_call_multi_return() {
    // Test a function that returns (tensor<3x6xf64>, tensor<i64>) via sret -- like closed_call
    let mlir = r#"
module @module {
  func.func private @compute(%arg0: tensor<3x6xf64>) -> (tensor<3x6xf64>, tensor<i64>) {
    %c = stablehlo.constant dense<42> : tensor<i64>
    %0 = stablehlo.add %arg0, %arg0 : tensor<3x6xf64>
    return %0, %c : tensor<3x6xf64>, tensor<i64>
  }
  func.func public @main(%arg0: tensor<3x6xf64>) -> (tensor<3x6xf64>, tensor<i64>) {
    %0:2 = call @compute(%arg0) : (tensor<3x6xf64>) -> (tensor<3x6xf64>, tensor<i64>)
    return %0#0, %0#1 : tensor<3x6xf64>, tensor<i64>
  }
}
"#;
    let a: Vec<f64> = (1..=18).map(|i| i as f64).collect();
    let in0 = f64_buf(&a);
    let out = run_mlir(mlir, &[&in0], &[144, 8]);
    let f64_result = read_f64s(&out[0]);
    let expected: Vec<f64> = a.iter().map(|x| x * 2.0).collect();
    assert_eq!(f64_result, expected);
    let i64_result = i64::from_le_bytes(out[1][..8].try_into().unwrap());
    assert_eq!(i64_result, 42);
}
