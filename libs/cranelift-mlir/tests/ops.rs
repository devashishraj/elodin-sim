use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

fn run_mlir(mlir: &str, inputs: &[&[u8]], output_sizes: &[usize]) -> Vec<Vec<u8>> {
    let module = parse_module(mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let fn_ptr = compiled.get_main_fn();
    let tick_fn: TickFn = unsafe { std::mem::transmute(fn_ptr) };

    let input_ptrs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
    let mut output_bufs: Vec<Vec<u8>> = output_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { tick_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()) };

    output_bufs
}

fn f64_buf(vals: &[f64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i64_buf(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_buf(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f64s(buf: &[u8]) -> Vec<f64> {
    buf.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i64s(buf: &[u8]) -> Vec<i64> {
    buf.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn assert_f64_close(actual: f64, expected: f64) {
    let diff = (actual - expected).abs();
    let denom = expected.abs().max(1e-15);
    assert!(
        diff / denom < 1e-10,
        "expected {expected}, got {actual} (relative error: {:.2e})",
        diff / denom
    );
}

fn assert_f64s_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let denom = e.abs().max(1e-15);
        assert!(
            diff / denom < 1e-10,
            "element {i}: expected {e}, got {a} (relative error: {:.2e})",
            diff / denom
        );
    }
}

fn reference_erf_inv(x: f64) -> f64 {
    let a = x.abs();
    if a >= 1.0 {
        return if x > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if a <= 0.7 {
        let x2 = x * x;
        let r = x
            * (1.0
                + x2 * (-0.14110320156679688
                    + x2 * (0.00536696519760513 + x2 * (-0.00012490409090537))));
        let s = 1.0
            + x2 * (-0.46997548816375946 + x2 * (0.03894404780498262 + x2 * -0.00056807290818498));
        r / s
    } else {
        let w = (-((1.0 - a) / 2.0).ln()).sqrt();
        let r = if w < 2.5 {
            let w1 = w - 1.6;
            (0.15504_70003_11693
                + w1 * (1.24016_81885_33806
                    + w1 * (0.22667_58861_00498
                        + w1 * (-0.02552_42513_12362
                            + w1 * (-0.00491_55570_37938 + w1 * 0.00033_70766_71552)))))
                / (1.0
                    + w1 * (0.39145_53073_58388
                        + w1 * (0.06580_20454_42746
                            + w1 * (-0.00597_53879_79153
                                + w1 * (-0.00041_52271_33582 + w1 * 0.00001_65478_12831)))))
        } else {
            let w1 = w - 3.0;
            (1.001_674_066_314_44
                + w1 * (4.42945_75023_12524
                    + w1 * (3.37760_09990_92073
                        + w1 * (-0.32709_33711_11814
                            + w1 * (-0.81891_49028_77613
                                + w1 * (-0.17256_76541_42671 + w1 * 0.00108_00138_76602))))))
                / (1.0
                    + w1 * (3.54388_92476_56405
                        + w1 * (3.84683_82938_07354
                            + w1 * (0.51600_22689_27052
                                + w1 * (-0.41951_56654_41421
                                    + w1 * (-0.05508_77290_78127 + w1 * 0.00312_06800_28513))))))
        };
        if x < 0.0 { -r } else { r }
    }
}

// ---- Arithmetic ops ----

#[test]
fn test_add() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let in1 = f64_buf(&[4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_subtract() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[5.0, 7.0, 9.0]);
    let in1 = f64_buf(&[1.0, 2.0, 3.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_multiply() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[2.0, 3.0, 4.0]);
    let in1 = f64_buf(&[5.0, 6.0, 7.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 18.0, 28.0]);
}

#[test]
fn test_divide() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[10.0]);
    let in1 = f64_buf(&[2.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 5.0);
}

#[test]
fn test_negate() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.negate %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-1.0, 0.0, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert!(result[1].abs() < 1e-15);
    assert_f64_close(result[2], -3.0);
}

#[test]
fn test_sqrt() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.sqrt %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[9.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 3.0);
}

#[test]
fn test_maximum() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[3.0]);
    let in1 = f64_buf(&[5.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 5.0);
}

// ---- Comparison ----

#[test]
fn test_compare_lt() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<i1> {
    %0 = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}
"#;
    let in0 = f64_buf(&[3.0]);
    let in1 = f64_buf(&[5.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[1]);
    assert_eq!(out[0][0], 1, "3.0 < 5.0 should be true");
}

// ---- Constants ----

#[test]
fn test_constant() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<2xf64> {
    %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let out = run_mlir(mlir, &[], &[16]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0]);
}

// ---- Shape ops ----

#[test]
fn test_reshape() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<1xf64>) -> tensor<f64> {
    %0 = stablehlo.reshape %arg0 : (tensor<1xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[42.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 42.0);
}

#[test]
fn test_broadcast_in_dim() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[7.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[7.0, 7.0, 7.0]);
}

#[test]
fn test_slice() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>) -> tensor<2xf64> {
    %0 = stablehlo.slice %arg0 [1:3] : (tensor<4xf64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0, 40.0]);
    let out = run_mlir(mlir, &[&in0], &[16]);
    assert_f64s_close(&read_f64s(&out[0]), &[20.0, 30.0]);
}

#[test]
fn test_concatenate() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2xf64>, %arg1: tensor<1xf64>) -> tensor<3xf64> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0]);
    let in1 = f64_buf(&[3.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

// ---- Type conversion ----

#[test]
fn test_convert_i64_to_f64() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = i64_buf(&[42]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 42.0);
}

#[test]
fn test_iota() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let out = run_mlir(mlir, &[], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[0.0, 1.0, 2.0]);
}

// ---- Integer bitwise ops ----

#[test]
fn test_xor() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.xor %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0xFF]);
    let in1 = i64_buf(&[0x0F]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 0xF0);
}

#[test]
fn test_shift_left() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.shift_left %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[1]);
    let in1 = i64_buf(&[3]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 8);
}

#[test]
fn test_shift_right_logical() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[16]);
    let in1 = i64_buf(&[2]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 4);
}

#[test]
fn test_or() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0xF0]);
    let in1 = i64_buf(&[0x0F]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 0xFF);
}

#[test]
fn test_and() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.and %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0xFF]);
    let in1 = i64_buf(&[0x0F]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 0x0F);
}

// ---- Dot product ----

#[test]
fn test_dot_general_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<f64> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let in1 = f64_buf(&[4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 32.0);
}

// ---- Reduce ----

#[test]
fn test_reduce_add() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 6.0);
}

// ---- Transcendental ----

#[test]
fn test_erf_inv() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = chlo.erf_inv %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let input = 0.5_f64;
    let expected = 0.4769362762044699_f64;
    let in0 = f64_buf(&[input]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], expected);
}

// ---- Function call ----

#[test]
fn test_call() {
    let mlir = r#"
module @module {
  func.func private @double(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = call @double(%arg0) : (tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[5.0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 10.0);
}

// ---- Case ----

#[test]
fn test_case_two_branch() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i32>) -> tensor<f64> {
    %0 = stablehlo.case(%arg0) ({
      %cst0 = stablehlo.constant dense<10.0> : tensor<f64>
      stablehlo.return %cst0 : tensor<f64>
    }, {
      %cst1 = stablehlo.constant dense<20.0> : tensor<f64>
      stablehlo.return %cst1 : tensor<f64>
    }) : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#;
    let in0 = i32_buf(&[0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 10.0);

    let in1 = i32_buf(&[1]);
    let out = run_mlir(mlir, &[&in1], &[8]);
    assert_f64_close(read_f64s(&out[0])[0], 20.0);
}

// ---- While loop ----

#[test]
fn test_while_loop() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.while(%iter = %arg0) : tensor<i64>
    cond {
      %limit = stablehlo.constant dense<5> : tensor<i64>
      %cmp = stablehlo.compare LT, %iter, %limit, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    }
    do {
      %one = stablehlo.constant dense<1> : tensor<i64>
      %next = stablehlo.add %iter, %one : tensor<i64>
      stablehlo.return %next : tensor<i64>
    }
    return %0 : tensor<i64>
  }
}
"#;
    let in0 = i64_buf(&[0]);
    let out = run_mlir(mlir, &[&in0], &[8]);
    assert_eq!(read_i64s(&out[0])[0], 5);
}

// ---- Transpose ----

#[test]
fn test_transpose_2d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}
"#;
    // Input: [[1,2,3],[4,5,6]]
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&in0], &[48]);
    // Expected: [[1,4],[2,5],[3,6]]
    let result = read_f64s(&out[0]);
    assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_3d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x4xf64>) -> tensor<3x2x4xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<2x3x4xf64>) -> tensor<3x2x4xf64>
    return %0 : tensor<3x2x4xf64>
  }
}
"#;
    // Input: 2x3x4 = 24 elements, row-major: group0=[0..12), group1=[12..24)
    let mut input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let in0 = f64_buf(&input);
    let out = run_mlir(mlir, &[&in0], &[192]);
    let result = read_f64s(&out[0]);
    // dims=[1,0,2]: output[j][i][k] = input[i][j][k]
    // output shape 3x2x4
    for j in 0..3 {
        for i in 0..2 {
            for k in 0..4 {
                let expected = (i * 3 * 4 + j * 4 + k) as f64;
                let got = result[j * 2 * 4 + i * 4 + k];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "transpose[{j}][{i}][{k}]: got {got}, expected {expected}"
                );
            }
        }
    }
}

// ---- Dynamic slice ----

#[test]
fn test_dynamic_slice_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<i64>) -> tensor<2xf64> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [2] : (tensor<5xf64>, tensor<i64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let idx = i64_buf(&[2]);
    let out = run_mlir(mlir, &[&in0, &idx], &[16]);
    assert_eq!(read_f64s(&out[0]), &[30.0, 40.0]);
}

#[test]
fn test_dynamic_slice_3d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3x4xf64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<1x3x4xf64> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [1, 3, 4] : (tensor<2x3x4xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x4xf64>
    return %0 : tensor<1x3x4xf64>
  }
}
"#;
    let input: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let in0 = f64_buf(&input);
    let idx0 = i64_buf(&[1]);
    let idx1 = i64_buf(&[0]);
    let idx2 = i64_buf(&[0]);
    let out = run_mlir(mlir, &[&in0, &idx0, &idx1, &idx2], &[96]);
    let result = read_f64s(&out[0]);
    // Slice starting at [1,0,0] with sizes [1,3,4] = elements 12..24
    let expected: Vec<f64> = (12..24).map(|i| i as f64).collect();
    assert_eq!(result, expected);
}

// ---- Dynamic update slice ----

#[test]
fn test_dynamic_update_slice_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<2xf64>, %arg2: tensor<i64>) -> tensor<5xf64> {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<5xf64>, tensor<2xf64>, tensor<i64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let base = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let update = f64_buf(&[99.0, 100.0]);
    let idx = i64_buf(&[1]);
    let out = run_mlir(mlir, &[&base, &update, &idx], &[40]);
    assert_eq!(read_f64s(&out[0]), &[1.0, 99.0, 100.0, 4.0, 5.0]);
}

// ---- Gather (row-select) ----

#[test]
fn test_gather_row_select() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x4xf64>, %arg1: tensor<2x1xui32>) -> tensor<2x4xf64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<3x4xf64>, tensor<2x1xui32>) -> tensor<2x4xf64>
    return %0 : tensor<2x4xf64>
  }
}
"#;
    // operand: [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    let operand = f64_buf(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    // indices: [[2],[0]] -- pick row 2 then row 0
    let indices: Vec<u8> = [2u32, 0u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = run_mlir(mlir, &[&operand, &indices], &[64]);
    let result = read_f64s(&out[0]);
    // Expected: row 2 = [9,10,11,12], row 0 = [1,2,3,4]
    assert_eq!(result, &[9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]);
}

// ---- Batched dot product ----

#[test]
fn test_dot_general_batched() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x4xf64>, %arg1: tensor<3x4xf64>) -> tensor<3xf64> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    // batch 0: [1,2,3,4]·[1,0,0,0] = 1
    // batch 1: [5,6,7,8]·[0,1,0,0] = 6
    // batch 2: [9,10,11,12]·[0,0,1,0] = 11
    let a = f64_buf(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    let b = f64_buf(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let out = run_mlir(mlir, &[&a, &b], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 6.0, 11.0]);
}
