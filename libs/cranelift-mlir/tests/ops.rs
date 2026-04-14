use cranelift_mlir::lower::{CompileConfig, compile_module, compile_module_with_config};
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

fn run_mlir_mem(mlir: &str, inputs: &[&[u8]], output_sizes: &[usize]) -> Vec<Vec<u8>> {
    let module = parse_module(mlir).expect("parse failed");
    let config = CompileConfig {
        force_pointer_abi_main: true,
    };
    let compiled =
        compile_module_with_config(&module, config).expect(&format!("compile failed (mem path)"));
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

// ---- LAPACK ops ----

#[test]
fn test_lapack_dgetrf_2x2() {
    // A = [[4, 3], [6, 3]] -> LU with partial pivoting
    // LAPACK should swap rows: pivot = [2, 2]
    // L = [[1, 0], [2/3, 1]], U = [[6, 3], [0, 1]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xi32>, tensor<i32>) {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<2x2xf64>, tensor<2xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<2x2xf64>, tensor<2xi32>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[4.0, 3.0, 6.0, 3.0]);
    let out = run_mlir(mlir, &[&a], &[32, 8, 4]);
    let lu = read_f64s(&out[0]);
    let pivots: Vec<i32> = out[1]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let info: i32 = i32::from_le_bytes(out[2][..4].try_into().unwrap());
    eprintln!("LU: {:?}", lu);
    eprintln!("pivots: {:?}", pivots);
    eprintln!("info: {}", info);
    assert_eq!(info, 0);
    // After pivot: row 0 <-> row 1 (pivot[0] = 2 in 1-indexed)
    assert_eq!(pivots[0], 2);
    // LU packed: [[6, 3], [2/3, 1]]
    assert_f64_close(lu[0], 6.0);
    assert_f64_close(lu[1], 3.0);
    assert_f64_close(lu[2], 2.0 / 3.0);
    assert_f64_close(lu[3], 1.0);
}

#[test]
fn test_lapack_dtrsm_lower_unit() {
    // L = [[1, 0], [2, 1]] (unit lower triangular), b = [[5], [8]]
    // Solve L*x = b -> x = [[5], [-2]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> tensor<2x1xf64> {
    %0 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %arg1) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}} -> tensor<2x1xf64>
    return %0 : tensor<2x1xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 0.0, 2.0, 1.0]);
    let b = f64_buf(&[5.0, 8.0]);
    let out = run_mlir(mlir, &[&a, &b], &[16]);
    let result = read_f64s(&out[0]);
    eprintln!("trsm result: {:?}", result);
    assert_f64s_close(&result, &[5.0, -2.0]);
}

#[test]
fn test_lapack_dtrsm_upper() {
    // U = [[3, 1], [0, 2]] (upper triangular), b = [[5], [4]]
    // Solve U*x = b -> x = [[1], [2]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> tensor<2x1xf64> {
    %0 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %arg1) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}} -> tensor<2x1xf64>
    return %0 : tensor<2x1xf64>
  }
}
"#;
    let a = f64_buf(&[3.0, 1.0, 0.0, 2.0]);
    let b = f64_buf(&[5.0, 4.0]);
    let out = run_mlir(mlir, &[&a, &b], &[16]);
    let result = read_f64s(&out[0]);
    eprintln!("trsm result: {:?}", result);
    assert_f64s_close(&result, &[1.0, 2.0]);
}

#[test]
fn test_lapack_svd_2x2() {
    // A = [[3, 0], [0, 2]]  -> SVD: U=I, S=[3,2], V=I
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xf64>, tensor<2x2xf64>, tensor<2x2xf64>, tensor<i32>) {
    %0:5 = stablehlo.custom_call @lapack_dgesdd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 65 : ui8}} -> (tensor<2x2xf64>, tensor<2xf64>, tensor<2x2xf64>, tensor<2x2xf64>, tensor<i32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<2x2xf64>, tensor<2xf64>, tensor<2x2xf64>, tensor<2x2xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[3.0, 0.0, 0.0, 2.0]);
    let out = run_mlir(mlir, &[&a], &[32, 16, 32, 32, 4]);
    let u = read_f64s(&out[0]);
    let s = read_f64s(&out[1]);
    let vt = read_f64s(&out[2]);
    eprintln!("U: {:?}", u);
    eprintln!("S: {:?}", s);
    eprintln!("VT: {:?}", vt);
    // S should be [3, 2] (descending)
    assert_f64_close(s[0], 3.0);
    assert_f64_close(s[1], 2.0);
    // U * diag(S) * VT should reconstruct A
    let mut reconstructed = [0.0f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                reconstructed[i * 2 + j] += u[i * 2 + k] * s[k] * vt[k * 2 + j];
            }
        }
    }
    eprintln!("Reconstructed: {:?}", reconstructed);
    assert_f64s_close(&reconstructed, &[3.0, 0.0, 0.0, 2.0]);
}

#[test]
fn test_lapack_cholesky_3x3() {
    // A = [[4, 2, 0], [2, 5, 3], [0, 3, 10]] -> L such that L*L^T = A
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<i32>) {
    %0:2 = stablehlo.custom_call @lapack_dpotrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {uplo = 76 : ui8}} -> (tensor<3x3xf64>, tensor<i32>)
    return %0#0, %0#1 : tensor<3x3xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0]);
    let out = run_mlir(mlir, &[&a], &[72, 4]);
    let l = read_f64s(&out[0]);
    let info: i32 = i32::from_le_bytes(out[1][..4].try_into().unwrap());
    eprintln!("L: {:?}", l);
    eprintln!("info: {}", info);
    assert_eq!(info, 0);
    // L should be lower triangular
    assert_f64_close(l[1], 0.0); // L[0,1]
    assert_f64_close(l[2], 0.0); // L[0,2]
    assert_f64_close(l[5], 0.0); // L[1,2]
    // Verify L*L^T = A
    let mut llt = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                llt[i * 3 + j] += l[i * 3 + k] * l[j * 3 + k];
            }
        }
    }
    eprintln!("L*L^T: {:?}", llt);
    assert_f64s_close(&llt, &[4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 10.0]);
}

#[test]
fn test_lapack_svd_3x3_nontrivial() {
    // A = [[1, 2, 0], [0, 3, 1], [2, 0, 4]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<i32>) {
    %0:5 = stablehlo.custom_call @lapack_dgesdd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 65 : ui8}} -> (tensor<3x3xf64>, tensor<3xf64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<i32>)
    return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<3x3xf64>, tensor<3xf64>, tensor<3x3xf64>, tensor<3x3xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0]);
    let out = run_mlir(mlir, &[&a], &[72, 24, 72, 72, 4]);
    let u = read_f64s(&out[0]);
    let s = read_f64s(&out[1]);
    let vt = read_f64s(&out[2]);
    eprintln!("U: {:?}", u);
    eprintln!("S: {:?}", s);
    eprintln!("VT: {:?}", vt);
    // Verify U * diag(S) * VT reconstructs A
    let mut reconstructed = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                reconstructed[i * 3 + j] += u[i * 3 + k] * s[k] * vt[k * 3 + j];
            }
        }
    }
    eprintln!("Reconstructed: {:?}", reconstructed);
    let expected = [1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0];
    for (i, (&a, &e)) in reconstructed.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < 1e-10,
            "element {i}: expected {e}, got {a} (abs_diff: {diff:.2e})"
        );
    }
}

#[test]
fn test_lapack_syevd_2x2() {
    // A = [[2, 1], [1, 3]] (symmetric) -> eigenvalues ~[1.382, 3.618]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xf64>, tensor<i32>) {
    %0:3 = stablehlo.custom_call @lapack_dsyevd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}} -> (tensor<2x2xf64>, tensor<2xf64>, tensor<i32>)
    return %0#0, %0#1, %0#2 : tensor<2x2xf64>, tensor<2xf64>, tensor<i32>
  }
}
"#;
    let a = f64_buf(&[2.0, 1.0, 1.0, 3.0]);
    let out = run_mlir(mlir, &[&a], &[32, 16, 4]);
    let eigvecs = read_f64s(&out[0]);
    let eigvals = read_f64s(&out[1]);
    let info: i32 = i32::from_le_bytes(out[2][..4].try_into().unwrap());
    eprintln!("eigvals: {:?}", eigvals);
    eprintln!("eigvecs: {:?}", eigvecs);
    eprintln!("info: {}", info);
    assert_eq!(info, 0);
    // Eigenvalues of [[2,1],[1,3]] are (5-sqrt(5))/2 and (5+sqrt(5))/2
    let sqrt5 = 5.0f64.sqrt();
    assert_f64_close(eigvals[0], (5.0 - sqrt5) / 2.0);
    assert_f64_close(eigvals[1], (5.0 + sqrt5) / 2.0);
    // Verify V * diag(lambda) * V^T = A
    let mut reconstructed = [0.0f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                reconstructed[i * 2 + j] += eigvecs[i * 2 + k] * eigvals[k] * eigvecs[j * 2 + k];
            }
        }
    }
    eprintln!("Reconstructed: {:?}", reconstructed);
    assert_f64s_close(&reconstructed, &[2.0, 1.0, 1.0, 3.0]);
}

#[test]
fn test_lapack_qr_3x3() {
    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>) {
    %0:2 = stablehlo.custom_call @lapack_dgeqrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<3x3xf64>, tensor<3xf64>)
    return %0#0, %0#1 : tensor<3x3xf64>, tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let out = run_mlir(mlir, &[&a], &[72, 24]);
    let qr_packed = read_f64s(&out[0]);
    let tau = read_f64s(&out[1]);
    eprintln!("QR packed: {:?}", qr_packed);
    eprintln!("tau: {:?}", tau);
    // Just verify the packed form and tau are populated (non-zero)
    assert!(tau[0].abs() > 1e-15, "tau[0] should be non-zero");
}

#[test]
fn test_lapack_qr_orgqr_roundtrip_3x3() {
    // Full QR roundtrip: A -> (QR, tau) -> (Q, R) -> Q*R should equal A
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0:2 = stablehlo.custom_call @lapack_dgeqrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<3x3xf64>, tensor<3xf64>)
    %1 = stablehlo.custom_call @lapack_dorgqr_ffi(%0#0, %0#1) {backend_config = "", mhlo.backend_config = {}} -> tensor<3x3xf64>
    return %1 : tensor<3x3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let out = run_mlir(mlir, &[&a], &[72]);
    let q = read_f64s(&out[0]);
    eprintln!("Q: {:?}", q);
    // Q should be orthogonal: Q^T * Q = I
    let mut qtq = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                qtq[i * 3 + j] += q[k * 3 + i] * q[k * 3 + j];
            }
        }
    }
    eprintln!("Q^T*Q: {:?}", qtq);
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for (i, (&a, &e)) in qtq.iter().zip(identity.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < 1e-10,
            "Q^T*Q[{}]: expected {e}, got {a} (abs_diff: {diff:.2e})",
            i
        );
    }
}

#[test]
fn test_solve_3x3_vector_rhs() {
    // Direct test: solve A*x = b where A = [[1.01, 0.00833, 0], [0, 1.01, 0.00833], [0, 0, 1.01]]
    // b = [1, 3, 5] -> x ~= [0.966, 2.929, 4.950]
    // Use the exact MLIR pattern from the linalg-iree solve path
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}} -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<3xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare GE, %0#2, %3, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %8 = stablehlo.select %7, %0#0, %6 : tensor<3x3xi1>, tensor<3x3xf64>
    %9 = stablehlo.iota dim = 0 : tensor<3xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %10:4 = stablehlo.while(%iterArg = %2, %iterArg_3 = %c_2, %iterArg_4 = %c_1, %iterArg_5 = %9) : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    cond {
      %c_6 = stablehlo.constant dense<3> : tensor<i64>
      %14 = stablehlo.compare LT, %iterArg_3, %c_6, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14:2 = func.call @closed_call(%iterArg, %iterArg_4, %iterArg_5) : (tensor<3xi32>, tensor<i64>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %15 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %15, %14#0, %14#1 : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    }
    %11 = call @_lu_solve(%8, %10#3, %arg1) : (tensor<3x3xf64>, tensor<3xi32>, tensor<3xf64>) -> tensor<3xf64>
    return %11 : tensor<3xf64>
  }
  func.func private @closed_call(%arg0: tensor<3xi32>, %arg1: tensor<i64>, %arg2: tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg1, %c : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare LT, %arg1, %c_0, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = stablehlo.convert %arg1 : tensor<i64>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %3 = stablehlo.add %2, %c_1 : tensor<i64>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<i1>, tensor<i64>
    %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<3xi32>, tensor<i64>) -> tensor<1xi32>
    %6 = stablehlo.reshape %5 : (tensor<1xi32>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.compare LT, %arg1, %c_2, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8 = stablehlo.convert %arg1 : tensor<i64>
    %c_3 = stablehlo.constant dense<3> : tensor<i64>
    %9 = stablehlo.add %8, %c_3 : tensor<i64>
    %10 = stablehlo.select %7, %9, %arg1 : tensor<i1>, tensor<i64>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<3xi32>, tensor<i64>) -> tensor<1xi32>
    %12 = stablehlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.compare LT, %6, %c_4, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<3> : tensor<i32>
    %14 = stablehlo.add %6, %c_5 : tensor<i32>
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32>
    %16 = stablehlo.dynamic_slice %arg2, %15, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %18 = stablehlo.compare LT, %arg1, %c_6, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_7 = stablehlo.constant dense<3> : tensor<i64>
    %19 = stablehlo.add %arg1, %c_7 : tensor<i64>
    %20 = stablehlo.select %18, %19, %arg1 : tensor<i1>, tensor<i64>
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = "stablehlo.scatter"(%arg2, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %24 = stablehlo.compare LT, %6, %c_8, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_9 = stablehlo.constant dense<3> : tensor<i32>
    %25 = stablehlo.add %6, %c_9 : tensor<i32>
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>
    return %0, %28 : tensor<i64>, tensor<3xi32>
  }
  func.func private @_lu_solve(%arg0: tensor<3x3xf64>, %arg1: tensor<3xi32>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.compare LT, %arg1, %1, SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<3xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<3xi1>, tensor<3xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x1xf64>, tensor<3x1xi32>) -> tensor<3x1xf64>
    %8 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %7) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}} -> tensor<3x1xf64>
    %9 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %8) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}} -> tensor<3x1xf64>
    %10 = stablehlo.slice %9 [0:3, 0:1] : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<3x1xf64>) -> tensor<3xf64>
    return %11 : tensor<3xf64>
  }
}
"#;
    let a = f64_buf(&[
        1.01,
        1.0 / 120.0,
        0.0,
        0.0,
        1.01,
        1.0 / 120.0,
        0.0,
        0.0,
        1.01,
    ]);
    let b = f64_buf(&[1.0, 3.0, 5.0]);
    let out = run_mlir(mlir, &[&a, &b], &[24]);
    let result = read_f64s(&out[0]);
    eprintln!("solve result: {:?}", result);
    // Expected: [0.966, 2.929, 4.950] approximately
    assert_f64_close(result[2], 5.0 / 1.01);
}

#[test]
#[ignore = "known issue: 3D gather/broadcast path produces wrong results for matrix-RHS solve"]
fn test_linalg_iree_one_tick() {
    let mlir = include_str!("../testdata/linalg-iree.stablehlo.mlir");
    let module = cranelift_mlir::parser::parse_module(mlir).expect("parse failed");
    let compiled = cranelift_mlir::lower::compile_module(&module).expect("compile failed");
    let fn_ptr = compiled.get_main_fn();
    let tick_fn: TickFn = unsafe { std::mem::transmute(fn_ptr) };

    // Inputs match the world() spawn in sim.py:
    // arg0: tick (i64) = 0
    let arg0 = i64_buf(&[0]);
    // arg1: mrhs_state (3x2 f64) = [[1,2],[3,4],[5,6]]
    let arg1 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // arg2: sm2_state (2 f64) = [1.0, 0.5]
    let arg2 = f64_buf(&[1.0, 0.5]);
    // arg3: sm2_cov (2x2 f64) = eye(2)*5 = [[5,0],[0,5]]
    let arg3 = f64_buf(&[5.0, 0.0, 0.0, 5.0]);
    // arg4: kf3_state (3 f64) = [0, 1, 0]
    let arg4 = f64_buf(&[0.0, 1.0, 0.0]);
    // arg5: kf3_cov (3x3 f64) = eye(3)*10
    let arg5 = f64_buf(&[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]);
    // arg6: kf3_info (5 f64) = zeros
    let arg6 = f64_buf(&[0.0, 0.0, 0.0, 0.0, 0.0]);
    // arg7: ekf6_state (6 f64) = [0,0,100,10,0,-5]
    let arg7 = f64_buf(&[0.0, 0.0, 100.0, 10.0, 0.0, -5.0]);
    // arg8: ekf6_cov (6x6 f64) = eye(6)*100
    let mut arg8_data = vec![0.0f64; 36];
    for i in 0..6 {
        arg8_data[i * 6 + i] = 100.0;
    }
    let arg8 = f64_buf(&arg8_data);
    // arg9: ekf6_info (4 f64) = zeros
    let arg9 = f64_buf(&[0.0, 0.0, 0.0, 0.0]);
    // arg10: mode_state (4 i64) = [0,0,0,0]
    let arg10 = i64_buf(&[0, 0, 0, 0]);

    let inputs: Vec<&[u8]> = vec![
        &arg0, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8, &arg9, &arg10,
    ];
    // Outputs (from return type):
    // result[0]: 4xf64 (ekf6_info)
    // result[1]: 3x3xf64 (kf3_cov)
    // result[2]: 2xf64 (sm2_state)
    // result[3]: 3x2xf64 (mrhs_state)
    // result[4]: i64 (tick)
    // result[5]: 2x2xf64 (sm2_cov)
    // result[6]: 4xi64 (mode_state)
    // result[7]: 3xf64 (kf3_state)
    // result[8]: 6xf64 (ekf6_state)
    // result[9]: 5xf64 (kf3_info)
    // result[10]: 6x6xf64 (ekf6_cov)
    let output_sizes = vec![32, 72, 16, 48, 8, 32, 32, 24, 48, 40, 288];
    let out = run_mlir(mlir, &inputs, &output_sizes);

    // Check result[3]: mrhs_state after one tick
    let mrhs = read_f64s(&out[3]);
    eprintln!("mrhs_state: {:?}", mrhs);
    // Expected: solve(F3 + 0.01*I, [[1,2],[3,4],[5,6]])
    // F3 + 0.01*I is upper triangular with diag ~1.01
    assert_f64_close(mrhs[0], 0.9659286191338475);
    assert_f64_close(mrhs[4], 4.9504950495049505);
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
fn test_add_mem() {
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
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[24]);
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

// ---- New elementwise / transcendental ops ----

#[test]
fn test_sine() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.sine %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 1.0);
    assert!(result[2].abs() < 1e-15);
}

#[test]
fn test_cosine() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.cosine %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert!(result[1].abs() < 1e-15);
    assert_f64_close(result[2], -1.0);
}

#[test]
fn test_atan2() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.atan2 %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let y = f64_buf(&[0.0, 1.0, -1.0]);
    let x = f64_buf(&[1.0, 0.0, 0.0]);
    let out = run_mlir(mlir, &[&y, &x], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], std::f64::consts::FRAC_PI_2);
    assert_f64_close(result[2], -std::f64::consts::FRAC_PI_2);
}

#[test]
fn test_abs_float() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.abs %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-3.0, 0.0, 5.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[3.0, 0.0, 5.0]);
}

#[test]
fn test_minimum() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 5.0, 3.0]);
    let in1 = f64_buf(&[4.0, 2.0, 3.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_sign() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.sign %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-3.5, 0.0, 7.2]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[-1.0, 0.0, 1.0]);
}

#[test]
fn test_remainder() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.remainder %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[7.0, 10.0, -5.5]);
    let in1 = f64_buf(&[3.0, 3.0, 2.0]);
    let out = run_mlir(mlir, &[&in0, &in1], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert_f64_close(result[1], 1.0);
    assert_f64_close(result[2], -1.5);
}

#[test]
fn test_acos() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = chlo.acos %arg0 : tensor<f64> -> tensor<f64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 0.0, -1.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], std::f64::consts::FRAC_PI_2);
    assert_f64_close(result[2], std::f64::consts::PI);
}

#[test]
fn test_exponential() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.exponential %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, 1.0, -1.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 1.0);
    assert_f64_close(result[1], std::f64::consts::E);
    assert_f64_close(result[2], 1.0 / std::f64::consts::E);
}

#[test]
fn test_log() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.log %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, std::f64::consts::E, 10.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 1.0);
    assert_f64_close(result[2], 10.0_f64.ln());
}

#[test]
fn test_clamp() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %lo = stablehlo.constant dense<-1.0> : tensor<f64>
    %hi = stablehlo.constant dense<1.0> : tensor<f64>
    %0 = stablehlo.clamp %lo, %arg0, %hi : (tensor<f64>, tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[-5.0, 0.5, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[-1.0, 0.5, 1.0]);
}

#[test]
fn test_power() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.power %arg0, %arg1 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let base = f64_buf(&[2.0, 3.0, 10.0]);
    let exp = f64_buf(&[3.0, 2.0, 0.5]);
    let out = run_mlir(mlir, &[&base, &exp], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 8.0);
    assert_f64_close(result[1], 9.0);
    assert_f64_close(result[2], 10.0_f64.sqrt());
}

#[test]
fn test_reverse() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<4xf64>
    return %0 : tensor<4xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let out = run_mlir(mlir, &[&in0], &[32]);
    assert_f64s_close(&read_f64s(&out[0]), &[4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_tanh() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.tanh %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[0.0, 1.0, -1.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert_f64_close(result[0], 0.0);
    assert_f64_close(result[1], 1.0_f64.tanh());
    assert_f64_close(result[2], (-1.0_f64).tanh());
}

#[test]
fn test_ssa_shadow_redefine() {
    let mlir = r#"
module @module {
  func.func private @use_pair(%arg0: tensor<2xui32>) -> tensor<2xui32> {
    return %arg0 : tensor<2xui32>
  }
  func.func public @main(%arg0: tensor<i64>) -> (tensor<2xui32>, tensor<i64>) {
    %c = stablehlo.constant dense<[42, 99]> : tensor<2xui32>
    %0 = call @use_pair(%c) : (tensor<2xui32>) -> tensor<2xui32>
    %c = stablehlo.constant dense<7> : tensor<i64>
    %1 = stablehlo.add %arg0, %c : tensor<i64>
    return %0, %1 : tensor<2xui32>, tensor<i64>
  }
}
"#;
    let input = 10i64.to_le_bytes().to_vec();
    let out = run_mlir(mlir, &[&input], &[8, 8]);
    let seed: Vec<u32> = out[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let sum = i64::from_le_bytes(out[1].as_slice().try_into().unwrap());
    assert_eq!(
        seed,
        vec![42, 99],
        "First %c definition corrupted by redefinition"
    );
    assert_eq!(sum, 17, "Second %c definition incorrect");
}

#[test]
fn test_gather_i32_indices_1d() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<3x1xi32>) -> tensor<3xf64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<5xf64>, tensor<3x1xi32>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let data = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let indices = i32_buf(&[0, 2, 4]);
    let out = run_mlir(mlir, &[&data, &indices], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 30.0, 50.0]);
}

#[test]
fn test_tan() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.tan %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[
        0.0,
        std::f64::consts::FRAC_PI_4,
        -std::f64::consts::FRAC_PI_4,
    ]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    let result = read_f64s(&out[0]);
    assert!(result[0].abs() < 1e-15);
    assert_f64_close(result[1], 1.0);
    assert_f64_close(result[2], -1.0);
}

#[test]
fn test_floor() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.floor %arg0 : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.7, -0.3, 3.0]);
    let out = run_mlir(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, -1.0, 3.0]);
}

#[test]
fn test_many_args_call_and_value_survival() {
    // Tests: 1) 32-arg function call works 2) return values survive past a 32-arg sret call
    let mlir = r#"
module @module {
  func.func private @noise(%arg0: tensor<f64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.add %arg1, %0 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
  func.func private @big_sret(%a0: tensor<f64>, %a1: tensor<f64>, %a2: tensor<f64>,
    %a3: tensor<3xf64>, %a4: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
    %s = stablehlo.add %a3, %a4 : tensor<3xf64>
    %b0 = stablehlo.broadcast_in_dim %a0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r0 = stablehlo.add %s, %b0 : tensor<3xf64>
    %b1 = stablehlo.broadcast_in_dim %a1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r1 = stablehlo.add %s, %b1 : tensor<3xf64>
    %b2 = stablehlo.broadcast_in_dim %a2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %r2 = stablehlo.add %s, %b2 : tensor<3xf64>
    return %r0, %r1, %r2 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }
  func.func public @main(%tick: tensor<f64>, %bias: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
    %noise_out = call @noise(%tick, %bias) : (tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    %c1 = stablehlo.constant dense<1.0> : tensor<f64>
    %c2 = stablehlo.constant dense<2.0> : tensor<f64>
    %c3 = stablehlo.constant dense<3.0> : tensor<f64>
    %sret_out:3 = call @big_sret(%c1, %c2, %c3, %noise_out, %bias) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
    %next_tick = stablehlo.add %tick, %c1 : tensor<f64>
    %noise_out2 = call @noise(%next_tick, %noise_out) : (tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    return %noise_out, %noise_out2 : tensor<3xf64>, tensor<3xf64>
  }
}
"#;
    let tick = f64_buf(&[0.0]);
    let bias = f64_buf(&[10.0, 20.0, 30.0]);
    let out = run_mlir(mlir, &[&tick, &bias], &[24, 24]);
    let r0 = read_f64s(&out[0]); // noise_out = bias + tick = [10, 20, 30]
    let r1 = read_f64s(&out[1]); // noise_out2 = noise_out + (tick+1) = [11, 21, 31]
    assert_f64s_close(&r0, &[10.0, 20.0, 30.0]);
    assert_f64s_close(&r1, &[11.0, 21.0, 31.0]);
}

#[test]
fn test_dynamic_slice_clamps_out_of_bounds() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4x3xf64>, %arg1: tensor<i32>) -> tensor<1x3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c0, sizes = [1, 3] : (tensor<4x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    return %0 : tensor<1x3xf64>
  }
}
"#;
    // 4x3 matrix: row0=[1,2,3], row1=[4,5,6], row2=[7,8,9], row3=[10,11,12]
    let data = f64_buf(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    // Index 10 is way out of bounds (max valid = 3), should clamp to row 3
    let idx = i32_buf(&[10]);
    let out = run_mlir(mlir, &[&data, &idx], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 11.0, 12.0]);

    // Index -5 is negative, should clamp to row 0
    let idx_neg = i32_buf(&[-5]);
    let out_neg = run_mlir(mlir, &[&data, &idx_neg], &[24]);
    assert_f64s_close(&read_f64s(&out_neg[0]), &[1.0, 2.0, 3.0]);
}

// ---- Pointer-ABI (memory-backed) path tests ----

fn mem_binop_test(op: &str, a: &[f64], b: &[f64], expected: &[f64]) {
    let n = a.len();
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>, %arg1: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = stablehlo.{op} %arg0, %arg1 : tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(a);
    let in1 = f64_buf(b);
    let out = run_mlir_mem(&mlir, &[&in0, &in1], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), expected);
}

fn mem_unop_test(op: &str, a: &[f64], expected: &[f64]) {
    let n = a.len();
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<{n}xf64>) -> tensor<{n}xf64> {{
    %0 = stablehlo.{op} %arg0 : tensor<{n}xf64>
    return %0 : tensor<{n}xf64>
  }}
}}"#
    );
    let in0 = f64_buf(a);
    let out = run_mlir_mem(&mlir, &[&in0], &[n * 8]);
    assert_f64s_close(&read_f64s(&out[0]), expected);
}

#[test]
fn test_subtract_mem() {
    mem_binop_test(
        "subtract",
        &[5.0, 7.0, 9.0],
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
    );
}

#[test]
fn test_multiply_mem() {
    mem_binop_test(
        "multiply",
        &[2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0],
        &[10.0, 18.0, 28.0],
    );
}

#[test]
fn test_divide_mem() {
    mem_binop_test(
        "divide",
        &[10.0, 18.0, 28.0],
        &[2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0],
    );
}

#[test]
fn test_maximum_mem() {
    mem_binop_test(
        "maximum",
        &[1.0, 5.0, 3.0],
        &[4.0, 2.0, 6.0],
        &[4.0, 5.0, 6.0],
    );
}

#[test]
fn test_minimum_mem() {
    mem_binop_test(
        "minimum",
        &[1.0, 5.0, 3.0],
        &[4.0, 2.0, 6.0],
        &[1.0, 2.0, 3.0],
    );
}

#[test]
fn test_negate_mem() {
    mem_unop_test("negate", &[1.0, -2.0, 3.0], &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_sqrt_mem() {
    mem_unop_test("sqrt", &[4.0, 9.0, 16.0], &[2.0, 3.0, 4.0]);
}

#[test]
fn test_floor_mem() {
    mem_unop_test("floor", &[1.7, 2.3, -0.5], &[1.0, 2.0, -1.0]);
}

#[test]
fn test_sine_mem() {
    mem_unop_test("sine", &[0.0, std::f64::consts::FRAC_PI_2], &[0.0, 1.0]);
}

#[test]
fn test_cosine_mem() {
    mem_unop_test("cosine", &[0.0, std::f64::consts::PI], &[1.0, -1.0]);
}

#[test]
fn test_exponential_mem() {
    mem_unop_test("exponential", &[0.0, 1.0], &[1.0, std::f64::consts::E]);
}

#[test]
fn test_log_mem() {
    mem_unop_test("log", &[1.0, std::f64::consts::E], &[0.0, 1.0]);
}

#[test]
fn test_tanh_mem() {
    mem_unop_test("tanh", &[0.0], &[0.0]);
}

#[test]
fn test_abs_mem() {
    mem_unop_test("abs", &[-3.0, 0.0, 5.0], &[3.0, 0.0, 5.0]);
}

#[test]
fn test_power_mem() {
    mem_binop_test("power", &[2.0, 3.0], &[3.0, 2.0], &[8.0, 9.0]);
}

#[test]
fn test_reshape_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<6xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<6xf64>) -> tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[48]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_constant_mem() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3xf64> {
    %0 = stablehlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_compare_lt_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xi1> {
    %0 = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 5.0, 3.0]);
    let in1 = f64_buf(&[2.0, 4.0, 3.0]);
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[3]);
    assert_eq!(out[0], vec![1, 0, 0]);
}

#[test]
fn test_select_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi1>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<3xi1>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let cond = vec![1u8, 0, 1];
    let in1 = f64_buf(&[10.0, 20.0, 30.0]);
    let in2 = f64_buf(&[100.0, 200.0, 300.0]);
    let out = run_mlir_mem(mlir, &[&cond, &in1, &in2], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[10.0, 200.0, 30.0]);
}

#[test]
fn test_convert_i64_to_f64_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi64>) -> tensor<3xf64> {
    %0 = stablehlo.convert %arg0 : (tensor<3xi64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = i64_buf(&[1, 2, 3]);
    let out = run_mlir_mem(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_iota_mem() {
    let mlir = r#"
module @module {
  func.func public @main() -> tensor<3x2xi64> {
    %0 = stablehlo.iota dim = 1 : tensor<3x2xi64>
    return %0 : tensor<3x2xi64>
  }
}
"#;
    let out = run_mlir_mem(mlir, &[], &[48]);
    assert_eq!(read_i64s(&out[0]), vec![0, 1, 0, 1, 0, 1]);
}

#[test]
fn test_broadcast_in_dim_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf64>) -> tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[48]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_transpose_2d_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[48]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_slice_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg0 [1:4] : (tensor<5xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[20.0, 30.0, 40.0]);
}

#[test]
fn test_concatenate_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2xf64>, %arg1: tensor<3xf64>) -> tensor<5xf64> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2xf64>, tensor<3xf64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0]);
    let in1 = f64_buf(&[3.0, 4.0, 5.0]);
    let out = run_mlir_mem(mlir, &[&in0, &in1], &[40]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_dynamic_slice_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<i64>) -> tensor<3xf64> {
    %0 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [3] : (tensor<5xf64>, tensor<i64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let data = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let idx = i64_buf(&[1]);
    let out = run_mlir_mem(mlir, &[&data, &idx], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[20.0, 30.0, 40.0]);
}

#[test]
fn test_dynamic_update_slice_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<2xf64>, %arg2: tensor<i64>) -> tensor<5xf64> {
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2 : (tensor<5xf64>, tensor<2xf64>, tensor<i64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let data = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let upd = f64_buf(&[90.0, 91.0]);
    let idx = i64_buf(&[2]);
    let out = run_mlir_mem(mlir, &[&data, &upd, &idx], &[40]);
    assert_f64s_close(&read_f64s(&out[0]), &[1.0, 2.0, 90.0, 91.0, 5.0]);
}

#[test]
fn test_dot_general_matmul_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0] :
      (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<2x2xf64>
    return %0 : tensor<2x2xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_buf(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let out = run_mlir_mem(mlir, &[&a, &b], &[32]);
    assert_f64s_close(&read_f64s(&out[0]), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_reduce_sum_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<6xf64>) -> tensor<2xf64> {
    %0 = stablehlo.reshape %arg0 : (tensor<6xf64>) -> tensor<2x3xf64>
    %init = stablehlo.constant dense<0.0> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
    return %1 : tensor<2xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[16]);
    assert_f64s_close(&read_f64s(&out[0]), &[6.0, 15.0]);
}

#[test]
fn test_while_mem_count1() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<f64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %c0) : tensor<f64>, tensor<i64>
      cond {
        %limit = stablehlo.constant dense<1> : tensor<i64>
        %cmp = stablehlo.compare LT, %iterArg_0, %limit, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
      } do {
        %inc = stablehlo.constant dense<10.0> : tensor<f64>
        %one = stablehlo.constant dense<1> : tensor<i64>
        %new_val = stablehlo.add %iterArg, %inc : tensor<f64>
        %new_idx = stablehlo.add %iterArg_0, %one : tensor<i64>
        stablehlo.return %new_val, %new_idx : tensor<f64>, tensor<i64>
      }
    return %0#0 : tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[5.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[8]);
    assert_f64s_close(&read_f64s(&out[0]), &[15.0]);
}

#[test]
fn test_while_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %c0) : tensor<3xf64>, tensor<i64>
      cond {
        %limit = stablehlo.constant dense<3> : tensor<i64>
        %cmp = stablehlo.compare LT, %iterArg_0, %limit, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cmp : tensor<i1>
      } do {
        %inc = stablehlo.constant dense<[1.0, 1.0, 1.0]> : tensor<3xf64>
        %one = stablehlo.constant dense<1> : tensor<i64>
        %new_val = stablehlo.add %iterArg, %inc : tensor<3xf64>
        %new_idx = stablehlo.add %iterArg_0, %one : tensor<i64>
        stablehlo.return %new_val, %new_idx : tensor<3xf64>, tensor<i64>
      }
    return %0#0 : tensor<3xf64>
  }
}
"#;
    let in0 = f64_buf(&[10.0, 20.0, 30.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[24]);
    assert_f64s_close(&read_f64s(&out[0]), &[13.0, 23.0, 33.0]);
}

#[test]
fn test_broadcast_i32_1d_to_2d_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xi32>) -> tensor<3x1xi32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    return %0 : tensor<3x1xi32>
  }
}
"#;
    let in0 = i32_buf(&[10, 20, 30]);
    let out = run_mlir_mem(mlir, &[&in0], &[12]);
    let result: Vec<i32> = out[0]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, vec![10, 20, 30]);
}

#[test]
fn test_gather_with_i32_indices_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>, %arg1: tensor<3xi32>) -> tensor<3xf64> {
    %idx = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %0 = "stablehlo.gather"(%arg0, %idx) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<5xf64>, tensor<3x1xi32>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
    let data = f64_buf(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let indices = i32_buf(&[0, 2, 4]);
    let out = run_mlir_mem(mlir, &[&data, &indices], &[24]);
    let result = read_f64s(&out[0]);
    assert!(
        !result.iter().any(|v| v.is_nan()),
        "gather with i32 indices produced NaN: {result:?}"
    );
    assert_f64s_close(&result, &[10.0, 30.0, 50.0]);
}

// ---- Drone @inner function regression test ----

#[test]
fn test_drone_inner_mem() {
    // This is the exact @inner function from the drone MLIR, inlined into @main.
    // It builds a 22x3 lookup table from 4 constant matrices, then uses
    // dynamic_slice with i32 indices to select a row.
    // With index=2, scale=1.0: row 2 of table = [-0.3, 0.4, 0.0] (from cst_2)
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, -0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_0 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<5x3xf64>
    %cst_1 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_2 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.2, 0.4, 0.0], [-0.3, 0.4, 0.0], [0.1, 0.1, 0.0], [0.3, -0.4, 0.0]]> : tensor<5x3xf64>
    %0 = stablehlo.concatenate %cst_2, %cst, %cst_0, %cst_1, dim = 0 : (tensor<5x3xf64>, tensor<6x3xf64>, tensor<5x3xf64>, tensor<6x3xf64>) -> tensor<22x3xf64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    %2 = stablehlo.multiply %1, %arg1 : tensor<f64>
    %3 = stablehlo.convert %2 : (tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare LT, %3, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<22> : tensor<i32>
    %5 = stablehlo.add %3, %c_3 : tensor<i32>
    %6 = stablehlo.select %4, %5, %3 : tensor<i1>, tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.dynamic_slice %0, %6, %c_4, sizes = [1, 3] : (tensor<22x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    %8 = stablehlo.reshape %7 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %8 : tensor<3xf64>
  }
}
"#;
    // index=2, scale=1.0 -> row 2 of concatenated table (cst_2 row 2) = [-0.3, 0.4, 0.0]
    let idx = i64_buf(&[2]);
    let scale = f64_buf(&[1.0]);
    let out = run_mlir_mem(mlir, &[&idx, &scale], &[24]);
    let result = read_f64s(&out[0]);
    assert!(
        !result.iter().any(|v| v.is_nan()),
        "drone @inner produced NaN: {result:?}"
    );
    assert_f64s_close(&result, &[-0.3, 0.4, 0.0]);
}

#[test]
fn test_drone_inner_cross_abi() {
    // Test the cross-ABI boundary: @main (scalar) calls @inner (pointer ABI)
    // @inner has tensor<22x3xf64> (66 elements > 64 threshold) so it's pointer ABI
    // @main passes scalar args, @inner returns tensor<3xf64>
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<3xf64> {
    %0 = call @inner(%arg0, %arg1) : (tensor<i64>, tensor<f64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
  func.func private @inner(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, -0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_0 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]> : tensor<5x3xf64>
    %cst_1 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, -0.2], [0.0, 0.0, -0.2], [0.0, 0.0, 0.0]]> : tensor<6x3xf64>
    %cst_2 = stablehlo.constant dense<[[0.0, 0.0, 0.0], [0.2, 0.4, 0.0], [-0.3, 0.4, 0.0], [0.1, 0.1, 0.0], [0.3, -0.4, 0.0]]> : tensor<5x3xf64>
    %0 = stablehlo.concatenate %cst_2, %cst, %cst_0, %cst_1, dim = 0 : (tensor<5x3xf64>, tensor<6x3xf64>, tensor<5x3xf64>, tensor<6x3xf64>) -> tensor<22x3xf64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    %2 = stablehlo.multiply %1, %arg1 : tensor<f64>
    %3 = stablehlo.convert %2 : (tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare LT, %3, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<22> : tensor<i32>
    %5 = stablehlo.add %3, %c_3 : tensor<i32>
    %6 = stablehlo.select %4, %5, %3 : tensor<i1>, tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.dynamic_slice %0, %6, %c_4, sizes = [1, 3] : (tensor<22x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    %8 = stablehlo.reshape %7 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %8 : tensor<3xf64>
  }
}
"#;
    // index=2, scale=1.0 -> row 2 = [-0.3, 0.4, 0.0]
    let idx = i64_buf(&[2]);
    let scale = f64_buf(&[1.0]);
    let out = run_mlir(mlir, &[&idx, &scale], &[24]);
    let result = read_f64s(&out[0]);
    assert!(
        !result.iter().any(|v| v.is_nan()),
        "cross-ABI @inner produced NaN: {result:?}"
    );
    assert_f64s_close(&result, &[-0.3, 0.4, 0.0]);
}

#[test]
fn test_divide_ui32_mem() {
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<4xui32>, %arg1: tensor<4xui32>) -> tensor<4xui32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xui32>
    %c = stablehlo.constant dense<2> : tensor<ui32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %2 = stablehlo.divide %0, %1 : tensor<4xui32>
    return %2 : tensor<4xui32>
  }
}
"#;
    let lo: Vec<u8> = [0u32, 10, 50, 100]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let hi: Vec<u8> = [120u32, 120, 120, 120]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = run_mlir_mem(mlir, &[&lo, &hi], &[16]);
    let result: Vec<u32> = out[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, vec![60, 65, 85, 110]);
}

#[test]
fn test_scatter_i32_index_mem() {
    // Scatter with i32 index: set element at position 2 of a 5-element vector to 99.0
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %idx = stablehlo.constant dense<[2]> : tensor<1xi32>
    %upd = stablehlo.constant dense<99.0> : tensor<f64>
    %0 = "stablehlo.scatter"(%arg0, %idx, %upd) <{
      indices_are_sorted = true,
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>,
      unique_indices = true
    }> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      stablehlo.return %b : tensor<f64>
    }) : (tensor<5xf64>, tensor<1xi32>, tensor<f64>) -> tensor<5xf64>
    return %0 : tensor<5xf64>
  }
}
"#;
    let in0 = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = run_mlir_mem(mlir, &[&in0], &[40]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 2.0, 99.0, 4.0, 5.0]);
}

#[test]
fn test_cross_abi_multi_result() {
    // Test pointer-ABI function returning multiple results via cross-ABI call
    // The callee has a large constant (>64 elements) forcing pointer ABI
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3xf64>) -> (tensor<3xf64>, tensor<f64>) {
    %0:2 = call @big_func(%arg0) : (tensor<3xf64>) -> (tensor<3xf64>, tensor<f64>)
    return %0#0, %0#1 : tensor<3xf64>, tensor<f64>
  }
  func.func private @big_func(%arg0: tensor<3xf64>) -> (tensor<3xf64>, tensor<f64>) {
    %big = stablehlo.constant dense<0.0> : tensor<100xf64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.multiply %arg0, %arg0 : tensor<3xf64>
    return %2, %1 : tensor<3xf64>, tensor<f64>
  }
}
"#;
    let in0 = f64_buf(&[2.0, 3.0, 4.0]);
    let out = run_mlir(mlir, &[&in0], &[24, 8]);
    let r0 = read_f64s(&out[0]);
    let r1 = read_f64s(&out[1]);
    assert_f64s_close(&r0, &[4.0, 9.0, 16.0]);
    assert_f64s_close(&r1, &[2.0]);
}

#[test]
fn test_concatenate_dim1_mem() {
    // Concatenate along dim 1: [[1,2],[3,4]] ++ [[5],[6]] -> [[1,2,5],[3,4,6]]
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x2xf64>, %arg1: tensor<2x1xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
"#;
    let a = f64_buf(&[1.0, 2.0, 3.0, 4.0]);
    let b = f64_buf(&[5.0, 6.0]);
    let out = run_mlir_mem(mlir, &[&a, &b], &[48]);
    let result = read_f64s(&out[0]);
    assert_f64s_close(&result, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
}
