use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

fn read_f64s(buf: &[u8]) -> Vec<f64> {
    buf.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn extract_func<'a>(mlir: &'a str, name: &str) -> &'a str {
    let search = format!("func.func private @{name}(");
    let start = mlir.find(&search).unwrap_or_else(|| panic!("function @{name} not found"));
    let body = &mlir[start..];
    let end = body.find("\n  }").unwrap() + 4;
    &mlir[start..start + end]
}

/// Test: call inner_147 twice in sequence (chained output -> input)
/// This mimics what @main does: inner_147(T, bias) then inner_222(T+1, result)
/// but using the SAME function to eliminate compilation differences.
#[test]
fn test_chained_noise_calls() {
    let full_mlir = include_str!("../testdata/drone.stablehlo.mlir");

    let deps = [
        "inner_147",
        "_threefry_fold_in", "_normal", "_normal_real", "_uniform",
        "threefry2x32", "threefry2x32_169",
        "closed_call_158", "closed_call_173",
    ];
    let funcs: String = deps.iter().map(|n| extract_func(full_mlir, n)).collect::<Vec<_>>().join("\n");

    // Call inner_147 twice: first with (T, bias), then with (T+1, result_of_first)
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {{
    %first = call @inner_147(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %next_tick = stablehlo.add %arg0, %c : tensor<i64>
    %second = call @inner_147(%next_tick, %first) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %first, %second : tensor<3xf64>, tensor<3xf64>
  }}
  {funcs}
}}"#
    );

    let module = parse_module(&mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let tick: i64 = 0;
    let bias = [0.0025f64, 0.0001, 0.0005];
    let tick_buf = tick.to_le_bytes().to_vec();
    let bias_buf: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_ptrs = [tick_buf.as_ptr(), bias_buf.as_ptr()];
    let mut out_first = vec![0u8; 24];
    let mut out_second = vec![0u8; 24];
    let mut out_ptrs = [out_first.as_mut_ptr(), out_second.as_mut_ptr()];

    unsafe { tick_fn(input_ptrs.as_ptr(), out_ptrs.as_mut_ptr()); }

    let first = read_f64s(&out_first);
    let second = read_f64s(&out_second);

    eprintln!("first call (tick=0): {:?}", first);
    eprintln!("second call (tick=1, input=first): {:?}", second);

    // Now do them separately with explicit known inputs
    let mlir_single = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {{
    %0 = call @inner_147(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }}
  {funcs}
}}"#
    );
    let module_s = parse_module(&mlir_single).expect("parse single failed");
    let compiled_s = compile_module(&module_s).expect("compile single failed");
    let fn_s: TickFn = unsafe { std::mem::transmute(compiled_s.get_main_fn()) };

    // Call 1: tick=0, bias
    let mut out_s1 = vec![0u8; 24];
    let mut out_ptr_s1 = [out_s1.as_mut_ptr()];
    unsafe { fn_s(input_ptrs.as_ptr(), out_ptr_s1.as_mut_ptr()); }
    let sep_first = read_f64s(&out_s1);

    // Call 2: tick=1, result of call 1
    let tick2: i64 = 1;
    let tick2_buf = tick2.to_le_bytes().to_vec();
    let input_ptrs2 = [tick2_buf.as_ptr(), out_s1.as_ptr()];
    let mut out_s2 = vec![0u8; 24];
    let mut out_ptr_s2 = [out_s2.as_mut_ptr()];
    unsafe { fn_s(input_ptrs2.as_ptr(), out_ptr_s2.as_mut_ptr()); }
    let sep_second = read_f64s(&out_s2);

    eprintln!("separate call 1 (tick=0): {:?}", sep_first);
    eprintln!("separate call 2 (tick=1, input=sep_first): {:?}", sep_second);

    // Chained and separate should be identical
    assert_eq!(first, sep_first, "first call should match");
    assert_eq!(second, sep_second,
        "chained second call differs from separate!\n  chained: {:?}\n  separate: {:?}",
        second, sep_second
    );
}

/// Test: call inner_147, then inner_189 (sret), then use inner_147's output again.
/// This tests whether the sret call corrupts earlier SSA values.
#[test]
fn test_sret_does_not_corrupt_earlier_values() {
    let full_mlir = include_str!("../testdata/drone.stablehlo.mlir");

    let deps = [
        "inner_147", "inner_189",
        "_threefry_fold_in", "_normal", "_normal_real", "_uniform",
        "threefry2x32", "threefry2x32_169",
        "closed_call_158", "closed_call_173",
    ];
    let funcs: String = deps.iter().map(|n| extract_func(full_mlir, n)).collect::<Vec<_>>().join("\n");

    // Main: call inner_147, then inner_189 (sret), then return inner_147's output
    // and also call inner_147 again with tick+1 and inner_147's output
    let mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>, %arg2: tensor<7xf64>, %arg3: tensor<6xf64>, %arg4: tensor<4x3xf64>, %arg5: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {{
    %noise = call @inner_147(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %sret_result:2 = call @inner_189(%arg0, %arg2, %arg3, %arg4, %noise, %arg5) : (tensor<i64>, tensor<7xf64>, tensor<6xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    %c = stablehlo.constant dense<1> : tensor<i64>
    %next_tick = stablehlo.add %arg0, %c : tensor<i64>
    %second_noise = call @inner_147(%next_tick, %noise) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %noise, %sret_result#1, %second_noise : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }}
  {funcs}
}}"#
    );

    let module = parse_module(&mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let tick_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let tick: i64 = 0;
    let bias = [0.0025f64, 0.0001, 0.0005];
    let world_pos = [0.0f64; 7]; // identity quaternion + origin
    world_pos[..].to_vec()[3] = 1.0; // w=1
    let forces = [0.0f64; 6];
    let motor_dirs = [0.0f64; 12]; // 4x3 zeros
    let gyro_bias = [0.0f64; 3];

    let tick_buf = tick.to_le_bytes().to_vec();
    let bias_buf: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wp_buf: Vec<u8> = world_pos.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f_buf: Vec<u8> = forces.iter().flat_map(|v| v.to_le_bytes()).collect();
    let md_buf: Vec<u8> = motor_dirs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let gb_buf: Vec<u8> = gyro_bias.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_ptrs = [tick_buf.as_ptr(), bias_buf.as_ptr(), wp_buf.as_ptr(), f_buf.as_ptr(), md_buf.as_ptr(), gb_buf.as_ptr()];
    let mut out_noise = vec![0u8; 24];
    let mut out_sret_1 = vec![0u8; 24];
    let mut out_second = vec![0u8; 24];
    let mut out_ptrs = [out_noise.as_mut_ptr(), out_sret_1.as_mut_ptr(), out_second.as_mut_ptr()];

    unsafe { tick_fn(input_ptrs.as_ptr(), out_ptrs.as_mut_ptr()); }

    let noise = read_f64s(&out_noise);
    let second = read_f64s(&out_second);

    // Now do without the sret call in between
    let mlir_no_sret = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {{
    %noise = call @inner_147(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %next_tick = stablehlo.add %arg0, %c : tensor<i64>
    %second_noise = call @inner_147(%next_tick, %noise) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %noise, %second_noise : tensor<3xf64>, tensor<3xf64>
  }}
  {funcs}
}}"#
    );

    let module_ns = parse_module(&mlir_no_sret).expect("parse no-sret failed");
    let compiled_ns = compile_module(&module_ns).expect("compile no-sret failed");
    let fn_ns: TickFn = unsafe { std::mem::transmute(compiled_ns.get_main_fn()) };

    let input_ptrs_ns = [tick_buf.as_ptr(), bias_buf.as_ptr()];
    let mut out_noise_ns = vec![0u8; 24];
    let mut out_second_ns = vec![0u8; 24];
    let mut out_ptrs_ns = [out_noise_ns.as_mut_ptr(), out_second_ns.as_mut_ptr()];

    unsafe { fn_ns(input_ptrs_ns.as_ptr(), out_ptrs_ns.as_mut_ptr()); }

    let noise_ns = read_f64s(&out_noise_ns);
    let second_ns = read_f64s(&out_second_ns);

    eprintln!("WITH sret call in between:");
    eprintln!("  noise: {:?}", noise);
    eprintln!("  second: {:?}", second);
    eprintln!("WITHOUT sret call:");
    eprintln!("  noise: {:?}", noise_ns);
    eprintln!("  second: {:?}", second_ns);

    assert_eq!(noise, noise_ns, "first call should match regardless of sret");
    assert_eq!(second, second_ns,
        "sret call corrupted earlier values!\n  with_sret: {:?}\n  without: {:?}",
        second, second_ns
    );
}

/// Test: compile the FULL drone main function and check output[12] (gyro_bias)
/// matches what inner_236 would produce when called standalone with the correct inputs.
#[test]
fn test_full_main_vs_standalone_inner_236() {
    let full_mlir = include_str!("../testdata/drone.stablehlo.mlir");

    // Compile the full module
    let module = parse_module(full_mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let full_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    // Prepare all-zero inputs (same as initial world state)
    // arg sizes from the main signature
    let arg_sizes: Vec<usize> = vec![
        8, 8, 24, 32, 24, 56, 24, 24, 72, 32, 32, 48, 56, 48, 24, 32, 32, 32, 48, 48,
        8, 24, 96, 96, 24, 24, 8, 24, 24, 24, 32,
    ];
    let mut input_bufs: Vec<Vec<u8>> = arg_sizes.iter().map(|&sz| vec![0u8; sz]).collect();

    // Set arg5 (world_pos = tensor<7xf64>): quaternion w=1.0 at offset 24 (index 3)
    let one_bytes = 1.0f64.to_le_bytes();
    input_bufs[5][24..32].copy_from_slice(&one_bytes);

    // Set arg1 (sim_time_step): from IREE baseline = 0x8efb1787814e6b3f
    let ts_bytes: [u8; 8] = [0x8e, 0xfb, 0x17, 0x87, 0x81, 0x4e, 0x6b, 0x3f];
    input_bufs[1].copy_from_slice(&ts_bytes);

    // Set arg12 (tensor<7xf64>): initial inertia
    let arg12_hex = "0612143fc6dcb53f60764f1e166abd3f9ca223b9fc87c43f0000000000000000000000000000000000000000000000000000000000000000";
    let arg12_bytes: Vec<u8> = (0..arg12_hex.len()/2).map(|i| u8::from_str_radix(&arg12_hex[i*2..i*2+2], 16).unwrap()).collect();
    let copy_len = arg12_bytes.len().min(input_bufs[12].len());
    input_bufs[12][..copy_len].copy_from_slice(&arg12_bytes[..copy_len]);

    // Set arg21 (tensor<3xf64>): gyro_bias initial = [0.0025, 0.0001, 0.0005]
    let gb = [0.0025f64, 0.0001, 0.0005];
    let gb_bytes: Vec<u8> = gb.iter().flat_map(|v| v.to_le_bytes()).collect();
    input_bufs[21].copy_from_slice(&gb_bytes);

    // Set arg26 (f64): 1.0 (from IREE baseline)
    input_bufs[26].copy_from_slice(&one_bytes);

    // Set arg28 (tensor<3xf64>): attitude_estimate = [0, 0, 0, 1] -> but it's 3xf64 so just zeros
    // (identity quaternion is [0,0,0] for euler)

    let input_ptrs: Vec<*const u8> = input_bufs.iter().map(|b| b.as_ptr()).collect();

    // Output buffers: 31 outputs, need sizes from return types
    let out_sizes: Vec<usize> = vec![
        48, 24, 8, 24, 24, 24, 8, 8, 72, 32, 24, 32, 24, 24, 32, 48, 32, 96, 24, 96,
        56, 24, 32, 56, 48, 48, 24, 24, 32, 32, 8,
    ];
    let mut output_bufs: Vec<Vec<u8>> = out_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    let mut output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

    unsafe { full_fn(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()); }

    // Read output[12] = gyro_bias (tensor<3xf64>)
    let gyro_bias = read_f64s(&output_bufs[12]);
    eprintln!("Full main output[12] (gyro_bias): {:?}", gyro_bias);

    // Now compile inner_147 standalone and call it with the SAME inputs
    // inner_147 takes (sensor_tick: i64, bias: 3xf64) -> 3xf64
    // sensor_tick at tick 0 becomes 1 after inner_146 increments it
    // BUT in the full main, %633 = call @inner_146(%arg20) where %arg20 = 0
    // inner_146 adds 1, so %633 = 1
    let deps = [
        "inner_147",
        "_threefry_fold_in", "_normal", "_normal_real", "_uniform",
        "threefry2x32", "threefry2x32_169",
        "closed_call_158", "closed_call_173",
    ];
    let funcs: String = deps.iter().map(|n| extract_func(full_mlir, n)).collect::<Vec<_>>().join("\n");
    let standalone_mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {{
    %0 = call @inner_147(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }}
  {funcs}
}}"#
    );
    let sa_module = parse_module(&standalone_mlir).expect("parse standalone failed");
    let sa_compiled = compile_module(&sa_module).expect("compile standalone failed");
    let sa_fn: TickFn = unsafe { std::mem::transmute(sa_compiled.get_main_fn()) };

    // Call with tick=1, bias=[0.0025, 0.0001, 0.0005]
    let tick: i64 = 1;
    let tick_buf = tick.to_le_bytes().to_vec();
    let sa_input_ptrs = [tick_buf.as_ptr(), gb_bytes.as_ptr()];
    let mut sa_out = vec![0u8; 24];
    let mut sa_out_ptrs = [sa_out.as_mut_ptr()];
    unsafe { sa_fn(sa_input_ptrs.as_ptr(), sa_out_ptrs.as_mut_ptr()); }

    let sa_result = read_f64s(&sa_out);
    eprintln!("Standalone inner_147(tick=1, bias): {:?}", sa_result);

    // Also read output[3] = %1 from full main (should be inner(0+1, sim_ts, arg2) output)
    let out3 = read_f64s(&output_bufs[3]);
    eprintln!("Full main output[3]: {:?}", out3);

    // Check what output[6] is (sensor_tick output = %0 = arg0+1 = 1)
    let out6 = i64::from_le_bytes(output_bufs[6][..8].try_into().unwrap());
    eprintln!("Full main output[6] (tick counter): {}", out6);

}

/// Test: compile truncated main (first 648 lines, returns %634 = inner_147 output)
/// and compare against standalone inner_147 with the same inputs.
#[test]
fn test_truncated_main_inner147_output() {
    let truncated_mlir = std::fs::read_to_string("/tmp/drone_just_inner147.mlir")
        .expect("run the Python script to generate /tmp/drone_just_inner147.mlir first");

    let module = parse_module(&truncated_mlir).expect("parse truncated failed");
    let compiled = compile_module(&module).expect("compile truncated failed");
    let trunc_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    // Same inputs as the full main test
    let arg_sizes: Vec<usize> = vec![
        8, 8, 24, 32, 24, 56, 24, 24, 72, 32, 32, 48, 56, 48, 24, 32, 32, 32, 48, 48,
        8, 24, 96, 96, 24, 24, 8, 24, 24, 24, 32,
    ];
    let mut input_bufs: Vec<Vec<u8>> = arg_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    input_bufs[5][24..32].copy_from_slice(&1.0f64.to_le_bytes());
    input_bufs[1].copy_from_slice(&[0x8e, 0xfb, 0x17, 0x87, 0x81, 0x4e, 0x6b, 0x3f]);
    let gb = [0.0025f64, 0.0001, 0.0005];
    let gb_bytes: Vec<u8> = gb.iter().flat_map(|v| v.to_le_bytes()).collect();
    input_bufs[21].copy_from_slice(&gb_bytes);
    input_bufs[26].copy_from_slice(&1.0f64.to_le_bytes());
    let arg12_hex = "0612143fc6dcb53f60764f1e166abd3f9ca223b9fc87c43f0000000000000000000000000000000000000000000000000000000000000000";
    let arg12_bytes: Vec<u8> = (0..arg12_hex.len()/2).map(|i| u8::from_str_radix(&arg12_hex[i*2..i*2+2], 16).unwrap()).collect();
    let copy_len = arg12_bytes.len().min(input_bufs[12].len());
    input_bufs[12][..copy_len].copy_from_slice(&arg12_bytes[..copy_len]);

    let input_ptrs: Vec<*const u8> = input_bufs.iter().map(|b| b.as_ptr()).collect();
    let mut out = vec![0u8; 24];
    let mut out_ptrs = [out.as_mut_ptr()];

    unsafe { trunc_fn(input_ptrs.as_ptr(), out_ptrs.as_mut_ptr()); }
    let trunc_result = read_f64s(&out);

    // Compare with standalone inner_147(tick=1, bias)
    let full_mlir = include_str!("../testdata/drone.stablehlo.mlir");
    let deps = [
        "inner_147",
        "_threefry_fold_in", "_normal", "_normal_real", "_uniform",
        "threefry2x32", "threefry2x32_169",
        "closed_call_158", "closed_call_173",
    ];
    let funcs: String = deps.iter().map(|n| extract_func(full_mlir, n)).collect::<Vec<_>>().join("\n");
    let sa_mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {{
    %0 = call @inner_147(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }}
  {funcs}
}}"#
    );
    let sa_module = parse_module(&sa_mlir).expect("parse standalone failed");
    let sa_compiled = compile_module(&sa_module).expect("compile standalone failed");
    let sa_fn: TickFn = unsafe { std::mem::transmute(sa_compiled.get_main_fn()) };

    let tick: i64 = 1;
    let tick_buf = tick.to_le_bytes().to_vec();
    let sa_input_ptrs = [tick_buf.as_ptr(), gb_bytes.as_ptr()];
    let mut sa_out = vec![0u8; 24];
    let mut sa_out_ptrs = [sa_out.as_mut_ptr()];
    unsafe { sa_fn(sa_input_ptrs.as_ptr(), sa_out_ptrs.as_mut_ptr()); }
    let sa_result = read_f64s(&sa_out);

    eprintln!("Truncated main (648 lines, returns %%634): {:?}", trunc_result);
    eprintln!("Standalone inner_147(tick=1, bias):         {:?}", sa_result);

    for i in 0..3 {
        let diff = (trunc_result[i] - sa_result[i]).abs();
        eprintln!("  [{}] trunc={:.17e} standalone={:.17e} diff={:.3e}", i, trunc_result[i], sa_result[i], diff);
        assert!(
            diff < 1e-15,
            "Truncated main produces different inner_147 output than standalone!\n  truncated: {:?}\n  standalone: {:?}",
            trunc_result, sa_result
        );
    }
}

/// Test: truncated main through inner_222 (1290 lines).
/// Returns %634 (inner_147), %1268 (inner_222), %1267 (tick for inner_222).
/// Compare inner_222 output against standalone call.
#[test]
fn test_truncated_main_inner222_output() {
    let trunc_mlir = std::fs::read_to_string("/tmp/drone_through_inner222.mlir")
        .expect("run Python to generate /tmp/drone_through_inner222.mlir");

    let module = parse_module(&trunc_mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let trunc_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let arg_sizes: Vec<usize> = vec![
        8, 8, 24, 32, 24, 56, 24, 24, 72, 32, 32, 48, 56, 48, 24, 32, 32, 32, 48, 48,
        8, 24, 96, 96, 24, 24, 8, 24, 24, 24, 32,
    ];
    let mut input_bufs: Vec<Vec<u8>> = arg_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    input_bufs[5][24..32].copy_from_slice(&1.0f64.to_le_bytes());
    input_bufs[1].copy_from_slice(&[0x8e, 0xfb, 0x17, 0x87, 0x81, 0x4e, 0x6b, 0x3f]);
    let gb = [0.0025f64, 0.0001, 0.0005];
    let gb_bytes: Vec<u8> = gb.iter().flat_map(|v| v.to_le_bytes()).collect();
    input_bufs[21].copy_from_slice(&gb_bytes);
    input_bufs[26].copy_from_slice(&1.0f64.to_le_bytes());

    let input_ptrs: Vec<*const u8> = input_bufs.iter().map(|b| b.as_ptr()).collect();
    let mut out_634 = vec![0u8; 24];
    let mut out_1268 = vec![0u8; 24];
    let mut out_1267 = vec![0u8; 8];
    let mut out_ptrs = [out_634.as_mut_ptr(), out_1268.as_mut_ptr(), out_1267.as_mut_ptr()];

    unsafe { trunc_fn(input_ptrs.as_ptr(), out_ptrs.as_mut_ptr()); }

    let val_634 = read_f64s(&out_634);
    let val_1268 = read_f64s(&out_1268);
    let val_1267 = i64::from_le_bytes(out_1267[..8].try_into().unwrap());

    eprintln!("Truncated main (1290 lines):");
    eprintln!("  %%634 (inner_147 output): {:?}", val_634);
    eprintln!("  %%1268 (inner_222 output): {:?}", val_1268);
    eprintln!("  %%1267 (tick for inner_222): {}", val_1267);

    // Now call inner_222 standalone with tick=val_1267, bias=val_634
    let full_mlir = include_str!("../testdata/drone.stablehlo.mlir");
    let deps = [
        "inner_222",
        "_threefry_fold_in", "_normal", "_normal_real", "_uniform",
        "threefry2x32", "threefry2x32_169",
        "closed_call_158", "closed_call_173",
    ];
    let funcs: String = deps.iter().map(|n| extract_func(full_mlir, n)).collect::<Vec<_>>().join("\n");
    let sa_mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {{
    %0 = call @inner_222(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }}
  {funcs}
}}"#
    );
    let sa_module = parse_module(&sa_mlir).expect("parse standalone failed");
    let sa_compiled = compile_module(&sa_module).expect("compile standalone failed");
    let sa_fn: TickFn = unsafe { std::mem::transmute(sa_compiled.get_main_fn()) };

    let tick_buf = val_1267.to_le_bytes().to_vec();
    let bias_buf = out_634.clone();
    let sa_input_ptrs = [tick_buf.as_ptr(), bias_buf.as_ptr()];
    let mut sa_out = vec![0u8; 24];
    let mut sa_out_ptrs = [sa_out.as_mut_ptr()];
    unsafe { sa_fn(sa_input_ptrs.as_ptr(), sa_out_ptrs.as_mut_ptr()); }
    let sa_result = read_f64s(&sa_out);

    eprintln!("Standalone inner_222(tick={}, bias={:?}): {:?}", val_1267, val_634, sa_result);

    // Compare
    for i in 0..3 {
        let diff = (val_1268[i] - sa_result[i]).abs();
        eprintln!("  [{}] trunc={:.17e} standalone={:.17e} diff={:.3e}", i, val_1268[i], sa_result[i], diff);
    }
    assert_eq!(
        out_1268, sa_out,
        "inner_222 in truncated main differs from standalone!\n  main: {:?}\n  standalone: {:?}",
        val_1268, sa_result
    );
}

/// Test: 1935 lines (through inner_236 = gyro_bias output).
#[test]
fn test_truncated_main_inner236_output() {
    let trunc_mlir = std::fs::read_to_string("/tmp/drone_through_inner236.mlir")
        .expect("run Python to generate /tmp/drone_through_inner236.mlir");

    let module = parse_module(&trunc_mlir).expect("parse failed");
    let compiled = compile_module(&module).expect("compile failed");
    let trunc_fn: TickFn = unsafe { std::mem::transmute(compiled.get_main_fn()) };

    let arg_sizes: Vec<usize> = vec![
        8, 8, 24, 32, 24, 56, 24, 24, 72, 32, 32, 48, 56, 48, 24, 32, 32, 32, 48, 48,
        8, 24, 96, 96, 24, 24, 8, 24, 24, 24, 32,
    ];
    let mut input_bufs: Vec<Vec<u8>> = arg_sizes.iter().map(|&sz| vec![0u8; sz]).collect();
    input_bufs[5][24..32].copy_from_slice(&1.0f64.to_le_bytes());
    input_bufs[1].copy_from_slice(&[0x8e, 0xfb, 0x17, 0x87, 0x81, 0x4e, 0x6b, 0x3f]);
    let gb = [0.0025f64, 0.0001, 0.0005];
    let gb_bytes: Vec<u8> = gb.iter().flat_map(|v| v.to_le_bytes()).collect();
    input_bufs[21].copy_from_slice(&gb_bytes);
    input_bufs[26].copy_from_slice(&1.0f64.to_le_bytes());

    let input_ptrs: Vec<*const u8> = input_bufs.iter().map(|b| b.as_ptr()).collect();
    let mut out_634 = vec![0u8; 24];
    let mut out_1268 = vec![0u8; 24];
    let mut out_1902 = vec![0u8; 24];
    let mut out_1901 = vec![0u8; 8];
    let mut out_ptrs = [out_634.as_mut_ptr(), out_1268.as_mut_ptr(), out_1902.as_mut_ptr(), out_1901.as_mut_ptr()];

    unsafe { trunc_fn(input_ptrs.as_ptr(), out_ptrs.as_mut_ptr()); }

    let val_634 = read_f64s(&out_634);
    let val_1268 = read_f64s(&out_1268);
    let val_1902 = read_f64s(&out_1902);
    let val_1901 = i64::from_le_bytes(out_1901[..8].try_into().unwrap());

    eprintln!("Truncated main (1935 lines):");
    eprintln!("  %%634 (inner_147): {:?}", val_634);
    eprintln!("  %%1268 (inner_222): {:?}", val_1268);
    eprintln!("  %%1902 (inner_236 = gyro_bias): {:?}", val_1902);
    eprintln!("  %%1901 (tick for inner_236): {}", val_1901);

    // Standalone inner_236 with tick=val_1901, bias=val_1268
    let full_mlir = include_str!("../testdata/drone.stablehlo.mlir");
    let deps = [
        "inner_236",
        "_threefry_fold_in", "_normal", "_normal_real", "_uniform",
        "threefry2x32", "threefry2x32_169",
        "closed_call_158", "closed_call_173",
    ];
    let funcs: String = deps.iter().map(|n| extract_func(full_mlir, n)).collect::<Vec<_>>().join("\n");
    let sa_mlir = format!(
        r#"module @module {{
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {{
    %0 = call @inner_236(%arg0, %arg1) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }}
  {funcs}
}}"#
    );
    let sa_module = parse_module(&sa_mlir).expect("parse standalone failed");
    let sa_compiled = compile_module(&sa_module).expect("compile standalone failed");
    let sa_fn: TickFn = unsafe { std::mem::transmute(sa_compiled.get_main_fn()) };

    let tick_buf = val_1901.to_le_bytes().to_vec();
    let sa_input_ptrs = [tick_buf.as_ptr(), out_1268.as_ptr()];
    let mut sa_out = vec![0u8; 24];
    let mut sa_out_ptrs = [sa_out.as_mut_ptr()];
    unsafe { sa_fn(sa_input_ptrs.as_ptr(), sa_out_ptrs.as_mut_ptr()); }
    let sa_result = read_f64s(&sa_out);

    eprintln!("Standalone inner_236(tick={}, bias={:?}): {:?}", val_1901, val_1268, sa_result);

    for i in 0..3 {
        let diff = (val_1902[i] - sa_result[i]).abs();
        eprintln!("  [{}] trunc={:.17e} standalone={:.17e} diff={:.3e}", i, val_1902[i], sa_result[i], diff);
    }
    assert_eq!(
        out_1902, sa_out,
        "inner_236 in truncated main differs from standalone!\n  main: {:?}\n  standalone: {:?}",
        val_1902, sa_result
    );
}

