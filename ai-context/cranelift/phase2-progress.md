# Cranelift Backend Progress: Phase 2 Summary

**Date:** April 13, 2026

## Results Summary

| Example | Status | Max Error | Cranelift RTF | IREE RTF | Speedup |
|---------|--------|-----------|---------------|----------|---------|
| ball | PASS | 1.8e-15 | 12,456x | 78.6x | **159x** |
| three-body | PASS | bit-identical | 4,423x | 29.0x | **153x** |
| rocket | PASS | 4.4e-13 | 18.4x | 2.3x | **8x** |
| drone | FAIL (22/37 files) | diverges at row 3 | 531x | 2.9x | **183x** |
| cube-sat | BLOCKED | compile hangs | -- | -- | -- |
| linalg-iree | NOT STARTED | -- | -- | -- | -- |

## What Was Implemented

### New StableHLO Ops (18 new, 51 total)

| Category | Ops |
|----------|-----|
| Trig | `sine`, `cosine`, `tan`, `atan2`, `acos` (chlo) |
| Elementwise | `abs`, `sign`, `minimum`, `exponential`, `log`, `power`, `tanh` |
| Rounding | `floor`, `round_nearest_even` |
| Clamping | `clamp` |
| Shape | `reverse`, `pad` |
| Mutation | `scatter` (simple index-set pattern) |
| Custom calls | `custom_call @lapack_dgesdd_ffi` (3x3 SVD via Jacobi) |
| Modular | `remainder` (float fmod + integer srem) |

### Infrastructure

- All new ops have golden-value tests in `tests/ops.rs` (51 tests total)
- E2E parse+compile tests for drone, rocket, cube-sat
- Testdata MLIR files for all three examples (`git add -f` for nix)
- `dup_scratch` buffer in `cranelift_exec.rs` to avoid null output pointer writes

### Bugs Found and Fixed

1. **N-dimensional `slice` was broken** (critical for rocket): Only 1D and 2D slicing worked. For 3D+ tensors, the fallback did a flat `vals[s..e]` which returned wrong elements. Fixed with proper N-dimensional coordinate-based slicing.

2. **Integer `remainder` called float `fmod`**: When `stablehlo.remainder` operated on `tensor<i64>`, the lowering called `fmod` (f64 -> f64) instead of `srem` (i64 -> i64), causing a Cranelift type verification failure.

3. **`reduce` always used `fadd`**: The `apply_reduce_op` helper used `fadd` for `ReduceOp::Add` regardless of the operand type. Now checks `val_type.is_float()` before choosing `fadd` vs `iadd`.

4. **Missing `I1 -> F64` conversion**: The `convert_value` function had no case for `(ElementType::I1, ElementType::F64)`, falling through to `_ => v` which returned the raw `i8` value. This caused `fsub.i8` type errors when boolean values were converted to float. Fixed by adding explicit `I1 -> F64`, `I1 -> F32`, `F64 -> I1`, `I64 -> I1`, `UI64 -> F64`, and other missing conversion paths.

## Remaining Issue 1: Drone Divergence in Full Main Function

### Symptoms

- 22 of 37 CSV output files diverge from IREE baseline (21 of 31 raw tick outputs differ)
- Divergence is deterministic (two Cranelift runs produce identical wrong values)
- Inputs to the tick function are byte-identical between IREE and Cranelift (verified via runtime logging)
- The divergence occurs even with `ticks_per_telemetry=1` (single tick per batch)

### Ruled Out

Through systematic investigation, the following have been definitively ruled out:

1. **Batch loop feedback** -- setting `ticks_per_telemetry=1` (no between-tick feedback) still produces divergent output
2. **Individual function compilation** -- `inner_147` and `inner_222` (identical PRNG noise functions) produce bit-identical output when called in isolation
3. **Chained function calls** -- calling inner_147 then inner_147 again with the first's output produces correct results in a test main function
4. **sret (struct return) corruption** -- calling inner_147 then inner_189 (sret, 15 elements) then using inner_147's output afterward produces correct results
5. **Cranelift optimization bugs** -- setting `opt_level=none` produces the same divergence
6. **Input data mismatch** -- runtime byte-dump of all 31 inputs confirms identical data between backends
7. **MLIR text differences** -- `diff` shows the MLIR is byte-identical between IREE and Cranelift dump
8. **SSA name shadowing** -- dedicated test proves shadowed names work correctly
9. **Missing type conversions** -- `I1->F64` and other conversions have been added and verified

### What Remains

The bug manifests **only** when the full drone `@main` function (~1900 instructions, ~80 function calls including 10 sret calls) is compiled as a single Cranelift function. Every sub-component tested in isolation produces correct results.

### MLIR Tracing

The return statement maps outputs to SSA values. Tracing the divergence chain:
- `output[12]` = `%1902 = call @inner_236(%1901, %1268)` -- gyro_bias (DIVERGES)
- `%1268 = call @inner_222(%1267, %634)` -- 2nd noise update (uses inner_147's output)
- `%634 = call @inner_147(%633, %arg21)` -- 1st noise update (accel_bias, MATCHES)
- `inner_147` and `inner_222` have **byte-identical** MLIR bodies with identical PRNG seeds

Since `inner_147` (output `%634`) matches IREE, and `inner_222` (output `%1268`) diverges, and both have identical code, the `%634` value must somehow be corrupted between its definition at line 647 and its use at line 1290 -- a span of 643 lines of intervening instructions including multiple sret calls.

### Recommended Next Steps

1. **Binary search within main**: Build a test that compiles the first N instructions of `@main` (up to a midpoint), returns intermediate values, and checks against IREE. Narrow down which instruction first produces a wrong value.

2. **Cranelift IR dump comparison**: Dump the Cranelift IR for the full `@main` function (using `CRANELIFT_DUMP_IR=1`), find the instructions corresponding to the `%634` -> `%1268` chain, and verify the SSA data flow is correct in the compiled IR.

3. **Value liveness test**: Create a test with ~100 "filler" function calls between defining `%634` and using it, to see if register spill/reload introduces errors under high register pressure.

4. **Compare with smaller function**: If splitting `@main` into two halves (each ~950 instructions) eliminates the divergence, the issue is register pressure or stack layout with very large functions.

## Remaining Issue 2: Cube-Sat Compilation Scalability

### Symptoms

- Cube-sat MLIR parses successfully (3,995 lines, 63 functions)
- Compilation hangs (>5 minutes, 100% CPU) in the `compile_cube_sat_mlir` test
- The I1->F64 fix resolved the `fsub.i8` type error that appeared before the hang

### Root Cause

The cube-sat simulation uses the EGM08 gravity model which operates on `tensor<65x65xf64>` matrices (4,225 elements each). Under the current scalar ABI, each matrix becomes 4,225 SSA values. Functions like `closed_call` (EGM08 scan body) have:
- 65x65 matrix as a parameter = 4,225 f64 SSA values
- Multiple 65x65 matrix multiplications (dot_general) within the function body
- While loops carrying 65x65 matrices as loop state

This exceeds Cranelift's practical capacity for the scalar ABI. The JIT compiler spends all its time in register allocation and instruction selection for thousands of SSA values per function.

### Recommended Approaches

**Option A: Pointer ABI for large tensors** (best long-term fix)
- For functions where any parameter or return type exceeds a threshold (e.g., 64 elements), switch to a pointer-based ABI where tensors are passed as `*const f64` pointers instead of individual SSA values.
- Store/load from stack slots at function boundaries.
- Would require changes to `lower_callee_body`, `lower_call`, and the sret handling.

**Option B: Inline EGM08 as a Rust runtime function**
- Detect `closed_call` functions that operate on large matrices.
- Replace the JIT-compiled while loop with a Rust native function call that performs the EGM08 accumulation.
- Similar to how we handle SVD via `custom_call`.

**Option C: Skip EGM08 scan unrolling**
- Use JAX's `jax.lax.scan` with a `while` loop that has fewer iterations.
- The EGM08 model accumulates spherical harmonics up to degree 64, resulting in 65 iterations of the scan.
- Each iteration works on 65x65 state -- this is where the 65x65 matrices come from.

## Remaining Issue 3: Linalg-IREE

Not started. This example intentionally exercises LAPACK operations (Cholesky, LU solve, QR, SVD, eigendecomposition, determinant). Implementation requires:

1. Dumping the MLIR to catalog all `stablehlo.custom_call @lapack_*` operations
2. Implementing each LAPACK operation as a Rust runtime function (like the SVD Jacobi approach)
3. For small matrices (typically 2x2 to 12x12 in Kalman filter contexts), pure Rust implementations are feasible
4. For larger matrices, linking to OpenBLAS or similar via FFI

The cube-sat scalability issue (65x65 matrices) should be resolved first since linalg-iree may have similar or larger matrices.

## Files Modified

| File | Lines Changed | Purpose |
|------|--------------|---------|
| `libs/cranelift-mlir/src/ir.rs` | +74 | 18 new Instruction variants |
| `libs/cranelift-mlir/src/parser.rs` | +322 | Parsers for all new ops, scatter, custom_call, pad |
| `libs/cranelift-mlir/src/lower.rs` | +720 | Lowering for all new ops, SVD runtime, type fixes |
| `libs/cranelift-mlir/tests/ops.rs` | +321 | 18 new golden-value tests |
| `libs/cranelift-mlir/tests/drone_e2e.rs` | new | Drone parse+compile test |
| `libs/cranelift-mlir/tests/rocket_e2e.rs` | new | Rocket parse+compile test |
| `libs/cranelift-mlir/tests/cube_sat_e2e.rs` | new | Cube-sat parse+compile test |
| `libs/cranelift-mlir/testdata/drone.stablehlo.mlir` | new (8,121 lines) | Drone MLIR testdata |
| `libs/cranelift-mlir/testdata/rocket.stablehlo.mlir` | new (4,481 lines) | Rocket MLIR testdata |
| `libs/cranelift-mlir/testdata/cube-sat.stablehlo.mlir` | new (3,995 lines) | Cube-sat MLIR testdata |
| `libs/nox-py/src/cranelift_exec.rs` | +7 | Scratch buffer for duplicate output slots |
