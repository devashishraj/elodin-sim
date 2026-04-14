# cranelift-mlir

StableHLO MLIR to Cranelift JIT compiler for Elodin simulations.

Parses StableHLO MLIR text (as emitted by `jax.jit().lower().compiler_ir(dialect="stablehlo")`)
and compiles it to native code via Cranelift JIT. Designed for small, CPU-bound physics
simulations where IREE overhead dominates.

## Performance

Validated against five regression examples:

| Example | Baseline RTF | Cranelift RTF | Speedup | Status |
|---------|-------------|---------------|---------|--------|
| ball | 79x | 10,767x | **137x** | PASS |
| three-body | 29x | 4,946x | **170x** | PASS |
| drone | 2.9x | 292x | **101x** | PASS |
| rocket | 2.3x | 32x | **14x** | PASS |
| cube-sat | 0.56x | 3.8x | **6.7x** | PASS |

## Architecture

```
StableHLO text --> parser.rs --> IR (ir.rs) --> lower.rs --> Cranelift IR --> native fn pointer
                                                    |
                                              tensor_rt.rs  (runtime library for N-D ops)
```

- **`ir.rs`** (~437 lines): Internal IR types -- `Module`, `FuncDef`, 35+ `Instruction` variants, `TensorType`, `GatherDims`.
- **`parser.rs`** (~2,276 lines): Winnow-based parser converting StableHLO MLIR text to IR. Supports child contexts for while/case scoping.
- **`lower.rs`** (~5,756 lines): Cranelift JIT compilation. Dual ABI: scalar path for small functions, pointer-ABI path for large-tensor functions.
- **`tensor_rt.rs`** (~1,125 lines): Runtime library for N-dimensional tensor operations (broadcast, slice, transpose, reduce, gather, scatter, matmul, etc.).

### Dual ABI

Functions are classified at parse time based on tensor sizes:

- **Scalar ABI** (all tensors <= 64 elements): Each tensor element is a separate Cranelift SSA value. Fast for small simulations. Functions returning > 8 elements use struct return (sret).
- **Pointer ABI** (any tensor > 64 elements): All tensors are stack-allocated buffers passed by pointer. Operations delegate to `tensor_rt` functions. Used for EGM08 gravity model (65x65 matrices).

Cross-ABI calls are marshaled automatically at call sites.

The compiled `main` function always uses a pointer ABI:
```
extern "C" fn(inputs: *const *const u8, outputs: *mut *mut u8)
```

### While Loop Scoping

While-loop cond/body blocks use `ctx.child()` parser contexts that inherit the parent's name-to-ValueId mappings. This allows while-loop bodies to reference outer-scope variables (function parameters, constants defined before the loop). The `iter_arg_ids` are stored in the IR and used by both the scalar and pointer-ABI While handlers.

## Testing

### 114+ tests across 17 test binaries

```bash
cargo test -p cranelift-mlir                          # all tests
cargo test -p cranelift-mlir --test ops               # 114 per-op and integration tests
cargo test -p cranelift-mlir --test checkpoint_test --release -- --ignored  # checkpoint verifier (needs CHECKPOINT_DIR)
```

### Per-op golden-value tests

Every supported StableHLO op has individual tests in `tests/ops.rs` covering both scalar and pointer-ABI (mem) paths. Tests include:
- Small-scale ops (3x3, 4x4 matrices)
- 65x65 matrix operations (transpose, broadcast, multiply, reduce)
- While loops with cross-ABI calls and outer-scope variable references
- Multi-function pointer-ABI chains (roll, scatter, broadcast, multiply, reduce)
- N-D gather with multi-element index vectors

### Tick Checkpoint Tool

A reusable diagnostic for comparing Cranelift JIT output against XLA reference values. Useful for diagnosing any compilation correctness bug.

**Generate a checkpoint** (captures inputs, Cranelift outputs, and XLA reference outputs):
```bash
ELODIN_BACKEND=cranelift \
ELODIN_CRANELIFT_CHECKPOINT_DIR=/tmp/ckpt \
  bash scripts/ci/regress.sh <example> examples/<example>/main.py
```

**Verify with fast Rust test** (no Python needed, ~0.06s):
```bash
CHECKPOINT_DIR=/path/to/checkpoint \
  cargo test -p cranelift-mlir --test checkpoint_test --release -- --ignored --nocapture
```

The verifier reports pass/fail for each output with element-level diffs:
```
output 0: FAIL at elem 3/12: got=-0.000236, want=-8.680
output 1: OK (1 elems)
...
9 of 28 outputs FAILED: [0, 8, 12, 14, 15, 16, 19, 23, 24]
```

Checkpoints can be committed to `testdata/checkpoints/<example>/` as permanent regression data.

### Adding a new op

1. Add the `Instruction` variant in `src/ir.rs`
2. Add the parser arm in `src/parser.rs` (wire it in `parse_op()`)
3. Add the lowering case in `src/lower.rs`:
   - Scalar path: `lower_instruction()` match arm
   - Pointer-ABI path: `lower_instruction_mem()` match arm
   - If the op needs an N-D runtime function, add it to `src/tensor_rt.rs` and register in `TensorRtIds`
4. Add golden-value tests in `tests/ops.rs` for both scalar and `_mem` paths
5. Run `cargo test -p cranelift-mlir`

### Adding a new simulation example

1. Dump the MLIR: `ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DUMP_DIR=/tmp/dump python examples/<name>/main.py run`
2. Catalog ops: `grep -oE '(stablehlo|chlo|func)\.[a-z_]+' /tmp/dump/.../stablehlo.mlir | sort | uniq -c | sort -rn`
3. Copy to testdata: `cp /tmp/dump/.../stablehlo.mlir testdata/<name>.stablehlo.mlir`
4. Create `tests/<name>_e2e.rs` with parse + compile tests
5. Implement any missing ops (see "Adding a new op" above)
6. Run regression: `ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh <name> examples/<name>/main.py`

### Debugging a failing example

1. Generate a checkpoint: `ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_CHECKPOINT_DIR=/tmp/ckpt bash scripts/ci/regress.sh <name> examples/<name>/main.py`
2. Run verifier: `CHECKPOINT_DIR=/tmp/ckpt cargo test -p cranelift-mlir --test checkpoint_test --release -- --ignored --nocapture`
3. The verifier shows which outputs diverge from XLA and the first mismatching element
4. To bisect: modify the checkpoint's `stablehlo.mlir` to expose intermediate values as additional returns, regenerate XLA reference, and re-run the verifier

## Dumping StableHLO MLIR

```bash
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DUMP_DIR=/tmp/cranelift_dump \
  python examples/ball/main.py run
```

Produces `stablehlo.mlir` and `compile_context.json` in a timestamped subdirectory.

## Op Coverage

| Op | Tested | Notes |
|----|--------|-------|
| **Arithmetic** | | |
| stablehlo.add | yes | float + integer |
| stablehlo.subtract | yes | float + integer |
| stablehlo.multiply | yes | float + integer |
| stablehlo.divide | yes | float + signed/unsigned integer |
| stablehlo.negate | yes | float + integer |
| stablehlo.sqrt | yes | inline Cranelift instruction |
| stablehlo.power | yes | via libm powf |
| stablehlo.maximum | yes | float |
| stablehlo.minimum | yes | float |
| stablehlo.abs | yes | float |
| stablehlo.floor | yes | float |
| stablehlo.sign | yes | float |
| stablehlo.remainder | yes | float |
| stablehlo.sine | yes | via libm |
| stablehlo.cosine | yes | via libm |
| stablehlo.tanh | yes | via libm |
| stablehlo.exponential | yes | via libm |
| stablehlo.log | yes | via libm |
| chlo.tan | yes | via libm |
| chlo.erf_inv | yes | Cephes ndtri-based, ~15 digits f64 |
| **Comparison / Select** | | |
| stablehlo.compare | yes | all directions, float/signed/unsigned, i32/i64 |
| stablehlo.select | yes | f64/i32/i64 with i1 mask |
| **Constants** | | |
| stablehlo.constant | yes | scalar, array, splat, hex blobs |
| **Shape** | | |
| stablehlo.reshape | yes | |
| stablehlo.broadcast_in_dim | yes | N-dimensional with proper dims mapping |
| stablehlo.slice | yes | N-dimensional |
| stablehlo.concatenate | yes | any dimension, N-dimensional |
| stablehlo.transpose | yes | 2D + N-D via tensor_rt |
| stablehlo.pad | yes | N-dimensional |
| stablehlo.reverse | yes | |
| **Dynamic** | | |
| stablehlo.dynamic_slice | yes | N-dimensional runtime indices |
| stablehlo.dynamic_update_slice | yes | N-dimensional runtime indices |
| **Type Conversion** | | |
| stablehlo.convert | yes | all numeric type pairs (f64/f32/i64/i32/ui32/ui64/i1) |
| stablehlo.bitcast_convert | yes | |
| stablehlo.iota | yes | N-dimensional, f64 + i64 |
| **Integer Bitwise** | | |
| stablehlo.xor | yes | |
| stablehlo.or | yes | |
| stablehlo.and | yes | |
| stablehlo.shift_left | yes | |
| stablehlo.shift_right_logical | yes | |
| **Linear Algebra** | | |
| stablehlo.dot_general | yes | scalar, 1D dot, matvec, matmul, batched dot |
| stablehlo.reduce | yes | add/min/max, all dimensions |
| **Indexing** | | |
| stablehlo.gather | yes | 1D row-select + N-D multi-index (index_vector_dim) |
| stablehlo.scatter | yes | 1D index-set with i32/i64 indices |
| **Control Flow** | | |
| stablehlo.while | yes | loop blocks with outer-scope variable access, cross-ABI calls |
| stablehlo.case | yes | branching with dispatch chain and merge block |
| func.call | yes | scalar ABI, sret, pointer ABI, cross-ABI marshaling |

## Known Issues

- **Debug-mode UB check**: In debug builds, some JIT functions trigger `ptr::copy_nonoverlapping` precondition checks. Use `--release` for checkpoint verification.
- **Stack overflow**: Complex functions (e.g., inner_375 with 5 while loops) may overflow the default test thread stack. The checkpoint test uses a 64MB thread stack.
- **No SIMD**: All operations are scalar Cranelift instructions. Future optimization opportunity via faer-rs.
