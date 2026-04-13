# cranelift-mlir

StableHLO MLIR to Cranelift JIT compiler for Elodin simulations.

Parses StableHLO MLIR text (as emitted by `jax.jit().lower().compiler_ir(dialect="stablehlo")`)
and compiles it to native code via Cranelift JIT. Designed for small, CPU-bound physics
simulations where IREE overhead dominates.

## Performance

Validated against two regression examples (bit-for-bit identical to IREE baselines):

| Example | IREE RTF | Cranelift RTF | Speedup | Compile time |
|---------|----------|---------------|---------|-------------|
| ball | 79x | 12,300x | **156x** | 7ms vs 4.7s |
| three-body | 29x | 4,660x | **161x** | 7ms vs 4.7s |

## Architecture

```
StableHLO text --> parser.rs --> IR (ir.rs) --> lower.rs --> Cranelift IR --> native fn pointer
```

- **`ir.rs`** (361 lines): Internal IR types -- `Module`, `FuncDef`, 33 `Instruction` variants, `TensorType`.
- **`parser.rs`** (1,924 lines): Winnow-based parser converting StableHLO MLIR text to IR.
- **`lower.rs`** (2,200 lines): Cranelift JIT compilation from IR to native function pointers.

### ABI

The compiled `main` function uses a pointer ABI:
```
extern "C" fn(inputs: *const *const u8, outputs: *mut *mut u8)
```
Each input/output pointer addresses the raw byte buffer for one tensor.

Non-main functions use a scalar ABI where each tensor element is a separate Cranelift SSA
value. Functions returning more than 8 scalar elements use struct return (sret) via pointer.

## Testing

### 68 tests across 13 test binaries

```bash
cargo test -p cranelift-mlir                          # all tests
cargo test -p cranelift-mlir --test ops               # 33 per-op golden-value tests
cargo test -p cranelift-mlir --test ball_e2e           # ball parse + compile
cargo test -p cranelift-mlir --test three_body_e2e     # three-body parse + compile
cargo test -p cranelift-mlir --test test_threefry_e2e  # PRNG bit-exact verification
```

### Per-op golden-value tests

Every supported StableHLO op has an individual test in `tests/ops.rs`. Each test:

1. Defines a minimal MLIR module containing only the op under test
2. Provides known input byte buffers
3. Provides expected output byte buffers (hand-computed or verified against JAX/NumPy)
4. Parses, compiles, executes, asserts outputs match

Tolerances: exact match for integers, `1e-10` relative tolerance for floats.

### Adding a new op

1. Add the `Instruction` variant in `src/ir.rs`
2. Add the parser arm in `src/parser.rs` (wire it in `parse_op()`)
3. Add the lowering case in `src/lower.rs` (add match arm in `lower_instruction()`)
4. Add a golden-value test in `tests/ops.rs` with MLIR snippet + known inputs/outputs
5. Run `cargo test -p cranelift-mlir`

### Adding a new simulation example

1. Dump the MLIR: `ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DUMP_DIR=/tmp/dump python examples/<name>/main.py run`
2. Catalog ops: `grep -oE '(stablehlo|chlo|func)\.[a-z_]+' /tmp/dump/.../stablehlo.mlir | sort | uniq -c | sort -rn`
3. Copy to testdata: `cp /tmp/dump/.../stablehlo.mlir testdata/<name>.stablehlo.mlir && git add -f testdata/<name>.stablehlo.mlir`
4. Create `tests/<name>_e2e.rs` with parse + compile tests
5. Implement any missing ops (see "Adding a new op" above)
6. Run regression: `ELODIN_BACKEND=cranelift bash ./scripts/ci/regress.sh <name> examples/<name>/main.py`

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
| stablehlo.maximum | yes | float (signed int comparison -- see known issues) |
| **Comparison / Select** | | |
| stablehlo.compare | yes | all directions, float/signed/unsigned types |
| stablehlo.select | -- | implemented, not independently tested |
| **Constants** | | |
| stablehlo.constant | yes | scalar, array, splat, hex blobs, hex bit patterns |
| **Shape** | | |
| stablehlo.reshape | yes | |
| stablehlo.broadcast_in_dim | yes | N-dimensional with proper dims mapping |
| stablehlo.slice | yes | 1D and 2D static slicing |
| stablehlo.concatenate | yes | any dimension (N-dimensional) |
| stablehlo.transpose | yes | arbitrary permutation |
| **Dynamic** | | |
| stablehlo.dynamic_slice | yes | runtime indices, stack-slot based |
| stablehlo.dynamic_update_slice | yes | runtime indices, stack-slot based |
| **Type Conversion** | | |
| stablehlo.convert | yes | all numeric type pairs (f64/f32/i64/i32/ui32/ui64/i1) |
| stablehlo.bitcast_convert | -- | implemented, not independently tested |
| stablehlo.iota | yes | |
| **Integer Bitwise** | | |
| stablehlo.xor | yes | |
| stablehlo.or | yes | |
| stablehlo.and | yes | |
| stablehlo.shift_left | yes | |
| stablehlo.shift_right_logical | yes | |
| **Linear Algebra** | | |
| stablehlo.dot_general | yes | scalar, 1D dot, matvec, matmul, batched dot |
| stablehlo.reduce | yes | add/min/max, dimensions [0] and [1] for 1D and 2D |
| **Indexing** | | |
| stablehlo.gather | yes | row-select pattern (start_index_map=[0], collapsed_slice_dims=[0]) |
| **Control Flow** | | |
| stablehlo.while | yes | real Cranelift loop blocks with header/body/exit |
| stablehlo.case | yes | real branching with dispatch chain and merge block |
| func.call | yes | scalar ABI + sret for large returns |
| **Special Functions** | | |
| chlo.erf_inv | yes | Cephes ndtri-based, ~15 digits f64 precision |

## Known Issues

- **`stablehlo.maximum` on unsigned integers**: Uses signed comparison, which is incorrect for `ui32`/`ui64`. Not triggered by current examples.
- **Gather**: Only the row-select pattern is implemented. Other gather dimension configurations will need extension.
- **Tensor representation**: Each tensor is `Vec<Value>` (one SSA value per element). Suitable for small simulation tensors but not large ML workloads.
- **No SIMD**: All operations are scalar Cranelift instructions.
