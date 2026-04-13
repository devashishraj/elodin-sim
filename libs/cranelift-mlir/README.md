# cranelift-mlir

StableHLO MLIR → Cranelift JIT compiler for Elodin simulations.

Parses StableHLO MLIR text (as emitted by `jax.jit().lower().compiler_ir(dialect="stablehlo")`)
and compiles it to native code via Cranelift JIT. Designed for small, CPU-bound physics
simulations where IREE overhead dominates.

## Architecture

```
StableHLO text → parser.rs → IR (ir.rs) → lower.rs → Cranelift IR → native fn pointer
```

- **`ir.rs`**: Internal IR types — `Module`, `FuncDef`, `Instruction`, `TensorType`, etc.
- **`parser.rs`**: Winnow-based parser converting StableHLO MLIR text to IR.
- **`lower.rs`**: Cranelift JIT compilation from IR to native function pointers.

### ABI

The compiled `main` function uses a pointer ABI:
```
extern "C" fn(inputs: *const *const u8, outputs: *mut *mut u8)
```
Each input/output pointer addresses the raw byte buffer for one tensor.

## Testing

### Per-op golden-value tests

Every supported StableHLO op has an individual test in `tests/ops/`. Each test:

1. Defines a minimal MLIR module containing only the op under test
2. Provides known input byte buffers
3. Provides expected output byte buffers (hand-computed or verified against JAX/NumPy)
4. Parses → compiles → executes → asserts outputs match

Tolerances: exact match for integers, `1e-10` relative tolerance for floats.

### Adding a new op

1. Add the `Instruction` variant in `src/ir.rs`
2. Add the parser arm in `src/parser.rs`
3. Add the lowering case in `src/lower.rs`
4. Create `tests/ops/test_<op_name>.rs` with MLIR snippet + golden inputs/outputs
5. Run `cargo test -p cranelift-mlir`

### End-to-end tests

`tests/ball_e2e.rs` parses and compiles the full ball simulation MLIR
(`testdata/ball.stablehlo.mlir`), verifying the entire pipeline end-to-end.

## Dumping StableHLO MLIR

To dump MLIR from a simulation for testing:
```bash
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DUMP_DIR=/tmp/cranelift_dump \
  python examples/ball/main.py run
```

## Running tests

```bash
cargo test -p cranelift-mlir                     # all tests
cargo test -p cranelift-mlir --test ball_e2e      # end-to-end only
cargo test -p cranelift-mlir -- ops::             # per-op tests only
```

## Op Coverage

| Op | Implemented | Tested |
|----|------------|--------|
| stablehlo.add | yes | yes |
| stablehlo.subtract | yes | yes |
| stablehlo.multiply | yes | yes |
| stablehlo.divide | yes | yes |
| stablehlo.negate | yes | yes |
| stablehlo.sqrt | yes | yes |
| stablehlo.maximum | yes | yes |
| stablehlo.compare | yes | yes |
| stablehlo.select | yes | - |
| stablehlo.constant | yes | yes |
| stablehlo.reshape | yes | yes |
| stablehlo.broadcast_in_dim | yes | yes |
| stablehlo.slice | yes | yes |
| stablehlo.concatenate | yes | yes |
| stablehlo.dot_general | yes | yes |
| stablehlo.reduce | yes | yes |
| stablehlo.convert | yes | yes |
| stablehlo.bitcast_convert | yes | - |
| stablehlo.iota | yes | yes |
| stablehlo.xor | yes | yes |
| stablehlo.or | yes | yes |
| stablehlo.and | yes | yes |
| stablehlo.shift_left | yes | yes |
| stablehlo.shift_right_logical | yes | yes |
| stablehlo.while | yes | yes |
| stablehlo.case | yes | yes |
| stablehlo.gather | yes | - |
| chlo.erf_inv | yes | yes |
| func.call | yes | yes |
