use cranelift_mlir::parser::parse_module;

#[test]
fn parse_rocket_mlir() {
    let mlir = include_str!("../testdata/rocket.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse");
    assert!(module.main_func().is_some(), "no main function found");
    assert!(
        module.functions.len() > 1,
        "expected multiple functions, got {}",
        module.functions.len()
    );
}

#[test]
fn compile_rocket_mlir() {
    let mlir = include_str!("../testdata/rocket.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse");
    let compiled = cranelift_mlir::lower::compile_module(&module).expect("failed to compile");
    assert!(!compiled.get_main_fn().is_null());
}
