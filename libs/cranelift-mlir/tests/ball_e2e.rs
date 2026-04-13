use cranelift_mlir::parser::parse_module;

#[test]
fn parse_ball_mlir() {
    let mlir = include_str!("../testdata/ball.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse ball MLIR");

    assert!(
        module.main_func().is_some(),
        "module should have a main function"
    );

    let main = module.main_func().unwrap();
    assert!(main.is_public, "main should be public");
    assert_eq!(main.params.len(), 9, "ball main has 9 inputs");
    assert_eq!(main.result_types.len(), 9, "ball main has 9 outputs");
    assert!(!main.body.is_empty(), "main body should have instructions");

    assert!(
        module.functions.len() > 1,
        "ball module has multiple functions (main + inner helpers)"
    );

    eprintln!(
        "parsed {} functions, main has {} instructions",
        module.functions.len(),
        main.body.len(),
    );
}

#[test]
fn compile_ball_mlir() {
    let mlir = include_str!("../testdata/ball.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse ball MLIR");
    let compiled =
        cranelift_mlir::lower::compile_module(&module).expect("failed to compile ball MLIR");
    let fn_ptr = compiled.get_main_fn();
    assert!(
        !fn_ptr.is_null(),
        "main function pointer should not be null"
    );
}
