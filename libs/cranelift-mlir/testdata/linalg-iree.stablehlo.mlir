module @module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3x2xf64>, %arg2: tensor<2xf64>, %arg3: tensor<2x2xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3x3xf64>, %arg6: tensor<5xf64>, %arg7: tensor<6xf64>, %arg8: tensor<6x6xf64>, %arg9: tensor<4xf64>, %arg10: tensor<4xi64>) -> (tensor<4xf64> {jax.result_info = "result[0]"}, tensor<3x3xf64> {jax.result_info = "result[1]"}, tensor<2xf64> {jax.result_info = "result[2]"}, tensor<3x2xf64> {jax.result_info = "result[3]"}, tensor<i64> {jax.result_info = "result[4]"}, tensor<2x2xf64> {jax.result_info = "result[5]"}, tensor<4xi64> {jax.result_info = "result[6]"}, tensor<3xf64> {jax.result_info = "result[7]"}, tensor<6xf64> {jax.result_info = "result[8]"}, tensor<5xf64> {jax.result_info = "result[9]"}, tensor<6x6xf64> {jax.result_info = "result[10]"}) {
    %0:3 = call @inner(%arg7, %arg8, %arg9) : (tensor<6xf64>, tensor<6x6xf64>, tensor<4xf64>) -> (tensor<6xf64>, tensor<6x6xf64>, tensor<4xf64>)
    %1:3 = call @inner_43(%arg4, %arg5, %arg6) : (tensor<3xf64>, tensor<3x3xf64>, tensor<5xf64>) -> (tensor<3xf64>, tensor<3x3xf64>, tensor<5xf64>)
    %2:2 = call @inner_146(%arg2, %arg3) : (tensor<2xf64>, tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2x2xf64>)
    %3 = call @inner_203(%arg1) : (tensor<3x2xf64>) -> tensor<3x2xf64>
    %4 = call @inner_219(%arg10) : (tensor<4xi64>) -> tensor<4xi64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %5 = stablehlo.add %arg0, %c : tensor<i64>
    return %0#2, %1#1, %2#0, %3, %5, %2#1, %4, %1#0, %0#0, %1#2, %0#1 : tensor<4xf64>, tensor<3x3xf64>, tensor<2xf64>, tensor<3x2xf64>, tensor<i64>, tensor<2x2xf64>, tensor<4xi64>, tensor<3xf64>, tensor<6xf64>, tensor<5xf64>, tensor<6x6xf64>
  }
  func.func private @inner(%arg0: tensor<6xf64>, %arg1: tensor<6x6xf64>, %arg2: tensor<4xf64>) -> (tensor<6xf64>, tensor<6x6xf64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.0083333333333333332, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.0083333333333333332, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.0083333333333333332], [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x6xf64>
    %cst_0 = stablehlo.constant dense<[[1.000000e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-02, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-02, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-02]]> : tensor<6x6xf64>
    %cst_1 = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<6x6xf64>
    %cst_2 = stablehlo.constant dense<[[1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e-01]]> : tensor<6x6xf64>
    %0 = stablehlo.dot_general %cst, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1 = stablehlo.dot_general %cst, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %2 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %3 = stablehlo.dot_general %1, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %4 = stablehlo.add %3, %cst_0 : tensor<6x6xf64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %cst_4 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %7 = stablehlo.multiply %6, %5 : tensor<6xf64>
    %8 = stablehlo.add %0, %7 : tensor<6xf64>
    %9 = stablehlo.dot_general %cst_1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %10 = stablehlo.subtract %8, %9 : tensor<6xf64>
    %11 = stablehlo.dot_general %cst_1, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %12 = stablehlo.transpose %cst_1, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %13 = stablehlo.dot_general %11, %12, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %14 = stablehlo.add %13, %cst_2 : tensor<6x6xf64>
    %15:3 = call @svd(%14) : (tensor<6x6xf64>) -> (tensor<6x6xf64>, tensor<6xf64>, tensor<6x6xf64>)
    %cst_5 = stablehlo.constant dense<9.9999999999999998E-13> : tensor<f64>
    %16 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %17 = stablehlo.compare  GT, %15#1, %16,  FLOAT : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xi1>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %18 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %19 = stablehlo.divide %18, %15#1 : tensor<6xf64>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %20 = call @_where(%17, %19, %cst_7) : (tensor<6xi1>, tensor<6xf64>, tensor<f64>) -> tensor<6xf64>
    %21 = stablehlo.transpose %15#2, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %22 = call @_diag(%20) : (tensor<6xf64>) -> tensor<6x6xf64>
    %23 = stablehlo.dot_general %21, %22, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %24 = stablehlo.transpose %15#0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %25 = stablehlo.dot_general %23, %24, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %26 = stablehlo.transpose %cst_1, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %27 = stablehlo.dot_general %4, %26, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %28 = stablehlo.dot_general %27, %25, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %29 = stablehlo.dot_general %28, %10, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %30 = stablehlo.add %0, %29 : tensor<6xf64>
    %31 = stablehlo.iota dim = 0 : tensor<6x6xi64>
    %32 = stablehlo.iota dim = 1 : tensor<6x6xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %33 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<6x6xi64>
    %34 = stablehlo.add %31, %33 : tensor<6x6xi64>
    %35 = stablehlo.compare  EQ, %34, %32,  SIGNED : (tensor<6x6xi64>, tensor<6x6xi64>) -> tensor<6x6xi1>
    %36 = stablehlo.convert %35 : (tensor<6x6xi1>) -> tensor<6x6xf64>
    %37 = stablehlo.dot_general %28, %cst_1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %38 = stablehlo.subtract %36, %37 : tensor<6x6xf64>
    %39 = stablehlo.dot_general %38, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %40 = stablehlo.transpose %38, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %41 = stablehlo.dot_general %39, %40, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %42 = stablehlo.dot_general %28, %cst_2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %43 = stablehlo.transpose %28, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %44 = stablehlo.dot_general %42, %43, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<6x6xf64>, tensor<6x6xf64>) -> tensor<6x6xf64>
    %45 = stablehlo.add %41, %44 : tensor<6x6xf64>
    %46:2 = call @eigh(%45) : (tensor<6x6xf64>) -> (tensor<6xf64>, tensor<6x6xf64>)
    %47 = call @norm(%10) : (tensor<6xf64>) -> tensor<f64>
    %cst_8 = stablehlo.constant dense<1.000000e+02> : tensor<f64>
    %48 = stablehlo.compare  LT, %47, %cst_8,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %49 = stablehlo.slice %46#0 [0:1] : (tensor<6xf64>) -> tensor<1xf64>
    %50 = stablehlo.reshape %49 : (tensor<1xf64>) -> tensor<f64>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %51 = stablehlo.compare  GT, %50, %cst_9,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %52 = stablehlo.and %48, %51 : tensor<i1>
    %53 = stablehlo.convert %52 : (tensor<i1>) -> tensor<i32>
    %54 = "stablehlo.case"(%53) ({
      stablehlo.return %30 : tensor<6xf64>
    }, {
      %65 = stablehlo.iota dim = 0 : tensor<6x6xi64>
      %66 = stablehlo.iota dim = 1 : tensor<6x6xi64>
      %c_12 = stablehlo.constant dense<0> : tensor<i64>
      %67 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<6x6xi64>
      %68 = stablehlo.add %65, %67 : tensor<6x6xi64>
      %69 = stablehlo.compare  EQ, %68, %66,  SIGNED : (tensor<6x6xi64>, tensor<6x6xi64>) -> tensor<6x6xi1>
      %70 = stablehlo.convert %69 : (tensor<6x6xi1>) -> tensor<6x6xf64>
      %cst_13 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
      %71 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<6x6xf64>
      %72 = stablehlo.multiply %71, %70 : tensor<6x6xf64>
      %73 = stablehlo.add %45, %72 : tensor<6x6xf64>
      %74 = func.call @solve(%73, %10) : (tensor<6x6xf64>, tensor<6xf64>) -> tensor<6xf64>
      %cst_14 = stablehlo.constant dense<9.9999999999999998E-13> : tensor<f64>
      %75 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f64>) -> tensor<6xf64>
      %76 = stablehlo.multiply %75, %74 : tensor<6xf64>
      %77 = stablehlo.add %30, %76 : tensor<6xf64>
      stablehlo.return %77 : tensor<6xf64>
    }) : (tensor<i32>) -> tensor<6xf64>
    %55 = stablehlo.slice %54 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %56 = call @norm_39(%55) : (tensor<3xf64>) -> tensor<f64>
    %57 = call @norm(%10) : (tensor<6xf64>) -> tensor<f64>
    %cst_10 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %58 = stablehlo.reduce(%46#0 init: %cst_10) applies stablehlo.maximum across dimensions = [0] : (tensor<6xf64>, tensor<f64>) -> tensor<f64>
    %cst_11 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %59 = stablehlo.reduce(%46#0 init: %cst_11) applies stablehlo.minimum across dimensions = [0] : (tensor<6xf64>, tensor<f64>) -> tensor<f64>
    %60 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %61 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.broadcast_in_dim %56, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %64 = stablehlo.concatenate %60, %61, %62, %63, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    return %54, %45, %64 : tensor<6xf64>, tensor<6x6xf64>, tensor<4xf64>
  }
  func.func private @svd(%arg0: tensor<6x6xf64>) -> (tensor<6x6xf64>, tensor<6xf64>, tensor<6x6xf64>) {
    %0:5 = stablehlo.custom_call @lapack_dgesdd_ffi(%arg0) {backend_config = "", mhlo.backend_config = {mode = 65 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], [n, o], [p, q], []) {i=6, j=6, k=6, l=6, m=6, n=6, o=6, p=6, q=6}, custom>} : (tensor<6x6xf64>) -> (tensor<6x6xf64>, tensor<6xf64>, tensor<6x6xf64>, tensor<6x6xf64>, tensor<i32>)
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %2 = stablehlo.compare  EQ, %0#4, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %5 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xi1>) -> tensor<6xi1>
    %6 = stablehlo.select %5, %0#1, %4 : tensor<6xi1>, tensor<6xf64>
    %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<6x6xf64>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<6x6xi1>
    %10 = stablehlo.select %9, %0#2, %8 : tensor<6x6xi1>, tensor<6x6xf64>
    %11 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst_1 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<6x6xf64>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<6x6xi1>
    %14 = stablehlo.select %13, %0#3, %12 : tensor<6x6xi1>, tensor<6x6xf64>
    return %10, %6, %14 : tensor<6x6xf64>, tensor<6xf64>, tensor<6x6xf64>
  }
  func.func private @_where(%arg0: tensor<6xi1>, %arg1: tensor<6xf64>, %arg2: tensor<f64>) -> tensor<6xf64> {
    %0 = stablehlo.convert %arg2 : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<6xi1>, tensor<6xf64>
    return %2 : tensor<6xf64>
  }
  func.func private @_diag(%arg0: tensor<6xf64>) -> tensor<6x6xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.pad %arg0, %cst, low = [0], high = [0], interior = [0] : (tensor<6xf64>, tensor<f64>) -> tensor<6xf64>
    %1 = stablehlo.iota dim = 0 : tensor<6x6xi64>
    %2 = stablehlo.iota dim = 1 : tensor<6x6xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<6x6xi64>
    %4 = stablehlo.add %1, %3 : tensor<6x6xi64>
    %5 = stablehlo.compare  EQ, %4, %2,  SIGNED : (tensor<6x6xi64>, tensor<6x6xi64>) -> tensor<6x6xi1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %7 = call @_where_5(%5, %0, %6) : (tensor<6x6xi1>, tensor<6xf64>, tensor<6xf64>) -> tensor<6x6xf64>
    return %7 : tensor<6x6xf64>
  }
  func.func private @_where_5(%arg0: tensor<6x6xi1>, %arg1: tensor<6xf64>, %arg2: tensor<6xf64>) -> tensor<6x6xf64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<6xf64>) -> tensor<6x6xf64>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<6xf64>) -> tensor<6x6xf64>
    %2 = stablehlo.select %arg0, %0, %1 : tensor<6x6xi1>, tensor<6x6xf64>
    return %2 : tensor<6x6xf64>
  }
  func.func private @eigh(%arg0: tensor<6x6xf64>) -> (tensor<6xf64>, tensor<6x6xf64>) {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %1 = stablehlo.add %arg0, %0 : tensor<6x6xf64>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<6x6xf64>
    %3 = stablehlo.divide %1, %2 : tensor<6x6xf64>
    %4:3 = stablehlo.custom_call @lapack_dsyevd_ffi(%3) {backend_config = "", mhlo.backend_config = {mode = 86 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=6, j=6, k=6, l=6, m=6}, custom>} : (tensor<6x6xf64>) -> (tensor<6x6xf64>, tensor<6xf64>, tensor<i32>)
    %c = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %6 = stablehlo.compare  EQ, %4#2, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<6x6xf64>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<6x6xi1>
    %10 = stablehlo.select %9, %4#0, %8 : tensor<6x6xi1>, tensor<6x6xf64>
    %11 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %cst_1 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %13 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi1>) -> tensor<6xi1>
    %14 = stablehlo.select %13, %4#1, %12 : tensor<6xi1>, tensor<6xf64>
    return %14, %10 : tensor<6xf64>, tensor<6x6xf64>
  }
  func.func private @norm(%arg0: tensor<6xf64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<6xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.sqrt %1 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @solve(%arg0: tensor<6x6xf64>, %arg1: tensor<6xf64>) -> tensor<6xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=6, j=6, k=6, l=6, m=6}, custom>} : (tensor<6x6xf64>) -> (tensor<6x6xf64>, tensor<6xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<6xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<6x6xf64>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<6x6xi1>
    %8 = stablehlo.select %7, %0#0, %6 : tensor<6x6xi1>, tensor<6x6xf64>
    %9 = stablehlo.iota dim = 0 : tensor<6xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %10:4 = stablehlo.while(%iterArg = %2, %iterArg_3 = %c_2, %iterArg_4 = %c_1, %iterArg_5 = %9) : tensor<6xi32>, tensor<i64>, tensor<i64>, tensor<6xi32>
    cond {
      %c_6 = stablehlo.constant dense<6> : tensor<i64>
      %12 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %12 : tensor<i1>
    } do {
      %12:2 = func.call @closed_call(%iterArg, %iterArg_4, %iterArg_5) : (tensor<6xi32>, tensor<i64>, tensor<6xi32>) -> (tensor<i64>, tensor<6xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %13 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %13, %12#0, %12#1 : tensor<6xi32>, tensor<i64>, tensor<i64>, tensor<6xi32>
    }
    %11 = call @_lu_solve(%8, %10#3, %arg1) : (tensor<6x6xf64>, tensor<6xi32>, tensor<6xf64>) -> tensor<6xf64>
    return %11 : tensor<6xf64>
  }
  func.func private @closed_call(%arg0: tensor<6xi32>, %arg1: tensor<i64>, %arg2: tensor<6xi32>) -> (tensor<i64>, tensor<6xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg1, %c : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare  LT, %arg1, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = stablehlo.convert %arg1 : tensor<i64>
    %c_1 = stablehlo.constant dense<6> : tensor<i64>
    %3 = stablehlo.add %2, %c_1 : tensor<i64>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<i1>, tensor<i64>
    %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<6xi32>, tensor<i64>) -> tensor<1xi32>
    %6 = stablehlo.reshape %5 : (tensor<1xi32>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.compare  LT, %arg1, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8 = stablehlo.convert %arg1 : tensor<i64>
    %c_3 = stablehlo.constant dense<6> : tensor<i64>
    %9 = stablehlo.add %8, %c_3 : tensor<i64>
    %10 = stablehlo.select %7, %9, %arg1 : tensor<i1>, tensor<i64>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<6xi32>, tensor<i64>) -> tensor<1xi32>
    %12 = stablehlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.compare  LT, %6, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<6> : tensor<i32>
    %14 = stablehlo.add %6, %c_5 : tensor<i32>
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32>
    %16 = stablehlo.dynamic_slice %arg2, %15, sizes = [1] : (tensor<6xi32>, tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %18 = stablehlo.compare  LT, %arg1, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_7 = stablehlo.constant dense<6> : tensor<i64>
    %19 = stablehlo.add %arg1, %c_7 : tensor<i64>
    %20 = stablehlo.select %18, %19, %arg1 : tensor<i1>, tensor<i64>
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = "stablehlo.scatter"(%arg2, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<6xi32>, tensor<1xi32>, tensor<i32>) -> tensor<6xi32>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %24 = stablehlo.compare  LT, %6, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_9 = stablehlo.constant dense<6> : tensor<i32>
    %25 = stablehlo.add %6, %c_9 : tensor<i32>
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<6xi32>, tensor<1xi32>, tensor<i32>) -> tensor<6xi32>
    return %0, %28 : tensor<i64>, tensor<6xi32>
  }
  func.func private @_lu_solve(%arg0: tensor<6x6xf64>, %arg1: tensor<6xi32>, %arg2: tensor<6xf64>) -> tensor<6xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<6xf64>) -> tensor<6x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<6xi32>, tensor<6xi32>) -> tensor<6xi1>
    %c_0 = stablehlo.constant dense<6> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<6xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<6xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<6xi1>, tensor<6xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<6xi32>) -> tensor<6x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<6x1xf64>, tensor<6x1xi32>) -> tensor<6x1xf64>
    %8 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %7) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=6, j=6, k=6, l=1, m=6, n=1}, custom>} : (tensor<6x6xf64>, tensor<6x1xf64>) -> tensor<6x1xf64>
    %9 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %8) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=6, j=6, k=6, l=1, m=6, n=1}, custom>} : (tensor<6x6xf64>, tensor<6x1xf64>) -> tensor<6x1xf64>
    %10 = stablehlo.slice %9 [0:6, 0:1] : (tensor<6x1xf64>) -> tensor<6x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<6x1xf64>) -> tensor<6xf64>
    return %11 : tensor<6xf64>
  }
  func.func private @norm_39(%arg0: tensor<3xf64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<3xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.sqrt %1 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @inner_43(%arg0: tensor<3xf64>, %arg1: tensor<3x3xf64>, %arg2: tensor<5xf64>) -> (tensor<3xf64>, tensor<3x3xf64>, tensor<5xf64>) {
    %cst = stablehlo.constant dense<[[1.000000e+00, 0.0083333333333333332, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.0083333333333333332], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    %cst_0 = stablehlo.constant dense<[[1.000000e-02, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e-02, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e-02]]> : tensor<3x3xf64>
    %cst_1 = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    %cst_2 = stablehlo.constant dense<[[1.000000e-01, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e-01]]> : tensor<3x3xf64>
    %0 = stablehlo.dot_general %cst, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1 = stablehlo.dot_general %cst, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %2 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %3 = stablehlo.dot_general %1, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %4 = stablehlo.add %3, %cst_0 : tensor<3x3xf64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %cst_4 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %7 = stablehlo.multiply %6, %5 : tensor<3xf64>
    %8 = stablehlo.add %0, %7 : tensor<3xf64>
    %9 = stablehlo.dot_general %cst_1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %10 = stablehlo.subtract %8, %9 : tensor<3xf64>
    %11 = stablehlo.dot_general %cst_1, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %12 = stablehlo.transpose %cst_1, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %13 = stablehlo.dot_general %11, %12, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %14 = stablehlo.add %13, %cst_2 : tensor<3x3xf64>
    %15 = call @cholesky(%14) : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %16 = stablehlo.transpose %15, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %17 = stablehlo.dot_general %15, %16, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %18 = stablehlo.transpose %14, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %19 = stablehlo.transpose %cst_1, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %20 = stablehlo.dot_general %4, %19, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %21 = stablehlo.transpose %20, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %22 = call @solve_59(%18, %21) : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %23 = stablehlo.transpose %22, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %24 = stablehlo.dot_general %23, %10, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %25 = stablehlo.add %0, %24 : tensor<3xf64>
    %26 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    %27 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %28 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %29 = stablehlo.add %26, %28 : tensor<3x3xi64>
    %30 = stablehlo.compare  EQ, %29, %27,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
    %31 = stablehlo.convert %30 : (tensor<3x3xi1>) -> tensor<3x3xf64>
    %32 = stablehlo.dot_general %23, %cst_1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %33 = stablehlo.subtract %31, %32 : tensor<3x3xf64>
    %34 = stablehlo.dot_general %33, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %35 = stablehlo.transpose %33, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %36 = stablehlo.dot_general %34, %35, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %37 = stablehlo.dot_general %23, %cst_2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %38 = stablehlo.transpose %23, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %39 = stablehlo.dot_general %37, %38, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %40 = stablehlo.add %36, %39 : tensor<3x3xf64>
    %41:2 = call @qr(%40) : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3x3xf64>)
    %42 = stablehlo.dot_general %41#0, %41#1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %43 = call @det(%14) : (tensor<3x3xf64>) -> tensor<f64>
    %44:2 = call @slogdet(%14) : (tensor<3x3xf64>) -> (tensor<f64>, tensor<f64>)
    %45 = call @solve_126(%14, %10) : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %cst_5 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %46 = stablehlo.log %cst_5 : tensor<f64>
    %cst_6 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %47 = stablehlo.multiply %cst_6, %46 : tensor<f64>
    %48 = stablehlo.convert %47 : tensor<f64>
    %49 = stablehlo.add %48, %44#1 : tensor<f64>
    %50 = stablehlo.dot_general %10, %45, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    %51 = stablehlo.add %49, %50 : tensor<f64>
    %cst_7 = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
    %52 = stablehlo.multiply %cst_7, %51 : tensor<f64>
    %53 = stablehlo.slice %arg0 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %54 = stablehlo.reshape %53 : (tensor<1xf64>) -> tensor<f64>
    %cst_8 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %55 = stablehlo.compare  GT, %54, %cst_8,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %56 = stablehlo.slice %arg0 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %57 = stablehlo.reshape %56 : (tensor<1xf64>) -> tensor<f64>
    %cst_9 = stablehlo.constant dense<-1.000000e+03> : tensor<f64>
    %58 = stablehlo.compare  GT, %57, %cst_9,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %59 = stablehlo.and %55, %58 : tensor<i1>
    %60 = call @norm_39(%25) : (tensor<3xf64>) -> tensor<f64>
    %cst_10 = stablehlo.constant dense<1.000000e+08> : tensor<f64>
    %61 = stablehlo.compare  LT, %60, %cst_10,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %62 = stablehlo.and %59, %61 : tensor<i1>
    %63 = stablehlo.convert %62 : (tensor<i1>) -> tensor<i32>
    %64 = "stablehlo.case"(%63) ({
      stablehlo.return %25 : tensor<3xf64>
    }, {
      %75 = stablehlo.iota dim = 0 : tensor<3x3xi64>
      %76 = stablehlo.iota dim = 1 : tensor<3x3xi64>
      %c_11 = stablehlo.constant dense<0> : tensor<i64>
      %77 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
      %78 = stablehlo.add %75, %77 : tensor<3x3xi64>
      %79 = stablehlo.compare  EQ, %78, %76,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
      %80 = stablehlo.convert %79 : (tensor<3x3xi1>) -> tensor<3x3xf64>
      %cst_12 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
      %81 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
      %82 = stablehlo.multiply %81, %80 : tensor<3x3xf64>
      %83 = stablehlo.add %14, %82 : tensor<3x3xf64>
      %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %84 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst_14 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
      %85 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %86 = stablehlo.multiply %85, %84 : tensor<3xf64>
      %87 = stablehlo.add %10, %86 : tensor<3xf64>
      %88 = func.call @solve_126(%83, %87) : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
      %89 = stablehlo.add %88, %25 : tensor<3xf64>
      %90 = stablehlo.slice %89 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %91 = stablehlo.reshape %90 : (tensor<1xf64>) -> tensor<f64>
      %92 = stablehlo.slice %89 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
      %cst_15 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %94 = stablehlo.add %93, %cst_15 : tensor<f64>
      %95 = stablehlo.atan2 %91, %94 : tensor<f64>
      %96 = stablehlo.slice %89 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
      %98 = stablehlo.slice %89 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
      %100 = stablehlo.slice %89 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %101 = stablehlo.reshape %100 : (tensor<1xf64>) -> tensor<f64>
      %102 = stablehlo.multiply %99, %101 : tensor<f64>
      %103 = stablehlo.slice %89 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
      %105 = stablehlo.slice %89 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %106 = stablehlo.reshape %105 : (tensor<1xf64>) -> tensor<f64>
      %107 = stablehlo.multiply %104, %106 : tensor<f64>
      %108 = stablehlo.add %102, %107 : tensor<f64>
      %109 = stablehlo.sqrt %108 : tensor<f64>
      %cst_16 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %110 = stablehlo.add %109, %cst_16 : tensor<f64>
      %111 = stablehlo.atan2 %97, %110 : tensor<f64>
      %112 = stablehlo.cosine %95 : tensor<f64>
      %113 = stablehlo.sine %95 : tensor<f64>
      %114 = stablehlo.cosine %111 : tensor<f64>
      %115 = stablehlo.sine %111 : tensor<f64>
      %116 = stablehlo.slice %89 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %117 = stablehlo.reshape %116 : (tensor<1xf64>) -> tensor<f64>
      %118 = stablehlo.multiply %117, %112 : tensor<f64>
      %119 = stablehlo.slice %89 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %120 = stablehlo.reshape %119 : (tensor<1xf64>) -> tensor<f64>
      %121 = stablehlo.multiply %120, %113 : tensor<f64>
      %122 = stablehlo.subtract %118, %121 : tensor<f64>
      %cst_17 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %123 = stablehlo.multiply %cst_17, %115 : tensor<f64>
      %124 = stablehlo.add %122, %123 : tensor<f64>
      %125 = stablehlo.slice %89 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %126 = stablehlo.reshape %125 : (tensor<1xf64>) -> tensor<f64>
      %127 = stablehlo.multiply %126, %113 : tensor<f64>
      %128 = stablehlo.slice %89 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %129 = stablehlo.reshape %128 : (tensor<1xf64>) -> tensor<f64>
      %130 = stablehlo.multiply %129, %112 : tensor<f64>
      %131 = stablehlo.add %127, %130 : tensor<f64>
      %cst_18 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %132 = stablehlo.multiply %cst_18, %114 : tensor<f64>
      %133 = stablehlo.add %131, %132 : tensor<f64>
      %134 = stablehlo.slice %89 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %135 = stablehlo.reshape %134 : (tensor<1xf64>) -> tensor<f64>
      %136 = stablehlo.multiply %135, %114 : tensor<f64>
      %137 = stablehlo.multiply %113, %112 : tensor<f64>
      %cst_19 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %138 = stablehlo.multiply %cst_19, %137 : tensor<f64>
      %139 = stablehlo.add %136, %138 : tensor<f64>
      %140 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %141 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %142 = stablehlo.broadcast_in_dim %139, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %143 = stablehlo.concatenate %140, %141, %142, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %144 = stablehlo.slice %143 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
      %146 = stablehlo.slice %143 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %147 = stablehlo.reshape %146 : (tensor<1xf64>) -> tensor<f64>
      %cst_20 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %148 = stablehlo.add %147, %cst_20 : tensor<f64>
      %149 = stablehlo.atan2 %145, %148 : tensor<f64>
      %150 = stablehlo.slice %143 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
      %152 = stablehlo.slice %143 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %153 = stablehlo.reshape %152 : (tensor<1xf64>) -> tensor<f64>
      %154 = stablehlo.slice %143 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %155 = stablehlo.reshape %154 : (tensor<1xf64>) -> tensor<f64>
      %156 = stablehlo.multiply %153, %155 : tensor<f64>
      %157 = stablehlo.slice %143 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %158 = stablehlo.reshape %157 : (tensor<1xf64>) -> tensor<f64>
      %159 = stablehlo.slice %143 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %160 = stablehlo.reshape %159 : (tensor<1xf64>) -> tensor<f64>
      %161 = stablehlo.multiply %158, %160 : tensor<f64>
      %162 = stablehlo.add %156, %161 : tensor<f64>
      %163 = stablehlo.sqrt %162 : tensor<f64>
      %cst_21 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %164 = stablehlo.add %163, %cst_21 : tensor<f64>
      %165 = stablehlo.atan2 %151, %164 : tensor<f64>
      %166 = stablehlo.cosine %149 : tensor<f64>
      %167 = stablehlo.sine %149 : tensor<f64>
      %168 = stablehlo.cosine %165 : tensor<f64>
      %169 = stablehlo.sine %165 : tensor<f64>
      %170 = stablehlo.slice %143 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %171 = stablehlo.reshape %170 : (tensor<1xf64>) -> tensor<f64>
      %172 = stablehlo.multiply %171, %166 : tensor<f64>
      %173 = stablehlo.slice %143 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %174 = stablehlo.reshape %173 : (tensor<1xf64>) -> tensor<f64>
      %175 = stablehlo.multiply %174, %167 : tensor<f64>
      %176 = stablehlo.subtract %172, %175 : tensor<f64>
      %cst_22 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %177 = stablehlo.multiply %cst_22, %169 : tensor<f64>
      %178 = stablehlo.add %176, %177 : tensor<f64>
      %179 = stablehlo.slice %143 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %180 = stablehlo.reshape %179 : (tensor<1xf64>) -> tensor<f64>
      %181 = stablehlo.multiply %180, %167 : tensor<f64>
      %182 = stablehlo.slice %143 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %183 = stablehlo.reshape %182 : (tensor<1xf64>) -> tensor<f64>
      %184 = stablehlo.multiply %183, %166 : tensor<f64>
      %185 = stablehlo.add %181, %184 : tensor<f64>
      %cst_23 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %186 = stablehlo.multiply %cst_23, %168 : tensor<f64>
      %187 = stablehlo.add %185, %186 : tensor<f64>
      %188 = stablehlo.slice %143 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %189 = stablehlo.reshape %188 : (tensor<1xf64>) -> tensor<f64>
      %190 = stablehlo.multiply %189, %168 : tensor<f64>
      %191 = stablehlo.multiply %167, %166 : tensor<f64>
      %cst_24 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %192 = stablehlo.multiply %cst_24, %191 : tensor<f64>
      %193 = stablehlo.add %190, %192 : tensor<f64>
      %194 = stablehlo.broadcast_in_dim %178, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %195 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %196 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %197 = stablehlo.concatenate %194, %195, %196, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %198 = stablehlo.slice %197 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %199 = stablehlo.reshape %198 : (tensor<1xf64>) -> tensor<f64>
      %200 = stablehlo.slice %197 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %201 = stablehlo.reshape %200 : (tensor<1xf64>) -> tensor<f64>
      %cst_25 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %202 = stablehlo.add %201, %cst_25 : tensor<f64>
      %203 = stablehlo.atan2 %199, %202 : tensor<f64>
      %204 = stablehlo.slice %197 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %205 = stablehlo.reshape %204 : (tensor<1xf64>) -> tensor<f64>
      %206 = stablehlo.slice %197 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %207 = stablehlo.reshape %206 : (tensor<1xf64>) -> tensor<f64>
      %208 = stablehlo.slice %197 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %209 = stablehlo.reshape %208 : (tensor<1xf64>) -> tensor<f64>
      %210 = stablehlo.multiply %207, %209 : tensor<f64>
      %211 = stablehlo.slice %197 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %212 = stablehlo.reshape %211 : (tensor<1xf64>) -> tensor<f64>
      %213 = stablehlo.slice %197 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %214 = stablehlo.reshape %213 : (tensor<1xf64>) -> tensor<f64>
      %215 = stablehlo.multiply %212, %214 : tensor<f64>
      %216 = stablehlo.add %210, %215 : tensor<f64>
      %217 = stablehlo.sqrt %216 : tensor<f64>
      %cst_26 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %218 = stablehlo.add %217, %cst_26 : tensor<f64>
      %219 = stablehlo.atan2 %205, %218 : tensor<f64>
      %220 = stablehlo.cosine %203 : tensor<f64>
      %221 = stablehlo.sine %203 : tensor<f64>
      %222 = stablehlo.cosine %219 : tensor<f64>
      %223 = stablehlo.sine %219 : tensor<f64>
      %224 = stablehlo.slice %197 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %225 = stablehlo.reshape %224 : (tensor<1xf64>) -> tensor<f64>
      %226 = stablehlo.multiply %225, %220 : tensor<f64>
      %227 = stablehlo.slice %197 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %228 = stablehlo.reshape %227 : (tensor<1xf64>) -> tensor<f64>
      %229 = stablehlo.multiply %228, %221 : tensor<f64>
      %230 = stablehlo.subtract %226, %229 : tensor<f64>
      %cst_27 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %231 = stablehlo.multiply %cst_27, %223 : tensor<f64>
      %232 = stablehlo.add %230, %231 : tensor<f64>
      %233 = stablehlo.slice %197 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %234 = stablehlo.reshape %233 : (tensor<1xf64>) -> tensor<f64>
      %235 = stablehlo.multiply %234, %221 : tensor<f64>
      %236 = stablehlo.slice %197 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %237 = stablehlo.reshape %236 : (tensor<1xf64>) -> tensor<f64>
      %238 = stablehlo.multiply %237, %220 : tensor<f64>
      %239 = stablehlo.add %235, %238 : tensor<f64>
      %cst_28 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %240 = stablehlo.multiply %cst_28, %222 : tensor<f64>
      %241 = stablehlo.add %239, %240 : tensor<f64>
      %242 = stablehlo.slice %197 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %243 = stablehlo.reshape %242 : (tensor<1xf64>) -> tensor<f64>
      %244 = stablehlo.multiply %243, %222 : tensor<f64>
      %245 = stablehlo.multiply %221, %220 : tensor<f64>
      %cst_29 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %246 = stablehlo.multiply %cst_29, %245 : tensor<f64>
      %247 = stablehlo.add %244, %246 : tensor<f64>
      %248 = stablehlo.broadcast_in_dim %232, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %249 = stablehlo.broadcast_in_dim %241, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %250 = stablehlo.broadcast_in_dim %247, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %251 = stablehlo.concatenate %248, %249, %250, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %252 = stablehlo.slice %251 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %253 = stablehlo.reshape %252 : (tensor<1xf64>) -> tensor<f64>
      %254 = stablehlo.slice %251 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %255 = stablehlo.reshape %254 : (tensor<1xf64>) -> tensor<f64>
      %cst_30 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %256 = stablehlo.add %255, %cst_30 : tensor<f64>
      %257 = stablehlo.atan2 %253, %256 : tensor<f64>
      %258 = stablehlo.slice %251 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %259 = stablehlo.reshape %258 : (tensor<1xf64>) -> tensor<f64>
      %260 = stablehlo.slice %251 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %261 = stablehlo.reshape %260 : (tensor<1xf64>) -> tensor<f64>
      %262 = stablehlo.slice %251 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %263 = stablehlo.reshape %262 : (tensor<1xf64>) -> tensor<f64>
      %264 = stablehlo.multiply %261, %263 : tensor<f64>
      %265 = stablehlo.slice %251 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %266 = stablehlo.reshape %265 : (tensor<1xf64>) -> tensor<f64>
      %267 = stablehlo.slice %251 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %268 = stablehlo.reshape %267 : (tensor<1xf64>) -> tensor<f64>
      %269 = stablehlo.multiply %266, %268 : tensor<f64>
      %270 = stablehlo.add %264, %269 : tensor<f64>
      %271 = stablehlo.sqrt %270 : tensor<f64>
      %cst_31 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %272 = stablehlo.add %271, %cst_31 : tensor<f64>
      %273 = stablehlo.atan2 %259, %272 : tensor<f64>
      %274 = stablehlo.cosine %257 : tensor<f64>
      %275 = stablehlo.sine %257 : tensor<f64>
      %276 = stablehlo.cosine %273 : tensor<f64>
      %277 = stablehlo.sine %273 : tensor<f64>
      %278 = stablehlo.slice %251 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %279 = stablehlo.reshape %278 : (tensor<1xf64>) -> tensor<f64>
      %280 = stablehlo.multiply %279, %274 : tensor<f64>
      %281 = stablehlo.slice %251 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %282 = stablehlo.reshape %281 : (tensor<1xf64>) -> tensor<f64>
      %283 = stablehlo.multiply %282, %275 : tensor<f64>
      %284 = stablehlo.subtract %280, %283 : tensor<f64>
      %cst_32 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %285 = stablehlo.multiply %cst_32, %277 : tensor<f64>
      %286 = stablehlo.add %284, %285 : tensor<f64>
      %287 = stablehlo.slice %251 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %288 = stablehlo.reshape %287 : (tensor<1xf64>) -> tensor<f64>
      %289 = stablehlo.multiply %288, %275 : tensor<f64>
      %290 = stablehlo.slice %251 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %291 = stablehlo.reshape %290 : (tensor<1xf64>) -> tensor<f64>
      %292 = stablehlo.multiply %291, %274 : tensor<f64>
      %293 = stablehlo.add %289, %292 : tensor<f64>
      %cst_33 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %294 = stablehlo.multiply %cst_33, %276 : tensor<f64>
      %295 = stablehlo.add %293, %294 : tensor<f64>
      %296 = stablehlo.slice %251 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %297 = stablehlo.reshape %296 : (tensor<1xf64>) -> tensor<f64>
      %298 = stablehlo.multiply %297, %276 : tensor<f64>
      %299 = stablehlo.multiply %275, %274 : tensor<f64>
      %cst_34 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %300 = stablehlo.multiply %cst_34, %299 : tensor<f64>
      %301 = stablehlo.add %298, %300 : tensor<f64>
      %302 = stablehlo.broadcast_in_dim %286, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %303 = stablehlo.broadcast_in_dim %295, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %304 = stablehlo.broadcast_in_dim %301, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %305 = stablehlo.concatenate %302, %303, %304, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %306 = stablehlo.slice %305 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %307 = stablehlo.reshape %306 : (tensor<1xf64>) -> tensor<f64>
      %308 = stablehlo.slice %305 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %309 = stablehlo.reshape %308 : (tensor<1xf64>) -> tensor<f64>
      %cst_35 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %310 = stablehlo.add %309, %cst_35 : tensor<f64>
      %311 = stablehlo.atan2 %307, %310 : tensor<f64>
      %312 = stablehlo.slice %305 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %313 = stablehlo.reshape %312 : (tensor<1xf64>) -> tensor<f64>
      %314 = stablehlo.slice %305 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %315 = stablehlo.reshape %314 : (tensor<1xf64>) -> tensor<f64>
      %316 = stablehlo.slice %305 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %317 = stablehlo.reshape %316 : (tensor<1xf64>) -> tensor<f64>
      %318 = stablehlo.multiply %315, %317 : tensor<f64>
      %319 = stablehlo.slice %305 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %320 = stablehlo.reshape %319 : (tensor<1xf64>) -> tensor<f64>
      %321 = stablehlo.slice %305 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %322 = stablehlo.reshape %321 : (tensor<1xf64>) -> tensor<f64>
      %323 = stablehlo.multiply %320, %322 : tensor<f64>
      %324 = stablehlo.add %318, %323 : tensor<f64>
      %325 = stablehlo.sqrt %324 : tensor<f64>
      %cst_36 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %326 = stablehlo.add %325, %cst_36 : tensor<f64>
      %327 = stablehlo.atan2 %313, %326 : tensor<f64>
      %328 = stablehlo.cosine %311 : tensor<f64>
      %329 = stablehlo.sine %311 : tensor<f64>
      %330 = stablehlo.cosine %327 : tensor<f64>
      %331 = stablehlo.sine %327 : tensor<f64>
      %332 = stablehlo.slice %305 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %333 = stablehlo.reshape %332 : (tensor<1xf64>) -> tensor<f64>
      %334 = stablehlo.multiply %333, %328 : tensor<f64>
      %335 = stablehlo.slice %305 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %336 = stablehlo.reshape %335 : (tensor<1xf64>) -> tensor<f64>
      %337 = stablehlo.multiply %336, %329 : tensor<f64>
      %338 = stablehlo.subtract %334, %337 : tensor<f64>
      %cst_37 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %339 = stablehlo.multiply %cst_37, %331 : tensor<f64>
      %340 = stablehlo.add %338, %339 : tensor<f64>
      %341 = stablehlo.slice %305 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %342 = stablehlo.reshape %341 : (tensor<1xf64>) -> tensor<f64>
      %343 = stablehlo.multiply %342, %329 : tensor<f64>
      %344 = stablehlo.slice %305 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %345 = stablehlo.reshape %344 : (tensor<1xf64>) -> tensor<f64>
      %346 = stablehlo.multiply %345, %328 : tensor<f64>
      %347 = stablehlo.add %343, %346 : tensor<f64>
      %cst_38 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %348 = stablehlo.multiply %cst_38, %330 : tensor<f64>
      %349 = stablehlo.add %347, %348 : tensor<f64>
      %350 = stablehlo.slice %305 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %351 = stablehlo.reshape %350 : (tensor<1xf64>) -> tensor<f64>
      %352 = stablehlo.multiply %351, %330 : tensor<f64>
      %353 = stablehlo.multiply %329, %328 : tensor<f64>
      %cst_39 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %354 = stablehlo.multiply %cst_39, %353 : tensor<f64>
      %355 = stablehlo.add %352, %354 : tensor<f64>
      %356 = stablehlo.broadcast_in_dim %340, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %357 = stablehlo.broadcast_in_dim %349, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %358 = stablehlo.broadcast_in_dim %355, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %359 = stablehlo.concatenate %356, %357, %358, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %360 = stablehlo.slice %359 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %361 = stablehlo.reshape %360 : (tensor<1xf64>) -> tensor<f64>
      %362 = stablehlo.slice %359 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %363 = stablehlo.reshape %362 : (tensor<1xf64>) -> tensor<f64>
      %cst_40 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %364 = stablehlo.add %363, %cst_40 : tensor<f64>
      %365 = stablehlo.atan2 %361, %364 : tensor<f64>
      %366 = stablehlo.slice %359 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %367 = stablehlo.reshape %366 : (tensor<1xf64>) -> tensor<f64>
      %368 = stablehlo.slice %359 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %369 = stablehlo.reshape %368 : (tensor<1xf64>) -> tensor<f64>
      %370 = stablehlo.slice %359 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %371 = stablehlo.reshape %370 : (tensor<1xf64>) -> tensor<f64>
      %372 = stablehlo.multiply %369, %371 : tensor<f64>
      %373 = stablehlo.slice %359 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %374 = stablehlo.reshape %373 : (tensor<1xf64>) -> tensor<f64>
      %375 = stablehlo.slice %359 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %376 = stablehlo.reshape %375 : (tensor<1xf64>) -> tensor<f64>
      %377 = stablehlo.multiply %374, %376 : tensor<f64>
      %378 = stablehlo.add %372, %377 : tensor<f64>
      %379 = stablehlo.sqrt %378 : tensor<f64>
      %cst_41 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %380 = stablehlo.add %379, %cst_41 : tensor<f64>
      %381 = stablehlo.atan2 %367, %380 : tensor<f64>
      %382 = stablehlo.cosine %365 : tensor<f64>
      %383 = stablehlo.sine %365 : tensor<f64>
      %384 = stablehlo.cosine %381 : tensor<f64>
      %385 = stablehlo.sine %381 : tensor<f64>
      %386 = stablehlo.slice %359 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %387 = stablehlo.reshape %386 : (tensor<1xf64>) -> tensor<f64>
      %388 = stablehlo.multiply %387, %382 : tensor<f64>
      %389 = stablehlo.slice %359 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %390 = stablehlo.reshape %389 : (tensor<1xf64>) -> tensor<f64>
      %391 = stablehlo.multiply %390, %383 : tensor<f64>
      %392 = stablehlo.subtract %388, %391 : tensor<f64>
      %cst_42 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %393 = stablehlo.multiply %cst_42, %385 : tensor<f64>
      %394 = stablehlo.add %392, %393 : tensor<f64>
      %395 = stablehlo.slice %359 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %396 = stablehlo.reshape %395 : (tensor<1xf64>) -> tensor<f64>
      %397 = stablehlo.multiply %396, %383 : tensor<f64>
      %398 = stablehlo.slice %359 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %399 = stablehlo.reshape %398 : (tensor<1xf64>) -> tensor<f64>
      %400 = stablehlo.multiply %399, %382 : tensor<f64>
      %401 = stablehlo.add %397, %400 : tensor<f64>
      %cst_43 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %402 = stablehlo.multiply %cst_43, %384 : tensor<f64>
      %403 = stablehlo.add %401, %402 : tensor<f64>
      %404 = stablehlo.slice %359 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %405 = stablehlo.reshape %404 : (tensor<1xf64>) -> tensor<f64>
      %406 = stablehlo.multiply %405, %384 : tensor<f64>
      %407 = stablehlo.multiply %383, %382 : tensor<f64>
      %cst_44 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %408 = stablehlo.multiply %cst_44, %407 : tensor<f64>
      %409 = stablehlo.add %406, %408 : tensor<f64>
      %410 = stablehlo.broadcast_in_dim %394, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %411 = stablehlo.broadcast_in_dim %403, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %412 = stablehlo.broadcast_in_dim %409, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %413 = stablehlo.concatenate %410, %411, %412, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %414 = stablehlo.slice %413 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %415 = stablehlo.reshape %414 : (tensor<1xf64>) -> tensor<f64>
      %416 = stablehlo.slice %413 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %417 = stablehlo.reshape %416 : (tensor<1xf64>) -> tensor<f64>
      %cst_45 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %418 = stablehlo.add %417, %cst_45 : tensor<f64>
      %419 = stablehlo.atan2 %415, %418 : tensor<f64>
      %420 = stablehlo.slice %413 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %421 = stablehlo.reshape %420 : (tensor<1xf64>) -> tensor<f64>
      %422 = stablehlo.slice %413 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %423 = stablehlo.reshape %422 : (tensor<1xf64>) -> tensor<f64>
      %424 = stablehlo.slice %413 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %425 = stablehlo.reshape %424 : (tensor<1xf64>) -> tensor<f64>
      %426 = stablehlo.multiply %423, %425 : tensor<f64>
      %427 = stablehlo.slice %413 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %428 = stablehlo.reshape %427 : (tensor<1xf64>) -> tensor<f64>
      %429 = stablehlo.slice %413 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %430 = stablehlo.reshape %429 : (tensor<1xf64>) -> tensor<f64>
      %431 = stablehlo.multiply %428, %430 : tensor<f64>
      %432 = stablehlo.add %426, %431 : tensor<f64>
      %433 = stablehlo.sqrt %432 : tensor<f64>
      %cst_46 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %434 = stablehlo.add %433, %cst_46 : tensor<f64>
      %435 = stablehlo.atan2 %421, %434 : tensor<f64>
      %436 = stablehlo.cosine %419 : tensor<f64>
      %437 = stablehlo.sine %419 : tensor<f64>
      %438 = stablehlo.cosine %435 : tensor<f64>
      %439 = stablehlo.sine %435 : tensor<f64>
      %440 = stablehlo.slice %413 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %441 = stablehlo.reshape %440 : (tensor<1xf64>) -> tensor<f64>
      %442 = stablehlo.multiply %441, %436 : tensor<f64>
      %443 = stablehlo.slice %413 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %444 = stablehlo.reshape %443 : (tensor<1xf64>) -> tensor<f64>
      %445 = stablehlo.multiply %444, %437 : tensor<f64>
      %446 = stablehlo.subtract %442, %445 : tensor<f64>
      %cst_47 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %447 = stablehlo.multiply %cst_47, %439 : tensor<f64>
      %448 = stablehlo.add %446, %447 : tensor<f64>
      %449 = stablehlo.slice %413 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %450 = stablehlo.reshape %449 : (tensor<1xf64>) -> tensor<f64>
      %451 = stablehlo.multiply %450, %437 : tensor<f64>
      %452 = stablehlo.slice %413 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %453 = stablehlo.reshape %452 : (tensor<1xf64>) -> tensor<f64>
      %454 = stablehlo.multiply %453, %436 : tensor<f64>
      %455 = stablehlo.add %451, %454 : tensor<f64>
      %cst_48 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %456 = stablehlo.multiply %cst_48, %438 : tensor<f64>
      %457 = stablehlo.add %455, %456 : tensor<f64>
      %458 = stablehlo.slice %413 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %459 = stablehlo.reshape %458 : (tensor<1xf64>) -> tensor<f64>
      %460 = stablehlo.multiply %459, %438 : tensor<f64>
      %461 = stablehlo.multiply %437, %436 : tensor<f64>
      %cst_49 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %462 = stablehlo.multiply %cst_49, %461 : tensor<f64>
      %463 = stablehlo.add %460, %462 : tensor<f64>
      %464 = stablehlo.broadcast_in_dim %448, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %465 = stablehlo.broadcast_in_dim %457, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %466 = stablehlo.broadcast_in_dim %463, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %467 = stablehlo.concatenate %464, %465, %466, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %468 = stablehlo.slice %467 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %469 = stablehlo.reshape %468 : (tensor<1xf64>) -> tensor<f64>
      %470 = stablehlo.slice %467 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %471 = stablehlo.reshape %470 : (tensor<1xf64>) -> tensor<f64>
      %cst_50 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %472 = stablehlo.add %471, %cst_50 : tensor<f64>
      %473 = stablehlo.atan2 %469, %472 : tensor<f64>
      %474 = stablehlo.slice %467 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %475 = stablehlo.reshape %474 : (tensor<1xf64>) -> tensor<f64>
      %476 = stablehlo.slice %467 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %477 = stablehlo.reshape %476 : (tensor<1xf64>) -> tensor<f64>
      %478 = stablehlo.slice %467 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %479 = stablehlo.reshape %478 : (tensor<1xf64>) -> tensor<f64>
      %480 = stablehlo.multiply %477, %479 : tensor<f64>
      %481 = stablehlo.slice %467 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %482 = stablehlo.reshape %481 : (tensor<1xf64>) -> tensor<f64>
      %483 = stablehlo.slice %467 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %484 = stablehlo.reshape %483 : (tensor<1xf64>) -> tensor<f64>
      %485 = stablehlo.multiply %482, %484 : tensor<f64>
      %486 = stablehlo.add %480, %485 : tensor<f64>
      %487 = stablehlo.sqrt %486 : tensor<f64>
      %cst_51 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %488 = stablehlo.add %487, %cst_51 : tensor<f64>
      %489 = stablehlo.atan2 %475, %488 : tensor<f64>
      %490 = stablehlo.cosine %473 : tensor<f64>
      %491 = stablehlo.sine %473 : tensor<f64>
      %492 = stablehlo.cosine %489 : tensor<f64>
      %493 = stablehlo.sine %489 : tensor<f64>
      %494 = stablehlo.slice %467 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %495 = stablehlo.reshape %494 : (tensor<1xf64>) -> tensor<f64>
      %496 = stablehlo.multiply %495, %490 : tensor<f64>
      %497 = stablehlo.slice %467 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %498 = stablehlo.reshape %497 : (tensor<1xf64>) -> tensor<f64>
      %499 = stablehlo.multiply %498, %491 : tensor<f64>
      %500 = stablehlo.subtract %496, %499 : tensor<f64>
      %cst_52 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %501 = stablehlo.multiply %cst_52, %493 : tensor<f64>
      %502 = stablehlo.add %500, %501 : tensor<f64>
      %503 = stablehlo.slice %467 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %504 = stablehlo.reshape %503 : (tensor<1xf64>) -> tensor<f64>
      %505 = stablehlo.multiply %504, %491 : tensor<f64>
      %506 = stablehlo.slice %467 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %507 = stablehlo.reshape %506 : (tensor<1xf64>) -> tensor<f64>
      %508 = stablehlo.multiply %507, %490 : tensor<f64>
      %509 = stablehlo.add %505, %508 : tensor<f64>
      %cst_53 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %510 = stablehlo.multiply %cst_53, %492 : tensor<f64>
      %511 = stablehlo.add %509, %510 : tensor<f64>
      %512 = stablehlo.slice %467 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %513 = stablehlo.reshape %512 : (tensor<1xf64>) -> tensor<f64>
      %514 = stablehlo.multiply %513, %492 : tensor<f64>
      %515 = stablehlo.multiply %491, %490 : tensor<f64>
      %cst_54 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %516 = stablehlo.multiply %cst_54, %515 : tensor<f64>
      %517 = stablehlo.add %514, %516 : tensor<f64>
      %518 = stablehlo.broadcast_in_dim %502, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %519 = stablehlo.broadcast_in_dim %511, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %520 = stablehlo.broadcast_in_dim %517, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %521 = stablehlo.concatenate %518, %519, %520, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %522 = stablehlo.slice %521 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %523 = stablehlo.reshape %522 : (tensor<1xf64>) -> tensor<f64>
      %524 = stablehlo.slice %521 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %525 = stablehlo.reshape %524 : (tensor<1xf64>) -> tensor<f64>
      %cst_55 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %526 = stablehlo.add %525, %cst_55 : tensor<f64>
      %527 = stablehlo.atan2 %523, %526 : tensor<f64>
      %528 = stablehlo.slice %521 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %529 = stablehlo.reshape %528 : (tensor<1xf64>) -> tensor<f64>
      %530 = stablehlo.slice %521 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %531 = stablehlo.reshape %530 : (tensor<1xf64>) -> tensor<f64>
      %532 = stablehlo.slice %521 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %533 = stablehlo.reshape %532 : (tensor<1xf64>) -> tensor<f64>
      %534 = stablehlo.multiply %531, %533 : tensor<f64>
      %535 = stablehlo.slice %521 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %536 = stablehlo.reshape %535 : (tensor<1xf64>) -> tensor<f64>
      %537 = stablehlo.slice %521 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %538 = stablehlo.reshape %537 : (tensor<1xf64>) -> tensor<f64>
      %539 = stablehlo.multiply %536, %538 : tensor<f64>
      %540 = stablehlo.add %534, %539 : tensor<f64>
      %541 = stablehlo.sqrt %540 : tensor<f64>
      %cst_56 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %542 = stablehlo.add %541, %cst_56 : tensor<f64>
      %543 = stablehlo.atan2 %529, %542 : tensor<f64>
      %544 = stablehlo.cosine %527 : tensor<f64>
      %545 = stablehlo.sine %527 : tensor<f64>
      %546 = stablehlo.cosine %543 : tensor<f64>
      %547 = stablehlo.sine %543 : tensor<f64>
      %548 = stablehlo.slice %521 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %549 = stablehlo.reshape %548 : (tensor<1xf64>) -> tensor<f64>
      %550 = stablehlo.multiply %549, %544 : tensor<f64>
      %551 = stablehlo.slice %521 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %552 = stablehlo.reshape %551 : (tensor<1xf64>) -> tensor<f64>
      %553 = stablehlo.multiply %552, %545 : tensor<f64>
      %554 = stablehlo.subtract %550, %553 : tensor<f64>
      %cst_57 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %555 = stablehlo.multiply %cst_57, %547 : tensor<f64>
      %556 = stablehlo.add %554, %555 : tensor<f64>
      %557 = stablehlo.slice %521 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %558 = stablehlo.reshape %557 : (tensor<1xf64>) -> tensor<f64>
      %559 = stablehlo.multiply %558, %545 : tensor<f64>
      %560 = stablehlo.slice %521 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %561 = stablehlo.reshape %560 : (tensor<1xf64>) -> tensor<f64>
      %562 = stablehlo.multiply %561, %544 : tensor<f64>
      %563 = stablehlo.add %559, %562 : tensor<f64>
      %cst_58 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %564 = stablehlo.multiply %cst_58, %546 : tensor<f64>
      %565 = stablehlo.add %563, %564 : tensor<f64>
      %566 = stablehlo.slice %521 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %567 = stablehlo.reshape %566 : (tensor<1xf64>) -> tensor<f64>
      %568 = stablehlo.multiply %567, %546 : tensor<f64>
      %569 = stablehlo.multiply %545, %544 : tensor<f64>
      %cst_59 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %570 = stablehlo.multiply %cst_59, %569 : tensor<f64>
      %571 = stablehlo.add %568, %570 : tensor<f64>
      %572 = stablehlo.broadcast_in_dim %556, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %573 = stablehlo.broadcast_in_dim %565, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %574 = stablehlo.broadcast_in_dim %571, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %575 = stablehlo.concatenate %572, %573, %574, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %576 = stablehlo.slice %575 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %577 = stablehlo.reshape %576 : (tensor<1xf64>) -> tensor<f64>
      %578 = stablehlo.slice %575 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %579 = stablehlo.reshape %578 : (tensor<1xf64>) -> tensor<f64>
      %cst_60 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %580 = stablehlo.add %579, %cst_60 : tensor<f64>
      %581 = stablehlo.atan2 %577, %580 : tensor<f64>
      %582 = stablehlo.slice %575 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %583 = stablehlo.reshape %582 : (tensor<1xf64>) -> tensor<f64>
      %584 = stablehlo.slice %575 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %585 = stablehlo.reshape %584 : (tensor<1xf64>) -> tensor<f64>
      %586 = stablehlo.slice %575 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %587 = stablehlo.reshape %586 : (tensor<1xf64>) -> tensor<f64>
      %588 = stablehlo.multiply %585, %587 : tensor<f64>
      %589 = stablehlo.slice %575 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %590 = stablehlo.reshape %589 : (tensor<1xf64>) -> tensor<f64>
      %591 = stablehlo.slice %575 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %592 = stablehlo.reshape %591 : (tensor<1xf64>) -> tensor<f64>
      %593 = stablehlo.multiply %590, %592 : tensor<f64>
      %594 = stablehlo.add %588, %593 : tensor<f64>
      %595 = stablehlo.sqrt %594 : tensor<f64>
      %cst_61 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %596 = stablehlo.add %595, %cst_61 : tensor<f64>
      %597 = stablehlo.atan2 %583, %596 : tensor<f64>
      %598 = stablehlo.cosine %581 : tensor<f64>
      %599 = stablehlo.sine %581 : tensor<f64>
      %600 = stablehlo.cosine %597 : tensor<f64>
      %601 = stablehlo.sine %597 : tensor<f64>
      %602 = stablehlo.slice %575 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %603 = stablehlo.reshape %602 : (tensor<1xf64>) -> tensor<f64>
      %604 = stablehlo.multiply %603, %598 : tensor<f64>
      %605 = stablehlo.slice %575 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %606 = stablehlo.reshape %605 : (tensor<1xf64>) -> tensor<f64>
      %607 = stablehlo.multiply %606, %599 : tensor<f64>
      %608 = stablehlo.subtract %604, %607 : tensor<f64>
      %cst_62 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %609 = stablehlo.multiply %cst_62, %601 : tensor<f64>
      %610 = stablehlo.add %608, %609 : tensor<f64>
      %611 = stablehlo.slice %575 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %612 = stablehlo.reshape %611 : (tensor<1xf64>) -> tensor<f64>
      %613 = stablehlo.multiply %612, %599 : tensor<f64>
      %614 = stablehlo.slice %575 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %615 = stablehlo.reshape %614 : (tensor<1xf64>) -> tensor<f64>
      %616 = stablehlo.multiply %615, %598 : tensor<f64>
      %617 = stablehlo.add %613, %616 : tensor<f64>
      %cst_63 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %618 = stablehlo.multiply %cst_63, %600 : tensor<f64>
      %619 = stablehlo.add %617, %618 : tensor<f64>
      %620 = stablehlo.slice %575 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %621 = stablehlo.reshape %620 : (tensor<1xf64>) -> tensor<f64>
      %622 = stablehlo.multiply %621, %600 : tensor<f64>
      %623 = stablehlo.multiply %599, %598 : tensor<f64>
      %cst_64 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %624 = stablehlo.multiply %cst_64, %623 : tensor<f64>
      %625 = stablehlo.add %622, %624 : tensor<f64>
      %626 = stablehlo.broadcast_in_dim %610, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %627 = stablehlo.broadcast_in_dim %619, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %628 = stablehlo.broadcast_in_dim %625, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %629 = stablehlo.concatenate %626, %627, %628, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %630 = stablehlo.slice %629 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %631 = stablehlo.reshape %630 : (tensor<1xf64>) -> tensor<f64>
      %632 = stablehlo.slice %629 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %633 = stablehlo.reshape %632 : (tensor<1xf64>) -> tensor<f64>
      %cst_65 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %634 = stablehlo.add %633, %cst_65 : tensor<f64>
      %635 = stablehlo.atan2 %631, %634 : tensor<f64>
      %636 = stablehlo.slice %629 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %637 = stablehlo.reshape %636 : (tensor<1xf64>) -> tensor<f64>
      %638 = stablehlo.slice %629 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %639 = stablehlo.reshape %638 : (tensor<1xf64>) -> tensor<f64>
      %640 = stablehlo.slice %629 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %641 = stablehlo.reshape %640 : (tensor<1xf64>) -> tensor<f64>
      %642 = stablehlo.multiply %639, %641 : tensor<f64>
      %643 = stablehlo.slice %629 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %644 = stablehlo.reshape %643 : (tensor<1xf64>) -> tensor<f64>
      %645 = stablehlo.slice %629 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %646 = stablehlo.reshape %645 : (tensor<1xf64>) -> tensor<f64>
      %647 = stablehlo.multiply %644, %646 : tensor<f64>
      %648 = stablehlo.add %642, %647 : tensor<f64>
      %649 = stablehlo.sqrt %648 : tensor<f64>
      %cst_66 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %650 = stablehlo.add %649, %cst_66 : tensor<f64>
      %651 = stablehlo.atan2 %637, %650 : tensor<f64>
      %652 = stablehlo.cosine %635 : tensor<f64>
      %653 = stablehlo.sine %635 : tensor<f64>
      %654 = stablehlo.cosine %651 : tensor<f64>
      %655 = stablehlo.sine %651 : tensor<f64>
      %656 = stablehlo.slice %629 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %657 = stablehlo.reshape %656 : (tensor<1xf64>) -> tensor<f64>
      %658 = stablehlo.multiply %657, %652 : tensor<f64>
      %659 = stablehlo.slice %629 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %660 = stablehlo.reshape %659 : (tensor<1xf64>) -> tensor<f64>
      %661 = stablehlo.multiply %660, %653 : tensor<f64>
      %662 = stablehlo.subtract %658, %661 : tensor<f64>
      %cst_67 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %663 = stablehlo.multiply %cst_67, %655 : tensor<f64>
      %664 = stablehlo.add %662, %663 : tensor<f64>
      %665 = stablehlo.slice %629 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %666 = stablehlo.reshape %665 : (tensor<1xf64>) -> tensor<f64>
      %667 = stablehlo.multiply %666, %653 : tensor<f64>
      %668 = stablehlo.slice %629 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %669 = stablehlo.reshape %668 : (tensor<1xf64>) -> tensor<f64>
      %670 = stablehlo.multiply %669, %652 : tensor<f64>
      %671 = stablehlo.add %667, %670 : tensor<f64>
      %cst_68 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %672 = stablehlo.multiply %cst_68, %654 : tensor<f64>
      %673 = stablehlo.add %671, %672 : tensor<f64>
      %674 = stablehlo.slice %629 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %675 = stablehlo.reshape %674 : (tensor<1xf64>) -> tensor<f64>
      %676 = stablehlo.multiply %675, %654 : tensor<f64>
      %677 = stablehlo.multiply %653, %652 : tensor<f64>
      %cst_69 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %678 = stablehlo.multiply %cst_69, %677 : tensor<f64>
      %679 = stablehlo.add %676, %678 : tensor<f64>
      %680 = stablehlo.broadcast_in_dim %664, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %681 = stablehlo.broadcast_in_dim %673, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %682 = stablehlo.broadcast_in_dim %679, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %683 = stablehlo.concatenate %680, %681, %682, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %684 = stablehlo.slice %683 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %685 = stablehlo.reshape %684 : (tensor<1xf64>) -> tensor<f64>
      %686 = stablehlo.slice %683 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %687 = stablehlo.reshape %686 : (tensor<1xf64>) -> tensor<f64>
      %cst_70 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %688 = stablehlo.add %687, %cst_70 : tensor<f64>
      %689 = stablehlo.atan2 %685, %688 : tensor<f64>
      %690 = stablehlo.slice %683 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %691 = stablehlo.reshape %690 : (tensor<1xf64>) -> tensor<f64>
      %692 = stablehlo.slice %683 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %693 = stablehlo.reshape %692 : (tensor<1xf64>) -> tensor<f64>
      %694 = stablehlo.slice %683 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %695 = stablehlo.reshape %694 : (tensor<1xf64>) -> tensor<f64>
      %696 = stablehlo.multiply %693, %695 : tensor<f64>
      %697 = stablehlo.slice %683 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %698 = stablehlo.reshape %697 : (tensor<1xf64>) -> tensor<f64>
      %699 = stablehlo.slice %683 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %700 = stablehlo.reshape %699 : (tensor<1xf64>) -> tensor<f64>
      %701 = stablehlo.multiply %698, %700 : tensor<f64>
      %702 = stablehlo.add %696, %701 : tensor<f64>
      %703 = stablehlo.sqrt %702 : tensor<f64>
      %cst_71 = stablehlo.constant dense<1.000000e-09> : tensor<f64>
      %704 = stablehlo.add %703, %cst_71 : tensor<f64>
      %705 = stablehlo.atan2 %691, %704 : tensor<f64>
      %706 = stablehlo.cosine %689 : tensor<f64>
      %707 = stablehlo.sine %689 : tensor<f64>
      %708 = stablehlo.cosine %705 : tensor<f64>
      %709 = stablehlo.sine %705 : tensor<f64>
      %710 = stablehlo.slice %683 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %711 = stablehlo.reshape %710 : (tensor<1xf64>) -> tensor<f64>
      %712 = stablehlo.multiply %711, %706 : tensor<f64>
      %713 = stablehlo.slice %683 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %714 = stablehlo.reshape %713 : (tensor<1xf64>) -> tensor<f64>
      %715 = stablehlo.multiply %714, %707 : tensor<f64>
      %716 = stablehlo.subtract %712, %715 : tensor<f64>
      %cst_72 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %717 = stablehlo.multiply %cst_72, %709 : tensor<f64>
      %718 = stablehlo.add %716, %717 : tensor<f64>
      %719 = stablehlo.slice %683 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %720 = stablehlo.reshape %719 : (tensor<1xf64>) -> tensor<f64>
      %721 = stablehlo.multiply %720, %707 : tensor<f64>
      %722 = stablehlo.slice %683 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %723 = stablehlo.reshape %722 : (tensor<1xf64>) -> tensor<f64>
      %724 = stablehlo.multiply %723, %706 : tensor<f64>
      %725 = stablehlo.add %721, %724 : tensor<f64>
      %cst_73 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %726 = stablehlo.multiply %cst_73, %708 : tensor<f64>
      %727 = stablehlo.add %725, %726 : tensor<f64>
      %728 = stablehlo.slice %683 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %729 = stablehlo.reshape %728 : (tensor<1xf64>) -> tensor<f64>
      %730 = stablehlo.multiply %729, %708 : tensor<f64>
      %731 = stablehlo.multiply %707, %706 : tensor<f64>
      %cst_74 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
      %732 = stablehlo.multiply %cst_74, %731 : tensor<f64>
      %733 = stablehlo.add %730, %732 : tensor<f64>
      %734 = stablehlo.broadcast_in_dim %718, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %735 = stablehlo.broadcast_in_dim %727, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %736 = stablehlo.broadcast_in_dim %733, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %737 = stablehlo.concatenate %734, %735, %736, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      %cst_75 = stablehlo.constant dense<9.9999999999999998E-13> : tensor<f64>
      %738 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %739 = stablehlo.multiply %738, %737 : tensor<3xf64>
      %740 = stablehlo.add %25, %739 : tensor<3xf64>
      stablehlo.return %740 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    %65 = call @norm_39(%64) : (tensor<3xf64>) -> tensor<f64>
    %66 = stablehlo.slice %23 [0:3, 0:1] : (tensor<3x3xf64>) -> tensor<3x1xf64>
    %67 = stablehlo.reshape %66 : (tensor<3x1xf64>) -> tensor<3xf64>
    %68 = call @norm_39(%67) : (tensor<3xf64>) -> tensor<f64>
    %69 = stablehlo.broadcast_in_dim %52, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %71 = stablehlo.broadcast_in_dim %44#0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %72 = stablehlo.broadcast_in_dim %65, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %73 = stablehlo.broadcast_in_dim %68, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %74 = stablehlo.concatenate %69, %70, %71, %72, %73, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<5xf64>
    return %64, %42, %74 : tensor<3xf64>, tensor<3x3xf64>, tensor<5xf64>
  }
  func.func private @cholesky(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %1 = stablehlo.add %arg0, %0 : tensor<3x3xf64>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %3 = stablehlo.divide %1, %2 : tensor<3x3xf64>
    %4:2 = stablehlo.custom_call @lapack_dpotrf_ffi(%3) {backend_config = "", mhlo.backend_config = {uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], []) {i=3, j=3, k=3, l=3}, custom>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<i32>)
    %c = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %6 = stablehlo.compare  EQ, %4#1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst_0 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %9 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %10 = stablehlo.select %9, %4#0, %8 : tensor<3x3xi1>, tensor<3x3xf64>
    %11 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %12 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %13 = stablehlo.add %11, %12 : tensor<3x3xi64>
    %14 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    %15 = stablehlo.compare  GE, %13, %14,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %17 = stablehlo.select %15, %10, %16 : tensor<3x3xi1>, tensor<3x3xf64>
    return %17 : tensor<3x3xf64>
  }
  func.func private @solve_59(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=3, k=3, l=3, m=3}, custom>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<3xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
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
      %14 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14:2 = func.call @closed_call_66(%iterArg, %iterArg_4, %iterArg_5) : (tensor<3xi32>, tensor<i64>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %15 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %15, %14#0, %14#1 : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    }
    %11 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %12 = call @_lu_solve_71(%8, %10#3, %11) : (tensor<3x3xf64>, tensor<3xi32>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    return %13 : tensor<3x3xf64>
  }
  func.func private @closed_call_66(%arg0: tensor<3xi32>, %arg1: tensor<i64>, %arg2: tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg1, %c : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare  LT, %arg1, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = stablehlo.convert %arg1 : tensor<i64>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %3 = stablehlo.add %2, %c_1 : tensor<i64>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<i1>, tensor<i64>
    %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<3xi32>, tensor<i64>) -> tensor<1xi32>
    %6 = stablehlo.reshape %5 : (tensor<1xi32>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.compare  LT, %arg1, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8 = stablehlo.convert %arg1 : tensor<i64>
    %c_3 = stablehlo.constant dense<3> : tensor<i64>
    %9 = stablehlo.add %8, %c_3 : tensor<i64>
    %10 = stablehlo.select %7, %9, %arg1 : tensor<i1>, tensor<i64>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<3xi32>, tensor<i64>) -> tensor<1xi32>
    %12 = stablehlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.compare  LT, %6, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<3> : tensor<i32>
    %14 = stablehlo.add %6, %c_5 : tensor<i32>
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32>
    %16 = stablehlo.dynamic_slice %arg2, %15, sizes = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %18 = stablehlo.compare  LT, %arg1, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
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
    %24 = stablehlo.compare  LT, %6, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
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
  func.func private @_lu_solve_71(%arg0: tensor<3x3xf64>, %arg1: tensor<3xi32>, %arg2: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<3x3xf64>) -> tensor<3x3x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<3xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<3xi1>, tensor<3xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 3, 1, 1>}> : (tensor<3x3x1xf64>, tensor<3x1xi32>) -> tensor<3x3x1xf64>
    %8 = stablehlo.transpose %7, dims = [1, 2, 0] : (tensor<3x3x1xf64>) -> tensor<3x1x3xf64>
    %9 = stablehlo.reshape %8 : (tensor<3x1x3xf64>) -> tensor<3x3xf64>
    %10 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %9) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=3, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %11 = stablehlo.reshape %10 : (tensor<3x3xf64>) -> tensor<3x1x3xf64>
    %12 = stablehlo.reshape %11 : (tensor<3x1x3xf64>) -> tensor<3x3xf64>
    %13 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %12) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=3, j=3, k=3, l=3, m=3, n=3}, custom>} : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
    %14 = stablehlo.reshape %13 : (tensor<3x3xf64>) -> tensor<3x1x3xf64>
    %15 = stablehlo.slice %14 [0:3, 0:1, 0:3] : (tensor<3x1x3xf64>) -> tensor<3x1x3xf64>
    %16 = stablehlo.transpose %15, dims = [2, 0, 1] : (tensor<3x1x3xf64>) -> tensor<3x3x1xf64>
    %17 = stablehlo.reshape %16 : (tensor<3x3x1xf64>) -> tensor<3x3xf64>
    return %17 : tensor<3x3xf64>
  }
  func.func private @qr(%arg0: tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3x3xf64>) {
    %0:2 = stablehlo.custom_call @lapack_dgeqrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m]) {i=3, j=3, k=3, l=3, m=3}, custom>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xf64>)
    %1 = stablehlo.custom_call @lapack_dorgqr_ffi(%0#0, %0#1) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k])->([l, m]) {i=3, j=3, k=3, l=3, m=3}, custom>} : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3x3xf64>
    %2 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    %c = stablehlo.constant dense<-1> : tensor<i64>
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %4 = stablehlo.add %2, %3 : tensor<3x3xi64>
    %5 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    %6 = stablehlo.compare  GE, %4, %5,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %8 = stablehlo.select %6, %7, %0#0 : tensor<3x3xi1>, tensor<3x3xf64>
    return %1, %8 : tensor<3x3xf64>, tensor<3x3xf64>
  }
  func.func private @det(%arg0: tensor<3x3xf64>) -> tensor<f64> {
    %0 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x1xf64>) -> tensor<f64>
    %2 = stablehlo.slice %arg0 [1:2, 1:2] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x1xf64>) -> tensor<f64>
    %4 = stablehlo.multiply %1, %3 : tensor<f64>
    %5 = stablehlo.slice %arg0 [2:3, 2:3] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1x1xf64>) -> tensor<f64>
    %7 = stablehlo.multiply %4, %6 : tensor<f64>
    %8 = stablehlo.slice %arg0 [0:1, 1:2] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %9 = stablehlo.reshape %8 : (tensor<1x1xf64>) -> tensor<f64>
    %10 = stablehlo.slice %arg0 [1:2, 2:3] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1x1xf64>) -> tensor<f64>
    %12 = stablehlo.multiply %9, %11 : tensor<f64>
    %13 = stablehlo.slice %arg0 [2:3, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %14 = stablehlo.reshape %13 : (tensor<1x1xf64>) -> tensor<f64>
    %15 = stablehlo.multiply %12, %14 : tensor<f64>
    %16 = stablehlo.add %7, %15 : tensor<f64>
    %17 = stablehlo.slice %arg0 [0:1, 2:3] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %18 = stablehlo.reshape %17 : (tensor<1x1xf64>) -> tensor<f64>
    %19 = stablehlo.slice %arg0 [1:2, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %20 = stablehlo.reshape %19 : (tensor<1x1xf64>) -> tensor<f64>
    %21 = stablehlo.multiply %18, %20 : tensor<f64>
    %22 = stablehlo.slice %arg0 [2:3, 1:2] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %23 = stablehlo.reshape %22 : (tensor<1x1xf64>) -> tensor<f64>
    %24 = stablehlo.multiply %21, %23 : tensor<f64>
    %25 = stablehlo.add %16, %24 : tensor<f64>
    %26 = stablehlo.slice %arg0 [0:1, 2:3] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %27 = stablehlo.reshape %26 : (tensor<1x1xf64>) -> tensor<f64>
    %28 = stablehlo.slice %arg0 [1:2, 1:2] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1x1xf64>) -> tensor<f64>
    %30 = stablehlo.multiply %27, %29 : tensor<f64>
    %31 = stablehlo.slice %arg0 [2:3, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %32 = stablehlo.reshape %31 : (tensor<1x1xf64>) -> tensor<f64>
    %33 = stablehlo.multiply %30, %32 : tensor<f64>
    %34 = stablehlo.subtract %25, %33 : tensor<f64>
    %35 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1x1xf64>) -> tensor<f64>
    %37 = stablehlo.slice %arg0 [1:2, 2:3] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %38 = stablehlo.reshape %37 : (tensor<1x1xf64>) -> tensor<f64>
    %39 = stablehlo.multiply %36, %38 : tensor<f64>
    %40 = stablehlo.slice %arg0 [2:3, 1:2] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1x1xf64>) -> tensor<f64>
    %42 = stablehlo.multiply %39, %41 : tensor<f64>
    %43 = stablehlo.subtract %34, %42 : tensor<f64>
    %44 = stablehlo.slice %arg0 [0:1, 1:2] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %45 = stablehlo.reshape %44 : (tensor<1x1xf64>) -> tensor<f64>
    %46 = stablehlo.slice %arg0 [1:2, 0:1] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %47 = stablehlo.reshape %46 : (tensor<1x1xf64>) -> tensor<f64>
    %48 = stablehlo.multiply %45, %47 : tensor<f64>
    %49 = stablehlo.slice %arg0 [2:3, 2:3] : (tensor<3x3xf64>) -> tensor<1x1xf64>
    %50 = stablehlo.reshape %49 : (tensor<1x1xf64>) -> tensor<f64>
    %51 = stablehlo.multiply %48, %50 : tensor<f64>
    %52 = stablehlo.subtract %43, %51 : tensor<f64>
    return %52 : tensor<f64>
  }
  func.func private @slogdet(%arg0: tensor<3x3xf64>) -> (tensor<f64>, tensor<f64>) {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=3, k=3, l=3, m=3}, custom>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<3xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<3x3xi1>
    %8 = stablehlo.select %7, %0#0, %6 : tensor<3x3xi1>, tensor<3x3xf64>
    %9 = stablehlo.iota dim = 0 : tensor<3xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %10:4 = stablehlo.while(%iterArg = %2, %iterArg_13 = %c_2, %iterArg_14 = %c_1, %iterArg_15 = %9) : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    cond {
      %c_16 = stablehlo.constant dense<3> : tensor<i64>
      %32 = stablehlo.compare  LT, %iterArg_13, %c_16,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %32 : tensor<i1>
    } do {
      %32:2 = func.call @closed_call_66(%iterArg, %iterArg_14, %iterArg_15) : (tensor<3xi32>, tensor<i64>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>)
      %c_16 = stablehlo.constant dense<1> : tensor<i64>
      %33 = stablehlo.add %iterArg_13, %c_16 : tensor<i64>
      stablehlo.return %iterArg, %33, %32#0, %32#1 : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    }
    %11 = call @diagonal(%8) : (tensor<3x3xf64>) -> tensor<3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %13 = stablehlo.compare  EQ, %11, %12,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %c_4 = stablehlo.constant dense<false> : tensor<i1>
    %14 = stablehlo.reduce(%13 init: %c_4) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %15 = stablehlo.iota dim = 0 : tensor<3xi32>
    %16 = stablehlo.compare  NE, %2, %15,  SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %17 = call @count_nonzero(%16) : (tensor<3xi1>) -> tensor<i64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %18 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %19 = stablehlo.compare  LT, %11, %18,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %20 = call @count_nonzero(%19) : (tensor<3xi1>) -> tensor<i64>
    %21 = stablehlo.add %17, %20 : tensor<i64>
    %c_6 = stablehlo.constant dense<2> : tensor<i64>
    %22 = call @remainder(%21, %c_6) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %c_7 = stablehlo.constant dense<-2> : tensor<i64>
    %23 = stablehlo.multiply %c_7, %22 : tensor<i64>
    %c_8 = stablehlo.constant dense<1> : tensor<i64>
    %24 = stablehlo.add %23, %c_8 : tensor<i64>
    %25 = stablehlo.convert %24 : (tensor<i64>) -> tensor<f64>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %26 = stablehlo.multiply %cst_9, %25 : tensor<f64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %27 = call @_where_124(%14, %cst_10, %26) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %28 = stablehlo.abs %11 : tensor<3xf64>
    %29 = stablehlo.log %28 : tensor<3xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %30 = stablehlo.reduce(%29 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %cst_12 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %31 = call @_where_124(%14, %cst_12, %30) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    return %27, %31 : tensor<f64>, tensor<f64>
  }
  func.func private @diagonal(%arg0: tensor<3x3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    %1 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi64>
    %4 = stablehlo.compare  EQ, %3, %1,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
    %5 = stablehlo.convert %4 : (tensor<3x3xi1>) -> tensor<3x3xf64>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %6 = "stablehlo.case"(%c_0) ({
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      stablehlo.return %c_1 : tensor<i32>
    }, {
      %c_1 = stablehlo.constant dense<0> : tensor<i32>
      stablehlo.return %c_1 : tensor<i32>
    }) : (tensor<i32>) -> tensor<i32>
    %7 = "stablehlo.case"(%6) ({
      %8 = stablehlo.iota dim = 0 : tensor<3xi64>
      %9 = stablehlo.iota dim = 0 : tensor<3xi64>
      %c_1 = stablehlo.constant dense<0> : tensor<i64>
      %10 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3xi64>
      %11 = stablehlo.compare  LT, %8, %10,  SIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
      %c_2 = stablehlo.constant dense<3> : tensor<i64>
      %12 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<3xi64>
      %13 = stablehlo.add %8, %12 : tensor<3xi64>
      %14 = stablehlo.select %11, %13, %8 : tensor<3xi1>, tensor<3xi64>
      %c_3 = stablehlo.constant dense<0> : tensor<i64>
      %15 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3xi64>
      %16 = stablehlo.compare  LT, %9, %15,  SIGNED : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
      %c_4 = stablehlo.constant dense<3> : tensor<i64>
      %17 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<3xi64>
      %18 = stablehlo.add %9, %17 : tensor<3xi64>
      %19 = stablehlo.select %16, %18, %9 : tensor<3xi1>, tensor<3xi64>
      %20 = stablehlo.convert %14 : (tensor<3xi64>) -> tensor<3xi32>
      %21 = stablehlo.convert %19 : (tensor<3xi64>) -> tensor<3xi32>
      %22 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
      %23 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
      %24 = stablehlo.concatenate %22, %23, dim = 1 : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x2xi32>
      %25 = "stablehlo.gather"(%arg0, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x3xf64>, tensor<3x2xi32>) -> tensor<3xf64>
      stablehlo.return %25 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    return %7 : tensor<3xf64>
  }
  func.func private @count_nonzero(%arg0: tensor<3xi1>) -> tensor<i64> {
    %c = stablehlo.constant dense<false> : tensor<i1>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
    %1 = stablehlo.compare  NE, %arg0, %0,  UNSIGNED : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
    %2 = stablehlo.convert %1 : (tensor<3xi1>) -> tensor<3xi32>
    %3 = stablehlo.convert %2 : (tensor<3xi32>) -> tensor<3xi64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %4 = stablehlo.reduce(%3 init: %c_0) applies stablehlo.add across dimensions = [0] : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    return %4 : tensor<i64>
  }
  func.func private @remainder(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.convert %arg1 : tensor<i64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare  EQ, %0, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %2 = call @_where_117(%1, %c_0, %0) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %3 = stablehlo.remainder %arg0, %2 : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %4 = stablehlo.compare  NE, %3, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %5 = stablehlo.compare  LT, %3, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %6 = stablehlo.compare  LT, %2, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %7 = stablehlo.compare  NE, %5, %6,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = stablehlo.and %7, %4 : tensor<i1>
    %9 = stablehlo.add %3, %2 : tensor<i64>
    %10 = stablehlo.select %8, %9, %3 : tensor<i1>, tensor<i64>
    return %10 : tensor<i64>
  }
  func.func private @_where_117(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @_where_124(%arg0: tensor<i1>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @solve_126(%arg0: tensor<3x3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=3, k=3, l=3, m=3}, custom>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<3xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
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
      %12 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %12 : tensor<i1>
    } do {
      %12:2 = func.call @closed_call_66(%iterArg, %iterArg_4, %iterArg_5) : (tensor<3xi32>, tensor<i64>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %13 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %13, %12#0, %12#1 : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    }
    %11 = call @_lu_solve_128(%8, %10#3, %arg1) : (tensor<3x3xf64>, tensor<3xi32>, tensor<3xf64>) -> tensor<3xf64>
    return %11 : tensor<3xf64>
  }
  func.func private @_lu_solve_128(%arg0: tensor<3x3xf64>, %arg1: tensor<3xi32>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<3xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<3xi1>, tensor<3xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<3x1xf64>, tensor<3x1xi32>) -> tensor<3x1xf64>
    %8 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %7) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=3, j=3, k=3, l=1, m=3, n=1}, custom>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %9 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %8) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=3, j=3, k=3, l=1, m=3, n=1}, custom>} : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x1xf64>
    %10 = stablehlo.slice %9 [0:3, 0:1] : (tensor<3x1xf64>) -> tensor<3x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<3x1xf64>) -> tensor<3xf64>
    return %11 : tensor<3xf64>
  }
  func.func private @inner_146(%arg0: tensor<2xf64>, %arg1: tensor<2x2xf64>) -> (tensor<2xf64>, tensor<2x2xf64>) {
    %cst = stablehlo.constant dense<[[1.000000e+00, 0.0083333333333333332], [0.000000e+00, 1.000000e+00]]> : tensor<2x2xf64>
    %cst_0 = stablehlo.constant dense<[[1.000000e-02, 0.000000e+00], [0.000000e+00, 1.000000e-02]]> : tensor<2x2xf64>
    %cst_1 = stablehlo.constant dense<[[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]> : tensor<2x2xf64>
    %cst_2 = stablehlo.constant dense<[[1.000000e-01, 0.000000e+00], [0.000000e+00, 1.000000e-01]]> : tensor<2x2xf64>
    %0 = stablehlo.dot_general %cst, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %1 = stablehlo.dot_general %cst, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %2 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %3 = stablehlo.dot_general %1, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %4 = stablehlo.add %3, %cst_0 : tensor<2x2xf64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %cst_4 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %7 = stablehlo.multiply %6, %5 : tensor<2xf64>
    %8 = stablehlo.add %0, %7 : tensor<2xf64>
    %9 = stablehlo.dot_general %cst_1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %10 = stablehlo.subtract %8, %9 : tensor<2xf64>
    %11 = stablehlo.dot_general %cst_1, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %12 = stablehlo.transpose %cst_1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %13 = stablehlo.dot_general %11, %12, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %14 = stablehlo.add %13, %cst_2 : tensor<2x2xf64>
    %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %16 = stablehlo.transpose %cst_1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %17 = stablehlo.dot_general %4, %16, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %18 = stablehlo.transpose %17, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %19 = call @solve_155(%15, %18) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %20 = stablehlo.transpose %19, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %21 = stablehlo.dot_general %20, %10, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %22 = stablehlo.add %0, %21 : tensor<2xf64>
    %23 = stablehlo.iota dim = 0 : tensor<2x2xi64>
    %24 = stablehlo.iota dim = 1 : tensor<2x2xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %25 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<2x2xi64>
    %26 = stablehlo.add %23, %25 : tensor<2x2xi64>
    %27 = stablehlo.compare  EQ, %26, %24,  SIGNED : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
    %28 = stablehlo.convert %27 : (tensor<2x2xi1>) -> tensor<2x2xf64>
    %29 = stablehlo.dot_general %20, %cst_1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %30 = stablehlo.subtract %28, %29 : tensor<2x2xf64>
    %31 = stablehlo.dot_general %30, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %32 = stablehlo.transpose %30, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %33 = stablehlo.dot_general %31, %32, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %34 = stablehlo.dot_general %20, %cst_2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %35 = stablehlo.transpose %20, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %36 = stablehlo.dot_general %34, %35, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %37 = stablehlo.add %33, %36 : tensor<2x2xf64>
    %38 = call @inv(%37) : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %39 = stablehlo.dot_general %38, %37, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %40 = call @norm_188(%10) : (tensor<2xf64>) -> tensor<f64>
    %cst_5 = stablehlo.constant dense<5.000000e+01> : tensor<f64>
    %41 = stablehlo.compare  LT, %40, %cst_5,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %42 = stablehlo.slice %arg0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %43 = stablehlo.reshape %42 : (tensor<1xf64>) -> tensor<f64>
    %cst_6 = stablehlo.constant dense<-1.000000e+06> : tensor<f64>
    %44 = stablehlo.compare  GT, %43, %cst_6,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %45 = stablehlo.and %41, %44 : tensor<i1>
    %46 = stablehlo.convert %45 : (tensor<i1>) -> tensor<i32>
    %47 = "stablehlo.case"(%46) ({
      stablehlo.return %22 : tensor<2xf64>
    }, {
      %48 = stablehlo.iota dim = 0 : tensor<2x2xi64>
      %49 = stablehlo.iota dim = 1 : tensor<2x2xi64>
      %c_7 = stablehlo.constant dense<0> : tensor<i64>
      %50 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i64>) -> tensor<2x2xi64>
      %51 = stablehlo.add %48, %50 : tensor<2x2xi64>
      %52 = stablehlo.compare  EQ, %51, %49,  SIGNED : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
      %53 = stablehlo.convert %52 : (tensor<2x2xi1>) -> tensor<2x2xf64>
      %cst_8 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
      %54 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
      %55 = stablehlo.multiply %54, %53 : tensor<2x2xf64>
      %56 = stablehlo.add %14, %55 : tensor<2x2xf64>
      %57 = func.call @solve_194(%56, %10) : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
      %cst_9 = stablehlo.constant dense<9.9999999999999998E-13> : tensor<f64>
      %58 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f64>) -> tensor<2xf64>
      %59 = stablehlo.multiply %58, %57 : tensor<2xf64>
      %60 = stablehlo.add %22, %59 : tensor<2xf64>
      stablehlo.return %60 : tensor<2xf64>
    }) : (tensor<i32>) -> tensor<2xf64>
    return %47, %37 : tensor<2xf64>, tensor<2x2xf64>
  }
  func.func private @solve_155(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=2, j=2, k=2, l=2, m=2}, custom>} : (tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<2xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<2x2xi1>
    %8 = stablehlo.select %7, %0#0, %6 : tensor<2x2xi1>, tensor<2x2xf64>
    %9 = stablehlo.iota dim = 0 : tensor<2xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %10:4 = stablehlo.while(%iterArg = %2, %iterArg_3 = %c_2, %iterArg_4 = %c_1, %iterArg_5 = %9) : tensor<2xi32>, tensor<i64>, tensor<i64>, tensor<2xi32>
    cond {
      %c_6 = stablehlo.constant dense<2> : tensor<i64>
      %14 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14:2 = func.call @closed_call_162(%iterArg, %iterArg_4, %iterArg_5) : (tensor<2xi32>, tensor<i64>, tensor<2xi32>) -> (tensor<i64>, tensor<2xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %15 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %15, %14#0, %14#1 : tensor<2xi32>, tensor<i64>, tensor<i64>, tensor<2xi32>
    }
    %11 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %12 = call @_lu_solve_167(%8, %10#3, %11) : (tensor<2x2xf64>, tensor<2xi32>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    return %13 : tensor<2x2xf64>
  }
  func.func private @closed_call_162(%arg0: tensor<2xi32>, %arg1: tensor<i64>, %arg2: tensor<2xi32>) -> (tensor<i64>, tensor<2xi32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg1, %c : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare  LT, %arg1, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = stablehlo.convert %arg1 : tensor<i64>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %3 = stablehlo.add %2, %c_1 : tensor<i64>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<i1>, tensor<i64>
    %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<2xi32>, tensor<i64>) -> tensor<1xi32>
    %6 = stablehlo.reshape %5 : (tensor<1xi32>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.compare  LT, %arg1, %c_2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8 = stablehlo.convert %arg1 : tensor<i64>
    %c_3 = stablehlo.constant dense<2> : tensor<i64>
    %9 = stablehlo.add %8, %c_3 : tensor<i64>
    %10 = stablehlo.select %7, %9, %arg1 : tensor<i1>, tensor<i64>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<2xi32>, tensor<i64>) -> tensor<1xi32>
    %12 = stablehlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %13 = stablehlo.compare  LT, %6, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<2> : tensor<i32>
    %14 = stablehlo.add %6, %c_5 : tensor<i32>
    %15 = stablehlo.select %13, %14, %6 : tensor<i1>, tensor<i32>
    %16 = stablehlo.dynamic_slice %arg2, %15, sizes = [1] : (tensor<2xi32>, tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %18 = stablehlo.compare  LT, %arg1, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_7 = stablehlo.constant dense<2> : tensor<i64>
    %19 = stablehlo.add %arg1, %c_7 : tensor<i64>
    %20 = stablehlo.select %18, %19, %arg1 : tensor<i1>, tensor<i64>
    %21 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %23 = "stablehlo.scatter"(%arg2, %22, %17) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %24 = stablehlo.compare  LT, %6, %c_8,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_9 = stablehlo.constant dense<2> : tensor<i32>
    %25 = stablehlo.add %6, %c_9 : tensor<i32>
    %26 = stablehlo.select %24, %25, %6 : tensor<i1>, tensor<i32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = "stablehlo.scatter"(%23, %27, %12) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      stablehlo.return %arg4 : tensor<i32>
    }) : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
    return %0, %28 : tensor<i64>, tensor<2xi32>
  }
  func.func private @_lu_solve_167(%arg0: tensor<2x2xf64>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<2x2xf64>) -> tensor<2x2x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<2xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<2xi1>, tensor<2xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1, 1>}> : (tensor<2x2x1xf64>, tensor<2x1xi32>) -> tensor<2x2x1xf64>
    %8 = stablehlo.transpose %7, dims = [1, 2, 0] : (tensor<2x2x1xf64>) -> tensor<2x1x2xf64>
    %9 = stablehlo.reshape %8 : (tensor<2x1x2xf64>) -> tensor<2x2xf64>
    %10 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %9) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=2, j=2, k=2, l=2, m=2, n=2}, custom>} : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %11 = stablehlo.reshape %10 : (tensor<2x2xf64>) -> tensor<2x1x2xf64>
    %12 = stablehlo.reshape %11 : (tensor<2x1x2xf64>) -> tensor<2x2xf64>
    %13 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %12) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=2, j=2, k=2, l=2, m=2, n=2}, custom>} : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %14 = stablehlo.reshape %13 : (tensor<2x2xf64>) -> tensor<2x1x2xf64>
    %15 = stablehlo.slice %14 [0:2, 0:1, 0:2] : (tensor<2x1x2xf64>) -> tensor<2x1x2xf64>
    %16 = stablehlo.transpose %15, dims = [2, 0, 1] : (tensor<2x1x2xf64>) -> tensor<2x2x1xf64>
    %17 = stablehlo.reshape %16 : (tensor<2x2x1xf64>) -> tensor<2x2xf64>
    return %17 : tensor<2x2xf64>
  }
  func.func private @inv(%arg0: tensor<2x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.iota dim = 0 : tensor<2x2xi64>
    %1 = stablehlo.iota dim = 1 : tensor<2x2xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<2x2xi64>
    %3 = stablehlo.add %0, %2 : tensor<2x2xi64>
    %4 = stablehlo.compare  EQ, %3, %1,  SIGNED : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
    %5 = stablehlo.convert %4 : (tensor<2x2xi1>) -> tensor<2x2xf64>
    %6 = call @solve_155(%arg0, %5) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    return %6 : tensor<2x2xf64>
  }
  func.func private @norm_188(%arg0: tensor<2xf64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<2xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.sqrt %1 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @solve_194(%arg0: tensor<2x2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=2, j=2, k=2, l=2, m=2}, custom>} : (tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<2xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<2xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<2x2xi1>
    %8 = stablehlo.select %7, %0#0, %6 : tensor<2x2xi1>, tensor<2x2xf64>
    %9 = stablehlo.iota dim = 0 : tensor<2xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %10:4 = stablehlo.while(%iterArg = %2, %iterArg_3 = %c_2, %iterArg_4 = %c_1, %iterArg_5 = %9) : tensor<2xi32>, tensor<i64>, tensor<i64>, tensor<2xi32>
    cond {
      %c_6 = stablehlo.constant dense<2> : tensor<i64>
      %12 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %12 : tensor<i1>
    } do {
      %12:2 = func.call @closed_call_162(%iterArg, %iterArg_4, %iterArg_5) : (tensor<2xi32>, tensor<i64>, tensor<2xi32>) -> (tensor<i64>, tensor<2xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %13 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %13, %12#0, %12#1 : tensor<2xi32>, tensor<i64>, tensor<i64>, tensor<2xi32>
    }
    %11 = call @_lu_solve_196(%8, %10#3, %arg1) : (tensor<2x2xf64>, tensor<2xi32>, tensor<2xf64>) -> tensor<2xf64>
    return %11 : tensor<2xf64>
  }
  func.func private @_lu_solve_196(%arg0: tensor<2x2xf64>, %arg1: tensor<2xi32>, %arg2: tensor<2xf64>) -> tensor<2xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<2xf64>) -> tensor<2x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<2xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<2xi1>, tensor<2xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<2x1xf64>, tensor<2x1xi32>) -> tensor<2x1xf64>
    %8 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %7) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=2, j=2, k=2, l=1, m=2, n=1}, custom>} : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
    %9 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %8) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=2, j=2, k=2, l=1, m=2, n=1}, custom>} : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
    %10 = stablehlo.slice %9 [0:2, 0:1] : (tensor<2x1xf64>) -> tensor<2x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<2x1xf64>) -> tensor<2xf64>
    return %11 : tensor<2xf64>
  }
  func.func private @inner_203(%arg0: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %cst = stablehlo.constant dense<[[1.000000e+00, 0.0083333333333333332, 0.000000e+00], [0.000000e+00, 1.000000e+00, 0.0083333333333333332], [0.000000e+00, 0.000000e+00, 1.000000e+00]]> : tensor<3x3xf64>
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi64>
    %1 = stablehlo.iota dim = 1 : tensor<3x3xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %3 = stablehlo.add %0, %2 : tensor<3x3xi64>
    %4 = stablehlo.compare  EQ, %3, %1,  SIGNED : (tensor<3x3xi64>, tensor<3x3xi64>) -> tensor<3x3xi1>
    %5 = stablehlo.convert %4 : (tensor<3x3xi1>) -> tensor<3x3xf64>
    %cst_0 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %7 = stablehlo.multiply %6, %5 : tensor<3x3xf64>
    %8 = stablehlo.add %cst, %7 : tensor<3x3xf64>
    %9 = call @solve_204(%8, %arg0) : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
    return %9 : tensor<3x2xf64>
  }
  func.func private @solve_204(%arg0: tensor<3x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0:3 = stablehlo.custom_call @lapack_dgetrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m], []) {i=3, j=3, k=3, l=3, m=3}, custom>} : (tensor<3x3xf64>) -> (tensor<3x3xf64>, tensor<3xi32>, tensor<i32>)
    %c = stablehlo.constant dense<1> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.subtract %0#1, %1 : tensor<3xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0#2, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
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
      %14 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14:2 = func.call @closed_call_66(%iterArg, %iterArg_4, %iterArg_5) : (tensor<3xi32>, tensor<i64>, tensor<3xi32>) -> (tensor<i64>, tensor<3xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %15 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %15, %14#0, %14#1 : tensor<3xi32>, tensor<i64>, tensor<i64>, tensor<3xi32>
    }
    %11 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %12 = call @_lu_solve_207(%8, %10#3, %11) : (tensor<3x3xf64>, tensor<3xi32>, tensor<2x3xf64>) -> tensor<2x3xf64>
    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %13 : tensor<3x2xf64>
  }
  func.func private @_lu_solve_207(%arg0: tensor<3x3xf64>, %arg1: tensor<3xi32>, %arg2: tensor<2x3xf64>) -> tensor<2x3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<2x3xf64>) -> tensor<2x3x1xf64>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %2 = stablehlo.compare  LT, %arg1, %1,  SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    %c_0 = stablehlo.constant dense<3> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3xi32>
    %4 = stablehlo.add %arg1, %3 : tensor<3xi32>
    %5 = stablehlo.select %2, %4, %arg1 : tensor<3xi1>, tensor<3xi32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<3xi32>) -> tensor<3x1xi32>
    %7 = "stablehlo.gather"(%0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1, 1>}> : (tensor<2x3x1xf64>, tensor<3x1xi32>) -> tensor<2x3x1xf64>
    %8 = stablehlo.transpose %7, dims = [1, 2, 0] : (tensor<2x3x1xf64>) -> tensor<3x1x2xf64>
    %9 = stablehlo.reshape %8 : (tensor<3x1x2xf64>) -> tensor<3x2xf64>
    %10 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %9) {backend_config = "", mhlo.backend_config = {diag = 85 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 76 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=3, j=3, k=3, l=2, m=3, n=2}, custom>} : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
    %11 = stablehlo.reshape %10 : (tensor<3x2xf64>) -> tensor<3x1x2xf64>
    %12 = stablehlo.reshape %11 : (tensor<3x1x2xf64>) -> tensor<3x2xf64>
    %13 = stablehlo.custom_call @lapack_dtrsm_ffi(%arg0, %12) {backend_config = "", mhlo.backend_config = {diag = 78 : ui8, side = 76 : ui8, trans_x = 78 : ui8, uplo = 85 : ui8}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=3, j=3, k=3, l=2, m=3, n=2}, custom>} : (tensor<3x3xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
    %14 = stablehlo.reshape %13 : (tensor<3x2xf64>) -> tensor<3x1x2xf64>
    %15 = stablehlo.slice %14 [0:3, 0:1, 0:2] : (tensor<3x1x2xf64>) -> tensor<3x1x2xf64>
    %16 = stablehlo.transpose %15, dims = [2, 0, 1] : (tensor<3x1x2xf64>) -> tensor<2x3x1xf64>
    %17 = stablehlo.reshape %16 : (tensor<2x3x1xf64>) -> tensor<2x3xf64>
    return %17 : tensor<2x3xf64>
  }
  func.func private @inner_219(%arg0: tensor<4xi64>) -> tensor<4xi64> {
    %c = stablehlo.constant dense<[1, 0, 0, 0]> : tensor<4xi64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<4xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %2 = stablehlo.compare  GT, %1, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %3 = stablehlo.slice %arg0 [1:2] : (tensor<4xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %5 = stablehlo.compare  EQ, %4, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %6 = stablehlo.and %2, %5 : tensor<i1>
    %7 = stablehlo.convert %6 : (tensor<i1>) -> tensor<i32>
    %8 = "stablehlo.case"(%7) ({
      stablehlo.return %arg0 : tensor<4xi64>
    }, {
      %19 = stablehlo.add %arg0, %c : tensor<4xi64>
      stablehlo.return %19 : tensor<4xi64>
    }) : (tensor<i32>) -> tensor<4xi64>
    %9 = stablehlo.slice %8 [0:1] : (tensor<4xi64>) -> tensor<1xi64>
    %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
    %c_2 = stablehlo.constant dense<4> : tensor<i64>
    %11 = call @remainder(%10, %c_2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %12 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<4xi64>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %13 = stablehlo.compare  LT, %11, %c_4,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<4> : tensor<i64>
    %14 = stablehlo.add %11, %c_5 : tensor<i64>
    %15 = stablehlo.select %13, %14, %11 : tensor<i1>, tensor<i64>
    %16 = stablehlo.convert %15 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %c_6 = stablehlo.constant dense<1> : tensor<i64>
    %18 = "stablehlo.scatter"(%12, %17, %c_6) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      stablehlo.return %arg2 : tensor<i64>
    }) : (tensor<4xi64>, tensor<1xi32>, tensor<i64>) -> tensor<4xi64>
    return %18 : tensor<4xi64>
  }
}
