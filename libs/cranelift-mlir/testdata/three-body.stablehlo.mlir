module @module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3x6xf64>, %arg2: tensor<3x7xf64>, %arg3: tensor<3x7xf64>, %arg4: tensor<3x6xf64>, %arg5: tensor<f64>, %arg6: tensor<3x6xf64>) -> (tensor<3x6xf64> {jax.result_info = "result[0]"}, tensor<f64> {jax.result_info = "result[1]"}, tensor<i64> {jax.result_info = "result[2]"}, tensor<3x6xf64> {jax.result_info = "result[3]"}, tensor<3x7xf64> {jax.result_info = "result[4]"}, tensor<3x7xf64> {jax.result_info = "result[5]"}, tensor<3x6xf64> {jax.result_info = "result[6]"}) {
    %0 = stablehlo.slice %arg2 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2 = stablehlo.reshape %arg5 : (tensor<f64>) -> tensor<f64>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %4 = stablehlo.multiply %1, %3 : tensor<3xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %6 = stablehlo.multiply %5, %arg6 : tensor<3x6xf64>
    %7 = stablehlo.slice %6 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %9 = stablehlo.divide %7, %8 : tensor<3x3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %11 = stablehlo.concatenate %9, %10, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %12 = stablehlo.slice %11 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %13 = stablehlo.reshape %12 : (tensor<3x1xf64>) -> tensor<3xf64>
    %14 = stablehlo.slice %0 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %15 = stablehlo.reshape %14 : (tensor<3x1xf64>) -> tensor<3xf64>
    %16 = stablehlo.multiply %13, %15 : tensor<3xf64>
    %17 = stablehlo.slice %11 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %18 = stablehlo.reshape %17 : (tensor<3x1xf64>) -> tensor<3xf64>
    %19 = stablehlo.slice %0 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %20 = stablehlo.reshape %19 : (tensor<3x1xf64>) -> tensor<3xf64>
    %21 = stablehlo.multiply %18, %20 : tensor<3xf64>
    %22 = stablehlo.add %16, %21 : tensor<3xf64>
    %23 = stablehlo.slice %11 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %24 = stablehlo.reshape %23 : (tensor<3x1xf64>) -> tensor<3xf64>
    %25 = stablehlo.slice %0 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %26 = stablehlo.reshape %25 : (tensor<3x1xf64>) -> tensor<3xf64>
    %27 = stablehlo.multiply %24, %26 : tensor<3xf64>
    %28 = stablehlo.add %22, %27 : tensor<3xf64>
    %29 = stablehlo.slice %11 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %30 = stablehlo.reshape %29 : (tensor<3x1xf64>) -> tensor<3xf64>
    %31 = stablehlo.slice %0 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %32 = stablehlo.reshape %31 : (tensor<3x1xf64>) -> tensor<3xf64>
    %33 = stablehlo.multiply %30, %32 : tensor<3xf64>
    %34 = stablehlo.subtract %28, %33 : tensor<3xf64>
    %35 = stablehlo.reshape %34 : (tensor<3xf64>) -> tensor<3x1xf64>
    %36 = stablehlo.multiply %13, %32 : tensor<3xf64>
    %37 = stablehlo.multiply %18, %26 : tensor<3xf64>
    %38 = stablehlo.subtract %36, %37 : tensor<3xf64>
    %39 = stablehlo.multiply %24, %20 : tensor<3xf64>
    %40 = stablehlo.add %38, %39 : tensor<3xf64>
    %41 = stablehlo.multiply %30, %15 : tensor<3xf64>
    %42 = stablehlo.add %40, %41 : tensor<3xf64>
    %43 = stablehlo.reshape %42 : (tensor<3xf64>) -> tensor<3x1xf64>
    %44 = stablehlo.multiply %13, %26 : tensor<3xf64>
    %45 = stablehlo.multiply %18, %32 : tensor<3xf64>
    %46 = stablehlo.add %44, %45 : tensor<3xf64>
    %47 = stablehlo.multiply %24, %15 : tensor<3xf64>
    %48 = stablehlo.subtract %46, %47 : tensor<3xf64>
    %49 = stablehlo.multiply %30, %20 : tensor<3xf64>
    %50 = stablehlo.add %48, %49 : tensor<3xf64>
    %51 = stablehlo.reshape %50 : (tensor<3xf64>) -> tensor<3x1xf64>
    %52 = stablehlo.multiply %13, %20 : tensor<3xf64>
    %53 = stablehlo.multiply %18, %15 : tensor<3xf64>
    %54 = stablehlo.subtract %52, %53 : tensor<3xf64>
    %55 = stablehlo.multiply %24, %32 : tensor<3xf64>
    %56 = stablehlo.subtract %54, %55 : tensor<3xf64>
    %57 = stablehlo.multiply %30, %26 : tensor<3xf64>
    %58 = stablehlo.subtract %56, %57 : tensor<3xf64>
    %59 = stablehlo.reshape %58 : (tensor<3xf64>) -> tensor<3x1xf64>
    %60 = stablehlo.concatenate %35, %43, %51, %59, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %61 = stablehlo.add %0, %60 : tensor<3x4xf64>
    %62 = stablehlo.dot_general %61, %61, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %63 = stablehlo.sqrt %62 : tensor<3xf64>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %65 = stablehlo.divide %61, %64 : tensor<3x4xf64>
    %66 = stablehlo.slice %arg2 [0:3, 4:7] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %67 = stablehlo.slice %6 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %68 = stablehlo.add %66, %67 : tensor<3x3xf64>
    %69 = stablehlo.concatenate %65, %68, dim = 1 : (tensor<3x4xf64>, tensor<3x3xf64>) -> tensor<3x7xf64>
    %70 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %71 = stablehlo.multiply %70, %arg4 : tensor<3x6xf64>
    %72 = stablehlo.add %arg6, %71 : tensor<3x6xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %73 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %74 = call @inner(%69, %arg3, %73) : (tensor<3x7xf64>, tensor<3x7xf64>, tensor<3x6xf64>) -> tensor<3x6xf64>
    %75 = stablehlo.slice %69 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %76 = stablehlo.slice %75 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %77 = stablehlo.reshape %76 : (tensor<3x1xf64>) -> tensor<3xf64>
    %78 = stablehlo.slice %75 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %79 = stablehlo.reshape %78 : (tensor<3x1xf64>) -> tensor<3xf64>
    %80 = stablehlo.negate %79 : tensor<3xf64>
    %81 = stablehlo.reshape %80 : (tensor<3xf64>) -> tensor<3x1xf64>
    %82 = stablehlo.slice %75 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %83 = stablehlo.reshape %82 : (tensor<3x1xf64>) -> tensor<3xf64>
    %84 = stablehlo.negate %83 : tensor<3xf64>
    %85 = stablehlo.reshape %84 : (tensor<3xf64>) -> tensor<3x1xf64>
    %86 = stablehlo.slice %75 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %87 = stablehlo.reshape %86 : (tensor<3x1xf64>) -> tensor<3xf64>
    %88 = stablehlo.negate %87 : tensor<3xf64>
    %89 = stablehlo.reshape %88 : (tensor<3xf64>) -> tensor<3x1xf64>
    %90 = stablehlo.slice %75 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %91 = stablehlo.reshape %90 : (tensor<3x1xf64>) -> tensor<3xf64>
    %92 = stablehlo.reshape %91 : (tensor<3xf64>) -> tensor<3x1xf64>
    %93 = stablehlo.concatenate %81, %85, %89, %92, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %94 = stablehlo.dot_general %75, %75, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %96 = stablehlo.divide %93, %95 : tensor<3x4xf64>
    %97 = stablehlo.slice %96 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %98 = stablehlo.reshape %97 : (tensor<3x1xf64>) -> tensor<3xf64>
    %99 = stablehlo.slice %74 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %100 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %101 = stablehlo.concatenate %99, %100, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %102 = stablehlo.slice %101 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %103 = stablehlo.reshape %102 : (tensor<3x1xf64>) -> tensor<3xf64>
    %104 = stablehlo.multiply %98, %103 : tensor<3xf64>
    %105 = stablehlo.slice %96 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %106 = stablehlo.reshape %105 : (tensor<3x1xf64>) -> tensor<3xf64>
    %107 = stablehlo.slice %101 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %108 = stablehlo.reshape %107 : (tensor<3x1xf64>) -> tensor<3xf64>
    %109 = stablehlo.multiply %106, %108 : tensor<3xf64>
    %110 = stablehlo.add %104, %109 : tensor<3xf64>
    %111 = stablehlo.slice %96 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %112 = stablehlo.reshape %111 : (tensor<3x1xf64>) -> tensor<3xf64>
    %113 = stablehlo.slice %101 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %114 = stablehlo.reshape %113 : (tensor<3x1xf64>) -> tensor<3xf64>
    %115 = stablehlo.multiply %112, %114 : tensor<3xf64>
    %116 = stablehlo.add %110, %115 : tensor<3xf64>
    %117 = stablehlo.slice %96 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %118 = stablehlo.reshape %117 : (tensor<3x1xf64>) -> tensor<3xf64>
    %119 = stablehlo.slice %101 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %120 = stablehlo.reshape %119 : (tensor<3x1xf64>) -> tensor<3xf64>
    %121 = stablehlo.multiply %118, %120 : tensor<3xf64>
    %122 = stablehlo.subtract %116, %121 : tensor<3xf64>
    %123 = stablehlo.reshape %122 : (tensor<3xf64>) -> tensor<3x1xf64>
    %124 = stablehlo.multiply %98, %120 : tensor<3xf64>
    %125 = stablehlo.multiply %106, %114 : tensor<3xf64>
    %126 = stablehlo.subtract %124, %125 : tensor<3xf64>
    %127 = stablehlo.multiply %112, %108 : tensor<3xf64>
    %128 = stablehlo.add %126, %127 : tensor<3xf64>
    %129 = stablehlo.multiply %118, %103 : tensor<3xf64>
    %130 = stablehlo.add %128, %129 : tensor<3xf64>
    %131 = stablehlo.reshape %130 : (tensor<3xf64>) -> tensor<3x1xf64>
    %132 = stablehlo.multiply %98, %114 : tensor<3xf64>
    %133 = stablehlo.multiply %106, %120 : tensor<3xf64>
    %134 = stablehlo.add %132, %133 : tensor<3xf64>
    %135 = stablehlo.multiply %112, %103 : tensor<3xf64>
    %136 = stablehlo.subtract %134, %135 : tensor<3xf64>
    %137 = stablehlo.multiply %118, %108 : tensor<3xf64>
    %138 = stablehlo.add %136, %137 : tensor<3xf64>
    %139 = stablehlo.reshape %138 : (tensor<3xf64>) -> tensor<3x1xf64>
    %140 = stablehlo.multiply %98, %108 : tensor<3xf64>
    %141 = stablehlo.multiply %106, %103 : tensor<3xf64>
    %142 = stablehlo.subtract %140, %141 : tensor<3xf64>
    %143 = stablehlo.multiply %112, %120 : tensor<3xf64>
    %144 = stablehlo.subtract %142, %143 : tensor<3xf64>
    %145 = stablehlo.multiply %118, %114 : tensor<3xf64>
    %146 = stablehlo.subtract %144, %145 : tensor<3xf64>
    %147 = stablehlo.reshape %146 : (tensor<3xf64>) -> tensor<3x1xf64>
    %148 = stablehlo.concatenate %123, %131, %139, %147, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %149 = stablehlo.slice %148 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %150 = stablehlo.reshape %149 : (tensor<3x1xf64>) -> tensor<3xf64>
    %151 = stablehlo.slice %96 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %152 = stablehlo.reshape %151 : (tensor<3x1xf64>) -> tensor<3xf64>
    %153 = stablehlo.negate %152 : tensor<3xf64>
    %154 = stablehlo.reshape %153 : (tensor<3xf64>) -> tensor<3x1xf64>
    %155 = stablehlo.slice %96 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %156 = stablehlo.reshape %155 : (tensor<3x1xf64>) -> tensor<3xf64>
    %157 = stablehlo.negate %156 : tensor<3xf64>
    %158 = stablehlo.reshape %157 : (tensor<3xf64>) -> tensor<3x1xf64>
    %159 = stablehlo.slice %96 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %160 = stablehlo.reshape %159 : (tensor<3x1xf64>) -> tensor<3xf64>
    %161 = stablehlo.negate %160 : tensor<3xf64>
    %162 = stablehlo.reshape %161 : (tensor<3xf64>) -> tensor<3x1xf64>
    %163 = stablehlo.slice %96 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %164 = stablehlo.reshape %163 : (tensor<3x1xf64>) -> tensor<3xf64>
    %165 = stablehlo.reshape %164 : (tensor<3xf64>) -> tensor<3x1xf64>
    %166 = stablehlo.concatenate %154, %158, %162, %165, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %167 = stablehlo.dot_general %96, %96, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %169 = stablehlo.divide %166, %168 : tensor<3x4xf64>
    %170 = stablehlo.slice %169 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %171 = stablehlo.reshape %170 : (tensor<3x1xf64>) -> tensor<3xf64>
    %172 = stablehlo.multiply %150, %171 : tensor<3xf64>
    %173 = stablehlo.slice %148 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %174 = stablehlo.reshape %173 : (tensor<3x1xf64>) -> tensor<3xf64>
    %175 = stablehlo.slice %169 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %176 = stablehlo.reshape %175 : (tensor<3x1xf64>) -> tensor<3xf64>
    %177 = stablehlo.multiply %174, %176 : tensor<3xf64>
    %178 = stablehlo.add %172, %177 : tensor<3xf64>
    %179 = stablehlo.slice %148 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %180 = stablehlo.reshape %179 : (tensor<3x1xf64>) -> tensor<3xf64>
    %181 = stablehlo.slice %169 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %182 = stablehlo.reshape %181 : (tensor<3x1xf64>) -> tensor<3xf64>
    %183 = stablehlo.multiply %180, %182 : tensor<3xf64>
    %184 = stablehlo.add %178, %183 : tensor<3xf64>
    %185 = stablehlo.slice %148 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %186 = stablehlo.reshape %185 : (tensor<3x1xf64>) -> tensor<3xf64>
    %187 = stablehlo.slice %169 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %188 = stablehlo.reshape %187 : (tensor<3x1xf64>) -> tensor<3xf64>
    %189 = stablehlo.multiply %186, %188 : tensor<3xf64>
    %190 = stablehlo.subtract %184, %189 : tensor<3xf64>
    %191 = stablehlo.reshape %190 : (tensor<3xf64>) -> tensor<3x1xf64>
    %192 = stablehlo.multiply %150, %188 : tensor<3xf64>
    %193 = stablehlo.multiply %174, %182 : tensor<3xf64>
    %194 = stablehlo.subtract %192, %193 : tensor<3xf64>
    %195 = stablehlo.multiply %180, %176 : tensor<3xf64>
    %196 = stablehlo.add %194, %195 : tensor<3xf64>
    %197 = stablehlo.multiply %186, %171 : tensor<3xf64>
    %198 = stablehlo.add %196, %197 : tensor<3xf64>
    %199 = stablehlo.reshape %198 : (tensor<3xf64>) -> tensor<3x1xf64>
    %200 = stablehlo.multiply %150, %182 : tensor<3xf64>
    %201 = stablehlo.multiply %174, %188 : tensor<3xf64>
    %202 = stablehlo.add %200, %201 : tensor<3xf64>
    %203 = stablehlo.multiply %180, %171 : tensor<3xf64>
    %204 = stablehlo.subtract %202, %203 : tensor<3xf64>
    %205 = stablehlo.multiply %186, %176 : tensor<3xf64>
    %206 = stablehlo.add %204, %205 : tensor<3xf64>
    %207 = stablehlo.reshape %206 : (tensor<3xf64>) -> tensor<3x1xf64>
    %208 = stablehlo.multiply %150, %176 : tensor<3xf64>
    %209 = stablehlo.multiply %174, %171 : tensor<3xf64>
    %210 = stablehlo.subtract %208, %209 : tensor<3xf64>
    %211 = stablehlo.multiply %180, %188 : tensor<3xf64>
    %212 = stablehlo.subtract %210, %211 : tensor<3xf64>
    %213 = stablehlo.multiply %186, %182 : tensor<3xf64>
    %214 = stablehlo.subtract %212, %213 : tensor<3xf64>
    %215 = stablehlo.reshape %214 : (tensor<3xf64>) -> tensor<3x1xf64>
    %216 = stablehlo.concatenate %191, %199, %207, %215, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %217 = stablehlo.slice %216 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %218 = stablehlo.reshape %217 : (tensor<3x1xf64>) -> tensor<3xf64>
    %219 = stablehlo.reshape %218 : (tensor<3xf64>) -> tensor<3x1xf64>
    %220 = stablehlo.slice %216 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %221 = stablehlo.reshape %220 : (tensor<3x1xf64>) -> tensor<3xf64>
    %222 = stablehlo.reshape %221 : (tensor<3xf64>) -> tensor<3x1xf64>
    %223 = stablehlo.slice %216 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %224 = stablehlo.reshape %223 : (tensor<3x1xf64>) -> tensor<3xf64>
    %225 = stablehlo.reshape %224 : (tensor<3xf64>) -> tensor<3x1xf64>
    %226 = stablehlo.concatenate %219, %222, %225, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %227 = stablehlo.slice %96 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %228 = stablehlo.reshape %227 : (tensor<3x1xf64>) -> tensor<3xf64>
    %229 = stablehlo.slice %74 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %230 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %231 = stablehlo.concatenate %229, %230, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %232 = stablehlo.slice %231 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %233 = stablehlo.reshape %232 : (tensor<3x1xf64>) -> tensor<3xf64>
    %234 = stablehlo.multiply %228, %233 : tensor<3xf64>
    %235 = stablehlo.slice %96 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %236 = stablehlo.reshape %235 : (tensor<3x1xf64>) -> tensor<3xf64>
    %237 = stablehlo.slice %231 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %238 = stablehlo.reshape %237 : (tensor<3x1xf64>) -> tensor<3xf64>
    %239 = stablehlo.multiply %236, %238 : tensor<3xf64>
    %240 = stablehlo.add %234, %239 : tensor<3xf64>
    %241 = stablehlo.slice %96 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %242 = stablehlo.reshape %241 : (tensor<3x1xf64>) -> tensor<3xf64>
    %243 = stablehlo.slice %231 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %244 = stablehlo.reshape %243 : (tensor<3x1xf64>) -> tensor<3xf64>
    %245 = stablehlo.multiply %242, %244 : tensor<3xf64>
    %246 = stablehlo.add %240, %245 : tensor<3xf64>
    %247 = stablehlo.slice %96 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %248 = stablehlo.reshape %247 : (tensor<3x1xf64>) -> tensor<3xf64>
    %249 = stablehlo.slice %231 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %250 = stablehlo.reshape %249 : (tensor<3x1xf64>) -> tensor<3xf64>
    %251 = stablehlo.multiply %248, %250 : tensor<3xf64>
    %252 = stablehlo.subtract %246, %251 : tensor<3xf64>
    %253 = stablehlo.reshape %252 : (tensor<3xf64>) -> tensor<3x1xf64>
    %254 = stablehlo.multiply %228, %250 : tensor<3xf64>
    %255 = stablehlo.multiply %236, %244 : tensor<3xf64>
    %256 = stablehlo.subtract %254, %255 : tensor<3xf64>
    %257 = stablehlo.multiply %242, %238 : tensor<3xf64>
    %258 = stablehlo.add %256, %257 : tensor<3xf64>
    %259 = stablehlo.multiply %248, %233 : tensor<3xf64>
    %260 = stablehlo.add %258, %259 : tensor<3xf64>
    %261 = stablehlo.reshape %260 : (tensor<3xf64>) -> tensor<3x1xf64>
    %262 = stablehlo.multiply %228, %244 : tensor<3xf64>
    %263 = stablehlo.multiply %236, %250 : tensor<3xf64>
    %264 = stablehlo.add %262, %263 : tensor<3xf64>
    %265 = stablehlo.multiply %242, %233 : tensor<3xf64>
    %266 = stablehlo.subtract %264, %265 : tensor<3xf64>
    %267 = stablehlo.multiply %248, %238 : tensor<3xf64>
    %268 = stablehlo.add %266, %267 : tensor<3xf64>
    %269 = stablehlo.reshape %268 : (tensor<3xf64>) -> tensor<3x1xf64>
    %270 = stablehlo.multiply %228, %238 : tensor<3xf64>
    %271 = stablehlo.multiply %236, %233 : tensor<3xf64>
    %272 = stablehlo.subtract %270, %271 : tensor<3xf64>
    %273 = stablehlo.multiply %242, %250 : tensor<3xf64>
    %274 = stablehlo.subtract %272, %273 : tensor<3xf64>
    %275 = stablehlo.multiply %248, %244 : tensor<3xf64>
    %276 = stablehlo.subtract %274, %275 : tensor<3xf64>
    %277 = stablehlo.reshape %276 : (tensor<3xf64>) -> tensor<3x1xf64>
    %278 = stablehlo.concatenate %253, %261, %269, %277, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %279 = stablehlo.slice %278 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %280 = stablehlo.reshape %279 : (tensor<3x1xf64>) -> tensor<3xf64>
    %281 = stablehlo.slice %96 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %282 = stablehlo.reshape %281 : (tensor<3x1xf64>) -> tensor<3xf64>
    %283 = stablehlo.negate %282 : tensor<3xf64>
    %284 = stablehlo.reshape %283 : (tensor<3xf64>) -> tensor<3x1xf64>
    %285 = stablehlo.slice %96 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %286 = stablehlo.reshape %285 : (tensor<3x1xf64>) -> tensor<3xf64>
    %287 = stablehlo.negate %286 : tensor<3xf64>
    %288 = stablehlo.reshape %287 : (tensor<3xf64>) -> tensor<3x1xf64>
    %289 = stablehlo.slice %96 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %290 = stablehlo.reshape %289 : (tensor<3x1xf64>) -> tensor<3xf64>
    %291 = stablehlo.negate %290 : tensor<3xf64>
    %292 = stablehlo.reshape %291 : (tensor<3xf64>) -> tensor<3x1xf64>
    %293 = stablehlo.slice %96 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %294 = stablehlo.reshape %293 : (tensor<3x1xf64>) -> tensor<3xf64>
    %295 = stablehlo.reshape %294 : (tensor<3xf64>) -> tensor<3x1xf64>
    %296 = stablehlo.concatenate %284, %288, %292, %295, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %297 = stablehlo.dot_general %96, %96, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %299 = stablehlo.divide %296, %298 : tensor<3x4xf64>
    %300 = stablehlo.slice %299 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %301 = stablehlo.reshape %300 : (tensor<3x1xf64>) -> tensor<3xf64>
    %302 = stablehlo.multiply %280, %301 : tensor<3xf64>
    %303 = stablehlo.slice %278 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %304 = stablehlo.reshape %303 : (tensor<3x1xf64>) -> tensor<3xf64>
    %305 = stablehlo.slice %299 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %306 = stablehlo.reshape %305 : (tensor<3x1xf64>) -> tensor<3xf64>
    %307 = stablehlo.multiply %304, %306 : tensor<3xf64>
    %308 = stablehlo.add %302, %307 : tensor<3xf64>
    %309 = stablehlo.slice %278 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %310 = stablehlo.reshape %309 : (tensor<3x1xf64>) -> tensor<3xf64>
    %311 = stablehlo.slice %299 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %312 = stablehlo.reshape %311 : (tensor<3x1xf64>) -> tensor<3xf64>
    %313 = stablehlo.multiply %310, %312 : tensor<3xf64>
    %314 = stablehlo.add %308, %313 : tensor<3xf64>
    %315 = stablehlo.slice %278 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %316 = stablehlo.reshape %315 : (tensor<3x1xf64>) -> tensor<3xf64>
    %317 = stablehlo.slice %299 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %318 = stablehlo.reshape %317 : (tensor<3x1xf64>) -> tensor<3xf64>
    %319 = stablehlo.multiply %316, %318 : tensor<3xf64>
    %320 = stablehlo.subtract %314, %319 : tensor<3xf64>
    %321 = stablehlo.reshape %320 : (tensor<3xf64>) -> tensor<3x1xf64>
    %322 = stablehlo.multiply %280, %318 : tensor<3xf64>
    %323 = stablehlo.multiply %304, %312 : tensor<3xf64>
    %324 = stablehlo.subtract %322, %323 : tensor<3xf64>
    %325 = stablehlo.multiply %310, %306 : tensor<3xf64>
    %326 = stablehlo.add %324, %325 : tensor<3xf64>
    %327 = stablehlo.multiply %316, %301 : tensor<3xf64>
    %328 = stablehlo.add %326, %327 : tensor<3xf64>
    %329 = stablehlo.reshape %328 : (tensor<3xf64>) -> tensor<3x1xf64>
    %330 = stablehlo.multiply %280, %312 : tensor<3xf64>
    %331 = stablehlo.multiply %304, %318 : tensor<3xf64>
    %332 = stablehlo.add %330, %331 : tensor<3xf64>
    %333 = stablehlo.multiply %310, %301 : tensor<3xf64>
    %334 = stablehlo.subtract %332, %333 : tensor<3xf64>
    %335 = stablehlo.multiply %316, %306 : tensor<3xf64>
    %336 = stablehlo.add %334, %335 : tensor<3xf64>
    %337 = stablehlo.reshape %336 : (tensor<3xf64>) -> tensor<3x1xf64>
    %338 = stablehlo.multiply %280, %306 : tensor<3xf64>
    %339 = stablehlo.multiply %304, %301 : tensor<3xf64>
    %340 = stablehlo.subtract %338, %339 : tensor<3xf64>
    %341 = stablehlo.multiply %310, %318 : tensor<3xf64>
    %342 = stablehlo.subtract %340, %341 : tensor<3xf64>
    %343 = stablehlo.multiply %316, %312 : tensor<3xf64>
    %344 = stablehlo.subtract %342, %343 : tensor<3xf64>
    %345 = stablehlo.reshape %344 : (tensor<3xf64>) -> tensor<3x1xf64>
    %346 = stablehlo.concatenate %321, %329, %337, %345, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %347 = stablehlo.slice %346 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %348 = stablehlo.reshape %347 : (tensor<3x1xf64>) -> tensor<3xf64>
    %349 = stablehlo.reshape %348 : (tensor<3xf64>) -> tensor<3x1xf64>
    %350 = stablehlo.slice %346 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %351 = stablehlo.reshape %350 : (tensor<3x1xf64>) -> tensor<3xf64>
    %352 = stablehlo.reshape %351 : (tensor<3xf64>) -> tensor<3x1xf64>
    %353 = stablehlo.slice %346 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %354 = stablehlo.reshape %353 : (tensor<3x1xf64>) -> tensor<3xf64>
    %355 = stablehlo.reshape %354 : (tensor<3xf64>) -> tensor<3x1xf64>
    %356 = stablehlo.concatenate %349, %352, %355, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %357 = stablehlo.concatenate %226, %356, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %358 = stablehlo.slice %357 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %359 = stablehlo.slice %arg3 [0:3, 0:3] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %360 = stablehlo.divide %358, %359 : tensor<3x3xf64>
    %361 = stablehlo.slice %357 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %362 = stablehlo.slice %arg3 [0:3, 6:7] : (tensor<3x7xf64>) -> tensor<3x1xf64>
    %363 = stablehlo.reshape %362 : (tensor<3x1xf64>) -> tensor<3xf64>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
    %365 = stablehlo.divide %361, %364 : tensor<3x3xf64>
    %366 = stablehlo.concatenate %360, %365, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %367 = stablehlo.slice %366 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %368 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %369 = stablehlo.concatenate %367, %368, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %370 = stablehlo.slice %369 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %371 = stablehlo.reshape %370 : (tensor<3x1xf64>) -> tensor<3xf64>
    %372 = stablehlo.multiply %77, %371 : tensor<3xf64>
    %373 = stablehlo.slice %75 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %374 = stablehlo.reshape %373 : (tensor<3x1xf64>) -> tensor<3xf64>
    %375 = stablehlo.slice %369 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %376 = stablehlo.reshape %375 : (tensor<3x1xf64>) -> tensor<3xf64>
    %377 = stablehlo.multiply %374, %376 : tensor<3xf64>
    %378 = stablehlo.add %372, %377 : tensor<3xf64>
    %379 = stablehlo.slice %75 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %380 = stablehlo.reshape %379 : (tensor<3x1xf64>) -> tensor<3xf64>
    %381 = stablehlo.slice %369 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %382 = stablehlo.reshape %381 : (tensor<3x1xf64>) -> tensor<3xf64>
    %383 = stablehlo.multiply %380, %382 : tensor<3xf64>
    %384 = stablehlo.add %378, %383 : tensor<3xf64>
    %385 = stablehlo.slice %75 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %386 = stablehlo.reshape %385 : (tensor<3x1xf64>) -> tensor<3xf64>
    %387 = stablehlo.slice %369 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %388 = stablehlo.reshape %387 : (tensor<3x1xf64>) -> tensor<3xf64>
    %389 = stablehlo.multiply %386, %388 : tensor<3xf64>
    %390 = stablehlo.subtract %384, %389 : tensor<3xf64>
    %391 = stablehlo.reshape %390 : (tensor<3xf64>) -> tensor<3x1xf64>
    %392 = stablehlo.multiply %77, %388 : tensor<3xf64>
    %393 = stablehlo.multiply %374, %382 : tensor<3xf64>
    %394 = stablehlo.subtract %392, %393 : tensor<3xf64>
    %395 = stablehlo.multiply %380, %376 : tensor<3xf64>
    %396 = stablehlo.add %394, %395 : tensor<3xf64>
    %397 = stablehlo.multiply %386, %371 : tensor<3xf64>
    %398 = stablehlo.add %396, %397 : tensor<3xf64>
    %399 = stablehlo.reshape %398 : (tensor<3xf64>) -> tensor<3x1xf64>
    %400 = stablehlo.multiply %77, %382 : tensor<3xf64>
    %401 = stablehlo.multiply %374, %388 : tensor<3xf64>
    %402 = stablehlo.add %400, %401 : tensor<3xf64>
    %403 = stablehlo.multiply %380, %371 : tensor<3xf64>
    %404 = stablehlo.subtract %402, %403 : tensor<3xf64>
    %405 = stablehlo.multiply %386, %376 : tensor<3xf64>
    %406 = stablehlo.add %404, %405 : tensor<3xf64>
    %407 = stablehlo.reshape %406 : (tensor<3xf64>) -> tensor<3x1xf64>
    %408 = stablehlo.multiply %77, %376 : tensor<3xf64>
    %409 = stablehlo.multiply %374, %371 : tensor<3xf64>
    %410 = stablehlo.subtract %408, %409 : tensor<3xf64>
    %411 = stablehlo.multiply %380, %388 : tensor<3xf64>
    %412 = stablehlo.subtract %410, %411 : tensor<3xf64>
    %413 = stablehlo.multiply %386, %382 : tensor<3xf64>
    %414 = stablehlo.subtract %412, %413 : tensor<3xf64>
    %415 = stablehlo.reshape %414 : (tensor<3xf64>) -> tensor<3x1xf64>
    %416 = stablehlo.concatenate %391, %399, %407, %415, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %417 = stablehlo.slice %416 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %418 = stablehlo.reshape %417 : (tensor<3x1xf64>) -> tensor<3xf64>
    %419 = stablehlo.slice %75 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %420 = stablehlo.reshape %419 : (tensor<3x1xf64>) -> tensor<3xf64>
    %421 = stablehlo.negate %420 : tensor<3xf64>
    %422 = stablehlo.reshape %421 : (tensor<3xf64>) -> tensor<3x1xf64>
    %423 = stablehlo.slice %75 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %424 = stablehlo.reshape %423 : (tensor<3x1xf64>) -> tensor<3xf64>
    %425 = stablehlo.negate %424 : tensor<3xf64>
    %426 = stablehlo.reshape %425 : (tensor<3xf64>) -> tensor<3x1xf64>
    %427 = stablehlo.slice %75 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %428 = stablehlo.reshape %427 : (tensor<3x1xf64>) -> tensor<3xf64>
    %429 = stablehlo.negate %428 : tensor<3xf64>
    %430 = stablehlo.reshape %429 : (tensor<3xf64>) -> tensor<3x1xf64>
    %431 = stablehlo.slice %75 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %432 = stablehlo.reshape %431 : (tensor<3x1xf64>) -> tensor<3xf64>
    %433 = stablehlo.reshape %432 : (tensor<3xf64>) -> tensor<3x1xf64>
    %434 = stablehlo.concatenate %422, %426, %430, %433, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %435 = stablehlo.dot_general %75, %75, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %436 = stablehlo.broadcast_in_dim %435, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %437 = stablehlo.divide %434, %436 : tensor<3x4xf64>
    %438 = stablehlo.slice %437 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %439 = stablehlo.reshape %438 : (tensor<3x1xf64>) -> tensor<3xf64>
    %440 = stablehlo.multiply %418, %439 : tensor<3xf64>
    %441 = stablehlo.slice %416 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %442 = stablehlo.reshape %441 : (tensor<3x1xf64>) -> tensor<3xf64>
    %443 = stablehlo.slice %437 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %444 = stablehlo.reshape %443 : (tensor<3x1xf64>) -> tensor<3xf64>
    %445 = stablehlo.multiply %442, %444 : tensor<3xf64>
    %446 = stablehlo.add %440, %445 : tensor<3xf64>
    %447 = stablehlo.slice %416 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %448 = stablehlo.reshape %447 : (tensor<3x1xf64>) -> tensor<3xf64>
    %449 = stablehlo.slice %437 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %450 = stablehlo.reshape %449 : (tensor<3x1xf64>) -> tensor<3xf64>
    %451 = stablehlo.multiply %448, %450 : tensor<3xf64>
    %452 = stablehlo.add %446, %451 : tensor<3xf64>
    %453 = stablehlo.slice %416 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %454 = stablehlo.reshape %453 : (tensor<3x1xf64>) -> tensor<3xf64>
    %455 = stablehlo.slice %437 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %456 = stablehlo.reshape %455 : (tensor<3x1xf64>) -> tensor<3xf64>
    %457 = stablehlo.multiply %454, %456 : tensor<3xf64>
    %458 = stablehlo.subtract %452, %457 : tensor<3xf64>
    %459 = stablehlo.reshape %458 : (tensor<3xf64>) -> tensor<3x1xf64>
    %460 = stablehlo.multiply %418, %456 : tensor<3xf64>
    %461 = stablehlo.multiply %442, %450 : tensor<3xf64>
    %462 = stablehlo.subtract %460, %461 : tensor<3xf64>
    %463 = stablehlo.multiply %448, %444 : tensor<3xf64>
    %464 = stablehlo.add %462, %463 : tensor<3xf64>
    %465 = stablehlo.multiply %454, %439 : tensor<3xf64>
    %466 = stablehlo.add %464, %465 : tensor<3xf64>
    %467 = stablehlo.reshape %466 : (tensor<3xf64>) -> tensor<3x1xf64>
    %468 = stablehlo.multiply %418, %450 : tensor<3xf64>
    %469 = stablehlo.multiply %442, %456 : tensor<3xf64>
    %470 = stablehlo.add %468, %469 : tensor<3xf64>
    %471 = stablehlo.multiply %448, %439 : tensor<3xf64>
    %472 = stablehlo.subtract %470, %471 : tensor<3xf64>
    %473 = stablehlo.multiply %454, %444 : tensor<3xf64>
    %474 = stablehlo.add %472, %473 : tensor<3xf64>
    %475 = stablehlo.reshape %474 : (tensor<3xf64>) -> tensor<3x1xf64>
    %476 = stablehlo.multiply %418, %444 : tensor<3xf64>
    %477 = stablehlo.multiply %442, %439 : tensor<3xf64>
    %478 = stablehlo.subtract %476, %477 : tensor<3xf64>
    %479 = stablehlo.multiply %448, %456 : tensor<3xf64>
    %480 = stablehlo.subtract %478, %479 : tensor<3xf64>
    %481 = stablehlo.multiply %454, %450 : tensor<3xf64>
    %482 = stablehlo.subtract %480, %481 : tensor<3xf64>
    %483 = stablehlo.reshape %482 : (tensor<3xf64>) -> tensor<3x1xf64>
    %484 = stablehlo.concatenate %459, %467, %475, %483, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %485 = stablehlo.slice %484 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %486 = stablehlo.reshape %485 : (tensor<3x1xf64>) -> tensor<3xf64>
    %487 = stablehlo.reshape %486 : (tensor<3xf64>) -> tensor<3x1xf64>
    %488 = stablehlo.slice %484 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %489 = stablehlo.reshape %488 : (tensor<3x1xf64>) -> tensor<3xf64>
    %490 = stablehlo.reshape %489 : (tensor<3xf64>) -> tensor<3x1xf64>
    %491 = stablehlo.slice %484 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %492 = stablehlo.reshape %491 : (tensor<3x1xf64>) -> tensor<3xf64>
    %493 = stablehlo.reshape %492 : (tensor<3xf64>) -> tensor<3x1xf64>
    %494 = stablehlo.concatenate %487, %490, %493, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %495 = stablehlo.slice %75 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %496 = stablehlo.reshape %495 : (tensor<3x1xf64>) -> tensor<3xf64>
    %497 = stablehlo.slice %366 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %498 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %499 = stablehlo.concatenate %497, %498, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %500 = stablehlo.slice %499 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %501 = stablehlo.reshape %500 : (tensor<3x1xf64>) -> tensor<3xf64>
    %502 = stablehlo.multiply %496, %501 : tensor<3xf64>
    %503 = stablehlo.slice %75 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %504 = stablehlo.reshape %503 : (tensor<3x1xf64>) -> tensor<3xf64>
    %505 = stablehlo.slice %499 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %506 = stablehlo.reshape %505 : (tensor<3x1xf64>) -> tensor<3xf64>
    %507 = stablehlo.multiply %504, %506 : tensor<3xf64>
    %508 = stablehlo.add %502, %507 : tensor<3xf64>
    %509 = stablehlo.slice %75 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %510 = stablehlo.reshape %509 : (tensor<3x1xf64>) -> tensor<3xf64>
    %511 = stablehlo.slice %499 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %512 = stablehlo.reshape %511 : (tensor<3x1xf64>) -> tensor<3xf64>
    %513 = stablehlo.multiply %510, %512 : tensor<3xf64>
    %514 = stablehlo.add %508, %513 : tensor<3xf64>
    %515 = stablehlo.slice %75 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %516 = stablehlo.reshape %515 : (tensor<3x1xf64>) -> tensor<3xf64>
    %517 = stablehlo.slice %499 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %518 = stablehlo.reshape %517 : (tensor<3x1xf64>) -> tensor<3xf64>
    %519 = stablehlo.multiply %516, %518 : tensor<3xf64>
    %520 = stablehlo.subtract %514, %519 : tensor<3xf64>
    %521 = stablehlo.reshape %520 : (tensor<3xf64>) -> tensor<3x1xf64>
    %522 = stablehlo.multiply %496, %518 : tensor<3xf64>
    %523 = stablehlo.multiply %504, %512 : tensor<3xf64>
    %524 = stablehlo.subtract %522, %523 : tensor<3xf64>
    %525 = stablehlo.multiply %510, %506 : tensor<3xf64>
    %526 = stablehlo.add %524, %525 : tensor<3xf64>
    %527 = stablehlo.multiply %516, %501 : tensor<3xf64>
    %528 = stablehlo.add %526, %527 : tensor<3xf64>
    %529 = stablehlo.reshape %528 : (tensor<3xf64>) -> tensor<3x1xf64>
    %530 = stablehlo.multiply %496, %512 : tensor<3xf64>
    %531 = stablehlo.multiply %504, %518 : tensor<3xf64>
    %532 = stablehlo.add %530, %531 : tensor<3xf64>
    %533 = stablehlo.multiply %510, %501 : tensor<3xf64>
    %534 = stablehlo.subtract %532, %533 : tensor<3xf64>
    %535 = stablehlo.multiply %516, %506 : tensor<3xf64>
    %536 = stablehlo.add %534, %535 : tensor<3xf64>
    %537 = stablehlo.reshape %536 : (tensor<3xf64>) -> tensor<3x1xf64>
    %538 = stablehlo.multiply %496, %506 : tensor<3xf64>
    %539 = stablehlo.multiply %504, %501 : tensor<3xf64>
    %540 = stablehlo.subtract %538, %539 : tensor<3xf64>
    %541 = stablehlo.multiply %510, %518 : tensor<3xf64>
    %542 = stablehlo.subtract %540, %541 : tensor<3xf64>
    %543 = stablehlo.multiply %516, %512 : tensor<3xf64>
    %544 = stablehlo.subtract %542, %543 : tensor<3xf64>
    %545 = stablehlo.reshape %544 : (tensor<3xf64>) -> tensor<3x1xf64>
    %546 = stablehlo.concatenate %521, %529, %537, %545, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %547 = stablehlo.slice %546 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %548 = stablehlo.reshape %547 : (tensor<3x1xf64>) -> tensor<3xf64>
    %549 = stablehlo.slice %75 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %550 = stablehlo.reshape %549 : (tensor<3x1xf64>) -> tensor<3xf64>
    %551 = stablehlo.negate %550 : tensor<3xf64>
    %552 = stablehlo.reshape %551 : (tensor<3xf64>) -> tensor<3x1xf64>
    %553 = stablehlo.slice %75 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %554 = stablehlo.reshape %553 : (tensor<3x1xf64>) -> tensor<3xf64>
    %555 = stablehlo.negate %554 : tensor<3xf64>
    %556 = stablehlo.reshape %555 : (tensor<3xf64>) -> tensor<3x1xf64>
    %557 = stablehlo.slice %75 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %558 = stablehlo.reshape %557 : (tensor<3x1xf64>) -> tensor<3xf64>
    %559 = stablehlo.negate %558 : tensor<3xf64>
    %560 = stablehlo.reshape %559 : (tensor<3xf64>) -> tensor<3x1xf64>
    %561 = stablehlo.slice %75 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %562 = stablehlo.reshape %561 : (tensor<3x1xf64>) -> tensor<3xf64>
    %563 = stablehlo.reshape %562 : (tensor<3xf64>) -> tensor<3x1xf64>
    %564 = stablehlo.concatenate %552, %556, %560, %563, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %565 = stablehlo.dot_general %75, %75, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %566 = stablehlo.broadcast_in_dim %565, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %567 = stablehlo.divide %564, %566 : tensor<3x4xf64>
    %568 = stablehlo.slice %567 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %569 = stablehlo.reshape %568 : (tensor<3x1xf64>) -> tensor<3xf64>
    %570 = stablehlo.multiply %548, %569 : tensor<3xf64>
    %571 = stablehlo.slice %546 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %572 = stablehlo.reshape %571 : (tensor<3x1xf64>) -> tensor<3xf64>
    %573 = stablehlo.slice %567 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %574 = stablehlo.reshape %573 : (tensor<3x1xf64>) -> tensor<3xf64>
    %575 = stablehlo.multiply %572, %574 : tensor<3xf64>
    %576 = stablehlo.add %570, %575 : tensor<3xf64>
    %577 = stablehlo.slice %546 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %578 = stablehlo.reshape %577 : (tensor<3x1xf64>) -> tensor<3xf64>
    %579 = stablehlo.slice %567 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %580 = stablehlo.reshape %579 : (tensor<3x1xf64>) -> tensor<3xf64>
    %581 = stablehlo.multiply %578, %580 : tensor<3xf64>
    %582 = stablehlo.add %576, %581 : tensor<3xf64>
    %583 = stablehlo.slice %546 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %584 = stablehlo.reshape %583 : (tensor<3x1xf64>) -> tensor<3xf64>
    %585 = stablehlo.slice %567 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %586 = stablehlo.reshape %585 : (tensor<3x1xf64>) -> tensor<3xf64>
    %587 = stablehlo.multiply %584, %586 : tensor<3xf64>
    %588 = stablehlo.subtract %582, %587 : tensor<3xf64>
    %589 = stablehlo.reshape %588 : (tensor<3xf64>) -> tensor<3x1xf64>
    %590 = stablehlo.multiply %548, %586 : tensor<3xf64>
    %591 = stablehlo.multiply %572, %580 : tensor<3xf64>
    %592 = stablehlo.subtract %590, %591 : tensor<3xf64>
    %593 = stablehlo.multiply %578, %574 : tensor<3xf64>
    %594 = stablehlo.add %592, %593 : tensor<3xf64>
    %595 = stablehlo.multiply %584, %569 : tensor<3xf64>
    %596 = stablehlo.add %594, %595 : tensor<3xf64>
    %597 = stablehlo.reshape %596 : (tensor<3xf64>) -> tensor<3x1xf64>
    %598 = stablehlo.multiply %548, %580 : tensor<3xf64>
    %599 = stablehlo.multiply %572, %586 : tensor<3xf64>
    %600 = stablehlo.add %598, %599 : tensor<3xf64>
    %601 = stablehlo.multiply %578, %569 : tensor<3xf64>
    %602 = stablehlo.subtract %600, %601 : tensor<3xf64>
    %603 = stablehlo.multiply %584, %574 : tensor<3xf64>
    %604 = stablehlo.add %602, %603 : tensor<3xf64>
    %605 = stablehlo.reshape %604 : (tensor<3xf64>) -> tensor<3x1xf64>
    %606 = stablehlo.multiply %548, %574 : tensor<3xf64>
    %607 = stablehlo.multiply %572, %569 : tensor<3xf64>
    %608 = stablehlo.subtract %606, %607 : tensor<3xf64>
    %609 = stablehlo.multiply %578, %586 : tensor<3xf64>
    %610 = stablehlo.subtract %608, %609 : tensor<3xf64>
    %611 = stablehlo.multiply %584, %580 : tensor<3xf64>
    %612 = stablehlo.subtract %610, %611 : tensor<3xf64>
    %613 = stablehlo.reshape %612 : (tensor<3xf64>) -> tensor<3x1xf64>
    %614 = stablehlo.concatenate %589, %597, %605, %613, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %615 = stablehlo.slice %614 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %616 = stablehlo.reshape %615 : (tensor<3x1xf64>) -> tensor<3xf64>
    %617 = stablehlo.reshape %616 : (tensor<3xf64>) -> tensor<3x1xf64>
    %618 = stablehlo.slice %614 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %619 = stablehlo.reshape %618 : (tensor<3x1xf64>) -> tensor<3xf64>
    %620 = stablehlo.reshape %619 : (tensor<3xf64>) -> tensor<3x1xf64>
    %621 = stablehlo.slice %614 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %622 = stablehlo.reshape %621 : (tensor<3x1xf64>) -> tensor<3xf64>
    %623 = stablehlo.reshape %622 : (tensor<3xf64>) -> tensor<3x1xf64>
    %624 = stablehlo.concatenate %617, %620, %623, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %625 = stablehlo.concatenate %494, %624, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %626 = stablehlo.slice %arg2 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %cst_7 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %627 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %628 = stablehlo.reshape %arg5 : (tensor<f64>) -> tensor<f64>
    %629 = stablehlo.broadcast_in_dim %628, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %630 = stablehlo.multiply %627, %629 : tensor<3xf64>
    %631 = stablehlo.broadcast_in_dim %630, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %632 = stablehlo.multiply %631, %arg6 : tensor<3x6xf64>
    %633 = stablehlo.slice %632 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_8 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %634 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %635 = stablehlo.divide %633, %634 : tensor<3x3xf64>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %636 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %637 = stablehlo.concatenate %635, %636, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %638 = stablehlo.slice %637 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %639 = stablehlo.reshape %638 : (tensor<3x1xf64>) -> tensor<3xf64>
    %640 = stablehlo.slice %626 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %641 = stablehlo.reshape %640 : (tensor<3x1xf64>) -> tensor<3xf64>
    %642 = stablehlo.multiply %639, %641 : tensor<3xf64>
    %643 = stablehlo.slice %637 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %644 = stablehlo.reshape %643 : (tensor<3x1xf64>) -> tensor<3xf64>
    %645 = stablehlo.slice %626 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %646 = stablehlo.reshape %645 : (tensor<3x1xf64>) -> tensor<3xf64>
    %647 = stablehlo.multiply %644, %646 : tensor<3xf64>
    %648 = stablehlo.add %642, %647 : tensor<3xf64>
    %649 = stablehlo.slice %637 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %650 = stablehlo.reshape %649 : (tensor<3x1xf64>) -> tensor<3xf64>
    %651 = stablehlo.slice %626 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %652 = stablehlo.reshape %651 : (tensor<3x1xf64>) -> tensor<3xf64>
    %653 = stablehlo.multiply %650, %652 : tensor<3xf64>
    %654 = stablehlo.add %648, %653 : tensor<3xf64>
    %655 = stablehlo.slice %637 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %656 = stablehlo.reshape %655 : (tensor<3x1xf64>) -> tensor<3xf64>
    %657 = stablehlo.slice %626 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %658 = stablehlo.reshape %657 : (tensor<3x1xf64>) -> tensor<3xf64>
    %659 = stablehlo.multiply %656, %658 : tensor<3xf64>
    %660 = stablehlo.subtract %654, %659 : tensor<3xf64>
    %661 = stablehlo.reshape %660 : (tensor<3xf64>) -> tensor<3x1xf64>
    %662 = stablehlo.multiply %639, %658 : tensor<3xf64>
    %663 = stablehlo.multiply %644, %652 : tensor<3xf64>
    %664 = stablehlo.subtract %662, %663 : tensor<3xf64>
    %665 = stablehlo.multiply %650, %646 : tensor<3xf64>
    %666 = stablehlo.add %664, %665 : tensor<3xf64>
    %667 = stablehlo.multiply %656, %641 : tensor<3xf64>
    %668 = stablehlo.add %666, %667 : tensor<3xf64>
    %669 = stablehlo.reshape %668 : (tensor<3xf64>) -> tensor<3x1xf64>
    %670 = stablehlo.multiply %639, %652 : tensor<3xf64>
    %671 = stablehlo.multiply %644, %658 : tensor<3xf64>
    %672 = stablehlo.add %670, %671 : tensor<3xf64>
    %673 = stablehlo.multiply %650, %641 : tensor<3xf64>
    %674 = stablehlo.subtract %672, %673 : tensor<3xf64>
    %675 = stablehlo.multiply %656, %646 : tensor<3xf64>
    %676 = stablehlo.add %674, %675 : tensor<3xf64>
    %677 = stablehlo.reshape %676 : (tensor<3xf64>) -> tensor<3x1xf64>
    %678 = stablehlo.multiply %639, %646 : tensor<3xf64>
    %679 = stablehlo.multiply %644, %641 : tensor<3xf64>
    %680 = stablehlo.subtract %678, %679 : tensor<3xf64>
    %681 = stablehlo.multiply %650, %658 : tensor<3xf64>
    %682 = stablehlo.subtract %680, %681 : tensor<3xf64>
    %683 = stablehlo.multiply %656, %652 : tensor<3xf64>
    %684 = stablehlo.subtract %682, %683 : tensor<3xf64>
    %685 = stablehlo.reshape %684 : (tensor<3xf64>) -> tensor<3x1xf64>
    %686 = stablehlo.concatenate %661, %669, %677, %685, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %687 = stablehlo.add %626, %686 : tensor<3x4xf64>
    %688 = stablehlo.dot_general %687, %687, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %689 = stablehlo.sqrt %688 : tensor<3xf64>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %691 = stablehlo.divide %687, %690 : tensor<3x4xf64>
    %692 = stablehlo.slice %arg2 [0:3, 4:7] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %693 = stablehlo.slice %632 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %694 = stablehlo.add %692, %693 : tensor<3x3xf64>
    %695 = stablehlo.concatenate %691, %694, dim = 1 : (tensor<3x4xf64>, tensor<3x3xf64>) -> tensor<3x7xf64>
    %696 = stablehlo.broadcast_in_dim %630, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %697 = stablehlo.multiply %696, %625 : tensor<3x6xf64>
    %698 = stablehlo.add %arg6, %697 : tensor<3x6xf64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %699 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %700 = call @inner(%695, %arg3, %699) : (tensor<3x7xf64>, tensor<3x7xf64>, tensor<3x6xf64>) -> tensor<3x6xf64>
    %701 = stablehlo.slice %695 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %702 = stablehlo.slice %701 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %703 = stablehlo.reshape %702 : (tensor<3x1xf64>) -> tensor<3xf64>
    %704 = stablehlo.slice %701 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %705 = stablehlo.reshape %704 : (tensor<3x1xf64>) -> tensor<3xf64>
    %706 = stablehlo.negate %705 : tensor<3xf64>
    %707 = stablehlo.reshape %706 : (tensor<3xf64>) -> tensor<3x1xf64>
    %708 = stablehlo.slice %701 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %709 = stablehlo.reshape %708 : (tensor<3x1xf64>) -> tensor<3xf64>
    %710 = stablehlo.negate %709 : tensor<3xf64>
    %711 = stablehlo.reshape %710 : (tensor<3xf64>) -> tensor<3x1xf64>
    %712 = stablehlo.slice %701 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %713 = stablehlo.reshape %712 : (tensor<3x1xf64>) -> tensor<3xf64>
    %714 = stablehlo.negate %713 : tensor<3xf64>
    %715 = stablehlo.reshape %714 : (tensor<3xf64>) -> tensor<3x1xf64>
    %716 = stablehlo.slice %701 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %717 = stablehlo.reshape %716 : (tensor<3x1xf64>) -> tensor<3xf64>
    %718 = stablehlo.reshape %717 : (tensor<3xf64>) -> tensor<3x1xf64>
    %719 = stablehlo.concatenate %707, %711, %715, %718, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %720 = stablehlo.dot_general %701, %701, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %721 = stablehlo.broadcast_in_dim %720, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %722 = stablehlo.divide %719, %721 : tensor<3x4xf64>
    %723 = stablehlo.slice %722 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %724 = stablehlo.reshape %723 : (tensor<3x1xf64>) -> tensor<3xf64>
    %725 = stablehlo.slice %700 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %726 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %727 = stablehlo.concatenate %725, %726, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %728 = stablehlo.slice %727 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %729 = stablehlo.reshape %728 : (tensor<3x1xf64>) -> tensor<3xf64>
    %730 = stablehlo.multiply %724, %729 : tensor<3xf64>
    %731 = stablehlo.slice %722 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %732 = stablehlo.reshape %731 : (tensor<3x1xf64>) -> tensor<3xf64>
    %733 = stablehlo.slice %727 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %734 = stablehlo.reshape %733 : (tensor<3x1xf64>) -> tensor<3xf64>
    %735 = stablehlo.multiply %732, %734 : tensor<3xf64>
    %736 = stablehlo.add %730, %735 : tensor<3xf64>
    %737 = stablehlo.slice %722 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %738 = stablehlo.reshape %737 : (tensor<3x1xf64>) -> tensor<3xf64>
    %739 = stablehlo.slice %727 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %740 = stablehlo.reshape %739 : (tensor<3x1xf64>) -> tensor<3xf64>
    %741 = stablehlo.multiply %738, %740 : tensor<3xf64>
    %742 = stablehlo.add %736, %741 : tensor<3xf64>
    %743 = stablehlo.slice %722 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %744 = stablehlo.reshape %743 : (tensor<3x1xf64>) -> tensor<3xf64>
    %745 = stablehlo.slice %727 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %746 = stablehlo.reshape %745 : (tensor<3x1xf64>) -> tensor<3xf64>
    %747 = stablehlo.multiply %744, %746 : tensor<3xf64>
    %748 = stablehlo.subtract %742, %747 : tensor<3xf64>
    %749 = stablehlo.reshape %748 : (tensor<3xf64>) -> tensor<3x1xf64>
    %750 = stablehlo.multiply %724, %746 : tensor<3xf64>
    %751 = stablehlo.multiply %732, %740 : tensor<3xf64>
    %752 = stablehlo.subtract %750, %751 : tensor<3xf64>
    %753 = stablehlo.multiply %738, %734 : tensor<3xf64>
    %754 = stablehlo.add %752, %753 : tensor<3xf64>
    %755 = stablehlo.multiply %744, %729 : tensor<3xf64>
    %756 = stablehlo.add %754, %755 : tensor<3xf64>
    %757 = stablehlo.reshape %756 : (tensor<3xf64>) -> tensor<3x1xf64>
    %758 = stablehlo.multiply %724, %740 : tensor<3xf64>
    %759 = stablehlo.multiply %732, %746 : tensor<3xf64>
    %760 = stablehlo.add %758, %759 : tensor<3xf64>
    %761 = stablehlo.multiply %738, %729 : tensor<3xf64>
    %762 = stablehlo.subtract %760, %761 : tensor<3xf64>
    %763 = stablehlo.multiply %744, %734 : tensor<3xf64>
    %764 = stablehlo.add %762, %763 : tensor<3xf64>
    %765 = stablehlo.reshape %764 : (tensor<3xf64>) -> tensor<3x1xf64>
    %766 = stablehlo.multiply %724, %734 : tensor<3xf64>
    %767 = stablehlo.multiply %732, %729 : tensor<3xf64>
    %768 = stablehlo.subtract %766, %767 : tensor<3xf64>
    %769 = stablehlo.multiply %738, %746 : tensor<3xf64>
    %770 = stablehlo.subtract %768, %769 : tensor<3xf64>
    %771 = stablehlo.multiply %744, %740 : tensor<3xf64>
    %772 = stablehlo.subtract %770, %771 : tensor<3xf64>
    %773 = stablehlo.reshape %772 : (tensor<3xf64>) -> tensor<3x1xf64>
    %774 = stablehlo.concatenate %749, %757, %765, %773, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %775 = stablehlo.slice %774 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %776 = stablehlo.reshape %775 : (tensor<3x1xf64>) -> tensor<3xf64>
    %777 = stablehlo.slice %722 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %778 = stablehlo.reshape %777 : (tensor<3x1xf64>) -> tensor<3xf64>
    %779 = stablehlo.negate %778 : tensor<3xf64>
    %780 = stablehlo.reshape %779 : (tensor<3xf64>) -> tensor<3x1xf64>
    %781 = stablehlo.slice %722 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %782 = stablehlo.reshape %781 : (tensor<3x1xf64>) -> tensor<3xf64>
    %783 = stablehlo.negate %782 : tensor<3xf64>
    %784 = stablehlo.reshape %783 : (tensor<3xf64>) -> tensor<3x1xf64>
    %785 = stablehlo.slice %722 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %786 = stablehlo.reshape %785 : (tensor<3x1xf64>) -> tensor<3xf64>
    %787 = stablehlo.negate %786 : tensor<3xf64>
    %788 = stablehlo.reshape %787 : (tensor<3xf64>) -> tensor<3x1xf64>
    %789 = stablehlo.slice %722 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %790 = stablehlo.reshape %789 : (tensor<3x1xf64>) -> tensor<3xf64>
    %791 = stablehlo.reshape %790 : (tensor<3xf64>) -> tensor<3x1xf64>
    %792 = stablehlo.concatenate %780, %784, %788, %791, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %793 = stablehlo.dot_general %722, %722, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %794 = stablehlo.broadcast_in_dim %793, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %795 = stablehlo.divide %792, %794 : tensor<3x4xf64>
    %796 = stablehlo.slice %795 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %797 = stablehlo.reshape %796 : (tensor<3x1xf64>) -> tensor<3xf64>
    %798 = stablehlo.multiply %776, %797 : tensor<3xf64>
    %799 = stablehlo.slice %774 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %800 = stablehlo.reshape %799 : (tensor<3x1xf64>) -> tensor<3xf64>
    %801 = stablehlo.slice %795 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %802 = stablehlo.reshape %801 : (tensor<3x1xf64>) -> tensor<3xf64>
    %803 = stablehlo.multiply %800, %802 : tensor<3xf64>
    %804 = stablehlo.add %798, %803 : tensor<3xf64>
    %805 = stablehlo.slice %774 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %806 = stablehlo.reshape %805 : (tensor<3x1xf64>) -> tensor<3xf64>
    %807 = stablehlo.slice %795 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %808 = stablehlo.reshape %807 : (tensor<3x1xf64>) -> tensor<3xf64>
    %809 = stablehlo.multiply %806, %808 : tensor<3xf64>
    %810 = stablehlo.add %804, %809 : tensor<3xf64>
    %811 = stablehlo.slice %774 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %812 = stablehlo.reshape %811 : (tensor<3x1xf64>) -> tensor<3xf64>
    %813 = stablehlo.slice %795 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %814 = stablehlo.reshape %813 : (tensor<3x1xf64>) -> tensor<3xf64>
    %815 = stablehlo.multiply %812, %814 : tensor<3xf64>
    %816 = stablehlo.subtract %810, %815 : tensor<3xf64>
    %817 = stablehlo.reshape %816 : (tensor<3xf64>) -> tensor<3x1xf64>
    %818 = stablehlo.multiply %776, %814 : tensor<3xf64>
    %819 = stablehlo.multiply %800, %808 : tensor<3xf64>
    %820 = stablehlo.subtract %818, %819 : tensor<3xf64>
    %821 = stablehlo.multiply %806, %802 : tensor<3xf64>
    %822 = stablehlo.add %820, %821 : tensor<3xf64>
    %823 = stablehlo.multiply %812, %797 : tensor<3xf64>
    %824 = stablehlo.add %822, %823 : tensor<3xf64>
    %825 = stablehlo.reshape %824 : (tensor<3xf64>) -> tensor<3x1xf64>
    %826 = stablehlo.multiply %776, %808 : tensor<3xf64>
    %827 = stablehlo.multiply %800, %814 : tensor<3xf64>
    %828 = stablehlo.add %826, %827 : tensor<3xf64>
    %829 = stablehlo.multiply %806, %797 : tensor<3xf64>
    %830 = stablehlo.subtract %828, %829 : tensor<3xf64>
    %831 = stablehlo.multiply %812, %802 : tensor<3xf64>
    %832 = stablehlo.add %830, %831 : tensor<3xf64>
    %833 = stablehlo.reshape %832 : (tensor<3xf64>) -> tensor<3x1xf64>
    %834 = stablehlo.multiply %776, %802 : tensor<3xf64>
    %835 = stablehlo.multiply %800, %797 : tensor<3xf64>
    %836 = stablehlo.subtract %834, %835 : tensor<3xf64>
    %837 = stablehlo.multiply %806, %814 : tensor<3xf64>
    %838 = stablehlo.subtract %836, %837 : tensor<3xf64>
    %839 = stablehlo.multiply %812, %808 : tensor<3xf64>
    %840 = stablehlo.subtract %838, %839 : tensor<3xf64>
    %841 = stablehlo.reshape %840 : (tensor<3xf64>) -> tensor<3x1xf64>
    %842 = stablehlo.concatenate %817, %825, %833, %841, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %843 = stablehlo.slice %842 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %844 = stablehlo.reshape %843 : (tensor<3x1xf64>) -> tensor<3xf64>
    %845 = stablehlo.reshape %844 : (tensor<3xf64>) -> tensor<3x1xf64>
    %846 = stablehlo.slice %842 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %847 = stablehlo.reshape %846 : (tensor<3x1xf64>) -> tensor<3xf64>
    %848 = stablehlo.reshape %847 : (tensor<3xf64>) -> tensor<3x1xf64>
    %849 = stablehlo.slice %842 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %850 = stablehlo.reshape %849 : (tensor<3x1xf64>) -> tensor<3xf64>
    %851 = stablehlo.reshape %850 : (tensor<3xf64>) -> tensor<3x1xf64>
    %852 = stablehlo.concatenate %845, %848, %851, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %853 = stablehlo.slice %722 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %854 = stablehlo.reshape %853 : (tensor<3x1xf64>) -> tensor<3xf64>
    %855 = stablehlo.slice %700 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %856 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %857 = stablehlo.concatenate %855, %856, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %858 = stablehlo.slice %857 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %859 = stablehlo.reshape %858 : (tensor<3x1xf64>) -> tensor<3xf64>
    %860 = stablehlo.multiply %854, %859 : tensor<3xf64>
    %861 = stablehlo.slice %722 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %862 = stablehlo.reshape %861 : (tensor<3x1xf64>) -> tensor<3xf64>
    %863 = stablehlo.slice %857 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %864 = stablehlo.reshape %863 : (tensor<3x1xf64>) -> tensor<3xf64>
    %865 = stablehlo.multiply %862, %864 : tensor<3xf64>
    %866 = stablehlo.add %860, %865 : tensor<3xf64>
    %867 = stablehlo.slice %722 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %868 = stablehlo.reshape %867 : (tensor<3x1xf64>) -> tensor<3xf64>
    %869 = stablehlo.slice %857 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %870 = stablehlo.reshape %869 : (tensor<3x1xf64>) -> tensor<3xf64>
    %871 = stablehlo.multiply %868, %870 : tensor<3xf64>
    %872 = stablehlo.add %866, %871 : tensor<3xf64>
    %873 = stablehlo.slice %722 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %874 = stablehlo.reshape %873 : (tensor<3x1xf64>) -> tensor<3xf64>
    %875 = stablehlo.slice %857 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %876 = stablehlo.reshape %875 : (tensor<3x1xf64>) -> tensor<3xf64>
    %877 = stablehlo.multiply %874, %876 : tensor<3xf64>
    %878 = stablehlo.subtract %872, %877 : tensor<3xf64>
    %879 = stablehlo.reshape %878 : (tensor<3xf64>) -> tensor<3x1xf64>
    %880 = stablehlo.multiply %854, %876 : tensor<3xf64>
    %881 = stablehlo.multiply %862, %870 : tensor<3xf64>
    %882 = stablehlo.subtract %880, %881 : tensor<3xf64>
    %883 = stablehlo.multiply %868, %864 : tensor<3xf64>
    %884 = stablehlo.add %882, %883 : tensor<3xf64>
    %885 = stablehlo.multiply %874, %859 : tensor<3xf64>
    %886 = stablehlo.add %884, %885 : tensor<3xf64>
    %887 = stablehlo.reshape %886 : (tensor<3xf64>) -> tensor<3x1xf64>
    %888 = stablehlo.multiply %854, %870 : tensor<3xf64>
    %889 = stablehlo.multiply %862, %876 : tensor<3xf64>
    %890 = stablehlo.add %888, %889 : tensor<3xf64>
    %891 = stablehlo.multiply %868, %859 : tensor<3xf64>
    %892 = stablehlo.subtract %890, %891 : tensor<3xf64>
    %893 = stablehlo.multiply %874, %864 : tensor<3xf64>
    %894 = stablehlo.add %892, %893 : tensor<3xf64>
    %895 = stablehlo.reshape %894 : (tensor<3xf64>) -> tensor<3x1xf64>
    %896 = stablehlo.multiply %854, %864 : tensor<3xf64>
    %897 = stablehlo.multiply %862, %859 : tensor<3xf64>
    %898 = stablehlo.subtract %896, %897 : tensor<3xf64>
    %899 = stablehlo.multiply %868, %876 : tensor<3xf64>
    %900 = stablehlo.subtract %898, %899 : tensor<3xf64>
    %901 = stablehlo.multiply %874, %870 : tensor<3xf64>
    %902 = stablehlo.subtract %900, %901 : tensor<3xf64>
    %903 = stablehlo.reshape %902 : (tensor<3xf64>) -> tensor<3x1xf64>
    %904 = stablehlo.concatenate %879, %887, %895, %903, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %905 = stablehlo.slice %904 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %906 = stablehlo.reshape %905 : (tensor<3x1xf64>) -> tensor<3xf64>
    %907 = stablehlo.slice %722 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %908 = stablehlo.reshape %907 : (tensor<3x1xf64>) -> tensor<3xf64>
    %909 = stablehlo.negate %908 : tensor<3xf64>
    %910 = stablehlo.reshape %909 : (tensor<3xf64>) -> tensor<3x1xf64>
    %911 = stablehlo.slice %722 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %912 = stablehlo.reshape %911 : (tensor<3x1xf64>) -> tensor<3xf64>
    %913 = stablehlo.negate %912 : tensor<3xf64>
    %914 = stablehlo.reshape %913 : (tensor<3xf64>) -> tensor<3x1xf64>
    %915 = stablehlo.slice %722 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %916 = stablehlo.reshape %915 : (tensor<3x1xf64>) -> tensor<3xf64>
    %917 = stablehlo.negate %916 : tensor<3xf64>
    %918 = stablehlo.reshape %917 : (tensor<3xf64>) -> tensor<3x1xf64>
    %919 = stablehlo.slice %722 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %920 = stablehlo.reshape %919 : (tensor<3x1xf64>) -> tensor<3xf64>
    %921 = stablehlo.reshape %920 : (tensor<3xf64>) -> tensor<3x1xf64>
    %922 = stablehlo.concatenate %910, %914, %918, %921, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %923 = stablehlo.dot_general %722, %722, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %924 = stablehlo.broadcast_in_dim %923, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %925 = stablehlo.divide %922, %924 : tensor<3x4xf64>
    %926 = stablehlo.slice %925 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %927 = stablehlo.reshape %926 : (tensor<3x1xf64>) -> tensor<3xf64>
    %928 = stablehlo.multiply %906, %927 : tensor<3xf64>
    %929 = stablehlo.slice %904 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %930 = stablehlo.reshape %929 : (tensor<3x1xf64>) -> tensor<3xf64>
    %931 = stablehlo.slice %925 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %932 = stablehlo.reshape %931 : (tensor<3x1xf64>) -> tensor<3xf64>
    %933 = stablehlo.multiply %930, %932 : tensor<3xf64>
    %934 = stablehlo.add %928, %933 : tensor<3xf64>
    %935 = stablehlo.slice %904 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %936 = stablehlo.reshape %935 : (tensor<3x1xf64>) -> tensor<3xf64>
    %937 = stablehlo.slice %925 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %938 = stablehlo.reshape %937 : (tensor<3x1xf64>) -> tensor<3xf64>
    %939 = stablehlo.multiply %936, %938 : tensor<3xf64>
    %940 = stablehlo.add %934, %939 : tensor<3xf64>
    %941 = stablehlo.slice %904 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %942 = stablehlo.reshape %941 : (tensor<3x1xf64>) -> tensor<3xf64>
    %943 = stablehlo.slice %925 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %944 = stablehlo.reshape %943 : (tensor<3x1xf64>) -> tensor<3xf64>
    %945 = stablehlo.multiply %942, %944 : tensor<3xf64>
    %946 = stablehlo.subtract %940, %945 : tensor<3xf64>
    %947 = stablehlo.reshape %946 : (tensor<3xf64>) -> tensor<3x1xf64>
    %948 = stablehlo.multiply %906, %944 : tensor<3xf64>
    %949 = stablehlo.multiply %930, %938 : tensor<3xf64>
    %950 = stablehlo.subtract %948, %949 : tensor<3xf64>
    %951 = stablehlo.multiply %936, %932 : tensor<3xf64>
    %952 = stablehlo.add %950, %951 : tensor<3xf64>
    %953 = stablehlo.multiply %942, %927 : tensor<3xf64>
    %954 = stablehlo.add %952, %953 : tensor<3xf64>
    %955 = stablehlo.reshape %954 : (tensor<3xf64>) -> tensor<3x1xf64>
    %956 = stablehlo.multiply %906, %938 : tensor<3xf64>
    %957 = stablehlo.multiply %930, %944 : tensor<3xf64>
    %958 = stablehlo.add %956, %957 : tensor<3xf64>
    %959 = stablehlo.multiply %936, %927 : tensor<3xf64>
    %960 = stablehlo.subtract %958, %959 : tensor<3xf64>
    %961 = stablehlo.multiply %942, %932 : tensor<3xf64>
    %962 = stablehlo.add %960, %961 : tensor<3xf64>
    %963 = stablehlo.reshape %962 : (tensor<3xf64>) -> tensor<3x1xf64>
    %964 = stablehlo.multiply %906, %932 : tensor<3xf64>
    %965 = stablehlo.multiply %930, %927 : tensor<3xf64>
    %966 = stablehlo.subtract %964, %965 : tensor<3xf64>
    %967 = stablehlo.multiply %936, %944 : tensor<3xf64>
    %968 = stablehlo.subtract %966, %967 : tensor<3xf64>
    %969 = stablehlo.multiply %942, %938 : tensor<3xf64>
    %970 = stablehlo.subtract %968, %969 : tensor<3xf64>
    %971 = stablehlo.reshape %970 : (tensor<3xf64>) -> tensor<3x1xf64>
    %972 = stablehlo.concatenate %947, %955, %963, %971, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %973 = stablehlo.slice %972 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %974 = stablehlo.reshape %973 : (tensor<3x1xf64>) -> tensor<3xf64>
    %975 = stablehlo.reshape %974 : (tensor<3xf64>) -> tensor<3x1xf64>
    %976 = stablehlo.slice %972 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %977 = stablehlo.reshape %976 : (tensor<3x1xf64>) -> tensor<3xf64>
    %978 = stablehlo.reshape %977 : (tensor<3xf64>) -> tensor<3x1xf64>
    %979 = stablehlo.slice %972 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %980 = stablehlo.reshape %979 : (tensor<3x1xf64>) -> tensor<3xf64>
    %981 = stablehlo.reshape %980 : (tensor<3xf64>) -> tensor<3x1xf64>
    %982 = stablehlo.concatenate %975, %978, %981, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %983 = stablehlo.concatenate %852, %982, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %984 = stablehlo.slice %983 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %985 = stablehlo.slice %arg3 [0:3, 0:3] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %986 = stablehlo.divide %984, %985 : tensor<3x3xf64>
    %987 = stablehlo.slice %983 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %988 = stablehlo.slice %arg3 [0:3, 6:7] : (tensor<3x7xf64>) -> tensor<3x1xf64>
    %989 = stablehlo.reshape %988 : (tensor<3x1xf64>) -> tensor<3xf64>
    %990 = stablehlo.broadcast_in_dim %989, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
    %991 = stablehlo.divide %987, %990 : tensor<3x3xf64>
    %992 = stablehlo.concatenate %986, %991, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %993 = stablehlo.slice %992 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %994 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %995 = stablehlo.concatenate %993, %994, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %996 = stablehlo.slice %995 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %997 = stablehlo.reshape %996 : (tensor<3x1xf64>) -> tensor<3xf64>
    %998 = stablehlo.multiply %703, %997 : tensor<3xf64>
    %999 = stablehlo.slice %701 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1000 = stablehlo.reshape %999 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1001 = stablehlo.slice %995 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1002 = stablehlo.reshape %1001 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1003 = stablehlo.multiply %1000, %1002 : tensor<3xf64>
    %1004 = stablehlo.add %998, %1003 : tensor<3xf64>
    %1005 = stablehlo.slice %701 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1006 = stablehlo.reshape %1005 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1007 = stablehlo.slice %995 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1008 = stablehlo.reshape %1007 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1009 = stablehlo.multiply %1006, %1008 : tensor<3xf64>
    %1010 = stablehlo.add %1004, %1009 : tensor<3xf64>
    %1011 = stablehlo.slice %701 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1012 = stablehlo.reshape %1011 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1013 = stablehlo.slice %995 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1014 = stablehlo.reshape %1013 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1015 = stablehlo.multiply %1012, %1014 : tensor<3xf64>
    %1016 = stablehlo.subtract %1010, %1015 : tensor<3xf64>
    %1017 = stablehlo.reshape %1016 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1018 = stablehlo.multiply %703, %1014 : tensor<3xf64>
    %1019 = stablehlo.multiply %1000, %1008 : tensor<3xf64>
    %1020 = stablehlo.subtract %1018, %1019 : tensor<3xf64>
    %1021 = stablehlo.multiply %1006, %1002 : tensor<3xf64>
    %1022 = stablehlo.add %1020, %1021 : tensor<3xf64>
    %1023 = stablehlo.multiply %1012, %997 : tensor<3xf64>
    %1024 = stablehlo.add %1022, %1023 : tensor<3xf64>
    %1025 = stablehlo.reshape %1024 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1026 = stablehlo.multiply %703, %1008 : tensor<3xf64>
    %1027 = stablehlo.multiply %1000, %1014 : tensor<3xf64>
    %1028 = stablehlo.add %1026, %1027 : tensor<3xf64>
    %1029 = stablehlo.multiply %1006, %997 : tensor<3xf64>
    %1030 = stablehlo.subtract %1028, %1029 : tensor<3xf64>
    %1031 = stablehlo.multiply %1012, %1002 : tensor<3xf64>
    %1032 = stablehlo.add %1030, %1031 : tensor<3xf64>
    %1033 = stablehlo.reshape %1032 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1034 = stablehlo.multiply %703, %1002 : tensor<3xf64>
    %1035 = stablehlo.multiply %1000, %997 : tensor<3xf64>
    %1036 = stablehlo.subtract %1034, %1035 : tensor<3xf64>
    %1037 = stablehlo.multiply %1006, %1014 : tensor<3xf64>
    %1038 = stablehlo.subtract %1036, %1037 : tensor<3xf64>
    %1039 = stablehlo.multiply %1012, %1008 : tensor<3xf64>
    %1040 = stablehlo.subtract %1038, %1039 : tensor<3xf64>
    %1041 = stablehlo.reshape %1040 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1042 = stablehlo.concatenate %1017, %1025, %1033, %1041, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1043 = stablehlo.slice %1042 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1044 = stablehlo.reshape %1043 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1045 = stablehlo.slice %701 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1046 = stablehlo.reshape %1045 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1047 = stablehlo.negate %1046 : tensor<3xf64>
    %1048 = stablehlo.reshape %1047 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1049 = stablehlo.slice %701 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1050 = stablehlo.reshape %1049 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1051 = stablehlo.negate %1050 : tensor<3xf64>
    %1052 = stablehlo.reshape %1051 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1053 = stablehlo.slice %701 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1054 = stablehlo.reshape %1053 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1055 = stablehlo.negate %1054 : tensor<3xf64>
    %1056 = stablehlo.reshape %1055 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1057 = stablehlo.slice %701 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1058 = stablehlo.reshape %1057 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1059 = stablehlo.reshape %1058 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1060 = stablehlo.concatenate %1048, %1052, %1056, %1059, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1061 = stablehlo.dot_general %701, %701, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1062 = stablehlo.broadcast_in_dim %1061, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1063 = stablehlo.divide %1060, %1062 : tensor<3x4xf64>
    %1064 = stablehlo.slice %1063 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1065 = stablehlo.reshape %1064 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1066 = stablehlo.multiply %1044, %1065 : tensor<3xf64>
    %1067 = stablehlo.slice %1042 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1068 = stablehlo.reshape %1067 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1069 = stablehlo.slice %1063 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1070 = stablehlo.reshape %1069 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1071 = stablehlo.multiply %1068, %1070 : tensor<3xf64>
    %1072 = stablehlo.add %1066, %1071 : tensor<3xf64>
    %1073 = stablehlo.slice %1042 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1074 = stablehlo.reshape %1073 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1075 = stablehlo.slice %1063 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1076 = stablehlo.reshape %1075 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1077 = stablehlo.multiply %1074, %1076 : tensor<3xf64>
    %1078 = stablehlo.add %1072, %1077 : tensor<3xf64>
    %1079 = stablehlo.slice %1042 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1080 = stablehlo.reshape %1079 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1081 = stablehlo.slice %1063 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1082 = stablehlo.reshape %1081 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1083 = stablehlo.multiply %1080, %1082 : tensor<3xf64>
    %1084 = stablehlo.subtract %1078, %1083 : tensor<3xf64>
    %1085 = stablehlo.reshape %1084 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1086 = stablehlo.multiply %1044, %1082 : tensor<3xf64>
    %1087 = stablehlo.multiply %1068, %1076 : tensor<3xf64>
    %1088 = stablehlo.subtract %1086, %1087 : tensor<3xf64>
    %1089 = stablehlo.multiply %1074, %1070 : tensor<3xf64>
    %1090 = stablehlo.add %1088, %1089 : tensor<3xf64>
    %1091 = stablehlo.multiply %1080, %1065 : tensor<3xf64>
    %1092 = stablehlo.add %1090, %1091 : tensor<3xf64>
    %1093 = stablehlo.reshape %1092 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1094 = stablehlo.multiply %1044, %1076 : tensor<3xf64>
    %1095 = stablehlo.multiply %1068, %1082 : tensor<3xf64>
    %1096 = stablehlo.add %1094, %1095 : tensor<3xf64>
    %1097 = stablehlo.multiply %1074, %1065 : tensor<3xf64>
    %1098 = stablehlo.subtract %1096, %1097 : tensor<3xf64>
    %1099 = stablehlo.multiply %1080, %1070 : tensor<3xf64>
    %1100 = stablehlo.add %1098, %1099 : tensor<3xf64>
    %1101 = stablehlo.reshape %1100 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1102 = stablehlo.multiply %1044, %1070 : tensor<3xf64>
    %1103 = stablehlo.multiply %1068, %1065 : tensor<3xf64>
    %1104 = stablehlo.subtract %1102, %1103 : tensor<3xf64>
    %1105 = stablehlo.multiply %1074, %1082 : tensor<3xf64>
    %1106 = stablehlo.subtract %1104, %1105 : tensor<3xf64>
    %1107 = stablehlo.multiply %1080, %1076 : tensor<3xf64>
    %1108 = stablehlo.subtract %1106, %1107 : tensor<3xf64>
    %1109 = stablehlo.reshape %1108 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1110 = stablehlo.concatenate %1085, %1093, %1101, %1109, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1111 = stablehlo.slice %1110 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1112 = stablehlo.reshape %1111 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1113 = stablehlo.reshape %1112 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1114 = stablehlo.slice %1110 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1115 = stablehlo.reshape %1114 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1116 = stablehlo.reshape %1115 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1117 = stablehlo.slice %1110 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1118 = stablehlo.reshape %1117 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1119 = stablehlo.reshape %1118 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1120 = stablehlo.concatenate %1113, %1116, %1119, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %1121 = stablehlo.slice %701 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1122 = stablehlo.reshape %1121 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1123 = stablehlo.slice %992 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1124 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1125 = stablehlo.concatenate %1123, %1124, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1126 = stablehlo.slice %1125 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1127 = stablehlo.reshape %1126 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1128 = stablehlo.multiply %1122, %1127 : tensor<3xf64>
    %1129 = stablehlo.slice %701 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1130 = stablehlo.reshape %1129 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1131 = stablehlo.slice %1125 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1132 = stablehlo.reshape %1131 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1133 = stablehlo.multiply %1130, %1132 : tensor<3xf64>
    %1134 = stablehlo.add %1128, %1133 : tensor<3xf64>
    %1135 = stablehlo.slice %701 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1136 = stablehlo.reshape %1135 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1137 = stablehlo.slice %1125 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1138 = stablehlo.reshape %1137 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1139 = stablehlo.multiply %1136, %1138 : tensor<3xf64>
    %1140 = stablehlo.add %1134, %1139 : tensor<3xf64>
    %1141 = stablehlo.slice %701 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1142 = stablehlo.reshape %1141 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1143 = stablehlo.slice %1125 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1144 = stablehlo.reshape %1143 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1145 = stablehlo.multiply %1142, %1144 : tensor<3xf64>
    %1146 = stablehlo.subtract %1140, %1145 : tensor<3xf64>
    %1147 = stablehlo.reshape %1146 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1148 = stablehlo.multiply %1122, %1144 : tensor<3xf64>
    %1149 = stablehlo.multiply %1130, %1138 : tensor<3xf64>
    %1150 = stablehlo.subtract %1148, %1149 : tensor<3xf64>
    %1151 = stablehlo.multiply %1136, %1132 : tensor<3xf64>
    %1152 = stablehlo.add %1150, %1151 : tensor<3xf64>
    %1153 = stablehlo.multiply %1142, %1127 : tensor<3xf64>
    %1154 = stablehlo.add %1152, %1153 : tensor<3xf64>
    %1155 = stablehlo.reshape %1154 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1156 = stablehlo.multiply %1122, %1138 : tensor<3xf64>
    %1157 = stablehlo.multiply %1130, %1144 : tensor<3xf64>
    %1158 = stablehlo.add %1156, %1157 : tensor<3xf64>
    %1159 = stablehlo.multiply %1136, %1127 : tensor<3xf64>
    %1160 = stablehlo.subtract %1158, %1159 : tensor<3xf64>
    %1161 = stablehlo.multiply %1142, %1132 : tensor<3xf64>
    %1162 = stablehlo.add %1160, %1161 : tensor<3xf64>
    %1163 = stablehlo.reshape %1162 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1164 = stablehlo.multiply %1122, %1132 : tensor<3xf64>
    %1165 = stablehlo.multiply %1130, %1127 : tensor<3xf64>
    %1166 = stablehlo.subtract %1164, %1165 : tensor<3xf64>
    %1167 = stablehlo.multiply %1136, %1144 : tensor<3xf64>
    %1168 = stablehlo.subtract %1166, %1167 : tensor<3xf64>
    %1169 = stablehlo.multiply %1142, %1138 : tensor<3xf64>
    %1170 = stablehlo.subtract %1168, %1169 : tensor<3xf64>
    %1171 = stablehlo.reshape %1170 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1172 = stablehlo.concatenate %1147, %1155, %1163, %1171, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1173 = stablehlo.slice %1172 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1174 = stablehlo.reshape %1173 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1175 = stablehlo.slice %701 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1176 = stablehlo.reshape %1175 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1177 = stablehlo.negate %1176 : tensor<3xf64>
    %1178 = stablehlo.reshape %1177 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1179 = stablehlo.slice %701 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1180 = stablehlo.reshape %1179 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1181 = stablehlo.negate %1180 : tensor<3xf64>
    %1182 = stablehlo.reshape %1181 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1183 = stablehlo.slice %701 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1184 = stablehlo.reshape %1183 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1185 = stablehlo.negate %1184 : tensor<3xf64>
    %1186 = stablehlo.reshape %1185 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1187 = stablehlo.slice %701 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1188 = stablehlo.reshape %1187 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1189 = stablehlo.reshape %1188 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1190 = stablehlo.concatenate %1178, %1182, %1186, %1189, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1191 = stablehlo.dot_general %701, %701, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1192 = stablehlo.broadcast_in_dim %1191, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1193 = stablehlo.divide %1190, %1192 : tensor<3x4xf64>
    %1194 = stablehlo.slice %1193 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1195 = stablehlo.reshape %1194 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1196 = stablehlo.multiply %1174, %1195 : tensor<3xf64>
    %1197 = stablehlo.slice %1172 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1198 = stablehlo.reshape %1197 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1199 = stablehlo.slice %1193 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1200 = stablehlo.reshape %1199 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1201 = stablehlo.multiply %1198, %1200 : tensor<3xf64>
    %1202 = stablehlo.add %1196, %1201 : tensor<3xf64>
    %1203 = stablehlo.slice %1172 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1204 = stablehlo.reshape %1203 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1205 = stablehlo.slice %1193 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1206 = stablehlo.reshape %1205 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1207 = stablehlo.multiply %1204, %1206 : tensor<3xf64>
    %1208 = stablehlo.add %1202, %1207 : tensor<3xf64>
    %1209 = stablehlo.slice %1172 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1210 = stablehlo.reshape %1209 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1211 = stablehlo.slice %1193 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1212 = stablehlo.reshape %1211 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1213 = stablehlo.multiply %1210, %1212 : tensor<3xf64>
    %1214 = stablehlo.subtract %1208, %1213 : tensor<3xf64>
    %1215 = stablehlo.reshape %1214 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1216 = stablehlo.multiply %1174, %1212 : tensor<3xf64>
    %1217 = stablehlo.multiply %1198, %1206 : tensor<3xf64>
    %1218 = stablehlo.subtract %1216, %1217 : tensor<3xf64>
    %1219 = stablehlo.multiply %1204, %1200 : tensor<3xf64>
    %1220 = stablehlo.add %1218, %1219 : tensor<3xf64>
    %1221 = stablehlo.multiply %1210, %1195 : tensor<3xf64>
    %1222 = stablehlo.add %1220, %1221 : tensor<3xf64>
    %1223 = stablehlo.reshape %1222 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1224 = stablehlo.multiply %1174, %1206 : tensor<3xf64>
    %1225 = stablehlo.multiply %1198, %1212 : tensor<3xf64>
    %1226 = stablehlo.add %1224, %1225 : tensor<3xf64>
    %1227 = stablehlo.multiply %1204, %1195 : tensor<3xf64>
    %1228 = stablehlo.subtract %1226, %1227 : tensor<3xf64>
    %1229 = stablehlo.multiply %1210, %1200 : tensor<3xf64>
    %1230 = stablehlo.add %1228, %1229 : tensor<3xf64>
    %1231 = stablehlo.reshape %1230 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1232 = stablehlo.multiply %1174, %1200 : tensor<3xf64>
    %1233 = stablehlo.multiply %1198, %1195 : tensor<3xf64>
    %1234 = stablehlo.subtract %1232, %1233 : tensor<3xf64>
    %1235 = stablehlo.multiply %1204, %1212 : tensor<3xf64>
    %1236 = stablehlo.subtract %1234, %1235 : tensor<3xf64>
    %1237 = stablehlo.multiply %1210, %1206 : tensor<3xf64>
    %1238 = stablehlo.subtract %1236, %1237 : tensor<3xf64>
    %1239 = stablehlo.reshape %1238 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1240 = stablehlo.concatenate %1215, %1223, %1231, %1239, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1241 = stablehlo.slice %1240 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1242 = stablehlo.reshape %1241 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1243 = stablehlo.reshape %1242 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1244 = stablehlo.slice %1240 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1245 = stablehlo.reshape %1244 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1246 = stablehlo.reshape %1245 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1247 = stablehlo.slice %1240 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1248 = stablehlo.reshape %1247 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1249 = stablehlo.reshape %1248 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1250 = stablehlo.concatenate %1243, %1246, %1249, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %1251 = stablehlo.concatenate %1120, %1250, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %1252 = stablehlo.slice %arg2 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %cst_15 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %1253 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1254 = stablehlo.reshape %arg5 : (tensor<f64>) -> tensor<f64>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1256 = stablehlo.multiply %1253, %1255 : tensor<3xf64>
    %1257 = stablehlo.broadcast_in_dim %1256, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %1258 = stablehlo.multiply %1257, %arg6 : tensor<3x6xf64>
    %1259 = stablehlo.slice %1258 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_16 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1260 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %1261 = stablehlo.divide %1259, %1260 : tensor<3x3xf64>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1262 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1263 = stablehlo.concatenate %1261, %1262, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1264 = stablehlo.slice %1263 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1265 = stablehlo.reshape %1264 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1266 = stablehlo.slice %1252 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1267 = stablehlo.reshape %1266 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1268 = stablehlo.multiply %1265, %1267 : tensor<3xf64>
    %1269 = stablehlo.slice %1263 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1270 = stablehlo.reshape %1269 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1271 = stablehlo.slice %1252 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1272 = stablehlo.reshape %1271 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1273 = stablehlo.multiply %1270, %1272 : tensor<3xf64>
    %1274 = stablehlo.add %1268, %1273 : tensor<3xf64>
    %1275 = stablehlo.slice %1263 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1276 = stablehlo.reshape %1275 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1277 = stablehlo.slice %1252 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1278 = stablehlo.reshape %1277 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1279 = stablehlo.multiply %1276, %1278 : tensor<3xf64>
    %1280 = stablehlo.add %1274, %1279 : tensor<3xf64>
    %1281 = stablehlo.slice %1263 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1282 = stablehlo.reshape %1281 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1283 = stablehlo.slice %1252 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1284 = stablehlo.reshape %1283 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1285 = stablehlo.multiply %1282, %1284 : tensor<3xf64>
    %1286 = stablehlo.subtract %1280, %1285 : tensor<3xf64>
    %1287 = stablehlo.reshape %1286 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1288 = stablehlo.multiply %1265, %1284 : tensor<3xf64>
    %1289 = stablehlo.multiply %1270, %1278 : tensor<3xf64>
    %1290 = stablehlo.subtract %1288, %1289 : tensor<3xf64>
    %1291 = stablehlo.multiply %1276, %1272 : tensor<3xf64>
    %1292 = stablehlo.add %1290, %1291 : tensor<3xf64>
    %1293 = stablehlo.multiply %1282, %1267 : tensor<3xf64>
    %1294 = stablehlo.add %1292, %1293 : tensor<3xf64>
    %1295 = stablehlo.reshape %1294 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1296 = stablehlo.multiply %1265, %1278 : tensor<3xf64>
    %1297 = stablehlo.multiply %1270, %1284 : tensor<3xf64>
    %1298 = stablehlo.add %1296, %1297 : tensor<3xf64>
    %1299 = stablehlo.multiply %1276, %1267 : tensor<3xf64>
    %1300 = stablehlo.subtract %1298, %1299 : tensor<3xf64>
    %1301 = stablehlo.multiply %1282, %1272 : tensor<3xf64>
    %1302 = stablehlo.add %1300, %1301 : tensor<3xf64>
    %1303 = stablehlo.reshape %1302 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1304 = stablehlo.multiply %1265, %1272 : tensor<3xf64>
    %1305 = stablehlo.multiply %1270, %1267 : tensor<3xf64>
    %1306 = stablehlo.subtract %1304, %1305 : tensor<3xf64>
    %1307 = stablehlo.multiply %1276, %1284 : tensor<3xf64>
    %1308 = stablehlo.subtract %1306, %1307 : tensor<3xf64>
    %1309 = stablehlo.multiply %1282, %1278 : tensor<3xf64>
    %1310 = stablehlo.subtract %1308, %1309 : tensor<3xf64>
    %1311 = stablehlo.reshape %1310 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1312 = stablehlo.concatenate %1287, %1295, %1303, %1311, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1313 = stablehlo.add %1252, %1312 : tensor<3x4xf64>
    %1314 = stablehlo.dot_general %1313, %1313, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1315 = stablehlo.sqrt %1314 : tensor<3xf64>
    %1316 = stablehlo.broadcast_in_dim %1315, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1317 = stablehlo.divide %1313, %1316 : tensor<3x4xf64>
    %1318 = stablehlo.slice %arg2 [0:3, 4:7] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %1319 = stablehlo.slice %1258 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %1320 = stablehlo.add %1318, %1319 : tensor<3x3xf64>
    %1321 = stablehlo.concatenate %1317, %1320, dim = 1 : (tensor<3x4xf64>, tensor<3x3xf64>) -> tensor<3x7xf64>
    %1322 = stablehlo.broadcast_in_dim %1256, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %1323 = stablehlo.multiply %1322, %1251 : tensor<3x6xf64>
    %1324 = stablehlo.add %arg6, %1323 : tensor<3x6xf64>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1325 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %1326 = call @inner(%1321, %arg3, %1325) : (tensor<3x7xf64>, tensor<3x7xf64>, tensor<3x6xf64>) -> tensor<3x6xf64>
    %1327 = stablehlo.slice %1321 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %1328 = stablehlo.slice %1327 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1329 = stablehlo.reshape %1328 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1330 = stablehlo.slice %1327 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1331 = stablehlo.reshape %1330 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1332 = stablehlo.negate %1331 : tensor<3xf64>
    %1333 = stablehlo.reshape %1332 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1334 = stablehlo.slice %1327 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1335 = stablehlo.reshape %1334 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1336 = stablehlo.negate %1335 : tensor<3xf64>
    %1337 = stablehlo.reshape %1336 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1338 = stablehlo.slice %1327 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1339 = stablehlo.reshape %1338 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1340 = stablehlo.negate %1339 : tensor<3xf64>
    %1341 = stablehlo.reshape %1340 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1342 = stablehlo.slice %1327 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1343 = stablehlo.reshape %1342 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1344 = stablehlo.reshape %1343 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1345 = stablehlo.concatenate %1333, %1337, %1341, %1344, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1346 = stablehlo.dot_general %1327, %1327, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1347 = stablehlo.broadcast_in_dim %1346, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1348 = stablehlo.divide %1345, %1347 : tensor<3x4xf64>
    %1349 = stablehlo.slice %1348 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1350 = stablehlo.reshape %1349 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1351 = stablehlo.slice %1326 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1352 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1353 = stablehlo.concatenate %1351, %1352, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1354 = stablehlo.slice %1353 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1355 = stablehlo.reshape %1354 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1356 = stablehlo.multiply %1350, %1355 : tensor<3xf64>
    %1357 = stablehlo.slice %1348 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1358 = stablehlo.reshape %1357 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1359 = stablehlo.slice %1353 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1360 = stablehlo.reshape %1359 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1361 = stablehlo.multiply %1358, %1360 : tensor<3xf64>
    %1362 = stablehlo.add %1356, %1361 : tensor<3xf64>
    %1363 = stablehlo.slice %1348 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1364 = stablehlo.reshape %1363 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1365 = stablehlo.slice %1353 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1366 = stablehlo.reshape %1365 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1367 = stablehlo.multiply %1364, %1366 : tensor<3xf64>
    %1368 = stablehlo.add %1362, %1367 : tensor<3xf64>
    %1369 = stablehlo.slice %1348 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1370 = stablehlo.reshape %1369 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1371 = stablehlo.slice %1353 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1372 = stablehlo.reshape %1371 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1373 = stablehlo.multiply %1370, %1372 : tensor<3xf64>
    %1374 = stablehlo.subtract %1368, %1373 : tensor<3xf64>
    %1375 = stablehlo.reshape %1374 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1376 = stablehlo.multiply %1350, %1372 : tensor<3xf64>
    %1377 = stablehlo.multiply %1358, %1366 : tensor<3xf64>
    %1378 = stablehlo.subtract %1376, %1377 : tensor<3xf64>
    %1379 = stablehlo.multiply %1364, %1360 : tensor<3xf64>
    %1380 = stablehlo.add %1378, %1379 : tensor<3xf64>
    %1381 = stablehlo.multiply %1370, %1355 : tensor<3xf64>
    %1382 = stablehlo.add %1380, %1381 : tensor<3xf64>
    %1383 = stablehlo.reshape %1382 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1384 = stablehlo.multiply %1350, %1366 : tensor<3xf64>
    %1385 = stablehlo.multiply %1358, %1372 : tensor<3xf64>
    %1386 = stablehlo.add %1384, %1385 : tensor<3xf64>
    %1387 = stablehlo.multiply %1364, %1355 : tensor<3xf64>
    %1388 = stablehlo.subtract %1386, %1387 : tensor<3xf64>
    %1389 = stablehlo.multiply %1370, %1360 : tensor<3xf64>
    %1390 = stablehlo.add %1388, %1389 : tensor<3xf64>
    %1391 = stablehlo.reshape %1390 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1392 = stablehlo.multiply %1350, %1360 : tensor<3xf64>
    %1393 = stablehlo.multiply %1358, %1355 : tensor<3xf64>
    %1394 = stablehlo.subtract %1392, %1393 : tensor<3xf64>
    %1395 = stablehlo.multiply %1364, %1372 : tensor<3xf64>
    %1396 = stablehlo.subtract %1394, %1395 : tensor<3xf64>
    %1397 = stablehlo.multiply %1370, %1366 : tensor<3xf64>
    %1398 = stablehlo.subtract %1396, %1397 : tensor<3xf64>
    %1399 = stablehlo.reshape %1398 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1400 = stablehlo.concatenate %1375, %1383, %1391, %1399, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1401 = stablehlo.slice %1400 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1402 = stablehlo.reshape %1401 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1403 = stablehlo.slice %1348 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1404 = stablehlo.reshape %1403 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1405 = stablehlo.negate %1404 : tensor<3xf64>
    %1406 = stablehlo.reshape %1405 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1407 = stablehlo.slice %1348 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1408 = stablehlo.reshape %1407 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1409 = stablehlo.negate %1408 : tensor<3xf64>
    %1410 = stablehlo.reshape %1409 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1411 = stablehlo.slice %1348 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1412 = stablehlo.reshape %1411 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1413 = stablehlo.negate %1412 : tensor<3xf64>
    %1414 = stablehlo.reshape %1413 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1415 = stablehlo.slice %1348 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1416 = stablehlo.reshape %1415 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1417 = stablehlo.reshape %1416 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1418 = stablehlo.concatenate %1406, %1410, %1414, %1417, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1419 = stablehlo.dot_general %1348, %1348, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1420 = stablehlo.broadcast_in_dim %1419, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1421 = stablehlo.divide %1418, %1420 : tensor<3x4xf64>
    %1422 = stablehlo.slice %1421 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1423 = stablehlo.reshape %1422 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1424 = stablehlo.multiply %1402, %1423 : tensor<3xf64>
    %1425 = stablehlo.slice %1400 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1426 = stablehlo.reshape %1425 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1427 = stablehlo.slice %1421 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1428 = stablehlo.reshape %1427 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1429 = stablehlo.multiply %1426, %1428 : tensor<3xf64>
    %1430 = stablehlo.add %1424, %1429 : tensor<3xf64>
    %1431 = stablehlo.slice %1400 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1432 = stablehlo.reshape %1431 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1433 = stablehlo.slice %1421 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1434 = stablehlo.reshape %1433 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1435 = stablehlo.multiply %1432, %1434 : tensor<3xf64>
    %1436 = stablehlo.add %1430, %1435 : tensor<3xf64>
    %1437 = stablehlo.slice %1400 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1438 = stablehlo.reshape %1437 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1439 = stablehlo.slice %1421 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1440 = stablehlo.reshape %1439 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1441 = stablehlo.multiply %1438, %1440 : tensor<3xf64>
    %1442 = stablehlo.subtract %1436, %1441 : tensor<3xf64>
    %1443 = stablehlo.reshape %1442 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1444 = stablehlo.multiply %1402, %1440 : tensor<3xf64>
    %1445 = stablehlo.multiply %1426, %1434 : tensor<3xf64>
    %1446 = stablehlo.subtract %1444, %1445 : tensor<3xf64>
    %1447 = stablehlo.multiply %1432, %1428 : tensor<3xf64>
    %1448 = stablehlo.add %1446, %1447 : tensor<3xf64>
    %1449 = stablehlo.multiply %1438, %1423 : tensor<3xf64>
    %1450 = stablehlo.add %1448, %1449 : tensor<3xf64>
    %1451 = stablehlo.reshape %1450 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1452 = stablehlo.multiply %1402, %1434 : tensor<3xf64>
    %1453 = stablehlo.multiply %1426, %1440 : tensor<3xf64>
    %1454 = stablehlo.add %1452, %1453 : tensor<3xf64>
    %1455 = stablehlo.multiply %1432, %1423 : tensor<3xf64>
    %1456 = stablehlo.subtract %1454, %1455 : tensor<3xf64>
    %1457 = stablehlo.multiply %1438, %1428 : tensor<3xf64>
    %1458 = stablehlo.add %1456, %1457 : tensor<3xf64>
    %1459 = stablehlo.reshape %1458 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1460 = stablehlo.multiply %1402, %1428 : tensor<3xf64>
    %1461 = stablehlo.multiply %1426, %1423 : tensor<3xf64>
    %1462 = stablehlo.subtract %1460, %1461 : tensor<3xf64>
    %1463 = stablehlo.multiply %1432, %1440 : tensor<3xf64>
    %1464 = stablehlo.subtract %1462, %1463 : tensor<3xf64>
    %1465 = stablehlo.multiply %1438, %1434 : tensor<3xf64>
    %1466 = stablehlo.subtract %1464, %1465 : tensor<3xf64>
    %1467 = stablehlo.reshape %1466 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1468 = stablehlo.concatenate %1443, %1451, %1459, %1467, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1469 = stablehlo.slice %1468 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1470 = stablehlo.reshape %1469 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1471 = stablehlo.reshape %1470 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1472 = stablehlo.slice %1468 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1473 = stablehlo.reshape %1472 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1474 = stablehlo.reshape %1473 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1475 = stablehlo.slice %1468 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1476 = stablehlo.reshape %1475 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1477 = stablehlo.reshape %1476 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1478 = stablehlo.concatenate %1471, %1474, %1477, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %1479 = stablehlo.slice %1348 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1480 = stablehlo.reshape %1479 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1481 = stablehlo.slice %1326 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1482 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1483 = stablehlo.concatenate %1481, %1482, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1484 = stablehlo.slice %1483 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1485 = stablehlo.reshape %1484 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1486 = stablehlo.multiply %1480, %1485 : tensor<3xf64>
    %1487 = stablehlo.slice %1348 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1488 = stablehlo.reshape %1487 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1489 = stablehlo.slice %1483 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1490 = stablehlo.reshape %1489 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1491 = stablehlo.multiply %1488, %1490 : tensor<3xf64>
    %1492 = stablehlo.add %1486, %1491 : tensor<3xf64>
    %1493 = stablehlo.slice %1348 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1494 = stablehlo.reshape %1493 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1495 = stablehlo.slice %1483 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1496 = stablehlo.reshape %1495 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1497 = stablehlo.multiply %1494, %1496 : tensor<3xf64>
    %1498 = stablehlo.add %1492, %1497 : tensor<3xf64>
    %1499 = stablehlo.slice %1348 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1500 = stablehlo.reshape %1499 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1501 = stablehlo.slice %1483 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1502 = stablehlo.reshape %1501 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1503 = stablehlo.multiply %1500, %1502 : tensor<3xf64>
    %1504 = stablehlo.subtract %1498, %1503 : tensor<3xf64>
    %1505 = stablehlo.reshape %1504 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1506 = stablehlo.multiply %1480, %1502 : tensor<3xf64>
    %1507 = stablehlo.multiply %1488, %1496 : tensor<3xf64>
    %1508 = stablehlo.subtract %1506, %1507 : tensor<3xf64>
    %1509 = stablehlo.multiply %1494, %1490 : tensor<3xf64>
    %1510 = stablehlo.add %1508, %1509 : tensor<3xf64>
    %1511 = stablehlo.multiply %1500, %1485 : tensor<3xf64>
    %1512 = stablehlo.add %1510, %1511 : tensor<3xf64>
    %1513 = stablehlo.reshape %1512 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1514 = stablehlo.multiply %1480, %1496 : tensor<3xf64>
    %1515 = stablehlo.multiply %1488, %1502 : tensor<3xf64>
    %1516 = stablehlo.add %1514, %1515 : tensor<3xf64>
    %1517 = stablehlo.multiply %1494, %1485 : tensor<3xf64>
    %1518 = stablehlo.subtract %1516, %1517 : tensor<3xf64>
    %1519 = stablehlo.multiply %1500, %1490 : tensor<3xf64>
    %1520 = stablehlo.add %1518, %1519 : tensor<3xf64>
    %1521 = stablehlo.reshape %1520 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1522 = stablehlo.multiply %1480, %1490 : tensor<3xf64>
    %1523 = stablehlo.multiply %1488, %1485 : tensor<3xf64>
    %1524 = stablehlo.subtract %1522, %1523 : tensor<3xf64>
    %1525 = stablehlo.multiply %1494, %1502 : tensor<3xf64>
    %1526 = stablehlo.subtract %1524, %1525 : tensor<3xf64>
    %1527 = stablehlo.multiply %1500, %1496 : tensor<3xf64>
    %1528 = stablehlo.subtract %1526, %1527 : tensor<3xf64>
    %1529 = stablehlo.reshape %1528 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1530 = stablehlo.concatenate %1505, %1513, %1521, %1529, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1531 = stablehlo.slice %1530 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1532 = stablehlo.reshape %1531 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1533 = stablehlo.slice %1348 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1534 = stablehlo.reshape %1533 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1535 = stablehlo.negate %1534 : tensor<3xf64>
    %1536 = stablehlo.reshape %1535 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1537 = stablehlo.slice %1348 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1538 = stablehlo.reshape %1537 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1539 = stablehlo.negate %1538 : tensor<3xf64>
    %1540 = stablehlo.reshape %1539 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1541 = stablehlo.slice %1348 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1542 = stablehlo.reshape %1541 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1543 = stablehlo.negate %1542 : tensor<3xf64>
    %1544 = stablehlo.reshape %1543 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1545 = stablehlo.slice %1348 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1546 = stablehlo.reshape %1545 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1547 = stablehlo.reshape %1546 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1548 = stablehlo.concatenate %1536, %1540, %1544, %1547, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1549 = stablehlo.dot_general %1348, %1348, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1550 = stablehlo.broadcast_in_dim %1549, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1551 = stablehlo.divide %1548, %1550 : tensor<3x4xf64>
    %1552 = stablehlo.slice %1551 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1553 = stablehlo.reshape %1552 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1554 = stablehlo.multiply %1532, %1553 : tensor<3xf64>
    %1555 = stablehlo.slice %1530 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1556 = stablehlo.reshape %1555 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1557 = stablehlo.slice %1551 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1558 = stablehlo.reshape %1557 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1559 = stablehlo.multiply %1556, %1558 : tensor<3xf64>
    %1560 = stablehlo.add %1554, %1559 : tensor<3xf64>
    %1561 = stablehlo.slice %1530 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1562 = stablehlo.reshape %1561 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1563 = stablehlo.slice %1551 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1564 = stablehlo.reshape %1563 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1565 = stablehlo.multiply %1562, %1564 : tensor<3xf64>
    %1566 = stablehlo.add %1560, %1565 : tensor<3xf64>
    %1567 = stablehlo.slice %1530 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1568 = stablehlo.reshape %1567 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1569 = stablehlo.slice %1551 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1570 = stablehlo.reshape %1569 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1571 = stablehlo.multiply %1568, %1570 : tensor<3xf64>
    %1572 = stablehlo.subtract %1566, %1571 : tensor<3xf64>
    %1573 = stablehlo.reshape %1572 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1574 = stablehlo.multiply %1532, %1570 : tensor<3xf64>
    %1575 = stablehlo.multiply %1556, %1564 : tensor<3xf64>
    %1576 = stablehlo.subtract %1574, %1575 : tensor<3xf64>
    %1577 = stablehlo.multiply %1562, %1558 : tensor<3xf64>
    %1578 = stablehlo.add %1576, %1577 : tensor<3xf64>
    %1579 = stablehlo.multiply %1568, %1553 : tensor<3xf64>
    %1580 = stablehlo.add %1578, %1579 : tensor<3xf64>
    %1581 = stablehlo.reshape %1580 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1582 = stablehlo.multiply %1532, %1564 : tensor<3xf64>
    %1583 = stablehlo.multiply %1556, %1570 : tensor<3xf64>
    %1584 = stablehlo.add %1582, %1583 : tensor<3xf64>
    %1585 = stablehlo.multiply %1562, %1553 : tensor<3xf64>
    %1586 = stablehlo.subtract %1584, %1585 : tensor<3xf64>
    %1587 = stablehlo.multiply %1568, %1558 : tensor<3xf64>
    %1588 = stablehlo.add %1586, %1587 : tensor<3xf64>
    %1589 = stablehlo.reshape %1588 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1590 = stablehlo.multiply %1532, %1558 : tensor<3xf64>
    %1591 = stablehlo.multiply %1556, %1553 : tensor<3xf64>
    %1592 = stablehlo.subtract %1590, %1591 : tensor<3xf64>
    %1593 = stablehlo.multiply %1562, %1570 : tensor<3xf64>
    %1594 = stablehlo.subtract %1592, %1593 : tensor<3xf64>
    %1595 = stablehlo.multiply %1568, %1564 : tensor<3xf64>
    %1596 = stablehlo.subtract %1594, %1595 : tensor<3xf64>
    %1597 = stablehlo.reshape %1596 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1598 = stablehlo.concatenate %1573, %1581, %1589, %1597, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1599 = stablehlo.slice %1598 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1600 = stablehlo.reshape %1599 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1601 = stablehlo.reshape %1600 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1602 = stablehlo.slice %1598 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1603 = stablehlo.reshape %1602 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1604 = stablehlo.reshape %1603 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1605 = stablehlo.slice %1598 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1606 = stablehlo.reshape %1605 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1607 = stablehlo.reshape %1606 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1608 = stablehlo.concatenate %1601, %1604, %1607, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %1609 = stablehlo.concatenate %1478, %1608, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %1610 = stablehlo.slice %1609 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %1611 = stablehlo.slice %arg3 [0:3, 0:3] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %1612 = stablehlo.divide %1610, %1611 : tensor<3x3xf64>
    %1613 = stablehlo.slice %1609 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %1614 = stablehlo.slice %arg3 [0:3, 6:7] : (tensor<3x7xf64>) -> tensor<3x1xf64>
    %1615 = stablehlo.reshape %1614 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1616 = stablehlo.broadcast_in_dim %1615, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
    %1617 = stablehlo.divide %1613, %1616 : tensor<3x3xf64>
    %1618 = stablehlo.concatenate %1612, %1617, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %1619 = stablehlo.slice %1618 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1620 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1621 = stablehlo.concatenate %1619, %1620, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1622 = stablehlo.slice %1621 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1623 = stablehlo.reshape %1622 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1624 = stablehlo.multiply %1329, %1623 : tensor<3xf64>
    %1625 = stablehlo.slice %1327 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1626 = stablehlo.reshape %1625 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1627 = stablehlo.slice %1621 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1628 = stablehlo.reshape %1627 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1629 = stablehlo.multiply %1626, %1628 : tensor<3xf64>
    %1630 = stablehlo.add %1624, %1629 : tensor<3xf64>
    %1631 = stablehlo.slice %1327 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1632 = stablehlo.reshape %1631 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1633 = stablehlo.slice %1621 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1634 = stablehlo.reshape %1633 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1635 = stablehlo.multiply %1632, %1634 : tensor<3xf64>
    %1636 = stablehlo.add %1630, %1635 : tensor<3xf64>
    %1637 = stablehlo.slice %1327 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1638 = stablehlo.reshape %1637 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1639 = stablehlo.slice %1621 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1640 = stablehlo.reshape %1639 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1641 = stablehlo.multiply %1638, %1640 : tensor<3xf64>
    %1642 = stablehlo.subtract %1636, %1641 : tensor<3xf64>
    %1643 = stablehlo.reshape %1642 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1644 = stablehlo.multiply %1329, %1640 : tensor<3xf64>
    %1645 = stablehlo.multiply %1626, %1634 : tensor<3xf64>
    %1646 = stablehlo.subtract %1644, %1645 : tensor<3xf64>
    %1647 = stablehlo.multiply %1632, %1628 : tensor<3xf64>
    %1648 = stablehlo.add %1646, %1647 : tensor<3xf64>
    %1649 = stablehlo.multiply %1638, %1623 : tensor<3xf64>
    %1650 = stablehlo.add %1648, %1649 : tensor<3xf64>
    %1651 = stablehlo.reshape %1650 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1652 = stablehlo.multiply %1329, %1634 : tensor<3xf64>
    %1653 = stablehlo.multiply %1626, %1640 : tensor<3xf64>
    %1654 = stablehlo.add %1652, %1653 : tensor<3xf64>
    %1655 = stablehlo.multiply %1632, %1623 : tensor<3xf64>
    %1656 = stablehlo.subtract %1654, %1655 : tensor<3xf64>
    %1657 = stablehlo.multiply %1638, %1628 : tensor<3xf64>
    %1658 = stablehlo.add %1656, %1657 : tensor<3xf64>
    %1659 = stablehlo.reshape %1658 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1660 = stablehlo.multiply %1329, %1628 : tensor<3xf64>
    %1661 = stablehlo.multiply %1626, %1623 : tensor<3xf64>
    %1662 = stablehlo.subtract %1660, %1661 : tensor<3xf64>
    %1663 = stablehlo.multiply %1632, %1640 : tensor<3xf64>
    %1664 = stablehlo.subtract %1662, %1663 : tensor<3xf64>
    %1665 = stablehlo.multiply %1638, %1634 : tensor<3xf64>
    %1666 = stablehlo.subtract %1664, %1665 : tensor<3xf64>
    %1667 = stablehlo.reshape %1666 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1668 = stablehlo.concatenate %1643, %1651, %1659, %1667, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1669 = stablehlo.slice %1668 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1670 = stablehlo.reshape %1669 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1671 = stablehlo.slice %1327 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1672 = stablehlo.reshape %1671 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1673 = stablehlo.negate %1672 : tensor<3xf64>
    %1674 = stablehlo.reshape %1673 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1675 = stablehlo.slice %1327 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1676 = stablehlo.reshape %1675 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1677 = stablehlo.negate %1676 : tensor<3xf64>
    %1678 = stablehlo.reshape %1677 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1679 = stablehlo.slice %1327 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1680 = stablehlo.reshape %1679 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1681 = stablehlo.negate %1680 : tensor<3xf64>
    %1682 = stablehlo.reshape %1681 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1683 = stablehlo.slice %1327 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1684 = stablehlo.reshape %1683 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1685 = stablehlo.reshape %1684 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1686 = stablehlo.concatenate %1674, %1678, %1682, %1685, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1687 = stablehlo.dot_general %1327, %1327, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1689 = stablehlo.divide %1686, %1688 : tensor<3x4xf64>
    %1690 = stablehlo.slice %1689 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1691 = stablehlo.reshape %1690 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1692 = stablehlo.multiply %1670, %1691 : tensor<3xf64>
    %1693 = stablehlo.slice %1668 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1694 = stablehlo.reshape %1693 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1695 = stablehlo.slice %1689 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1696 = stablehlo.reshape %1695 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1697 = stablehlo.multiply %1694, %1696 : tensor<3xf64>
    %1698 = stablehlo.add %1692, %1697 : tensor<3xf64>
    %1699 = stablehlo.slice %1668 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1700 = stablehlo.reshape %1699 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1701 = stablehlo.slice %1689 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1702 = stablehlo.reshape %1701 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1703 = stablehlo.multiply %1700, %1702 : tensor<3xf64>
    %1704 = stablehlo.add %1698, %1703 : tensor<3xf64>
    %1705 = stablehlo.slice %1668 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1706 = stablehlo.reshape %1705 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1707 = stablehlo.slice %1689 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1708 = stablehlo.reshape %1707 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1709 = stablehlo.multiply %1706, %1708 : tensor<3xf64>
    %1710 = stablehlo.subtract %1704, %1709 : tensor<3xf64>
    %1711 = stablehlo.reshape %1710 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1712 = stablehlo.multiply %1670, %1708 : tensor<3xf64>
    %1713 = stablehlo.multiply %1694, %1702 : tensor<3xf64>
    %1714 = stablehlo.subtract %1712, %1713 : tensor<3xf64>
    %1715 = stablehlo.multiply %1700, %1696 : tensor<3xf64>
    %1716 = stablehlo.add %1714, %1715 : tensor<3xf64>
    %1717 = stablehlo.multiply %1706, %1691 : tensor<3xf64>
    %1718 = stablehlo.add %1716, %1717 : tensor<3xf64>
    %1719 = stablehlo.reshape %1718 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1720 = stablehlo.multiply %1670, %1702 : tensor<3xf64>
    %1721 = stablehlo.multiply %1694, %1708 : tensor<3xf64>
    %1722 = stablehlo.add %1720, %1721 : tensor<3xf64>
    %1723 = stablehlo.multiply %1700, %1691 : tensor<3xf64>
    %1724 = stablehlo.subtract %1722, %1723 : tensor<3xf64>
    %1725 = stablehlo.multiply %1706, %1696 : tensor<3xf64>
    %1726 = stablehlo.add %1724, %1725 : tensor<3xf64>
    %1727 = stablehlo.reshape %1726 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1728 = stablehlo.multiply %1670, %1696 : tensor<3xf64>
    %1729 = stablehlo.multiply %1694, %1691 : tensor<3xf64>
    %1730 = stablehlo.subtract %1728, %1729 : tensor<3xf64>
    %1731 = stablehlo.multiply %1700, %1708 : tensor<3xf64>
    %1732 = stablehlo.subtract %1730, %1731 : tensor<3xf64>
    %1733 = stablehlo.multiply %1706, %1702 : tensor<3xf64>
    %1734 = stablehlo.subtract %1732, %1733 : tensor<3xf64>
    %1735 = stablehlo.reshape %1734 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1736 = stablehlo.concatenate %1711, %1719, %1727, %1735, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1737 = stablehlo.slice %1736 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1738 = stablehlo.reshape %1737 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1739 = stablehlo.reshape %1738 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1740 = stablehlo.slice %1736 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1741 = stablehlo.reshape %1740 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1742 = stablehlo.reshape %1741 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1743 = stablehlo.slice %1736 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1744 = stablehlo.reshape %1743 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1745 = stablehlo.reshape %1744 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1746 = stablehlo.concatenate %1739, %1742, %1745, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %1747 = stablehlo.slice %1327 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1748 = stablehlo.reshape %1747 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1749 = stablehlo.slice %1618 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1750 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1751 = stablehlo.concatenate %1749, %1750, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1752 = stablehlo.slice %1751 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1753 = stablehlo.reshape %1752 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1754 = stablehlo.multiply %1748, %1753 : tensor<3xf64>
    %1755 = stablehlo.slice %1327 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1756 = stablehlo.reshape %1755 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1757 = stablehlo.slice %1751 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1758 = stablehlo.reshape %1757 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1759 = stablehlo.multiply %1756, %1758 : tensor<3xf64>
    %1760 = stablehlo.add %1754, %1759 : tensor<3xf64>
    %1761 = stablehlo.slice %1327 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1762 = stablehlo.reshape %1761 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1763 = stablehlo.slice %1751 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1764 = stablehlo.reshape %1763 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1765 = stablehlo.multiply %1762, %1764 : tensor<3xf64>
    %1766 = stablehlo.add %1760, %1765 : tensor<3xf64>
    %1767 = stablehlo.slice %1327 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1768 = stablehlo.reshape %1767 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1769 = stablehlo.slice %1751 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1770 = stablehlo.reshape %1769 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1771 = stablehlo.multiply %1768, %1770 : tensor<3xf64>
    %1772 = stablehlo.subtract %1766, %1771 : tensor<3xf64>
    %1773 = stablehlo.reshape %1772 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1774 = stablehlo.multiply %1748, %1770 : tensor<3xf64>
    %1775 = stablehlo.multiply %1756, %1764 : tensor<3xf64>
    %1776 = stablehlo.subtract %1774, %1775 : tensor<3xf64>
    %1777 = stablehlo.multiply %1762, %1758 : tensor<3xf64>
    %1778 = stablehlo.add %1776, %1777 : tensor<3xf64>
    %1779 = stablehlo.multiply %1768, %1753 : tensor<3xf64>
    %1780 = stablehlo.add %1778, %1779 : tensor<3xf64>
    %1781 = stablehlo.reshape %1780 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1782 = stablehlo.multiply %1748, %1764 : tensor<3xf64>
    %1783 = stablehlo.multiply %1756, %1770 : tensor<3xf64>
    %1784 = stablehlo.add %1782, %1783 : tensor<3xf64>
    %1785 = stablehlo.multiply %1762, %1753 : tensor<3xf64>
    %1786 = stablehlo.subtract %1784, %1785 : tensor<3xf64>
    %1787 = stablehlo.multiply %1768, %1758 : tensor<3xf64>
    %1788 = stablehlo.add %1786, %1787 : tensor<3xf64>
    %1789 = stablehlo.reshape %1788 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1790 = stablehlo.multiply %1748, %1758 : tensor<3xf64>
    %1791 = stablehlo.multiply %1756, %1753 : tensor<3xf64>
    %1792 = stablehlo.subtract %1790, %1791 : tensor<3xf64>
    %1793 = stablehlo.multiply %1762, %1770 : tensor<3xf64>
    %1794 = stablehlo.subtract %1792, %1793 : tensor<3xf64>
    %1795 = stablehlo.multiply %1768, %1764 : tensor<3xf64>
    %1796 = stablehlo.subtract %1794, %1795 : tensor<3xf64>
    %1797 = stablehlo.reshape %1796 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1798 = stablehlo.concatenate %1773, %1781, %1789, %1797, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1799 = stablehlo.slice %1798 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1800 = stablehlo.reshape %1799 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1801 = stablehlo.slice %1327 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1802 = stablehlo.reshape %1801 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1803 = stablehlo.negate %1802 : tensor<3xf64>
    %1804 = stablehlo.reshape %1803 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1805 = stablehlo.slice %1327 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1806 = stablehlo.reshape %1805 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1807 = stablehlo.negate %1806 : tensor<3xf64>
    %1808 = stablehlo.reshape %1807 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1809 = stablehlo.slice %1327 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1810 = stablehlo.reshape %1809 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1811 = stablehlo.negate %1810 : tensor<3xf64>
    %1812 = stablehlo.reshape %1811 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1813 = stablehlo.slice %1327 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1814 = stablehlo.reshape %1813 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1815 = stablehlo.reshape %1814 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1816 = stablehlo.concatenate %1804, %1808, %1812, %1815, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1817 = stablehlo.dot_general %1327, %1327, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1818 = stablehlo.broadcast_in_dim %1817, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1819 = stablehlo.divide %1816, %1818 : tensor<3x4xf64>
    %1820 = stablehlo.slice %1819 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1821 = stablehlo.reshape %1820 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1822 = stablehlo.multiply %1800, %1821 : tensor<3xf64>
    %1823 = stablehlo.slice %1798 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1824 = stablehlo.reshape %1823 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1825 = stablehlo.slice %1819 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1826 = stablehlo.reshape %1825 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1827 = stablehlo.multiply %1824, %1826 : tensor<3xf64>
    %1828 = stablehlo.add %1822, %1827 : tensor<3xf64>
    %1829 = stablehlo.slice %1798 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1830 = stablehlo.reshape %1829 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1831 = stablehlo.slice %1819 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1832 = stablehlo.reshape %1831 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1833 = stablehlo.multiply %1830, %1832 : tensor<3xf64>
    %1834 = stablehlo.add %1828, %1833 : tensor<3xf64>
    %1835 = stablehlo.slice %1798 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1836 = stablehlo.reshape %1835 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1837 = stablehlo.slice %1819 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1838 = stablehlo.reshape %1837 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1839 = stablehlo.multiply %1836, %1838 : tensor<3xf64>
    %1840 = stablehlo.subtract %1834, %1839 : tensor<3xf64>
    %1841 = stablehlo.reshape %1840 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1842 = stablehlo.multiply %1800, %1838 : tensor<3xf64>
    %1843 = stablehlo.multiply %1824, %1832 : tensor<3xf64>
    %1844 = stablehlo.subtract %1842, %1843 : tensor<3xf64>
    %1845 = stablehlo.multiply %1830, %1826 : tensor<3xf64>
    %1846 = stablehlo.add %1844, %1845 : tensor<3xf64>
    %1847 = stablehlo.multiply %1836, %1821 : tensor<3xf64>
    %1848 = stablehlo.add %1846, %1847 : tensor<3xf64>
    %1849 = stablehlo.reshape %1848 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1850 = stablehlo.multiply %1800, %1832 : tensor<3xf64>
    %1851 = stablehlo.multiply %1824, %1838 : tensor<3xf64>
    %1852 = stablehlo.add %1850, %1851 : tensor<3xf64>
    %1853 = stablehlo.multiply %1830, %1821 : tensor<3xf64>
    %1854 = stablehlo.subtract %1852, %1853 : tensor<3xf64>
    %1855 = stablehlo.multiply %1836, %1826 : tensor<3xf64>
    %1856 = stablehlo.add %1854, %1855 : tensor<3xf64>
    %1857 = stablehlo.reshape %1856 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1858 = stablehlo.multiply %1800, %1826 : tensor<3xf64>
    %1859 = stablehlo.multiply %1824, %1821 : tensor<3xf64>
    %1860 = stablehlo.subtract %1858, %1859 : tensor<3xf64>
    %1861 = stablehlo.multiply %1830, %1838 : tensor<3xf64>
    %1862 = stablehlo.subtract %1860, %1861 : tensor<3xf64>
    %1863 = stablehlo.multiply %1836, %1832 : tensor<3xf64>
    %1864 = stablehlo.subtract %1862, %1863 : tensor<3xf64>
    %1865 = stablehlo.reshape %1864 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1866 = stablehlo.concatenate %1841, %1849, %1857, %1865, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1867 = stablehlo.slice %1866 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1868 = stablehlo.reshape %1867 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1869 = stablehlo.reshape %1868 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1870 = stablehlo.slice %1866 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1871 = stablehlo.reshape %1870 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1872 = stablehlo.reshape %1871 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1873 = stablehlo.slice %1866 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1874 = stablehlo.reshape %1873 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1875 = stablehlo.reshape %1874 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1876 = stablehlo.concatenate %1869, %1872, %1875, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %1877 = stablehlo.concatenate %1746, %1876, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %1878 = stablehlo.slice %arg2 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1879 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1880 = stablehlo.reshape %arg5 : (tensor<f64>) -> tensor<f64>
    %1881 = stablehlo.broadcast_in_dim %1880, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1882 = stablehlo.multiply %1879, %1881 : tensor<3xf64>
    %1883 = stablehlo.broadcast_in_dim %1882, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %1884 = stablehlo.multiply %1883, %arg6 : tensor<3x6xf64>
    %1885 = stablehlo.slice %1884 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_24 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1886 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %1887 = stablehlo.divide %1885, %1886 : tensor<3x3xf64>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1888 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1889 = stablehlo.concatenate %1887, %1888, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1890 = stablehlo.slice %1889 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1891 = stablehlo.reshape %1890 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1892 = stablehlo.slice %1878 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1893 = stablehlo.reshape %1892 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1894 = stablehlo.multiply %1891, %1893 : tensor<3xf64>
    %1895 = stablehlo.slice %1889 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1896 = stablehlo.reshape %1895 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1897 = stablehlo.slice %1878 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1898 = stablehlo.reshape %1897 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1899 = stablehlo.multiply %1896, %1898 : tensor<3xf64>
    %1900 = stablehlo.add %1894, %1899 : tensor<3xf64>
    %1901 = stablehlo.slice %1889 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1902 = stablehlo.reshape %1901 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1903 = stablehlo.slice %1878 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1904 = stablehlo.reshape %1903 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1905 = stablehlo.multiply %1902, %1904 : tensor<3xf64>
    %1906 = stablehlo.add %1900, %1905 : tensor<3xf64>
    %1907 = stablehlo.slice %1889 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1908 = stablehlo.reshape %1907 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1909 = stablehlo.slice %1878 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1910 = stablehlo.reshape %1909 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1911 = stablehlo.multiply %1908, %1910 : tensor<3xf64>
    %1912 = stablehlo.subtract %1906, %1911 : tensor<3xf64>
    %1913 = stablehlo.reshape %1912 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1914 = stablehlo.multiply %1891, %1910 : tensor<3xf64>
    %1915 = stablehlo.multiply %1896, %1904 : tensor<3xf64>
    %1916 = stablehlo.subtract %1914, %1915 : tensor<3xf64>
    %1917 = stablehlo.multiply %1902, %1898 : tensor<3xf64>
    %1918 = stablehlo.add %1916, %1917 : tensor<3xf64>
    %1919 = stablehlo.multiply %1908, %1893 : tensor<3xf64>
    %1920 = stablehlo.add %1918, %1919 : tensor<3xf64>
    %1921 = stablehlo.reshape %1920 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1922 = stablehlo.multiply %1891, %1904 : tensor<3xf64>
    %1923 = stablehlo.multiply %1896, %1910 : tensor<3xf64>
    %1924 = stablehlo.add %1922, %1923 : tensor<3xf64>
    %1925 = stablehlo.multiply %1902, %1893 : tensor<3xf64>
    %1926 = stablehlo.subtract %1924, %1925 : tensor<3xf64>
    %1927 = stablehlo.multiply %1908, %1898 : tensor<3xf64>
    %1928 = stablehlo.add %1926, %1927 : tensor<3xf64>
    %1929 = stablehlo.reshape %1928 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1930 = stablehlo.multiply %1891, %1898 : tensor<3xf64>
    %1931 = stablehlo.multiply %1896, %1893 : tensor<3xf64>
    %1932 = stablehlo.subtract %1930, %1931 : tensor<3xf64>
    %1933 = stablehlo.multiply %1902, %1910 : tensor<3xf64>
    %1934 = stablehlo.subtract %1932, %1933 : tensor<3xf64>
    %1935 = stablehlo.multiply %1908, %1904 : tensor<3xf64>
    %1936 = stablehlo.subtract %1934, %1935 : tensor<3xf64>
    %1937 = stablehlo.reshape %1936 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1938 = stablehlo.concatenate %1913, %1921, %1929, %1937, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1939 = stablehlo.add %1878, %1938 : tensor<3x4xf64>
    %1940 = stablehlo.dot_general %1939, %1939, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1941 = stablehlo.sqrt %1940 : tensor<3xf64>
    %1942 = stablehlo.broadcast_in_dim %1941, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1943 = stablehlo.divide %1939, %1942 : tensor<3x4xf64>
    %1944 = stablehlo.slice %arg2 [0:3, 4:7] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %1945 = stablehlo.slice %1884 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %1946 = stablehlo.add %1944, %1945 : tensor<3x3xf64>
    %1947 = stablehlo.concatenate %1943, %1946, dim = 1 : (tensor<3x4xf64>, tensor<3x3xf64>) -> tensor<3x7xf64>
    %1948 = stablehlo.broadcast_in_dim %1882, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %1949 = stablehlo.multiply %1948, %1877 : tensor<3x6xf64>
    %1950 = stablehlo.add %arg6, %1949 : tensor<3x6xf64>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1951 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %1952 = call @inner(%1947, %arg3, %1951) : (tensor<3x7xf64>, tensor<3x7xf64>, tensor<3x6xf64>) -> tensor<3x6xf64>
    %1953 = stablehlo.slice %1947 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %1954 = stablehlo.slice %1953 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1955 = stablehlo.reshape %1954 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1956 = stablehlo.slice %1953 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1957 = stablehlo.reshape %1956 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1958 = stablehlo.negate %1957 : tensor<3xf64>
    %1959 = stablehlo.reshape %1958 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1960 = stablehlo.slice %1953 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1961 = stablehlo.reshape %1960 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1962 = stablehlo.negate %1961 : tensor<3xf64>
    %1963 = stablehlo.reshape %1962 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1964 = stablehlo.slice %1953 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1965 = stablehlo.reshape %1964 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1966 = stablehlo.negate %1965 : tensor<3xf64>
    %1967 = stablehlo.reshape %1966 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1968 = stablehlo.slice %1953 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1969 = stablehlo.reshape %1968 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1970 = stablehlo.reshape %1969 : (tensor<3xf64>) -> tensor<3x1xf64>
    %1971 = stablehlo.concatenate %1959, %1963, %1967, %1970, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1972 = stablehlo.dot_general %1953, %1953, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %1973 = stablehlo.broadcast_in_dim %1972, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %1974 = stablehlo.divide %1971, %1973 : tensor<3x4xf64>
    %1975 = stablehlo.slice %1974 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1976 = stablehlo.reshape %1975 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1977 = stablehlo.slice %1952 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1978 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %1979 = stablehlo.concatenate %1977, %1978, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %1980 = stablehlo.slice %1979 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1981 = stablehlo.reshape %1980 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1982 = stablehlo.multiply %1976, %1981 : tensor<3xf64>
    %1983 = stablehlo.slice %1974 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1984 = stablehlo.reshape %1983 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1985 = stablehlo.slice %1979 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1986 = stablehlo.reshape %1985 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1987 = stablehlo.multiply %1984, %1986 : tensor<3xf64>
    %1988 = stablehlo.add %1982, %1987 : tensor<3xf64>
    %1989 = stablehlo.slice %1974 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1990 = stablehlo.reshape %1989 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1991 = stablehlo.slice %1979 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1992 = stablehlo.reshape %1991 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1993 = stablehlo.multiply %1990, %1992 : tensor<3xf64>
    %1994 = stablehlo.add %1988, %1993 : tensor<3xf64>
    %1995 = stablehlo.slice %1974 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1996 = stablehlo.reshape %1995 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1997 = stablehlo.slice %1979 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %1998 = stablehlo.reshape %1997 : (tensor<3x1xf64>) -> tensor<3xf64>
    %1999 = stablehlo.multiply %1996, %1998 : tensor<3xf64>
    %2000 = stablehlo.subtract %1994, %1999 : tensor<3xf64>
    %2001 = stablehlo.reshape %2000 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2002 = stablehlo.multiply %1976, %1998 : tensor<3xf64>
    %2003 = stablehlo.multiply %1984, %1992 : tensor<3xf64>
    %2004 = stablehlo.subtract %2002, %2003 : tensor<3xf64>
    %2005 = stablehlo.multiply %1990, %1986 : tensor<3xf64>
    %2006 = stablehlo.add %2004, %2005 : tensor<3xf64>
    %2007 = stablehlo.multiply %1996, %1981 : tensor<3xf64>
    %2008 = stablehlo.add %2006, %2007 : tensor<3xf64>
    %2009 = stablehlo.reshape %2008 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2010 = stablehlo.multiply %1976, %1992 : tensor<3xf64>
    %2011 = stablehlo.multiply %1984, %1998 : tensor<3xf64>
    %2012 = stablehlo.add %2010, %2011 : tensor<3xf64>
    %2013 = stablehlo.multiply %1990, %1981 : tensor<3xf64>
    %2014 = stablehlo.subtract %2012, %2013 : tensor<3xf64>
    %2015 = stablehlo.multiply %1996, %1986 : tensor<3xf64>
    %2016 = stablehlo.add %2014, %2015 : tensor<3xf64>
    %2017 = stablehlo.reshape %2016 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2018 = stablehlo.multiply %1976, %1986 : tensor<3xf64>
    %2019 = stablehlo.multiply %1984, %1981 : tensor<3xf64>
    %2020 = stablehlo.subtract %2018, %2019 : tensor<3xf64>
    %2021 = stablehlo.multiply %1990, %1998 : tensor<3xf64>
    %2022 = stablehlo.subtract %2020, %2021 : tensor<3xf64>
    %2023 = stablehlo.multiply %1996, %1992 : tensor<3xf64>
    %2024 = stablehlo.subtract %2022, %2023 : tensor<3xf64>
    %2025 = stablehlo.reshape %2024 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2026 = stablehlo.concatenate %2001, %2009, %2017, %2025, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2027 = stablehlo.slice %2026 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2028 = stablehlo.reshape %2027 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2029 = stablehlo.slice %1974 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2030 = stablehlo.reshape %2029 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2031 = stablehlo.negate %2030 : tensor<3xf64>
    %2032 = stablehlo.reshape %2031 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2033 = stablehlo.slice %1974 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2034 = stablehlo.reshape %2033 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2035 = stablehlo.negate %2034 : tensor<3xf64>
    %2036 = stablehlo.reshape %2035 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2037 = stablehlo.slice %1974 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2038 = stablehlo.reshape %2037 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2039 = stablehlo.negate %2038 : tensor<3xf64>
    %2040 = stablehlo.reshape %2039 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2041 = stablehlo.slice %1974 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2042 = stablehlo.reshape %2041 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2043 = stablehlo.reshape %2042 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2044 = stablehlo.concatenate %2032, %2036, %2040, %2043, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2045 = stablehlo.dot_general %1974, %1974, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %2046 = stablehlo.broadcast_in_dim %2045, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %2047 = stablehlo.divide %2044, %2046 : tensor<3x4xf64>
    %2048 = stablehlo.slice %2047 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2049 = stablehlo.reshape %2048 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2050 = stablehlo.multiply %2028, %2049 : tensor<3xf64>
    %2051 = stablehlo.slice %2026 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2052 = stablehlo.reshape %2051 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2053 = stablehlo.slice %2047 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2054 = stablehlo.reshape %2053 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2055 = stablehlo.multiply %2052, %2054 : tensor<3xf64>
    %2056 = stablehlo.add %2050, %2055 : tensor<3xf64>
    %2057 = stablehlo.slice %2026 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2058 = stablehlo.reshape %2057 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2059 = stablehlo.slice %2047 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2060 = stablehlo.reshape %2059 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2061 = stablehlo.multiply %2058, %2060 : tensor<3xf64>
    %2062 = stablehlo.add %2056, %2061 : tensor<3xf64>
    %2063 = stablehlo.slice %2026 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2064 = stablehlo.reshape %2063 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2065 = stablehlo.slice %2047 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2066 = stablehlo.reshape %2065 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2067 = stablehlo.multiply %2064, %2066 : tensor<3xf64>
    %2068 = stablehlo.subtract %2062, %2067 : tensor<3xf64>
    %2069 = stablehlo.reshape %2068 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2070 = stablehlo.multiply %2028, %2066 : tensor<3xf64>
    %2071 = stablehlo.multiply %2052, %2060 : tensor<3xf64>
    %2072 = stablehlo.subtract %2070, %2071 : tensor<3xf64>
    %2073 = stablehlo.multiply %2058, %2054 : tensor<3xf64>
    %2074 = stablehlo.add %2072, %2073 : tensor<3xf64>
    %2075 = stablehlo.multiply %2064, %2049 : tensor<3xf64>
    %2076 = stablehlo.add %2074, %2075 : tensor<3xf64>
    %2077 = stablehlo.reshape %2076 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2078 = stablehlo.multiply %2028, %2060 : tensor<3xf64>
    %2079 = stablehlo.multiply %2052, %2066 : tensor<3xf64>
    %2080 = stablehlo.add %2078, %2079 : tensor<3xf64>
    %2081 = stablehlo.multiply %2058, %2049 : tensor<3xf64>
    %2082 = stablehlo.subtract %2080, %2081 : tensor<3xf64>
    %2083 = stablehlo.multiply %2064, %2054 : tensor<3xf64>
    %2084 = stablehlo.add %2082, %2083 : tensor<3xf64>
    %2085 = stablehlo.reshape %2084 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2086 = stablehlo.multiply %2028, %2054 : tensor<3xf64>
    %2087 = stablehlo.multiply %2052, %2049 : tensor<3xf64>
    %2088 = stablehlo.subtract %2086, %2087 : tensor<3xf64>
    %2089 = stablehlo.multiply %2058, %2066 : tensor<3xf64>
    %2090 = stablehlo.subtract %2088, %2089 : tensor<3xf64>
    %2091 = stablehlo.multiply %2064, %2060 : tensor<3xf64>
    %2092 = stablehlo.subtract %2090, %2091 : tensor<3xf64>
    %2093 = stablehlo.reshape %2092 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2094 = stablehlo.concatenate %2069, %2077, %2085, %2093, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2095 = stablehlo.slice %2094 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2096 = stablehlo.reshape %2095 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2097 = stablehlo.reshape %2096 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2098 = stablehlo.slice %2094 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2099 = stablehlo.reshape %2098 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2100 = stablehlo.reshape %2099 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2101 = stablehlo.slice %2094 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2102 = stablehlo.reshape %2101 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2103 = stablehlo.reshape %2102 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2104 = stablehlo.concatenate %2097, %2100, %2103, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %2105 = stablehlo.slice %1974 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2106 = stablehlo.reshape %2105 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2107 = stablehlo.slice %1952 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2108 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %2109 = stablehlo.concatenate %2107, %2108, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2110 = stablehlo.slice %2109 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2111 = stablehlo.reshape %2110 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2112 = stablehlo.multiply %2106, %2111 : tensor<3xf64>
    %2113 = stablehlo.slice %1974 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2114 = stablehlo.reshape %2113 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2115 = stablehlo.slice %2109 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2116 = stablehlo.reshape %2115 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2117 = stablehlo.multiply %2114, %2116 : tensor<3xf64>
    %2118 = stablehlo.add %2112, %2117 : tensor<3xf64>
    %2119 = stablehlo.slice %1974 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2120 = stablehlo.reshape %2119 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2121 = stablehlo.slice %2109 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2122 = stablehlo.reshape %2121 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2123 = stablehlo.multiply %2120, %2122 : tensor<3xf64>
    %2124 = stablehlo.add %2118, %2123 : tensor<3xf64>
    %2125 = stablehlo.slice %1974 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2126 = stablehlo.reshape %2125 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2127 = stablehlo.slice %2109 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2128 = stablehlo.reshape %2127 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2129 = stablehlo.multiply %2126, %2128 : tensor<3xf64>
    %2130 = stablehlo.subtract %2124, %2129 : tensor<3xf64>
    %2131 = stablehlo.reshape %2130 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2132 = stablehlo.multiply %2106, %2128 : tensor<3xf64>
    %2133 = stablehlo.multiply %2114, %2122 : tensor<3xf64>
    %2134 = stablehlo.subtract %2132, %2133 : tensor<3xf64>
    %2135 = stablehlo.multiply %2120, %2116 : tensor<3xf64>
    %2136 = stablehlo.add %2134, %2135 : tensor<3xf64>
    %2137 = stablehlo.multiply %2126, %2111 : tensor<3xf64>
    %2138 = stablehlo.add %2136, %2137 : tensor<3xf64>
    %2139 = stablehlo.reshape %2138 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2140 = stablehlo.multiply %2106, %2122 : tensor<3xf64>
    %2141 = stablehlo.multiply %2114, %2128 : tensor<3xf64>
    %2142 = stablehlo.add %2140, %2141 : tensor<3xf64>
    %2143 = stablehlo.multiply %2120, %2111 : tensor<3xf64>
    %2144 = stablehlo.subtract %2142, %2143 : tensor<3xf64>
    %2145 = stablehlo.multiply %2126, %2116 : tensor<3xf64>
    %2146 = stablehlo.add %2144, %2145 : tensor<3xf64>
    %2147 = stablehlo.reshape %2146 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2148 = stablehlo.multiply %2106, %2116 : tensor<3xf64>
    %2149 = stablehlo.multiply %2114, %2111 : tensor<3xf64>
    %2150 = stablehlo.subtract %2148, %2149 : tensor<3xf64>
    %2151 = stablehlo.multiply %2120, %2128 : tensor<3xf64>
    %2152 = stablehlo.subtract %2150, %2151 : tensor<3xf64>
    %2153 = stablehlo.multiply %2126, %2122 : tensor<3xf64>
    %2154 = stablehlo.subtract %2152, %2153 : tensor<3xf64>
    %2155 = stablehlo.reshape %2154 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2156 = stablehlo.concatenate %2131, %2139, %2147, %2155, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2157 = stablehlo.slice %2156 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2158 = stablehlo.reshape %2157 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2159 = stablehlo.slice %1974 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2160 = stablehlo.reshape %2159 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2161 = stablehlo.negate %2160 : tensor<3xf64>
    %2162 = stablehlo.reshape %2161 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2163 = stablehlo.slice %1974 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2164 = stablehlo.reshape %2163 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2165 = stablehlo.negate %2164 : tensor<3xf64>
    %2166 = stablehlo.reshape %2165 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2167 = stablehlo.slice %1974 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2168 = stablehlo.reshape %2167 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2169 = stablehlo.negate %2168 : tensor<3xf64>
    %2170 = stablehlo.reshape %2169 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2171 = stablehlo.slice %1974 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2172 = stablehlo.reshape %2171 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2173 = stablehlo.reshape %2172 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2174 = stablehlo.concatenate %2162, %2166, %2170, %2173, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2175 = stablehlo.dot_general %1974, %1974, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %2176 = stablehlo.broadcast_in_dim %2175, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %2177 = stablehlo.divide %2174, %2176 : tensor<3x4xf64>
    %2178 = stablehlo.slice %2177 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2179 = stablehlo.reshape %2178 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2180 = stablehlo.multiply %2158, %2179 : tensor<3xf64>
    %2181 = stablehlo.slice %2156 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2182 = stablehlo.reshape %2181 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2183 = stablehlo.slice %2177 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2184 = stablehlo.reshape %2183 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2185 = stablehlo.multiply %2182, %2184 : tensor<3xf64>
    %2186 = stablehlo.add %2180, %2185 : tensor<3xf64>
    %2187 = stablehlo.slice %2156 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2188 = stablehlo.reshape %2187 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2189 = stablehlo.slice %2177 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2190 = stablehlo.reshape %2189 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2191 = stablehlo.multiply %2188, %2190 : tensor<3xf64>
    %2192 = stablehlo.add %2186, %2191 : tensor<3xf64>
    %2193 = stablehlo.slice %2156 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2194 = stablehlo.reshape %2193 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2195 = stablehlo.slice %2177 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2196 = stablehlo.reshape %2195 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2197 = stablehlo.multiply %2194, %2196 : tensor<3xf64>
    %2198 = stablehlo.subtract %2192, %2197 : tensor<3xf64>
    %2199 = stablehlo.reshape %2198 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2200 = stablehlo.multiply %2158, %2196 : tensor<3xf64>
    %2201 = stablehlo.multiply %2182, %2190 : tensor<3xf64>
    %2202 = stablehlo.subtract %2200, %2201 : tensor<3xf64>
    %2203 = stablehlo.multiply %2188, %2184 : tensor<3xf64>
    %2204 = stablehlo.add %2202, %2203 : tensor<3xf64>
    %2205 = stablehlo.multiply %2194, %2179 : tensor<3xf64>
    %2206 = stablehlo.add %2204, %2205 : tensor<3xf64>
    %2207 = stablehlo.reshape %2206 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2208 = stablehlo.multiply %2158, %2190 : tensor<3xf64>
    %2209 = stablehlo.multiply %2182, %2196 : tensor<3xf64>
    %2210 = stablehlo.add %2208, %2209 : tensor<3xf64>
    %2211 = stablehlo.multiply %2188, %2179 : tensor<3xf64>
    %2212 = stablehlo.subtract %2210, %2211 : tensor<3xf64>
    %2213 = stablehlo.multiply %2194, %2184 : tensor<3xf64>
    %2214 = stablehlo.add %2212, %2213 : tensor<3xf64>
    %2215 = stablehlo.reshape %2214 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2216 = stablehlo.multiply %2158, %2184 : tensor<3xf64>
    %2217 = stablehlo.multiply %2182, %2179 : tensor<3xf64>
    %2218 = stablehlo.subtract %2216, %2217 : tensor<3xf64>
    %2219 = stablehlo.multiply %2188, %2196 : tensor<3xf64>
    %2220 = stablehlo.subtract %2218, %2219 : tensor<3xf64>
    %2221 = stablehlo.multiply %2194, %2190 : tensor<3xf64>
    %2222 = stablehlo.subtract %2220, %2221 : tensor<3xf64>
    %2223 = stablehlo.reshape %2222 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2224 = stablehlo.concatenate %2199, %2207, %2215, %2223, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2225 = stablehlo.slice %2224 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2226 = stablehlo.reshape %2225 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2227 = stablehlo.reshape %2226 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2228 = stablehlo.slice %2224 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2229 = stablehlo.reshape %2228 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2230 = stablehlo.reshape %2229 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2231 = stablehlo.slice %2224 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2232 = stablehlo.reshape %2231 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2233 = stablehlo.reshape %2232 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2234 = stablehlo.concatenate %2227, %2230, %2233, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %2235 = stablehlo.concatenate %2104, %2234, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %2236 = stablehlo.slice %2235 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %2237 = stablehlo.slice %arg3 [0:3, 0:3] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %2238 = stablehlo.divide %2236, %2237 : tensor<3x3xf64>
    %2239 = stablehlo.slice %2235 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %2240 = stablehlo.slice %arg3 [0:3, 6:7] : (tensor<3x7xf64>) -> tensor<3x1xf64>
    %2241 = stablehlo.reshape %2240 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2242 = stablehlo.broadcast_in_dim %2241, dims = [0] : (tensor<3xf64>) -> tensor<3x3xf64>
    %2243 = stablehlo.divide %2239, %2242 : tensor<3x3xf64>
    %2244 = stablehlo.concatenate %2238, %2243, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %2245 = stablehlo.slice %2244 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2246 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %2247 = stablehlo.concatenate %2245, %2246, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2248 = stablehlo.slice %2247 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2249 = stablehlo.reshape %2248 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2250 = stablehlo.multiply %1955, %2249 : tensor<3xf64>
    %2251 = stablehlo.slice %1953 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2252 = stablehlo.reshape %2251 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2253 = stablehlo.slice %2247 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2254 = stablehlo.reshape %2253 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2255 = stablehlo.multiply %2252, %2254 : tensor<3xf64>
    %2256 = stablehlo.add %2250, %2255 : tensor<3xf64>
    %2257 = stablehlo.slice %1953 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2258 = stablehlo.reshape %2257 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2259 = stablehlo.slice %2247 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2260 = stablehlo.reshape %2259 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2261 = stablehlo.multiply %2258, %2260 : tensor<3xf64>
    %2262 = stablehlo.add %2256, %2261 : tensor<3xf64>
    %2263 = stablehlo.slice %1953 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2264 = stablehlo.reshape %2263 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2265 = stablehlo.slice %2247 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2266 = stablehlo.reshape %2265 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2267 = stablehlo.multiply %2264, %2266 : tensor<3xf64>
    %2268 = stablehlo.subtract %2262, %2267 : tensor<3xf64>
    %2269 = stablehlo.reshape %2268 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2270 = stablehlo.multiply %1955, %2266 : tensor<3xf64>
    %2271 = stablehlo.multiply %2252, %2260 : tensor<3xf64>
    %2272 = stablehlo.subtract %2270, %2271 : tensor<3xf64>
    %2273 = stablehlo.multiply %2258, %2254 : tensor<3xf64>
    %2274 = stablehlo.add %2272, %2273 : tensor<3xf64>
    %2275 = stablehlo.multiply %2264, %2249 : tensor<3xf64>
    %2276 = stablehlo.add %2274, %2275 : tensor<3xf64>
    %2277 = stablehlo.reshape %2276 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2278 = stablehlo.multiply %1955, %2260 : tensor<3xf64>
    %2279 = stablehlo.multiply %2252, %2266 : tensor<3xf64>
    %2280 = stablehlo.add %2278, %2279 : tensor<3xf64>
    %2281 = stablehlo.multiply %2258, %2249 : tensor<3xf64>
    %2282 = stablehlo.subtract %2280, %2281 : tensor<3xf64>
    %2283 = stablehlo.multiply %2264, %2254 : tensor<3xf64>
    %2284 = stablehlo.add %2282, %2283 : tensor<3xf64>
    %2285 = stablehlo.reshape %2284 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2286 = stablehlo.multiply %1955, %2254 : tensor<3xf64>
    %2287 = stablehlo.multiply %2252, %2249 : tensor<3xf64>
    %2288 = stablehlo.subtract %2286, %2287 : tensor<3xf64>
    %2289 = stablehlo.multiply %2258, %2266 : tensor<3xf64>
    %2290 = stablehlo.subtract %2288, %2289 : tensor<3xf64>
    %2291 = stablehlo.multiply %2264, %2260 : tensor<3xf64>
    %2292 = stablehlo.subtract %2290, %2291 : tensor<3xf64>
    %2293 = stablehlo.reshape %2292 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2294 = stablehlo.concatenate %2269, %2277, %2285, %2293, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2295 = stablehlo.slice %2294 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2296 = stablehlo.reshape %2295 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2297 = stablehlo.slice %1953 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2298 = stablehlo.reshape %2297 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2299 = stablehlo.negate %2298 : tensor<3xf64>
    %2300 = stablehlo.reshape %2299 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2301 = stablehlo.slice %1953 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2302 = stablehlo.reshape %2301 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2303 = stablehlo.negate %2302 : tensor<3xf64>
    %2304 = stablehlo.reshape %2303 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2305 = stablehlo.slice %1953 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2306 = stablehlo.reshape %2305 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2307 = stablehlo.negate %2306 : tensor<3xf64>
    %2308 = stablehlo.reshape %2307 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2309 = stablehlo.slice %1953 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2310 = stablehlo.reshape %2309 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2311 = stablehlo.reshape %2310 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2312 = stablehlo.concatenate %2300, %2304, %2308, %2311, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2313 = stablehlo.dot_general %1953, %1953, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %2314 = stablehlo.broadcast_in_dim %2313, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %2315 = stablehlo.divide %2312, %2314 : tensor<3x4xf64>
    %2316 = stablehlo.slice %2315 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2317 = stablehlo.reshape %2316 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2318 = stablehlo.multiply %2296, %2317 : tensor<3xf64>
    %2319 = stablehlo.slice %2294 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2320 = stablehlo.reshape %2319 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2321 = stablehlo.slice %2315 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2322 = stablehlo.reshape %2321 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2323 = stablehlo.multiply %2320, %2322 : tensor<3xf64>
    %2324 = stablehlo.add %2318, %2323 : tensor<3xf64>
    %2325 = stablehlo.slice %2294 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2326 = stablehlo.reshape %2325 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2327 = stablehlo.slice %2315 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2328 = stablehlo.reshape %2327 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2329 = stablehlo.multiply %2326, %2328 : tensor<3xf64>
    %2330 = stablehlo.add %2324, %2329 : tensor<3xf64>
    %2331 = stablehlo.slice %2294 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2332 = stablehlo.reshape %2331 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2333 = stablehlo.slice %2315 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2334 = stablehlo.reshape %2333 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2335 = stablehlo.multiply %2332, %2334 : tensor<3xf64>
    %2336 = stablehlo.subtract %2330, %2335 : tensor<3xf64>
    %2337 = stablehlo.reshape %2336 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2338 = stablehlo.multiply %2296, %2334 : tensor<3xf64>
    %2339 = stablehlo.multiply %2320, %2328 : tensor<3xf64>
    %2340 = stablehlo.subtract %2338, %2339 : tensor<3xf64>
    %2341 = stablehlo.multiply %2326, %2322 : tensor<3xf64>
    %2342 = stablehlo.add %2340, %2341 : tensor<3xf64>
    %2343 = stablehlo.multiply %2332, %2317 : tensor<3xf64>
    %2344 = stablehlo.add %2342, %2343 : tensor<3xf64>
    %2345 = stablehlo.reshape %2344 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2346 = stablehlo.multiply %2296, %2328 : tensor<3xf64>
    %2347 = stablehlo.multiply %2320, %2334 : tensor<3xf64>
    %2348 = stablehlo.add %2346, %2347 : tensor<3xf64>
    %2349 = stablehlo.multiply %2326, %2317 : tensor<3xf64>
    %2350 = stablehlo.subtract %2348, %2349 : tensor<3xf64>
    %2351 = stablehlo.multiply %2332, %2322 : tensor<3xf64>
    %2352 = stablehlo.add %2350, %2351 : tensor<3xf64>
    %2353 = stablehlo.reshape %2352 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2354 = stablehlo.multiply %2296, %2322 : tensor<3xf64>
    %2355 = stablehlo.multiply %2320, %2317 : tensor<3xf64>
    %2356 = stablehlo.subtract %2354, %2355 : tensor<3xf64>
    %2357 = stablehlo.multiply %2326, %2334 : tensor<3xf64>
    %2358 = stablehlo.subtract %2356, %2357 : tensor<3xf64>
    %2359 = stablehlo.multiply %2332, %2328 : tensor<3xf64>
    %2360 = stablehlo.subtract %2358, %2359 : tensor<3xf64>
    %2361 = stablehlo.reshape %2360 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2362 = stablehlo.concatenate %2337, %2345, %2353, %2361, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2363 = stablehlo.slice %2362 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2364 = stablehlo.reshape %2363 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2365 = stablehlo.reshape %2364 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2366 = stablehlo.slice %2362 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2367 = stablehlo.reshape %2366 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2368 = stablehlo.reshape %2367 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2369 = stablehlo.slice %2362 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2370 = stablehlo.reshape %2369 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2371 = stablehlo.reshape %2370 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2372 = stablehlo.concatenate %2365, %2368, %2371, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %2373 = stablehlo.slice %1953 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2374 = stablehlo.reshape %2373 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2375 = stablehlo.slice %2244 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2376 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %2377 = stablehlo.concatenate %2375, %2376, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2378 = stablehlo.slice %2377 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2379 = stablehlo.reshape %2378 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2380 = stablehlo.multiply %2374, %2379 : tensor<3xf64>
    %2381 = stablehlo.slice %1953 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2382 = stablehlo.reshape %2381 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2383 = stablehlo.slice %2377 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2384 = stablehlo.reshape %2383 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2385 = stablehlo.multiply %2382, %2384 : tensor<3xf64>
    %2386 = stablehlo.add %2380, %2385 : tensor<3xf64>
    %2387 = stablehlo.slice %1953 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2388 = stablehlo.reshape %2387 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2389 = stablehlo.slice %2377 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2390 = stablehlo.reshape %2389 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2391 = stablehlo.multiply %2388, %2390 : tensor<3xf64>
    %2392 = stablehlo.add %2386, %2391 : tensor<3xf64>
    %2393 = stablehlo.slice %1953 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2394 = stablehlo.reshape %2393 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2395 = stablehlo.slice %2377 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2396 = stablehlo.reshape %2395 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2397 = stablehlo.multiply %2394, %2396 : tensor<3xf64>
    %2398 = stablehlo.subtract %2392, %2397 : tensor<3xf64>
    %2399 = stablehlo.reshape %2398 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2400 = stablehlo.multiply %2374, %2396 : tensor<3xf64>
    %2401 = stablehlo.multiply %2382, %2390 : tensor<3xf64>
    %2402 = stablehlo.subtract %2400, %2401 : tensor<3xf64>
    %2403 = stablehlo.multiply %2388, %2384 : tensor<3xf64>
    %2404 = stablehlo.add %2402, %2403 : tensor<3xf64>
    %2405 = stablehlo.multiply %2394, %2379 : tensor<3xf64>
    %2406 = stablehlo.add %2404, %2405 : tensor<3xf64>
    %2407 = stablehlo.reshape %2406 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2408 = stablehlo.multiply %2374, %2390 : tensor<3xf64>
    %2409 = stablehlo.multiply %2382, %2396 : tensor<3xf64>
    %2410 = stablehlo.add %2408, %2409 : tensor<3xf64>
    %2411 = stablehlo.multiply %2388, %2379 : tensor<3xf64>
    %2412 = stablehlo.subtract %2410, %2411 : tensor<3xf64>
    %2413 = stablehlo.multiply %2394, %2384 : tensor<3xf64>
    %2414 = stablehlo.add %2412, %2413 : tensor<3xf64>
    %2415 = stablehlo.reshape %2414 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2416 = stablehlo.multiply %2374, %2384 : tensor<3xf64>
    %2417 = stablehlo.multiply %2382, %2379 : tensor<3xf64>
    %2418 = stablehlo.subtract %2416, %2417 : tensor<3xf64>
    %2419 = stablehlo.multiply %2388, %2396 : tensor<3xf64>
    %2420 = stablehlo.subtract %2418, %2419 : tensor<3xf64>
    %2421 = stablehlo.multiply %2394, %2390 : tensor<3xf64>
    %2422 = stablehlo.subtract %2420, %2421 : tensor<3xf64>
    %2423 = stablehlo.reshape %2422 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2424 = stablehlo.concatenate %2399, %2407, %2415, %2423, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2425 = stablehlo.slice %2424 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2426 = stablehlo.reshape %2425 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2427 = stablehlo.slice %1953 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2428 = stablehlo.reshape %2427 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2429 = stablehlo.negate %2428 : tensor<3xf64>
    %2430 = stablehlo.reshape %2429 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2431 = stablehlo.slice %1953 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2432 = stablehlo.reshape %2431 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2433 = stablehlo.negate %2432 : tensor<3xf64>
    %2434 = stablehlo.reshape %2433 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2435 = stablehlo.slice %1953 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2436 = stablehlo.reshape %2435 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2437 = stablehlo.negate %2436 : tensor<3xf64>
    %2438 = stablehlo.reshape %2437 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2439 = stablehlo.slice %1953 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2440 = stablehlo.reshape %2439 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2441 = stablehlo.reshape %2440 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2442 = stablehlo.concatenate %2430, %2434, %2438, %2441, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2443 = stablehlo.dot_general %1953, %1953, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %2444 = stablehlo.broadcast_in_dim %2443, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %2445 = stablehlo.divide %2442, %2444 : tensor<3x4xf64>
    %2446 = stablehlo.slice %2445 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2447 = stablehlo.reshape %2446 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2448 = stablehlo.multiply %2426, %2447 : tensor<3xf64>
    %2449 = stablehlo.slice %2424 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2450 = stablehlo.reshape %2449 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2451 = stablehlo.slice %2445 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2452 = stablehlo.reshape %2451 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2453 = stablehlo.multiply %2450, %2452 : tensor<3xf64>
    %2454 = stablehlo.add %2448, %2453 : tensor<3xf64>
    %2455 = stablehlo.slice %2424 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2456 = stablehlo.reshape %2455 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2457 = stablehlo.slice %2445 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2458 = stablehlo.reshape %2457 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2459 = stablehlo.multiply %2456, %2458 : tensor<3xf64>
    %2460 = stablehlo.add %2454, %2459 : tensor<3xf64>
    %2461 = stablehlo.slice %2424 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2462 = stablehlo.reshape %2461 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2463 = stablehlo.slice %2445 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2464 = stablehlo.reshape %2463 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2465 = stablehlo.multiply %2462, %2464 : tensor<3xf64>
    %2466 = stablehlo.subtract %2460, %2465 : tensor<3xf64>
    %2467 = stablehlo.reshape %2466 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2468 = stablehlo.multiply %2426, %2464 : tensor<3xf64>
    %2469 = stablehlo.multiply %2450, %2458 : tensor<3xf64>
    %2470 = stablehlo.subtract %2468, %2469 : tensor<3xf64>
    %2471 = stablehlo.multiply %2456, %2452 : tensor<3xf64>
    %2472 = stablehlo.add %2470, %2471 : tensor<3xf64>
    %2473 = stablehlo.multiply %2462, %2447 : tensor<3xf64>
    %2474 = stablehlo.add %2472, %2473 : tensor<3xf64>
    %2475 = stablehlo.reshape %2474 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2476 = stablehlo.multiply %2426, %2458 : tensor<3xf64>
    %2477 = stablehlo.multiply %2450, %2464 : tensor<3xf64>
    %2478 = stablehlo.add %2476, %2477 : tensor<3xf64>
    %2479 = stablehlo.multiply %2456, %2447 : tensor<3xf64>
    %2480 = stablehlo.subtract %2478, %2479 : tensor<3xf64>
    %2481 = stablehlo.multiply %2462, %2452 : tensor<3xf64>
    %2482 = stablehlo.add %2480, %2481 : tensor<3xf64>
    %2483 = stablehlo.reshape %2482 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2484 = stablehlo.multiply %2426, %2452 : tensor<3xf64>
    %2485 = stablehlo.multiply %2450, %2447 : tensor<3xf64>
    %2486 = stablehlo.subtract %2484, %2485 : tensor<3xf64>
    %2487 = stablehlo.multiply %2456, %2464 : tensor<3xf64>
    %2488 = stablehlo.subtract %2486, %2487 : tensor<3xf64>
    %2489 = stablehlo.multiply %2462, %2458 : tensor<3xf64>
    %2490 = stablehlo.subtract %2488, %2489 : tensor<3xf64>
    %2491 = stablehlo.reshape %2490 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2492 = stablehlo.concatenate %2467, %2475, %2483, %2491, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2493 = stablehlo.slice %2492 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2494 = stablehlo.reshape %2493 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2495 = stablehlo.reshape %2494 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2496 = stablehlo.slice %2492 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2497 = stablehlo.reshape %2496 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2498 = stablehlo.reshape %2497 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2499 = stablehlo.slice %2492 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2500 = stablehlo.reshape %2499 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2501 = stablehlo.reshape %2500 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2502 = stablehlo.concatenate %2495, %2498, %2501, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x3xf64>
    %2503 = stablehlo.concatenate %2372, %2502, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %cst_31 = stablehlo.constant dense<0.16666666666666666> : tensor<f64>
    %2504 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2505 = stablehlo.reshape %arg5 : (tensor<f64>) -> tensor<f64>
    %2506 = stablehlo.broadcast_in_dim %2505, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2507 = stablehlo.multiply %2504, %2506 : tensor<3xf64>
    %2508 = stablehlo.broadcast_in_dim %2507, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %cst_32 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2509 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %2510 = stablehlo.multiply %2509, %1251 : tensor<3x6xf64>
    %2511 = stablehlo.add %625, %2510 : tensor<3x6xf64>
    %cst_33 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2512 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %2513 = stablehlo.multiply %2512, %1877 : tensor<3x6xf64>
    %2514 = stablehlo.add %2511, %2513 : tensor<3x6xf64>
    %2515 = stablehlo.add %2514, %2503 : tensor<3x6xf64>
    %2516 = stablehlo.multiply %2508, %2515 : tensor<3x6xf64>
    %2517 = stablehlo.add %arg6, %2516 : tensor<3x6xf64>
    %2518 = stablehlo.slice %arg2 [0:3, 0:4] : (tensor<3x7xf64>) -> tensor<3x4xf64>
    %2519 = stablehlo.broadcast_in_dim %2507, dims = [0] : (tensor<3xf64>) -> tensor<3x6xf64>
    %cst_34 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2520 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %2521 = stablehlo.multiply %2520, %698 : tensor<3x6xf64>
    %2522 = stablehlo.add %72, %2521 : tensor<3x6xf64>
    %cst_35 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2523 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f64>) -> tensor<3x6xf64>
    %2524 = stablehlo.multiply %2523, %1324 : tensor<3x6xf64>
    %2525 = stablehlo.add %2522, %2524 : tensor<3x6xf64>
    %2526 = stablehlo.add %2525, %1950 : tensor<3x6xf64>
    %2527 = stablehlo.multiply %2519, %2526 : tensor<3x6xf64>
    %2528 = stablehlo.slice %2527 [0:3, 0:3] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %cst_36 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2529 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f64>) -> tensor<3x3xf64>
    %2530 = stablehlo.divide %2528, %2529 : tensor<3x3xf64>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2531 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f64>) -> tensor<3x1xf64>
    %2532 = stablehlo.concatenate %2530, %2531, dim = 1 : (tensor<3x3xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2533 = stablehlo.slice %2532 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2534 = stablehlo.reshape %2533 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2535 = stablehlo.slice %2518 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2536 = stablehlo.reshape %2535 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2537 = stablehlo.multiply %2534, %2536 : tensor<3xf64>
    %2538 = stablehlo.slice %2532 [0:3, 0:1] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2539 = stablehlo.reshape %2538 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2540 = stablehlo.slice %2518 [0:3, 3:4] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2541 = stablehlo.reshape %2540 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2542 = stablehlo.multiply %2539, %2541 : tensor<3xf64>
    %2543 = stablehlo.add %2537, %2542 : tensor<3xf64>
    %2544 = stablehlo.slice %2532 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2545 = stablehlo.reshape %2544 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2546 = stablehlo.slice %2518 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2547 = stablehlo.reshape %2546 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2548 = stablehlo.multiply %2545, %2547 : tensor<3xf64>
    %2549 = stablehlo.add %2543, %2548 : tensor<3xf64>
    %2550 = stablehlo.slice %2532 [0:3, 2:3] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2551 = stablehlo.reshape %2550 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2552 = stablehlo.slice %2518 [0:3, 1:2] : (tensor<3x4xf64>) -> tensor<3x1xf64>
    %2553 = stablehlo.reshape %2552 : (tensor<3x1xf64>) -> tensor<3xf64>
    %2554 = stablehlo.multiply %2551, %2553 : tensor<3xf64>
    %2555 = stablehlo.subtract %2549, %2554 : tensor<3xf64>
    %2556 = stablehlo.reshape %2555 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2557 = stablehlo.multiply %2534, %2553 : tensor<3xf64>
    %2558 = stablehlo.multiply %2539, %2547 : tensor<3xf64>
    %2559 = stablehlo.subtract %2557, %2558 : tensor<3xf64>
    %2560 = stablehlo.multiply %2545, %2541 : tensor<3xf64>
    %2561 = stablehlo.add %2559, %2560 : tensor<3xf64>
    %2562 = stablehlo.multiply %2551, %2536 : tensor<3xf64>
    %2563 = stablehlo.add %2561, %2562 : tensor<3xf64>
    %2564 = stablehlo.reshape %2563 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2565 = stablehlo.multiply %2534, %2547 : tensor<3xf64>
    %2566 = stablehlo.multiply %2539, %2553 : tensor<3xf64>
    %2567 = stablehlo.add %2565, %2566 : tensor<3xf64>
    %2568 = stablehlo.multiply %2545, %2536 : tensor<3xf64>
    %2569 = stablehlo.subtract %2567, %2568 : tensor<3xf64>
    %2570 = stablehlo.multiply %2551, %2541 : tensor<3xf64>
    %2571 = stablehlo.add %2569, %2570 : tensor<3xf64>
    %2572 = stablehlo.reshape %2571 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2573 = stablehlo.multiply %2534, %2541 : tensor<3xf64>
    %2574 = stablehlo.multiply %2539, %2536 : tensor<3xf64>
    %2575 = stablehlo.subtract %2573, %2574 : tensor<3xf64>
    %2576 = stablehlo.multiply %2545, %2553 : tensor<3xf64>
    %2577 = stablehlo.subtract %2575, %2576 : tensor<3xf64>
    %2578 = stablehlo.multiply %2551, %2547 : tensor<3xf64>
    %2579 = stablehlo.subtract %2577, %2578 : tensor<3xf64>
    %2580 = stablehlo.reshape %2579 : (tensor<3xf64>) -> tensor<3x1xf64>
    %2581 = stablehlo.concatenate %2556, %2564, %2572, %2580, dim = 1 : (tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>, tensor<3x1xf64>) -> tensor<3x4xf64>
    %2582 = stablehlo.add %2518, %2581 : tensor<3x4xf64>
    %2583 = stablehlo.dot_general %2582, %2582, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3xf64>
    %2584 = stablehlo.sqrt %2583 : tensor<3xf64>
    %2585 = stablehlo.broadcast_in_dim %2584, dims = [0] : (tensor<3xf64>) -> tensor<3x4xf64>
    %2586 = stablehlo.divide %2582, %2585 : tensor<3x4xf64>
    %2587 = stablehlo.slice %arg2 [0:3, 4:7] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %2588 = stablehlo.slice %2527 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %2589 = stablehlo.add %2587, %2588 : tensor<3x3xf64>
    %2590 = stablehlo.concatenate %2586, %2589, dim = 1 : (tensor<3x4xf64>, tensor<3x3xf64>) -> tensor<3x7xf64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %2591 = stablehlo.add %arg0, %c : tensor<i64>
    return %2503, %arg5, %2591, %2517, %2590, %arg3, %1952 : tensor<3x6xf64>, tensor<f64>, tensor<i64>, tensor<3x6xf64>, tensor<3x7xf64>, tensor<3x7xf64>, tensor<3x6xf64>
  }
  func.func private @inner(%arg0: tensor<3x7xf64>, %arg1: tensor<3x7xf64>, %arg2: tensor<3x6xf64>) -> tensor<3x6xf64> {
    %c = stablehlo.constant dense<0> : tensor<1xui32>
    %c_0 = stablehlo.constant dense<1> : tensor<1xui32>
    %c_1 = stablehlo.constant dense<2> : tensor<1xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<1xui32>
    %c_3 = stablehlo.constant dense<1> : tensor<1xui32>
    %c_4 = stablehlo.constant dense<2> : tensor<1xui32>
    %c_5 = stablehlo.constant dense<[1, 2]> : tensor<2xui32>
    %c_6 = stablehlo.constant dense<[0, 2]> : tensor<2xui32>
    %c_7 = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
    %c_8 = stablehlo.constant dense<[1, 2]> : tensor<2xui32>
    %c_9 = stablehlo.constant dense<[0, 2]> : tensor<2xui32>
    %c_10 = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    %2 = stablehlo.reshape %1 : (tensor<1x7xf64>) -> tensor<7xf64>
    %3 = stablehlo.reshape %2 : (tensor<7xf64>) -> tensor<1x7xf64>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %5 = "stablehlo.gather"(%arg0, %4) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    %6 = stablehlo.reshape %5 : (tensor<1x7xf64>) -> tensor<7xf64>
    %7 = stablehlo.reshape %6 : (tensor<7xf64>) -> tensor<1x7xf64>
    %8 = stablehlo.concatenate %3, %7, dim = 0 : (tensor<1x7xf64>, tensor<1x7xf64>) -> tensor<2x7xf64>
    %9 = stablehlo.broadcast_in_dim %c_1, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %10 = "stablehlo.gather"(%arg0, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    %11 = stablehlo.reshape %10 : (tensor<1x7xf64>) -> tensor<7xf64>
    %12 = stablehlo.reshape %11 : (tensor<7xf64>) -> tensor<1x7xf64>
    %13 = stablehlo.concatenate %8, %12, dim = 0 : (tensor<2x7xf64>, tensor<1x7xf64>) -> tensor<3x7xf64>
    %14 = stablehlo.broadcast_in_dim %c_2, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %15 = "stablehlo.gather"(%arg1, %14) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    %16 = stablehlo.reshape %15 : (tensor<1x7xf64>) -> tensor<7xf64>
    %17 = stablehlo.reshape %16 : (tensor<7xf64>) -> tensor<1x7xf64>
    %18 = stablehlo.broadcast_in_dim %c_3, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %19 = "stablehlo.gather"(%arg1, %18) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    %20 = stablehlo.reshape %19 : (tensor<1x7xf64>) -> tensor<7xf64>
    %21 = stablehlo.reshape %20 : (tensor<7xf64>) -> tensor<1x7xf64>
    %22 = stablehlo.concatenate %17, %21, dim = 0 : (tensor<1x7xf64>, tensor<1x7xf64>) -> tensor<2x7xf64>
    %23 = stablehlo.broadcast_in_dim %c_4, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %24 = "stablehlo.gather"(%arg1, %23) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    %25 = stablehlo.reshape %24 : (tensor<1x7xf64>) -> tensor<7xf64>
    %26 = stablehlo.reshape %25 : (tensor<7xf64>) -> tensor<1x7xf64>
    %27 = stablehlo.concatenate %22, %26, dim = 0 : (tensor<2x7xf64>, tensor<1x7xf64>) -> tensor<3x7xf64>
    %28 = stablehlo.broadcast_in_dim %c_5, dims = [0] : (tensor<2xui32>) -> tensor<2x1xui32>
    %29 = "stablehlo.gather"(%arg0, %28) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<2x1xui32>) -> tensor<2x7xf64>
    %30 = stablehlo.reshape %29 : (tensor<2x7xf64>) -> tensor<1x2x7xf64>
    %31 = stablehlo.broadcast_in_dim %c_6, dims = [0] : (tensor<2xui32>) -> tensor<2x1xui32>
    %32 = "stablehlo.gather"(%arg0, %31) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<2x1xui32>) -> tensor<2x7xf64>
    %33 = stablehlo.reshape %32 : (tensor<2x7xf64>) -> tensor<1x2x7xf64>
    %34 = stablehlo.concatenate %30, %33, dim = 0 : (tensor<1x2x7xf64>, tensor<1x2x7xf64>) -> tensor<2x2x7xf64>
    %35 = stablehlo.broadcast_in_dim %c_7, dims = [0] : (tensor<2xui32>) -> tensor<2x1xui32>
    %36 = "stablehlo.gather"(%arg0, %35) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<2x1xui32>) -> tensor<2x7xf64>
    %37 = stablehlo.reshape %36 : (tensor<2x7xf64>) -> tensor<1x2x7xf64>
    %38 = stablehlo.concatenate %34, %37, dim = 0 : (tensor<2x2x7xf64>, tensor<1x2x7xf64>) -> tensor<3x2x7xf64>
    %39 = stablehlo.broadcast_in_dim %c_8, dims = [0] : (tensor<2xui32>) -> tensor<2x1xui32>
    %40 = "stablehlo.gather"(%arg1, %39) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<2x1xui32>) -> tensor<2x7xf64>
    %41 = stablehlo.reshape %40 : (tensor<2x7xf64>) -> tensor<1x2x7xf64>
    %42 = stablehlo.broadcast_in_dim %c_9, dims = [0] : (tensor<2xui32>) -> tensor<2x1xui32>
    %43 = "stablehlo.gather"(%arg1, %42) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<2x1xui32>) -> tensor<2x7xf64>
    %44 = stablehlo.reshape %43 : (tensor<2x7xf64>) -> tensor<1x2x7xf64>
    %45 = stablehlo.concatenate %41, %44, dim = 0 : (tensor<1x2x7xf64>, tensor<1x2x7xf64>) -> tensor<2x2x7xf64>
    %46 = stablehlo.broadcast_in_dim %c_10, dims = [0] : (tensor<2xui32>) -> tensor<2x1xui32>
    %47 = "stablehlo.gather"(%arg1, %46) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<2x1xui32>) -> tensor<2x7xf64>
    %48 = stablehlo.reshape %47 : (tensor<2x7xf64>) -> tensor<1x2x7xf64>
    %49 = stablehlo.concatenate %45, %48, dim = 0 : (tensor<2x2x7xf64>, tensor<1x2x7xf64>) -> tensor<3x2x7xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %50 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %51 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %52 = stablehlo.concatenate %50, %51, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %53 = stablehlo.broadcast_in_dim %52, dims = [1] : (tensor<6xf64>) -> tensor<3x6xf64>
    %54 = stablehlo.transpose %38, dims = [1, 0, 2] : (tensor<3x2x7xf64>) -> tensor<2x3x7xf64>
    %55 = stablehlo.transpose %49, dims = [1, 0, 2] : (tensor<3x2x7xf64>) -> tensor<2x3x7xf64>
    %c_12 = stablehlo.constant dense<0> : tensor<i64>
    %56 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %c_13 = stablehlo.constant dense<0> : tensor<i64>
    %57:7 = stablehlo.while(%iterArg = %54, %iterArg_14 = %55, %iterArg_15 = %13, %iterArg_16 = %27, %iterArg_17 = %c_13, %iterArg_18 = %53, %iterArg_19 = %56) : tensor<2x3x7xf64>, tensor<2x3x7xf64>, tensor<3x7xf64>, tensor<3x7xf64>, tensor<i64>, tensor<3x6xf64>, tensor<2xi64>
    cond {
      %c_20 = stablehlo.constant dense<2> : tensor<i64>
      %58 = stablehlo.compare  LT, %iterArg_17, %c_20,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %58 : tensor<i1>
    } do {
      %c_20 = stablehlo.constant dense<0> : tensor<i64>
      %c_21 = stablehlo.constant dense<0> : tensor<i64>
      %58 = stablehlo.dynamic_slice %iterArg, %iterArg_17, %c_20, %c_21, sizes = [1, 3, 7] : (tensor<2x3x7xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x7xf64>
      %59 = stablehlo.reshape %58 : (tensor<1x3x7xf64>) -> tensor<3x7xf64>
      %c_22 = stablehlo.constant dense<0> : tensor<i64>
      %c_23 = stablehlo.constant dense<0> : tensor<i64>
      %60 = stablehlo.dynamic_slice %iterArg_14, %iterArg_17, %c_22, %c_23, sizes = [1, 3, 7] : (tensor<2x3x7xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x3x7xf64>
      %61 = stablehlo.reshape %60 : (tensor<1x3x7xf64>) -> tensor<3x7xf64>
      %62:2 = func.call @closed_call(%iterArg_15, %iterArg_16, %iterArg_18, %59, %61) : (tensor<3x7xf64>, tensor<3x7xf64>, tensor<3x6xf64>, tensor<3x7xf64>, tensor<3x7xf64>) -> (tensor<3x6xf64>, tensor<i64>)
      %63 = stablehlo.broadcast_in_dim %62#1, dims = [] : (tensor<i64>) -> tensor<1xi64>
      %64 = stablehlo.dynamic_update_slice %iterArg_19, %63, %iterArg_17 : (tensor<2xi64>, tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
      %c_24 = stablehlo.constant dense<1> : tensor<i64>
      %65 = stablehlo.add %iterArg_17, %c_24 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_14, %iterArg_15, %iterArg_16, %65, %62#0, %64 : tensor<2x3x7xf64>, tensor<2x3x7xf64>, tensor<3x7xf64>, tensor<3x7xf64>, tensor<i64>, tensor<3x6xf64>, tensor<2xi64>
    }
    return %57#5 : tensor<3x6xf64>
  }
  func.func private @closed_call(%arg0: tensor<3x7xf64>, %arg1: tensor<3x7xf64>, %arg2: tensor<3x6xf64>, %arg3: tensor<3x7xf64>, %arg4: tensor<3x7xf64>) -> (tensor<3x6xf64>, tensor<i64>) {
    %0 = stablehlo.slice %arg0 [0:3, 4:7] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %1 = stablehlo.slice %arg3 [0:3, 4:7] : (tensor<3x7xf64>) -> tensor<3x3xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<3x3xf64>
    %3 = stablehlo.slice %arg1 [0:3, 6:7] : (tensor<3x7xf64>) -> tensor<3x1xf64>
    %4 = stablehlo.reshape %3 : (tensor<3x1xf64>) -> tensor<3xf64>
    %5 = stablehlo.slice %arg4 [0:3, 6:7] : (tensor<3x7xf64>) -> tensor<3x1xf64>
    %6 = stablehlo.reshape %5 : (tensor<3x1xf64>) -> tensor<3xf64>
    %7 = call @norm(%2) : (tensor<3x3xf64>) -> tensor<3xf64>
    %cst = stablehlo.constant dense<6.6742999999999994E-11> : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %9 = stablehlo.multiply %8, %6 : tensor<3xf64>
    %10 = stablehlo.multiply %9, %4 : tensor<3xf64>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x3xf64>
    %13 = stablehlo.multiply %12, %2 : tensor<3x3xf64>
    %14 = stablehlo.multiply %7, %7 : tensor<3xf64>
    %15 = stablehlo.multiply %14, %7 : tensor<3xf64>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<3xf64>) -> tensor<3x1xf64>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x3xf64>
    %18 = stablehlo.divide %13, %17 : tensor<3x3xf64>
    %19 = stablehlo.slice %arg2 [0:3, 3:6] : (tensor<3x6xf64>) -> tensor<3x3xf64>
    %20 = stablehlo.subtract %19, %18 : tensor<3x3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<3xf64>) -> tensor<3x3xf64>
    %23 = stablehlo.concatenate %22, %20, dim = 1 : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x6xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    return %23, %c : tensor<3x6xf64>, tensor<i64>
  }
  func.func private @norm(%arg0: tensor<3x3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<3x3xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<3x3xf64>, tensor<f64>) -> tensor<3xf64>
    %2 = stablehlo.sqrt %1 : tensor<3xf64>
    return %2 : tensor<3xf64>
  }
}
