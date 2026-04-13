module @module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<3xf64>, %arg3: tensor<7xf64>, %arg4: tensor<6xf64>, %arg5: tensor<6xf64>, %arg6: tensor<7xf64>, %arg7: tensor<6xf64>, %arg8: tensor<f64>) -> (tensor<6xf64> {jax.result_info = "result[0]"}, tensor<f64> {jax.result_info = "result[1]"}, tensor<i64> {jax.result_info = "result[2]"}, tensor<3xf64> {jax.result_info = "result[3]"}, tensor<i64> {jax.result_info = "result[4]"}, tensor<6xf64> {jax.result_info = "result[5]"}, tensor<7xf64> {jax.result_info = "result[6]"}, tensor<7xf64> {jax.result_info = "result[7]"}, tensor<6xf64> {jax.result_info = "result[8]"}) {
    %0 = call @inner(%arg1, %arg2) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %1 = call @inner_20(%arg3, %arg4) : (tensor<7xf64>, tensor<6xf64>) -> tensor<6xf64>
    %2 = stablehlo.slice %arg3 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %3 = stablehlo.reshape %arg8 : (tensor<f64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %4 = stablehlo.multiply %cst, %3 : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %6 = stablehlo.multiply %5, %1 : tensor<6xf64>
    %7 = stablehlo.slice %6 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %9 = stablehlo.divide %7, %8 : tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %12 = stablehlo.slice %11 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %13 = stablehlo.reshape %12 : (tensor<1xf64>) -> tensor<f64>
    %14 = stablehlo.slice %2 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.multiply %13, %15 : tensor<f64>
    %17 = stablehlo.slice %11 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %18 = stablehlo.reshape %17 : (tensor<1xf64>) -> tensor<f64>
    %19 = stablehlo.slice %2 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %20 = stablehlo.reshape %19 : (tensor<1xf64>) -> tensor<f64>
    %21 = stablehlo.multiply %18, %20 : tensor<f64>
    %22 = stablehlo.add %16, %21 : tensor<f64>
    %23 = stablehlo.slice %11 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %24 = stablehlo.reshape %23 : (tensor<1xf64>) -> tensor<f64>
    %25 = stablehlo.slice %2 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %27 = stablehlo.multiply %24, %26 : tensor<f64>
    %28 = stablehlo.add %22, %27 : tensor<f64>
    %29 = stablehlo.slice %11 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.slice %2 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %32 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
    %33 = stablehlo.multiply %30, %32 : tensor<f64>
    %34 = stablehlo.subtract %28, %33 : tensor<f64>
    %35 = stablehlo.reshape %34 : (tensor<f64>) -> tensor<1xf64>
    %36 = stablehlo.multiply %13, %32 : tensor<f64>
    %37 = stablehlo.multiply %18, %26 : tensor<f64>
    %38 = stablehlo.subtract %36, %37 : tensor<f64>
    %39 = stablehlo.multiply %24, %20 : tensor<f64>
    %40 = stablehlo.add %38, %39 : tensor<f64>
    %41 = stablehlo.multiply %30, %15 : tensor<f64>
    %42 = stablehlo.add %40, %41 : tensor<f64>
    %43 = stablehlo.reshape %42 : (tensor<f64>) -> tensor<1xf64>
    %44 = stablehlo.multiply %13, %26 : tensor<f64>
    %45 = stablehlo.multiply %18, %32 : tensor<f64>
    %46 = stablehlo.add %44, %45 : tensor<f64>
    %47 = stablehlo.multiply %24, %15 : tensor<f64>
    %48 = stablehlo.subtract %46, %47 : tensor<f64>
    %49 = stablehlo.multiply %30, %20 : tensor<f64>
    %50 = stablehlo.add %48, %49 : tensor<f64>
    %51 = stablehlo.reshape %50 : (tensor<f64>) -> tensor<1xf64>
    %52 = stablehlo.multiply %13, %20 : tensor<f64>
    %53 = stablehlo.multiply %18, %15 : tensor<f64>
    %54 = stablehlo.subtract %52, %53 : tensor<f64>
    %55 = stablehlo.multiply %24, %32 : tensor<f64>
    %56 = stablehlo.subtract %54, %55 : tensor<f64>
    %57 = stablehlo.multiply %30, %26 : tensor<f64>
    %58 = stablehlo.subtract %56, %57 : tensor<f64>
    %59 = stablehlo.reshape %58 : (tensor<f64>) -> tensor<1xf64>
    %60 = stablehlo.concatenate %35, %43, %51, %59, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %61 = stablehlo.add %2, %60 : tensor<4xf64>
    %62 = stablehlo.dot_general %61, %61, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %63 = stablehlo.sqrt %62 : tensor<f64>
    %64 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %65 = stablehlo.divide %61, %64 : tensor<4xf64>
    %66 = stablehlo.slice %arg3 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %67 = stablehlo.slice %6 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %68 = stablehlo.add %66, %67 : tensor<3xf64>
    %69 = stablehlo.concatenate %65, %68, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %70 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %71 = stablehlo.multiply %70, %arg7 : tensor<6xf64>
    %72 = stablehlo.add %1, %71 : tensor<6xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %73 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %74 = call @inner_55(%73, %arg6) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %75 = call @inner_57(%0, %72, %74) : (tensor<3xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %76 = stablehlo.slice %69 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %77 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %80 = stablehlo.reshape %79 : (tensor<1xf64>) -> tensor<f64>
    %81 = stablehlo.negate %80 : tensor<f64>
    %82 = stablehlo.reshape %81 : (tensor<f64>) -> tensor<1xf64>
    %83 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %84 = stablehlo.reshape %83 : (tensor<1xf64>) -> tensor<f64>
    %85 = stablehlo.negate %84 : tensor<f64>
    %86 = stablehlo.reshape %85 : (tensor<f64>) -> tensor<1xf64>
    %87 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %88 = stablehlo.reshape %87 : (tensor<1xf64>) -> tensor<f64>
    %89 = stablehlo.negate %88 : tensor<f64>
    %90 = stablehlo.reshape %89 : (tensor<f64>) -> tensor<1xf64>
    %91 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %92 = stablehlo.reshape %91 : (tensor<1xf64>) -> tensor<f64>
    %93 = stablehlo.reshape %92 : (tensor<f64>) -> tensor<1xf64>
    %94 = stablehlo.concatenate %82, %86, %90, %93, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %95 = stablehlo.dot_general %76, %76, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %96 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %97 = stablehlo.divide %94, %96 : tensor<4xf64>
    %98 = stablehlo.slice %97 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.slice %75 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %101 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %102 = stablehlo.concatenate %100, %101, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %103 = stablehlo.slice %102 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
    %105 = stablehlo.multiply %99, %104 : tensor<f64>
    %106 = stablehlo.slice %97 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %107 = stablehlo.reshape %106 : (tensor<1xf64>) -> tensor<f64>
    %108 = stablehlo.slice %102 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.multiply %107, %109 : tensor<f64>
    %111 = stablehlo.add %105, %110 : tensor<f64>
    %112 = stablehlo.slice %97 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %113 = stablehlo.reshape %112 : (tensor<1xf64>) -> tensor<f64>
    %114 = stablehlo.slice %102 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %115 = stablehlo.reshape %114 : (tensor<1xf64>) -> tensor<f64>
    %116 = stablehlo.multiply %113, %115 : tensor<f64>
    %117 = stablehlo.add %111, %116 : tensor<f64>
    %118 = stablehlo.slice %97 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %119 = stablehlo.reshape %118 : (tensor<1xf64>) -> tensor<f64>
    %120 = stablehlo.slice %102 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %121 = stablehlo.reshape %120 : (tensor<1xf64>) -> tensor<f64>
    %122 = stablehlo.multiply %119, %121 : tensor<f64>
    %123 = stablehlo.subtract %117, %122 : tensor<f64>
    %124 = stablehlo.reshape %123 : (tensor<f64>) -> tensor<1xf64>
    %125 = stablehlo.multiply %99, %121 : tensor<f64>
    %126 = stablehlo.multiply %107, %115 : tensor<f64>
    %127 = stablehlo.subtract %125, %126 : tensor<f64>
    %128 = stablehlo.multiply %113, %109 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.multiply %119, %104 : tensor<f64>
    %131 = stablehlo.add %129, %130 : tensor<f64>
    %132 = stablehlo.reshape %131 : (tensor<f64>) -> tensor<1xf64>
    %133 = stablehlo.multiply %99, %115 : tensor<f64>
    %134 = stablehlo.multiply %107, %121 : tensor<f64>
    %135 = stablehlo.add %133, %134 : tensor<f64>
    %136 = stablehlo.multiply %113, %104 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.multiply %119, %109 : tensor<f64>
    %139 = stablehlo.add %137, %138 : tensor<f64>
    %140 = stablehlo.reshape %139 : (tensor<f64>) -> tensor<1xf64>
    %141 = stablehlo.multiply %99, %109 : tensor<f64>
    %142 = stablehlo.multiply %107, %104 : tensor<f64>
    %143 = stablehlo.subtract %141, %142 : tensor<f64>
    %144 = stablehlo.multiply %113, %121 : tensor<f64>
    %145 = stablehlo.subtract %143, %144 : tensor<f64>
    %146 = stablehlo.multiply %119, %115 : tensor<f64>
    %147 = stablehlo.subtract %145, %146 : tensor<f64>
    %148 = stablehlo.reshape %147 : (tensor<f64>) -> tensor<1xf64>
    %149 = stablehlo.concatenate %124, %132, %140, %148, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %150 = stablehlo.slice %149 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
    %152 = stablehlo.slice %97 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %153 = stablehlo.reshape %152 : (tensor<1xf64>) -> tensor<f64>
    %154 = stablehlo.negate %153 : tensor<f64>
    %155 = stablehlo.reshape %154 : (tensor<f64>) -> tensor<1xf64>
    %156 = stablehlo.slice %97 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %157 = stablehlo.reshape %156 : (tensor<1xf64>) -> tensor<f64>
    %158 = stablehlo.negate %157 : tensor<f64>
    %159 = stablehlo.reshape %158 : (tensor<f64>) -> tensor<1xf64>
    %160 = stablehlo.slice %97 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %161 = stablehlo.reshape %160 : (tensor<1xf64>) -> tensor<f64>
    %162 = stablehlo.negate %161 : tensor<f64>
    %163 = stablehlo.reshape %162 : (tensor<f64>) -> tensor<1xf64>
    %164 = stablehlo.slice %97 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %165 = stablehlo.reshape %164 : (tensor<1xf64>) -> tensor<f64>
    %166 = stablehlo.reshape %165 : (tensor<f64>) -> tensor<1xf64>
    %167 = stablehlo.concatenate %155, %159, %163, %166, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %168 = stablehlo.dot_general %97, %97, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %169 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %170 = stablehlo.divide %167, %169 : tensor<4xf64>
    %171 = stablehlo.slice %170 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %172 = stablehlo.reshape %171 : (tensor<1xf64>) -> tensor<f64>
    %173 = stablehlo.multiply %151, %172 : tensor<f64>
    %174 = stablehlo.slice %149 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %175 = stablehlo.reshape %174 : (tensor<1xf64>) -> tensor<f64>
    %176 = stablehlo.slice %170 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %177 = stablehlo.reshape %176 : (tensor<1xf64>) -> tensor<f64>
    %178 = stablehlo.multiply %175, %177 : tensor<f64>
    %179 = stablehlo.add %173, %178 : tensor<f64>
    %180 = stablehlo.slice %149 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %181 = stablehlo.reshape %180 : (tensor<1xf64>) -> tensor<f64>
    %182 = stablehlo.slice %170 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %183 = stablehlo.reshape %182 : (tensor<1xf64>) -> tensor<f64>
    %184 = stablehlo.multiply %181, %183 : tensor<f64>
    %185 = stablehlo.add %179, %184 : tensor<f64>
    %186 = stablehlo.slice %149 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %187 = stablehlo.reshape %186 : (tensor<1xf64>) -> tensor<f64>
    %188 = stablehlo.slice %170 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %189 = stablehlo.reshape %188 : (tensor<1xf64>) -> tensor<f64>
    %190 = stablehlo.multiply %187, %189 : tensor<f64>
    %191 = stablehlo.subtract %185, %190 : tensor<f64>
    %192 = stablehlo.reshape %191 : (tensor<f64>) -> tensor<1xf64>
    %193 = stablehlo.multiply %151, %189 : tensor<f64>
    %194 = stablehlo.multiply %175, %183 : tensor<f64>
    %195 = stablehlo.subtract %193, %194 : tensor<f64>
    %196 = stablehlo.multiply %181, %177 : tensor<f64>
    %197 = stablehlo.add %195, %196 : tensor<f64>
    %198 = stablehlo.multiply %187, %172 : tensor<f64>
    %199 = stablehlo.add %197, %198 : tensor<f64>
    %200 = stablehlo.reshape %199 : (tensor<f64>) -> tensor<1xf64>
    %201 = stablehlo.multiply %151, %183 : tensor<f64>
    %202 = stablehlo.multiply %175, %189 : tensor<f64>
    %203 = stablehlo.add %201, %202 : tensor<f64>
    %204 = stablehlo.multiply %181, %172 : tensor<f64>
    %205 = stablehlo.subtract %203, %204 : tensor<f64>
    %206 = stablehlo.multiply %187, %177 : tensor<f64>
    %207 = stablehlo.add %205, %206 : tensor<f64>
    %208 = stablehlo.reshape %207 : (tensor<f64>) -> tensor<1xf64>
    %209 = stablehlo.multiply %151, %177 : tensor<f64>
    %210 = stablehlo.multiply %175, %172 : tensor<f64>
    %211 = stablehlo.subtract %209, %210 : tensor<f64>
    %212 = stablehlo.multiply %181, %189 : tensor<f64>
    %213 = stablehlo.subtract %211, %212 : tensor<f64>
    %214 = stablehlo.multiply %187, %183 : tensor<f64>
    %215 = stablehlo.subtract %213, %214 : tensor<f64>
    %216 = stablehlo.reshape %215 : (tensor<f64>) -> tensor<1xf64>
    %217 = stablehlo.concatenate %192, %200, %208, %216, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %218 = stablehlo.slice %217 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %219 = stablehlo.reshape %218 : (tensor<1xf64>) -> tensor<f64>
    %220 = stablehlo.reshape %219 : (tensor<f64>) -> tensor<1xf64>
    %221 = stablehlo.slice %217 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %222 = stablehlo.reshape %221 : (tensor<1xf64>) -> tensor<f64>
    %223 = stablehlo.reshape %222 : (tensor<f64>) -> tensor<1xf64>
    %224 = stablehlo.slice %217 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %225 = stablehlo.reshape %224 : (tensor<1xf64>) -> tensor<f64>
    %226 = stablehlo.reshape %225 : (tensor<f64>) -> tensor<1xf64>
    %227 = stablehlo.concatenate %220, %223, %226, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %228 = stablehlo.slice %97 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %229 = stablehlo.reshape %228 : (tensor<1xf64>) -> tensor<f64>
    %230 = stablehlo.slice %75 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %231 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %232 = stablehlo.concatenate %230, %231, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %233 = stablehlo.slice %232 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %234 = stablehlo.reshape %233 : (tensor<1xf64>) -> tensor<f64>
    %235 = stablehlo.multiply %229, %234 : tensor<f64>
    %236 = stablehlo.slice %97 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %237 = stablehlo.reshape %236 : (tensor<1xf64>) -> tensor<f64>
    %238 = stablehlo.slice %232 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %239 = stablehlo.reshape %238 : (tensor<1xf64>) -> tensor<f64>
    %240 = stablehlo.multiply %237, %239 : tensor<f64>
    %241 = stablehlo.add %235, %240 : tensor<f64>
    %242 = stablehlo.slice %97 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %243 = stablehlo.reshape %242 : (tensor<1xf64>) -> tensor<f64>
    %244 = stablehlo.slice %232 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %245 = stablehlo.reshape %244 : (tensor<1xf64>) -> tensor<f64>
    %246 = stablehlo.multiply %243, %245 : tensor<f64>
    %247 = stablehlo.add %241, %246 : tensor<f64>
    %248 = stablehlo.slice %97 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %249 = stablehlo.reshape %248 : (tensor<1xf64>) -> tensor<f64>
    %250 = stablehlo.slice %232 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %251 = stablehlo.reshape %250 : (tensor<1xf64>) -> tensor<f64>
    %252 = stablehlo.multiply %249, %251 : tensor<f64>
    %253 = stablehlo.subtract %247, %252 : tensor<f64>
    %254 = stablehlo.reshape %253 : (tensor<f64>) -> tensor<1xf64>
    %255 = stablehlo.multiply %229, %251 : tensor<f64>
    %256 = stablehlo.multiply %237, %245 : tensor<f64>
    %257 = stablehlo.subtract %255, %256 : tensor<f64>
    %258 = stablehlo.multiply %243, %239 : tensor<f64>
    %259 = stablehlo.add %257, %258 : tensor<f64>
    %260 = stablehlo.multiply %249, %234 : tensor<f64>
    %261 = stablehlo.add %259, %260 : tensor<f64>
    %262 = stablehlo.reshape %261 : (tensor<f64>) -> tensor<1xf64>
    %263 = stablehlo.multiply %229, %245 : tensor<f64>
    %264 = stablehlo.multiply %237, %251 : tensor<f64>
    %265 = stablehlo.add %263, %264 : tensor<f64>
    %266 = stablehlo.multiply %243, %234 : tensor<f64>
    %267 = stablehlo.subtract %265, %266 : tensor<f64>
    %268 = stablehlo.multiply %249, %239 : tensor<f64>
    %269 = stablehlo.add %267, %268 : tensor<f64>
    %270 = stablehlo.reshape %269 : (tensor<f64>) -> tensor<1xf64>
    %271 = stablehlo.multiply %229, %239 : tensor<f64>
    %272 = stablehlo.multiply %237, %234 : tensor<f64>
    %273 = stablehlo.subtract %271, %272 : tensor<f64>
    %274 = stablehlo.multiply %243, %251 : tensor<f64>
    %275 = stablehlo.subtract %273, %274 : tensor<f64>
    %276 = stablehlo.multiply %249, %245 : tensor<f64>
    %277 = stablehlo.subtract %275, %276 : tensor<f64>
    %278 = stablehlo.reshape %277 : (tensor<f64>) -> tensor<1xf64>
    %279 = stablehlo.concatenate %254, %262, %270, %278, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %280 = stablehlo.slice %279 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %281 = stablehlo.reshape %280 : (tensor<1xf64>) -> tensor<f64>
    %282 = stablehlo.slice %97 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %283 = stablehlo.reshape %282 : (tensor<1xf64>) -> tensor<f64>
    %284 = stablehlo.negate %283 : tensor<f64>
    %285 = stablehlo.reshape %284 : (tensor<f64>) -> tensor<1xf64>
    %286 = stablehlo.slice %97 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %287 = stablehlo.reshape %286 : (tensor<1xf64>) -> tensor<f64>
    %288 = stablehlo.negate %287 : tensor<f64>
    %289 = stablehlo.reshape %288 : (tensor<f64>) -> tensor<1xf64>
    %290 = stablehlo.slice %97 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %291 = stablehlo.reshape %290 : (tensor<1xf64>) -> tensor<f64>
    %292 = stablehlo.negate %291 : tensor<f64>
    %293 = stablehlo.reshape %292 : (tensor<f64>) -> tensor<1xf64>
    %294 = stablehlo.slice %97 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %295 = stablehlo.reshape %294 : (tensor<1xf64>) -> tensor<f64>
    %296 = stablehlo.reshape %295 : (tensor<f64>) -> tensor<1xf64>
    %297 = stablehlo.concatenate %285, %289, %293, %296, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %298 = stablehlo.dot_general %97, %97, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %299 = stablehlo.broadcast_in_dim %298, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %300 = stablehlo.divide %297, %299 : tensor<4xf64>
    %301 = stablehlo.slice %300 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %302 = stablehlo.reshape %301 : (tensor<1xf64>) -> tensor<f64>
    %303 = stablehlo.multiply %281, %302 : tensor<f64>
    %304 = stablehlo.slice %279 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %305 = stablehlo.reshape %304 : (tensor<1xf64>) -> tensor<f64>
    %306 = stablehlo.slice %300 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %307 = stablehlo.reshape %306 : (tensor<1xf64>) -> tensor<f64>
    %308 = stablehlo.multiply %305, %307 : tensor<f64>
    %309 = stablehlo.add %303, %308 : tensor<f64>
    %310 = stablehlo.slice %279 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %311 = stablehlo.reshape %310 : (tensor<1xf64>) -> tensor<f64>
    %312 = stablehlo.slice %300 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %313 = stablehlo.reshape %312 : (tensor<1xf64>) -> tensor<f64>
    %314 = stablehlo.multiply %311, %313 : tensor<f64>
    %315 = stablehlo.add %309, %314 : tensor<f64>
    %316 = stablehlo.slice %279 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %317 = stablehlo.reshape %316 : (tensor<1xf64>) -> tensor<f64>
    %318 = stablehlo.slice %300 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %319 = stablehlo.reshape %318 : (tensor<1xf64>) -> tensor<f64>
    %320 = stablehlo.multiply %317, %319 : tensor<f64>
    %321 = stablehlo.subtract %315, %320 : tensor<f64>
    %322 = stablehlo.reshape %321 : (tensor<f64>) -> tensor<1xf64>
    %323 = stablehlo.multiply %281, %319 : tensor<f64>
    %324 = stablehlo.multiply %305, %313 : tensor<f64>
    %325 = stablehlo.subtract %323, %324 : tensor<f64>
    %326 = stablehlo.multiply %311, %307 : tensor<f64>
    %327 = stablehlo.add %325, %326 : tensor<f64>
    %328 = stablehlo.multiply %317, %302 : tensor<f64>
    %329 = stablehlo.add %327, %328 : tensor<f64>
    %330 = stablehlo.reshape %329 : (tensor<f64>) -> tensor<1xf64>
    %331 = stablehlo.multiply %281, %313 : tensor<f64>
    %332 = stablehlo.multiply %305, %319 : tensor<f64>
    %333 = stablehlo.add %331, %332 : tensor<f64>
    %334 = stablehlo.multiply %311, %302 : tensor<f64>
    %335 = stablehlo.subtract %333, %334 : tensor<f64>
    %336 = stablehlo.multiply %317, %307 : tensor<f64>
    %337 = stablehlo.add %335, %336 : tensor<f64>
    %338 = stablehlo.reshape %337 : (tensor<f64>) -> tensor<1xf64>
    %339 = stablehlo.multiply %281, %307 : tensor<f64>
    %340 = stablehlo.multiply %305, %302 : tensor<f64>
    %341 = stablehlo.subtract %339, %340 : tensor<f64>
    %342 = stablehlo.multiply %311, %319 : tensor<f64>
    %343 = stablehlo.subtract %341, %342 : tensor<f64>
    %344 = stablehlo.multiply %317, %313 : tensor<f64>
    %345 = stablehlo.subtract %343, %344 : tensor<f64>
    %346 = stablehlo.reshape %345 : (tensor<f64>) -> tensor<1xf64>
    %347 = stablehlo.concatenate %322, %330, %338, %346, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %348 = stablehlo.slice %347 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %349 = stablehlo.reshape %348 : (tensor<1xf64>) -> tensor<f64>
    %350 = stablehlo.reshape %349 : (tensor<f64>) -> tensor<1xf64>
    %351 = stablehlo.slice %347 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %352 = stablehlo.reshape %351 : (tensor<1xf64>) -> tensor<f64>
    %353 = stablehlo.reshape %352 : (tensor<f64>) -> tensor<1xf64>
    %354 = stablehlo.slice %347 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %355 = stablehlo.reshape %354 : (tensor<1xf64>) -> tensor<f64>
    %356 = stablehlo.reshape %355 : (tensor<f64>) -> tensor<1xf64>
    %357 = stablehlo.concatenate %350, %353, %356, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %358 = stablehlo.concatenate %227, %357, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %359 = stablehlo.slice %358 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %360 = stablehlo.slice %arg6 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %361 = stablehlo.divide %359, %360 : tensor<3xf64>
    %362 = stablehlo.slice %358 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %363 = stablehlo.slice %arg6 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %364 = stablehlo.reshape %363 : (tensor<1xf64>) -> tensor<f64>
    %365 = stablehlo.broadcast_in_dim %364, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %366 = stablehlo.divide %362, %365 : tensor<3xf64>
    %367 = stablehlo.concatenate %361, %366, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %368 = stablehlo.slice %367 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %369 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %370 = stablehlo.concatenate %368, %369, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %371 = stablehlo.slice %370 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %372 = stablehlo.reshape %371 : (tensor<1xf64>) -> tensor<f64>
    %373 = stablehlo.multiply %78, %372 : tensor<f64>
    %374 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %375 = stablehlo.reshape %374 : (tensor<1xf64>) -> tensor<f64>
    %376 = stablehlo.slice %370 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %377 = stablehlo.reshape %376 : (tensor<1xf64>) -> tensor<f64>
    %378 = stablehlo.multiply %375, %377 : tensor<f64>
    %379 = stablehlo.add %373, %378 : tensor<f64>
    %380 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %381 = stablehlo.reshape %380 : (tensor<1xf64>) -> tensor<f64>
    %382 = stablehlo.slice %370 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %383 = stablehlo.reshape %382 : (tensor<1xf64>) -> tensor<f64>
    %384 = stablehlo.multiply %381, %383 : tensor<f64>
    %385 = stablehlo.add %379, %384 : tensor<f64>
    %386 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %387 = stablehlo.reshape %386 : (tensor<1xf64>) -> tensor<f64>
    %388 = stablehlo.slice %370 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %389 = stablehlo.reshape %388 : (tensor<1xf64>) -> tensor<f64>
    %390 = stablehlo.multiply %387, %389 : tensor<f64>
    %391 = stablehlo.subtract %385, %390 : tensor<f64>
    %392 = stablehlo.reshape %391 : (tensor<f64>) -> tensor<1xf64>
    %393 = stablehlo.multiply %78, %389 : tensor<f64>
    %394 = stablehlo.multiply %375, %383 : tensor<f64>
    %395 = stablehlo.subtract %393, %394 : tensor<f64>
    %396 = stablehlo.multiply %381, %377 : tensor<f64>
    %397 = stablehlo.add %395, %396 : tensor<f64>
    %398 = stablehlo.multiply %387, %372 : tensor<f64>
    %399 = stablehlo.add %397, %398 : tensor<f64>
    %400 = stablehlo.reshape %399 : (tensor<f64>) -> tensor<1xf64>
    %401 = stablehlo.multiply %78, %383 : tensor<f64>
    %402 = stablehlo.multiply %375, %389 : tensor<f64>
    %403 = stablehlo.add %401, %402 : tensor<f64>
    %404 = stablehlo.multiply %381, %372 : tensor<f64>
    %405 = stablehlo.subtract %403, %404 : tensor<f64>
    %406 = stablehlo.multiply %387, %377 : tensor<f64>
    %407 = stablehlo.add %405, %406 : tensor<f64>
    %408 = stablehlo.reshape %407 : (tensor<f64>) -> tensor<1xf64>
    %409 = stablehlo.multiply %78, %377 : tensor<f64>
    %410 = stablehlo.multiply %375, %372 : tensor<f64>
    %411 = stablehlo.subtract %409, %410 : tensor<f64>
    %412 = stablehlo.multiply %381, %389 : tensor<f64>
    %413 = stablehlo.subtract %411, %412 : tensor<f64>
    %414 = stablehlo.multiply %387, %383 : tensor<f64>
    %415 = stablehlo.subtract %413, %414 : tensor<f64>
    %416 = stablehlo.reshape %415 : (tensor<f64>) -> tensor<1xf64>
    %417 = stablehlo.concatenate %392, %400, %408, %416, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %418 = stablehlo.slice %417 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %419 = stablehlo.reshape %418 : (tensor<1xf64>) -> tensor<f64>
    %420 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %421 = stablehlo.reshape %420 : (tensor<1xf64>) -> tensor<f64>
    %422 = stablehlo.negate %421 : tensor<f64>
    %423 = stablehlo.reshape %422 : (tensor<f64>) -> tensor<1xf64>
    %424 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %425 = stablehlo.reshape %424 : (tensor<1xf64>) -> tensor<f64>
    %426 = stablehlo.negate %425 : tensor<f64>
    %427 = stablehlo.reshape %426 : (tensor<f64>) -> tensor<1xf64>
    %428 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %429 = stablehlo.reshape %428 : (tensor<1xf64>) -> tensor<f64>
    %430 = stablehlo.negate %429 : tensor<f64>
    %431 = stablehlo.reshape %430 : (tensor<f64>) -> tensor<1xf64>
    %432 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %433 = stablehlo.reshape %432 : (tensor<1xf64>) -> tensor<f64>
    %434 = stablehlo.reshape %433 : (tensor<f64>) -> tensor<1xf64>
    %435 = stablehlo.concatenate %423, %427, %431, %434, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %436 = stablehlo.dot_general %76, %76, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %437 = stablehlo.broadcast_in_dim %436, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %438 = stablehlo.divide %435, %437 : tensor<4xf64>
    %439 = stablehlo.slice %438 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %440 = stablehlo.reshape %439 : (tensor<1xf64>) -> tensor<f64>
    %441 = stablehlo.multiply %419, %440 : tensor<f64>
    %442 = stablehlo.slice %417 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %443 = stablehlo.reshape %442 : (tensor<1xf64>) -> tensor<f64>
    %444 = stablehlo.slice %438 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %445 = stablehlo.reshape %444 : (tensor<1xf64>) -> tensor<f64>
    %446 = stablehlo.multiply %443, %445 : tensor<f64>
    %447 = stablehlo.add %441, %446 : tensor<f64>
    %448 = stablehlo.slice %417 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %449 = stablehlo.reshape %448 : (tensor<1xf64>) -> tensor<f64>
    %450 = stablehlo.slice %438 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %451 = stablehlo.reshape %450 : (tensor<1xf64>) -> tensor<f64>
    %452 = stablehlo.multiply %449, %451 : tensor<f64>
    %453 = stablehlo.add %447, %452 : tensor<f64>
    %454 = stablehlo.slice %417 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %455 = stablehlo.reshape %454 : (tensor<1xf64>) -> tensor<f64>
    %456 = stablehlo.slice %438 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %457 = stablehlo.reshape %456 : (tensor<1xf64>) -> tensor<f64>
    %458 = stablehlo.multiply %455, %457 : tensor<f64>
    %459 = stablehlo.subtract %453, %458 : tensor<f64>
    %460 = stablehlo.reshape %459 : (tensor<f64>) -> tensor<1xf64>
    %461 = stablehlo.multiply %419, %457 : tensor<f64>
    %462 = stablehlo.multiply %443, %451 : tensor<f64>
    %463 = stablehlo.subtract %461, %462 : tensor<f64>
    %464 = stablehlo.multiply %449, %445 : tensor<f64>
    %465 = stablehlo.add %463, %464 : tensor<f64>
    %466 = stablehlo.multiply %455, %440 : tensor<f64>
    %467 = stablehlo.add %465, %466 : tensor<f64>
    %468 = stablehlo.reshape %467 : (tensor<f64>) -> tensor<1xf64>
    %469 = stablehlo.multiply %419, %451 : tensor<f64>
    %470 = stablehlo.multiply %443, %457 : tensor<f64>
    %471 = stablehlo.add %469, %470 : tensor<f64>
    %472 = stablehlo.multiply %449, %440 : tensor<f64>
    %473 = stablehlo.subtract %471, %472 : tensor<f64>
    %474 = stablehlo.multiply %455, %445 : tensor<f64>
    %475 = stablehlo.add %473, %474 : tensor<f64>
    %476 = stablehlo.reshape %475 : (tensor<f64>) -> tensor<1xf64>
    %477 = stablehlo.multiply %419, %445 : tensor<f64>
    %478 = stablehlo.multiply %443, %440 : tensor<f64>
    %479 = stablehlo.subtract %477, %478 : tensor<f64>
    %480 = stablehlo.multiply %449, %457 : tensor<f64>
    %481 = stablehlo.subtract %479, %480 : tensor<f64>
    %482 = stablehlo.multiply %455, %451 : tensor<f64>
    %483 = stablehlo.subtract %481, %482 : tensor<f64>
    %484 = stablehlo.reshape %483 : (tensor<f64>) -> tensor<1xf64>
    %485 = stablehlo.concatenate %460, %468, %476, %484, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %486 = stablehlo.slice %485 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %487 = stablehlo.reshape %486 : (tensor<1xf64>) -> tensor<f64>
    %488 = stablehlo.reshape %487 : (tensor<f64>) -> tensor<1xf64>
    %489 = stablehlo.slice %485 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %490 = stablehlo.reshape %489 : (tensor<1xf64>) -> tensor<f64>
    %491 = stablehlo.reshape %490 : (tensor<f64>) -> tensor<1xf64>
    %492 = stablehlo.slice %485 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %493 = stablehlo.reshape %492 : (tensor<1xf64>) -> tensor<f64>
    %494 = stablehlo.reshape %493 : (tensor<f64>) -> tensor<1xf64>
    %495 = stablehlo.concatenate %488, %491, %494, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %496 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %497 = stablehlo.reshape %496 : (tensor<1xf64>) -> tensor<f64>
    %498 = stablehlo.slice %367 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %499 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %500 = stablehlo.concatenate %498, %499, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %501 = stablehlo.slice %500 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %502 = stablehlo.reshape %501 : (tensor<1xf64>) -> tensor<f64>
    %503 = stablehlo.multiply %497, %502 : tensor<f64>
    %504 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %505 = stablehlo.reshape %504 : (tensor<1xf64>) -> tensor<f64>
    %506 = stablehlo.slice %500 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %507 = stablehlo.reshape %506 : (tensor<1xf64>) -> tensor<f64>
    %508 = stablehlo.multiply %505, %507 : tensor<f64>
    %509 = stablehlo.add %503, %508 : tensor<f64>
    %510 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %511 = stablehlo.reshape %510 : (tensor<1xf64>) -> tensor<f64>
    %512 = stablehlo.slice %500 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %513 = stablehlo.reshape %512 : (tensor<1xf64>) -> tensor<f64>
    %514 = stablehlo.multiply %511, %513 : tensor<f64>
    %515 = stablehlo.add %509, %514 : tensor<f64>
    %516 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %517 = stablehlo.reshape %516 : (tensor<1xf64>) -> tensor<f64>
    %518 = stablehlo.slice %500 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %519 = stablehlo.reshape %518 : (tensor<1xf64>) -> tensor<f64>
    %520 = stablehlo.multiply %517, %519 : tensor<f64>
    %521 = stablehlo.subtract %515, %520 : tensor<f64>
    %522 = stablehlo.reshape %521 : (tensor<f64>) -> tensor<1xf64>
    %523 = stablehlo.multiply %497, %519 : tensor<f64>
    %524 = stablehlo.multiply %505, %513 : tensor<f64>
    %525 = stablehlo.subtract %523, %524 : tensor<f64>
    %526 = stablehlo.multiply %511, %507 : tensor<f64>
    %527 = stablehlo.add %525, %526 : tensor<f64>
    %528 = stablehlo.multiply %517, %502 : tensor<f64>
    %529 = stablehlo.add %527, %528 : tensor<f64>
    %530 = stablehlo.reshape %529 : (tensor<f64>) -> tensor<1xf64>
    %531 = stablehlo.multiply %497, %513 : tensor<f64>
    %532 = stablehlo.multiply %505, %519 : tensor<f64>
    %533 = stablehlo.add %531, %532 : tensor<f64>
    %534 = stablehlo.multiply %511, %502 : tensor<f64>
    %535 = stablehlo.subtract %533, %534 : tensor<f64>
    %536 = stablehlo.multiply %517, %507 : tensor<f64>
    %537 = stablehlo.add %535, %536 : tensor<f64>
    %538 = stablehlo.reshape %537 : (tensor<f64>) -> tensor<1xf64>
    %539 = stablehlo.multiply %497, %507 : tensor<f64>
    %540 = stablehlo.multiply %505, %502 : tensor<f64>
    %541 = stablehlo.subtract %539, %540 : tensor<f64>
    %542 = stablehlo.multiply %511, %519 : tensor<f64>
    %543 = stablehlo.subtract %541, %542 : tensor<f64>
    %544 = stablehlo.multiply %517, %513 : tensor<f64>
    %545 = stablehlo.subtract %543, %544 : tensor<f64>
    %546 = stablehlo.reshape %545 : (tensor<f64>) -> tensor<1xf64>
    %547 = stablehlo.concatenate %522, %530, %538, %546, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %548 = stablehlo.slice %547 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %549 = stablehlo.reshape %548 : (tensor<1xf64>) -> tensor<f64>
    %550 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %551 = stablehlo.reshape %550 : (tensor<1xf64>) -> tensor<f64>
    %552 = stablehlo.negate %551 : tensor<f64>
    %553 = stablehlo.reshape %552 : (tensor<f64>) -> tensor<1xf64>
    %554 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %555 = stablehlo.reshape %554 : (tensor<1xf64>) -> tensor<f64>
    %556 = stablehlo.negate %555 : tensor<f64>
    %557 = stablehlo.reshape %556 : (tensor<f64>) -> tensor<1xf64>
    %558 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %559 = stablehlo.reshape %558 : (tensor<1xf64>) -> tensor<f64>
    %560 = stablehlo.negate %559 : tensor<f64>
    %561 = stablehlo.reshape %560 : (tensor<f64>) -> tensor<1xf64>
    %562 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %563 = stablehlo.reshape %562 : (tensor<1xf64>) -> tensor<f64>
    %564 = stablehlo.reshape %563 : (tensor<f64>) -> tensor<1xf64>
    %565 = stablehlo.concatenate %553, %557, %561, %564, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %566 = stablehlo.dot_general %76, %76, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %567 = stablehlo.broadcast_in_dim %566, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %568 = stablehlo.divide %565, %567 : tensor<4xf64>
    %569 = stablehlo.slice %568 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %570 = stablehlo.reshape %569 : (tensor<1xf64>) -> tensor<f64>
    %571 = stablehlo.multiply %549, %570 : tensor<f64>
    %572 = stablehlo.slice %547 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %573 = stablehlo.reshape %572 : (tensor<1xf64>) -> tensor<f64>
    %574 = stablehlo.slice %568 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %575 = stablehlo.reshape %574 : (tensor<1xf64>) -> tensor<f64>
    %576 = stablehlo.multiply %573, %575 : tensor<f64>
    %577 = stablehlo.add %571, %576 : tensor<f64>
    %578 = stablehlo.slice %547 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %579 = stablehlo.reshape %578 : (tensor<1xf64>) -> tensor<f64>
    %580 = stablehlo.slice %568 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %581 = stablehlo.reshape %580 : (tensor<1xf64>) -> tensor<f64>
    %582 = stablehlo.multiply %579, %581 : tensor<f64>
    %583 = stablehlo.add %577, %582 : tensor<f64>
    %584 = stablehlo.slice %547 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %585 = stablehlo.reshape %584 : (tensor<1xf64>) -> tensor<f64>
    %586 = stablehlo.slice %568 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %587 = stablehlo.reshape %586 : (tensor<1xf64>) -> tensor<f64>
    %588 = stablehlo.multiply %585, %587 : tensor<f64>
    %589 = stablehlo.subtract %583, %588 : tensor<f64>
    %590 = stablehlo.reshape %589 : (tensor<f64>) -> tensor<1xf64>
    %591 = stablehlo.multiply %549, %587 : tensor<f64>
    %592 = stablehlo.multiply %573, %581 : tensor<f64>
    %593 = stablehlo.subtract %591, %592 : tensor<f64>
    %594 = stablehlo.multiply %579, %575 : tensor<f64>
    %595 = stablehlo.add %593, %594 : tensor<f64>
    %596 = stablehlo.multiply %585, %570 : tensor<f64>
    %597 = stablehlo.add %595, %596 : tensor<f64>
    %598 = stablehlo.reshape %597 : (tensor<f64>) -> tensor<1xf64>
    %599 = stablehlo.multiply %549, %581 : tensor<f64>
    %600 = stablehlo.multiply %573, %587 : tensor<f64>
    %601 = stablehlo.add %599, %600 : tensor<f64>
    %602 = stablehlo.multiply %579, %570 : tensor<f64>
    %603 = stablehlo.subtract %601, %602 : tensor<f64>
    %604 = stablehlo.multiply %585, %575 : tensor<f64>
    %605 = stablehlo.add %603, %604 : tensor<f64>
    %606 = stablehlo.reshape %605 : (tensor<f64>) -> tensor<1xf64>
    %607 = stablehlo.multiply %549, %575 : tensor<f64>
    %608 = stablehlo.multiply %573, %570 : tensor<f64>
    %609 = stablehlo.subtract %607, %608 : tensor<f64>
    %610 = stablehlo.multiply %579, %587 : tensor<f64>
    %611 = stablehlo.subtract %609, %610 : tensor<f64>
    %612 = stablehlo.multiply %585, %581 : tensor<f64>
    %613 = stablehlo.subtract %611, %612 : tensor<f64>
    %614 = stablehlo.reshape %613 : (tensor<f64>) -> tensor<1xf64>
    %615 = stablehlo.concatenate %590, %598, %606, %614, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %616 = stablehlo.slice %615 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %617 = stablehlo.reshape %616 : (tensor<1xf64>) -> tensor<f64>
    %618 = stablehlo.reshape %617 : (tensor<f64>) -> tensor<1xf64>
    %619 = stablehlo.slice %615 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %620 = stablehlo.reshape %619 : (tensor<1xf64>) -> tensor<f64>
    %621 = stablehlo.reshape %620 : (tensor<f64>) -> tensor<1xf64>
    %622 = stablehlo.slice %615 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %623 = stablehlo.reshape %622 : (tensor<1xf64>) -> tensor<f64>
    %624 = stablehlo.reshape %623 : (tensor<f64>) -> tensor<1xf64>
    %625 = stablehlo.concatenate %618, %621, %624, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %626 = stablehlo.concatenate %495, %625, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %627 = stablehlo.slice %arg3 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %628 = stablehlo.reshape %arg8 : (tensor<f64>) -> tensor<f64>
    %cst_7 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %629 = stablehlo.multiply %cst_7, %628 : tensor<f64>
    %630 = stablehlo.broadcast_in_dim %629, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %631 = stablehlo.multiply %630, %1 : tensor<6xf64>
    %632 = stablehlo.slice %631 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_8 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %633 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %634 = stablehlo.divide %632, %633 : tensor<3xf64>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %635 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %636 = stablehlo.concatenate %634, %635, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %637 = stablehlo.slice %636 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %638 = stablehlo.reshape %637 : (tensor<1xf64>) -> tensor<f64>
    %639 = stablehlo.slice %627 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %640 = stablehlo.reshape %639 : (tensor<1xf64>) -> tensor<f64>
    %641 = stablehlo.multiply %638, %640 : tensor<f64>
    %642 = stablehlo.slice %636 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %643 = stablehlo.reshape %642 : (tensor<1xf64>) -> tensor<f64>
    %644 = stablehlo.slice %627 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %645 = stablehlo.reshape %644 : (tensor<1xf64>) -> tensor<f64>
    %646 = stablehlo.multiply %643, %645 : tensor<f64>
    %647 = stablehlo.add %641, %646 : tensor<f64>
    %648 = stablehlo.slice %636 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %649 = stablehlo.reshape %648 : (tensor<1xf64>) -> tensor<f64>
    %650 = stablehlo.slice %627 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %651 = stablehlo.reshape %650 : (tensor<1xf64>) -> tensor<f64>
    %652 = stablehlo.multiply %649, %651 : tensor<f64>
    %653 = stablehlo.add %647, %652 : tensor<f64>
    %654 = stablehlo.slice %636 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %655 = stablehlo.reshape %654 : (tensor<1xf64>) -> tensor<f64>
    %656 = stablehlo.slice %627 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %657 = stablehlo.reshape %656 : (tensor<1xf64>) -> tensor<f64>
    %658 = stablehlo.multiply %655, %657 : tensor<f64>
    %659 = stablehlo.subtract %653, %658 : tensor<f64>
    %660 = stablehlo.reshape %659 : (tensor<f64>) -> tensor<1xf64>
    %661 = stablehlo.multiply %638, %657 : tensor<f64>
    %662 = stablehlo.multiply %643, %651 : tensor<f64>
    %663 = stablehlo.subtract %661, %662 : tensor<f64>
    %664 = stablehlo.multiply %649, %645 : tensor<f64>
    %665 = stablehlo.add %663, %664 : tensor<f64>
    %666 = stablehlo.multiply %655, %640 : tensor<f64>
    %667 = stablehlo.add %665, %666 : tensor<f64>
    %668 = stablehlo.reshape %667 : (tensor<f64>) -> tensor<1xf64>
    %669 = stablehlo.multiply %638, %651 : tensor<f64>
    %670 = stablehlo.multiply %643, %657 : tensor<f64>
    %671 = stablehlo.add %669, %670 : tensor<f64>
    %672 = stablehlo.multiply %649, %640 : tensor<f64>
    %673 = stablehlo.subtract %671, %672 : tensor<f64>
    %674 = stablehlo.multiply %655, %645 : tensor<f64>
    %675 = stablehlo.add %673, %674 : tensor<f64>
    %676 = stablehlo.reshape %675 : (tensor<f64>) -> tensor<1xf64>
    %677 = stablehlo.multiply %638, %645 : tensor<f64>
    %678 = stablehlo.multiply %643, %640 : tensor<f64>
    %679 = stablehlo.subtract %677, %678 : tensor<f64>
    %680 = stablehlo.multiply %649, %657 : tensor<f64>
    %681 = stablehlo.subtract %679, %680 : tensor<f64>
    %682 = stablehlo.multiply %655, %651 : tensor<f64>
    %683 = stablehlo.subtract %681, %682 : tensor<f64>
    %684 = stablehlo.reshape %683 : (tensor<f64>) -> tensor<1xf64>
    %685 = stablehlo.concatenate %660, %668, %676, %684, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %686 = stablehlo.add %627, %685 : tensor<4xf64>
    %687 = stablehlo.dot_general %686, %686, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %688 = stablehlo.sqrt %687 : tensor<f64>
    %689 = stablehlo.broadcast_in_dim %688, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %690 = stablehlo.divide %686, %689 : tensor<4xf64>
    %691 = stablehlo.slice %arg3 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %692 = stablehlo.slice %631 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %693 = stablehlo.add %691, %692 : tensor<3xf64>
    %694 = stablehlo.concatenate %690, %693, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %695 = stablehlo.broadcast_in_dim %629, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %696 = stablehlo.multiply %695, %626 : tensor<6xf64>
    %697 = stablehlo.add %1, %696 : tensor<6xf64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %698 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %699 = call @inner_55(%698, %arg6) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %700 = call @inner_57(%0, %697, %699) : (tensor<3xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %701 = stablehlo.slice %694 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %702 = stablehlo.slice %701 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %703 = stablehlo.reshape %702 : (tensor<1xf64>) -> tensor<f64>
    %704 = stablehlo.slice %701 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %705 = stablehlo.reshape %704 : (tensor<1xf64>) -> tensor<f64>
    %706 = stablehlo.negate %705 : tensor<f64>
    %707 = stablehlo.reshape %706 : (tensor<f64>) -> tensor<1xf64>
    %708 = stablehlo.slice %701 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %709 = stablehlo.reshape %708 : (tensor<1xf64>) -> tensor<f64>
    %710 = stablehlo.negate %709 : tensor<f64>
    %711 = stablehlo.reshape %710 : (tensor<f64>) -> tensor<1xf64>
    %712 = stablehlo.slice %701 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %713 = stablehlo.reshape %712 : (tensor<1xf64>) -> tensor<f64>
    %714 = stablehlo.negate %713 : tensor<f64>
    %715 = stablehlo.reshape %714 : (tensor<f64>) -> tensor<1xf64>
    %716 = stablehlo.slice %701 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %717 = stablehlo.reshape %716 : (tensor<1xf64>) -> tensor<f64>
    %718 = stablehlo.reshape %717 : (tensor<f64>) -> tensor<1xf64>
    %719 = stablehlo.concatenate %707, %711, %715, %718, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %720 = stablehlo.dot_general %701, %701, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %721 = stablehlo.broadcast_in_dim %720, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %722 = stablehlo.divide %719, %721 : tensor<4xf64>
    %723 = stablehlo.slice %722 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %724 = stablehlo.reshape %723 : (tensor<1xf64>) -> tensor<f64>
    %725 = stablehlo.slice %700 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %726 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %727 = stablehlo.concatenate %725, %726, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %728 = stablehlo.slice %727 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %729 = stablehlo.reshape %728 : (tensor<1xf64>) -> tensor<f64>
    %730 = stablehlo.multiply %724, %729 : tensor<f64>
    %731 = stablehlo.slice %722 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %732 = stablehlo.reshape %731 : (tensor<1xf64>) -> tensor<f64>
    %733 = stablehlo.slice %727 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %734 = stablehlo.reshape %733 : (tensor<1xf64>) -> tensor<f64>
    %735 = stablehlo.multiply %732, %734 : tensor<f64>
    %736 = stablehlo.add %730, %735 : tensor<f64>
    %737 = stablehlo.slice %722 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %738 = stablehlo.reshape %737 : (tensor<1xf64>) -> tensor<f64>
    %739 = stablehlo.slice %727 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %740 = stablehlo.reshape %739 : (tensor<1xf64>) -> tensor<f64>
    %741 = stablehlo.multiply %738, %740 : tensor<f64>
    %742 = stablehlo.add %736, %741 : tensor<f64>
    %743 = stablehlo.slice %722 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %744 = stablehlo.reshape %743 : (tensor<1xf64>) -> tensor<f64>
    %745 = stablehlo.slice %727 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %746 = stablehlo.reshape %745 : (tensor<1xf64>) -> tensor<f64>
    %747 = stablehlo.multiply %744, %746 : tensor<f64>
    %748 = stablehlo.subtract %742, %747 : tensor<f64>
    %749 = stablehlo.reshape %748 : (tensor<f64>) -> tensor<1xf64>
    %750 = stablehlo.multiply %724, %746 : tensor<f64>
    %751 = stablehlo.multiply %732, %740 : tensor<f64>
    %752 = stablehlo.subtract %750, %751 : tensor<f64>
    %753 = stablehlo.multiply %738, %734 : tensor<f64>
    %754 = stablehlo.add %752, %753 : tensor<f64>
    %755 = stablehlo.multiply %744, %729 : tensor<f64>
    %756 = stablehlo.add %754, %755 : tensor<f64>
    %757 = stablehlo.reshape %756 : (tensor<f64>) -> tensor<1xf64>
    %758 = stablehlo.multiply %724, %740 : tensor<f64>
    %759 = stablehlo.multiply %732, %746 : tensor<f64>
    %760 = stablehlo.add %758, %759 : tensor<f64>
    %761 = stablehlo.multiply %738, %729 : tensor<f64>
    %762 = stablehlo.subtract %760, %761 : tensor<f64>
    %763 = stablehlo.multiply %744, %734 : tensor<f64>
    %764 = stablehlo.add %762, %763 : tensor<f64>
    %765 = stablehlo.reshape %764 : (tensor<f64>) -> tensor<1xf64>
    %766 = stablehlo.multiply %724, %734 : tensor<f64>
    %767 = stablehlo.multiply %732, %729 : tensor<f64>
    %768 = stablehlo.subtract %766, %767 : tensor<f64>
    %769 = stablehlo.multiply %738, %746 : tensor<f64>
    %770 = stablehlo.subtract %768, %769 : tensor<f64>
    %771 = stablehlo.multiply %744, %740 : tensor<f64>
    %772 = stablehlo.subtract %770, %771 : tensor<f64>
    %773 = stablehlo.reshape %772 : (tensor<f64>) -> tensor<1xf64>
    %774 = stablehlo.concatenate %749, %757, %765, %773, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %775 = stablehlo.slice %774 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %776 = stablehlo.reshape %775 : (tensor<1xf64>) -> tensor<f64>
    %777 = stablehlo.slice %722 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %778 = stablehlo.reshape %777 : (tensor<1xf64>) -> tensor<f64>
    %779 = stablehlo.negate %778 : tensor<f64>
    %780 = stablehlo.reshape %779 : (tensor<f64>) -> tensor<1xf64>
    %781 = stablehlo.slice %722 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %782 = stablehlo.reshape %781 : (tensor<1xf64>) -> tensor<f64>
    %783 = stablehlo.negate %782 : tensor<f64>
    %784 = stablehlo.reshape %783 : (tensor<f64>) -> tensor<1xf64>
    %785 = stablehlo.slice %722 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %786 = stablehlo.reshape %785 : (tensor<1xf64>) -> tensor<f64>
    %787 = stablehlo.negate %786 : tensor<f64>
    %788 = stablehlo.reshape %787 : (tensor<f64>) -> tensor<1xf64>
    %789 = stablehlo.slice %722 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %790 = stablehlo.reshape %789 : (tensor<1xf64>) -> tensor<f64>
    %791 = stablehlo.reshape %790 : (tensor<f64>) -> tensor<1xf64>
    %792 = stablehlo.concatenate %780, %784, %788, %791, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %793 = stablehlo.dot_general %722, %722, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %794 = stablehlo.broadcast_in_dim %793, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %795 = stablehlo.divide %792, %794 : tensor<4xf64>
    %796 = stablehlo.slice %795 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %797 = stablehlo.reshape %796 : (tensor<1xf64>) -> tensor<f64>
    %798 = stablehlo.multiply %776, %797 : tensor<f64>
    %799 = stablehlo.slice %774 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %800 = stablehlo.reshape %799 : (tensor<1xf64>) -> tensor<f64>
    %801 = stablehlo.slice %795 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %802 = stablehlo.reshape %801 : (tensor<1xf64>) -> tensor<f64>
    %803 = stablehlo.multiply %800, %802 : tensor<f64>
    %804 = stablehlo.add %798, %803 : tensor<f64>
    %805 = stablehlo.slice %774 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %806 = stablehlo.reshape %805 : (tensor<1xf64>) -> tensor<f64>
    %807 = stablehlo.slice %795 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %808 = stablehlo.reshape %807 : (tensor<1xf64>) -> tensor<f64>
    %809 = stablehlo.multiply %806, %808 : tensor<f64>
    %810 = stablehlo.add %804, %809 : tensor<f64>
    %811 = stablehlo.slice %774 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %812 = stablehlo.reshape %811 : (tensor<1xf64>) -> tensor<f64>
    %813 = stablehlo.slice %795 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %814 = stablehlo.reshape %813 : (tensor<1xf64>) -> tensor<f64>
    %815 = stablehlo.multiply %812, %814 : tensor<f64>
    %816 = stablehlo.subtract %810, %815 : tensor<f64>
    %817 = stablehlo.reshape %816 : (tensor<f64>) -> tensor<1xf64>
    %818 = stablehlo.multiply %776, %814 : tensor<f64>
    %819 = stablehlo.multiply %800, %808 : tensor<f64>
    %820 = stablehlo.subtract %818, %819 : tensor<f64>
    %821 = stablehlo.multiply %806, %802 : tensor<f64>
    %822 = stablehlo.add %820, %821 : tensor<f64>
    %823 = stablehlo.multiply %812, %797 : tensor<f64>
    %824 = stablehlo.add %822, %823 : tensor<f64>
    %825 = stablehlo.reshape %824 : (tensor<f64>) -> tensor<1xf64>
    %826 = stablehlo.multiply %776, %808 : tensor<f64>
    %827 = stablehlo.multiply %800, %814 : tensor<f64>
    %828 = stablehlo.add %826, %827 : tensor<f64>
    %829 = stablehlo.multiply %806, %797 : tensor<f64>
    %830 = stablehlo.subtract %828, %829 : tensor<f64>
    %831 = stablehlo.multiply %812, %802 : tensor<f64>
    %832 = stablehlo.add %830, %831 : tensor<f64>
    %833 = stablehlo.reshape %832 : (tensor<f64>) -> tensor<1xf64>
    %834 = stablehlo.multiply %776, %802 : tensor<f64>
    %835 = stablehlo.multiply %800, %797 : tensor<f64>
    %836 = stablehlo.subtract %834, %835 : tensor<f64>
    %837 = stablehlo.multiply %806, %814 : tensor<f64>
    %838 = stablehlo.subtract %836, %837 : tensor<f64>
    %839 = stablehlo.multiply %812, %808 : tensor<f64>
    %840 = stablehlo.subtract %838, %839 : tensor<f64>
    %841 = stablehlo.reshape %840 : (tensor<f64>) -> tensor<1xf64>
    %842 = stablehlo.concatenate %817, %825, %833, %841, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %843 = stablehlo.slice %842 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %844 = stablehlo.reshape %843 : (tensor<1xf64>) -> tensor<f64>
    %845 = stablehlo.reshape %844 : (tensor<f64>) -> tensor<1xf64>
    %846 = stablehlo.slice %842 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %847 = stablehlo.reshape %846 : (tensor<1xf64>) -> tensor<f64>
    %848 = stablehlo.reshape %847 : (tensor<f64>) -> tensor<1xf64>
    %849 = stablehlo.slice %842 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %850 = stablehlo.reshape %849 : (tensor<1xf64>) -> tensor<f64>
    %851 = stablehlo.reshape %850 : (tensor<f64>) -> tensor<1xf64>
    %852 = stablehlo.concatenate %845, %848, %851, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %853 = stablehlo.slice %722 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %854 = stablehlo.reshape %853 : (tensor<1xf64>) -> tensor<f64>
    %855 = stablehlo.slice %700 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %856 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %857 = stablehlo.concatenate %855, %856, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %858 = stablehlo.slice %857 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %859 = stablehlo.reshape %858 : (tensor<1xf64>) -> tensor<f64>
    %860 = stablehlo.multiply %854, %859 : tensor<f64>
    %861 = stablehlo.slice %722 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %862 = stablehlo.reshape %861 : (tensor<1xf64>) -> tensor<f64>
    %863 = stablehlo.slice %857 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %864 = stablehlo.reshape %863 : (tensor<1xf64>) -> tensor<f64>
    %865 = stablehlo.multiply %862, %864 : tensor<f64>
    %866 = stablehlo.add %860, %865 : tensor<f64>
    %867 = stablehlo.slice %722 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %868 = stablehlo.reshape %867 : (tensor<1xf64>) -> tensor<f64>
    %869 = stablehlo.slice %857 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %870 = stablehlo.reshape %869 : (tensor<1xf64>) -> tensor<f64>
    %871 = stablehlo.multiply %868, %870 : tensor<f64>
    %872 = stablehlo.add %866, %871 : tensor<f64>
    %873 = stablehlo.slice %722 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %874 = stablehlo.reshape %873 : (tensor<1xf64>) -> tensor<f64>
    %875 = stablehlo.slice %857 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %876 = stablehlo.reshape %875 : (tensor<1xf64>) -> tensor<f64>
    %877 = stablehlo.multiply %874, %876 : tensor<f64>
    %878 = stablehlo.subtract %872, %877 : tensor<f64>
    %879 = stablehlo.reshape %878 : (tensor<f64>) -> tensor<1xf64>
    %880 = stablehlo.multiply %854, %876 : tensor<f64>
    %881 = stablehlo.multiply %862, %870 : tensor<f64>
    %882 = stablehlo.subtract %880, %881 : tensor<f64>
    %883 = stablehlo.multiply %868, %864 : tensor<f64>
    %884 = stablehlo.add %882, %883 : tensor<f64>
    %885 = stablehlo.multiply %874, %859 : tensor<f64>
    %886 = stablehlo.add %884, %885 : tensor<f64>
    %887 = stablehlo.reshape %886 : (tensor<f64>) -> tensor<1xf64>
    %888 = stablehlo.multiply %854, %870 : tensor<f64>
    %889 = stablehlo.multiply %862, %876 : tensor<f64>
    %890 = stablehlo.add %888, %889 : tensor<f64>
    %891 = stablehlo.multiply %868, %859 : tensor<f64>
    %892 = stablehlo.subtract %890, %891 : tensor<f64>
    %893 = stablehlo.multiply %874, %864 : tensor<f64>
    %894 = stablehlo.add %892, %893 : tensor<f64>
    %895 = stablehlo.reshape %894 : (tensor<f64>) -> tensor<1xf64>
    %896 = stablehlo.multiply %854, %864 : tensor<f64>
    %897 = stablehlo.multiply %862, %859 : tensor<f64>
    %898 = stablehlo.subtract %896, %897 : tensor<f64>
    %899 = stablehlo.multiply %868, %876 : tensor<f64>
    %900 = stablehlo.subtract %898, %899 : tensor<f64>
    %901 = stablehlo.multiply %874, %870 : tensor<f64>
    %902 = stablehlo.subtract %900, %901 : tensor<f64>
    %903 = stablehlo.reshape %902 : (tensor<f64>) -> tensor<1xf64>
    %904 = stablehlo.concatenate %879, %887, %895, %903, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %905 = stablehlo.slice %904 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %906 = stablehlo.reshape %905 : (tensor<1xf64>) -> tensor<f64>
    %907 = stablehlo.slice %722 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %908 = stablehlo.reshape %907 : (tensor<1xf64>) -> tensor<f64>
    %909 = stablehlo.negate %908 : tensor<f64>
    %910 = stablehlo.reshape %909 : (tensor<f64>) -> tensor<1xf64>
    %911 = stablehlo.slice %722 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %912 = stablehlo.reshape %911 : (tensor<1xf64>) -> tensor<f64>
    %913 = stablehlo.negate %912 : tensor<f64>
    %914 = stablehlo.reshape %913 : (tensor<f64>) -> tensor<1xf64>
    %915 = stablehlo.slice %722 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %916 = stablehlo.reshape %915 : (tensor<1xf64>) -> tensor<f64>
    %917 = stablehlo.negate %916 : tensor<f64>
    %918 = stablehlo.reshape %917 : (tensor<f64>) -> tensor<1xf64>
    %919 = stablehlo.slice %722 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %920 = stablehlo.reshape %919 : (tensor<1xf64>) -> tensor<f64>
    %921 = stablehlo.reshape %920 : (tensor<f64>) -> tensor<1xf64>
    %922 = stablehlo.concatenate %910, %914, %918, %921, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %923 = stablehlo.dot_general %722, %722, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %924 = stablehlo.broadcast_in_dim %923, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %925 = stablehlo.divide %922, %924 : tensor<4xf64>
    %926 = stablehlo.slice %925 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %927 = stablehlo.reshape %926 : (tensor<1xf64>) -> tensor<f64>
    %928 = stablehlo.multiply %906, %927 : tensor<f64>
    %929 = stablehlo.slice %904 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %930 = stablehlo.reshape %929 : (tensor<1xf64>) -> tensor<f64>
    %931 = stablehlo.slice %925 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %932 = stablehlo.reshape %931 : (tensor<1xf64>) -> tensor<f64>
    %933 = stablehlo.multiply %930, %932 : tensor<f64>
    %934 = stablehlo.add %928, %933 : tensor<f64>
    %935 = stablehlo.slice %904 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %936 = stablehlo.reshape %935 : (tensor<1xf64>) -> tensor<f64>
    %937 = stablehlo.slice %925 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %938 = stablehlo.reshape %937 : (tensor<1xf64>) -> tensor<f64>
    %939 = stablehlo.multiply %936, %938 : tensor<f64>
    %940 = stablehlo.add %934, %939 : tensor<f64>
    %941 = stablehlo.slice %904 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %942 = stablehlo.reshape %941 : (tensor<1xf64>) -> tensor<f64>
    %943 = stablehlo.slice %925 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %944 = stablehlo.reshape %943 : (tensor<1xf64>) -> tensor<f64>
    %945 = stablehlo.multiply %942, %944 : tensor<f64>
    %946 = stablehlo.subtract %940, %945 : tensor<f64>
    %947 = stablehlo.reshape %946 : (tensor<f64>) -> tensor<1xf64>
    %948 = stablehlo.multiply %906, %944 : tensor<f64>
    %949 = stablehlo.multiply %930, %938 : tensor<f64>
    %950 = stablehlo.subtract %948, %949 : tensor<f64>
    %951 = stablehlo.multiply %936, %932 : tensor<f64>
    %952 = stablehlo.add %950, %951 : tensor<f64>
    %953 = stablehlo.multiply %942, %927 : tensor<f64>
    %954 = stablehlo.add %952, %953 : tensor<f64>
    %955 = stablehlo.reshape %954 : (tensor<f64>) -> tensor<1xf64>
    %956 = stablehlo.multiply %906, %938 : tensor<f64>
    %957 = stablehlo.multiply %930, %944 : tensor<f64>
    %958 = stablehlo.add %956, %957 : tensor<f64>
    %959 = stablehlo.multiply %936, %927 : tensor<f64>
    %960 = stablehlo.subtract %958, %959 : tensor<f64>
    %961 = stablehlo.multiply %942, %932 : tensor<f64>
    %962 = stablehlo.add %960, %961 : tensor<f64>
    %963 = stablehlo.reshape %962 : (tensor<f64>) -> tensor<1xf64>
    %964 = stablehlo.multiply %906, %932 : tensor<f64>
    %965 = stablehlo.multiply %930, %927 : tensor<f64>
    %966 = stablehlo.subtract %964, %965 : tensor<f64>
    %967 = stablehlo.multiply %936, %944 : tensor<f64>
    %968 = stablehlo.subtract %966, %967 : tensor<f64>
    %969 = stablehlo.multiply %942, %938 : tensor<f64>
    %970 = stablehlo.subtract %968, %969 : tensor<f64>
    %971 = stablehlo.reshape %970 : (tensor<f64>) -> tensor<1xf64>
    %972 = stablehlo.concatenate %947, %955, %963, %971, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %973 = stablehlo.slice %972 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %974 = stablehlo.reshape %973 : (tensor<1xf64>) -> tensor<f64>
    %975 = stablehlo.reshape %974 : (tensor<f64>) -> tensor<1xf64>
    %976 = stablehlo.slice %972 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %977 = stablehlo.reshape %976 : (tensor<1xf64>) -> tensor<f64>
    %978 = stablehlo.reshape %977 : (tensor<f64>) -> tensor<1xf64>
    %979 = stablehlo.slice %972 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %980 = stablehlo.reshape %979 : (tensor<1xf64>) -> tensor<f64>
    %981 = stablehlo.reshape %980 : (tensor<f64>) -> tensor<1xf64>
    %982 = stablehlo.concatenate %975, %978, %981, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %983 = stablehlo.concatenate %852, %982, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %984 = stablehlo.slice %983 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %985 = stablehlo.slice %arg6 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %986 = stablehlo.divide %984, %985 : tensor<3xf64>
    %987 = stablehlo.slice %983 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %988 = stablehlo.slice %arg6 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %989 = stablehlo.reshape %988 : (tensor<1xf64>) -> tensor<f64>
    %990 = stablehlo.broadcast_in_dim %989, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %991 = stablehlo.divide %987, %990 : tensor<3xf64>
    %992 = stablehlo.concatenate %986, %991, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %993 = stablehlo.slice %992 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %994 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %995 = stablehlo.concatenate %993, %994, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %996 = stablehlo.slice %995 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %997 = stablehlo.reshape %996 : (tensor<1xf64>) -> tensor<f64>
    %998 = stablehlo.multiply %703, %997 : tensor<f64>
    %999 = stablehlo.slice %701 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1000 = stablehlo.reshape %999 : (tensor<1xf64>) -> tensor<f64>
    %1001 = stablehlo.slice %995 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1002 = stablehlo.reshape %1001 : (tensor<1xf64>) -> tensor<f64>
    %1003 = stablehlo.multiply %1000, %1002 : tensor<f64>
    %1004 = stablehlo.add %998, %1003 : tensor<f64>
    %1005 = stablehlo.slice %701 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1006 = stablehlo.reshape %1005 : (tensor<1xf64>) -> tensor<f64>
    %1007 = stablehlo.slice %995 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1008 = stablehlo.reshape %1007 : (tensor<1xf64>) -> tensor<f64>
    %1009 = stablehlo.multiply %1006, %1008 : tensor<f64>
    %1010 = stablehlo.add %1004, %1009 : tensor<f64>
    %1011 = stablehlo.slice %701 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1012 = stablehlo.reshape %1011 : (tensor<1xf64>) -> tensor<f64>
    %1013 = stablehlo.slice %995 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1014 = stablehlo.reshape %1013 : (tensor<1xf64>) -> tensor<f64>
    %1015 = stablehlo.multiply %1012, %1014 : tensor<f64>
    %1016 = stablehlo.subtract %1010, %1015 : tensor<f64>
    %1017 = stablehlo.reshape %1016 : (tensor<f64>) -> tensor<1xf64>
    %1018 = stablehlo.multiply %703, %1014 : tensor<f64>
    %1019 = stablehlo.multiply %1000, %1008 : tensor<f64>
    %1020 = stablehlo.subtract %1018, %1019 : tensor<f64>
    %1021 = stablehlo.multiply %1006, %1002 : tensor<f64>
    %1022 = stablehlo.add %1020, %1021 : tensor<f64>
    %1023 = stablehlo.multiply %1012, %997 : tensor<f64>
    %1024 = stablehlo.add %1022, %1023 : tensor<f64>
    %1025 = stablehlo.reshape %1024 : (tensor<f64>) -> tensor<1xf64>
    %1026 = stablehlo.multiply %703, %1008 : tensor<f64>
    %1027 = stablehlo.multiply %1000, %1014 : tensor<f64>
    %1028 = stablehlo.add %1026, %1027 : tensor<f64>
    %1029 = stablehlo.multiply %1006, %997 : tensor<f64>
    %1030 = stablehlo.subtract %1028, %1029 : tensor<f64>
    %1031 = stablehlo.multiply %1012, %1002 : tensor<f64>
    %1032 = stablehlo.add %1030, %1031 : tensor<f64>
    %1033 = stablehlo.reshape %1032 : (tensor<f64>) -> tensor<1xf64>
    %1034 = stablehlo.multiply %703, %1002 : tensor<f64>
    %1035 = stablehlo.multiply %1000, %997 : tensor<f64>
    %1036 = stablehlo.subtract %1034, %1035 : tensor<f64>
    %1037 = stablehlo.multiply %1006, %1014 : tensor<f64>
    %1038 = stablehlo.subtract %1036, %1037 : tensor<f64>
    %1039 = stablehlo.multiply %1012, %1008 : tensor<f64>
    %1040 = stablehlo.subtract %1038, %1039 : tensor<f64>
    %1041 = stablehlo.reshape %1040 : (tensor<f64>) -> tensor<1xf64>
    %1042 = stablehlo.concatenate %1017, %1025, %1033, %1041, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1043 = stablehlo.slice %1042 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1044 = stablehlo.reshape %1043 : (tensor<1xf64>) -> tensor<f64>
    %1045 = stablehlo.slice %701 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1046 = stablehlo.reshape %1045 : (tensor<1xf64>) -> tensor<f64>
    %1047 = stablehlo.negate %1046 : tensor<f64>
    %1048 = stablehlo.reshape %1047 : (tensor<f64>) -> tensor<1xf64>
    %1049 = stablehlo.slice %701 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1050 = stablehlo.reshape %1049 : (tensor<1xf64>) -> tensor<f64>
    %1051 = stablehlo.negate %1050 : tensor<f64>
    %1052 = stablehlo.reshape %1051 : (tensor<f64>) -> tensor<1xf64>
    %1053 = stablehlo.slice %701 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1054 = stablehlo.reshape %1053 : (tensor<1xf64>) -> tensor<f64>
    %1055 = stablehlo.negate %1054 : tensor<f64>
    %1056 = stablehlo.reshape %1055 : (tensor<f64>) -> tensor<1xf64>
    %1057 = stablehlo.slice %701 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1058 = stablehlo.reshape %1057 : (tensor<1xf64>) -> tensor<f64>
    %1059 = stablehlo.reshape %1058 : (tensor<f64>) -> tensor<1xf64>
    %1060 = stablehlo.concatenate %1048, %1052, %1056, %1059, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1061 = stablehlo.dot_general %701, %701, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1062 = stablehlo.broadcast_in_dim %1061, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1063 = stablehlo.divide %1060, %1062 : tensor<4xf64>
    %1064 = stablehlo.slice %1063 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1065 = stablehlo.reshape %1064 : (tensor<1xf64>) -> tensor<f64>
    %1066 = stablehlo.multiply %1044, %1065 : tensor<f64>
    %1067 = stablehlo.slice %1042 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1068 = stablehlo.reshape %1067 : (tensor<1xf64>) -> tensor<f64>
    %1069 = stablehlo.slice %1063 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1070 = stablehlo.reshape %1069 : (tensor<1xf64>) -> tensor<f64>
    %1071 = stablehlo.multiply %1068, %1070 : tensor<f64>
    %1072 = stablehlo.add %1066, %1071 : tensor<f64>
    %1073 = stablehlo.slice %1042 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1074 = stablehlo.reshape %1073 : (tensor<1xf64>) -> tensor<f64>
    %1075 = stablehlo.slice %1063 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1076 = stablehlo.reshape %1075 : (tensor<1xf64>) -> tensor<f64>
    %1077 = stablehlo.multiply %1074, %1076 : tensor<f64>
    %1078 = stablehlo.add %1072, %1077 : tensor<f64>
    %1079 = stablehlo.slice %1042 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1080 = stablehlo.reshape %1079 : (tensor<1xf64>) -> tensor<f64>
    %1081 = stablehlo.slice %1063 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1082 = stablehlo.reshape %1081 : (tensor<1xf64>) -> tensor<f64>
    %1083 = stablehlo.multiply %1080, %1082 : tensor<f64>
    %1084 = stablehlo.subtract %1078, %1083 : tensor<f64>
    %1085 = stablehlo.reshape %1084 : (tensor<f64>) -> tensor<1xf64>
    %1086 = stablehlo.multiply %1044, %1082 : tensor<f64>
    %1087 = stablehlo.multiply %1068, %1076 : tensor<f64>
    %1088 = stablehlo.subtract %1086, %1087 : tensor<f64>
    %1089 = stablehlo.multiply %1074, %1070 : tensor<f64>
    %1090 = stablehlo.add %1088, %1089 : tensor<f64>
    %1091 = stablehlo.multiply %1080, %1065 : tensor<f64>
    %1092 = stablehlo.add %1090, %1091 : tensor<f64>
    %1093 = stablehlo.reshape %1092 : (tensor<f64>) -> tensor<1xf64>
    %1094 = stablehlo.multiply %1044, %1076 : tensor<f64>
    %1095 = stablehlo.multiply %1068, %1082 : tensor<f64>
    %1096 = stablehlo.add %1094, %1095 : tensor<f64>
    %1097 = stablehlo.multiply %1074, %1065 : tensor<f64>
    %1098 = stablehlo.subtract %1096, %1097 : tensor<f64>
    %1099 = stablehlo.multiply %1080, %1070 : tensor<f64>
    %1100 = stablehlo.add %1098, %1099 : tensor<f64>
    %1101 = stablehlo.reshape %1100 : (tensor<f64>) -> tensor<1xf64>
    %1102 = stablehlo.multiply %1044, %1070 : tensor<f64>
    %1103 = stablehlo.multiply %1068, %1065 : tensor<f64>
    %1104 = stablehlo.subtract %1102, %1103 : tensor<f64>
    %1105 = stablehlo.multiply %1074, %1082 : tensor<f64>
    %1106 = stablehlo.subtract %1104, %1105 : tensor<f64>
    %1107 = stablehlo.multiply %1080, %1076 : tensor<f64>
    %1108 = stablehlo.subtract %1106, %1107 : tensor<f64>
    %1109 = stablehlo.reshape %1108 : (tensor<f64>) -> tensor<1xf64>
    %1110 = stablehlo.concatenate %1085, %1093, %1101, %1109, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1111 = stablehlo.slice %1110 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1112 = stablehlo.reshape %1111 : (tensor<1xf64>) -> tensor<f64>
    %1113 = stablehlo.reshape %1112 : (tensor<f64>) -> tensor<1xf64>
    %1114 = stablehlo.slice %1110 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1115 = stablehlo.reshape %1114 : (tensor<1xf64>) -> tensor<f64>
    %1116 = stablehlo.reshape %1115 : (tensor<f64>) -> tensor<1xf64>
    %1117 = stablehlo.slice %1110 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1118 = stablehlo.reshape %1117 : (tensor<1xf64>) -> tensor<f64>
    %1119 = stablehlo.reshape %1118 : (tensor<f64>) -> tensor<1xf64>
    %1120 = stablehlo.concatenate %1113, %1116, %1119, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1121 = stablehlo.slice %701 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1122 = stablehlo.reshape %1121 : (tensor<1xf64>) -> tensor<f64>
    %1123 = stablehlo.slice %992 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1124 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1125 = stablehlo.concatenate %1123, %1124, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1126 = stablehlo.slice %1125 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1127 = stablehlo.reshape %1126 : (tensor<1xf64>) -> tensor<f64>
    %1128 = stablehlo.multiply %1122, %1127 : tensor<f64>
    %1129 = stablehlo.slice %701 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1130 = stablehlo.reshape %1129 : (tensor<1xf64>) -> tensor<f64>
    %1131 = stablehlo.slice %1125 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1132 = stablehlo.reshape %1131 : (tensor<1xf64>) -> tensor<f64>
    %1133 = stablehlo.multiply %1130, %1132 : tensor<f64>
    %1134 = stablehlo.add %1128, %1133 : tensor<f64>
    %1135 = stablehlo.slice %701 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1136 = stablehlo.reshape %1135 : (tensor<1xf64>) -> tensor<f64>
    %1137 = stablehlo.slice %1125 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1138 = stablehlo.reshape %1137 : (tensor<1xf64>) -> tensor<f64>
    %1139 = stablehlo.multiply %1136, %1138 : tensor<f64>
    %1140 = stablehlo.add %1134, %1139 : tensor<f64>
    %1141 = stablehlo.slice %701 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1142 = stablehlo.reshape %1141 : (tensor<1xf64>) -> tensor<f64>
    %1143 = stablehlo.slice %1125 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1144 = stablehlo.reshape %1143 : (tensor<1xf64>) -> tensor<f64>
    %1145 = stablehlo.multiply %1142, %1144 : tensor<f64>
    %1146 = stablehlo.subtract %1140, %1145 : tensor<f64>
    %1147 = stablehlo.reshape %1146 : (tensor<f64>) -> tensor<1xf64>
    %1148 = stablehlo.multiply %1122, %1144 : tensor<f64>
    %1149 = stablehlo.multiply %1130, %1138 : tensor<f64>
    %1150 = stablehlo.subtract %1148, %1149 : tensor<f64>
    %1151 = stablehlo.multiply %1136, %1132 : tensor<f64>
    %1152 = stablehlo.add %1150, %1151 : tensor<f64>
    %1153 = stablehlo.multiply %1142, %1127 : tensor<f64>
    %1154 = stablehlo.add %1152, %1153 : tensor<f64>
    %1155 = stablehlo.reshape %1154 : (tensor<f64>) -> tensor<1xf64>
    %1156 = stablehlo.multiply %1122, %1138 : tensor<f64>
    %1157 = stablehlo.multiply %1130, %1144 : tensor<f64>
    %1158 = stablehlo.add %1156, %1157 : tensor<f64>
    %1159 = stablehlo.multiply %1136, %1127 : tensor<f64>
    %1160 = stablehlo.subtract %1158, %1159 : tensor<f64>
    %1161 = stablehlo.multiply %1142, %1132 : tensor<f64>
    %1162 = stablehlo.add %1160, %1161 : tensor<f64>
    %1163 = stablehlo.reshape %1162 : (tensor<f64>) -> tensor<1xf64>
    %1164 = stablehlo.multiply %1122, %1132 : tensor<f64>
    %1165 = stablehlo.multiply %1130, %1127 : tensor<f64>
    %1166 = stablehlo.subtract %1164, %1165 : tensor<f64>
    %1167 = stablehlo.multiply %1136, %1144 : tensor<f64>
    %1168 = stablehlo.subtract %1166, %1167 : tensor<f64>
    %1169 = stablehlo.multiply %1142, %1138 : tensor<f64>
    %1170 = stablehlo.subtract %1168, %1169 : tensor<f64>
    %1171 = stablehlo.reshape %1170 : (tensor<f64>) -> tensor<1xf64>
    %1172 = stablehlo.concatenate %1147, %1155, %1163, %1171, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1173 = stablehlo.slice %1172 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1174 = stablehlo.reshape %1173 : (tensor<1xf64>) -> tensor<f64>
    %1175 = stablehlo.slice %701 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1176 = stablehlo.reshape %1175 : (tensor<1xf64>) -> tensor<f64>
    %1177 = stablehlo.negate %1176 : tensor<f64>
    %1178 = stablehlo.reshape %1177 : (tensor<f64>) -> tensor<1xf64>
    %1179 = stablehlo.slice %701 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1180 = stablehlo.reshape %1179 : (tensor<1xf64>) -> tensor<f64>
    %1181 = stablehlo.negate %1180 : tensor<f64>
    %1182 = stablehlo.reshape %1181 : (tensor<f64>) -> tensor<1xf64>
    %1183 = stablehlo.slice %701 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1184 = stablehlo.reshape %1183 : (tensor<1xf64>) -> tensor<f64>
    %1185 = stablehlo.negate %1184 : tensor<f64>
    %1186 = stablehlo.reshape %1185 : (tensor<f64>) -> tensor<1xf64>
    %1187 = stablehlo.slice %701 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1188 = stablehlo.reshape %1187 : (tensor<1xf64>) -> tensor<f64>
    %1189 = stablehlo.reshape %1188 : (tensor<f64>) -> tensor<1xf64>
    %1190 = stablehlo.concatenate %1178, %1182, %1186, %1189, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1191 = stablehlo.dot_general %701, %701, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1192 = stablehlo.broadcast_in_dim %1191, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1193 = stablehlo.divide %1190, %1192 : tensor<4xf64>
    %1194 = stablehlo.slice %1193 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1195 = stablehlo.reshape %1194 : (tensor<1xf64>) -> tensor<f64>
    %1196 = stablehlo.multiply %1174, %1195 : tensor<f64>
    %1197 = stablehlo.slice %1172 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1198 = stablehlo.reshape %1197 : (tensor<1xf64>) -> tensor<f64>
    %1199 = stablehlo.slice %1193 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1200 = stablehlo.reshape %1199 : (tensor<1xf64>) -> tensor<f64>
    %1201 = stablehlo.multiply %1198, %1200 : tensor<f64>
    %1202 = stablehlo.add %1196, %1201 : tensor<f64>
    %1203 = stablehlo.slice %1172 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1204 = stablehlo.reshape %1203 : (tensor<1xf64>) -> tensor<f64>
    %1205 = stablehlo.slice %1193 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1206 = stablehlo.reshape %1205 : (tensor<1xf64>) -> tensor<f64>
    %1207 = stablehlo.multiply %1204, %1206 : tensor<f64>
    %1208 = stablehlo.add %1202, %1207 : tensor<f64>
    %1209 = stablehlo.slice %1172 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1210 = stablehlo.reshape %1209 : (tensor<1xf64>) -> tensor<f64>
    %1211 = stablehlo.slice %1193 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1212 = stablehlo.reshape %1211 : (tensor<1xf64>) -> tensor<f64>
    %1213 = stablehlo.multiply %1210, %1212 : tensor<f64>
    %1214 = stablehlo.subtract %1208, %1213 : tensor<f64>
    %1215 = stablehlo.reshape %1214 : (tensor<f64>) -> tensor<1xf64>
    %1216 = stablehlo.multiply %1174, %1212 : tensor<f64>
    %1217 = stablehlo.multiply %1198, %1206 : tensor<f64>
    %1218 = stablehlo.subtract %1216, %1217 : tensor<f64>
    %1219 = stablehlo.multiply %1204, %1200 : tensor<f64>
    %1220 = stablehlo.add %1218, %1219 : tensor<f64>
    %1221 = stablehlo.multiply %1210, %1195 : tensor<f64>
    %1222 = stablehlo.add %1220, %1221 : tensor<f64>
    %1223 = stablehlo.reshape %1222 : (tensor<f64>) -> tensor<1xf64>
    %1224 = stablehlo.multiply %1174, %1206 : tensor<f64>
    %1225 = stablehlo.multiply %1198, %1212 : tensor<f64>
    %1226 = stablehlo.add %1224, %1225 : tensor<f64>
    %1227 = stablehlo.multiply %1204, %1195 : tensor<f64>
    %1228 = stablehlo.subtract %1226, %1227 : tensor<f64>
    %1229 = stablehlo.multiply %1210, %1200 : tensor<f64>
    %1230 = stablehlo.add %1228, %1229 : tensor<f64>
    %1231 = stablehlo.reshape %1230 : (tensor<f64>) -> tensor<1xf64>
    %1232 = stablehlo.multiply %1174, %1200 : tensor<f64>
    %1233 = stablehlo.multiply %1198, %1195 : tensor<f64>
    %1234 = stablehlo.subtract %1232, %1233 : tensor<f64>
    %1235 = stablehlo.multiply %1204, %1212 : tensor<f64>
    %1236 = stablehlo.subtract %1234, %1235 : tensor<f64>
    %1237 = stablehlo.multiply %1210, %1206 : tensor<f64>
    %1238 = stablehlo.subtract %1236, %1237 : tensor<f64>
    %1239 = stablehlo.reshape %1238 : (tensor<f64>) -> tensor<1xf64>
    %1240 = stablehlo.concatenate %1215, %1223, %1231, %1239, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1241 = stablehlo.slice %1240 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1242 = stablehlo.reshape %1241 : (tensor<1xf64>) -> tensor<f64>
    %1243 = stablehlo.reshape %1242 : (tensor<f64>) -> tensor<1xf64>
    %1244 = stablehlo.slice %1240 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1245 = stablehlo.reshape %1244 : (tensor<1xf64>) -> tensor<f64>
    %1246 = stablehlo.reshape %1245 : (tensor<f64>) -> tensor<1xf64>
    %1247 = stablehlo.slice %1240 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1248 = stablehlo.reshape %1247 : (tensor<1xf64>) -> tensor<f64>
    %1249 = stablehlo.reshape %1248 : (tensor<f64>) -> tensor<1xf64>
    %1250 = stablehlo.concatenate %1243, %1246, %1249, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1251 = stablehlo.concatenate %1120, %1250, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1252 = stablehlo.slice %arg3 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1253 = stablehlo.reshape %arg8 : (tensor<f64>) -> tensor<f64>
    %cst_15 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %1254 = stablehlo.multiply %cst_15, %1253 : tensor<f64>
    %1255 = stablehlo.broadcast_in_dim %1254, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1256 = stablehlo.multiply %1255, %1 : tensor<6xf64>
    %1257 = stablehlo.slice %1256 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_16 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1258 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1259 = stablehlo.divide %1257, %1258 : tensor<3xf64>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1260 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1261 = stablehlo.concatenate %1259, %1260, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1262 = stablehlo.slice %1261 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1263 = stablehlo.reshape %1262 : (tensor<1xf64>) -> tensor<f64>
    %1264 = stablehlo.slice %1252 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1265 = stablehlo.reshape %1264 : (tensor<1xf64>) -> tensor<f64>
    %1266 = stablehlo.multiply %1263, %1265 : tensor<f64>
    %1267 = stablehlo.slice %1261 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1268 = stablehlo.reshape %1267 : (tensor<1xf64>) -> tensor<f64>
    %1269 = stablehlo.slice %1252 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1270 = stablehlo.reshape %1269 : (tensor<1xf64>) -> tensor<f64>
    %1271 = stablehlo.multiply %1268, %1270 : tensor<f64>
    %1272 = stablehlo.add %1266, %1271 : tensor<f64>
    %1273 = stablehlo.slice %1261 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1274 = stablehlo.reshape %1273 : (tensor<1xf64>) -> tensor<f64>
    %1275 = stablehlo.slice %1252 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1276 = stablehlo.reshape %1275 : (tensor<1xf64>) -> tensor<f64>
    %1277 = stablehlo.multiply %1274, %1276 : tensor<f64>
    %1278 = stablehlo.add %1272, %1277 : tensor<f64>
    %1279 = stablehlo.slice %1261 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1280 = stablehlo.reshape %1279 : (tensor<1xf64>) -> tensor<f64>
    %1281 = stablehlo.slice %1252 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1282 = stablehlo.reshape %1281 : (tensor<1xf64>) -> tensor<f64>
    %1283 = stablehlo.multiply %1280, %1282 : tensor<f64>
    %1284 = stablehlo.subtract %1278, %1283 : tensor<f64>
    %1285 = stablehlo.reshape %1284 : (tensor<f64>) -> tensor<1xf64>
    %1286 = stablehlo.multiply %1263, %1282 : tensor<f64>
    %1287 = stablehlo.multiply %1268, %1276 : tensor<f64>
    %1288 = stablehlo.subtract %1286, %1287 : tensor<f64>
    %1289 = stablehlo.multiply %1274, %1270 : tensor<f64>
    %1290 = stablehlo.add %1288, %1289 : tensor<f64>
    %1291 = stablehlo.multiply %1280, %1265 : tensor<f64>
    %1292 = stablehlo.add %1290, %1291 : tensor<f64>
    %1293 = stablehlo.reshape %1292 : (tensor<f64>) -> tensor<1xf64>
    %1294 = stablehlo.multiply %1263, %1276 : tensor<f64>
    %1295 = stablehlo.multiply %1268, %1282 : tensor<f64>
    %1296 = stablehlo.add %1294, %1295 : tensor<f64>
    %1297 = stablehlo.multiply %1274, %1265 : tensor<f64>
    %1298 = stablehlo.subtract %1296, %1297 : tensor<f64>
    %1299 = stablehlo.multiply %1280, %1270 : tensor<f64>
    %1300 = stablehlo.add %1298, %1299 : tensor<f64>
    %1301 = stablehlo.reshape %1300 : (tensor<f64>) -> tensor<1xf64>
    %1302 = stablehlo.multiply %1263, %1270 : tensor<f64>
    %1303 = stablehlo.multiply %1268, %1265 : tensor<f64>
    %1304 = stablehlo.subtract %1302, %1303 : tensor<f64>
    %1305 = stablehlo.multiply %1274, %1282 : tensor<f64>
    %1306 = stablehlo.subtract %1304, %1305 : tensor<f64>
    %1307 = stablehlo.multiply %1280, %1276 : tensor<f64>
    %1308 = stablehlo.subtract %1306, %1307 : tensor<f64>
    %1309 = stablehlo.reshape %1308 : (tensor<f64>) -> tensor<1xf64>
    %1310 = stablehlo.concatenate %1285, %1293, %1301, %1309, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1311 = stablehlo.add %1252, %1310 : tensor<4xf64>
    %1312 = stablehlo.dot_general %1311, %1311, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1313 = stablehlo.sqrt %1312 : tensor<f64>
    %1314 = stablehlo.broadcast_in_dim %1313, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1315 = stablehlo.divide %1311, %1314 : tensor<4xf64>
    %1316 = stablehlo.slice %arg3 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1317 = stablehlo.slice %1256 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1318 = stablehlo.add %1316, %1317 : tensor<3xf64>
    %1319 = stablehlo.concatenate %1315, %1318, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %1320 = stablehlo.broadcast_in_dim %1254, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1321 = stablehlo.multiply %1320, %1251 : tensor<6xf64>
    %1322 = stablehlo.add %1, %1321 : tensor<6xf64>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1323 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1324 = call @inner_55(%1323, %arg6) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %1325 = call @inner_57(%0, %1322, %1324) : (tensor<3xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1326 = stablehlo.slice %1319 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1327 = stablehlo.slice %1326 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1328 = stablehlo.reshape %1327 : (tensor<1xf64>) -> tensor<f64>
    %1329 = stablehlo.slice %1326 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1330 = stablehlo.reshape %1329 : (tensor<1xf64>) -> tensor<f64>
    %1331 = stablehlo.negate %1330 : tensor<f64>
    %1332 = stablehlo.reshape %1331 : (tensor<f64>) -> tensor<1xf64>
    %1333 = stablehlo.slice %1326 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1334 = stablehlo.reshape %1333 : (tensor<1xf64>) -> tensor<f64>
    %1335 = stablehlo.negate %1334 : tensor<f64>
    %1336 = stablehlo.reshape %1335 : (tensor<f64>) -> tensor<1xf64>
    %1337 = stablehlo.slice %1326 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1338 = stablehlo.reshape %1337 : (tensor<1xf64>) -> tensor<f64>
    %1339 = stablehlo.negate %1338 : tensor<f64>
    %1340 = stablehlo.reshape %1339 : (tensor<f64>) -> tensor<1xf64>
    %1341 = stablehlo.slice %1326 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1342 = stablehlo.reshape %1341 : (tensor<1xf64>) -> tensor<f64>
    %1343 = stablehlo.reshape %1342 : (tensor<f64>) -> tensor<1xf64>
    %1344 = stablehlo.concatenate %1332, %1336, %1340, %1343, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1345 = stablehlo.dot_general %1326, %1326, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1346 = stablehlo.broadcast_in_dim %1345, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1347 = stablehlo.divide %1344, %1346 : tensor<4xf64>
    %1348 = stablehlo.slice %1347 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1349 = stablehlo.reshape %1348 : (tensor<1xf64>) -> tensor<f64>
    %1350 = stablehlo.slice %1325 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1351 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1352 = stablehlo.concatenate %1350, %1351, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1353 = stablehlo.slice %1352 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1354 = stablehlo.reshape %1353 : (tensor<1xf64>) -> tensor<f64>
    %1355 = stablehlo.multiply %1349, %1354 : tensor<f64>
    %1356 = stablehlo.slice %1347 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1357 = stablehlo.reshape %1356 : (tensor<1xf64>) -> tensor<f64>
    %1358 = stablehlo.slice %1352 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1359 = stablehlo.reshape %1358 : (tensor<1xf64>) -> tensor<f64>
    %1360 = stablehlo.multiply %1357, %1359 : tensor<f64>
    %1361 = stablehlo.add %1355, %1360 : tensor<f64>
    %1362 = stablehlo.slice %1347 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1363 = stablehlo.reshape %1362 : (tensor<1xf64>) -> tensor<f64>
    %1364 = stablehlo.slice %1352 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1365 = stablehlo.reshape %1364 : (tensor<1xf64>) -> tensor<f64>
    %1366 = stablehlo.multiply %1363, %1365 : tensor<f64>
    %1367 = stablehlo.add %1361, %1366 : tensor<f64>
    %1368 = stablehlo.slice %1347 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1369 = stablehlo.reshape %1368 : (tensor<1xf64>) -> tensor<f64>
    %1370 = stablehlo.slice %1352 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1371 = stablehlo.reshape %1370 : (tensor<1xf64>) -> tensor<f64>
    %1372 = stablehlo.multiply %1369, %1371 : tensor<f64>
    %1373 = stablehlo.subtract %1367, %1372 : tensor<f64>
    %1374 = stablehlo.reshape %1373 : (tensor<f64>) -> tensor<1xf64>
    %1375 = stablehlo.multiply %1349, %1371 : tensor<f64>
    %1376 = stablehlo.multiply %1357, %1365 : tensor<f64>
    %1377 = stablehlo.subtract %1375, %1376 : tensor<f64>
    %1378 = stablehlo.multiply %1363, %1359 : tensor<f64>
    %1379 = stablehlo.add %1377, %1378 : tensor<f64>
    %1380 = stablehlo.multiply %1369, %1354 : tensor<f64>
    %1381 = stablehlo.add %1379, %1380 : tensor<f64>
    %1382 = stablehlo.reshape %1381 : (tensor<f64>) -> tensor<1xf64>
    %1383 = stablehlo.multiply %1349, %1365 : tensor<f64>
    %1384 = stablehlo.multiply %1357, %1371 : tensor<f64>
    %1385 = stablehlo.add %1383, %1384 : tensor<f64>
    %1386 = stablehlo.multiply %1363, %1354 : tensor<f64>
    %1387 = stablehlo.subtract %1385, %1386 : tensor<f64>
    %1388 = stablehlo.multiply %1369, %1359 : tensor<f64>
    %1389 = stablehlo.add %1387, %1388 : tensor<f64>
    %1390 = stablehlo.reshape %1389 : (tensor<f64>) -> tensor<1xf64>
    %1391 = stablehlo.multiply %1349, %1359 : tensor<f64>
    %1392 = stablehlo.multiply %1357, %1354 : tensor<f64>
    %1393 = stablehlo.subtract %1391, %1392 : tensor<f64>
    %1394 = stablehlo.multiply %1363, %1371 : tensor<f64>
    %1395 = stablehlo.subtract %1393, %1394 : tensor<f64>
    %1396 = stablehlo.multiply %1369, %1365 : tensor<f64>
    %1397 = stablehlo.subtract %1395, %1396 : tensor<f64>
    %1398 = stablehlo.reshape %1397 : (tensor<f64>) -> tensor<1xf64>
    %1399 = stablehlo.concatenate %1374, %1382, %1390, %1398, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1400 = stablehlo.slice %1399 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1401 = stablehlo.reshape %1400 : (tensor<1xf64>) -> tensor<f64>
    %1402 = stablehlo.slice %1347 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1403 = stablehlo.reshape %1402 : (tensor<1xf64>) -> tensor<f64>
    %1404 = stablehlo.negate %1403 : tensor<f64>
    %1405 = stablehlo.reshape %1404 : (tensor<f64>) -> tensor<1xf64>
    %1406 = stablehlo.slice %1347 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1407 = stablehlo.reshape %1406 : (tensor<1xf64>) -> tensor<f64>
    %1408 = stablehlo.negate %1407 : tensor<f64>
    %1409 = stablehlo.reshape %1408 : (tensor<f64>) -> tensor<1xf64>
    %1410 = stablehlo.slice %1347 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1411 = stablehlo.reshape %1410 : (tensor<1xf64>) -> tensor<f64>
    %1412 = stablehlo.negate %1411 : tensor<f64>
    %1413 = stablehlo.reshape %1412 : (tensor<f64>) -> tensor<1xf64>
    %1414 = stablehlo.slice %1347 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1415 = stablehlo.reshape %1414 : (tensor<1xf64>) -> tensor<f64>
    %1416 = stablehlo.reshape %1415 : (tensor<f64>) -> tensor<1xf64>
    %1417 = stablehlo.concatenate %1405, %1409, %1413, %1416, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1418 = stablehlo.dot_general %1347, %1347, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1419 = stablehlo.broadcast_in_dim %1418, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1420 = stablehlo.divide %1417, %1419 : tensor<4xf64>
    %1421 = stablehlo.slice %1420 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1422 = stablehlo.reshape %1421 : (tensor<1xf64>) -> tensor<f64>
    %1423 = stablehlo.multiply %1401, %1422 : tensor<f64>
    %1424 = stablehlo.slice %1399 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1425 = stablehlo.reshape %1424 : (tensor<1xf64>) -> tensor<f64>
    %1426 = stablehlo.slice %1420 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1427 = stablehlo.reshape %1426 : (tensor<1xf64>) -> tensor<f64>
    %1428 = stablehlo.multiply %1425, %1427 : tensor<f64>
    %1429 = stablehlo.add %1423, %1428 : tensor<f64>
    %1430 = stablehlo.slice %1399 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1431 = stablehlo.reshape %1430 : (tensor<1xf64>) -> tensor<f64>
    %1432 = stablehlo.slice %1420 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1433 = stablehlo.reshape %1432 : (tensor<1xf64>) -> tensor<f64>
    %1434 = stablehlo.multiply %1431, %1433 : tensor<f64>
    %1435 = stablehlo.add %1429, %1434 : tensor<f64>
    %1436 = stablehlo.slice %1399 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1437 = stablehlo.reshape %1436 : (tensor<1xf64>) -> tensor<f64>
    %1438 = stablehlo.slice %1420 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1439 = stablehlo.reshape %1438 : (tensor<1xf64>) -> tensor<f64>
    %1440 = stablehlo.multiply %1437, %1439 : tensor<f64>
    %1441 = stablehlo.subtract %1435, %1440 : tensor<f64>
    %1442 = stablehlo.reshape %1441 : (tensor<f64>) -> tensor<1xf64>
    %1443 = stablehlo.multiply %1401, %1439 : tensor<f64>
    %1444 = stablehlo.multiply %1425, %1433 : tensor<f64>
    %1445 = stablehlo.subtract %1443, %1444 : tensor<f64>
    %1446 = stablehlo.multiply %1431, %1427 : tensor<f64>
    %1447 = stablehlo.add %1445, %1446 : tensor<f64>
    %1448 = stablehlo.multiply %1437, %1422 : tensor<f64>
    %1449 = stablehlo.add %1447, %1448 : tensor<f64>
    %1450 = stablehlo.reshape %1449 : (tensor<f64>) -> tensor<1xf64>
    %1451 = stablehlo.multiply %1401, %1433 : tensor<f64>
    %1452 = stablehlo.multiply %1425, %1439 : tensor<f64>
    %1453 = stablehlo.add %1451, %1452 : tensor<f64>
    %1454 = stablehlo.multiply %1431, %1422 : tensor<f64>
    %1455 = stablehlo.subtract %1453, %1454 : tensor<f64>
    %1456 = stablehlo.multiply %1437, %1427 : tensor<f64>
    %1457 = stablehlo.add %1455, %1456 : tensor<f64>
    %1458 = stablehlo.reshape %1457 : (tensor<f64>) -> tensor<1xf64>
    %1459 = stablehlo.multiply %1401, %1427 : tensor<f64>
    %1460 = stablehlo.multiply %1425, %1422 : tensor<f64>
    %1461 = stablehlo.subtract %1459, %1460 : tensor<f64>
    %1462 = stablehlo.multiply %1431, %1439 : tensor<f64>
    %1463 = stablehlo.subtract %1461, %1462 : tensor<f64>
    %1464 = stablehlo.multiply %1437, %1433 : tensor<f64>
    %1465 = stablehlo.subtract %1463, %1464 : tensor<f64>
    %1466 = stablehlo.reshape %1465 : (tensor<f64>) -> tensor<1xf64>
    %1467 = stablehlo.concatenate %1442, %1450, %1458, %1466, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1468 = stablehlo.slice %1467 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1469 = stablehlo.reshape %1468 : (tensor<1xf64>) -> tensor<f64>
    %1470 = stablehlo.reshape %1469 : (tensor<f64>) -> tensor<1xf64>
    %1471 = stablehlo.slice %1467 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1472 = stablehlo.reshape %1471 : (tensor<1xf64>) -> tensor<f64>
    %1473 = stablehlo.reshape %1472 : (tensor<f64>) -> tensor<1xf64>
    %1474 = stablehlo.slice %1467 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1475 = stablehlo.reshape %1474 : (tensor<1xf64>) -> tensor<f64>
    %1476 = stablehlo.reshape %1475 : (tensor<f64>) -> tensor<1xf64>
    %1477 = stablehlo.concatenate %1470, %1473, %1476, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1478 = stablehlo.slice %1347 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1479 = stablehlo.reshape %1478 : (tensor<1xf64>) -> tensor<f64>
    %1480 = stablehlo.slice %1325 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1481 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1482 = stablehlo.concatenate %1480, %1481, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1483 = stablehlo.slice %1482 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1484 = stablehlo.reshape %1483 : (tensor<1xf64>) -> tensor<f64>
    %1485 = stablehlo.multiply %1479, %1484 : tensor<f64>
    %1486 = stablehlo.slice %1347 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1487 = stablehlo.reshape %1486 : (tensor<1xf64>) -> tensor<f64>
    %1488 = stablehlo.slice %1482 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1489 = stablehlo.reshape %1488 : (tensor<1xf64>) -> tensor<f64>
    %1490 = stablehlo.multiply %1487, %1489 : tensor<f64>
    %1491 = stablehlo.add %1485, %1490 : tensor<f64>
    %1492 = stablehlo.slice %1347 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1493 = stablehlo.reshape %1492 : (tensor<1xf64>) -> tensor<f64>
    %1494 = stablehlo.slice %1482 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1495 = stablehlo.reshape %1494 : (tensor<1xf64>) -> tensor<f64>
    %1496 = stablehlo.multiply %1493, %1495 : tensor<f64>
    %1497 = stablehlo.add %1491, %1496 : tensor<f64>
    %1498 = stablehlo.slice %1347 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1499 = stablehlo.reshape %1498 : (tensor<1xf64>) -> tensor<f64>
    %1500 = stablehlo.slice %1482 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1501 = stablehlo.reshape %1500 : (tensor<1xf64>) -> tensor<f64>
    %1502 = stablehlo.multiply %1499, %1501 : tensor<f64>
    %1503 = stablehlo.subtract %1497, %1502 : tensor<f64>
    %1504 = stablehlo.reshape %1503 : (tensor<f64>) -> tensor<1xf64>
    %1505 = stablehlo.multiply %1479, %1501 : tensor<f64>
    %1506 = stablehlo.multiply %1487, %1495 : tensor<f64>
    %1507 = stablehlo.subtract %1505, %1506 : tensor<f64>
    %1508 = stablehlo.multiply %1493, %1489 : tensor<f64>
    %1509 = stablehlo.add %1507, %1508 : tensor<f64>
    %1510 = stablehlo.multiply %1499, %1484 : tensor<f64>
    %1511 = stablehlo.add %1509, %1510 : tensor<f64>
    %1512 = stablehlo.reshape %1511 : (tensor<f64>) -> tensor<1xf64>
    %1513 = stablehlo.multiply %1479, %1495 : tensor<f64>
    %1514 = stablehlo.multiply %1487, %1501 : tensor<f64>
    %1515 = stablehlo.add %1513, %1514 : tensor<f64>
    %1516 = stablehlo.multiply %1493, %1484 : tensor<f64>
    %1517 = stablehlo.subtract %1515, %1516 : tensor<f64>
    %1518 = stablehlo.multiply %1499, %1489 : tensor<f64>
    %1519 = stablehlo.add %1517, %1518 : tensor<f64>
    %1520 = stablehlo.reshape %1519 : (tensor<f64>) -> tensor<1xf64>
    %1521 = stablehlo.multiply %1479, %1489 : tensor<f64>
    %1522 = stablehlo.multiply %1487, %1484 : tensor<f64>
    %1523 = stablehlo.subtract %1521, %1522 : tensor<f64>
    %1524 = stablehlo.multiply %1493, %1501 : tensor<f64>
    %1525 = stablehlo.subtract %1523, %1524 : tensor<f64>
    %1526 = stablehlo.multiply %1499, %1495 : tensor<f64>
    %1527 = stablehlo.subtract %1525, %1526 : tensor<f64>
    %1528 = stablehlo.reshape %1527 : (tensor<f64>) -> tensor<1xf64>
    %1529 = stablehlo.concatenate %1504, %1512, %1520, %1528, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1530 = stablehlo.slice %1529 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1531 = stablehlo.reshape %1530 : (tensor<1xf64>) -> tensor<f64>
    %1532 = stablehlo.slice %1347 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1533 = stablehlo.reshape %1532 : (tensor<1xf64>) -> tensor<f64>
    %1534 = stablehlo.negate %1533 : tensor<f64>
    %1535 = stablehlo.reshape %1534 : (tensor<f64>) -> tensor<1xf64>
    %1536 = stablehlo.slice %1347 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1537 = stablehlo.reshape %1536 : (tensor<1xf64>) -> tensor<f64>
    %1538 = stablehlo.negate %1537 : tensor<f64>
    %1539 = stablehlo.reshape %1538 : (tensor<f64>) -> tensor<1xf64>
    %1540 = stablehlo.slice %1347 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1541 = stablehlo.reshape %1540 : (tensor<1xf64>) -> tensor<f64>
    %1542 = stablehlo.negate %1541 : tensor<f64>
    %1543 = stablehlo.reshape %1542 : (tensor<f64>) -> tensor<1xf64>
    %1544 = stablehlo.slice %1347 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1545 = stablehlo.reshape %1544 : (tensor<1xf64>) -> tensor<f64>
    %1546 = stablehlo.reshape %1545 : (tensor<f64>) -> tensor<1xf64>
    %1547 = stablehlo.concatenate %1535, %1539, %1543, %1546, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1548 = stablehlo.dot_general %1347, %1347, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1549 = stablehlo.broadcast_in_dim %1548, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1550 = stablehlo.divide %1547, %1549 : tensor<4xf64>
    %1551 = stablehlo.slice %1550 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1552 = stablehlo.reshape %1551 : (tensor<1xf64>) -> tensor<f64>
    %1553 = stablehlo.multiply %1531, %1552 : tensor<f64>
    %1554 = stablehlo.slice %1529 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1555 = stablehlo.reshape %1554 : (tensor<1xf64>) -> tensor<f64>
    %1556 = stablehlo.slice %1550 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1557 = stablehlo.reshape %1556 : (tensor<1xf64>) -> tensor<f64>
    %1558 = stablehlo.multiply %1555, %1557 : tensor<f64>
    %1559 = stablehlo.add %1553, %1558 : tensor<f64>
    %1560 = stablehlo.slice %1529 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1561 = stablehlo.reshape %1560 : (tensor<1xf64>) -> tensor<f64>
    %1562 = stablehlo.slice %1550 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1563 = stablehlo.reshape %1562 : (tensor<1xf64>) -> tensor<f64>
    %1564 = stablehlo.multiply %1561, %1563 : tensor<f64>
    %1565 = stablehlo.add %1559, %1564 : tensor<f64>
    %1566 = stablehlo.slice %1529 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1567 = stablehlo.reshape %1566 : (tensor<1xf64>) -> tensor<f64>
    %1568 = stablehlo.slice %1550 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1569 = stablehlo.reshape %1568 : (tensor<1xf64>) -> tensor<f64>
    %1570 = stablehlo.multiply %1567, %1569 : tensor<f64>
    %1571 = stablehlo.subtract %1565, %1570 : tensor<f64>
    %1572 = stablehlo.reshape %1571 : (tensor<f64>) -> tensor<1xf64>
    %1573 = stablehlo.multiply %1531, %1569 : tensor<f64>
    %1574 = stablehlo.multiply %1555, %1563 : tensor<f64>
    %1575 = stablehlo.subtract %1573, %1574 : tensor<f64>
    %1576 = stablehlo.multiply %1561, %1557 : tensor<f64>
    %1577 = stablehlo.add %1575, %1576 : tensor<f64>
    %1578 = stablehlo.multiply %1567, %1552 : tensor<f64>
    %1579 = stablehlo.add %1577, %1578 : tensor<f64>
    %1580 = stablehlo.reshape %1579 : (tensor<f64>) -> tensor<1xf64>
    %1581 = stablehlo.multiply %1531, %1563 : tensor<f64>
    %1582 = stablehlo.multiply %1555, %1569 : tensor<f64>
    %1583 = stablehlo.add %1581, %1582 : tensor<f64>
    %1584 = stablehlo.multiply %1561, %1552 : tensor<f64>
    %1585 = stablehlo.subtract %1583, %1584 : tensor<f64>
    %1586 = stablehlo.multiply %1567, %1557 : tensor<f64>
    %1587 = stablehlo.add %1585, %1586 : tensor<f64>
    %1588 = stablehlo.reshape %1587 : (tensor<f64>) -> tensor<1xf64>
    %1589 = stablehlo.multiply %1531, %1557 : tensor<f64>
    %1590 = stablehlo.multiply %1555, %1552 : tensor<f64>
    %1591 = stablehlo.subtract %1589, %1590 : tensor<f64>
    %1592 = stablehlo.multiply %1561, %1569 : tensor<f64>
    %1593 = stablehlo.subtract %1591, %1592 : tensor<f64>
    %1594 = stablehlo.multiply %1567, %1563 : tensor<f64>
    %1595 = stablehlo.subtract %1593, %1594 : tensor<f64>
    %1596 = stablehlo.reshape %1595 : (tensor<f64>) -> tensor<1xf64>
    %1597 = stablehlo.concatenate %1572, %1580, %1588, %1596, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1598 = stablehlo.slice %1597 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1599 = stablehlo.reshape %1598 : (tensor<1xf64>) -> tensor<f64>
    %1600 = stablehlo.reshape %1599 : (tensor<f64>) -> tensor<1xf64>
    %1601 = stablehlo.slice %1597 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1602 = stablehlo.reshape %1601 : (tensor<1xf64>) -> tensor<f64>
    %1603 = stablehlo.reshape %1602 : (tensor<f64>) -> tensor<1xf64>
    %1604 = stablehlo.slice %1597 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1605 = stablehlo.reshape %1604 : (tensor<1xf64>) -> tensor<f64>
    %1606 = stablehlo.reshape %1605 : (tensor<f64>) -> tensor<1xf64>
    %1607 = stablehlo.concatenate %1600, %1603, %1606, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1608 = stablehlo.concatenate %1477, %1607, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1609 = stablehlo.slice %1608 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1610 = stablehlo.slice %arg6 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %1611 = stablehlo.divide %1609, %1610 : tensor<3xf64>
    %1612 = stablehlo.slice %1608 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1613 = stablehlo.slice %arg6 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1614 = stablehlo.reshape %1613 : (tensor<1xf64>) -> tensor<f64>
    %1615 = stablehlo.broadcast_in_dim %1614, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1616 = stablehlo.divide %1612, %1615 : tensor<3xf64>
    %1617 = stablehlo.concatenate %1611, %1616, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1618 = stablehlo.slice %1617 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1619 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1620 = stablehlo.concatenate %1618, %1619, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1621 = stablehlo.slice %1620 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1622 = stablehlo.reshape %1621 : (tensor<1xf64>) -> tensor<f64>
    %1623 = stablehlo.multiply %1328, %1622 : tensor<f64>
    %1624 = stablehlo.slice %1326 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1625 = stablehlo.reshape %1624 : (tensor<1xf64>) -> tensor<f64>
    %1626 = stablehlo.slice %1620 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1627 = stablehlo.reshape %1626 : (tensor<1xf64>) -> tensor<f64>
    %1628 = stablehlo.multiply %1625, %1627 : tensor<f64>
    %1629 = stablehlo.add %1623, %1628 : tensor<f64>
    %1630 = stablehlo.slice %1326 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1631 = stablehlo.reshape %1630 : (tensor<1xf64>) -> tensor<f64>
    %1632 = stablehlo.slice %1620 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1633 = stablehlo.reshape %1632 : (tensor<1xf64>) -> tensor<f64>
    %1634 = stablehlo.multiply %1631, %1633 : tensor<f64>
    %1635 = stablehlo.add %1629, %1634 : tensor<f64>
    %1636 = stablehlo.slice %1326 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1637 = stablehlo.reshape %1636 : (tensor<1xf64>) -> tensor<f64>
    %1638 = stablehlo.slice %1620 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1639 = stablehlo.reshape %1638 : (tensor<1xf64>) -> tensor<f64>
    %1640 = stablehlo.multiply %1637, %1639 : tensor<f64>
    %1641 = stablehlo.subtract %1635, %1640 : tensor<f64>
    %1642 = stablehlo.reshape %1641 : (tensor<f64>) -> tensor<1xf64>
    %1643 = stablehlo.multiply %1328, %1639 : tensor<f64>
    %1644 = stablehlo.multiply %1625, %1633 : tensor<f64>
    %1645 = stablehlo.subtract %1643, %1644 : tensor<f64>
    %1646 = stablehlo.multiply %1631, %1627 : tensor<f64>
    %1647 = stablehlo.add %1645, %1646 : tensor<f64>
    %1648 = stablehlo.multiply %1637, %1622 : tensor<f64>
    %1649 = stablehlo.add %1647, %1648 : tensor<f64>
    %1650 = stablehlo.reshape %1649 : (tensor<f64>) -> tensor<1xf64>
    %1651 = stablehlo.multiply %1328, %1633 : tensor<f64>
    %1652 = stablehlo.multiply %1625, %1639 : tensor<f64>
    %1653 = stablehlo.add %1651, %1652 : tensor<f64>
    %1654 = stablehlo.multiply %1631, %1622 : tensor<f64>
    %1655 = stablehlo.subtract %1653, %1654 : tensor<f64>
    %1656 = stablehlo.multiply %1637, %1627 : tensor<f64>
    %1657 = stablehlo.add %1655, %1656 : tensor<f64>
    %1658 = stablehlo.reshape %1657 : (tensor<f64>) -> tensor<1xf64>
    %1659 = stablehlo.multiply %1328, %1627 : tensor<f64>
    %1660 = stablehlo.multiply %1625, %1622 : tensor<f64>
    %1661 = stablehlo.subtract %1659, %1660 : tensor<f64>
    %1662 = stablehlo.multiply %1631, %1639 : tensor<f64>
    %1663 = stablehlo.subtract %1661, %1662 : tensor<f64>
    %1664 = stablehlo.multiply %1637, %1633 : tensor<f64>
    %1665 = stablehlo.subtract %1663, %1664 : tensor<f64>
    %1666 = stablehlo.reshape %1665 : (tensor<f64>) -> tensor<1xf64>
    %1667 = stablehlo.concatenate %1642, %1650, %1658, %1666, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1668 = stablehlo.slice %1667 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1669 = stablehlo.reshape %1668 : (tensor<1xf64>) -> tensor<f64>
    %1670 = stablehlo.slice %1326 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1671 = stablehlo.reshape %1670 : (tensor<1xf64>) -> tensor<f64>
    %1672 = stablehlo.negate %1671 : tensor<f64>
    %1673 = stablehlo.reshape %1672 : (tensor<f64>) -> tensor<1xf64>
    %1674 = stablehlo.slice %1326 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1675 = stablehlo.reshape %1674 : (tensor<1xf64>) -> tensor<f64>
    %1676 = stablehlo.negate %1675 : tensor<f64>
    %1677 = stablehlo.reshape %1676 : (tensor<f64>) -> tensor<1xf64>
    %1678 = stablehlo.slice %1326 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1679 = stablehlo.reshape %1678 : (tensor<1xf64>) -> tensor<f64>
    %1680 = stablehlo.negate %1679 : tensor<f64>
    %1681 = stablehlo.reshape %1680 : (tensor<f64>) -> tensor<1xf64>
    %1682 = stablehlo.slice %1326 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1683 = stablehlo.reshape %1682 : (tensor<1xf64>) -> tensor<f64>
    %1684 = stablehlo.reshape %1683 : (tensor<f64>) -> tensor<1xf64>
    %1685 = stablehlo.concatenate %1673, %1677, %1681, %1684, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1686 = stablehlo.dot_general %1326, %1326, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1687 = stablehlo.broadcast_in_dim %1686, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1688 = stablehlo.divide %1685, %1687 : tensor<4xf64>
    %1689 = stablehlo.slice %1688 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1690 = stablehlo.reshape %1689 : (tensor<1xf64>) -> tensor<f64>
    %1691 = stablehlo.multiply %1669, %1690 : tensor<f64>
    %1692 = stablehlo.slice %1667 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1693 = stablehlo.reshape %1692 : (tensor<1xf64>) -> tensor<f64>
    %1694 = stablehlo.slice %1688 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1695 = stablehlo.reshape %1694 : (tensor<1xf64>) -> tensor<f64>
    %1696 = stablehlo.multiply %1693, %1695 : tensor<f64>
    %1697 = stablehlo.add %1691, %1696 : tensor<f64>
    %1698 = stablehlo.slice %1667 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1699 = stablehlo.reshape %1698 : (tensor<1xf64>) -> tensor<f64>
    %1700 = stablehlo.slice %1688 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1701 = stablehlo.reshape %1700 : (tensor<1xf64>) -> tensor<f64>
    %1702 = stablehlo.multiply %1699, %1701 : tensor<f64>
    %1703 = stablehlo.add %1697, %1702 : tensor<f64>
    %1704 = stablehlo.slice %1667 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1705 = stablehlo.reshape %1704 : (tensor<1xf64>) -> tensor<f64>
    %1706 = stablehlo.slice %1688 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1707 = stablehlo.reshape %1706 : (tensor<1xf64>) -> tensor<f64>
    %1708 = stablehlo.multiply %1705, %1707 : tensor<f64>
    %1709 = stablehlo.subtract %1703, %1708 : tensor<f64>
    %1710 = stablehlo.reshape %1709 : (tensor<f64>) -> tensor<1xf64>
    %1711 = stablehlo.multiply %1669, %1707 : tensor<f64>
    %1712 = stablehlo.multiply %1693, %1701 : tensor<f64>
    %1713 = stablehlo.subtract %1711, %1712 : tensor<f64>
    %1714 = stablehlo.multiply %1699, %1695 : tensor<f64>
    %1715 = stablehlo.add %1713, %1714 : tensor<f64>
    %1716 = stablehlo.multiply %1705, %1690 : tensor<f64>
    %1717 = stablehlo.add %1715, %1716 : tensor<f64>
    %1718 = stablehlo.reshape %1717 : (tensor<f64>) -> tensor<1xf64>
    %1719 = stablehlo.multiply %1669, %1701 : tensor<f64>
    %1720 = stablehlo.multiply %1693, %1707 : tensor<f64>
    %1721 = stablehlo.add %1719, %1720 : tensor<f64>
    %1722 = stablehlo.multiply %1699, %1690 : tensor<f64>
    %1723 = stablehlo.subtract %1721, %1722 : tensor<f64>
    %1724 = stablehlo.multiply %1705, %1695 : tensor<f64>
    %1725 = stablehlo.add %1723, %1724 : tensor<f64>
    %1726 = stablehlo.reshape %1725 : (tensor<f64>) -> tensor<1xf64>
    %1727 = stablehlo.multiply %1669, %1695 : tensor<f64>
    %1728 = stablehlo.multiply %1693, %1690 : tensor<f64>
    %1729 = stablehlo.subtract %1727, %1728 : tensor<f64>
    %1730 = stablehlo.multiply %1699, %1707 : tensor<f64>
    %1731 = stablehlo.subtract %1729, %1730 : tensor<f64>
    %1732 = stablehlo.multiply %1705, %1701 : tensor<f64>
    %1733 = stablehlo.subtract %1731, %1732 : tensor<f64>
    %1734 = stablehlo.reshape %1733 : (tensor<f64>) -> tensor<1xf64>
    %1735 = stablehlo.concatenate %1710, %1718, %1726, %1734, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1736 = stablehlo.slice %1735 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1737 = stablehlo.reshape %1736 : (tensor<1xf64>) -> tensor<f64>
    %1738 = stablehlo.reshape %1737 : (tensor<f64>) -> tensor<1xf64>
    %1739 = stablehlo.slice %1735 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1740 = stablehlo.reshape %1739 : (tensor<1xf64>) -> tensor<f64>
    %1741 = stablehlo.reshape %1740 : (tensor<f64>) -> tensor<1xf64>
    %1742 = stablehlo.slice %1735 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1743 = stablehlo.reshape %1742 : (tensor<1xf64>) -> tensor<f64>
    %1744 = stablehlo.reshape %1743 : (tensor<f64>) -> tensor<1xf64>
    %1745 = stablehlo.concatenate %1738, %1741, %1744, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1746 = stablehlo.slice %1326 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1747 = stablehlo.reshape %1746 : (tensor<1xf64>) -> tensor<f64>
    %1748 = stablehlo.slice %1617 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1749 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1750 = stablehlo.concatenate %1748, %1749, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1751 = stablehlo.slice %1750 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1752 = stablehlo.reshape %1751 : (tensor<1xf64>) -> tensor<f64>
    %1753 = stablehlo.multiply %1747, %1752 : tensor<f64>
    %1754 = stablehlo.slice %1326 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1755 = stablehlo.reshape %1754 : (tensor<1xf64>) -> tensor<f64>
    %1756 = stablehlo.slice %1750 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1757 = stablehlo.reshape %1756 : (tensor<1xf64>) -> tensor<f64>
    %1758 = stablehlo.multiply %1755, %1757 : tensor<f64>
    %1759 = stablehlo.add %1753, %1758 : tensor<f64>
    %1760 = stablehlo.slice %1326 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1761 = stablehlo.reshape %1760 : (tensor<1xf64>) -> tensor<f64>
    %1762 = stablehlo.slice %1750 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1763 = stablehlo.reshape %1762 : (tensor<1xf64>) -> tensor<f64>
    %1764 = stablehlo.multiply %1761, %1763 : tensor<f64>
    %1765 = stablehlo.add %1759, %1764 : tensor<f64>
    %1766 = stablehlo.slice %1326 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1767 = stablehlo.reshape %1766 : (tensor<1xf64>) -> tensor<f64>
    %1768 = stablehlo.slice %1750 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1769 = stablehlo.reshape %1768 : (tensor<1xf64>) -> tensor<f64>
    %1770 = stablehlo.multiply %1767, %1769 : tensor<f64>
    %1771 = stablehlo.subtract %1765, %1770 : tensor<f64>
    %1772 = stablehlo.reshape %1771 : (tensor<f64>) -> tensor<1xf64>
    %1773 = stablehlo.multiply %1747, %1769 : tensor<f64>
    %1774 = stablehlo.multiply %1755, %1763 : tensor<f64>
    %1775 = stablehlo.subtract %1773, %1774 : tensor<f64>
    %1776 = stablehlo.multiply %1761, %1757 : tensor<f64>
    %1777 = stablehlo.add %1775, %1776 : tensor<f64>
    %1778 = stablehlo.multiply %1767, %1752 : tensor<f64>
    %1779 = stablehlo.add %1777, %1778 : tensor<f64>
    %1780 = stablehlo.reshape %1779 : (tensor<f64>) -> tensor<1xf64>
    %1781 = stablehlo.multiply %1747, %1763 : tensor<f64>
    %1782 = stablehlo.multiply %1755, %1769 : tensor<f64>
    %1783 = stablehlo.add %1781, %1782 : tensor<f64>
    %1784 = stablehlo.multiply %1761, %1752 : tensor<f64>
    %1785 = stablehlo.subtract %1783, %1784 : tensor<f64>
    %1786 = stablehlo.multiply %1767, %1757 : tensor<f64>
    %1787 = stablehlo.add %1785, %1786 : tensor<f64>
    %1788 = stablehlo.reshape %1787 : (tensor<f64>) -> tensor<1xf64>
    %1789 = stablehlo.multiply %1747, %1757 : tensor<f64>
    %1790 = stablehlo.multiply %1755, %1752 : tensor<f64>
    %1791 = stablehlo.subtract %1789, %1790 : tensor<f64>
    %1792 = stablehlo.multiply %1761, %1769 : tensor<f64>
    %1793 = stablehlo.subtract %1791, %1792 : tensor<f64>
    %1794 = stablehlo.multiply %1767, %1763 : tensor<f64>
    %1795 = stablehlo.subtract %1793, %1794 : tensor<f64>
    %1796 = stablehlo.reshape %1795 : (tensor<f64>) -> tensor<1xf64>
    %1797 = stablehlo.concatenate %1772, %1780, %1788, %1796, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1798 = stablehlo.slice %1797 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1799 = stablehlo.reshape %1798 : (tensor<1xf64>) -> tensor<f64>
    %1800 = stablehlo.slice %1326 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1801 = stablehlo.reshape %1800 : (tensor<1xf64>) -> tensor<f64>
    %1802 = stablehlo.negate %1801 : tensor<f64>
    %1803 = stablehlo.reshape %1802 : (tensor<f64>) -> tensor<1xf64>
    %1804 = stablehlo.slice %1326 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1805 = stablehlo.reshape %1804 : (tensor<1xf64>) -> tensor<f64>
    %1806 = stablehlo.negate %1805 : tensor<f64>
    %1807 = stablehlo.reshape %1806 : (tensor<f64>) -> tensor<1xf64>
    %1808 = stablehlo.slice %1326 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1809 = stablehlo.reshape %1808 : (tensor<1xf64>) -> tensor<f64>
    %1810 = stablehlo.negate %1809 : tensor<f64>
    %1811 = stablehlo.reshape %1810 : (tensor<f64>) -> tensor<1xf64>
    %1812 = stablehlo.slice %1326 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1813 = stablehlo.reshape %1812 : (tensor<1xf64>) -> tensor<f64>
    %1814 = stablehlo.reshape %1813 : (tensor<f64>) -> tensor<1xf64>
    %1815 = stablehlo.concatenate %1803, %1807, %1811, %1814, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1816 = stablehlo.dot_general %1326, %1326, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1817 = stablehlo.broadcast_in_dim %1816, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1818 = stablehlo.divide %1815, %1817 : tensor<4xf64>
    %1819 = stablehlo.slice %1818 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1820 = stablehlo.reshape %1819 : (tensor<1xf64>) -> tensor<f64>
    %1821 = stablehlo.multiply %1799, %1820 : tensor<f64>
    %1822 = stablehlo.slice %1797 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1823 = stablehlo.reshape %1822 : (tensor<1xf64>) -> tensor<f64>
    %1824 = stablehlo.slice %1818 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1825 = stablehlo.reshape %1824 : (tensor<1xf64>) -> tensor<f64>
    %1826 = stablehlo.multiply %1823, %1825 : tensor<f64>
    %1827 = stablehlo.add %1821, %1826 : tensor<f64>
    %1828 = stablehlo.slice %1797 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1829 = stablehlo.reshape %1828 : (tensor<1xf64>) -> tensor<f64>
    %1830 = stablehlo.slice %1818 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1831 = stablehlo.reshape %1830 : (tensor<1xf64>) -> tensor<f64>
    %1832 = stablehlo.multiply %1829, %1831 : tensor<f64>
    %1833 = stablehlo.add %1827, %1832 : tensor<f64>
    %1834 = stablehlo.slice %1797 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1835 = stablehlo.reshape %1834 : (tensor<1xf64>) -> tensor<f64>
    %1836 = stablehlo.slice %1818 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1837 = stablehlo.reshape %1836 : (tensor<1xf64>) -> tensor<f64>
    %1838 = stablehlo.multiply %1835, %1837 : tensor<f64>
    %1839 = stablehlo.subtract %1833, %1838 : tensor<f64>
    %1840 = stablehlo.reshape %1839 : (tensor<f64>) -> tensor<1xf64>
    %1841 = stablehlo.multiply %1799, %1837 : tensor<f64>
    %1842 = stablehlo.multiply %1823, %1831 : tensor<f64>
    %1843 = stablehlo.subtract %1841, %1842 : tensor<f64>
    %1844 = stablehlo.multiply %1829, %1825 : tensor<f64>
    %1845 = stablehlo.add %1843, %1844 : tensor<f64>
    %1846 = stablehlo.multiply %1835, %1820 : tensor<f64>
    %1847 = stablehlo.add %1845, %1846 : tensor<f64>
    %1848 = stablehlo.reshape %1847 : (tensor<f64>) -> tensor<1xf64>
    %1849 = stablehlo.multiply %1799, %1831 : tensor<f64>
    %1850 = stablehlo.multiply %1823, %1837 : tensor<f64>
    %1851 = stablehlo.add %1849, %1850 : tensor<f64>
    %1852 = stablehlo.multiply %1829, %1820 : tensor<f64>
    %1853 = stablehlo.subtract %1851, %1852 : tensor<f64>
    %1854 = stablehlo.multiply %1835, %1825 : tensor<f64>
    %1855 = stablehlo.add %1853, %1854 : tensor<f64>
    %1856 = stablehlo.reshape %1855 : (tensor<f64>) -> tensor<1xf64>
    %1857 = stablehlo.multiply %1799, %1825 : tensor<f64>
    %1858 = stablehlo.multiply %1823, %1820 : tensor<f64>
    %1859 = stablehlo.subtract %1857, %1858 : tensor<f64>
    %1860 = stablehlo.multiply %1829, %1837 : tensor<f64>
    %1861 = stablehlo.subtract %1859, %1860 : tensor<f64>
    %1862 = stablehlo.multiply %1835, %1831 : tensor<f64>
    %1863 = stablehlo.subtract %1861, %1862 : tensor<f64>
    %1864 = stablehlo.reshape %1863 : (tensor<f64>) -> tensor<1xf64>
    %1865 = stablehlo.concatenate %1840, %1848, %1856, %1864, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1866 = stablehlo.slice %1865 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1867 = stablehlo.reshape %1866 : (tensor<1xf64>) -> tensor<f64>
    %1868 = stablehlo.reshape %1867 : (tensor<f64>) -> tensor<1xf64>
    %1869 = stablehlo.slice %1865 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1870 = stablehlo.reshape %1869 : (tensor<1xf64>) -> tensor<f64>
    %1871 = stablehlo.reshape %1870 : (tensor<f64>) -> tensor<1xf64>
    %1872 = stablehlo.slice %1865 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1873 = stablehlo.reshape %1872 : (tensor<1xf64>) -> tensor<f64>
    %1874 = stablehlo.reshape %1873 : (tensor<f64>) -> tensor<1xf64>
    %1875 = stablehlo.concatenate %1868, %1871, %1874, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1876 = stablehlo.concatenate %1745, %1875, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1877 = stablehlo.slice %arg3 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1878 = stablehlo.reshape %arg8 : (tensor<f64>) -> tensor<f64>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1879 = stablehlo.multiply %cst_23, %1878 : tensor<f64>
    %1880 = stablehlo.broadcast_in_dim %1879, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1881 = stablehlo.multiply %1880, %1 : tensor<6xf64>
    %1882 = stablehlo.slice %1881 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_24 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1883 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1884 = stablehlo.divide %1882, %1883 : tensor<3xf64>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1885 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1886 = stablehlo.concatenate %1884, %1885, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1887 = stablehlo.slice %1886 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1888 = stablehlo.reshape %1887 : (tensor<1xf64>) -> tensor<f64>
    %1889 = stablehlo.slice %1877 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1890 = stablehlo.reshape %1889 : (tensor<1xf64>) -> tensor<f64>
    %1891 = stablehlo.multiply %1888, %1890 : tensor<f64>
    %1892 = stablehlo.slice %1886 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1893 = stablehlo.reshape %1892 : (tensor<1xf64>) -> tensor<f64>
    %1894 = stablehlo.slice %1877 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1895 = stablehlo.reshape %1894 : (tensor<1xf64>) -> tensor<f64>
    %1896 = stablehlo.multiply %1893, %1895 : tensor<f64>
    %1897 = stablehlo.add %1891, %1896 : tensor<f64>
    %1898 = stablehlo.slice %1886 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1899 = stablehlo.reshape %1898 : (tensor<1xf64>) -> tensor<f64>
    %1900 = stablehlo.slice %1877 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1901 = stablehlo.reshape %1900 : (tensor<1xf64>) -> tensor<f64>
    %1902 = stablehlo.multiply %1899, %1901 : tensor<f64>
    %1903 = stablehlo.add %1897, %1902 : tensor<f64>
    %1904 = stablehlo.slice %1886 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1905 = stablehlo.reshape %1904 : (tensor<1xf64>) -> tensor<f64>
    %1906 = stablehlo.slice %1877 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1907 = stablehlo.reshape %1906 : (tensor<1xf64>) -> tensor<f64>
    %1908 = stablehlo.multiply %1905, %1907 : tensor<f64>
    %1909 = stablehlo.subtract %1903, %1908 : tensor<f64>
    %1910 = stablehlo.reshape %1909 : (tensor<f64>) -> tensor<1xf64>
    %1911 = stablehlo.multiply %1888, %1907 : tensor<f64>
    %1912 = stablehlo.multiply %1893, %1901 : tensor<f64>
    %1913 = stablehlo.subtract %1911, %1912 : tensor<f64>
    %1914 = stablehlo.multiply %1899, %1895 : tensor<f64>
    %1915 = stablehlo.add %1913, %1914 : tensor<f64>
    %1916 = stablehlo.multiply %1905, %1890 : tensor<f64>
    %1917 = stablehlo.add %1915, %1916 : tensor<f64>
    %1918 = stablehlo.reshape %1917 : (tensor<f64>) -> tensor<1xf64>
    %1919 = stablehlo.multiply %1888, %1901 : tensor<f64>
    %1920 = stablehlo.multiply %1893, %1907 : tensor<f64>
    %1921 = stablehlo.add %1919, %1920 : tensor<f64>
    %1922 = stablehlo.multiply %1899, %1890 : tensor<f64>
    %1923 = stablehlo.subtract %1921, %1922 : tensor<f64>
    %1924 = stablehlo.multiply %1905, %1895 : tensor<f64>
    %1925 = stablehlo.add %1923, %1924 : tensor<f64>
    %1926 = stablehlo.reshape %1925 : (tensor<f64>) -> tensor<1xf64>
    %1927 = stablehlo.multiply %1888, %1895 : tensor<f64>
    %1928 = stablehlo.multiply %1893, %1890 : tensor<f64>
    %1929 = stablehlo.subtract %1927, %1928 : tensor<f64>
    %1930 = stablehlo.multiply %1899, %1907 : tensor<f64>
    %1931 = stablehlo.subtract %1929, %1930 : tensor<f64>
    %1932 = stablehlo.multiply %1905, %1901 : tensor<f64>
    %1933 = stablehlo.subtract %1931, %1932 : tensor<f64>
    %1934 = stablehlo.reshape %1933 : (tensor<f64>) -> tensor<1xf64>
    %1935 = stablehlo.concatenate %1910, %1918, %1926, %1934, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1936 = stablehlo.add %1877, %1935 : tensor<4xf64>
    %1937 = stablehlo.dot_general %1936, %1936, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1938 = stablehlo.sqrt %1937 : tensor<f64>
    %1939 = stablehlo.broadcast_in_dim %1938, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1940 = stablehlo.divide %1936, %1939 : tensor<4xf64>
    %1941 = stablehlo.slice %arg3 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1942 = stablehlo.slice %1881 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1943 = stablehlo.add %1941, %1942 : tensor<3xf64>
    %1944 = stablehlo.concatenate %1940, %1943, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %1945 = stablehlo.broadcast_in_dim %1879, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1946 = stablehlo.multiply %1945, %1876 : tensor<6xf64>
    %1947 = stablehlo.add %1, %1946 : tensor<6xf64>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1948 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1949 = call @inner_55(%1948, %arg6) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %1950 = call @inner_57(%0, %1947, %1949) : (tensor<3xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1951 = stablehlo.slice %1944 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1952 = stablehlo.slice %1951 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1953 = stablehlo.reshape %1952 : (tensor<1xf64>) -> tensor<f64>
    %1954 = stablehlo.slice %1951 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1955 = stablehlo.reshape %1954 : (tensor<1xf64>) -> tensor<f64>
    %1956 = stablehlo.negate %1955 : tensor<f64>
    %1957 = stablehlo.reshape %1956 : (tensor<f64>) -> tensor<1xf64>
    %1958 = stablehlo.slice %1951 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1959 = stablehlo.reshape %1958 : (tensor<1xf64>) -> tensor<f64>
    %1960 = stablehlo.negate %1959 : tensor<f64>
    %1961 = stablehlo.reshape %1960 : (tensor<f64>) -> tensor<1xf64>
    %1962 = stablehlo.slice %1951 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1963 = stablehlo.reshape %1962 : (tensor<1xf64>) -> tensor<f64>
    %1964 = stablehlo.negate %1963 : tensor<f64>
    %1965 = stablehlo.reshape %1964 : (tensor<f64>) -> tensor<1xf64>
    %1966 = stablehlo.slice %1951 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1967 = stablehlo.reshape %1966 : (tensor<1xf64>) -> tensor<f64>
    %1968 = stablehlo.reshape %1967 : (tensor<f64>) -> tensor<1xf64>
    %1969 = stablehlo.concatenate %1957, %1961, %1965, %1968, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1970 = stablehlo.dot_general %1951, %1951, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1971 = stablehlo.broadcast_in_dim %1970, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1972 = stablehlo.divide %1969, %1971 : tensor<4xf64>
    %1973 = stablehlo.slice %1972 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1974 = stablehlo.reshape %1973 : (tensor<1xf64>) -> tensor<f64>
    %1975 = stablehlo.slice %1950 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1976 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1977 = stablehlo.concatenate %1975, %1976, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1978 = stablehlo.slice %1977 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1979 = stablehlo.reshape %1978 : (tensor<1xf64>) -> tensor<f64>
    %1980 = stablehlo.multiply %1974, %1979 : tensor<f64>
    %1981 = stablehlo.slice %1972 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1982 = stablehlo.reshape %1981 : (tensor<1xf64>) -> tensor<f64>
    %1983 = stablehlo.slice %1977 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1984 = stablehlo.reshape %1983 : (tensor<1xf64>) -> tensor<f64>
    %1985 = stablehlo.multiply %1982, %1984 : tensor<f64>
    %1986 = stablehlo.add %1980, %1985 : tensor<f64>
    %1987 = stablehlo.slice %1972 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1988 = stablehlo.reshape %1987 : (tensor<1xf64>) -> tensor<f64>
    %1989 = stablehlo.slice %1977 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1990 = stablehlo.reshape %1989 : (tensor<1xf64>) -> tensor<f64>
    %1991 = stablehlo.multiply %1988, %1990 : tensor<f64>
    %1992 = stablehlo.add %1986, %1991 : tensor<f64>
    %1993 = stablehlo.slice %1972 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1994 = stablehlo.reshape %1993 : (tensor<1xf64>) -> tensor<f64>
    %1995 = stablehlo.slice %1977 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1996 = stablehlo.reshape %1995 : (tensor<1xf64>) -> tensor<f64>
    %1997 = stablehlo.multiply %1994, %1996 : tensor<f64>
    %1998 = stablehlo.subtract %1992, %1997 : tensor<f64>
    %1999 = stablehlo.reshape %1998 : (tensor<f64>) -> tensor<1xf64>
    %2000 = stablehlo.multiply %1974, %1996 : tensor<f64>
    %2001 = stablehlo.multiply %1982, %1990 : tensor<f64>
    %2002 = stablehlo.subtract %2000, %2001 : tensor<f64>
    %2003 = stablehlo.multiply %1988, %1984 : tensor<f64>
    %2004 = stablehlo.add %2002, %2003 : tensor<f64>
    %2005 = stablehlo.multiply %1994, %1979 : tensor<f64>
    %2006 = stablehlo.add %2004, %2005 : tensor<f64>
    %2007 = stablehlo.reshape %2006 : (tensor<f64>) -> tensor<1xf64>
    %2008 = stablehlo.multiply %1974, %1990 : tensor<f64>
    %2009 = stablehlo.multiply %1982, %1996 : tensor<f64>
    %2010 = stablehlo.add %2008, %2009 : tensor<f64>
    %2011 = stablehlo.multiply %1988, %1979 : tensor<f64>
    %2012 = stablehlo.subtract %2010, %2011 : tensor<f64>
    %2013 = stablehlo.multiply %1994, %1984 : tensor<f64>
    %2014 = stablehlo.add %2012, %2013 : tensor<f64>
    %2015 = stablehlo.reshape %2014 : (tensor<f64>) -> tensor<1xf64>
    %2016 = stablehlo.multiply %1974, %1984 : tensor<f64>
    %2017 = stablehlo.multiply %1982, %1979 : tensor<f64>
    %2018 = stablehlo.subtract %2016, %2017 : tensor<f64>
    %2019 = stablehlo.multiply %1988, %1996 : tensor<f64>
    %2020 = stablehlo.subtract %2018, %2019 : tensor<f64>
    %2021 = stablehlo.multiply %1994, %1990 : tensor<f64>
    %2022 = stablehlo.subtract %2020, %2021 : tensor<f64>
    %2023 = stablehlo.reshape %2022 : (tensor<f64>) -> tensor<1xf64>
    %2024 = stablehlo.concatenate %1999, %2007, %2015, %2023, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2025 = stablehlo.slice %2024 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2026 = stablehlo.reshape %2025 : (tensor<1xf64>) -> tensor<f64>
    %2027 = stablehlo.slice %1972 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2028 = stablehlo.reshape %2027 : (tensor<1xf64>) -> tensor<f64>
    %2029 = stablehlo.negate %2028 : tensor<f64>
    %2030 = stablehlo.reshape %2029 : (tensor<f64>) -> tensor<1xf64>
    %2031 = stablehlo.slice %1972 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2032 = stablehlo.reshape %2031 : (tensor<1xf64>) -> tensor<f64>
    %2033 = stablehlo.negate %2032 : tensor<f64>
    %2034 = stablehlo.reshape %2033 : (tensor<f64>) -> tensor<1xf64>
    %2035 = stablehlo.slice %1972 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2036 = stablehlo.reshape %2035 : (tensor<1xf64>) -> tensor<f64>
    %2037 = stablehlo.negate %2036 : tensor<f64>
    %2038 = stablehlo.reshape %2037 : (tensor<f64>) -> tensor<1xf64>
    %2039 = stablehlo.slice %1972 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2040 = stablehlo.reshape %2039 : (tensor<1xf64>) -> tensor<f64>
    %2041 = stablehlo.reshape %2040 : (tensor<f64>) -> tensor<1xf64>
    %2042 = stablehlo.concatenate %2030, %2034, %2038, %2041, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2043 = stablehlo.dot_general %1972, %1972, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2044 = stablehlo.broadcast_in_dim %2043, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2045 = stablehlo.divide %2042, %2044 : tensor<4xf64>
    %2046 = stablehlo.slice %2045 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2047 = stablehlo.reshape %2046 : (tensor<1xf64>) -> tensor<f64>
    %2048 = stablehlo.multiply %2026, %2047 : tensor<f64>
    %2049 = stablehlo.slice %2024 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2050 = stablehlo.reshape %2049 : (tensor<1xf64>) -> tensor<f64>
    %2051 = stablehlo.slice %2045 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2052 = stablehlo.reshape %2051 : (tensor<1xf64>) -> tensor<f64>
    %2053 = stablehlo.multiply %2050, %2052 : tensor<f64>
    %2054 = stablehlo.add %2048, %2053 : tensor<f64>
    %2055 = stablehlo.slice %2024 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2056 = stablehlo.reshape %2055 : (tensor<1xf64>) -> tensor<f64>
    %2057 = stablehlo.slice %2045 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2058 = stablehlo.reshape %2057 : (tensor<1xf64>) -> tensor<f64>
    %2059 = stablehlo.multiply %2056, %2058 : tensor<f64>
    %2060 = stablehlo.add %2054, %2059 : tensor<f64>
    %2061 = stablehlo.slice %2024 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2062 = stablehlo.reshape %2061 : (tensor<1xf64>) -> tensor<f64>
    %2063 = stablehlo.slice %2045 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2064 = stablehlo.reshape %2063 : (tensor<1xf64>) -> tensor<f64>
    %2065 = stablehlo.multiply %2062, %2064 : tensor<f64>
    %2066 = stablehlo.subtract %2060, %2065 : tensor<f64>
    %2067 = stablehlo.reshape %2066 : (tensor<f64>) -> tensor<1xf64>
    %2068 = stablehlo.multiply %2026, %2064 : tensor<f64>
    %2069 = stablehlo.multiply %2050, %2058 : tensor<f64>
    %2070 = stablehlo.subtract %2068, %2069 : tensor<f64>
    %2071 = stablehlo.multiply %2056, %2052 : tensor<f64>
    %2072 = stablehlo.add %2070, %2071 : tensor<f64>
    %2073 = stablehlo.multiply %2062, %2047 : tensor<f64>
    %2074 = stablehlo.add %2072, %2073 : tensor<f64>
    %2075 = stablehlo.reshape %2074 : (tensor<f64>) -> tensor<1xf64>
    %2076 = stablehlo.multiply %2026, %2058 : tensor<f64>
    %2077 = stablehlo.multiply %2050, %2064 : tensor<f64>
    %2078 = stablehlo.add %2076, %2077 : tensor<f64>
    %2079 = stablehlo.multiply %2056, %2047 : tensor<f64>
    %2080 = stablehlo.subtract %2078, %2079 : tensor<f64>
    %2081 = stablehlo.multiply %2062, %2052 : tensor<f64>
    %2082 = stablehlo.add %2080, %2081 : tensor<f64>
    %2083 = stablehlo.reshape %2082 : (tensor<f64>) -> tensor<1xf64>
    %2084 = stablehlo.multiply %2026, %2052 : tensor<f64>
    %2085 = stablehlo.multiply %2050, %2047 : tensor<f64>
    %2086 = stablehlo.subtract %2084, %2085 : tensor<f64>
    %2087 = stablehlo.multiply %2056, %2064 : tensor<f64>
    %2088 = stablehlo.subtract %2086, %2087 : tensor<f64>
    %2089 = stablehlo.multiply %2062, %2058 : tensor<f64>
    %2090 = stablehlo.subtract %2088, %2089 : tensor<f64>
    %2091 = stablehlo.reshape %2090 : (tensor<f64>) -> tensor<1xf64>
    %2092 = stablehlo.concatenate %2067, %2075, %2083, %2091, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2093 = stablehlo.slice %2092 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2094 = stablehlo.reshape %2093 : (tensor<1xf64>) -> tensor<f64>
    %2095 = stablehlo.reshape %2094 : (tensor<f64>) -> tensor<1xf64>
    %2096 = stablehlo.slice %2092 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2097 = stablehlo.reshape %2096 : (tensor<1xf64>) -> tensor<f64>
    %2098 = stablehlo.reshape %2097 : (tensor<f64>) -> tensor<1xf64>
    %2099 = stablehlo.slice %2092 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2100 = stablehlo.reshape %2099 : (tensor<1xf64>) -> tensor<f64>
    %2101 = stablehlo.reshape %2100 : (tensor<f64>) -> tensor<1xf64>
    %2102 = stablehlo.concatenate %2095, %2098, %2101, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2103 = stablehlo.slice %1972 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2104 = stablehlo.reshape %2103 : (tensor<1xf64>) -> tensor<f64>
    %2105 = stablehlo.slice %1950 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2106 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2107 = stablehlo.concatenate %2105, %2106, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2108 = stablehlo.slice %2107 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2109 = stablehlo.reshape %2108 : (tensor<1xf64>) -> tensor<f64>
    %2110 = stablehlo.multiply %2104, %2109 : tensor<f64>
    %2111 = stablehlo.slice %1972 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2112 = stablehlo.reshape %2111 : (tensor<1xf64>) -> tensor<f64>
    %2113 = stablehlo.slice %2107 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2114 = stablehlo.reshape %2113 : (tensor<1xf64>) -> tensor<f64>
    %2115 = stablehlo.multiply %2112, %2114 : tensor<f64>
    %2116 = stablehlo.add %2110, %2115 : tensor<f64>
    %2117 = stablehlo.slice %1972 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2118 = stablehlo.reshape %2117 : (tensor<1xf64>) -> tensor<f64>
    %2119 = stablehlo.slice %2107 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2120 = stablehlo.reshape %2119 : (tensor<1xf64>) -> tensor<f64>
    %2121 = stablehlo.multiply %2118, %2120 : tensor<f64>
    %2122 = stablehlo.add %2116, %2121 : tensor<f64>
    %2123 = stablehlo.slice %1972 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2124 = stablehlo.reshape %2123 : (tensor<1xf64>) -> tensor<f64>
    %2125 = stablehlo.slice %2107 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2126 = stablehlo.reshape %2125 : (tensor<1xf64>) -> tensor<f64>
    %2127 = stablehlo.multiply %2124, %2126 : tensor<f64>
    %2128 = stablehlo.subtract %2122, %2127 : tensor<f64>
    %2129 = stablehlo.reshape %2128 : (tensor<f64>) -> tensor<1xf64>
    %2130 = stablehlo.multiply %2104, %2126 : tensor<f64>
    %2131 = stablehlo.multiply %2112, %2120 : tensor<f64>
    %2132 = stablehlo.subtract %2130, %2131 : tensor<f64>
    %2133 = stablehlo.multiply %2118, %2114 : tensor<f64>
    %2134 = stablehlo.add %2132, %2133 : tensor<f64>
    %2135 = stablehlo.multiply %2124, %2109 : tensor<f64>
    %2136 = stablehlo.add %2134, %2135 : tensor<f64>
    %2137 = stablehlo.reshape %2136 : (tensor<f64>) -> tensor<1xf64>
    %2138 = stablehlo.multiply %2104, %2120 : tensor<f64>
    %2139 = stablehlo.multiply %2112, %2126 : tensor<f64>
    %2140 = stablehlo.add %2138, %2139 : tensor<f64>
    %2141 = stablehlo.multiply %2118, %2109 : tensor<f64>
    %2142 = stablehlo.subtract %2140, %2141 : tensor<f64>
    %2143 = stablehlo.multiply %2124, %2114 : tensor<f64>
    %2144 = stablehlo.add %2142, %2143 : tensor<f64>
    %2145 = stablehlo.reshape %2144 : (tensor<f64>) -> tensor<1xf64>
    %2146 = stablehlo.multiply %2104, %2114 : tensor<f64>
    %2147 = stablehlo.multiply %2112, %2109 : tensor<f64>
    %2148 = stablehlo.subtract %2146, %2147 : tensor<f64>
    %2149 = stablehlo.multiply %2118, %2126 : tensor<f64>
    %2150 = stablehlo.subtract %2148, %2149 : tensor<f64>
    %2151 = stablehlo.multiply %2124, %2120 : tensor<f64>
    %2152 = stablehlo.subtract %2150, %2151 : tensor<f64>
    %2153 = stablehlo.reshape %2152 : (tensor<f64>) -> tensor<1xf64>
    %2154 = stablehlo.concatenate %2129, %2137, %2145, %2153, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2155 = stablehlo.slice %2154 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2156 = stablehlo.reshape %2155 : (tensor<1xf64>) -> tensor<f64>
    %2157 = stablehlo.slice %1972 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2158 = stablehlo.reshape %2157 : (tensor<1xf64>) -> tensor<f64>
    %2159 = stablehlo.negate %2158 : tensor<f64>
    %2160 = stablehlo.reshape %2159 : (tensor<f64>) -> tensor<1xf64>
    %2161 = stablehlo.slice %1972 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2162 = stablehlo.reshape %2161 : (tensor<1xf64>) -> tensor<f64>
    %2163 = stablehlo.negate %2162 : tensor<f64>
    %2164 = stablehlo.reshape %2163 : (tensor<f64>) -> tensor<1xf64>
    %2165 = stablehlo.slice %1972 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2166 = stablehlo.reshape %2165 : (tensor<1xf64>) -> tensor<f64>
    %2167 = stablehlo.negate %2166 : tensor<f64>
    %2168 = stablehlo.reshape %2167 : (tensor<f64>) -> tensor<1xf64>
    %2169 = stablehlo.slice %1972 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2170 = stablehlo.reshape %2169 : (tensor<1xf64>) -> tensor<f64>
    %2171 = stablehlo.reshape %2170 : (tensor<f64>) -> tensor<1xf64>
    %2172 = stablehlo.concatenate %2160, %2164, %2168, %2171, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2173 = stablehlo.dot_general %1972, %1972, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2174 = stablehlo.broadcast_in_dim %2173, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2175 = stablehlo.divide %2172, %2174 : tensor<4xf64>
    %2176 = stablehlo.slice %2175 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2177 = stablehlo.reshape %2176 : (tensor<1xf64>) -> tensor<f64>
    %2178 = stablehlo.multiply %2156, %2177 : tensor<f64>
    %2179 = stablehlo.slice %2154 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2180 = stablehlo.reshape %2179 : (tensor<1xf64>) -> tensor<f64>
    %2181 = stablehlo.slice %2175 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2182 = stablehlo.reshape %2181 : (tensor<1xf64>) -> tensor<f64>
    %2183 = stablehlo.multiply %2180, %2182 : tensor<f64>
    %2184 = stablehlo.add %2178, %2183 : tensor<f64>
    %2185 = stablehlo.slice %2154 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2186 = stablehlo.reshape %2185 : (tensor<1xf64>) -> tensor<f64>
    %2187 = stablehlo.slice %2175 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2188 = stablehlo.reshape %2187 : (tensor<1xf64>) -> tensor<f64>
    %2189 = stablehlo.multiply %2186, %2188 : tensor<f64>
    %2190 = stablehlo.add %2184, %2189 : tensor<f64>
    %2191 = stablehlo.slice %2154 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2192 = stablehlo.reshape %2191 : (tensor<1xf64>) -> tensor<f64>
    %2193 = stablehlo.slice %2175 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2194 = stablehlo.reshape %2193 : (tensor<1xf64>) -> tensor<f64>
    %2195 = stablehlo.multiply %2192, %2194 : tensor<f64>
    %2196 = stablehlo.subtract %2190, %2195 : tensor<f64>
    %2197 = stablehlo.reshape %2196 : (tensor<f64>) -> tensor<1xf64>
    %2198 = stablehlo.multiply %2156, %2194 : tensor<f64>
    %2199 = stablehlo.multiply %2180, %2188 : tensor<f64>
    %2200 = stablehlo.subtract %2198, %2199 : tensor<f64>
    %2201 = stablehlo.multiply %2186, %2182 : tensor<f64>
    %2202 = stablehlo.add %2200, %2201 : tensor<f64>
    %2203 = stablehlo.multiply %2192, %2177 : tensor<f64>
    %2204 = stablehlo.add %2202, %2203 : tensor<f64>
    %2205 = stablehlo.reshape %2204 : (tensor<f64>) -> tensor<1xf64>
    %2206 = stablehlo.multiply %2156, %2188 : tensor<f64>
    %2207 = stablehlo.multiply %2180, %2194 : tensor<f64>
    %2208 = stablehlo.add %2206, %2207 : tensor<f64>
    %2209 = stablehlo.multiply %2186, %2177 : tensor<f64>
    %2210 = stablehlo.subtract %2208, %2209 : tensor<f64>
    %2211 = stablehlo.multiply %2192, %2182 : tensor<f64>
    %2212 = stablehlo.add %2210, %2211 : tensor<f64>
    %2213 = stablehlo.reshape %2212 : (tensor<f64>) -> tensor<1xf64>
    %2214 = stablehlo.multiply %2156, %2182 : tensor<f64>
    %2215 = stablehlo.multiply %2180, %2177 : tensor<f64>
    %2216 = stablehlo.subtract %2214, %2215 : tensor<f64>
    %2217 = stablehlo.multiply %2186, %2194 : tensor<f64>
    %2218 = stablehlo.subtract %2216, %2217 : tensor<f64>
    %2219 = stablehlo.multiply %2192, %2188 : tensor<f64>
    %2220 = stablehlo.subtract %2218, %2219 : tensor<f64>
    %2221 = stablehlo.reshape %2220 : (tensor<f64>) -> tensor<1xf64>
    %2222 = stablehlo.concatenate %2197, %2205, %2213, %2221, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2223 = stablehlo.slice %2222 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2224 = stablehlo.reshape %2223 : (tensor<1xf64>) -> tensor<f64>
    %2225 = stablehlo.reshape %2224 : (tensor<f64>) -> tensor<1xf64>
    %2226 = stablehlo.slice %2222 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2227 = stablehlo.reshape %2226 : (tensor<1xf64>) -> tensor<f64>
    %2228 = stablehlo.reshape %2227 : (tensor<f64>) -> tensor<1xf64>
    %2229 = stablehlo.slice %2222 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2230 = stablehlo.reshape %2229 : (tensor<1xf64>) -> tensor<f64>
    %2231 = stablehlo.reshape %2230 : (tensor<f64>) -> tensor<1xf64>
    %2232 = stablehlo.concatenate %2225, %2228, %2231, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2233 = stablehlo.concatenate %2102, %2232, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2234 = stablehlo.slice %2233 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %2235 = stablehlo.slice %arg6 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %2236 = stablehlo.divide %2234, %2235 : tensor<3xf64>
    %2237 = stablehlo.slice %2233 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %2238 = stablehlo.slice %arg6 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %2239 = stablehlo.reshape %2238 : (tensor<1xf64>) -> tensor<f64>
    %2240 = stablehlo.broadcast_in_dim %2239, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2241 = stablehlo.divide %2237, %2240 : tensor<3xf64>
    %2242 = stablehlo.concatenate %2236, %2241, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2243 = stablehlo.slice %2242 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2244 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2245 = stablehlo.concatenate %2243, %2244, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2246 = stablehlo.slice %2245 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2247 = stablehlo.reshape %2246 : (tensor<1xf64>) -> tensor<f64>
    %2248 = stablehlo.multiply %1953, %2247 : tensor<f64>
    %2249 = stablehlo.slice %1951 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2250 = stablehlo.reshape %2249 : (tensor<1xf64>) -> tensor<f64>
    %2251 = stablehlo.slice %2245 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2252 = stablehlo.reshape %2251 : (tensor<1xf64>) -> tensor<f64>
    %2253 = stablehlo.multiply %2250, %2252 : tensor<f64>
    %2254 = stablehlo.add %2248, %2253 : tensor<f64>
    %2255 = stablehlo.slice %1951 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2256 = stablehlo.reshape %2255 : (tensor<1xf64>) -> tensor<f64>
    %2257 = stablehlo.slice %2245 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2258 = stablehlo.reshape %2257 : (tensor<1xf64>) -> tensor<f64>
    %2259 = stablehlo.multiply %2256, %2258 : tensor<f64>
    %2260 = stablehlo.add %2254, %2259 : tensor<f64>
    %2261 = stablehlo.slice %1951 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2262 = stablehlo.reshape %2261 : (tensor<1xf64>) -> tensor<f64>
    %2263 = stablehlo.slice %2245 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2264 = stablehlo.reshape %2263 : (tensor<1xf64>) -> tensor<f64>
    %2265 = stablehlo.multiply %2262, %2264 : tensor<f64>
    %2266 = stablehlo.subtract %2260, %2265 : tensor<f64>
    %2267 = stablehlo.reshape %2266 : (tensor<f64>) -> tensor<1xf64>
    %2268 = stablehlo.multiply %1953, %2264 : tensor<f64>
    %2269 = stablehlo.multiply %2250, %2258 : tensor<f64>
    %2270 = stablehlo.subtract %2268, %2269 : tensor<f64>
    %2271 = stablehlo.multiply %2256, %2252 : tensor<f64>
    %2272 = stablehlo.add %2270, %2271 : tensor<f64>
    %2273 = stablehlo.multiply %2262, %2247 : tensor<f64>
    %2274 = stablehlo.add %2272, %2273 : tensor<f64>
    %2275 = stablehlo.reshape %2274 : (tensor<f64>) -> tensor<1xf64>
    %2276 = stablehlo.multiply %1953, %2258 : tensor<f64>
    %2277 = stablehlo.multiply %2250, %2264 : tensor<f64>
    %2278 = stablehlo.add %2276, %2277 : tensor<f64>
    %2279 = stablehlo.multiply %2256, %2247 : tensor<f64>
    %2280 = stablehlo.subtract %2278, %2279 : tensor<f64>
    %2281 = stablehlo.multiply %2262, %2252 : tensor<f64>
    %2282 = stablehlo.add %2280, %2281 : tensor<f64>
    %2283 = stablehlo.reshape %2282 : (tensor<f64>) -> tensor<1xf64>
    %2284 = stablehlo.multiply %1953, %2252 : tensor<f64>
    %2285 = stablehlo.multiply %2250, %2247 : tensor<f64>
    %2286 = stablehlo.subtract %2284, %2285 : tensor<f64>
    %2287 = stablehlo.multiply %2256, %2264 : tensor<f64>
    %2288 = stablehlo.subtract %2286, %2287 : tensor<f64>
    %2289 = stablehlo.multiply %2262, %2258 : tensor<f64>
    %2290 = stablehlo.subtract %2288, %2289 : tensor<f64>
    %2291 = stablehlo.reshape %2290 : (tensor<f64>) -> tensor<1xf64>
    %2292 = stablehlo.concatenate %2267, %2275, %2283, %2291, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2293 = stablehlo.slice %2292 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2294 = stablehlo.reshape %2293 : (tensor<1xf64>) -> tensor<f64>
    %2295 = stablehlo.slice %1951 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2296 = stablehlo.reshape %2295 : (tensor<1xf64>) -> tensor<f64>
    %2297 = stablehlo.negate %2296 : tensor<f64>
    %2298 = stablehlo.reshape %2297 : (tensor<f64>) -> tensor<1xf64>
    %2299 = stablehlo.slice %1951 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2300 = stablehlo.reshape %2299 : (tensor<1xf64>) -> tensor<f64>
    %2301 = stablehlo.negate %2300 : tensor<f64>
    %2302 = stablehlo.reshape %2301 : (tensor<f64>) -> tensor<1xf64>
    %2303 = stablehlo.slice %1951 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2304 = stablehlo.reshape %2303 : (tensor<1xf64>) -> tensor<f64>
    %2305 = stablehlo.negate %2304 : tensor<f64>
    %2306 = stablehlo.reshape %2305 : (tensor<f64>) -> tensor<1xf64>
    %2307 = stablehlo.slice %1951 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2308 = stablehlo.reshape %2307 : (tensor<1xf64>) -> tensor<f64>
    %2309 = stablehlo.reshape %2308 : (tensor<f64>) -> tensor<1xf64>
    %2310 = stablehlo.concatenate %2298, %2302, %2306, %2309, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2311 = stablehlo.dot_general %1951, %1951, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2312 = stablehlo.broadcast_in_dim %2311, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2313 = stablehlo.divide %2310, %2312 : tensor<4xf64>
    %2314 = stablehlo.slice %2313 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2315 = stablehlo.reshape %2314 : (tensor<1xf64>) -> tensor<f64>
    %2316 = stablehlo.multiply %2294, %2315 : tensor<f64>
    %2317 = stablehlo.slice %2292 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2318 = stablehlo.reshape %2317 : (tensor<1xf64>) -> tensor<f64>
    %2319 = stablehlo.slice %2313 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2320 = stablehlo.reshape %2319 : (tensor<1xf64>) -> tensor<f64>
    %2321 = stablehlo.multiply %2318, %2320 : tensor<f64>
    %2322 = stablehlo.add %2316, %2321 : tensor<f64>
    %2323 = stablehlo.slice %2292 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2324 = stablehlo.reshape %2323 : (tensor<1xf64>) -> tensor<f64>
    %2325 = stablehlo.slice %2313 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2326 = stablehlo.reshape %2325 : (tensor<1xf64>) -> tensor<f64>
    %2327 = stablehlo.multiply %2324, %2326 : tensor<f64>
    %2328 = stablehlo.add %2322, %2327 : tensor<f64>
    %2329 = stablehlo.slice %2292 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2330 = stablehlo.reshape %2329 : (tensor<1xf64>) -> tensor<f64>
    %2331 = stablehlo.slice %2313 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2332 = stablehlo.reshape %2331 : (tensor<1xf64>) -> tensor<f64>
    %2333 = stablehlo.multiply %2330, %2332 : tensor<f64>
    %2334 = stablehlo.subtract %2328, %2333 : tensor<f64>
    %2335 = stablehlo.reshape %2334 : (tensor<f64>) -> tensor<1xf64>
    %2336 = stablehlo.multiply %2294, %2332 : tensor<f64>
    %2337 = stablehlo.multiply %2318, %2326 : tensor<f64>
    %2338 = stablehlo.subtract %2336, %2337 : tensor<f64>
    %2339 = stablehlo.multiply %2324, %2320 : tensor<f64>
    %2340 = stablehlo.add %2338, %2339 : tensor<f64>
    %2341 = stablehlo.multiply %2330, %2315 : tensor<f64>
    %2342 = stablehlo.add %2340, %2341 : tensor<f64>
    %2343 = stablehlo.reshape %2342 : (tensor<f64>) -> tensor<1xf64>
    %2344 = stablehlo.multiply %2294, %2326 : tensor<f64>
    %2345 = stablehlo.multiply %2318, %2332 : tensor<f64>
    %2346 = stablehlo.add %2344, %2345 : tensor<f64>
    %2347 = stablehlo.multiply %2324, %2315 : tensor<f64>
    %2348 = stablehlo.subtract %2346, %2347 : tensor<f64>
    %2349 = stablehlo.multiply %2330, %2320 : tensor<f64>
    %2350 = stablehlo.add %2348, %2349 : tensor<f64>
    %2351 = stablehlo.reshape %2350 : (tensor<f64>) -> tensor<1xf64>
    %2352 = stablehlo.multiply %2294, %2320 : tensor<f64>
    %2353 = stablehlo.multiply %2318, %2315 : tensor<f64>
    %2354 = stablehlo.subtract %2352, %2353 : tensor<f64>
    %2355 = stablehlo.multiply %2324, %2332 : tensor<f64>
    %2356 = stablehlo.subtract %2354, %2355 : tensor<f64>
    %2357 = stablehlo.multiply %2330, %2326 : tensor<f64>
    %2358 = stablehlo.subtract %2356, %2357 : tensor<f64>
    %2359 = stablehlo.reshape %2358 : (tensor<f64>) -> tensor<1xf64>
    %2360 = stablehlo.concatenate %2335, %2343, %2351, %2359, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2361 = stablehlo.slice %2360 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2362 = stablehlo.reshape %2361 : (tensor<1xf64>) -> tensor<f64>
    %2363 = stablehlo.reshape %2362 : (tensor<f64>) -> tensor<1xf64>
    %2364 = stablehlo.slice %2360 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2365 = stablehlo.reshape %2364 : (tensor<1xf64>) -> tensor<f64>
    %2366 = stablehlo.reshape %2365 : (tensor<f64>) -> tensor<1xf64>
    %2367 = stablehlo.slice %2360 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2368 = stablehlo.reshape %2367 : (tensor<1xf64>) -> tensor<f64>
    %2369 = stablehlo.reshape %2368 : (tensor<f64>) -> tensor<1xf64>
    %2370 = stablehlo.concatenate %2363, %2366, %2369, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2371 = stablehlo.slice %1951 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2372 = stablehlo.reshape %2371 : (tensor<1xf64>) -> tensor<f64>
    %2373 = stablehlo.slice %2242 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2374 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2375 = stablehlo.concatenate %2373, %2374, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2376 = stablehlo.slice %2375 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2377 = stablehlo.reshape %2376 : (tensor<1xf64>) -> tensor<f64>
    %2378 = stablehlo.multiply %2372, %2377 : tensor<f64>
    %2379 = stablehlo.slice %1951 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2380 = stablehlo.reshape %2379 : (tensor<1xf64>) -> tensor<f64>
    %2381 = stablehlo.slice %2375 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2382 = stablehlo.reshape %2381 : (tensor<1xf64>) -> tensor<f64>
    %2383 = stablehlo.multiply %2380, %2382 : tensor<f64>
    %2384 = stablehlo.add %2378, %2383 : tensor<f64>
    %2385 = stablehlo.slice %1951 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2386 = stablehlo.reshape %2385 : (tensor<1xf64>) -> tensor<f64>
    %2387 = stablehlo.slice %2375 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2388 = stablehlo.reshape %2387 : (tensor<1xf64>) -> tensor<f64>
    %2389 = stablehlo.multiply %2386, %2388 : tensor<f64>
    %2390 = stablehlo.add %2384, %2389 : tensor<f64>
    %2391 = stablehlo.slice %1951 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2392 = stablehlo.reshape %2391 : (tensor<1xf64>) -> tensor<f64>
    %2393 = stablehlo.slice %2375 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2394 = stablehlo.reshape %2393 : (tensor<1xf64>) -> tensor<f64>
    %2395 = stablehlo.multiply %2392, %2394 : tensor<f64>
    %2396 = stablehlo.subtract %2390, %2395 : tensor<f64>
    %2397 = stablehlo.reshape %2396 : (tensor<f64>) -> tensor<1xf64>
    %2398 = stablehlo.multiply %2372, %2394 : tensor<f64>
    %2399 = stablehlo.multiply %2380, %2388 : tensor<f64>
    %2400 = stablehlo.subtract %2398, %2399 : tensor<f64>
    %2401 = stablehlo.multiply %2386, %2382 : tensor<f64>
    %2402 = stablehlo.add %2400, %2401 : tensor<f64>
    %2403 = stablehlo.multiply %2392, %2377 : tensor<f64>
    %2404 = stablehlo.add %2402, %2403 : tensor<f64>
    %2405 = stablehlo.reshape %2404 : (tensor<f64>) -> tensor<1xf64>
    %2406 = stablehlo.multiply %2372, %2388 : tensor<f64>
    %2407 = stablehlo.multiply %2380, %2394 : tensor<f64>
    %2408 = stablehlo.add %2406, %2407 : tensor<f64>
    %2409 = stablehlo.multiply %2386, %2377 : tensor<f64>
    %2410 = stablehlo.subtract %2408, %2409 : tensor<f64>
    %2411 = stablehlo.multiply %2392, %2382 : tensor<f64>
    %2412 = stablehlo.add %2410, %2411 : tensor<f64>
    %2413 = stablehlo.reshape %2412 : (tensor<f64>) -> tensor<1xf64>
    %2414 = stablehlo.multiply %2372, %2382 : tensor<f64>
    %2415 = stablehlo.multiply %2380, %2377 : tensor<f64>
    %2416 = stablehlo.subtract %2414, %2415 : tensor<f64>
    %2417 = stablehlo.multiply %2386, %2394 : tensor<f64>
    %2418 = stablehlo.subtract %2416, %2417 : tensor<f64>
    %2419 = stablehlo.multiply %2392, %2388 : tensor<f64>
    %2420 = stablehlo.subtract %2418, %2419 : tensor<f64>
    %2421 = stablehlo.reshape %2420 : (tensor<f64>) -> tensor<1xf64>
    %2422 = stablehlo.concatenate %2397, %2405, %2413, %2421, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2423 = stablehlo.slice %2422 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2424 = stablehlo.reshape %2423 : (tensor<1xf64>) -> tensor<f64>
    %2425 = stablehlo.slice %1951 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2426 = stablehlo.reshape %2425 : (tensor<1xf64>) -> tensor<f64>
    %2427 = stablehlo.negate %2426 : tensor<f64>
    %2428 = stablehlo.reshape %2427 : (tensor<f64>) -> tensor<1xf64>
    %2429 = stablehlo.slice %1951 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2430 = stablehlo.reshape %2429 : (tensor<1xf64>) -> tensor<f64>
    %2431 = stablehlo.negate %2430 : tensor<f64>
    %2432 = stablehlo.reshape %2431 : (tensor<f64>) -> tensor<1xf64>
    %2433 = stablehlo.slice %1951 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2434 = stablehlo.reshape %2433 : (tensor<1xf64>) -> tensor<f64>
    %2435 = stablehlo.negate %2434 : tensor<f64>
    %2436 = stablehlo.reshape %2435 : (tensor<f64>) -> tensor<1xf64>
    %2437 = stablehlo.slice %1951 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2438 = stablehlo.reshape %2437 : (tensor<1xf64>) -> tensor<f64>
    %2439 = stablehlo.reshape %2438 : (tensor<f64>) -> tensor<1xf64>
    %2440 = stablehlo.concatenate %2428, %2432, %2436, %2439, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2441 = stablehlo.dot_general %1951, %1951, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2442 = stablehlo.broadcast_in_dim %2441, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2443 = stablehlo.divide %2440, %2442 : tensor<4xf64>
    %2444 = stablehlo.slice %2443 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2445 = stablehlo.reshape %2444 : (tensor<1xf64>) -> tensor<f64>
    %2446 = stablehlo.multiply %2424, %2445 : tensor<f64>
    %2447 = stablehlo.slice %2422 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2448 = stablehlo.reshape %2447 : (tensor<1xf64>) -> tensor<f64>
    %2449 = stablehlo.slice %2443 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2450 = stablehlo.reshape %2449 : (tensor<1xf64>) -> tensor<f64>
    %2451 = stablehlo.multiply %2448, %2450 : tensor<f64>
    %2452 = stablehlo.add %2446, %2451 : tensor<f64>
    %2453 = stablehlo.slice %2422 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2454 = stablehlo.reshape %2453 : (tensor<1xf64>) -> tensor<f64>
    %2455 = stablehlo.slice %2443 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2456 = stablehlo.reshape %2455 : (tensor<1xf64>) -> tensor<f64>
    %2457 = stablehlo.multiply %2454, %2456 : tensor<f64>
    %2458 = stablehlo.add %2452, %2457 : tensor<f64>
    %2459 = stablehlo.slice %2422 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2460 = stablehlo.reshape %2459 : (tensor<1xf64>) -> tensor<f64>
    %2461 = stablehlo.slice %2443 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2462 = stablehlo.reshape %2461 : (tensor<1xf64>) -> tensor<f64>
    %2463 = stablehlo.multiply %2460, %2462 : tensor<f64>
    %2464 = stablehlo.subtract %2458, %2463 : tensor<f64>
    %2465 = stablehlo.reshape %2464 : (tensor<f64>) -> tensor<1xf64>
    %2466 = stablehlo.multiply %2424, %2462 : tensor<f64>
    %2467 = stablehlo.multiply %2448, %2456 : tensor<f64>
    %2468 = stablehlo.subtract %2466, %2467 : tensor<f64>
    %2469 = stablehlo.multiply %2454, %2450 : tensor<f64>
    %2470 = stablehlo.add %2468, %2469 : tensor<f64>
    %2471 = stablehlo.multiply %2460, %2445 : tensor<f64>
    %2472 = stablehlo.add %2470, %2471 : tensor<f64>
    %2473 = stablehlo.reshape %2472 : (tensor<f64>) -> tensor<1xf64>
    %2474 = stablehlo.multiply %2424, %2456 : tensor<f64>
    %2475 = stablehlo.multiply %2448, %2462 : tensor<f64>
    %2476 = stablehlo.add %2474, %2475 : tensor<f64>
    %2477 = stablehlo.multiply %2454, %2445 : tensor<f64>
    %2478 = stablehlo.subtract %2476, %2477 : tensor<f64>
    %2479 = stablehlo.multiply %2460, %2450 : tensor<f64>
    %2480 = stablehlo.add %2478, %2479 : tensor<f64>
    %2481 = stablehlo.reshape %2480 : (tensor<f64>) -> tensor<1xf64>
    %2482 = stablehlo.multiply %2424, %2450 : tensor<f64>
    %2483 = stablehlo.multiply %2448, %2445 : tensor<f64>
    %2484 = stablehlo.subtract %2482, %2483 : tensor<f64>
    %2485 = stablehlo.multiply %2454, %2462 : tensor<f64>
    %2486 = stablehlo.subtract %2484, %2485 : tensor<f64>
    %2487 = stablehlo.multiply %2460, %2456 : tensor<f64>
    %2488 = stablehlo.subtract %2486, %2487 : tensor<f64>
    %2489 = stablehlo.reshape %2488 : (tensor<f64>) -> tensor<1xf64>
    %2490 = stablehlo.concatenate %2465, %2473, %2481, %2489, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2491 = stablehlo.slice %2490 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2492 = stablehlo.reshape %2491 : (tensor<1xf64>) -> tensor<f64>
    %2493 = stablehlo.reshape %2492 : (tensor<f64>) -> tensor<1xf64>
    %2494 = stablehlo.slice %2490 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2495 = stablehlo.reshape %2494 : (tensor<1xf64>) -> tensor<f64>
    %2496 = stablehlo.reshape %2495 : (tensor<f64>) -> tensor<1xf64>
    %2497 = stablehlo.slice %2490 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2498 = stablehlo.reshape %2497 : (tensor<1xf64>) -> tensor<f64>
    %2499 = stablehlo.reshape %2498 : (tensor<f64>) -> tensor<1xf64>
    %2500 = stablehlo.concatenate %2493, %2496, %2499, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2501 = stablehlo.concatenate %2370, %2500, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2502 = stablehlo.reshape %arg8 : (tensor<f64>) -> tensor<f64>
    %cst_31 = stablehlo.constant dense<0.16666666666666666> : tensor<f64>
    %2503 = stablehlo.multiply %cst_31, %2502 : tensor<f64>
    %2504 = stablehlo.broadcast_in_dim %2503, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %cst_32 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2505 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2506 = stablehlo.multiply %2505, %1251 : tensor<6xf64>
    %2507 = stablehlo.add %626, %2506 : tensor<6xf64>
    %cst_33 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2508 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2509 = stablehlo.multiply %2508, %1876 : tensor<6xf64>
    %2510 = stablehlo.add %2507, %2509 : tensor<6xf64>
    %2511 = stablehlo.add %2510, %2501 : tensor<6xf64>
    %2512 = stablehlo.multiply %2504, %2511 : tensor<6xf64>
    %2513 = stablehlo.add %1, %2512 : tensor<6xf64>
    %2514 = stablehlo.slice %arg3 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2515 = stablehlo.broadcast_in_dim %2503, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %cst_34 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2516 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2517 = stablehlo.multiply %2516, %697 : tensor<6xf64>
    %2518 = stablehlo.add %72, %2517 : tensor<6xf64>
    %cst_35 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2519 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2520 = stablehlo.multiply %2519, %1322 : tensor<6xf64>
    %2521 = stablehlo.add %2518, %2520 : tensor<6xf64>
    %2522 = stablehlo.add %2521, %1947 : tensor<6xf64>
    %2523 = stablehlo.multiply %2515, %2522 : tensor<6xf64>
    %2524 = stablehlo.slice %2523 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_36 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2525 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2526 = stablehlo.divide %2524, %2525 : tensor<3xf64>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2527 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2528 = stablehlo.concatenate %2526, %2527, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2529 = stablehlo.slice %2528 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2530 = stablehlo.reshape %2529 : (tensor<1xf64>) -> tensor<f64>
    %2531 = stablehlo.slice %2514 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2532 = stablehlo.reshape %2531 : (tensor<1xf64>) -> tensor<f64>
    %2533 = stablehlo.multiply %2530, %2532 : tensor<f64>
    %2534 = stablehlo.slice %2528 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2535 = stablehlo.reshape %2534 : (tensor<1xf64>) -> tensor<f64>
    %2536 = stablehlo.slice %2514 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2537 = stablehlo.reshape %2536 : (tensor<1xf64>) -> tensor<f64>
    %2538 = stablehlo.multiply %2535, %2537 : tensor<f64>
    %2539 = stablehlo.add %2533, %2538 : tensor<f64>
    %2540 = stablehlo.slice %2528 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2541 = stablehlo.reshape %2540 : (tensor<1xf64>) -> tensor<f64>
    %2542 = stablehlo.slice %2514 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2543 = stablehlo.reshape %2542 : (tensor<1xf64>) -> tensor<f64>
    %2544 = stablehlo.multiply %2541, %2543 : tensor<f64>
    %2545 = stablehlo.add %2539, %2544 : tensor<f64>
    %2546 = stablehlo.slice %2528 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2547 = stablehlo.reshape %2546 : (tensor<1xf64>) -> tensor<f64>
    %2548 = stablehlo.slice %2514 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2549 = stablehlo.reshape %2548 : (tensor<1xf64>) -> tensor<f64>
    %2550 = stablehlo.multiply %2547, %2549 : tensor<f64>
    %2551 = stablehlo.subtract %2545, %2550 : tensor<f64>
    %2552 = stablehlo.reshape %2551 : (tensor<f64>) -> tensor<1xf64>
    %2553 = stablehlo.multiply %2530, %2549 : tensor<f64>
    %2554 = stablehlo.multiply %2535, %2543 : tensor<f64>
    %2555 = stablehlo.subtract %2553, %2554 : tensor<f64>
    %2556 = stablehlo.multiply %2541, %2537 : tensor<f64>
    %2557 = stablehlo.add %2555, %2556 : tensor<f64>
    %2558 = stablehlo.multiply %2547, %2532 : tensor<f64>
    %2559 = stablehlo.add %2557, %2558 : tensor<f64>
    %2560 = stablehlo.reshape %2559 : (tensor<f64>) -> tensor<1xf64>
    %2561 = stablehlo.multiply %2530, %2543 : tensor<f64>
    %2562 = stablehlo.multiply %2535, %2549 : tensor<f64>
    %2563 = stablehlo.add %2561, %2562 : tensor<f64>
    %2564 = stablehlo.multiply %2541, %2532 : tensor<f64>
    %2565 = stablehlo.subtract %2563, %2564 : tensor<f64>
    %2566 = stablehlo.multiply %2547, %2537 : tensor<f64>
    %2567 = stablehlo.add %2565, %2566 : tensor<f64>
    %2568 = stablehlo.reshape %2567 : (tensor<f64>) -> tensor<1xf64>
    %2569 = stablehlo.multiply %2530, %2537 : tensor<f64>
    %2570 = stablehlo.multiply %2535, %2532 : tensor<f64>
    %2571 = stablehlo.subtract %2569, %2570 : tensor<f64>
    %2572 = stablehlo.multiply %2541, %2549 : tensor<f64>
    %2573 = stablehlo.subtract %2571, %2572 : tensor<f64>
    %2574 = stablehlo.multiply %2547, %2543 : tensor<f64>
    %2575 = stablehlo.subtract %2573, %2574 : tensor<f64>
    %2576 = stablehlo.reshape %2575 : (tensor<f64>) -> tensor<1xf64>
    %2577 = stablehlo.concatenate %2552, %2560, %2568, %2576, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2578 = stablehlo.add %2514, %2577 : tensor<4xf64>
    %2579 = stablehlo.dot_general %2578, %2578, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2580 = stablehlo.sqrt %2579 : tensor<f64>
    %2581 = stablehlo.broadcast_in_dim %2580, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2582 = stablehlo.divide %2578, %2581 : tensor<4xf64>
    %2583 = stablehlo.slice %arg3 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %2584 = stablehlo.slice %2523 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %2585 = stablehlo.add %2583, %2584 : tensor<3xf64>
    %2586 = stablehlo.concatenate %2582, %2585, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %2587 = stablehlo.add %arg0, %c : tensor<i64>
    return %2501, %arg8, %2587, %0, %arg1, %2513, %2586, %arg6, %1950 : tensor<6xf64>, tensor<f64>, tensor<i64>, tensor<3xf64>, tensor<i64>, tensor<6xf64>, tensor<7xf64>, tensor<7xf64>, tensor<6xf64>
  }
  func.func private @inner(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<32> : tensor<i64>
    %0 = stablehlo.shift_right_logical %arg0, %c : tensor<i64>
    %1 = stablehlo.convert %0 : (tensor<i64>) -> tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %c_0 = stablehlo.constant dense<4294967295> : tensor<i64>
    %3 = stablehlo.and %arg0, %c_0 : tensor<i64>
    %4 = stablehlo.convert %3 : (tensor<i64>) -> tensor<ui32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %6 = stablehlo.concatenate %2, %5, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %7 = call @_normal(%6) : (tensor<2xui32>) -> tensor<3xf64>
    return %7 : tensor<3xf64>
  }
  func.func private @_normal(%arg0: tensor<2xui32>) -> tensor<3xf64> {
    %0 = call @_normal_real(%arg0) : (tensor<2xui32>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
  func.func private @_normal_real(%arg0: tensor<2xui32>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<-0.99999999999999988> : tensor<f64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = call @_uniform(%arg0, %cst, %cst_0) : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<3xf64>
    %1 = chlo.erf_inv %0 : tensor<3xf64> -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<1.4142135623730951> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.multiply %2, %1 : tensor<3xf64>
    return %3 : tensor<3xf64>
  }
  func.func private @_uniform(%arg0: tensor<2xui32>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %3 = stablehlo.reshape %2 : (tensor<1xui32>) -> tensor<ui32>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %5 = stablehlo.reshape %4 : (tensor<1xui32>) -> tensor<ui32>
    %6 = stablehlo.iota dim = 0 : tensor<3xui64>
    %c = stablehlo.constant dense<1> : tensor<ui64>
    %7 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<3xui64>
    %8 = stablehlo.multiply %7, %6 : tensor<3xui64>
    %c_0 = stablehlo.constant dense<32> : tensor<ui64>
    %9 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<ui64>) -> tensor<3xui64>
    %10 = stablehlo.shift_right_logical %8, %9 : tensor<3xui64>
    %11 = stablehlo.convert %8 : (tensor<3xui64>) -> tensor<3xui32>
    %12 = stablehlo.convert %10 : (tensor<3xui64>) -> tensor<3xui32>
    %13:2 = call @threefry2x32(%3, %5, %12, %11) : (tensor<ui32>, tensor<ui32>, tensor<3xui32>, tensor<3xui32>) -> (tensor<3xui32>, tensor<3xui32>)
    %14 = stablehlo.convert %13#0 : (tensor<3xui32>) -> tensor<3xui64>
    %15 = stablehlo.convert %13#1 : (tensor<3xui32>) -> tensor<3xui64>
    %c_1 = stablehlo.constant dense<32> : tensor<ui64>
    %16 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<3xui64>
    %17 = stablehlo.shift_left %14, %16 : tensor<3xui64>
    %18 = stablehlo.or %17, %15 : tensor<3xui64>
    %c_2 = stablehlo.constant dense<12> : tensor<ui64>
    %19 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<ui64>) -> tensor<3xui64>
    %20 = stablehlo.shift_right_logical %18, %19 : tensor<3xui64>
    %c_3 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %21 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<ui64>) -> tensor<3xui64>
    %22 = stablehlo.or %20, %21 : tensor<3xui64>
    %23 = stablehlo.bitcast_convert %22 : (tensor<3xui64>) -> tensor<3xf64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %25 = stablehlo.subtract %23, %24 : tensor<3xf64>
    %26 = stablehlo.subtract %1, %0 : tensor<1xf64>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0] : (tensor<1xf64>) -> tensor<3xf64>
    %28 = stablehlo.multiply %25, %27 : tensor<3xf64>
    %29 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<3xf64>
    %30 = stablehlo.add %28, %29 : tensor<3xf64>
    %31 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<3xf64>
    %32 = stablehlo.maximum %31, %30 : tensor<3xf64>
    return %32 : tensor<3xf64>
  }
  func.func private @threefry2x32(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<3xui32>, %arg3: tensor<3xui32>) -> (tensor<3xui32>, tensor<3xui32>) {
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %1 = stablehlo.xor %0, %c_1 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<3xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<3xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %6:9 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c_2, %iterArg_5 = %3, %iterArg_6 = %5, %iterArg_7 = %arg1, %iterArg_8 = %1, %iterArg_9 = %arg0, %iterArg_10 = %c, %iterArg_11 = %c_0) : tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    cond {
      %c_12 = stablehlo.constant dense<5> : tensor<i64>
      %7 = stablehlo.compare  LT, %iterArg, %c_12,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %7 : tensor<i1>
    } do {
      %7:8 = func.call @closed_call(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_12 = stablehlo.constant dense<1> : tensor<i64>
      %8 = stablehlo.add %iterArg, %c_12 : tensor<i64>
      stablehlo.return %8, %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6, %7#7 : tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    return %6#2, %6#3 : tensor<3xui32>, tensor<3xui32>
  }
  func.func private @closed_call(%arg0: tensor<i64>, %arg1: tensor<3xui32>, %arg2: tensor<3xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<3xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<3xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<3xui32>
    %9 = stablehlo.or %5, %8 : tensor<3xui32>
    %10 = stablehlo.xor %3, %9 : tensor<3xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<3xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<3xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<3xui32>
    %19 = stablehlo.or %15, %18 : tensor<3xui32>
    %20 = stablehlo.xor %13, %19 : tensor<3xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<3xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<3xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<3xui32>
    %29 = stablehlo.or %25, %28 : tensor<3xui32>
    %30 = stablehlo.xor %23, %29 : tensor<3xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<3xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<3xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<3xui32>
    %39 = stablehlo.or %35, %38 : tensor<3xui32>
    %40 = stablehlo.xor %33, %39 : tensor<3xui32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %42 = stablehlo.add %33, %41 : tensor<3xui32>
    %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %44 = stablehlo.add %40, %43 : tensor<3xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %46 = stablehlo.convert %45 : (tensor<i64>) -> tensor<ui32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<3xui32>
    %48 = stablehlo.add %44, %47 : tensor<3xui32>
    return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
  }
  func.func private @inner_20(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[1.000000e+00, 1.000000e+00, -1.000000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg0 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %0 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
    %3 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %4 = stablehlo.slice %3 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.maximum %2, %5 : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.compare  LT, %6, %cst_0,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %8 = stablehlo.convert %7 : (tensor<i1>) -> tensor<i32>
    %9 = "stablehlo.case"(%8) ({
      stablehlo.return %arg1 : tensor<6xf64>
    }, {
      %10 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
      %11 = stablehlo.multiply %10, %cst : tensor<3xf64>
      %cst_1 = stablehlo.constant dense<8.500000e-01> : tensor<f64>
      %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %13 = stablehlo.multiply %11, %12 : tensor<3xf64>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %14 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %15 = stablehlo.concatenate %14, %13, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %16 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %17 = stablehlo.concatenate %16, %13, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
      stablehlo.return %17 : tensor<6xf64>
    }) : (tensor<i32>) -> tensor<6xf64>
    return %9 : tensor<6xf64>
  }
  func.func private @inner_55(%arg0: tensor<6xf64>, %arg1: tensor<7xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, -9.810000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg1 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.multiply %cst, %2 : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %5 = stablehlo.concatenate %4, %3, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %6 = stablehlo.add %arg0, %5 : tensor<6xf64>
    return %6 : tensor<6xf64>
  }
  func.func private @inner_57(%arg0: tensor<3xf64>, %arg1: tensor<6xf64>, %arg2: tensor<6xf64>) -> tensor<6xf64> {
    %0 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.subtract %arg0, %0 : tensor<3xf64>
    %2 = call @norm(%1) : (tensor<3xf64>) -> tensor<f64>
    %3 = stablehlo.multiply %2, %2 : tensor<f64>
    %cst = stablehlo.constant dense<6.125000e-01> : tensor<f64>
    %4 = stablehlo.multiply %cst, %3 : tensor<f64>
    %cst_0 = stablehlo.constant dense<0.25132000000000004> : tensor<f64>
    %5 = stablehlo.multiply %4, %cst_0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %6 = stablehlo.multiply %cst_1, %5 : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.divide %1, %7 : tensor<3xf64>
    %9 = stablehlo.slice %arg2 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %10 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %11 = stablehlo.multiply %10, %8 : tensor<3xf64>
    %12 = stablehlo.add %9, %11 : tensor<3xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %13 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %14 = stablehlo.concatenate %13, %12, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    return %14 : tensor<6xf64>
  }
  func.func private @norm(%arg0: tensor<3xf64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<3xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.sqrt %1 : tensor<f64>
    return %2 : tensor<f64>
  }
}
