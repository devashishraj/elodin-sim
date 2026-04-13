module @module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<f64>, %arg2: tensor<3xf64>, %arg3: tensor<4xf64>, %arg4: tensor<3xf64>, %arg5: tensor<7xf64>, %arg6: tensor<3xf64>, %arg7: tensor<3xf64>, %arg8: tensor<3x3xf64>, %arg9: tensor<4xf64>, %arg10: tensor<4xf64>, %arg11: tensor<6xf64>, %arg12: tensor<7xf64>, %arg13: tensor<6xf64>, %arg14: tensor<3xf64>, %arg15: tensor<4xf64>, %arg16: tensor<4xf64>, %arg17: tensor<4xf64>, %arg18: tensor<6xf64>, %arg19: tensor<6xf64>, %arg20: tensor<i64>, %arg21: tensor<3xf64>, %arg22: tensor<4x3xf64>, %arg23: tensor<4x3xf64>, %arg24: tensor<3xf64>, %arg25: tensor<3xf64>, %arg26: tensor<f64>, %arg27: tensor<3xf64>, %arg28: tensor<3xf64>, %arg29: tensor<3xf64>, %arg30: tensor<4xf64>) -> (tensor<6xf64> {jax.result_info = "result[0]"}, tensor<3xf64> {jax.result_info = "result[1]"}, tensor<f64> {jax.result_info = "result[2]"}, tensor<3xf64> {jax.result_info = "result[3]"}, tensor<3xf64> {jax.result_info = "result[4]"}, tensor<3xf64> {jax.result_info = "result[5]"}, tensor<i64> {jax.result_info = "result[6]"}, tensor<f64> {jax.result_info = "result[7]"}, tensor<3x3xf64> {jax.result_info = "result[8]"}, tensor<4xf64> {jax.result_info = "result[9]"}, tensor<3xf64> {jax.result_info = "result[10]"}, tensor<4xf64> {jax.result_info = "result[11]"}, tensor<3xf64> {jax.result_info = "result[12]"}, tensor<3xf64> {jax.result_info = "result[13]"}, tensor<4xf64> {jax.result_info = "result[14]"}, tensor<6xf64> {jax.result_info = "result[15]"}, tensor<4xf64> {jax.result_info = "result[16]"}, tensor<4x3xf64> {jax.result_info = "result[17]"}, tensor<3xf64> {jax.result_info = "result[18]"}, tensor<4x3xf64> {jax.result_info = "result[19]"}, tensor<7xf64> {jax.result_info = "result[20]"}, tensor<3xf64> {jax.result_info = "result[21]"}, tensor<4xf64> {jax.result_info = "result[22]"}, tensor<7xf64> {jax.result_info = "result[23]"}, tensor<6xf64> {jax.result_info = "result[24]"}, tensor<6xf64> {jax.result_info = "result[25]"}, tensor<3xf64> {jax.result_info = "result[26]"}, tensor<3xf64> {jax.result_info = "result[27]"}, tensor<4xf64> {jax.result_info = "result[28]"}, tensor<4xf64> {jax.result_info = "result[29]"}, tensor<i64> {jax.result_info = "result[30]"}) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = call @inner(%0, %arg1, %arg2) : (tensor<i64>, tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    %2:2 = call @inner_2(%1, %arg3, %arg4) : (tensor<3xf64>, tensor<4xf64>, tensor<3xf64>) -> (tensor<4xf64>, tensor<3xf64>)
    %3 = call @inner_39(%arg5, %arg6, %2#0, %2#1, %arg7) : (tensor<7xf64>, tensor<3xf64>, tensor<4xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %4 = call @inner_62(%arg8, %3, %arg6) : (tensor<3x3xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3x3xf64>
    %5 = call @inner_66(%4, %arg9) : (tensor<3x3xf64>, tensor<4xf64>) -> tensor<4xf64>
    %6 = call @inner_69(%5, %arg10) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %8:3 = call @inner_96(%6, %arg15, %arg16, %arg17) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    %9 = call @inner_128(%arg12, %7) : (tensor<7xf64>, tensor<6xf64>) -> tensor<6xf64>
    %10 = call @inner_133(%arg13, %arg14) : (tensor<6xf64>, tensor<3xf64>) -> tensor<3xf64>
    %11 = call @inner_136(%8#0, %8#1, %arg18) : (tensor<4xf64>, tensor<4xf64>, tensor<6xf64>) -> tensor<6xf64>
    %12 = call @inner_140(%11, %10, %arg5, %9) : (tensor<6xf64>, tensor<3xf64>, tensor<7xf64>, tensor<6xf64>) -> tensor<6xf64>
    %13 = stablehlo.slice %arg5 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %14 = stablehlo.slice %13 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.slice %13 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %17 = stablehlo.reshape %16 : (tensor<1xf64>) -> tensor<f64>
    %18 = stablehlo.negate %17 : tensor<f64>
    %19 = stablehlo.reshape %18 : (tensor<f64>) -> tensor<1xf64>
    %20 = stablehlo.slice %13 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.negate %21 : tensor<f64>
    %23 = stablehlo.reshape %22 : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.slice %13 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.negate %25 : tensor<f64>
    %27 = stablehlo.reshape %26 : (tensor<f64>) -> tensor<1xf64>
    %28 = stablehlo.slice %13 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
    %30 = stablehlo.reshape %29 : (tensor<f64>) -> tensor<1xf64>
    %31 = stablehlo.concatenate %19, %23, %27, %30, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %32 = stablehlo.dot_general %13, %13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %34 = stablehlo.divide %31, %33 : tensor<4xf64>
    %35 = stablehlo.slice %34 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1xf64>) -> tensor<f64>
    %37 = stablehlo.slice %12 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %38 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %39 = stablehlo.concatenate %37, %38, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %40 = stablehlo.slice %39 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.multiply %36, %41 : tensor<f64>
    %43 = stablehlo.slice %34 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %44 = stablehlo.reshape %43 : (tensor<1xf64>) -> tensor<f64>
    %45 = stablehlo.slice %39 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %46 = stablehlo.reshape %45 : (tensor<1xf64>) -> tensor<f64>
    %47 = stablehlo.multiply %44, %46 : tensor<f64>
    %48 = stablehlo.add %42, %47 : tensor<f64>
    %49 = stablehlo.slice %34 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %50 = stablehlo.reshape %49 : (tensor<1xf64>) -> tensor<f64>
    %51 = stablehlo.slice %39 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %52 = stablehlo.reshape %51 : (tensor<1xf64>) -> tensor<f64>
    %53 = stablehlo.multiply %50, %52 : tensor<f64>
    %54 = stablehlo.add %48, %53 : tensor<f64>
    %55 = stablehlo.slice %34 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %56 = stablehlo.reshape %55 : (tensor<1xf64>) -> tensor<f64>
    %57 = stablehlo.slice %39 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %58 = stablehlo.reshape %57 : (tensor<1xf64>) -> tensor<f64>
    %59 = stablehlo.multiply %56, %58 : tensor<f64>
    %60 = stablehlo.subtract %54, %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.multiply %36, %58 : tensor<f64>
    %63 = stablehlo.multiply %44, %52 : tensor<f64>
    %64 = stablehlo.subtract %62, %63 : tensor<f64>
    %65 = stablehlo.multiply %50, %46 : tensor<f64>
    %66 = stablehlo.add %64, %65 : tensor<f64>
    %67 = stablehlo.multiply %56, %41 : tensor<f64>
    %68 = stablehlo.add %66, %67 : tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.multiply %36, %52 : tensor<f64>
    %71 = stablehlo.multiply %44, %58 : tensor<f64>
    %72 = stablehlo.add %70, %71 : tensor<f64>
    %73 = stablehlo.multiply %50, %41 : tensor<f64>
    %74 = stablehlo.subtract %72, %73 : tensor<f64>
    %75 = stablehlo.multiply %56, %46 : tensor<f64>
    %76 = stablehlo.add %74, %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.multiply %36, %46 : tensor<f64>
    %79 = stablehlo.multiply %44, %41 : tensor<f64>
    %80 = stablehlo.subtract %78, %79 : tensor<f64>
    %81 = stablehlo.multiply %50, %58 : tensor<f64>
    %82 = stablehlo.subtract %80, %81 : tensor<f64>
    %83 = stablehlo.multiply %56, %52 : tensor<f64>
    %84 = stablehlo.subtract %82, %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.concatenate %61, %69, %77, %85, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %87 = stablehlo.slice %86 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %88 = stablehlo.reshape %87 : (tensor<1xf64>) -> tensor<f64>
    %89 = stablehlo.slice %34 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %90 = stablehlo.reshape %89 : (tensor<1xf64>) -> tensor<f64>
    %91 = stablehlo.negate %90 : tensor<f64>
    %92 = stablehlo.reshape %91 : (tensor<f64>) -> tensor<1xf64>
    %93 = stablehlo.slice %34 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %94 = stablehlo.reshape %93 : (tensor<1xf64>) -> tensor<f64>
    %95 = stablehlo.negate %94 : tensor<f64>
    %96 = stablehlo.reshape %95 : (tensor<f64>) -> tensor<1xf64>
    %97 = stablehlo.slice %34 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.negate %98 : tensor<f64>
    %100 = stablehlo.reshape %99 : (tensor<f64>) -> tensor<1xf64>
    %101 = stablehlo.slice %34 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %102 = stablehlo.reshape %101 : (tensor<1xf64>) -> tensor<f64>
    %103 = stablehlo.reshape %102 : (tensor<f64>) -> tensor<1xf64>
    %104 = stablehlo.concatenate %92, %96, %100, %103, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %105 = stablehlo.dot_general %34, %34, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %106 = stablehlo.broadcast_in_dim %105, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %107 = stablehlo.divide %104, %106 : tensor<4xf64>
    %108 = stablehlo.slice %107 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.multiply %88, %109 : tensor<f64>
    %111 = stablehlo.slice %86 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %112 = stablehlo.reshape %111 : (tensor<1xf64>) -> tensor<f64>
    %113 = stablehlo.slice %107 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %114 = stablehlo.reshape %113 : (tensor<1xf64>) -> tensor<f64>
    %115 = stablehlo.multiply %112, %114 : tensor<f64>
    %116 = stablehlo.add %110, %115 : tensor<f64>
    %117 = stablehlo.slice %86 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %118 = stablehlo.reshape %117 : (tensor<1xf64>) -> tensor<f64>
    %119 = stablehlo.slice %107 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %120 = stablehlo.reshape %119 : (tensor<1xf64>) -> tensor<f64>
    %121 = stablehlo.multiply %118, %120 : tensor<f64>
    %122 = stablehlo.add %116, %121 : tensor<f64>
    %123 = stablehlo.slice %86 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %124 = stablehlo.reshape %123 : (tensor<1xf64>) -> tensor<f64>
    %125 = stablehlo.slice %107 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %126 = stablehlo.reshape %125 : (tensor<1xf64>) -> tensor<f64>
    %127 = stablehlo.multiply %124, %126 : tensor<f64>
    %128 = stablehlo.subtract %122, %127 : tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.multiply %88, %126 : tensor<f64>
    %131 = stablehlo.multiply %112, %120 : tensor<f64>
    %132 = stablehlo.subtract %130, %131 : tensor<f64>
    %133 = stablehlo.multiply %118, %114 : tensor<f64>
    %134 = stablehlo.add %132, %133 : tensor<f64>
    %135 = stablehlo.multiply %124, %109 : tensor<f64>
    %136 = stablehlo.add %134, %135 : tensor<f64>
    %137 = stablehlo.reshape %136 : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.multiply %88, %120 : tensor<f64>
    %139 = stablehlo.multiply %112, %126 : tensor<f64>
    %140 = stablehlo.add %138, %139 : tensor<f64>
    %141 = stablehlo.multiply %118, %109 : tensor<f64>
    %142 = stablehlo.subtract %140, %141 : tensor<f64>
    %143 = stablehlo.multiply %124, %114 : tensor<f64>
    %144 = stablehlo.add %142, %143 : tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.multiply %88, %114 : tensor<f64>
    %147 = stablehlo.multiply %112, %109 : tensor<f64>
    %148 = stablehlo.subtract %146, %147 : tensor<f64>
    %149 = stablehlo.multiply %118, %126 : tensor<f64>
    %150 = stablehlo.subtract %148, %149 : tensor<f64>
    %151 = stablehlo.multiply %124, %120 : tensor<f64>
    %152 = stablehlo.subtract %150, %151 : tensor<f64>
    %153 = stablehlo.reshape %152 : (tensor<f64>) -> tensor<1xf64>
    %154 = stablehlo.concatenate %129, %137, %145, %153, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %155 = stablehlo.slice %154 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %156 = stablehlo.reshape %155 : (tensor<1xf64>) -> tensor<f64>
    %157 = stablehlo.reshape %156 : (tensor<f64>) -> tensor<1xf64>
    %158 = stablehlo.slice %154 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %159 = stablehlo.reshape %158 : (tensor<1xf64>) -> tensor<f64>
    %160 = stablehlo.reshape %159 : (tensor<f64>) -> tensor<1xf64>
    %161 = stablehlo.slice %154 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %162 = stablehlo.reshape %161 : (tensor<1xf64>) -> tensor<f64>
    %163 = stablehlo.reshape %162 : (tensor<f64>) -> tensor<1xf64>
    %164 = stablehlo.concatenate %157, %160, %163, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %165 = stablehlo.slice %34 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %12 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %168 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %169 = stablehlo.concatenate %167, %168, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %170 = stablehlo.slice %169 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %171 = stablehlo.reshape %170 : (tensor<1xf64>) -> tensor<f64>
    %172 = stablehlo.multiply %166, %171 : tensor<f64>
    %173 = stablehlo.slice %34 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %174 = stablehlo.reshape %173 : (tensor<1xf64>) -> tensor<f64>
    %175 = stablehlo.slice %169 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %176 = stablehlo.reshape %175 : (tensor<1xf64>) -> tensor<f64>
    %177 = stablehlo.multiply %174, %176 : tensor<f64>
    %178 = stablehlo.add %172, %177 : tensor<f64>
    %179 = stablehlo.slice %34 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %180 = stablehlo.reshape %179 : (tensor<1xf64>) -> tensor<f64>
    %181 = stablehlo.slice %169 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %182 = stablehlo.reshape %181 : (tensor<1xf64>) -> tensor<f64>
    %183 = stablehlo.multiply %180, %182 : tensor<f64>
    %184 = stablehlo.add %178, %183 : tensor<f64>
    %185 = stablehlo.slice %34 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %186 = stablehlo.reshape %185 : (tensor<1xf64>) -> tensor<f64>
    %187 = stablehlo.slice %169 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %188 = stablehlo.reshape %187 : (tensor<1xf64>) -> tensor<f64>
    %189 = stablehlo.multiply %186, %188 : tensor<f64>
    %190 = stablehlo.subtract %184, %189 : tensor<f64>
    %191 = stablehlo.reshape %190 : (tensor<f64>) -> tensor<1xf64>
    %192 = stablehlo.multiply %166, %188 : tensor<f64>
    %193 = stablehlo.multiply %174, %182 : tensor<f64>
    %194 = stablehlo.subtract %192, %193 : tensor<f64>
    %195 = stablehlo.multiply %180, %176 : tensor<f64>
    %196 = stablehlo.add %194, %195 : tensor<f64>
    %197 = stablehlo.multiply %186, %171 : tensor<f64>
    %198 = stablehlo.add %196, %197 : tensor<f64>
    %199 = stablehlo.reshape %198 : (tensor<f64>) -> tensor<1xf64>
    %200 = stablehlo.multiply %166, %182 : tensor<f64>
    %201 = stablehlo.multiply %174, %188 : tensor<f64>
    %202 = stablehlo.add %200, %201 : tensor<f64>
    %203 = stablehlo.multiply %180, %171 : tensor<f64>
    %204 = stablehlo.subtract %202, %203 : tensor<f64>
    %205 = stablehlo.multiply %186, %176 : tensor<f64>
    %206 = stablehlo.add %204, %205 : tensor<f64>
    %207 = stablehlo.reshape %206 : (tensor<f64>) -> tensor<1xf64>
    %208 = stablehlo.multiply %166, %176 : tensor<f64>
    %209 = stablehlo.multiply %174, %171 : tensor<f64>
    %210 = stablehlo.subtract %208, %209 : tensor<f64>
    %211 = stablehlo.multiply %180, %188 : tensor<f64>
    %212 = stablehlo.subtract %210, %211 : tensor<f64>
    %213 = stablehlo.multiply %186, %182 : tensor<f64>
    %214 = stablehlo.subtract %212, %213 : tensor<f64>
    %215 = stablehlo.reshape %214 : (tensor<f64>) -> tensor<1xf64>
    %216 = stablehlo.concatenate %191, %199, %207, %215, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %217 = stablehlo.slice %216 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %218 = stablehlo.reshape %217 : (tensor<1xf64>) -> tensor<f64>
    %219 = stablehlo.slice %34 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %220 = stablehlo.reshape %219 : (tensor<1xf64>) -> tensor<f64>
    %221 = stablehlo.negate %220 : tensor<f64>
    %222 = stablehlo.reshape %221 : (tensor<f64>) -> tensor<1xf64>
    %223 = stablehlo.slice %34 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %224 = stablehlo.reshape %223 : (tensor<1xf64>) -> tensor<f64>
    %225 = stablehlo.negate %224 : tensor<f64>
    %226 = stablehlo.reshape %225 : (tensor<f64>) -> tensor<1xf64>
    %227 = stablehlo.slice %34 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %228 = stablehlo.reshape %227 : (tensor<1xf64>) -> tensor<f64>
    %229 = stablehlo.negate %228 : tensor<f64>
    %230 = stablehlo.reshape %229 : (tensor<f64>) -> tensor<1xf64>
    %231 = stablehlo.slice %34 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %232 = stablehlo.reshape %231 : (tensor<1xf64>) -> tensor<f64>
    %233 = stablehlo.reshape %232 : (tensor<f64>) -> tensor<1xf64>
    %234 = stablehlo.concatenate %222, %226, %230, %233, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %235 = stablehlo.dot_general %34, %34, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %236 = stablehlo.broadcast_in_dim %235, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %237 = stablehlo.divide %234, %236 : tensor<4xf64>
    %238 = stablehlo.slice %237 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %239 = stablehlo.reshape %238 : (tensor<1xf64>) -> tensor<f64>
    %240 = stablehlo.multiply %218, %239 : tensor<f64>
    %241 = stablehlo.slice %216 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %242 = stablehlo.reshape %241 : (tensor<1xf64>) -> tensor<f64>
    %243 = stablehlo.slice %237 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %244 = stablehlo.reshape %243 : (tensor<1xf64>) -> tensor<f64>
    %245 = stablehlo.multiply %242, %244 : tensor<f64>
    %246 = stablehlo.add %240, %245 : tensor<f64>
    %247 = stablehlo.slice %216 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %248 = stablehlo.reshape %247 : (tensor<1xf64>) -> tensor<f64>
    %249 = stablehlo.slice %237 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %250 = stablehlo.reshape %249 : (tensor<1xf64>) -> tensor<f64>
    %251 = stablehlo.multiply %248, %250 : tensor<f64>
    %252 = stablehlo.add %246, %251 : tensor<f64>
    %253 = stablehlo.slice %216 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %254 = stablehlo.reshape %253 : (tensor<1xf64>) -> tensor<f64>
    %255 = stablehlo.slice %237 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %256 = stablehlo.reshape %255 : (tensor<1xf64>) -> tensor<f64>
    %257 = stablehlo.multiply %254, %256 : tensor<f64>
    %258 = stablehlo.subtract %252, %257 : tensor<f64>
    %259 = stablehlo.reshape %258 : (tensor<f64>) -> tensor<1xf64>
    %260 = stablehlo.multiply %218, %256 : tensor<f64>
    %261 = stablehlo.multiply %242, %250 : tensor<f64>
    %262 = stablehlo.subtract %260, %261 : tensor<f64>
    %263 = stablehlo.multiply %248, %244 : tensor<f64>
    %264 = stablehlo.add %262, %263 : tensor<f64>
    %265 = stablehlo.multiply %254, %239 : tensor<f64>
    %266 = stablehlo.add %264, %265 : tensor<f64>
    %267 = stablehlo.reshape %266 : (tensor<f64>) -> tensor<1xf64>
    %268 = stablehlo.multiply %218, %250 : tensor<f64>
    %269 = stablehlo.multiply %242, %256 : tensor<f64>
    %270 = stablehlo.add %268, %269 : tensor<f64>
    %271 = stablehlo.multiply %248, %239 : tensor<f64>
    %272 = stablehlo.subtract %270, %271 : tensor<f64>
    %273 = stablehlo.multiply %254, %244 : tensor<f64>
    %274 = stablehlo.add %272, %273 : tensor<f64>
    %275 = stablehlo.reshape %274 : (tensor<f64>) -> tensor<1xf64>
    %276 = stablehlo.multiply %218, %244 : tensor<f64>
    %277 = stablehlo.multiply %242, %239 : tensor<f64>
    %278 = stablehlo.subtract %276, %277 : tensor<f64>
    %279 = stablehlo.multiply %248, %256 : tensor<f64>
    %280 = stablehlo.subtract %278, %279 : tensor<f64>
    %281 = stablehlo.multiply %254, %250 : tensor<f64>
    %282 = stablehlo.subtract %280, %281 : tensor<f64>
    %283 = stablehlo.reshape %282 : (tensor<f64>) -> tensor<1xf64>
    %284 = stablehlo.concatenate %259, %267, %275, %283, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %285 = stablehlo.slice %284 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %286 = stablehlo.reshape %285 : (tensor<1xf64>) -> tensor<f64>
    %287 = stablehlo.reshape %286 : (tensor<f64>) -> tensor<1xf64>
    %288 = stablehlo.slice %284 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %289 = stablehlo.reshape %288 : (tensor<1xf64>) -> tensor<f64>
    %290 = stablehlo.reshape %289 : (tensor<f64>) -> tensor<1xf64>
    %291 = stablehlo.slice %284 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %292 = stablehlo.reshape %291 : (tensor<1xf64>) -> tensor<f64>
    %293 = stablehlo.reshape %292 : (tensor<f64>) -> tensor<1xf64>
    %294 = stablehlo.concatenate %287, %290, %293, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %295 = stablehlo.concatenate %164, %294, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %296 = stablehlo.slice %295 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %297 = stablehlo.slice %arg12 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %298 = stablehlo.divide %296, %297 : tensor<3xf64>
    %299 = stablehlo.slice %295 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %300 = stablehlo.slice %arg12 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %301 = stablehlo.reshape %300 : (tensor<1xf64>) -> tensor<f64>
    %302 = stablehlo.broadcast_in_dim %301, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %303 = stablehlo.divide %299, %302 : tensor<3xf64>
    %304 = stablehlo.concatenate %298, %303, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %305 = stablehlo.slice %304 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %306 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %307 = stablehlo.concatenate %305, %306, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %308 = stablehlo.slice %307 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %309 = stablehlo.reshape %308 : (tensor<1xf64>) -> tensor<f64>
    %310 = stablehlo.multiply %15, %309 : tensor<f64>
    %311 = stablehlo.slice %13 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %312 = stablehlo.reshape %311 : (tensor<1xf64>) -> tensor<f64>
    %313 = stablehlo.slice %307 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %314 = stablehlo.reshape %313 : (tensor<1xf64>) -> tensor<f64>
    %315 = stablehlo.multiply %312, %314 : tensor<f64>
    %316 = stablehlo.add %310, %315 : tensor<f64>
    %317 = stablehlo.slice %13 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %318 = stablehlo.reshape %317 : (tensor<1xf64>) -> tensor<f64>
    %319 = stablehlo.slice %307 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %320 = stablehlo.reshape %319 : (tensor<1xf64>) -> tensor<f64>
    %321 = stablehlo.multiply %318, %320 : tensor<f64>
    %322 = stablehlo.add %316, %321 : tensor<f64>
    %323 = stablehlo.slice %13 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %324 = stablehlo.reshape %323 : (tensor<1xf64>) -> tensor<f64>
    %325 = stablehlo.slice %307 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %326 = stablehlo.reshape %325 : (tensor<1xf64>) -> tensor<f64>
    %327 = stablehlo.multiply %324, %326 : tensor<f64>
    %328 = stablehlo.subtract %322, %327 : tensor<f64>
    %329 = stablehlo.reshape %328 : (tensor<f64>) -> tensor<1xf64>
    %330 = stablehlo.multiply %15, %326 : tensor<f64>
    %331 = stablehlo.multiply %312, %320 : tensor<f64>
    %332 = stablehlo.subtract %330, %331 : tensor<f64>
    %333 = stablehlo.multiply %318, %314 : tensor<f64>
    %334 = stablehlo.add %332, %333 : tensor<f64>
    %335 = stablehlo.multiply %324, %309 : tensor<f64>
    %336 = stablehlo.add %334, %335 : tensor<f64>
    %337 = stablehlo.reshape %336 : (tensor<f64>) -> tensor<1xf64>
    %338 = stablehlo.multiply %15, %320 : tensor<f64>
    %339 = stablehlo.multiply %312, %326 : tensor<f64>
    %340 = stablehlo.add %338, %339 : tensor<f64>
    %341 = stablehlo.multiply %318, %309 : tensor<f64>
    %342 = stablehlo.subtract %340, %341 : tensor<f64>
    %343 = stablehlo.multiply %324, %314 : tensor<f64>
    %344 = stablehlo.add %342, %343 : tensor<f64>
    %345 = stablehlo.reshape %344 : (tensor<f64>) -> tensor<1xf64>
    %346 = stablehlo.multiply %15, %314 : tensor<f64>
    %347 = stablehlo.multiply %312, %309 : tensor<f64>
    %348 = stablehlo.subtract %346, %347 : tensor<f64>
    %349 = stablehlo.multiply %318, %326 : tensor<f64>
    %350 = stablehlo.subtract %348, %349 : tensor<f64>
    %351 = stablehlo.multiply %324, %320 : tensor<f64>
    %352 = stablehlo.subtract %350, %351 : tensor<f64>
    %353 = stablehlo.reshape %352 : (tensor<f64>) -> tensor<1xf64>
    %354 = stablehlo.concatenate %329, %337, %345, %353, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %355 = stablehlo.slice %354 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %356 = stablehlo.reshape %355 : (tensor<1xf64>) -> tensor<f64>
    %357 = stablehlo.slice %13 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %358 = stablehlo.reshape %357 : (tensor<1xf64>) -> tensor<f64>
    %359 = stablehlo.negate %358 : tensor<f64>
    %360 = stablehlo.reshape %359 : (tensor<f64>) -> tensor<1xf64>
    %361 = stablehlo.slice %13 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %362 = stablehlo.reshape %361 : (tensor<1xf64>) -> tensor<f64>
    %363 = stablehlo.negate %362 : tensor<f64>
    %364 = stablehlo.reshape %363 : (tensor<f64>) -> tensor<1xf64>
    %365 = stablehlo.slice %13 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %366 = stablehlo.reshape %365 : (tensor<1xf64>) -> tensor<f64>
    %367 = stablehlo.negate %366 : tensor<f64>
    %368 = stablehlo.reshape %367 : (tensor<f64>) -> tensor<1xf64>
    %369 = stablehlo.slice %13 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %370 = stablehlo.reshape %369 : (tensor<1xf64>) -> tensor<f64>
    %371 = stablehlo.reshape %370 : (tensor<f64>) -> tensor<1xf64>
    %372 = stablehlo.concatenate %360, %364, %368, %371, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %373 = stablehlo.dot_general %13, %13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %374 = stablehlo.broadcast_in_dim %373, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %375 = stablehlo.divide %372, %374 : tensor<4xf64>
    %376 = stablehlo.slice %375 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %377 = stablehlo.reshape %376 : (tensor<1xf64>) -> tensor<f64>
    %378 = stablehlo.multiply %356, %377 : tensor<f64>
    %379 = stablehlo.slice %354 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %380 = stablehlo.reshape %379 : (tensor<1xf64>) -> tensor<f64>
    %381 = stablehlo.slice %375 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %382 = stablehlo.reshape %381 : (tensor<1xf64>) -> tensor<f64>
    %383 = stablehlo.multiply %380, %382 : tensor<f64>
    %384 = stablehlo.add %378, %383 : tensor<f64>
    %385 = stablehlo.slice %354 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %386 = stablehlo.reshape %385 : (tensor<1xf64>) -> tensor<f64>
    %387 = stablehlo.slice %375 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %388 = stablehlo.reshape %387 : (tensor<1xf64>) -> tensor<f64>
    %389 = stablehlo.multiply %386, %388 : tensor<f64>
    %390 = stablehlo.add %384, %389 : tensor<f64>
    %391 = stablehlo.slice %354 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %392 = stablehlo.reshape %391 : (tensor<1xf64>) -> tensor<f64>
    %393 = stablehlo.slice %375 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %394 = stablehlo.reshape %393 : (tensor<1xf64>) -> tensor<f64>
    %395 = stablehlo.multiply %392, %394 : tensor<f64>
    %396 = stablehlo.subtract %390, %395 : tensor<f64>
    %397 = stablehlo.reshape %396 : (tensor<f64>) -> tensor<1xf64>
    %398 = stablehlo.multiply %356, %394 : tensor<f64>
    %399 = stablehlo.multiply %380, %388 : tensor<f64>
    %400 = stablehlo.subtract %398, %399 : tensor<f64>
    %401 = stablehlo.multiply %386, %382 : tensor<f64>
    %402 = stablehlo.add %400, %401 : tensor<f64>
    %403 = stablehlo.multiply %392, %377 : tensor<f64>
    %404 = stablehlo.add %402, %403 : tensor<f64>
    %405 = stablehlo.reshape %404 : (tensor<f64>) -> tensor<1xf64>
    %406 = stablehlo.multiply %356, %388 : tensor<f64>
    %407 = stablehlo.multiply %380, %394 : tensor<f64>
    %408 = stablehlo.add %406, %407 : tensor<f64>
    %409 = stablehlo.multiply %386, %377 : tensor<f64>
    %410 = stablehlo.subtract %408, %409 : tensor<f64>
    %411 = stablehlo.multiply %392, %382 : tensor<f64>
    %412 = stablehlo.add %410, %411 : tensor<f64>
    %413 = stablehlo.reshape %412 : (tensor<f64>) -> tensor<1xf64>
    %414 = stablehlo.multiply %356, %382 : tensor<f64>
    %415 = stablehlo.multiply %380, %377 : tensor<f64>
    %416 = stablehlo.subtract %414, %415 : tensor<f64>
    %417 = stablehlo.multiply %386, %394 : tensor<f64>
    %418 = stablehlo.subtract %416, %417 : tensor<f64>
    %419 = stablehlo.multiply %392, %388 : tensor<f64>
    %420 = stablehlo.subtract %418, %419 : tensor<f64>
    %421 = stablehlo.reshape %420 : (tensor<f64>) -> tensor<1xf64>
    %422 = stablehlo.concatenate %397, %405, %413, %421, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %423 = stablehlo.slice %422 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %424 = stablehlo.reshape %423 : (tensor<1xf64>) -> tensor<f64>
    %425 = stablehlo.reshape %424 : (tensor<f64>) -> tensor<1xf64>
    %426 = stablehlo.slice %422 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %427 = stablehlo.reshape %426 : (tensor<1xf64>) -> tensor<f64>
    %428 = stablehlo.reshape %427 : (tensor<f64>) -> tensor<1xf64>
    %429 = stablehlo.slice %422 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %430 = stablehlo.reshape %429 : (tensor<1xf64>) -> tensor<f64>
    %431 = stablehlo.reshape %430 : (tensor<f64>) -> tensor<1xf64>
    %432 = stablehlo.concatenate %425, %428, %431, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %433 = stablehlo.slice %13 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %434 = stablehlo.reshape %433 : (tensor<1xf64>) -> tensor<f64>
    %435 = stablehlo.slice %304 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %436 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %437 = stablehlo.concatenate %435, %436, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %438 = stablehlo.slice %437 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %439 = stablehlo.reshape %438 : (tensor<1xf64>) -> tensor<f64>
    %440 = stablehlo.multiply %434, %439 : tensor<f64>
    %441 = stablehlo.slice %13 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %442 = stablehlo.reshape %441 : (tensor<1xf64>) -> tensor<f64>
    %443 = stablehlo.slice %437 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %444 = stablehlo.reshape %443 : (tensor<1xf64>) -> tensor<f64>
    %445 = stablehlo.multiply %442, %444 : tensor<f64>
    %446 = stablehlo.add %440, %445 : tensor<f64>
    %447 = stablehlo.slice %13 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %448 = stablehlo.reshape %447 : (tensor<1xf64>) -> tensor<f64>
    %449 = stablehlo.slice %437 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %450 = stablehlo.reshape %449 : (tensor<1xf64>) -> tensor<f64>
    %451 = stablehlo.multiply %448, %450 : tensor<f64>
    %452 = stablehlo.add %446, %451 : tensor<f64>
    %453 = stablehlo.slice %13 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %454 = stablehlo.reshape %453 : (tensor<1xf64>) -> tensor<f64>
    %455 = stablehlo.slice %437 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %456 = stablehlo.reshape %455 : (tensor<1xf64>) -> tensor<f64>
    %457 = stablehlo.multiply %454, %456 : tensor<f64>
    %458 = stablehlo.subtract %452, %457 : tensor<f64>
    %459 = stablehlo.reshape %458 : (tensor<f64>) -> tensor<1xf64>
    %460 = stablehlo.multiply %434, %456 : tensor<f64>
    %461 = stablehlo.multiply %442, %450 : tensor<f64>
    %462 = stablehlo.subtract %460, %461 : tensor<f64>
    %463 = stablehlo.multiply %448, %444 : tensor<f64>
    %464 = stablehlo.add %462, %463 : tensor<f64>
    %465 = stablehlo.multiply %454, %439 : tensor<f64>
    %466 = stablehlo.add %464, %465 : tensor<f64>
    %467 = stablehlo.reshape %466 : (tensor<f64>) -> tensor<1xf64>
    %468 = stablehlo.multiply %434, %450 : tensor<f64>
    %469 = stablehlo.multiply %442, %456 : tensor<f64>
    %470 = stablehlo.add %468, %469 : tensor<f64>
    %471 = stablehlo.multiply %448, %439 : tensor<f64>
    %472 = stablehlo.subtract %470, %471 : tensor<f64>
    %473 = stablehlo.multiply %454, %444 : tensor<f64>
    %474 = stablehlo.add %472, %473 : tensor<f64>
    %475 = stablehlo.reshape %474 : (tensor<f64>) -> tensor<1xf64>
    %476 = stablehlo.multiply %434, %444 : tensor<f64>
    %477 = stablehlo.multiply %442, %439 : tensor<f64>
    %478 = stablehlo.subtract %476, %477 : tensor<f64>
    %479 = stablehlo.multiply %448, %456 : tensor<f64>
    %480 = stablehlo.subtract %478, %479 : tensor<f64>
    %481 = stablehlo.multiply %454, %450 : tensor<f64>
    %482 = stablehlo.subtract %480, %481 : tensor<f64>
    %483 = stablehlo.reshape %482 : (tensor<f64>) -> tensor<1xf64>
    %484 = stablehlo.concatenate %459, %467, %475, %483, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %485 = stablehlo.slice %484 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %486 = stablehlo.reshape %485 : (tensor<1xf64>) -> tensor<f64>
    %487 = stablehlo.slice %13 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %488 = stablehlo.reshape %487 : (tensor<1xf64>) -> tensor<f64>
    %489 = stablehlo.negate %488 : tensor<f64>
    %490 = stablehlo.reshape %489 : (tensor<f64>) -> tensor<1xf64>
    %491 = stablehlo.slice %13 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %492 = stablehlo.reshape %491 : (tensor<1xf64>) -> tensor<f64>
    %493 = stablehlo.negate %492 : tensor<f64>
    %494 = stablehlo.reshape %493 : (tensor<f64>) -> tensor<1xf64>
    %495 = stablehlo.slice %13 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %496 = stablehlo.reshape %495 : (tensor<1xf64>) -> tensor<f64>
    %497 = stablehlo.negate %496 : tensor<f64>
    %498 = stablehlo.reshape %497 : (tensor<f64>) -> tensor<1xf64>
    %499 = stablehlo.slice %13 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %500 = stablehlo.reshape %499 : (tensor<1xf64>) -> tensor<f64>
    %501 = stablehlo.reshape %500 : (tensor<f64>) -> tensor<1xf64>
    %502 = stablehlo.concatenate %490, %494, %498, %501, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %503 = stablehlo.dot_general %13, %13, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %504 = stablehlo.broadcast_in_dim %503, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %505 = stablehlo.divide %502, %504 : tensor<4xf64>
    %506 = stablehlo.slice %505 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %507 = stablehlo.reshape %506 : (tensor<1xf64>) -> tensor<f64>
    %508 = stablehlo.multiply %486, %507 : tensor<f64>
    %509 = stablehlo.slice %484 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %510 = stablehlo.reshape %509 : (tensor<1xf64>) -> tensor<f64>
    %511 = stablehlo.slice %505 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %512 = stablehlo.reshape %511 : (tensor<1xf64>) -> tensor<f64>
    %513 = stablehlo.multiply %510, %512 : tensor<f64>
    %514 = stablehlo.add %508, %513 : tensor<f64>
    %515 = stablehlo.slice %484 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %516 = stablehlo.reshape %515 : (tensor<1xf64>) -> tensor<f64>
    %517 = stablehlo.slice %505 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %518 = stablehlo.reshape %517 : (tensor<1xf64>) -> tensor<f64>
    %519 = stablehlo.multiply %516, %518 : tensor<f64>
    %520 = stablehlo.add %514, %519 : tensor<f64>
    %521 = stablehlo.slice %484 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %522 = stablehlo.reshape %521 : (tensor<1xf64>) -> tensor<f64>
    %523 = stablehlo.slice %505 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %524 = stablehlo.reshape %523 : (tensor<1xf64>) -> tensor<f64>
    %525 = stablehlo.multiply %522, %524 : tensor<f64>
    %526 = stablehlo.subtract %520, %525 : tensor<f64>
    %527 = stablehlo.reshape %526 : (tensor<f64>) -> tensor<1xf64>
    %528 = stablehlo.multiply %486, %524 : tensor<f64>
    %529 = stablehlo.multiply %510, %518 : tensor<f64>
    %530 = stablehlo.subtract %528, %529 : tensor<f64>
    %531 = stablehlo.multiply %516, %512 : tensor<f64>
    %532 = stablehlo.add %530, %531 : tensor<f64>
    %533 = stablehlo.multiply %522, %507 : tensor<f64>
    %534 = stablehlo.add %532, %533 : tensor<f64>
    %535 = stablehlo.reshape %534 : (tensor<f64>) -> tensor<1xf64>
    %536 = stablehlo.multiply %486, %518 : tensor<f64>
    %537 = stablehlo.multiply %510, %524 : tensor<f64>
    %538 = stablehlo.add %536, %537 : tensor<f64>
    %539 = stablehlo.multiply %516, %507 : tensor<f64>
    %540 = stablehlo.subtract %538, %539 : tensor<f64>
    %541 = stablehlo.multiply %522, %512 : tensor<f64>
    %542 = stablehlo.add %540, %541 : tensor<f64>
    %543 = stablehlo.reshape %542 : (tensor<f64>) -> tensor<1xf64>
    %544 = stablehlo.multiply %486, %512 : tensor<f64>
    %545 = stablehlo.multiply %510, %507 : tensor<f64>
    %546 = stablehlo.subtract %544, %545 : tensor<f64>
    %547 = stablehlo.multiply %516, %524 : tensor<f64>
    %548 = stablehlo.subtract %546, %547 : tensor<f64>
    %549 = stablehlo.multiply %522, %518 : tensor<f64>
    %550 = stablehlo.subtract %548, %549 : tensor<f64>
    %551 = stablehlo.reshape %550 : (tensor<f64>) -> tensor<1xf64>
    %552 = stablehlo.concatenate %527, %535, %543, %551, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %553 = stablehlo.slice %552 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %554 = stablehlo.reshape %553 : (tensor<1xf64>) -> tensor<f64>
    %555 = stablehlo.reshape %554 : (tensor<f64>) -> tensor<1xf64>
    %556 = stablehlo.slice %552 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %557 = stablehlo.reshape %556 : (tensor<1xf64>) -> tensor<f64>
    %558 = stablehlo.reshape %557 : (tensor<f64>) -> tensor<1xf64>
    %559 = stablehlo.slice %552 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %560 = stablehlo.reshape %559 : (tensor<1xf64>) -> tensor<f64>
    %561 = stablehlo.reshape %560 : (tensor<f64>) -> tensor<1xf64>
    %562 = stablehlo.concatenate %555, %558, %561, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %563 = stablehlo.concatenate %432, %562, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %cst_4 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %564 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %565 = stablehlo.multiply %564, %563 : tensor<6xf64>
    %566 = stablehlo.add %arg13, %565 : tensor<6xf64>
    %567 = stablehlo.slice %arg5 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %cst_5 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %568 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %569 = stablehlo.multiply %568, %566 : tensor<6xf64>
    %570 = stablehlo.slice %569 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_6 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %571 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %572 = stablehlo.divide %570, %571 : tensor<3xf64>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %573 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %574 = stablehlo.concatenate %572, %573, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %575 = stablehlo.slice %574 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %576 = stablehlo.reshape %575 : (tensor<1xf64>) -> tensor<f64>
    %577 = stablehlo.slice %567 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %578 = stablehlo.reshape %577 : (tensor<1xf64>) -> tensor<f64>
    %579 = stablehlo.multiply %576, %578 : tensor<f64>
    %580 = stablehlo.slice %574 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %581 = stablehlo.reshape %580 : (tensor<1xf64>) -> tensor<f64>
    %582 = stablehlo.slice %567 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %583 = stablehlo.reshape %582 : (tensor<1xf64>) -> tensor<f64>
    %584 = stablehlo.multiply %581, %583 : tensor<f64>
    %585 = stablehlo.add %579, %584 : tensor<f64>
    %586 = stablehlo.slice %574 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %587 = stablehlo.reshape %586 : (tensor<1xf64>) -> tensor<f64>
    %588 = stablehlo.slice %567 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %589 = stablehlo.reshape %588 : (tensor<1xf64>) -> tensor<f64>
    %590 = stablehlo.multiply %587, %589 : tensor<f64>
    %591 = stablehlo.add %585, %590 : tensor<f64>
    %592 = stablehlo.slice %574 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %593 = stablehlo.reshape %592 : (tensor<1xf64>) -> tensor<f64>
    %594 = stablehlo.slice %567 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %595 = stablehlo.reshape %594 : (tensor<1xf64>) -> tensor<f64>
    %596 = stablehlo.multiply %593, %595 : tensor<f64>
    %597 = stablehlo.subtract %591, %596 : tensor<f64>
    %598 = stablehlo.reshape %597 : (tensor<f64>) -> tensor<1xf64>
    %599 = stablehlo.multiply %576, %595 : tensor<f64>
    %600 = stablehlo.multiply %581, %589 : tensor<f64>
    %601 = stablehlo.subtract %599, %600 : tensor<f64>
    %602 = stablehlo.multiply %587, %583 : tensor<f64>
    %603 = stablehlo.add %601, %602 : tensor<f64>
    %604 = stablehlo.multiply %593, %578 : tensor<f64>
    %605 = stablehlo.add %603, %604 : tensor<f64>
    %606 = stablehlo.reshape %605 : (tensor<f64>) -> tensor<1xf64>
    %607 = stablehlo.multiply %576, %589 : tensor<f64>
    %608 = stablehlo.multiply %581, %595 : tensor<f64>
    %609 = stablehlo.add %607, %608 : tensor<f64>
    %610 = stablehlo.multiply %587, %578 : tensor<f64>
    %611 = stablehlo.subtract %609, %610 : tensor<f64>
    %612 = stablehlo.multiply %593, %583 : tensor<f64>
    %613 = stablehlo.add %611, %612 : tensor<f64>
    %614 = stablehlo.reshape %613 : (tensor<f64>) -> tensor<1xf64>
    %615 = stablehlo.multiply %576, %583 : tensor<f64>
    %616 = stablehlo.multiply %581, %578 : tensor<f64>
    %617 = stablehlo.subtract %615, %616 : tensor<f64>
    %618 = stablehlo.multiply %587, %595 : tensor<f64>
    %619 = stablehlo.subtract %617, %618 : tensor<f64>
    %620 = stablehlo.multiply %593, %589 : tensor<f64>
    %621 = stablehlo.subtract %619, %620 : tensor<f64>
    %622 = stablehlo.reshape %621 : (tensor<f64>) -> tensor<1xf64>
    %623 = stablehlo.concatenate %598, %606, %614, %622, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %624 = stablehlo.add %567, %623 : tensor<4xf64>
    %625 = stablehlo.dot_general %624, %624, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %626 = stablehlo.sqrt %625 : tensor<f64>
    %627 = stablehlo.broadcast_in_dim %626, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %628 = stablehlo.divide %624, %627 : tensor<4xf64>
    %629 = stablehlo.slice %arg5 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %630 = stablehlo.slice %569 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %631 = stablehlo.add %629, %630 : tensor<3xf64>
    %632 = stablehlo.concatenate %628, %631, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %633 = call @inner_146(%arg20) : (tensor<i64>) -> tensor<i64>
    %634 = call @inner_147(%633, %arg21) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %635:2 = call @inner_189(%633, %632, %566, %arg22, %634, %arg6) : (tensor<i64>, tensor<7xf64>, tensor<6xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    %636:2 = call @inner_201(%633, %632, %563, %arg23, %arg24, %arg25) : (tensor<i64>, tensor<7xf64>, tensor<6xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    %637 = call @inner_203(%636#1, %635#1, %arg26) : (tensor<3xf64>, tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %638 = call @inner_204(%633, %632, %arg27, %arg28) : (tensor<i64>, tensor<7xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %639 = call @inner_214(%632, %566, %arg29) : (tensor<7xf64>, tensor<6xf64>, tensor<3xf64>) -> tensor<3xf64>
    %640 = call @inner_215(%8#2, %arg30) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %641 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %642:3 = call @inner_216(%6, %8#0, %8#1, %8#2) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    %643 = call @inner_217(%arg12, %641) : (tensor<7xf64>, tensor<6xf64>) -> tensor<6xf64>
    %644 = call @inner_218(%566, %10) : (tensor<6xf64>, tensor<3xf64>) -> tensor<3xf64>
    %645 = call @inner_219(%642#0, %642#1, %11) : (tensor<4xf64>, tensor<4xf64>, tensor<6xf64>) -> tensor<6xf64>
    %646 = call @inner_220(%645, %644, %632, %643) : (tensor<6xf64>, tensor<3xf64>, tensor<7xf64>, tensor<6xf64>) -> tensor<6xf64>
    %647 = stablehlo.slice %632 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %648 = stablehlo.slice %647 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %649 = stablehlo.reshape %648 : (tensor<1xf64>) -> tensor<f64>
    %650 = stablehlo.slice %647 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %651 = stablehlo.reshape %650 : (tensor<1xf64>) -> tensor<f64>
    %652 = stablehlo.negate %651 : tensor<f64>
    %653 = stablehlo.reshape %652 : (tensor<f64>) -> tensor<1xf64>
    %654 = stablehlo.slice %647 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %655 = stablehlo.reshape %654 : (tensor<1xf64>) -> tensor<f64>
    %656 = stablehlo.negate %655 : tensor<f64>
    %657 = stablehlo.reshape %656 : (tensor<f64>) -> tensor<1xf64>
    %658 = stablehlo.slice %647 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %659 = stablehlo.reshape %658 : (tensor<1xf64>) -> tensor<f64>
    %660 = stablehlo.negate %659 : tensor<f64>
    %661 = stablehlo.reshape %660 : (tensor<f64>) -> tensor<1xf64>
    %662 = stablehlo.slice %647 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %663 = stablehlo.reshape %662 : (tensor<1xf64>) -> tensor<f64>
    %664 = stablehlo.reshape %663 : (tensor<f64>) -> tensor<1xf64>
    %665 = stablehlo.concatenate %653, %657, %661, %664, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %666 = stablehlo.dot_general %647, %647, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %667 = stablehlo.broadcast_in_dim %666, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %668 = stablehlo.divide %665, %667 : tensor<4xf64>
    %669 = stablehlo.slice %668 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %670 = stablehlo.reshape %669 : (tensor<1xf64>) -> tensor<f64>
    %671 = stablehlo.slice %646 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %672 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %673 = stablehlo.concatenate %671, %672, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %674 = stablehlo.slice %673 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %675 = stablehlo.reshape %674 : (tensor<1xf64>) -> tensor<f64>
    %676 = stablehlo.multiply %670, %675 : tensor<f64>
    %677 = stablehlo.slice %668 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %678 = stablehlo.reshape %677 : (tensor<1xf64>) -> tensor<f64>
    %679 = stablehlo.slice %673 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %680 = stablehlo.reshape %679 : (tensor<1xf64>) -> tensor<f64>
    %681 = stablehlo.multiply %678, %680 : tensor<f64>
    %682 = stablehlo.add %676, %681 : tensor<f64>
    %683 = stablehlo.slice %668 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %684 = stablehlo.reshape %683 : (tensor<1xf64>) -> tensor<f64>
    %685 = stablehlo.slice %673 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %686 = stablehlo.reshape %685 : (tensor<1xf64>) -> tensor<f64>
    %687 = stablehlo.multiply %684, %686 : tensor<f64>
    %688 = stablehlo.add %682, %687 : tensor<f64>
    %689 = stablehlo.slice %668 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %690 = stablehlo.reshape %689 : (tensor<1xf64>) -> tensor<f64>
    %691 = stablehlo.slice %673 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %692 = stablehlo.reshape %691 : (tensor<1xf64>) -> tensor<f64>
    %693 = stablehlo.multiply %690, %692 : tensor<f64>
    %694 = stablehlo.subtract %688, %693 : tensor<f64>
    %695 = stablehlo.reshape %694 : (tensor<f64>) -> tensor<1xf64>
    %696 = stablehlo.multiply %670, %692 : tensor<f64>
    %697 = stablehlo.multiply %678, %686 : tensor<f64>
    %698 = stablehlo.subtract %696, %697 : tensor<f64>
    %699 = stablehlo.multiply %684, %680 : tensor<f64>
    %700 = stablehlo.add %698, %699 : tensor<f64>
    %701 = stablehlo.multiply %690, %675 : tensor<f64>
    %702 = stablehlo.add %700, %701 : tensor<f64>
    %703 = stablehlo.reshape %702 : (tensor<f64>) -> tensor<1xf64>
    %704 = stablehlo.multiply %670, %686 : tensor<f64>
    %705 = stablehlo.multiply %678, %692 : tensor<f64>
    %706 = stablehlo.add %704, %705 : tensor<f64>
    %707 = stablehlo.multiply %684, %675 : tensor<f64>
    %708 = stablehlo.subtract %706, %707 : tensor<f64>
    %709 = stablehlo.multiply %690, %680 : tensor<f64>
    %710 = stablehlo.add %708, %709 : tensor<f64>
    %711 = stablehlo.reshape %710 : (tensor<f64>) -> tensor<1xf64>
    %712 = stablehlo.multiply %670, %680 : tensor<f64>
    %713 = stablehlo.multiply %678, %675 : tensor<f64>
    %714 = stablehlo.subtract %712, %713 : tensor<f64>
    %715 = stablehlo.multiply %684, %692 : tensor<f64>
    %716 = stablehlo.subtract %714, %715 : tensor<f64>
    %717 = stablehlo.multiply %690, %686 : tensor<f64>
    %718 = stablehlo.subtract %716, %717 : tensor<f64>
    %719 = stablehlo.reshape %718 : (tensor<f64>) -> tensor<1xf64>
    %720 = stablehlo.concatenate %695, %703, %711, %719, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %721 = stablehlo.slice %720 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %722 = stablehlo.reshape %721 : (tensor<1xf64>) -> tensor<f64>
    %723 = stablehlo.slice %668 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %724 = stablehlo.reshape %723 : (tensor<1xf64>) -> tensor<f64>
    %725 = stablehlo.negate %724 : tensor<f64>
    %726 = stablehlo.reshape %725 : (tensor<f64>) -> tensor<1xf64>
    %727 = stablehlo.slice %668 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %728 = stablehlo.reshape %727 : (tensor<1xf64>) -> tensor<f64>
    %729 = stablehlo.negate %728 : tensor<f64>
    %730 = stablehlo.reshape %729 : (tensor<f64>) -> tensor<1xf64>
    %731 = stablehlo.slice %668 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %732 = stablehlo.reshape %731 : (tensor<1xf64>) -> tensor<f64>
    %733 = stablehlo.negate %732 : tensor<f64>
    %734 = stablehlo.reshape %733 : (tensor<f64>) -> tensor<1xf64>
    %735 = stablehlo.slice %668 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %736 = stablehlo.reshape %735 : (tensor<1xf64>) -> tensor<f64>
    %737 = stablehlo.reshape %736 : (tensor<f64>) -> tensor<1xf64>
    %738 = stablehlo.concatenate %726, %730, %734, %737, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %739 = stablehlo.dot_general %668, %668, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %740 = stablehlo.broadcast_in_dim %739, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %741 = stablehlo.divide %738, %740 : tensor<4xf64>
    %742 = stablehlo.slice %741 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %743 = stablehlo.reshape %742 : (tensor<1xf64>) -> tensor<f64>
    %744 = stablehlo.multiply %722, %743 : tensor<f64>
    %745 = stablehlo.slice %720 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %746 = stablehlo.reshape %745 : (tensor<1xf64>) -> tensor<f64>
    %747 = stablehlo.slice %741 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %748 = stablehlo.reshape %747 : (tensor<1xf64>) -> tensor<f64>
    %749 = stablehlo.multiply %746, %748 : tensor<f64>
    %750 = stablehlo.add %744, %749 : tensor<f64>
    %751 = stablehlo.slice %720 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %752 = stablehlo.reshape %751 : (tensor<1xf64>) -> tensor<f64>
    %753 = stablehlo.slice %741 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %754 = stablehlo.reshape %753 : (tensor<1xf64>) -> tensor<f64>
    %755 = stablehlo.multiply %752, %754 : tensor<f64>
    %756 = stablehlo.add %750, %755 : tensor<f64>
    %757 = stablehlo.slice %720 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %758 = stablehlo.reshape %757 : (tensor<1xf64>) -> tensor<f64>
    %759 = stablehlo.slice %741 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %760 = stablehlo.reshape %759 : (tensor<1xf64>) -> tensor<f64>
    %761 = stablehlo.multiply %758, %760 : tensor<f64>
    %762 = stablehlo.subtract %756, %761 : tensor<f64>
    %763 = stablehlo.reshape %762 : (tensor<f64>) -> tensor<1xf64>
    %764 = stablehlo.multiply %722, %760 : tensor<f64>
    %765 = stablehlo.multiply %746, %754 : tensor<f64>
    %766 = stablehlo.subtract %764, %765 : tensor<f64>
    %767 = stablehlo.multiply %752, %748 : tensor<f64>
    %768 = stablehlo.add %766, %767 : tensor<f64>
    %769 = stablehlo.multiply %758, %743 : tensor<f64>
    %770 = stablehlo.add %768, %769 : tensor<f64>
    %771 = stablehlo.reshape %770 : (tensor<f64>) -> tensor<1xf64>
    %772 = stablehlo.multiply %722, %754 : tensor<f64>
    %773 = stablehlo.multiply %746, %760 : tensor<f64>
    %774 = stablehlo.add %772, %773 : tensor<f64>
    %775 = stablehlo.multiply %752, %743 : tensor<f64>
    %776 = stablehlo.subtract %774, %775 : tensor<f64>
    %777 = stablehlo.multiply %758, %748 : tensor<f64>
    %778 = stablehlo.add %776, %777 : tensor<f64>
    %779 = stablehlo.reshape %778 : (tensor<f64>) -> tensor<1xf64>
    %780 = stablehlo.multiply %722, %748 : tensor<f64>
    %781 = stablehlo.multiply %746, %743 : tensor<f64>
    %782 = stablehlo.subtract %780, %781 : tensor<f64>
    %783 = stablehlo.multiply %752, %760 : tensor<f64>
    %784 = stablehlo.subtract %782, %783 : tensor<f64>
    %785 = stablehlo.multiply %758, %754 : tensor<f64>
    %786 = stablehlo.subtract %784, %785 : tensor<f64>
    %787 = stablehlo.reshape %786 : (tensor<f64>) -> tensor<1xf64>
    %788 = stablehlo.concatenate %763, %771, %779, %787, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %789 = stablehlo.slice %788 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %790 = stablehlo.reshape %789 : (tensor<1xf64>) -> tensor<f64>
    %791 = stablehlo.reshape %790 : (tensor<f64>) -> tensor<1xf64>
    %792 = stablehlo.slice %788 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %793 = stablehlo.reshape %792 : (tensor<1xf64>) -> tensor<f64>
    %794 = stablehlo.reshape %793 : (tensor<f64>) -> tensor<1xf64>
    %795 = stablehlo.slice %788 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %796 = stablehlo.reshape %795 : (tensor<1xf64>) -> tensor<f64>
    %797 = stablehlo.reshape %796 : (tensor<f64>) -> tensor<1xf64>
    %798 = stablehlo.concatenate %791, %794, %797, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %799 = stablehlo.slice %668 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %800 = stablehlo.reshape %799 : (tensor<1xf64>) -> tensor<f64>
    %801 = stablehlo.slice %646 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %802 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %803 = stablehlo.concatenate %801, %802, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %804 = stablehlo.slice %803 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %805 = stablehlo.reshape %804 : (tensor<1xf64>) -> tensor<f64>
    %806 = stablehlo.multiply %800, %805 : tensor<f64>
    %807 = stablehlo.slice %668 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %808 = stablehlo.reshape %807 : (tensor<1xf64>) -> tensor<f64>
    %809 = stablehlo.slice %803 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %810 = stablehlo.reshape %809 : (tensor<1xf64>) -> tensor<f64>
    %811 = stablehlo.multiply %808, %810 : tensor<f64>
    %812 = stablehlo.add %806, %811 : tensor<f64>
    %813 = stablehlo.slice %668 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %814 = stablehlo.reshape %813 : (tensor<1xf64>) -> tensor<f64>
    %815 = stablehlo.slice %803 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %816 = stablehlo.reshape %815 : (tensor<1xf64>) -> tensor<f64>
    %817 = stablehlo.multiply %814, %816 : tensor<f64>
    %818 = stablehlo.add %812, %817 : tensor<f64>
    %819 = stablehlo.slice %668 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %820 = stablehlo.reshape %819 : (tensor<1xf64>) -> tensor<f64>
    %821 = stablehlo.slice %803 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %822 = stablehlo.reshape %821 : (tensor<1xf64>) -> tensor<f64>
    %823 = stablehlo.multiply %820, %822 : tensor<f64>
    %824 = stablehlo.subtract %818, %823 : tensor<f64>
    %825 = stablehlo.reshape %824 : (tensor<f64>) -> tensor<1xf64>
    %826 = stablehlo.multiply %800, %822 : tensor<f64>
    %827 = stablehlo.multiply %808, %816 : tensor<f64>
    %828 = stablehlo.subtract %826, %827 : tensor<f64>
    %829 = stablehlo.multiply %814, %810 : tensor<f64>
    %830 = stablehlo.add %828, %829 : tensor<f64>
    %831 = stablehlo.multiply %820, %805 : tensor<f64>
    %832 = stablehlo.add %830, %831 : tensor<f64>
    %833 = stablehlo.reshape %832 : (tensor<f64>) -> tensor<1xf64>
    %834 = stablehlo.multiply %800, %816 : tensor<f64>
    %835 = stablehlo.multiply %808, %822 : tensor<f64>
    %836 = stablehlo.add %834, %835 : tensor<f64>
    %837 = stablehlo.multiply %814, %805 : tensor<f64>
    %838 = stablehlo.subtract %836, %837 : tensor<f64>
    %839 = stablehlo.multiply %820, %810 : tensor<f64>
    %840 = stablehlo.add %838, %839 : tensor<f64>
    %841 = stablehlo.reshape %840 : (tensor<f64>) -> tensor<1xf64>
    %842 = stablehlo.multiply %800, %810 : tensor<f64>
    %843 = stablehlo.multiply %808, %805 : tensor<f64>
    %844 = stablehlo.subtract %842, %843 : tensor<f64>
    %845 = stablehlo.multiply %814, %822 : tensor<f64>
    %846 = stablehlo.subtract %844, %845 : tensor<f64>
    %847 = stablehlo.multiply %820, %816 : tensor<f64>
    %848 = stablehlo.subtract %846, %847 : tensor<f64>
    %849 = stablehlo.reshape %848 : (tensor<f64>) -> tensor<1xf64>
    %850 = stablehlo.concatenate %825, %833, %841, %849, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %851 = stablehlo.slice %850 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %852 = stablehlo.reshape %851 : (tensor<1xf64>) -> tensor<f64>
    %853 = stablehlo.slice %668 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %854 = stablehlo.reshape %853 : (tensor<1xf64>) -> tensor<f64>
    %855 = stablehlo.negate %854 : tensor<f64>
    %856 = stablehlo.reshape %855 : (tensor<f64>) -> tensor<1xf64>
    %857 = stablehlo.slice %668 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %858 = stablehlo.reshape %857 : (tensor<1xf64>) -> tensor<f64>
    %859 = stablehlo.negate %858 : tensor<f64>
    %860 = stablehlo.reshape %859 : (tensor<f64>) -> tensor<1xf64>
    %861 = stablehlo.slice %668 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %862 = stablehlo.reshape %861 : (tensor<1xf64>) -> tensor<f64>
    %863 = stablehlo.negate %862 : tensor<f64>
    %864 = stablehlo.reshape %863 : (tensor<f64>) -> tensor<1xf64>
    %865 = stablehlo.slice %668 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %866 = stablehlo.reshape %865 : (tensor<1xf64>) -> tensor<f64>
    %867 = stablehlo.reshape %866 : (tensor<f64>) -> tensor<1xf64>
    %868 = stablehlo.concatenate %856, %860, %864, %867, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %869 = stablehlo.dot_general %668, %668, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %870 = stablehlo.broadcast_in_dim %869, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %871 = stablehlo.divide %868, %870 : tensor<4xf64>
    %872 = stablehlo.slice %871 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %873 = stablehlo.reshape %872 : (tensor<1xf64>) -> tensor<f64>
    %874 = stablehlo.multiply %852, %873 : tensor<f64>
    %875 = stablehlo.slice %850 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %876 = stablehlo.reshape %875 : (tensor<1xf64>) -> tensor<f64>
    %877 = stablehlo.slice %871 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %878 = stablehlo.reshape %877 : (tensor<1xf64>) -> tensor<f64>
    %879 = stablehlo.multiply %876, %878 : tensor<f64>
    %880 = stablehlo.add %874, %879 : tensor<f64>
    %881 = stablehlo.slice %850 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %882 = stablehlo.reshape %881 : (tensor<1xf64>) -> tensor<f64>
    %883 = stablehlo.slice %871 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %884 = stablehlo.reshape %883 : (tensor<1xf64>) -> tensor<f64>
    %885 = stablehlo.multiply %882, %884 : tensor<f64>
    %886 = stablehlo.add %880, %885 : tensor<f64>
    %887 = stablehlo.slice %850 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %888 = stablehlo.reshape %887 : (tensor<1xf64>) -> tensor<f64>
    %889 = stablehlo.slice %871 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %890 = stablehlo.reshape %889 : (tensor<1xf64>) -> tensor<f64>
    %891 = stablehlo.multiply %888, %890 : tensor<f64>
    %892 = stablehlo.subtract %886, %891 : tensor<f64>
    %893 = stablehlo.reshape %892 : (tensor<f64>) -> tensor<1xf64>
    %894 = stablehlo.multiply %852, %890 : tensor<f64>
    %895 = stablehlo.multiply %876, %884 : tensor<f64>
    %896 = stablehlo.subtract %894, %895 : tensor<f64>
    %897 = stablehlo.multiply %882, %878 : tensor<f64>
    %898 = stablehlo.add %896, %897 : tensor<f64>
    %899 = stablehlo.multiply %888, %873 : tensor<f64>
    %900 = stablehlo.add %898, %899 : tensor<f64>
    %901 = stablehlo.reshape %900 : (tensor<f64>) -> tensor<1xf64>
    %902 = stablehlo.multiply %852, %884 : tensor<f64>
    %903 = stablehlo.multiply %876, %890 : tensor<f64>
    %904 = stablehlo.add %902, %903 : tensor<f64>
    %905 = stablehlo.multiply %882, %873 : tensor<f64>
    %906 = stablehlo.subtract %904, %905 : tensor<f64>
    %907 = stablehlo.multiply %888, %878 : tensor<f64>
    %908 = stablehlo.add %906, %907 : tensor<f64>
    %909 = stablehlo.reshape %908 : (tensor<f64>) -> tensor<1xf64>
    %910 = stablehlo.multiply %852, %878 : tensor<f64>
    %911 = stablehlo.multiply %876, %873 : tensor<f64>
    %912 = stablehlo.subtract %910, %911 : tensor<f64>
    %913 = stablehlo.multiply %882, %890 : tensor<f64>
    %914 = stablehlo.subtract %912, %913 : tensor<f64>
    %915 = stablehlo.multiply %888, %884 : tensor<f64>
    %916 = stablehlo.subtract %914, %915 : tensor<f64>
    %917 = stablehlo.reshape %916 : (tensor<f64>) -> tensor<1xf64>
    %918 = stablehlo.concatenate %893, %901, %909, %917, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %919 = stablehlo.slice %918 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %920 = stablehlo.reshape %919 : (tensor<1xf64>) -> tensor<f64>
    %921 = stablehlo.reshape %920 : (tensor<f64>) -> tensor<1xf64>
    %922 = stablehlo.slice %918 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %923 = stablehlo.reshape %922 : (tensor<1xf64>) -> tensor<f64>
    %924 = stablehlo.reshape %923 : (tensor<f64>) -> tensor<1xf64>
    %925 = stablehlo.slice %918 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %926 = stablehlo.reshape %925 : (tensor<1xf64>) -> tensor<f64>
    %927 = stablehlo.reshape %926 : (tensor<f64>) -> tensor<1xf64>
    %928 = stablehlo.concatenate %921, %924, %927, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %929 = stablehlo.concatenate %798, %928, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %930 = stablehlo.slice %929 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %931 = stablehlo.slice %arg12 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %932 = stablehlo.divide %930, %931 : tensor<3xf64>
    %933 = stablehlo.slice %929 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %934 = stablehlo.slice %arg12 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %935 = stablehlo.reshape %934 : (tensor<1xf64>) -> tensor<f64>
    %936 = stablehlo.broadcast_in_dim %935, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %937 = stablehlo.divide %933, %936 : tensor<3xf64>
    %938 = stablehlo.concatenate %932, %937, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %939 = stablehlo.slice %938 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %940 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %941 = stablehlo.concatenate %939, %940, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %942 = stablehlo.slice %941 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %943 = stablehlo.reshape %942 : (tensor<1xf64>) -> tensor<f64>
    %944 = stablehlo.multiply %649, %943 : tensor<f64>
    %945 = stablehlo.slice %647 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %946 = stablehlo.reshape %945 : (tensor<1xf64>) -> tensor<f64>
    %947 = stablehlo.slice %941 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %948 = stablehlo.reshape %947 : (tensor<1xf64>) -> tensor<f64>
    %949 = stablehlo.multiply %946, %948 : tensor<f64>
    %950 = stablehlo.add %944, %949 : tensor<f64>
    %951 = stablehlo.slice %647 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %952 = stablehlo.reshape %951 : (tensor<1xf64>) -> tensor<f64>
    %953 = stablehlo.slice %941 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %954 = stablehlo.reshape %953 : (tensor<1xf64>) -> tensor<f64>
    %955 = stablehlo.multiply %952, %954 : tensor<f64>
    %956 = stablehlo.add %950, %955 : tensor<f64>
    %957 = stablehlo.slice %647 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %958 = stablehlo.reshape %957 : (tensor<1xf64>) -> tensor<f64>
    %959 = stablehlo.slice %941 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %960 = stablehlo.reshape %959 : (tensor<1xf64>) -> tensor<f64>
    %961 = stablehlo.multiply %958, %960 : tensor<f64>
    %962 = stablehlo.subtract %956, %961 : tensor<f64>
    %963 = stablehlo.reshape %962 : (tensor<f64>) -> tensor<1xf64>
    %964 = stablehlo.multiply %649, %960 : tensor<f64>
    %965 = stablehlo.multiply %946, %954 : tensor<f64>
    %966 = stablehlo.subtract %964, %965 : tensor<f64>
    %967 = stablehlo.multiply %952, %948 : tensor<f64>
    %968 = stablehlo.add %966, %967 : tensor<f64>
    %969 = stablehlo.multiply %958, %943 : tensor<f64>
    %970 = stablehlo.add %968, %969 : tensor<f64>
    %971 = stablehlo.reshape %970 : (tensor<f64>) -> tensor<1xf64>
    %972 = stablehlo.multiply %649, %954 : tensor<f64>
    %973 = stablehlo.multiply %946, %960 : tensor<f64>
    %974 = stablehlo.add %972, %973 : tensor<f64>
    %975 = stablehlo.multiply %952, %943 : tensor<f64>
    %976 = stablehlo.subtract %974, %975 : tensor<f64>
    %977 = stablehlo.multiply %958, %948 : tensor<f64>
    %978 = stablehlo.add %976, %977 : tensor<f64>
    %979 = stablehlo.reshape %978 : (tensor<f64>) -> tensor<1xf64>
    %980 = stablehlo.multiply %649, %948 : tensor<f64>
    %981 = stablehlo.multiply %946, %943 : tensor<f64>
    %982 = stablehlo.subtract %980, %981 : tensor<f64>
    %983 = stablehlo.multiply %952, %960 : tensor<f64>
    %984 = stablehlo.subtract %982, %983 : tensor<f64>
    %985 = stablehlo.multiply %958, %954 : tensor<f64>
    %986 = stablehlo.subtract %984, %985 : tensor<f64>
    %987 = stablehlo.reshape %986 : (tensor<f64>) -> tensor<1xf64>
    %988 = stablehlo.concatenate %963, %971, %979, %987, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %989 = stablehlo.slice %988 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %990 = stablehlo.reshape %989 : (tensor<1xf64>) -> tensor<f64>
    %991 = stablehlo.slice %647 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %992 = stablehlo.reshape %991 : (tensor<1xf64>) -> tensor<f64>
    %993 = stablehlo.negate %992 : tensor<f64>
    %994 = stablehlo.reshape %993 : (tensor<f64>) -> tensor<1xf64>
    %995 = stablehlo.slice %647 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %996 = stablehlo.reshape %995 : (tensor<1xf64>) -> tensor<f64>
    %997 = stablehlo.negate %996 : tensor<f64>
    %998 = stablehlo.reshape %997 : (tensor<f64>) -> tensor<1xf64>
    %999 = stablehlo.slice %647 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1000 = stablehlo.reshape %999 : (tensor<1xf64>) -> tensor<f64>
    %1001 = stablehlo.negate %1000 : tensor<f64>
    %1002 = stablehlo.reshape %1001 : (tensor<f64>) -> tensor<1xf64>
    %1003 = stablehlo.slice %647 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1004 = stablehlo.reshape %1003 : (tensor<1xf64>) -> tensor<f64>
    %1005 = stablehlo.reshape %1004 : (tensor<f64>) -> tensor<1xf64>
    %1006 = stablehlo.concatenate %994, %998, %1002, %1005, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1007 = stablehlo.dot_general %647, %647, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1008 = stablehlo.broadcast_in_dim %1007, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1009 = stablehlo.divide %1006, %1008 : tensor<4xf64>
    %1010 = stablehlo.slice %1009 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1011 = stablehlo.reshape %1010 : (tensor<1xf64>) -> tensor<f64>
    %1012 = stablehlo.multiply %990, %1011 : tensor<f64>
    %1013 = stablehlo.slice %988 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1014 = stablehlo.reshape %1013 : (tensor<1xf64>) -> tensor<f64>
    %1015 = stablehlo.slice %1009 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1016 = stablehlo.reshape %1015 : (tensor<1xf64>) -> tensor<f64>
    %1017 = stablehlo.multiply %1014, %1016 : tensor<f64>
    %1018 = stablehlo.add %1012, %1017 : tensor<f64>
    %1019 = stablehlo.slice %988 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1020 = stablehlo.reshape %1019 : (tensor<1xf64>) -> tensor<f64>
    %1021 = stablehlo.slice %1009 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1022 = stablehlo.reshape %1021 : (tensor<1xf64>) -> tensor<f64>
    %1023 = stablehlo.multiply %1020, %1022 : tensor<f64>
    %1024 = stablehlo.add %1018, %1023 : tensor<f64>
    %1025 = stablehlo.slice %988 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1026 = stablehlo.reshape %1025 : (tensor<1xf64>) -> tensor<f64>
    %1027 = stablehlo.slice %1009 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1028 = stablehlo.reshape %1027 : (tensor<1xf64>) -> tensor<f64>
    %1029 = stablehlo.multiply %1026, %1028 : tensor<f64>
    %1030 = stablehlo.subtract %1024, %1029 : tensor<f64>
    %1031 = stablehlo.reshape %1030 : (tensor<f64>) -> tensor<1xf64>
    %1032 = stablehlo.multiply %990, %1028 : tensor<f64>
    %1033 = stablehlo.multiply %1014, %1022 : tensor<f64>
    %1034 = stablehlo.subtract %1032, %1033 : tensor<f64>
    %1035 = stablehlo.multiply %1020, %1016 : tensor<f64>
    %1036 = stablehlo.add %1034, %1035 : tensor<f64>
    %1037 = stablehlo.multiply %1026, %1011 : tensor<f64>
    %1038 = stablehlo.add %1036, %1037 : tensor<f64>
    %1039 = stablehlo.reshape %1038 : (tensor<f64>) -> tensor<1xf64>
    %1040 = stablehlo.multiply %990, %1022 : tensor<f64>
    %1041 = stablehlo.multiply %1014, %1028 : tensor<f64>
    %1042 = stablehlo.add %1040, %1041 : tensor<f64>
    %1043 = stablehlo.multiply %1020, %1011 : tensor<f64>
    %1044 = stablehlo.subtract %1042, %1043 : tensor<f64>
    %1045 = stablehlo.multiply %1026, %1016 : tensor<f64>
    %1046 = stablehlo.add %1044, %1045 : tensor<f64>
    %1047 = stablehlo.reshape %1046 : (tensor<f64>) -> tensor<1xf64>
    %1048 = stablehlo.multiply %990, %1016 : tensor<f64>
    %1049 = stablehlo.multiply %1014, %1011 : tensor<f64>
    %1050 = stablehlo.subtract %1048, %1049 : tensor<f64>
    %1051 = stablehlo.multiply %1020, %1028 : tensor<f64>
    %1052 = stablehlo.subtract %1050, %1051 : tensor<f64>
    %1053 = stablehlo.multiply %1026, %1022 : tensor<f64>
    %1054 = stablehlo.subtract %1052, %1053 : tensor<f64>
    %1055 = stablehlo.reshape %1054 : (tensor<f64>) -> tensor<1xf64>
    %1056 = stablehlo.concatenate %1031, %1039, %1047, %1055, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1057 = stablehlo.slice %1056 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1058 = stablehlo.reshape %1057 : (tensor<1xf64>) -> tensor<f64>
    %1059 = stablehlo.reshape %1058 : (tensor<f64>) -> tensor<1xf64>
    %1060 = stablehlo.slice %1056 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1061 = stablehlo.reshape %1060 : (tensor<1xf64>) -> tensor<f64>
    %1062 = stablehlo.reshape %1061 : (tensor<f64>) -> tensor<1xf64>
    %1063 = stablehlo.slice %1056 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1064 = stablehlo.reshape %1063 : (tensor<1xf64>) -> tensor<f64>
    %1065 = stablehlo.reshape %1064 : (tensor<f64>) -> tensor<1xf64>
    %1066 = stablehlo.concatenate %1059, %1062, %1065, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1067 = stablehlo.slice %647 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1068 = stablehlo.reshape %1067 : (tensor<1xf64>) -> tensor<f64>
    %1069 = stablehlo.slice %938 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1070 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1071 = stablehlo.concatenate %1069, %1070, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1072 = stablehlo.slice %1071 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1073 = stablehlo.reshape %1072 : (tensor<1xf64>) -> tensor<f64>
    %1074 = stablehlo.multiply %1068, %1073 : tensor<f64>
    %1075 = stablehlo.slice %647 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1076 = stablehlo.reshape %1075 : (tensor<1xf64>) -> tensor<f64>
    %1077 = stablehlo.slice %1071 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1078 = stablehlo.reshape %1077 : (tensor<1xf64>) -> tensor<f64>
    %1079 = stablehlo.multiply %1076, %1078 : tensor<f64>
    %1080 = stablehlo.add %1074, %1079 : tensor<f64>
    %1081 = stablehlo.slice %647 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1082 = stablehlo.reshape %1081 : (tensor<1xf64>) -> tensor<f64>
    %1083 = stablehlo.slice %1071 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1084 = stablehlo.reshape %1083 : (tensor<1xf64>) -> tensor<f64>
    %1085 = stablehlo.multiply %1082, %1084 : tensor<f64>
    %1086 = stablehlo.add %1080, %1085 : tensor<f64>
    %1087 = stablehlo.slice %647 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1088 = stablehlo.reshape %1087 : (tensor<1xf64>) -> tensor<f64>
    %1089 = stablehlo.slice %1071 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1090 = stablehlo.reshape %1089 : (tensor<1xf64>) -> tensor<f64>
    %1091 = stablehlo.multiply %1088, %1090 : tensor<f64>
    %1092 = stablehlo.subtract %1086, %1091 : tensor<f64>
    %1093 = stablehlo.reshape %1092 : (tensor<f64>) -> tensor<1xf64>
    %1094 = stablehlo.multiply %1068, %1090 : tensor<f64>
    %1095 = stablehlo.multiply %1076, %1084 : tensor<f64>
    %1096 = stablehlo.subtract %1094, %1095 : tensor<f64>
    %1097 = stablehlo.multiply %1082, %1078 : tensor<f64>
    %1098 = stablehlo.add %1096, %1097 : tensor<f64>
    %1099 = stablehlo.multiply %1088, %1073 : tensor<f64>
    %1100 = stablehlo.add %1098, %1099 : tensor<f64>
    %1101 = stablehlo.reshape %1100 : (tensor<f64>) -> tensor<1xf64>
    %1102 = stablehlo.multiply %1068, %1084 : tensor<f64>
    %1103 = stablehlo.multiply %1076, %1090 : tensor<f64>
    %1104 = stablehlo.add %1102, %1103 : tensor<f64>
    %1105 = stablehlo.multiply %1082, %1073 : tensor<f64>
    %1106 = stablehlo.subtract %1104, %1105 : tensor<f64>
    %1107 = stablehlo.multiply %1088, %1078 : tensor<f64>
    %1108 = stablehlo.add %1106, %1107 : tensor<f64>
    %1109 = stablehlo.reshape %1108 : (tensor<f64>) -> tensor<1xf64>
    %1110 = stablehlo.multiply %1068, %1078 : tensor<f64>
    %1111 = stablehlo.multiply %1076, %1073 : tensor<f64>
    %1112 = stablehlo.subtract %1110, %1111 : tensor<f64>
    %1113 = stablehlo.multiply %1082, %1090 : tensor<f64>
    %1114 = stablehlo.subtract %1112, %1113 : tensor<f64>
    %1115 = stablehlo.multiply %1088, %1084 : tensor<f64>
    %1116 = stablehlo.subtract %1114, %1115 : tensor<f64>
    %1117 = stablehlo.reshape %1116 : (tensor<f64>) -> tensor<1xf64>
    %1118 = stablehlo.concatenate %1093, %1101, %1109, %1117, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1119 = stablehlo.slice %1118 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1120 = stablehlo.reshape %1119 : (tensor<1xf64>) -> tensor<f64>
    %1121 = stablehlo.slice %647 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1122 = stablehlo.reshape %1121 : (tensor<1xf64>) -> tensor<f64>
    %1123 = stablehlo.negate %1122 : tensor<f64>
    %1124 = stablehlo.reshape %1123 : (tensor<f64>) -> tensor<1xf64>
    %1125 = stablehlo.slice %647 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1126 = stablehlo.reshape %1125 : (tensor<1xf64>) -> tensor<f64>
    %1127 = stablehlo.negate %1126 : tensor<f64>
    %1128 = stablehlo.reshape %1127 : (tensor<f64>) -> tensor<1xf64>
    %1129 = stablehlo.slice %647 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1130 = stablehlo.reshape %1129 : (tensor<1xf64>) -> tensor<f64>
    %1131 = stablehlo.negate %1130 : tensor<f64>
    %1132 = stablehlo.reshape %1131 : (tensor<f64>) -> tensor<1xf64>
    %1133 = stablehlo.slice %647 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1134 = stablehlo.reshape %1133 : (tensor<1xf64>) -> tensor<f64>
    %1135 = stablehlo.reshape %1134 : (tensor<f64>) -> tensor<1xf64>
    %1136 = stablehlo.concatenate %1124, %1128, %1132, %1135, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1137 = stablehlo.dot_general %647, %647, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1138 = stablehlo.broadcast_in_dim %1137, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1139 = stablehlo.divide %1136, %1138 : tensor<4xf64>
    %1140 = stablehlo.slice %1139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1141 = stablehlo.reshape %1140 : (tensor<1xf64>) -> tensor<f64>
    %1142 = stablehlo.multiply %1120, %1141 : tensor<f64>
    %1143 = stablehlo.slice %1118 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1144 = stablehlo.reshape %1143 : (tensor<1xf64>) -> tensor<f64>
    %1145 = stablehlo.slice %1139 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1146 = stablehlo.reshape %1145 : (tensor<1xf64>) -> tensor<f64>
    %1147 = stablehlo.multiply %1144, %1146 : tensor<f64>
    %1148 = stablehlo.add %1142, %1147 : tensor<f64>
    %1149 = stablehlo.slice %1118 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1150 = stablehlo.reshape %1149 : (tensor<1xf64>) -> tensor<f64>
    %1151 = stablehlo.slice %1139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1152 = stablehlo.reshape %1151 : (tensor<1xf64>) -> tensor<f64>
    %1153 = stablehlo.multiply %1150, %1152 : tensor<f64>
    %1154 = stablehlo.add %1148, %1153 : tensor<f64>
    %1155 = stablehlo.slice %1118 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1156 = stablehlo.reshape %1155 : (tensor<1xf64>) -> tensor<f64>
    %1157 = stablehlo.slice %1139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1158 = stablehlo.reshape %1157 : (tensor<1xf64>) -> tensor<f64>
    %1159 = stablehlo.multiply %1156, %1158 : tensor<f64>
    %1160 = stablehlo.subtract %1154, %1159 : tensor<f64>
    %1161 = stablehlo.reshape %1160 : (tensor<f64>) -> tensor<1xf64>
    %1162 = stablehlo.multiply %1120, %1158 : tensor<f64>
    %1163 = stablehlo.multiply %1144, %1152 : tensor<f64>
    %1164 = stablehlo.subtract %1162, %1163 : tensor<f64>
    %1165 = stablehlo.multiply %1150, %1146 : tensor<f64>
    %1166 = stablehlo.add %1164, %1165 : tensor<f64>
    %1167 = stablehlo.multiply %1156, %1141 : tensor<f64>
    %1168 = stablehlo.add %1166, %1167 : tensor<f64>
    %1169 = stablehlo.reshape %1168 : (tensor<f64>) -> tensor<1xf64>
    %1170 = stablehlo.multiply %1120, %1152 : tensor<f64>
    %1171 = stablehlo.multiply %1144, %1158 : tensor<f64>
    %1172 = stablehlo.add %1170, %1171 : tensor<f64>
    %1173 = stablehlo.multiply %1150, %1141 : tensor<f64>
    %1174 = stablehlo.subtract %1172, %1173 : tensor<f64>
    %1175 = stablehlo.multiply %1156, %1146 : tensor<f64>
    %1176 = stablehlo.add %1174, %1175 : tensor<f64>
    %1177 = stablehlo.reshape %1176 : (tensor<f64>) -> tensor<1xf64>
    %1178 = stablehlo.multiply %1120, %1146 : tensor<f64>
    %1179 = stablehlo.multiply %1144, %1141 : tensor<f64>
    %1180 = stablehlo.subtract %1178, %1179 : tensor<f64>
    %1181 = stablehlo.multiply %1150, %1158 : tensor<f64>
    %1182 = stablehlo.subtract %1180, %1181 : tensor<f64>
    %1183 = stablehlo.multiply %1156, %1152 : tensor<f64>
    %1184 = stablehlo.subtract %1182, %1183 : tensor<f64>
    %1185 = stablehlo.reshape %1184 : (tensor<f64>) -> tensor<1xf64>
    %1186 = stablehlo.concatenate %1161, %1169, %1177, %1185, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1187 = stablehlo.slice %1186 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1188 = stablehlo.reshape %1187 : (tensor<1xf64>) -> tensor<f64>
    %1189 = stablehlo.reshape %1188 : (tensor<f64>) -> tensor<1xf64>
    %1190 = stablehlo.slice %1186 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1191 = stablehlo.reshape %1190 : (tensor<1xf64>) -> tensor<f64>
    %1192 = stablehlo.reshape %1191 : (tensor<f64>) -> tensor<1xf64>
    %1193 = stablehlo.slice %1186 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1194 = stablehlo.reshape %1193 : (tensor<1xf64>) -> tensor<f64>
    %1195 = stablehlo.reshape %1194 : (tensor<f64>) -> tensor<1xf64>
    %1196 = stablehlo.concatenate %1189, %1192, %1195, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1197 = stablehlo.concatenate %1066, %1196, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %cst_13 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %1198 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1199 = stablehlo.multiply %1198, %1197 : tensor<6xf64>
    %1200 = stablehlo.add %566, %1199 : tensor<6xf64>
    %1201 = stablehlo.slice %632 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %cst_14 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %1202 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1203 = stablehlo.multiply %1202, %1200 : tensor<6xf64>
    %1204 = stablehlo.slice %1203 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_15 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1205 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1206 = stablehlo.divide %1204, %1205 : tensor<3xf64>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1207 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1208 = stablehlo.concatenate %1206, %1207, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1209 = stablehlo.slice %1208 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1210 = stablehlo.reshape %1209 : (tensor<1xf64>) -> tensor<f64>
    %1211 = stablehlo.slice %1201 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1212 = stablehlo.reshape %1211 : (tensor<1xf64>) -> tensor<f64>
    %1213 = stablehlo.multiply %1210, %1212 : tensor<f64>
    %1214 = stablehlo.slice %1208 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1215 = stablehlo.reshape %1214 : (tensor<1xf64>) -> tensor<f64>
    %1216 = stablehlo.slice %1201 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1217 = stablehlo.reshape %1216 : (tensor<1xf64>) -> tensor<f64>
    %1218 = stablehlo.multiply %1215, %1217 : tensor<f64>
    %1219 = stablehlo.add %1213, %1218 : tensor<f64>
    %1220 = stablehlo.slice %1208 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1221 = stablehlo.reshape %1220 : (tensor<1xf64>) -> tensor<f64>
    %1222 = stablehlo.slice %1201 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1223 = stablehlo.reshape %1222 : (tensor<1xf64>) -> tensor<f64>
    %1224 = stablehlo.multiply %1221, %1223 : tensor<f64>
    %1225 = stablehlo.add %1219, %1224 : tensor<f64>
    %1226 = stablehlo.slice %1208 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1227 = stablehlo.reshape %1226 : (tensor<1xf64>) -> tensor<f64>
    %1228 = stablehlo.slice %1201 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1229 = stablehlo.reshape %1228 : (tensor<1xf64>) -> tensor<f64>
    %1230 = stablehlo.multiply %1227, %1229 : tensor<f64>
    %1231 = stablehlo.subtract %1225, %1230 : tensor<f64>
    %1232 = stablehlo.reshape %1231 : (tensor<f64>) -> tensor<1xf64>
    %1233 = stablehlo.multiply %1210, %1229 : tensor<f64>
    %1234 = stablehlo.multiply %1215, %1223 : tensor<f64>
    %1235 = stablehlo.subtract %1233, %1234 : tensor<f64>
    %1236 = stablehlo.multiply %1221, %1217 : tensor<f64>
    %1237 = stablehlo.add %1235, %1236 : tensor<f64>
    %1238 = stablehlo.multiply %1227, %1212 : tensor<f64>
    %1239 = stablehlo.add %1237, %1238 : tensor<f64>
    %1240 = stablehlo.reshape %1239 : (tensor<f64>) -> tensor<1xf64>
    %1241 = stablehlo.multiply %1210, %1223 : tensor<f64>
    %1242 = stablehlo.multiply %1215, %1229 : tensor<f64>
    %1243 = stablehlo.add %1241, %1242 : tensor<f64>
    %1244 = stablehlo.multiply %1221, %1212 : tensor<f64>
    %1245 = stablehlo.subtract %1243, %1244 : tensor<f64>
    %1246 = stablehlo.multiply %1227, %1217 : tensor<f64>
    %1247 = stablehlo.add %1245, %1246 : tensor<f64>
    %1248 = stablehlo.reshape %1247 : (tensor<f64>) -> tensor<1xf64>
    %1249 = stablehlo.multiply %1210, %1217 : tensor<f64>
    %1250 = stablehlo.multiply %1215, %1212 : tensor<f64>
    %1251 = stablehlo.subtract %1249, %1250 : tensor<f64>
    %1252 = stablehlo.multiply %1221, %1229 : tensor<f64>
    %1253 = stablehlo.subtract %1251, %1252 : tensor<f64>
    %1254 = stablehlo.multiply %1227, %1223 : tensor<f64>
    %1255 = stablehlo.subtract %1253, %1254 : tensor<f64>
    %1256 = stablehlo.reshape %1255 : (tensor<f64>) -> tensor<1xf64>
    %1257 = stablehlo.concatenate %1232, %1240, %1248, %1256, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1258 = stablehlo.add %1201, %1257 : tensor<4xf64>
    %1259 = stablehlo.dot_general %1258, %1258, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1260 = stablehlo.sqrt %1259 : tensor<f64>
    %1261 = stablehlo.broadcast_in_dim %1260, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1262 = stablehlo.divide %1258, %1261 : tensor<4xf64>
    %1263 = stablehlo.slice %632 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1264 = stablehlo.slice %1203 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1265 = stablehlo.add %1263, %1264 : tensor<3xf64>
    %1266 = stablehlo.concatenate %1262, %1265, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %1267 = call @inner_221(%633) : (tensor<i64>) -> tensor<i64>
    %1268 = call @inner_222(%1267, %634) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %1269:2 = call @inner_223(%1267, %1266, %1200, %635#0, %1268, %635#1) : (tensor<i64>, tensor<7xf64>, tensor<6xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    %1270:2 = call @inner_224(%1267, %1266, %1197, %636#0, %arg24, %636#1) : (tensor<i64>, tensor<7xf64>, tensor<6xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    %1271 = call @inner_225(%1270#1, %1269#1, %637) : (tensor<3xf64>, tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %1272 = call @inner_226(%1267, %1266, %arg27, %638) : (tensor<i64>, tensor<7xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1273 = call @inner_228(%1266, %1200, %639) : (tensor<7xf64>, tensor<6xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1274 = call @inner_229(%642#2, %640) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1275 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1276:3 = call @inner_230(%6, %642#0, %642#1, %642#2) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
    %1277 = call @inner_231(%arg12, %1275) : (tensor<7xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1278 = call @inner_232(%1200, %644) : (tensor<6xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1279 = call @inner_233(%1276#0, %1276#1, %645) : (tensor<4xf64>, tensor<4xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1280 = call @inner_234(%1279, %1278, %1266, %1277) : (tensor<6xf64>, tensor<3xf64>, tensor<7xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1281 = stablehlo.slice %1266 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1282 = stablehlo.slice %1281 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1283 = stablehlo.reshape %1282 : (tensor<1xf64>) -> tensor<f64>
    %1284 = stablehlo.slice %1281 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1285 = stablehlo.reshape %1284 : (tensor<1xf64>) -> tensor<f64>
    %1286 = stablehlo.negate %1285 : tensor<f64>
    %1287 = stablehlo.reshape %1286 : (tensor<f64>) -> tensor<1xf64>
    %1288 = stablehlo.slice %1281 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1289 = stablehlo.reshape %1288 : (tensor<1xf64>) -> tensor<f64>
    %1290 = stablehlo.negate %1289 : tensor<f64>
    %1291 = stablehlo.reshape %1290 : (tensor<f64>) -> tensor<1xf64>
    %1292 = stablehlo.slice %1281 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1293 = stablehlo.reshape %1292 : (tensor<1xf64>) -> tensor<f64>
    %1294 = stablehlo.negate %1293 : tensor<f64>
    %1295 = stablehlo.reshape %1294 : (tensor<f64>) -> tensor<1xf64>
    %1296 = stablehlo.slice %1281 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1297 = stablehlo.reshape %1296 : (tensor<1xf64>) -> tensor<f64>
    %1298 = stablehlo.reshape %1297 : (tensor<f64>) -> tensor<1xf64>
    %1299 = stablehlo.concatenate %1287, %1291, %1295, %1298, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1300 = stablehlo.dot_general %1281, %1281, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1301 = stablehlo.broadcast_in_dim %1300, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1302 = stablehlo.divide %1299, %1301 : tensor<4xf64>
    %1303 = stablehlo.slice %1302 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1304 = stablehlo.reshape %1303 : (tensor<1xf64>) -> tensor<f64>
    %1305 = stablehlo.slice %1280 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1306 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1307 = stablehlo.concatenate %1305, %1306, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1308 = stablehlo.slice %1307 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1309 = stablehlo.reshape %1308 : (tensor<1xf64>) -> tensor<f64>
    %1310 = stablehlo.multiply %1304, %1309 : tensor<f64>
    %1311 = stablehlo.slice %1302 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1312 = stablehlo.reshape %1311 : (tensor<1xf64>) -> tensor<f64>
    %1313 = stablehlo.slice %1307 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1314 = stablehlo.reshape %1313 : (tensor<1xf64>) -> tensor<f64>
    %1315 = stablehlo.multiply %1312, %1314 : tensor<f64>
    %1316 = stablehlo.add %1310, %1315 : tensor<f64>
    %1317 = stablehlo.slice %1302 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1318 = stablehlo.reshape %1317 : (tensor<1xf64>) -> tensor<f64>
    %1319 = stablehlo.slice %1307 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1320 = stablehlo.reshape %1319 : (tensor<1xf64>) -> tensor<f64>
    %1321 = stablehlo.multiply %1318, %1320 : tensor<f64>
    %1322 = stablehlo.add %1316, %1321 : tensor<f64>
    %1323 = stablehlo.slice %1302 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1324 = stablehlo.reshape %1323 : (tensor<1xf64>) -> tensor<f64>
    %1325 = stablehlo.slice %1307 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1326 = stablehlo.reshape %1325 : (tensor<1xf64>) -> tensor<f64>
    %1327 = stablehlo.multiply %1324, %1326 : tensor<f64>
    %1328 = stablehlo.subtract %1322, %1327 : tensor<f64>
    %1329 = stablehlo.reshape %1328 : (tensor<f64>) -> tensor<1xf64>
    %1330 = stablehlo.multiply %1304, %1326 : tensor<f64>
    %1331 = stablehlo.multiply %1312, %1320 : tensor<f64>
    %1332 = stablehlo.subtract %1330, %1331 : tensor<f64>
    %1333 = stablehlo.multiply %1318, %1314 : tensor<f64>
    %1334 = stablehlo.add %1332, %1333 : tensor<f64>
    %1335 = stablehlo.multiply %1324, %1309 : tensor<f64>
    %1336 = stablehlo.add %1334, %1335 : tensor<f64>
    %1337 = stablehlo.reshape %1336 : (tensor<f64>) -> tensor<1xf64>
    %1338 = stablehlo.multiply %1304, %1320 : tensor<f64>
    %1339 = stablehlo.multiply %1312, %1326 : tensor<f64>
    %1340 = stablehlo.add %1338, %1339 : tensor<f64>
    %1341 = stablehlo.multiply %1318, %1309 : tensor<f64>
    %1342 = stablehlo.subtract %1340, %1341 : tensor<f64>
    %1343 = stablehlo.multiply %1324, %1314 : tensor<f64>
    %1344 = stablehlo.add %1342, %1343 : tensor<f64>
    %1345 = stablehlo.reshape %1344 : (tensor<f64>) -> tensor<1xf64>
    %1346 = stablehlo.multiply %1304, %1314 : tensor<f64>
    %1347 = stablehlo.multiply %1312, %1309 : tensor<f64>
    %1348 = stablehlo.subtract %1346, %1347 : tensor<f64>
    %1349 = stablehlo.multiply %1318, %1326 : tensor<f64>
    %1350 = stablehlo.subtract %1348, %1349 : tensor<f64>
    %1351 = stablehlo.multiply %1324, %1320 : tensor<f64>
    %1352 = stablehlo.subtract %1350, %1351 : tensor<f64>
    %1353 = stablehlo.reshape %1352 : (tensor<f64>) -> tensor<1xf64>
    %1354 = stablehlo.concatenate %1329, %1337, %1345, %1353, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1355 = stablehlo.slice %1354 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1356 = stablehlo.reshape %1355 : (tensor<1xf64>) -> tensor<f64>
    %1357 = stablehlo.slice %1302 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1358 = stablehlo.reshape %1357 : (tensor<1xf64>) -> tensor<f64>
    %1359 = stablehlo.negate %1358 : tensor<f64>
    %1360 = stablehlo.reshape %1359 : (tensor<f64>) -> tensor<1xf64>
    %1361 = stablehlo.slice %1302 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1362 = stablehlo.reshape %1361 : (tensor<1xf64>) -> tensor<f64>
    %1363 = stablehlo.negate %1362 : tensor<f64>
    %1364 = stablehlo.reshape %1363 : (tensor<f64>) -> tensor<1xf64>
    %1365 = stablehlo.slice %1302 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1366 = stablehlo.reshape %1365 : (tensor<1xf64>) -> tensor<f64>
    %1367 = stablehlo.negate %1366 : tensor<f64>
    %1368 = stablehlo.reshape %1367 : (tensor<f64>) -> tensor<1xf64>
    %1369 = stablehlo.slice %1302 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1370 = stablehlo.reshape %1369 : (tensor<1xf64>) -> tensor<f64>
    %1371 = stablehlo.reshape %1370 : (tensor<f64>) -> tensor<1xf64>
    %1372 = stablehlo.concatenate %1360, %1364, %1368, %1371, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1373 = stablehlo.dot_general %1302, %1302, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1374 = stablehlo.broadcast_in_dim %1373, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1375 = stablehlo.divide %1372, %1374 : tensor<4xf64>
    %1376 = stablehlo.slice %1375 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1377 = stablehlo.reshape %1376 : (tensor<1xf64>) -> tensor<f64>
    %1378 = stablehlo.multiply %1356, %1377 : tensor<f64>
    %1379 = stablehlo.slice %1354 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1380 = stablehlo.reshape %1379 : (tensor<1xf64>) -> tensor<f64>
    %1381 = stablehlo.slice %1375 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1382 = stablehlo.reshape %1381 : (tensor<1xf64>) -> tensor<f64>
    %1383 = stablehlo.multiply %1380, %1382 : tensor<f64>
    %1384 = stablehlo.add %1378, %1383 : tensor<f64>
    %1385 = stablehlo.slice %1354 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1386 = stablehlo.reshape %1385 : (tensor<1xf64>) -> tensor<f64>
    %1387 = stablehlo.slice %1375 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1388 = stablehlo.reshape %1387 : (tensor<1xf64>) -> tensor<f64>
    %1389 = stablehlo.multiply %1386, %1388 : tensor<f64>
    %1390 = stablehlo.add %1384, %1389 : tensor<f64>
    %1391 = stablehlo.slice %1354 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1392 = stablehlo.reshape %1391 : (tensor<1xf64>) -> tensor<f64>
    %1393 = stablehlo.slice %1375 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1394 = stablehlo.reshape %1393 : (tensor<1xf64>) -> tensor<f64>
    %1395 = stablehlo.multiply %1392, %1394 : tensor<f64>
    %1396 = stablehlo.subtract %1390, %1395 : tensor<f64>
    %1397 = stablehlo.reshape %1396 : (tensor<f64>) -> tensor<1xf64>
    %1398 = stablehlo.multiply %1356, %1394 : tensor<f64>
    %1399 = stablehlo.multiply %1380, %1388 : tensor<f64>
    %1400 = stablehlo.subtract %1398, %1399 : tensor<f64>
    %1401 = stablehlo.multiply %1386, %1382 : tensor<f64>
    %1402 = stablehlo.add %1400, %1401 : tensor<f64>
    %1403 = stablehlo.multiply %1392, %1377 : tensor<f64>
    %1404 = stablehlo.add %1402, %1403 : tensor<f64>
    %1405 = stablehlo.reshape %1404 : (tensor<f64>) -> tensor<1xf64>
    %1406 = stablehlo.multiply %1356, %1388 : tensor<f64>
    %1407 = stablehlo.multiply %1380, %1394 : tensor<f64>
    %1408 = stablehlo.add %1406, %1407 : tensor<f64>
    %1409 = stablehlo.multiply %1386, %1377 : tensor<f64>
    %1410 = stablehlo.subtract %1408, %1409 : tensor<f64>
    %1411 = stablehlo.multiply %1392, %1382 : tensor<f64>
    %1412 = stablehlo.add %1410, %1411 : tensor<f64>
    %1413 = stablehlo.reshape %1412 : (tensor<f64>) -> tensor<1xf64>
    %1414 = stablehlo.multiply %1356, %1382 : tensor<f64>
    %1415 = stablehlo.multiply %1380, %1377 : tensor<f64>
    %1416 = stablehlo.subtract %1414, %1415 : tensor<f64>
    %1417 = stablehlo.multiply %1386, %1394 : tensor<f64>
    %1418 = stablehlo.subtract %1416, %1417 : tensor<f64>
    %1419 = stablehlo.multiply %1392, %1388 : tensor<f64>
    %1420 = stablehlo.subtract %1418, %1419 : tensor<f64>
    %1421 = stablehlo.reshape %1420 : (tensor<f64>) -> tensor<1xf64>
    %1422 = stablehlo.concatenate %1397, %1405, %1413, %1421, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1423 = stablehlo.slice %1422 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1424 = stablehlo.reshape %1423 : (tensor<1xf64>) -> tensor<f64>
    %1425 = stablehlo.reshape %1424 : (tensor<f64>) -> tensor<1xf64>
    %1426 = stablehlo.slice %1422 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1427 = stablehlo.reshape %1426 : (tensor<1xf64>) -> tensor<f64>
    %1428 = stablehlo.reshape %1427 : (tensor<f64>) -> tensor<1xf64>
    %1429 = stablehlo.slice %1422 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1430 = stablehlo.reshape %1429 : (tensor<1xf64>) -> tensor<f64>
    %1431 = stablehlo.reshape %1430 : (tensor<f64>) -> tensor<1xf64>
    %1432 = stablehlo.concatenate %1425, %1428, %1431, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1433 = stablehlo.slice %1302 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1434 = stablehlo.reshape %1433 : (tensor<1xf64>) -> tensor<f64>
    %1435 = stablehlo.slice %1280 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1436 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1437 = stablehlo.concatenate %1435, %1436, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1438 = stablehlo.slice %1437 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1439 = stablehlo.reshape %1438 : (tensor<1xf64>) -> tensor<f64>
    %1440 = stablehlo.multiply %1434, %1439 : tensor<f64>
    %1441 = stablehlo.slice %1302 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1442 = stablehlo.reshape %1441 : (tensor<1xf64>) -> tensor<f64>
    %1443 = stablehlo.slice %1437 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1444 = stablehlo.reshape %1443 : (tensor<1xf64>) -> tensor<f64>
    %1445 = stablehlo.multiply %1442, %1444 : tensor<f64>
    %1446 = stablehlo.add %1440, %1445 : tensor<f64>
    %1447 = stablehlo.slice %1302 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1448 = stablehlo.reshape %1447 : (tensor<1xf64>) -> tensor<f64>
    %1449 = stablehlo.slice %1437 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1450 = stablehlo.reshape %1449 : (tensor<1xf64>) -> tensor<f64>
    %1451 = stablehlo.multiply %1448, %1450 : tensor<f64>
    %1452 = stablehlo.add %1446, %1451 : tensor<f64>
    %1453 = stablehlo.slice %1302 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1454 = stablehlo.reshape %1453 : (tensor<1xf64>) -> tensor<f64>
    %1455 = stablehlo.slice %1437 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1456 = stablehlo.reshape %1455 : (tensor<1xf64>) -> tensor<f64>
    %1457 = stablehlo.multiply %1454, %1456 : tensor<f64>
    %1458 = stablehlo.subtract %1452, %1457 : tensor<f64>
    %1459 = stablehlo.reshape %1458 : (tensor<f64>) -> tensor<1xf64>
    %1460 = stablehlo.multiply %1434, %1456 : tensor<f64>
    %1461 = stablehlo.multiply %1442, %1450 : tensor<f64>
    %1462 = stablehlo.subtract %1460, %1461 : tensor<f64>
    %1463 = stablehlo.multiply %1448, %1444 : tensor<f64>
    %1464 = stablehlo.add %1462, %1463 : tensor<f64>
    %1465 = stablehlo.multiply %1454, %1439 : tensor<f64>
    %1466 = stablehlo.add %1464, %1465 : tensor<f64>
    %1467 = stablehlo.reshape %1466 : (tensor<f64>) -> tensor<1xf64>
    %1468 = stablehlo.multiply %1434, %1450 : tensor<f64>
    %1469 = stablehlo.multiply %1442, %1456 : tensor<f64>
    %1470 = stablehlo.add %1468, %1469 : tensor<f64>
    %1471 = stablehlo.multiply %1448, %1439 : tensor<f64>
    %1472 = stablehlo.subtract %1470, %1471 : tensor<f64>
    %1473 = stablehlo.multiply %1454, %1444 : tensor<f64>
    %1474 = stablehlo.add %1472, %1473 : tensor<f64>
    %1475 = stablehlo.reshape %1474 : (tensor<f64>) -> tensor<1xf64>
    %1476 = stablehlo.multiply %1434, %1444 : tensor<f64>
    %1477 = stablehlo.multiply %1442, %1439 : tensor<f64>
    %1478 = stablehlo.subtract %1476, %1477 : tensor<f64>
    %1479 = stablehlo.multiply %1448, %1456 : tensor<f64>
    %1480 = stablehlo.subtract %1478, %1479 : tensor<f64>
    %1481 = stablehlo.multiply %1454, %1450 : tensor<f64>
    %1482 = stablehlo.subtract %1480, %1481 : tensor<f64>
    %1483 = stablehlo.reshape %1482 : (tensor<f64>) -> tensor<1xf64>
    %1484 = stablehlo.concatenate %1459, %1467, %1475, %1483, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1485 = stablehlo.slice %1484 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1486 = stablehlo.reshape %1485 : (tensor<1xf64>) -> tensor<f64>
    %1487 = stablehlo.slice %1302 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1488 = stablehlo.reshape %1487 : (tensor<1xf64>) -> tensor<f64>
    %1489 = stablehlo.negate %1488 : tensor<f64>
    %1490 = stablehlo.reshape %1489 : (tensor<f64>) -> tensor<1xf64>
    %1491 = stablehlo.slice %1302 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1492 = stablehlo.reshape %1491 : (tensor<1xf64>) -> tensor<f64>
    %1493 = stablehlo.negate %1492 : tensor<f64>
    %1494 = stablehlo.reshape %1493 : (tensor<f64>) -> tensor<1xf64>
    %1495 = stablehlo.slice %1302 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1496 = stablehlo.reshape %1495 : (tensor<1xf64>) -> tensor<f64>
    %1497 = stablehlo.negate %1496 : tensor<f64>
    %1498 = stablehlo.reshape %1497 : (tensor<f64>) -> tensor<1xf64>
    %1499 = stablehlo.slice %1302 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1500 = stablehlo.reshape %1499 : (tensor<1xf64>) -> tensor<f64>
    %1501 = stablehlo.reshape %1500 : (tensor<f64>) -> tensor<1xf64>
    %1502 = stablehlo.concatenate %1490, %1494, %1498, %1501, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1503 = stablehlo.dot_general %1302, %1302, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1504 = stablehlo.broadcast_in_dim %1503, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1505 = stablehlo.divide %1502, %1504 : tensor<4xf64>
    %1506 = stablehlo.slice %1505 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1507 = stablehlo.reshape %1506 : (tensor<1xf64>) -> tensor<f64>
    %1508 = stablehlo.multiply %1486, %1507 : tensor<f64>
    %1509 = stablehlo.slice %1484 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1510 = stablehlo.reshape %1509 : (tensor<1xf64>) -> tensor<f64>
    %1511 = stablehlo.slice %1505 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1512 = stablehlo.reshape %1511 : (tensor<1xf64>) -> tensor<f64>
    %1513 = stablehlo.multiply %1510, %1512 : tensor<f64>
    %1514 = stablehlo.add %1508, %1513 : tensor<f64>
    %1515 = stablehlo.slice %1484 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1516 = stablehlo.reshape %1515 : (tensor<1xf64>) -> tensor<f64>
    %1517 = stablehlo.slice %1505 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1518 = stablehlo.reshape %1517 : (tensor<1xf64>) -> tensor<f64>
    %1519 = stablehlo.multiply %1516, %1518 : tensor<f64>
    %1520 = stablehlo.add %1514, %1519 : tensor<f64>
    %1521 = stablehlo.slice %1484 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1522 = stablehlo.reshape %1521 : (tensor<1xf64>) -> tensor<f64>
    %1523 = stablehlo.slice %1505 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1524 = stablehlo.reshape %1523 : (tensor<1xf64>) -> tensor<f64>
    %1525 = stablehlo.multiply %1522, %1524 : tensor<f64>
    %1526 = stablehlo.subtract %1520, %1525 : tensor<f64>
    %1527 = stablehlo.reshape %1526 : (tensor<f64>) -> tensor<1xf64>
    %1528 = stablehlo.multiply %1486, %1524 : tensor<f64>
    %1529 = stablehlo.multiply %1510, %1518 : tensor<f64>
    %1530 = stablehlo.subtract %1528, %1529 : tensor<f64>
    %1531 = stablehlo.multiply %1516, %1512 : tensor<f64>
    %1532 = stablehlo.add %1530, %1531 : tensor<f64>
    %1533 = stablehlo.multiply %1522, %1507 : tensor<f64>
    %1534 = stablehlo.add %1532, %1533 : tensor<f64>
    %1535 = stablehlo.reshape %1534 : (tensor<f64>) -> tensor<1xf64>
    %1536 = stablehlo.multiply %1486, %1518 : tensor<f64>
    %1537 = stablehlo.multiply %1510, %1524 : tensor<f64>
    %1538 = stablehlo.add %1536, %1537 : tensor<f64>
    %1539 = stablehlo.multiply %1516, %1507 : tensor<f64>
    %1540 = stablehlo.subtract %1538, %1539 : tensor<f64>
    %1541 = stablehlo.multiply %1522, %1512 : tensor<f64>
    %1542 = stablehlo.add %1540, %1541 : tensor<f64>
    %1543 = stablehlo.reshape %1542 : (tensor<f64>) -> tensor<1xf64>
    %1544 = stablehlo.multiply %1486, %1512 : tensor<f64>
    %1545 = stablehlo.multiply %1510, %1507 : tensor<f64>
    %1546 = stablehlo.subtract %1544, %1545 : tensor<f64>
    %1547 = stablehlo.multiply %1516, %1524 : tensor<f64>
    %1548 = stablehlo.subtract %1546, %1547 : tensor<f64>
    %1549 = stablehlo.multiply %1522, %1518 : tensor<f64>
    %1550 = stablehlo.subtract %1548, %1549 : tensor<f64>
    %1551 = stablehlo.reshape %1550 : (tensor<f64>) -> tensor<1xf64>
    %1552 = stablehlo.concatenate %1527, %1535, %1543, %1551, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1553 = stablehlo.slice %1552 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1554 = stablehlo.reshape %1553 : (tensor<1xf64>) -> tensor<f64>
    %1555 = stablehlo.reshape %1554 : (tensor<f64>) -> tensor<1xf64>
    %1556 = stablehlo.slice %1552 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1557 = stablehlo.reshape %1556 : (tensor<1xf64>) -> tensor<f64>
    %1558 = stablehlo.reshape %1557 : (tensor<f64>) -> tensor<1xf64>
    %1559 = stablehlo.slice %1552 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1560 = stablehlo.reshape %1559 : (tensor<1xf64>) -> tensor<f64>
    %1561 = stablehlo.reshape %1560 : (tensor<f64>) -> tensor<1xf64>
    %1562 = stablehlo.concatenate %1555, %1558, %1561, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1563 = stablehlo.concatenate %1432, %1562, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1564 = stablehlo.slice %1563 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1565 = stablehlo.slice %arg12 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %1566 = stablehlo.divide %1564, %1565 : tensor<3xf64>
    %1567 = stablehlo.slice %1563 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1568 = stablehlo.slice %arg12 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1569 = stablehlo.reshape %1568 : (tensor<1xf64>) -> tensor<f64>
    %1570 = stablehlo.broadcast_in_dim %1569, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1571 = stablehlo.divide %1567, %1570 : tensor<3xf64>
    %1572 = stablehlo.concatenate %1566, %1571, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1573 = stablehlo.slice %1572 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1574 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1575 = stablehlo.concatenate %1573, %1574, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1576 = stablehlo.slice %1575 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1577 = stablehlo.reshape %1576 : (tensor<1xf64>) -> tensor<f64>
    %1578 = stablehlo.multiply %1283, %1577 : tensor<f64>
    %1579 = stablehlo.slice %1281 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1580 = stablehlo.reshape %1579 : (tensor<1xf64>) -> tensor<f64>
    %1581 = stablehlo.slice %1575 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1582 = stablehlo.reshape %1581 : (tensor<1xf64>) -> tensor<f64>
    %1583 = stablehlo.multiply %1580, %1582 : tensor<f64>
    %1584 = stablehlo.add %1578, %1583 : tensor<f64>
    %1585 = stablehlo.slice %1281 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1586 = stablehlo.reshape %1585 : (tensor<1xf64>) -> tensor<f64>
    %1587 = stablehlo.slice %1575 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1588 = stablehlo.reshape %1587 : (tensor<1xf64>) -> tensor<f64>
    %1589 = stablehlo.multiply %1586, %1588 : tensor<f64>
    %1590 = stablehlo.add %1584, %1589 : tensor<f64>
    %1591 = stablehlo.slice %1281 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1592 = stablehlo.reshape %1591 : (tensor<1xf64>) -> tensor<f64>
    %1593 = stablehlo.slice %1575 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1594 = stablehlo.reshape %1593 : (tensor<1xf64>) -> tensor<f64>
    %1595 = stablehlo.multiply %1592, %1594 : tensor<f64>
    %1596 = stablehlo.subtract %1590, %1595 : tensor<f64>
    %1597 = stablehlo.reshape %1596 : (tensor<f64>) -> tensor<1xf64>
    %1598 = stablehlo.multiply %1283, %1594 : tensor<f64>
    %1599 = stablehlo.multiply %1580, %1588 : tensor<f64>
    %1600 = stablehlo.subtract %1598, %1599 : tensor<f64>
    %1601 = stablehlo.multiply %1586, %1582 : tensor<f64>
    %1602 = stablehlo.add %1600, %1601 : tensor<f64>
    %1603 = stablehlo.multiply %1592, %1577 : tensor<f64>
    %1604 = stablehlo.add %1602, %1603 : tensor<f64>
    %1605 = stablehlo.reshape %1604 : (tensor<f64>) -> tensor<1xf64>
    %1606 = stablehlo.multiply %1283, %1588 : tensor<f64>
    %1607 = stablehlo.multiply %1580, %1594 : tensor<f64>
    %1608 = stablehlo.add %1606, %1607 : tensor<f64>
    %1609 = stablehlo.multiply %1586, %1577 : tensor<f64>
    %1610 = stablehlo.subtract %1608, %1609 : tensor<f64>
    %1611 = stablehlo.multiply %1592, %1582 : tensor<f64>
    %1612 = stablehlo.add %1610, %1611 : tensor<f64>
    %1613 = stablehlo.reshape %1612 : (tensor<f64>) -> tensor<1xf64>
    %1614 = stablehlo.multiply %1283, %1582 : tensor<f64>
    %1615 = stablehlo.multiply %1580, %1577 : tensor<f64>
    %1616 = stablehlo.subtract %1614, %1615 : tensor<f64>
    %1617 = stablehlo.multiply %1586, %1594 : tensor<f64>
    %1618 = stablehlo.subtract %1616, %1617 : tensor<f64>
    %1619 = stablehlo.multiply %1592, %1588 : tensor<f64>
    %1620 = stablehlo.subtract %1618, %1619 : tensor<f64>
    %1621 = stablehlo.reshape %1620 : (tensor<f64>) -> tensor<1xf64>
    %1622 = stablehlo.concatenate %1597, %1605, %1613, %1621, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1623 = stablehlo.slice %1622 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1624 = stablehlo.reshape %1623 : (tensor<1xf64>) -> tensor<f64>
    %1625 = stablehlo.slice %1281 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1626 = stablehlo.reshape %1625 : (tensor<1xf64>) -> tensor<f64>
    %1627 = stablehlo.negate %1626 : tensor<f64>
    %1628 = stablehlo.reshape %1627 : (tensor<f64>) -> tensor<1xf64>
    %1629 = stablehlo.slice %1281 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1630 = stablehlo.reshape %1629 : (tensor<1xf64>) -> tensor<f64>
    %1631 = stablehlo.negate %1630 : tensor<f64>
    %1632 = stablehlo.reshape %1631 : (tensor<f64>) -> tensor<1xf64>
    %1633 = stablehlo.slice %1281 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1634 = stablehlo.reshape %1633 : (tensor<1xf64>) -> tensor<f64>
    %1635 = stablehlo.negate %1634 : tensor<f64>
    %1636 = stablehlo.reshape %1635 : (tensor<f64>) -> tensor<1xf64>
    %1637 = stablehlo.slice %1281 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1638 = stablehlo.reshape %1637 : (tensor<1xf64>) -> tensor<f64>
    %1639 = stablehlo.reshape %1638 : (tensor<f64>) -> tensor<1xf64>
    %1640 = stablehlo.concatenate %1628, %1632, %1636, %1639, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1641 = stablehlo.dot_general %1281, %1281, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1642 = stablehlo.broadcast_in_dim %1641, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1643 = stablehlo.divide %1640, %1642 : tensor<4xf64>
    %1644 = stablehlo.slice %1643 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1645 = stablehlo.reshape %1644 : (tensor<1xf64>) -> tensor<f64>
    %1646 = stablehlo.multiply %1624, %1645 : tensor<f64>
    %1647 = stablehlo.slice %1622 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1648 = stablehlo.reshape %1647 : (tensor<1xf64>) -> tensor<f64>
    %1649 = stablehlo.slice %1643 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1650 = stablehlo.reshape %1649 : (tensor<1xf64>) -> tensor<f64>
    %1651 = stablehlo.multiply %1648, %1650 : tensor<f64>
    %1652 = stablehlo.add %1646, %1651 : tensor<f64>
    %1653 = stablehlo.slice %1622 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1654 = stablehlo.reshape %1653 : (tensor<1xf64>) -> tensor<f64>
    %1655 = stablehlo.slice %1643 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1656 = stablehlo.reshape %1655 : (tensor<1xf64>) -> tensor<f64>
    %1657 = stablehlo.multiply %1654, %1656 : tensor<f64>
    %1658 = stablehlo.add %1652, %1657 : tensor<f64>
    %1659 = stablehlo.slice %1622 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1660 = stablehlo.reshape %1659 : (tensor<1xf64>) -> tensor<f64>
    %1661 = stablehlo.slice %1643 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1662 = stablehlo.reshape %1661 : (tensor<1xf64>) -> tensor<f64>
    %1663 = stablehlo.multiply %1660, %1662 : tensor<f64>
    %1664 = stablehlo.subtract %1658, %1663 : tensor<f64>
    %1665 = stablehlo.reshape %1664 : (tensor<f64>) -> tensor<1xf64>
    %1666 = stablehlo.multiply %1624, %1662 : tensor<f64>
    %1667 = stablehlo.multiply %1648, %1656 : tensor<f64>
    %1668 = stablehlo.subtract %1666, %1667 : tensor<f64>
    %1669 = stablehlo.multiply %1654, %1650 : tensor<f64>
    %1670 = stablehlo.add %1668, %1669 : tensor<f64>
    %1671 = stablehlo.multiply %1660, %1645 : tensor<f64>
    %1672 = stablehlo.add %1670, %1671 : tensor<f64>
    %1673 = stablehlo.reshape %1672 : (tensor<f64>) -> tensor<1xf64>
    %1674 = stablehlo.multiply %1624, %1656 : tensor<f64>
    %1675 = stablehlo.multiply %1648, %1662 : tensor<f64>
    %1676 = stablehlo.add %1674, %1675 : tensor<f64>
    %1677 = stablehlo.multiply %1654, %1645 : tensor<f64>
    %1678 = stablehlo.subtract %1676, %1677 : tensor<f64>
    %1679 = stablehlo.multiply %1660, %1650 : tensor<f64>
    %1680 = stablehlo.add %1678, %1679 : tensor<f64>
    %1681 = stablehlo.reshape %1680 : (tensor<f64>) -> tensor<1xf64>
    %1682 = stablehlo.multiply %1624, %1650 : tensor<f64>
    %1683 = stablehlo.multiply %1648, %1645 : tensor<f64>
    %1684 = stablehlo.subtract %1682, %1683 : tensor<f64>
    %1685 = stablehlo.multiply %1654, %1662 : tensor<f64>
    %1686 = stablehlo.subtract %1684, %1685 : tensor<f64>
    %1687 = stablehlo.multiply %1660, %1656 : tensor<f64>
    %1688 = stablehlo.subtract %1686, %1687 : tensor<f64>
    %1689 = stablehlo.reshape %1688 : (tensor<f64>) -> tensor<1xf64>
    %1690 = stablehlo.concatenate %1665, %1673, %1681, %1689, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1691 = stablehlo.slice %1690 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1692 = stablehlo.reshape %1691 : (tensor<1xf64>) -> tensor<f64>
    %1693 = stablehlo.reshape %1692 : (tensor<f64>) -> tensor<1xf64>
    %1694 = stablehlo.slice %1690 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1695 = stablehlo.reshape %1694 : (tensor<1xf64>) -> tensor<f64>
    %1696 = stablehlo.reshape %1695 : (tensor<f64>) -> tensor<1xf64>
    %1697 = stablehlo.slice %1690 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1698 = stablehlo.reshape %1697 : (tensor<1xf64>) -> tensor<f64>
    %1699 = stablehlo.reshape %1698 : (tensor<f64>) -> tensor<1xf64>
    %1700 = stablehlo.concatenate %1693, %1696, %1699, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1701 = stablehlo.slice %1281 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1702 = stablehlo.reshape %1701 : (tensor<1xf64>) -> tensor<f64>
    %1703 = stablehlo.slice %1572 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1704 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1705 = stablehlo.concatenate %1703, %1704, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1706 = stablehlo.slice %1705 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1707 = stablehlo.reshape %1706 : (tensor<1xf64>) -> tensor<f64>
    %1708 = stablehlo.multiply %1702, %1707 : tensor<f64>
    %1709 = stablehlo.slice %1281 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1710 = stablehlo.reshape %1709 : (tensor<1xf64>) -> tensor<f64>
    %1711 = stablehlo.slice %1705 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1712 = stablehlo.reshape %1711 : (tensor<1xf64>) -> tensor<f64>
    %1713 = stablehlo.multiply %1710, %1712 : tensor<f64>
    %1714 = stablehlo.add %1708, %1713 : tensor<f64>
    %1715 = stablehlo.slice %1281 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1716 = stablehlo.reshape %1715 : (tensor<1xf64>) -> tensor<f64>
    %1717 = stablehlo.slice %1705 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1718 = stablehlo.reshape %1717 : (tensor<1xf64>) -> tensor<f64>
    %1719 = stablehlo.multiply %1716, %1718 : tensor<f64>
    %1720 = stablehlo.add %1714, %1719 : tensor<f64>
    %1721 = stablehlo.slice %1281 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1722 = stablehlo.reshape %1721 : (tensor<1xf64>) -> tensor<f64>
    %1723 = stablehlo.slice %1705 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1724 = stablehlo.reshape %1723 : (tensor<1xf64>) -> tensor<f64>
    %1725 = stablehlo.multiply %1722, %1724 : tensor<f64>
    %1726 = stablehlo.subtract %1720, %1725 : tensor<f64>
    %1727 = stablehlo.reshape %1726 : (tensor<f64>) -> tensor<1xf64>
    %1728 = stablehlo.multiply %1702, %1724 : tensor<f64>
    %1729 = stablehlo.multiply %1710, %1718 : tensor<f64>
    %1730 = stablehlo.subtract %1728, %1729 : tensor<f64>
    %1731 = stablehlo.multiply %1716, %1712 : tensor<f64>
    %1732 = stablehlo.add %1730, %1731 : tensor<f64>
    %1733 = stablehlo.multiply %1722, %1707 : tensor<f64>
    %1734 = stablehlo.add %1732, %1733 : tensor<f64>
    %1735 = stablehlo.reshape %1734 : (tensor<f64>) -> tensor<1xf64>
    %1736 = stablehlo.multiply %1702, %1718 : tensor<f64>
    %1737 = stablehlo.multiply %1710, %1724 : tensor<f64>
    %1738 = stablehlo.add %1736, %1737 : tensor<f64>
    %1739 = stablehlo.multiply %1716, %1707 : tensor<f64>
    %1740 = stablehlo.subtract %1738, %1739 : tensor<f64>
    %1741 = stablehlo.multiply %1722, %1712 : tensor<f64>
    %1742 = stablehlo.add %1740, %1741 : tensor<f64>
    %1743 = stablehlo.reshape %1742 : (tensor<f64>) -> tensor<1xf64>
    %1744 = stablehlo.multiply %1702, %1712 : tensor<f64>
    %1745 = stablehlo.multiply %1710, %1707 : tensor<f64>
    %1746 = stablehlo.subtract %1744, %1745 : tensor<f64>
    %1747 = stablehlo.multiply %1716, %1724 : tensor<f64>
    %1748 = stablehlo.subtract %1746, %1747 : tensor<f64>
    %1749 = stablehlo.multiply %1722, %1718 : tensor<f64>
    %1750 = stablehlo.subtract %1748, %1749 : tensor<f64>
    %1751 = stablehlo.reshape %1750 : (tensor<f64>) -> tensor<1xf64>
    %1752 = stablehlo.concatenate %1727, %1735, %1743, %1751, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1753 = stablehlo.slice %1752 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1754 = stablehlo.reshape %1753 : (tensor<1xf64>) -> tensor<f64>
    %1755 = stablehlo.slice %1281 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1756 = stablehlo.reshape %1755 : (tensor<1xf64>) -> tensor<f64>
    %1757 = stablehlo.negate %1756 : tensor<f64>
    %1758 = stablehlo.reshape %1757 : (tensor<f64>) -> tensor<1xf64>
    %1759 = stablehlo.slice %1281 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1760 = stablehlo.reshape %1759 : (tensor<1xf64>) -> tensor<f64>
    %1761 = stablehlo.negate %1760 : tensor<f64>
    %1762 = stablehlo.reshape %1761 : (tensor<f64>) -> tensor<1xf64>
    %1763 = stablehlo.slice %1281 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1764 = stablehlo.reshape %1763 : (tensor<1xf64>) -> tensor<f64>
    %1765 = stablehlo.negate %1764 : tensor<f64>
    %1766 = stablehlo.reshape %1765 : (tensor<f64>) -> tensor<1xf64>
    %1767 = stablehlo.slice %1281 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1768 = stablehlo.reshape %1767 : (tensor<1xf64>) -> tensor<f64>
    %1769 = stablehlo.reshape %1768 : (tensor<f64>) -> tensor<1xf64>
    %1770 = stablehlo.concatenate %1758, %1762, %1766, %1769, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1771 = stablehlo.dot_general %1281, %1281, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1772 = stablehlo.broadcast_in_dim %1771, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1773 = stablehlo.divide %1770, %1772 : tensor<4xf64>
    %1774 = stablehlo.slice %1773 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1775 = stablehlo.reshape %1774 : (tensor<1xf64>) -> tensor<f64>
    %1776 = stablehlo.multiply %1754, %1775 : tensor<f64>
    %1777 = stablehlo.slice %1752 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1778 = stablehlo.reshape %1777 : (tensor<1xf64>) -> tensor<f64>
    %1779 = stablehlo.slice %1773 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1780 = stablehlo.reshape %1779 : (tensor<1xf64>) -> tensor<f64>
    %1781 = stablehlo.multiply %1778, %1780 : tensor<f64>
    %1782 = stablehlo.add %1776, %1781 : tensor<f64>
    %1783 = stablehlo.slice %1752 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1784 = stablehlo.reshape %1783 : (tensor<1xf64>) -> tensor<f64>
    %1785 = stablehlo.slice %1773 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1786 = stablehlo.reshape %1785 : (tensor<1xf64>) -> tensor<f64>
    %1787 = stablehlo.multiply %1784, %1786 : tensor<f64>
    %1788 = stablehlo.add %1782, %1787 : tensor<f64>
    %1789 = stablehlo.slice %1752 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1790 = stablehlo.reshape %1789 : (tensor<1xf64>) -> tensor<f64>
    %1791 = stablehlo.slice %1773 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1792 = stablehlo.reshape %1791 : (tensor<1xf64>) -> tensor<f64>
    %1793 = stablehlo.multiply %1790, %1792 : tensor<f64>
    %1794 = stablehlo.subtract %1788, %1793 : tensor<f64>
    %1795 = stablehlo.reshape %1794 : (tensor<f64>) -> tensor<1xf64>
    %1796 = stablehlo.multiply %1754, %1792 : tensor<f64>
    %1797 = stablehlo.multiply %1778, %1786 : tensor<f64>
    %1798 = stablehlo.subtract %1796, %1797 : tensor<f64>
    %1799 = stablehlo.multiply %1784, %1780 : tensor<f64>
    %1800 = stablehlo.add %1798, %1799 : tensor<f64>
    %1801 = stablehlo.multiply %1790, %1775 : tensor<f64>
    %1802 = stablehlo.add %1800, %1801 : tensor<f64>
    %1803 = stablehlo.reshape %1802 : (tensor<f64>) -> tensor<1xf64>
    %1804 = stablehlo.multiply %1754, %1786 : tensor<f64>
    %1805 = stablehlo.multiply %1778, %1792 : tensor<f64>
    %1806 = stablehlo.add %1804, %1805 : tensor<f64>
    %1807 = stablehlo.multiply %1784, %1775 : tensor<f64>
    %1808 = stablehlo.subtract %1806, %1807 : tensor<f64>
    %1809 = stablehlo.multiply %1790, %1780 : tensor<f64>
    %1810 = stablehlo.add %1808, %1809 : tensor<f64>
    %1811 = stablehlo.reshape %1810 : (tensor<f64>) -> tensor<1xf64>
    %1812 = stablehlo.multiply %1754, %1780 : tensor<f64>
    %1813 = stablehlo.multiply %1778, %1775 : tensor<f64>
    %1814 = stablehlo.subtract %1812, %1813 : tensor<f64>
    %1815 = stablehlo.multiply %1784, %1792 : tensor<f64>
    %1816 = stablehlo.subtract %1814, %1815 : tensor<f64>
    %1817 = stablehlo.multiply %1790, %1786 : tensor<f64>
    %1818 = stablehlo.subtract %1816, %1817 : tensor<f64>
    %1819 = stablehlo.reshape %1818 : (tensor<f64>) -> tensor<1xf64>
    %1820 = stablehlo.concatenate %1795, %1803, %1811, %1819, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1821 = stablehlo.slice %1820 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1822 = stablehlo.reshape %1821 : (tensor<1xf64>) -> tensor<f64>
    %1823 = stablehlo.reshape %1822 : (tensor<f64>) -> tensor<1xf64>
    %1824 = stablehlo.slice %1820 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1825 = stablehlo.reshape %1824 : (tensor<1xf64>) -> tensor<f64>
    %1826 = stablehlo.reshape %1825 : (tensor<f64>) -> tensor<1xf64>
    %1827 = stablehlo.slice %1820 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1828 = stablehlo.reshape %1827 : (tensor<1xf64>) -> tensor<f64>
    %1829 = stablehlo.reshape %1828 : (tensor<f64>) -> tensor<1xf64>
    %1830 = stablehlo.concatenate %1823, %1826, %1829, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1831 = stablehlo.concatenate %1700, %1830, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %cst_22 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %1832 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1833 = stablehlo.multiply %1832, %1831 : tensor<6xf64>
    %1834 = stablehlo.add %1200, %1833 : tensor<6xf64>
    %1835 = stablehlo.slice %1266 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %cst_23 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %1836 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1837 = stablehlo.multiply %1836, %1834 : tensor<6xf64>
    %1838 = stablehlo.slice %1837 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_24 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1839 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1840 = stablehlo.divide %1838, %1839 : tensor<3xf64>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1841 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1842 = stablehlo.concatenate %1840, %1841, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1843 = stablehlo.slice %1842 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1844 = stablehlo.reshape %1843 : (tensor<1xf64>) -> tensor<f64>
    %1845 = stablehlo.slice %1835 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1846 = stablehlo.reshape %1845 : (tensor<1xf64>) -> tensor<f64>
    %1847 = stablehlo.multiply %1844, %1846 : tensor<f64>
    %1848 = stablehlo.slice %1842 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1849 = stablehlo.reshape %1848 : (tensor<1xf64>) -> tensor<f64>
    %1850 = stablehlo.slice %1835 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1851 = stablehlo.reshape %1850 : (tensor<1xf64>) -> tensor<f64>
    %1852 = stablehlo.multiply %1849, %1851 : tensor<f64>
    %1853 = stablehlo.add %1847, %1852 : tensor<f64>
    %1854 = stablehlo.slice %1842 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1855 = stablehlo.reshape %1854 : (tensor<1xf64>) -> tensor<f64>
    %1856 = stablehlo.slice %1835 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1857 = stablehlo.reshape %1856 : (tensor<1xf64>) -> tensor<f64>
    %1858 = stablehlo.multiply %1855, %1857 : tensor<f64>
    %1859 = stablehlo.add %1853, %1858 : tensor<f64>
    %1860 = stablehlo.slice %1842 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1861 = stablehlo.reshape %1860 : (tensor<1xf64>) -> tensor<f64>
    %1862 = stablehlo.slice %1835 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1863 = stablehlo.reshape %1862 : (tensor<1xf64>) -> tensor<f64>
    %1864 = stablehlo.multiply %1861, %1863 : tensor<f64>
    %1865 = stablehlo.subtract %1859, %1864 : tensor<f64>
    %1866 = stablehlo.reshape %1865 : (tensor<f64>) -> tensor<1xf64>
    %1867 = stablehlo.multiply %1844, %1863 : tensor<f64>
    %1868 = stablehlo.multiply %1849, %1857 : tensor<f64>
    %1869 = stablehlo.subtract %1867, %1868 : tensor<f64>
    %1870 = stablehlo.multiply %1855, %1851 : tensor<f64>
    %1871 = stablehlo.add %1869, %1870 : tensor<f64>
    %1872 = stablehlo.multiply %1861, %1846 : tensor<f64>
    %1873 = stablehlo.add %1871, %1872 : tensor<f64>
    %1874 = stablehlo.reshape %1873 : (tensor<f64>) -> tensor<1xf64>
    %1875 = stablehlo.multiply %1844, %1857 : tensor<f64>
    %1876 = stablehlo.multiply %1849, %1863 : tensor<f64>
    %1877 = stablehlo.add %1875, %1876 : tensor<f64>
    %1878 = stablehlo.multiply %1855, %1846 : tensor<f64>
    %1879 = stablehlo.subtract %1877, %1878 : tensor<f64>
    %1880 = stablehlo.multiply %1861, %1851 : tensor<f64>
    %1881 = stablehlo.add %1879, %1880 : tensor<f64>
    %1882 = stablehlo.reshape %1881 : (tensor<f64>) -> tensor<1xf64>
    %1883 = stablehlo.multiply %1844, %1851 : tensor<f64>
    %1884 = stablehlo.multiply %1849, %1846 : tensor<f64>
    %1885 = stablehlo.subtract %1883, %1884 : tensor<f64>
    %1886 = stablehlo.multiply %1855, %1863 : tensor<f64>
    %1887 = stablehlo.subtract %1885, %1886 : tensor<f64>
    %1888 = stablehlo.multiply %1861, %1857 : tensor<f64>
    %1889 = stablehlo.subtract %1887, %1888 : tensor<f64>
    %1890 = stablehlo.reshape %1889 : (tensor<f64>) -> tensor<1xf64>
    %1891 = stablehlo.concatenate %1866, %1874, %1882, %1890, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1892 = stablehlo.add %1835, %1891 : tensor<4xf64>
    %1893 = stablehlo.dot_general %1892, %1892, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1894 = stablehlo.sqrt %1893 : tensor<f64>
    %1895 = stablehlo.broadcast_in_dim %1894, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1896 = stablehlo.divide %1892, %1895 : tensor<4xf64>
    %1897 = stablehlo.slice %1266 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1898 = stablehlo.slice %1837 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1899 = stablehlo.add %1897, %1898 : tensor<3xf64>
    %1900 = stablehlo.concatenate %1896, %1899, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %1901 = call @inner_235(%1267) : (tensor<i64>) -> tensor<i64>
    %1902 = call @inner_236(%1901, %1268) : (tensor<i64>, tensor<3xf64>) -> tensor<3xf64>
    %1903:2 = call @inner_237(%1901, %1900, %1834, %1269#0, %1902, %1269#1) : (tensor<i64>, tensor<7xf64>, tensor<6xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    %1904:2 = call @inner_238(%1901, %1900, %1831, %1270#0, %arg24, %1270#1) : (tensor<i64>, tensor<7xf64>, tensor<6xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>)
    %1905 = call @inner_239(%1904#1, %1903#1, %1271) : (tensor<3xf64>, tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %1906 = call @inner_240(%1901, %1900, %arg27, %1272) : (tensor<i64>, tensor<7xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1907 = call @inner_242(%1900, %1834, %1273) : (tensor<7xf64>, tensor<6xf64>, tensor<3xf64>) -> tensor<3xf64>
    %1908 = call @inner_243(%1276#2, %1274) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %1831, %1906, %arg1, %1, %1904#1, %3, %0, %1905, %4, %1276#1, %arg24, %5, %1902, %1907, %2#0, %1834, %1276#0, %1904#0, %1903#1, %1903#0, %1900, %arg27, %1276#2, %arg12, %1280, %1279, %1278, %2#1, %6, %1908, %1901 : tensor<6xf64>, tensor<3xf64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<i64>, tensor<f64>, tensor<3x3xf64>, tensor<4xf64>, tensor<3xf64>, tensor<4xf64>, tensor<3xf64>, tensor<3xf64>, tensor<4xf64>, tensor<6xf64>, tensor<4xf64>, tensor<4x3xf64>, tensor<3xf64>, tensor<4x3xf64>, tensor<7xf64>, tensor<3xf64>, tensor<4xf64>, tensor<7xf64>, tensor<6xf64>, tensor<6xf64>, tensor<3xf64>, tensor<3xf64>, tensor<4xf64>, tensor<4xf64>, tensor<i64>
  }
  func.func private @inner(%arg0: tensor<i64>, %arg1: tensor<f64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 3.000000e-01, 0.000000e+00], [0.000000e+00, -2.000000e-01, 0.000000e+00], [0.000000e+00, -2.000000e-01, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x3xf64>
    %cst_0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00], [-2.000000e-01, 0.000000e+00, 0.000000e+00], [4.000000e-01, 0.000000e+00, 0.000000e+00], [-2.000000e-01, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<5x3xf64>
    %cst_1 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 1.000000e-01], [0.000000e+00, 0.000000e+00, 1.000000e-01], [0.000000e+00, 0.000000e+00, -2.000000e-01], [0.000000e+00, 0.000000e+00, -2.000000e-01], [0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x3xf64>
    %cst_2 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00], [2.000000e-01, 4.000000e-01, 0.000000e+00], [-3.000000e-01, 4.000000e-01, 0.000000e+00], [1.000000e-01, 1.000000e-01, 0.000000e+00], [3.000000e-01, -4.000000e-01, 0.000000e+00]]> : tensor<5x3xf64>
    %0 = stablehlo.concatenate %cst_2, %cst, %cst_0, %cst_1, dim = 0 : (tensor<5x3xf64>, tensor<6x3xf64>, tensor<5x3xf64>, tensor<6x3xf64>) -> tensor<22x3xf64>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    %2 = stablehlo.multiply %1, %arg1 : tensor<f64>
    %3 = stablehlo.convert %2 : (tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.compare  LT, %3, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_3 = stablehlo.constant dense<22> : tensor<i32>
    %5 = stablehlo.add %3, %c_3 : tensor<i32>
    %6 = stablehlo.select %4, %5, %3 : tensor<i1>, tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.dynamic_slice %0, %6, %c_4, sizes = [1, 3] : (tensor<22x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    %8 = stablehlo.reshape %7 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %8 : tensor<3xf64>
  }
  func.func private @inner_2(%arg0: tensor<3xf64>, %arg1: tensor<4xf64>, %arg2: tensor<3xf64>) -> (tensor<4xf64>, tensor<3xf64>) {
    %cst = stablehlo.constant dense<[1.100000e+05, 1.100000e+05, 2.700000e+04]> : tensor<3xf64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.slice %arg0 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %arg1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.slice %arg1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %9 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
    %10 = stablehlo.slice %arg1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.slice %arg1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %13 = stablehlo.reshape %12 : (tensor<1xf64>) -> tensor<f64>
    %14 = stablehlo.multiply %13, %7 : tensor<f64>
    %15 = stablehlo.multiply %9, %11 : tensor<f64>
    %16 = stablehlo.add %14, %15 : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %17 = stablehlo.multiply %cst_0, %16 : tensor<f64>
    %18 = stablehlo.multiply %7, %7 : tensor<f64>
    %19 = stablehlo.multiply %9, %9 : tensor<f64>
    %20 = stablehlo.add %18, %19 : tensor<f64>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %21 = stablehlo.multiply %cst_1, %20 : tensor<f64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %22 = stablehlo.subtract %cst_2, %21 : tensor<f64>
    %23 = stablehlo.atan2 %17, %22 : tensor<f64>
    %24 = stablehlo.multiply %13, %9 : tensor<f64>
    %25 = stablehlo.multiply %7, %11 : tensor<f64>
    %26 = stablehlo.subtract %24, %25 : tensor<f64>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %27 = stablehlo.multiply %cst_3, %26 : tensor<f64>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %28 = stablehlo.add %cst_4, %27 : tensor<f64>
    %29 = stablehlo.sqrt %28 : tensor<f64>
    %30 = stablehlo.multiply %13, %9 : tensor<f64>
    %31 = stablehlo.multiply %7, %11 : tensor<f64>
    %32 = stablehlo.subtract %30, %31 : tensor<f64>
    %cst_5 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %33 = stablehlo.multiply %cst_5, %32 : tensor<f64>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %34 = stablehlo.subtract %cst_6, %33 : tensor<f64>
    %35 = stablehlo.sqrt %34 : tensor<f64>
    %36 = stablehlo.atan2 %29, %35 : tensor<f64>
    %cst_7 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %37 = stablehlo.multiply %cst_7, %36 : tensor<f64>
    %cst_8 = stablehlo.constant dense<1.5707963267948966> : tensor<f64>
    %38 = stablehlo.subtract %37, %cst_8 : tensor<f64>
    %39 = stablehlo.multiply %13, %11 : tensor<f64>
    %40 = stablehlo.multiply %7, %9 : tensor<f64>
    %41 = stablehlo.add %39, %40 : tensor<f64>
    %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %42 = stablehlo.multiply %cst_9, %41 : tensor<f64>
    %43 = stablehlo.multiply %9, %9 : tensor<f64>
    %44 = stablehlo.multiply %11, %11 : tensor<f64>
    %45 = stablehlo.add %43, %44 : tensor<f64>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %46 = stablehlo.multiply %cst_10, %45 : tensor<f64>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %47 = stablehlo.subtract %cst_11, %46 : tensor<f64>
    %48 = stablehlo.atan2 %42, %47 : tensor<f64>
    %49 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %50 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %51 = stablehlo.broadcast_in_dim %48, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %52 = stablehlo.concatenate %49, %50, %51, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %53 = stablehlo.slice %52 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %54 = stablehlo.reshape %53 : (tensor<1xf64>) -> tensor<f64>
    %55 = stablehlo.slice %52 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %56 = stablehlo.reshape %55 : (tensor<1xf64>) -> tensor<f64>
    %57 = stablehlo.slice %52 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %58 = stablehlo.reshape %57 : (tensor<1xf64>) -> tensor<f64>
    %59 = stablehlo.slice %arg2 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %60 = stablehlo.reshape %59 : (tensor<1xf64>) -> tensor<f64>
    %61 = stablehlo.slice %arg2 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %62 = stablehlo.reshape %61 : (tensor<1xf64>) -> tensor<f64>
    %63 = stablehlo.slice %arg2 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %64 = stablehlo.reshape %63 : (tensor<1xf64>) -> tensor<f64>
    %cst_12 = stablehlo.constant dense<1.000000e-02> : tensor<f64>
    %65 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %66 = stablehlo.multiply %cst, %65 : tensor<3xf64>
    %cst_13 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %67 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %68 = stablehlo.multiply %66, %67 : tensor<3xf64>
    %cst_14 = stablehlo.constant dense<1.800000e+02> : tensor<f64>
    %69 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %70 = stablehlo.divide %68, %69 : tensor<3xf64>
    %71 = stablehlo.slice %70 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %72 = stablehlo.reshape %71 : (tensor<1xf64>) -> tensor<f64>
    %73 = stablehlo.slice %70 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.slice %70 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %76 = stablehlo.reshape %75 : (tensor<1xf64>) -> tensor<f64>
    %77 = stablehlo.slice %arg1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.slice %arg1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %80 = stablehlo.reshape %79 : (tensor<1xf64>) -> tensor<f64>
    %81 = stablehlo.slice %arg1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %82 = stablehlo.reshape %81 : (tensor<1xf64>) -> tensor<f64>
    %83 = stablehlo.slice %arg1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %84 = stablehlo.reshape %83 : (tensor<1xf64>) -> tensor<f64>
    %85 = stablehlo.multiply %84, %78 : tensor<f64>
    %86 = stablehlo.multiply %80, %82 : tensor<f64>
    %87 = stablehlo.add %85, %86 : tensor<f64>
    %cst_15 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %88 = stablehlo.multiply %cst_15, %87 : tensor<f64>
    %89 = stablehlo.multiply %78, %78 : tensor<f64>
    %90 = stablehlo.multiply %80, %80 : tensor<f64>
    %91 = stablehlo.add %89, %90 : tensor<f64>
    %cst_16 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %92 = stablehlo.multiply %cst_16, %91 : tensor<f64>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %93 = stablehlo.subtract %cst_17, %92 : tensor<f64>
    %94 = stablehlo.atan2 %88, %93 : tensor<f64>
    %95 = stablehlo.multiply %84, %80 : tensor<f64>
    %96 = stablehlo.multiply %78, %82 : tensor<f64>
    %97 = stablehlo.subtract %95, %96 : tensor<f64>
    %cst_18 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %98 = stablehlo.multiply %cst_18, %97 : tensor<f64>
    %cst_19 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %99 = stablehlo.add %cst_19, %98 : tensor<f64>
    %100 = stablehlo.sqrt %99 : tensor<f64>
    %101 = stablehlo.multiply %84, %80 : tensor<f64>
    %102 = stablehlo.multiply %78, %82 : tensor<f64>
    %103 = stablehlo.subtract %101, %102 : tensor<f64>
    %cst_20 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %104 = stablehlo.multiply %cst_20, %103 : tensor<f64>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %105 = stablehlo.subtract %cst_21, %104 : tensor<f64>
    %106 = stablehlo.sqrt %105 : tensor<f64>
    %107 = stablehlo.atan2 %100, %106 : tensor<f64>
    %cst_22 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %108 = stablehlo.multiply %cst_22, %107 : tensor<f64>
    %cst_23 = stablehlo.constant dense<1.5707963267948966> : tensor<f64>
    %109 = stablehlo.subtract %108, %cst_23 : tensor<f64>
    %110 = stablehlo.multiply %84, %82 : tensor<f64>
    %111 = stablehlo.multiply %78, %80 : tensor<f64>
    %112 = stablehlo.add %110, %111 : tensor<f64>
    %cst_24 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %113 = stablehlo.multiply %cst_24, %112 : tensor<f64>
    %114 = stablehlo.multiply %80, %80 : tensor<f64>
    %115 = stablehlo.multiply %82, %82 : tensor<f64>
    %116 = stablehlo.add %114, %115 : tensor<f64>
    %cst_25 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %117 = stablehlo.multiply %cst_25, %116 : tensor<f64>
    %cst_26 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %118 = stablehlo.subtract %cst_26, %117 : tensor<f64>
    %119 = stablehlo.atan2 %113, %118 : tensor<f64>
    %120 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %121 = stablehlo.broadcast_in_dim %109, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %122 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.concatenate %120, %121, %122, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %124 = stablehlo.slice %123 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %125 = stablehlo.reshape %124 : (tensor<1xf64>) -> tensor<f64>
    %126 = stablehlo.slice %123 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %127 = stablehlo.reshape %126 : (tensor<1xf64>) -> tensor<f64>
    %128 = stablehlo.slice %123 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %129 = stablehlo.reshape %128 : (tensor<1xf64>) -> tensor<f64>
    %130 = stablehlo.sine %125 : tensor<f64>
    %131 = stablehlo.abs %130 : tensor<f64>
    %cst_27 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_28 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %132 = call @clip(%131, %cst_27, %cst_28) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %133 = stablehlo.cosine %125 : tensor<f64>
    %134 = stablehlo.abs %133 : tensor<f64>
    %cst_29 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_30 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %135 = call @clip(%134, %cst_29, %cst_30) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %136 = stablehlo.sine %127 : tensor<f64>
    %137 = stablehlo.abs %136 : tensor<f64>
    %cst_31 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_32 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %138 = call @clip(%137, %cst_31, %cst_32) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %139 = stablehlo.cosine %127 : tensor<f64>
    %140 = stablehlo.abs %139 : tensor<f64>
    %cst_33 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %cst_34 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %141 = call @clip(%140, %cst_33, %cst_34) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %142 = stablehlo.divide %74, %135 : tensor<f64>
    %143 = stablehlo.divide %76, %132 : tensor<f64>
    %144 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %145 = stablehlo.broadcast_in_dim %143, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.concatenate %144, %145, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    %cst_35 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %147 = stablehlo.reduce(%146 init: %cst_35) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %148 = stablehlo.divide %72, %138 : tensor<f64>
    %149 = stablehlo.multiply %132, %141 : tensor<f64>
    %150 = stablehlo.divide %74, %149 : tensor<f64>
    %151 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %152 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %153 = stablehlo.concatenate %151, %152, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    %cst_36 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %154 = stablehlo.reduce(%153 init: %cst_36) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %155 = stablehlo.multiply %135, %141 : tensor<f64>
    %156 = stablehlo.divide %76, %155 : tensor<f64>
    %157 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %158 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %159 = stablehlo.concatenate %157, %158, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    %cst_37 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %160 = stablehlo.reduce(%159 init: %cst_37) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %161 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %162 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %163 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %164 = stablehlo.concatenate %161, %162, %163, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %165 = stablehlo.slice %164 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %164 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %168 = stablehlo.reshape %167 : (tensor<1xf64>) -> tensor<f64>
    %169 = stablehlo.slice %164 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %170 = stablehlo.reshape %169 : (tensor<1xf64>) -> tensor<f64>
    %171 = stablehlo.subtract %1, %54 : tensor<f64>
    %cst_38 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %172 = call @remainder(%171, %cst_38) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_39 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %173 = stablehlo.compare  LT, %172, %cst_39,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_40 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %174 = stablehlo.add %172, %cst_40 : tensor<f64>
    %175 = call @_where(%173, %174, %172) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_41 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %176 = stablehlo.compare  GT, %175, %cst_41,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_42 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %177 = stablehlo.subtract %175, %cst_42 : tensor<f64>
    %178 = call @_where(%176, %177, %175) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_43 = stablehlo.constant dense<2.500000e+01> : tensor<f64>
    %179 = stablehlo.divide %166, %cst_43 : tensor<f64>
    %180 = stablehlo.abs %178 : tensor<f64>
    %181 = stablehlo.compare  GT, %180, %179,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %182 = stablehlo.sign %178 : tensor<f64>
    %cst_44 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %183 = stablehlo.multiply %cst_44, %166 : tensor<f64>
    %184 = stablehlo.sign %178 : tensor<f64>
    %185 = stablehlo.multiply %184, %178 : tensor<f64>
    %cst_45 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %186 = stablehlo.divide %179, %cst_45 : tensor<f64>
    %187 = stablehlo.subtract %185, %186 : tensor<f64>
    %188 = stablehlo.multiply %183, %187 : tensor<f64>
    %189 = stablehlo.sqrt %188 : tensor<f64>
    %190 = stablehlo.multiply %182, %189 : tensor<f64>
    %cst_46 = stablehlo.constant dense<5.000000e+00> : tensor<f64>
    %191 = stablehlo.multiply %178, %cst_46 : tensor<f64>
    %192 = call @_where(%181, %190, %191) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %193 = stablehlo.abs %178 : tensor<f64>
    %194 = stablehlo.negate %193 : tensor<f64>
    %cst_47 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %195 = stablehlo.divide %194, %cst_47 : tensor<f64>
    %196 = stablehlo.abs %178 : tensor<f64>
    %cst_48 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %197 = stablehlo.divide %196, %cst_48 : tensor<f64>
    %198 = call @clip_19(%192, %195, %197) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %199 = stablehlo.subtract %198, %60 : tensor<f64>
    %cst_49 = stablehlo.constant dense<1.000000e+02> : tensor<f64>
    %200 = stablehlo.multiply %199, %cst_49 : tensor<f64>
    %201 = stablehlo.abs %199 : tensor<f64>
    %202 = stablehlo.negate %201 : tensor<f64>
    %cst_50 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %203 = stablehlo.divide %202, %cst_50 : tensor<f64>
    %204 = stablehlo.abs %199 : tensor<f64>
    %cst_51 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %205 = stablehlo.divide %204, %cst_51 : tensor<f64>
    %206 = call @clip_19(%200, %203, %205) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_52 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %207 = stablehlo.multiply %206, %cst_52 : tensor<f64>
    %208 = stablehlo.add %60, %207 : tensor<f64>
    %cst_53 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %209 = stablehlo.multiply %166, %cst_53 : tensor<f64>
    %210 = stablehlo.subtract %60, %209 : tensor<f64>
    %211 = stablehlo.add %60, %209 : tensor<f64>
    %212 = call @clip_19(%208, %210, %211) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %213 = stablehlo.subtract %3, %56 : tensor<f64>
    %cst_54 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %214 = call @remainder(%213, %cst_54) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %215 = stablehlo.compare  LT, %214, %cst_55,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_56 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %216 = stablehlo.add %214, %cst_56 : tensor<f64>
    %217 = call @_where(%215, %216, %214) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_57 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %218 = stablehlo.compare  GT, %217, %cst_57,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_58 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %219 = stablehlo.subtract %217, %cst_58 : tensor<f64>
    %220 = call @_where(%218, %219, %217) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_59 = stablehlo.constant dense<2.500000e+01> : tensor<f64>
    %221 = stablehlo.divide %168, %cst_59 : tensor<f64>
    %222 = stablehlo.abs %220 : tensor<f64>
    %223 = stablehlo.compare  GT, %222, %221,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %224 = stablehlo.sign %220 : tensor<f64>
    %cst_60 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %225 = stablehlo.multiply %cst_60, %168 : tensor<f64>
    %226 = stablehlo.sign %220 : tensor<f64>
    %227 = stablehlo.multiply %226, %220 : tensor<f64>
    %cst_61 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %228 = stablehlo.divide %221, %cst_61 : tensor<f64>
    %229 = stablehlo.subtract %227, %228 : tensor<f64>
    %230 = stablehlo.multiply %225, %229 : tensor<f64>
    %231 = stablehlo.sqrt %230 : tensor<f64>
    %232 = stablehlo.multiply %224, %231 : tensor<f64>
    %cst_62 = stablehlo.constant dense<5.000000e+00> : tensor<f64>
    %233 = stablehlo.multiply %220, %cst_62 : tensor<f64>
    %234 = call @_where(%223, %232, %233) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %235 = stablehlo.abs %220 : tensor<f64>
    %236 = stablehlo.negate %235 : tensor<f64>
    %cst_63 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %237 = stablehlo.divide %236, %cst_63 : tensor<f64>
    %238 = stablehlo.abs %220 : tensor<f64>
    %cst_64 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %239 = stablehlo.divide %238, %cst_64 : tensor<f64>
    %240 = call @clip_19(%234, %237, %239) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %241 = stablehlo.subtract %240, %62 : tensor<f64>
    %cst_65 = stablehlo.constant dense<1.000000e+02> : tensor<f64>
    %242 = stablehlo.multiply %241, %cst_65 : tensor<f64>
    %243 = stablehlo.abs %241 : tensor<f64>
    %244 = stablehlo.negate %243 : tensor<f64>
    %cst_66 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %245 = stablehlo.divide %244, %cst_66 : tensor<f64>
    %246 = stablehlo.abs %241 : tensor<f64>
    %cst_67 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %247 = stablehlo.divide %246, %cst_67 : tensor<f64>
    %248 = call @clip_19(%242, %245, %247) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_68 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %249 = stablehlo.multiply %248, %cst_68 : tensor<f64>
    %250 = stablehlo.add %62, %249 : tensor<f64>
    %cst_69 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %251 = stablehlo.multiply %168, %cst_69 : tensor<f64>
    %252 = stablehlo.subtract %62, %251 : tensor<f64>
    %253 = stablehlo.add %62, %251 : tensor<f64>
    %254 = call @clip_19(%250, %252, %253) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %255 = stablehlo.subtract %5, %64 : tensor<f64>
    %cst_70 = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %256 = stablehlo.multiply %255, %cst_70 : tensor<f64>
    %257 = stablehlo.abs %255 : tensor<f64>
    %258 = stablehlo.negate %257 : tensor<f64>
    %cst_71 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %259 = stablehlo.divide %258, %cst_71 : tensor<f64>
    %260 = stablehlo.abs %255 : tensor<f64>
    %cst_72 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %261 = stablehlo.divide %260, %cst_72 : tensor<f64>
    %262 = call @clip_19(%256, %259, %261) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_73 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %263 = stablehlo.multiply %262, %cst_73 : tensor<f64>
    %264 = stablehlo.add %64, %263 : tensor<f64>
    %cst_74 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %265 = stablehlo.multiply %170, %cst_74 : tensor<f64>
    %266 = stablehlo.subtract %64, %265 : tensor<f64>
    %267 = stablehlo.add %64, %265 : tensor<f64>
    %268 = call @clip_19(%264, %266, %267) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %269 = stablehlo.broadcast_in_dim %212, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %270 = stablehlo.broadcast_in_dim %254, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %271 = stablehlo.broadcast_in_dim %268, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %272 = stablehlo.concatenate %269, %270, %271, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %273 = stablehlo.slice %arg1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %274 = stablehlo.reshape %273 : (tensor<1xf64>) -> tensor<f64>
    %275 = stablehlo.slice %arg1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %276 = stablehlo.reshape %275 : (tensor<1xf64>) -> tensor<f64>
    %277 = stablehlo.slice %arg1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %278 = stablehlo.reshape %277 : (tensor<1xf64>) -> tensor<f64>
    %279 = stablehlo.slice %arg1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %280 = stablehlo.reshape %279 : (tensor<1xf64>) -> tensor<f64>
    %281 = stablehlo.multiply %280, %274 : tensor<f64>
    %282 = stablehlo.multiply %276, %278 : tensor<f64>
    %283 = stablehlo.add %281, %282 : tensor<f64>
    %cst_75 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %284 = stablehlo.multiply %cst_75, %283 : tensor<f64>
    %285 = stablehlo.multiply %274, %274 : tensor<f64>
    %286 = stablehlo.multiply %276, %276 : tensor<f64>
    %287 = stablehlo.add %285, %286 : tensor<f64>
    %cst_76 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %288 = stablehlo.multiply %cst_76, %287 : tensor<f64>
    %cst_77 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %289 = stablehlo.subtract %cst_77, %288 : tensor<f64>
    %290 = stablehlo.atan2 %284, %289 : tensor<f64>
    %291 = stablehlo.multiply %280, %276 : tensor<f64>
    %292 = stablehlo.multiply %274, %278 : tensor<f64>
    %293 = stablehlo.subtract %291, %292 : tensor<f64>
    %cst_78 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %294 = stablehlo.multiply %cst_78, %293 : tensor<f64>
    %cst_79 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %295 = stablehlo.add %cst_79, %294 : tensor<f64>
    %296 = stablehlo.sqrt %295 : tensor<f64>
    %297 = stablehlo.multiply %280, %276 : tensor<f64>
    %298 = stablehlo.multiply %274, %278 : tensor<f64>
    %299 = stablehlo.subtract %297, %298 : tensor<f64>
    %cst_80 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %300 = stablehlo.multiply %cst_80, %299 : tensor<f64>
    %cst_81 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %301 = stablehlo.subtract %cst_81, %300 : tensor<f64>
    %302 = stablehlo.sqrt %301 : tensor<f64>
    %303 = stablehlo.atan2 %296, %302 : tensor<f64>
    %cst_82 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %304 = stablehlo.multiply %cst_82, %303 : tensor<f64>
    %cst_83 = stablehlo.constant dense<1.5707963267948966> : tensor<f64>
    %305 = stablehlo.subtract %304, %cst_83 : tensor<f64>
    %306 = stablehlo.multiply %280, %278 : tensor<f64>
    %307 = stablehlo.multiply %274, %276 : tensor<f64>
    %308 = stablehlo.add %306, %307 : tensor<f64>
    %cst_84 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %309 = stablehlo.multiply %cst_84, %308 : tensor<f64>
    %310 = stablehlo.multiply %276, %276 : tensor<f64>
    %311 = stablehlo.multiply %278, %278 : tensor<f64>
    %312 = stablehlo.add %310, %311 : tensor<f64>
    %cst_85 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %313 = stablehlo.multiply %cst_85, %312 : tensor<f64>
    %cst_86 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %314 = stablehlo.subtract %cst_86, %313 : tensor<f64>
    %315 = stablehlo.atan2 %309, %314 : tensor<f64>
    %316 = stablehlo.broadcast_in_dim %290, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %317 = stablehlo.broadcast_in_dim %305, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %318 = stablehlo.broadcast_in_dim %315, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %319 = stablehlo.concatenate %316, %317, %318, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %320 = stablehlo.slice %319 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %321 = stablehlo.reshape %320 : (tensor<1xf64>) -> tensor<f64>
    %322 = stablehlo.slice %319 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %323 = stablehlo.reshape %322 : (tensor<1xf64>) -> tensor<f64>
    %324 = stablehlo.slice %319 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %325 = stablehlo.reshape %324 : (tensor<1xf64>) -> tensor<f64>
    %326 = stablehlo.sine %323 : tensor<f64>
    %327 = stablehlo.negate %326 : tensor<f64>
    %328 = stablehlo.cosine %321 : tensor<f64>
    %329 = stablehlo.sine %321 : tensor<f64>
    %330 = stablehlo.cosine %323 : tensor<f64>
    %331 = stablehlo.multiply %329, %330 : tensor<f64>
    %332 = stablehlo.sine %321 : tensor<f64>
    %333 = stablehlo.negate %332 : tensor<f64>
    %334 = stablehlo.cosine %321 : tensor<f64>
    %335 = stablehlo.cosine %323 : tensor<f64>
    %336 = stablehlo.multiply %334, %335 : tensor<f64>
    %cst_87 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %337 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %338 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %339 = stablehlo.broadcast_in_dim %327, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %340 = stablehlo.concatenate %337, %338, %339, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %341 = stablehlo.broadcast_in_dim %340, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %342 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %343 = stablehlo.broadcast_in_dim %328, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %344 = stablehlo.broadcast_in_dim %331, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %345 = stablehlo.concatenate %342, %343, %344, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %346 = stablehlo.broadcast_in_dim %345, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %347 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %348 = stablehlo.broadcast_in_dim %333, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %349 = stablehlo.broadcast_in_dim %336, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %350 = stablehlo.concatenate %347, %348, %349, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %351 = stablehlo.broadcast_in_dim %350, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %352 = stablehlo.concatenate %341, %346, %351, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<3x3xf64>
    %353 = stablehlo.dot_general %352, %272, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %354 = call @nan_to_num(%353) : (tensor<3xf64>) -> tensor<3xf64>
    %cst_91 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %355 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %356 = stablehlo.multiply %354, %355 : tensor<3xf64>
    %357 = call @norm(%356) : (tensor<3xf64>) -> tensor<f64>
    %cst_92 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %358 = stablehlo.compare  LT, %357, %cst_92,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %359 = stablehlo.convert %358 : (tensor<i1>) -> tensor<i32>
    %360 = "stablehlo.case"(%359) ({
      %410 = stablehlo.broadcast_in_dim %357, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %411 = stablehlo.divide %356, %410 : tensor<3xf64>
      %412 = stablehlo.dot_general %411, %411, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
      %413 = stablehlo.sqrt %412 : tensor<f64>
      %414 = stablehlo.broadcast_in_dim %413, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %415 = stablehlo.divide %411, %414 : tensor<3xf64>
      %cst_93 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %416 = stablehlo.divide %357, %cst_93 : tensor<f64>
      %417 = stablehlo.sine %416 : tensor<f64>
      %418 = stablehlo.broadcast_in_dim %417, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %419 = stablehlo.multiply %415, %418 : tensor<3xf64>
      %420 = stablehlo.cosine %416 : tensor<f64>
      %421 = stablehlo.broadcast_in_dim %420, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %422 = stablehlo.concatenate %419, %421, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      %423 = stablehlo.dot_general %411, %411, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
      %424 = stablehlo.sqrt %423 : tensor<f64>
      %425 = stablehlo.broadcast_in_dim %424, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %426 = stablehlo.divide %411, %425 : tensor<3xf64>
      %cst_94 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %427 = stablehlo.divide %357, %cst_94 : tensor<f64>
      %428 = stablehlo.sine %427 : tensor<f64>
      %429 = stablehlo.broadcast_in_dim %428, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %430 = stablehlo.multiply %426, %429 : tensor<3xf64>
      %431 = stablehlo.cosine %427 : tensor<f64>
      %432 = stablehlo.broadcast_in_dim %431, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %433 = stablehlo.concatenate %430, %432, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      stablehlo.return %433 : tensor<4xf64>
    }, {
      %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %410 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst_94 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %411 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %412 = stablehlo.concatenate %410, %411, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %413 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst_96 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %414 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %415 = stablehlo.concatenate %413, %414, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      stablehlo.return %415 : tensor<4xf64>
    }) : (tensor<i32>) -> tensor<4xf64>
    %361 = stablehlo.slice %arg1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %362 = stablehlo.reshape %361 : (tensor<1xf64>) -> tensor<f64>
    %363 = stablehlo.slice %360 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %364 = stablehlo.reshape %363 : (tensor<1xf64>) -> tensor<f64>
    %365 = stablehlo.multiply %362, %364 : tensor<f64>
    %366 = stablehlo.slice %arg1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %367 = stablehlo.reshape %366 : (tensor<1xf64>) -> tensor<f64>
    %368 = stablehlo.slice %360 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %369 = stablehlo.reshape %368 : (tensor<1xf64>) -> tensor<f64>
    %370 = stablehlo.multiply %367, %369 : tensor<f64>
    %371 = stablehlo.add %365, %370 : tensor<f64>
    %372 = stablehlo.slice %arg1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %373 = stablehlo.reshape %372 : (tensor<1xf64>) -> tensor<f64>
    %374 = stablehlo.slice %360 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %375 = stablehlo.reshape %374 : (tensor<1xf64>) -> tensor<f64>
    %376 = stablehlo.multiply %373, %375 : tensor<f64>
    %377 = stablehlo.add %371, %376 : tensor<f64>
    %378 = stablehlo.slice %arg1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %379 = stablehlo.reshape %378 : (tensor<1xf64>) -> tensor<f64>
    %380 = stablehlo.slice %360 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %381 = stablehlo.reshape %380 : (tensor<1xf64>) -> tensor<f64>
    %382 = stablehlo.multiply %379, %381 : tensor<f64>
    %383 = stablehlo.subtract %377, %382 : tensor<f64>
    %384 = stablehlo.reshape %383 : (tensor<f64>) -> tensor<1xf64>
    %385 = stablehlo.multiply %362, %381 : tensor<f64>
    %386 = stablehlo.multiply %367, %375 : tensor<f64>
    %387 = stablehlo.subtract %385, %386 : tensor<f64>
    %388 = stablehlo.multiply %373, %369 : tensor<f64>
    %389 = stablehlo.add %387, %388 : tensor<f64>
    %390 = stablehlo.multiply %379, %364 : tensor<f64>
    %391 = stablehlo.add %389, %390 : tensor<f64>
    %392 = stablehlo.reshape %391 : (tensor<f64>) -> tensor<1xf64>
    %393 = stablehlo.multiply %362, %375 : tensor<f64>
    %394 = stablehlo.multiply %367, %381 : tensor<f64>
    %395 = stablehlo.add %393, %394 : tensor<f64>
    %396 = stablehlo.multiply %373, %364 : tensor<f64>
    %397 = stablehlo.subtract %395, %396 : tensor<f64>
    %398 = stablehlo.multiply %379, %369 : tensor<f64>
    %399 = stablehlo.add %397, %398 : tensor<f64>
    %400 = stablehlo.reshape %399 : (tensor<f64>) -> tensor<1xf64>
    %401 = stablehlo.multiply %362, %369 : tensor<f64>
    %402 = stablehlo.multiply %367, %364 : tensor<f64>
    %403 = stablehlo.subtract %401, %402 : tensor<f64>
    %404 = stablehlo.multiply %373, %381 : tensor<f64>
    %405 = stablehlo.subtract %403, %404 : tensor<f64>
    %406 = stablehlo.multiply %379, %375 : tensor<f64>
    %407 = stablehlo.subtract %405, %406 : tensor<f64>
    %408 = stablehlo.reshape %407 : (tensor<f64>) -> tensor<1xf64>
    %409 = stablehlo.concatenate %384, %392, %400, %408, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    return %409, %272 : tensor<4xf64>, tensor<3xf64>
  }
  func.func private @clip(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f64>
    %2 = stablehlo.convert %arg2 : tensor<f64>
    %3 = stablehlo.minimum %2, %1 : tensor<f64>
    return %3 : tensor<f64>
  }
  func.func private @remainder(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.remainder %arg0, %0 : tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = stablehlo.compare  NE, %1, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %3 = stablehlo.compare  LT, %1, %cst_0,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %4 = stablehlo.compare  LT, %0, %cst_1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %5 = stablehlo.compare  NE, %3, %4,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = stablehlo.and %5, %2 : tensor<i1>
    %7 = stablehlo.add %1, %0 : tensor<f64>
    %8 = stablehlo.select %6, %7, %1 : tensor<i1>, tensor<f64>
    return %8 : tensor<f64>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @clip_19(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<f64>
    %1 = stablehlo.minimum %arg2, %0 : tensor<f64>
    return %1 : tensor<f64>
  }
  func.func private @nan_to_num(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = call @_where_23(%0, %cst, %arg0) : (tensor<3xi1>, tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.compare  EQ, %1, %2,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %cst_1 = stablehlo.constant dense<1.7976931348623157E+308> : tensor<f64>
    %4 = call @_where_23(%3, %cst_1, %1) : (tensor<3xi1>, tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    %cst_2 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.compare  EQ, %4, %5,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %cst_3 = stablehlo.constant dense<-1.7976931348623157E+308> : tensor<f64>
    %7 = call @_where_23(%6, %cst_3, %4) : (tensor<3xi1>, tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    return %7 : tensor<3xf64>
  }
  func.func private @_where_23(%arg0: tensor<3xi1>, %arg1: tensor<f64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<3xi1>, tensor<3xf64>
    return %1 : tensor<3xf64>
  }
  func.func private @norm(%arg0: tensor<3xf64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<3xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.sqrt %1 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @inner_39(%arg0: tensor<7xf64>, %arg1: tensor<3xf64>, %arg2: tensor<4xf64>, %arg3: tensor<3xf64>, %arg4: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00]> : tensor<3xf64>
    %cst_1 = stablehlo.constant dense<[4.000000e+00, 4.000000e+00, 1.000000e+00]> : tensor<3xf64>
    %cst_2 = stablehlo.constant dense<2.000000e+01> : tensor<3xf64>
    %0 = stablehlo.slice %cst [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.slice %cst [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.slice %cst [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %6 = stablehlo.multiply %5, %cst_3 : tensor<f64>
    %7 = stablehlo.cosine %6 : tensor<f64>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %8 = stablehlo.multiply %5, %cst_4 : tensor<f64>
    %9 = stablehlo.sine %8 : tensor<f64>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %10 = stablehlo.multiply %3, %cst_5 : tensor<f64>
    %11 = stablehlo.cosine %10 : tensor<f64>
    %cst_6 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %12 = stablehlo.multiply %3, %cst_6 : tensor<f64>
    %13 = stablehlo.sine %12 : tensor<f64>
    %cst_7 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %14 = stablehlo.multiply %1, %cst_7 : tensor<f64>
    %15 = stablehlo.cosine %14 : tensor<f64>
    %cst_8 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %16 = stablehlo.multiply %1, %cst_8 : tensor<f64>
    %17 = stablehlo.sine %16 : tensor<f64>
    %18 = stablehlo.multiply %15, %11 : tensor<f64>
    %19 = stablehlo.multiply %18, %7 : tensor<f64>
    %20 = stablehlo.multiply %17, %13 : tensor<f64>
    %21 = stablehlo.multiply %20, %9 : tensor<f64>
    %22 = stablehlo.add %19, %21 : tensor<f64>
    %23 = stablehlo.multiply %17, %11 : tensor<f64>
    %24 = stablehlo.multiply %23, %7 : tensor<f64>
    %25 = stablehlo.multiply %15, %13 : tensor<f64>
    %26 = stablehlo.multiply %25, %9 : tensor<f64>
    %27 = stablehlo.subtract %24, %26 : tensor<f64>
    %28 = stablehlo.multiply %15, %13 : tensor<f64>
    %29 = stablehlo.multiply %28, %7 : tensor<f64>
    %30 = stablehlo.multiply %17, %11 : tensor<f64>
    %31 = stablehlo.multiply %30, %9 : tensor<f64>
    %32 = stablehlo.add %29, %31 : tensor<f64>
    %33 = stablehlo.multiply %15, %11 : tensor<f64>
    %34 = stablehlo.multiply %33, %9 : tensor<f64>
    %35 = stablehlo.multiply %17, %13 : tensor<f64>
    %36 = stablehlo.multiply %35, %7 : tensor<f64>
    %37 = stablehlo.subtract %34, %36 : tensor<f64>
    %38 = stablehlo.broadcast_in_dim %27, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %39 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %40 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %41 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %42 = stablehlo.concatenate %38, %39, %40, %41, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %43 = stablehlo.slice %arg2 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %44 = stablehlo.reshape %43 : (tensor<1xf64>) -> tensor<f64>
    %45 = stablehlo.slice %42 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %46 = stablehlo.reshape %45 : (tensor<1xf64>) -> tensor<f64>
    %47 = stablehlo.multiply %44, %46 : tensor<f64>
    %48 = stablehlo.slice %arg2 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %49 = stablehlo.reshape %48 : (tensor<1xf64>) -> tensor<f64>
    %50 = stablehlo.slice %42 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %51 = stablehlo.reshape %50 : (tensor<1xf64>) -> tensor<f64>
    %52 = stablehlo.multiply %49, %51 : tensor<f64>
    %53 = stablehlo.add %47, %52 : tensor<f64>
    %54 = stablehlo.slice %arg2 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %55 = stablehlo.reshape %54 : (tensor<1xf64>) -> tensor<f64>
    %56 = stablehlo.slice %42 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %57 = stablehlo.reshape %56 : (tensor<1xf64>) -> tensor<f64>
    %58 = stablehlo.multiply %55, %57 : tensor<f64>
    %59 = stablehlo.add %53, %58 : tensor<f64>
    %60 = stablehlo.slice %arg2 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %61 = stablehlo.reshape %60 : (tensor<1xf64>) -> tensor<f64>
    %62 = stablehlo.slice %42 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %63 = stablehlo.reshape %62 : (tensor<1xf64>) -> tensor<f64>
    %64 = stablehlo.multiply %61, %63 : tensor<f64>
    %65 = stablehlo.subtract %59, %64 : tensor<f64>
    %66 = stablehlo.reshape %65 : (tensor<f64>) -> tensor<1xf64>
    %67 = stablehlo.multiply %44, %63 : tensor<f64>
    %68 = stablehlo.multiply %49, %57 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.multiply %55, %51 : tensor<f64>
    %71 = stablehlo.add %69, %70 : tensor<f64>
    %72 = stablehlo.multiply %61, %46 : tensor<f64>
    %73 = stablehlo.add %71, %72 : tensor<f64>
    %74 = stablehlo.reshape %73 : (tensor<f64>) -> tensor<1xf64>
    %75 = stablehlo.multiply %44, %57 : tensor<f64>
    %76 = stablehlo.multiply %49, %63 : tensor<f64>
    %77 = stablehlo.add %75, %76 : tensor<f64>
    %78 = stablehlo.multiply %55, %46 : tensor<f64>
    %79 = stablehlo.subtract %77, %78 : tensor<f64>
    %80 = stablehlo.multiply %61, %51 : tensor<f64>
    %81 = stablehlo.add %79, %80 : tensor<f64>
    %82 = stablehlo.reshape %81 : (tensor<f64>) -> tensor<1xf64>
    %83 = stablehlo.multiply %44, %51 : tensor<f64>
    %84 = stablehlo.multiply %49, %46 : tensor<f64>
    %85 = stablehlo.subtract %83, %84 : tensor<f64>
    %86 = stablehlo.multiply %55, %63 : tensor<f64>
    %87 = stablehlo.subtract %85, %86 : tensor<f64>
    %88 = stablehlo.multiply %61, %57 : tensor<f64>
    %89 = stablehlo.subtract %87, %88 : tensor<f64>
    %90 = stablehlo.reshape %89 : (tensor<f64>) -> tensor<1xf64>
    %91 = stablehlo.concatenate %66, %74, %82, %90, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %92 = stablehlo.slice %91 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.slice %91 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %95 = stablehlo.reshape %94 : (tensor<1xf64>) -> tensor<f64>
    %96 = stablehlo.slice %91 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.slice %91 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.multiply %99, %93 : tensor<f64>
    %101 = stablehlo.multiply %95, %97 : tensor<f64>
    %102 = stablehlo.add %100, %101 : tensor<f64>
    %cst_9 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %103 = stablehlo.multiply %cst_9, %102 : tensor<f64>
    %104 = stablehlo.multiply %93, %93 : tensor<f64>
    %105 = stablehlo.multiply %95, %95 : tensor<f64>
    %106 = stablehlo.add %104, %105 : tensor<f64>
    %cst_10 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %107 = stablehlo.multiply %cst_10, %106 : tensor<f64>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %108 = stablehlo.subtract %cst_11, %107 : tensor<f64>
    %109 = stablehlo.atan2 %103, %108 : tensor<f64>
    %110 = stablehlo.multiply %99, %95 : tensor<f64>
    %111 = stablehlo.multiply %93, %97 : tensor<f64>
    %112 = stablehlo.subtract %110, %111 : tensor<f64>
    %cst_12 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %113 = stablehlo.multiply %cst_12, %112 : tensor<f64>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %114 = stablehlo.add %cst_13, %113 : tensor<f64>
    %115 = stablehlo.sqrt %114 : tensor<f64>
    %116 = stablehlo.multiply %99, %95 : tensor<f64>
    %117 = stablehlo.multiply %93, %97 : tensor<f64>
    %118 = stablehlo.subtract %116, %117 : tensor<f64>
    %cst_14 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %119 = stablehlo.multiply %cst_14, %118 : tensor<f64>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %120 = stablehlo.subtract %cst_15, %119 : tensor<f64>
    %121 = stablehlo.sqrt %120 : tensor<f64>
    %122 = stablehlo.atan2 %115, %121 : tensor<f64>
    %cst_16 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %123 = stablehlo.multiply %cst_16, %122 : tensor<f64>
    %cst_17 = stablehlo.constant dense<1.5707963267948966> : tensor<f64>
    %124 = stablehlo.subtract %123, %cst_17 : tensor<f64>
    %125 = stablehlo.multiply %99, %97 : tensor<f64>
    %126 = stablehlo.multiply %93, %95 : tensor<f64>
    %127 = stablehlo.add %125, %126 : tensor<f64>
    %cst_18 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %128 = stablehlo.multiply %cst_18, %127 : tensor<f64>
    %129 = stablehlo.multiply %95, %95 : tensor<f64>
    %130 = stablehlo.multiply %97, %97 : tensor<f64>
    %131 = stablehlo.add %129, %130 : tensor<f64>
    %cst_19 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %132 = stablehlo.multiply %cst_19, %131 : tensor<f64>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %133 = stablehlo.subtract %cst_20, %132 : tensor<f64>
    %134 = stablehlo.atan2 %128, %133 : tensor<f64>
    %135 = stablehlo.broadcast_in_dim %109, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %136 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %137 = stablehlo.broadcast_in_dim %134, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.concatenate %135, %136, %137, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %139 = stablehlo.slice %138 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.slice %138 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %142 = stablehlo.reshape %141 : (tensor<1xf64>) -> tensor<f64>
    %143 = stablehlo.slice %138 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %144 = stablehlo.reshape %143 : (tensor<1xf64>) -> tensor<f64>
    %145 = stablehlo.sine %142 : tensor<f64>
    %146 = stablehlo.negate %145 : tensor<f64>
    %147 = stablehlo.cosine %140 : tensor<f64>
    %148 = stablehlo.sine %140 : tensor<f64>
    %149 = stablehlo.cosine %142 : tensor<f64>
    %150 = stablehlo.multiply %148, %149 : tensor<f64>
    %151 = stablehlo.sine %140 : tensor<f64>
    %152 = stablehlo.negate %151 : tensor<f64>
    %153 = stablehlo.cosine %140 : tensor<f64>
    %154 = stablehlo.cosine %142 : tensor<f64>
    %155 = stablehlo.multiply %153, %154 : tensor<f64>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %156 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %157 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %158 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %159 = stablehlo.concatenate %156, %157, %158, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %160 = stablehlo.broadcast_in_dim %159, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %161 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %162 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %163 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %164 = stablehlo.concatenate %161, %162, %163, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %165 = stablehlo.broadcast_in_dim %164, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %166 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %167 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %168 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %169 = stablehlo.concatenate %166, %167, %168, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %170 = stablehlo.broadcast_in_dim %169, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %171 = stablehlo.concatenate %160, %165, %170, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<3x3xf64>
    %172 = stablehlo.dot_general %171, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %173 = call @nan_to_num(%172) : (tensor<3xf64>) -> tensor<3xf64>
    %174 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %175 = stablehlo.slice %174 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %176 = stablehlo.reshape %175 : (tensor<1xf64>) -> tensor<f64>
    %177 = stablehlo.negate %176 : tensor<f64>
    %178 = stablehlo.reshape %177 : (tensor<f64>) -> tensor<1xf64>
    %179 = stablehlo.slice %174 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %180 = stablehlo.reshape %179 : (tensor<1xf64>) -> tensor<f64>
    %181 = stablehlo.negate %180 : tensor<f64>
    %182 = stablehlo.reshape %181 : (tensor<f64>) -> tensor<1xf64>
    %183 = stablehlo.slice %174 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %184 = stablehlo.reshape %183 : (tensor<1xf64>) -> tensor<f64>
    %185 = stablehlo.negate %184 : tensor<f64>
    %186 = stablehlo.reshape %185 : (tensor<f64>) -> tensor<1xf64>
    %187 = stablehlo.slice %174 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %188 = stablehlo.reshape %187 : (tensor<1xf64>) -> tensor<f64>
    %189 = stablehlo.reshape %188 : (tensor<f64>) -> tensor<1xf64>
    %190 = stablehlo.concatenate %178, %182, %186, %189, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %191 = stablehlo.dot_general %174, %174, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %192 = stablehlo.broadcast_in_dim %191, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %193 = stablehlo.divide %190, %192 : tensor<4xf64>
    %194 = stablehlo.slice %193 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %195 = stablehlo.reshape %194 : (tensor<1xf64>) -> tensor<f64>
    %196 = stablehlo.slice %arg2 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %197 = stablehlo.reshape %196 : (tensor<1xf64>) -> tensor<f64>
    %198 = stablehlo.slice %42 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %199 = stablehlo.reshape %198 : (tensor<1xf64>) -> tensor<f64>
    %200 = stablehlo.multiply %197, %199 : tensor<f64>
    %201 = stablehlo.slice %arg2 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %202 = stablehlo.reshape %201 : (tensor<1xf64>) -> tensor<f64>
    %203 = stablehlo.slice %42 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %204 = stablehlo.reshape %203 : (tensor<1xf64>) -> tensor<f64>
    %205 = stablehlo.multiply %202, %204 : tensor<f64>
    %206 = stablehlo.add %200, %205 : tensor<f64>
    %207 = stablehlo.slice %arg2 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %208 = stablehlo.reshape %207 : (tensor<1xf64>) -> tensor<f64>
    %209 = stablehlo.slice %42 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %210 = stablehlo.reshape %209 : (tensor<1xf64>) -> tensor<f64>
    %211 = stablehlo.multiply %208, %210 : tensor<f64>
    %212 = stablehlo.add %206, %211 : tensor<f64>
    %213 = stablehlo.slice %arg2 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %214 = stablehlo.reshape %213 : (tensor<1xf64>) -> tensor<f64>
    %215 = stablehlo.slice %42 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %216 = stablehlo.reshape %215 : (tensor<1xf64>) -> tensor<f64>
    %217 = stablehlo.multiply %214, %216 : tensor<f64>
    %218 = stablehlo.subtract %212, %217 : tensor<f64>
    %219 = stablehlo.reshape %218 : (tensor<f64>) -> tensor<1xf64>
    %220 = stablehlo.multiply %197, %216 : tensor<f64>
    %221 = stablehlo.multiply %202, %210 : tensor<f64>
    %222 = stablehlo.subtract %220, %221 : tensor<f64>
    %223 = stablehlo.multiply %208, %204 : tensor<f64>
    %224 = stablehlo.add %222, %223 : tensor<f64>
    %225 = stablehlo.multiply %214, %199 : tensor<f64>
    %226 = stablehlo.add %224, %225 : tensor<f64>
    %227 = stablehlo.reshape %226 : (tensor<f64>) -> tensor<1xf64>
    %228 = stablehlo.multiply %197, %210 : tensor<f64>
    %229 = stablehlo.multiply %202, %216 : tensor<f64>
    %230 = stablehlo.add %228, %229 : tensor<f64>
    %231 = stablehlo.multiply %208, %199 : tensor<f64>
    %232 = stablehlo.subtract %230, %231 : tensor<f64>
    %233 = stablehlo.multiply %214, %204 : tensor<f64>
    %234 = stablehlo.add %232, %233 : tensor<f64>
    %235 = stablehlo.reshape %234 : (tensor<f64>) -> tensor<1xf64>
    %236 = stablehlo.multiply %197, %204 : tensor<f64>
    %237 = stablehlo.multiply %202, %199 : tensor<f64>
    %238 = stablehlo.subtract %236, %237 : tensor<f64>
    %239 = stablehlo.multiply %208, %216 : tensor<f64>
    %240 = stablehlo.subtract %238, %239 : tensor<f64>
    %241 = stablehlo.multiply %214, %210 : tensor<f64>
    %242 = stablehlo.subtract %240, %241 : tensor<f64>
    %243 = stablehlo.reshape %242 : (tensor<f64>) -> tensor<1xf64>
    %244 = stablehlo.concatenate %219, %227, %235, %243, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %245 = stablehlo.slice %244 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %246 = stablehlo.reshape %245 : (tensor<1xf64>) -> tensor<f64>
    %247 = stablehlo.multiply %195, %246 : tensor<f64>
    %248 = stablehlo.slice %193 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %249 = stablehlo.reshape %248 : (tensor<1xf64>) -> tensor<f64>
    %250 = stablehlo.slice %244 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %251 = stablehlo.reshape %250 : (tensor<1xf64>) -> tensor<f64>
    %252 = stablehlo.multiply %249, %251 : tensor<f64>
    %253 = stablehlo.add %247, %252 : tensor<f64>
    %254 = stablehlo.slice %193 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %255 = stablehlo.reshape %254 : (tensor<1xf64>) -> tensor<f64>
    %256 = stablehlo.slice %244 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %257 = stablehlo.reshape %256 : (tensor<1xf64>) -> tensor<f64>
    %258 = stablehlo.multiply %255, %257 : tensor<f64>
    %259 = stablehlo.add %253, %258 : tensor<f64>
    %260 = stablehlo.slice %193 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %261 = stablehlo.reshape %260 : (tensor<1xf64>) -> tensor<f64>
    %262 = stablehlo.slice %244 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %263 = stablehlo.reshape %262 : (tensor<1xf64>) -> tensor<f64>
    %264 = stablehlo.multiply %261, %263 : tensor<f64>
    %265 = stablehlo.subtract %259, %264 : tensor<f64>
    %266 = stablehlo.reshape %265 : (tensor<f64>) -> tensor<1xf64>
    %267 = stablehlo.multiply %195, %263 : tensor<f64>
    %268 = stablehlo.multiply %249, %257 : tensor<f64>
    %269 = stablehlo.subtract %267, %268 : tensor<f64>
    %270 = stablehlo.multiply %255, %251 : tensor<f64>
    %271 = stablehlo.add %269, %270 : tensor<f64>
    %272 = stablehlo.multiply %261, %246 : tensor<f64>
    %273 = stablehlo.add %271, %272 : tensor<f64>
    %274 = stablehlo.reshape %273 : (tensor<f64>) -> tensor<1xf64>
    %275 = stablehlo.multiply %195, %257 : tensor<f64>
    %276 = stablehlo.multiply %249, %263 : tensor<f64>
    %277 = stablehlo.add %275, %276 : tensor<f64>
    %278 = stablehlo.multiply %255, %246 : tensor<f64>
    %279 = stablehlo.subtract %277, %278 : tensor<f64>
    %280 = stablehlo.multiply %261, %251 : tensor<f64>
    %281 = stablehlo.add %279, %280 : tensor<f64>
    %282 = stablehlo.reshape %281 : (tensor<f64>) -> tensor<1xf64>
    %283 = stablehlo.multiply %195, %251 : tensor<f64>
    %284 = stablehlo.multiply %249, %246 : tensor<f64>
    %285 = stablehlo.subtract %283, %284 : tensor<f64>
    %286 = stablehlo.multiply %255, %263 : tensor<f64>
    %287 = stablehlo.subtract %285, %286 : tensor<f64>
    %288 = stablehlo.multiply %261, %257 : tensor<f64>
    %289 = stablehlo.subtract %287, %288 : tensor<f64>
    %290 = stablehlo.reshape %289 : (tensor<f64>) -> tensor<1xf64>
    %291 = stablehlo.concatenate %266, %274, %282, %290, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %292 = stablehlo.slice %291 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %293 = stablehlo.reshape %292 : (tensor<1xf64>) -> tensor<f64>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %294 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %295 = stablehlo.concatenate %173, %294, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %296 = stablehlo.slice %295 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %297 = stablehlo.reshape %296 : (tensor<1xf64>) -> tensor<f64>
    %298 = stablehlo.multiply %293, %297 : tensor<f64>
    %299 = stablehlo.slice %291 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %300 = stablehlo.reshape %299 : (tensor<1xf64>) -> tensor<f64>
    %301 = stablehlo.slice %295 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %302 = stablehlo.reshape %301 : (tensor<1xf64>) -> tensor<f64>
    %303 = stablehlo.multiply %300, %302 : tensor<f64>
    %304 = stablehlo.add %298, %303 : tensor<f64>
    %305 = stablehlo.slice %291 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %306 = stablehlo.reshape %305 : (tensor<1xf64>) -> tensor<f64>
    %307 = stablehlo.slice %295 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %308 = stablehlo.reshape %307 : (tensor<1xf64>) -> tensor<f64>
    %309 = stablehlo.multiply %306, %308 : tensor<f64>
    %310 = stablehlo.add %304, %309 : tensor<f64>
    %311 = stablehlo.slice %291 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %312 = stablehlo.reshape %311 : (tensor<1xf64>) -> tensor<f64>
    %313 = stablehlo.slice %295 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %314 = stablehlo.reshape %313 : (tensor<1xf64>) -> tensor<f64>
    %315 = stablehlo.multiply %312, %314 : tensor<f64>
    %316 = stablehlo.subtract %310, %315 : tensor<f64>
    %317 = stablehlo.reshape %316 : (tensor<f64>) -> tensor<1xf64>
    %318 = stablehlo.multiply %293, %314 : tensor<f64>
    %319 = stablehlo.multiply %300, %308 : tensor<f64>
    %320 = stablehlo.subtract %318, %319 : tensor<f64>
    %321 = stablehlo.multiply %306, %302 : tensor<f64>
    %322 = stablehlo.add %320, %321 : tensor<f64>
    %323 = stablehlo.multiply %312, %297 : tensor<f64>
    %324 = stablehlo.add %322, %323 : tensor<f64>
    %325 = stablehlo.reshape %324 : (tensor<f64>) -> tensor<1xf64>
    %326 = stablehlo.multiply %293, %308 : tensor<f64>
    %327 = stablehlo.multiply %300, %314 : tensor<f64>
    %328 = stablehlo.add %326, %327 : tensor<f64>
    %329 = stablehlo.multiply %306, %297 : tensor<f64>
    %330 = stablehlo.subtract %328, %329 : tensor<f64>
    %331 = stablehlo.multiply %312, %302 : tensor<f64>
    %332 = stablehlo.add %330, %331 : tensor<f64>
    %333 = stablehlo.reshape %332 : (tensor<f64>) -> tensor<1xf64>
    %334 = stablehlo.multiply %293, %302 : tensor<f64>
    %335 = stablehlo.multiply %300, %297 : tensor<f64>
    %336 = stablehlo.subtract %334, %335 : tensor<f64>
    %337 = stablehlo.multiply %306, %314 : tensor<f64>
    %338 = stablehlo.subtract %336, %337 : tensor<f64>
    %339 = stablehlo.multiply %312, %308 : tensor<f64>
    %340 = stablehlo.subtract %338, %339 : tensor<f64>
    %341 = stablehlo.reshape %340 : (tensor<f64>) -> tensor<1xf64>
    %342 = stablehlo.concatenate %317, %325, %333, %341, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %343 = stablehlo.slice %342 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %344 = stablehlo.reshape %343 : (tensor<1xf64>) -> tensor<f64>
    %345 = stablehlo.slice %291 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %346 = stablehlo.reshape %345 : (tensor<1xf64>) -> tensor<f64>
    %347 = stablehlo.negate %346 : tensor<f64>
    %348 = stablehlo.reshape %347 : (tensor<f64>) -> tensor<1xf64>
    %349 = stablehlo.slice %291 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %350 = stablehlo.reshape %349 : (tensor<1xf64>) -> tensor<f64>
    %351 = stablehlo.negate %350 : tensor<f64>
    %352 = stablehlo.reshape %351 : (tensor<f64>) -> tensor<1xf64>
    %353 = stablehlo.slice %291 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %354 = stablehlo.reshape %353 : (tensor<1xf64>) -> tensor<f64>
    %355 = stablehlo.negate %354 : tensor<f64>
    %356 = stablehlo.reshape %355 : (tensor<f64>) -> tensor<1xf64>
    %357 = stablehlo.slice %291 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %358 = stablehlo.reshape %357 : (tensor<1xf64>) -> tensor<f64>
    %359 = stablehlo.reshape %358 : (tensor<f64>) -> tensor<1xf64>
    %360 = stablehlo.concatenate %348, %352, %356, %359, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %361 = stablehlo.dot_general %291, %291, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %362 = stablehlo.broadcast_in_dim %361, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %363 = stablehlo.divide %360, %362 : tensor<4xf64>
    %364 = stablehlo.slice %363 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %365 = stablehlo.reshape %364 : (tensor<1xf64>) -> tensor<f64>
    %366 = stablehlo.multiply %344, %365 : tensor<f64>
    %367 = stablehlo.slice %342 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %368 = stablehlo.reshape %367 : (tensor<1xf64>) -> tensor<f64>
    %369 = stablehlo.slice %363 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %370 = stablehlo.reshape %369 : (tensor<1xf64>) -> tensor<f64>
    %371 = stablehlo.multiply %368, %370 : tensor<f64>
    %372 = stablehlo.add %366, %371 : tensor<f64>
    %373 = stablehlo.slice %342 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %374 = stablehlo.reshape %373 : (tensor<1xf64>) -> tensor<f64>
    %375 = stablehlo.slice %363 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %376 = stablehlo.reshape %375 : (tensor<1xf64>) -> tensor<f64>
    %377 = stablehlo.multiply %374, %376 : tensor<f64>
    %378 = stablehlo.add %372, %377 : tensor<f64>
    %379 = stablehlo.slice %342 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %380 = stablehlo.reshape %379 : (tensor<1xf64>) -> tensor<f64>
    %381 = stablehlo.slice %363 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %382 = stablehlo.reshape %381 : (tensor<1xf64>) -> tensor<f64>
    %383 = stablehlo.multiply %380, %382 : tensor<f64>
    %384 = stablehlo.subtract %378, %383 : tensor<f64>
    %385 = stablehlo.reshape %384 : (tensor<f64>) -> tensor<1xf64>
    %386 = stablehlo.multiply %344, %382 : tensor<f64>
    %387 = stablehlo.multiply %368, %376 : tensor<f64>
    %388 = stablehlo.subtract %386, %387 : tensor<f64>
    %389 = stablehlo.multiply %374, %370 : tensor<f64>
    %390 = stablehlo.add %388, %389 : tensor<f64>
    %391 = stablehlo.multiply %380, %365 : tensor<f64>
    %392 = stablehlo.add %390, %391 : tensor<f64>
    %393 = stablehlo.reshape %392 : (tensor<f64>) -> tensor<1xf64>
    %394 = stablehlo.multiply %344, %376 : tensor<f64>
    %395 = stablehlo.multiply %368, %382 : tensor<f64>
    %396 = stablehlo.add %394, %395 : tensor<f64>
    %397 = stablehlo.multiply %374, %365 : tensor<f64>
    %398 = stablehlo.subtract %396, %397 : tensor<f64>
    %399 = stablehlo.multiply %380, %370 : tensor<f64>
    %400 = stablehlo.add %398, %399 : tensor<f64>
    %401 = stablehlo.reshape %400 : (tensor<f64>) -> tensor<1xf64>
    %402 = stablehlo.multiply %344, %370 : tensor<f64>
    %403 = stablehlo.multiply %368, %365 : tensor<f64>
    %404 = stablehlo.subtract %402, %403 : tensor<f64>
    %405 = stablehlo.multiply %374, %382 : tensor<f64>
    %406 = stablehlo.subtract %404, %405 : tensor<f64>
    %407 = stablehlo.multiply %380, %376 : tensor<f64>
    %408 = stablehlo.subtract %406, %407 : tensor<f64>
    %409 = stablehlo.reshape %408 : (tensor<f64>) -> tensor<1xf64>
    %410 = stablehlo.concatenate %385, %393, %401, %409, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %411 = stablehlo.slice %410 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %412 = stablehlo.reshape %411 : (tensor<1xf64>) -> tensor<f64>
    %413 = stablehlo.reshape %412 : (tensor<f64>) -> tensor<1xf64>
    %414 = stablehlo.slice %410 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %415 = stablehlo.reshape %414 : (tensor<1xf64>) -> tensor<f64>
    %416 = stablehlo.reshape %415 : (tensor<f64>) -> tensor<1xf64>
    %417 = stablehlo.slice %410 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %418 = stablehlo.reshape %417 : (tensor<1xf64>) -> tensor<f64>
    %419 = stablehlo.reshape %418 : (tensor<f64>) -> tensor<1xf64>
    %420 = stablehlo.concatenate %413, %416, %419, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %421 = stablehlo.slice %arg2 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %422 = stablehlo.reshape %421 : (tensor<1xf64>) -> tensor<f64>
    %423 = stablehlo.slice %42 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %424 = stablehlo.reshape %423 : (tensor<1xf64>) -> tensor<f64>
    %425 = stablehlo.multiply %422, %424 : tensor<f64>
    %426 = stablehlo.slice %arg2 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %427 = stablehlo.reshape %426 : (tensor<1xf64>) -> tensor<f64>
    %428 = stablehlo.slice %42 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %429 = stablehlo.reshape %428 : (tensor<1xf64>) -> tensor<f64>
    %430 = stablehlo.multiply %427, %429 : tensor<f64>
    %431 = stablehlo.add %425, %430 : tensor<f64>
    %432 = stablehlo.slice %arg2 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %433 = stablehlo.reshape %432 : (tensor<1xf64>) -> tensor<f64>
    %434 = stablehlo.slice %42 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %435 = stablehlo.reshape %434 : (tensor<1xf64>) -> tensor<f64>
    %436 = stablehlo.multiply %433, %435 : tensor<f64>
    %437 = stablehlo.add %431, %436 : tensor<f64>
    %438 = stablehlo.slice %arg2 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %439 = stablehlo.reshape %438 : (tensor<1xf64>) -> tensor<f64>
    %440 = stablehlo.slice %42 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %441 = stablehlo.reshape %440 : (tensor<1xf64>) -> tensor<f64>
    %442 = stablehlo.multiply %439, %441 : tensor<f64>
    %443 = stablehlo.subtract %437, %442 : tensor<f64>
    %444 = stablehlo.reshape %443 : (tensor<f64>) -> tensor<1xf64>
    %445 = stablehlo.multiply %422, %441 : tensor<f64>
    %446 = stablehlo.multiply %427, %435 : tensor<f64>
    %447 = stablehlo.subtract %445, %446 : tensor<f64>
    %448 = stablehlo.multiply %433, %429 : tensor<f64>
    %449 = stablehlo.add %447, %448 : tensor<f64>
    %450 = stablehlo.multiply %439, %424 : tensor<f64>
    %451 = stablehlo.add %449, %450 : tensor<f64>
    %452 = stablehlo.reshape %451 : (tensor<f64>) -> tensor<1xf64>
    %453 = stablehlo.multiply %422, %435 : tensor<f64>
    %454 = stablehlo.multiply %427, %441 : tensor<f64>
    %455 = stablehlo.add %453, %454 : tensor<f64>
    %456 = stablehlo.multiply %433, %424 : tensor<f64>
    %457 = stablehlo.subtract %455, %456 : tensor<f64>
    %458 = stablehlo.multiply %439, %429 : tensor<f64>
    %459 = stablehlo.add %457, %458 : tensor<f64>
    %460 = stablehlo.reshape %459 : (tensor<f64>) -> tensor<1xf64>
    %461 = stablehlo.multiply %422, %429 : tensor<f64>
    %462 = stablehlo.multiply %427, %424 : tensor<f64>
    %463 = stablehlo.subtract %461, %462 : tensor<f64>
    %464 = stablehlo.multiply %433, %441 : tensor<f64>
    %465 = stablehlo.subtract %463, %464 : tensor<f64>
    %466 = stablehlo.multiply %439, %435 : tensor<f64>
    %467 = stablehlo.subtract %465, %466 : tensor<f64>
    %468 = stablehlo.reshape %467 : (tensor<f64>) -> tensor<1xf64>
    %469 = stablehlo.concatenate %444, %452, %460, %468, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %470 = stablehlo.slice %469 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %471 = stablehlo.reshape %470 : (tensor<1xf64>) -> tensor<f64>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %472 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %473 = stablehlo.concatenate %cst_0, %472, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %474 = stablehlo.slice %473 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %475 = stablehlo.reshape %474 : (tensor<1xf64>) -> tensor<f64>
    %476 = stablehlo.multiply %471, %475 : tensor<f64>
    %477 = stablehlo.slice %469 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %478 = stablehlo.reshape %477 : (tensor<1xf64>) -> tensor<f64>
    %479 = stablehlo.slice %473 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %480 = stablehlo.reshape %479 : (tensor<1xf64>) -> tensor<f64>
    %481 = stablehlo.multiply %478, %480 : tensor<f64>
    %482 = stablehlo.add %476, %481 : tensor<f64>
    %483 = stablehlo.slice %469 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %484 = stablehlo.reshape %483 : (tensor<1xf64>) -> tensor<f64>
    %485 = stablehlo.slice %473 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %486 = stablehlo.reshape %485 : (tensor<1xf64>) -> tensor<f64>
    %487 = stablehlo.multiply %484, %486 : tensor<f64>
    %488 = stablehlo.add %482, %487 : tensor<f64>
    %489 = stablehlo.slice %469 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %490 = stablehlo.reshape %489 : (tensor<1xf64>) -> tensor<f64>
    %491 = stablehlo.slice %473 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %492 = stablehlo.reshape %491 : (tensor<1xf64>) -> tensor<f64>
    %493 = stablehlo.multiply %490, %492 : tensor<f64>
    %494 = stablehlo.subtract %488, %493 : tensor<f64>
    %495 = stablehlo.reshape %494 : (tensor<f64>) -> tensor<1xf64>
    %496 = stablehlo.multiply %471, %492 : tensor<f64>
    %497 = stablehlo.multiply %478, %486 : tensor<f64>
    %498 = stablehlo.subtract %496, %497 : tensor<f64>
    %499 = stablehlo.multiply %484, %480 : tensor<f64>
    %500 = stablehlo.add %498, %499 : tensor<f64>
    %501 = stablehlo.multiply %490, %475 : tensor<f64>
    %502 = stablehlo.add %500, %501 : tensor<f64>
    %503 = stablehlo.reshape %502 : (tensor<f64>) -> tensor<1xf64>
    %504 = stablehlo.multiply %471, %486 : tensor<f64>
    %505 = stablehlo.multiply %478, %492 : tensor<f64>
    %506 = stablehlo.add %504, %505 : tensor<f64>
    %507 = stablehlo.multiply %484, %475 : tensor<f64>
    %508 = stablehlo.subtract %506, %507 : tensor<f64>
    %509 = stablehlo.multiply %490, %480 : tensor<f64>
    %510 = stablehlo.add %508, %509 : tensor<f64>
    %511 = stablehlo.reshape %510 : (tensor<f64>) -> tensor<1xf64>
    %512 = stablehlo.multiply %471, %480 : tensor<f64>
    %513 = stablehlo.multiply %478, %475 : tensor<f64>
    %514 = stablehlo.subtract %512, %513 : tensor<f64>
    %515 = stablehlo.multiply %484, %492 : tensor<f64>
    %516 = stablehlo.subtract %514, %515 : tensor<f64>
    %517 = stablehlo.multiply %490, %486 : tensor<f64>
    %518 = stablehlo.subtract %516, %517 : tensor<f64>
    %519 = stablehlo.reshape %518 : (tensor<f64>) -> tensor<1xf64>
    %520 = stablehlo.concatenate %495, %503, %511, %519, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %521 = stablehlo.slice %520 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %522 = stablehlo.reshape %521 : (tensor<1xf64>) -> tensor<f64>
    %523 = stablehlo.slice %469 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %524 = stablehlo.reshape %523 : (tensor<1xf64>) -> tensor<f64>
    %525 = stablehlo.negate %524 : tensor<f64>
    %526 = stablehlo.reshape %525 : (tensor<f64>) -> tensor<1xf64>
    %527 = stablehlo.slice %469 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %528 = stablehlo.reshape %527 : (tensor<1xf64>) -> tensor<f64>
    %529 = stablehlo.negate %528 : tensor<f64>
    %530 = stablehlo.reshape %529 : (tensor<f64>) -> tensor<1xf64>
    %531 = stablehlo.slice %469 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %532 = stablehlo.reshape %531 : (tensor<1xf64>) -> tensor<f64>
    %533 = stablehlo.negate %532 : tensor<f64>
    %534 = stablehlo.reshape %533 : (tensor<f64>) -> tensor<1xf64>
    %535 = stablehlo.slice %469 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %536 = stablehlo.reshape %535 : (tensor<1xf64>) -> tensor<f64>
    %537 = stablehlo.reshape %536 : (tensor<f64>) -> tensor<1xf64>
    %538 = stablehlo.concatenate %526, %530, %534, %537, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %539 = stablehlo.dot_general %469, %469, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %540 = stablehlo.broadcast_in_dim %539, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %541 = stablehlo.divide %538, %540 : tensor<4xf64>
    %542 = stablehlo.slice %541 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %543 = stablehlo.reshape %542 : (tensor<1xf64>) -> tensor<f64>
    %544 = stablehlo.multiply %522, %543 : tensor<f64>
    %545 = stablehlo.slice %520 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %546 = stablehlo.reshape %545 : (tensor<1xf64>) -> tensor<f64>
    %547 = stablehlo.slice %541 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %548 = stablehlo.reshape %547 : (tensor<1xf64>) -> tensor<f64>
    %549 = stablehlo.multiply %546, %548 : tensor<f64>
    %550 = stablehlo.add %544, %549 : tensor<f64>
    %551 = stablehlo.slice %520 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %552 = stablehlo.reshape %551 : (tensor<1xf64>) -> tensor<f64>
    %553 = stablehlo.slice %541 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %554 = stablehlo.reshape %553 : (tensor<1xf64>) -> tensor<f64>
    %555 = stablehlo.multiply %552, %554 : tensor<f64>
    %556 = stablehlo.add %550, %555 : tensor<f64>
    %557 = stablehlo.slice %520 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %558 = stablehlo.reshape %557 : (tensor<1xf64>) -> tensor<f64>
    %559 = stablehlo.slice %541 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %560 = stablehlo.reshape %559 : (tensor<1xf64>) -> tensor<f64>
    %561 = stablehlo.multiply %558, %560 : tensor<f64>
    %562 = stablehlo.subtract %556, %561 : tensor<f64>
    %563 = stablehlo.reshape %562 : (tensor<f64>) -> tensor<1xf64>
    %564 = stablehlo.multiply %522, %560 : tensor<f64>
    %565 = stablehlo.multiply %546, %554 : tensor<f64>
    %566 = stablehlo.subtract %564, %565 : tensor<f64>
    %567 = stablehlo.multiply %552, %548 : tensor<f64>
    %568 = stablehlo.add %566, %567 : tensor<f64>
    %569 = stablehlo.multiply %558, %543 : tensor<f64>
    %570 = stablehlo.add %568, %569 : tensor<f64>
    %571 = stablehlo.reshape %570 : (tensor<f64>) -> tensor<1xf64>
    %572 = stablehlo.multiply %522, %554 : tensor<f64>
    %573 = stablehlo.multiply %546, %560 : tensor<f64>
    %574 = stablehlo.add %572, %573 : tensor<f64>
    %575 = stablehlo.multiply %552, %543 : tensor<f64>
    %576 = stablehlo.subtract %574, %575 : tensor<f64>
    %577 = stablehlo.multiply %558, %548 : tensor<f64>
    %578 = stablehlo.add %576, %577 : tensor<f64>
    %579 = stablehlo.reshape %578 : (tensor<f64>) -> tensor<1xf64>
    %580 = stablehlo.multiply %522, %548 : tensor<f64>
    %581 = stablehlo.multiply %546, %543 : tensor<f64>
    %582 = stablehlo.subtract %580, %581 : tensor<f64>
    %583 = stablehlo.multiply %552, %560 : tensor<f64>
    %584 = stablehlo.subtract %582, %583 : tensor<f64>
    %585 = stablehlo.multiply %558, %554 : tensor<f64>
    %586 = stablehlo.subtract %584, %585 : tensor<f64>
    %587 = stablehlo.reshape %586 : (tensor<f64>) -> tensor<1xf64>
    %588 = stablehlo.concatenate %563, %571, %579, %587, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %589 = stablehlo.slice %588 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %590 = stablehlo.reshape %589 : (tensor<1xf64>) -> tensor<f64>
    %591 = stablehlo.reshape %590 : (tensor<f64>) -> tensor<1xf64>
    %592 = stablehlo.slice %588 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %593 = stablehlo.reshape %592 : (tensor<1xf64>) -> tensor<f64>
    %594 = stablehlo.reshape %593 : (tensor<f64>) -> tensor<1xf64>
    %595 = stablehlo.slice %588 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %596 = stablehlo.reshape %595 : (tensor<1xf64>) -> tensor<f64>
    %597 = stablehlo.reshape %596 : (tensor<f64>) -> tensor<1xf64>
    %598 = stablehlo.concatenate %591, %594, %597, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %599 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %600 = stablehlo.slice %599 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %601 = stablehlo.reshape %600 : (tensor<1xf64>) -> tensor<f64>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %602 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %603 = stablehlo.concatenate %cst_0, %602, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %604 = stablehlo.slice %603 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %605 = stablehlo.reshape %604 : (tensor<1xf64>) -> tensor<f64>
    %606 = stablehlo.multiply %601, %605 : tensor<f64>
    %607 = stablehlo.slice %599 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %608 = stablehlo.reshape %607 : (tensor<1xf64>) -> tensor<f64>
    %609 = stablehlo.slice %603 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %610 = stablehlo.reshape %609 : (tensor<1xf64>) -> tensor<f64>
    %611 = stablehlo.multiply %608, %610 : tensor<f64>
    %612 = stablehlo.add %606, %611 : tensor<f64>
    %613 = stablehlo.slice %599 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %614 = stablehlo.reshape %613 : (tensor<1xf64>) -> tensor<f64>
    %615 = stablehlo.slice %603 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %616 = stablehlo.reshape %615 : (tensor<1xf64>) -> tensor<f64>
    %617 = stablehlo.multiply %614, %616 : tensor<f64>
    %618 = stablehlo.add %612, %617 : tensor<f64>
    %619 = stablehlo.slice %599 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %620 = stablehlo.reshape %619 : (tensor<1xf64>) -> tensor<f64>
    %621 = stablehlo.slice %603 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %622 = stablehlo.reshape %621 : (tensor<1xf64>) -> tensor<f64>
    %623 = stablehlo.multiply %620, %622 : tensor<f64>
    %624 = stablehlo.subtract %618, %623 : tensor<f64>
    %625 = stablehlo.reshape %624 : (tensor<f64>) -> tensor<1xf64>
    %626 = stablehlo.multiply %601, %622 : tensor<f64>
    %627 = stablehlo.multiply %608, %616 : tensor<f64>
    %628 = stablehlo.subtract %626, %627 : tensor<f64>
    %629 = stablehlo.multiply %614, %610 : tensor<f64>
    %630 = stablehlo.add %628, %629 : tensor<f64>
    %631 = stablehlo.multiply %620, %605 : tensor<f64>
    %632 = stablehlo.add %630, %631 : tensor<f64>
    %633 = stablehlo.reshape %632 : (tensor<f64>) -> tensor<1xf64>
    %634 = stablehlo.multiply %601, %616 : tensor<f64>
    %635 = stablehlo.multiply %608, %622 : tensor<f64>
    %636 = stablehlo.add %634, %635 : tensor<f64>
    %637 = stablehlo.multiply %614, %605 : tensor<f64>
    %638 = stablehlo.subtract %636, %637 : tensor<f64>
    %639 = stablehlo.multiply %620, %610 : tensor<f64>
    %640 = stablehlo.add %638, %639 : tensor<f64>
    %641 = stablehlo.reshape %640 : (tensor<f64>) -> tensor<1xf64>
    %642 = stablehlo.multiply %601, %610 : tensor<f64>
    %643 = stablehlo.multiply %608, %605 : tensor<f64>
    %644 = stablehlo.subtract %642, %643 : tensor<f64>
    %645 = stablehlo.multiply %614, %622 : tensor<f64>
    %646 = stablehlo.subtract %644, %645 : tensor<f64>
    %647 = stablehlo.multiply %620, %616 : tensor<f64>
    %648 = stablehlo.subtract %646, %647 : tensor<f64>
    %649 = stablehlo.reshape %648 : (tensor<f64>) -> tensor<1xf64>
    %650 = stablehlo.concatenate %625, %633, %641, %649, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %651 = stablehlo.slice %650 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %652 = stablehlo.reshape %651 : (tensor<1xf64>) -> tensor<f64>
    %653 = stablehlo.slice %599 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %654 = stablehlo.reshape %653 : (tensor<1xf64>) -> tensor<f64>
    %655 = stablehlo.negate %654 : tensor<f64>
    %656 = stablehlo.reshape %655 : (tensor<f64>) -> tensor<1xf64>
    %657 = stablehlo.slice %599 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %658 = stablehlo.reshape %657 : (tensor<1xf64>) -> tensor<f64>
    %659 = stablehlo.negate %658 : tensor<f64>
    %660 = stablehlo.reshape %659 : (tensor<f64>) -> tensor<1xf64>
    %661 = stablehlo.slice %599 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %662 = stablehlo.reshape %661 : (tensor<1xf64>) -> tensor<f64>
    %663 = stablehlo.negate %662 : tensor<f64>
    %664 = stablehlo.reshape %663 : (tensor<f64>) -> tensor<1xf64>
    %665 = stablehlo.slice %599 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %666 = stablehlo.reshape %665 : (tensor<1xf64>) -> tensor<f64>
    %667 = stablehlo.reshape %666 : (tensor<f64>) -> tensor<1xf64>
    %668 = stablehlo.concatenate %656, %660, %664, %667, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %669 = stablehlo.dot_general %599, %599, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %670 = stablehlo.broadcast_in_dim %669, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %671 = stablehlo.divide %668, %670 : tensor<4xf64>
    %672 = stablehlo.slice %671 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %673 = stablehlo.reshape %672 : (tensor<1xf64>) -> tensor<f64>
    %674 = stablehlo.multiply %652, %673 : tensor<f64>
    %675 = stablehlo.slice %650 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %676 = stablehlo.reshape %675 : (tensor<1xf64>) -> tensor<f64>
    %677 = stablehlo.slice %671 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %678 = stablehlo.reshape %677 : (tensor<1xf64>) -> tensor<f64>
    %679 = stablehlo.multiply %676, %678 : tensor<f64>
    %680 = stablehlo.add %674, %679 : tensor<f64>
    %681 = stablehlo.slice %650 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %682 = stablehlo.reshape %681 : (tensor<1xf64>) -> tensor<f64>
    %683 = stablehlo.slice %671 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %684 = stablehlo.reshape %683 : (tensor<1xf64>) -> tensor<f64>
    %685 = stablehlo.multiply %682, %684 : tensor<f64>
    %686 = stablehlo.add %680, %685 : tensor<f64>
    %687 = stablehlo.slice %650 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %688 = stablehlo.reshape %687 : (tensor<1xf64>) -> tensor<f64>
    %689 = stablehlo.slice %671 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %690 = stablehlo.reshape %689 : (tensor<1xf64>) -> tensor<f64>
    %691 = stablehlo.multiply %688, %690 : tensor<f64>
    %692 = stablehlo.subtract %686, %691 : tensor<f64>
    %693 = stablehlo.reshape %692 : (tensor<f64>) -> tensor<1xf64>
    %694 = stablehlo.multiply %652, %690 : tensor<f64>
    %695 = stablehlo.multiply %676, %684 : tensor<f64>
    %696 = stablehlo.subtract %694, %695 : tensor<f64>
    %697 = stablehlo.multiply %682, %678 : tensor<f64>
    %698 = stablehlo.add %696, %697 : tensor<f64>
    %699 = stablehlo.multiply %688, %673 : tensor<f64>
    %700 = stablehlo.add %698, %699 : tensor<f64>
    %701 = stablehlo.reshape %700 : (tensor<f64>) -> tensor<1xf64>
    %702 = stablehlo.multiply %652, %684 : tensor<f64>
    %703 = stablehlo.multiply %676, %690 : tensor<f64>
    %704 = stablehlo.add %702, %703 : tensor<f64>
    %705 = stablehlo.multiply %682, %673 : tensor<f64>
    %706 = stablehlo.subtract %704, %705 : tensor<f64>
    %707 = stablehlo.multiply %688, %678 : tensor<f64>
    %708 = stablehlo.add %706, %707 : tensor<f64>
    %709 = stablehlo.reshape %708 : (tensor<f64>) -> tensor<1xf64>
    %710 = stablehlo.multiply %652, %678 : tensor<f64>
    %711 = stablehlo.multiply %676, %673 : tensor<f64>
    %712 = stablehlo.subtract %710, %711 : tensor<f64>
    %713 = stablehlo.multiply %682, %690 : tensor<f64>
    %714 = stablehlo.subtract %712, %713 : tensor<f64>
    %715 = stablehlo.multiply %688, %684 : tensor<f64>
    %716 = stablehlo.subtract %714, %715 : tensor<f64>
    %717 = stablehlo.reshape %716 : (tensor<f64>) -> tensor<1xf64>
    %718 = stablehlo.concatenate %693, %701, %709, %717, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %719 = stablehlo.slice %718 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %720 = stablehlo.reshape %719 : (tensor<1xf64>) -> tensor<f64>
    %721 = stablehlo.reshape %720 : (tensor<f64>) -> tensor<1xf64>
    %722 = stablehlo.slice %718 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %723 = stablehlo.reshape %722 : (tensor<1xf64>) -> tensor<f64>
    %724 = stablehlo.reshape %723 : (tensor<f64>) -> tensor<1xf64>
    %725 = stablehlo.slice %718 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %726 = stablehlo.reshape %725 : (tensor<1xf64>) -> tensor<f64>
    %727 = stablehlo.reshape %726 : (tensor<f64>) -> tensor<1xf64>
    %728 = stablehlo.concatenate %721, %724, %727, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %729 = stablehlo.dot_general %728, %598, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    %cst_28 = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %730 = call @clip(%729, %cst_28, %cst_29) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %731 = chlo.acos %730 : tensor<f64> -> tensor<f64>
    %732 = call @cross(%728, %598) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %733 = call @norm(%732) : (tensor<3xf64>) -> tensor<f64>
    %734 = stablehlo.broadcast_in_dim %733, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %735 = stablehlo.broadcast_in_dim %731, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %736 = stablehlo.concatenate %734, %735, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    %cst_30 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %737 = stablehlo.reduce(%736 init: %cst_30) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %cst_31 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %738 = stablehlo.compare  GT, %737, %cst_31,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %739 = stablehlo.convert %738 : (tensor<i1>) -> tensor<i32>
    %740 = "stablehlo.case"(%739) ({
      stablehlo.return %cst_0 : tensor<3xf64>
    }, {
      %1153 = stablehlo.broadcast_in_dim %733, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1154 = stablehlo.divide %732, %1153 : tensor<3xf64>
      stablehlo.return %1154 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    %741 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %742 = stablehlo.slice %741 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %743 = stablehlo.reshape %742 : (tensor<1xf64>) -> tensor<f64>
    %744 = stablehlo.negate %743 : tensor<f64>
    %745 = stablehlo.reshape %744 : (tensor<f64>) -> tensor<1xf64>
    %746 = stablehlo.slice %741 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %747 = stablehlo.reshape %746 : (tensor<1xf64>) -> tensor<f64>
    %748 = stablehlo.negate %747 : tensor<f64>
    %749 = stablehlo.reshape %748 : (tensor<f64>) -> tensor<1xf64>
    %750 = stablehlo.slice %741 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %751 = stablehlo.reshape %750 : (tensor<1xf64>) -> tensor<f64>
    %752 = stablehlo.negate %751 : tensor<f64>
    %753 = stablehlo.reshape %752 : (tensor<f64>) -> tensor<1xf64>
    %754 = stablehlo.slice %741 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %755 = stablehlo.reshape %754 : (tensor<1xf64>) -> tensor<f64>
    %756 = stablehlo.reshape %755 : (tensor<f64>) -> tensor<1xf64>
    %757 = stablehlo.concatenate %745, %749, %753, %756, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %758 = stablehlo.dot_general %741, %741, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %759 = stablehlo.broadcast_in_dim %758, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %760 = stablehlo.divide %757, %759 : tensor<4xf64>
    %761 = stablehlo.slice %760 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %762 = stablehlo.reshape %761 : (tensor<1xf64>) -> tensor<f64>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %763 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %764 = stablehlo.concatenate %740, %763, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %765 = stablehlo.slice %764 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %766 = stablehlo.reshape %765 : (tensor<1xf64>) -> tensor<f64>
    %767 = stablehlo.multiply %762, %766 : tensor<f64>
    %768 = stablehlo.slice %760 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %769 = stablehlo.reshape %768 : (tensor<1xf64>) -> tensor<f64>
    %770 = stablehlo.slice %764 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %771 = stablehlo.reshape %770 : (tensor<1xf64>) -> tensor<f64>
    %772 = stablehlo.multiply %769, %771 : tensor<f64>
    %773 = stablehlo.add %767, %772 : tensor<f64>
    %774 = stablehlo.slice %760 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %775 = stablehlo.reshape %774 : (tensor<1xf64>) -> tensor<f64>
    %776 = stablehlo.slice %764 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %777 = stablehlo.reshape %776 : (tensor<1xf64>) -> tensor<f64>
    %778 = stablehlo.multiply %775, %777 : tensor<f64>
    %779 = stablehlo.add %773, %778 : tensor<f64>
    %780 = stablehlo.slice %760 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %781 = stablehlo.reshape %780 : (tensor<1xf64>) -> tensor<f64>
    %782 = stablehlo.slice %764 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %783 = stablehlo.reshape %782 : (tensor<1xf64>) -> tensor<f64>
    %784 = stablehlo.multiply %781, %783 : tensor<f64>
    %785 = stablehlo.subtract %779, %784 : tensor<f64>
    %786 = stablehlo.reshape %785 : (tensor<f64>) -> tensor<1xf64>
    %787 = stablehlo.multiply %762, %783 : tensor<f64>
    %788 = stablehlo.multiply %769, %777 : tensor<f64>
    %789 = stablehlo.subtract %787, %788 : tensor<f64>
    %790 = stablehlo.multiply %775, %771 : tensor<f64>
    %791 = stablehlo.add %789, %790 : tensor<f64>
    %792 = stablehlo.multiply %781, %766 : tensor<f64>
    %793 = stablehlo.add %791, %792 : tensor<f64>
    %794 = stablehlo.reshape %793 : (tensor<f64>) -> tensor<1xf64>
    %795 = stablehlo.multiply %762, %777 : tensor<f64>
    %796 = stablehlo.multiply %769, %783 : tensor<f64>
    %797 = stablehlo.add %795, %796 : tensor<f64>
    %798 = stablehlo.multiply %775, %766 : tensor<f64>
    %799 = stablehlo.subtract %797, %798 : tensor<f64>
    %800 = stablehlo.multiply %781, %771 : tensor<f64>
    %801 = stablehlo.add %799, %800 : tensor<f64>
    %802 = stablehlo.reshape %801 : (tensor<f64>) -> tensor<1xf64>
    %803 = stablehlo.multiply %762, %771 : tensor<f64>
    %804 = stablehlo.multiply %769, %766 : tensor<f64>
    %805 = stablehlo.subtract %803, %804 : tensor<f64>
    %806 = stablehlo.multiply %775, %783 : tensor<f64>
    %807 = stablehlo.subtract %805, %806 : tensor<f64>
    %808 = stablehlo.multiply %781, %777 : tensor<f64>
    %809 = stablehlo.subtract %807, %808 : tensor<f64>
    %810 = stablehlo.reshape %809 : (tensor<f64>) -> tensor<1xf64>
    %811 = stablehlo.concatenate %786, %794, %802, %810, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %812 = stablehlo.slice %811 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %813 = stablehlo.reshape %812 : (tensor<1xf64>) -> tensor<f64>
    %814 = stablehlo.slice %760 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %815 = stablehlo.reshape %814 : (tensor<1xf64>) -> tensor<f64>
    %816 = stablehlo.negate %815 : tensor<f64>
    %817 = stablehlo.reshape %816 : (tensor<f64>) -> tensor<1xf64>
    %818 = stablehlo.slice %760 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %819 = stablehlo.reshape %818 : (tensor<1xf64>) -> tensor<f64>
    %820 = stablehlo.negate %819 : tensor<f64>
    %821 = stablehlo.reshape %820 : (tensor<f64>) -> tensor<1xf64>
    %822 = stablehlo.slice %760 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %823 = stablehlo.reshape %822 : (tensor<1xf64>) -> tensor<f64>
    %824 = stablehlo.negate %823 : tensor<f64>
    %825 = stablehlo.reshape %824 : (tensor<f64>) -> tensor<1xf64>
    %826 = stablehlo.slice %760 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %827 = stablehlo.reshape %826 : (tensor<1xf64>) -> tensor<f64>
    %828 = stablehlo.reshape %827 : (tensor<f64>) -> tensor<1xf64>
    %829 = stablehlo.concatenate %817, %821, %825, %828, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %830 = stablehlo.dot_general %760, %760, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %831 = stablehlo.broadcast_in_dim %830, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %832 = stablehlo.divide %829, %831 : tensor<4xf64>
    %833 = stablehlo.slice %832 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %834 = stablehlo.reshape %833 : (tensor<1xf64>) -> tensor<f64>
    %835 = stablehlo.multiply %813, %834 : tensor<f64>
    %836 = stablehlo.slice %811 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %837 = stablehlo.reshape %836 : (tensor<1xf64>) -> tensor<f64>
    %838 = stablehlo.slice %832 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %839 = stablehlo.reshape %838 : (tensor<1xf64>) -> tensor<f64>
    %840 = stablehlo.multiply %837, %839 : tensor<f64>
    %841 = stablehlo.add %835, %840 : tensor<f64>
    %842 = stablehlo.slice %811 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %843 = stablehlo.reshape %842 : (tensor<1xf64>) -> tensor<f64>
    %844 = stablehlo.slice %832 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %845 = stablehlo.reshape %844 : (tensor<1xf64>) -> tensor<f64>
    %846 = stablehlo.multiply %843, %845 : tensor<f64>
    %847 = stablehlo.add %841, %846 : tensor<f64>
    %848 = stablehlo.slice %811 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %849 = stablehlo.reshape %848 : (tensor<1xf64>) -> tensor<f64>
    %850 = stablehlo.slice %832 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %851 = stablehlo.reshape %850 : (tensor<1xf64>) -> tensor<f64>
    %852 = stablehlo.multiply %849, %851 : tensor<f64>
    %853 = stablehlo.subtract %847, %852 : tensor<f64>
    %854 = stablehlo.reshape %853 : (tensor<f64>) -> tensor<1xf64>
    %855 = stablehlo.multiply %813, %851 : tensor<f64>
    %856 = stablehlo.multiply %837, %845 : tensor<f64>
    %857 = stablehlo.subtract %855, %856 : tensor<f64>
    %858 = stablehlo.multiply %843, %839 : tensor<f64>
    %859 = stablehlo.add %857, %858 : tensor<f64>
    %860 = stablehlo.multiply %849, %834 : tensor<f64>
    %861 = stablehlo.add %859, %860 : tensor<f64>
    %862 = stablehlo.reshape %861 : (tensor<f64>) -> tensor<1xf64>
    %863 = stablehlo.multiply %813, %845 : tensor<f64>
    %864 = stablehlo.multiply %837, %851 : tensor<f64>
    %865 = stablehlo.add %863, %864 : tensor<f64>
    %866 = stablehlo.multiply %843, %834 : tensor<f64>
    %867 = stablehlo.subtract %865, %866 : tensor<f64>
    %868 = stablehlo.multiply %849, %839 : tensor<f64>
    %869 = stablehlo.add %867, %868 : tensor<f64>
    %870 = stablehlo.reshape %869 : (tensor<f64>) -> tensor<1xf64>
    %871 = stablehlo.multiply %813, %839 : tensor<f64>
    %872 = stablehlo.multiply %837, %834 : tensor<f64>
    %873 = stablehlo.subtract %871, %872 : tensor<f64>
    %874 = stablehlo.multiply %843, %851 : tensor<f64>
    %875 = stablehlo.subtract %873, %874 : tensor<f64>
    %876 = stablehlo.multiply %849, %845 : tensor<f64>
    %877 = stablehlo.subtract %875, %876 : tensor<f64>
    %878 = stablehlo.reshape %877 : (tensor<f64>) -> tensor<1xf64>
    %879 = stablehlo.concatenate %854, %862, %870, %878, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %880 = stablehlo.slice %879 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %881 = stablehlo.reshape %880 : (tensor<1xf64>) -> tensor<f64>
    %882 = stablehlo.reshape %881 : (tensor<f64>) -> tensor<1xf64>
    %883 = stablehlo.slice %879 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %884 = stablehlo.reshape %883 : (tensor<1xf64>) -> tensor<f64>
    %885 = stablehlo.reshape %884 : (tensor<f64>) -> tensor<1xf64>
    %886 = stablehlo.slice %879 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %887 = stablehlo.reshape %886 : (tensor<1xf64>) -> tensor<f64>
    %888 = stablehlo.reshape %887 : (tensor<f64>) -> tensor<1xf64>
    %889 = stablehlo.concatenate %882, %885, %888, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %cst_33 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %890 = stablehlo.compare  GT, %731, %cst_33,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %891 = stablehlo.convert %890 : (tensor<i1>) -> tensor<i32>
    %892 = "stablehlo.case"(%891) ({
      %cst_41 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1153 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst_42 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1154 = stablehlo.broadcast_in_dim %cst_42, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %1155 = stablehlo.concatenate %1153, %1154, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1156 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst_44 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1157 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %1158 = stablehlo.concatenate %1156, %1157, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      stablehlo.return %1158 : tensor<4xf64>
    }, {
      %1153 = stablehlo.dot_general %889, %889, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
      %1154 = stablehlo.sqrt %1153 : tensor<f64>
      %1155 = stablehlo.broadcast_in_dim %1154, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1156 = stablehlo.divide %889, %1155 : tensor<3xf64>
      %cst_41 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %1157 = stablehlo.divide %731, %cst_41 : tensor<f64>
      %1158 = stablehlo.sine %1157 : tensor<f64>
      %1159 = stablehlo.broadcast_in_dim %1158, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1160 = stablehlo.multiply %1156, %1159 : tensor<3xf64>
      %1161 = stablehlo.cosine %1157 : tensor<f64>
      %1162 = stablehlo.broadcast_in_dim %1161, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %1163 = stablehlo.concatenate %1160, %1162, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      %1164 = stablehlo.dot_general %889, %889, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
      %1165 = stablehlo.sqrt %1164 : tensor<f64>
      %1166 = stablehlo.broadcast_in_dim %1165, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1167 = stablehlo.divide %889, %1166 : tensor<3xf64>
      %cst_42 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %1168 = stablehlo.divide %731, %cst_42 : tensor<f64>
      %1169 = stablehlo.sine %1168 : tensor<f64>
      %1170 = stablehlo.broadcast_in_dim %1169, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1171 = stablehlo.multiply %1167, %1170 : tensor<3xf64>
      %1172 = stablehlo.cosine %1168 : tensor<f64>
      %1173 = stablehlo.broadcast_in_dim %1172, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %1174 = stablehlo.concatenate %1171, %1173, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
      stablehlo.return %1174 : tensor<4xf64>
    }) : (tensor<i32>) -> tensor<4xf64>
    %893 = stablehlo.slice %892 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %894 = stablehlo.reshape %893 : (tensor<1xf64>) -> tensor<f64>
    %895 = stablehlo.slice %892 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %896 = stablehlo.reshape %895 : (tensor<1xf64>) -> tensor<f64>
    %897 = stablehlo.slice %892 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %898 = stablehlo.reshape %897 : (tensor<1xf64>) -> tensor<f64>
    %899 = stablehlo.slice %892 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %900 = stablehlo.reshape %899 : (tensor<1xf64>) -> tensor<f64>
    %901 = stablehlo.multiply %894, %894 : tensor<f64>
    %902 = stablehlo.multiply %896, %896 : tensor<f64>
    %903 = stablehlo.add %901, %902 : tensor<f64>
    %904 = stablehlo.multiply %898, %898 : tensor<f64>
    %905 = stablehlo.add %903, %904 : tensor<f64>
    %906 = stablehlo.sqrt %905 : tensor<f64>
    %907 = stablehlo.broadcast_in_dim %894, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %908 = stablehlo.broadcast_in_dim %896, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %909 = stablehlo.broadcast_in_dim %898, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %910 = stablehlo.concatenate %907, %908, %909, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %cst_34 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %911 = stablehlo.compare  LT, %906, %cst_34,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %912 = stablehlo.convert %911 : (tensor<i1>) -> tensor<i32>
    %913 = "stablehlo.case"(%912) ({
      %1153 = stablehlo.broadcast_in_dim %906, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1154 = stablehlo.divide %910, %1153 : tensor<3xf64>
      %1155 = stablehlo.atan2 %906, %900 : tensor<f64>
      %cst_41 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %1156 = stablehlo.multiply %cst_41, %1155 : tensor<f64>
      %cst_42 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
      %1157 = func.call @remainder(%1156, %cst_42) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1158 = stablehlo.compare  LT, %1157, %cst_43,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_44 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
      %1159 = stablehlo.add %1157, %cst_44 : tensor<f64>
      %1160 = func.call @_where(%1158, %1159, %1157) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %cst_45 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
      %1161 = stablehlo.compare  GT, %1160, %cst_45,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_46 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
      %1162 = stablehlo.subtract %1160, %cst_46 : tensor<f64>
      %1163 = func.call @_where(%1161, %1162, %1160) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %1164 = stablehlo.broadcast_in_dim %1163, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1165 = stablehlo.multiply %1154, %1164 : tensor<3xf64>
      stablehlo.return %1165 : tensor<3xf64>
    }, {
      stablehlo.return %910 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    %914 = stablehlo.slice %913 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %915 = stablehlo.reshape %914 : (tensor<1xf64>) -> tensor<f64>
    %916 = stablehlo.slice %913 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %917 = stablehlo.reshape %916 : (tensor<1xf64>) -> tensor<f64>
    %918 = stablehlo.slice %913 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %919 = stablehlo.reshape %918 : (tensor<1xf64>) -> tensor<f64>
    %920 = stablehlo.slice %892 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %921 = stablehlo.reshape %920 : (tensor<1xf64>) -> tensor<f64>
    %922 = stablehlo.negate %921 : tensor<f64>
    %923 = stablehlo.reshape %922 : (tensor<f64>) -> tensor<1xf64>
    %924 = stablehlo.slice %892 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %925 = stablehlo.reshape %924 : (tensor<1xf64>) -> tensor<f64>
    %926 = stablehlo.negate %925 : tensor<f64>
    %927 = stablehlo.reshape %926 : (tensor<f64>) -> tensor<1xf64>
    %928 = stablehlo.slice %892 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %929 = stablehlo.reshape %928 : (tensor<1xf64>) -> tensor<f64>
    %930 = stablehlo.negate %929 : tensor<f64>
    %931 = stablehlo.reshape %930 : (tensor<f64>) -> tensor<1xf64>
    %932 = stablehlo.slice %892 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %933 = stablehlo.reshape %932 : (tensor<1xf64>) -> tensor<f64>
    %934 = stablehlo.reshape %933 : (tensor<f64>) -> tensor<1xf64>
    %935 = stablehlo.concatenate %923, %927, %931, %934, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %936 = stablehlo.dot_general %892, %892, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %937 = stablehlo.broadcast_in_dim %936, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %938 = stablehlo.divide %935, %937 : tensor<4xf64>
    %939 = stablehlo.slice %938 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %940 = stablehlo.reshape %939 : (tensor<1xf64>) -> tensor<f64>
    %941 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %942 = stablehlo.slice %941 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %943 = stablehlo.reshape %942 : (tensor<1xf64>) -> tensor<f64>
    %944 = stablehlo.negate %943 : tensor<f64>
    %945 = stablehlo.reshape %944 : (tensor<f64>) -> tensor<1xf64>
    %946 = stablehlo.slice %941 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %947 = stablehlo.reshape %946 : (tensor<1xf64>) -> tensor<f64>
    %948 = stablehlo.negate %947 : tensor<f64>
    %949 = stablehlo.reshape %948 : (tensor<f64>) -> tensor<1xf64>
    %950 = stablehlo.slice %941 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %951 = stablehlo.reshape %950 : (tensor<1xf64>) -> tensor<f64>
    %952 = stablehlo.negate %951 : tensor<f64>
    %953 = stablehlo.reshape %952 : (tensor<f64>) -> tensor<1xf64>
    %954 = stablehlo.slice %941 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %955 = stablehlo.reshape %954 : (tensor<1xf64>) -> tensor<f64>
    %956 = stablehlo.reshape %955 : (tensor<f64>) -> tensor<1xf64>
    %957 = stablehlo.concatenate %945, %949, %953, %956, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %958 = stablehlo.dot_general %941, %941, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %959 = stablehlo.broadcast_in_dim %958, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %960 = stablehlo.divide %957, %959 : tensor<4xf64>
    %961 = stablehlo.slice %960 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %962 = stablehlo.reshape %961 : (tensor<1xf64>) -> tensor<f64>
    %963 = stablehlo.multiply %940, %962 : tensor<f64>
    %964 = stablehlo.slice %938 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %965 = stablehlo.reshape %964 : (tensor<1xf64>) -> tensor<f64>
    %966 = stablehlo.slice %960 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %967 = stablehlo.reshape %966 : (tensor<1xf64>) -> tensor<f64>
    %968 = stablehlo.multiply %965, %967 : tensor<f64>
    %969 = stablehlo.add %963, %968 : tensor<f64>
    %970 = stablehlo.slice %938 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %971 = stablehlo.reshape %970 : (tensor<1xf64>) -> tensor<f64>
    %972 = stablehlo.slice %960 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %973 = stablehlo.reshape %972 : (tensor<1xf64>) -> tensor<f64>
    %974 = stablehlo.multiply %971, %973 : tensor<f64>
    %975 = stablehlo.add %969, %974 : tensor<f64>
    %976 = stablehlo.slice %938 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %977 = stablehlo.reshape %976 : (tensor<1xf64>) -> tensor<f64>
    %978 = stablehlo.slice %960 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %979 = stablehlo.reshape %978 : (tensor<1xf64>) -> tensor<f64>
    %980 = stablehlo.multiply %977, %979 : tensor<f64>
    %981 = stablehlo.subtract %975, %980 : tensor<f64>
    %982 = stablehlo.reshape %981 : (tensor<f64>) -> tensor<1xf64>
    %983 = stablehlo.multiply %940, %979 : tensor<f64>
    %984 = stablehlo.multiply %965, %973 : tensor<f64>
    %985 = stablehlo.subtract %983, %984 : tensor<f64>
    %986 = stablehlo.multiply %971, %967 : tensor<f64>
    %987 = stablehlo.add %985, %986 : tensor<f64>
    %988 = stablehlo.multiply %977, %962 : tensor<f64>
    %989 = stablehlo.add %987, %988 : tensor<f64>
    %990 = stablehlo.reshape %989 : (tensor<f64>) -> tensor<1xf64>
    %991 = stablehlo.multiply %940, %973 : tensor<f64>
    %992 = stablehlo.multiply %965, %979 : tensor<f64>
    %993 = stablehlo.add %991, %992 : tensor<f64>
    %994 = stablehlo.multiply %971, %962 : tensor<f64>
    %995 = stablehlo.subtract %993, %994 : tensor<f64>
    %996 = stablehlo.multiply %977, %967 : tensor<f64>
    %997 = stablehlo.add %995, %996 : tensor<f64>
    %998 = stablehlo.reshape %997 : (tensor<f64>) -> tensor<1xf64>
    %999 = stablehlo.multiply %940, %967 : tensor<f64>
    %1000 = stablehlo.multiply %965, %962 : tensor<f64>
    %1001 = stablehlo.subtract %999, %1000 : tensor<f64>
    %1002 = stablehlo.multiply %971, %979 : tensor<f64>
    %1003 = stablehlo.subtract %1001, %1002 : tensor<f64>
    %1004 = stablehlo.multiply %977, %973 : tensor<f64>
    %1005 = stablehlo.subtract %1003, %1004 : tensor<f64>
    %1006 = stablehlo.reshape %1005 : (tensor<f64>) -> tensor<1xf64>
    %1007 = stablehlo.concatenate %982, %990, %998, %1006, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1008 = stablehlo.slice %1007 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1009 = stablehlo.reshape %1008 : (tensor<1xf64>) -> tensor<f64>
    %1010 = stablehlo.slice %arg2 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1011 = stablehlo.reshape %1010 : (tensor<1xf64>) -> tensor<f64>
    %1012 = stablehlo.slice %42 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1013 = stablehlo.reshape %1012 : (tensor<1xf64>) -> tensor<f64>
    %1014 = stablehlo.multiply %1011, %1013 : tensor<f64>
    %1015 = stablehlo.slice %arg2 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1016 = stablehlo.reshape %1015 : (tensor<1xf64>) -> tensor<f64>
    %1017 = stablehlo.slice %42 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1018 = stablehlo.reshape %1017 : (tensor<1xf64>) -> tensor<f64>
    %1019 = stablehlo.multiply %1016, %1018 : tensor<f64>
    %1020 = stablehlo.add %1014, %1019 : tensor<f64>
    %1021 = stablehlo.slice %arg2 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1022 = stablehlo.reshape %1021 : (tensor<1xf64>) -> tensor<f64>
    %1023 = stablehlo.slice %42 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1024 = stablehlo.reshape %1023 : (tensor<1xf64>) -> tensor<f64>
    %1025 = stablehlo.multiply %1022, %1024 : tensor<f64>
    %1026 = stablehlo.add %1020, %1025 : tensor<f64>
    %1027 = stablehlo.slice %arg2 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1028 = stablehlo.reshape %1027 : (tensor<1xf64>) -> tensor<f64>
    %1029 = stablehlo.slice %42 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1030 = stablehlo.reshape %1029 : (tensor<1xf64>) -> tensor<f64>
    %1031 = stablehlo.multiply %1028, %1030 : tensor<f64>
    %1032 = stablehlo.subtract %1026, %1031 : tensor<f64>
    %1033 = stablehlo.reshape %1032 : (tensor<f64>) -> tensor<1xf64>
    %1034 = stablehlo.multiply %1011, %1030 : tensor<f64>
    %1035 = stablehlo.multiply %1016, %1024 : tensor<f64>
    %1036 = stablehlo.subtract %1034, %1035 : tensor<f64>
    %1037 = stablehlo.multiply %1022, %1018 : tensor<f64>
    %1038 = stablehlo.add %1036, %1037 : tensor<f64>
    %1039 = stablehlo.multiply %1028, %1013 : tensor<f64>
    %1040 = stablehlo.add %1038, %1039 : tensor<f64>
    %1041 = stablehlo.reshape %1040 : (tensor<f64>) -> tensor<1xf64>
    %1042 = stablehlo.multiply %1011, %1024 : tensor<f64>
    %1043 = stablehlo.multiply %1016, %1030 : tensor<f64>
    %1044 = stablehlo.add %1042, %1043 : tensor<f64>
    %1045 = stablehlo.multiply %1022, %1013 : tensor<f64>
    %1046 = stablehlo.subtract %1044, %1045 : tensor<f64>
    %1047 = stablehlo.multiply %1028, %1018 : tensor<f64>
    %1048 = stablehlo.add %1046, %1047 : tensor<f64>
    %1049 = stablehlo.reshape %1048 : (tensor<f64>) -> tensor<1xf64>
    %1050 = stablehlo.multiply %1011, %1018 : tensor<f64>
    %1051 = stablehlo.multiply %1016, %1013 : tensor<f64>
    %1052 = stablehlo.subtract %1050, %1051 : tensor<f64>
    %1053 = stablehlo.multiply %1022, %1030 : tensor<f64>
    %1054 = stablehlo.subtract %1052, %1053 : tensor<f64>
    %1055 = stablehlo.multiply %1028, %1024 : tensor<f64>
    %1056 = stablehlo.subtract %1054, %1055 : tensor<f64>
    %1057 = stablehlo.reshape %1056 : (tensor<f64>) -> tensor<1xf64>
    %1058 = stablehlo.concatenate %1033, %1041, %1049, %1057, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1059 = stablehlo.slice %1058 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1060 = stablehlo.reshape %1059 : (tensor<1xf64>) -> tensor<f64>
    %1061 = stablehlo.multiply %1009, %1060 : tensor<f64>
    %1062 = stablehlo.slice %1007 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1063 = stablehlo.reshape %1062 : (tensor<1xf64>) -> tensor<f64>
    %1064 = stablehlo.slice %1058 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1065 = stablehlo.reshape %1064 : (tensor<1xf64>) -> tensor<f64>
    %1066 = stablehlo.multiply %1063, %1065 : tensor<f64>
    %1067 = stablehlo.add %1061, %1066 : tensor<f64>
    %1068 = stablehlo.slice %1007 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1069 = stablehlo.reshape %1068 : (tensor<1xf64>) -> tensor<f64>
    %1070 = stablehlo.slice %1058 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1071 = stablehlo.reshape %1070 : (tensor<1xf64>) -> tensor<f64>
    %1072 = stablehlo.multiply %1069, %1071 : tensor<f64>
    %1073 = stablehlo.add %1067, %1072 : tensor<f64>
    %1074 = stablehlo.slice %1007 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1075 = stablehlo.reshape %1074 : (tensor<1xf64>) -> tensor<f64>
    %1076 = stablehlo.slice %1058 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1077 = stablehlo.reshape %1076 : (tensor<1xf64>) -> tensor<f64>
    %1078 = stablehlo.multiply %1075, %1077 : tensor<f64>
    %1079 = stablehlo.subtract %1073, %1078 : tensor<f64>
    %1080 = stablehlo.reshape %1079 : (tensor<f64>) -> tensor<1xf64>
    %1081 = stablehlo.multiply %1009, %1077 : tensor<f64>
    %1082 = stablehlo.multiply %1063, %1071 : tensor<f64>
    %1083 = stablehlo.subtract %1081, %1082 : tensor<f64>
    %1084 = stablehlo.multiply %1069, %1065 : tensor<f64>
    %1085 = stablehlo.add %1083, %1084 : tensor<f64>
    %1086 = stablehlo.multiply %1075, %1060 : tensor<f64>
    %1087 = stablehlo.add %1085, %1086 : tensor<f64>
    %1088 = stablehlo.reshape %1087 : (tensor<f64>) -> tensor<1xf64>
    %1089 = stablehlo.multiply %1009, %1071 : tensor<f64>
    %1090 = stablehlo.multiply %1063, %1077 : tensor<f64>
    %1091 = stablehlo.add %1089, %1090 : tensor<f64>
    %1092 = stablehlo.multiply %1069, %1060 : tensor<f64>
    %1093 = stablehlo.subtract %1091, %1092 : tensor<f64>
    %1094 = stablehlo.multiply %1075, %1065 : tensor<f64>
    %1095 = stablehlo.add %1093, %1094 : tensor<f64>
    %1096 = stablehlo.reshape %1095 : (tensor<f64>) -> tensor<1xf64>
    %1097 = stablehlo.multiply %1009, %1065 : tensor<f64>
    %1098 = stablehlo.multiply %1063, %1060 : tensor<f64>
    %1099 = stablehlo.subtract %1097, %1098 : tensor<f64>
    %1100 = stablehlo.multiply %1069, %1077 : tensor<f64>
    %1101 = stablehlo.subtract %1099, %1100 : tensor<f64>
    %1102 = stablehlo.multiply %1075, %1071 : tensor<f64>
    %1103 = stablehlo.subtract %1101, %1102 : tensor<f64>
    %1104 = stablehlo.reshape %1103 : (tensor<f64>) -> tensor<1xf64>
    %1105 = stablehlo.concatenate %1080, %1088, %1096, %1104, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1106 = stablehlo.slice %1105 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1107 = stablehlo.reshape %1106 : (tensor<1xf64>) -> tensor<f64>
    %1108 = stablehlo.slice %1105 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1109 = stablehlo.reshape %1108 : (tensor<1xf64>) -> tensor<f64>
    %1110 = stablehlo.slice %1105 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1111 = stablehlo.reshape %1110 : (tensor<1xf64>) -> tensor<f64>
    %1112 = stablehlo.slice %1105 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1113 = stablehlo.reshape %1112 : (tensor<1xf64>) -> tensor<f64>
    %1114 = stablehlo.multiply %1107, %1107 : tensor<f64>
    %1115 = stablehlo.multiply %1109, %1109 : tensor<f64>
    %1116 = stablehlo.add %1114, %1115 : tensor<f64>
    %1117 = stablehlo.multiply %1111, %1111 : tensor<f64>
    %1118 = stablehlo.add %1116, %1117 : tensor<f64>
    %1119 = stablehlo.sqrt %1118 : tensor<f64>
    %1120 = stablehlo.broadcast_in_dim %1107, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1121 = stablehlo.broadcast_in_dim %1109, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1122 = stablehlo.broadcast_in_dim %1111, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1123 = stablehlo.concatenate %1120, %1121, %1122, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %cst_35 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %1124 = stablehlo.compare  LT, %1119, %cst_35,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1125 = stablehlo.convert %1124 : (tensor<i1>) -> tensor<i32>
    %1126 = "stablehlo.case"(%1125) ({
      %1153 = stablehlo.broadcast_in_dim %1119, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1154 = stablehlo.divide %1123, %1153 : tensor<3xf64>
      %1155 = stablehlo.atan2 %1119, %1113 : tensor<f64>
      %cst_41 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %1156 = stablehlo.multiply %cst_41, %1155 : tensor<f64>
      %cst_42 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
      %1157 = func.call @remainder(%1156, %cst_42) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %cst_43 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1158 = stablehlo.compare  LT, %1157, %cst_43,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_44 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
      %1159 = stablehlo.add %1157, %cst_44 : tensor<f64>
      %1160 = func.call @_where(%1158, %1159, %1157) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %cst_45 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
      %1161 = stablehlo.compare  GT, %1160, %cst_45,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %cst_46 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
      %1162 = stablehlo.subtract %1160, %cst_46 : tensor<f64>
      %1163 = func.call @_where(%1161, %1162, %1160) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %1164 = stablehlo.broadcast_in_dim %1163, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %1165 = stablehlo.multiply %1154, %1164 : tensor<3xf64>
      stablehlo.return %1165 : tensor<3xf64>
    }, {
      stablehlo.return %1123 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    %1127 = stablehlo.slice %1126 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %1128 = stablehlo.reshape %1127 : (tensor<1xf64>) -> tensor<f64>
    %1129 = stablehlo.slice %1126 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %1130 = stablehlo.reshape %1129 : (tensor<1xf64>) -> tensor<f64>
    %1131 = stablehlo.slice %1126 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %1132 = stablehlo.reshape %1131 : (tensor<1xf64>) -> tensor<f64>
    %1133 = stablehlo.broadcast_in_dim %915, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1134 = stablehlo.broadcast_in_dim %917, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1135 = stablehlo.broadcast_in_dim %1132, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1136 = stablehlo.concatenate %1133, %1134, %1135, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1137 = stablehlo.multiply %1136, %cst_1 : tensor<3xf64>
    %cst_36 = stablehlo.constant dense<1.0471975511965976> : tensor<f64>
    %1138 = stablehlo.compare  GT, %731, %cst_36,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %1139 = stablehlo.convert %1138 : (tensor<i1>) -> tensor<i32>
    %1140 = "stablehlo.case"(%1139) ({
      %cst_41 = stablehlo.constant dense<0.52359877559829882> : tensor<f64>
      %1153 = stablehlo.compare  GT, %731, %cst_41,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %1154 = stablehlo.convert %1153 : (tensor<i1>) -> tensor<i32>
      %1155 = "stablehlo.case"(%1154) ({
        %1156 = stablehlo.add %1137, %420 : tensor<3xf64>
        stablehlo.return %1156 : tensor<3xf64>
      }, {
        %cst_42 = stablehlo.constant dense<0.52359877559829882> : tensor<f64>
        %1156 = stablehlo.subtract %731, %cst_42 : tensor<f64>
        %cst_43 = stablehlo.constant dense<0.52359877559829882> : tensor<f64>
        %1157 = stablehlo.divide %1156, %cst_43 : tensor<f64>
        %cst_44 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %1158 = stablehlo.subtract %cst_44, %1157 : tensor<f64>
        %1159 = stablehlo.slice %420 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
        %1160 = stablehlo.reshape %1159 : (tensor<1xf64>) -> tensor<f64>
        %1161 = stablehlo.multiply %1160, %1158 : tensor<f64>
        %1162 = stablehlo.slice %420 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
        %1163 = stablehlo.reshape %1162 : (tensor<1xf64>) -> tensor<f64>
        %1164 = stablehlo.multiply %1163, %1158 : tensor<f64>
        %1165 = stablehlo.slice %420 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
        %1166 = stablehlo.reshape %1165 : (tensor<1xf64>) -> tensor<f64>
        %1167 = stablehlo.broadcast_in_dim %1161, dims = [] : (tensor<f64>) -> tensor<1xf64>
        %1168 = stablehlo.broadcast_in_dim %1164, dims = [] : (tensor<f64>) -> tensor<1xf64>
        %1169 = stablehlo.broadcast_in_dim %1166, dims = [] : (tensor<f64>) -> tensor<1xf64>
        %1170 = stablehlo.concatenate %1167, %1168, %1169, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
        %1171 = stablehlo.add %1137, %1170 : tensor<3xf64>
        %1172 = stablehlo.slice %arg1 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
        %1173 = stablehlo.reshape %1172 : (tensor<1xf64>) -> tensor<f64>
        %cst_45 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %1174 = stablehlo.subtract %cst_45, %1158 : tensor<f64>
        %1175 = stablehlo.multiply %1173, %1174 : tensor<f64>
        %1176 = stablehlo.slice %1171 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
        %1177 = stablehlo.reshape %1176 : (tensor<1xf64>) -> tensor<f64>
        %1178 = stablehlo.multiply %1177, %1158 : tensor<f64>
        %1179 = stablehlo.add %1175, %1178 : tensor<f64>
        %1180 = stablehlo.slice %1171 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
        %1181 = stablehlo.reshape %1180 : (tensor<1xf64>) -> tensor<f64>
        %1182 = stablehlo.slice %1171 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
        %1183 = stablehlo.reshape %1182 : (tensor<1xf64>) -> tensor<f64>
        %1184 = stablehlo.broadcast_in_dim %1181, dims = [] : (tensor<f64>) -> tensor<1xf64>
        %1185 = stablehlo.broadcast_in_dim %1183, dims = [] : (tensor<f64>) -> tensor<1xf64>
        %1186 = stablehlo.broadcast_in_dim %1179, dims = [] : (tensor<f64>) -> tensor<1xf64>
        %1187 = stablehlo.concatenate %1184, %1185, %1186, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
        stablehlo.return %1187 : tensor<3xf64>
      }) : (tensor<i32>) -> tensor<3xf64>
      stablehlo.return %1155 : tensor<3xf64>
    }, {
      %1153 = stablehlo.slice %1137 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
      %1154 = stablehlo.reshape %1153 : (tensor<1xf64>) -> tensor<f64>
      %1155 = stablehlo.slice %1137 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
      %1156 = stablehlo.reshape %1155 : (tensor<1xf64>) -> tensor<f64>
      %1157 = stablehlo.slice %arg1 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
      %1158 = stablehlo.reshape %1157 : (tensor<1xf64>) -> tensor<f64>
      %1159 = stablehlo.broadcast_in_dim %1154, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %1160 = stablehlo.broadcast_in_dim %1156, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %1161 = stablehlo.broadcast_in_dim %1158, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %1162 = stablehlo.concatenate %1159, %1160, %1161, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
      stablehlo.return %1162 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    %cst_37 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %1141 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1142 = stablehlo.multiply %1141, %cst_2 : tensor<3xf64>
    %cst_38 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1143 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1144 = stablehlo.divide %1143, %1142 : tensor<3xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1145 = call @nan_to_num_58(%1144, %c) : (tensor<3xf64>, tensor<i64>) -> tensor<3xf64>
    %cst_39 = stablehlo.constant dense<3.000000e+02> : tensor<f64>
    %1146 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1147 = stablehlo.add %1145, %1146 : tensor<3xf64>
    %cst_40 = stablehlo.constant dense<3.000000e+02> : tensor<f64>
    %1148 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1149 = stablehlo.divide %1148, %1147 : tensor<3xf64>
    %1150 = stablehlo.subtract %1140, %arg4 : tensor<3xf64>
    %1151 = stablehlo.multiply %1149, %1150 : tensor<3xf64>
    %1152 = stablehlo.add %arg4, %1151 : tensor<3xf64>
    return %1152 : tensor<3xf64>
  }
  func.func private @cross(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.slice %arg0 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %arg1 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.slice %arg1 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %9 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
    %10 = stablehlo.slice %arg1 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.multiply %3, %11 : tensor<f64>
    %13 = stablehlo.multiply %5, %9 : tensor<f64>
    %14 = stablehlo.subtract %12, %13 : tensor<f64>
    %15 = stablehlo.multiply %5, %7 : tensor<f64>
    %16 = stablehlo.multiply %1, %11 : tensor<f64>
    %17 = stablehlo.subtract %15, %16 : tensor<f64>
    %18 = stablehlo.multiply %1, %9 : tensor<f64>
    %19 = stablehlo.multiply %3, %7 : tensor<f64>
    %20 = stablehlo.subtract %18, %19 : tensor<f64>
    %21 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %22 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %23 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.concatenate %21, %22, %23, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %24 : tensor<3xf64>
  }
  func.func private @nan_to_num_58(%arg0: tensor<3xf64>, %arg1: tensor<i64>) -> tensor<3xf64> {
    %0 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = call @_where_23(%0, %cst, %arg0) : (tensor<3xi1>, tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.compare  EQ, %1, %2,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %4 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<f64>
    %5 = call @_where_23(%3, %4, %1) : (tensor<3xi1>, tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %7 = stablehlo.compare  EQ, %5, %6,  FLOAT : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xi1>
    %cst_2 = stablehlo.constant dense<-1.7976931348623157E+308> : tensor<f64>
    %8 = call @_where_23(%7, %cst_2, %5) : (tensor<3xi1>, tensor<f64>, tensor<3xf64>) -> tensor<3xf64>
    return %8 : tensor<3xf64>
  }
  func.func private @inner_62(%arg0: tensor<3x3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>) -> tensor<3x3xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 2.500000e+00]> : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<[1.000000e+01, 1.000000e+01, 0.000000e+00]> : tensor<3xf64>
    %cst_1 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.multiply %0, %cst : tensor<3xf64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.divide %2, %1 : tensor<3xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %4 = call @nan_to_num_58(%3, %c) : (tensor<3xf64>, tensor<i64>) -> tensor<3xf64>
    %cst_3 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.add %4, %5 : tensor<3xf64>
    %cst_4 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.divide %7, %6 : tensor<3xf64>
    %cst_5 = stablehlo.constant dense<6.2831853071795862> : tensor<f64>
    %9 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %10 = stablehlo.multiply %9, %cst_0 : tensor<3xf64>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %11 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %12 = stablehlo.divide %11, %10 : tensor<3xf64>
    %c_7 = stablehlo.constant dense<0> : tensor<i64>
    %13 = call @nan_to_num_58(%12, %c_7) : (tensor<3xf64>, tensor<i64>) -> tensor<3xf64>
    %cst_8 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %14 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %15 = stablehlo.add %13, %14 : tensor<3xf64>
    %cst_9 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %16 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %17 = stablehlo.divide %16, %15 : tensor<3xf64>
    %18 = stablehlo.slice %arg0 [0:1, 0:3] : (tensor<3x3xf64>) -> tensor<1x3xf64>
    %19 = stablehlo.reshape %18 : (tensor<1x3xf64>) -> tensor<3xf64>
    %20 = stablehlo.slice %arg0 [1:2, 0:3] : (tensor<3x3xf64>) -> tensor<1x3xf64>
    %21 = stablehlo.reshape %20 : (tensor<1x3xf64>) -> tensor<3xf64>
    %22 = stablehlo.slice %arg0 [2:3, 0:3] : (tensor<3x3xf64>) -> tensor<1x3xf64>
    %23 = stablehlo.reshape %22 : (tensor<1x3xf64>) -> tensor<3xf64>
    %24 = stablehlo.subtract %arg1, %arg2 : tensor<3xf64>
    %25 = stablehlo.subtract %24, %19 : tensor<3xf64>
    %26 = stablehlo.multiply %8, %25 : tensor<3xf64>
    %27 = stablehlo.add %19, %26 : tensor<3xf64>
    %cst_10 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %28 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %29 = stablehlo.multiply %27, %28 : tensor<3xf64>
    %30 = stablehlo.add %21, %29 : tensor<3xf64>
    %31 = stablehlo.subtract %27, %19 : tensor<3xf64>
    %cst_11 = stablehlo.constant dense<0.0033333333333333335> : tensor<f64>
    %32 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %33 = stablehlo.divide %31, %32 : tensor<3xf64>
    %34 = stablehlo.subtract %33, %23 : tensor<3xf64>
    %35 = stablehlo.multiply %17, %34 : tensor<3xf64>
    %36 = stablehlo.add %23, %35 : tensor<3xf64>
    %37 = stablehlo.broadcast_in_dim %27, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %38 = stablehlo.broadcast_in_dim %30, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %39 = stablehlo.broadcast_in_dim %36, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %40 = stablehlo.concatenate %37, %38, %39, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<3x3xf64>
    return %40 : tensor<3x3xf64>
  }
  func.func private @inner_66(%arg0: tensor<3x3xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %cst = stablehlo.constant dense<[[3.200000e-01, 3.200000e-01, 1.100000e+00], [5.000000e-02, 5.000000e-02, 8.000000e-02], [1.200000e-01, 8.000000e-02, 1.000000e-02]]> : tensor<3x3xf64>
    %0 = stablehlo.multiply %arg0, %cst : tensor<3x3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<3x3xf64>, tensor<f64>) -> tensor<3xf64>
    %2 = stablehlo.slice %1 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.slice %1 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %1 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %9 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_1 = stablehlo.constant dense<7.390000e-01> : tensor<f64>
    %11 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %12 = stablehlo.concatenate %8, %9, %10, %11, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    return %12 : tensor<4xf64>
  }
  func.func private @inner_69(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %cst = stablehlo.constant dense<[-0.49999999999999994, 0.49999999999999994, 0.49999999999999994, -5.000000e-01]> : tensor<4xf64>
    %cst_0 = stablehlo.constant dense<[-0.49999999999999994, 5.000000e-01, -5.000000e-01, 0.49999999999999994]> : tensor<4xf64>
    %cst_1 = stablehlo.constant dense<[-5.000000e-01, -5.000000e-01, 5.000000e-01, 5.000000e-01]> : tensor<4xf64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<4xf64>
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.slice %arg0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %arg0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %8 = stablehlo.multiply %cst_3, %7 : tensor<f64>
    %cst_4 = stablehlo.constant dense<3.445000e-01> : tensor<f64>
    %9 = stablehlo.add %cst_4, %8 : tensor<f64>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %10 = call @clip_70(%9, %7, %cst_5) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_6 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %11 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %12 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %13 = stablehlo.concatenate %11, %12, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    %cst_7 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %14 = stablehlo.reduce(%13 init: %cst_7) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %15 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %16 = stablehlo.multiply %15, %cst : tensor<4xf64>
    %17 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %18 = stablehlo.multiply %17, %cst_0 : tensor<4xf64>
    %19 = stablehlo.add %16, %18 : tensor<4xf64>
    %20 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %21 = stablehlo.add %19, %20 : tensor<4xf64>
    %22 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %23 = stablehlo.multiply %22, %cst_1 : tensor<4xf64>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %24 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %25 = stablehlo.subtract %24, %21 : tensor<4xf64>
    %26 = call @_where_75(%23, %25, %21) : (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %27 = call @clip_78(%26, %cst_9) : (tensor<4xf64>, tensor<f64>) -> tensor<4xf64>
    %28 = stablehlo.abs %cst_1 : tensor<4xf64>
    %29 = stablehlo.divide %27, %28 : tensor<4xf64>
    %cst_10 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %30 = stablehlo.reduce(%29 init: %cst_10) applies stablehlo.minimum across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    %31 = stablehlo.negate %30 : tensor<f64>
    %32 = call @clip_19(%5, %31, %30) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %34 = stablehlo.multiply %33, %cst_1 : tensor<4xf64>
    %35 = stablehlo.add %19, %34 : tensor<4xf64>
    %cst_11 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %36 = stablehlo.reduce(%35 init: %cst_11) applies stablehlo.minimum across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    %cst_12 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %37 = stablehlo.reduce(%35 init: %cst_12) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    %38 = stablehlo.subtract %37, %36 : tensor<f64>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %39 = stablehlo.compare  GT, %38, %cst_13,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %40 = stablehlo.convert %39 : (tensor<i1>) -> tensor<i32>
    %41 = "stablehlo.case"(%40) ({
      %cst_27 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      stablehlo.return %cst_27 : tensor<f64>
    }, {
      %79 = stablehlo.subtract %37, %36 : tensor<f64>
      %cst_27 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %80 = stablehlo.divide %cst_27, %79 : tensor<f64>
      stablehlo.return %80 : tensor<f64>
    }) : (tensor<i32>) -> tensor<f64>
    %42 = stablehlo.add %10, %36 : tensor<f64>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %43 = stablehlo.compare  LT, %42, %cst_14,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %44 = stablehlo.convert %43 : (tensor<i1>) -> tensor<i32>
    %45 = "stablehlo.case"(%44) ({
      stablehlo.return %41 : tensor<f64>
    }, {
      %79 = stablehlo.negate %10 : tensor<f64>
      %80 = stablehlo.divide %79, %36 : tensor<f64>
      %81 = stablehlo.convert %41 : tensor<f64>
      %82 = stablehlo.broadcast_in_dim %81, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %83 = stablehlo.broadcast_in_dim %80, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %84 = stablehlo.concatenate %82, %83, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
      %cst_27 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
      %85 = stablehlo.reduce(%84 init: %cst_27) applies stablehlo.minimum across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
      stablehlo.return %85 : tensor<f64>
    }) : (tensor<i32>) -> tensor<f64>
    %46 = stablehlo.convert %45 : tensor<f64>
    %47 = stablehlo.multiply %36, %46 : tensor<f64>
    %48 = stablehlo.convert %45 : tensor<f64>
    %49 = stablehlo.multiply %37, %48 : tensor<f64>
    %50 = stablehlo.negate %47 : tensor<f64>
    %51 = stablehlo.subtract %7, %50 : tensor<f64>
    %cst_15 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %52 = stablehlo.compare  LT, %45, %cst_15,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %53 = call @_where_85(%52, %cst_16, %51) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %54 = stablehlo.add %50, %49 : tensor<f64>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %55 = stablehlo.subtract %cst_17, %54 : tensor<f64>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %56 = call @clip_87(%53, %cst_18, %55) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %57 = stablehlo.add %50, %56 : tensor<f64>
    %58 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %59 = stablehlo.multiply %58, %cst_2 : tensor<4xf64>
    %60 = stablehlo.convert %45 : tensor<f64>
    %61 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %62 = stablehlo.multiply %35, %61 : tensor<4xf64>
    %63 = stablehlo.add %59, %62 : tensor<4xf64>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_20 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %64 = call @clip_89(%63, %cst_19, %cst_20) : (tensor<4xf64>, tensor<f64>, tensor<f64>) -> tensor<4xf64>
    %65 = stablehlo.negate %64 : tensor<4xf64>
    %cst_21 = stablehlo.constant dense<3.332000e+00> : tensor<f64>
    %66 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %67 = stablehlo.multiply %66, %65 : tensor<4xf64>
    %cst_22 = stablehlo.constant dense<0.027889000000000011> : tensor<f64>
    %68 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %69 = stablehlo.subtract %68, %67 : tensor<4xf64>
    %70 = stablehlo.sqrt %69 : tensor<4xf64>
    %cst_23 = stablehlo.constant dense<-0.16700000000000004> : tensor<f64>
    %71 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %72 = stablehlo.add %71, %70 : tensor<4xf64>
    %cst_24 = stablehlo.constant dense<1.666000e+00> : tensor<f64>
    %73 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %74 = stablehlo.divide %72, %73 : tensor<4xf64>
    %cst_25 = stablehlo.constant dense<7.055000e+02> : tensor<f64>
    %75 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %76 = stablehlo.multiply %74, %75 : tensor<4xf64>
    %cst_26 = stablehlo.constant dense<1.152000e+03> : tensor<f64>
    %77 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %78 = stablehlo.add %76, %77 : tensor<4xf64>
    return %78 : tensor<4xf64>
  }
  func.func private @clip_70(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<f64>
    %1 = stablehlo.convert %arg2 : tensor<f64>
    %2 = stablehlo.minimum %1, %0 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @_where_75(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1 = stablehlo.compare  NE, %arg0, %0,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %2 = stablehlo.select %1, %arg1, %arg2 : tensor<4xi1>, tensor<4xf64>
    return %2 : tensor<4xf64>
  }
  func.func private @clip_78(%arg0: tensor<4xf64>, %arg1: tensor<f64>) -> tensor<4xf64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2 = stablehlo.maximum %1, %arg0 : tensor<4xf64>
    return %2 : tensor<4xf64>
  }
  func.func private @_where_85(%arg0: tensor<i1>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @clip_87(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f64>
    %2 = stablehlo.minimum %arg2, %1 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @clip_89(%arg0: tensor<4xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<4xf64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2 = stablehlo.maximum %1, %arg0 : tensor<4xf64>
    %3 = stablehlo.convert %arg2 : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %5 = stablehlo.minimum %4, %2 : tensor<4xf64>
    return %5 : tensor<4xf64>
  }
  func.func private @inner_96(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<"0x4F621058E9089240DCB5847C78179240CE66D5E79A369240AA8BDB689C46924071CE88D2026692409A081B9E2A769240FE21FDF6CD9492402AF697DDD3A492405A17B7D134C49240837CD0B391D39240C1CAA1454AF39240565BB1BFF4019340A267B3EAEB229340B8AF03E71C30934098DD93875D5193408E31772DA55E9340F1F44A597E809340E0E995B2208D93407AA52C430CB093401CC9E53FF8BC9340184850FC3CDF9340D2BCE31401EC9340014D840D270E944026E4839EF91A944075E09C112D3E944052FC1873034B94405917B7D1346D9440131DC9E59B7A9440B6627FD9B19B944010A5BDC16BA994403A234A7BCBCA94408BB96B093DD894406B787AA540FA944012C7BAB851079540780B2428A229954029A913D0B0369540DE02098AC7589540B27BF2B0EC6595404BEA04344D88954082C0CAA1DD95954061C3D32B01B89540B71E85EB45C59540E926310898E79540DE9387859EF495409E5E29CBF416964006F01648282496403108AC1C16469640265C8FC2ED5396404703780B3C75964088B0E1E955829640D95F764F72A496401EF46C56ADB19640A2B437F8A6D496406054522780E0964063AA6054BA0397406B09F9A0AB0F974076BE9F1AEB33974066B3EA73A53E974055302AA96F649740FBA9F1D2616D97402A3A92CB0F94974060545227949C97404950FC182BC49740A635CD3B12CC9740DF718A8E5CF49740F97E6ABCFCFA9740A0CDAACF2D249840F6E461A13E2A9840635DDC46B3539840EDC9C342795A98407D3F355E4A8398402EFF21FD028A98406F3480B7B8B1984043696FF001B99840FF43FAEDD3E09840273108AC4CE89840C64B378919109940A60A4625C5169940E07A14AEC33E9940A64E4013F94499409E3C2CD4D66D9940AE03E78C00749940287E8CB95B9D99405EC3D32BC5A2994070F085C924CD9940FACBEEC933D19940B27BF2B034FC9940769CA223E5FF9940BBB88D06902B9A402EDD24062D2F9A40F6065F98AC5A9A400B4FAF94915E9A4025287E8CCD8A9A408DB96B09F58D9A405B423EE8F5B99A40ADFA5C6D79BD9A40302AA91354E99A40156A4DF346EC9A4050AF9465C4189B40BB96900FD61B9B402B1895D4A1479B40FDB27BF2984B9B4036CD3B4E25779B40FD1873D78A7A9B402506819547A69B402D431CEBD6A99B403892CB7FD4D59B403D2CD49A76D99B40F0C9C34231069C40AEB6627FE9089C4058CA32C4E9359C409DCDAACFF5379C402575029A84649C405A643BDF93679C4063105839C8939C400D0BB5A681979C405A17B7D1DCC29C40BE9F1A2F3DC69C4062E5D0223BF29C40557424976BF59C40"> : tensor<120xf64>
    %cst_0 = stablehlo.constant dense<"0x0000000000ACAA40000000000038AD400000000000CCAE400000000000C8B04000000000009AB1400000000000E7B2400000000000B4B3400000000000F5B4400000000000FCB54000000000000DB740000000000016B840000000000003B94000000000003ABA400000000000F1BA40000000000043BC40000000000007BD40000000000043BE400000000000D2BE40000000008023C040000000000063C040000000000019C140000000008059C140000000000018C24000000000003EC24000000000800FC340000000008034C3400000000080FFC34000000000001EC4400000000000E7C4400000000080E8C4400000000000CBC5400000000080D8C5400000000000A5C6400000000000AFC640000000000083C740000000008089C74000000000006DC84000000000804CC84000000000003FC940000000008013C94000000000800FCA400000000000DAC9400000000000D6CA400000000080A1CA4000000000009FCB4000000000806BCB4000000000006BCC40000000008027CC40000000000033CD400000000000E9CC400000000080EECD400000000080A4CD400000000000B8CE4000000000005DCE40000000000076CF4000000000801BCF40000000004019D0400000000080CDCF40000000008079D04000000000C03FD04000000000C0E2D04000000000C0B1D040000000004041D140000000008001D14000000000C0ABD14000000000805ED14000000000800BD2400000000000C6D140000000000072D24000000000802FD2400000000000DBD240000000000096D24000000000403FD3400000000080EDD2400000000040AAD34000000000C04FD34000000000C00AD4400000000080B5D34000000000406FD440000000000022D4400000000040D4D44000000000C07CD44000000000C041D5400000000000E5D4400000000080AED54000000000C04AD540000000008065D64000000000C0C8D540000000000007D74000000000C09AD640000000008078D740000000004006D7400000000080E1D74000000000C076D740000000004055D8400000000000F6D7400000000000D6D84000000000006FD840000000008044D9400000000080E3D8400000000000BBD94000000000C051D940000000008024DA400000000040BCD940000000008081DA40000000008025DA400000000080E6DA40000000004081DA40000000008046DB400000000000E7DA40000000004098DB40000000000042DB4000000000C0EBDB40000000004097DB4000000000C03ADC400000000080F0DB4000000000007FDC40000000004039DC400000000040C4DC4000000000C087DC40"> : tensor<120xf64>
    %cst_1 = stablehlo.constant dense<[-5.000000e-01, -5.000000e-01, 5.000000e-01, 5.000000e-01]> : tensor<4xf64>
    %0 = call @_interp(%arg0, %cst, %cst_0) : (tensor<4xf64>, tensor<120xf64>, tensor<120xf64>) -> tensor<4xf64>
    %1 = stablehlo.subtract %0, %arg3 : tensor<4xf64>
    %cst_2 = stablehlo.constant dense<0.01098901098901099> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %3 = stablehlo.multiply %2, %1 : tensor<4xf64>
    %4 = stablehlo.add %arg3, %3 : tensor<4xf64>
    %5 = stablehlo.multiply %4, %4 : tensor<4xf64>
    %cst_3 = stablehlo.constant dense<9.9068131782640682E-9> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %7 = stablehlo.multiply %5, %6 : tensor<4xf64>
    %8 = stablehlo.multiply %4, %4 : tensor<4xf64>
    %cst_4 = stablehlo.constant dense<9.8192338453001589E-11> : tensor<f64>
    %9 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %10 = stablehlo.multiply %8, %9 : tensor<4xf64>
    %11 = stablehlo.multiply %10, %cst_1 : tensor<4xf64>
    return %7, %11, %4 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
  }
  func.func private @_interp(%arg0: tensor<4xf64>, %arg1: tensor<120xf64>, %arg2: tensor<120xf64>) -> tensor<4xf64> {
    %0 = call @searchsorted(%arg1, %arg0) : (tensor<120xf64>, tensor<4xf64>) -> tensor<4xi32>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<119> : tensor<i64>
    %1 = call @clip_112(%0, %c, %c_0) : (tensor<4xi32>, tensor<i64>, tensor<i64>) -> tensor<4xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %3 = stablehlo.compare  LT, %1, %2,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_2 = stablehlo.constant dense<120> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %5 = stablehlo.add %1, %4 : tensor<4xi32>
    %6 = stablehlo.select %3, %5, %1 : tensor<4xi1>, tensor<4xi32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %8 = "stablehlo.gather"(%arg2, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %10 = stablehlo.subtract %1, %9 : tensor<4xi32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %12 = stablehlo.compare  LT, %10, %11,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_5 = stablehlo.constant dense<120> : tensor<i32>
    %13 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %14 = stablehlo.add %10, %13 : tensor<4xi32>
    %15 = stablehlo.select %12, %14, %10 : tensor<4xi1>, tensor<4xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %17 = "stablehlo.gather"(%arg2, %16) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %18 = stablehlo.subtract %8, %17 : tensor<4xf64>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %20 = stablehlo.compare  LT, %1, %19,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_7 = stablehlo.constant dense<120> : tensor<i32>
    %21 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %22 = stablehlo.add %1, %21 : tensor<4xi32>
    %23 = stablehlo.select %20, %22, %1 : tensor<4xi1>, tensor<4xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %25 = "stablehlo.gather"(%arg1, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %c_8 = stablehlo.constant dense<1> : tensor<i32>
    %26 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %27 = stablehlo.subtract %1, %26 : tensor<4xi32>
    %c_9 = stablehlo.constant dense<0> : tensor<i32>
    %28 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %29 = stablehlo.compare  LT, %27, %28,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_10 = stablehlo.constant dense<120> : tensor<i32>
    %30 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %31 = stablehlo.add %27, %30 : tensor<4xi32>
    %32 = stablehlo.select %29, %31, %27 : tensor<4xi1>, tensor<4xi32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %34 = "stablehlo.gather"(%arg1, %33) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %35 = stablehlo.subtract %25, %34 : tensor<4xf64>
    %c_11 = stablehlo.constant dense<1> : tensor<i32>
    %36 = stablehlo.broadcast_in_dim %c_11, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %37 = stablehlo.subtract %1, %36 : tensor<4xi32>
    %c_12 = stablehlo.constant dense<0> : tensor<i32>
    %38 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %39 = stablehlo.compare  LT, %37, %38,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_13 = stablehlo.constant dense<120> : tensor<i32>
    %40 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %41 = stablehlo.add %37, %40 : tensor<4xi32>
    %42 = stablehlo.select %39, %41, %37 : tensor<4xi1>, tensor<4xi32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %44 = "stablehlo.gather"(%arg1, %43) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %45 = stablehlo.subtract %arg0, %44 : tensor<4xf64>
    %46 = stablehlo.abs %35 : tensor<4xf64>
    %cst = stablehlo.constant dense<4.9303806576313238E-32> : tensor<f64>
    %47 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %48 = stablehlo.compare  LE, %46, %47,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %c_14 = stablehlo.constant dense<1> : tensor<i32>
    %49 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %50 = stablehlo.subtract %1, %49 : tensor<4xi32>
    %c_15 = stablehlo.constant dense<0> : tensor<i32>
    %51 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %52 = stablehlo.compare  LT, %50, %51,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_16 = stablehlo.constant dense<120> : tensor<i32>
    %53 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %54 = stablehlo.add %50, %53 : tensor<4xi32>
    %55 = stablehlo.select %52, %54, %50 : tensor<4xi1>, tensor<4xi32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %57 = "stablehlo.gather"(%arg2, %56) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %c_17 = stablehlo.constant dense<1> : tensor<i32>
    %58 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %59 = stablehlo.subtract %1, %58 : tensor<4xi32>
    %c_18 = stablehlo.constant dense<0> : tensor<i32>
    %60 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %61 = stablehlo.compare  LT, %59, %60,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_19 = stablehlo.constant dense<120> : tensor<i32>
    %62 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %63 = stablehlo.add %59, %62 : tensor<4xi32>
    %64 = stablehlo.select %61, %63, %59 : tensor<4xi1>, tensor<4xi32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %66 = "stablehlo.gather"(%arg2, %65) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4xf64>
    %c_20 = stablehlo.constant dense<1> : tensor<i64>
    %67 = call @_where_119(%48, %c_20, %35) : (tensor<4xi1>, tensor<i64>, tensor<4xf64>) -> tensor<4xf64>
    %68 = stablehlo.divide %45, %67 : tensor<4xf64>
    %69 = stablehlo.multiply %68, %18 : tensor<4xf64>
    %70 = stablehlo.add %66, %69 : tensor<4xf64>
    %71 = call @_where_121(%48, %57, %70) : (tensor<4xi1>, tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    %72 = stablehlo.slice %arg2 [0:1] : (tensor<120xf64>) -> tensor<1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1xf64>) -> tensor<f64>
    %74 = stablehlo.slice %arg1 [0:1] : (tensor<120xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %77 = stablehlo.compare  LT, %arg0, %76,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %78 = call @_where_124(%77, %73, %71) : (tensor<4xi1>, tensor<f64>, tensor<4xf64>) -> tensor<4xf64>
    %c_21 = stablehlo.constant dense<119> : tensor<i64>
    %79 = stablehlo.dynamic_slice %arg2, %c_21, sizes = [1] : (tensor<120xf64>, tensor<i64>) -> tensor<1xf64>
    %80 = stablehlo.reshape %79 : (tensor<1xf64>) -> tensor<f64>
    %c_22 = stablehlo.constant dense<119> : tensor<i64>
    %81 = stablehlo.dynamic_slice %arg1, %c_22, sizes = [1] : (tensor<120xf64>, tensor<i64>) -> tensor<1xf64>
    %82 = stablehlo.reshape %81 : (tensor<1xf64>) -> tensor<f64>
    %83 = stablehlo.broadcast_in_dim %82, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %84 = stablehlo.compare  GT, %arg0, %83,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %85 = call @_where_124(%84, %80, %78) : (tensor<4xi1>, tensor<f64>, tensor<4xf64>) -> tensor<4xf64>
    return %85 : tensor<4xf64>
  }
  func.func private @searchsorted(%arg0: tensor<120xf64>, %arg1: tensor<4xf64>) -> tensor<4xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %c_0 = stablehlo.constant dense<120> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %2:5 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %c_1, %iterArg_4 = %0, %iterArg_5 = %1) : tensor<120xf64>, tensor<4xf64>, tensor<i64>, tensor<4xi32>, tensor<4xi32>
    cond {
      %c_6 = stablehlo.constant dense<7> : tensor<i64>
      %3 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3:2 = func.call @closed_call(%iterArg, %iterArg_2, %iterArg_4, %iterArg_5) : (tensor<120xf64>, tensor<4xf64>, tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %4 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_2, %4, %3#0, %3#1 : tensor<120xf64>, tensor<4xf64>, tensor<i64>, tensor<4xi32>, tensor<4xi32>
    }
    return %2#4 : tensor<4xi32>
  }
  func.func private @closed_call(%arg0: tensor<120xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xi32>, %arg3: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %0 = stablehlo.convert %arg2 : (tensor<4xi32>) -> tensor<4xui32>
    %1 = stablehlo.convert %arg3 : (tensor<4xi32>) -> tensor<4xui32>
    %2 = stablehlo.add %0, %1 : tensor<4xui32>
    %c = stablehlo.constant dense<2> : tensor<ui32>
    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %4 = stablehlo.divide %2, %3 : tensor<4xui32>
    %5 = stablehlo.convert %4 : (tensor<4xui32>) -> tensor<4xi32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %7 = stablehlo.compare  LT, %5, %6,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    %c_1 = stablehlo.constant dense<120> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %9 = stablehlo.add %5, %8 : tensor<4xi32>
    %10 = stablehlo.select %7, %9, %5 : tensor<4xi1>, tensor<4xi32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %12 = "stablehlo.gather"(%arg0, %11) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<120xf64>, tensor<4x1xi32>) -> tensor<4x1xf64>
    %13 = stablehlo.reshape %12 : (tensor<4x1xf64>) -> tensor<4xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %15 = stablehlo.compare  EQ, %arg1, %14,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %17 = stablehlo.select %15, %16, %arg1 : tensor<4xi1>, tensor<4xf64>
    %18 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %20 = stablehlo.select %18, %19, %17 : tensor<4xi1>, tensor<4xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %21 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %22 = stablehlo.compare  EQ, %13, %21,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %23 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %24 = stablehlo.select %22, %23, %13 : tensor<4xi1>, tensor<4xf64>
    %25 = stablehlo.compare  NE, %13, %13,  FLOAT : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %cst_6 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %26 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %27 = stablehlo.select %25, %26, %24 : tensor<4xi1>, tensor<4xf64>
    %28 = stablehlo.compare  LT, %20, %27,  TOTALORDER : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xi1>
    %29 = call @_where_110(%28, %arg2, %5) : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    %30 = call @_where_110(%28, %5, %arg3) : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %29, %30 : tensor<4xi32>, tensor<4xi32>
  }
  func.func private @_where_110(%arg0: tensor<4xi1>, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>) -> tensor<4xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<4xi1>, tensor<4xi32>
    return %0 : tensor<4xi32>
  }
  func.func private @clip_112(%arg0: tensor<4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<4xi32> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<4xi32>
    %3 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<4xi32>
    return %5 : tensor<4xi32>
  }
  func.func private @_where_119(%arg0: tensor<4xi1>, %arg1: tensor<i64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<4xi1>, tensor<4xf64>
    return %2 : tensor<4xf64>
  }
  func.func private @_where_121(%arg0: tensor<4xi1>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<4xi1>, tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  func.func private @_where_124(%arg0: tensor<4xi1>, %arg1: tensor<f64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<4xi1>, tensor<4xf64>
    return %1 : tensor<4xf64>
  }
  func.func private @inner_128(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, -9.810000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg0 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.multiply %cst, %2 : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %5 = stablehlo.concatenate %4, %3, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %6 = stablehlo.add %arg1, %5 : tensor<6xf64>
    return %6 : tensor<6xf64>
  }
  func.func private @inner_133(%arg0: tensor<6xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.negate %0 : tensor<3xf64>
    %2 = call @norm(%1) : (tensor<3xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %4 = stablehlo.multiply %3, %1 : tensor<3xf64>
    %5 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.multiply %4, %5 : tensor<3xf64>
    return %6 : tensor<3xf64>
  }
  func.func private @inner_136(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[[0.000000e+00, -0.087155742747658165, 0.99619469809174543], [-0.065403129230143062, 0.086969135612238901, 0.99406176877383478], [0.000000e+00, 0.087155742747658165, 0.99619469809174543], [-0.065403129230143062, -0.086969135612238901, 0.99406176877383478]]> : tensor<4x3xf64>
    %cst_0 = stablehlo.constant dense<[[-0.20858424832311181, -0.25901062150385384, -0.022660493114391125], [0.19843360999226459, 0.25704989260274902, -0.0094332447193082886], [0.20858424832311181, -0.25901062150385384, 0.022660493114391125], [-0.19843360999226459, 0.25704989260274902, 0.0094332447193082886]]> : tensor<4x3xf64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %2 = stablehlo.multiply %cst, %1 : tensor<4x3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %3 = stablehlo.reduce(%2 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %6 = stablehlo.multiply %cst, %5 : tensor<4x3xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.reduce(%6 init: %cst_2) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %10 = stablehlo.multiply %cst_0, %9 : tensor<4x3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %11 = stablehlo.reduce(%10 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %13 = stablehlo.concatenate %12, %3, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %15 = stablehlo.concatenate %7, %14, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %16 = stablehlo.add %13, %15 : tensor<6xf64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %17 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %18 = stablehlo.concatenate %11, %17, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %19 = stablehlo.add %16, %18 : tensor<6xf64>
    return %19 : tensor<6xf64>
  }
  func.func private @inner_140(%arg0: tensor<6xf64>, %arg1: tensor<3xf64>, %arg2: tensor<7xf64>, %arg3: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.concatenate %0, %arg1, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<6xf64>
    %3 = stablehlo.slice %arg2 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %4 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %arg0 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %9 = stablehlo.slice %8 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
    %11 = stablehlo.multiply %5, %10 : tensor<f64>
    %12 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %13 = stablehlo.reshape %12 : (tensor<1xf64>) -> tensor<f64>
    %14 = stablehlo.slice %8 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.multiply %13, %15 : tensor<f64>
    %17 = stablehlo.add %11, %16 : tensor<f64>
    %18 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.slice %8 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.multiply %19, %21 : tensor<f64>
    %23 = stablehlo.add %17, %22 : tensor<f64>
    %24 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.slice %8 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
    %28 = stablehlo.multiply %25, %27 : tensor<f64>
    %29 = stablehlo.subtract %23, %28 : tensor<f64>
    %30 = stablehlo.reshape %29 : (tensor<f64>) -> tensor<1xf64>
    %31 = stablehlo.multiply %5, %27 : tensor<f64>
    %32 = stablehlo.multiply %13, %21 : tensor<f64>
    %33 = stablehlo.subtract %31, %32 : tensor<f64>
    %34 = stablehlo.multiply %19, %15 : tensor<f64>
    %35 = stablehlo.add %33, %34 : tensor<f64>
    %36 = stablehlo.multiply %25, %10 : tensor<f64>
    %37 = stablehlo.add %35, %36 : tensor<f64>
    %38 = stablehlo.reshape %37 : (tensor<f64>) -> tensor<1xf64>
    %39 = stablehlo.multiply %5, %21 : tensor<f64>
    %40 = stablehlo.multiply %13, %27 : tensor<f64>
    %41 = stablehlo.add %39, %40 : tensor<f64>
    %42 = stablehlo.multiply %19, %10 : tensor<f64>
    %43 = stablehlo.subtract %41, %42 : tensor<f64>
    %44 = stablehlo.multiply %25, %15 : tensor<f64>
    %45 = stablehlo.add %43, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %5, %15 : tensor<f64>
    %48 = stablehlo.multiply %13, %10 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %19, %27 : tensor<f64>
    %51 = stablehlo.subtract %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %25, %21 : tensor<f64>
    %53 = stablehlo.subtract %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.concatenate %30, %38, %46, %54, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %56 = stablehlo.slice %55 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %57 = stablehlo.reshape %56 : (tensor<1xf64>) -> tensor<f64>
    %58 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %59 = stablehlo.reshape %58 : (tensor<1xf64>) -> tensor<f64>
    %60 = stablehlo.negate %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %63 = stablehlo.reshape %62 : (tensor<1xf64>) -> tensor<f64>
    %64 = stablehlo.negate %63 : tensor<f64>
    %65 = stablehlo.reshape %64 : (tensor<f64>) -> tensor<1xf64>
    %66 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %67 = stablehlo.reshape %66 : (tensor<1xf64>) -> tensor<f64>
    %68 = stablehlo.negate %67 : tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %71 = stablehlo.reshape %70 : (tensor<1xf64>) -> tensor<f64>
    %72 = stablehlo.reshape %71 : (tensor<f64>) -> tensor<1xf64>
    %73 = stablehlo.concatenate %61, %65, %69, %72, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %74 = stablehlo.dot_general %3, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %76 = stablehlo.divide %73, %75 : tensor<4xf64>
    %77 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.multiply %57, %78 : tensor<f64>
    %80 = stablehlo.slice %55 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %81 = stablehlo.reshape %80 : (tensor<1xf64>) -> tensor<f64>
    %82 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.multiply %81, %83 : tensor<f64>
    %85 = stablehlo.add %79, %84 : tensor<f64>
    %86 = stablehlo.slice %55 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %89 = stablehlo.reshape %88 : (tensor<1xf64>) -> tensor<f64>
    %90 = stablehlo.multiply %87, %89 : tensor<f64>
    %91 = stablehlo.add %85, %90 : tensor<f64>
    %92 = stablehlo.slice %55 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %95 = stablehlo.reshape %94 : (tensor<1xf64>) -> tensor<f64>
    %96 = stablehlo.multiply %93, %95 : tensor<f64>
    %97 = stablehlo.subtract %91, %96 : tensor<f64>
    %98 = stablehlo.reshape %97 : (tensor<f64>) -> tensor<1xf64>
    %99 = stablehlo.multiply %57, %95 : tensor<f64>
    %100 = stablehlo.multiply %81, %89 : tensor<f64>
    %101 = stablehlo.subtract %99, %100 : tensor<f64>
    %102 = stablehlo.multiply %87, %83 : tensor<f64>
    %103 = stablehlo.add %101, %102 : tensor<f64>
    %104 = stablehlo.multiply %93, %78 : tensor<f64>
    %105 = stablehlo.add %103, %104 : tensor<f64>
    %106 = stablehlo.reshape %105 : (tensor<f64>) -> tensor<1xf64>
    %107 = stablehlo.multiply %57, %89 : tensor<f64>
    %108 = stablehlo.multiply %81, %95 : tensor<f64>
    %109 = stablehlo.add %107, %108 : tensor<f64>
    %110 = stablehlo.multiply %87, %78 : tensor<f64>
    %111 = stablehlo.subtract %109, %110 : tensor<f64>
    %112 = stablehlo.multiply %93, %83 : tensor<f64>
    %113 = stablehlo.add %111, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %57, %83 : tensor<f64>
    %116 = stablehlo.multiply %81, %78 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %87, %95 : tensor<f64>
    %119 = stablehlo.subtract %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %93, %89 : tensor<f64>
    %121 = stablehlo.subtract %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.concatenate %98, %106, %114, %122, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %124 = stablehlo.slice %123 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %125 = stablehlo.reshape %124 : (tensor<1xf64>) -> tensor<f64>
    %126 = stablehlo.reshape %125 : (tensor<f64>) -> tensor<1xf64>
    %127 = stablehlo.slice %123 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %128 = stablehlo.reshape %127 : (tensor<1xf64>) -> tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.slice %123 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %131 = stablehlo.reshape %130 : (tensor<1xf64>) -> tensor<f64>
    %132 = stablehlo.reshape %131 : (tensor<f64>) -> tensor<1xf64>
    %133 = stablehlo.concatenate %126, %129, %132, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %134 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %135 = stablehlo.reshape %134 : (tensor<1xf64>) -> tensor<f64>
    %136 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %137 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.concatenate %136, %137, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %139 = stablehlo.slice %138 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.multiply %135, %140 : tensor<f64>
    %142 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %143 = stablehlo.reshape %142 : (tensor<1xf64>) -> tensor<f64>
    %144 = stablehlo.slice %138 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.multiply %143, %145 : tensor<f64>
    %147 = stablehlo.add %141, %146 : tensor<f64>
    %148 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %149 = stablehlo.reshape %148 : (tensor<1xf64>) -> tensor<f64>
    %150 = stablehlo.slice %138 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
    %152 = stablehlo.multiply %149, %151 : tensor<f64>
    %153 = stablehlo.add %147, %152 : tensor<f64>
    %154 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %155 = stablehlo.reshape %154 : (tensor<1xf64>) -> tensor<f64>
    %156 = stablehlo.slice %138 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %157 = stablehlo.reshape %156 : (tensor<1xf64>) -> tensor<f64>
    %158 = stablehlo.multiply %155, %157 : tensor<f64>
    %159 = stablehlo.subtract %153, %158 : tensor<f64>
    %160 = stablehlo.reshape %159 : (tensor<f64>) -> tensor<1xf64>
    %161 = stablehlo.multiply %135, %157 : tensor<f64>
    %162 = stablehlo.multiply %143, %151 : tensor<f64>
    %163 = stablehlo.subtract %161, %162 : tensor<f64>
    %164 = stablehlo.multiply %149, %145 : tensor<f64>
    %165 = stablehlo.add %163, %164 : tensor<f64>
    %166 = stablehlo.multiply %155, %140 : tensor<f64>
    %167 = stablehlo.add %165, %166 : tensor<f64>
    %168 = stablehlo.reshape %167 : (tensor<f64>) -> tensor<1xf64>
    %169 = stablehlo.multiply %135, %151 : tensor<f64>
    %170 = stablehlo.multiply %143, %157 : tensor<f64>
    %171 = stablehlo.add %169, %170 : tensor<f64>
    %172 = stablehlo.multiply %149, %140 : tensor<f64>
    %173 = stablehlo.subtract %171, %172 : tensor<f64>
    %174 = stablehlo.multiply %155, %145 : tensor<f64>
    %175 = stablehlo.add %173, %174 : tensor<f64>
    %176 = stablehlo.reshape %175 : (tensor<f64>) -> tensor<1xf64>
    %177 = stablehlo.multiply %135, %145 : tensor<f64>
    %178 = stablehlo.multiply %143, %140 : tensor<f64>
    %179 = stablehlo.subtract %177, %178 : tensor<f64>
    %180 = stablehlo.multiply %149, %157 : tensor<f64>
    %181 = stablehlo.subtract %179, %180 : tensor<f64>
    %182 = stablehlo.multiply %155, %151 : tensor<f64>
    %183 = stablehlo.subtract %181, %182 : tensor<f64>
    %184 = stablehlo.reshape %183 : (tensor<f64>) -> tensor<1xf64>
    %185 = stablehlo.concatenate %160, %168, %176, %184, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %186 = stablehlo.slice %185 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %187 = stablehlo.reshape %186 : (tensor<1xf64>) -> tensor<f64>
    %188 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %189 = stablehlo.reshape %188 : (tensor<1xf64>) -> tensor<f64>
    %190 = stablehlo.negate %189 : tensor<f64>
    %191 = stablehlo.reshape %190 : (tensor<f64>) -> tensor<1xf64>
    %192 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %193 = stablehlo.reshape %192 : (tensor<1xf64>) -> tensor<f64>
    %194 = stablehlo.negate %193 : tensor<f64>
    %195 = stablehlo.reshape %194 : (tensor<f64>) -> tensor<1xf64>
    %196 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %197 = stablehlo.reshape %196 : (tensor<1xf64>) -> tensor<f64>
    %198 = stablehlo.negate %197 : tensor<f64>
    %199 = stablehlo.reshape %198 : (tensor<f64>) -> tensor<1xf64>
    %200 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %201 = stablehlo.reshape %200 : (tensor<1xf64>) -> tensor<f64>
    %202 = stablehlo.reshape %201 : (tensor<f64>) -> tensor<1xf64>
    %203 = stablehlo.concatenate %191, %195, %199, %202, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %204 = stablehlo.dot_general %3, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %205 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %206 = stablehlo.divide %203, %205 : tensor<4xf64>
    %207 = stablehlo.slice %206 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %208 = stablehlo.reshape %207 : (tensor<1xf64>) -> tensor<f64>
    %209 = stablehlo.multiply %187, %208 : tensor<f64>
    %210 = stablehlo.slice %185 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %211 = stablehlo.reshape %210 : (tensor<1xf64>) -> tensor<f64>
    %212 = stablehlo.slice %206 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %213 = stablehlo.reshape %212 : (tensor<1xf64>) -> tensor<f64>
    %214 = stablehlo.multiply %211, %213 : tensor<f64>
    %215 = stablehlo.add %209, %214 : tensor<f64>
    %216 = stablehlo.slice %185 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %217 = stablehlo.reshape %216 : (tensor<1xf64>) -> tensor<f64>
    %218 = stablehlo.slice %206 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %219 = stablehlo.reshape %218 : (tensor<1xf64>) -> tensor<f64>
    %220 = stablehlo.multiply %217, %219 : tensor<f64>
    %221 = stablehlo.add %215, %220 : tensor<f64>
    %222 = stablehlo.slice %185 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %223 = stablehlo.reshape %222 : (tensor<1xf64>) -> tensor<f64>
    %224 = stablehlo.slice %206 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %225 = stablehlo.reshape %224 : (tensor<1xf64>) -> tensor<f64>
    %226 = stablehlo.multiply %223, %225 : tensor<f64>
    %227 = stablehlo.subtract %221, %226 : tensor<f64>
    %228 = stablehlo.reshape %227 : (tensor<f64>) -> tensor<1xf64>
    %229 = stablehlo.multiply %187, %225 : tensor<f64>
    %230 = stablehlo.multiply %211, %219 : tensor<f64>
    %231 = stablehlo.subtract %229, %230 : tensor<f64>
    %232 = stablehlo.multiply %217, %213 : tensor<f64>
    %233 = stablehlo.add %231, %232 : tensor<f64>
    %234 = stablehlo.multiply %223, %208 : tensor<f64>
    %235 = stablehlo.add %233, %234 : tensor<f64>
    %236 = stablehlo.reshape %235 : (tensor<f64>) -> tensor<1xf64>
    %237 = stablehlo.multiply %187, %219 : tensor<f64>
    %238 = stablehlo.multiply %211, %225 : tensor<f64>
    %239 = stablehlo.add %237, %238 : tensor<f64>
    %240 = stablehlo.multiply %217, %208 : tensor<f64>
    %241 = stablehlo.subtract %239, %240 : tensor<f64>
    %242 = stablehlo.multiply %223, %213 : tensor<f64>
    %243 = stablehlo.add %241, %242 : tensor<f64>
    %244 = stablehlo.reshape %243 : (tensor<f64>) -> tensor<1xf64>
    %245 = stablehlo.multiply %187, %213 : tensor<f64>
    %246 = stablehlo.multiply %211, %208 : tensor<f64>
    %247 = stablehlo.subtract %245, %246 : tensor<f64>
    %248 = stablehlo.multiply %217, %225 : tensor<f64>
    %249 = stablehlo.subtract %247, %248 : tensor<f64>
    %250 = stablehlo.multiply %223, %219 : tensor<f64>
    %251 = stablehlo.subtract %249, %250 : tensor<f64>
    %252 = stablehlo.reshape %251 : (tensor<f64>) -> tensor<1xf64>
    %253 = stablehlo.concatenate %228, %236, %244, %252, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %254 = stablehlo.slice %253 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %255 = stablehlo.reshape %254 : (tensor<1xf64>) -> tensor<f64>
    %256 = stablehlo.reshape %255 : (tensor<f64>) -> tensor<1xf64>
    %257 = stablehlo.slice %253 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %258 = stablehlo.reshape %257 : (tensor<1xf64>) -> tensor<f64>
    %259 = stablehlo.reshape %258 : (tensor<f64>) -> tensor<1xf64>
    %260 = stablehlo.slice %253 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %261 = stablehlo.reshape %260 : (tensor<1xf64>) -> tensor<f64>
    %262 = stablehlo.reshape %261 : (tensor<f64>) -> tensor<1xf64>
    %263 = stablehlo.concatenate %256, %259, %262, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %264 = stablehlo.concatenate %133, %263, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %265 = stablehlo.add %2, %264 : tensor<6xf64>
    return %265 : tensor<6xf64>
  }
  func.func private @inner_146(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @inner_147(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<[1797259609, 2579123966]> : tensor<2xui32>
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %1 = call @_threefry_fold_in(%c, %0) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %2 = stablehlo.sqrt %cst : tensor<f64>
    %3 = call @_normal(%1) : (tensor<2xui32>) -> tensor<3xf64>
    %4 = stablehlo.convert %2 : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.multiply %5, %3 : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.multiply %6, %7 : tensor<3xf64>
    %9 = stablehlo.add %arg1, %8 : tensor<3xf64>
    return %9 : tensor<3xf64>
  }
  func.func private @_threefry_fold_in(%arg0: tensor<2xui32>, %arg1: tensor<ui32>) -> tensor<2xui32> {
    %c = stablehlo.constant dense<32> : tensor<ui32>
    %0 = stablehlo.shift_right_logical %arg1, %c : tensor<ui32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %c_0 = stablehlo.constant dense<4294967295> : tensor<ui32>
    %2 = stablehlo.and %arg1, %c_0 : tensor<ui32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %4 = stablehlo.concatenate %1, %3, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %5 = stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %6 = stablehlo.reshape %5 : (tensor<1xui32>) -> tensor<ui32>
    %7 = stablehlo.slice %arg0 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %8 = stablehlo.reshape %7 : (tensor<1xui32>) -> tensor<ui32>
    %9 = stablehlo.slice %4 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
    %10 = stablehlo.slice %4 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
    %11:2 = call @threefry2x32(%6, %8, %9, %10) : (tensor<ui32>, tensor<ui32>, tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>, tensor<1xui32>)
    %12 = stablehlo.concatenate %11#0, %11#1, dim = 0 : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    return %12 : tensor<2xui32>
  }
  func.func private @threefry2x32(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<1xui32>, %arg3: tensor<1xui32>) -> (tensor<1xui32>, tensor<1xui32>) {
    %c = stablehlo.constant dense<[13, 15, 26, 6]> : tensor<4xui32>
    %c_0 = stablehlo.constant dense<[17, 29, 16, 24]> : tensor<4xui32>
    %0 = stablehlo.xor %arg0, %arg1 : tensor<ui32>
    %c_1 = stablehlo.constant dense<466688986> : tensor<ui32>
    %1 = stablehlo.xor %0, %c_1 : tensor<ui32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %3 = stablehlo.add %arg2, %2 : tensor<1xui32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %5 = stablehlo.add %arg3, %4 : tensor<1xui32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %6:9 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c_2, %iterArg_5 = %3, %iterArg_6 = %5, %iterArg_7 = %arg1, %iterArg_8 = %1, %iterArg_9 = %arg0, %iterArg_10 = %c, %iterArg_11 = %c_0) : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    cond {
      %c_12 = stablehlo.constant dense<5> : tensor<i64>
      %7 = stablehlo.compare  LT, %iterArg, %c_12,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %7 : tensor<i1>
    } do {
      %7:8 = func.call @closed_call_158(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_12 = stablehlo.constant dense<1> : tensor<i64>
      %8 = stablehlo.add %iterArg, %c_12 : tensor<i64>
      stablehlo.return %8, %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6, %7#7 : tensor<i64>, tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    return %6#2, %6#3 : tensor<1xui32>, tensor<1xui32>
  }
  func.func private @closed_call_158(%arg0: tensor<i64>, %arg1: tensor<1xui32>, %arg2: tensor<1xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1 = stablehlo.slice %arg6 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
    %2 = stablehlo.reshape %1 : (tensor<1xui32>) -> tensor<ui32>
    %3 = stablehlo.add %arg1, %arg2 : tensor<1xui32>
    %4 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %5 = stablehlo.shift_left %arg2, %4 : tensor<1xui32>
    %c_0 = stablehlo.constant dense<32> : tensor<ui32>
    %6 = stablehlo.subtract %c_0, %2 : tensor<ui32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %8 = stablehlo.shift_right_logical %arg2, %7 : tensor<1xui32>
    %9 = stablehlo.or %5, %8 : tensor<1xui32>
    %10 = stablehlo.xor %3, %9 : tensor<1xui32>
    %11 = stablehlo.slice %arg6 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
    %12 = stablehlo.reshape %11 : (tensor<1xui32>) -> tensor<ui32>
    %13 = stablehlo.add %3, %10 : tensor<1xui32>
    %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %15 = stablehlo.shift_left %10, %14 : tensor<1xui32>
    %c_1 = stablehlo.constant dense<32> : tensor<ui32>
    %16 = stablehlo.subtract %c_1, %12 : tensor<ui32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %18 = stablehlo.shift_right_logical %10, %17 : tensor<1xui32>
    %19 = stablehlo.or %15, %18 : tensor<1xui32>
    %20 = stablehlo.xor %13, %19 : tensor<1xui32>
    %21 = stablehlo.slice %arg6 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
    %22 = stablehlo.reshape %21 : (tensor<1xui32>) -> tensor<ui32>
    %23 = stablehlo.add %13, %20 : tensor<1xui32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %25 = stablehlo.shift_left %20, %24 : tensor<1xui32>
    %c_2 = stablehlo.constant dense<32> : tensor<ui32>
    %26 = stablehlo.subtract %c_2, %22 : tensor<ui32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %28 = stablehlo.shift_right_logical %20, %27 : tensor<1xui32>
    %29 = stablehlo.or %25, %28 : tensor<1xui32>
    %30 = stablehlo.xor %23, %29 : tensor<1xui32>
    %31 = stablehlo.slice %arg6 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
    %32 = stablehlo.reshape %31 : (tensor<1xui32>) -> tensor<ui32>
    %33 = stablehlo.add %23, %30 : tensor<1xui32>
    %34 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %35 = stablehlo.shift_left %30, %34 : tensor<1xui32>
    %c_3 = stablehlo.constant dense<32> : tensor<ui32>
    %36 = stablehlo.subtract %c_3, %32 : tensor<ui32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %38 = stablehlo.shift_right_logical %30, %37 : tensor<1xui32>
    %39 = stablehlo.or %35, %38 : tensor<1xui32>
    %40 = stablehlo.xor %33, %39 : tensor<1xui32>
    %41 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %42 = stablehlo.add %33, %41 : tensor<1xui32>
    %43 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %44 = stablehlo.add %40, %43 : tensor<1xui32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %45 = stablehlo.add %arg0, %c_4 : tensor<i64>
    %46 = stablehlo.convert %45 : (tensor<i64>) -> tensor<ui32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<ui32>) -> tensor<1xui32>
    %48 = stablehlo.add %44, %47 : tensor<1xui32>
    return %0, %42, %48, %arg4, %arg5, %arg3, %arg7, %arg6 : tensor<i64>, tensor<1xui32>, tensor<1xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
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
    %13:2 = call @threefry2x32_169(%3, %5, %12, %11) : (tensor<ui32>, tensor<ui32>, tensor<3xui32>, tensor<3xui32>) -> (tensor<3xui32>, tensor<3xui32>)
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
  func.func private @threefry2x32_169(%arg0: tensor<ui32>, %arg1: tensor<ui32>, %arg2: tensor<3xui32>, %arg3: tensor<3xui32>) -> (tensor<3xui32>, tensor<3xui32>) {
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
      %7:8 = func.call @closed_call_173(%iterArg_4, %iterArg_5, %iterArg_6, %iterArg_7, %iterArg_8, %iterArg_9, %iterArg_10, %iterArg_11) : (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %c_12 = stablehlo.constant dense<1> : tensor<i64>
      %8 = stablehlo.add %iterArg, %c_12 : tensor<i64>
      stablehlo.return %8, %7#0, %7#1, %7#2, %7#3, %7#4, %7#5, %7#6, %7#7 : tensor<i64>, tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>
    }
    return %6#2, %6#3 : tensor<3xui32>, tensor<3xui32>
  }
  func.func private @closed_call_173(%arg0: tensor<i64>, %arg1: tensor<3xui32>, %arg2: tensor<3xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>) -> (tensor<i64>, tensor<3xui32>, tensor<3xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) {
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
  func.func private @inner_189(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<6xf64>, %arg3: tensor<4x3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %c = stablehlo.constant dense<[1797259609, 2579123966]> : tensor<2xui32>
    %cst = stablehlo.constant dense<[0.016209783477834406, 0.032419566955668812, 0.016209783477834406, -1.6089340340646645, 0.67377316797600217]> : tensor<5xf64>
    %0 = stablehlo.slice %arg2 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2 = stablehlo.slice %1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.negate %3 : tensor<f64>
    %5 = stablehlo.reshape %4 : (tensor<f64>) -> tensor<1xf64>
    %6 = stablehlo.slice %1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.reshape %15 : (tensor<f64>) -> tensor<1xf64>
    %17 = stablehlo.concatenate %5, %9, %13, %16, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %18 = stablehlo.dot_general %1, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %20 = stablehlo.divide %17, %19 : tensor<4xf64>
    %21 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %23 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.concatenate %0, %23, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %25 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %27 = stablehlo.multiply %22, %26 : tensor<f64>
    %28 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
    %30 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.multiply %29, %31 : tensor<f64>
    %33 = stablehlo.add %27, %32 : tensor<f64>
    %34 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %37 = stablehlo.reshape %36 : (tensor<1xf64>) -> tensor<f64>
    %38 = stablehlo.multiply %35, %37 : tensor<f64>
    %39 = stablehlo.add %33, %38 : tensor<f64>
    %40 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %43 = stablehlo.reshape %42 : (tensor<1xf64>) -> tensor<f64>
    %44 = stablehlo.multiply %41, %43 : tensor<f64>
    %45 = stablehlo.subtract %39, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %22, %43 : tensor<f64>
    %48 = stablehlo.multiply %29, %37 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %35, %31 : tensor<f64>
    %51 = stablehlo.add %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %41, %26 : tensor<f64>
    %53 = stablehlo.add %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.multiply %22, %37 : tensor<f64>
    %56 = stablehlo.multiply %29, %43 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.multiply %35, %26 : tensor<f64>
    %59 = stablehlo.subtract %57, %58 : tensor<f64>
    %60 = stablehlo.multiply %41, %31 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.reshape %61 : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.multiply %22, %31 : tensor<f64>
    %64 = stablehlo.multiply %29, %26 : tensor<f64>
    %65 = stablehlo.subtract %63, %64 : tensor<f64>
    %66 = stablehlo.multiply %35, %43 : tensor<f64>
    %67 = stablehlo.subtract %65, %66 : tensor<f64>
    %68 = stablehlo.multiply %41, %37 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.reshape %69 : (tensor<f64>) -> tensor<1xf64>
    %71 = stablehlo.concatenate %46, %54, %62, %70, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %72 = stablehlo.slice %71 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1xf64>) -> tensor<f64>
    %74 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.negate %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.reshape %87 : (tensor<f64>) -> tensor<1xf64>
    %89 = stablehlo.concatenate %77, %81, %85, %88, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %90 = stablehlo.dot_general %20, %20, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %92 = stablehlo.divide %89, %91 : tensor<4xf64>
    %93 = stablehlo.slice %92 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %94 = stablehlo.reshape %93 : (tensor<1xf64>) -> tensor<f64>
    %95 = stablehlo.multiply %73, %94 : tensor<f64>
    %96 = stablehlo.slice %71 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.slice %92 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.multiply %97, %99 : tensor<f64>
    %101 = stablehlo.add %95, %100 : tensor<f64>
    %102 = stablehlo.slice %71 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.slice %92 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %105 = stablehlo.reshape %104 : (tensor<1xf64>) -> tensor<f64>
    %106 = stablehlo.multiply %103, %105 : tensor<f64>
    %107 = stablehlo.add %101, %106 : tensor<f64>
    %108 = stablehlo.slice %71 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.slice %92 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %111 = stablehlo.reshape %110 : (tensor<1xf64>) -> tensor<f64>
    %112 = stablehlo.multiply %109, %111 : tensor<f64>
    %113 = stablehlo.subtract %107, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %73, %111 : tensor<f64>
    %116 = stablehlo.multiply %97, %105 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %103, %99 : tensor<f64>
    %119 = stablehlo.add %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %109, %94 : tensor<f64>
    %121 = stablehlo.add %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.multiply %73, %105 : tensor<f64>
    %124 = stablehlo.multiply %97, %111 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.multiply %103, %94 : tensor<f64>
    %127 = stablehlo.subtract %125, %126 : tensor<f64>
    %128 = stablehlo.multiply %109, %99 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.reshape %129 : (tensor<f64>) -> tensor<1xf64>
    %131 = stablehlo.multiply %73, %99 : tensor<f64>
    %132 = stablehlo.multiply %97, %94 : tensor<f64>
    %133 = stablehlo.subtract %131, %132 : tensor<f64>
    %134 = stablehlo.multiply %103, %111 : tensor<f64>
    %135 = stablehlo.subtract %133, %134 : tensor<f64>
    %136 = stablehlo.multiply %109, %105 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.reshape %137 : (tensor<f64>) -> tensor<1xf64>
    %139 = stablehlo.concatenate %114, %122, %130, %138, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %140 = stablehlo.slice %139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %141 = stablehlo.reshape %140 : (tensor<1xf64>) -> tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.slice %139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %144 = stablehlo.reshape %143 : (tensor<1xf64>) -> tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.slice %139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %147 = stablehlo.reshape %146 : (tensor<1xf64>) -> tensor<f64>
    %148 = stablehlo.reshape %147 : (tensor<f64>) -> tensor<1xf64>
    %149 = stablehlo.concatenate %142, %145, %148, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %150 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %151 = call @_threefry_fold_in(%c, %150) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_1 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %152 = stablehlo.sqrt %cst_1 : tensor<f64>
    %153 = call @_normal(%151) : (tensor<2xui32>) -> tensor<3xf64>
    %154 = stablehlo.convert %152 : tensor<f64>
    %155 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %156 = stablehlo.multiply %155, %153 : tensor<3xf64>
    %157 = stablehlo.add %149, %156 : tensor<3xf64>
    %158 = stablehlo.add %157, %arg4 : tensor<3xf64>
    %159 = stablehlo.slice %cst [0:1] : (tensor<5xf64>) -> tensor<1xf64>
    %160 = stablehlo.reshape %159 : (tensor<1xf64>) -> tensor<f64>
    %161 = stablehlo.slice %cst [1:2] : (tensor<5xf64>) -> tensor<1xf64>
    %162 = stablehlo.reshape %161 : (tensor<1xf64>) -> tensor<f64>
    %163 = stablehlo.slice %cst [2:3] : (tensor<5xf64>) -> tensor<1xf64>
    %164 = stablehlo.reshape %163 : (tensor<1xf64>) -> tensor<f64>
    %165 = stablehlo.slice %cst [3:4] : (tensor<5xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %cst [4:5] : (tensor<5xf64>) -> tensor<1xf64>
    %168 = stablehlo.reshape %167 : (tensor<1xf64>) -> tensor<f64>
    %169 = stablehlo.slice %arg3 [0:1, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %170 = stablehlo.reshape %169 : (tensor<1x3xf64>) -> tensor<3xf64>
    %171 = stablehlo.slice %arg3 [1:2, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %172 = stablehlo.reshape %171 : (tensor<1x3xf64>) -> tensor<3xf64>
    %173 = stablehlo.slice %arg3 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %174 = stablehlo.reshape %173 : (tensor<1x3xf64>) -> tensor<3xf64>
    %175 = stablehlo.slice %arg3 [3:4, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %176 = stablehlo.reshape %175 : (tensor<1x3xf64>) -> tensor<3xf64>
    %177 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %178 = stablehlo.multiply %177, %158 : tensor<3xf64>
    %179 = stablehlo.broadcast_in_dim %162, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %180 = stablehlo.multiply %179, %170 : tensor<3xf64>
    %181 = stablehlo.add %178, %180 : tensor<3xf64>
    %182 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %183 = stablehlo.multiply %182, %172 : tensor<3xf64>
    %184 = stablehlo.add %181, %183 : tensor<3xf64>
    %185 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %186 = stablehlo.multiply %185, %174 : tensor<3xf64>
    %187 = stablehlo.subtract %184, %186 : tensor<3xf64>
    %188 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %189 = stablehlo.multiply %188, %176 : tensor<3xf64>
    %190 = stablehlo.subtract %187, %189 : tensor<3xf64>
    %191 = stablehlo.broadcast_in_dim %158, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %192 = stablehlo.broadcast_in_dim %170, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %193 = stablehlo.broadcast_in_dim %190, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %194 = stablehlo.broadcast_in_dim %174, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %195 = stablehlo.concatenate %191, %192, %193, %194, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<4x3xf64>
    %196 = stablehlo.slice %195 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %197 = stablehlo.reshape %196 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %195, %197 : tensor<4x3xf64>, tensor<3xf64>
  }
  func.func private @inner_201(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<6xf64>, %arg3: tensor<4x3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %c = stablehlo.constant dense<[0, 0, 1]> : tensor<3xi64>
    %c_0 = stablehlo.constant dense<[928981903, 3453687069]> : tensor<2xui32>
    %cst = stablehlo.constant dense<[0.0044300075115303239, 0.0088600150230606477, 0.0044300075115303239, -1.8030932880476023, 0.82081331809372371]> : tensor<5xf64>
    %0 = stablehlo.slice %arg2 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<9.810000e+00> : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2 = stablehlo.divide %0, %1 : tensor<3xf64>
    %3 = stablehlo.convert %c : (tensor<3xi64>) -> tensor<3xf64>
    %4 = stablehlo.add %2, %3 : tensor<3xf64>
    %5 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %6 = stablehlo.slice %5 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %5 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %5 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.negate %15 : tensor<f64>
    %17 = stablehlo.reshape %16 : (tensor<f64>) -> tensor<1xf64>
    %18 = stablehlo.slice %5 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.reshape %19 : (tensor<f64>) -> tensor<1xf64>
    %21 = stablehlo.concatenate %9, %13, %17, %20, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %22 = stablehlo.dot_general %5, %5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %24 = stablehlo.divide %21, %23 : tensor<4xf64>
    %25 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %27 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %28 = stablehlo.concatenate %4, %27, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %29 = stablehlo.slice %28 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.multiply %26, %30 : tensor<f64>
    %32 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %33 = stablehlo.reshape %32 : (tensor<1xf64>) -> tensor<f64>
    %34 = stablehlo.slice %28 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.multiply %33, %35 : tensor<f64>
    %37 = stablehlo.add %31, %36 : tensor<f64>
    %38 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %39 = stablehlo.reshape %38 : (tensor<1xf64>) -> tensor<f64>
    %40 = stablehlo.slice %28 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.multiply %39, %41 : tensor<f64>
    %43 = stablehlo.add %37, %42 : tensor<f64>
    %44 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %45 = stablehlo.reshape %44 : (tensor<1xf64>) -> tensor<f64>
    %46 = stablehlo.slice %28 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %47 = stablehlo.reshape %46 : (tensor<1xf64>) -> tensor<f64>
    %48 = stablehlo.multiply %45, %47 : tensor<f64>
    %49 = stablehlo.subtract %43, %48 : tensor<f64>
    %50 = stablehlo.reshape %49 : (tensor<f64>) -> tensor<1xf64>
    %51 = stablehlo.multiply %26, %47 : tensor<f64>
    %52 = stablehlo.multiply %33, %41 : tensor<f64>
    %53 = stablehlo.subtract %51, %52 : tensor<f64>
    %54 = stablehlo.multiply %39, %35 : tensor<f64>
    %55 = stablehlo.add %53, %54 : tensor<f64>
    %56 = stablehlo.multiply %45, %30 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.reshape %57 : (tensor<f64>) -> tensor<1xf64>
    %59 = stablehlo.multiply %26, %41 : tensor<f64>
    %60 = stablehlo.multiply %33, %47 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.multiply %39, %30 : tensor<f64>
    %63 = stablehlo.subtract %61, %62 : tensor<f64>
    %64 = stablehlo.multiply %45, %35 : tensor<f64>
    %65 = stablehlo.add %63, %64 : tensor<f64>
    %66 = stablehlo.reshape %65 : (tensor<f64>) -> tensor<1xf64>
    %67 = stablehlo.multiply %26, %35 : tensor<f64>
    %68 = stablehlo.multiply %33, %30 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.multiply %39, %47 : tensor<f64>
    %71 = stablehlo.subtract %69, %70 : tensor<f64>
    %72 = stablehlo.multiply %45, %41 : tensor<f64>
    %73 = stablehlo.subtract %71, %72 : tensor<f64>
    %74 = stablehlo.reshape %73 : (tensor<f64>) -> tensor<1xf64>
    %75 = stablehlo.concatenate %50, %58, %66, %74, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %76 = stablehlo.slice %75 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %77 = stablehlo.reshape %76 : (tensor<1xf64>) -> tensor<f64>
    %78 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.negate %87 : tensor<f64>
    %89 = stablehlo.reshape %88 : (tensor<f64>) -> tensor<1xf64>
    %90 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %91 = stablehlo.reshape %90 : (tensor<1xf64>) -> tensor<f64>
    %92 = stablehlo.reshape %91 : (tensor<f64>) -> tensor<1xf64>
    %93 = stablehlo.concatenate %81, %85, %89, %92, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %94 = stablehlo.dot_general %24, %24, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %96 = stablehlo.divide %93, %95 : tensor<4xf64>
    %97 = stablehlo.slice %96 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.multiply %77, %98 : tensor<f64>
    %100 = stablehlo.slice %75 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %101 = stablehlo.reshape %100 : (tensor<1xf64>) -> tensor<f64>
    %102 = stablehlo.slice %96 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.multiply %101, %103 : tensor<f64>
    %105 = stablehlo.add %99, %104 : tensor<f64>
    %106 = stablehlo.slice %75 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %107 = stablehlo.reshape %106 : (tensor<1xf64>) -> tensor<f64>
    %108 = stablehlo.slice %96 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.multiply %107, %109 : tensor<f64>
    %111 = stablehlo.add %105, %110 : tensor<f64>
    %112 = stablehlo.slice %75 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %113 = stablehlo.reshape %112 : (tensor<1xf64>) -> tensor<f64>
    %114 = stablehlo.slice %96 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %115 = stablehlo.reshape %114 : (tensor<1xf64>) -> tensor<f64>
    %116 = stablehlo.multiply %113, %115 : tensor<f64>
    %117 = stablehlo.subtract %111, %116 : tensor<f64>
    %118 = stablehlo.reshape %117 : (tensor<f64>) -> tensor<1xf64>
    %119 = stablehlo.multiply %77, %115 : tensor<f64>
    %120 = stablehlo.multiply %101, %109 : tensor<f64>
    %121 = stablehlo.subtract %119, %120 : tensor<f64>
    %122 = stablehlo.multiply %107, %103 : tensor<f64>
    %123 = stablehlo.add %121, %122 : tensor<f64>
    %124 = stablehlo.multiply %113, %98 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.reshape %125 : (tensor<f64>) -> tensor<1xf64>
    %127 = stablehlo.multiply %77, %109 : tensor<f64>
    %128 = stablehlo.multiply %101, %115 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.multiply %107, %98 : tensor<f64>
    %131 = stablehlo.subtract %129, %130 : tensor<f64>
    %132 = stablehlo.multiply %113, %103 : tensor<f64>
    %133 = stablehlo.add %131, %132 : tensor<f64>
    %134 = stablehlo.reshape %133 : (tensor<f64>) -> tensor<1xf64>
    %135 = stablehlo.multiply %77, %103 : tensor<f64>
    %136 = stablehlo.multiply %101, %98 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.multiply %107, %115 : tensor<f64>
    %139 = stablehlo.subtract %137, %138 : tensor<f64>
    %140 = stablehlo.multiply %113, %109 : tensor<f64>
    %141 = stablehlo.subtract %139, %140 : tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.concatenate %118, %126, %134, %142, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %144 = stablehlo.slice %143 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.reshape %145 : (tensor<f64>) -> tensor<1xf64>
    %147 = stablehlo.slice %143 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %148 = stablehlo.reshape %147 : (tensor<1xf64>) -> tensor<f64>
    %149 = stablehlo.reshape %148 : (tensor<f64>) -> tensor<1xf64>
    %150 = stablehlo.slice %143 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
    %152 = stablehlo.reshape %151 : (tensor<f64>) -> tensor<1xf64>
    %153 = stablehlo.concatenate %146, %149, %152, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %154 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %155 = call @_threefry_fold_in(%c_0, %154) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_3 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %156 = stablehlo.sqrt %cst_3 : tensor<f64>
    %157 = call @_normal(%155) : (tensor<2xui32>) -> tensor<3xf64>
    %158 = stablehlo.convert %156 : tensor<f64>
    %159 = stablehlo.broadcast_in_dim %158, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %160 = stablehlo.multiply %159, %157 : tensor<3xf64>
    %161 = stablehlo.add %153, %160 : tensor<3xf64>
    %162 = stablehlo.add %161, %arg4 : tensor<3xf64>
    %163 = stablehlo.slice %cst [0:1] : (tensor<5xf64>) -> tensor<1xf64>
    %164 = stablehlo.reshape %163 : (tensor<1xf64>) -> tensor<f64>
    %165 = stablehlo.slice %cst [1:2] : (tensor<5xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %cst [2:3] : (tensor<5xf64>) -> tensor<1xf64>
    %168 = stablehlo.reshape %167 : (tensor<1xf64>) -> tensor<f64>
    %169 = stablehlo.slice %cst [3:4] : (tensor<5xf64>) -> tensor<1xf64>
    %170 = stablehlo.reshape %169 : (tensor<1xf64>) -> tensor<f64>
    %171 = stablehlo.slice %cst [4:5] : (tensor<5xf64>) -> tensor<1xf64>
    %172 = stablehlo.reshape %171 : (tensor<1xf64>) -> tensor<f64>
    %173 = stablehlo.slice %arg3 [0:1, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %174 = stablehlo.reshape %173 : (tensor<1x3xf64>) -> tensor<3xf64>
    %175 = stablehlo.slice %arg3 [1:2, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %176 = stablehlo.reshape %175 : (tensor<1x3xf64>) -> tensor<3xf64>
    %177 = stablehlo.slice %arg3 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %178 = stablehlo.reshape %177 : (tensor<1x3xf64>) -> tensor<3xf64>
    %179 = stablehlo.slice %arg3 [3:4, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %180 = stablehlo.reshape %179 : (tensor<1x3xf64>) -> tensor<3xf64>
    %181 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %182 = stablehlo.multiply %181, %162 : tensor<3xf64>
    %183 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %184 = stablehlo.multiply %183, %174 : tensor<3xf64>
    %185 = stablehlo.add %182, %184 : tensor<3xf64>
    %186 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %187 = stablehlo.multiply %186, %176 : tensor<3xf64>
    %188 = stablehlo.add %185, %187 : tensor<3xf64>
    %189 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %190 = stablehlo.multiply %189, %178 : tensor<3xf64>
    %191 = stablehlo.subtract %188, %190 : tensor<3xf64>
    %192 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %193 = stablehlo.multiply %192, %180 : tensor<3xf64>
    %194 = stablehlo.subtract %191, %193 : tensor<3xf64>
    %195 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %196 = stablehlo.broadcast_in_dim %174, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %197 = stablehlo.broadcast_in_dim %194, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %198 = stablehlo.broadcast_in_dim %178, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %199 = stablehlo.concatenate %195, %196, %197, %198, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<4x3xf64>
    %200 = stablehlo.slice %199 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %201 = stablehlo.reshape %200 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %199, %201 : tensor<4x3xf64>, tensor<3xf64>
  }
  func.func private @inner_203(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = call @norm(%arg0) : (tensor<3xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.subtract %0, %cst : tensor<f64>
    %2 = stablehlo.abs %1 : tensor<f64>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %3 = stablehlo.divide %2, %cst_0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %4 = call @clip(%3, %cst_1, %cst_2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.subtract %cst_3, %4 : tensor<f64>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %6 = stablehlo.multiply %cst_4, %5 : tensor<f64>
    %7 = call @norm(%arg1) : (tensor<3xf64>) -> tensor<f64>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %8 = stablehlo.divide %7, %cst_5 : tensor<f64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %9 = call @clip(%8, %cst_6, %cst_7) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %10 = stablehlo.subtract %cst_8, %9 : tensor<f64>
    %11 = stablehlo.multiply %6, %10 : tensor<f64>
    return %11 : tensor<f64>
  }
  func.func private @inner_204(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<3xf64>
    %c = stablehlo.constant dense<[4146024105, 2718843009]> : tensor<2xui32>
    %0 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
    %3 = stablehlo.negate %2 : tensor<f64>
    %4 = stablehlo.reshape %3 : (tensor<f64>) -> tensor<1xf64>
    %5 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %7 = stablehlo.negate %6 : tensor<f64>
    %8 = stablehlo.reshape %7 : (tensor<f64>) -> tensor<1xf64>
    %9 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
    %11 = stablehlo.negate %10 : tensor<f64>
    %12 = stablehlo.reshape %11 : (tensor<f64>) -> tensor<1xf64>
    %13 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %14 = stablehlo.reshape %13 : (tensor<1xf64>) -> tensor<f64>
    %15 = stablehlo.reshape %14 : (tensor<f64>) -> tensor<1xf64>
    %16 = stablehlo.concatenate %4, %8, %12, %15, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %17 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %19 = stablehlo.divide %16, %18 : tensor<4xf64>
    %20 = stablehlo.slice %19 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %23 = stablehlo.concatenate %cst, %22, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %24 = stablehlo.slice %23 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.multiply %21, %25 : tensor<f64>
    %27 = stablehlo.slice %19 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %28 = stablehlo.reshape %27 : (tensor<1xf64>) -> tensor<f64>
    %29 = stablehlo.slice %23 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.multiply %28, %30 : tensor<f64>
    %32 = stablehlo.add %26, %31 : tensor<f64>
    %33 = stablehlo.slice %19 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %34 = stablehlo.reshape %33 : (tensor<1xf64>) -> tensor<f64>
    %35 = stablehlo.slice %23 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1xf64>) -> tensor<f64>
    %37 = stablehlo.multiply %34, %36 : tensor<f64>
    %38 = stablehlo.add %32, %37 : tensor<f64>
    %39 = stablehlo.slice %19 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %40 = stablehlo.reshape %39 : (tensor<1xf64>) -> tensor<f64>
    %41 = stablehlo.slice %23 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.multiply %40, %42 : tensor<f64>
    %44 = stablehlo.subtract %38, %43 : tensor<f64>
    %45 = stablehlo.reshape %44 : (tensor<f64>) -> tensor<1xf64>
    %46 = stablehlo.multiply %21, %42 : tensor<f64>
    %47 = stablehlo.multiply %28, %36 : tensor<f64>
    %48 = stablehlo.subtract %46, %47 : tensor<f64>
    %49 = stablehlo.multiply %34, %30 : tensor<f64>
    %50 = stablehlo.add %48, %49 : tensor<f64>
    %51 = stablehlo.multiply %40, %25 : tensor<f64>
    %52 = stablehlo.add %50, %51 : tensor<f64>
    %53 = stablehlo.reshape %52 : (tensor<f64>) -> tensor<1xf64>
    %54 = stablehlo.multiply %21, %36 : tensor<f64>
    %55 = stablehlo.multiply %28, %42 : tensor<f64>
    %56 = stablehlo.add %54, %55 : tensor<f64>
    %57 = stablehlo.multiply %34, %25 : tensor<f64>
    %58 = stablehlo.subtract %56, %57 : tensor<f64>
    %59 = stablehlo.multiply %40, %30 : tensor<f64>
    %60 = stablehlo.add %58, %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.multiply %21, %30 : tensor<f64>
    %63 = stablehlo.multiply %28, %25 : tensor<f64>
    %64 = stablehlo.subtract %62, %63 : tensor<f64>
    %65 = stablehlo.multiply %34, %42 : tensor<f64>
    %66 = stablehlo.subtract %64, %65 : tensor<f64>
    %67 = stablehlo.multiply %40, %36 : tensor<f64>
    %68 = stablehlo.subtract %66, %67 : tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.concatenate %45, %53, %61, %69, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %71 = stablehlo.slice %70 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %72 = stablehlo.reshape %71 : (tensor<1xf64>) -> tensor<f64>
    %73 = stablehlo.slice %19 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.negate %74 : tensor<f64>
    %76 = stablehlo.reshape %75 : (tensor<f64>) -> tensor<1xf64>
    %77 = stablehlo.slice %19 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.negate %78 : tensor<f64>
    %80 = stablehlo.reshape %79 : (tensor<f64>) -> tensor<1xf64>
    %81 = stablehlo.slice %19 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %82 = stablehlo.reshape %81 : (tensor<1xf64>) -> tensor<f64>
    %83 = stablehlo.negate %82 : tensor<f64>
    %84 = stablehlo.reshape %83 : (tensor<f64>) -> tensor<1xf64>
    %85 = stablehlo.slice %19 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %86 = stablehlo.reshape %85 : (tensor<1xf64>) -> tensor<f64>
    %87 = stablehlo.reshape %86 : (tensor<f64>) -> tensor<1xf64>
    %88 = stablehlo.concatenate %76, %80, %84, %87, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %89 = stablehlo.dot_general %19, %19, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %91 = stablehlo.divide %88, %90 : tensor<4xf64>
    %92 = stablehlo.slice %91 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.multiply %72, %93 : tensor<f64>
    %95 = stablehlo.slice %70 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %96 = stablehlo.reshape %95 : (tensor<1xf64>) -> tensor<f64>
    %97 = stablehlo.slice %91 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.multiply %96, %98 : tensor<f64>
    %100 = stablehlo.add %94, %99 : tensor<f64>
    %101 = stablehlo.slice %70 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %102 = stablehlo.reshape %101 : (tensor<1xf64>) -> tensor<f64>
    %103 = stablehlo.slice %91 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
    %105 = stablehlo.multiply %102, %104 : tensor<f64>
    %106 = stablehlo.add %100, %105 : tensor<f64>
    %107 = stablehlo.slice %70 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %108 = stablehlo.reshape %107 : (tensor<1xf64>) -> tensor<f64>
    %109 = stablehlo.slice %91 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %110 = stablehlo.reshape %109 : (tensor<1xf64>) -> tensor<f64>
    %111 = stablehlo.multiply %108, %110 : tensor<f64>
    %112 = stablehlo.subtract %106, %111 : tensor<f64>
    %113 = stablehlo.reshape %112 : (tensor<f64>) -> tensor<1xf64>
    %114 = stablehlo.multiply %72, %110 : tensor<f64>
    %115 = stablehlo.multiply %96, %104 : tensor<f64>
    %116 = stablehlo.subtract %114, %115 : tensor<f64>
    %117 = stablehlo.multiply %102, %98 : tensor<f64>
    %118 = stablehlo.add %116, %117 : tensor<f64>
    %119 = stablehlo.multiply %108, %93 : tensor<f64>
    %120 = stablehlo.add %118, %119 : tensor<f64>
    %121 = stablehlo.reshape %120 : (tensor<f64>) -> tensor<1xf64>
    %122 = stablehlo.multiply %72, %104 : tensor<f64>
    %123 = stablehlo.multiply %96, %110 : tensor<f64>
    %124 = stablehlo.add %122, %123 : tensor<f64>
    %125 = stablehlo.multiply %102, %93 : tensor<f64>
    %126 = stablehlo.subtract %124, %125 : tensor<f64>
    %127 = stablehlo.multiply %108, %98 : tensor<f64>
    %128 = stablehlo.add %126, %127 : tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.multiply %72, %98 : tensor<f64>
    %131 = stablehlo.multiply %96, %93 : tensor<f64>
    %132 = stablehlo.subtract %130, %131 : tensor<f64>
    %133 = stablehlo.multiply %102, %110 : tensor<f64>
    %134 = stablehlo.subtract %132, %133 : tensor<f64>
    %135 = stablehlo.multiply %108, %104 : tensor<f64>
    %136 = stablehlo.subtract %134, %135 : tensor<f64>
    %137 = stablehlo.reshape %136 : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.concatenate %113, %121, %129, %137, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %139 = stablehlo.slice %138 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.reshape %140 : (tensor<f64>) -> tensor<1xf64>
    %142 = stablehlo.slice %138 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %143 = stablehlo.reshape %142 : (tensor<1xf64>) -> tensor<f64>
    %144 = stablehlo.reshape %143 : (tensor<f64>) -> tensor<1xf64>
    %145 = stablehlo.slice %138 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %146 = stablehlo.reshape %145 : (tensor<1xf64>) -> tensor<f64>
    %147 = stablehlo.reshape %146 : (tensor<f64>) -> tensor<1xf64>
    %148 = stablehlo.concatenate %141, %144, %147, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %149 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %150 = call @_threefry_fold_in(%c, %149) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_1 = stablehlo.constant dense<1.000000e-04> : tensor<f64>
    %151 = stablehlo.sqrt %cst_1 : tensor<f64>
    %152 = call @_normal(%150) : (tensor<2xui32>) -> tensor<3xf64>
    %153 = stablehlo.convert %151 : tensor<f64>
    %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %155 = stablehlo.multiply %154, %152 : tensor<3xf64>
    %156 = stablehlo.add %148, %155 : tensor<3xf64>
    %157 = stablehlo.add %156, %arg2 : tensor<3xf64>
    %c_2 = stablehlo.constant dense<9> : tensor<i64>
    %158 = call @remainder_205(%arg0, %c_2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %159 = stablehlo.compare  EQ, %158, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %160 = stablehlo.convert %159 : (tensor<i1>) -> tensor<i32>
    %161 = "stablehlo.case"(%160) ({
      stablehlo.return %arg3 : tensor<3xf64>
    }, {
      stablehlo.return %157 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    return %161 : tensor<3xf64>
  }
  func.func private @remainder_205(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.convert %arg1 : tensor<i64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare  EQ, %0, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %2 = call @_where_208(%1, %c_0, %0) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
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
  func.func private @_where_208(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @inner_214(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg1 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2 = stablehlo.slice %1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.negate %3 : tensor<f64>
    %5 = stablehlo.reshape %4 : (tensor<f64>) -> tensor<1xf64>
    %6 = stablehlo.slice %1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.reshape %15 : (tensor<f64>) -> tensor<1xf64>
    %17 = stablehlo.concatenate %5, %9, %13, %16, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %18 = stablehlo.dot_general %1, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %20 = stablehlo.divide %17, %19 : tensor<4xf64>
    %21 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %23 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.concatenate %0, %23, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %25 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %27 = stablehlo.multiply %22, %26 : tensor<f64>
    %28 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
    %30 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.multiply %29, %31 : tensor<f64>
    %33 = stablehlo.add %27, %32 : tensor<f64>
    %34 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %37 = stablehlo.reshape %36 : (tensor<1xf64>) -> tensor<f64>
    %38 = stablehlo.multiply %35, %37 : tensor<f64>
    %39 = stablehlo.add %33, %38 : tensor<f64>
    %40 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %43 = stablehlo.reshape %42 : (tensor<1xf64>) -> tensor<f64>
    %44 = stablehlo.multiply %41, %43 : tensor<f64>
    %45 = stablehlo.subtract %39, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %22, %43 : tensor<f64>
    %48 = stablehlo.multiply %29, %37 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %35, %31 : tensor<f64>
    %51 = stablehlo.add %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %41, %26 : tensor<f64>
    %53 = stablehlo.add %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.multiply %22, %37 : tensor<f64>
    %56 = stablehlo.multiply %29, %43 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.multiply %35, %26 : tensor<f64>
    %59 = stablehlo.subtract %57, %58 : tensor<f64>
    %60 = stablehlo.multiply %41, %31 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.reshape %61 : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.multiply %22, %31 : tensor<f64>
    %64 = stablehlo.multiply %29, %26 : tensor<f64>
    %65 = stablehlo.subtract %63, %64 : tensor<f64>
    %66 = stablehlo.multiply %35, %43 : tensor<f64>
    %67 = stablehlo.subtract %65, %66 : tensor<f64>
    %68 = stablehlo.multiply %41, %37 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.reshape %69 : (tensor<f64>) -> tensor<1xf64>
    %71 = stablehlo.concatenate %46, %54, %62, %70, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %72 = stablehlo.slice %71 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1xf64>) -> tensor<f64>
    %74 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.negate %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.reshape %87 : (tensor<f64>) -> tensor<1xf64>
    %89 = stablehlo.concatenate %77, %81, %85, %88, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %90 = stablehlo.dot_general %20, %20, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %92 = stablehlo.divide %89, %91 : tensor<4xf64>
    %93 = stablehlo.slice %92 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %94 = stablehlo.reshape %93 : (tensor<1xf64>) -> tensor<f64>
    %95 = stablehlo.multiply %73, %94 : tensor<f64>
    %96 = stablehlo.slice %71 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.slice %92 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.multiply %97, %99 : tensor<f64>
    %101 = stablehlo.add %95, %100 : tensor<f64>
    %102 = stablehlo.slice %71 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.slice %92 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %105 = stablehlo.reshape %104 : (tensor<1xf64>) -> tensor<f64>
    %106 = stablehlo.multiply %103, %105 : tensor<f64>
    %107 = stablehlo.add %101, %106 : tensor<f64>
    %108 = stablehlo.slice %71 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.slice %92 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %111 = stablehlo.reshape %110 : (tensor<1xf64>) -> tensor<f64>
    %112 = stablehlo.multiply %109, %111 : tensor<f64>
    %113 = stablehlo.subtract %107, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %73, %111 : tensor<f64>
    %116 = stablehlo.multiply %97, %105 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %103, %99 : tensor<f64>
    %119 = stablehlo.add %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %109, %94 : tensor<f64>
    %121 = stablehlo.add %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.multiply %73, %105 : tensor<f64>
    %124 = stablehlo.multiply %97, %111 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.multiply %103, %94 : tensor<f64>
    %127 = stablehlo.subtract %125, %126 : tensor<f64>
    %128 = stablehlo.multiply %109, %99 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.reshape %129 : (tensor<f64>) -> tensor<1xf64>
    %131 = stablehlo.multiply %73, %99 : tensor<f64>
    %132 = stablehlo.multiply %97, %94 : tensor<f64>
    %133 = stablehlo.subtract %131, %132 : tensor<f64>
    %134 = stablehlo.multiply %103, %111 : tensor<f64>
    %135 = stablehlo.subtract %133, %134 : tensor<f64>
    %136 = stablehlo.multiply %109, %105 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.reshape %137 : (tensor<f64>) -> tensor<1xf64>
    %139 = stablehlo.concatenate %114, %122, %130, %138, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %140 = stablehlo.slice %139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %141 = stablehlo.reshape %140 : (tensor<1xf64>) -> tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.slice %139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %144 = stablehlo.reshape %143 : (tensor<1xf64>) -> tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.slice %139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %147 = stablehlo.reshape %146 : (tensor<1xf64>) -> tensor<f64>
    %148 = stablehlo.reshape %147 : (tensor<f64>) -> tensor<1xf64>
    %149 = stablehlo.concatenate %142, %145, %148, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %149 : tensor<3xf64>
  }
  func.func private @inner_215(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4xf64>
    %cst_0 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %3 = stablehlo.multiply %1, %2 : tensor<4xf64>
    %cst_1 = stablehlo.constant dense<6.000000e+01> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %5 = stablehlo.divide %3, %4 : tensor<4xf64>
    return %5 : tensor<4xf64>
  }
  func.func private @inner_216(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<"0x4F621058E9089240DCB5847C78179240CE66D5E79A369240AA8BDB689C46924071CE88D2026692409A081B9E2A769240FE21FDF6CD9492402AF697DDD3A492405A17B7D134C49240837CD0B391D39240C1CAA1454AF39240565BB1BFF4019340A267B3EAEB229340B8AF03E71C30934098DD93875D5193408E31772DA55E9340F1F44A597E809340E0E995B2208D93407AA52C430CB093401CC9E53FF8BC9340184850FC3CDF9340D2BCE31401EC9340014D840D270E944026E4839EF91A944075E09C112D3E944052FC1873034B94405917B7D1346D9440131DC9E59B7A9440B6627FD9B19B944010A5BDC16BA994403A234A7BCBCA94408BB96B093DD894406B787AA540FA944012C7BAB851079540780B2428A229954029A913D0B0369540DE02098AC7589540B27BF2B0EC6595404BEA04344D88954082C0CAA1DD95954061C3D32B01B89540B71E85EB45C59540E926310898E79540DE9387859EF495409E5E29CBF416964006F01648282496403108AC1C16469640265C8FC2ED5396404703780B3C75964088B0E1E955829640D95F764F72A496401EF46C56ADB19640A2B437F8A6D496406054522780E0964063AA6054BA0397406B09F9A0AB0F974076BE9F1AEB33974066B3EA73A53E974055302AA96F649740FBA9F1D2616D97402A3A92CB0F94974060545227949C97404950FC182BC49740A635CD3B12CC9740DF718A8E5CF49740F97E6ABCFCFA9740A0CDAACF2D249840F6E461A13E2A9840635DDC46B3539840EDC9C342795A98407D3F355E4A8398402EFF21FD028A98406F3480B7B8B1984043696FF001B99840FF43FAEDD3E09840273108AC4CE89840C64B378919109940A60A4625C5169940E07A14AEC33E9940A64E4013F94499409E3C2CD4D66D9940AE03E78C00749940287E8CB95B9D99405EC3D32BC5A2994070F085C924CD9940FACBEEC933D19940B27BF2B034FC9940769CA223E5FF9940BBB88D06902B9A402EDD24062D2F9A40F6065F98AC5A9A400B4FAF94915E9A4025287E8CCD8A9A408DB96B09F58D9A405B423EE8F5B99A40ADFA5C6D79BD9A40302AA91354E99A40156A4DF346EC9A4050AF9465C4189B40BB96900FD61B9B402B1895D4A1479B40FDB27BF2984B9B4036CD3B4E25779B40FD1873D78A7A9B402506819547A69B402D431CEBD6A99B403892CB7FD4D59B403D2CD49A76D99B40F0C9C34231069C40AEB6627FE9089C4058CA32C4E9359C409DCDAACFF5379C402575029A84649C405A643BDF93679C4063105839C8939C400D0BB5A681979C405A17B7D1DCC29C40BE9F1A2F3DC69C4062E5D0223BF29C40557424976BF59C40"> : tensor<120xf64>
    %cst_0 = stablehlo.constant dense<"0x0000000000ACAA40000000000038AD400000000000CCAE400000000000C8B04000000000009AB1400000000000E7B2400000000000B4B3400000000000F5B4400000000000FCB54000000000000DB740000000000016B840000000000003B94000000000003ABA400000000000F1BA40000000000043BC40000000000007BD40000000000043BE400000000000D2BE40000000008023C040000000000063C040000000000019C140000000008059C140000000000018C24000000000003EC24000000000800FC340000000008034C3400000000080FFC34000000000001EC4400000000000E7C4400000000080E8C4400000000000CBC5400000000080D8C5400000000000A5C6400000000000AFC640000000000083C740000000008089C74000000000006DC84000000000804CC84000000000003FC940000000008013C94000000000800FCA400000000000DAC9400000000000D6CA400000000080A1CA4000000000009FCB4000000000806BCB4000000000006BCC40000000008027CC40000000000033CD400000000000E9CC400000000080EECD400000000080A4CD400000000000B8CE4000000000005DCE40000000000076CF4000000000801BCF40000000004019D0400000000080CDCF40000000008079D04000000000C03FD04000000000C0E2D04000000000C0B1D040000000004041D140000000008001D14000000000C0ABD14000000000805ED14000000000800BD2400000000000C6D140000000000072D24000000000802FD2400000000000DBD240000000000096D24000000000403FD3400000000080EDD2400000000040AAD34000000000C04FD34000000000C00AD4400000000080B5D34000000000406FD440000000000022D4400000000040D4D44000000000C07CD44000000000C041D5400000000000E5D4400000000080AED54000000000C04AD540000000008065D64000000000C0C8D540000000000007D74000000000C09AD640000000008078D740000000004006D7400000000080E1D74000000000C076D740000000004055D8400000000000F6D7400000000000D6D84000000000006FD840000000008044D9400000000080E3D8400000000000BBD94000000000C051D940000000008024DA400000000040BCD940000000008081DA40000000008025DA400000000080E6DA40000000004081DA40000000008046DB400000000000E7DA40000000004098DB40000000000042DB4000000000C0EBDB40000000004097DB4000000000C03ADC400000000080F0DB4000000000007FDC40000000004039DC400000000040C4DC4000000000C087DC40"> : tensor<120xf64>
    %cst_1 = stablehlo.constant dense<[-5.000000e-01, -5.000000e-01, 5.000000e-01, 5.000000e-01]> : tensor<4xf64>
    %0 = call @_interp(%arg0, %cst, %cst_0) : (tensor<4xf64>, tensor<120xf64>, tensor<120xf64>) -> tensor<4xf64>
    %1 = stablehlo.subtract %0, %arg3 : tensor<4xf64>
    %cst_2 = stablehlo.constant dense<0.01098901098901099> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %3 = stablehlo.multiply %2, %1 : tensor<4xf64>
    %4 = stablehlo.add %arg3, %3 : tensor<4xf64>
    %5 = stablehlo.multiply %4, %4 : tensor<4xf64>
    %cst_3 = stablehlo.constant dense<9.9068131782640682E-9> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %7 = stablehlo.multiply %5, %6 : tensor<4xf64>
    %8 = stablehlo.multiply %4, %4 : tensor<4xf64>
    %cst_4 = stablehlo.constant dense<9.8192338453001589E-11> : tensor<f64>
    %9 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %10 = stablehlo.multiply %8, %9 : tensor<4xf64>
    %11 = stablehlo.multiply %10, %cst_1 : tensor<4xf64>
    return %7, %11, %4 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
  }
  func.func private @inner_217(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, -9.810000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg0 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.multiply %cst, %2 : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %5 = stablehlo.concatenate %4, %3, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %6 = stablehlo.add %arg1, %5 : tensor<6xf64>
    return %6 : tensor<6xf64>
  }
  func.func private @inner_218(%arg0: tensor<6xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.negate %0 : tensor<3xf64>
    %2 = call @norm(%1) : (tensor<3xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %4 = stablehlo.multiply %3, %1 : tensor<3xf64>
    %5 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.multiply %4, %5 : tensor<3xf64>
    return %6 : tensor<3xf64>
  }
  func.func private @inner_219(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[[0.000000e+00, -0.087155742747658165, 0.99619469809174543], [-0.065403129230143062, 0.086969135612238901, 0.99406176877383478], [0.000000e+00, 0.087155742747658165, 0.99619469809174543], [-0.065403129230143062, -0.086969135612238901, 0.99406176877383478]]> : tensor<4x3xf64>
    %cst_0 = stablehlo.constant dense<[[-0.20858424832311181, -0.25901062150385384, -0.022660493114391125], [0.19843360999226459, 0.25704989260274902, -0.0094332447193082886], [0.20858424832311181, -0.25901062150385384, 0.022660493114391125], [-0.19843360999226459, 0.25704989260274902, 0.0094332447193082886]]> : tensor<4x3xf64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %2 = stablehlo.multiply %cst, %1 : tensor<4x3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %3 = stablehlo.reduce(%2 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %6 = stablehlo.multiply %cst, %5 : tensor<4x3xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.reduce(%6 init: %cst_2) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %10 = stablehlo.multiply %cst_0, %9 : tensor<4x3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %11 = stablehlo.reduce(%10 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %13 = stablehlo.concatenate %12, %3, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %15 = stablehlo.concatenate %7, %14, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %16 = stablehlo.add %13, %15 : tensor<6xf64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %17 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %18 = stablehlo.concatenate %11, %17, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %19 = stablehlo.add %16, %18 : tensor<6xf64>
    return %19 : tensor<6xf64>
  }
  func.func private @inner_220(%arg0: tensor<6xf64>, %arg1: tensor<3xf64>, %arg2: tensor<7xf64>, %arg3: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.concatenate %0, %arg1, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<6xf64>
    %3 = stablehlo.slice %arg2 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %4 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %arg0 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %9 = stablehlo.slice %8 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
    %11 = stablehlo.multiply %5, %10 : tensor<f64>
    %12 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %13 = stablehlo.reshape %12 : (tensor<1xf64>) -> tensor<f64>
    %14 = stablehlo.slice %8 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.multiply %13, %15 : tensor<f64>
    %17 = stablehlo.add %11, %16 : tensor<f64>
    %18 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.slice %8 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.multiply %19, %21 : tensor<f64>
    %23 = stablehlo.add %17, %22 : tensor<f64>
    %24 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.slice %8 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
    %28 = stablehlo.multiply %25, %27 : tensor<f64>
    %29 = stablehlo.subtract %23, %28 : tensor<f64>
    %30 = stablehlo.reshape %29 : (tensor<f64>) -> tensor<1xf64>
    %31 = stablehlo.multiply %5, %27 : tensor<f64>
    %32 = stablehlo.multiply %13, %21 : tensor<f64>
    %33 = stablehlo.subtract %31, %32 : tensor<f64>
    %34 = stablehlo.multiply %19, %15 : tensor<f64>
    %35 = stablehlo.add %33, %34 : tensor<f64>
    %36 = stablehlo.multiply %25, %10 : tensor<f64>
    %37 = stablehlo.add %35, %36 : tensor<f64>
    %38 = stablehlo.reshape %37 : (tensor<f64>) -> tensor<1xf64>
    %39 = stablehlo.multiply %5, %21 : tensor<f64>
    %40 = stablehlo.multiply %13, %27 : tensor<f64>
    %41 = stablehlo.add %39, %40 : tensor<f64>
    %42 = stablehlo.multiply %19, %10 : tensor<f64>
    %43 = stablehlo.subtract %41, %42 : tensor<f64>
    %44 = stablehlo.multiply %25, %15 : tensor<f64>
    %45 = stablehlo.add %43, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %5, %15 : tensor<f64>
    %48 = stablehlo.multiply %13, %10 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %19, %27 : tensor<f64>
    %51 = stablehlo.subtract %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %25, %21 : tensor<f64>
    %53 = stablehlo.subtract %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.concatenate %30, %38, %46, %54, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %56 = stablehlo.slice %55 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %57 = stablehlo.reshape %56 : (tensor<1xf64>) -> tensor<f64>
    %58 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %59 = stablehlo.reshape %58 : (tensor<1xf64>) -> tensor<f64>
    %60 = stablehlo.negate %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %63 = stablehlo.reshape %62 : (tensor<1xf64>) -> tensor<f64>
    %64 = stablehlo.negate %63 : tensor<f64>
    %65 = stablehlo.reshape %64 : (tensor<f64>) -> tensor<1xf64>
    %66 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %67 = stablehlo.reshape %66 : (tensor<1xf64>) -> tensor<f64>
    %68 = stablehlo.negate %67 : tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %71 = stablehlo.reshape %70 : (tensor<1xf64>) -> tensor<f64>
    %72 = stablehlo.reshape %71 : (tensor<f64>) -> tensor<1xf64>
    %73 = stablehlo.concatenate %61, %65, %69, %72, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %74 = stablehlo.dot_general %3, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %76 = stablehlo.divide %73, %75 : tensor<4xf64>
    %77 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.multiply %57, %78 : tensor<f64>
    %80 = stablehlo.slice %55 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %81 = stablehlo.reshape %80 : (tensor<1xf64>) -> tensor<f64>
    %82 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.multiply %81, %83 : tensor<f64>
    %85 = stablehlo.add %79, %84 : tensor<f64>
    %86 = stablehlo.slice %55 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %89 = stablehlo.reshape %88 : (tensor<1xf64>) -> tensor<f64>
    %90 = stablehlo.multiply %87, %89 : tensor<f64>
    %91 = stablehlo.add %85, %90 : tensor<f64>
    %92 = stablehlo.slice %55 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %95 = stablehlo.reshape %94 : (tensor<1xf64>) -> tensor<f64>
    %96 = stablehlo.multiply %93, %95 : tensor<f64>
    %97 = stablehlo.subtract %91, %96 : tensor<f64>
    %98 = stablehlo.reshape %97 : (tensor<f64>) -> tensor<1xf64>
    %99 = stablehlo.multiply %57, %95 : tensor<f64>
    %100 = stablehlo.multiply %81, %89 : tensor<f64>
    %101 = stablehlo.subtract %99, %100 : tensor<f64>
    %102 = stablehlo.multiply %87, %83 : tensor<f64>
    %103 = stablehlo.add %101, %102 : tensor<f64>
    %104 = stablehlo.multiply %93, %78 : tensor<f64>
    %105 = stablehlo.add %103, %104 : tensor<f64>
    %106 = stablehlo.reshape %105 : (tensor<f64>) -> tensor<1xf64>
    %107 = stablehlo.multiply %57, %89 : tensor<f64>
    %108 = stablehlo.multiply %81, %95 : tensor<f64>
    %109 = stablehlo.add %107, %108 : tensor<f64>
    %110 = stablehlo.multiply %87, %78 : tensor<f64>
    %111 = stablehlo.subtract %109, %110 : tensor<f64>
    %112 = stablehlo.multiply %93, %83 : tensor<f64>
    %113 = stablehlo.add %111, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %57, %83 : tensor<f64>
    %116 = stablehlo.multiply %81, %78 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %87, %95 : tensor<f64>
    %119 = stablehlo.subtract %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %93, %89 : tensor<f64>
    %121 = stablehlo.subtract %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.concatenate %98, %106, %114, %122, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %124 = stablehlo.slice %123 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %125 = stablehlo.reshape %124 : (tensor<1xf64>) -> tensor<f64>
    %126 = stablehlo.reshape %125 : (tensor<f64>) -> tensor<1xf64>
    %127 = stablehlo.slice %123 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %128 = stablehlo.reshape %127 : (tensor<1xf64>) -> tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.slice %123 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %131 = stablehlo.reshape %130 : (tensor<1xf64>) -> tensor<f64>
    %132 = stablehlo.reshape %131 : (tensor<f64>) -> tensor<1xf64>
    %133 = stablehlo.concatenate %126, %129, %132, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %134 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %135 = stablehlo.reshape %134 : (tensor<1xf64>) -> tensor<f64>
    %136 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %137 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.concatenate %136, %137, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %139 = stablehlo.slice %138 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.multiply %135, %140 : tensor<f64>
    %142 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %143 = stablehlo.reshape %142 : (tensor<1xf64>) -> tensor<f64>
    %144 = stablehlo.slice %138 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.multiply %143, %145 : tensor<f64>
    %147 = stablehlo.add %141, %146 : tensor<f64>
    %148 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %149 = stablehlo.reshape %148 : (tensor<1xf64>) -> tensor<f64>
    %150 = stablehlo.slice %138 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
    %152 = stablehlo.multiply %149, %151 : tensor<f64>
    %153 = stablehlo.add %147, %152 : tensor<f64>
    %154 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %155 = stablehlo.reshape %154 : (tensor<1xf64>) -> tensor<f64>
    %156 = stablehlo.slice %138 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %157 = stablehlo.reshape %156 : (tensor<1xf64>) -> tensor<f64>
    %158 = stablehlo.multiply %155, %157 : tensor<f64>
    %159 = stablehlo.subtract %153, %158 : tensor<f64>
    %160 = stablehlo.reshape %159 : (tensor<f64>) -> tensor<1xf64>
    %161 = stablehlo.multiply %135, %157 : tensor<f64>
    %162 = stablehlo.multiply %143, %151 : tensor<f64>
    %163 = stablehlo.subtract %161, %162 : tensor<f64>
    %164 = stablehlo.multiply %149, %145 : tensor<f64>
    %165 = stablehlo.add %163, %164 : tensor<f64>
    %166 = stablehlo.multiply %155, %140 : tensor<f64>
    %167 = stablehlo.add %165, %166 : tensor<f64>
    %168 = stablehlo.reshape %167 : (tensor<f64>) -> tensor<1xf64>
    %169 = stablehlo.multiply %135, %151 : tensor<f64>
    %170 = stablehlo.multiply %143, %157 : tensor<f64>
    %171 = stablehlo.add %169, %170 : tensor<f64>
    %172 = stablehlo.multiply %149, %140 : tensor<f64>
    %173 = stablehlo.subtract %171, %172 : tensor<f64>
    %174 = stablehlo.multiply %155, %145 : tensor<f64>
    %175 = stablehlo.add %173, %174 : tensor<f64>
    %176 = stablehlo.reshape %175 : (tensor<f64>) -> tensor<1xf64>
    %177 = stablehlo.multiply %135, %145 : tensor<f64>
    %178 = stablehlo.multiply %143, %140 : tensor<f64>
    %179 = stablehlo.subtract %177, %178 : tensor<f64>
    %180 = stablehlo.multiply %149, %157 : tensor<f64>
    %181 = stablehlo.subtract %179, %180 : tensor<f64>
    %182 = stablehlo.multiply %155, %151 : tensor<f64>
    %183 = stablehlo.subtract %181, %182 : tensor<f64>
    %184 = stablehlo.reshape %183 : (tensor<f64>) -> tensor<1xf64>
    %185 = stablehlo.concatenate %160, %168, %176, %184, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %186 = stablehlo.slice %185 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %187 = stablehlo.reshape %186 : (tensor<1xf64>) -> tensor<f64>
    %188 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %189 = stablehlo.reshape %188 : (tensor<1xf64>) -> tensor<f64>
    %190 = stablehlo.negate %189 : tensor<f64>
    %191 = stablehlo.reshape %190 : (tensor<f64>) -> tensor<1xf64>
    %192 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %193 = stablehlo.reshape %192 : (tensor<1xf64>) -> tensor<f64>
    %194 = stablehlo.negate %193 : tensor<f64>
    %195 = stablehlo.reshape %194 : (tensor<f64>) -> tensor<1xf64>
    %196 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %197 = stablehlo.reshape %196 : (tensor<1xf64>) -> tensor<f64>
    %198 = stablehlo.negate %197 : tensor<f64>
    %199 = stablehlo.reshape %198 : (tensor<f64>) -> tensor<1xf64>
    %200 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %201 = stablehlo.reshape %200 : (tensor<1xf64>) -> tensor<f64>
    %202 = stablehlo.reshape %201 : (tensor<f64>) -> tensor<1xf64>
    %203 = stablehlo.concatenate %191, %195, %199, %202, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %204 = stablehlo.dot_general %3, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %205 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %206 = stablehlo.divide %203, %205 : tensor<4xf64>
    %207 = stablehlo.slice %206 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %208 = stablehlo.reshape %207 : (tensor<1xf64>) -> tensor<f64>
    %209 = stablehlo.multiply %187, %208 : tensor<f64>
    %210 = stablehlo.slice %185 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %211 = stablehlo.reshape %210 : (tensor<1xf64>) -> tensor<f64>
    %212 = stablehlo.slice %206 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %213 = stablehlo.reshape %212 : (tensor<1xf64>) -> tensor<f64>
    %214 = stablehlo.multiply %211, %213 : tensor<f64>
    %215 = stablehlo.add %209, %214 : tensor<f64>
    %216 = stablehlo.slice %185 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %217 = stablehlo.reshape %216 : (tensor<1xf64>) -> tensor<f64>
    %218 = stablehlo.slice %206 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %219 = stablehlo.reshape %218 : (tensor<1xf64>) -> tensor<f64>
    %220 = stablehlo.multiply %217, %219 : tensor<f64>
    %221 = stablehlo.add %215, %220 : tensor<f64>
    %222 = stablehlo.slice %185 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %223 = stablehlo.reshape %222 : (tensor<1xf64>) -> tensor<f64>
    %224 = stablehlo.slice %206 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %225 = stablehlo.reshape %224 : (tensor<1xf64>) -> tensor<f64>
    %226 = stablehlo.multiply %223, %225 : tensor<f64>
    %227 = stablehlo.subtract %221, %226 : tensor<f64>
    %228 = stablehlo.reshape %227 : (tensor<f64>) -> tensor<1xf64>
    %229 = stablehlo.multiply %187, %225 : tensor<f64>
    %230 = stablehlo.multiply %211, %219 : tensor<f64>
    %231 = stablehlo.subtract %229, %230 : tensor<f64>
    %232 = stablehlo.multiply %217, %213 : tensor<f64>
    %233 = stablehlo.add %231, %232 : tensor<f64>
    %234 = stablehlo.multiply %223, %208 : tensor<f64>
    %235 = stablehlo.add %233, %234 : tensor<f64>
    %236 = stablehlo.reshape %235 : (tensor<f64>) -> tensor<1xf64>
    %237 = stablehlo.multiply %187, %219 : tensor<f64>
    %238 = stablehlo.multiply %211, %225 : tensor<f64>
    %239 = stablehlo.add %237, %238 : tensor<f64>
    %240 = stablehlo.multiply %217, %208 : tensor<f64>
    %241 = stablehlo.subtract %239, %240 : tensor<f64>
    %242 = stablehlo.multiply %223, %213 : tensor<f64>
    %243 = stablehlo.add %241, %242 : tensor<f64>
    %244 = stablehlo.reshape %243 : (tensor<f64>) -> tensor<1xf64>
    %245 = stablehlo.multiply %187, %213 : tensor<f64>
    %246 = stablehlo.multiply %211, %208 : tensor<f64>
    %247 = stablehlo.subtract %245, %246 : tensor<f64>
    %248 = stablehlo.multiply %217, %225 : tensor<f64>
    %249 = stablehlo.subtract %247, %248 : tensor<f64>
    %250 = stablehlo.multiply %223, %219 : tensor<f64>
    %251 = stablehlo.subtract %249, %250 : tensor<f64>
    %252 = stablehlo.reshape %251 : (tensor<f64>) -> tensor<1xf64>
    %253 = stablehlo.concatenate %228, %236, %244, %252, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %254 = stablehlo.slice %253 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %255 = stablehlo.reshape %254 : (tensor<1xf64>) -> tensor<f64>
    %256 = stablehlo.reshape %255 : (tensor<f64>) -> tensor<1xf64>
    %257 = stablehlo.slice %253 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %258 = stablehlo.reshape %257 : (tensor<1xf64>) -> tensor<f64>
    %259 = stablehlo.reshape %258 : (tensor<f64>) -> tensor<1xf64>
    %260 = stablehlo.slice %253 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %261 = stablehlo.reshape %260 : (tensor<1xf64>) -> tensor<f64>
    %262 = stablehlo.reshape %261 : (tensor<f64>) -> tensor<1xf64>
    %263 = stablehlo.concatenate %256, %259, %262, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %264 = stablehlo.concatenate %133, %263, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %265 = stablehlo.add %2, %264 : tensor<6xf64>
    return %265 : tensor<6xf64>
  }
  func.func private @inner_221(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @inner_222(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<[1797259609, 2579123966]> : tensor<2xui32>
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %1 = call @_threefry_fold_in(%c, %0) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %2 = stablehlo.sqrt %cst : tensor<f64>
    %3 = call @_normal(%1) : (tensor<2xui32>) -> tensor<3xf64>
    %4 = stablehlo.convert %2 : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.multiply %5, %3 : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.multiply %6, %7 : tensor<3xf64>
    %9 = stablehlo.add %arg1, %8 : tensor<3xf64>
    return %9 : tensor<3xf64>
  }
  func.func private @inner_223(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<6xf64>, %arg3: tensor<4x3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %c = stablehlo.constant dense<[1797259609, 2579123966]> : tensor<2xui32>
    %cst = stablehlo.constant dense<[0.016209783477834406, 0.032419566955668812, 0.016209783477834406, -1.6089340340646645, 0.67377316797600217]> : tensor<5xf64>
    %0 = stablehlo.slice %arg2 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2 = stablehlo.slice %1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.negate %3 : tensor<f64>
    %5 = stablehlo.reshape %4 : (tensor<f64>) -> tensor<1xf64>
    %6 = stablehlo.slice %1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.reshape %15 : (tensor<f64>) -> tensor<1xf64>
    %17 = stablehlo.concatenate %5, %9, %13, %16, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %18 = stablehlo.dot_general %1, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %20 = stablehlo.divide %17, %19 : tensor<4xf64>
    %21 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %23 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.concatenate %0, %23, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %25 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %27 = stablehlo.multiply %22, %26 : tensor<f64>
    %28 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
    %30 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.multiply %29, %31 : tensor<f64>
    %33 = stablehlo.add %27, %32 : tensor<f64>
    %34 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %37 = stablehlo.reshape %36 : (tensor<1xf64>) -> tensor<f64>
    %38 = stablehlo.multiply %35, %37 : tensor<f64>
    %39 = stablehlo.add %33, %38 : tensor<f64>
    %40 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %43 = stablehlo.reshape %42 : (tensor<1xf64>) -> tensor<f64>
    %44 = stablehlo.multiply %41, %43 : tensor<f64>
    %45 = stablehlo.subtract %39, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %22, %43 : tensor<f64>
    %48 = stablehlo.multiply %29, %37 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %35, %31 : tensor<f64>
    %51 = stablehlo.add %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %41, %26 : tensor<f64>
    %53 = stablehlo.add %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.multiply %22, %37 : tensor<f64>
    %56 = stablehlo.multiply %29, %43 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.multiply %35, %26 : tensor<f64>
    %59 = stablehlo.subtract %57, %58 : tensor<f64>
    %60 = stablehlo.multiply %41, %31 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.reshape %61 : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.multiply %22, %31 : tensor<f64>
    %64 = stablehlo.multiply %29, %26 : tensor<f64>
    %65 = stablehlo.subtract %63, %64 : tensor<f64>
    %66 = stablehlo.multiply %35, %43 : tensor<f64>
    %67 = stablehlo.subtract %65, %66 : tensor<f64>
    %68 = stablehlo.multiply %41, %37 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.reshape %69 : (tensor<f64>) -> tensor<1xf64>
    %71 = stablehlo.concatenate %46, %54, %62, %70, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %72 = stablehlo.slice %71 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1xf64>) -> tensor<f64>
    %74 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.negate %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.reshape %87 : (tensor<f64>) -> tensor<1xf64>
    %89 = stablehlo.concatenate %77, %81, %85, %88, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %90 = stablehlo.dot_general %20, %20, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %92 = stablehlo.divide %89, %91 : tensor<4xf64>
    %93 = stablehlo.slice %92 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %94 = stablehlo.reshape %93 : (tensor<1xf64>) -> tensor<f64>
    %95 = stablehlo.multiply %73, %94 : tensor<f64>
    %96 = stablehlo.slice %71 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.slice %92 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.multiply %97, %99 : tensor<f64>
    %101 = stablehlo.add %95, %100 : tensor<f64>
    %102 = stablehlo.slice %71 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.slice %92 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %105 = stablehlo.reshape %104 : (tensor<1xf64>) -> tensor<f64>
    %106 = stablehlo.multiply %103, %105 : tensor<f64>
    %107 = stablehlo.add %101, %106 : tensor<f64>
    %108 = stablehlo.slice %71 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.slice %92 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %111 = stablehlo.reshape %110 : (tensor<1xf64>) -> tensor<f64>
    %112 = stablehlo.multiply %109, %111 : tensor<f64>
    %113 = stablehlo.subtract %107, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %73, %111 : tensor<f64>
    %116 = stablehlo.multiply %97, %105 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %103, %99 : tensor<f64>
    %119 = stablehlo.add %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %109, %94 : tensor<f64>
    %121 = stablehlo.add %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.multiply %73, %105 : tensor<f64>
    %124 = stablehlo.multiply %97, %111 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.multiply %103, %94 : tensor<f64>
    %127 = stablehlo.subtract %125, %126 : tensor<f64>
    %128 = stablehlo.multiply %109, %99 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.reshape %129 : (tensor<f64>) -> tensor<1xf64>
    %131 = stablehlo.multiply %73, %99 : tensor<f64>
    %132 = stablehlo.multiply %97, %94 : tensor<f64>
    %133 = stablehlo.subtract %131, %132 : tensor<f64>
    %134 = stablehlo.multiply %103, %111 : tensor<f64>
    %135 = stablehlo.subtract %133, %134 : tensor<f64>
    %136 = stablehlo.multiply %109, %105 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.reshape %137 : (tensor<f64>) -> tensor<1xf64>
    %139 = stablehlo.concatenate %114, %122, %130, %138, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %140 = stablehlo.slice %139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %141 = stablehlo.reshape %140 : (tensor<1xf64>) -> tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.slice %139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %144 = stablehlo.reshape %143 : (tensor<1xf64>) -> tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.slice %139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %147 = stablehlo.reshape %146 : (tensor<1xf64>) -> tensor<f64>
    %148 = stablehlo.reshape %147 : (tensor<f64>) -> tensor<1xf64>
    %149 = stablehlo.concatenate %142, %145, %148, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %150 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %151 = call @_threefry_fold_in(%c, %150) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_1 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %152 = stablehlo.sqrt %cst_1 : tensor<f64>
    %153 = call @_normal(%151) : (tensor<2xui32>) -> tensor<3xf64>
    %154 = stablehlo.convert %152 : tensor<f64>
    %155 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %156 = stablehlo.multiply %155, %153 : tensor<3xf64>
    %157 = stablehlo.add %149, %156 : tensor<3xf64>
    %158 = stablehlo.add %157, %arg4 : tensor<3xf64>
    %159 = stablehlo.slice %cst [0:1] : (tensor<5xf64>) -> tensor<1xf64>
    %160 = stablehlo.reshape %159 : (tensor<1xf64>) -> tensor<f64>
    %161 = stablehlo.slice %cst [1:2] : (tensor<5xf64>) -> tensor<1xf64>
    %162 = stablehlo.reshape %161 : (tensor<1xf64>) -> tensor<f64>
    %163 = stablehlo.slice %cst [2:3] : (tensor<5xf64>) -> tensor<1xf64>
    %164 = stablehlo.reshape %163 : (tensor<1xf64>) -> tensor<f64>
    %165 = stablehlo.slice %cst [3:4] : (tensor<5xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %cst [4:5] : (tensor<5xf64>) -> tensor<1xf64>
    %168 = stablehlo.reshape %167 : (tensor<1xf64>) -> tensor<f64>
    %169 = stablehlo.slice %arg3 [0:1, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %170 = stablehlo.reshape %169 : (tensor<1x3xf64>) -> tensor<3xf64>
    %171 = stablehlo.slice %arg3 [1:2, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %172 = stablehlo.reshape %171 : (tensor<1x3xf64>) -> tensor<3xf64>
    %173 = stablehlo.slice %arg3 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %174 = stablehlo.reshape %173 : (tensor<1x3xf64>) -> tensor<3xf64>
    %175 = stablehlo.slice %arg3 [3:4, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %176 = stablehlo.reshape %175 : (tensor<1x3xf64>) -> tensor<3xf64>
    %177 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %178 = stablehlo.multiply %177, %158 : tensor<3xf64>
    %179 = stablehlo.broadcast_in_dim %162, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %180 = stablehlo.multiply %179, %170 : tensor<3xf64>
    %181 = stablehlo.add %178, %180 : tensor<3xf64>
    %182 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %183 = stablehlo.multiply %182, %172 : tensor<3xf64>
    %184 = stablehlo.add %181, %183 : tensor<3xf64>
    %185 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %186 = stablehlo.multiply %185, %174 : tensor<3xf64>
    %187 = stablehlo.subtract %184, %186 : tensor<3xf64>
    %188 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %189 = stablehlo.multiply %188, %176 : tensor<3xf64>
    %190 = stablehlo.subtract %187, %189 : tensor<3xf64>
    %191 = stablehlo.broadcast_in_dim %158, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %192 = stablehlo.broadcast_in_dim %170, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %193 = stablehlo.broadcast_in_dim %190, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %194 = stablehlo.broadcast_in_dim %174, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %195 = stablehlo.concatenate %191, %192, %193, %194, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<4x3xf64>
    %196 = stablehlo.slice %195 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %197 = stablehlo.reshape %196 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %195, %197 : tensor<4x3xf64>, tensor<3xf64>
  }
  func.func private @inner_224(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<6xf64>, %arg3: tensor<4x3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %c = stablehlo.constant dense<[0, 0, 1]> : tensor<3xi64>
    %c_0 = stablehlo.constant dense<[928981903, 3453687069]> : tensor<2xui32>
    %cst = stablehlo.constant dense<[0.0044300075115303239, 0.0088600150230606477, 0.0044300075115303239, -1.8030932880476023, 0.82081331809372371]> : tensor<5xf64>
    %0 = stablehlo.slice %arg2 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<9.810000e+00> : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2 = stablehlo.divide %0, %1 : tensor<3xf64>
    %3 = stablehlo.convert %c : (tensor<3xi64>) -> tensor<3xf64>
    %4 = stablehlo.add %2, %3 : tensor<3xf64>
    %5 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %6 = stablehlo.slice %5 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %5 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %5 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.negate %15 : tensor<f64>
    %17 = stablehlo.reshape %16 : (tensor<f64>) -> tensor<1xf64>
    %18 = stablehlo.slice %5 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.reshape %19 : (tensor<f64>) -> tensor<1xf64>
    %21 = stablehlo.concatenate %9, %13, %17, %20, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %22 = stablehlo.dot_general %5, %5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %24 = stablehlo.divide %21, %23 : tensor<4xf64>
    %25 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %27 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %28 = stablehlo.concatenate %4, %27, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %29 = stablehlo.slice %28 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.multiply %26, %30 : tensor<f64>
    %32 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %33 = stablehlo.reshape %32 : (tensor<1xf64>) -> tensor<f64>
    %34 = stablehlo.slice %28 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.multiply %33, %35 : tensor<f64>
    %37 = stablehlo.add %31, %36 : tensor<f64>
    %38 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %39 = stablehlo.reshape %38 : (tensor<1xf64>) -> tensor<f64>
    %40 = stablehlo.slice %28 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.multiply %39, %41 : tensor<f64>
    %43 = stablehlo.add %37, %42 : tensor<f64>
    %44 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %45 = stablehlo.reshape %44 : (tensor<1xf64>) -> tensor<f64>
    %46 = stablehlo.slice %28 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %47 = stablehlo.reshape %46 : (tensor<1xf64>) -> tensor<f64>
    %48 = stablehlo.multiply %45, %47 : tensor<f64>
    %49 = stablehlo.subtract %43, %48 : tensor<f64>
    %50 = stablehlo.reshape %49 : (tensor<f64>) -> tensor<1xf64>
    %51 = stablehlo.multiply %26, %47 : tensor<f64>
    %52 = stablehlo.multiply %33, %41 : tensor<f64>
    %53 = stablehlo.subtract %51, %52 : tensor<f64>
    %54 = stablehlo.multiply %39, %35 : tensor<f64>
    %55 = stablehlo.add %53, %54 : tensor<f64>
    %56 = stablehlo.multiply %45, %30 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.reshape %57 : (tensor<f64>) -> tensor<1xf64>
    %59 = stablehlo.multiply %26, %41 : tensor<f64>
    %60 = stablehlo.multiply %33, %47 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.multiply %39, %30 : tensor<f64>
    %63 = stablehlo.subtract %61, %62 : tensor<f64>
    %64 = stablehlo.multiply %45, %35 : tensor<f64>
    %65 = stablehlo.add %63, %64 : tensor<f64>
    %66 = stablehlo.reshape %65 : (tensor<f64>) -> tensor<1xf64>
    %67 = stablehlo.multiply %26, %35 : tensor<f64>
    %68 = stablehlo.multiply %33, %30 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.multiply %39, %47 : tensor<f64>
    %71 = stablehlo.subtract %69, %70 : tensor<f64>
    %72 = stablehlo.multiply %45, %41 : tensor<f64>
    %73 = stablehlo.subtract %71, %72 : tensor<f64>
    %74 = stablehlo.reshape %73 : (tensor<f64>) -> tensor<1xf64>
    %75 = stablehlo.concatenate %50, %58, %66, %74, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %76 = stablehlo.slice %75 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %77 = stablehlo.reshape %76 : (tensor<1xf64>) -> tensor<f64>
    %78 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.negate %87 : tensor<f64>
    %89 = stablehlo.reshape %88 : (tensor<f64>) -> tensor<1xf64>
    %90 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %91 = stablehlo.reshape %90 : (tensor<1xf64>) -> tensor<f64>
    %92 = stablehlo.reshape %91 : (tensor<f64>) -> tensor<1xf64>
    %93 = stablehlo.concatenate %81, %85, %89, %92, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %94 = stablehlo.dot_general %24, %24, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %96 = stablehlo.divide %93, %95 : tensor<4xf64>
    %97 = stablehlo.slice %96 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.multiply %77, %98 : tensor<f64>
    %100 = stablehlo.slice %75 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %101 = stablehlo.reshape %100 : (tensor<1xf64>) -> tensor<f64>
    %102 = stablehlo.slice %96 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.multiply %101, %103 : tensor<f64>
    %105 = stablehlo.add %99, %104 : tensor<f64>
    %106 = stablehlo.slice %75 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %107 = stablehlo.reshape %106 : (tensor<1xf64>) -> tensor<f64>
    %108 = stablehlo.slice %96 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.multiply %107, %109 : tensor<f64>
    %111 = stablehlo.add %105, %110 : tensor<f64>
    %112 = stablehlo.slice %75 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %113 = stablehlo.reshape %112 : (tensor<1xf64>) -> tensor<f64>
    %114 = stablehlo.slice %96 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %115 = stablehlo.reshape %114 : (tensor<1xf64>) -> tensor<f64>
    %116 = stablehlo.multiply %113, %115 : tensor<f64>
    %117 = stablehlo.subtract %111, %116 : tensor<f64>
    %118 = stablehlo.reshape %117 : (tensor<f64>) -> tensor<1xf64>
    %119 = stablehlo.multiply %77, %115 : tensor<f64>
    %120 = stablehlo.multiply %101, %109 : tensor<f64>
    %121 = stablehlo.subtract %119, %120 : tensor<f64>
    %122 = stablehlo.multiply %107, %103 : tensor<f64>
    %123 = stablehlo.add %121, %122 : tensor<f64>
    %124 = stablehlo.multiply %113, %98 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.reshape %125 : (tensor<f64>) -> tensor<1xf64>
    %127 = stablehlo.multiply %77, %109 : tensor<f64>
    %128 = stablehlo.multiply %101, %115 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.multiply %107, %98 : tensor<f64>
    %131 = stablehlo.subtract %129, %130 : tensor<f64>
    %132 = stablehlo.multiply %113, %103 : tensor<f64>
    %133 = stablehlo.add %131, %132 : tensor<f64>
    %134 = stablehlo.reshape %133 : (tensor<f64>) -> tensor<1xf64>
    %135 = stablehlo.multiply %77, %103 : tensor<f64>
    %136 = stablehlo.multiply %101, %98 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.multiply %107, %115 : tensor<f64>
    %139 = stablehlo.subtract %137, %138 : tensor<f64>
    %140 = stablehlo.multiply %113, %109 : tensor<f64>
    %141 = stablehlo.subtract %139, %140 : tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.concatenate %118, %126, %134, %142, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %144 = stablehlo.slice %143 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.reshape %145 : (tensor<f64>) -> tensor<1xf64>
    %147 = stablehlo.slice %143 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %148 = stablehlo.reshape %147 : (tensor<1xf64>) -> tensor<f64>
    %149 = stablehlo.reshape %148 : (tensor<f64>) -> tensor<1xf64>
    %150 = stablehlo.slice %143 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
    %152 = stablehlo.reshape %151 : (tensor<f64>) -> tensor<1xf64>
    %153 = stablehlo.concatenate %146, %149, %152, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %154 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %155 = call @_threefry_fold_in(%c_0, %154) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_3 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %156 = stablehlo.sqrt %cst_3 : tensor<f64>
    %157 = call @_normal(%155) : (tensor<2xui32>) -> tensor<3xf64>
    %158 = stablehlo.convert %156 : tensor<f64>
    %159 = stablehlo.broadcast_in_dim %158, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %160 = stablehlo.multiply %159, %157 : tensor<3xf64>
    %161 = stablehlo.add %153, %160 : tensor<3xf64>
    %162 = stablehlo.add %161, %arg4 : tensor<3xf64>
    %163 = stablehlo.slice %cst [0:1] : (tensor<5xf64>) -> tensor<1xf64>
    %164 = stablehlo.reshape %163 : (tensor<1xf64>) -> tensor<f64>
    %165 = stablehlo.slice %cst [1:2] : (tensor<5xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %cst [2:3] : (tensor<5xf64>) -> tensor<1xf64>
    %168 = stablehlo.reshape %167 : (tensor<1xf64>) -> tensor<f64>
    %169 = stablehlo.slice %cst [3:4] : (tensor<5xf64>) -> tensor<1xf64>
    %170 = stablehlo.reshape %169 : (tensor<1xf64>) -> tensor<f64>
    %171 = stablehlo.slice %cst [4:5] : (tensor<5xf64>) -> tensor<1xf64>
    %172 = stablehlo.reshape %171 : (tensor<1xf64>) -> tensor<f64>
    %173 = stablehlo.slice %arg3 [0:1, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %174 = stablehlo.reshape %173 : (tensor<1x3xf64>) -> tensor<3xf64>
    %175 = stablehlo.slice %arg3 [1:2, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %176 = stablehlo.reshape %175 : (tensor<1x3xf64>) -> tensor<3xf64>
    %177 = stablehlo.slice %arg3 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %178 = stablehlo.reshape %177 : (tensor<1x3xf64>) -> tensor<3xf64>
    %179 = stablehlo.slice %arg3 [3:4, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %180 = stablehlo.reshape %179 : (tensor<1x3xf64>) -> tensor<3xf64>
    %181 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %182 = stablehlo.multiply %181, %162 : tensor<3xf64>
    %183 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %184 = stablehlo.multiply %183, %174 : tensor<3xf64>
    %185 = stablehlo.add %182, %184 : tensor<3xf64>
    %186 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %187 = stablehlo.multiply %186, %176 : tensor<3xf64>
    %188 = stablehlo.add %185, %187 : tensor<3xf64>
    %189 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %190 = stablehlo.multiply %189, %178 : tensor<3xf64>
    %191 = stablehlo.subtract %188, %190 : tensor<3xf64>
    %192 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %193 = stablehlo.multiply %192, %180 : tensor<3xf64>
    %194 = stablehlo.subtract %191, %193 : tensor<3xf64>
    %195 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %196 = stablehlo.broadcast_in_dim %174, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %197 = stablehlo.broadcast_in_dim %194, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %198 = stablehlo.broadcast_in_dim %178, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %199 = stablehlo.concatenate %195, %196, %197, %198, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<4x3xf64>
    %200 = stablehlo.slice %199 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %201 = stablehlo.reshape %200 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %199, %201 : tensor<4x3xf64>, tensor<3xf64>
  }
  func.func private @inner_225(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = call @norm(%arg0) : (tensor<3xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.subtract %0, %cst : tensor<f64>
    %2 = stablehlo.abs %1 : tensor<f64>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %3 = stablehlo.divide %2, %cst_0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %4 = call @clip(%3, %cst_1, %cst_2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.subtract %cst_3, %4 : tensor<f64>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %6 = stablehlo.multiply %cst_4, %5 : tensor<f64>
    %7 = call @norm(%arg1) : (tensor<3xf64>) -> tensor<f64>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %8 = stablehlo.divide %7, %cst_5 : tensor<f64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %9 = call @clip(%8, %cst_6, %cst_7) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %10 = stablehlo.subtract %cst_8, %9 : tensor<f64>
    %11 = stablehlo.multiply %6, %10 : tensor<f64>
    return %11 : tensor<f64>
  }
  func.func private @inner_226(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<3xf64>
    %c = stablehlo.constant dense<[4146024105, 2718843009]> : tensor<2xui32>
    %0 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
    %3 = stablehlo.negate %2 : tensor<f64>
    %4 = stablehlo.reshape %3 : (tensor<f64>) -> tensor<1xf64>
    %5 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %7 = stablehlo.negate %6 : tensor<f64>
    %8 = stablehlo.reshape %7 : (tensor<f64>) -> tensor<1xf64>
    %9 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
    %11 = stablehlo.negate %10 : tensor<f64>
    %12 = stablehlo.reshape %11 : (tensor<f64>) -> tensor<1xf64>
    %13 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %14 = stablehlo.reshape %13 : (tensor<1xf64>) -> tensor<f64>
    %15 = stablehlo.reshape %14 : (tensor<f64>) -> tensor<1xf64>
    %16 = stablehlo.concatenate %4, %8, %12, %15, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %17 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %19 = stablehlo.divide %16, %18 : tensor<4xf64>
    %20 = stablehlo.slice %19 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %23 = stablehlo.concatenate %cst, %22, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %24 = stablehlo.slice %23 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.multiply %21, %25 : tensor<f64>
    %27 = stablehlo.slice %19 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %28 = stablehlo.reshape %27 : (tensor<1xf64>) -> tensor<f64>
    %29 = stablehlo.slice %23 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.multiply %28, %30 : tensor<f64>
    %32 = stablehlo.add %26, %31 : tensor<f64>
    %33 = stablehlo.slice %19 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %34 = stablehlo.reshape %33 : (tensor<1xf64>) -> tensor<f64>
    %35 = stablehlo.slice %23 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1xf64>) -> tensor<f64>
    %37 = stablehlo.multiply %34, %36 : tensor<f64>
    %38 = stablehlo.add %32, %37 : tensor<f64>
    %39 = stablehlo.slice %19 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %40 = stablehlo.reshape %39 : (tensor<1xf64>) -> tensor<f64>
    %41 = stablehlo.slice %23 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.multiply %40, %42 : tensor<f64>
    %44 = stablehlo.subtract %38, %43 : tensor<f64>
    %45 = stablehlo.reshape %44 : (tensor<f64>) -> tensor<1xf64>
    %46 = stablehlo.multiply %21, %42 : tensor<f64>
    %47 = stablehlo.multiply %28, %36 : tensor<f64>
    %48 = stablehlo.subtract %46, %47 : tensor<f64>
    %49 = stablehlo.multiply %34, %30 : tensor<f64>
    %50 = stablehlo.add %48, %49 : tensor<f64>
    %51 = stablehlo.multiply %40, %25 : tensor<f64>
    %52 = stablehlo.add %50, %51 : tensor<f64>
    %53 = stablehlo.reshape %52 : (tensor<f64>) -> tensor<1xf64>
    %54 = stablehlo.multiply %21, %36 : tensor<f64>
    %55 = stablehlo.multiply %28, %42 : tensor<f64>
    %56 = stablehlo.add %54, %55 : tensor<f64>
    %57 = stablehlo.multiply %34, %25 : tensor<f64>
    %58 = stablehlo.subtract %56, %57 : tensor<f64>
    %59 = stablehlo.multiply %40, %30 : tensor<f64>
    %60 = stablehlo.add %58, %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.multiply %21, %30 : tensor<f64>
    %63 = stablehlo.multiply %28, %25 : tensor<f64>
    %64 = stablehlo.subtract %62, %63 : tensor<f64>
    %65 = stablehlo.multiply %34, %42 : tensor<f64>
    %66 = stablehlo.subtract %64, %65 : tensor<f64>
    %67 = stablehlo.multiply %40, %36 : tensor<f64>
    %68 = stablehlo.subtract %66, %67 : tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.concatenate %45, %53, %61, %69, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %71 = stablehlo.slice %70 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %72 = stablehlo.reshape %71 : (tensor<1xf64>) -> tensor<f64>
    %73 = stablehlo.slice %19 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.negate %74 : tensor<f64>
    %76 = stablehlo.reshape %75 : (tensor<f64>) -> tensor<1xf64>
    %77 = stablehlo.slice %19 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.negate %78 : tensor<f64>
    %80 = stablehlo.reshape %79 : (tensor<f64>) -> tensor<1xf64>
    %81 = stablehlo.slice %19 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %82 = stablehlo.reshape %81 : (tensor<1xf64>) -> tensor<f64>
    %83 = stablehlo.negate %82 : tensor<f64>
    %84 = stablehlo.reshape %83 : (tensor<f64>) -> tensor<1xf64>
    %85 = stablehlo.slice %19 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %86 = stablehlo.reshape %85 : (tensor<1xf64>) -> tensor<f64>
    %87 = stablehlo.reshape %86 : (tensor<f64>) -> tensor<1xf64>
    %88 = stablehlo.concatenate %76, %80, %84, %87, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %89 = stablehlo.dot_general %19, %19, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %91 = stablehlo.divide %88, %90 : tensor<4xf64>
    %92 = stablehlo.slice %91 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.multiply %72, %93 : tensor<f64>
    %95 = stablehlo.slice %70 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %96 = stablehlo.reshape %95 : (tensor<1xf64>) -> tensor<f64>
    %97 = stablehlo.slice %91 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.multiply %96, %98 : tensor<f64>
    %100 = stablehlo.add %94, %99 : tensor<f64>
    %101 = stablehlo.slice %70 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %102 = stablehlo.reshape %101 : (tensor<1xf64>) -> tensor<f64>
    %103 = stablehlo.slice %91 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
    %105 = stablehlo.multiply %102, %104 : tensor<f64>
    %106 = stablehlo.add %100, %105 : tensor<f64>
    %107 = stablehlo.slice %70 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %108 = stablehlo.reshape %107 : (tensor<1xf64>) -> tensor<f64>
    %109 = stablehlo.slice %91 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %110 = stablehlo.reshape %109 : (tensor<1xf64>) -> tensor<f64>
    %111 = stablehlo.multiply %108, %110 : tensor<f64>
    %112 = stablehlo.subtract %106, %111 : tensor<f64>
    %113 = stablehlo.reshape %112 : (tensor<f64>) -> tensor<1xf64>
    %114 = stablehlo.multiply %72, %110 : tensor<f64>
    %115 = stablehlo.multiply %96, %104 : tensor<f64>
    %116 = stablehlo.subtract %114, %115 : tensor<f64>
    %117 = stablehlo.multiply %102, %98 : tensor<f64>
    %118 = stablehlo.add %116, %117 : tensor<f64>
    %119 = stablehlo.multiply %108, %93 : tensor<f64>
    %120 = stablehlo.add %118, %119 : tensor<f64>
    %121 = stablehlo.reshape %120 : (tensor<f64>) -> tensor<1xf64>
    %122 = stablehlo.multiply %72, %104 : tensor<f64>
    %123 = stablehlo.multiply %96, %110 : tensor<f64>
    %124 = stablehlo.add %122, %123 : tensor<f64>
    %125 = stablehlo.multiply %102, %93 : tensor<f64>
    %126 = stablehlo.subtract %124, %125 : tensor<f64>
    %127 = stablehlo.multiply %108, %98 : tensor<f64>
    %128 = stablehlo.add %126, %127 : tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.multiply %72, %98 : tensor<f64>
    %131 = stablehlo.multiply %96, %93 : tensor<f64>
    %132 = stablehlo.subtract %130, %131 : tensor<f64>
    %133 = stablehlo.multiply %102, %110 : tensor<f64>
    %134 = stablehlo.subtract %132, %133 : tensor<f64>
    %135 = stablehlo.multiply %108, %104 : tensor<f64>
    %136 = stablehlo.subtract %134, %135 : tensor<f64>
    %137 = stablehlo.reshape %136 : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.concatenate %113, %121, %129, %137, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %139 = stablehlo.slice %138 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.reshape %140 : (tensor<f64>) -> tensor<1xf64>
    %142 = stablehlo.slice %138 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %143 = stablehlo.reshape %142 : (tensor<1xf64>) -> tensor<f64>
    %144 = stablehlo.reshape %143 : (tensor<f64>) -> tensor<1xf64>
    %145 = stablehlo.slice %138 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %146 = stablehlo.reshape %145 : (tensor<1xf64>) -> tensor<f64>
    %147 = stablehlo.reshape %146 : (tensor<f64>) -> tensor<1xf64>
    %148 = stablehlo.concatenate %141, %144, %147, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %149 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %150 = call @_threefry_fold_in(%c, %149) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_1 = stablehlo.constant dense<1.000000e-04> : tensor<f64>
    %151 = stablehlo.sqrt %cst_1 : tensor<f64>
    %152 = call @_normal(%150) : (tensor<2xui32>) -> tensor<3xf64>
    %153 = stablehlo.convert %151 : tensor<f64>
    %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %155 = stablehlo.multiply %154, %152 : tensor<3xf64>
    %156 = stablehlo.add %148, %155 : tensor<3xf64>
    %157 = stablehlo.add %156, %arg2 : tensor<3xf64>
    %c_2 = stablehlo.constant dense<9> : tensor<i64>
    %158 = call @remainder_205(%arg0, %c_2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %159 = stablehlo.compare  EQ, %158, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %160 = stablehlo.convert %159 : (tensor<i1>) -> tensor<i32>
    %161 = "stablehlo.case"(%160) ({
      stablehlo.return %arg3 : tensor<3xf64>
    }, {
      stablehlo.return %157 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    return %161 : tensor<3xf64>
  }
  func.func private @inner_228(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg1 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2 = stablehlo.slice %1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.negate %3 : tensor<f64>
    %5 = stablehlo.reshape %4 : (tensor<f64>) -> tensor<1xf64>
    %6 = stablehlo.slice %1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.reshape %15 : (tensor<f64>) -> tensor<1xf64>
    %17 = stablehlo.concatenate %5, %9, %13, %16, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %18 = stablehlo.dot_general %1, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %20 = stablehlo.divide %17, %19 : tensor<4xf64>
    %21 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %23 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.concatenate %0, %23, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %25 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %27 = stablehlo.multiply %22, %26 : tensor<f64>
    %28 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
    %30 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.multiply %29, %31 : tensor<f64>
    %33 = stablehlo.add %27, %32 : tensor<f64>
    %34 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %37 = stablehlo.reshape %36 : (tensor<1xf64>) -> tensor<f64>
    %38 = stablehlo.multiply %35, %37 : tensor<f64>
    %39 = stablehlo.add %33, %38 : tensor<f64>
    %40 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %43 = stablehlo.reshape %42 : (tensor<1xf64>) -> tensor<f64>
    %44 = stablehlo.multiply %41, %43 : tensor<f64>
    %45 = stablehlo.subtract %39, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %22, %43 : tensor<f64>
    %48 = stablehlo.multiply %29, %37 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %35, %31 : tensor<f64>
    %51 = stablehlo.add %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %41, %26 : tensor<f64>
    %53 = stablehlo.add %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.multiply %22, %37 : tensor<f64>
    %56 = stablehlo.multiply %29, %43 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.multiply %35, %26 : tensor<f64>
    %59 = stablehlo.subtract %57, %58 : tensor<f64>
    %60 = stablehlo.multiply %41, %31 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.reshape %61 : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.multiply %22, %31 : tensor<f64>
    %64 = stablehlo.multiply %29, %26 : tensor<f64>
    %65 = stablehlo.subtract %63, %64 : tensor<f64>
    %66 = stablehlo.multiply %35, %43 : tensor<f64>
    %67 = stablehlo.subtract %65, %66 : tensor<f64>
    %68 = stablehlo.multiply %41, %37 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.reshape %69 : (tensor<f64>) -> tensor<1xf64>
    %71 = stablehlo.concatenate %46, %54, %62, %70, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %72 = stablehlo.slice %71 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1xf64>) -> tensor<f64>
    %74 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.negate %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.reshape %87 : (tensor<f64>) -> tensor<1xf64>
    %89 = stablehlo.concatenate %77, %81, %85, %88, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %90 = stablehlo.dot_general %20, %20, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %92 = stablehlo.divide %89, %91 : tensor<4xf64>
    %93 = stablehlo.slice %92 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %94 = stablehlo.reshape %93 : (tensor<1xf64>) -> tensor<f64>
    %95 = stablehlo.multiply %73, %94 : tensor<f64>
    %96 = stablehlo.slice %71 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.slice %92 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.multiply %97, %99 : tensor<f64>
    %101 = stablehlo.add %95, %100 : tensor<f64>
    %102 = stablehlo.slice %71 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.slice %92 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %105 = stablehlo.reshape %104 : (tensor<1xf64>) -> tensor<f64>
    %106 = stablehlo.multiply %103, %105 : tensor<f64>
    %107 = stablehlo.add %101, %106 : tensor<f64>
    %108 = stablehlo.slice %71 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.slice %92 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %111 = stablehlo.reshape %110 : (tensor<1xf64>) -> tensor<f64>
    %112 = stablehlo.multiply %109, %111 : tensor<f64>
    %113 = stablehlo.subtract %107, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %73, %111 : tensor<f64>
    %116 = stablehlo.multiply %97, %105 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %103, %99 : tensor<f64>
    %119 = stablehlo.add %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %109, %94 : tensor<f64>
    %121 = stablehlo.add %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.multiply %73, %105 : tensor<f64>
    %124 = stablehlo.multiply %97, %111 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.multiply %103, %94 : tensor<f64>
    %127 = stablehlo.subtract %125, %126 : tensor<f64>
    %128 = stablehlo.multiply %109, %99 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.reshape %129 : (tensor<f64>) -> tensor<1xf64>
    %131 = stablehlo.multiply %73, %99 : tensor<f64>
    %132 = stablehlo.multiply %97, %94 : tensor<f64>
    %133 = stablehlo.subtract %131, %132 : tensor<f64>
    %134 = stablehlo.multiply %103, %111 : tensor<f64>
    %135 = stablehlo.subtract %133, %134 : tensor<f64>
    %136 = stablehlo.multiply %109, %105 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.reshape %137 : (tensor<f64>) -> tensor<1xf64>
    %139 = stablehlo.concatenate %114, %122, %130, %138, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %140 = stablehlo.slice %139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %141 = stablehlo.reshape %140 : (tensor<1xf64>) -> tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.slice %139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %144 = stablehlo.reshape %143 : (tensor<1xf64>) -> tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.slice %139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %147 = stablehlo.reshape %146 : (tensor<1xf64>) -> tensor<f64>
    %148 = stablehlo.reshape %147 : (tensor<f64>) -> tensor<1xf64>
    %149 = stablehlo.concatenate %142, %145, %148, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %149 : tensor<3xf64>
  }
  func.func private @inner_229(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4xf64>
    %cst_0 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %3 = stablehlo.multiply %1, %2 : tensor<4xf64>
    %cst_1 = stablehlo.constant dense<6.000000e+01> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %5 = stablehlo.divide %3, %4 : tensor<4xf64>
    return %5 : tensor<4xf64>
  }
  func.func private @inner_230(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
    %cst = stablehlo.constant dense<"0x4F621058E9089240DCB5847C78179240CE66D5E79A369240AA8BDB689C46924071CE88D2026692409A081B9E2A769240FE21FDF6CD9492402AF697DDD3A492405A17B7D134C49240837CD0B391D39240C1CAA1454AF39240565BB1BFF4019340A267B3EAEB229340B8AF03E71C30934098DD93875D5193408E31772DA55E9340F1F44A597E809340E0E995B2208D93407AA52C430CB093401CC9E53FF8BC9340184850FC3CDF9340D2BCE31401EC9340014D840D270E944026E4839EF91A944075E09C112D3E944052FC1873034B94405917B7D1346D9440131DC9E59B7A9440B6627FD9B19B944010A5BDC16BA994403A234A7BCBCA94408BB96B093DD894406B787AA540FA944012C7BAB851079540780B2428A229954029A913D0B0369540DE02098AC7589540B27BF2B0EC6595404BEA04344D88954082C0CAA1DD95954061C3D32B01B89540B71E85EB45C59540E926310898E79540DE9387859EF495409E5E29CBF416964006F01648282496403108AC1C16469640265C8FC2ED5396404703780B3C75964088B0E1E955829640D95F764F72A496401EF46C56ADB19640A2B437F8A6D496406054522780E0964063AA6054BA0397406B09F9A0AB0F974076BE9F1AEB33974066B3EA73A53E974055302AA96F649740FBA9F1D2616D97402A3A92CB0F94974060545227949C97404950FC182BC49740A635CD3B12CC9740DF718A8E5CF49740F97E6ABCFCFA9740A0CDAACF2D249840F6E461A13E2A9840635DDC46B3539840EDC9C342795A98407D3F355E4A8398402EFF21FD028A98406F3480B7B8B1984043696FF001B99840FF43FAEDD3E09840273108AC4CE89840C64B378919109940A60A4625C5169940E07A14AEC33E9940A64E4013F94499409E3C2CD4D66D9940AE03E78C00749940287E8CB95B9D99405EC3D32BC5A2994070F085C924CD9940FACBEEC933D19940B27BF2B034FC9940769CA223E5FF9940BBB88D06902B9A402EDD24062D2F9A40F6065F98AC5A9A400B4FAF94915E9A4025287E8CCD8A9A408DB96B09F58D9A405B423EE8F5B99A40ADFA5C6D79BD9A40302AA91354E99A40156A4DF346EC9A4050AF9465C4189B40BB96900FD61B9B402B1895D4A1479B40FDB27BF2984B9B4036CD3B4E25779B40FD1873D78A7A9B402506819547A69B402D431CEBD6A99B403892CB7FD4D59B403D2CD49A76D99B40F0C9C34231069C40AEB6627FE9089C4058CA32C4E9359C409DCDAACFF5379C402575029A84649C405A643BDF93679C4063105839C8939C400D0BB5A681979C405A17B7D1DCC29C40BE9F1A2F3DC69C4062E5D0223BF29C40557424976BF59C40"> : tensor<120xf64>
    %cst_0 = stablehlo.constant dense<"0x0000000000ACAA40000000000038AD400000000000CCAE400000000000C8B04000000000009AB1400000000000E7B2400000000000B4B3400000000000F5B4400000000000FCB54000000000000DB740000000000016B840000000000003B94000000000003ABA400000000000F1BA40000000000043BC40000000000007BD40000000000043BE400000000000D2BE40000000008023C040000000000063C040000000000019C140000000008059C140000000000018C24000000000003EC24000000000800FC340000000008034C3400000000080FFC34000000000001EC4400000000000E7C4400000000080E8C4400000000000CBC5400000000080D8C5400000000000A5C6400000000000AFC640000000000083C740000000008089C74000000000006DC84000000000804CC84000000000003FC940000000008013C94000000000800FCA400000000000DAC9400000000000D6CA400000000080A1CA4000000000009FCB4000000000806BCB4000000000006BCC40000000008027CC40000000000033CD400000000000E9CC400000000080EECD400000000080A4CD400000000000B8CE4000000000005DCE40000000000076CF4000000000801BCF40000000004019D0400000000080CDCF40000000008079D04000000000C03FD04000000000C0E2D04000000000C0B1D040000000004041D140000000008001D14000000000C0ABD14000000000805ED14000000000800BD2400000000000C6D140000000000072D24000000000802FD2400000000000DBD240000000000096D24000000000403FD3400000000080EDD2400000000040AAD34000000000C04FD34000000000C00AD4400000000080B5D34000000000406FD440000000000022D4400000000040D4D44000000000C07CD44000000000C041D5400000000000E5D4400000000080AED54000000000C04AD540000000008065D64000000000C0C8D540000000000007D74000000000C09AD640000000008078D740000000004006D7400000000080E1D74000000000C076D740000000004055D8400000000000F6D7400000000000D6D84000000000006FD840000000008044D9400000000080E3D8400000000000BBD94000000000C051D940000000008024DA400000000040BCD940000000008081DA40000000008025DA400000000080E6DA40000000004081DA40000000008046DB400000000000E7DA40000000004098DB40000000000042DB4000000000C0EBDB40000000004097DB4000000000C03ADC400000000080F0DB4000000000007FDC40000000004039DC400000000040C4DC4000000000C087DC40"> : tensor<120xf64>
    %cst_1 = stablehlo.constant dense<[-5.000000e-01, -5.000000e-01, 5.000000e-01, 5.000000e-01]> : tensor<4xf64>
    %0 = call @_interp(%arg0, %cst, %cst_0) : (tensor<4xf64>, tensor<120xf64>, tensor<120xf64>) -> tensor<4xf64>
    %1 = stablehlo.subtract %0, %arg3 : tensor<4xf64>
    %cst_2 = stablehlo.constant dense<0.01098901098901099> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %3 = stablehlo.multiply %2, %1 : tensor<4xf64>
    %4 = stablehlo.add %arg3, %3 : tensor<4xf64>
    %5 = stablehlo.multiply %4, %4 : tensor<4xf64>
    %cst_3 = stablehlo.constant dense<9.9068131782640682E-9> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %7 = stablehlo.multiply %5, %6 : tensor<4xf64>
    %8 = stablehlo.multiply %4, %4 : tensor<4xf64>
    %cst_4 = stablehlo.constant dense<9.8192338453001589E-11> : tensor<f64>
    %9 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %10 = stablehlo.multiply %8, %9 : tensor<4xf64>
    %11 = stablehlo.multiply %10, %cst_1 : tensor<4xf64>
    return %7, %11, %4 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
  }
  func.func private @inner_231(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, -9.810000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg0 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %3 = stablehlo.multiply %cst, %2 : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %5 = stablehlo.concatenate %4, %3, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %6 = stablehlo.add %arg1, %5 : tensor<6xf64>
    return %6 : tensor<6xf64>
  }
  func.func private @inner_232(%arg0: tensor<6xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.negate %0 : tensor<3xf64>
    %2 = call @norm(%1) : (tensor<3xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %4 = stablehlo.multiply %3, %1 : tensor<3xf64>
    %5 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.multiply %4, %5 : tensor<3xf64>
    return %6 : tensor<3xf64>
  }
  func.func private @inner_233(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[[0.000000e+00, -0.087155742747658165, 0.99619469809174543], [-0.065403129230143062, 0.086969135612238901, 0.99406176877383478], [0.000000e+00, 0.087155742747658165, 0.99619469809174543], [-0.065403129230143062, -0.086969135612238901, 0.99406176877383478]]> : tensor<4x3xf64>
    %cst_0 = stablehlo.constant dense<[[-0.20858424832311181, -0.25901062150385384, -0.022660493114391125], [0.19843360999226459, 0.25704989260274902, -0.0094332447193082886], [0.20858424832311181, -0.25901062150385384, 0.022660493114391125], [-0.19843360999226459, 0.25704989260274902, 0.0094332447193082886]]> : tensor<4x3xf64>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %2 = stablehlo.multiply %cst, %1 : tensor<4x3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %3 = stablehlo.reduce(%2 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %6 = stablehlo.multiply %cst, %5 : tensor<4x3xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.reduce(%6 init: %cst_2) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf64>) -> tensor<4x1xf64>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<4x1xf64>) -> tensor<4x3xf64>
    %10 = stablehlo.multiply %cst_0, %9 : tensor<4x3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %11 = stablehlo.reduce(%10 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<4x3xf64>, tensor<f64>) -> tensor<3xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %13 = stablehlo.concatenate %12, %3, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %15 = stablehlo.concatenate %7, %14, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %16 = stablehlo.add %13, %15 : tensor<6xf64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %17 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %18 = stablehlo.concatenate %11, %17, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %19 = stablehlo.add %16, %18 : tensor<6xf64>
    return %19 : tensor<6xf64>
  }
  func.func private @inner_234(%arg0: tensor<6xf64>, %arg1: tensor<3xf64>, %arg2: tensor<7xf64>, %arg3: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1 = stablehlo.concatenate %0, %arg1, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<6xf64>
    %3 = stablehlo.slice %arg2 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %4 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %arg0 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %9 = stablehlo.slice %8 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
    %11 = stablehlo.multiply %5, %10 : tensor<f64>
    %12 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %13 = stablehlo.reshape %12 : (tensor<1xf64>) -> tensor<f64>
    %14 = stablehlo.slice %8 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.multiply %13, %15 : tensor<f64>
    %17 = stablehlo.add %11, %16 : tensor<f64>
    %18 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.slice %8 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.multiply %19, %21 : tensor<f64>
    %23 = stablehlo.add %17, %22 : tensor<f64>
    %24 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.slice %8 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
    %28 = stablehlo.multiply %25, %27 : tensor<f64>
    %29 = stablehlo.subtract %23, %28 : tensor<f64>
    %30 = stablehlo.reshape %29 : (tensor<f64>) -> tensor<1xf64>
    %31 = stablehlo.multiply %5, %27 : tensor<f64>
    %32 = stablehlo.multiply %13, %21 : tensor<f64>
    %33 = stablehlo.subtract %31, %32 : tensor<f64>
    %34 = stablehlo.multiply %19, %15 : tensor<f64>
    %35 = stablehlo.add %33, %34 : tensor<f64>
    %36 = stablehlo.multiply %25, %10 : tensor<f64>
    %37 = stablehlo.add %35, %36 : tensor<f64>
    %38 = stablehlo.reshape %37 : (tensor<f64>) -> tensor<1xf64>
    %39 = stablehlo.multiply %5, %21 : tensor<f64>
    %40 = stablehlo.multiply %13, %27 : tensor<f64>
    %41 = stablehlo.add %39, %40 : tensor<f64>
    %42 = stablehlo.multiply %19, %10 : tensor<f64>
    %43 = stablehlo.subtract %41, %42 : tensor<f64>
    %44 = stablehlo.multiply %25, %15 : tensor<f64>
    %45 = stablehlo.add %43, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %5, %15 : tensor<f64>
    %48 = stablehlo.multiply %13, %10 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %19, %27 : tensor<f64>
    %51 = stablehlo.subtract %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %25, %21 : tensor<f64>
    %53 = stablehlo.subtract %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.concatenate %30, %38, %46, %54, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %56 = stablehlo.slice %55 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %57 = stablehlo.reshape %56 : (tensor<1xf64>) -> tensor<f64>
    %58 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %59 = stablehlo.reshape %58 : (tensor<1xf64>) -> tensor<f64>
    %60 = stablehlo.negate %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %63 = stablehlo.reshape %62 : (tensor<1xf64>) -> tensor<f64>
    %64 = stablehlo.negate %63 : tensor<f64>
    %65 = stablehlo.reshape %64 : (tensor<f64>) -> tensor<1xf64>
    %66 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %67 = stablehlo.reshape %66 : (tensor<1xf64>) -> tensor<f64>
    %68 = stablehlo.negate %67 : tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %71 = stablehlo.reshape %70 : (tensor<1xf64>) -> tensor<f64>
    %72 = stablehlo.reshape %71 : (tensor<f64>) -> tensor<1xf64>
    %73 = stablehlo.concatenate %61, %65, %69, %72, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %74 = stablehlo.dot_general %3, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %75 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %76 = stablehlo.divide %73, %75 : tensor<4xf64>
    %77 = stablehlo.slice %76 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.multiply %57, %78 : tensor<f64>
    %80 = stablehlo.slice %55 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %81 = stablehlo.reshape %80 : (tensor<1xf64>) -> tensor<f64>
    %82 = stablehlo.slice %76 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.multiply %81, %83 : tensor<f64>
    %85 = stablehlo.add %79, %84 : tensor<f64>
    %86 = stablehlo.slice %55 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.slice %76 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %89 = stablehlo.reshape %88 : (tensor<1xf64>) -> tensor<f64>
    %90 = stablehlo.multiply %87, %89 : tensor<f64>
    %91 = stablehlo.add %85, %90 : tensor<f64>
    %92 = stablehlo.slice %55 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.slice %76 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %95 = stablehlo.reshape %94 : (tensor<1xf64>) -> tensor<f64>
    %96 = stablehlo.multiply %93, %95 : tensor<f64>
    %97 = stablehlo.subtract %91, %96 : tensor<f64>
    %98 = stablehlo.reshape %97 : (tensor<f64>) -> tensor<1xf64>
    %99 = stablehlo.multiply %57, %95 : tensor<f64>
    %100 = stablehlo.multiply %81, %89 : tensor<f64>
    %101 = stablehlo.subtract %99, %100 : tensor<f64>
    %102 = stablehlo.multiply %87, %83 : tensor<f64>
    %103 = stablehlo.add %101, %102 : tensor<f64>
    %104 = stablehlo.multiply %93, %78 : tensor<f64>
    %105 = stablehlo.add %103, %104 : tensor<f64>
    %106 = stablehlo.reshape %105 : (tensor<f64>) -> tensor<1xf64>
    %107 = stablehlo.multiply %57, %89 : tensor<f64>
    %108 = stablehlo.multiply %81, %95 : tensor<f64>
    %109 = stablehlo.add %107, %108 : tensor<f64>
    %110 = stablehlo.multiply %87, %78 : tensor<f64>
    %111 = stablehlo.subtract %109, %110 : tensor<f64>
    %112 = stablehlo.multiply %93, %83 : tensor<f64>
    %113 = stablehlo.add %111, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %57, %83 : tensor<f64>
    %116 = stablehlo.multiply %81, %78 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %87, %95 : tensor<f64>
    %119 = stablehlo.subtract %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %93, %89 : tensor<f64>
    %121 = stablehlo.subtract %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.concatenate %98, %106, %114, %122, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %124 = stablehlo.slice %123 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %125 = stablehlo.reshape %124 : (tensor<1xf64>) -> tensor<f64>
    %126 = stablehlo.reshape %125 : (tensor<f64>) -> tensor<1xf64>
    %127 = stablehlo.slice %123 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %128 = stablehlo.reshape %127 : (tensor<1xf64>) -> tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.slice %123 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %131 = stablehlo.reshape %130 : (tensor<1xf64>) -> tensor<f64>
    %132 = stablehlo.reshape %131 : (tensor<f64>) -> tensor<1xf64>
    %133 = stablehlo.concatenate %126, %129, %132, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %134 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %135 = stablehlo.reshape %134 : (tensor<1xf64>) -> tensor<f64>
    %136 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %137 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.concatenate %136, %137, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %139 = stablehlo.slice %138 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.multiply %135, %140 : tensor<f64>
    %142 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %143 = stablehlo.reshape %142 : (tensor<1xf64>) -> tensor<f64>
    %144 = stablehlo.slice %138 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.multiply %143, %145 : tensor<f64>
    %147 = stablehlo.add %141, %146 : tensor<f64>
    %148 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %149 = stablehlo.reshape %148 : (tensor<1xf64>) -> tensor<f64>
    %150 = stablehlo.slice %138 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
    %152 = stablehlo.multiply %149, %151 : tensor<f64>
    %153 = stablehlo.add %147, %152 : tensor<f64>
    %154 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %155 = stablehlo.reshape %154 : (tensor<1xf64>) -> tensor<f64>
    %156 = stablehlo.slice %138 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %157 = stablehlo.reshape %156 : (tensor<1xf64>) -> tensor<f64>
    %158 = stablehlo.multiply %155, %157 : tensor<f64>
    %159 = stablehlo.subtract %153, %158 : tensor<f64>
    %160 = stablehlo.reshape %159 : (tensor<f64>) -> tensor<1xf64>
    %161 = stablehlo.multiply %135, %157 : tensor<f64>
    %162 = stablehlo.multiply %143, %151 : tensor<f64>
    %163 = stablehlo.subtract %161, %162 : tensor<f64>
    %164 = stablehlo.multiply %149, %145 : tensor<f64>
    %165 = stablehlo.add %163, %164 : tensor<f64>
    %166 = stablehlo.multiply %155, %140 : tensor<f64>
    %167 = stablehlo.add %165, %166 : tensor<f64>
    %168 = stablehlo.reshape %167 : (tensor<f64>) -> tensor<1xf64>
    %169 = stablehlo.multiply %135, %151 : tensor<f64>
    %170 = stablehlo.multiply %143, %157 : tensor<f64>
    %171 = stablehlo.add %169, %170 : tensor<f64>
    %172 = stablehlo.multiply %149, %140 : tensor<f64>
    %173 = stablehlo.subtract %171, %172 : tensor<f64>
    %174 = stablehlo.multiply %155, %145 : tensor<f64>
    %175 = stablehlo.add %173, %174 : tensor<f64>
    %176 = stablehlo.reshape %175 : (tensor<f64>) -> tensor<1xf64>
    %177 = stablehlo.multiply %135, %145 : tensor<f64>
    %178 = stablehlo.multiply %143, %140 : tensor<f64>
    %179 = stablehlo.subtract %177, %178 : tensor<f64>
    %180 = stablehlo.multiply %149, %157 : tensor<f64>
    %181 = stablehlo.subtract %179, %180 : tensor<f64>
    %182 = stablehlo.multiply %155, %151 : tensor<f64>
    %183 = stablehlo.subtract %181, %182 : tensor<f64>
    %184 = stablehlo.reshape %183 : (tensor<f64>) -> tensor<1xf64>
    %185 = stablehlo.concatenate %160, %168, %176, %184, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %186 = stablehlo.slice %185 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %187 = stablehlo.reshape %186 : (tensor<1xf64>) -> tensor<f64>
    %188 = stablehlo.slice %3 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %189 = stablehlo.reshape %188 : (tensor<1xf64>) -> tensor<f64>
    %190 = stablehlo.negate %189 : tensor<f64>
    %191 = stablehlo.reshape %190 : (tensor<f64>) -> tensor<1xf64>
    %192 = stablehlo.slice %3 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %193 = stablehlo.reshape %192 : (tensor<1xf64>) -> tensor<f64>
    %194 = stablehlo.negate %193 : tensor<f64>
    %195 = stablehlo.reshape %194 : (tensor<f64>) -> tensor<1xf64>
    %196 = stablehlo.slice %3 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %197 = stablehlo.reshape %196 : (tensor<1xf64>) -> tensor<f64>
    %198 = stablehlo.negate %197 : tensor<f64>
    %199 = stablehlo.reshape %198 : (tensor<f64>) -> tensor<1xf64>
    %200 = stablehlo.slice %3 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %201 = stablehlo.reshape %200 : (tensor<1xf64>) -> tensor<f64>
    %202 = stablehlo.reshape %201 : (tensor<f64>) -> tensor<1xf64>
    %203 = stablehlo.concatenate %191, %195, %199, %202, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %204 = stablehlo.dot_general %3, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %205 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %206 = stablehlo.divide %203, %205 : tensor<4xf64>
    %207 = stablehlo.slice %206 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %208 = stablehlo.reshape %207 : (tensor<1xf64>) -> tensor<f64>
    %209 = stablehlo.multiply %187, %208 : tensor<f64>
    %210 = stablehlo.slice %185 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %211 = stablehlo.reshape %210 : (tensor<1xf64>) -> tensor<f64>
    %212 = stablehlo.slice %206 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %213 = stablehlo.reshape %212 : (tensor<1xf64>) -> tensor<f64>
    %214 = stablehlo.multiply %211, %213 : tensor<f64>
    %215 = stablehlo.add %209, %214 : tensor<f64>
    %216 = stablehlo.slice %185 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %217 = stablehlo.reshape %216 : (tensor<1xf64>) -> tensor<f64>
    %218 = stablehlo.slice %206 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %219 = stablehlo.reshape %218 : (tensor<1xf64>) -> tensor<f64>
    %220 = stablehlo.multiply %217, %219 : tensor<f64>
    %221 = stablehlo.add %215, %220 : tensor<f64>
    %222 = stablehlo.slice %185 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %223 = stablehlo.reshape %222 : (tensor<1xf64>) -> tensor<f64>
    %224 = stablehlo.slice %206 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %225 = stablehlo.reshape %224 : (tensor<1xf64>) -> tensor<f64>
    %226 = stablehlo.multiply %223, %225 : tensor<f64>
    %227 = stablehlo.subtract %221, %226 : tensor<f64>
    %228 = stablehlo.reshape %227 : (tensor<f64>) -> tensor<1xf64>
    %229 = stablehlo.multiply %187, %225 : tensor<f64>
    %230 = stablehlo.multiply %211, %219 : tensor<f64>
    %231 = stablehlo.subtract %229, %230 : tensor<f64>
    %232 = stablehlo.multiply %217, %213 : tensor<f64>
    %233 = stablehlo.add %231, %232 : tensor<f64>
    %234 = stablehlo.multiply %223, %208 : tensor<f64>
    %235 = stablehlo.add %233, %234 : tensor<f64>
    %236 = stablehlo.reshape %235 : (tensor<f64>) -> tensor<1xf64>
    %237 = stablehlo.multiply %187, %219 : tensor<f64>
    %238 = stablehlo.multiply %211, %225 : tensor<f64>
    %239 = stablehlo.add %237, %238 : tensor<f64>
    %240 = stablehlo.multiply %217, %208 : tensor<f64>
    %241 = stablehlo.subtract %239, %240 : tensor<f64>
    %242 = stablehlo.multiply %223, %213 : tensor<f64>
    %243 = stablehlo.add %241, %242 : tensor<f64>
    %244 = stablehlo.reshape %243 : (tensor<f64>) -> tensor<1xf64>
    %245 = stablehlo.multiply %187, %213 : tensor<f64>
    %246 = stablehlo.multiply %211, %208 : tensor<f64>
    %247 = stablehlo.subtract %245, %246 : tensor<f64>
    %248 = stablehlo.multiply %217, %225 : tensor<f64>
    %249 = stablehlo.subtract %247, %248 : tensor<f64>
    %250 = stablehlo.multiply %223, %219 : tensor<f64>
    %251 = stablehlo.subtract %249, %250 : tensor<f64>
    %252 = stablehlo.reshape %251 : (tensor<f64>) -> tensor<1xf64>
    %253 = stablehlo.concatenate %228, %236, %244, %252, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %254 = stablehlo.slice %253 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %255 = stablehlo.reshape %254 : (tensor<1xf64>) -> tensor<f64>
    %256 = stablehlo.reshape %255 : (tensor<f64>) -> tensor<1xf64>
    %257 = stablehlo.slice %253 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %258 = stablehlo.reshape %257 : (tensor<1xf64>) -> tensor<f64>
    %259 = stablehlo.reshape %258 : (tensor<f64>) -> tensor<1xf64>
    %260 = stablehlo.slice %253 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %261 = stablehlo.reshape %260 : (tensor<1xf64>) -> tensor<f64>
    %262 = stablehlo.reshape %261 : (tensor<f64>) -> tensor<1xf64>
    %263 = stablehlo.concatenate %256, %259, %262, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %264 = stablehlo.concatenate %133, %263, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %265 = stablehlo.add %2, %264 : tensor<6xf64>
    return %265 : tensor<6xf64>
  }
  func.func private @inner_235(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    return %0 : tensor<i64>
  }
  func.func private @inner_236(%arg0: tensor<i64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<[1797259609, 2579123966]> : tensor<2xui32>
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %1 = call @_threefry_fold_in(%c, %0) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %2 = stablehlo.sqrt %cst : tensor<f64>
    %3 = call @_normal(%1) : (tensor<2xui32>) -> tensor<3xf64>
    %4 = stablehlo.convert %2 : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %6 = stablehlo.multiply %5, %3 : tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.0011111111111111111> : tensor<f64>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %8 = stablehlo.multiply %6, %7 : tensor<3xf64>
    %9 = stablehlo.add %arg1, %8 : tensor<3xf64>
    return %9 : tensor<3xf64>
  }
  func.func private @inner_237(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<6xf64>, %arg3: tensor<4x3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %c = stablehlo.constant dense<[1797259609, 2579123966]> : tensor<2xui32>
    %cst = stablehlo.constant dense<[0.016209783477834406, 0.032419566955668812, 0.016209783477834406, -1.6089340340646645, 0.67377316797600217]> : tensor<5xf64>
    %0 = stablehlo.slice %arg2 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2 = stablehlo.slice %1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.negate %3 : tensor<f64>
    %5 = stablehlo.reshape %4 : (tensor<f64>) -> tensor<1xf64>
    %6 = stablehlo.slice %1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.reshape %15 : (tensor<f64>) -> tensor<1xf64>
    %17 = stablehlo.concatenate %5, %9, %13, %16, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %18 = stablehlo.dot_general %1, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %20 = stablehlo.divide %17, %19 : tensor<4xf64>
    %21 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %23 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.concatenate %0, %23, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %25 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %27 = stablehlo.multiply %22, %26 : tensor<f64>
    %28 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
    %30 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.multiply %29, %31 : tensor<f64>
    %33 = stablehlo.add %27, %32 : tensor<f64>
    %34 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %37 = stablehlo.reshape %36 : (tensor<1xf64>) -> tensor<f64>
    %38 = stablehlo.multiply %35, %37 : tensor<f64>
    %39 = stablehlo.add %33, %38 : tensor<f64>
    %40 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %43 = stablehlo.reshape %42 : (tensor<1xf64>) -> tensor<f64>
    %44 = stablehlo.multiply %41, %43 : tensor<f64>
    %45 = stablehlo.subtract %39, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %22, %43 : tensor<f64>
    %48 = stablehlo.multiply %29, %37 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %35, %31 : tensor<f64>
    %51 = stablehlo.add %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %41, %26 : tensor<f64>
    %53 = stablehlo.add %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.multiply %22, %37 : tensor<f64>
    %56 = stablehlo.multiply %29, %43 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.multiply %35, %26 : tensor<f64>
    %59 = stablehlo.subtract %57, %58 : tensor<f64>
    %60 = stablehlo.multiply %41, %31 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.reshape %61 : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.multiply %22, %31 : tensor<f64>
    %64 = stablehlo.multiply %29, %26 : tensor<f64>
    %65 = stablehlo.subtract %63, %64 : tensor<f64>
    %66 = stablehlo.multiply %35, %43 : tensor<f64>
    %67 = stablehlo.subtract %65, %66 : tensor<f64>
    %68 = stablehlo.multiply %41, %37 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.reshape %69 : (tensor<f64>) -> tensor<1xf64>
    %71 = stablehlo.concatenate %46, %54, %62, %70, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %72 = stablehlo.slice %71 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1xf64>) -> tensor<f64>
    %74 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.negate %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.reshape %87 : (tensor<f64>) -> tensor<1xf64>
    %89 = stablehlo.concatenate %77, %81, %85, %88, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %90 = stablehlo.dot_general %20, %20, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %92 = stablehlo.divide %89, %91 : tensor<4xf64>
    %93 = stablehlo.slice %92 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %94 = stablehlo.reshape %93 : (tensor<1xf64>) -> tensor<f64>
    %95 = stablehlo.multiply %73, %94 : tensor<f64>
    %96 = stablehlo.slice %71 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.slice %92 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.multiply %97, %99 : tensor<f64>
    %101 = stablehlo.add %95, %100 : tensor<f64>
    %102 = stablehlo.slice %71 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.slice %92 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %105 = stablehlo.reshape %104 : (tensor<1xf64>) -> tensor<f64>
    %106 = stablehlo.multiply %103, %105 : tensor<f64>
    %107 = stablehlo.add %101, %106 : tensor<f64>
    %108 = stablehlo.slice %71 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.slice %92 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %111 = stablehlo.reshape %110 : (tensor<1xf64>) -> tensor<f64>
    %112 = stablehlo.multiply %109, %111 : tensor<f64>
    %113 = stablehlo.subtract %107, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %73, %111 : tensor<f64>
    %116 = stablehlo.multiply %97, %105 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %103, %99 : tensor<f64>
    %119 = stablehlo.add %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %109, %94 : tensor<f64>
    %121 = stablehlo.add %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.multiply %73, %105 : tensor<f64>
    %124 = stablehlo.multiply %97, %111 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.multiply %103, %94 : tensor<f64>
    %127 = stablehlo.subtract %125, %126 : tensor<f64>
    %128 = stablehlo.multiply %109, %99 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.reshape %129 : (tensor<f64>) -> tensor<1xf64>
    %131 = stablehlo.multiply %73, %99 : tensor<f64>
    %132 = stablehlo.multiply %97, %94 : tensor<f64>
    %133 = stablehlo.subtract %131, %132 : tensor<f64>
    %134 = stablehlo.multiply %103, %111 : tensor<f64>
    %135 = stablehlo.subtract %133, %134 : tensor<f64>
    %136 = stablehlo.multiply %109, %105 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.reshape %137 : (tensor<f64>) -> tensor<1xf64>
    %139 = stablehlo.concatenate %114, %122, %130, %138, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %140 = stablehlo.slice %139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %141 = stablehlo.reshape %140 : (tensor<1xf64>) -> tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.slice %139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %144 = stablehlo.reshape %143 : (tensor<1xf64>) -> tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.slice %139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %147 = stablehlo.reshape %146 : (tensor<1xf64>) -> tensor<f64>
    %148 = stablehlo.reshape %147 : (tensor<f64>) -> tensor<1xf64>
    %149 = stablehlo.concatenate %142, %145, %148, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %150 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %151 = call @_threefry_fold_in(%c, %150) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_1 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %152 = stablehlo.sqrt %cst_1 : tensor<f64>
    %153 = call @_normal(%151) : (tensor<2xui32>) -> tensor<3xf64>
    %154 = stablehlo.convert %152 : tensor<f64>
    %155 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %156 = stablehlo.multiply %155, %153 : tensor<3xf64>
    %157 = stablehlo.add %149, %156 : tensor<3xf64>
    %158 = stablehlo.add %157, %arg4 : tensor<3xf64>
    %159 = stablehlo.slice %cst [0:1] : (tensor<5xf64>) -> tensor<1xf64>
    %160 = stablehlo.reshape %159 : (tensor<1xf64>) -> tensor<f64>
    %161 = stablehlo.slice %cst [1:2] : (tensor<5xf64>) -> tensor<1xf64>
    %162 = stablehlo.reshape %161 : (tensor<1xf64>) -> tensor<f64>
    %163 = stablehlo.slice %cst [2:3] : (tensor<5xf64>) -> tensor<1xf64>
    %164 = stablehlo.reshape %163 : (tensor<1xf64>) -> tensor<f64>
    %165 = stablehlo.slice %cst [3:4] : (tensor<5xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %cst [4:5] : (tensor<5xf64>) -> tensor<1xf64>
    %168 = stablehlo.reshape %167 : (tensor<1xf64>) -> tensor<f64>
    %169 = stablehlo.slice %arg3 [0:1, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %170 = stablehlo.reshape %169 : (tensor<1x3xf64>) -> tensor<3xf64>
    %171 = stablehlo.slice %arg3 [1:2, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %172 = stablehlo.reshape %171 : (tensor<1x3xf64>) -> tensor<3xf64>
    %173 = stablehlo.slice %arg3 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %174 = stablehlo.reshape %173 : (tensor<1x3xf64>) -> tensor<3xf64>
    %175 = stablehlo.slice %arg3 [3:4, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %176 = stablehlo.reshape %175 : (tensor<1x3xf64>) -> tensor<3xf64>
    %177 = stablehlo.broadcast_in_dim %160, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %178 = stablehlo.multiply %177, %158 : tensor<3xf64>
    %179 = stablehlo.broadcast_in_dim %162, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %180 = stablehlo.multiply %179, %170 : tensor<3xf64>
    %181 = stablehlo.add %178, %180 : tensor<3xf64>
    %182 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %183 = stablehlo.multiply %182, %172 : tensor<3xf64>
    %184 = stablehlo.add %181, %183 : tensor<3xf64>
    %185 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %186 = stablehlo.multiply %185, %174 : tensor<3xf64>
    %187 = stablehlo.subtract %184, %186 : tensor<3xf64>
    %188 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %189 = stablehlo.multiply %188, %176 : tensor<3xf64>
    %190 = stablehlo.subtract %187, %189 : tensor<3xf64>
    %191 = stablehlo.broadcast_in_dim %158, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %192 = stablehlo.broadcast_in_dim %170, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %193 = stablehlo.broadcast_in_dim %190, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %194 = stablehlo.broadcast_in_dim %174, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %195 = stablehlo.concatenate %191, %192, %193, %194, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<4x3xf64>
    %196 = stablehlo.slice %195 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %197 = stablehlo.reshape %196 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %195, %197 : tensor<4x3xf64>, tensor<3xf64>
  }
  func.func private @inner_238(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<6xf64>, %arg3: tensor<4x3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>) -> (tensor<4x3xf64>, tensor<3xf64>) {
    %c = stablehlo.constant dense<[0, 0, 1]> : tensor<3xi64>
    %c_0 = stablehlo.constant dense<[928981903, 3453687069]> : tensor<2xui32>
    %cst = stablehlo.constant dense<[0.0044300075115303239, 0.0088600150230606477, 0.0044300075115303239, -1.8030932880476023, 0.82081331809372371]> : tensor<5xf64>
    %0 = stablehlo.slice %arg2 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_1 = stablehlo.constant dense<9.810000e+00> : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2 = stablehlo.divide %0, %1 : tensor<3xf64>
    %3 = stablehlo.convert %c : (tensor<3xi64>) -> tensor<3xf64>
    %4 = stablehlo.add %2, %3 : tensor<3xf64>
    %5 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %6 = stablehlo.slice %5 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %5 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %5 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.negate %15 : tensor<f64>
    %17 = stablehlo.reshape %16 : (tensor<f64>) -> tensor<1xf64>
    %18 = stablehlo.slice %5 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %19 = stablehlo.reshape %18 : (tensor<1xf64>) -> tensor<f64>
    %20 = stablehlo.reshape %19 : (tensor<f64>) -> tensor<1xf64>
    %21 = stablehlo.concatenate %9, %13, %17, %20, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %22 = stablehlo.dot_general %5, %5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %24 = stablehlo.divide %21, %23 : tensor<4xf64>
    %25 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %27 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %28 = stablehlo.concatenate %4, %27, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %29 = stablehlo.slice %28 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.multiply %26, %30 : tensor<f64>
    %32 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %33 = stablehlo.reshape %32 : (tensor<1xf64>) -> tensor<f64>
    %34 = stablehlo.slice %28 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.multiply %33, %35 : tensor<f64>
    %37 = stablehlo.add %31, %36 : tensor<f64>
    %38 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %39 = stablehlo.reshape %38 : (tensor<1xf64>) -> tensor<f64>
    %40 = stablehlo.slice %28 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.multiply %39, %41 : tensor<f64>
    %43 = stablehlo.add %37, %42 : tensor<f64>
    %44 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %45 = stablehlo.reshape %44 : (tensor<1xf64>) -> tensor<f64>
    %46 = stablehlo.slice %28 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %47 = stablehlo.reshape %46 : (tensor<1xf64>) -> tensor<f64>
    %48 = stablehlo.multiply %45, %47 : tensor<f64>
    %49 = stablehlo.subtract %43, %48 : tensor<f64>
    %50 = stablehlo.reshape %49 : (tensor<f64>) -> tensor<1xf64>
    %51 = stablehlo.multiply %26, %47 : tensor<f64>
    %52 = stablehlo.multiply %33, %41 : tensor<f64>
    %53 = stablehlo.subtract %51, %52 : tensor<f64>
    %54 = stablehlo.multiply %39, %35 : tensor<f64>
    %55 = stablehlo.add %53, %54 : tensor<f64>
    %56 = stablehlo.multiply %45, %30 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.reshape %57 : (tensor<f64>) -> tensor<1xf64>
    %59 = stablehlo.multiply %26, %41 : tensor<f64>
    %60 = stablehlo.multiply %33, %47 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.multiply %39, %30 : tensor<f64>
    %63 = stablehlo.subtract %61, %62 : tensor<f64>
    %64 = stablehlo.multiply %45, %35 : tensor<f64>
    %65 = stablehlo.add %63, %64 : tensor<f64>
    %66 = stablehlo.reshape %65 : (tensor<f64>) -> tensor<1xf64>
    %67 = stablehlo.multiply %26, %35 : tensor<f64>
    %68 = stablehlo.multiply %33, %30 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.multiply %39, %47 : tensor<f64>
    %71 = stablehlo.subtract %69, %70 : tensor<f64>
    %72 = stablehlo.multiply %45, %41 : tensor<f64>
    %73 = stablehlo.subtract %71, %72 : tensor<f64>
    %74 = stablehlo.reshape %73 : (tensor<f64>) -> tensor<1xf64>
    %75 = stablehlo.concatenate %50, %58, %66, %74, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %76 = stablehlo.slice %75 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %77 = stablehlo.reshape %76 : (tensor<1xf64>) -> tensor<f64>
    %78 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.negate %87 : tensor<f64>
    %89 = stablehlo.reshape %88 : (tensor<f64>) -> tensor<1xf64>
    %90 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %91 = stablehlo.reshape %90 : (tensor<1xf64>) -> tensor<f64>
    %92 = stablehlo.reshape %91 : (tensor<f64>) -> tensor<1xf64>
    %93 = stablehlo.concatenate %81, %85, %89, %92, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %94 = stablehlo.dot_general %24, %24, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %95 = stablehlo.broadcast_in_dim %94, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %96 = stablehlo.divide %93, %95 : tensor<4xf64>
    %97 = stablehlo.slice %96 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.multiply %77, %98 : tensor<f64>
    %100 = stablehlo.slice %75 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %101 = stablehlo.reshape %100 : (tensor<1xf64>) -> tensor<f64>
    %102 = stablehlo.slice %96 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.multiply %101, %103 : tensor<f64>
    %105 = stablehlo.add %99, %104 : tensor<f64>
    %106 = stablehlo.slice %75 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %107 = stablehlo.reshape %106 : (tensor<1xf64>) -> tensor<f64>
    %108 = stablehlo.slice %96 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.multiply %107, %109 : tensor<f64>
    %111 = stablehlo.add %105, %110 : tensor<f64>
    %112 = stablehlo.slice %75 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %113 = stablehlo.reshape %112 : (tensor<1xf64>) -> tensor<f64>
    %114 = stablehlo.slice %96 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %115 = stablehlo.reshape %114 : (tensor<1xf64>) -> tensor<f64>
    %116 = stablehlo.multiply %113, %115 : tensor<f64>
    %117 = stablehlo.subtract %111, %116 : tensor<f64>
    %118 = stablehlo.reshape %117 : (tensor<f64>) -> tensor<1xf64>
    %119 = stablehlo.multiply %77, %115 : tensor<f64>
    %120 = stablehlo.multiply %101, %109 : tensor<f64>
    %121 = stablehlo.subtract %119, %120 : tensor<f64>
    %122 = stablehlo.multiply %107, %103 : tensor<f64>
    %123 = stablehlo.add %121, %122 : tensor<f64>
    %124 = stablehlo.multiply %113, %98 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.reshape %125 : (tensor<f64>) -> tensor<1xf64>
    %127 = stablehlo.multiply %77, %109 : tensor<f64>
    %128 = stablehlo.multiply %101, %115 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.multiply %107, %98 : tensor<f64>
    %131 = stablehlo.subtract %129, %130 : tensor<f64>
    %132 = stablehlo.multiply %113, %103 : tensor<f64>
    %133 = stablehlo.add %131, %132 : tensor<f64>
    %134 = stablehlo.reshape %133 : (tensor<f64>) -> tensor<1xf64>
    %135 = stablehlo.multiply %77, %103 : tensor<f64>
    %136 = stablehlo.multiply %101, %98 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.multiply %107, %115 : tensor<f64>
    %139 = stablehlo.subtract %137, %138 : tensor<f64>
    %140 = stablehlo.multiply %113, %109 : tensor<f64>
    %141 = stablehlo.subtract %139, %140 : tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.concatenate %118, %126, %134, %142, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %144 = stablehlo.slice %143 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.reshape %145 : (tensor<f64>) -> tensor<1xf64>
    %147 = stablehlo.slice %143 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %148 = stablehlo.reshape %147 : (tensor<1xf64>) -> tensor<f64>
    %149 = stablehlo.reshape %148 : (tensor<f64>) -> tensor<1xf64>
    %150 = stablehlo.slice %143 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %151 = stablehlo.reshape %150 : (tensor<1xf64>) -> tensor<f64>
    %152 = stablehlo.reshape %151 : (tensor<f64>) -> tensor<1xf64>
    %153 = stablehlo.concatenate %146, %149, %152, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %154 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %155 = call @_threefry_fold_in(%c_0, %154) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_3 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %156 = stablehlo.sqrt %cst_3 : tensor<f64>
    %157 = call @_normal(%155) : (tensor<2xui32>) -> tensor<3xf64>
    %158 = stablehlo.convert %156 : tensor<f64>
    %159 = stablehlo.broadcast_in_dim %158, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %160 = stablehlo.multiply %159, %157 : tensor<3xf64>
    %161 = stablehlo.add %153, %160 : tensor<3xf64>
    %162 = stablehlo.add %161, %arg4 : tensor<3xf64>
    %163 = stablehlo.slice %cst [0:1] : (tensor<5xf64>) -> tensor<1xf64>
    %164 = stablehlo.reshape %163 : (tensor<1xf64>) -> tensor<f64>
    %165 = stablehlo.slice %cst [1:2] : (tensor<5xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.slice %cst [2:3] : (tensor<5xf64>) -> tensor<1xf64>
    %168 = stablehlo.reshape %167 : (tensor<1xf64>) -> tensor<f64>
    %169 = stablehlo.slice %cst [3:4] : (tensor<5xf64>) -> tensor<1xf64>
    %170 = stablehlo.reshape %169 : (tensor<1xf64>) -> tensor<f64>
    %171 = stablehlo.slice %cst [4:5] : (tensor<5xf64>) -> tensor<1xf64>
    %172 = stablehlo.reshape %171 : (tensor<1xf64>) -> tensor<f64>
    %173 = stablehlo.slice %arg3 [0:1, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %174 = stablehlo.reshape %173 : (tensor<1x3xf64>) -> tensor<3xf64>
    %175 = stablehlo.slice %arg3 [1:2, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %176 = stablehlo.reshape %175 : (tensor<1x3xf64>) -> tensor<3xf64>
    %177 = stablehlo.slice %arg3 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %178 = stablehlo.reshape %177 : (tensor<1x3xf64>) -> tensor<3xf64>
    %179 = stablehlo.slice %arg3 [3:4, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %180 = stablehlo.reshape %179 : (tensor<1x3xf64>) -> tensor<3xf64>
    %181 = stablehlo.broadcast_in_dim %164, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %182 = stablehlo.multiply %181, %162 : tensor<3xf64>
    %183 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %184 = stablehlo.multiply %183, %174 : tensor<3xf64>
    %185 = stablehlo.add %182, %184 : tensor<3xf64>
    %186 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %187 = stablehlo.multiply %186, %176 : tensor<3xf64>
    %188 = stablehlo.add %185, %187 : tensor<3xf64>
    %189 = stablehlo.broadcast_in_dim %170, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %190 = stablehlo.multiply %189, %178 : tensor<3xf64>
    %191 = stablehlo.subtract %188, %190 : tensor<3xf64>
    %192 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %193 = stablehlo.multiply %192, %180 : tensor<3xf64>
    %194 = stablehlo.subtract %191, %193 : tensor<3xf64>
    %195 = stablehlo.broadcast_in_dim %162, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %196 = stablehlo.broadcast_in_dim %174, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %197 = stablehlo.broadcast_in_dim %194, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %198 = stablehlo.broadcast_in_dim %178, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
    %199 = stablehlo.concatenate %195, %196, %197, %198, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>, tensor<1x3xf64>) -> tensor<4x3xf64>
    %200 = stablehlo.slice %199 [2:3, 0:3] : (tensor<4x3xf64>) -> tensor<1x3xf64>
    %201 = stablehlo.reshape %200 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %199, %201 : tensor<4x3xf64>, tensor<3xf64>
  }
  func.func private @inner_239(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = call @norm(%arg0) : (tensor<3xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.subtract %0, %cst : tensor<f64>
    %2 = stablehlo.abs %1 : tensor<f64>
    %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %3 = stablehlo.divide %2, %cst_0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %4 = call @clip(%3, %cst_1, %cst_2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %5 = stablehlo.subtract %cst_3, %4 : tensor<f64>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %6 = stablehlo.multiply %cst_4, %5 : tensor<f64>
    %7 = call @norm(%arg1) : (tensor<3xf64>) -> tensor<f64>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %8 = stablehlo.divide %7, %cst_5 : tensor<f64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %9 = call @clip(%8, %cst_6, %cst_7) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %10 = stablehlo.subtract %cst_8, %9 : tensor<f64>
    %11 = stablehlo.multiply %6, %10 : tensor<f64>
    return %11 : tensor<f64>
  }
  func.func private @inner_240(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<3xf64>
    %c = stablehlo.constant dense<[4146024105, 2718843009]> : tensor<2xui32>
    %0 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
    %3 = stablehlo.negate %2 : tensor<f64>
    %4 = stablehlo.reshape %3 : (tensor<f64>) -> tensor<1xf64>
    %5 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %7 = stablehlo.negate %6 : tensor<f64>
    %8 = stablehlo.reshape %7 : (tensor<f64>) -> tensor<1xf64>
    %9 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
    %11 = stablehlo.negate %10 : tensor<f64>
    %12 = stablehlo.reshape %11 : (tensor<f64>) -> tensor<1xf64>
    %13 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %14 = stablehlo.reshape %13 : (tensor<1xf64>) -> tensor<f64>
    %15 = stablehlo.reshape %14 : (tensor<f64>) -> tensor<1xf64>
    %16 = stablehlo.concatenate %4, %8, %12, %15, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %17 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %19 = stablehlo.divide %16, %18 : tensor<4xf64>
    %20 = stablehlo.slice %19 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %22 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %23 = stablehlo.concatenate %cst, %22, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %24 = stablehlo.slice %23 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.multiply %21, %25 : tensor<f64>
    %27 = stablehlo.slice %19 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %28 = stablehlo.reshape %27 : (tensor<1xf64>) -> tensor<f64>
    %29 = stablehlo.slice %23 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.multiply %28, %30 : tensor<f64>
    %32 = stablehlo.add %26, %31 : tensor<f64>
    %33 = stablehlo.slice %19 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %34 = stablehlo.reshape %33 : (tensor<1xf64>) -> tensor<f64>
    %35 = stablehlo.slice %23 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1xf64>) -> tensor<f64>
    %37 = stablehlo.multiply %34, %36 : tensor<f64>
    %38 = stablehlo.add %32, %37 : tensor<f64>
    %39 = stablehlo.slice %19 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %40 = stablehlo.reshape %39 : (tensor<1xf64>) -> tensor<f64>
    %41 = stablehlo.slice %23 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.multiply %40, %42 : tensor<f64>
    %44 = stablehlo.subtract %38, %43 : tensor<f64>
    %45 = stablehlo.reshape %44 : (tensor<f64>) -> tensor<1xf64>
    %46 = stablehlo.multiply %21, %42 : tensor<f64>
    %47 = stablehlo.multiply %28, %36 : tensor<f64>
    %48 = stablehlo.subtract %46, %47 : tensor<f64>
    %49 = stablehlo.multiply %34, %30 : tensor<f64>
    %50 = stablehlo.add %48, %49 : tensor<f64>
    %51 = stablehlo.multiply %40, %25 : tensor<f64>
    %52 = stablehlo.add %50, %51 : tensor<f64>
    %53 = stablehlo.reshape %52 : (tensor<f64>) -> tensor<1xf64>
    %54 = stablehlo.multiply %21, %36 : tensor<f64>
    %55 = stablehlo.multiply %28, %42 : tensor<f64>
    %56 = stablehlo.add %54, %55 : tensor<f64>
    %57 = stablehlo.multiply %34, %25 : tensor<f64>
    %58 = stablehlo.subtract %56, %57 : tensor<f64>
    %59 = stablehlo.multiply %40, %30 : tensor<f64>
    %60 = stablehlo.add %58, %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.multiply %21, %30 : tensor<f64>
    %63 = stablehlo.multiply %28, %25 : tensor<f64>
    %64 = stablehlo.subtract %62, %63 : tensor<f64>
    %65 = stablehlo.multiply %34, %42 : tensor<f64>
    %66 = stablehlo.subtract %64, %65 : tensor<f64>
    %67 = stablehlo.multiply %40, %36 : tensor<f64>
    %68 = stablehlo.subtract %66, %67 : tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.concatenate %45, %53, %61, %69, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %71 = stablehlo.slice %70 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %72 = stablehlo.reshape %71 : (tensor<1xf64>) -> tensor<f64>
    %73 = stablehlo.slice %19 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.negate %74 : tensor<f64>
    %76 = stablehlo.reshape %75 : (tensor<f64>) -> tensor<1xf64>
    %77 = stablehlo.slice %19 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.negate %78 : tensor<f64>
    %80 = stablehlo.reshape %79 : (tensor<f64>) -> tensor<1xf64>
    %81 = stablehlo.slice %19 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %82 = stablehlo.reshape %81 : (tensor<1xf64>) -> tensor<f64>
    %83 = stablehlo.negate %82 : tensor<f64>
    %84 = stablehlo.reshape %83 : (tensor<f64>) -> tensor<1xf64>
    %85 = stablehlo.slice %19 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %86 = stablehlo.reshape %85 : (tensor<1xf64>) -> tensor<f64>
    %87 = stablehlo.reshape %86 : (tensor<f64>) -> tensor<1xf64>
    %88 = stablehlo.concatenate %76, %80, %84, %87, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %89 = stablehlo.dot_general %19, %19, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %91 = stablehlo.divide %88, %90 : tensor<4xf64>
    %92 = stablehlo.slice %91 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.multiply %72, %93 : tensor<f64>
    %95 = stablehlo.slice %70 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %96 = stablehlo.reshape %95 : (tensor<1xf64>) -> tensor<f64>
    %97 = stablehlo.slice %91 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.multiply %96, %98 : tensor<f64>
    %100 = stablehlo.add %94, %99 : tensor<f64>
    %101 = stablehlo.slice %70 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %102 = stablehlo.reshape %101 : (tensor<1xf64>) -> tensor<f64>
    %103 = stablehlo.slice %91 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
    %105 = stablehlo.multiply %102, %104 : tensor<f64>
    %106 = stablehlo.add %100, %105 : tensor<f64>
    %107 = stablehlo.slice %70 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %108 = stablehlo.reshape %107 : (tensor<1xf64>) -> tensor<f64>
    %109 = stablehlo.slice %91 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %110 = stablehlo.reshape %109 : (tensor<1xf64>) -> tensor<f64>
    %111 = stablehlo.multiply %108, %110 : tensor<f64>
    %112 = stablehlo.subtract %106, %111 : tensor<f64>
    %113 = stablehlo.reshape %112 : (tensor<f64>) -> tensor<1xf64>
    %114 = stablehlo.multiply %72, %110 : tensor<f64>
    %115 = stablehlo.multiply %96, %104 : tensor<f64>
    %116 = stablehlo.subtract %114, %115 : tensor<f64>
    %117 = stablehlo.multiply %102, %98 : tensor<f64>
    %118 = stablehlo.add %116, %117 : tensor<f64>
    %119 = stablehlo.multiply %108, %93 : tensor<f64>
    %120 = stablehlo.add %118, %119 : tensor<f64>
    %121 = stablehlo.reshape %120 : (tensor<f64>) -> tensor<1xf64>
    %122 = stablehlo.multiply %72, %104 : tensor<f64>
    %123 = stablehlo.multiply %96, %110 : tensor<f64>
    %124 = stablehlo.add %122, %123 : tensor<f64>
    %125 = stablehlo.multiply %102, %93 : tensor<f64>
    %126 = stablehlo.subtract %124, %125 : tensor<f64>
    %127 = stablehlo.multiply %108, %98 : tensor<f64>
    %128 = stablehlo.add %126, %127 : tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.multiply %72, %98 : tensor<f64>
    %131 = stablehlo.multiply %96, %93 : tensor<f64>
    %132 = stablehlo.subtract %130, %131 : tensor<f64>
    %133 = stablehlo.multiply %102, %110 : tensor<f64>
    %134 = stablehlo.subtract %132, %133 : tensor<f64>
    %135 = stablehlo.multiply %108, %104 : tensor<f64>
    %136 = stablehlo.subtract %134, %135 : tensor<f64>
    %137 = stablehlo.reshape %136 : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.concatenate %113, %121, %129, %137, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %139 = stablehlo.slice %138 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.reshape %140 : (tensor<f64>) -> tensor<1xf64>
    %142 = stablehlo.slice %138 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %143 = stablehlo.reshape %142 : (tensor<1xf64>) -> tensor<f64>
    %144 = stablehlo.reshape %143 : (tensor<f64>) -> tensor<1xf64>
    %145 = stablehlo.slice %138 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %146 = stablehlo.reshape %145 : (tensor<1xf64>) -> tensor<f64>
    %147 = stablehlo.reshape %146 : (tensor<f64>) -> tensor<1xf64>
    %148 = stablehlo.concatenate %141, %144, %147, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %149 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<ui32>
    %150 = call @_threefry_fold_in(%c, %149) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
    %cst_1 = stablehlo.constant dense<1.000000e-04> : tensor<f64>
    %151 = stablehlo.sqrt %cst_1 : tensor<f64>
    %152 = call @_normal(%150) : (tensor<2xui32>) -> tensor<3xf64>
    %153 = stablehlo.convert %151 : tensor<f64>
    %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %155 = stablehlo.multiply %154, %152 : tensor<3xf64>
    %156 = stablehlo.add %148, %155 : tensor<3xf64>
    %157 = stablehlo.add %156, %arg2 : tensor<3xf64>
    %c_2 = stablehlo.constant dense<9> : tensor<i64>
    %158 = call @remainder_205(%arg0, %c_2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %159 = stablehlo.compare  EQ, %158, %c_3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %160 = stablehlo.convert %159 : (tensor<i1>) -> tensor<i32>
    %161 = "stablehlo.case"(%160) ({
      stablehlo.return %arg3 : tensor<3xf64>
    }, {
      stablehlo.return %157 : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    return %161 : tensor<3xf64>
  }
  func.func private @inner_242(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg1 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2 = stablehlo.slice %1 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.negate %3 : tensor<f64>
    %5 = stablehlo.reshape %4 : (tensor<f64>) -> tensor<1xf64>
    %6 = stablehlo.slice %1 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.negate %7 : tensor<f64>
    %9 = stablehlo.reshape %8 : (tensor<f64>) -> tensor<1xf64>
    %10 = stablehlo.slice %1 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.negate %11 : tensor<f64>
    %13 = stablehlo.reshape %12 : (tensor<f64>) -> tensor<1xf64>
    %14 = stablehlo.slice %1 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.reshape %15 : (tensor<f64>) -> tensor<1xf64>
    %17 = stablehlo.concatenate %5, %9, %13, %16, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %18 = stablehlo.dot_general %1, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %20 = stablehlo.divide %17, %19 : tensor<4xf64>
    %21 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %23 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.concatenate %0, %23, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %25 = stablehlo.slice %24 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %26 = stablehlo.reshape %25 : (tensor<1xf64>) -> tensor<f64>
    %27 = stablehlo.multiply %22, %26 : tensor<f64>
    %28 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
    %30 = stablehlo.slice %24 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.multiply %29, %31 : tensor<f64>
    %33 = stablehlo.add %27, %32 : tensor<f64>
    %34 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
    %36 = stablehlo.slice %24 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %37 = stablehlo.reshape %36 : (tensor<1xf64>) -> tensor<f64>
    %38 = stablehlo.multiply %35, %37 : tensor<f64>
    %39 = stablehlo.add %33, %38 : tensor<f64>
    %40 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %41 = stablehlo.reshape %40 : (tensor<1xf64>) -> tensor<f64>
    %42 = stablehlo.slice %24 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %43 = stablehlo.reshape %42 : (tensor<1xf64>) -> tensor<f64>
    %44 = stablehlo.multiply %41, %43 : tensor<f64>
    %45 = stablehlo.subtract %39, %44 : tensor<f64>
    %46 = stablehlo.reshape %45 : (tensor<f64>) -> tensor<1xf64>
    %47 = stablehlo.multiply %22, %43 : tensor<f64>
    %48 = stablehlo.multiply %29, %37 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.multiply %35, %31 : tensor<f64>
    %51 = stablehlo.add %49, %50 : tensor<f64>
    %52 = stablehlo.multiply %41, %26 : tensor<f64>
    %53 = stablehlo.add %51, %52 : tensor<f64>
    %54 = stablehlo.reshape %53 : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.multiply %22, %37 : tensor<f64>
    %56 = stablehlo.multiply %29, %43 : tensor<f64>
    %57 = stablehlo.add %55, %56 : tensor<f64>
    %58 = stablehlo.multiply %35, %26 : tensor<f64>
    %59 = stablehlo.subtract %57, %58 : tensor<f64>
    %60 = stablehlo.multiply %41, %31 : tensor<f64>
    %61 = stablehlo.add %59, %60 : tensor<f64>
    %62 = stablehlo.reshape %61 : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.multiply %22, %31 : tensor<f64>
    %64 = stablehlo.multiply %29, %26 : tensor<f64>
    %65 = stablehlo.subtract %63, %64 : tensor<f64>
    %66 = stablehlo.multiply %35, %43 : tensor<f64>
    %67 = stablehlo.subtract %65, %66 : tensor<f64>
    %68 = stablehlo.multiply %41, %37 : tensor<f64>
    %69 = stablehlo.subtract %67, %68 : tensor<f64>
    %70 = stablehlo.reshape %69 : (tensor<f64>) -> tensor<1xf64>
    %71 = stablehlo.concatenate %46, %54, %62, %70, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %72 = stablehlo.slice %71 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1xf64>) -> tensor<f64>
    %74 = stablehlo.slice %20 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.negate %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.slice %20 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.negate %79 : tensor<f64>
    %81 = stablehlo.reshape %80 : (tensor<f64>) -> tensor<1xf64>
    %82 = stablehlo.slice %20 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.negate %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.slice %20 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1xf64>) -> tensor<f64>
    %88 = stablehlo.reshape %87 : (tensor<f64>) -> tensor<1xf64>
    %89 = stablehlo.concatenate %77, %81, %85, %88, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %90 = stablehlo.dot_general %20, %20, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %92 = stablehlo.divide %89, %91 : tensor<4xf64>
    %93 = stablehlo.slice %92 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %94 = stablehlo.reshape %93 : (tensor<1xf64>) -> tensor<f64>
    %95 = stablehlo.multiply %73, %94 : tensor<f64>
    %96 = stablehlo.slice %71 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.slice %92 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %99 = stablehlo.reshape %98 : (tensor<1xf64>) -> tensor<f64>
    %100 = stablehlo.multiply %97, %99 : tensor<f64>
    %101 = stablehlo.add %95, %100 : tensor<f64>
    %102 = stablehlo.slice %71 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %103 = stablehlo.reshape %102 : (tensor<1xf64>) -> tensor<f64>
    %104 = stablehlo.slice %92 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %105 = stablehlo.reshape %104 : (tensor<1xf64>) -> tensor<f64>
    %106 = stablehlo.multiply %103, %105 : tensor<f64>
    %107 = stablehlo.add %101, %106 : tensor<f64>
    %108 = stablehlo.slice %71 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %109 = stablehlo.reshape %108 : (tensor<1xf64>) -> tensor<f64>
    %110 = stablehlo.slice %92 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %111 = stablehlo.reshape %110 : (tensor<1xf64>) -> tensor<f64>
    %112 = stablehlo.multiply %109, %111 : tensor<f64>
    %113 = stablehlo.subtract %107, %112 : tensor<f64>
    %114 = stablehlo.reshape %113 : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.multiply %73, %111 : tensor<f64>
    %116 = stablehlo.multiply %97, %105 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.multiply %103, %99 : tensor<f64>
    %119 = stablehlo.add %117, %118 : tensor<f64>
    %120 = stablehlo.multiply %109, %94 : tensor<f64>
    %121 = stablehlo.add %119, %120 : tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.multiply %73, %105 : tensor<f64>
    %124 = stablehlo.multiply %97, %111 : tensor<f64>
    %125 = stablehlo.add %123, %124 : tensor<f64>
    %126 = stablehlo.multiply %103, %94 : tensor<f64>
    %127 = stablehlo.subtract %125, %126 : tensor<f64>
    %128 = stablehlo.multiply %109, %99 : tensor<f64>
    %129 = stablehlo.add %127, %128 : tensor<f64>
    %130 = stablehlo.reshape %129 : (tensor<f64>) -> tensor<1xf64>
    %131 = stablehlo.multiply %73, %99 : tensor<f64>
    %132 = stablehlo.multiply %97, %94 : tensor<f64>
    %133 = stablehlo.subtract %131, %132 : tensor<f64>
    %134 = stablehlo.multiply %103, %111 : tensor<f64>
    %135 = stablehlo.subtract %133, %134 : tensor<f64>
    %136 = stablehlo.multiply %109, %105 : tensor<f64>
    %137 = stablehlo.subtract %135, %136 : tensor<f64>
    %138 = stablehlo.reshape %137 : (tensor<f64>) -> tensor<1xf64>
    %139 = stablehlo.concatenate %114, %122, %130, %138, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %140 = stablehlo.slice %139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %141 = stablehlo.reshape %140 : (tensor<1xf64>) -> tensor<f64>
    %142 = stablehlo.reshape %141 : (tensor<f64>) -> tensor<1xf64>
    %143 = stablehlo.slice %139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %144 = stablehlo.reshape %143 : (tensor<1xf64>) -> tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.slice %139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %147 = stablehlo.reshape %146 : (tensor<1xf64>) -> tensor<f64>
    %148 = stablehlo.reshape %147 : (tensor<f64>) -> tensor<1xf64>
    %149 = stablehlo.concatenate %142, %145, %148, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %149 : tensor<3xf64>
  }
  func.func private @inner_243(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<4xf64>
    %cst_0 = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %3 = stablehlo.multiply %1, %2 : tensor<4xf64>
    %cst_1 = stablehlo.constant dense<6.000000e+01> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %5 = stablehlo.divide %3, %4 : tensor<4xf64>
    return %5 : tensor<4xf64>
  }
}
