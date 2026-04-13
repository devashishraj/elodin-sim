module @module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<7xf64>, %arg2: tensor<6xf64>, %arg3: tensor<3xf64>, %arg4: tensor<3xf64>, %arg5: tensor<f64>, %arg6: tensor<f64>, %arg7: tensor<f64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<6xf64>, %arg11: tensor<3xf64>, %arg12: tensor<480x3xf64>, %arg13: tensor<3xf64>, %arg14: tensor<3xf64>, %arg15: tensor<3xf64>, %arg16: tensor<f64>, %arg17: tensor<f64>, %arg18: tensor<f64>, %arg19: tensor<6xf64>, %arg20: tensor<f64>, %arg21: tensor<6xf64>, %arg22: tensor<f64>, %arg23: tensor<f64>, %arg24: tensor<f64>, %arg25: tensor<6xf64>, %arg26: tensor<7xf64>) -> (tensor<6xf64> {jax.result_info = "result[0]"}, tensor<480x3xf64> {jax.result_info = "result[1]"}, tensor<f64> {jax.result_info = "result[2]"}, tensor<f64> {jax.result_info = "result[3]"}, tensor<2xf64> {jax.result_info = "result[4]"}, tensor<f64> {jax.result_info = "result[5]"}, tensor<i64> {jax.result_info = "result[6]"}, tensor<f64> {jax.result_info = "result[7]"}, tensor<3xf64> {jax.result_info = "result[8]"}, tensor<3xf64> {jax.result_info = "result[9]"}, tensor<6xf64> {jax.result_info = "result[10]"}, tensor<3xf64> {jax.result_info = "result[11]"}, tensor<f64> {jax.result_info = "result[12]"}, tensor<3xf64> {jax.result_info = "result[13]"}, tensor<6xf64> {jax.result_info = "result[14]"}, tensor<f64> {jax.result_info = "result[15]"}, tensor<2xf64> {jax.result_info = "result[16]"}, tensor<f64> {jax.result_info = "result[17]"}, tensor<7xf64> {jax.result_info = "result[18]"}, tensor<7xf64> {jax.result_info = "result[19]"}, tensor<f64> {jax.result_info = "result[20]"}, tensor<6xf64> {jax.result_info = "result[21]"}, tensor<f64> {jax.result_info = "result[22]"}, tensor<6xf64> {jax.result_info = "result[23]"}, tensor<3xf64> {jax.result_info = "result[24]"}, tensor<f64> {jax.result_info = "result[25]"}, tensor<3xf64> {jax.result_info = "result[26]"}) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.add %arg0, %c : tensor<i64>
    %1:2 = call @inner(%arg1, %arg2, %arg3, %arg5, %arg6) : (tensor<7xf64>, tensor<6xf64>, tensor<3xf64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    %2 = call @inner_24(%arg1, %arg2, %arg3, %arg4) : (tensor<7xf64>, tensor<6xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %3 = call @inner_36(%arg1, %arg2, %arg3, %arg7) : (tensor<7xf64>, tensor<6xf64>, tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %4 = call @inner_38(%arg8, %arg9) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %5 = call @inner_42(%arg2, %arg10, %arg11) : (tensor<6xf64>, tensor<6xf64>, tensor<3xf64>) -> tensor<3xf64>
    %6 = call @inner_51(%5, %arg12) : (tensor<3xf64>, tensor<480x3xf64>) -> tensor<480x3xf64>
    %7 = call @inner_55(%6, %arg13) : (tensor<480x3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %8 = call @inner_78(%4, %7, %arg14) : (tensor<2xf64>, tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %9 = call @inner_82(%arg15, %8, %arg16) : (tensor<3xf64>, tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %10 = call @inner_83(%arg17, %9, %1#0) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %11 = call @inner_84(%1#0, %3, %10, %arg18, %arg19) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<6xf64>) -> tensor<6xf64>
    %12 = call @inner_96(%11, %arg20, %1#1, %arg21) : (tensor<6xf64>, tensor<f64>, tensor<f64>, tensor<6xf64>) -> tensor<6xf64>
    %13 = call @inner_105(%0, %arg22, %arg23, %arg24) : (tensor<i64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %14 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %15 = stablehlo.reshape %arg22 : (tensor<f64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %16 = stablehlo.multiply %cst, %15 : tensor<f64>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %18 = stablehlo.multiply %17, %arg2 : tensor<6xf64>
    %19 = stablehlo.slice %18 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %20 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %21 = stablehlo.divide %19, %20 : tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %22 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %23 = stablehlo.concatenate %21, %22, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %24 = stablehlo.slice %23 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %25 = stablehlo.reshape %24 : (tensor<1xf64>) -> tensor<f64>
    %26 = stablehlo.slice %14 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
    %28 = stablehlo.multiply %25, %27 : tensor<f64>
    %29 = stablehlo.slice %23 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.slice %14 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %32 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
    %33 = stablehlo.multiply %30, %32 : tensor<f64>
    %34 = stablehlo.add %28, %33 : tensor<f64>
    %35 = stablehlo.slice %23 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1xf64>) -> tensor<f64>
    %37 = stablehlo.slice %14 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %38 = stablehlo.reshape %37 : (tensor<1xf64>) -> tensor<f64>
    %39 = stablehlo.multiply %36, %38 : tensor<f64>
    %40 = stablehlo.add %34, %39 : tensor<f64>
    %41 = stablehlo.slice %23 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.slice %14 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %44 = stablehlo.reshape %43 : (tensor<1xf64>) -> tensor<f64>
    %45 = stablehlo.multiply %42, %44 : tensor<f64>
    %46 = stablehlo.subtract %40, %45 : tensor<f64>
    %47 = stablehlo.reshape %46 : (tensor<f64>) -> tensor<1xf64>
    %48 = stablehlo.multiply %25, %44 : tensor<f64>
    %49 = stablehlo.multiply %30, %38 : tensor<f64>
    %50 = stablehlo.subtract %48, %49 : tensor<f64>
    %51 = stablehlo.multiply %36, %32 : tensor<f64>
    %52 = stablehlo.add %50, %51 : tensor<f64>
    %53 = stablehlo.multiply %42, %27 : tensor<f64>
    %54 = stablehlo.add %52, %53 : tensor<f64>
    %55 = stablehlo.reshape %54 : (tensor<f64>) -> tensor<1xf64>
    %56 = stablehlo.multiply %25, %38 : tensor<f64>
    %57 = stablehlo.multiply %30, %44 : tensor<f64>
    %58 = stablehlo.add %56, %57 : tensor<f64>
    %59 = stablehlo.multiply %36, %27 : tensor<f64>
    %60 = stablehlo.subtract %58, %59 : tensor<f64>
    %61 = stablehlo.multiply %42, %32 : tensor<f64>
    %62 = stablehlo.add %60, %61 : tensor<f64>
    %63 = stablehlo.reshape %62 : (tensor<f64>) -> tensor<1xf64>
    %64 = stablehlo.multiply %25, %32 : tensor<f64>
    %65 = stablehlo.multiply %30, %27 : tensor<f64>
    %66 = stablehlo.subtract %64, %65 : tensor<f64>
    %67 = stablehlo.multiply %36, %44 : tensor<f64>
    %68 = stablehlo.subtract %66, %67 : tensor<f64>
    %69 = stablehlo.multiply %42, %38 : tensor<f64>
    %70 = stablehlo.subtract %68, %69 : tensor<f64>
    %71 = stablehlo.reshape %70 : (tensor<f64>) -> tensor<1xf64>
    %72 = stablehlo.concatenate %47, %55, %63, %71, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %73 = stablehlo.add %14, %72 : tensor<4xf64>
    %74 = stablehlo.dot_general %73, %73, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %75 = stablehlo.sqrt %74 : tensor<f64>
    %76 = stablehlo.broadcast_in_dim %75, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %77 = stablehlo.divide %73, %76 : tensor<4xf64>
    %78 = stablehlo.slice %arg1 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %79 = stablehlo.slice %18 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %80 = stablehlo.add %78, %79 : tensor<3xf64>
    %81 = stablehlo.concatenate %77, %80, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %82 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %83 = stablehlo.multiply %82, %arg10 : tensor<6xf64>
    %84 = stablehlo.add %arg2, %83 : tensor<6xf64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %85 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %86 = call @inner_124(%85, %arg26) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %87 = call @inner_126(%13, %86, %81) : (tensor<f64>, tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %88 = call @inner_127(%81, %12, %87) : (tensor<7xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %89 = stablehlo.slice %81 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %90 = stablehlo.slice %89 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %91 = stablehlo.reshape %90 : (tensor<1xf64>) -> tensor<f64>
    %92 = stablehlo.slice %89 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %93 = stablehlo.reshape %92 : (tensor<1xf64>) -> tensor<f64>
    %94 = stablehlo.negate %93 : tensor<f64>
    %95 = stablehlo.reshape %94 : (tensor<f64>) -> tensor<1xf64>
    %96 = stablehlo.slice %89 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %97 = stablehlo.reshape %96 : (tensor<1xf64>) -> tensor<f64>
    %98 = stablehlo.negate %97 : tensor<f64>
    %99 = stablehlo.reshape %98 : (tensor<f64>) -> tensor<1xf64>
    %100 = stablehlo.slice %89 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %101 = stablehlo.reshape %100 : (tensor<1xf64>) -> tensor<f64>
    %102 = stablehlo.negate %101 : tensor<f64>
    %103 = stablehlo.reshape %102 : (tensor<f64>) -> tensor<1xf64>
    %104 = stablehlo.slice %89 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %105 = stablehlo.reshape %104 : (tensor<1xf64>) -> tensor<f64>
    %106 = stablehlo.reshape %105 : (tensor<f64>) -> tensor<1xf64>
    %107 = stablehlo.concatenate %95, %99, %103, %106, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %108 = stablehlo.dot_general %89, %89, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %109 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %110 = stablehlo.divide %107, %109 : tensor<4xf64>
    %111 = stablehlo.slice %110 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %112 = stablehlo.reshape %111 : (tensor<1xf64>) -> tensor<f64>
    %113 = stablehlo.slice %88 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %114 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %115 = stablehlo.concatenate %113, %114, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %116 = stablehlo.slice %115 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %117 = stablehlo.reshape %116 : (tensor<1xf64>) -> tensor<f64>
    %118 = stablehlo.multiply %112, %117 : tensor<f64>
    %119 = stablehlo.slice %110 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %120 = stablehlo.reshape %119 : (tensor<1xf64>) -> tensor<f64>
    %121 = stablehlo.slice %115 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %122 = stablehlo.reshape %121 : (tensor<1xf64>) -> tensor<f64>
    %123 = stablehlo.multiply %120, %122 : tensor<f64>
    %124 = stablehlo.add %118, %123 : tensor<f64>
    %125 = stablehlo.slice %110 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %126 = stablehlo.reshape %125 : (tensor<1xf64>) -> tensor<f64>
    %127 = stablehlo.slice %115 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %128 = stablehlo.reshape %127 : (tensor<1xf64>) -> tensor<f64>
    %129 = stablehlo.multiply %126, %128 : tensor<f64>
    %130 = stablehlo.add %124, %129 : tensor<f64>
    %131 = stablehlo.slice %110 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %132 = stablehlo.reshape %131 : (tensor<1xf64>) -> tensor<f64>
    %133 = stablehlo.slice %115 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %134 = stablehlo.reshape %133 : (tensor<1xf64>) -> tensor<f64>
    %135 = stablehlo.multiply %132, %134 : tensor<f64>
    %136 = stablehlo.subtract %130, %135 : tensor<f64>
    %137 = stablehlo.reshape %136 : (tensor<f64>) -> tensor<1xf64>
    %138 = stablehlo.multiply %112, %134 : tensor<f64>
    %139 = stablehlo.multiply %120, %128 : tensor<f64>
    %140 = stablehlo.subtract %138, %139 : tensor<f64>
    %141 = stablehlo.multiply %126, %122 : tensor<f64>
    %142 = stablehlo.add %140, %141 : tensor<f64>
    %143 = stablehlo.multiply %132, %117 : tensor<f64>
    %144 = stablehlo.add %142, %143 : tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.multiply %112, %128 : tensor<f64>
    %147 = stablehlo.multiply %120, %134 : tensor<f64>
    %148 = stablehlo.add %146, %147 : tensor<f64>
    %149 = stablehlo.multiply %126, %117 : tensor<f64>
    %150 = stablehlo.subtract %148, %149 : tensor<f64>
    %151 = stablehlo.multiply %132, %122 : tensor<f64>
    %152 = stablehlo.add %150, %151 : tensor<f64>
    %153 = stablehlo.reshape %152 : (tensor<f64>) -> tensor<1xf64>
    %154 = stablehlo.multiply %112, %122 : tensor<f64>
    %155 = stablehlo.multiply %120, %117 : tensor<f64>
    %156 = stablehlo.subtract %154, %155 : tensor<f64>
    %157 = stablehlo.multiply %126, %134 : tensor<f64>
    %158 = stablehlo.subtract %156, %157 : tensor<f64>
    %159 = stablehlo.multiply %132, %128 : tensor<f64>
    %160 = stablehlo.subtract %158, %159 : tensor<f64>
    %161 = stablehlo.reshape %160 : (tensor<f64>) -> tensor<1xf64>
    %162 = stablehlo.concatenate %137, %145, %153, %161, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %163 = stablehlo.slice %162 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %164 = stablehlo.reshape %163 : (tensor<1xf64>) -> tensor<f64>
    %165 = stablehlo.slice %110 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %166 = stablehlo.reshape %165 : (tensor<1xf64>) -> tensor<f64>
    %167 = stablehlo.negate %166 : tensor<f64>
    %168 = stablehlo.reshape %167 : (tensor<f64>) -> tensor<1xf64>
    %169 = stablehlo.slice %110 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %170 = stablehlo.reshape %169 : (tensor<1xf64>) -> tensor<f64>
    %171 = stablehlo.negate %170 : tensor<f64>
    %172 = stablehlo.reshape %171 : (tensor<f64>) -> tensor<1xf64>
    %173 = stablehlo.slice %110 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %174 = stablehlo.reshape %173 : (tensor<1xf64>) -> tensor<f64>
    %175 = stablehlo.negate %174 : tensor<f64>
    %176 = stablehlo.reshape %175 : (tensor<f64>) -> tensor<1xf64>
    %177 = stablehlo.slice %110 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %178 = stablehlo.reshape %177 : (tensor<1xf64>) -> tensor<f64>
    %179 = stablehlo.reshape %178 : (tensor<f64>) -> tensor<1xf64>
    %180 = stablehlo.concatenate %168, %172, %176, %179, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %181 = stablehlo.dot_general %110, %110, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %182 = stablehlo.broadcast_in_dim %181, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %183 = stablehlo.divide %180, %182 : tensor<4xf64>
    %184 = stablehlo.slice %183 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %185 = stablehlo.reshape %184 : (tensor<1xf64>) -> tensor<f64>
    %186 = stablehlo.multiply %164, %185 : tensor<f64>
    %187 = stablehlo.slice %162 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %188 = stablehlo.reshape %187 : (tensor<1xf64>) -> tensor<f64>
    %189 = stablehlo.slice %183 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %190 = stablehlo.reshape %189 : (tensor<1xf64>) -> tensor<f64>
    %191 = stablehlo.multiply %188, %190 : tensor<f64>
    %192 = stablehlo.add %186, %191 : tensor<f64>
    %193 = stablehlo.slice %162 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %194 = stablehlo.reshape %193 : (tensor<1xf64>) -> tensor<f64>
    %195 = stablehlo.slice %183 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %196 = stablehlo.reshape %195 : (tensor<1xf64>) -> tensor<f64>
    %197 = stablehlo.multiply %194, %196 : tensor<f64>
    %198 = stablehlo.add %192, %197 : tensor<f64>
    %199 = stablehlo.slice %162 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %200 = stablehlo.reshape %199 : (tensor<1xf64>) -> tensor<f64>
    %201 = stablehlo.slice %183 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %202 = stablehlo.reshape %201 : (tensor<1xf64>) -> tensor<f64>
    %203 = stablehlo.multiply %200, %202 : tensor<f64>
    %204 = stablehlo.subtract %198, %203 : tensor<f64>
    %205 = stablehlo.reshape %204 : (tensor<f64>) -> tensor<1xf64>
    %206 = stablehlo.multiply %164, %202 : tensor<f64>
    %207 = stablehlo.multiply %188, %196 : tensor<f64>
    %208 = stablehlo.subtract %206, %207 : tensor<f64>
    %209 = stablehlo.multiply %194, %190 : tensor<f64>
    %210 = stablehlo.add %208, %209 : tensor<f64>
    %211 = stablehlo.multiply %200, %185 : tensor<f64>
    %212 = stablehlo.add %210, %211 : tensor<f64>
    %213 = stablehlo.reshape %212 : (tensor<f64>) -> tensor<1xf64>
    %214 = stablehlo.multiply %164, %196 : tensor<f64>
    %215 = stablehlo.multiply %188, %202 : tensor<f64>
    %216 = stablehlo.add %214, %215 : tensor<f64>
    %217 = stablehlo.multiply %194, %185 : tensor<f64>
    %218 = stablehlo.subtract %216, %217 : tensor<f64>
    %219 = stablehlo.multiply %200, %190 : tensor<f64>
    %220 = stablehlo.add %218, %219 : tensor<f64>
    %221 = stablehlo.reshape %220 : (tensor<f64>) -> tensor<1xf64>
    %222 = stablehlo.multiply %164, %190 : tensor<f64>
    %223 = stablehlo.multiply %188, %185 : tensor<f64>
    %224 = stablehlo.subtract %222, %223 : tensor<f64>
    %225 = stablehlo.multiply %194, %202 : tensor<f64>
    %226 = stablehlo.subtract %224, %225 : tensor<f64>
    %227 = stablehlo.multiply %200, %196 : tensor<f64>
    %228 = stablehlo.subtract %226, %227 : tensor<f64>
    %229 = stablehlo.reshape %228 : (tensor<f64>) -> tensor<1xf64>
    %230 = stablehlo.concatenate %205, %213, %221, %229, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %231 = stablehlo.slice %230 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %232 = stablehlo.reshape %231 : (tensor<1xf64>) -> tensor<f64>
    %233 = stablehlo.reshape %232 : (tensor<f64>) -> tensor<1xf64>
    %234 = stablehlo.slice %230 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %235 = stablehlo.reshape %234 : (tensor<1xf64>) -> tensor<f64>
    %236 = stablehlo.reshape %235 : (tensor<f64>) -> tensor<1xf64>
    %237 = stablehlo.slice %230 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %238 = stablehlo.reshape %237 : (tensor<1xf64>) -> tensor<f64>
    %239 = stablehlo.reshape %238 : (tensor<f64>) -> tensor<1xf64>
    %240 = stablehlo.concatenate %233, %236, %239, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %241 = stablehlo.slice %110 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %242 = stablehlo.reshape %241 : (tensor<1xf64>) -> tensor<f64>
    %243 = stablehlo.slice %88 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %244 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %245 = stablehlo.concatenate %243, %244, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %246 = stablehlo.slice %245 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %247 = stablehlo.reshape %246 : (tensor<1xf64>) -> tensor<f64>
    %248 = stablehlo.multiply %242, %247 : tensor<f64>
    %249 = stablehlo.slice %110 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %250 = stablehlo.reshape %249 : (tensor<1xf64>) -> tensor<f64>
    %251 = stablehlo.slice %245 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %252 = stablehlo.reshape %251 : (tensor<1xf64>) -> tensor<f64>
    %253 = stablehlo.multiply %250, %252 : tensor<f64>
    %254 = stablehlo.add %248, %253 : tensor<f64>
    %255 = stablehlo.slice %110 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %256 = stablehlo.reshape %255 : (tensor<1xf64>) -> tensor<f64>
    %257 = stablehlo.slice %245 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %258 = stablehlo.reshape %257 : (tensor<1xf64>) -> tensor<f64>
    %259 = stablehlo.multiply %256, %258 : tensor<f64>
    %260 = stablehlo.add %254, %259 : tensor<f64>
    %261 = stablehlo.slice %110 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %262 = stablehlo.reshape %261 : (tensor<1xf64>) -> tensor<f64>
    %263 = stablehlo.slice %245 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %264 = stablehlo.reshape %263 : (tensor<1xf64>) -> tensor<f64>
    %265 = stablehlo.multiply %262, %264 : tensor<f64>
    %266 = stablehlo.subtract %260, %265 : tensor<f64>
    %267 = stablehlo.reshape %266 : (tensor<f64>) -> tensor<1xf64>
    %268 = stablehlo.multiply %242, %264 : tensor<f64>
    %269 = stablehlo.multiply %250, %258 : tensor<f64>
    %270 = stablehlo.subtract %268, %269 : tensor<f64>
    %271 = stablehlo.multiply %256, %252 : tensor<f64>
    %272 = stablehlo.add %270, %271 : tensor<f64>
    %273 = stablehlo.multiply %262, %247 : tensor<f64>
    %274 = stablehlo.add %272, %273 : tensor<f64>
    %275 = stablehlo.reshape %274 : (tensor<f64>) -> tensor<1xf64>
    %276 = stablehlo.multiply %242, %258 : tensor<f64>
    %277 = stablehlo.multiply %250, %264 : tensor<f64>
    %278 = stablehlo.add %276, %277 : tensor<f64>
    %279 = stablehlo.multiply %256, %247 : tensor<f64>
    %280 = stablehlo.subtract %278, %279 : tensor<f64>
    %281 = stablehlo.multiply %262, %252 : tensor<f64>
    %282 = stablehlo.add %280, %281 : tensor<f64>
    %283 = stablehlo.reshape %282 : (tensor<f64>) -> tensor<1xf64>
    %284 = stablehlo.multiply %242, %252 : tensor<f64>
    %285 = stablehlo.multiply %250, %247 : tensor<f64>
    %286 = stablehlo.subtract %284, %285 : tensor<f64>
    %287 = stablehlo.multiply %256, %264 : tensor<f64>
    %288 = stablehlo.subtract %286, %287 : tensor<f64>
    %289 = stablehlo.multiply %262, %258 : tensor<f64>
    %290 = stablehlo.subtract %288, %289 : tensor<f64>
    %291 = stablehlo.reshape %290 : (tensor<f64>) -> tensor<1xf64>
    %292 = stablehlo.concatenate %267, %275, %283, %291, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %293 = stablehlo.slice %292 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %294 = stablehlo.reshape %293 : (tensor<1xf64>) -> tensor<f64>
    %295 = stablehlo.slice %110 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %296 = stablehlo.reshape %295 : (tensor<1xf64>) -> tensor<f64>
    %297 = stablehlo.negate %296 : tensor<f64>
    %298 = stablehlo.reshape %297 : (tensor<f64>) -> tensor<1xf64>
    %299 = stablehlo.slice %110 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %300 = stablehlo.reshape %299 : (tensor<1xf64>) -> tensor<f64>
    %301 = stablehlo.negate %300 : tensor<f64>
    %302 = stablehlo.reshape %301 : (tensor<f64>) -> tensor<1xf64>
    %303 = stablehlo.slice %110 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %304 = stablehlo.reshape %303 : (tensor<1xf64>) -> tensor<f64>
    %305 = stablehlo.negate %304 : tensor<f64>
    %306 = stablehlo.reshape %305 : (tensor<f64>) -> tensor<1xf64>
    %307 = stablehlo.slice %110 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %308 = stablehlo.reshape %307 : (tensor<1xf64>) -> tensor<f64>
    %309 = stablehlo.reshape %308 : (tensor<f64>) -> tensor<1xf64>
    %310 = stablehlo.concatenate %298, %302, %306, %309, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %311 = stablehlo.dot_general %110, %110, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %312 = stablehlo.broadcast_in_dim %311, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %313 = stablehlo.divide %310, %312 : tensor<4xf64>
    %314 = stablehlo.slice %313 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %315 = stablehlo.reshape %314 : (tensor<1xf64>) -> tensor<f64>
    %316 = stablehlo.multiply %294, %315 : tensor<f64>
    %317 = stablehlo.slice %292 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %318 = stablehlo.reshape %317 : (tensor<1xf64>) -> tensor<f64>
    %319 = stablehlo.slice %313 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %320 = stablehlo.reshape %319 : (tensor<1xf64>) -> tensor<f64>
    %321 = stablehlo.multiply %318, %320 : tensor<f64>
    %322 = stablehlo.add %316, %321 : tensor<f64>
    %323 = stablehlo.slice %292 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %324 = stablehlo.reshape %323 : (tensor<1xf64>) -> tensor<f64>
    %325 = stablehlo.slice %313 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %326 = stablehlo.reshape %325 : (tensor<1xf64>) -> tensor<f64>
    %327 = stablehlo.multiply %324, %326 : tensor<f64>
    %328 = stablehlo.add %322, %327 : tensor<f64>
    %329 = stablehlo.slice %292 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %330 = stablehlo.reshape %329 : (tensor<1xf64>) -> tensor<f64>
    %331 = stablehlo.slice %313 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %332 = stablehlo.reshape %331 : (tensor<1xf64>) -> tensor<f64>
    %333 = stablehlo.multiply %330, %332 : tensor<f64>
    %334 = stablehlo.subtract %328, %333 : tensor<f64>
    %335 = stablehlo.reshape %334 : (tensor<f64>) -> tensor<1xf64>
    %336 = stablehlo.multiply %294, %332 : tensor<f64>
    %337 = stablehlo.multiply %318, %326 : tensor<f64>
    %338 = stablehlo.subtract %336, %337 : tensor<f64>
    %339 = stablehlo.multiply %324, %320 : tensor<f64>
    %340 = stablehlo.add %338, %339 : tensor<f64>
    %341 = stablehlo.multiply %330, %315 : tensor<f64>
    %342 = stablehlo.add %340, %341 : tensor<f64>
    %343 = stablehlo.reshape %342 : (tensor<f64>) -> tensor<1xf64>
    %344 = stablehlo.multiply %294, %326 : tensor<f64>
    %345 = stablehlo.multiply %318, %332 : tensor<f64>
    %346 = stablehlo.add %344, %345 : tensor<f64>
    %347 = stablehlo.multiply %324, %315 : tensor<f64>
    %348 = stablehlo.subtract %346, %347 : tensor<f64>
    %349 = stablehlo.multiply %330, %320 : tensor<f64>
    %350 = stablehlo.add %348, %349 : tensor<f64>
    %351 = stablehlo.reshape %350 : (tensor<f64>) -> tensor<1xf64>
    %352 = stablehlo.multiply %294, %320 : tensor<f64>
    %353 = stablehlo.multiply %318, %315 : tensor<f64>
    %354 = stablehlo.subtract %352, %353 : tensor<f64>
    %355 = stablehlo.multiply %324, %332 : tensor<f64>
    %356 = stablehlo.subtract %354, %355 : tensor<f64>
    %357 = stablehlo.multiply %330, %326 : tensor<f64>
    %358 = stablehlo.subtract %356, %357 : tensor<f64>
    %359 = stablehlo.reshape %358 : (tensor<f64>) -> tensor<1xf64>
    %360 = stablehlo.concatenate %335, %343, %351, %359, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %361 = stablehlo.slice %360 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %362 = stablehlo.reshape %361 : (tensor<1xf64>) -> tensor<f64>
    %363 = stablehlo.reshape %362 : (tensor<f64>) -> tensor<1xf64>
    %364 = stablehlo.slice %360 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %365 = stablehlo.reshape %364 : (tensor<1xf64>) -> tensor<f64>
    %366 = stablehlo.reshape %365 : (tensor<f64>) -> tensor<1xf64>
    %367 = stablehlo.slice %360 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %368 = stablehlo.reshape %367 : (tensor<1xf64>) -> tensor<f64>
    %369 = stablehlo.reshape %368 : (tensor<f64>) -> tensor<1xf64>
    %370 = stablehlo.concatenate %363, %366, %369, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %371 = stablehlo.concatenate %240, %370, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %372 = stablehlo.slice %371 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %373 = stablehlo.slice %arg26 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %374 = stablehlo.divide %372, %373 : tensor<3xf64>
    %375 = stablehlo.slice %371 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %376 = stablehlo.slice %arg26 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %377 = stablehlo.reshape %376 : (tensor<1xf64>) -> tensor<f64>
    %378 = stablehlo.broadcast_in_dim %377, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %379 = stablehlo.divide %375, %378 : tensor<3xf64>
    %380 = stablehlo.concatenate %374, %379, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %381 = stablehlo.slice %380 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %382 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %383 = stablehlo.concatenate %381, %382, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %384 = stablehlo.slice %383 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %385 = stablehlo.reshape %384 : (tensor<1xf64>) -> tensor<f64>
    %386 = stablehlo.multiply %91, %385 : tensor<f64>
    %387 = stablehlo.slice %89 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %388 = stablehlo.reshape %387 : (tensor<1xf64>) -> tensor<f64>
    %389 = stablehlo.slice %383 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %390 = stablehlo.reshape %389 : (tensor<1xf64>) -> tensor<f64>
    %391 = stablehlo.multiply %388, %390 : tensor<f64>
    %392 = stablehlo.add %386, %391 : tensor<f64>
    %393 = stablehlo.slice %89 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %394 = stablehlo.reshape %393 : (tensor<1xf64>) -> tensor<f64>
    %395 = stablehlo.slice %383 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %396 = stablehlo.reshape %395 : (tensor<1xf64>) -> tensor<f64>
    %397 = stablehlo.multiply %394, %396 : tensor<f64>
    %398 = stablehlo.add %392, %397 : tensor<f64>
    %399 = stablehlo.slice %89 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %400 = stablehlo.reshape %399 : (tensor<1xf64>) -> tensor<f64>
    %401 = stablehlo.slice %383 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %402 = stablehlo.reshape %401 : (tensor<1xf64>) -> tensor<f64>
    %403 = stablehlo.multiply %400, %402 : tensor<f64>
    %404 = stablehlo.subtract %398, %403 : tensor<f64>
    %405 = stablehlo.reshape %404 : (tensor<f64>) -> tensor<1xf64>
    %406 = stablehlo.multiply %91, %402 : tensor<f64>
    %407 = stablehlo.multiply %388, %396 : tensor<f64>
    %408 = stablehlo.subtract %406, %407 : tensor<f64>
    %409 = stablehlo.multiply %394, %390 : tensor<f64>
    %410 = stablehlo.add %408, %409 : tensor<f64>
    %411 = stablehlo.multiply %400, %385 : tensor<f64>
    %412 = stablehlo.add %410, %411 : tensor<f64>
    %413 = stablehlo.reshape %412 : (tensor<f64>) -> tensor<1xf64>
    %414 = stablehlo.multiply %91, %396 : tensor<f64>
    %415 = stablehlo.multiply %388, %402 : tensor<f64>
    %416 = stablehlo.add %414, %415 : tensor<f64>
    %417 = stablehlo.multiply %394, %385 : tensor<f64>
    %418 = stablehlo.subtract %416, %417 : tensor<f64>
    %419 = stablehlo.multiply %400, %390 : tensor<f64>
    %420 = stablehlo.add %418, %419 : tensor<f64>
    %421 = stablehlo.reshape %420 : (tensor<f64>) -> tensor<1xf64>
    %422 = stablehlo.multiply %91, %390 : tensor<f64>
    %423 = stablehlo.multiply %388, %385 : tensor<f64>
    %424 = stablehlo.subtract %422, %423 : tensor<f64>
    %425 = stablehlo.multiply %394, %402 : tensor<f64>
    %426 = stablehlo.subtract %424, %425 : tensor<f64>
    %427 = stablehlo.multiply %400, %396 : tensor<f64>
    %428 = stablehlo.subtract %426, %427 : tensor<f64>
    %429 = stablehlo.reshape %428 : (tensor<f64>) -> tensor<1xf64>
    %430 = stablehlo.concatenate %405, %413, %421, %429, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %431 = stablehlo.slice %430 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %432 = stablehlo.reshape %431 : (tensor<1xf64>) -> tensor<f64>
    %433 = stablehlo.slice %89 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %434 = stablehlo.reshape %433 : (tensor<1xf64>) -> tensor<f64>
    %435 = stablehlo.negate %434 : tensor<f64>
    %436 = stablehlo.reshape %435 : (tensor<f64>) -> tensor<1xf64>
    %437 = stablehlo.slice %89 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %438 = stablehlo.reshape %437 : (tensor<1xf64>) -> tensor<f64>
    %439 = stablehlo.negate %438 : tensor<f64>
    %440 = stablehlo.reshape %439 : (tensor<f64>) -> tensor<1xf64>
    %441 = stablehlo.slice %89 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %442 = stablehlo.reshape %441 : (tensor<1xf64>) -> tensor<f64>
    %443 = stablehlo.negate %442 : tensor<f64>
    %444 = stablehlo.reshape %443 : (tensor<f64>) -> tensor<1xf64>
    %445 = stablehlo.slice %89 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %446 = stablehlo.reshape %445 : (tensor<1xf64>) -> tensor<f64>
    %447 = stablehlo.reshape %446 : (tensor<f64>) -> tensor<1xf64>
    %448 = stablehlo.concatenate %436, %440, %444, %447, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %449 = stablehlo.dot_general %89, %89, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %450 = stablehlo.broadcast_in_dim %449, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %451 = stablehlo.divide %448, %450 : tensor<4xf64>
    %452 = stablehlo.slice %451 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %453 = stablehlo.reshape %452 : (tensor<1xf64>) -> tensor<f64>
    %454 = stablehlo.multiply %432, %453 : tensor<f64>
    %455 = stablehlo.slice %430 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %456 = stablehlo.reshape %455 : (tensor<1xf64>) -> tensor<f64>
    %457 = stablehlo.slice %451 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %458 = stablehlo.reshape %457 : (tensor<1xf64>) -> tensor<f64>
    %459 = stablehlo.multiply %456, %458 : tensor<f64>
    %460 = stablehlo.add %454, %459 : tensor<f64>
    %461 = stablehlo.slice %430 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %462 = stablehlo.reshape %461 : (tensor<1xf64>) -> tensor<f64>
    %463 = stablehlo.slice %451 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %464 = stablehlo.reshape %463 : (tensor<1xf64>) -> tensor<f64>
    %465 = stablehlo.multiply %462, %464 : tensor<f64>
    %466 = stablehlo.add %460, %465 : tensor<f64>
    %467 = stablehlo.slice %430 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %468 = stablehlo.reshape %467 : (tensor<1xf64>) -> tensor<f64>
    %469 = stablehlo.slice %451 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %470 = stablehlo.reshape %469 : (tensor<1xf64>) -> tensor<f64>
    %471 = stablehlo.multiply %468, %470 : tensor<f64>
    %472 = stablehlo.subtract %466, %471 : tensor<f64>
    %473 = stablehlo.reshape %472 : (tensor<f64>) -> tensor<1xf64>
    %474 = stablehlo.multiply %432, %470 : tensor<f64>
    %475 = stablehlo.multiply %456, %464 : tensor<f64>
    %476 = stablehlo.subtract %474, %475 : tensor<f64>
    %477 = stablehlo.multiply %462, %458 : tensor<f64>
    %478 = stablehlo.add %476, %477 : tensor<f64>
    %479 = stablehlo.multiply %468, %453 : tensor<f64>
    %480 = stablehlo.add %478, %479 : tensor<f64>
    %481 = stablehlo.reshape %480 : (tensor<f64>) -> tensor<1xf64>
    %482 = stablehlo.multiply %432, %464 : tensor<f64>
    %483 = stablehlo.multiply %456, %470 : tensor<f64>
    %484 = stablehlo.add %482, %483 : tensor<f64>
    %485 = stablehlo.multiply %462, %453 : tensor<f64>
    %486 = stablehlo.subtract %484, %485 : tensor<f64>
    %487 = stablehlo.multiply %468, %458 : tensor<f64>
    %488 = stablehlo.add %486, %487 : tensor<f64>
    %489 = stablehlo.reshape %488 : (tensor<f64>) -> tensor<1xf64>
    %490 = stablehlo.multiply %432, %458 : tensor<f64>
    %491 = stablehlo.multiply %456, %453 : tensor<f64>
    %492 = stablehlo.subtract %490, %491 : tensor<f64>
    %493 = stablehlo.multiply %462, %470 : tensor<f64>
    %494 = stablehlo.subtract %492, %493 : tensor<f64>
    %495 = stablehlo.multiply %468, %464 : tensor<f64>
    %496 = stablehlo.subtract %494, %495 : tensor<f64>
    %497 = stablehlo.reshape %496 : (tensor<f64>) -> tensor<1xf64>
    %498 = stablehlo.concatenate %473, %481, %489, %497, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %499 = stablehlo.slice %498 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %500 = stablehlo.reshape %499 : (tensor<1xf64>) -> tensor<f64>
    %501 = stablehlo.reshape %500 : (tensor<f64>) -> tensor<1xf64>
    %502 = stablehlo.slice %498 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %503 = stablehlo.reshape %502 : (tensor<1xf64>) -> tensor<f64>
    %504 = stablehlo.reshape %503 : (tensor<f64>) -> tensor<1xf64>
    %505 = stablehlo.slice %498 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %506 = stablehlo.reshape %505 : (tensor<1xf64>) -> tensor<f64>
    %507 = stablehlo.reshape %506 : (tensor<f64>) -> tensor<1xf64>
    %508 = stablehlo.concatenate %501, %504, %507, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %509 = stablehlo.slice %89 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %510 = stablehlo.reshape %509 : (tensor<1xf64>) -> tensor<f64>
    %511 = stablehlo.slice %380 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %512 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %513 = stablehlo.concatenate %511, %512, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %514 = stablehlo.slice %513 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %515 = stablehlo.reshape %514 : (tensor<1xf64>) -> tensor<f64>
    %516 = stablehlo.multiply %510, %515 : tensor<f64>
    %517 = stablehlo.slice %89 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %518 = stablehlo.reshape %517 : (tensor<1xf64>) -> tensor<f64>
    %519 = stablehlo.slice %513 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %520 = stablehlo.reshape %519 : (tensor<1xf64>) -> tensor<f64>
    %521 = stablehlo.multiply %518, %520 : tensor<f64>
    %522 = stablehlo.add %516, %521 : tensor<f64>
    %523 = stablehlo.slice %89 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %524 = stablehlo.reshape %523 : (tensor<1xf64>) -> tensor<f64>
    %525 = stablehlo.slice %513 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %526 = stablehlo.reshape %525 : (tensor<1xf64>) -> tensor<f64>
    %527 = stablehlo.multiply %524, %526 : tensor<f64>
    %528 = stablehlo.add %522, %527 : tensor<f64>
    %529 = stablehlo.slice %89 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %530 = stablehlo.reshape %529 : (tensor<1xf64>) -> tensor<f64>
    %531 = stablehlo.slice %513 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %532 = stablehlo.reshape %531 : (tensor<1xf64>) -> tensor<f64>
    %533 = stablehlo.multiply %530, %532 : tensor<f64>
    %534 = stablehlo.subtract %528, %533 : tensor<f64>
    %535 = stablehlo.reshape %534 : (tensor<f64>) -> tensor<1xf64>
    %536 = stablehlo.multiply %510, %532 : tensor<f64>
    %537 = stablehlo.multiply %518, %526 : tensor<f64>
    %538 = stablehlo.subtract %536, %537 : tensor<f64>
    %539 = stablehlo.multiply %524, %520 : tensor<f64>
    %540 = stablehlo.add %538, %539 : tensor<f64>
    %541 = stablehlo.multiply %530, %515 : tensor<f64>
    %542 = stablehlo.add %540, %541 : tensor<f64>
    %543 = stablehlo.reshape %542 : (tensor<f64>) -> tensor<1xf64>
    %544 = stablehlo.multiply %510, %526 : tensor<f64>
    %545 = stablehlo.multiply %518, %532 : tensor<f64>
    %546 = stablehlo.add %544, %545 : tensor<f64>
    %547 = stablehlo.multiply %524, %515 : tensor<f64>
    %548 = stablehlo.subtract %546, %547 : tensor<f64>
    %549 = stablehlo.multiply %530, %520 : tensor<f64>
    %550 = stablehlo.add %548, %549 : tensor<f64>
    %551 = stablehlo.reshape %550 : (tensor<f64>) -> tensor<1xf64>
    %552 = stablehlo.multiply %510, %520 : tensor<f64>
    %553 = stablehlo.multiply %518, %515 : tensor<f64>
    %554 = stablehlo.subtract %552, %553 : tensor<f64>
    %555 = stablehlo.multiply %524, %532 : tensor<f64>
    %556 = stablehlo.subtract %554, %555 : tensor<f64>
    %557 = stablehlo.multiply %530, %526 : tensor<f64>
    %558 = stablehlo.subtract %556, %557 : tensor<f64>
    %559 = stablehlo.reshape %558 : (tensor<f64>) -> tensor<1xf64>
    %560 = stablehlo.concatenate %535, %543, %551, %559, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %561 = stablehlo.slice %560 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %562 = stablehlo.reshape %561 : (tensor<1xf64>) -> tensor<f64>
    %563 = stablehlo.slice %89 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %564 = stablehlo.reshape %563 : (tensor<1xf64>) -> tensor<f64>
    %565 = stablehlo.negate %564 : tensor<f64>
    %566 = stablehlo.reshape %565 : (tensor<f64>) -> tensor<1xf64>
    %567 = stablehlo.slice %89 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %568 = stablehlo.reshape %567 : (tensor<1xf64>) -> tensor<f64>
    %569 = stablehlo.negate %568 : tensor<f64>
    %570 = stablehlo.reshape %569 : (tensor<f64>) -> tensor<1xf64>
    %571 = stablehlo.slice %89 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %572 = stablehlo.reshape %571 : (tensor<1xf64>) -> tensor<f64>
    %573 = stablehlo.negate %572 : tensor<f64>
    %574 = stablehlo.reshape %573 : (tensor<f64>) -> tensor<1xf64>
    %575 = stablehlo.slice %89 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %576 = stablehlo.reshape %575 : (tensor<1xf64>) -> tensor<f64>
    %577 = stablehlo.reshape %576 : (tensor<f64>) -> tensor<1xf64>
    %578 = stablehlo.concatenate %566, %570, %574, %577, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %579 = stablehlo.dot_general %89, %89, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %580 = stablehlo.broadcast_in_dim %579, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %581 = stablehlo.divide %578, %580 : tensor<4xf64>
    %582 = stablehlo.slice %581 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %583 = stablehlo.reshape %582 : (tensor<1xf64>) -> tensor<f64>
    %584 = stablehlo.multiply %562, %583 : tensor<f64>
    %585 = stablehlo.slice %560 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %586 = stablehlo.reshape %585 : (tensor<1xf64>) -> tensor<f64>
    %587 = stablehlo.slice %581 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %588 = stablehlo.reshape %587 : (tensor<1xf64>) -> tensor<f64>
    %589 = stablehlo.multiply %586, %588 : tensor<f64>
    %590 = stablehlo.add %584, %589 : tensor<f64>
    %591 = stablehlo.slice %560 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %592 = stablehlo.reshape %591 : (tensor<1xf64>) -> tensor<f64>
    %593 = stablehlo.slice %581 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %594 = stablehlo.reshape %593 : (tensor<1xf64>) -> tensor<f64>
    %595 = stablehlo.multiply %592, %594 : tensor<f64>
    %596 = stablehlo.add %590, %595 : tensor<f64>
    %597 = stablehlo.slice %560 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %598 = stablehlo.reshape %597 : (tensor<1xf64>) -> tensor<f64>
    %599 = stablehlo.slice %581 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %600 = stablehlo.reshape %599 : (tensor<1xf64>) -> tensor<f64>
    %601 = stablehlo.multiply %598, %600 : tensor<f64>
    %602 = stablehlo.subtract %596, %601 : tensor<f64>
    %603 = stablehlo.reshape %602 : (tensor<f64>) -> tensor<1xf64>
    %604 = stablehlo.multiply %562, %600 : tensor<f64>
    %605 = stablehlo.multiply %586, %594 : tensor<f64>
    %606 = stablehlo.subtract %604, %605 : tensor<f64>
    %607 = stablehlo.multiply %592, %588 : tensor<f64>
    %608 = stablehlo.add %606, %607 : tensor<f64>
    %609 = stablehlo.multiply %598, %583 : tensor<f64>
    %610 = stablehlo.add %608, %609 : tensor<f64>
    %611 = stablehlo.reshape %610 : (tensor<f64>) -> tensor<1xf64>
    %612 = stablehlo.multiply %562, %594 : tensor<f64>
    %613 = stablehlo.multiply %586, %600 : tensor<f64>
    %614 = stablehlo.add %612, %613 : tensor<f64>
    %615 = stablehlo.multiply %592, %583 : tensor<f64>
    %616 = stablehlo.subtract %614, %615 : tensor<f64>
    %617 = stablehlo.multiply %598, %588 : tensor<f64>
    %618 = stablehlo.add %616, %617 : tensor<f64>
    %619 = stablehlo.reshape %618 : (tensor<f64>) -> tensor<1xf64>
    %620 = stablehlo.multiply %562, %588 : tensor<f64>
    %621 = stablehlo.multiply %586, %583 : tensor<f64>
    %622 = stablehlo.subtract %620, %621 : tensor<f64>
    %623 = stablehlo.multiply %592, %600 : tensor<f64>
    %624 = stablehlo.subtract %622, %623 : tensor<f64>
    %625 = stablehlo.multiply %598, %594 : tensor<f64>
    %626 = stablehlo.subtract %624, %625 : tensor<f64>
    %627 = stablehlo.reshape %626 : (tensor<f64>) -> tensor<1xf64>
    %628 = stablehlo.concatenate %603, %611, %619, %627, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %629 = stablehlo.slice %628 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %630 = stablehlo.reshape %629 : (tensor<1xf64>) -> tensor<f64>
    %631 = stablehlo.reshape %630 : (tensor<f64>) -> tensor<1xf64>
    %632 = stablehlo.slice %628 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %633 = stablehlo.reshape %632 : (tensor<1xf64>) -> tensor<f64>
    %634 = stablehlo.reshape %633 : (tensor<f64>) -> tensor<1xf64>
    %635 = stablehlo.slice %628 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %636 = stablehlo.reshape %635 : (tensor<1xf64>) -> tensor<f64>
    %637 = stablehlo.reshape %636 : (tensor<f64>) -> tensor<1xf64>
    %638 = stablehlo.concatenate %631, %634, %637, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %639 = stablehlo.concatenate %508, %638, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %640 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %641 = stablehlo.reshape %arg22 : (tensor<f64>) -> tensor<f64>
    %cst_7 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %642 = stablehlo.multiply %cst_7, %641 : tensor<f64>
    %643 = stablehlo.broadcast_in_dim %642, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %644 = stablehlo.multiply %643, %arg2 : tensor<6xf64>
    %645 = stablehlo.slice %644 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_8 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %646 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %647 = stablehlo.divide %645, %646 : tensor<3xf64>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %648 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %649 = stablehlo.concatenate %647, %648, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %650 = stablehlo.slice %649 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %651 = stablehlo.reshape %650 : (tensor<1xf64>) -> tensor<f64>
    %652 = stablehlo.slice %640 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %653 = stablehlo.reshape %652 : (tensor<1xf64>) -> tensor<f64>
    %654 = stablehlo.multiply %651, %653 : tensor<f64>
    %655 = stablehlo.slice %649 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %656 = stablehlo.reshape %655 : (tensor<1xf64>) -> tensor<f64>
    %657 = stablehlo.slice %640 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %658 = stablehlo.reshape %657 : (tensor<1xf64>) -> tensor<f64>
    %659 = stablehlo.multiply %656, %658 : tensor<f64>
    %660 = stablehlo.add %654, %659 : tensor<f64>
    %661 = stablehlo.slice %649 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %662 = stablehlo.reshape %661 : (tensor<1xf64>) -> tensor<f64>
    %663 = stablehlo.slice %640 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %664 = stablehlo.reshape %663 : (tensor<1xf64>) -> tensor<f64>
    %665 = stablehlo.multiply %662, %664 : tensor<f64>
    %666 = stablehlo.add %660, %665 : tensor<f64>
    %667 = stablehlo.slice %649 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %668 = stablehlo.reshape %667 : (tensor<1xf64>) -> tensor<f64>
    %669 = stablehlo.slice %640 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %670 = stablehlo.reshape %669 : (tensor<1xf64>) -> tensor<f64>
    %671 = stablehlo.multiply %668, %670 : tensor<f64>
    %672 = stablehlo.subtract %666, %671 : tensor<f64>
    %673 = stablehlo.reshape %672 : (tensor<f64>) -> tensor<1xf64>
    %674 = stablehlo.multiply %651, %670 : tensor<f64>
    %675 = stablehlo.multiply %656, %664 : tensor<f64>
    %676 = stablehlo.subtract %674, %675 : tensor<f64>
    %677 = stablehlo.multiply %662, %658 : tensor<f64>
    %678 = stablehlo.add %676, %677 : tensor<f64>
    %679 = stablehlo.multiply %668, %653 : tensor<f64>
    %680 = stablehlo.add %678, %679 : tensor<f64>
    %681 = stablehlo.reshape %680 : (tensor<f64>) -> tensor<1xf64>
    %682 = stablehlo.multiply %651, %664 : tensor<f64>
    %683 = stablehlo.multiply %656, %670 : tensor<f64>
    %684 = stablehlo.add %682, %683 : tensor<f64>
    %685 = stablehlo.multiply %662, %653 : tensor<f64>
    %686 = stablehlo.subtract %684, %685 : tensor<f64>
    %687 = stablehlo.multiply %668, %658 : tensor<f64>
    %688 = stablehlo.add %686, %687 : tensor<f64>
    %689 = stablehlo.reshape %688 : (tensor<f64>) -> tensor<1xf64>
    %690 = stablehlo.multiply %651, %658 : tensor<f64>
    %691 = stablehlo.multiply %656, %653 : tensor<f64>
    %692 = stablehlo.subtract %690, %691 : tensor<f64>
    %693 = stablehlo.multiply %662, %670 : tensor<f64>
    %694 = stablehlo.subtract %692, %693 : tensor<f64>
    %695 = stablehlo.multiply %668, %664 : tensor<f64>
    %696 = stablehlo.subtract %694, %695 : tensor<f64>
    %697 = stablehlo.reshape %696 : (tensor<f64>) -> tensor<1xf64>
    %698 = stablehlo.concatenate %673, %681, %689, %697, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %699 = stablehlo.add %640, %698 : tensor<4xf64>
    %700 = stablehlo.dot_general %699, %699, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %701 = stablehlo.sqrt %700 : tensor<f64>
    %702 = stablehlo.broadcast_in_dim %701, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %703 = stablehlo.divide %699, %702 : tensor<4xf64>
    %704 = stablehlo.slice %arg1 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %705 = stablehlo.slice %644 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %706 = stablehlo.add %704, %705 : tensor<3xf64>
    %707 = stablehlo.concatenate %703, %706, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %708 = stablehlo.broadcast_in_dim %642, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %709 = stablehlo.multiply %708, %639 : tensor<6xf64>
    %710 = stablehlo.add %arg2, %709 : tensor<6xf64>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %711 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %712 = call @inner_124(%711, %arg26) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %713 = call @inner_126(%13, %712, %707) : (tensor<f64>, tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %714 = call @inner_127(%707, %12, %713) : (tensor<7xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %715 = stablehlo.slice %707 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %716 = stablehlo.slice %715 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %717 = stablehlo.reshape %716 : (tensor<1xf64>) -> tensor<f64>
    %718 = stablehlo.slice %715 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %719 = stablehlo.reshape %718 : (tensor<1xf64>) -> tensor<f64>
    %720 = stablehlo.negate %719 : tensor<f64>
    %721 = stablehlo.reshape %720 : (tensor<f64>) -> tensor<1xf64>
    %722 = stablehlo.slice %715 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %723 = stablehlo.reshape %722 : (tensor<1xf64>) -> tensor<f64>
    %724 = stablehlo.negate %723 : tensor<f64>
    %725 = stablehlo.reshape %724 : (tensor<f64>) -> tensor<1xf64>
    %726 = stablehlo.slice %715 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %727 = stablehlo.reshape %726 : (tensor<1xf64>) -> tensor<f64>
    %728 = stablehlo.negate %727 : tensor<f64>
    %729 = stablehlo.reshape %728 : (tensor<f64>) -> tensor<1xf64>
    %730 = stablehlo.slice %715 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %731 = stablehlo.reshape %730 : (tensor<1xf64>) -> tensor<f64>
    %732 = stablehlo.reshape %731 : (tensor<f64>) -> tensor<1xf64>
    %733 = stablehlo.concatenate %721, %725, %729, %732, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %734 = stablehlo.dot_general %715, %715, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %735 = stablehlo.broadcast_in_dim %734, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %736 = stablehlo.divide %733, %735 : tensor<4xf64>
    %737 = stablehlo.slice %736 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %738 = stablehlo.reshape %737 : (tensor<1xf64>) -> tensor<f64>
    %739 = stablehlo.slice %714 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %740 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %741 = stablehlo.concatenate %739, %740, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %742 = stablehlo.slice %741 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %743 = stablehlo.reshape %742 : (tensor<1xf64>) -> tensor<f64>
    %744 = stablehlo.multiply %738, %743 : tensor<f64>
    %745 = stablehlo.slice %736 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %746 = stablehlo.reshape %745 : (tensor<1xf64>) -> tensor<f64>
    %747 = stablehlo.slice %741 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %748 = stablehlo.reshape %747 : (tensor<1xf64>) -> tensor<f64>
    %749 = stablehlo.multiply %746, %748 : tensor<f64>
    %750 = stablehlo.add %744, %749 : tensor<f64>
    %751 = stablehlo.slice %736 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %752 = stablehlo.reshape %751 : (tensor<1xf64>) -> tensor<f64>
    %753 = stablehlo.slice %741 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %754 = stablehlo.reshape %753 : (tensor<1xf64>) -> tensor<f64>
    %755 = stablehlo.multiply %752, %754 : tensor<f64>
    %756 = stablehlo.add %750, %755 : tensor<f64>
    %757 = stablehlo.slice %736 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %758 = stablehlo.reshape %757 : (tensor<1xf64>) -> tensor<f64>
    %759 = stablehlo.slice %741 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %760 = stablehlo.reshape %759 : (tensor<1xf64>) -> tensor<f64>
    %761 = stablehlo.multiply %758, %760 : tensor<f64>
    %762 = stablehlo.subtract %756, %761 : tensor<f64>
    %763 = stablehlo.reshape %762 : (tensor<f64>) -> tensor<1xf64>
    %764 = stablehlo.multiply %738, %760 : tensor<f64>
    %765 = stablehlo.multiply %746, %754 : tensor<f64>
    %766 = stablehlo.subtract %764, %765 : tensor<f64>
    %767 = stablehlo.multiply %752, %748 : tensor<f64>
    %768 = stablehlo.add %766, %767 : tensor<f64>
    %769 = stablehlo.multiply %758, %743 : tensor<f64>
    %770 = stablehlo.add %768, %769 : tensor<f64>
    %771 = stablehlo.reshape %770 : (tensor<f64>) -> tensor<1xf64>
    %772 = stablehlo.multiply %738, %754 : tensor<f64>
    %773 = stablehlo.multiply %746, %760 : tensor<f64>
    %774 = stablehlo.add %772, %773 : tensor<f64>
    %775 = stablehlo.multiply %752, %743 : tensor<f64>
    %776 = stablehlo.subtract %774, %775 : tensor<f64>
    %777 = stablehlo.multiply %758, %748 : tensor<f64>
    %778 = stablehlo.add %776, %777 : tensor<f64>
    %779 = stablehlo.reshape %778 : (tensor<f64>) -> tensor<1xf64>
    %780 = stablehlo.multiply %738, %748 : tensor<f64>
    %781 = stablehlo.multiply %746, %743 : tensor<f64>
    %782 = stablehlo.subtract %780, %781 : tensor<f64>
    %783 = stablehlo.multiply %752, %760 : tensor<f64>
    %784 = stablehlo.subtract %782, %783 : tensor<f64>
    %785 = stablehlo.multiply %758, %754 : tensor<f64>
    %786 = stablehlo.subtract %784, %785 : tensor<f64>
    %787 = stablehlo.reshape %786 : (tensor<f64>) -> tensor<1xf64>
    %788 = stablehlo.concatenate %763, %771, %779, %787, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %789 = stablehlo.slice %788 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %790 = stablehlo.reshape %789 : (tensor<1xf64>) -> tensor<f64>
    %791 = stablehlo.slice %736 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %792 = stablehlo.reshape %791 : (tensor<1xf64>) -> tensor<f64>
    %793 = stablehlo.negate %792 : tensor<f64>
    %794 = stablehlo.reshape %793 : (tensor<f64>) -> tensor<1xf64>
    %795 = stablehlo.slice %736 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %796 = stablehlo.reshape %795 : (tensor<1xf64>) -> tensor<f64>
    %797 = stablehlo.negate %796 : tensor<f64>
    %798 = stablehlo.reshape %797 : (tensor<f64>) -> tensor<1xf64>
    %799 = stablehlo.slice %736 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %800 = stablehlo.reshape %799 : (tensor<1xf64>) -> tensor<f64>
    %801 = stablehlo.negate %800 : tensor<f64>
    %802 = stablehlo.reshape %801 : (tensor<f64>) -> tensor<1xf64>
    %803 = stablehlo.slice %736 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %804 = stablehlo.reshape %803 : (tensor<1xf64>) -> tensor<f64>
    %805 = stablehlo.reshape %804 : (tensor<f64>) -> tensor<1xf64>
    %806 = stablehlo.concatenate %794, %798, %802, %805, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %807 = stablehlo.dot_general %736, %736, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %808 = stablehlo.broadcast_in_dim %807, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %809 = stablehlo.divide %806, %808 : tensor<4xf64>
    %810 = stablehlo.slice %809 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %811 = stablehlo.reshape %810 : (tensor<1xf64>) -> tensor<f64>
    %812 = stablehlo.multiply %790, %811 : tensor<f64>
    %813 = stablehlo.slice %788 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %814 = stablehlo.reshape %813 : (tensor<1xf64>) -> tensor<f64>
    %815 = stablehlo.slice %809 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %816 = stablehlo.reshape %815 : (tensor<1xf64>) -> tensor<f64>
    %817 = stablehlo.multiply %814, %816 : tensor<f64>
    %818 = stablehlo.add %812, %817 : tensor<f64>
    %819 = stablehlo.slice %788 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %820 = stablehlo.reshape %819 : (tensor<1xf64>) -> tensor<f64>
    %821 = stablehlo.slice %809 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %822 = stablehlo.reshape %821 : (tensor<1xf64>) -> tensor<f64>
    %823 = stablehlo.multiply %820, %822 : tensor<f64>
    %824 = stablehlo.add %818, %823 : tensor<f64>
    %825 = stablehlo.slice %788 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %826 = stablehlo.reshape %825 : (tensor<1xf64>) -> tensor<f64>
    %827 = stablehlo.slice %809 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %828 = stablehlo.reshape %827 : (tensor<1xf64>) -> tensor<f64>
    %829 = stablehlo.multiply %826, %828 : tensor<f64>
    %830 = stablehlo.subtract %824, %829 : tensor<f64>
    %831 = stablehlo.reshape %830 : (tensor<f64>) -> tensor<1xf64>
    %832 = stablehlo.multiply %790, %828 : tensor<f64>
    %833 = stablehlo.multiply %814, %822 : tensor<f64>
    %834 = stablehlo.subtract %832, %833 : tensor<f64>
    %835 = stablehlo.multiply %820, %816 : tensor<f64>
    %836 = stablehlo.add %834, %835 : tensor<f64>
    %837 = stablehlo.multiply %826, %811 : tensor<f64>
    %838 = stablehlo.add %836, %837 : tensor<f64>
    %839 = stablehlo.reshape %838 : (tensor<f64>) -> tensor<1xf64>
    %840 = stablehlo.multiply %790, %822 : tensor<f64>
    %841 = stablehlo.multiply %814, %828 : tensor<f64>
    %842 = stablehlo.add %840, %841 : tensor<f64>
    %843 = stablehlo.multiply %820, %811 : tensor<f64>
    %844 = stablehlo.subtract %842, %843 : tensor<f64>
    %845 = stablehlo.multiply %826, %816 : tensor<f64>
    %846 = stablehlo.add %844, %845 : tensor<f64>
    %847 = stablehlo.reshape %846 : (tensor<f64>) -> tensor<1xf64>
    %848 = stablehlo.multiply %790, %816 : tensor<f64>
    %849 = stablehlo.multiply %814, %811 : tensor<f64>
    %850 = stablehlo.subtract %848, %849 : tensor<f64>
    %851 = stablehlo.multiply %820, %828 : tensor<f64>
    %852 = stablehlo.subtract %850, %851 : tensor<f64>
    %853 = stablehlo.multiply %826, %822 : tensor<f64>
    %854 = stablehlo.subtract %852, %853 : tensor<f64>
    %855 = stablehlo.reshape %854 : (tensor<f64>) -> tensor<1xf64>
    %856 = stablehlo.concatenate %831, %839, %847, %855, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %857 = stablehlo.slice %856 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %858 = stablehlo.reshape %857 : (tensor<1xf64>) -> tensor<f64>
    %859 = stablehlo.reshape %858 : (tensor<f64>) -> tensor<1xf64>
    %860 = stablehlo.slice %856 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %861 = stablehlo.reshape %860 : (tensor<1xf64>) -> tensor<f64>
    %862 = stablehlo.reshape %861 : (tensor<f64>) -> tensor<1xf64>
    %863 = stablehlo.slice %856 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %864 = stablehlo.reshape %863 : (tensor<1xf64>) -> tensor<f64>
    %865 = stablehlo.reshape %864 : (tensor<f64>) -> tensor<1xf64>
    %866 = stablehlo.concatenate %859, %862, %865, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %867 = stablehlo.slice %736 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %868 = stablehlo.reshape %867 : (tensor<1xf64>) -> tensor<f64>
    %869 = stablehlo.slice %714 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %870 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %871 = stablehlo.concatenate %869, %870, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %872 = stablehlo.slice %871 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %873 = stablehlo.reshape %872 : (tensor<1xf64>) -> tensor<f64>
    %874 = stablehlo.multiply %868, %873 : tensor<f64>
    %875 = stablehlo.slice %736 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %876 = stablehlo.reshape %875 : (tensor<1xf64>) -> tensor<f64>
    %877 = stablehlo.slice %871 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %878 = stablehlo.reshape %877 : (tensor<1xf64>) -> tensor<f64>
    %879 = stablehlo.multiply %876, %878 : tensor<f64>
    %880 = stablehlo.add %874, %879 : tensor<f64>
    %881 = stablehlo.slice %736 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %882 = stablehlo.reshape %881 : (tensor<1xf64>) -> tensor<f64>
    %883 = stablehlo.slice %871 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %884 = stablehlo.reshape %883 : (tensor<1xf64>) -> tensor<f64>
    %885 = stablehlo.multiply %882, %884 : tensor<f64>
    %886 = stablehlo.add %880, %885 : tensor<f64>
    %887 = stablehlo.slice %736 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %888 = stablehlo.reshape %887 : (tensor<1xf64>) -> tensor<f64>
    %889 = stablehlo.slice %871 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %890 = stablehlo.reshape %889 : (tensor<1xf64>) -> tensor<f64>
    %891 = stablehlo.multiply %888, %890 : tensor<f64>
    %892 = stablehlo.subtract %886, %891 : tensor<f64>
    %893 = stablehlo.reshape %892 : (tensor<f64>) -> tensor<1xf64>
    %894 = stablehlo.multiply %868, %890 : tensor<f64>
    %895 = stablehlo.multiply %876, %884 : tensor<f64>
    %896 = stablehlo.subtract %894, %895 : tensor<f64>
    %897 = stablehlo.multiply %882, %878 : tensor<f64>
    %898 = stablehlo.add %896, %897 : tensor<f64>
    %899 = stablehlo.multiply %888, %873 : tensor<f64>
    %900 = stablehlo.add %898, %899 : tensor<f64>
    %901 = stablehlo.reshape %900 : (tensor<f64>) -> tensor<1xf64>
    %902 = stablehlo.multiply %868, %884 : tensor<f64>
    %903 = stablehlo.multiply %876, %890 : tensor<f64>
    %904 = stablehlo.add %902, %903 : tensor<f64>
    %905 = stablehlo.multiply %882, %873 : tensor<f64>
    %906 = stablehlo.subtract %904, %905 : tensor<f64>
    %907 = stablehlo.multiply %888, %878 : tensor<f64>
    %908 = stablehlo.add %906, %907 : tensor<f64>
    %909 = stablehlo.reshape %908 : (tensor<f64>) -> tensor<1xf64>
    %910 = stablehlo.multiply %868, %878 : tensor<f64>
    %911 = stablehlo.multiply %876, %873 : tensor<f64>
    %912 = stablehlo.subtract %910, %911 : tensor<f64>
    %913 = stablehlo.multiply %882, %890 : tensor<f64>
    %914 = stablehlo.subtract %912, %913 : tensor<f64>
    %915 = stablehlo.multiply %888, %884 : tensor<f64>
    %916 = stablehlo.subtract %914, %915 : tensor<f64>
    %917 = stablehlo.reshape %916 : (tensor<f64>) -> tensor<1xf64>
    %918 = stablehlo.concatenate %893, %901, %909, %917, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %919 = stablehlo.slice %918 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %920 = stablehlo.reshape %919 : (tensor<1xf64>) -> tensor<f64>
    %921 = stablehlo.slice %736 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %922 = stablehlo.reshape %921 : (tensor<1xf64>) -> tensor<f64>
    %923 = stablehlo.negate %922 : tensor<f64>
    %924 = stablehlo.reshape %923 : (tensor<f64>) -> tensor<1xf64>
    %925 = stablehlo.slice %736 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %926 = stablehlo.reshape %925 : (tensor<1xf64>) -> tensor<f64>
    %927 = stablehlo.negate %926 : tensor<f64>
    %928 = stablehlo.reshape %927 : (tensor<f64>) -> tensor<1xf64>
    %929 = stablehlo.slice %736 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %930 = stablehlo.reshape %929 : (tensor<1xf64>) -> tensor<f64>
    %931 = stablehlo.negate %930 : tensor<f64>
    %932 = stablehlo.reshape %931 : (tensor<f64>) -> tensor<1xf64>
    %933 = stablehlo.slice %736 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %934 = stablehlo.reshape %933 : (tensor<1xf64>) -> tensor<f64>
    %935 = stablehlo.reshape %934 : (tensor<f64>) -> tensor<1xf64>
    %936 = stablehlo.concatenate %924, %928, %932, %935, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %937 = stablehlo.dot_general %736, %736, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %938 = stablehlo.broadcast_in_dim %937, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %939 = stablehlo.divide %936, %938 : tensor<4xf64>
    %940 = stablehlo.slice %939 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %941 = stablehlo.reshape %940 : (tensor<1xf64>) -> tensor<f64>
    %942 = stablehlo.multiply %920, %941 : tensor<f64>
    %943 = stablehlo.slice %918 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %944 = stablehlo.reshape %943 : (tensor<1xf64>) -> tensor<f64>
    %945 = stablehlo.slice %939 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %946 = stablehlo.reshape %945 : (tensor<1xf64>) -> tensor<f64>
    %947 = stablehlo.multiply %944, %946 : tensor<f64>
    %948 = stablehlo.add %942, %947 : tensor<f64>
    %949 = stablehlo.slice %918 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %950 = stablehlo.reshape %949 : (tensor<1xf64>) -> tensor<f64>
    %951 = stablehlo.slice %939 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %952 = stablehlo.reshape %951 : (tensor<1xf64>) -> tensor<f64>
    %953 = stablehlo.multiply %950, %952 : tensor<f64>
    %954 = stablehlo.add %948, %953 : tensor<f64>
    %955 = stablehlo.slice %918 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %956 = stablehlo.reshape %955 : (tensor<1xf64>) -> tensor<f64>
    %957 = stablehlo.slice %939 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %958 = stablehlo.reshape %957 : (tensor<1xf64>) -> tensor<f64>
    %959 = stablehlo.multiply %956, %958 : tensor<f64>
    %960 = stablehlo.subtract %954, %959 : tensor<f64>
    %961 = stablehlo.reshape %960 : (tensor<f64>) -> tensor<1xf64>
    %962 = stablehlo.multiply %920, %958 : tensor<f64>
    %963 = stablehlo.multiply %944, %952 : tensor<f64>
    %964 = stablehlo.subtract %962, %963 : tensor<f64>
    %965 = stablehlo.multiply %950, %946 : tensor<f64>
    %966 = stablehlo.add %964, %965 : tensor<f64>
    %967 = stablehlo.multiply %956, %941 : tensor<f64>
    %968 = stablehlo.add %966, %967 : tensor<f64>
    %969 = stablehlo.reshape %968 : (tensor<f64>) -> tensor<1xf64>
    %970 = stablehlo.multiply %920, %952 : tensor<f64>
    %971 = stablehlo.multiply %944, %958 : tensor<f64>
    %972 = stablehlo.add %970, %971 : tensor<f64>
    %973 = stablehlo.multiply %950, %941 : tensor<f64>
    %974 = stablehlo.subtract %972, %973 : tensor<f64>
    %975 = stablehlo.multiply %956, %946 : tensor<f64>
    %976 = stablehlo.add %974, %975 : tensor<f64>
    %977 = stablehlo.reshape %976 : (tensor<f64>) -> tensor<1xf64>
    %978 = stablehlo.multiply %920, %946 : tensor<f64>
    %979 = stablehlo.multiply %944, %941 : tensor<f64>
    %980 = stablehlo.subtract %978, %979 : tensor<f64>
    %981 = stablehlo.multiply %950, %958 : tensor<f64>
    %982 = stablehlo.subtract %980, %981 : tensor<f64>
    %983 = stablehlo.multiply %956, %952 : tensor<f64>
    %984 = stablehlo.subtract %982, %983 : tensor<f64>
    %985 = stablehlo.reshape %984 : (tensor<f64>) -> tensor<1xf64>
    %986 = stablehlo.concatenate %961, %969, %977, %985, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %987 = stablehlo.slice %986 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %988 = stablehlo.reshape %987 : (tensor<1xf64>) -> tensor<f64>
    %989 = stablehlo.reshape %988 : (tensor<f64>) -> tensor<1xf64>
    %990 = stablehlo.slice %986 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %991 = stablehlo.reshape %990 : (tensor<1xf64>) -> tensor<f64>
    %992 = stablehlo.reshape %991 : (tensor<f64>) -> tensor<1xf64>
    %993 = stablehlo.slice %986 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %994 = stablehlo.reshape %993 : (tensor<1xf64>) -> tensor<f64>
    %995 = stablehlo.reshape %994 : (tensor<f64>) -> tensor<1xf64>
    %996 = stablehlo.concatenate %989, %992, %995, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %997 = stablehlo.concatenate %866, %996, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %998 = stablehlo.slice %997 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %999 = stablehlo.slice %arg26 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %1000 = stablehlo.divide %998, %999 : tensor<3xf64>
    %1001 = stablehlo.slice %997 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1002 = stablehlo.slice %arg26 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1003 = stablehlo.reshape %1002 : (tensor<1xf64>) -> tensor<f64>
    %1004 = stablehlo.broadcast_in_dim %1003, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1005 = stablehlo.divide %1001, %1004 : tensor<3xf64>
    %1006 = stablehlo.concatenate %1000, %1005, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1007 = stablehlo.slice %1006 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1008 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1009 = stablehlo.concatenate %1007, %1008, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1010 = stablehlo.slice %1009 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1011 = stablehlo.reshape %1010 : (tensor<1xf64>) -> tensor<f64>
    %1012 = stablehlo.multiply %717, %1011 : tensor<f64>
    %1013 = stablehlo.slice %715 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1014 = stablehlo.reshape %1013 : (tensor<1xf64>) -> tensor<f64>
    %1015 = stablehlo.slice %1009 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1016 = stablehlo.reshape %1015 : (tensor<1xf64>) -> tensor<f64>
    %1017 = stablehlo.multiply %1014, %1016 : tensor<f64>
    %1018 = stablehlo.add %1012, %1017 : tensor<f64>
    %1019 = stablehlo.slice %715 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1020 = stablehlo.reshape %1019 : (tensor<1xf64>) -> tensor<f64>
    %1021 = stablehlo.slice %1009 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1022 = stablehlo.reshape %1021 : (tensor<1xf64>) -> tensor<f64>
    %1023 = stablehlo.multiply %1020, %1022 : tensor<f64>
    %1024 = stablehlo.add %1018, %1023 : tensor<f64>
    %1025 = stablehlo.slice %715 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1026 = stablehlo.reshape %1025 : (tensor<1xf64>) -> tensor<f64>
    %1027 = stablehlo.slice %1009 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1028 = stablehlo.reshape %1027 : (tensor<1xf64>) -> tensor<f64>
    %1029 = stablehlo.multiply %1026, %1028 : tensor<f64>
    %1030 = stablehlo.subtract %1024, %1029 : tensor<f64>
    %1031 = stablehlo.reshape %1030 : (tensor<f64>) -> tensor<1xf64>
    %1032 = stablehlo.multiply %717, %1028 : tensor<f64>
    %1033 = stablehlo.multiply %1014, %1022 : tensor<f64>
    %1034 = stablehlo.subtract %1032, %1033 : tensor<f64>
    %1035 = stablehlo.multiply %1020, %1016 : tensor<f64>
    %1036 = stablehlo.add %1034, %1035 : tensor<f64>
    %1037 = stablehlo.multiply %1026, %1011 : tensor<f64>
    %1038 = stablehlo.add %1036, %1037 : tensor<f64>
    %1039 = stablehlo.reshape %1038 : (tensor<f64>) -> tensor<1xf64>
    %1040 = stablehlo.multiply %717, %1022 : tensor<f64>
    %1041 = stablehlo.multiply %1014, %1028 : tensor<f64>
    %1042 = stablehlo.add %1040, %1041 : tensor<f64>
    %1043 = stablehlo.multiply %1020, %1011 : tensor<f64>
    %1044 = stablehlo.subtract %1042, %1043 : tensor<f64>
    %1045 = stablehlo.multiply %1026, %1016 : tensor<f64>
    %1046 = stablehlo.add %1044, %1045 : tensor<f64>
    %1047 = stablehlo.reshape %1046 : (tensor<f64>) -> tensor<1xf64>
    %1048 = stablehlo.multiply %717, %1016 : tensor<f64>
    %1049 = stablehlo.multiply %1014, %1011 : tensor<f64>
    %1050 = stablehlo.subtract %1048, %1049 : tensor<f64>
    %1051 = stablehlo.multiply %1020, %1028 : tensor<f64>
    %1052 = stablehlo.subtract %1050, %1051 : tensor<f64>
    %1053 = stablehlo.multiply %1026, %1022 : tensor<f64>
    %1054 = stablehlo.subtract %1052, %1053 : tensor<f64>
    %1055 = stablehlo.reshape %1054 : (tensor<f64>) -> tensor<1xf64>
    %1056 = stablehlo.concatenate %1031, %1039, %1047, %1055, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1057 = stablehlo.slice %1056 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1058 = stablehlo.reshape %1057 : (tensor<1xf64>) -> tensor<f64>
    %1059 = stablehlo.slice %715 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1060 = stablehlo.reshape %1059 : (tensor<1xf64>) -> tensor<f64>
    %1061 = stablehlo.negate %1060 : tensor<f64>
    %1062 = stablehlo.reshape %1061 : (tensor<f64>) -> tensor<1xf64>
    %1063 = stablehlo.slice %715 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1064 = stablehlo.reshape %1063 : (tensor<1xf64>) -> tensor<f64>
    %1065 = stablehlo.negate %1064 : tensor<f64>
    %1066 = stablehlo.reshape %1065 : (tensor<f64>) -> tensor<1xf64>
    %1067 = stablehlo.slice %715 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1068 = stablehlo.reshape %1067 : (tensor<1xf64>) -> tensor<f64>
    %1069 = stablehlo.negate %1068 : tensor<f64>
    %1070 = stablehlo.reshape %1069 : (tensor<f64>) -> tensor<1xf64>
    %1071 = stablehlo.slice %715 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1072 = stablehlo.reshape %1071 : (tensor<1xf64>) -> tensor<f64>
    %1073 = stablehlo.reshape %1072 : (tensor<f64>) -> tensor<1xf64>
    %1074 = stablehlo.concatenate %1062, %1066, %1070, %1073, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1075 = stablehlo.dot_general %715, %715, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1076 = stablehlo.broadcast_in_dim %1075, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1077 = stablehlo.divide %1074, %1076 : tensor<4xf64>
    %1078 = stablehlo.slice %1077 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1079 = stablehlo.reshape %1078 : (tensor<1xf64>) -> tensor<f64>
    %1080 = stablehlo.multiply %1058, %1079 : tensor<f64>
    %1081 = stablehlo.slice %1056 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1082 = stablehlo.reshape %1081 : (tensor<1xf64>) -> tensor<f64>
    %1083 = stablehlo.slice %1077 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1084 = stablehlo.reshape %1083 : (tensor<1xf64>) -> tensor<f64>
    %1085 = stablehlo.multiply %1082, %1084 : tensor<f64>
    %1086 = stablehlo.add %1080, %1085 : tensor<f64>
    %1087 = stablehlo.slice %1056 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1088 = stablehlo.reshape %1087 : (tensor<1xf64>) -> tensor<f64>
    %1089 = stablehlo.slice %1077 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1090 = stablehlo.reshape %1089 : (tensor<1xf64>) -> tensor<f64>
    %1091 = stablehlo.multiply %1088, %1090 : tensor<f64>
    %1092 = stablehlo.add %1086, %1091 : tensor<f64>
    %1093 = stablehlo.slice %1056 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1094 = stablehlo.reshape %1093 : (tensor<1xf64>) -> tensor<f64>
    %1095 = stablehlo.slice %1077 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1096 = stablehlo.reshape %1095 : (tensor<1xf64>) -> tensor<f64>
    %1097 = stablehlo.multiply %1094, %1096 : tensor<f64>
    %1098 = stablehlo.subtract %1092, %1097 : tensor<f64>
    %1099 = stablehlo.reshape %1098 : (tensor<f64>) -> tensor<1xf64>
    %1100 = stablehlo.multiply %1058, %1096 : tensor<f64>
    %1101 = stablehlo.multiply %1082, %1090 : tensor<f64>
    %1102 = stablehlo.subtract %1100, %1101 : tensor<f64>
    %1103 = stablehlo.multiply %1088, %1084 : tensor<f64>
    %1104 = stablehlo.add %1102, %1103 : tensor<f64>
    %1105 = stablehlo.multiply %1094, %1079 : tensor<f64>
    %1106 = stablehlo.add %1104, %1105 : tensor<f64>
    %1107 = stablehlo.reshape %1106 : (tensor<f64>) -> tensor<1xf64>
    %1108 = stablehlo.multiply %1058, %1090 : tensor<f64>
    %1109 = stablehlo.multiply %1082, %1096 : tensor<f64>
    %1110 = stablehlo.add %1108, %1109 : tensor<f64>
    %1111 = stablehlo.multiply %1088, %1079 : tensor<f64>
    %1112 = stablehlo.subtract %1110, %1111 : tensor<f64>
    %1113 = stablehlo.multiply %1094, %1084 : tensor<f64>
    %1114 = stablehlo.add %1112, %1113 : tensor<f64>
    %1115 = stablehlo.reshape %1114 : (tensor<f64>) -> tensor<1xf64>
    %1116 = stablehlo.multiply %1058, %1084 : tensor<f64>
    %1117 = stablehlo.multiply %1082, %1079 : tensor<f64>
    %1118 = stablehlo.subtract %1116, %1117 : tensor<f64>
    %1119 = stablehlo.multiply %1088, %1096 : tensor<f64>
    %1120 = stablehlo.subtract %1118, %1119 : tensor<f64>
    %1121 = stablehlo.multiply %1094, %1090 : tensor<f64>
    %1122 = stablehlo.subtract %1120, %1121 : tensor<f64>
    %1123 = stablehlo.reshape %1122 : (tensor<f64>) -> tensor<1xf64>
    %1124 = stablehlo.concatenate %1099, %1107, %1115, %1123, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1125 = stablehlo.slice %1124 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1126 = stablehlo.reshape %1125 : (tensor<1xf64>) -> tensor<f64>
    %1127 = stablehlo.reshape %1126 : (tensor<f64>) -> tensor<1xf64>
    %1128 = stablehlo.slice %1124 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1129 = stablehlo.reshape %1128 : (tensor<1xf64>) -> tensor<f64>
    %1130 = stablehlo.reshape %1129 : (tensor<f64>) -> tensor<1xf64>
    %1131 = stablehlo.slice %1124 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1132 = stablehlo.reshape %1131 : (tensor<1xf64>) -> tensor<f64>
    %1133 = stablehlo.reshape %1132 : (tensor<f64>) -> tensor<1xf64>
    %1134 = stablehlo.concatenate %1127, %1130, %1133, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1135 = stablehlo.slice %715 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1136 = stablehlo.reshape %1135 : (tensor<1xf64>) -> tensor<f64>
    %1137 = stablehlo.slice %1006 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1138 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1139 = stablehlo.concatenate %1137, %1138, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1140 = stablehlo.slice %1139 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1141 = stablehlo.reshape %1140 : (tensor<1xf64>) -> tensor<f64>
    %1142 = stablehlo.multiply %1136, %1141 : tensor<f64>
    %1143 = stablehlo.slice %715 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1144 = stablehlo.reshape %1143 : (tensor<1xf64>) -> tensor<f64>
    %1145 = stablehlo.slice %1139 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1146 = stablehlo.reshape %1145 : (tensor<1xf64>) -> tensor<f64>
    %1147 = stablehlo.multiply %1144, %1146 : tensor<f64>
    %1148 = stablehlo.add %1142, %1147 : tensor<f64>
    %1149 = stablehlo.slice %715 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1150 = stablehlo.reshape %1149 : (tensor<1xf64>) -> tensor<f64>
    %1151 = stablehlo.slice %1139 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1152 = stablehlo.reshape %1151 : (tensor<1xf64>) -> tensor<f64>
    %1153 = stablehlo.multiply %1150, %1152 : tensor<f64>
    %1154 = stablehlo.add %1148, %1153 : tensor<f64>
    %1155 = stablehlo.slice %715 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1156 = stablehlo.reshape %1155 : (tensor<1xf64>) -> tensor<f64>
    %1157 = stablehlo.slice %1139 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1158 = stablehlo.reshape %1157 : (tensor<1xf64>) -> tensor<f64>
    %1159 = stablehlo.multiply %1156, %1158 : tensor<f64>
    %1160 = stablehlo.subtract %1154, %1159 : tensor<f64>
    %1161 = stablehlo.reshape %1160 : (tensor<f64>) -> tensor<1xf64>
    %1162 = stablehlo.multiply %1136, %1158 : tensor<f64>
    %1163 = stablehlo.multiply %1144, %1152 : tensor<f64>
    %1164 = stablehlo.subtract %1162, %1163 : tensor<f64>
    %1165 = stablehlo.multiply %1150, %1146 : tensor<f64>
    %1166 = stablehlo.add %1164, %1165 : tensor<f64>
    %1167 = stablehlo.multiply %1156, %1141 : tensor<f64>
    %1168 = stablehlo.add %1166, %1167 : tensor<f64>
    %1169 = stablehlo.reshape %1168 : (tensor<f64>) -> tensor<1xf64>
    %1170 = stablehlo.multiply %1136, %1152 : tensor<f64>
    %1171 = stablehlo.multiply %1144, %1158 : tensor<f64>
    %1172 = stablehlo.add %1170, %1171 : tensor<f64>
    %1173 = stablehlo.multiply %1150, %1141 : tensor<f64>
    %1174 = stablehlo.subtract %1172, %1173 : tensor<f64>
    %1175 = stablehlo.multiply %1156, %1146 : tensor<f64>
    %1176 = stablehlo.add %1174, %1175 : tensor<f64>
    %1177 = stablehlo.reshape %1176 : (tensor<f64>) -> tensor<1xf64>
    %1178 = stablehlo.multiply %1136, %1146 : tensor<f64>
    %1179 = stablehlo.multiply %1144, %1141 : tensor<f64>
    %1180 = stablehlo.subtract %1178, %1179 : tensor<f64>
    %1181 = stablehlo.multiply %1150, %1158 : tensor<f64>
    %1182 = stablehlo.subtract %1180, %1181 : tensor<f64>
    %1183 = stablehlo.multiply %1156, %1152 : tensor<f64>
    %1184 = stablehlo.subtract %1182, %1183 : tensor<f64>
    %1185 = stablehlo.reshape %1184 : (tensor<f64>) -> tensor<1xf64>
    %1186 = stablehlo.concatenate %1161, %1169, %1177, %1185, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1187 = stablehlo.slice %1186 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1188 = stablehlo.reshape %1187 : (tensor<1xf64>) -> tensor<f64>
    %1189 = stablehlo.slice %715 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1190 = stablehlo.reshape %1189 : (tensor<1xf64>) -> tensor<f64>
    %1191 = stablehlo.negate %1190 : tensor<f64>
    %1192 = stablehlo.reshape %1191 : (tensor<f64>) -> tensor<1xf64>
    %1193 = stablehlo.slice %715 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1194 = stablehlo.reshape %1193 : (tensor<1xf64>) -> tensor<f64>
    %1195 = stablehlo.negate %1194 : tensor<f64>
    %1196 = stablehlo.reshape %1195 : (tensor<f64>) -> tensor<1xf64>
    %1197 = stablehlo.slice %715 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1198 = stablehlo.reshape %1197 : (tensor<1xf64>) -> tensor<f64>
    %1199 = stablehlo.negate %1198 : tensor<f64>
    %1200 = stablehlo.reshape %1199 : (tensor<f64>) -> tensor<1xf64>
    %1201 = stablehlo.slice %715 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1202 = stablehlo.reshape %1201 : (tensor<1xf64>) -> tensor<f64>
    %1203 = stablehlo.reshape %1202 : (tensor<f64>) -> tensor<1xf64>
    %1204 = stablehlo.concatenate %1192, %1196, %1200, %1203, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1205 = stablehlo.dot_general %715, %715, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1206 = stablehlo.broadcast_in_dim %1205, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1207 = stablehlo.divide %1204, %1206 : tensor<4xf64>
    %1208 = stablehlo.slice %1207 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1209 = stablehlo.reshape %1208 : (tensor<1xf64>) -> tensor<f64>
    %1210 = stablehlo.multiply %1188, %1209 : tensor<f64>
    %1211 = stablehlo.slice %1186 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1212 = stablehlo.reshape %1211 : (tensor<1xf64>) -> tensor<f64>
    %1213 = stablehlo.slice %1207 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1214 = stablehlo.reshape %1213 : (tensor<1xf64>) -> tensor<f64>
    %1215 = stablehlo.multiply %1212, %1214 : tensor<f64>
    %1216 = stablehlo.add %1210, %1215 : tensor<f64>
    %1217 = stablehlo.slice %1186 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1218 = stablehlo.reshape %1217 : (tensor<1xf64>) -> tensor<f64>
    %1219 = stablehlo.slice %1207 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1220 = stablehlo.reshape %1219 : (tensor<1xf64>) -> tensor<f64>
    %1221 = stablehlo.multiply %1218, %1220 : tensor<f64>
    %1222 = stablehlo.add %1216, %1221 : tensor<f64>
    %1223 = stablehlo.slice %1186 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1224 = stablehlo.reshape %1223 : (tensor<1xf64>) -> tensor<f64>
    %1225 = stablehlo.slice %1207 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1226 = stablehlo.reshape %1225 : (tensor<1xf64>) -> tensor<f64>
    %1227 = stablehlo.multiply %1224, %1226 : tensor<f64>
    %1228 = stablehlo.subtract %1222, %1227 : tensor<f64>
    %1229 = stablehlo.reshape %1228 : (tensor<f64>) -> tensor<1xf64>
    %1230 = stablehlo.multiply %1188, %1226 : tensor<f64>
    %1231 = stablehlo.multiply %1212, %1220 : tensor<f64>
    %1232 = stablehlo.subtract %1230, %1231 : tensor<f64>
    %1233 = stablehlo.multiply %1218, %1214 : tensor<f64>
    %1234 = stablehlo.add %1232, %1233 : tensor<f64>
    %1235 = stablehlo.multiply %1224, %1209 : tensor<f64>
    %1236 = stablehlo.add %1234, %1235 : tensor<f64>
    %1237 = stablehlo.reshape %1236 : (tensor<f64>) -> tensor<1xf64>
    %1238 = stablehlo.multiply %1188, %1220 : tensor<f64>
    %1239 = stablehlo.multiply %1212, %1226 : tensor<f64>
    %1240 = stablehlo.add %1238, %1239 : tensor<f64>
    %1241 = stablehlo.multiply %1218, %1209 : tensor<f64>
    %1242 = stablehlo.subtract %1240, %1241 : tensor<f64>
    %1243 = stablehlo.multiply %1224, %1214 : tensor<f64>
    %1244 = stablehlo.add %1242, %1243 : tensor<f64>
    %1245 = stablehlo.reshape %1244 : (tensor<f64>) -> tensor<1xf64>
    %1246 = stablehlo.multiply %1188, %1214 : tensor<f64>
    %1247 = stablehlo.multiply %1212, %1209 : tensor<f64>
    %1248 = stablehlo.subtract %1246, %1247 : tensor<f64>
    %1249 = stablehlo.multiply %1218, %1226 : tensor<f64>
    %1250 = stablehlo.subtract %1248, %1249 : tensor<f64>
    %1251 = stablehlo.multiply %1224, %1220 : tensor<f64>
    %1252 = stablehlo.subtract %1250, %1251 : tensor<f64>
    %1253 = stablehlo.reshape %1252 : (tensor<f64>) -> tensor<1xf64>
    %1254 = stablehlo.concatenate %1229, %1237, %1245, %1253, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1255 = stablehlo.slice %1254 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1256 = stablehlo.reshape %1255 : (tensor<1xf64>) -> tensor<f64>
    %1257 = stablehlo.reshape %1256 : (tensor<f64>) -> tensor<1xf64>
    %1258 = stablehlo.slice %1254 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1259 = stablehlo.reshape %1258 : (tensor<1xf64>) -> tensor<f64>
    %1260 = stablehlo.reshape %1259 : (tensor<f64>) -> tensor<1xf64>
    %1261 = stablehlo.slice %1254 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1262 = stablehlo.reshape %1261 : (tensor<1xf64>) -> tensor<f64>
    %1263 = stablehlo.reshape %1262 : (tensor<f64>) -> tensor<1xf64>
    %1264 = stablehlo.concatenate %1257, %1260, %1263, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1265 = stablehlo.concatenate %1134, %1264, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1266 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1267 = stablehlo.reshape %arg22 : (tensor<f64>) -> tensor<f64>
    %cst_15 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %1268 = stablehlo.multiply %cst_15, %1267 : tensor<f64>
    %1269 = stablehlo.broadcast_in_dim %1268, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1270 = stablehlo.multiply %1269, %arg2 : tensor<6xf64>
    %1271 = stablehlo.slice %1270 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_16 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1272 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1273 = stablehlo.divide %1271, %1272 : tensor<3xf64>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1274 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1275 = stablehlo.concatenate %1273, %1274, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1276 = stablehlo.slice %1275 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1277 = stablehlo.reshape %1276 : (tensor<1xf64>) -> tensor<f64>
    %1278 = stablehlo.slice %1266 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1279 = stablehlo.reshape %1278 : (tensor<1xf64>) -> tensor<f64>
    %1280 = stablehlo.multiply %1277, %1279 : tensor<f64>
    %1281 = stablehlo.slice %1275 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1282 = stablehlo.reshape %1281 : (tensor<1xf64>) -> tensor<f64>
    %1283 = stablehlo.slice %1266 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1284 = stablehlo.reshape %1283 : (tensor<1xf64>) -> tensor<f64>
    %1285 = stablehlo.multiply %1282, %1284 : tensor<f64>
    %1286 = stablehlo.add %1280, %1285 : tensor<f64>
    %1287 = stablehlo.slice %1275 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1288 = stablehlo.reshape %1287 : (tensor<1xf64>) -> tensor<f64>
    %1289 = stablehlo.slice %1266 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1290 = stablehlo.reshape %1289 : (tensor<1xf64>) -> tensor<f64>
    %1291 = stablehlo.multiply %1288, %1290 : tensor<f64>
    %1292 = stablehlo.add %1286, %1291 : tensor<f64>
    %1293 = stablehlo.slice %1275 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1294 = stablehlo.reshape %1293 : (tensor<1xf64>) -> tensor<f64>
    %1295 = stablehlo.slice %1266 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1296 = stablehlo.reshape %1295 : (tensor<1xf64>) -> tensor<f64>
    %1297 = stablehlo.multiply %1294, %1296 : tensor<f64>
    %1298 = stablehlo.subtract %1292, %1297 : tensor<f64>
    %1299 = stablehlo.reshape %1298 : (tensor<f64>) -> tensor<1xf64>
    %1300 = stablehlo.multiply %1277, %1296 : tensor<f64>
    %1301 = stablehlo.multiply %1282, %1290 : tensor<f64>
    %1302 = stablehlo.subtract %1300, %1301 : tensor<f64>
    %1303 = stablehlo.multiply %1288, %1284 : tensor<f64>
    %1304 = stablehlo.add %1302, %1303 : tensor<f64>
    %1305 = stablehlo.multiply %1294, %1279 : tensor<f64>
    %1306 = stablehlo.add %1304, %1305 : tensor<f64>
    %1307 = stablehlo.reshape %1306 : (tensor<f64>) -> tensor<1xf64>
    %1308 = stablehlo.multiply %1277, %1290 : tensor<f64>
    %1309 = stablehlo.multiply %1282, %1296 : tensor<f64>
    %1310 = stablehlo.add %1308, %1309 : tensor<f64>
    %1311 = stablehlo.multiply %1288, %1279 : tensor<f64>
    %1312 = stablehlo.subtract %1310, %1311 : tensor<f64>
    %1313 = stablehlo.multiply %1294, %1284 : tensor<f64>
    %1314 = stablehlo.add %1312, %1313 : tensor<f64>
    %1315 = stablehlo.reshape %1314 : (tensor<f64>) -> tensor<1xf64>
    %1316 = stablehlo.multiply %1277, %1284 : tensor<f64>
    %1317 = stablehlo.multiply %1282, %1279 : tensor<f64>
    %1318 = stablehlo.subtract %1316, %1317 : tensor<f64>
    %1319 = stablehlo.multiply %1288, %1296 : tensor<f64>
    %1320 = stablehlo.subtract %1318, %1319 : tensor<f64>
    %1321 = stablehlo.multiply %1294, %1290 : tensor<f64>
    %1322 = stablehlo.subtract %1320, %1321 : tensor<f64>
    %1323 = stablehlo.reshape %1322 : (tensor<f64>) -> tensor<1xf64>
    %1324 = stablehlo.concatenate %1299, %1307, %1315, %1323, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1325 = stablehlo.add %1266, %1324 : tensor<4xf64>
    %1326 = stablehlo.dot_general %1325, %1325, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1327 = stablehlo.sqrt %1326 : tensor<f64>
    %1328 = stablehlo.broadcast_in_dim %1327, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1329 = stablehlo.divide %1325, %1328 : tensor<4xf64>
    %1330 = stablehlo.slice %arg1 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1331 = stablehlo.slice %1270 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1332 = stablehlo.add %1330, %1331 : tensor<3xf64>
    %1333 = stablehlo.concatenate %1329, %1332, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %1334 = stablehlo.broadcast_in_dim %1268, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1335 = stablehlo.multiply %1334, %1265 : tensor<6xf64>
    %1336 = stablehlo.add %arg2, %1335 : tensor<6xf64>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1337 = stablehlo.broadcast_in_dim %cst_18, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1338 = call @inner_124(%1337, %arg26) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %1339 = call @inner_126(%13, %1338, %1333) : (tensor<f64>, tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %1340 = call @inner_127(%1333, %12, %1339) : (tensor<7xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1341 = stablehlo.slice %1333 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1342 = stablehlo.slice %1341 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1343 = stablehlo.reshape %1342 : (tensor<1xf64>) -> tensor<f64>
    %1344 = stablehlo.slice %1341 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1345 = stablehlo.reshape %1344 : (tensor<1xf64>) -> tensor<f64>
    %1346 = stablehlo.negate %1345 : tensor<f64>
    %1347 = stablehlo.reshape %1346 : (tensor<f64>) -> tensor<1xf64>
    %1348 = stablehlo.slice %1341 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1349 = stablehlo.reshape %1348 : (tensor<1xf64>) -> tensor<f64>
    %1350 = stablehlo.negate %1349 : tensor<f64>
    %1351 = stablehlo.reshape %1350 : (tensor<f64>) -> tensor<1xf64>
    %1352 = stablehlo.slice %1341 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1353 = stablehlo.reshape %1352 : (tensor<1xf64>) -> tensor<f64>
    %1354 = stablehlo.negate %1353 : tensor<f64>
    %1355 = stablehlo.reshape %1354 : (tensor<f64>) -> tensor<1xf64>
    %1356 = stablehlo.slice %1341 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1357 = stablehlo.reshape %1356 : (tensor<1xf64>) -> tensor<f64>
    %1358 = stablehlo.reshape %1357 : (tensor<f64>) -> tensor<1xf64>
    %1359 = stablehlo.concatenate %1347, %1351, %1355, %1358, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1360 = stablehlo.dot_general %1341, %1341, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1362 = stablehlo.divide %1359, %1361 : tensor<4xf64>
    %1363 = stablehlo.slice %1362 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1364 = stablehlo.reshape %1363 : (tensor<1xf64>) -> tensor<f64>
    %1365 = stablehlo.slice %1340 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1366 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1367 = stablehlo.concatenate %1365, %1366, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1368 = stablehlo.slice %1367 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1369 = stablehlo.reshape %1368 : (tensor<1xf64>) -> tensor<f64>
    %1370 = stablehlo.multiply %1364, %1369 : tensor<f64>
    %1371 = stablehlo.slice %1362 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1372 = stablehlo.reshape %1371 : (tensor<1xf64>) -> tensor<f64>
    %1373 = stablehlo.slice %1367 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1374 = stablehlo.reshape %1373 : (tensor<1xf64>) -> tensor<f64>
    %1375 = stablehlo.multiply %1372, %1374 : tensor<f64>
    %1376 = stablehlo.add %1370, %1375 : tensor<f64>
    %1377 = stablehlo.slice %1362 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1378 = stablehlo.reshape %1377 : (tensor<1xf64>) -> tensor<f64>
    %1379 = stablehlo.slice %1367 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1380 = stablehlo.reshape %1379 : (tensor<1xf64>) -> tensor<f64>
    %1381 = stablehlo.multiply %1378, %1380 : tensor<f64>
    %1382 = stablehlo.add %1376, %1381 : tensor<f64>
    %1383 = stablehlo.slice %1362 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1384 = stablehlo.reshape %1383 : (tensor<1xf64>) -> tensor<f64>
    %1385 = stablehlo.slice %1367 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1386 = stablehlo.reshape %1385 : (tensor<1xf64>) -> tensor<f64>
    %1387 = stablehlo.multiply %1384, %1386 : tensor<f64>
    %1388 = stablehlo.subtract %1382, %1387 : tensor<f64>
    %1389 = stablehlo.reshape %1388 : (tensor<f64>) -> tensor<1xf64>
    %1390 = stablehlo.multiply %1364, %1386 : tensor<f64>
    %1391 = stablehlo.multiply %1372, %1380 : tensor<f64>
    %1392 = stablehlo.subtract %1390, %1391 : tensor<f64>
    %1393 = stablehlo.multiply %1378, %1374 : tensor<f64>
    %1394 = stablehlo.add %1392, %1393 : tensor<f64>
    %1395 = stablehlo.multiply %1384, %1369 : tensor<f64>
    %1396 = stablehlo.add %1394, %1395 : tensor<f64>
    %1397 = stablehlo.reshape %1396 : (tensor<f64>) -> tensor<1xf64>
    %1398 = stablehlo.multiply %1364, %1380 : tensor<f64>
    %1399 = stablehlo.multiply %1372, %1386 : tensor<f64>
    %1400 = stablehlo.add %1398, %1399 : tensor<f64>
    %1401 = stablehlo.multiply %1378, %1369 : tensor<f64>
    %1402 = stablehlo.subtract %1400, %1401 : tensor<f64>
    %1403 = stablehlo.multiply %1384, %1374 : tensor<f64>
    %1404 = stablehlo.add %1402, %1403 : tensor<f64>
    %1405 = stablehlo.reshape %1404 : (tensor<f64>) -> tensor<1xf64>
    %1406 = stablehlo.multiply %1364, %1374 : tensor<f64>
    %1407 = stablehlo.multiply %1372, %1369 : tensor<f64>
    %1408 = stablehlo.subtract %1406, %1407 : tensor<f64>
    %1409 = stablehlo.multiply %1378, %1386 : tensor<f64>
    %1410 = stablehlo.subtract %1408, %1409 : tensor<f64>
    %1411 = stablehlo.multiply %1384, %1380 : tensor<f64>
    %1412 = stablehlo.subtract %1410, %1411 : tensor<f64>
    %1413 = stablehlo.reshape %1412 : (tensor<f64>) -> tensor<1xf64>
    %1414 = stablehlo.concatenate %1389, %1397, %1405, %1413, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1415 = stablehlo.slice %1414 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1416 = stablehlo.reshape %1415 : (tensor<1xf64>) -> tensor<f64>
    %1417 = stablehlo.slice %1362 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1418 = stablehlo.reshape %1417 : (tensor<1xf64>) -> tensor<f64>
    %1419 = stablehlo.negate %1418 : tensor<f64>
    %1420 = stablehlo.reshape %1419 : (tensor<f64>) -> tensor<1xf64>
    %1421 = stablehlo.slice %1362 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1422 = stablehlo.reshape %1421 : (tensor<1xf64>) -> tensor<f64>
    %1423 = stablehlo.negate %1422 : tensor<f64>
    %1424 = stablehlo.reshape %1423 : (tensor<f64>) -> tensor<1xf64>
    %1425 = stablehlo.slice %1362 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1426 = stablehlo.reshape %1425 : (tensor<1xf64>) -> tensor<f64>
    %1427 = stablehlo.negate %1426 : tensor<f64>
    %1428 = stablehlo.reshape %1427 : (tensor<f64>) -> tensor<1xf64>
    %1429 = stablehlo.slice %1362 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1430 = stablehlo.reshape %1429 : (tensor<1xf64>) -> tensor<f64>
    %1431 = stablehlo.reshape %1430 : (tensor<f64>) -> tensor<1xf64>
    %1432 = stablehlo.concatenate %1420, %1424, %1428, %1431, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1433 = stablehlo.dot_general %1362, %1362, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1434 = stablehlo.broadcast_in_dim %1433, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1435 = stablehlo.divide %1432, %1434 : tensor<4xf64>
    %1436 = stablehlo.slice %1435 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1437 = stablehlo.reshape %1436 : (tensor<1xf64>) -> tensor<f64>
    %1438 = stablehlo.multiply %1416, %1437 : tensor<f64>
    %1439 = stablehlo.slice %1414 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1440 = stablehlo.reshape %1439 : (tensor<1xf64>) -> tensor<f64>
    %1441 = stablehlo.slice %1435 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1442 = stablehlo.reshape %1441 : (tensor<1xf64>) -> tensor<f64>
    %1443 = stablehlo.multiply %1440, %1442 : tensor<f64>
    %1444 = stablehlo.add %1438, %1443 : tensor<f64>
    %1445 = stablehlo.slice %1414 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1446 = stablehlo.reshape %1445 : (tensor<1xf64>) -> tensor<f64>
    %1447 = stablehlo.slice %1435 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1448 = stablehlo.reshape %1447 : (tensor<1xf64>) -> tensor<f64>
    %1449 = stablehlo.multiply %1446, %1448 : tensor<f64>
    %1450 = stablehlo.add %1444, %1449 : tensor<f64>
    %1451 = stablehlo.slice %1414 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1452 = stablehlo.reshape %1451 : (tensor<1xf64>) -> tensor<f64>
    %1453 = stablehlo.slice %1435 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1454 = stablehlo.reshape %1453 : (tensor<1xf64>) -> tensor<f64>
    %1455 = stablehlo.multiply %1452, %1454 : tensor<f64>
    %1456 = stablehlo.subtract %1450, %1455 : tensor<f64>
    %1457 = stablehlo.reshape %1456 : (tensor<f64>) -> tensor<1xf64>
    %1458 = stablehlo.multiply %1416, %1454 : tensor<f64>
    %1459 = stablehlo.multiply %1440, %1448 : tensor<f64>
    %1460 = stablehlo.subtract %1458, %1459 : tensor<f64>
    %1461 = stablehlo.multiply %1446, %1442 : tensor<f64>
    %1462 = stablehlo.add %1460, %1461 : tensor<f64>
    %1463 = stablehlo.multiply %1452, %1437 : tensor<f64>
    %1464 = stablehlo.add %1462, %1463 : tensor<f64>
    %1465 = stablehlo.reshape %1464 : (tensor<f64>) -> tensor<1xf64>
    %1466 = stablehlo.multiply %1416, %1448 : tensor<f64>
    %1467 = stablehlo.multiply %1440, %1454 : tensor<f64>
    %1468 = stablehlo.add %1466, %1467 : tensor<f64>
    %1469 = stablehlo.multiply %1446, %1437 : tensor<f64>
    %1470 = stablehlo.subtract %1468, %1469 : tensor<f64>
    %1471 = stablehlo.multiply %1452, %1442 : tensor<f64>
    %1472 = stablehlo.add %1470, %1471 : tensor<f64>
    %1473 = stablehlo.reshape %1472 : (tensor<f64>) -> tensor<1xf64>
    %1474 = stablehlo.multiply %1416, %1442 : tensor<f64>
    %1475 = stablehlo.multiply %1440, %1437 : tensor<f64>
    %1476 = stablehlo.subtract %1474, %1475 : tensor<f64>
    %1477 = stablehlo.multiply %1446, %1454 : tensor<f64>
    %1478 = stablehlo.subtract %1476, %1477 : tensor<f64>
    %1479 = stablehlo.multiply %1452, %1448 : tensor<f64>
    %1480 = stablehlo.subtract %1478, %1479 : tensor<f64>
    %1481 = stablehlo.reshape %1480 : (tensor<f64>) -> tensor<1xf64>
    %1482 = stablehlo.concatenate %1457, %1465, %1473, %1481, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1483 = stablehlo.slice %1482 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1484 = stablehlo.reshape %1483 : (tensor<1xf64>) -> tensor<f64>
    %1485 = stablehlo.reshape %1484 : (tensor<f64>) -> tensor<1xf64>
    %1486 = stablehlo.slice %1482 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1487 = stablehlo.reshape %1486 : (tensor<1xf64>) -> tensor<f64>
    %1488 = stablehlo.reshape %1487 : (tensor<f64>) -> tensor<1xf64>
    %1489 = stablehlo.slice %1482 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1490 = stablehlo.reshape %1489 : (tensor<1xf64>) -> tensor<f64>
    %1491 = stablehlo.reshape %1490 : (tensor<f64>) -> tensor<1xf64>
    %1492 = stablehlo.concatenate %1485, %1488, %1491, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1493 = stablehlo.slice %1362 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1494 = stablehlo.reshape %1493 : (tensor<1xf64>) -> tensor<f64>
    %1495 = stablehlo.slice %1340 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1496 = stablehlo.broadcast_in_dim %cst_20, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1497 = stablehlo.concatenate %1495, %1496, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1498 = stablehlo.slice %1497 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1499 = stablehlo.reshape %1498 : (tensor<1xf64>) -> tensor<f64>
    %1500 = stablehlo.multiply %1494, %1499 : tensor<f64>
    %1501 = stablehlo.slice %1362 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1502 = stablehlo.reshape %1501 : (tensor<1xf64>) -> tensor<f64>
    %1503 = stablehlo.slice %1497 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1504 = stablehlo.reshape %1503 : (tensor<1xf64>) -> tensor<f64>
    %1505 = stablehlo.multiply %1502, %1504 : tensor<f64>
    %1506 = stablehlo.add %1500, %1505 : tensor<f64>
    %1507 = stablehlo.slice %1362 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1508 = stablehlo.reshape %1507 : (tensor<1xf64>) -> tensor<f64>
    %1509 = stablehlo.slice %1497 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1510 = stablehlo.reshape %1509 : (tensor<1xf64>) -> tensor<f64>
    %1511 = stablehlo.multiply %1508, %1510 : tensor<f64>
    %1512 = stablehlo.add %1506, %1511 : tensor<f64>
    %1513 = stablehlo.slice %1362 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1514 = stablehlo.reshape %1513 : (tensor<1xf64>) -> tensor<f64>
    %1515 = stablehlo.slice %1497 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1516 = stablehlo.reshape %1515 : (tensor<1xf64>) -> tensor<f64>
    %1517 = stablehlo.multiply %1514, %1516 : tensor<f64>
    %1518 = stablehlo.subtract %1512, %1517 : tensor<f64>
    %1519 = stablehlo.reshape %1518 : (tensor<f64>) -> tensor<1xf64>
    %1520 = stablehlo.multiply %1494, %1516 : tensor<f64>
    %1521 = stablehlo.multiply %1502, %1510 : tensor<f64>
    %1522 = stablehlo.subtract %1520, %1521 : tensor<f64>
    %1523 = stablehlo.multiply %1508, %1504 : tensor<f64>
    %1524 = stablehlo.add %1522, %1523 : tensor<f64>
    %1525 = stablehlo.multiply %1514, %1499 : tensor<f64>
    %1526 = stablehlo.add %1524, %1525 : tensor<f64>
    %1527 = stablehlo.reshape %1526 : (tensor<f64>) -> tensor<1xf64>
    %1528 = stablehlo.multiply %1494, %1510 : tensor<f64>
    %1529 = stablehlo.multiply %1502, %1516 : tensor<f64>
    %1530 = stablehlo.add %1528, %1529 : tensor<f64>
    %1531 = stablehlo.multiply %1508, %1499 : tensor<f64>
    %1532 = stablehlo.subtract %1530, %1531 : tensor<f64>
    %1533 = stablehlo.multiply %1514, %1504 : tensor<f64>
    %1534 = stablehlo.add %1532, %1533 : tensor<f64>
    %1535 = stablehlo.reshape %1534 : (tensor<f64>) -> tensor<1xf64>
    %1536 = stablehlo.multiply %1494, %1504 : tensor<f64>
    %1537 = stablehlo.multiply %1502, %1499 : tensor<f64>
    %1538 = stablehlo.subtract %1536, %1537 : tensor<f64>
    %1539 = stablehlo.multiply %1508, %1516 : tensor<f64>
    %1540 = stablehlo.subtract %1538, %1539 : tensor<f64>
    %1541 = stablehlo.multiply %1514, %1510 : tensor<f64>
    %1542 = stablehlo.subtract %1540, %1541 : tensor<f64>
    %1543 = stablehlo.reshape %1542 : (tensor<f64>) -> tensor<1xf64>
    %1544 = stablehlo.concatenate %1519, %1527, %1535, %1543, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1545 = stablehlo.slice %1544 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1546 = stablehlo.reshape %1545 : (tensor<1xf64>) -> tensor<f64>
    %1547 = stablehlo.slice %1362 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1548 = stablehlo.reshape %1547 : (tensor<1xf64>) -> tensor<f64>
    %1549 = stablehlo.negate %1548 : tensor<f64>
    %1550 = stablehlo.reshape %1549 : (tensor<f64>) -> tensor<1xf64>
    %1551 = stablehlo.slice %1362 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1552 = stablehlo.reshape %1551 : (tensor<1xf64>) -> tensor<f64>
    %1553 = stablehlo.negate %1552 : tensor<f64>
    %1554 = stablehlo.reshape %1553 : (tensor<f64>) -> tensor<1xf64>
    %1555 = stablehlo.slice %1362 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1556 = stablehlo.reshape %1555 : (tensor<1xf64>) -> tensor<f64>
    %1557 = stablehlo.negate %1556 : tensor<f64>
    %1558 = stablehlo.reshape %1557 : (tensor<f64>) -> tensor<1xf64>
    %1559 = stablehlo.slice %1362 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1560 = stablehlo.reshape %1559 : (tensor<1xf64>) -> tensor<f64>
    %1561 = stablehlo.reshape %1560 : (tensor<f64>) -> tensor<1xf64>
    %1562 = stablehlo.concatenate %1550, %1554, %1558, %1561, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1563 = stablehlo.dot_general %1362, %1362, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1564 = stablehlo.broadcast_in_dim %1563, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1565 = stablehlo.divide %1562, %1564 : tensor<4xf64>
    %1566 = stablehlo.slice %1565 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1567 = stablehlo.reshape %1566 : (tensor<1xf64>) -> tensor<f64>
    %1568 = stablehlo.multiply %1546, %1567 : tensor<f64>
    %1569 = stablehlo.slice %1544 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1570 = stablehlo.reshape %1569 : (tensor<1xf64>) -> tensor<f64>
    %1571 = stablehlo.slice %1565 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1572 = stablehlo.reshape %1571 : (tensor<1xf64>) -> tensor<f64>
    %1573 = stablehlo.multiply %1570, %1572 : tensor<f64>
    %1574 = stablehlo.add %1568, %1573 : tensor<f64>
    %1575 = stablehlo.slice %1544 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1576 = stablehlo.reshape %1575 : (tensor<1xf64>) -> tensor<f64>
    %1577 = stablehlo.slice %1565 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1578 = stablehlo.reshape %1577 : (tensor<1xf64>) -> tensor<f64>
    %1579 = stablehlo.multiply %1576, %1578 : tensor<f64>
    %1580 = stablehlo.add %1574, %1579 : tensor<f64>
    %1581 = stablehlo.slice %1544 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1582 = stablehlo.reshape %1581 : (tensor<1xf64>) -> tensor<f64>
    %1583 = stablehlo.slice %1565 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1584 = stablehlo.reshape %1583 : (tensor<1xf64>) -> tensor<f64>
    %1585 = stablehlo.multiply %1582, %1584 : tensor<f64>
    %1586 = stablehlo.subtract %1580, %1585 : tensor<f64>
    %1587 = stablehlo.reshape %1586 : (tensor<f64>) -> tensor<1xf64>
    %1588 = stablehlo.multiply %1546, %1584 : tensor<f64>
    %1589 = stablehlo.multiply %1570, %1578 : tensor<f64>
    %1590 = stablehlo.subtract %1588, %1589 : tensor<f64>
    %1591 = stablehlo.multiply %1576, %1572 : tensor<f64>
    %1592 = stablehlo.add %1590, %1591 : tensor<f64>
    %1593 = stablehlo.multiply %1582, %1567 : tensor<f64>
    %1594 = stablehlo.add %1592, %1593 : tensor<f64>
    %1595 = stablehlo.reshape %1594 : (tensor<f64>) -> tensor<1xf64>
    %1596 = stablehlo.multiply %1546, %1578 : tensor<f64>
    %1597 = stablehlo.multiply %1570, %1584 : tensor<f64>
    %1598 = stablehlo.add %1596, %1597 : tensor<f64>
    %1599 = stablehlo.multiply %1576, %1567 : tensor<f64>
    %1600 = stablehlo.subtract %1598, %1599 : tensor<f64>
    %1601 = stablehlo.multiply %1582, %1572 : tensor<f64>
    %1602 = stablehlo.add %1600, %1601 : tensor<f64>
    %1603 = stablehlo.reshape %1602 : (tensor<f64>) -> tensor<1xf64>
    %1604 = stablehlo.multiply %1546, %1572 : tensor<f64>
    %1605 = stablehlo.multiply %1570, %1567 : tensor<f64>
    %1606 = stablehlo.subtract %1604, %1605 : tensor<f64>
    %1607 = stablehlo.multiply %1576, %1584 : tensor<f64>
    %1608 = stablehlo.subtract %1606, %1607 : tensor<f64>
    %1609 = stablehlo.multiply %1582, %1578 : tensor<f64>
    %1610 = stablehlo.subtract %1608, %1609 : tensor<f64>
    %1611 = stablehlo.reshape %1610 : (tensor<f64>) -> tensor<1xf64>
    %1612 = stablehlo.concatenate %1587, %1595, %1603, %1611, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1613 = stablehlo.slice %1612 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1614 = stablehlo.reshape %1613 : (tensor<1xf64>) -> tensor<f64>
    %1615 = stablehlo.reshape %1614 : (tensor<f64>) -> tensor<1xf64>
    %1616 = stablehlo.slice %1612 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1617 = stablehlo.reshape %1616 : (tensor<1xf64>) -> tensor<f64>
    %1618 = stablehlo.reshape %1617 : (tensor<f64>) -> tensor<1xf64>
    %1619 = stablehlo.slice %1612 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1620 = stablehlo.reshape %1619 : (tensor<1xf64>) -> tensor<f64>
    %1621 = stablehlo.reshape %1620 : (tensor<f64>) -> tensor<1xf64>
    %1622 = stablehlo.concatenate %1615, %1618, %1621, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1623 = stablehlo.concatenate %1492, %1622, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1624 = stablehlo.slice %1623 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %1625 = stablehlo.slice %arg26 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %1626 = stablehlo.divide %1624, %1625 : tensor<3xf64>
    %1627 = stablehlo.slice %1623 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1628 = stablehlo.slice %arg26 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %1629 = stablehlo.reshape %1628 : (tensor<1xf64>) -> tensor<f64>
    %1630 = stablehlo.broadcast_in_dim %1629, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1631 = stablehlo.divide %1627, %1630 : tensor<3xf64>
    %1632 = stablehlo.concatenate %1626, %1631, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1633 = stablehlo.slice %1632 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1634 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1635 = stablehlo.concatenate %1633, %1634, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1636 = stablehlo.slice %1635 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1637 = stablehlo.reshape %1636 : (tensor<1xf64>) -> tensor<f64>
    %1638 = stablehlo.multiply %1343, %1637 : tensor<f64>
    %1639 = stablehlo.slice %1341 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1640 = stablehlo.reshape %1639 : (tensor<1xf64>) -> tensor<f64>
    %1641 = stablehlo.slice %1635 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1642 = stablehlo.reshape %1641 : (tensor<1xf64>) -> tensor<f64>
    %1643 = stablehlo.multiply %1640, %1642 : tensor<f64>
    %1644 = stablehlo.add %1638, %1643 : tensor<f64>
    %1645 = stablehlo.slice %1341 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1646 = stablehlo.reshape %1645 : (tensor<1xf64>) -> tensor<f64>
    %1647 = stablehlo.slice %1635 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1648 = stablehlo.reshape %1647 : (tensor<1xf64>) -> tensor<f64>
    %1649 = stablehlo.multiply %1646, %1648 : tensor<f64>
    %1650 = stablehlo.add %1644, %1649 : tensor<f64>
    %1651 = stablehlo.slice %1341 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1652 = stablehlo.reshape %1651 : (tensor<1xf64>) -> tensor<f64>
    %1653 = stablehlo.slice %1635 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1654 = stablehlo.reshape %1653 : (tensor<1xf64>) -> tensor<f64>
    %1655 = stablehlo.multiply %1652, %1654 : tensor<f64>
    %1656 = stablehlo.subtract %1650, %1655 : tensor<f64>
    %1657 = stablehlo.reshape %1656 : (tensor<f64>) -> tensor<1xf64>
    %1658 = stablehlo.multiply %1343, %1654 : tensor<f64>
    %1659 = stablehlo.multiply %1640, %1648 : tensor<f64>
    %1660 = stablehlo.subtract %1658, %1659 : tensor<f64>
    %1661 = stablehlo.multiply %1646, %1642 : tensor<f64>
    %1662 = stablehlo.add %1660, %1661 : tensor<f64>
    %1663 = stablehlo.multiply %1652, %1637 : tensor<f64>
    %1664 = stablehlo.add %1662, %1663 : tensor<f64>
    %1665 = stablehlo.reshape %1664 : (tensor<f64>) -> tensor<1xf64>
    %1666 = stablehlo.multiply %1343, %1648 : tensor<f64>
    %1667 = stablehlo.multiply %1640, %1654 : tensor<f64>
    %1668 = stablehlo.add %1666, %1667 : tensor<f64>
    %1669 = stablehlo.multiply %1646, %1637 : tensor<f64>
    %1670 = stablehlo.subtract %1668, %1669 : tensor<f64>
    %1671 = stablehlo.multiply %1652, %1642 : tensor<f64>
    %1672 = stablehlo.add %1670, %1671 : tensor<f64>
    %1673 = stablehlo.reshape %1672 : (tensor<f64>) -> tensor<1xf64>
    %1674 = stablehlo.multiply %1343, %1642 : tensor<f64>
    %1675 = stablehlo.multiply %1640, %1637 : tensor<f64>
    %1676 = stablehlo.subtract %1674, %1675 : tensor<f64>
    %1677 = stablehlo.multiply %1646, %1654 : tensor<f64>
    %1678 = stablehlo.subtract %1676, %1677 : tensor<f64>
    %1679 = stablehlo.multiply %1652, %1648 : tensor<f64>
    %1680 = stablehlo.subtract %1678, %1679 : tensor<f64>
    %1681 = stablehlo.reshape %1680 : (tensor<f64>) -> tensor<1xf64>
    %1682 = stablehlo.concatenate %1657, %1665, %1673, %1681, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1683 = stablehlo.slice %1682 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1684 = stablehlo.reshape %1683 : (tensor<1xf64>) -> tensor<f64>
    %1685 = stablehlo.slice %1341 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1686 = stablehlo.reshape %1685 : (tensor<1xf64>) -> tensor<f64>
    %1687 = stablehlo.negate %1686 : tensor<f64>
    %1688 = stablehlo.reshape %1687 : (tensor<f64>) -> tensor<1xf64>
    %1689 = stablehlo.slice %1341 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1690 = stablehlo.reshape %1689 : (tensor<1xf64>) -> tensor<f64>
    %1691 = stablehlo.negate %1690 : tensor<f64>
    %1692 = stablehlo.reshape %1691 : (tensor<f64>) -> tensor<1xf64>
    %1693 = stablehlo.slice %1341 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1694 = stablehlo.reshape %1693 : (tensor<1xf64>) -> tensor<f64>
    %1695 = stablehlo.negate %1694 : tensor<f64>
    %1696 = stablehlo.reshape %1695 : (tensor<f64>) -> tensor<1xf64>
    %1697 = stablehlo.slice %1341 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1698 = stablehlo.reshape %1697 : (tensor<1xf64>) -> tensor<f64>
    %1699 = stablehlo.reshape %1698 : (tensor<f64>) -> tensor<1xf64>
    %1700 = stablehlo.concatenate %1688, %1692, %1696, %1699, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1701 = stablehlo.dot_general %1341, %1341, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1702 = stablehlo.broadcast_in_dim %1701, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1703 = stablehlo.divide %1700, %1702 : tensor<4xf64>
    %1704 = stablehlo.slice %1703 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1705 = stablehlo.reshape %1704 : (tensor<1xf64>) -> tensor<f64>
    %1706 = stablehlo.multiply %1684, %1705 : tensor<f64>
    %1707 = stablehlo.slice %1682 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1708 = stablehlo.reshape %1707 : (tensor<1xf64>) -> tensor<f64>
    %1709 = stablehlo.slice %1703 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1710 = stablehlo.reshape %1709 : (tensor<1xf64>) -> tensor<f64>
    %1711 = stablehlo.multiply %1708, %1710 : tensor<f64>
    %1712 = stablehlo.add %1706, %1711 : tensor<f64>
    %1713 = stablehlo.slice %1682 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1714 = stablehlo.reshape %1713 : (tensor<1xf64>) -> tensor<f64>
    %1715 = stablehlo.slice %1703 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1716 = stablehlo.reshape %1715 : (tensor<1xf64>) -> tensor<f64>
    %1717 = stablehlo.multiply %1714, %1716 : tensor<f64>
    %1718 = stablehlo.add %1712, %1717 : tensor<f64>
    %1719 = stablehlo.slice %1682 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1720 = stablehlo.reshape %1719 : (tensor<1xf64>) -> tensor<f64>
    %1721 = stablehlo.slice %1703 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1722 = stablehlo.reshape %1721 : (tensor<1xf64>) -> tensor<f64>
    %1723 = stablehlo.multiply %1720, %1722 : tensor<f64>
    %1724 = stablehlo.subtract %1718, %1723 : tensor<f64>
    %1725 = stablehlo.reshape %1724 : (tensor<f64>) -> tensor<1xf64>
    %1726 = stablehlo.multiply %1684, %1722 : tensor<f64>
    %1727 = stablehlo.multiply %1708, %1716 : tensor<f64>
    %1728 = stablehlo.subtract %1726, %1727 : tensor<f64>
    %1729 = stablehlo.multiply %1714, %1710 : tensor<f64>
    %1730 = stablehlo.add %1728, %1729 : tensor<f64>
    %1731 = stablehlo.multiply %1720, %1705 : tensor<f64>
    %1732 = stablehlo.add %1730, %1731 : tensor<f64>
    %1733 = stablehlo.reshape %1732 : (tensor<f64>) -> tensor<1xf64>
    %1734 = stablehlo.multiply %1684, %1716 : tensor<f64>
    %1735 = stablehlo.multiply %1708, %1722 : tensor<f64>
    %1736 = stablehlo.add %1734, %1735 : tensor<f64>
    %1737 = stablehlo.multiply %1714, %1705 : tensor<f64>
    %1738 = stablehlo.subtract %1736, %1737 : tensor<f64>
    %1739 = stablehlo.multiply %1720, %1710 : tensor<f64>
    %1740 = stablehlo.add %1738, %1739 : tensor<f64>
    %1741 = stablehlo.reshape %1740 : (tensor<f64>) -> tensor<1xf64>
    %1742 = stablehlo.multiply %1684, %1710 : tensor<f64>
    %1743 = stablehlo.multiply %1708, %1705 : tensor<f64>
    %1744 = stablehlo.subtract %1742, %1743 : tensor<f64>
    %1745 = stablehlo.multiply %1714, %1722 : tensor<f64>
    %1746 = stablehlo.subtract %1744, %1745 : tensor<f64>
    %1747 = stablehlo.multiply %1720, %1716 : tensor<f64>
    %1748 = stablehlo.subtract %1746, %1747 : tensor<f64>
    %1749 = stablehlo.reshape %1748 : (tensor<f64>) -> tensor<1xf64>
    %1750 = stablehlo.concatenate %1725, %1733, %1741, %1749, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1751 = stablehlo.slice %1750 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1752 = stablehlo.reshape %1751 : (tensor<1xf64>) -> tensor<f64>
    %1753 = stablehlo.reshape %1752 : (tensor<f64>) -> tensor<1xf64>
    %1754 = stablehlo.slice %1750 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1755 = stablehlo.reshape %1754 : (tensor<1xf64>) -> tensor<f64>
    %1756 = stablehlo.reshape %1755 : (tensor<f64>) -> tensor<1xf64>
    %1757 = stablehlo.slice %1750 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1758 = stablehlo.reshape %1757 : (tensor<1xf64>) -> tensor<f64>
    %1759 = stablehlo.reshape %1758 : (tensor<f64>) -> tensor<1xf64>
    %1760 = stablehlo.concatenate %1753, %1756, %1759, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1761 = stablehlo.slice %1341 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1762 = stablehlo.reshape %1761 : (tensor<1xf64>) -> tensor<f64>
    %1763 = stablehlo.slice %1632 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_22 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1764 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1765 = stablehlo.concatenate %1763, %1764, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1766 = stablehlo.slice %1765 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1767 = stablehlo.reshape %1766 : (tensor<1xf64>) -> tensor<f64>
    %1768 = stablehlo.multiply %1762, %1767 : tensor<f64>
    %1769 = stablehlo.slice %1341 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1770 = stablehlo.reshape %1769 : (tensor<1xf64>) -> tensor<f64>
    %1771 = stablehlo.slice %1765 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1772 = stablehlo.reshape %1771 : (tensor<1xf64>) -> tensor<f64>
    %1773 = stablehlo.multiply %1770, %1772 : tensor<f64>
    %1774 = stablehlo.add %1768, %1773 : tensor<f64>
    %1775 = stablehlo.slice %1341 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1776 = stablehlo.reshape %1775 : (tensor<1xf64>) -> tensor<f64>
    %1777 = stablehlo.slice %1765 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1778 = stablehlo.reshape %1777 : (tensor<1xf64>) -> tensor<f64>
    %1779 = stablehlo.multiply %1776, %1778 : tensor<f64>
    %1780 = stablehlo.add %1774, %1779 : tensor<f64>
    %1781 = stablehlo.slice %1341 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1782 = stablehlo.reshape %1781 : (tensor<1xf64>) -> tensor<f64>
    %1783 = stablehlo.slice %1765 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1784 = stablehlo.reshape %1783 : (tensor<1xf64>) -> tensor<f64>
    %1785 = stablehlo.multiply %1782, %1784 : tensor<f64>
    %1786 = stablehlo.subtract %1780, %1785 : tensor<f64>
    %1787 = stablehlo.reshape %1786 : (tensor<f64>) -> tensor<1xf64>
    %1788 = stablehlo.multiply %1762, %1784 : tensor<f64>
    %1789 = stablehlo.multiply %1770, %1778 : tensor<f64>
    %1790 = stablehlo.subtract %1788, %1789 : tensor<f64>
    %1791 = stablehlo.multiply %1776, %1772 : tensor<f64>
    %1792 = stablehlo.add %1790, %1791 : tensor<f64>
    %1793 = stablehlo.multiply %1782, %1767 : tensor<f64>
    %1794 = stablehlo.add %1792, %1793 : tensor<f64>
    %1795 = stablehlo.reshape %1794 : (tensor<f64>) -> tensor<1xf64>
    %1796 = stablehlo.multiply %1762, %1778 : tensor<f64>
    %1797 = stablehlo.multiply %1770, %1784 : tensor<f64>
    %1798 = stablehlo.add %1796, %1797 : tensor<f64>
    %1799 = stablehlo.multiply %1776, %1767 : tensor<f64>
    %1800 = stablehlo.subtract %1798, %1799 : tensor<f64>
    %1801 = stablehlo.multiply %1782, %1772 : tensor<f64>
    %1802 = stablehlo.add %1800, %1801 : tensor<f64>
    %1803 = stablehlo.reshape %1802 : (tensor<f64>) -> tensor<1xf64>
    %1804 = stablehlo.multiply %1762, %1772 : tensor<f64>
    %1805 = stablehlo.multiply %1770, %1767 : tensor<f64>
    %1806 = stablehlo.subtract %1804, %1805 : tensor<f64>
    %1807 = stablehlo.multiply %1776, %1784 : tensor<f64>
    %1808 = stablehlo.subtract %1806, %1807 : tensor<f64>
    %1809 = stablehlo.multiply %1782, %1778 : tensor<f64>
    %1810 = stablehlo.subtract %1808, %1809 : tensor<f64>
    %1811 = stablehlo.reshape %1810 : (tensor<f64>) -> tensor<1xf64>
    %1812 = stablehlo.concatenate %1787, %1795, %1803, %1811, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1813 = stablehlo.slice %1812 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1814 = stablehlo.reshape %1813 : (tensor<1xf64>) -> tensor<f64>
    %1815 = stablehlo.slice %1341 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1816 = stablehlo.reshape %1815 : (tensor<1xf64>) -> tensor<f64>
    %1817 = stablehlo.negate %1816 : tensor<f64>
    %1818 = stablehlo.reshape %1817 : (tensor<f64>) -> tensor<1xf64>
    %1819 = stablehlo.slice %1341 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1820 = stablehlo.reshape %1819 : (tensor<1xf64>) -> tensor<f64>
    %1821 = stablehlo.negate %1820 : tensor<f64>
    %1822 = stablehlo.reshape %1821 : (tensor<f64>) -> tensor<1xf64>
    %1823 = stablehlo.slice %1341 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1824 = stablehlo.reshape %1823 : (tensor<1xf64>) -> tensor<f64>
    %1825 = stablehlo.negate %1824 : tensor<f64>
    %1826 = stablehlo.reshape %1825 : (tensor<f64>) -> tensor<1xf64>
    %1827 = stablehlo.slice %1341 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1828 = stablehlo.reshape %1827 : (tensor<1xf64>) -> tensor<f64>
    %1829 = stablehlo.reshape %1828 : (tensor<f64>) -> tensor<1xf64>
    %1830 = stablehlo.concatenate %1818, %1822, %1826, %1829, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1831 = stablehlo.dot_general %1341, %1341, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1832 = stablehlo.broadcast_in_dim %1831, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1833 = stablehlo.divide %1830, %1832 : tensor<4xf64>
    %1834 = stablehlo.slice %1833 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1835 = stablehlo.reshape %1834 : (tensor<1xf64>) -> tensor<f64>
    %1836 = stablehlo.multiply %1814, %1835 : tensor<f64>
    %1837 = stablehlo.slice %1812 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1838 = stablehlo.reshape %1837 : (tensor<1xf64>) -> tensor<f64>
    %1839 = stablehlo.slice %1833 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1840 = stablehlo.reshape %1839 : (tensor<1xf64>) -> tensor<f64>
    %1841 = stablehlo.multiply %1838, %1840 : tensor<f64>
    %1842 = stablehlo.add %1836, %1841 : tensor<f64>
    %1843 = stablehlo.slice %1812 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1844 = stablehlo.reshape %1843 : (tensor<1xf64>) -> tensor<f64>
    %1845 = stablehlo.slice %1833 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1846 = stablehlo.reshape %1845 : (tensor<1xf64>) -> tensor<f64>
    %1847 = stablehlo.multiply %1844, %1846 : tensor<f64>
    %1848 = stablehlo.add %1842, %1847 : tensor<f64>
    %1849 = stablehlo.slice %1812 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1850 = stablehlo.reshape %1849 : (tensor<1xf64>) -> tensor<f64>
    %1851 = stablehlo.slice %1833 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1852 = stablehlo.reshape %1851 : (tensor<1xf64>) -> tensor<f64>
    %1853 = stablehlo.multiply %1850, %1852 : tensor<f64>
    %1854 = stablehlo.subtract %1848, %1853 : tensor<f64>
    %1855 = stablehlo.reshape %1854 : (tensor<f64>) -> tensor<1xf64>
    %1856 = stablehlo.multiply %1814, %1852 : tensor<f64>
    %1857 = stablehlo.multiply %1838, %1846 : tensor<f64>
    %1858 = stablehlo.subtract %1856, %1857 : tensor<f64>
    %1859 = stablehlo.multiply %1844, %1840 : tensor<f64>
    %1860 = stablehlo.add %1858, %1859 : tensor<f64>
    %1861 = stablehlo.multiply %1850, %1835 : tensor<f64>
    %1862 = stablehlo.add %1860, %1861 : tensor<f64>
    %1863 = stablehlo.reshape %1862 : (tensor<f64>) -> tensor<1xf64>
    %1864 = stablehlo.multiply %1814, %1846 : tensor<f64>
    %1865 = stablehlo.multiply %1838, %1852 : tensor<f64>
    %1866 = stablehlo.add %1864, %1865 : tensor<f64>
    %1867 = stablehlo.multiply %1844, %1835 : tensor<f64>
    %1868 = stablehlo.subtract %1866, %1867 : tensor<f64>
    %1869 = stablehlo.multiply %1850, %1840 : tensor<f64>
    %1870 = stablehlo.add %1868, %1869 : tensor<f64>
    %1871 = stablehlo.reshape %1870 : (tensor<f64>) -> tensor<1xf64>
    %1872 = stablehlo.multiply %1814, %1840 : tensor<f64>
    %1873 = stablehlo.multiply %1838, %1835 : tensor<f64>
    %1874 = stablehlo.subtract %1872, %1873 : tensor<f64>
    %1875 = stablehlo.multiply %1844, %1852 : tensor<f64>
    %1876 = stablehlo.subtract %1874, %1875 : tensor<f64>
    %1877 = stablehlo.multiply %1850, %1846 : tensor<f64>
    %1878 = stablehlo.subtract %1876, %1877 : tensor<f64>
    %1879 = stablehlo.reshape %1878 : (tensor<f64>) -> tensor<1xf64>
    %1880 = stablehlo.concatenate %1855, %1863, %1871, %1879, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1881 = stablehlo.slice %1880 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1882 = stablehlo.reshape %1881 : (tensor<1xf64>) -> tensor<f64>
    %1883 = stablehlo.reshape %1882 : (tensor<f64>) -> tensor<1xf64>
    %1884 = stablehlo.slice %1880 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1885 = stablehlo.reshape %1884 : (tensor<1xf64>) -> tensor<f64>
    %1886 = stablehlo.reshape %1885 : (tensor<f64>) -> tensor<1xf64>
    %1887 = stablehlo.slice %1880 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1888 = stablehlo.reshape %1887 : (tensor<1xf64>) -> tensor<f64>
    %1889 = stablehlo.reshape %1888 : (tensor<f64>) -> tensor<1xf64>
    %1890 = stablehlo.concatenate %1883, %1886, %1889, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %1891 = stablehlo.concatenate %1760, %1890, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %1892 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1893 = stablehlo.reshape %arg22 : (tensor<f64>) -> tensor<f64>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1894 = stablehlo.multiply %cst_23, %1893 : tensor<f64>
    %1895 = stablehlo.broadcast_in_dim %1894, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1896 = stablehlo.multiply %1895, %arg2 : tensor<6xf64>
    %1897 = stablehlo.slice %1896 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_24 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %1898 = stablehlo.broadcast_in_dim %cst_24, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %1899 = stablehlo.divide %1897, %1898 : tensor<3xf64>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1900 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1901 = stablehlo.concatenate %1899, %1900, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1902 = stablehlo.slice %1901 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1903 = stablehlo.reshape %1902 : (tensor<1xf64>) -> tensor<f64>
    %1904 = stablehlo.slice %1892 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1905 = stablehlo.reshape %1904 : (tensor<1xf64>) -> tensor<f64>
    %1906 = stablehlo.multiply %1903, %1905 : tensor<f64>
    %1907 = stablehlo.slice %1901 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1908 = stablehlo.reshape %1907 : (tensor<1xf64>) -> tensor<f64>
    %1909 = stablehlo.slice %1892 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1910 = stablehlo.reshape %1909 : (tensor<1xf64>) -> tensor<f64>
    %1911 = stablehlo.multiply %1908, %1910 : tensor<f64>
    %1912 = stablehlo.add %1906, %1911 : tensor<f64>
    %1913 = stablehlo.slice %1901 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1914 = stablehlo.reshape %1913 : (tensor<1xf64>) -> tensor<f64>
    %1915 = stablehlo.slice %1892 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1916 = stablehlo.reshape %1915 : (tensor<1xf64>) -> tensor<f64>
    %1917 = stablehlo.multiply %1914, %1916 : tensor<f64>
    %1918 = stablehlo.add %1912, %1917 : tensor<f64>
    %1919 = stablehlo.slice %1901 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1920 = stablehlo.reshape %1919 : (tensor<1xf64>) -> tensor<f64>
    %1921 = stablehlo.slice %1892 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1922 = stablehlo.reshape %1921 : (tensor<1xf64>) -> tensor<f64>
    %1923 = stablehlo.multiply %1920, %1922 : tensor<f64>
    %1924 = stablehlo.subtract %1918, %1923 : tensor<f64>
    %1925 = stablehlo.reshape %1924 : (tensor<f64>) -> tensor<1xf64>
    %1926 = stablehlo.multiply %1903, %1922 : tensor<f64>
    %1927 = stablehlo.multiply %1908, %1916 : tensor<f64>
    %1928 = stablehlo.subtract %1926, %1927 : tensor<f64>
    %1929 = stablehlo.multiply %1914, %1910 : tensor<f64>
    %1930 = stablehlo.add %1928, %1929 : tensor<f64>
    %1931 = stablehlo.multiply %1920, %1905 : tensor<f64>
    %1932 = stablehlo.add %1930, %1931 : tensor<f64>
    %1933 = stablehlo.reshape %1932 : (tensor<f64>) -> tensor<1xf64>
    %1934 = stablehlo.multiply %1903, %1916 : tensor<f64>
    %1935 = stablehlo.multiply %1908, %1922 : tensor<f64>
    %1936 = stablehlo.add %1934, %1935 : tensor<f64>
    %1937 = stablehlo.multiply %1914, %1905 : tensor<f64>
    %1938 = stablehlo.subtract %1936, %1937 : tensor<f64>
    %1939 = stablehlo.multiply %1920, %1910 : tensor<f64>
    %1940 = stablehlo.add %1938, %1939 : tensor<f64>
    %1941 = stablehlo.reshape %1940 : (tensor<f64>) -> tensor<1xf64>
    %1942 = stablehlo.multiply %1903, %1910 : tensor<f64>
    %1943 = stablehlo.multiply %1908, %1905 : tensor<f64>
    %1944 = stablehlo.subtract %1942, %1943 : tensor<f64>
    %1945 = stablehlo.multiply %1914, %1922 : tensor<f64>
    %1946 = stablehlo.subtract %1944, %1945 : tensor<f64>
    %1947 = stablehlo.multiply %1920, %1916 : tensor<f64>
    %1948 = stablehlo.subtract %1946, %1947 : tensor<f64>
    %1949 = stablehlo.reshape %1948 : (tensor<f64>) -> tensor<1xf64>
    %1950 = stablehlo.concatenate %1925, %1933, %1941, %1949, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1951 = stablehlo.add %1892, %1950 : tensor<4xf64>
    %1952 = stablehlo.dot_general %1951, %1951, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1953 = stablehlo.sqrt %1952 : tensor<f64>
    %1954 = stablehlo.broadcast_in_dim %1953, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1955 = stablehlo.divide %1951, %1954 : tensor<4xf64>
    %1956 = stablehlo.slice %arg1 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1957 = stablehlo.slice %1896 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1958 = stablehlo.add %1956, %1957 : tensor<3xf64>
    %1959 = stablehlo.concatenate %1955, %1958, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    %1960 = stablehlo.broadcast_in_dim %1894, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1961 = stablehlo.multiply %1960, %1891 : tensor<6xf64>
    %1962 = stablehlo.add %arg2, %1961 : tensor<6xf64>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1963 = stablehlo.broadcast_in_dim %cst_26, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %1964 = call @inner_124(%1963, %arg26) : (tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %1965 = call @inner_126(%13, %1964, %1959) : (tensor<f64>, tensor<6xf64>, tensor<7xf64>) -> tensor<6xf64>
    %1966 = call @inner_127(%1959, %12, %1965) : (tensor<7xf64>, tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>
    %1967 = stablehlo.slice %1959 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1968 = stablehlo.slice %1967 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1969 = stablehlo.reshape %1968 : (tensor<1xf64>) -> tensor<f64>
    %1970 = stablehlo.slice %1967 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1971 = stablehlo.reshape %1970 : (tensor<1xf64>) -> tensor<f64>
    %1972 = stablehlo.negate %1971 : tensor<f64>
    %1973 = stablehlo.reshape %1972 : (tensor<f64>) -> tensor<1xf64>
    %1974 = stablehlo.slice %1967 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %1975 = stablehlo.reshape %1974 : (tensor<1xf64>) -> tensor<f64>
    %1976 = stablehlo.negate %1975 : tensor<f64>
    %1977 = stablehlo.reshape %1976 : (tensor<f64>) -> tensor<1xf64>
    %1978 = stablehlo.slice %1967 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %1979 = stablehlo.reshape %1978 : (tensor<1xf64>) -> tensor<f64>
    %1980 = stablehlo.negate %1979 : tensor<f64>
    %1981 = stablehlo.reshape %1980 : (tensor<f64>) -> tensor<1xf64>
    %1982 = stablehlo.slice %1967 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1983 = stablehlo.reshape %1982 : (tensor<1xf64>) -> tensor<f64>
    %1984 = stablehlo.reshape %1983 : (tensor<f64>) -> tensor<1xf64>
    %1985 = stablehlo.concatenate %1973, %1977, %1981, %1984, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1986 = stablehlo.dot_general %1967, %1967, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %1987 = stablehlo.broadcast_in_dim %1986, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1988 = stablehlo.divide %1985, %1987 : tensor<4xf64>
    %1989 = stablehlo.slice %1988 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %1990 = stablehlo.reshape %1989 : (tensor<1xf64>) -> tensor<f64>
    %1991 = stablehlo.slice %1966 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1992 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %1993 = stablehlo.concatenate %1991, %1992, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %1994 = stablehlo.slice %1993 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1995 = stablehlo.reshape %1994 : (tensor<1xf64>) -> tensor<f64>
    %1996 = stablehlo.multiply %1990, %1995 : tensor<f64>
    %1997 = stablehlo.slice %1988 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %1998 = stablehlo.reshape %1997 : (tensor<1xf64>) -> tensor<f64>
    %1999 = stablehlo.slice %1993 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2000 = stablehlo.reshape %1999 : (tensor<1xf64>) -> tensor<f64>
    %2001 = stablehlo.multiply %1998, %2000 : tensor<f64>
    %2002 = stablehlo.add %1996, %2001 : tensor<f64>
    %2003 = stablehlo.slice %1988 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2004 = stablehlo.reshape %2003 : (tensor<1xf64>) -> tensor<f64>
    %2005 = stablehlo.slice %1993 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2006 = stablehlo.reshape %2005 : (tensor<1xf64>) -> tensor<f64>
    %2007 = stablehlo.multiply %2004, %2006 : tensor<f64>
    %2008 = stablehlo.add %2002, %2007 : tensor<f64>
    %2009 = stablehlo.slice %1988 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2010 = stablehlo.reshape %2009 : (tensor<1xf64>) -> tensor<f64>
    %2011 = stablehlo.slice %1993 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2012 = stablehlo.reshape %2011 : (tensor<1xf64>) -> tensor<f64>
    %2013 = stablehlo.multiply %2010, %2012 : tensor<f64>
    %2014 = stablehlo.subtract %2008, %2013 : tensor<f64>
    %2015 = stablehlo.reshape %2014 : (tensor<f64>) -> tensor<1xf64>
    %2016 = stablehlo.multiply %1990, %2012 : tensor<f64>
    %2017 = stablehlo.multiply %1998, %2006 : tensor<f64>
    %2018 = stablehlo.subtract %2016, %2017 : tensor<f64>
    %2019 = stablehlo.multiply %2004, %2000 : tensor<f64>
    %2020 = stablehlo.add %2018, %2019 : tensor<f64>
    %2021 = stablehlo.multiply %2010, %1995 : tensor<f64>
    %2022 = stablehlo.add %2020, %2021 : tensor<f64>
    %2023 = stablehlo.reshape %2022 : (tensor<f64>) -> tensor<1xf64>
    %2024 = stablehlo.multiply %1990, %2006 : tensor<f64>
    %2025 = stablehlo.multiply %1998, %2012 : tensor<f64>
    %2026 = stablehlo.add %2024, %2025 : tensor<f64>
    %2027 = stablehlo.multiply %2004, %1995 : tensor<f64>
    %2028 = stablehlo.subtract %2026, %2027 : tensor<f64>
    %2029 = stablehlo.multiply %2010, %2000 : tensor<f64>
    %2030 = stablehlo.add %2028, %2029 : tensor<f64>
    %2031 = stablehlo.reshape %2030 : (tensor<f64>) -> tensor<1xf64>
    %2032 = stablehlo.multiply %1990, %2000 : tensor<f64>
    %2033 = stablehlo.multiply %1998, %1995 : tensor<f64>
    %2034 = stablehlo.subtract %2032, %2033 : tensor<f64>
    %2035 = stablehlo.multiply %2004, %2012 : tensor<f64>
    %2036 = stablehlo.subtract %2034, %2035 : tensor<f64>
    %2037 = stablehlo.multiply %2010, %2006 : tensor<f64>
    %2038 = stablehlo.subtract %2036, %2037 : tensor<f64>
    %2039 = stablehlo.reshape %2038 : (tensor<f64>) -> tensor<1xf64>
    %2040 = stablehlo.concatenate %2015, %2023, %2031, %2039, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2041 = stablehlo.slice %2040 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2042 = stablehlo.reshape %2041 : (tensor<1xf64>) -> tensor<f64>
    %2043 = stablehlo.slice %1988 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2044 = stablehlo.reshape %2043 : (tensor<1xf64>) -> tensor<f64>
    %2045 = stablehlo.negate %2044 : tensor<f64>
    %2046 = stablehlo.reshape %2045 : (tensor<f64>) -> tensor<1xf64>
    %2047 = stablehlo.slice %1988 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2048 = stablehlo.reshape %2047 : (tensor<1xf64>) -> tensor<f64>
    %2049 = stablehlo.negate %2048 : tensor<f64>
    %2050 = stablehlo.reshape %2049 : (tensor<f64>) -> tensor<1xf64>
    %2051 = stablehlo.slice %1988 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2052 = stablehlo.reshape %2051 : (tensor<1xf64>) -> tensor<f64>
    %2053 = stablehlo.negate %2052 : tensor<f64>
    %2054 = stablehlo.reshape %2053 : (tensor<f64>) -> tensor<1xf64>
    %2055 = stablehlo.slice %1988 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2056 = stablehlo.reshape %2055 : (tensor<1xf64>) -> tensor<f64>
    %2057 = stablehlo.reshape %2056 : (tensor<f64>) -> tensor<1xf64>
    %2058 = stablehlo.concatenate %2046, %2050, %2054, %2057, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2059 = stablehlo.dot_general %1988, %1988, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2060 = stablehlo.broadcast_in_dim %2059, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2061 = stablehlo.divide %2058, %2060 : tensor<4xf64>
    %2062 = stablehlo.slice %2061 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2063 = stablehlo.reshape %2062 : (tensor<1xf64>) -> tensor<f64>
    %2064 = stablehlo.multiply %2042, %2063 : tensor<f64>
    %2065 = stablehlo.slice %2040 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2066 = stablehlo.reshape %2065 : (tensor<1xf64>) -> tensor<f64>
    %2067 = stablehlo.slice %2061 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2068 = stablehlo.reshape %2067 : (tensor<1xf64>) -> tensor<f64>
    %2069 = stablehlo.multiply %2066, %2068 : tensor<f64>
    %2070 = stablehlo.add %2064, %2069 : tensor<f64>
    %2071 = stablehlo.slice %2040 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2072 = stablehlo.reshape %2071 : (tensor<1xf64>) -> tensor<f64>
    %2073 = stablehlo.slice %2061 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2074 = stablehlo.reshape %2073 : (tensor<1xf64>) -> tensor<f64>
    %2075 = stablehlo.multiply %2072, %2074 : tensor<f64>
    %2076 = stablehlo.add %2070, %2075 : tensor<f64>
    %2077 = stablehlo.slice %2040 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2078 = stablehlo.reshape %2077 : (tensor<1xf64>) -> tensor<f64>
    %2079 = stablehlo.slice %2061 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2080 = stablehlo.reshape %2079 : (tensor<1xf64>) -> tensor<f64>
    %2081 = stablehlo.multiply %2078, %2080 : tensor<f64>
    %2082 = stablehlo.subtract %2076, %2081 : tensor<f64>
    %2083 = stablehlo.reshape %2082 : (tensor<f64>) -> tensor<1xf64>
    %2084 = stablehlo.multiply %2042, %2080 : tensor<f64>
    %2085 = stablehlo.multiply %2066, %2074 : tensor<f64>
    %2086 = stablehlo.subtract %2084, %2085 : tensor<f64>
    %2087 = stablehlo.multiply %2072, %2068 : tensor<f64>
    %2088 = stablehlo.add %2086, %2087 : tensor<f64>
    %2089 = stablehlo.multiply %2078, %2063 : tensor<f64>
    %2090 = stablehlo.add %2088, %2089 : tensor<f64>
    %2091 = stablehlo.reshape %2090 : (tensor<f64>) -> tensor<1xf64>
    %2092 = stablehlo.multiply %2042, %2074 : tensor<f64>
    %2093 = stablehlo.multiply %2066, %2080 : tensor<f64>
    %2094 = stablehlo.add %2092, %2093 : tensor<f64>
    %2095 = stablehlo.multiply %2072, %2063 : tensor<f64>
    %2096 = stablehlo.subtract %2094, %2095 : tensor<f64>
    %2097 = stablehlo.multiply %2078, %2068 : tensor<f64>
    %2098 = stablehlo.add %2096, %2097 : tensor<f64>
    %2099 = stablehlo.reshape %2098 : (tensor<f64>) -> tensor<1xf64>
    %2100 = stablehlo.multiply %2042, %2068 : tensor<f64>
    %2101 = stablehlo.multiply %2066, %2063 : tensor<f64>
    %2102 = stablehlo.subtract %2100, %2101 : tensor<f64>
    %2103 = stablehlo.multiply %2072, %2080 : tensor<f64>
    %2104 = stablehlo.subtract %2102, %2103 : tensor<f64>
    %2105 = stablehlo.multiply %2078, %2074 : tensor<f64>
    %2106 = stablehlo.subtract %2104, %2105 : tensor<f64>
    %2107 = stablehlo.reshape %2106 : (tensor<f64>) -> tensor<1xf64>
    %2108 = stablehlo.concatenate %2083, %2091, %2099, %2107, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2109 = stablehlo.slice %2108 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2110 = stablehlo.reshape %2109 : (tensor<1xf64>) -> tensor<f64>
    %2111 = stablehlo.reshape %2110 : (tensor<f64>) -> tensor<1xf64>
    %2112 = stablehlo.slice %2108 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2113 = stablehlo.reshape %2112 : (tensor<1xf64>) -> tensor<f64>
    %2114 = stablehlo.reshape %2113 : (tensor<f64>) -> tensor<1xf64>
    %2115 = stablehlo.slice %2108 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2116 = stablehlo.reshape %2115 : (tensor<1xf64>) -> tensor<f64>
    %2117 = stablehlo.reshape %2116 : (tensor<f64>) -> tensor<1xf64>
    %2118 = stablehlo.concatenate %2111, %2114, %2117, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2119 = stablehlo.slice %1988 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2120 = stablehlo.reshape %2119 : (tensor<1xf64>) -> tensor<f64>
    %2121 = stablehlo.slice %1966 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2122 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2123 = stablehlo.concatenate %2121, %2122, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2124 = stablehlo.slice %2123 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2125 = stablehlo.reshape %2124 : (tensor<1xf64>) -> tensor<f64>
    %2126 = stablehlo.multiply %2120, %2125 : tensor<f64>
    %2127 = stablehlo.slice %1988 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2128 = stablehlo.reshape %2127 : (tensor<1xf64>) -> tensor<f64>
    %2129 = stablehlo.slice %2123 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2130 = stablehlo.reshape %2129 : (tensor<1xf64>) -> tensor<f64>
    %2131 = stablehlo.multiply %2128, %2130 : tensor<f64>
    %2132 = stablehlo.add %2126, %2131 : tensor<f64>
    %2133 = stablehlo.slice %1988 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2134 = stablehlo.reshape %2133 : (tensor<1xf64>) -> tensor<f64>
    %2135 = stablehlo.slice %2123 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2136 = stablehlo.reshape %2135 : (tensor<1xf64>) -> tensor<f64>
    %2137 = stablehlo.multiply %2134, %2136 : tensor<f64>
    %2138 = stablehlo.add %2132, %2137 : tensor<f64>
    %2139 = stablehlo.slice %1988 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2140 = stablehlo.reshape %2139 : (tensor<1xf64>) -> tensor<f64>
    %2141 = stablehlo.slice %2123 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2142 = stablehlo.reshape %2141 : (tensor<1xf64>) -> tensor<f64>
    %2143 = stablehlo.multiply %2140, %2142 : tensor<f64>
    %2144 = stablehlo.subtract %2138, %2143 : tensor<f64>
    %2145 = stablehlo.reshape %2144 : (tensor<f64>) -> tensor<1xf64>
    %2146 = stablehlo.multiply %2120, %2142 : tensor<f64>
    %2147 = stablehlo.multiply %2128, %2136 : tensor<f64>
    %2148 = stablehlo.subtract %2146, %2147 : tensor<f64>
    %2149 = stablehlo.multiply %2134, %2130 : tensor<f64>
    %2150 = stablehlo.add %2148, %2149 : tensor<f64>
    %2151 = stablehlo.multiply %2140, %2125 : tensor<f64>
    %2152 = stablehlo.add %2150, %2151 : tensor<f64>
    %2153 = stablehlo.reshape %2152 : (tensor<f64>) -> tensor<1xf64>
    %2154 = stablehlo.multiply %2120, %2136 : tensor<f64>
    %2155 = stablehlo.multiply %2128, %2142 : tensor<f64>
    %2156 = stablehlo.add %2154, %2155 : tensor<f64>
    %2157 = stablehlo.multiply %2134, %2125 : tensor<f64>
    %2158 = stablehlo.subtract %2156, %2157 : tensor<f64>
    %2159 = stablehlo.multiply %2140, %2130 : tensor<f64>
    %2160 = stablehlo.add %2158, %2159 : tensor<f64>
    %2161 = stablehlo.reshape %2160 : (tensor<f64>) -> tensor<1xf64>
    %2162 = stablehlo.multiply %2120, %2130 : tensor<f64>
    %2163 = stablehlo.multiply %2128, %2125 : tensor<f64>
    %2164 = stablehlo.subtract %2162, %2163 : tensor<f64>
    %2165 = stablehlo.multiply %2134, %2142 : tensor<f64>
    %2166 = stablehlo.subtract %2164, %2165 : tensor<f64>
    %2167 = stablehlo.multiply %2140, %2136 : tensor<f64>
    %2168 = stablehlo.subtract %2166, %2167 : tensor<f64>
    %2169 = stablehlo.reshape %2168 : (tensor<f64>) -> tensor<1xf64>
    %2170 = stablehlo.concatenate %2145, %2153, %2161, %2169, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2171 = stablehlo.slice %2170 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2172 = stablehlo.reshape %2171 : (tensor<1xf64>) -> tensor<f64>
    %2173 = stablehlo.slice %1988 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2174 = stablehlo.reshape %2173 : (tensor<1xf64>) -> tensor<f64>
    %2175 = stablehlo.negate %2174 : tensor<f64>
    %2176 = stablehlo.reshape %2175 : (tensor<f64>) -> tensor<1xf64>
    %2177 = stablehlo.slice %1988 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2178 = stablehlo.reshape %2177 : (tensor<1xf64>) -> tensor<f64>
    %2179 = stablehlo.negate %2178 : tensor<f64>
    %2180 = stablehlo.reshape %2179 : (tensor<f64>) -> tensor<1xf64>
    %2181 = stablehlo.slice %1988 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2182 = stablehlo.reshape %2181 : (tensor<1xf64>) -> tensor<f64>
    %2183 = stablehlo.negate %2182 : tensor<f64>
    %2184 = stablehlo.reshape %2183 : (tensor<f64>) -> tensor<1xf64>
    %2185 = stablehlo.slice %1988 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2186 = stablehlo.reshape %2185 : (tensor<1xf64>) -> tensor<f64>
    %2187 = stablehlo.reshape %2186 : (tensor<f64>) -> tensor<1xf64>
    %2188 = stablehlo.concatenate %2176, %2180, %2184, %2187, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2189 = stablehlo.dot_general %1988, %1988, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2190 = stablehlo.broadcast_in_dim %2189, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2191 = stablehlo.divide %2188, %2190 : tensor<4xf64>
    %2192 = stablehlo.slice %2191 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2193 = stablehlo.reshape %2192 : (tensor<1xf64>) -> tensor<f64>
    %2194 = stablehlo.multiply %2172, %2193 : tensor<f64>
    %2195 = stablehlo.slice %2170 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2196 = stablehlo.reshape %2195 : (tensor<1xf64>) -> tensor<f64>
    %2197 = stablehlo.slice %2191 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2198 = stablehlo.reshape %2197 : (tensor<1xf64>) -> tensor<f64>
    %2199 = stablehlo.multiply %2196, %2198 : tensor<f64>
    %2200 = stablehlo.add %2194, %2199 : tensor<f64>
    %2201 = stablehlo.slice %2170 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2202 = stablehlo.reshape %2201 : (tensor<1xf64>) -> tensor<f64>
    %2203 = stablehlo.slice %2191 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2204 = stablehlo.reshape %2203 : (tensor<1xf64>) -> tensor<f64>
    %2205 = stablehlo.multiply %2202, %2204 : tensor<f64>
    %2206 = stablehlo.add %2200, %2205 : tensor<f64>
    %2207 = stablehlo.slice %2170 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2208 = stablehlo.reshape %2207 : (tensor<1xf64>) -> tensor<f64>
    %2209 = stablehlo.slice %2191 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2210 = stablehlo.reshape %2209 : (tensor<1xf64>) -> tensor<f64>
    %2211 = stablehlo.multiply %2208, %2210 : tensor<f64>
    %2212 = stablehlo.subtract %2206, %2211 : tensor<f64>
    %2213 = stablehlo.reshape %2212 : (tensor<f64>) -> tensor<1xf64>
    %2214 = stablehlo.multiply %2172, %2210 : tensor<f64>
    %2215 = stablehlo.multiply %2196, %2204 : tensor<f64>
    %2216 = stablehlo.subtract %2214, %2215 : tensor<f64>
    %2217 = stablehlo.multiply %2202, %2198 : tensor<f64>
    %2218 = stablehlo.add %2216, %2217 : tensor<f64>
    %2219 = stablehlo.multiply %2208, %2193 : tensor<f64>
    %2220 = stablehlo.add %2218, %2219 : tensor<f64>
    %2221 = stablehlo.reshape %2220 : (tensor<f64>) -> tensor<1xf64>
    %2222 = stablehlo.multiply %2172, %2204 : tensor<f64>
    %2223 = stablehlo.multiply %2196, %2210 : tensor<f64>
    %2224 = stablehlo.add %2222, %2223 : tensor<f64>
    %2225 = stablehlo.multiply %2202, %2193 : tensor<f64>
    %2226 = stablehlo.subtract %2224, %2225 : tensor<f64>
    %2227 = stablehlo.multiply %2208, %2198 : tensor<f64>
    %2228 = stablehlo.add %2226, %2227 : tensor<f64>
    %2229 = stablehlo.reshape %2228 : (tensor<f64>) -> tensor<1xf64>
    %2230 = stablehlo.multiply %2172, %2198 : tensor<f64>
    %2231 = stablehlo.multiply %2196, %2193 : tensor<f64>
    %2232 = stablehlo.subtract %2230, %2231 : tensor<f64>
    %2233 = stablehlo.multiply %2202, %2210 : tensor<f64>
    %2234 = stablehlo.subtract %2232, %2233 : tensor<f64>
    %2235 = stablehlo.multiply %2208, %2204 : tensor<f64>
    %2236 = stablehlo.subtract %2234, %2235 : tensor<f64>
    %2237 = stablehlo.reshape %2236 : (tensor<f64>) -> tensor<1xf64>
    %2238 = stablehlo.concatenate %2213, %2221, %2229, %2237, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2239 = stablehlo.slice %2238 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2240 = stablehlo.reshape %2239 : (tensor<1xf64>) -> tensor<f64>
    %2241 = stablehlo.reshape %2240 : (tensor<f64>) -> tensor<1xf64>
    %2242 = stablehlo.slice %2238 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2243 = stablehlo.reshape %2242 : (tensor<1xf64>) -> tensor<f64>
    %2244 = stablehlo.reshape %2243 : (tensor<f64>) -> tensor<1xf64>
    %2245 = stablehlo.slice %2238 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2246 = stablehlo.reshape %2245 : (tensor<1xf64>) -> tensor<f64>
    %2247 = stablehlo.reshape %2246 : (tensor<f64>) -> tensor<1xf64>
    %2248 = stablehlo.concatenate %2241, %2244, %2247, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2249 = stablehlo.concatenate %2118, %2248, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2250 = stablehlo.slice %2249 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %2251 = stablehlo.slice %arg26 [0:3] : (tensor<7xf64>) -> tensor<3xf64>
    %2252 = stablehlo.divide %2250, %2251 : tensor<3xf64>
    %2253 = stablehlo.slice %2249 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %2254 = stablehlo.slice %arg26 [6:7] : (tensor<7xf64>) -> tensor<1xf64>
    %2255 = stablehlo.reshape %2254 : (tensor<1xf64>) -> tensor<f64>
    %2256 = stablehlo.broadcast_in_dim %2255, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2257 = stablehlo.divide %2253, %2256 : tensor<3xf64>
    %2258 = stablehlo.concatenate %2252, %2257, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2259 = stablehlo.slice %2258 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_29 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2260 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2261 = stablehlo.concatenate %2259, %2260, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2262 = stablehlo.slice %2261 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2263 = stablehlo.reshape %2262 : (tensor<1xf64>) -> tensor<f64>
    %2264 = stablehlo.multiply %1969, %2263 : tensor<f64>
    %2265 = stablehlo.slice %1967 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2266 = stablehlo.reshape %2265 : (tensor<1xf64>) -> tensor<f64>
    %2267 = stablehlo.slice %2261 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2268 = stablehlo.reshape %2267 : (tensor<1xf64>) -> tensor<f64>
    %2269 = stablehlo.multiply %2266, %2268 : tensor<f64>
    %2270 = stablehlo.add %2264, %2269 : tensor<f64>
    %2271 = stablehlo.slice %1967 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2272 = stablehlo.reshape %2271 : (tensor<1xf64>) -> tensor<f64>
    %2273 = stablehlo.slice %2261 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2274 = stablehlo.reshape %2273 : (tensor<1xf64>) -> tensor<f64>
    %2275 = stablehlo.multiply %2272, %2274 : tensor<f64>
    %2276 = stablehlo.add %2270, %2275 : tensor<f64>
    %2277 = stablehlo.slice %1967 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2278 = stablehlo.reshape %2277 : (tensor<1xf64>) -> tensor<f64>
    %2279 = stablehlo.slice %2261 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2280 = stablehlo.reshape %2279 : (tensor<1xf64>) -> tensor<f64>
    %2281 = stablehlo.multiply %2278, %2280 : tensor<f64>
    %2282 = stablehlo.subtract %2276, %2281 : tensor<f64>
    %2283 = stablehlo.reshape %2282 : (tensor<f64>) -> tensor<1xf64>
    %2284 = stablehlo.multiply %1969, %2280 : tensor<f64>
    %2285 = stablehlo.multiply %2266, %2274 : tensor<f64>
    %2286 = stablehlo.subtract %2284, %2285 : tensor<f64>
    %2287 = stablehlo.multiply %2272, %2268 : tensor<f64>
    %2288 = stablehlo.add %2286, %2287 : tensor<f64>
    %2289 = stablehlo.multiply %2278, %2263 : tensor<f64>
    %2290 = stablehlo.add %2288, %2289 : tensor<f64>
    %2291 = stablehlo.reshape %2290 : (tensor<f64>) -> tensor<1xf64>
    %2292 = stablehlo.multiply %1969, %2274 : tensor<f64>
    %2293 = stablehlo.multiply %2266, %2280 : tensor<f64>
    %2294 = stablehlo.add %2292, %2293 : tensor<f64>
    %2295 = stablehlo.multiply %2272, %2263 : tensor<f64>
    %2296 = stablehlo.subtract %2294, %2295 : tensor<f64>
    %2297 = stablehlo.multiply %2278, %2268 : tensor<f64>
    %2298 = stablehlo.add %2296, %2297 : tensor<f64>
    %2299 = stablehlo.reshape %2298 : (tensor<f64>) -> tensor<1xf64>
    %2300 = stablehlo.multiply %1969, %2268 : tensor<f64>
    %2301 = stablehlo.multiply %2266, %2263 : tensor<f64>
    %2302 = stablehlo.subtract %2300, %2301 : tensor<f64>
    %2303 = stablehlo.multiply %2272, %2280 : tensor<f64>
    %2304 = stablehlo.subtract %2302, %2303 : tensor<f64>
    %2305 = stablehlo.multiply %2278, %2274 : tensor<f64>
    %2306 = stablehlo.subtract %2304, %2305 : tensor<f64>
    %2307 = stablehlo.reshape %2306 : (tensor<f64>) -> tensor<1xf64>
    %2308 = stablehlo.concatenate %2283, %2291, %2299, %2307, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2309 = stablehlo.slice %2308 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2310 = stablehlo.reshape %2309 : (tensor<1xf64>) -> tensor<f64>
    %2311 = stablehlo.slice %1967 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2312 = stablehlo.reshape %2311 : (tensor<1xf64>) -> tensor<f64>
    %2313 = stablehlo.negate %2312 : tensor<f64>
    %2314 = stablehlo.reshape %2313 : (tensor<f64>) -> tensor<1xf64>
    %2315 = stablehlo.slice %1967 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2316 = stablehlo.reshape %2315 : (tensor<1xf64>) -> tensor<f64>
    %2317 = stablehlo.negate %2316 : tensor<f64>
    %2318 = stablehlo.reshape %2317 : (tensor<f64>) -> tensor<1xf64>
    %2319 = stablehlo.slice %1967 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2320 = stablehlo.reshape %2319 : (tensor<1xf64>) -> tensor<f64>
    %2321 = stablehlo.negate %2320 : tensor<f64>
    %2322 = stablehlo.reshape %2321 : (tensor<f64>) -> tensor<1xf64>
    %2323 = stablehlo.slice %1967 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2324 = stablehlo.reshape %2323 : (tensor<1xf64>) -> tensor<f64>
    %2325 = stablehlo.reshape %2324 : (tensor<f64>) -> tensor<1xf64>
    %2326 = stablehlo.concatenate %2314, %2318, %2322, %2325, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2327 = stablehlo.dot_general %1967, %1967, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2328 = stablehlo.broadcast_in_dim %2327, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2329 = stablehlo.divide %2326, %2328 : tensor<4xf64>
    %2330 = stablehlo.slice %2329 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2331 = stablehlo.reshape %2330 : (tensor<1xf64>) -> tensor<f64>
    %2332 = stablehlo.multiply %2310, %2331 : tensor<f64>
    %2333 = stablehlo.slice %2308 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2334 = stablehlo.reshape %2333 : (tensor<1xf64>) -> tensor<f64>
    %2335 = stablehlo.slice %2329 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2336 = stablehlo.reshape %2335 : (tensor<1xf64>) -> tensor<f64>
    %2337 = stablehlo.multiply %2334, %2336 : tensor<f64>
    %2338 = stablehlo.add %2332, %2337 : tensor<f64>
    %2339 = stablehlo.slice %2308 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2340 = stablehlo.reshape %2339 : (tensor<1xf64>) -> tensor<f64>
    %2341 = stablehlo.slice %2329 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2342 = stablehlo.reshape %2341 : (tensor<1xf64>) -> tensor<f64>
    %2343 = stablehlo.multiply %2340, %2342 : tensor<f64>
    %2344 = stablehlo.add %2338, %2343 : tensor<f64>
    %2345 = stablehlo.slice %2308 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2346 = stablehlo.reshape %2345 : (tensor<1xf64>) -> tensor<f64>
    %2347 = stablehlo.slice %2329 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2348 = stablehlo.reshape %2347 : (tensor<1xf64>) -> tensor<f64>
    %2349 = stablehlo.multiply %2346, %2348 : tensor<f64>
    %2350 = stablehlo.subtract %2344, %2349 : tensor<f64>
    %2351 = stablehlo.reshape %2350 : (tensor<f64>) -> tensor<1xf64>
    %2352 = stablehlo.multiply %2310, %2348 : tensor<f64>
    %2353 = stablehlo.multiply %2334, %2342 : tensor<f64>
    %2354 = stablehlo.subtract %2352, %2353 : tensor<f64>
    %2355 = stablehlo.multiply %2340, %2336 : tensor<f64>
    %2356 = stablehlo.add %2354, %2355 : tensor<f64>
    %2357 = stablehlo.multiply %2346, %2331 : tensor<f64>
    %2358 = stablehlo.add %2356, %2357 : tensor<f64>
    %2359 = stablehlo.reshape %2358 : (tensor<f64>) -> tensor<1xf64>
    %2360 = stablehlo.multiply %2310, %2342 : tensor<f64>
    %2361 = stablehlo.multiply %2334, %2348 : tensor<f64>
    %2362 = stablehlo.add %2360, %2361 : tensor<f64>
    %2363 = stablehlo.multiply %2340, %2331 : tensor<f64>
    %2364 = stablehlo.subtract %2362, %2363 : tensor<f64>
    %2365 = stablehlo.multiply %2346, %2336 : tensor<f64>
    %2366 = stablehlo.add %2364, %2365 : tensor<f64>
    %2367 = stablehlo.reshape %2366 : (tensor<f64>) -> tensor<1xf64>
    %2368 = stablehlo.multiply %2310, %2336 : tensor<f64>
    %2369 = stablehlo.multiply %2334, %2331 : tensor<f64>
    %2370 = stablehlo.subtract %2368, %2369 : tensor<f64>
    %2371 = stablehlo.multiply %2340, %2348 : tensor<f64>
    %2372 = stablehlo.subtract %2370, %2371 : tensor<f64>
    %2373 = stablehlo.multiply %2346, %2342 : tensor<f64>
    %2374 = stablehlo.subtract %2372, %2373 : tensor<f64>
    %2375 = stablehlo.reshape %2374 : (tensor<f64>) -> tensor<1xf64>
    %2376 = stablehlo.concatenate %2351, %2359, %2367, %2375, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2377 = stablehlo.slice %2376 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2378 = stablehlo.reshape %2377 : (tensor<1xf64>) -> tensor<f64>
    %2379 = stablehlo.reshape %2378 : (tensor<f64>) -> tensor<1xf64>
    %2380 = stablehlo.slice %2376 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2381 = stablehlo.reshape %2380 : (tensor<1xf64>) -> tensor<f64>
    %2382 = stablehlo.reshape %2381 : (tensor<f64>) -> tensor<1xf64>
    %2383 = stablehlo.slice %2376 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2384 = stablehlo.reshape %2383 : (tensor<1xf64>) -> tensor<f64>
    %2385 = stablehlo.reshape %2384 : (tensor<f64>) -> tensor<1xf64>
    %2386 = stablehlo.concatenate %2379, %2382, %2385, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2387 = stablehlo.slice %1967 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2388 = stablehlo.reshape %2387 : (tensor<1xf64>) -> tensor<f64>
    %2389 = stablehlo.slice %2258 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2390 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2391 = stablehlo.concatenate %2389, %2390, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2392 = stablehlo.slice %2391 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2393 = stablehlo.reshape %2392 : (tensor<1xf64>) -> tensor<f64>
    %2394 = stablehlo.multiply %2388, %2393 : tensor<f64>
    %2395 = stablehlo.slice %1967 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2396 = stablehlo.reshape %2395 : (tensor<1xf64>) -> tensor<f64>
    %2397 = stablehlo.slice %2391 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2398 = stablehlo.reshape %2397 : (tensor<1xf64>) -> tensor<f64>
    %2399 = stablehlo.multiply %2396, %2398 : tensor<f64>
    %2400 = stablehlo.add %2394, %2399 : tensor<f64>
    %2401 = stablehlo.slice %1967 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2402 = stablehlo.reshape %2401 : (tensor<1xf64>) -> tensor<f64>
    %2403 = stablehlo.slice %2391 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2404 = stablehlo.reshape %2403 : (tensor<1xf64>) -> tensor<f64>
    %2405 = stablehlo.multiply %2402, %2404 : tensor<f64>
    %2406 = stablehlo.add %2400, %2405 : tensor<f64>
    %2407 = stablehlo.slice %1967 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2408 = stablehlo.reshape %2407 : (tensor<1xf64>) -> tensor<f64>
    %2409 = stablehlo.slice %2391 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2410 = stablehlo.reshape %2409 : (tensor<1xf64>) -> tensor<f64>
    %2411 = stablehlo.multiply %2408, %2410 : tensor<f64>
    %2412 = stablehlo.subtract %2406, %2411 : tensor<f64>
    %2413 = stablehlo.reshape %2412 : (tensor<f64>) -> tensor<1xf64>
    %2414 = stablehlo.multiply %2388, %2410 : tensor<f64>
    %2415 = stablehlo.multiply %2396, %2404 : tensor<f64>
    %2416 = stablehlo.subtract %2414, %2415 : tensor<f64>
    %2417 = stablehlo.multiply %2402, %2398 : tensor<f64>
    %2418 = stablehlo.add %2416, %2417 : tensor<f64>
    %2419 = stablehlo.multiply %2408, %2393 : tensor<f64>
    %2420 = stablehlo.add %2418, %2419 : tensor<f64>
    %2421 = stablehlo.reshape %2420 : (tensor<f64>) -> tensor<1xf64>
    %2422 = stablehlo.multiply %2388, %2404 : tensor<f64>
    %2423 = stablehlo.multiply %2396, %2410 : tensor<f64>
    %2424 = stablehlo.add %2422, %2423 : tensor<f64>
    %2425 = stablehlo.multiply %2402, %2393 : tensor<f64>
    %2426 = stablehlo.subtract %2424, %2425 : tensor<f64>
    %2427 = stablehlo.multiply %2408, %2398 : tensor<f64>
    %2428 = stablehlo.add %2426, %2427 : tensor<f64>
    %2429 = stablehlo.reshape %2428 : (tensor<f64>) -> tensor<1xf64>
    %2430 = stablehlo.multiply %2388, %2398 : tensor<f64>
    %2431 = stablehlo.multiply %2396, %2393 : tensor<f64>
    %2432 = stablehlo.subtract %2430, %2431 : tensor<f64>
    %2433 = stablehlo.multiply %2402, %2410 : tensor<f64>
    %2434 = stablehlo.subtract %2432, %2433 : tensor<f64>
    %2435 = stablehlo.multiply %2408, %2404 : tensor<f64>
    %2436 = stablehlo.subtract %2434, %2435 : tensor<f64>
    %2437 = stablehlo.reshape %2436 : (tensor<f64>) -> tensor<1xf64>
    %2438 = stablehlo.concatenate %2413, %2421, %2429, %2437, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2439 = stablehlo.slice %2438 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2440 = stablehlo.reshape %2439 : (tensor<1xf64>) -> tensor<f64>
    %2441 = stablehlo.slice %1967 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2442 = stablehlo.reshape %2441 : (tensor<1xf64>) -> tensor<f64>
    %2443 = stablehlo.negate %2442 : tensor<f64>
    %2444 = stablehlo.reshape %2443 : (tensor<f64>) -> tensor<1xf64>
    %2445 = stablehlo.slice %1967 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2446 = stablehlo.reshape %2445 : (tensor<1xf64>) -> tensor<f64>
    %2447 = stablehlo.negate %2446 : tensor<f64>
    %2448 = stablehlo.reshape %2447 : (tensor<f64>) -> tensor<1xf64>
    %2449 = stablehlo.slice %1967 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2450 = stablehlo.reshape %2449 : (tensor<1xf64>) -> tensor<f64>
    %2451 = stablehlo.negate %2450 : tensor<f64>
    %2452 = stablehlo.reshape %2451 : (tensor<f64>) -> tensor<1xf64>
    %2453 = stablehlo.slice %1967 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2454 = stablehlo.reshape %2453 : (tensor<1xf64>) -> tensor<f64>
    %2455 = stablehlo.reshape %2454 : (tensor<f64>) -> tensor<1xf64>
    %2456 = stablehlo.concatenate %2444, %2448, %2452, %2455, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2457 = stablehlo.dot_general %1967, %1967, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2458 = stablehlo.broadcast_in_dim %2457, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2459 = stablehlo.divide %2456, %2458 : tensor<4xf64>
    %2460 = stablehlo.slice %2459 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2461 = stablehlo.reshape %2460 : (tensor<1xf64>) -> tensor<f64>
    %2462 = stablehlo.multiply %2440, %2461 : tensor<f64>
    %2463 = stablehlo.slice %2438 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2464 = stablehlo.reshape %2463 : (tensor<1xf64>) -> tensor<f64>
    %2465 = stablehlo.slice %2459 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2466 = stablehlo.reshape %2465 : (tensor<1xf64>) -> tensor<f64>
    %2467 = stablehlo.multiply %2464, %2466 : tensor<f64>
    %2468 = stablehlo.add %2462, %2467 : tensor<f64>
    %2469 = stablehlo.slice %2438 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2470 = stablehlo.reshape %2469 : (tensor<1xf64>) -> tensor<f64>
    %2471 = stablehlo.slice %2459 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2472 = stablehlo.reshape %2471 : (tensor<1xf64>) -> tensor<f64>
    %2473 = stablehlo.multiply %2470, %2472 : tensor<f64>
    %2474 = stablehlo.add %2468, %2473 : tensor<f64>
    %2475 = stablehlo.slice %2438 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2476 = stablehlo.reshape %2475 : (tensor<1xf64>) -> tensor<f64>
    %2477 = stablehlo.slice %2459 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2478 = stablehlo.reshape %2477 : (tensor<1xf64>) -> tensor<f64>
    %2479 = stablehlo.multiply %2476, %2478 : tensor<f64>
    %2480 = stablehlo.subtract %2474, %2479 : tensor<f64>
    %2481 = stablehlo.reshape %2480 : (tensor<f64>) -> tensor<1xf64>
    %2482 = stablehlo.multiply %2440, %2478 : tensor<f64>
    %2483 = stablehlo.multiply %2464, %2472 : tensor<f64>
    %2484 = stablehlo.subtract %2482, %2483 : tensor<f64>
    %2485 = stablehlo.multiply %2470, %2466 : tensor<f64>
    %2486 = stablehlo.add %2484, %2485 : tensor<f64>
    %2487 = stablehlo.multiply %2476, %2461 : tensor<f64>
    %2488 = stablehlo.add %2486, %2487 : tensor<f64>
    %2489 = stablehlo.reshape %2488 : (tensor<f64>) -> tensor<1xf64>
    %2490 = stablehlo.multiply %2440, %2472 : tensor<f64>
    %2491 = stablehlo.multiply %2464, %2478 : tensor<f64>
    %2492 = stablehlo.add %2490, %2491 : tensor<f64>
    %2493 = stablehlo.multiply %2470, %2461 : tensor<f64>
    %2494 = stablehlo.subtract %2492, %2493 : tensor<f64>
    %2495 = stablehlo.multiply %2476, %2466 : tensor<f64>
    %2496 = stablehlo.add %2494, %2495 : tensor<f64>
    %2497 = stablehlo.reshape %2496 : (tensor<f64>) -> tensor<1xf64>
    %2498 = stablehlo.multiply %2440, %2466 : tensor<f64>
    %2499 = stablehlo.multiply %2464, %2461 : tensor<f64>
    %2500 = stablehlo.subtract %2498, %2499 : tensor<f64>
    %2501 = stablehlo.multiply %2470, %2478 : tensor<f64>
    %2502 = stablehlo.subtract %2500, %2501 : tensor<f64>
    %2503 = stablehlo.multiply %2476, %2472 : tensor<f64>
    %2504 = stablehlo.subtract %2502, %2503 : tensor<f64>
    %2505 = stablehlo.reshape %2504 : (tensor<f64>) -> tensor<1xf64>
    %2506 = stablehlo.concatenate %2481, %2489, %2497, %2505, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2507 = stablehlo.slice %2506 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2508 = stablehlo.reshape %2507 : (tensor<1xf64>) -> tensor<f64>
    %2509 = stablehlo.reshape %2508 : (tensor<f64>) -> tensor<1xf64>
    %2510 = stablehlo.slice %2506 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2511 = stablehlo.reshape %2510 : (tensor<1xf64>) -> tensor<f64>
    %2512 = stablehlo.reshape %2511 : (tensor<f64>) -> tensor<1xf64>
    %2513 = stablehlo.slice %2506 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2514 = stablehlo.reshape %2513 : (tensor<1xf64>) -> tensor<f64>
    %2515 = stablehlo.reshape %2514 : (tensor<f64>) -> tensor<1xf64>
    %2516 = stablehlo.concatenate %2509, %2512, %2515, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %2517 = stablehlo.concatenate %2386, %2516, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %2518 = stablehlo.reshape %arg22 : (tensor<f64>) -> tensor<f64>
    %cst_31 = stablehlo.constant dense<0.16666666666666666> : tensor<f64>
    %2519 = stablehlo.multiply %cst_31, %2518 : tensor<f64>
    %2520 = stablehlo.broadcast_in_dim %2519, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %cst_32 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2521 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2522 = stablehlo.multiply %2521, %1265 : tensor<6xf64>
    %2523 = stablehlo.add %639, %2522 : tensor<6xf64>
    %cst_33 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2524 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2525 = stablehlo.multiply %2524, %1891 : tensor<6xf64>
    %2526 = stablehlo.add %2523, %2525 : tensor<6xf64>
    %2527 = stablehlo.add %2526, %2517 : tensor<6xf64>
    %2528 = stablehlo.multiply %2520, %2527 : tensor<6xf64>
    %2529 = stablehlo.add %arg2, %2528 : tensor<6xf64>
    %2530 = stablehlo.slice %arg1 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %2531 = stablehlo.broadcast_in_dim %2519, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %cst_34 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2532 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2533 = stablehlo.multiply %2532, %710 : tensor<6xf64>
    %2534 = stablehlo.add %84, %2533 : tensor<6xf64>
    %cst_35 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2535 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f64>) -> tensor<6xf64>
    %2536 = stablehlo.multiply %2535, %1336 : tensor<6xf64>
    %2537 = stablehlo.add %2534, %2536 : tensor<6xf64>
    %2538 = stablehlo.add %2537, %1962 : tensor<6xf64>
    %2539 = stablehlo.multiply %2531, %2538 : tensor<6xf64>
    %2540 = stablehlo.slice %2539 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_36 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2541 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2542 = stablehlo.divide %2540, %2541 : tensor<3xf64>
    %cst_37 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2543 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %2544 = stablehlo.concatenate %2542, %2543, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2545 = stablehlo.slice %2544 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2546 = stablehlo.reshape %2545 : (tensor<1xf64>) -> tensor<f64>
    %2547 = stablehlo.slice %2530 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2548 = stablehlo.reshape %2547 : (tensor<1xf64>) -> tensor<f64>
    %2549 = stablehlo.multiply %2546, %2548 : tensor<f64>
    %2550 = stablehlo.slice %2544 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %2551 = stablehlo.reshape %2550 : (tensor<1xf64>) -> tensor<f64>
    %2552 = stablehlo.slice %2530 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2553 = stablehlo.reshape %2552 : (tensor<1xf64>) -> tensor<f64>
    %2554 = stablehlo.multiply %2551, %2553 : tensor<f64>
    %2555 = stablehlo.add %2549, %2554 : tensor<f64>
    %2556 = stablehlo.slice %2544 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2557 = stablehlo.reshape %2556 : (tensor<1xf64>) -> tensor<f64>
    %2558 = stablehlo.slice %2530 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2559 = stablehlo.reshape %2558 : (tensor<1xf64>) -> tensor<f64>
    %2560 = stablehlo.multiply %2557, %2559 : tensor<f64>
    %2561 = stablehlo.add %2555, %2560 : tensor<f64>
    %2562 = stablehlo.slice %2544 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %2563 = stablehlo.reshape %2562 : (tensor<1xf64>) -> tensor<f64>
    %2564 = stablehlo.slice %2530 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %2565 = stablehlo.reshape %2564 : (tensor<1xf64>) -> tensor<f64>
    %2566 = stablehlo.multiply %2563, %2565 : tensor<f64>
    %2567 = stablehlo.subtract %2561, %2566 : tensor<f64>
    %2568 = stablehlo.reshape %2567 : (tensor<f64>) -> tensor<1xf64>
    %2569 = stablehlo.multiply %2546, %2565 : tensor<f64>
    %2570 = stablehlo.multiply %2551, %2559 : tensor<f64>
    %2571 = stablehlo.subtract %2569, %2570 : tensor<f64>
    %2572 = stablehlo.multiply %2557, %2553 : tensor<f64>
    %2573 = stablehlo.add %2571, %2572 : tensor<f64>
    %2574 = stablehlo.multiply %2563, %2548 : tensor<f64>
    %2575 = stablehlo.add %2573, %2574 : tensor<f64>
    %2576 = stablehlo.reshape %2575 : (tensor<f64>) -> tensor<1xf64>
    %2577 = stablehlo.multiply %2546, %2559 : tensor<f64>
    %2578 = stablehlo.multiply %2551, %2565 : tensor<f64>
    %2579 = stablehlo.add %2577, %2578 : tensor<f64>
    %2580 = stablehlo.multiply %2557, %2548 : tensor<f64>
    %2581 = stablehlo.subtract %2579, %2580 : tensor<f64>
    %2582 = stablehlo.multiply %2563, %2553 : tensor<f64>
    %2583 = stablehlo.add %2581, %2582 : tensor<f64>
    %2584 = stablehlo.reshape %2583 : (tensor<f64>) -> tensor<1xf64>
    %2585 = stablehlo.multiply %2546, %2553 : tensor<f64>
    %2586 = stablehlo.multiply %2551, %2548 : tensor<f64>
    %2587 = stablehlo.subtract %2585, %2586 : tensor<f64>
    %2588 = stablehlo.multiply %2557, %2565 : tensor<f64>
    %2589 = stablehlo.subtract %2587, %2588 : tensor<f64>
    %2590 = stablehlo.multiply %2563, %2559 : tensor<f64>
    %2591 = stablehlo.subtract %2589, %2590 : tensor<f64>
    %2592 = stablehlo.reshape %2591 : (tensor<f64>) -> tensor<1xf64>
    %2593 = stablehlo.concatenate %2568, %2576, %2584, %2592, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %2594 = stablehlo.add %2530, %2593 : tensor<4xf64>
    %2595 = stablehlo.dot_general %2594, %2594, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %2596 = stablehlo.sqrt %2595 : tensor<f64>
    %2597 = stablehlo.broadcast_in_dim %2596, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %2598 = stablehlo.divide %2594, %2597 : tensor<4xf64>
    %2599 = stablehlo.slice %arg1 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %2600 = stablehlo.slice %2539 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %2601 = stablehlo.add %2599, %2600 : tensor<3xf64>
    %2602 = stablehlo.concatenate %2598, %2601, dim = 0 : (tensor<4xf64>, tensor<3xf64>) -> tensor<7xf64>
    return %2517, %6, %1#1, %arg22, %arg8, %9, %0, %1#0, %arg3, %8, %12, %arg15, %arg23, %5, %2529, %13, %4, %3, %2602, %arg26, %arg20, %1966, %arg18, %11, %2, %10, %7 : tensor<6xf64>, tensor<480x3xf64>, tensor<f64>, tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>, tensor<6xf64>, tensor<3xf64>, tensor<f64>, tensor<3xf64>, tensor<6xf64>, tensor<f64>, tensor<2xf64>, tensor<f64>, tensor<7xf64>, tensor<7xf64>, tensor<f64>, tensor<6xf64>, tensor<f64>, tensor<6xf64>, tensor<3xf64>, tensor<f64>, tensor<3xf64>
  }
  func.func private @inner(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>, %arg2: tensor<3xf64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %cst = stablehlo.constant dense<[0.000000e+00, 1.100000e+04, 2.000000e+04, 3.200000e+04, 4.700000e+04, 5.100000e+04, 7.100000e+04, 8.485200e+04]> : tensor<8xf64>
    %cst_0 = stablehlo.constant dense<[1.500000e+01, -5.650000e+01, -5.650000e+01, -4.450000e+01, -2.500000e+00, -2.500000e+00, -5.850000e+01, -8.620000e+01]> : tensor<8xf64>
    %cst_1 = stablehlo.constant dense<[1.225000e+00, 3.639000e-01, 0.087999999999999994, 1.320000e-02, 1.400000e-03, 9.000000e-04, 1.000000e-04, 0.000000e+00]> : tensor<8xf64>
    %0 = stablehlo.slice %arg0 [4:7] : (tensor<7xf64>) -> tensor<3xf64>
    %1 = stablehlo.slice %0 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
    %3 = call @_interp(%2, %cst, %cst_0) : (tensor<f64>, tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %cst_2 = stablehlo.constant dense<2.731500e+02> : tensor<f64>
    %4 = stablehlo.add %3, %cst_2 : tensor<f64>
    %5 = call @_interp(%2, %cst, %cst_1) : (tensor<f64>, tensor<8xf64>, tensor<8xf64>) -> tensor<f64>
    %cst_3 = stablehlo.constant dense<4.018700e+02> : tensor<f64>
    %6 = stablehlo.multiply %cst_3, %4 : tensor<f64>
    %7 = stablehlo.sqrt %6 : tensor<f64>
    %8 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %9 = stablehlo.subtract %8, %arg2 : tensor<3xf64>
    %10 = call @norm(%9) : (tensor<3xf64>) -> tensor<f64>
    %11 = stablehlo.divide %10, %7 : tensor<f64>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %12 = stablehlo.multiply %cst_4, %5 : tensor<f64>
    %13 = stablehlo.multiply %10, %10 : tensor<f64>
    %14 = stablehlo.multiply %12, %13 : tensor<f64>
    %cst_5 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %15 = call @clip_21(%14, %cst_5) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    return %11, %15 : tensor<f64>, tensor<f64>
  }
  func.func private @_interp(%arg0: tensor<f64>, %arg1: tensor<8xf64>, %arg2: tensor<8xf64>) -> tensor<f64> {
    %0 = call @searchsorted(%arg1, %arg0) : (tensor<8xf64>, tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<7> : tensor<i64>
    %1 = call @clip(%0, %c, %c_0) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.compare  LT, %1, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<8> : tensor<i32>
    %3 = stablehlo.add %1, %c_2 : tensor<i32>
    %4 = stablehlo.select %2, %3, %1 : tensor<i1>, tensor<i32>
    %5 = stablehlo.dynamic_slice %arg2, %4, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %7 = stablehlo.subtract %1, %c_3 : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %7, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<8> : tensor<i32>
    %9 = stablehlo.add %7, %c_5 : tensor<i32>
    %10 = stablehlo.select %8, %9, %7 : tensor<i1>, tensor<i32>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
    %13 = stablehlo.subtract %6, %12 : tensor<f64>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %14 = stablehlo.compare  LT, %1, %c_6,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_7 = stablehlo.constant dense<8> : tensor<i32>
    %15 = stablehlo.add %1, %c_7 : tensor<i32>
    %16 = stablehlo.select %14, %15, %1 : tensor<i1>, tensor<i32>
    %17 = stablehlo.dynamic_slice %arg1, %16, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %18 = stablehlo.reshape %17 : (tensor<1xf64>) -> tensor<f64>
    %c_8 = stablehlo.constant dense<1> : tensor<i32>
    %19 = stablehlo.subtract %1, %c_8 : tensor<i32>
    %c_9 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.compare  LT, %19, %c_9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_10 = stablehlo.constant dense<8> : tensor<i32>
    %21 = stablehlo.add %19, %c_10 : tensor<i32>
    %22 = stablehlo.select %20, %21, %19 : tensor<i1>, tensor<i32>
    %23 = stablehlo.dynamic_slice %arg1, %22, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %24 = stablehlo.reshape %23 : (tensor<1xf64>) -> tensor<f64>
    %25 = stablehlo.subtract %18, %24 : tensor<f64>
    %c_11 = stablehlo.constant dense<1> : tensor<i32>
    %26 = stablehlo.subtract %1, %c_11 : tensor<i32>
    %c_12 = stablehlo.constant dense<0> : tensor<i32>
    %27 = stablehlo.compare  LT, %26, %c_12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_13 = stablehlo.constant dense<8> : tensor<i32>
    %28 = stablehlo.add %26, %c_13 : tensor<i32>
    %29 = stablehlo.select %27, %28, %26 : tensor<i1>, tensor<i32>
    %30 = stablehlo.dynamic_slice %arg1, %29, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.subtract %arg0, %31 : tensor<f64>
    %33 = stablehlo.abs %25 : tensor<f64>
    %cst = stablehlo.constant dense<4.9303806576313238E-32> : tensor<f64>
    %34 = stablehlo.compare  LE, %33, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %c_14 = stablehlo.constant dense<1> : tensor<i32>
    %35 = stablehlo.subtract %1, %c_14 : tensor<i32>
    %c_15 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.compare  LT, %35, %c_15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_16 = stablehlo.constant dense<8> : tensor<i32>
    %37 = stablehlo.add %35, %c_16 : tensor<i32>
    %38 = stablehlo.select %36, %37, %35 : tensor<i1>, tensor<i32>
    %39 = stablehlo.dynamic_slice %arg2, %38, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %40 = stablehlo.reshape %39 : (tensor<1xf64>) -> tensor<f64>
    %c_17 = stablehlo.constant dense<1> : tensor<i32>
    %41 = stablehlo.subtract %1, %c_17 : tensor<i32>
    %c_18 = stablehlo.constant dense<0> : tensor<i32>
    %42 = stablehlo.compare  LT, %41, %c_18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_19 = stablehlo.constant dense<8> : tensor<i32>
    %43 = stablehlo.add %41, %c_19 : tensor<i32>
    %44 = stablehlo.select %42, %43, %41 : tensor<i1>, tensor<i32>
    %45 = stablehlo.dynamic_slice %arg2, %44, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %46 = stablehlo.reshape %45 : (tensor<1xf64>) -> tensor<f64>
    %c_20 = stablehlo.constant dense<1> : tensor<i64>
    %47 = call @_where_9(%34, %c_20, %25) : (tensor<i1>, tensor<i64>, tensor<f64>) -> tensor<f64>
    %48 = stablehlo.divide %32, %47 : tensor<f64>
    %49 = stablehlo.multiply %48, %13 : tensor<f64>
    %50 = stablehlo.add %46, %49 : tensor<f64>
    %51 = call @_where_13(%34, %40, %50) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %52 = stablehlo.slice %arg2 [0:1] : (tensor<8xf64>) -> tensor<1xf64>
    %53 = stablehlo.reshape %52 : (tensor<1xf64>) -> tensor<f64>
    %54 = stablehlo.slice %arg1 [0:1] : (tensor<8xf64>) -> tensor<1xf64>
    %55 = stablehlo.reshape %54 : (tensor<1xf64>) -> tensor<f64>
    %56 = stablehlo.compare  LT, %arg0, %55,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %57 = call @_where_13(%56, %53, %51) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %c_21 = stablehlo.constant dense<7> : tensor<i64>
    %58 = stablehlo.dynamic_slice %arg2, %c_21, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %59 = stablehlo.reshape %58 : (tensor<1xf64>) -> tensor<f64>
    %c_22 = stablehlo.constant dense<7> : tensor<i64>
    %60 = stablehlo.dynamic_slice %arg1, %c_22, sizes = [1] : (tensor<8xf64>, tensor<i64>) -> tensor<1xf64>
    %61 = stablehlo.reshape %60 : (tensor<1xf64>) -> tensor<f64>
    %62 = stablehlo.compare  GT, %arg0, %61,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %63 = call @_where_13(%62, %59, %57) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    return %63 : tensor<f64>
  }
  func.func private @searchsorted(%arg0: tensor<8xf64>, %arg1: tensor<f64>) -> tensor<i32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<8> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:5 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %c_1, %iterArg_4 = %c, %iterArg_5 = %c_0) : tensor<8xf64>, tensor<f64>, tensor<i64>, tensor<i32>, tensor<i32>
    cond {
      %c_6 = stablehlo.constant dense<4> : tensor<i64>
      %1 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1:2 = func.call @closed_call(%iterArg, %iterArg_2, %iterArg_4, %iterArg_5) : (tensor<8xf64>, tensor<f64>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %2 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_2, %2, %1#0, %1#1 : tensor<8xf64>, tensor<f64>, tensor<i64>, tensor<i32>, tensor<i32>
    }
    return %0#4 : tensor<i32>
  }
  func.func private @closed_call(%arg0: tensor<8xf64>, %arg1: tensor<f64>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<ui32>
    %1 = stablehlo.convert %arg3 : (tensor<i32>) -> tensor<ui32>
    %2 = stablehlo.add %0, %1 : tensor<ui32>
    %c = stablehlo.constant dense<2> : tensor<ui32>
    %3 = stablehlo.divide %2, %c : tensor<ui32>
    %4 = stablehlo.convert %3 : (tensor<ui32>) -> tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.compare  LT, %4, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_1 = stablehlo.constant dense<8> : tensor<i32>
    %6 = stablehlo.add %4, %c_1 : tensor<i32>
    %7 = stablehlo.select %5, %6, %4 : tensor<i1>, tensor<i32>
    %8 = stablehlo.dynamic_slice %arg0, %7, sizes = [1] : (tensor<8xf64>, tensor<i32>) -> tensor<1xf64>
    %9 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %10 = stablehlo.compare  EQ, %arg1, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %11 = stablehlo.select %10, %cst_2, %arg1 : tensor<i1>, tensor<f64>
    %12 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %13 = stablehlo.select %12, %cst_3, %11 : tensor<i1>, tensor<f64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.compare  EQ, %9, %cst_4,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %15 = stablehlo.select %14, %cst_5, %9 : tensor<i1>, tensor<f64>
    %16 = stablehlo.compare  NE, %9, %9,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_6 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %17 = stablehlo.select %16, %cst_6, %15 : tensor<i1>, tensor<f64>
    %18 = stablehlo.compare  LT, %13, %17,  TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %19 = call @_where(%18, %arg2, %4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %20 = call @_where(%18, %4, %arg3) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    return %19, %20 : tensor<i32>, tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @clip(%arg0: tensor<i32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i32> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.maximum %0, %arg0 : tensor<i32>
    %2 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.minimum %2, %1 : tensor<i32>
    return %3 : tensor<i32>
  }
  func.func private @_where_9(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<i1>, tensor<f64>
    return %1 : tensor<f64>
  }
  func.func private @_where_13(%arg0: tensor<i1>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @norm(%arg0: tensor<3xf64>) -> tensor<f64> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<3xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.sqrt %1 : tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @clip_21(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f64>
    return %1 : tensor<f64>
  }
  func.func private @inner_24(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>, %arg2: tensor<3xf64>, %arg3: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.subtract %0, %arg2 : tensor<3xf64>
    %2 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %3 = stablehlo.slice %2 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
    %5 = stablehlo.negate %4 : tensor<f64>
    %6 = stablehlo.reshape %5 : (tensor<f64>) -> tensor<1xf64>
    %7 = stablehlo.slice %2 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %8 = stablehlo.reshape %7 : (tensor<1xf64>) -> tensor<f64>
    %9 = stablehlo.negate %8 : tensor<f64>
    %10 = stablehlo.reshape %9 : (tensor<f64>) -> tensor<1xf64>
    %11 = stablehlo.slice %2 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
    %13 = stablehlo.negate %12 : tensor<f64>
    %14 = stablehlo.reshape %13 : (tensor<f64>) -> tensor<1xf64>
    %15 = stablehlo.slice %2 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %16 = stablehlo.reshape %15 : (tensor<1xf64>) -> tensor<f64>
    %17 = stablehlo.reshape %16 : (tensor<f64>) -> tensor<1xf64>
    %18 = stablehlo.concatenate %6, %10, %14, %17, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %19 = stablehlo.dot_general %2, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %21 = stablehlo.divide %18, %20 : tensor<4xf64>
    %22 = stablehlo.slice %21 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %24 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %25 = stablehlo.concatenate %1, %24, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %26 = stablehlo.slice %25 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
    %28 = stablehlo.multiply %23, %27 : tensor<f64>
    %29 = stablehlo.slice %21 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.slice %25 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %32 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
    %33 = stablehlo.multiply %30, %32 : tensor<f64>
    %34 = stablehlo.add %28, %33 : tensor<f64>
    %35 = stablehlo.slice %21 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1xf64>) -> tensor<f64>
    %37 = stablehlo.slice %25 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %38 = stablehlo.reshape %37 : (tensor<1xf64>) -> tensor<f64>
    %39 = stablehlo.multiply %36, %38 : tensor<f64>
    %40 = stablehlo.add %34, %39 : tensor<f64>
    %41 = stablehlo.slice %21 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.slice %25 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %44 = stablehlo.reshape %43 : (tensor<1xf64>) -> tensor<f64>
    %45 = stablehlo.multiply %42, %44 : tensor<f64>
    %46 = stablehlo.subtract %40, %45 : tensor<f64>
    %47 = stablehlo.reshape %46 : (tensor<f64>) -> tensor<1xf64>
    %48 = stablehlo.multiply %23, %44 : tensor<f64>
    %49 = stablehlo.multiply %30, %38 : tensor<f64>
    %50 = stablehlo.subtract %48, %49 : tensor<f64>
    %51 = stablehlo.multiply %36, %32 : tensor<f64>
    %52 = stablehlo.add %50, %51 : tensor<f64>
    %53 = stablehlo.multiply %42, %27 : tensor<f64>
    %54 = stablehlo.add %52, %53 : tensor<f64>
    %55 = stablehlo.reshape %54 : (tensor<f64>) -> tensor<1xf64>
    %56 = stablehlo.multiply %23, %38 : tensor<f64>
    %57 = stablehlo.multiply %30, %44 : tensor<f64>
    %58 = stablehlo.add %56, %57 : tensor<f64>
    %59 = stablehlo.multiply %36, %27 : tensor<f64>
    %60 = stablehlo.subtract %58, %59 : tensor<f64>
    %61 = stablehlo.multiply %42, %32 : tensor<f64>
    %62 = stablehlo.add %60, %61 : tensor<f64>
    %63 = stablehlo.reshape %62 : (tensor<f64>) -> tensor<1xf64>
    %64 = stablehlo.multiply %23, %32 : tensor<f64>
    %65 = stablehlo.multiply %30, %27 : tensor<f64>
    %66 = stablehlo.subtract %64, %65 : tensor<f64>
    %67 = stablehlo.multiply %36, %44 : tensor<f64>
    %68 = stablehlo.subtract %66, %67 : tensor<f64>
    %69 = stablehlo.multiply %42, %38 : tensor<f64>
    %70 = stablehlo.subtract %68, %69 : tensor<f64>
    %71 = stablehlo.reshape %70 : (tensor<f64>) -> tensor<1xf64>
    %72 = stablehlo.concatenate %47, %55, %63, %71, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %73 = stablehlo.slice %72 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.slice %21 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %76 = stablehlo.reshape %75 : (tensor<1xf64>) -> tensor<f64>
    %77 = stablehlo.negate %76 : tensor<f64>
    %78 = stablehlo.reshape %77 : (tensor<f64>) -> tensor<1xf64>
    %79 = stablehlo.slice %21 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %80 = stablehlo.reshape %79 : (tensor<1xf64>) -> tensor<f64>
    %81 = stablehlo.negate %80 : tensor<f64>
    %82 = stablehlo.reshape %81 : (tensor<f64>) -> tensor<1xf64>
    %83 = stablehlo.slice %21 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %84 = stablehlo.reshape %83 : (tensor<1xf64>) -> tensor<f64>
    %85 = stablehlo.negate %84 : tensor<f64>
    %86 = stablehlo.reshape %85 : (tensor<f64>) -> tensor<1xf64>
    %87 = stablehlo.slice %21 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %88 = stablehlo.reshape %87 : (tensor<1xf64>) -> tensor<f64>
    %89 = stablehlo.reshape %88 : (tensor<f64>) -> tensor<1xf64>
    %90 = stablehlo.concatenate %78, %82, %86, %89, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %91 = stablehlo.dot_general %21, %21, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %92 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %93 = stablehlo.divide %90, %92 : tensor<4xf64>
    %94 = stablehlo.slice %93 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %95 = stablehlo.reshape %94 : (tensor<1xf64>) -> tensor<f64>
    %96 = stablehlo.multiply %74, %95 : tensor<f64>
    %97 = stablehlo.slice %72 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.slice %93 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %100 = stablehlo.reshape %99 : (tensor<1xf64>) -> tensor<f64>
    %101 = stablehlo.multiply %98, %100 : tensor<f64>
    %102 = stablehlo.add %96, %101 : tensor<f64>
    %103 = stablehlo.slice %72 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
    %105 = stablehlo.slice %93 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %106 = stablehlo.reshape %105 : (tensor<1xf64>) -> tensor<f64>
    %107 = stablehlo.multiply %104, %106 : tensor<f64>
    %108 = stablehlo.add %102, %107 : tensor<f64>
    %109 = stablehlo.slice %72 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %110 = stablehlo.reshape %109 : (tensor<1xf64>) -> tensor<f64>
    %111 = stablehlo.slice %93 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %112 = stablehlo.reshape %111 : (tensor<1xf64>) -> tensor<f64>
    %113 = stablehlo.multiply %110, %112 : tensor<f64>
    %114 = stablehlo.subtract %108, %113 : tensor<f64>
    %115 = stablehlo.reshape %114 : (tensor<f64>) -> tensor<1xf64>
    %116 = stablehlo.multiply %74, %112 : tensor<f64>
    %117 = stablehlo.multiply %98, %106 : tensor<f64>
    %118 = stablehlo.subtract %116, %117 : tensor<f64>
    %119 = stablehlo.multiply %104, %100 : tensor<f64>
    %120 = stablehlo.add %118, %119 : tensor<f64>
    %121 = stablehlo.multiply %110, %95 : tensor<f64>
    %122 = stablehlo.add %120, %121 : tensor<f64>
    %123 = stablehlo.reshape %122 : (tensor<f64>) -> tensor<1xf64>
    %124 = stablehlo.multiply %74, %106 : tensor<f64>
    %125 = stablehlo.multiply %98, %112 : tensor<f64>
    %126 = stablehlo.add %124, %125 : tensor<f64>
    %127 = stablehlo.multiply %104, %95 : tensor<f64>
    %128 = stablehlo.subtract %126, %127 : tensor<f64>
    %129 = stablehlo.multiply %110, %100 : tensor<f64>
    %130 = stablehlo.add %128, %129 : tensor<f64>
    %131 = stablehlo.reshape %130 : (tensor<f64>) -> tensor<1xf64>
    %132 = stablehlo.multiply %74, %100 : tensor<f64>
    %133 = stablehlo.multiply %98, %95 : tensor<f64>
    %134 = stablehlo.subtract %132, %133 : tensor<f64>
    %135 = stablehlo.multiply %104, %112 : tensor<f64>
    %136 = stablehlo.subtract %134, %135 : tensor<f64>
    %137 = stablehlo.multiply %110, %106 : tensor<f64>
    %138 = stablehlo.subtract %136, %137 : tensor<f64>
    %139 = stablehlo.reshape %138 : (tensor<f64>) -> tensor<1xf64>
    %140 = stablehlo.concatenate %115, %123, %131, %139, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %141 = stablehlo.slice %140 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %142 = stablehlo.reshape %141 : (tensor<1xf64>) -> tensor<f64>
    %143 = stablehlo.reshape %142 : (tensor<f64>) -> tensor<1xf64>
    %144 = stablehlo.slice %140 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.reshape %145 : (tensor<f64>) -> tensor<1xf64>
    %147 = stablehlo.slice %140 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %148 = stablehlo.reshape %147 : (tensor<1xf64>) -> tensor<f64>
    %149 = stablehlo.reshape %148 : (tensor<f64>) -> tensor<1xf64>
    %150 = stablehlo.concatenate %143, %146, %149, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %150 : tensor<3xf64>
  }
  func.func private @inner_36(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>, %arg2: tensor<3xf64>, %arg3: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<[-1.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = stablehlo.subtract %0, %arg2 : tensor<3xf64>
    %2 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %3 = stablehlo.slice %2 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
    %5 = stablehlo.negate %4 : tensor<f64>
    %6 = stablehlo.reshape %5 : (tensor<f64>) -> tensor<1xf64>
    %7 = stablehlo.slice %2 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %8 = stablehlo.reshape %7 : (tensor<1xf64>) -> tensor<f64>
    %9 = stablehlo.negate %8 : tensor<f64>
    %10 = stablehlo.reshape %9 : (tensor<f64>) -> tensor<1xf64>
    %11 = stablehlo.slice %2 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
    %13 = stablehlo.negate %12 : tensor<f64>
    %14 = stablehlo.reshape %13 : (tensor<f64>) -> tensor<1xf64>
    %15 = stablehlo.slice %2 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %16 = stablehlo.reshape %15 : (tensor<1xf64>) -> tensor<f64>
    %17 = stablehlo.reshape %16 : (tensor<f64>) -> tensor<1xf64>
    %18 = stablehlo.concatenate %6, %10, %14, %17, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %19 = stablehlo.dot_general %2, %2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %21 = stablehlo.divide %18, %20 : tensor<4xf64>
    %22 = stablehlo.slice %21 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %24 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %25 = stablehlo.concatenate %1, %24, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %26 = stablehlo.slice %25 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %27 = stablehlo.reshape %26 : (tensor<1xf64>) -> tensor<f64>
    %28 = stablehlo.multiply %23, %27 : tensor<f64>
    %29 = stablehlo.slice %21 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %30 = stablehlo.reshape %29 : (tensor<1xf64>) -> tensor<f64>
    %31 = stablehlo.slice %25 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %32 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
    %33 = stablehlo.multiply %30, %32 : tensor<f64>
    %34 = stablehlo.add %28, %33 : tensor<f64>
    %35 = stablehlo.slice %21 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %36 = stablehlo.reshape %35 : (tensor<1xf64>) -> tensor<f64>
    %37 = stablehlo.slice %25 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %38 = stablehlo.reshape %37 : (tensor<1xf64>) -> tensor<f64>
    %39 = stablehlo.multiply %36, %38 : tensor<f64>
    %40 = stablehlo.add %34, %39 : tensor<f64>
    %41 = stablehlo.slice %21 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.slice %25 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %44 = stablehlo.reshape %43 : (tensor<1xf64>) -> tensor<f64>
    %45 = stablehlo.multiply %42, %44 : tensor<f64>
    %46 = stablehlo.subtract %40, %45 : tensor<f64>
    %47 = stablehlo.reshape %46 : (tensor<f64>) -> tensor<1xf64>
    %48 = stablehlo.multiply %23, %44 : tensor<f64>
    %49 = stablehlo.multiply %30, %38 : tensor<f64>
    %50 = stablehlo.subtract %48, %49 : tensor<f64>
    %51 = stablehlo.multiply %36, %32 : tensor<f64>
    %52 = stablehlo.add %50, %51 : tensor<f64>
    %53 = stablehlo.multiply %42, %27 : tensor<f64>
    %54 = stablehlo.add %52, %53 : tensor<f64>
    %55 = stablehlo.reshape %54 : (tensor<f64>) -> tensor<1xf64>
    %56 = stablehlo.multiply %23, %38 : tensor<f64>
    %57 = stablehlo.multiply %30, %44 : tensor<f64>
    %58 = stablehlo.add %56, %57 : tensor<f64>
    %59 = stablehlo.multiply %36, %27 : tensor<f64>
    %60 = stablehlo.subtract %58, %59 : tensor<f64>
    %61 = stablehlo.multiply %42, %32 : tensor<f64>
    %62 = stablehlo.add %60, %61 : tensor<f64>
    %63 = stablehlo.reshape %62 : (tensor<f64>) -> tensor<1xf64>
    %64 = stablehlo.multiply %23, %32 : tensor<f64>
    %65 = stablehlo.multiply %30, %27 : tensor<f64>
    %66 = stablehlo.subtract %64, %65 : tensor<f64>
    %67 = stablehlo.multiply %36, %44 : tensor<f64>
    %68 = stablehlo.subtract %66, %67 : tensor<f64>
    %69 = stablehlo.multiply %42, %38 : tensor<f64>
    %70 = stablehlo.subtract %68, %69 : tensor<f64>
    %71 = stablehlo.reshape %70 : (tensor<f64>) -> tensor<1xf64>
    %72 = stablehlo.concatenate %47, %55, %63, %71, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %73 = stablehlo.slice %72 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.slice %21 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %76 = stablehlo.reshape %75 : (tensor<1xf64>) -> tensor<f64>
    %77 = stablehlo.negate %76 : tensor<f64>
    %78 = stablehlo.reshape %77 : (tensor<f64>) -> tensor<1xf64>
    %79 = stablehlo.slice %21 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %80 = stablehlo.reshape %79 : (tensor<1xf64>) -> tensor<f64>
    %81 = stablehlo.negate %80 : tensor<f64>
    %82 = stablehlo.reshape %81 : (tensor<f64>) -> tensor<1xf64>
    %83 = stablehlo.slice %21 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %84 = stablehlo.reshape %83 : (tensor<1xf64>) -> tensor<f64>
    %85 = stablehlo.negate %84 : tensor<f64>
    %86 = stablehlo.reshape %85 : (tensor<f64>) -> tensor<1xf64>
    %87 = stablehlo.slice %21 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %88 = stablehlo.reshape %87 : (tensor<1xf64>) -> tensor<f64>
    %89 = stablehlo.reshape %88 : (tensor<f64>) -> tensor<1xf64>
    %90 = stablehlo.concatenate %78, %82, %86, %89, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %91 = stablehlo.dot_general %21, %21, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %92 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %93 = stablehlo.divide %90, %92 : tensor<4xf64>
    %94 = stablehlo.slice %93 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %95 = stablehlo.reshape %94 : (tensor<1xf64>) -> tensor<f64>
    %96 = stablehlo.multiply %74, %95 : tensor<f64>
    %97 = stablehlo.slice %72 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %98 = stablehlo.reshape %97 : (tensor<1xf64>) -> tensor<f64>
    %99 = stablehlo.slice %93 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %100 = stablehlo.reshape %99 : (tensor<1xf64>) -> tensor<f64>
    %101 = stablehlo.multiply %98, %100 : tensor<f64>
    %102 = stablehlo.add %96, %101 : tensor<f64>
    %103 = stablehlo.slice %72 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
    %105 = stablehlo.slice %93 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %106 = stablehlo.reshape %105 : (tensor<1xf64>) -> tensor<f64>
    %107 = stablehlo.multiply %104, %106 : tensor<f64>
    %108 = stablehlo.add %102, %107 : tensor<f64>
    %109 = stablehlo.slice %72 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %110 = stablehlo.reshape %109 : (tensor<1xf64>) -> tensor<f64>
    %111 = stablehlo.slice %93 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %112 = stablehlo.reshape %111 : (tensor<1xf64>) -> tensor<f64>
    %113 = stablehlo.multiply %110, %112 : tensor<f64>
    %114 = stablehlo.subtract %108, %113 : tensor<f64>
    %115 = stablehlo.reshape %114 : (tensor<f64>) -> tensor<1xf64>
    %116 = stablehlo.multiply %74, %112 : tensor<f64>
    %117 = stablehlo.multiply %98, %106 : tensor<f64>
    %118 = stablehlo.subtract %116, %117 : tensor<f64>
    %119 = stablehlo.multiply %104, %100 : tensor<f64>
    %120 = stablehlo.add %118, %119 : tensor<f64>
    %121 = stablehlo.multiply %110, %95 : tensor<f64>
    %122 = stablehlo.add %120, %121 : tensor<f64>
    %123 = stablehlo.reshape %122 : (tensor<f64>) -> tensor<1xf64>
    %124 = stablehlo.multiply %74, %106 : tensor<f64>
    %125 = stablehlo.multiply %98, %112 : tensor<f64>
    %126 = stablehlo.add %124, %125 : tensor<f64>
    %127 = stablehlo.multiply %104, %95 : tensor<f64>
    %128 = stablehlo.subtract %126, %127 : tensor<f64>
    %129 = stablehlo.multiply %110, %100 : tensor<f64>
    %130 = stablehlo.add %128, %129 : tensor<f64>
    %131 = stablehlo.reshape %130 : (tensor<f64>) -> tensor<1xf64>
    %132 = stablehlo.multiply %74, %100 : tensor<f64>
    %133 = stablehlo.multiply %98, %95 : tensor<f64>
    %134 = stablehlo.subtract %132, %133 : tensor<f64>
    %135 = stablehlo.multiply %104, %112 : tensor<f64>
    %136 = stablehlo.subtract %134, %135 : tensor<f64>
    %137 = stablehlo.multiply %110, %106 : tensor<f64>
    %138 = stablehlo.subtract %136, %137 : tensor<f64>
    %139 = stablehlo.reshape %138 : (tensor<f64>) -> tensor<1xf64>
    %140 = stablehlo.concatenate %115, %123, %131, %139, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %141 = stablehlo.slice %140 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %142 = stablehlo.reshape %141 : (tensor<1xf64>) -> tensor<f64>
    %143 = stablehlo.reshape %142 : (tensor<f64>) -> tensor<1xf64>
    %144 = stablehlo.slice %140 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %145 = stablehlo.reshape %144 : (tensor<1xf64>) -> tensor<f64>
    %146 = stablehlo.reshape %145 : (tensor<f64>) -> tensor<1xf64>
    %147 = stablehlo.slice %140 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %148 = stablehlo.reshape %147 : (tensor<1xf64>) -> tensor<f64>
    %149 = stablehlo.reshape %148 : (tensor<f64>) -> tensor<1xf64>
    %150 = stablehlo.concatenate %143, %146, %149, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %151 = stablehlo.dot_general %150, %cst, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    %152 = call @norm(%150) : (tensor<3xf64>) -> tensor<f64>
    %cst_1 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %153 = call @clip_21(%152, %cst_1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %154 = stablehlo.divide %151, %153 : tensor<f64>
    %155 = chlo.acos %154 : tensor<f64> -> tensor<f64>
    %cst_2 = stablehlo.constant dense<57.295779513082323> : tensor<f64>
    %156 = stablehlo.multiply %155, %cst_2 : tensor<f64>
    %157 = stablehlo.slice %150 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %158 = stablehlo.reshape %157 : (tensor<1xf64>) -> tensor<f64>
    %159 = stablehlo.sign %158 : tensor<f64>
    %160 = stablehlo.negate %159 : tensor<f64>
    %161 = stablehlo.multiply %156, %160 : tensor<f64>
    return %161 : tensor<f64>
  }
  func.func private @inner_38(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<2xf64>
    %cst = stablehlo.constant dense<-0.0041666666666666666> : tensor<f64>
    %1 = stablehlo.exponential %cst : tensor<f64>
    %2 = stablehlo.convert %1 : tensor<f64>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %4 = stablehlo.multiply %0, %3 : tensor<2xf64>
    %5 = stablehlo.add %arg1, %4 : tensor<2xf64>
    return %5 : tensor<2xf64>
  }
  func.func private @inner_42(%arg0: tensor<6xf64>, %arg1: tensor<6xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<[-1.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %1 = call @norm(%0) : (tensor<3xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %2 = stablehlo.compare  LT, %1, %cst_0,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %3 = stablehlo.convert %2 : (tensor<i1>) -> tensor<i32>
    %4 = "stablehlo.case"(%3) ({
      %181 = stablehlo.slice %arg0 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
      stablehlo.return %181 : tensor<3xf64>
    }, {
      stablehlo.return %cst : tensor<3xf64>
    }) : (tensor<i32>) -> tensor<3xf64>
    %5 = call @norm(%cst) : (tensor<3xf64>) -> tensor<f64>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %7 = stablehlo.divide %cst, %6 : tensor<3xf64>
    %8 = call @norm(%4) : (tensor<3xf64>) -> tensor<f64>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %10 = stablehlo.divide %4, %9 : tensor<3xf64>
    %11 = call @cross(%7, %10) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %12 = stablehlo.dot_general %10, %10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    %13 = stablehlo.dot_general %7, %7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    %14 = stablehlo.multiply %12, %13 : tensor<f64>
    %15 = stablehlo.dot_general %7, %10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
    %16 = stablehlo.add %14, %15 : tensor<f64>
    %17 = stablehlo.slice %11 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %18 = stablehlo.reshape %17 : (tensor<1xf64>) -> tensor<f64>
    %19 = stablehlo.slice %11 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %20 = stablehlo.reshape %19 : (tensor<1xf64>) -> tensor<f64>
    %21 = stablehlo.slice %11 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %23 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %24 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %25 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %26 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %27 = stablehlo.concatenate %23, %24, %25, %26, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %28 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %29 = stablehlo.dot_general %27, %27, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %30 = stablehlo.sqrt %29 : tensor<f64>
    %31 = stablehlo.broadcast_in_dim %30, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %32 = stablehlo.divide %27, %31 : tensor<4xf64>
    %33 = stablehlo.slice %32 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %34 = stablehlo.reshape %33 : (tensor<1xf64>) -> tensor<f64>
    %35 = stablehlo.negate %34 : tensor<f64>
    %36 = stablehlo.reshape %35 : (tensor<f64>) -> tensor<1xf64>
    %37 = stablehlo.slice %32 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %38 = stablehlo.reshape %37 : (tensor<1xf64>) -> tensor<f64>
    %39 = stablehlo.negate %38 : tensor<f64>
    %40 = stablehlo.reshape %39 : (tensor<f64>) -> tensor<1xf64>
    %41 = stablehlo.slice %32 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.negate %42 : tensor<f64>
    %44 = stablehlo.reshape %43 : (tensor<f64>) -> tensor<1xf64>
    %45 = stablehlo.slice %32 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %46 = stablehlo.reshape %45 : (tensor<1xf64>) -> tensor<f64>
    %47 = stablehlo.reshape %46 : (tensor<f64>) -> tensor<1xf64>
    %48 = stablehlo.concatenate %36, %40, %44, %47, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %49 = stablehlo.dot_general %32, %32, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %50 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %51 = stablehlo.divide %48, %50 : tensor<4xf64>
    %52 = stablehlo.slice %51 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %53 = stablehlo.reshape %52 : (tensor<1xf64>) -> tensor<f64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %54 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %55 = stablehlo.concatenate %28, %54, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %56 = stablehlo.slice %55 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %57 = stablehlo.reshape %56 : (tensor<1xf64>) -> tensor<f64>
    %58 = stablehlo.multiply %53, %57 : tensor<f64>
    %59 = stablehlo.slice %51 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %60 = stablehlo.reshape %59 : (tensor<1xf64>) -> tensor<f64>
    %61 = stablehlo.slice %55 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %62 = stablehlo.reshape %61 : (tensor<1xf64>) -> tensor<f64>
    %63 = stablehlo.multiply %60, %62 : tensor<f64>
    %64 = stablehlo.add %58, %63 : tensor<f64>
    %65 = stablehlo.slice %51 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %66 = stablehlo.reshape %65 : (tensor<1xf64>) -> tensor<f64>
    %67 = stablehlo.slice %55 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %68 = stablehlo.reshape %67 : (tensor<1xf64>) -> tensor<f64>
    %69 = stablehlo.multiply %66, %68 : tensor<f64>
    %70 = stablehlo.add %64, %69 : tensor<f64>
    %71 = stablehlo.slice %51 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %72 = stablehlo.reshape %71 : (tensor<1xf64>) -> tensor<f64>
    %73 = stablehlo.slice %55 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.multiply %72, %74 : tensor<f64>
    %76 = stablehlo.subtract %70, %75 : tensor<f64>
    %77 = stablehlo.reshape %76 : (tensor<f64>) -> tensor<1xf64>
    %78 = stablehlo.multiply %53, %74 : tensor<f64>
    %79 = stablehlo.multiply %60, %68 : tensor<f64>
    %80 = stablehlo.subtract %78, %79 : tensor<f64>
    %81 = stablehlo.multiply %66, %62 : tensor<f64>
    %82 = stablehlo.add %80, %81 : tensor<f64>
    %83 = stablehlo.multiply %72, %57 : tensor<f64>
    %84 = stablehlo.add %82, %83 : tensor<f64>
    %85 = stablehlo.reshape %84 : (tensor<f64>) -> tensor<1xf64>
    %86 = stablehlo.multiply %53, %68 : tensor<f64>
    %87 = stablehlo.multiply %60, %74 : tensor<f64>
    %88 = stablehlo.add %86, %87 : tensor<f64>
    %89 = stablehlo.multiply %66, %57 : tensor<f64>
    %90 = stablehlo.subtract %88, %89 : tensor<f64>
    %91 = stablehlo.multiply %72, %62 : tensor<f64>
    %92 = stablehlo.add %90, %91 : tensor<f64>
    %93 = stablehlo.reshape %92 : (tensor<f64>) -> tensor<1xf64>
    %94 = stablehlo.multiply %53, %62 : tensor<f64>
    %95 = stablehlo.multiply %60, %57 : tensor<f64>
    %96 = stablehlo.subtract %94, %95 : tensor<f64>
    %97 = stablehlo.multiply %66, %74 : tensor<f64>
    %98 = stablehlo.subtract %96, %97 : tensor<f64>
    %99 = stablehlo.multiply %72, %68 : tensor<f64>
    %100 = stablehlo.subtract %98, %99 : tensor<f64>
    %101 = stablehlo.reshape %100 : (tensor<f64>) -> tensor<1xf64>
    %102 = stablehlo.concatenate %77, %85, %93, %101, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %103 = stablehlo.slice %102 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %104 = stablehlo.reshape %103 : (tensor<1xf64>) -> tensor<f64>
    %105 = stablehlo.slice %51 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %106 = stablehlo.reshape %105 : (tensor<1xf64>) -> tensor<f64>
    %107 = stablehlo.negate %106 : tensor<f64>
    %108 = stablehlo.reshape %107 : (tensor<f64>) -> tensor<1xf64>
    %109 = stablehlo.slice %51 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %110 = stablehlo.reshape %109 : (tensor<1xf64>) -> tensor<f64>
    %111 = stablehlo.negate %110 : tensor<f64>
    %112 = stablehlo.reshape %111 : (tensor<f64>) -> tensor<1xf64>
    %113 = stablehlo.slice %51 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %114 = stablehlo.reshape %113 : (tensor<1xf64>) -> tensor<f64>
    %115 = stablehlo.negate %114 : tensor<f64>
    %116 = stablehlo.reshape %115 : (tensor<f64>) -> tensor<1xf64>
    %117 = stablehlo.slice %51 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %118 = stablehlo.reshape %117 : (tensor<1xf64>) -> tensor<f64>
    %119 = stablehlo.reshape %118 : (tensor<f64>) -> tensor<1xf64>
    %120 = stablehlo.concatenate %108, %112, %116, %119, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %121 = stablehlo.dot_general %51, %51, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %122 = stablehlo.broadcast_in_dim %121, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %123 = stablehlo.divide %120, %122 : tensor<4xf64>
    %124 = stablehlo.slice %123 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %125 = stablehlo.reshape %124 : (tensor<1xf64>) -> tensor<f64>
    %126 = stablehlo.multiply %104, %125 : tensor<f64>
    %127 = stablehlo.slice %102 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %128 = stablehlo.reshape %127 : (tensor<1xf64>) -> tensor<f64>
    %129 = stablehlo.slice %123 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %130 = stablehlo.reshape %129 : (tensor<1xf64>) -> tensor<f64>
    %131 = stablehlo.multiply %128, %130 : tensor<f64>
    %132 = stablehlo.add %126, %131 : tensor<f64>
    %133 = stablehlo.slice %102 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %134 = stablehlo.reshape %133 : (tensor<1xf64>) -> tensor<f64>
    %135 = stablehlo.slice %123 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %136 = stablehlo.reshape %135 : (tensor<1xf64>) -> tensor<f64>
    %137 = stablehlo.multiply %134, %136 : tensor<f64>
    %138 = stablehlo.add %132, %137 : tensor<f64>
    %139 = stablehlo.slice %102 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.slice %123 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %142 = stablehlo.reshape %141 : (tensor<1xf64>) -> tensor<f64>
    %143 = stablehlo.multiply %140, %142 : tensor<f64>
    %144 = stablehlo.subtract %138, %143 : tensor<f64>
    %145 = stablehlo.reshape %144 : (tensor<f64>) -> tensor<1xf64>
    %146 = stablehlo.multiply %104, %142 : tensor<f64>
    %147 = stablehlo.multiply %128, %136 : tensor<f64>
    %148 = stablehlo.subtract %146, %147 : tensor<f64>
    %149 = stablehlo.multiply %134, %130 : tensor<f64>
    %150 = stablehlo.add %148, %149 : tensor<f64>
    %151 = stablehlo.multiply %140, %125 : tensor<f64>
    %152 = stablehlo.add %150, %151 : tensor<f64>
    %153 = stablehlo.reshape %152 : (tensor<f64>) -> tensor<1xf64>
    %154 = stablehlo.multiply %104, %136 : tensor<f64>
    %155 = stablehlo.multiply %128, %142 : tensor<f64>
    %156 = stablehlo.add %154, %155 : tensor<f64>
    %157 = stablehlo.multiply %134, %125 : tensor<f64>
    %158 = stablehlo.subtract %156, %157 : tensor<f64>
    %159 = stablehlo.multiply %140, %130 : tensor<f64>
    %160 = stablehlo.add %158, %159 : tensor<f64>
    %161 = stablehlo.reshape %160 : (tensor<f64>) -> tensor<1xf64>
    %162 = stablehlo.multiply %104, %130 : tensor<f64>
    %163 = stablehlo.multiply %128, %125 : tensor<f64>
    %164 = stablehlo.subtract %162, %163 : tensor<f64>
    %165 = stablehlo.multiply %134, %142 : tensor<f64>
    %166 = stablehlo.subtract %164, %165 : tensor<f64>
    %167 = stablehlo.multiply %140, %136 : tensor<f64>
    %168 = stablehlo.subtract %166, %167 : tensor<f64>
    %169 = stablehlo.reshape %168 : (tensor<f64>) -> tensor<1xf64>
    %170 = stablehlo.concatenate %145, %153, %161, %169, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %171 = stablehlo.slice %170 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %172 = stablehlo.reshape %171 : (tensor<1xf64>) -> tensor<f64>
    %173 = stablehlo.reshape %172 : (tensor<f64>) -> tensor<1xf64>
    %174 = stablehlo.slice %170 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %175 = stablehlo.reshape %174 : (tensor<1xf64>) -> tensor<f64>
    %176 = stablehlo.reshape %175 : (tensor<f64>) -> tensor<1xf64>
    %177 = stablehlo.slice %170 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %178 = stablehlo.reshape %177 : (tensor<1xf64>) -> tensor<f64>
    %179 = stablehlo.reshape %178 : (tensor<f64>) -> tensor<1xf64>
    %180 = stablehlo.concatenate %173, %176, %179, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %180 : tensor<3xf64>
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
  func.func private @inner_51(%arg0: tensor<3xf64>, %arg1: tensor<480x3xf64>) -> tensor<480x3xf64> {
    %0 = stablehlo.slice %arg1 [1:480, 0:3] : (tensor<480x3xf64>) -> tensor<479x3xf64>
    %1 = stablehlo.reshape %arg0 : (tensor<3xf64>) -> tensor<1x3xf64>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<479x3xf64>, tensor<1x3xf64>) -> tensor<480x3xf64>
    return %2 : tensor<480x3xf64>
  }
  func.func private @inner_55(%arg0: tensor<480x3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %cst = stablehlo.constant dense<0.026179938779914941> : tensor<f64>
    %0 = stablehlo.tan %cst : tensor<f64>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = stablehlo.divide %cst_0, %0 : tensor<f64>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %2 = stablehlo.sqrt %cst_1 : tensor<f64>
    %3 = stablehlo.multiply %2, %1 : tensor<f64>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %4 = stablehlo.add %cst_2, %3 : tensor<f64>
    %5 = stablehlo.multiply %1, %1 : tensor<f64>
    %6 = stablehlo.add %4, %5 : tensor<f64>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %7 = stablehlo.divide %cst_3, %6 : tensor<f64>
    %cst_4 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %8 = stablehlo.multiply %cst_4, %7 : tensor<f64>
    %9 = stablehlo.multiply %1, %1 : tensor<f64>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %10 = stablehlo.subtract %9, %cst_5 : tensor<f64>
    %cst_6 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %11 = stablehlo.multiply %cst_6, %10 : tensor<f64>
    %12 = stablehlo.multiply %11, %7 : tensor<f64>
    %13 = stablehlo.multiply %2, %1 : tensor<f64>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %14 = stablehlo.subtract %cst_7, %13 : tensor<f64>
    %15 = stablehlo.multiply %1, %1 : tensor<f64>
    %16 = stablehlo.add %14, %15 : tensor<f64>
    %17 = stablehlo.negate %16 : tensor<f64>
    %18 = stablehlo.multiply %17, %7 : tensor<f64>
    %19 = stablehlo.slice %arg0 [1:2, 0:3] : (tensor<480x3xf64>) -> tensor<1x3xf64>
    %20 = stablehlo.reshape %19 : (tensor<1x3xf64>) -> tensor<3xf64>
    %21 = stablehlo.slice %arg0 [0:1, 0:3] : (tensor<480x3xf64>) -> tensor<1x3xf64>
    %22 = stablehlo.reshape %21 : (tensor<1x3xf64>) -> tensor<3xf64>
    %23 = stablehlo.slice %arg0 [2:480, 0:3] : (tensor<480x3xf64>) -> tensor<478x3xf64>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %24 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f64>) -> tensor<478x3xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %25:11 = stablehlo.while(%iterArg = %23, %iterArg_15 = %7, %iterArg_16 = %8, %iterArg_17 = %12, %iterArg_18 = %18, %iterArg_19 = %c, %iterArg_20 = %20, %iterArg_21 = %22, %iterArg_22 = %20, %iterArg_23 = %22, %iterArg_24 = %24) : tensor<478x3xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<478x3xf64>
    cond {
      %c_25 = stablehlo.constant dense<478> : tensor<i64>
      %33 = stablehlo.compare  LT, %iterArg_19, %c_25,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %33 : tensor<i1>
    } do {
      %c_25 = stablehlo.constant dense<0> : tensor<i64>
      %33 = stablehlo.dynamic_slice %iterArg, %iterArg_19, %c_25, sizes = [1, 3] : (tensor<478x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
      %34 = stablehlo.reshape %33 : (tensor<1x3xf64>) -> tensor<3xf64>
      %35:5 = func.call @closed_call_71(%iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %iterArg_20, %iterArg_21, %iterArg_22, %iterArg_23, %34) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>)
      %36 = stablehlo.broadcast_in_dim %35#4, dims = [1] : (tensor<3xf64>) -> tensor<1x3xf64>
      %c_26 = stablehlo.constant dense<0> : tensor<i64>
      %37 = stablehlo.dynamic_update_slice %iterArg_24, %36, %iterArg_19, %c_26 : (tensor<478x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<478x3xf64>
      %c_27 = stablehlo.constant dense<1> : tensor<i64>
      %38 = stablehlo.add %iterArg_19, %c_27 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_15, %iterArg_16, %iterArg_17, %iterArg_18, %38, %35#0, %35#1, %35#2, %35#3, %37 : tensor<478x3xf64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<478x3xf64>
    }
    %26 = stablehlo.slice %25#10 [0:1, 0:3] : (tensor<478x3xf64>) -> tensor<1x3xf64>
    %27 = stablehlo.concatenate %26, %26, %25#10, dim = 0 : (tensor<1x3xf64>, tensor<1x3xf64>, tensor<478x3xf64>) -> tensor<480x3xf64>
    %c_9 = stablehlo.constant dense<-1> : tensor<i32>
    %c_10 = stablehlo.constant dense<0> : tensor<i32>
    %28 = stablehlo.compare  LT, %c_9, %c_10,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_11 = stablehlo.constant dense<-1> : tensor<i32>
    %c_12 = stablehlo.constant dense<480> : tensor<i32>
    %29 = stablehlo.add %c_11, %c_12 : tensor<i32>
    %c_13 = stablehlo.constant dense<-1> : tensor<i32>
    %30 = stablehlo.select %28, %29, %c_13 : tensor<i1>, tensor<i32>
    %c_14 = stablehlo.constant dense<0> : tensor<i32>
    %31 = stablehlo.dynamic_slice %27, %30, %c_14, sizes = [1, 3] : (tensor<480x3xf64>, tensor<i32>, tensor<i32>) -> tensor<1x3xf64>
    %32 = stablehlo.reshape %31 : (tensor<1x3xf64>) -> tensor<3xf64>
    return %32 : tensor<3xf64>
  }
  func.func private @closed_call_71(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<3xf64>, %arg5: tensor<3xf64>, %arg6: tensor<3xf64>, %arg7: tensor<3xf64>, %arg8: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>) {
    %0 = stablehlo.convert %arg0 : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %2 = stablehlo.multiply %1, %arg8 : tensor<3xf64>
    %3 = stablehlo.convert %arg1 : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %5 = stablehlo.multiply %4, %arg4 : tensor<3xf64>
    %6 = stablehlo.add %2, %5 : tensor<3xf64>
    %7 = stablehlo.convert %arg0 : tensor<f64>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %9 = stablehlo.multiply %8, %arg5 : tensor<3xf64>
    %10 = stablehlo.add %6, %9 : tensor<3xf64>
    %11 = stablehlo.convert %arg2 : tensor<f64>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %13 = stablehlo.multiply %12, %arg6 : tensor<3xf64>
    %14 = stablehlo.add %10, %13 : tensor<3xf64>
    %15 = stablehlo.convert %arg3 : tensor<f64>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %17 = stablehlo.multiply %16, %arg7 : tensor<3xf64>
    %18 = stablehlo.add %14, %17 : tensor<3xf64>
    return %arg8, %arg4, %18, %arg6, %18 : tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>, tensor<3xf64>
  }
  func.func private @inner_78(%arg0: tensor<2xf64>, %arg1: tensor<3xf64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.slice %arg1 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.subtract %1, %3 : tensor<f64>
    %5 = stablehlo.slice %arg2 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.0083333333333333332> : tensor<f64>
    %7 = stablehlo.multiply %4, %cst : tensor<f64>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %8 = stablehlo.multiply %7, %cst_0 : tensor<f64>
    %9 = stablehlo.add %6, %8 : tensor<f64>
    %cst_1 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %10 = call @clip_80(%9, %cst_1, %cst_2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %11 = stablehlo.slice %arg2 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
    %13 = stablehlo.subtract %4, %12 : tensor<f64>
    %14 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %15 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %16 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %17 = stablehlo.concatenate %14, %15, %16, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    return %17 : tensor<3xf64>
  }
  func.func private @clip_80(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.maximum %0, %arg0 : tensor<f64>
    %2 = stablehlo.convert %arg2 : tensor<f64>
    %3 = stablehlo.minimum %2, %1 : tensor<f64>
    return %3 : tensor<f64>
  }
  func.func private @inner_82(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<f64>) -> tensor<f64> {
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
    %12 = stablehlo.multiply %1, %7 : tensor<f64>
    %13 = stablehlo.multiply %3, %9 : tensor<f64>
    %14 = stablehlo.add %12, %13 : tensor<f64>
    %15 = stablehlo.multiply %5, %11 : tensor<f64>
    %16 = stablehlo.add %14, %15 : tensor<f64>
    %cst = stablehlo.constant dense<0.0083333333333333332> : tensor<f64>
    %17 = stablehlo.multiply %16, %cst : tensor<f64>
    return %17 : tensor<f64>
  }
  func.func private @inner_83(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %0 = stablehlo.add %cst, %arg2 : tensor<f64>
    %1 = stablehlo.divide %arg1, %0 : tensor<f64>
    %cst_0 = stablehlo.constant dense<-2.000000e-01> : tensor<f64>
    %cst_1 = stablehlo.constant dense<2.000000e-01> : tensor<f64>
    %2 = call @clip_80(%1, %cst_0, %cst_1) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %3 = stablehlo.add %arg0, %2 : tensor<f64>
    %cst_2 = stablehlo.constant dense<-4.000000e+01> : tensor<f64>
    %cst_3 = stablehlo.constant dense<4.000000e+01> : tensor<f64>
    %4 = call @clip_80(%3, %cst_2, %cst_3) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    return %4 : tensor<f64>
  }
  func.func private @inner_84(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<6xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<"0xB0726891EDFC17C0560E2DB29DEFF13F46B6F3FDD478F1BF1F85EB51B89E1BC0A69BC420B072F03F19E25817B7D1D8BFB81E85EB517820C0FCA9F1D24D62EE3F567DAEB6627FD93F295C8FC2F5A825C0029A081B9E5EEF3FA8C64B378941F23FC3F5285C8F4215C0E5D022DBF97EE43FA8C64B378941F2BFD578E926310818C07DD0B359F5B9E23F7B832F4CA60ADABF79E9263108AC17C0B515FBCBEEC9DB3F8FE4F21FD26FE73F4260E5D022DB16C02DB29DEFA7C6CB3F9CC420B0726801400000000000000000C0EC9E3C2CD4D23F0000000000000000355EBA490C02F53FDE02098A1F63D23F60E5D022DBF9F03FAE47E17A14AE0240BE30992A1895D03F8B6CE7FBA9F1024086C954C1A8A4DA3FA60A46257502CA3FC1CAA145B6F30B40C3F5285C8F421540E5D022DBF97EE43FA8C64B378941F23F894160E5D0220D409A081B9E5E29E33F60E5D022DBF9F83F37894160E5D0074052B81E85EB51E43F77BE9F1A2FDD034060E5D022DBF9F03F4ED1915CFE43E43F1F85EB51B81E0D40B0726891EDFC1740560E2DB29DEFF13F46B6F3FDD478F13FB0726891ED7C1540713D0AD7A370F33F39B4C876BE9FFC3FDD24068195C31040560E2DB29DEFF33F04560E2DB29D0440E9263108AC1CFE3F1283C0CAA145F43F713D0AD7A3700D40FA7E6ABC74131DC0AC1C5A643BDFF33F75931804560EF3BFE5D022DBF9BE20C0FED478E92631F23FFE65F7E461A1DABFE5D022DBF9BE23C037894160E5D0F03FAC1C5A643BDFDB3F5C8FC2F528DC29C085EB51B81E85F13FD578E9263108F43F986E1283C04A19C0D95F764F1E16E63FFCA9F1D24D62F4BF85EB51B81E051DC0BBB88D06F016E43FBADA8AFD65F7DCBF8FC2F5285C8F1CC03EE8D9ACFA5CDD3F13F241CF66D5E93FFCA9F1D24D621BC0A323B9FC87F4CB3FDD240681954303400000000000000000F46C567DAEB6D23F00000000000000002DB29DEFA7C6F73FD734EF384547D23F0C022B8716D9F23FA69BC420B072054003780B24287ED03F0AD7A3703D0A05407B14AE47E17ADC3FEC51B81E85EBC93F0AD7A3703D0A0F40986E1283C04A1940D95F764F1E16E63FFCA9F1D24D62F43F5A643BDF4F0D114082E2C798BB96E43F2DB29DEFA7C6FB3F4E62105839B40B40AAF1D24D6210E63F0AD7A3703D0A06405A643BDF4F8DF33F41F163CC5D4BE63F79E92631082C1040FA7E6ABC74131D40AC1C5A643BDFF33F75931804560EF33F2FDD24068195194004560E2DB29DF53FC520B0726891FF3F273108AC1CDA13401283C0CAA145F63F8D976E1283C00640295C8FC2F5280240CDCCCCCCCCCCF63F48E17A14AE4710408FC2F5285C0F27C004560E2DB29DFD3F5839B4C876BEF9BF7B14AE47E1FA28C0C1CAA145B6F3FB3FE71DA7E8482EEBBFC3F5285C8FC22BC0560E2DB29DEFF93F89D2DEE00B93C13FEC51B81E856B2FC0AE47E17A14AEF73F5EBA490C022BF53FE9263108AC1C22C0151DC9E53FA4EF3FD34D62105839FABFD578E92631C821C0265305A3923AEB3FDD2406819543E1BF4C37894160E520C075931804560EE53F3B014D840D4FED3F8FC2F5285C4F20C0C364AA605452DB3F6F1283C0CAA10540000000000000000079E9263108ACDC3F000000000000000023DBF97E6ABCFE3F6A4DF38E5374DC3F77BE9F1A2FDDF43FF4FDD478E9260940022B8716D9CEDB3FF4FDD478E92607404E62105839B4F23F8D976E1283C0DA3F1283C0CAA1C51140E9263108AC1C2240151DC9E53FA4EF3FD34D62105839FA3F4A0C022B87D62040F6285C8FC2F5F03FA8C64B37894104401B2FDD2406811C402DB29DEFA7C6F13FCFF753E3A59B0C40B6F3FDD478E91040105839B4C876F23F068195438BEC13408FC2F5285C0F274004560E2DB29DFD3F5839B4C876BEF93F48E17A14AE472440C74B37894160FF3F77BE9F1A2FDD0340E17A14AE476120403D0AD7A3703D00400C022B8716D90B40C3F5285C8FC2134062105839B4C800405839B4C8763E1340"> : tensor<3x5x4x3xf64>
    %0 = stablehlo.add %arg2, %arg3 : tensor<f64>
    %cst_0 = stablehlo.constant dense<-4.000000e+01> : tensor<f64>
    %cst_1 = stablehlo.constant dense<4.000000e+01> : tensor<f64>
    %1 = call @clip_80(%0, %cst_0, %cst_1) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.transpose %cst, dims = [3, 0, 1, 2] : (tensor<3x5x4x3xf64>) -> tensor<3x3x5x4xf64>
    %3 = stablehlo.abs %arg1 : tensor<f64>
    %cst_2 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %4 = stablehlo.compare  LT, %3, %cst_2,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %5 = stablehlo.convert %4 : (tensor<i1>) -> tensor<i32>
    %6 = "stablehlo.case"(%5) ({
      %53 = stablehlo.sign %arg1 : tensor<f64>
      stablehlo.return %53 : tensor<f64>
    }, {
      %cst_18 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      stablehlo.return %cst_18 : tensor<f64>
    }) : (tensor<i32>) -> tensor<f64>
    %7 = stablehlo.multiply %1, %6 : tensor<f64>
    %cst_3 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %8 = stablehlo.subtract %arg0, %cst_3 : tensor<f64>
    %cst_4 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %9 = stablehlo.multiply %8, %cst_4 : tensor<f64>
    %cst_5 = stablehlo.constant dense<8.000000e-01> : tensor<f64>
    %cst_6 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %10 = call @clip_86(%cst_5, %cst_6) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %11 = stablehlo.convert %10 : tensor<f64>
    %12 = stablehlo.divide %9, %11 : tensor<f64>
    %cst_7 = stablehlo.constant dense<-4.000000e+01> : tensor<f64>
    %13 = stablehlo.subtract %7, %cst_7 : tensor<f64>
    %cst_8 = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %14 = stablehlo.multiply %13, %cst_8 : tensor<f64>
    %cst_9 = stablehlo.constant dense<8.000000e+01> : tensor<f64>
    %cst_10 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %15 = call @clip_86(%cst_9, %cst_10) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %16 = stablehlo.convert %15 : tensor<f64>
    %17 = stablehlo.divide %14, %16 : tensor<f64>
    %18 = stablehlo.abs %arg1 : tensor<f64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %19 = stablehlo.subtract %18, %cst_11 : tensor<f64>
    %cst_12 = stablehlo.constant dense<3.000000e+00> : tensor<f64>
    %20 = stablehlo.multiply %19, %cst_12 : tensor<f64>
    %cst_13 = stablehlo.constant dense<1.500000e+01> : tensor<f64>
    %cst_14 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
    %21 = call @clip_86(%cst_13, %cst_14) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %22 = stablehlo.convert %21 : tensor<f64>
    %23 = stablehlo.divide %20, %22 : tensor<f64>
    %24 = stablehlo.slice %2 [0:1, 0:3, 0:5, 0:4] : (tensor<3x3x5x4xf64>) -> tensor<1x3x5x4xf64>
    %25 = stablehlo.reshape %24 : (tensor<1x3x5x4xf64>) -> tensor<3x5x4xf64>
    %26 = call @_map_coordinates(%25, %12, %17, %23) : (tensor<3x5x4xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %27 = stablehlo.slice %2 [1:2, 0:3, 0:5, 0:4] : (tensor<3x3x5x4xf64>) -> tensor<1x3x5x4xf64>
    %28 = stablehlo.reshape %27 : (tensor<1x3x5x4xf64>) -> tensor<3x5x4xf64>
    %29 = call @_map_coordinates(%28, %12, %17, %23) : (tensor<3x5x4xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %30 = stablehlo.slice %2 [2:3, 0:3, 0:5, 0:4] : (tensor<3x3x5x4xf64>) -> tensor<1x3x5x4xf64>
    %31 = stablehlo.reshape %30 : (tensor<1x3x5x4xf64>) -> tensor<3x5x4xf64>
    %32 = call @_map_coordinates(%31, %12, %17, %23) : (tensor<3x5x4xf64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %33 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %34 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %35 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %36 = stablehlo.concatenate %33, %34, %35, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %cst_15 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %37 = stablehlo.multiply %arg3, %cst_15 : tensor<f64>
    %38 = stablehlo.slice %36 [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %39 = stablehlo.reshape %38 : (tensor<1xf64>) -> tensor<f64>
    %40 = stablehlo.multiply %39, %6 : tensor<f64>
    %41 = stablehlo.slice %36 [1:2] : (tensor<3xf64>) -> tensor<1xf64>
    %42 = stablehlo.reshape %41 : (tensor<1xf64>) -> tensor<f64>
    %43 = stablehlo.slice %36 [2:3] : (tensor<3xf64>) -> tensor<1xf64>
    %44 = stablehlo.reshape %43 : (tensor<1xf64>) -> tensor<f64>
    %45 = stablehlo.multiply %44, %6 : tensor<f64>
    %46 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %47 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %48 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %49 = stablehlo.broadcast_in_dim %42, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %50 = stablehlo.broadcast_in_dim %45, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %51 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %52 = stablehlo.concatenate %46, %47, %48, %49, %50, %51, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<6xf64>
    return %52 : tensor<6xf64>
  }
  func.func private @clip_86(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.maximum %arg1, %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
  func.func private @_map_coordinates(%arg0: tensor<3x5x4xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.floor %arg1 : tensor<f64>
    %1 = stablehlo.subtract %arg1, %0 : tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %2 = stablehlo.subtract %cst, %1 : tensor<f64>
    %3 = stablehlo.convert %0 : (tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.add %3, %c : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<2> : tensor<i64>
    %5 = call @clip(%3, %c_0, %c_1) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<2> : tensor<i64>
    %6 = call @clip(%4, %c_2, %c_3) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %7 = stablehlo.floor %arg2 : tensor<f64>
    %8 = stablehlo.subtract %arg2, %7 : tensor<f64>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %9 = stablehlo.subtract %cst_4, %8 : tensor<f64>
    %10 = stablehlo.convert %7 : (tensor<f64>) -> tensor<i32>
    %c_5 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.add %10, %c_5 : tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %c_7 = stablehlo.constant dense<4> : tensor<i64>
    %12 = call @clip(%10, %c_6, %c_7) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %c_8 = stablehlo.constant dense<0> : tensor<i64>
    %c_9 = stablehlo.constant dense<4> : tensor<i64>
    %13 = call @clip(%11, %c_8, %c_9) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %14 = stablehlo.floor %arg3 : tensor<f64>
    %15 = stablehlo.subtract %arg3, %14 : tensor<f64>
    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %16 = stablehlo.subtract %cst_10, %15 : tensor<f64>
    %17 = stablehlo.convert %14 : (tensor<f64>) -> tensor<i32>
    %c_11 = stablehlo.constant dense<1> : tensor<i32>
    %18 = stablehlo.add %17, %c_11 : tensor<i32>
    %c_12 = stablehlo.constant dense<0> : tensor<i64>
    %c_13 = stablehlo.constant dense<3> : tensor<i64>
    %19 = call @clip(%17, %c_12, %c_13) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %c_14 = stablehlo.constant dense<0> : tensor<i64>
    %c_15 = stablehlo.constant dense<3> : tensor<i64>
    %20 = call @clip(%18, %c_14, %c_15) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %c_16 = stablehlo.constant dense<0> : tensor<i32>
    %21 = stablehlo.compare  LT, %5, %c_16,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_17 = stablehlo.constant dense<3> : tensor<i32>
    %22 = stablehlo.add %5, %c_17 : tensor<i32>
    %23 = stablehlo.select %21, %22, %5 : tensor<i1>, tensor<i32>
    %c_18 = stablehlo.constant dense<0> : tensor<i32>
    %24 = stablehlo.compare  LT, %12, %c_18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_19 = stablehlo.constant dense<5> : tensor<i32>
    %25 = stablehlo.add %12, %c_19 : tensor<i32>
    %26 = stablehlo.select %24, %25, %12 : tensor<i1>, tensor<i32>
    %c_20 = stablehlo.constant dense<0> : tensor<i32>
    %27 = stablehlo.compare  LT, %19, %c_20,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_21 = stablehlo.constant dense<4> : tensor<i32>
    %28 = stablehlo.add %19, %c_21 : tensor<i32>
    %29 = stablehlo.select %27, %28, %19 : tensor<i1>, tensor<i32>
    %30 = stablehlo.dynamic_slice %arg0, %23, %26, %29, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1x1x1xf64>) -> tensor<f64>
    %32 = stablehlo.multiply %2, %9 : tensor<f64>
    %33 = stablehlo.multiply %32, %16 : tensor<f64>
    %34 = stablehlo.multiply %33, %31 : tensor<f64>
    %c_22 = stablehlo.constant dense<0> : tensor<i32>
    %35 = stablehlo.compare  LT, %5, %c_22,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_23 = stablehlo.constant dense<3> : tensor<i32>
    %36 = stablehlo.add %5, %c_23 : tensor<i32>
    %37 = stablehlo.select %35, %36, %5 : tensor<i1>, tensor<i32>
    %c_24 = stablehlo.constant dense<0> : tensor<i32>
    %38 = stablehlo.compare  LT, %12, %c_24,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_25 = stablehlo.constant dense<5> : tensor<i32>
    %39 = stablehlo.add %12, %c_25 : tensor<i32>
    %40 = stablehlo.select %38, %39, %12 : tensor<i1>, tensor<i32>
    %c_26 = stablehlo.constant dense<0> : tensor<i32>
    %41 = stablehlo.compare  LT, %20, %c_26,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_27 = stablehlo.constant dense<4> : tensor<i32>
    %42 = stablehlo.add %20, %c_27 : tensor<i32>
    %43 = stablehlo.select %41, %42, %20 : tensor<i1>, tensor<i32>
    %44 = stablehlo.dynamic_slice %arg0, %37, %40, %43, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %45 = stablehlo.reshape %44 : (tensor<1x1x1xf64>) -> tensor<f64>
    %46 = stablehlo.multiply %2, %9 : tensor<f64>
    %47 = stablehlo.multiply %46, %15 : tensor<f64>
    %48 = stablehlo.multiply %47, %45 : tensor<f64>
    %c_28 = stablehlo.constant dense<0> : tensor<i32>
    %49 = stablehlo.compare  LT, %5, %c_28,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_29 = stablehlo.constant dense<3> : tensor<i32>
    %50 = stablehlo.add %5, %c_29 : tensor<i32>
    %51 = stablehlo.select %49, %50, %5 : tensor<i1>, tensor<i32>
    %c_30 = stablehlo.constant dense<0> : tensor<i32>
    %52 = stablehlo.compare  LT, %13, %c_30,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_31 = stablehlo.constant dense<5> : tensor<i32>
    %53 = stablehlo.add %13, %c_31 : tensor<i32>
    %54 = stablehlo.select %52, %53, %13 : tensor<i1>, tensor<i32>
    %c_32 = stablehlo.constant dense<0> : tensor<i32>
    %55 = stablehlo.compare  LT, %19, %c_32,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_33 = stablehlo.constant dense<4> : tensor<i32>
    %56 = stablehlo.add %19, %c_33 : tensor<i32>
    %57 = stablehlo.select %55, %56, %19 : tensor<i1>, tensor<i32>
    %58 = stablehlo.dynamic_slice %arg0, %51, %54, %57, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %59 = stablehlo.reshape %58 : (tensor<1x1x1xf64>) -> tensor<f64>
    %60 = stablehlo.multiply %2, %8 : tensor<f64>
    %61 = stablehlo.multiply %60, %16 : tensor<f64>
    %62 = stablehlo.multiply %61, %59 : tensor<f64>
    %c_34 = stablehlo.constant dense<0> : tensor<i32>
    %63 = stablehlo.compare  LT, %5, %c_34,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_35 = stablehlo.constant dense<3> : tensor<i32>
    %64 = stablehlo.add %5, %c_35 : tensor<i32>
    %65 = stablehlo.select %63, %64, %5 : tensor<i1>, tensor<i32>
    %c_36 = stablehlo.constant dense<0> : tensor<i32>
    %66 = stablehlo.compare  LT, %13, %c_36,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_37 = stablehlo.constant dense<5> : tensor<i32>
    %67 = stablehlo.add %13, %c_37 : tensor<i32>
    %68 = stablehlo.select %66, %67, %13 : tensor<i1>, tensor<i32>
    %c_38 = stablehlo.constant dense<0> : tensor<i32>
    %69 = stablehlo.compare  LT, %20, %c_38,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_39 = stablehlo.constant dense<4> : tensor<i32>
    %70 = stablehlo.add %20, %c_39 : tensor<i32>
    %71 = stablehlo.select %69, %70, %20 : tensor<i1>, tensor<i32>
    %72 = stablehlo.dynamic_slice %arg0, %65, %68, %71, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %73 = stablehlo.reshape %72 : (tensor<1x1x1xf64>) -> tensor<f64>
    %74 = stablehlo.multiply %2, %8 : tensor<f64>
    %75 = stablehlo.multiply %74, %15 : tensor<f64>
    %76 = stablehlo.multiply %75, %73 : tensor<f64>
    %c_40 = stablehlo.constant dense<0> : tensor<i32>
    %77 = stablehlo.compare  LT, %6, %c_40,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_41 = stablehlo.constant dense<3> : tensor<i32>
    %78 = stablehlo.add %6, %c_41 : tensor<i32>
    %79 = stablehlo.select %77, %78, %6 : tensor<i1>, tensor<i32>
    %c_42 = stablehlo.constant dense<0> : tensor<i32>
    %80 = stablehlo.compare  LT, %12, %c_42,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_43 = stablehlo.constant dense<5> : tensor<i32>
    %81 = stablehlo.add %12, %c_43 : tensor<i32>
    %82 = stablehlo.select %80, %81, %12 : tensor<i1>, tensor<i32>
    %c_44 = stablehlo.constant dense<0> : tensor<i32>
    %83 = stablehlo.compare  LT, %19, %c_44,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_45 = stablehlo.constant dense<4> : tensor<i32>
    %84 = stablehlo.add %19, %c_45 : tensor<i32>
    %85 = stablehlo.select %83, %84, %19 : tensor<i1>, tensor<i32>
    %86 = stablehlo.dynamic_slice %arg0, %79, %82, %85, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %87 = stablehlo.reshape %86 : (tensor<1x1x1xf64>) -> tensor<f64>
    %88 = stablehlo.multiply %1, %9 : tensor<f64>
    %89 = stablehlo.multiply %88, %16 : tensor<f64>
    %90 = stablehlo.multiply %89, %87 : tensor<f64>
    %c_46 = stablehlo.constant dense<0> : tensor<i32>
    %91 = stablehlo.compare  LT, %6, %c_46,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_47 = stablehlo.constant dense<3> : tensor<i32>
    %92 = stablehlo.add %6, %c_47 : tensor<i32>
    %93 = stablehlo.select %91, %92, %6 : tensor<i1>, tensor<i32>
    %c_48 = stablehlo.constant dense<0> : tensor<i32>
    %94 = stablehlo.compare  LT, %12, %c_48,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_49 = stablehlo.constant dense<5> : tensor<i32>
    %95 = stablehlo.add %12, %c_49 : tensor<i32>
    %96 = stablehlo.select %94, %95, %12 : tensor<i1>, tensor<i32>
    %c_50 = stablehlo.constant dense<0> : tensor<i32>
    %97 = stablehlo.compare  LT, %20, %c_50,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_51 = stablehlo.constant dense<4> : tensor<i32>
    %98 = stablehlo.add %20, %c_51 : tensor<i32>
    %99 = stablehlo.select %97, %98, %20 : tensor<i1>, tensor<i32>
    %100 = stablehlo.dynamic_slice %arg0, %93, %96, %99, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %101 = stablehlo.reshape %100 : (tensor<1x1x1xf64>) -> tensor<f64>
    %102 = stablehlo.multiply %1, %9 : tensor<f64>
    %103 = stablehlo.multiply %102, %15 : tensor<f64>
    %104 = stablehlo.multiply %103, %101 : tensor<f64>
    %c_52 = stablehlo.constant dense<0> : tensor<i32>
    %105 = stablehlo.compare  LT, %6, %c_52,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_53 = stablehlo.constant dense<3> : tensor<i32>
    %106 = stablehlo.add %6, %c_53 : tensor<i32>
    %107 = stablehlo.select %105, %106, %6 : tensor<i1>, tensor<i32>
    %c_54 = stablehlo.constant dense<0> : tensor<i32>
    %108 = stablehlo.compare  LT, %13, %c_54,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_55 = stablehlo.constant dense<5> : tensor<i32>
    %109 = stablehlo.add %13, %c_55 : tensor<i32>
    %110 = stablehlo.select %108, %109, %13 : tensor<i1>, tensor<i32>
    %c_56 = stablehlo.constant dense<0> : tensor<i32>
    %111 = stablehlo.compare  LT, %19, %c_56,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_57 = stablehlo.constant dense<4> : tensor<i32>
    %112 = stablehlo.add %19, %c_57 : tensor<i32>
    %113 = stablehlo.select %111, %112, %19 : tensor<i1>, tensor<i32>
    %114 = stablehlo.dynamic_slice %arg0, %107, %110, %113, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %115 = stablehlo.reshape %114 : (tensor<1x1x1xf64>) -> tensor<f64>
    %116 = stablehlo.multiply %1, %8 : tensor<f64>
    %117 = stablehlo.multiply %116, %16 : tensor<f64>
    %118 = stablehlo.multiply %117, %115 : tensor<f64>
    %c_58 = stablehlo.constant dense<0> : tensor<i32>
    %119 = stablehlo.compare  LT, %6, %c_58,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_59 = stablehlo.constant dense<3> : tensor<i32>
    %120 = stablehlo.add %6, %c_59 : tensor<i32>
    %121 = stablehlo.select %119, %120, %6 : tensor<i1>, tensor<i32>
    %c_60 = stablehlo.constant dense<0> : tensor<i32>
    %122 = stablehlo.compare  LT, %13, %c_60,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_61 = stablehlo.constant dense<5> : tensor<i32>
    %123 = stablehlo.add %13, %c_61 : tensor<i32>
    %124 = stablehlo.select %122, %123, %13 : tensor<i1>, tensor<i32>
    %c_62 = stablehlo.constant dense<0> : tensor<i32>
    %125 = stablehlo.compare  LT, %20, %c_62,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_63 = stablehlo.constant dense<4> : tensor<i32>
    %126 = stablehlo.add %20, %c_63 : tensor<i32>
    %127 = stablehlo.select %125, %126, %20 : tensor<i1>, tensor<i32>
    %128 = stablehlo.dynamic_slice %arg0, %121, %124, %127, sizes = [1, 1, 1] : (tensor<3x5x4xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1xf64>
    %129 = stablehlo.reshape %128 : (tensor<1x1x1xf64>) -> tensor<f64>
    %130 = stablehlo.multiply %1, %8 : tensor<f64>
    %131 = stablehlo.multiply %130, %15 : tensor<f64>
    %132 = stablehlo.multiply %131, %129 : tensor<f64>
    %133 = stablehlo.add %34, %48 : tensor<f64>
    %134 = stablehlo.add %133, %62 : tensor<f64>
    %135 = stablehlo.add %134, %76 : tensor<f64>
    %136 = stablehlo.add %135, %90 : tensor<f64>
    %137 = stablehlo.add %136, %104 : tensor<f64>
    %138 = stablehlo.add %137, %118 : tensor<f64>
    %139 = stablehlo.add %138, %132 : tensor<f64>
    return %139 : tensor<f64>
  }
  func.func private @inner_96(%arg0: tensor<6xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<6xf64>) -> tensor<6xf64> {
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<6xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.slice %arg0 [1:2] : (tensor<6xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.slice %arg0 [2:3] : (tensor<6xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.slice %arg0 [3:4] : (tensor<6xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.slice %arg0 [4:5] : (tensor<6xf64>) -> tensor<1xf64>
    %9 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
    %10 = stablehlo.slice %arg0 [5:6] : (tensor<6xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<4.038700e-01> : tensor<f64>
    %12 = stablehlo.subtract %arg1, %cst : tensor<f64>
    %13 = stablehlo.multiply %9, %12 : tensor<f64>
    %cst_0 = stablehlo.constant dense<5.434000e-02> : tensor<f64>
    %14 = stablehlo.divide %13, %cst_0 : tensor<f64>
    %15 = stablehlo.subtract %5, %14 : tensor<f64>
    %cst_1 = stablehlo.constant dense<4.038700e-01> : tensor<f64>
    %16 = stablehlo.subtract %arg1, %cst_1 : tensor<f64>
    %17 = stablehlo.multiply %11, %16 : tensor<f64>
    %cst_2 = stablehlo.constant dense<5.434000e-02> : tensor<f64>
    %18 = stablehlo.divide %17, %cst_2 : tensor<f64>
    %19 = stablehlo.subtract %3, %18 : tensor<f64>
    %20 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %21 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %22 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %23 = stablehlo.concatenate %20, %21, %22, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %24 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %25 = stablehlo.multiply %23, %24 : tensor<3xf64>
    %cst_3 = stablehlo.constant dense<2.489130e-03> : tensor<f64>
    %26 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %27 = stablehlo.multiply %25, %26 : tensor<3xf64>
    %28 = stablehlo.negate %15 : tensor<f64>
    %29 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %30 = stablehlo.broadcast_in_dim %28, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %31 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %32 = stablehlo.concatenate %29, %30, %31, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %33 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %34 = stablehlo.multiply %32, %33 : tensor<3xf64>
    %cst_4 = stablehlo.constant dense<2.489130e-03> : tensor<f64>
    %35 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %36 = stablehlo.multiply %34, %35 : tensor<3xf64>
    %cst_5 = stablehlo.constant dense<5.434000e-02> : tensor<f64>
    %37 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %38 = stablehlo.multiply %36, %37 : tensor<3xf64>
    %39 = stablehlo.concatenate %38, %27, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    return %39 : tensor<6xf64>
  }
  func.func private @inner_105(%arg0: tensor<i64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<[1.000000e-02, 6.700000e-01, 1.330000e+00, 1.990000e+00, 2.650000e+00, 3.310000e+00, 3.970000e+00, 4.630000e+00, 5.290000e+00, 5.950000e+00, 6.610000e+00, 7.2699999999999996, 7.9299999999999997, 8.5899999999999999, 9.250000e+00, 9.910000e+00, 1.057000e+01, 1.123000e+01, 1.189000e+01, 1.255000e+01, 1.321000e+01, 1.387000e+01, 1.453000e+01, 1.519000e+01, 1.585000e+01, 1.651000e+01, 1.717000e+01, 1.783000e+01, 1.849000e+01, 1.915000e+01, 1.981000e+01, 2.047000e+01, 2.113000e+01, 2.179000e+01, 2.245000e+01, 2.311000e+01, 2.377000e+01, 2.443000e+01, 2.509000e+01, 2.575000e+01, 2.641000e+01, 2.707000e+01, 2.773000e+01, 2.839000e+01, 2.905000e+01, 2.971000e+01, 3.037000e+01, 3.103000e+01, 3.169000e+01, 3.215000e+01]> : tensor<50xf64>
    %cst_0 = stablehlo.constant dense<[3.221480e+02, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 8.842600e+01, 0.000000e+00]> : tensor<50xf64>
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.multiply %0, %arg1 : tensor<f64>
    %2 = call @_interp_107(%1, %cst, %cst_0) : (tensor<f64>, tensor<50xf64>, tensor<50xf64>) -> tensor<f64>
    return %2 : tensor<f64>
  }
  func.func private @_interp_107(%arg0: tensor<f64>, %arg1: tensor<50xf64>, %arg2: tensor<50xf64>) -> tensor<f64> {
    %0 = call @searchsorted_108(%arg1, %arg0) : (tensor<50xf64>, tensor<f64>) -> tensor<i32>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<49> : tensor<i64>
    %1 = call @clip(%0, %c, %c_0) : (tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.compare  LT, %1, %c_1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_2 = stablehlo.constant dense<50> : tensor<i32>
    %3 = stablehlo.add %1, %c_2 : tensor<i32>
    %4 = stablehlo.select %2, %3, %1 : tensor<i1>, tensor<i32>
    %5 = stablehlo.dynamic_slice %arg2, %4, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %7 = stablehlo.subtract %1, %c_3 : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.compare  LT, %7, %c_4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_5 = stablehlo.constant dense<50> : tensor<i32>
    %9 = stablehlo.add %7, %c_5 : tensor<i32>
    %10 = stablehlo.select %8, %9, %7 : tensor<i1>, tensor<i32>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
    %13 = stablehlo.subtract %6, %12 : tensor<f64>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %14 = stablehlo.compare  LT, %1, %c_6,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_7 = stablehlo.constant dense<50> : tensor<i32>
    %15 = stablehlo.add %1, %c_7 : tensor<i32>
    %16 = stablehlo.select %14, %15, %1 : tensor<i1>, tensor<i32>
    %17 = stablehlo.dynamic_slice %arg1, %16, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %18 = stablehlo.reshape %17 : (tensor<1xf64>) -> tensor<f64>
    %c_8 = stablehlo.constant dense<1> : tensor<i32>
    %19 = stablehlo.subtract %1, %c_8 : tensor<i32>
    %c_9 = stablehlo.constant dense<0> : tensor<i32>
    %20 = stablehlo.compare  LT, %19, %c_9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_10 = stablehlo.constant dense<50> : tensor<i32>
    %21 = stablehlo.add %19, %c_10 : tensor<i32>
    %22 = stablehlo.select %20, %21, %19 : tensor<i1>, tensor<i32>
    %23 = stablehlo.dynamic_slice %arg1, %22, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %24 = stablehlo.reshape %23 : (tensor<1xf64>) -> tensor<f64>
    %25 = stablehlo.subtract %18, %24 : tensor<f64>
    %c_11 = stablehlo.constant dense<1> : tensor<i32>
    %26 = stablehlo.subtract %1, %c_11 : tensor<i32>
    %c_12 = stablehlo.constant dense<0> : tensor<i32>
    %27 = stablehlo.compare  LT, %26, %c_12,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_13 = stablehlo.constant dense<50> : tensor<i32>
    %28 = stablehlo.add %26, %c_13 : tensor<i32>
    %29 = stablehlo.select %27, %28, %26 : tensor<i1>, tensor<i32>
    %30 = stablehlo.dynamic_slice %arg1, %29, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %31 = stablehlo.reshape %30 : (tensor<1xf64>) -> tensor<f64>
    %32 = stablehlo.subtract %arg0, %31 : tensor<f64>
    %33 = stablehlo.abs %25 : tensor<f64>
    %cst = stablehlo.constant dense<4.9303806576313238E-32> : tensor<f64>
    %34 = stablehlo.compare  LE, %33, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %c_14 = stablehlo.constant dense<1> : tensor<i32>
    %35 = stablehlo.subtract %1, %c_14 : tensor<i32>
    %c_15 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.compare  LT, %35, %c_15,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_16 = stablehlo.constant dense<50> : tensor<i32>
    %37 = stablehlo.add %35, %c_16 : tensor<i32>
    %38 = stablehlo.select %36, %37, %35 : tensor<i1>, tensor<i32>
    %39 = stablehlo.dynamic_slice %arg2, %38, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %40 = stablehlo.reshape %39 : (tensor<1xf64>) -> tensor<f64>
    %c_17 = stablehlo.constant dense<1> : tensor<i32>
    %41 = stablehlo.subtract %1, %c_17 : tensor<i32>
    %c_18 = stablehlo.constant dense<0> : tensor<i32>
    %42 = stablehlo.compare  LT, %41, %c_18,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_19 = stablehlo.constant dense<50> : tensor<i32>
    %43 = stablehlo.add %41, %c_19 : tensor<i32>
    %44 = stablehlo.select %42, %43, %41 : tensor<i1>, tensor<i32>
    %45 = stablehlo.dynamic_slice %arg2, %44, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %46 = stablehlo.reshape %45 : (tensor<1xf64>) -> tensor<f64>
    %c_20 = stablehlo.constant dense<1> : tensor<i64>
    %47 = call @_where_9(%34, %c_20, %25) : (tensor<i1>, tensor<i64>, tensor<f64>) -> tensor<f64>
    %48 = stablehlo.divide %32, %47 : tensor<f64>
    %49 = stablehlo.multiply %48, %13 : tensor<f64>
    %50 = stablehlo.add %46, %49 : tensor<f64>
    %51 = call @_where_13(%34, %40, %50) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %52 = stablehlo.slice %arg2 [0:1] : (tensor<50xf64>) -> tensor<1xf64>
    %53 = stablehlo.reshape %52 : (tensor<1xf64>) -> tensor<f64>
    %54 = stablehlo.slice %arg1 [0:1] : (tensor<50xf64>) -> tensor<1xf64>
    %55 = stablehlo.reshape %54 : (tensor<1xf64>) -> tensor<f64>
    %56 = stablehlo.compare  LT, %arg0, %55,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %57 = call @_where_13(%56, %53, %51) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %c_21 = stablehlo.constant dense<49> : tensor<i64>
    %58 = stablehlo.dynamic_slice %arg2, %c_21, sizes = [1] : (tensor<50xf64>, tensor<i64>) -> tensor<1xf64>
    %59 = stablehlo.reshape %58 : (tensor<1xf64>) -> tensor<f64>
    %c_22 = stablehlo.constant dense<49> : tensor<i64>
    %60 = stablehlo.dynamic_slice %arg1, %c_22, sizes = [1] : (tensor<50xf64>, tensor<i64>) -> tensor<1xf64>
    %61 = stablehlo.reshape %60 : (tensor<1xf64>) -> tensor<f64>
    %62 = stablehlo.compare  GT, %arg0, %61,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %63 = call @_where_13(%62, %59, %57) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
    return %63 : tensor<f64>
  }
  func.func private @searchsorted_108(%arg0: tensor<50xf64>, %arg1: tensor<f64>) -> tensor<i32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<50> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:5 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %c_1, %iterArg_4 = %c, %iterArg_5 = %c_0) : tensor<50xf64>, tensor<f64>, tensor<i64>, tensor<i32>, tensor<i32>
    cond {
      %c_6 = stablehlo.constant dense<6> : tensor<i64>
      %1 = stablehlo.compare  LT, %iterArg_3, %c_6,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1:2 = func.call @closed_call_111(%iterArg, %iterArg_2, %iterArg_4, %iterArg_5) : (tensor<50xf64>, tensor<f64>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
      %c_6 = stablehlo.constant dense<1> : tensor<i64>
      %2 = stablehlo.add %iterArg_3, %c_6 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_2, %2, %1#0, %1#1 : tensor<50xf64>, tensor<f64>, tensor<i64>, tensor<i32>, tensor<i32>
    }
    return %0#4 : tensor<i32>
  }
  func.func private @closed_call_111(%arg0: tensor<50xf64>, %arg1: tensor<f64>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<ui32>
    %1 = stablehlo.convert %arg3 : (tensor<i32>) -> tensor<ui32>
    %2 = stablehlo.add %0, %1 : tensor<ui32>
    %c = stablehlo.constant dense<2> : tensor<ui32>
    %3 = stablehlo.divide %2, %c : tensor<ui32>
    %4 = stablehlo.convert %3 : (tensor<ui32>) -> tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.compare  LT, %4, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_1 = stablehlo.constant dense<50> : tensor<i32>
    %6 = stablehlo.add %4, %c_1 : tensor<i32>
    %7 = stablehlo.select %5, %6, %4 : tensor<i1>, tensor<i32>
    %8 = stablehlo.dynamic_slice %arg0, %7, sizes = [1] : (tensor<50xf64>, tensor<i32>) -> tensor<1xf64>
    %9 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %10 = stablehlo.compare  EQ, %arg1, %cst,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %11 = stablehlo.select %10, %cst_2, %arg1 : tensor<i1>, tensor<f64>
    %12 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_3 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %13 = stablehlo.select %12, %cst_3, %11 : tensor<i1>, tensor<f64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %14 = stablehlo.compare  EQ, %9, %cst_4,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %15 = stablehlo.select %14, %cst_5, %9 : tensor<i1>, tensor<f64>
    %16 = stablehlo.compare  NE, %9, %9,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %cst_6 = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
    %17 = stablehlo.select %16, %cst_6, %15 : tensor<i1>, tensor<f64>
    %18 = stablehlo.compare  LT, %13, %17,  TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %19 = call @_where(%18, %arg2, %4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %20 = call @_where(%18, %4, %arg3) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    return %19, %20 : tensor<i32>, tensor<i32>
  }
  func.func private @inner_124(%arg0: tensor<6xf64>, %arg1: tensor<7xf64>) -> tensor<6xf64> {
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
  func.func private @inner_126(%arg0: tensor<f64>, %arg1: tensor<6xf64>, %arg2: tensor<7xf64>) -> tensor<6xf64> {
    %cst = stablehlo.constant dense<[-1.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<3xf64>
    %0 = stablehlo.slice %arg2 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %4 = stablehlo.concatenate %cst, %3, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %5 = stablehlo.slice %4 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
    %7 = stablehlo.multiply %2, %6 : tensor<f64>
    %8 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %9 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
    %10 = stablehlo.slice %4 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %11 = stablehlo.reshape %10 : (tensor<1xf64>) -> tensor<f64>
    %12 = stablehlo.multiply %9, %11 : tensor<f64>
    %13 = stablehlo.add %7, %12 : tensor<f64>
    %14 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %15 = stablehlo.reshape %14 : (tensor<1xf64>) -> tensor<f64>
    %16 = stablehlo.slice %4 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %17 = stablehlo.reshape %16 : (tensor<1xf64>) -> tensor<f64>
    %18 = stablehlo.multiply %15, %17 : tensor<f64>
    %19 = stablehlo.add %13, %18 : tensor<f64>
    %20 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %21 = stablehlo.reshape %20 : (tensor<1xf64>) -> tensor<f64>
    %22 = stablehlo.slice %4 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %23 = stablehlo.reshape %22 : (tensor<1xf64>) -> tensor<f64>
    %24 = stablehlo.multiply %21, %23 : tensor<f64>
    %25 = stablehlo.subtract %19, %24 : tensor<f64>
    %26 = stablehlo.reshape %25 : (tensor<f64>) -> tensor<1xf64>
    %27 = stablehlo.multiply %2, %23 : tensor<f64>
    %28 = stablehlo.multiply %9, %17 : tensor<f64>
    %29 = stablehlo.subtract %27, %28 : tensor<f64>
    %30 = stablehlo.multiply %15, %11 : tensor<f64>
    %31 = stablehlo.add %29, %30 : tensor<f64>
    %32 = stablehlo.multiply %21, %6 : tensor<f64>
    %33 = stablehlo.add %31, %32 : tensor<f64>
    %34 = stablehlo.reshape %33 : (tensor<f64>) -> tensor<1xf64>
    %35 = stablehlo.multiply %2, %17 : tensor<f64>
    %36 = stablehlo.multiply %9, %23 : tensor<f64>
    %37 = stablehlo.add %35, %36 : tensor<f64>
    %38 = stablehlo.multiply %15, %6 : tensor<f64>
    %39 = stablehlo.subtract %37, %38 : tensor<f64>
    %40 = stablehlo.multiply %21, %11 : tensor<f64>
    %41 = stablehlo.add %39, %40 : tensor<f64>
    %42 = stablehlo.reshape %41 : (tensor<f64>) -> tensor<1xf64>
    %43 = stablehlo.multiply %2, %11 : tensor<f64>
    %44 = stablehlo.multiply %9, %6 : tensor<f64>
    %45 = stablehlo.subtract %43, %44 : tensor<f64>
    %46 = stablehlo.multiply %15, %23 : tensor<f64>
    %47 = stablehlo.subtract %45, %46 : tensor<f64>
    %48 = stablehlo.multiply %21, %17 : tensor<f64>
    %49 = stablehlo.subtract %47, %48 : tensor<f64>
    %50 = stablehlo.reshape %49 : (tensor<f64>) -> tensor<1xf64>
    %51 = stablehlo.concatenate %26, %34, %42, %50, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %52 = stablehlo.slice %51 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %53 = stablehlo.reshape %52 : (tensor<1xf64>) -> tensor<f64>
    %54 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %55 = stablehlo.reshape %54 : (tensor<1xf64>) -> tensor<f64>
    %56 = stablehlo.negate %55 : tensor<f64>
    %57 = stablehlo.reshape %56 : (tensor<f64>) -> tensor<1xf64>
    %58 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %59 = stablehlo.reshape %58 : (tensor<1xf64>) -> tensor<f64>
    %60 = stablehlo.negate %59 : tensor<f64>
    %61 = stablehlo.reshape %60 : (tensor<f64>) -> tensor<1xf64>
    %62 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %63 = stablehlo.reshape %62 : (tensor<1xf64>) -> tensor<f64>
    %64 = stablehlo.negate %63 : tensor<f64>
    %65 = stablehlo.reshape %64 : (tensor<f64>) -> tensor<1xf64>
    %66 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %67 = stablehlo.reshape %66 : (tensor<1xf64>) -> tensor<f64>
    %68 = stablehlo.reshape %67 : (tensor<f64>) -> tensor<1xf64>
    %69 = stablehlo.concatenate %57, %61, %65, %68, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %70 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %71 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %72 = stablehlo.divide %69, %71 : tensor<4xf64>
    %73 = stablehlo.slice %72 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %74 = stablehlo.reshape %73 : (tensor<1xf64>) -> tensor<f64>
    %75 = stablehlo.multiply %53, %74 : tensor<f64>
    %76 = stablehlo.slice %51 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %77 = stablehlo.reshape %76 : (tensor<1xf64>) -> tensor<f64>
    %78 = stablehlo.slice %72 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %79 = stablehlo.reshape %78 : (tensor<1xf64>) -> tensor<f64>
    %80 = stablehlo.multiply %77, %79 : tensor<f64>
    %81 = stablehlo.add %75, %80 : tensor<f64>
    %82 = stablehlo.slice %51 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %83 = stablehlo.reshape %82 : (tensor<1xf64>) -> tensor<f64>
    %84 = stablehlo.slice %72 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %85 = stablehlo.reshape %84 : (tensor<1xf64>) -> tensor<f64>
    %86 = stablehlo.multiply %83, %85 : tensor<f64>
    %87 = stablehlo.add %81, %86 : tensor<f64>
    %88 = stablehlo.slice %51 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %89 = stablehlo.reshape %88 : (tensor<1xf64>) -> tensor<f64>
    %90 = stablehlo.slice %72 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %91 = stablehlo.reshape %90 : (tensor<1xf64>) -> tensor<f64>
    %92 = stablehlo.multiply %89, %91 : tensor<f64>
    %93 = stablehlo.subtract %87, %92 : tensor<f64>
    %94 = stablehlo.reshape %93 : (tensor<f64>) -> tensor<1xf64>
    %95 = stablehlo.multiply %53, %91 : tensor<f64>
    %96 = stablehlo.multiply %77, %85 : tensor<f64>
    %97 = stablehlo.subtract %95, %96 : tensor<f64>
    %98 = stablehlo.multiply %83, %79 : tensor<f64>
    %99 = stablehlo.add %97, %98 : tensor<f64>
    %100 = stablehlo.multiply %89, %74 : tensor<f64>
    %101 = stablehlo.add %99, %100 : tensor<f64>
    %102 = stablehlo.reshape %101 : (tensor<f64>) -> tensor<1xf64>
    %103 = stablehlo.multiply %53, %85 : tensor<f64>
    %104 = stablehlo.multiply %77, %91 : tensor<f64>
    %105 = stablehlo.add %103, %104 : tensor<f64>
    %106 = stablehlo.multiply %83, %74 : tensor<f64>
    %107 = stablehlo.subtract %105, %106 : tensor<f64>
    %108 = stablehlo.multiply %89, %79 : tensor<f64>
    %109 = stablehlo.add %107, %108 : tensor<f64>
    %110 = stablehlo.reshape %109 : (tensor<f64>) -> tensor<1xf64>
    %111 = stablehlo.multiply %53, %79 : tensor<f64>
    %112 = stablehlo.multiply %77, %74 : tensor<f64>
    %113 = stablehlo.subtract %111, %112 : tensor<f64>
    %114 = stablehlo.multiply %83, %91 : tensor<f64>
    %115 = stablehlo.subtract %113, %114 : tensor<f64>
    %116 = stablehlo.multiply %89, %85 : tensor<f64>
    %117 = stablehlo.subtract %115, %116 : tensor<f64>
    %118 = stablehlo.reshape %117 : (tensor<f64>) -> tensor<1xf64>
    %119 = stablehlo.concatenate %94, %102, %110, %118, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %120 = stablehlo.slice %119 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %121 = stablehlo.reshape %120 : (tensor<1xf64>) -> tensor<f64>
    %122 = stablehlo.reshape %121 : (tensor<f64>) -> tensor<1xf64>
    %123 = stablehlo.slice %119 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %124 = stablehlo.reshape %123 : (tensor<1xf64>) -> tensor<f64>
    %125 = stablehlo.reshape %124 : (tensor<f64>) -> tensor<1xf64>
    %126 = stablehlo.slice %119 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %127 = stablehlo.reshape %126 : (tensor<1xf64>) -> tensor<f64>
    %128 = stablehlo.reshape %127 : (tensor<f64>) -> tensor<1xf64>
    %129 = stablehlo.concatenate %122, %125, %128, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %130 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %131 = stablehlo.multiply %129, %130 : tensor<3xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %132 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<3xf64>
    %133 = stablehlo.concatenate %132, %131, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %134 = stablehlo.add %arg1, %133 : tensor<6xf64>
    return %134 : tensor<6xf64>
  }
  func.func private @inner_127(%arg0: tensor<7xf64>, %arg1: tensor<6xf64>, %arg2: tensor<6xf64>) -> tensor<6xf64> {
    %0 = stablehlo.slice %arg0 [0:4] : (tensor<7xf64>) -> tensor<4xf64>
    %1 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
    %3 = stablehlo.slice %arg1 [0:3] : (tensor<6xf64>) -> tensor<3xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %6 = stablehlo.slice %5 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %7 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
    %8 = stablehlo.multiply %2, %7 : tensor<f64>
    %9 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %10 = stablehlo.reshape %9 : (tensor<1xf64>) -> tensor<f64>
    %11 = stablehlo.slice %5 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
    %13 = stablehlo.multiply %10, %12 : tensor<f64>
    %14 = stablehlo.add %8, %13 : tensor<f64>
    %15 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %16 = stablehlo.reshape %15 : (tensor<1xf64>) -> tensor<f64>
    %17 = stablehlo.slice %5 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %18 = stablehlo.reshape %17 : (tensor<1xf64>) -> tensor<f64>
    %19 = stablehlo.multiply %16, %18 : tensor<f64>
    %20 = stablehlo.add %14, %19 : tensor<f64>
    %21 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
    %23 = stablehlo.slice %5 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %24 = stablehlo.reshape %23 : (tensor<1xf64>) -> tensor<f64>
    %25 = stablehlo.multiply %22, %24 : tensor<f64>
    %26 = stablehlo.subtract %20, %25 : tensor<f64>
    %27 = stablehlo.reshape %26 : (tensor<f64>) -> tensor<1xf64>
    %28 = stablehlo.multiply %2, %24 : tensor<f64>
    %29 = stablehlo.multiply %10, %18 : tensor<f64>
    %30 = stablehlo.subtract %28, %29 : tensor<f64>
    %31 = stablehlo.multiply %16, %12 : tensor<f64>
    %32 = stablehlo.add %30, %31 : tensor<f64>
    %33 = stablehlo.multiply %22, %7 : tensor<f64>
    %34 = stablehlo.add %32, %33 : tensor<f64>
    %35 = stablehlo.reshape %34 : (tensor<f64>) -> tensor<1xf64>
    %36 = stablehlo.multiply %2, %18 : tensor<f64>
    %37 = stablehlo.multiply %10, %24 : tensor<f64>
    %38 = stablehlo.add %36, %37 : tensor<f64>
    %39 = stablehlo.multiply %16, %7 : tensor<f64>
    %40 = stablehlo.subtract %38, %39 : tensor<f64>
    %41 = stablehlo.multiply %22, %12 : tensor<f64>
    %42 = stablehlo.add %40, %41 : tensor<f64>
    %43 = stablehlo.reshape %42 : (tensor<f64>) -> tensor<1xf64>
    %44 = stablehlo.multiply %2, %12 : tensor<f64>
    %45 = stablehlo.multiply %10, %7 : tensor<f64>
    %46 = stablehlo.subtract %44, %45 : tensor<f64>
    %47 = stablehlo.multiply %16, %24 : tensor<f64>
    %48 = stablehlo.subtract %46, %47 : tensor<f64>
    %49 = stablehlo.multiply %22, %18 : tensor<f64>
    %50 = stablehlo.subtract %48, %49 : tensor<f64>
    %51 = stablehlo.reshape %50 : (tensor<f64>) -> tensor<1xf64>
    %52 = stablehlo.concatenate %27, %35, %43, %51, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %53 = stablehlo.slice %52 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %54 = stablehlo.reshape %53 : (tensor<1xf64>) -> tensor<f64>
    %55 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %56 = stablehlo.reshape %55 : (tensor<1xf64>) -> tensor<f64>
    %57 = stablehlo.negate %56 : tensor<f64>
    %58 = stablehlo.reshape %57 : (tensor<f64>) -> tensor<1xf64>
    %59 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %60 = stablehlo.reshape %59 : (tensor<1xf64>) -> tensor<f64>
    %61 = stablehlo.negate %60 : tensor<f64>
    %62 = stablehlo.reshape %61 : (tensor<f64>) -> tensor<1xf64>
    %63 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %64 = stablehlo.reshape %63 : (tensor<1xf64>) -> tensor<f64>
    %65 = stablehlo.negate %64 : tensor<f64>
    %66 = stablehlo.reshape %65 : (tensor<f64>) -> tensor<1xf64>
    %67 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %68 = stablehlo.reshape %67 : (tensor<1xf64>) -> tensor<f64>
    %69 = stablehlo.reshape %68 : (tensor<f64>) -> tensor<1xf64>
    %70 = stablehlo.concatenate %58, %62, %66, %69, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %71 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %72 = stablehlo.broadcast_in_dim %71, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %73 = stablehlo.divide %70, %72 : tensor<4xf64>
    %74 = stablehlo.slice %73 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %75 = stablehlo.reshape %74 : (tensor<1xf64>) -> tensor<f64>
    %76 = stablehlo.multiply %54, %75 : tensor<f64>
    %77 = stablehlo.slice %52 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %78 = stablehlo.reshape %77 : (tensor<1xf64>) -> tensor<f64>
    %79 = stablehlo.slice %73 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %80 = stablehlo.reshape %79 : (tensor<1xf64>) -> tensor<f64>
    %81 = stablehlo.multiply %78, %80 : tensor<f64>
    %82 = stablehlo.add %76, %81 : tensor<f64>
    %83 = stablehlo.slice %52 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %84 = stablehlo.reshape %83 : (tensor<1xf64>) -> tensor<f64>
    %85 = stablehlo.slice %73 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %86 = stablehlo.reshape %85 : (tensor<1xf64>) -> tensor<f64>
    %87 = stablehlo.multiply %84, %86 : tensor<f64>
    %88 = stablehlo.add %82, %87 : tensor<f64>
    %89 = stablehlo.slice %52 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %90 = stablehlo.reshape %89 : (tensor<1xf64>) -> tensor<f64>
    %91 = stablehlo.slice %73 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %92 = stablehlo.reshape %91 : (tensor<1xf64>) -> tensor<f64>
    %93 = stablehlo.multiply %90, %92 : tensor<f64>
    %94 = stablehlo.subtract %88, %93 : tensor<f64>
    %95 = stablehlo.reshape %94 : (tensor<f64>) -> tensor<1xf64>
    %96 = stablehlo.multiply %54, %92 : tensor<f64>
    %97 = stablehlo.multiply %78, %86 : tensor<f64>
    %98 = stablehlo.subtract %96, %97 : tensor<f64>
    %99 = stablehlo.multiply %84, %80 : tensor<f64>
    %100 = stablehlo.add %98, %99 : tensor<f64>
    %101 = stablehlo.multiply %90, %75 : tensor<f64>
    %102 = stablehlo.add %100, %101 : tensor<f64>
    %103 = stablehlo.reshape %102 : (tensor<f64>) -> tensor<1xf64>
    %104 = stablehlo.multiply %54, %86 : tensor<f64>
    %105 = stablehlo.multiply %78, %92 : tensor<f64>
    %106 = stablehlo.add %104, %105 : tensor<f64>
    %107 = stablehlo.multiply %84, %75 : tensor<f64>
    %108 = stablehlo.subtract %106, %107 : tensor<f64>
    %109 = stablehlo.multiply %90, %80 : tensor<f64>
    %110 = stablehlo.add %108, %109 : tensor<f64>
    %111 = stablehlo.reshape %110 : (tensor<f64>) -> tensor<1xf64>
    %112 = stablehlo.multiply %54, %80 : tensor<f64>
    %113 = stablehlo.multiply %78, %75 : tensor<f64>
    %114 = stablehlo.subtract %112, %113 : tensor<f64>
    %115 = stablehlo.multiply %84, %92 : tensor<f64>
    %116 = stablehlo.subtract %114, %115 : tensor<f64>
    %117 = stablehlo.multiply %90, %86 : tensor<f64>
    %118 = stablehlo.subtract %116, %117 : tensor<f64>
    %119 = stablehlo.reshape %118 : (tensor<f64>) -> tensor<1xf64>
    %120 = stablehlo.concatenate %95, %103, %111, %119, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %121 = stablehlo.slice %120 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %122 = stablehlo.reshape %121 : (tensor<1xf64>) -> tensor<f64>
    %123 = stablehlo.reshape %122 : (tensor<f64>) -> tensor<1xf64>
    %124 = stablehlo.slice %120 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %125 = stablehlo.reshape %124 : (tensor<1xf64>) -> tensor<f64>
    %126 = stablehlo.reshape %125 : (tensor<f64>) -> tensor<1xf64>
    %127 = stablehlo.slice %120 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %128 = stablehlo.reshape %127 : (tensor<1xf64>) -> tensor<f64>
    %129 = stablehlo.reshape %128 : (tensor<f64>) -> tensor<1xf64>
    %130 = stablehlo.concatenate %123, %126, %129, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %131 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %132 = stablehlo.reshape %131 : (tensor<1xf64>) -> tensor<f64>
    %133 = stablehlo.slice %arg1 [3:6] : (tensor<6xf64>) -> tensor<3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %134 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1xf64>
    %135 = stablehlo.concatenate %133, %134, dim = 0 : (tensor<3xf64>, tensor<1xf64>) -> tensor<4xf64>
    %136 = stablehlo.slice %135 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %137 = stablehlo.reshape %136 : (tensor<1xf64>) -> tensor<f64>
    %138 = stablehlo.multiply %132, %137 : tensor<f64>
    %139 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %140 = stablehlo.reshape %139 : (tensor<1xf64>) -> tensor<f64>
    %141 = stablehlo.slice %135 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %142 = stablehlo.reshape %141 : (tensor<1xf64>) -> tensor<f64>
    %143 = stablehlo.multiply %140, %142 : tensor<f64>
    %144 = stablehlo.add %138, %143 : tensor<f64>
    %145 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %146 = stablehlo.reshape %145 : (tensor<1xf64>) -> tensor<f64>
    %147 = stablehlo.slice %135 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %148 = stablehlo.reshape %147 : (tensor<1xf64>) -> tensor<f64>
    %149 = stablehlo.multiply %146, %148 : tensor<f64>
    %150 = stablehlo.add %144, %149 : tensor<f64>
    %151 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %152 = stablehlo.reshape %151 : (tensor<1xf64>) -> tensor<f64>
    %153 = stablehlo.slice %135 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %154 = stablehlo.reshape %153 : (tensor<1xf64>) -> tensor<f64>
    %155 = stablehlo.multiply %152, %154 : tensor<f64>
    %156 = stablehlo.subtract %150, %155 : tensor<f64>
    %157 = stablehlo.reshape %156 : (tensor<f64>) -> tensor<1xf64>
    %158 = stablehlo.multiply %132, %154 : tensor<f64>
    %159 = stablehlo.multiply %140, %148 : tensor<f64>
    %160 = stablehlo.subtract %158, %159 : tensor<f64>
    %161 = stablehlo.multiply %146, %142 : tensor<f64>
    %162 = stablehlo.add %160, %161 : tensor<f64>
    %163 = stablehlo.multiply %152, %137 : tensor<f64>
    %164 = stablehlo.add %162, %163 : tensor<f64>
    %165 = stablehlo.reshape %164 : (tensor<f64>) -> tensor<1xf64>
    %166 = stablehlo.multiply %132, %148 : tensor<f64>
    %167 = stablehlo.multiply %140, %154 : tensor<f64>
    %168 = stablehlo.add %166, %167 : tensor<f64>
    %169 = stablehlo.multiply %146, %137 : tensor<f64>
    %170 = stablehlo.subtract %168, %169 : tensor<f64>
    %171 = stablehlo.multiply %152, %142 : tensor<f64>
    %172 = stablehlo.add %170, %171 : tensor<f64>
    %173 = stablehlo.reshape %172 : (tensor<f64>) -> tensor<1xf64>
    %174 = stablehlo.multiply %132, %142 : tensor<f64>
    %175 = stablehlo.multiply %140, %137 : tensor<f64>
    %176 = stablehlo.subtract %174, %175 : tensor<f64>
    %177 = stablehlo.multiply %146, %154 : tensor<f64>
    %178 = stablehlo.subtract %176, %177 : tensor<f64>
    %179 = stablehlo.multiply %152, %148 : tensor<f64>
    %180 = stablehlo.subtract %178, %179 : tensor<f64>
    %181 = stablehlo.reshape %180 : (tensor<f64>) -> tensor<1xf64>
    %182 = stablehlo.concatenate %157, %165, %173, %181, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %183 = stablehlo.slice %182 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %184 = stablehlo.reshape %183 : (tensor<1xf64>) -> tensor<f64>
    %185 = stablehlo.slice %0 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %186 = stablehlo.reshape %185 : (tensor<1xf64>) -> tensor<f64>
    %187 = stablehlo.negate %186 : tensor<f64>
    %188 = stablehlo.reshape %187 : (tensor<f64>) -> tensor<1xf64>
    %189 = stablehlo.slice %0 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %190 = stablehlo.reshape %189 : (tensor<1xf64>) -> tensor<f64>
    %191 = stablehlo.negate %190 : tensor<f64>
    %192 = stablehlo.reshape %191 : (tensor<f64>) -> tensor<1xf64>
    %193 = stablehlo.slice %0 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %194 = stablehlo.reshape %193 : (tensor<1xf64>) -> tensor<f64>
    %195 = stablehlo.negate %194 : tensor<f64>
    %196 = stablehlo.reshape %195 : (tensor<f64>) -> tensor<1xf64>
    %197 = stablehlo.slice %0 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %198 = stablehlo.reshape %197 : (tensor<1xf64>) -> tensor<f64>
    %199 = stablehlo.reshape %198 : (tensor<f64>) -> tensor<1xf64>
    %200 = stablehlo.concatenate %188, %192, %196, %199, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %201 = stablehlo.dot_general %0, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4xf64>, tensor<4xf64>) -> tensor<f64>
    %202 = stablehlo.broadcast_in_dim %201, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %203 = stablehlo.divide %200, %202 : tensor<4xf64>
    %204 = stablehlo.slice %203 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %205 = stablehlo.reshape %204 : (tensor<1xf64>) -> tensor<f64>
    %206 = stablehlo.multiply %184, %205 : tensor<f64>
    %207 = stablehlo.slice %182 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %208 = stablehlo.reshape %207 : (tensor<1xf64>) -> tensor<f64>
    %209 = stablehlo.slice %203 [3:4] : (tensor<4xf64>) -> tensor<1xf64>
    %210 = stablehlo.reshape %209 : (tensor<1xf64>) -> tensor<f64>
    %211 = stablehlo.multiply %208, %210 : tensor<f64>
    %212 = stablehlo.add %206, %211 : tensor<f64>
    %213 = stablehlo.slice %182 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %214 = stablehlo.reshape %213 : (tensor<1xf64>) -> tensor<f64>
    %215 = stablehlo.slice %203 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %216 = stablehlo.reshape %215 : (tensor<1xf64>) -> tensor<f64>
    %217 = stablehlo.multiply %214, %216 : tensor<f64>
    %218 = stablehlo.add %212, %217 : tensor<f64>
    %219 = stablehlo.slice %182 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %220 = stablehlo.reshape %219 : (tensor<1xf64>) -> tensor<f64>
    %221 = stablehlo.slice %203 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %222 = stablehlo.reshape %221 : (tensor<1xf64>) -> tensor<f64>
    %223 = stablehlo.multiply %220, %222 : tensor<f64>
    %224 = stablehlo.subtract %218, %223 : tensor<f64>
    %225 = stablehlo.reshape %224 : (tensor<f64>) -> tensor<1xf64>
    %226 = stablehlo.multiply %184, %222 : tensor<f64>
    %227 = stablehlo.multiply %208, %216 : tensor<f64>
    %228 = stablehlo.subtract %226, %227 : tensor<f64>
    %229 = stablehlo.multiply %214, %210 : tensor<f64>
    %230 = stablehlo.add %228, %229 : tensor<f64>
    %231 = stablehlo.multiply %220, %205 : tensor<f64>
    %232 = stablehlo.add %230, %231 : tensor<f64>
    %233 = stablehlo.reshape %232 : (tensor<f64>) -> tensor<1xf64>
    %234 = stablehlo.multiply %184, %216 : tensor<f64>
    %235 = stablehlo.multiply %208, %222 : tensor<f64>
    %236 = stablehlo.add %234, %235 : tensor<f64>
    %237 = stablehlo.multiply %214, %205 : tensor<f64>
    %238 = stablehlo.subtract %236, %237 : tensor<f64>
    %239 = stablehlo.multiply %220, %210 : tensor<f64>
    %240 = stablehlo.add %238, %239 : tensor<f64>
    %241 = stablehlo.reshape %240 : (tensor<f64>) -> tensor<1xf64>
    %242 = stablehlo.multiply %184, %210 : tensor<f64>
    %243 = stablehlo.multiply %208, %205 : tensor<f64>
    %244 = stablehlo.subtract %242, %243 : tensor<f64>
    %245 = stablehlo.multiply %214, %222 : tensor<f64>
    %246 = stablehlo.subtract %244, %245 : tensor<f64>
    %247 = stablehlo.multiply %220, %216 : tensor<f64>
    %248 = stablehlo.subtract %246, %247 : tensor<f64>
    %249 = stablehlo.reshape %248 : (tensor<f64>) -> tensor<1xf64>
    %250 = stablehlo.concatenate %225, %233, %241, %249, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<4xf64>
    %251 = stablehlo.slice %250 [0:1] : (tensor<4xf64>) -> tensor<1xf64>
    %252 = stablehlo.reshape %251 : (tensor<1xf64>) -> tensor<f64>
    %253 = stablehlo.reshape %252 : (tensor<f64>) -> tensor<1xf64>
    %254 = stablehlo.slice %250 [1:2] : (tensor<4xf64>) -> tensor<1xf64>
    %255 = stablehlo.reshape %254 : (tensor<1xf64>) -> tensor<f64>
    %256 = stablehlo.reshape %255 : (tensor<f64>) -> tensor<1xf64>
    %257 = stablehlo.slice %250 [2:3] : (tensor<4xf64>) -> tensor<1xf64>
    %258 = stablehlo.reshape %257 : (tensor<1xf64>) -> tensor<f64>
    %259 = stablehlo.reshape %258 : (tensor<f64>) -> tensor<1xf64>
    %260 = stablehlo.concatenate %253, %256, %259, dim = 0 : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<3xf64>
    %261 = stablehlo.concatenate %130, %260, dim = 0 : (tensor<3xf64>, tensor<3xf64>) -> tensor<6xf64>
    %262 = stablehlo.add %arg2, %261 : tensor<6xf64>
    return %262 : tensor<6xf64>
  }
}
