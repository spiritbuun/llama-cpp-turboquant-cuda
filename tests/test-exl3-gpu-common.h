#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// Probe the backend's standalone executor, not the independent CPU/cache bridge.
// In particular, HIP wave64 devices intentionally decline this wave32 path.
static bool exl3_gpu_supported(ggml_backend_t backend) {
    auto * ctx = ggml_init({4096, nullptr, true});
    auto * w = ggml_new_tensor_2d(ctx, ggml_exl3_type(2, 2), 128, 128);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
    auto * y = ggml_mul_mat(ctx, w, x);
    const bool supported = ggml_backend_supports_op(backend, y);
    ggml_free(ctx);
    return supported;
}
