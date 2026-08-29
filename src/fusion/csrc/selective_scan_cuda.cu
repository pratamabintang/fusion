#include <torch/extension.h>
#include <vector>
#include "selective_scan.h"

// Dummy implementation since this is just packaging/TDD
std::vector<torch::Tensor> selective_scan_fwd_cuda(
    const torch::Tensor &u,
    const torch::Tensor &delta,
    const torch::Tensor &A,
    const torch::Tensor &B,
    const torch::Tensor &C,
    const c10::optional<torch::Tensor> &D,
    const c10::optional<torch::Tensor> &z,
    const c10::optional<torch::Tensor> &delta_bias,
    bool delta_softplus
) {
    return {u, u, u, u, u, u};
}

std::vector<torch::Tensor> selective_scan_bwd_cuda(
    const torch::Tensor &u,
    const torch::Tensor &delta,
    const torch::Tensor &A,
    const torch::Tensor &B,
    const torch::Tensor &C,
    const c10::optional<torch::Tensor> &D,
    const c10::optional<torch::Tensor> &z,
    const c10::optional<torch::Tensor> &delta_bias,
    const torch::Tensor &dout,
    const c10::optional<torch::Tensor> &x,
    const c10::optional<torch::Tensor> &out,
    bool delta_softplus
) {
    return {u, delta, A, B, C, u, u, u};
}
