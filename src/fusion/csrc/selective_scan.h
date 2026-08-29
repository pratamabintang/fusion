#pragma once
#include <torch/extension.h>
#include <vector>

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
);

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
);
