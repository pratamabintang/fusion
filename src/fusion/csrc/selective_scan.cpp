#include "selective_scan.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan_fwd_cuda", &selective_scan_fwd_cuda, "Selective scan forward (CUDA)");
    m.def("selective_scan_bwd_cuda", &selective_scan_bwd_cuda, "Selective scan backward (CUDA)");
}
