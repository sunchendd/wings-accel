#include <torch/extension.h>

void batch_gemm_softmax(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor D,
    torch::Tensor Norm,
    torch::Tensor Sum,
    torch::Tensor Softmax,
    int batchCount,
    int m,
    int n,
    int k,
    float alpha,
    float beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "batch_gemm_softmax",
        &batch_gemm_softmax,
        "Batch GEMM Softmax (CUDA)",
        py::arg("A"),
        py::arg("B"),
        py::arg("D"),
        py::arg("Norm"),
        py::arg("Sum"),
        py::arg("Softmax"),
        py::arg("batchCount"),
        py::arg("m"),
        py::arg("n"),
        py::arg("k"),
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f);
}
