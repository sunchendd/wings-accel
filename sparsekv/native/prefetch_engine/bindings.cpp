#pragma GCC diagnostic push
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#pragma GCC diagnostic pop

#include "kvcache_pre.h"

namespace bmsaprefetch {

PYBIND11_MODULE(_prefetch_engine, m)
{
    pybind11::class_<bmsaprefetch::BMSAPrefetchEngineC>(m, "BMSAPrefetchEngineC")
        .def(pybind11::init<torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
                            std::vector<uint32_t>&, bool, bool, int, int, int, bool>())
        .def("set_blocks_map", &bmsaprefetch::BMSAPrefetchEngineC::SetBlocksMap)
        .def("set_blocks_map_multilayer",
             &bmsaprefetch::BMSAPrefetchEngineC::SetBlocksMapMultiLayer)
        .def("add_blocks_map", &bmsaprefetch::BMSAPrefetchEngineC::AddBlocksMap)
        .def("del_blocks_map", &bmsaprefetch::BMSAPrefetchEngineC::DelBlocksMap)
        .def("run_async_prefetch_bs", &bmsaprefetch::BMSAPrefetchEngineC::RunAsyncPrefetchBs)
        .def("set_blocks_table_info", &bmsaprefetch::BMSAPrefetchEngineC::SetBlockTableInfo)
        .def("get_prefetch_status", &bmsaprefetch::BMSAPrefetchEngineC::GetPrefetchStatus)
        .def("set_prefetch_status", &bmsaprefetch::BMSAPrefetchEngineC::SetPrefetchStatus)
        .def("set_modelrunning_status",
             &bmsaprefetch::BMSAPrefetchEngineC::SetModelRunningStatus)
        .def("obtain_load_blocks", &bmsaprefetch::BMSAPrefetchEngineC::ObtainLoadBlocks)
        .def("obtain_miss_idxs", &bmsaprefetch::BMSAPrefetchEngineC::ObtainMissIdxs)
        .def("obtain_docs_map", &bmsaprefetch::BMSAPrefetchEngineC::ObtainDocsMap)
        .def("obtain_blocks_map", &bmsaprefetch::BMSAPrefetchEngineC::ObtainBlocksMap);
}

} // namespace bmsaprefetch
