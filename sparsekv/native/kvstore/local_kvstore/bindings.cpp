#include <pybind11/pybind11.h>
#include "localstore.h"

namespace py = pybind11;

namespace UC {

#ifndef SPARSE_PROJECT_NAME
#define SPARSE_PROJECT_NAME "vllm"  // NOLINT: conditional compile override
#endif
#ifndef SPARSE_PROJECT_VERSION
#define SPARSE_PROJECT_VERSION "0"  // NOLINT: conditional compile override
#endif
#ifndef SPARSE_COMMIT_ID
#define SPARSE_COMMIT_ID ""  // NOLINT: conditional compile override
#endif
#ifndef SPARSE_BUILD_TYPE
#define SPARSE_BUILD_TYPE ""  // NOLINT: conditional compile override
#endif

class LocalStorePy : public LocalStore {
public:
    void* CCStoreImpl() { return this; }
    py::list AllocBatch(const py::list& blocks)
    {
        py::list results;
        for (auto& block : blocks) { results.append(this->Alloc(block.cast<std::string>())); }
        return results;
    }
    py::list LookupBatch(const py::list& blocks)
    {
        py::list founds;
        for (auto& block : blocks) { founds.append(this->Lookup(block.cast<std::string>())); }
        return founds;
    }
    void CommitBatch(const py::list& blocks, const bool success)
    {
        for (auto& block : blocks) { this->Commit(block.cast<std::string>(), success); }
    }
    py::tuple CheckPy(const size_t task)
    {
        auto finish = false;
        auto ret = this->Check(task, finish);
        return py::make_tuple(ret, finish);
    }
    size_t Load(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, Task::Type::LOAD,
                              Task::Location::DEVICE, "LOCAL::S2D");
    }
    size_t Dump(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                const py::list& lengths)
    {
        return this->SubmitPy(blockIds, offsets, addresses, lengths, Task::Type::DUMP,
                              Task::Location::DEVICE, "LOCAL::D2S");
    }

private:
    size_t SubmitPy(const py::list& blockIds, const py::list& offsets, const py::list& addresses,
                    const py::list& lengths, Task::Type&& type, Task::Location&& location,
                    std::string&& brief)
    {
        Task task{std::move(type), std::move(location), std::move(brief)};
        auto blockId = blockIds.begin();
        auto offset = offsets.begin();
        auto address = addresses.begin();
        auto length = lengths.begin();
        while ((blockId != blockIds.end()) && (offset != offsets.end()) &&
               (address != addresses.end()) && (length != lengths.end())) {
            task.Append(blockId->cast<std::string>(), offset->cast<size_t>(),
                        address->cast<uintptr_t>(), length->cast<size_t>());
            blockId++;
            offset++;
            address++;
            length++;
        }
        return this->Submit(std::move(task));
    }
};

} // namespace UC

PYBIND11_MODULE(_local_kvstore, module)
{
    module.attr("project") = SPARSE_PROJECT_NAME;
    module.attr("version") = SPARSE_PROJECT_VERSION;
    module.attr("commit_id") = SPARSE_COMMIT_ID;
    module.attr("build_type") = SPARSE_BUILD_TYPE;
    auto store = py::class_<UC::LocalStorePy>(module, "LocalStore");
    store.def(py::init<>());
    auto config = py::class_<UC::LocalStorePy::Config>(store, "Config");
    config.def(py::init<const size_t, const size_t>(), py::arg("ioSize"), py::arg("capacity"));
    config.def_readwrite("ioSize", &UC::LocalStorePy::Config::ioSize);
    config.def_readwrite("capacity", &UC::LocalStorePy::Config::capacity);
    config.def_readwrite("backend", &UC::LocalStorePy::Config::backend);
    config.def_readwrite("deviceId", &UC::LocalStorePy::Config::deviceId);
    store.def("CCStoreImpl", &UC::LocalStorePy::CCStoreImpl);
    store.def("Setup", &UC::LocalStorePy::Setup);
    store.def("Alloc", py::overload_cast<const std::string&>(&UC::LocalStorePy::Alloc));
    store.def("AllocBatch", &UC::LocalStorePy::AllocBatch);
    store.def("Lookup", py::overload_cast<const std::string&>(&UC::LocalStorePy::Lookup));
    store.def("LookupBatch", &UC::LocalStorePy::LookupBatch);
    store.def("Load", &UC::LocalStorePy::Load);
    store.def("Dump", &UC::LocalStorePy::Dump);
    store.def("Wait", &UC::LocalStorePy::Wait);
    store.def("CheckPy", &UC::LocalStorePy::CheckPy);
    store.def("Check", &UC::LocalStorePy::Check);
    store.def("Commit",
              py::overload_cast<const std::string&, const bool>(&UC::LocalStorePy::Commit));
    store.def("CommitBatch", &UC::LocalStorePy::CommitBatch);
}
