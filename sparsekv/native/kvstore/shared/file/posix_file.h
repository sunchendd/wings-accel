#ifndef UNIFIEDCACHE_POSIX_FILE_H
#define UNIFIEDCACHE_POSIX_FILE_H

#include "ifile.h"

namespace UC {

class PosixFile : public IFile {
public:
    PosixFile(const std::string& path) : IFile{path}, handle_{-1} {}
    ~PosixFile() override;
    Status MkDir() override;
    Status RmDir() override;
    Status Rename(const std::string& newName) override;
    Status Access(const int32_t mode) override;
    Status Open(const uint32_t flags) override;
    void Close() override;
    void Remove() override;
    Status Read(void* buffer, size_t size, off64_t offset = -1) override;
    Status Write(const void* buffer, size_t size, off64_t offset = -1) override;
    Status Truncate(size_t length) override;
    Status Stat(FileStat& st) override;
    Status ShmOpen(const uint32_t flags) override;
    Status MMap(void*& addr, size_t size, bool write, bool read, bool shared) override;
    void MUnmap(void* addr, size_t size) override;
    void ShmUnlink() override;
    Status UpdateTime() override;

private:
    int32_t handle_;
};

} // namespace UC

#endif // UNIFIEDCACHE_POSIX_FILE_H
