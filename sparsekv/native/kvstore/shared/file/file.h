#ifndef UNIFIEDCACHE_FILE_H
#define UNIFIEDCACHE_FILE_H

#include <memory>
#include "ifile.h"

namespace UC {

class File {
public:
    static std::unique_ptr<IFile> Make(const std::string& path);
    static Status MkDir(const std::string& path);
    static Status RmDir(const std::string& path);
    static Status Rename(const std::string& path, const std::string& newName);
    static Status Access(const std::string& path, const int32_t mode);
    static Status Stat(const std::string& path, IFile::FileStat& st);
    static Status Read(const std::string& path, const size_t offset, const size_t length,
                       uintptr_t address, const bool directIo = false);
    static Status Write(const std::string& path, const size_t offset, const size_t length,
                        const uintptr_t address, const bool directIo = false,
                        const bool create = false);
    static void MUnmap(void* addr, size_t size);
    static void ShmUnlink(const std::string& path);
    static void Remove(const std::string& path);
};

} // namespace UC

#endif
