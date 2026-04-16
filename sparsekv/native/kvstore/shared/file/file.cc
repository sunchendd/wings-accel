#include "file.h"
#include "logger/logger.h"
#include "posix_file.h"

namespace UC {

using FileImpl = PosixFile;

std::unique_ptr<IFile> File::Make(const std::string& path)
{
    try {
        return std::make_unique<FileImpl>(path);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make file({}) pointer.", e.what(), path);
        return nullptr;
    }
}

Status File::MkDir(const std::string& path) { return FileImpl{path}.MkDir(); }

Status File::RmDir(const std::string& path) { return FileImpl{path}.RmDir(); }

Status File::Rename(const std::string& path, const std::string& newName)
{
    return FileImpl{path}.Rename(newName);
}

Status File::Access(const std::string& path, const int32_t mode)
{
    return FileImpl{path}.Access(mode);
}

Status File::Stat(const std::string& path, IFile::FileStat& st)
{
    FileImpl file{path};
    auto status = file.Open(IFile::OpenFlag::READ_ONLY);
    if (status.Failure()) { return status; }
    status = file.Stat(st);
    file.Close();
    return status;
}

Status File::Read(const std::string& path, const size_t offset, const size_t length,
                  uintptr_t address, const bool directIo)
{
    FileImpl file{path};
    Status status = Status::OK();
    auto flags = directIo ? IFile::OpenFlag::READ_ONLY | IFile::OpenFlag::DIRECT
                          : IFile::OpenFlag::READ_ONLY;
    if ((status = file.Open(flags)).Failure()) { return status; }
    if ((status = file.Read((void*)address, length, offset)).Failure()) { return status; }
    return status;
}

Status File::Write(const std::string& path, const size_t offset, const size_t length,
                   const uintptr_t address, const bool directIo, const bool create)
{
    FileImpl file{path};
    Status status = Status::OK();
    auto flags = IFile::OpenFlag::WRITE_ONLY;
    if (directIo) { flags |= IFile::OpenFlag::DIRECT; }
    if (create) { flags |= IFile::OpenFlag::CREATE; }
    if ((status = file.Open(flags)).Failure()) { return status; }
    if ((status = file.Write((const void*)address, length, offset)).Failure()) { return status; }
    return status;
}

void File::MUnmap(void* addr, size_t size) { FileImpl{{}}.MUnmap(addr, size); }

void File::ShmUnlink(const std::string& path) { FileImpl{path}.ShmUnlink(); }

void File::Remove(const std::string& path) { FileImpl{path}.Remove(); }

} // namespace UC
