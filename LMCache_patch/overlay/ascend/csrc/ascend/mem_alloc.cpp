#include "mem_alloc.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

void* checked_malloc(size_t size) {
  void* ptr = std::malloc(size);
  if (ptr == nullptr) {
    throw std::runtime_error("malloc failed");
  }
  std::memset(ptr, 0, size);
  return ptr;
}

}  // namespace

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  (void)flags;
  return reinterpret_cast<uintptr_t>(checked_malloc(size));
}

uintptr_t alloc_numa_ptr(size_t size, int node) {
  (void)node;
  return reinterpret_cast<uintptr_t>(checked_malloc(size));
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  (void)node;
  return reinterpret_cast<uintptr_t>(checked_malloc(size));
}

uintptr_t alloc_shm_pinned_ptr(size_t size, const std::string& shm_name) {
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0) {
    throw std::runtime_error("shm_open failed");
  }
  if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
    close(fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error("ftruncate failed");
  }
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error("mmap failed");
  }
  std::memset(ptr, 0, size);
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  std::free(reinterpret_cast<void*>(ptr));
}

void free_numa_ptr(uintptr_t ptr, size_t size) {
  (void)size;
  std::free(reinterpret_cast<void*>(ptr));
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  (void)size;
  std::free(reinterpret_cast<void*>(ptr));
}

void free_shm_pinned_ptr(uintptr_t ptr, size_t size, const std::string& shm_name) {
  munmap(reinterpret_cast<void*>(ptr), size);
  shm_unlink(shm_name.c_str());
}
