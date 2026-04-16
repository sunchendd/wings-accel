#ifndef UNIFIEDCACHE_INFRA_SINGLETON_H
#define UNIFIEDCACHE_INFRA_SINGLETON_H

namespace UC {

template <typename T>
class Singleton {
public:
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    static T* Instance()
    {
        static T t;
        return &t;
    }

private:
    Singleton() = default;
};

} // namespace UC

#endif
