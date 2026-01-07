#include <QiVision/Platform/Memory.h>

#include <cstdlib>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace Qi::Vision::Platform {

void* AlignedAlloc(size_t size, size_t alignment) {
    if (size == 0) return nullptr;

#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void AlignedFree(void* ptr) {
    if (ptr == nullptr) return;

#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace Qi::Vision::Platform
