#define STARPU_SIMGRID
#define STARPU_MAXIMPLEMENTATIONS 4
#define STARPU_NMAXBUFS 8
#define STARPU_MAXNODES 2
#define STARPU_NMAXWORKERS 16

#ifndef _MSC_VER
#include <stdint.h>
#else
#include <windows.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef UINT_PTR uintptr_t;
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef INT_PTR intptr_t;
#endif

#ifdef _MSC_VER
typedef long starpu_ssize_t;
#define __starpu_func__ __FUNCTION__
#else
#  include <sys/types.h>
typedef ssize_t starpu_ssize_t;
#define __starpu_func__ __func__
#endif

#if defined(c_plusplus) || defined(__cplusplus)
/* inline is part of C++ */
#  define __starpu_inline inline
#elif defined(_MSC_VER) || defined(__HP_cc)
#  define __starpu_inline __inline
#else
#  define __starpu_inline __inline__
#endif

