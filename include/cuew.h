/*
 * Copyright 2011-2014 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef __CUEW_H__
#define __CUEW_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

/* Defines. */
#define CUEW_VERSION_MAJOR 2
#define CUEW_VERSION_MINOR 0

#define CUDA_VERSION 9020
#define CU_IPC_HANDLE_SIZE 64
#define CU_STREAM_LEGACY ((CUstream)0x1)
#define CU_STREAM_PER_THREAD ((CUstream)0x2)
#define CU_MEMHOSTALLOC_PORTABLE 0x01
#define CU_MEMHOSTALLOC_DEVICEMAP 0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x04
#define CU_MEMHOSTREGISTER_PORTABLE 0x01
#define CU_MEMHOSTREGISTER_DEVICEMAP 0x02
#define CU_MEMHOSTREGISTER_IOMEMORY 0x04
#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC 0x01
#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC 0x02
#define CUDA_ARRAY3D_LAYERED 0x01
#define CUDA_ARRAY3D_2DARRAY 0x01
#define CUDA_ARRAY3D_SURFACE_LDST 0x02
#define CUDA_ARRAY3D_CUBEMAP 0x04
#define CUDA_ARRAY3D_TEXTURE_GATHER 0x08
#define CUDA_ARRAY3D_DEPTH_TEXTURE 0x10
#define CU_TRSA_OVERRIDE_FORMAT 0x01
#define CU_TRSF_READ_AS_INTEGER 0x01
#define CU_TRSF_NORMALIZED_COORDINATES 0x02
#define CU_TRSF_SRGB 0x10
#define CU_LAUNCH_PARAM_END ((void*)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define CU_PARAM_TR_DEFAULT -1
#define CU_DEVICE_CPU ((CUdevice)-1)
#define CU_DEVICE_INVALID ((CUdevice)-2)
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 1
#define CUDNN_PATCHLEVEL 4
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
#define CUDNN_DIM_MAX 8
#define CUDNN_LRN_MIN_N 1 /* minimum allowed lrnN */
#define CUDNN_LRN_MAX_N 16 /* maximum allowed lrnN */
#define CUDNN_LRN_MIN_K 1e-5 /* minimum allowed lrnK */
#define CUDNN_LRN_MIN_BETA 0.01 /* minimum allowed lrnBeta */
#define CUDNN_BN_MIN_EPSILON 1e-5 /* Minimum epsilon allowed to be used in the Batch Normalization formula */
#define CUDNN_SEV_ERROR_EN (1U << CUDNN_SEV_ERROR)
#define CUDNN_SEV_WARNING_EN (1U << CUDNN_SEV_WARNING)
#define CUDNN_SEV_INFO_EN (1U << CUDNN_SEV_INFO)

/* Functions which changed 3.1 -> 3.2 for 64 bit stuff,
 * the cuda library has both the old ones for compatibility and new
 * ones with _v2 postfix,
 */
#define cuDeviceTotalMem cuDeviceTotalMem_v2
#define cuCtxCreate cuCtxCreate_v2
#define cuModuleGetGlobal cuModuleGetGlobal_v2
#define cuMemGetInfo cuMemGetInfo_v2
#define cuMemAlloc cuMemAlloc_v2
#define cuMemAllocPitch cuMemAllocPitch_v2
#define cuMemFree cuMemFree_v2
#define cuMemGetAddressRange cuMemGetAddressRange_v2
#define cuMemAllocHost cuMemAllocHost_v2
#define cuMemHostGetDevicePointer cuMemHostGetDevicePointer_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemcpyDtoH cuMemcpyDtoH_v2
#define cuMemcpyDtoD cuMemcpyDtoD_v2
#define cuMemcpyDtoA cuMemcpyDtoA_v2
#define cuMemcpyAtoD cuMemcpyAtoD_v2
#define cuMemcpyHtoA cuMemcpyHtoA_v2
#define cuMemcpyAtoH cuMemcpyAtoH_v2
#define cuMemcpyAtoA cuMemcpyAtoA_v2
#define cuMemcpyHtoAAsync cuMemcpyHtoAAsync_v2
#define cuMemcpyAtoHAsync cuMemcpyAtoHAsync_v2
#define cuMemcpy2D cuMemcpy2D_v2
#define cuMemcpy2DUnaligned cuMemcpy2DUnaligned_v2
#define cuMemcpy3D cuMemcpy3D_v2
#define cuMemcpyHtoDAsync cuMemcpyHtoDAsync_v2
#define cuMemcpyDtoHAsync cuMemcpyDtoHAsync_v2
#define cuMemcpyDtoDAsync cuMemcpyDtoDAsync_v2
#define cuMemcpy2DAsync cuMemcpy2DAsync_v2
#define cuMemcpy3DAsync cuMemcpy3DAsync_v2
#define cuMemsetD8 cuMemsetD8_v2
#define cuMemsetD16 cuMemsetD16_v2
#define cuMemsetD32 cuMemsetD32_v2
#define cuMemsetD2D8 cuMemsetD2D8_v2
#define cuMemsetD2D16 cuMemsetD2D16_v2
#define cuMemsetD2D32 cuMemsetD2D32_v2
#define cuArrayCreate cuArrayCreate_v2
#define cuArrayGetDescriptor cuArrayGetDescriptor_v2
#define cuArray3DCreate cuArray3DCreate_v2
#define cuArray3DGetDescriptor cuArray3DGetDescriptor_v2
#define cuTexRefSetAddress cuTexRefSetAddress_v2
#define cuTexRefGetAddress cuTexRefGetAddress_v2
#define cuGraphicsResourceGetMappedPointer cuGraphicsResourceGetMappedPointer_v2
#define cuCtxDestroy cuCtxDestroy_v2
#define cuCtxPopCurrent cuCtxPopCurrent_v2
#define cuCtxPushCurrent cuCtxPushCurrent_v2
#define cuStreamDestroy cuStreamDestroy_v2
#define cuEventDestroy cuEventDestroy_v2
#define cuLinkCreate cuLinkCreate_v2
#define cuLinkAddData cuLinkAddData_v2
#define cuLinkAddFile cuLinkAddFile_v2
#define cuMemHostRegister cuMemHostRegister_v2
#define cuGraphicsResourceSetMapFlags cuGraphicsResourceSetMapFlags_v2
#define cuTexRefSetAddress2D cuTexRefSetAddress2D_v2
#define cuGLCtxCreate cuGLCtxCreate_v2
#define cuGLMapBufferObject cuGLMapBufferObject_v2
#define cuGLMapBufferObjectAsync cuGLMapBufferObjectAsync_v2
#define cuGLGetDevices cuGLGetDevices_v2

/* Types. */
#ifdef _MSC_VER
typedef unsigned __int32 cuuint32_t;
typedef unsigned __int64 cuuint64_t;
#else
#include <stdint.h>
typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;
#endif

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined (__aarch64__)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

#ifdef _WIN32
#  define CUDAAPI __stdcall
#  define CUDA_CB __stdcall
#else
#  define CUDAAPI
#  define CUDA_CB
#endif

#if !defined(__CUDACC__)
#  define __device_builtin__
#else
#  define __device_builtin__ __location__(device_builtin)
#endif

typedef __device_builtin__ struct CUstream_st *cudaStream_t;

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUarray_st* CUarray;
typedef struct CUmipmappedArray_st* CUmipmappedArray;
typedef struct CUtexref_st* CUtexref;
typedef struct CUsurfref_st* CUsurfref;
typedef struct CUevent_st* CUevent;
typedef struct CUstream_st* CUstream;
typedef struct CUgraphicsResource_st* CUgraphicsResource;
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

typedef struct CUuuid_st {
  char bytes[16];
} CUuuid;

typedef struct CUipcEventHandle_st {
  char reserved[CU_IPC_HANDLE_SIZE];
} CUipcEventHandle;

typedef struct CUipcMemHandle_st {
  char reserved[CU_IPC_HANDLE_SIZE];
} CUipcMemHandle;

typedef enum CUipcMem_flags_enum {
  CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1,
} CUipcMem_flags;

typedef enum CUmemAttach_flags_enum {
  CU_MEM_ATTACH_GLOBAL = 0x1,
  CU_MEM_ATTACH_HOST = 0x2,
  CU_MEM_ATTACH_SINGLE = 0x4,
} CUmemAttach_flags;

typedef enum CUctx_flags_enum {
  CU_CTX_SCHED_AUTO = 0x00,
  CU_CTX_SCHED_SPIN = 0x01,
  CU_CTX_SCHED_YIELD = 0x02,
  CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
  CU_CTX_BLOCKING_SYNC = 0x04,
  CU_CTX_SCHED_MASK = 0x07,
  CU_CTX_MAP_HOST = 0x08,
  CU_CTX_LMEM_RESIZE_TO_MAX = 0x10,
  CU_CTX_FLAGS_MASK = 0x1f,
} CUctx_flags;

typedef enum CUstream_flags_enum {
  CU_STREAM_DEFAULT = 0x0,
  CU_STREAM_NON_BLOCKING = 0x1,
} CUstream_flags;

typedef enum CUevent_flags_enum {
  CU_EVENT_DEFAULT = 0x0,
  CU_EVENT_BLOCKING_SYNC = 0x1,
  CU_EVENT_DISABLE_TIMING = 0x2,
  CU_EVENT_INTERPROCESS = 0x4,
} CUevent_flags;

typedef enum CUstreamWaitValue_flags_enum {
  CU_STREAM_WAIT_VALUE_GEQ = 0x0,
  CU_STREAM_WAIT_VALUE_EQ = 0x1,
  CU_STREAM_WAIT_VALUE_AND = 0x2,
  CU_STREAM_WAIT_VALUE_NOR = 0x3,
  CU_STREAM_WAIT_VALUE_FLUSH = (1 << 30),
} CUstreamWaitValue_flags;

typedef enum CUstreamWriteValue_flags_enum {
  CU_STREAM_WRITE_VALUE_DEFAULT = 0x0,
  CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1,
} CUstreamWriteValue_flags;

typedef enum CUstreamBatchMemOpType_enum {
  CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
  CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
  CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
  CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
  CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
} CUstreamBatchMemOpType;

typedef union CUstreamBatchMemOpParams_union {
  CUstreamBatchMemOpType operation;
  struct CUstreamMemOpWaitValueParams_st {
    CUstreamBatchMemOpType operation;
    CUdeviceptr address;
    union {
      cuuint32_t value;
      cuuint64_t value64;
    };
    unsigned int flags;
    CUdeviceptr alias;
  } waitValue;
  struct CUstreamMemOpWriteValueParams_st {
    CUstreamBatchMemOpType operation;
    CUdeviceptr address;
    union {
      cuuint32_t value;
      cuuint64_t value64;
    };
    unsigned int flags;
    CUdeviceptr alias;
  } writeValue;
  struct CUstreamMemOpFlushRemoteWritesParams_st {
    CUstreamBatchMemOpType operation;
    unsigned int flags;
  } flushRemoteWrites;
  cuuint64_t pad[6];
} CUstreamBatchMemOpParams;

typedef enum CUoccupancy_flags_enum {
  CU_OCCUPANCY_DEFAULT = 0x0,
  CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1,
} CUoccupancy_flags;

typedef enum CUarray_format_enum {
  CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
  CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
  CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
  CU_AD_FORMAT_SIGNED_INT8 = 0x08,
  CU_AD_FORMAT_SIGNED_INT16 = 0x09,
  CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
  CU_AD_FORMAT_HALF = 0x10,
  CU_AD_FORMAT_FLOAT = 0x20,
} CUarray_format;

typedef enum CUaddress_mode_enum {
  CU_TR_ADDRESS_MODE_WRAP = 0,
  CU_TR_ADDRESS_MODE_CLAMP = 1,
  CU_TR_ADDRESS_MODE_MIRROR = 2,
  CU_TR_ADDRESS_MODE_BORDER = 3,
} CUaddress_mode;

typedef enum CUfilter_mode_enum {
  CU_TR_FILTER_MODE_POINT = 0,
  CU_TR_FILTER_MODE_LINEAR = 1,
} CUfilter_mode;

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
  CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
  CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
  CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
  CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
  CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
  CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
  CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
  CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
  CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
  CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
  CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
  CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
  CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
  CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
  CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
  CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
  CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
  CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
  CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
  CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
  CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
  CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
  CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
  CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
  CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
  CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
  CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
  CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
  CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
  CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
  CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
  CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
  CU_DEVICE_ATTRIBUTE_MAX,
} CUdevice_attribute;

typedef struct CUdevprop_st {
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int sharedMemPerBlock;
  int totalConstantMemory;
  int SIMDWidth;
  int memPitch;
  int regsPerBlock;
  int clockRate;
  int textureAlign;
} CUdevprop;

typedef enum CUpointer_attribute_enum {
  CU_POINTER_ATTRIBUTE_CONTEXT = 1,
  CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
  CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
  CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
  CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
  CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
  CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
  CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
} CUpointer_attribute;

typedef enum CUfunction_attribute_enum {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  CU_FUNC_ATTRIBUTE_MAX,
} CUfunction_attribute;

typedef enum CUfunc_cache_enum {
  CU_FUNC_CACHE_PREFER_NONE = 0x00,
  CU_FUNC_CACHE_PREFER_SHARED = 0x01,
  CU_FUNC_CACHE_PREFER_L1 = 0x02,
  CU_FUNC_CACHE_PREFER_EQUAL = 0x03,
} CUfunc_cache;

typedef enum CUsharedconfig_enum {
  CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00,
  CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01,
  CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02,
} CUsharedconfig;

typedef enum CUshared_carveout_enum {
  CU_SHAREDMEM_CARVEOUT_DEFAULT,
  CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100,
  CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0,
} CUshared_carveout;

typedef enum CUmemorytype_enum {
  CU_MEMORYTYPE_HOST = 0x01,
  CU_MEMORYTYPE_DEVICE = 0x02,
  CU_MEMORYTYPE_ARRAY = 0x03,
  CU_MEMORYTYPE_UNIFIED = 0x04,
} CUmemorytype;

typedef enum CUcomputemode_enum {
  CU_COMPUTEMODE_DEFAULT = 0,
  CU_COMPUTEMODE_PROHIBITED = 2,
  CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,
} CUcomputemode;

typedef enum CUmem_advise_enum {
  CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
  CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
  CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
  CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
  CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
  CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6,
} CUmem_advise;

typedef enum CUmem_range_attribute_enum {
  CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
  CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
  CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
  CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
} CUmem_range_attribute;

typedef enum CUjit_option_enum {
  CU_JIT_MAX_REGISTERS = 0,
  CU_JIT_THREADS_PER_BLOCK,
  CU_JIT_WALL_TIME,
  CU_JIT_INFO_LOG_BUFFER,
  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
  CU_JIT_ERROR_LOG_BUFFER,
  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  CU_JIT_OPTIMIZATION_LEVEL,
  CU_JIT_TARGET_FROM_CUCONTEXT,
  CU_JIT_TARGET,
  CU_JIT_FALLBACK_STRATEGY,
  CU_JIT_GENERATE_DEBUG_INFO,
  CU_JIT_LOG_VERBOSE,
  CU_JIT_GENERATE_LINE_INFO,
  CU_JIT_CACHE_MODE,
  CU_JIT_NEW_SM3X_OPT,
  CU_JIT_FAST_COMPILE,
  CU_JIT_NUM_OPTIONS,
} CUjit_option;

typedef enum CUjit_target_enum {
  CU_TARGET_COMPUTE_20 = 20,
  CU_TARGET_COMPUTE_21 = 21,
  CU_TARGET_COMPUTE_30 = 30,
  CU_TARGET_COMPUTE_32 = 32,
  CU_TARGET_COMPUTE_35 = 35,
  CU_TARGET_COMPUTE_37 = 37,
  CU_TARGET_COMPUTE_50 = 50,
  CU_TARGET_COMPUTE_52 = 52,
  CU_TARGET_COMPUTE_53 = 53,
  CU_TARGET_COMPUTE_60 = 60,
  CU_TARGET_COMPUTE_61 = 61,
  CU_TARGET_COMPUTE_62 = 62,
  CU_TARGET_COMPUTE_70 = 70,
  CU_TARGET_COMPUTE_73 = 73,
  CU_TARGET_COMPUTE_75 = 75,
} CUjit_target;

typedef enum CUjit_fallback_enum {
  CU_PREFER_PTX = 0,
  CU_PREFER_BINARY,
} CUjit_fallback;

typedef enum CUjit_cacheMode_enum {
  CU_JIT_CACHE_OPTION_NONE = 0,
  CU_JIT_CACHE_OPTION_CG,
  CU_JIT_CACHE_OPTION_CA,
} CUjit_cacheMode;

typedef enum CUjitInputType_enum {
  CU_JIT_INPUT_CUBIN = 0,
  CU_JIT_INPUT_PTX,
  CU_JIT_INPUT_FATBINARY,
  CU_JIT_INPUT_OBJECT,
  CU_JIT_INPUT_LIBRARY,
  CU_JIT_NUM_INPUT_TYPES,
} CUjitInputType;

typedef struct CUlinkState_st* CUlinkState;

typedef enum CUgraphicsRegisterFlags_enum {
  CU_GRAPHICS_REGISTER_FLAGS_NONE = 0x00,
  CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 0x01,
  CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02,
  CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04,
  CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08,
} CUgraphicsRegisterFlags;

typedef enum CUgraphicsMapResourceFlags_enum {
  CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00,
  CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01,
  CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02,
} CUgraphicsMapResourceFlags;

typedef enum CUarray_cubemap_face_enum {
  CU_CUBEMAP_FACE_POSITIVE_X = 0x00,
  CU_CUBEMAP_FACE_NEGATIVE_X = 0x01,
  CU_CUBEMAP_FACE_POSITIVE_Y = 0x02,
  CU_CUBEMAP_FACE_NEGATIVE_Y = 0x03,
  CU_CUBEMAP_FACE_POSITIVE_Z = 0x04,
  CU_CUBEMAP_FACE_NEGATIVE_Z = 0x05,
} CUarray_cubemap_face;

typedef enum CUlimit_enum {
  CU_LIMIT_STACK_SIZE = 0x00,
  CU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
  CU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
  CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
  CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
  CU_LIMIT_MAX,
} CUlimit;

typedef enum CUresourcetype_enum {
  CU_RESOURCE_TYPE_ARRAY = 0x00,
  CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
  CU_RESOURCE_TYPE_LINEAR = 0x02,
  CU_RESOURCE_TYPE_PITCH2D = 0x03,
} CUresourcetype;

typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_PROFILER_DISABLED = 5,
  CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
  CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
  CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
  CUDA_ERROR_NO_DEVICE = 100,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_INVALID_IMAGE = 200,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
  CUDA_ERROR_MAP_FAILED = 205,
  CUDA_ERROR_UNMAP_FAILED = 206,
  CUDA_ERROR_ARRAY_IS_MAPPED = 207,
  CUDA_ERROR_ALREADY_MAPPED = 208,
  CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
  CUDA_ERROR_ALREADY_ACQUIRED = 210,
  CUDA_ERROR_NOT_MAPPED = 211,
  CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
  CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
  CUDA_ERROR_ECC_UNCORRECTABLE = 214,
  CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
  CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
  CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
  CUDA_ERROR_INVALID_PTX = 218,
  CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
  CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
  CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
  CUDA_ERROR_INVALID_SOURCE = 300,
  CUDA_ERROR_FILE_NOT_FOUND = 301,
  CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
  CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
  CUDA_ERROR_OPERATING_SYSTEM = 304,
  CUDA_ERROR_INVALID_HANDLE = 400,
  CUDA_ERROR_NOT_FOUND = 500,
  CUDA_ERROR_NOT_READY = 600,
  CUDA_ERROR_ILLEGAL_ADDRESS = 700,
  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
  CUDA_ERROR_LAUNCH_TIMEOUT = 702,
  CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
  CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
  CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
  CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
  CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
  CUDA_ERROR_ASSERT = 710,
  CUDA_ERROR_TOO_MANY_PEERS = 711,
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
  CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
  CUDA_ERROR_MISALIGNED_ADDRESS = 716,
  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
  CUDA_ERROR_INVALID_PC = 718,
  CUDA_ERROR_LAUNCH_FAILED = 719,
  CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
  CUDA_ERROR_NOT_PERMITTED = 800,
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CUDA_ERROR_UNKNOWN = 999,
} CUresult;

typedef enum CUdevice_P2PAttribute_enum {
  CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01,
  CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02,
  CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03,
  CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED = 0x04,
} CUdevice_P2PAttribute;

typedef void (CUDA_CB *CUstreamCallback)(CUstream hStream, CUresult status, void* userData);
typedef size_t (CUDA_CB *CUoccupancyB2DSize)(int blockSize);

typedef struct CUDA_MEMCPY2D_st {
  size_t srcXInBytes;
  size_t srcY;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  size_t srcPitch;
  size_t dstXInBytes;
  size_t dstY;
  CUmemorytype dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  size_t dstPitch;
  size_t WidthInBytes;
  size_t Height;
} CUDA_MEMCPY2D;

typedef struct CUDA_MEMCPY3D_st {
  size_t srcXInBytes;
  size_t srcY;
  size_t srcZ;
  size_t srcLOD;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  void* reserved0;
  size_t srcPitch;
  size_t srcHeight;
  size_t dstXInBytes;
  size_t dstY;
  size_t dstZ;
  size_t dstLOD;
  CUmemorytype dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  void* reserved1;
  size_t dstPitch;
  size_t dstHeight;
  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
} CUDA_MEMCPY3D;

typedef struct CUDA_MEMCPY3D_PEER_st {
  size_t srcXInBytes;
  size_t srcY;
  size_t srcZ;
  size_t srcLOD;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  CUcontext srcContext;
  size_t srcPitch;
  size_t srcHeight;
  size_t dstXInBytes;
  size_t dstY;
  size_t dstZ;
  size_t dstLOD;
  CUmemorytype dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  CUcontext dstContext;
  size_t dstPitch;
  size_t dstHeight;
  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
} CUDA_MEMCPY3D_PEER;

typedef struct CUDA_ARRAY_DESCRIPTOR_st {
  size_t Width;
  size_t Height;
  CUarray_format Format;
  unsigned int NumChannels;
} CUDA_ARRAY_DESCRIPTOR;

typedef struct CUDA_ARRAY3D_DESCRIPTOR_st {
  size_t Width;
  size_t Height;
  size_t Depth;
  CUarray_format Format;
  unsigned int NumChannels;
  unsigned int Flags;
} CUDA_ARRAY3D_DESCRIPTOR;

typedef struct CUDA_RESOURCE_DESC_st {
  CUresourcetype resType;
  union {
    struct {
      CUarray hArray;
    } array;
    struct {
      CUmipmappedArray hMipmappedArray;
    } mipmap;
    struct {
      CUdeviceptr devPtr;
      CUarray_format format;
      unsigned int numChannels;
      size_t sizeInBytes;
    } linear;
    struct {
      CUdeviceptr devPtr;
      CUarray_format format;
      unsigned int numChannels;
      size_t width;
      size_t height;
      size_t pitchInBytes;
    } pitch2D;
    struct {
      int reserved[32];
    } reserved;
  } res;
  unsigned int flags;
} CUDA_RESOURCE_DESC;

typedef struct CUDA_TEXTURE_DESC_st {
  CUaddress_mode addressMode[3];
  CUfilter_mode filterMode;
  unsigned int flags;
  unsigned int maxAnisotropy;
  CUfilter_mode mipmapFilterMode;
  float mipmapLevelBias;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;
  float borderColor[4];
  int reserved[12];
} CUDA_TEXTURE_DESC;

typedef enum CUresourceViewFormat_enum {
  CU_RES_VIEW_FORMAT_NONE = 0x00,
  CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01,
  CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02,
  CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03,
  CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04,
  CU_RES_VIEW_FORMAT_SINT_2X8 = 0x05,
  CU_RES_VIEW_FORMAT_SINT_4X8 = 0x06,
  CU_RES_VIEW_FORMAT_UINT_1X16 = 0x07,
  CU_RES_VIEW_FORMAT_UINT_2X16 = 0x08,
  CU_RES_VIEW_FORMAT_UINT_4X16 = 0x09,
  CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0a,
  CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0b,
  CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0c,
  CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0d,
  CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0e,
  CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0f,
  CU_RES_VIEW_FORMAT_SINT_1X32 = 0x10,
  CU_RES_VIEW_FORMAT_SINT_2X32 = 0x11,
  CU_RES_VIEW_FORMAT_SINT_4X32 = 0x12,
  CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13,
  CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14,
  CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15,
  CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16,
  CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17,
  CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18,
  CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19,
  CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a,
  CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b,
  CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c,
  CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d,
  CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e,
  CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f,
  CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20,
  CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21,
  CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22,
} CUresourceViewFormat;

typedef struct CUDA_RESOURCE_VIEW_DESC_st {
  CUresourceViewFormat format;
  size_t width;
  size_t height;
  size_t depth;
  unsigned int firstMipmapLevel;
  unsigned int lastMipmapLevel;
  unsigned int firstLayer;
  unsigned int lastLayer;
  unsigned int reserved[16];
} CUDA_RESOURCE_VIEW_DESC;

typedef struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
  unsigned long long p2pToken;
  unsigned int vaSpaceToken;
} CUDA_POINTER_ATTRIBUTE_P2P_TOKENS;

typedef struct CUDA_LAUNCH_PARAMS_st {
  CUfunction function;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  void** kernelParams;
} CUDA_LAUNCH_PARAMS;
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;

typedef enum CUGLDeviceList_enum {
  CU_GL_DEVICE_LIST_ALL = 0x01,
  CU_GL_DEVICE_LIST_CURRENT_FRAME = 0x02,
  CU_GL_DEVICE_LIST_NEXT_FRAME = 0x03,
} CUGLDeviceList;

typedef enum CUGLmap_flags_enum {
  CU_GL_MAP_RESOURCE_FLAGS_NONE = 0x00,
  CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01,
  CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02,
} CUGLmap_flags;

typedef enum  {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  NVRTC_ERROR_INTERNAL_ERROR = 11,
} nvrtcResult;

typedef struct _nvrtcProgram* nvrtcProgram;
typedef struct cudnnContext* cudnnHandle_t;

typedef enum  {
  CUDNN_STATUS_SUCCESS = 0,
  CUDNN_STATUS_NOT_INITIALIZED = 1,
  CUDNN_STATUS_ALLOC_FAILED = 2,
  CUDNN_STATUS_BAD_PARAM = 3,
  CUDNN_STATUS_INTERNAL_ERROR = 4,
  CUDNN_STATUS_INVALID_VALUE = 5,
  CUDNN_STATUS_ARCH_MISMATCH = 6,
  CUDNN_STATUS_MAPPING_ERROR = 7,
  CUDNN_STATUS_EXECUTION_FAILED = 8,
  CUDNN_STATUS_NOT_SUPPORTED = 9,
  CUDNN_STATUS_LICENSE_ERROR = 10,
  CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
  CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
  CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
} cudnnStatus_t;

typedef struct cudnnRuntimeTag_t {
} cudnnRuntimeTag_t;

typedef enum  {
  CUDNN_ERRQUERY_RAWCODE = 0,
  CUDNN_ERRQUERY_NONBLOCKING = 1,
  CUDNN_ERRQUERY_BLOCKING = 2,
} cudnnErrQueryMode_t;

typedef enum libraryPropertyType_t {
  MAJOR_VERSION,
  MINOR_VERSION,
  PATCH_LEVEL,
} libraryPropertyType;

typedef struct cudnnTensorStruct* cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct* cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct* cudnnFilterDescriptor_t;
typedef struct cudnnLRNStruct* cudnnLRNDescriptor_t;
typedef struct cudnnActivationStruct* cudnnActivationDescriptor_t;
typedef struct cudnnSpatialTransformerStruct* cudnnSpatialTransformerDescriptor_t;
typedef struct cudnnOpTensorStruct* cudnnOpTensorDescriptor_t;
typedef struct cudnnReduceTensorStruct* cudnnReduceTensorDescriptor_t;
typedef struct cudnnCTCLossStruct* cudnnCTCLossDescriptor_t;

typedef enum  {
  CUDNN_DATA_FLOAT = 0,
  CUDNN_DATA_DOUBLE = 1,
  CUDNN_DATA_HALF = 2,
  CUDNN_DATA_INT8 = 3,
  CUDNN_DATA_INT32 = 4,
  CUDNN_DATA_INT8x4 = 5,
  CUDNN_DATA_UINT8 = 6,
  CUDNN_DATA_UINT8x4 = 7,
} cudnnDataType_t;

typedef enum  {
  CUDNN_DEFAULT_MATH = 0,
  CUDNN_TENSOR_OP_MATH = 1,
} cudnnMathType_t;

typedef enum  {
  CUDNN_NOT_PROPAGATE_NAN = 0,
  CUDNN_PROPAGATE_NAN = 1,
} cudnnNanPropagation_t;

typedef enum  {
  CUDNN_NON_DETERMINISTIC = 0,
  CUDNN_DETERMINISTIC = 1,
} cudnnDeterminism_t;

typedef enum  {
  CUDNN_TENSOR_NCHW = 0,
  CUDNN_TENSOR_NHWC = 1,
  CUDNN_TENSOR_NCHW_VECT_C = 2,
} cudnnTensorFormat_t;

typedef enum  {
  CUDNN_OP_TENSOR_ADD = 0,
  CUDNN_OP_TENSOR_MUL = 1,
  CUDNN_OP_TENSOR_MIN = 2,
  CUDNN_OP_TENSOR_MAX = 3,
  CUDNN_OP_TENSOR_SQRT = 4,
  CUDNN_OP_TENSOR_NOT = 5,
} cudnnOpTensorOp_t;

typedef enum  {
  CUDNN_REDUCE_TENSOR_ADD = 0,
  CUDNN_REDUCE_TENSOR_MUL = 1,
  CUDNN_REDUCE_TENSOR_MIN = 2,
  CUDNN_REDUCE_TENSOR_MAX = 3,
  CUDNN_REDUCE_TENSOR_AMAX = 4,
  CUDNN_REDUCE_TENSOR_AVG = 5,
  CUDNN_REDUCE_TENSOR_NORM1 = 6,
  CUDNN_REDUCE_TENSOR_NORM2 = 7,
  CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} cudnnReduceTensorOp_t;

typedef enum  {
  CUDNN_REDUCE_TENSOR_NO_INDICES = 0,
  CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} cudnnReduceTensorIndices_t;

typedef enum  {
  CUDNN_32BIT_INDICES = 0,
  CUDNN_64BIT_INDICES = 1,
  CUDNN_16BIT_INDICES = 2,
  CUDNN_8BIT_INDICES = 3,
} cudnnIndicesType_t;

typedef enum  {
  CUDNN_CONVOLUTION = 0,
  CUDNN_CROSS_CORRELATION = 1,
} cudnnConvolutionMode_t;

typedef enum  {
  CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
  CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;

typedef enum  {
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
  CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
  CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
  CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
  CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
  CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
} cudnnConvolutionFwdAlgo_t;

typedef struct {
  cudnnConvolutionFwdAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];
} cudnnConvolutionFwdAlgoPerf_t;

typedef enum  {
  CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
  CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
  CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdFilterPreference_t;

typedef enum  {
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
  CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7,
} cudnnConvolutionBwdFilterAlgo_t;

typedef struct {
  cudnnConvolutionBwdFilterAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];
} cudnnConvolutionBwdFilterAlgoPerf_t;

typedef enum  {
  CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
  CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
  CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdDataPreference_t;

typedef enum  {
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
  CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6,
} cudnnConvolutionBwdDataAlgo_t;

typedef struct {
  cudnnConvolutionBwdDataAlgo_t algo;
  cudnnStatus_t status;
  float time;
  size_t memory;
  cudnnDeterminism_t determinism;
  cudnnMathType_t mathType;
  int reserved[3];
} cudnnConvolutionBwdDataAlgoPerf_t;

typedef enum  {
  CUDNN_SOFTMAX_FAST = 0,
  CUDNN_SOFTMAX_ACCURATE = 1,
  CUDNN_SOFTMAX_LOG = 2,
} cudnnSoftmaxAlgorithm_t;

typedef enum  {
  CUDNN_SOFTMAX_MODE_INSTANCE = 0,
  CUDNN_SOFTMAX_MODE_CHANNEL = 1,
} cudnnSoftmaxMode_t;

typedef enum  {
  CUDNN_POOLING_MAX = 0,
  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
  CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
  CUDNN_POOLING_MAX_DETERMINISTIC = 3,
} cudnnPoolingMode_t;

typedef enum  {
  CUDNN_ACTIVATION_SIGMOID = 0,
  CUDNN_ACTIVATION_RELU = 1,
  CUDNN_ACTIVATION_TANH = 2,
  CUDNN_ACTIVATION_CLIPPED_RELU = 3,
  CUDNN_ACTIVATION_ELU = 4,
  CUDNN_ACTIVATION_IDENTITY = 5,
} cudnnActivationMode_t;

typedef enum  {
  CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0,
} cudnnLRNMode_t;

typedef enum  {
  CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
} cudnnDivNormMode_t;

typedef enum  {
  CUDNN_BATCHNORM_PER_ACTIVATION = 0,
  CUDNN_BATCHNORM_SPATIAL = 1,
  CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
} cudnnBatchNormMode_t;

typedef enum  {
  CUDNN_SAMPLER_BILINEAR = 0,
} cudnnSamplerType_t;

typedef struct cudnnDropoutStruct* cudnnDropoutDescriptor_t;

typedef enum  {
  CUDNN_RNN_RELU = 0,
  CUDNN_RNN_TANH = 1,
  CUDNN_LSTM = 2,
  CUDNN_GRU = 3,
} cudnnRNNMode_t;

typedef enum  {
  CUDNN_UNIDIRECTIONAL = 0,
  CUDNN_BIDIRECTIONAL = 1,
} cudnnDirectionMode_t;

typedef enum  {
  CUDNN_LINEAR_INPUT = 0,
  CUDNN_SKIP_INPUT = 1,
} cudnnRNNInputMode_t;

typedef enum  {
  CUDNN_RNN_ALGO_STANDARD = 0,
  CUDNN_RNN_ALGO_PERSIST_STATIC = 1,
  CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2,
  CUDNN_RNN_ALGO_COUNT = 3,
} cudnnRNNAlgo_t;

typedef struct cudnnAlgorithmStruct* cudnnAlgorithmDescriptor_t;
typedef struct cudnnAlgorithmPerformanceStruct* cudnnAlgorithmPerformance_t;
typedef struct cudnnRNNStruct* cudnnRNNDescriptor_t;
typedef struct cudnnPersistentRNNPlan* cudnnPersistentRNNPlan_t;

typedef enum  {
  CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0,
  CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1,
} cudnnCTCLossAlgo_t;

typedef struct {
  union Algorithm {
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
    cudnnRNNAlgo_t RNNAlgo;
    cudnnCTCLossAlgo_t CTCLossAlgo;
  } algo;
} cudnnAlgorithm_t;

typedef enum  {
  CUDNN_SEV_FATAL = 0,
  CUDNN_SEV_ERROR = 1,
  CUDNN_SEV_WARNING = 2,
  CUDNN_SEV_INFO = 3,
} cudnnSeverity_t;

typedef struct {
  unsigned cudnn_version;
  cudnnStatus_t cudnnStatus;
  unsigned time_sec;
  unsigned time_usec;
  unsigned time_delta;
  cudnnHandle_t handle;
  cudaStream_t stream;
  unsigned long long pid;
  unsigned long long tid;
  int cudaDeviceId;
  int reserved[15];
} cudnnDebug_t;

typedef void (CUDA_CB *cudnnCallback_t)(cudnnSeverity_t sev, void* udata, const cudnnDebug_t* dbg, const char* msg);


/* Function types. */
typedef CUresult CUDAAPI tcuGetErrorString(CUresult error, const char** pStr);
typedef CUresult CUDAAPI tcuGetErrorName(CUresult error, const char** pStr);
typedef CUresult CUDAAPI tcuInit(unsigned int Flags);
typedef CUresult CUDAAPI tcuDriverGetVersion(int* driverVersion);
typedef CUresult CUDAAPI tcuDeviceGet(CUdevice* device, int ordinal);
typedef CUresult CUDAAPI tcuDeviceGetCount(int* count);
typedef CUresult CUDAAPI tcuDeviceGetName(char* name, int len, CUdevice dev);
typedef CUresult CUDAAPI tcuDeviceGetUuid(CUuuid* uuid, CUdevice dev);
typedef CUresult CUDAAPI tcuDeviceTotalMem_v2(size_t* bytes, CUdevice dev);
typedef CUresult CUDAAPI tcuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult CUDAAPI tcuDeviceGetProperties(CUdevprop* prop, CUdevice dev);
typedef CUresult CUDAAPI tcuDeviceComputeCapability(int* major, int* minor, CUdevice dev);
typedef CUresult CUDAAPI tcuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
typedef CUresult CUDAAPI tcuDevicePrimaryCtxRelease(CUdevice dev);
typedef CUresult CUDAAPI tcuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
typedef CUresult CUDAAPI tcuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active);
typedef CUresult CUDAAPI tcuDevicePrimaryCtxReset(CUdevice dev);
typedef CUresult CUDAAPI tcuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev);
typedef CUresult CUDAAPI tcuCtxDestroy_v2(CUcontext ctx);
typedef CUresult CUDAAPI tcuCtxPushCurrent_v2(CUcontext ctx);
typedef CUresult CUDAAPI tcuCtxPopCurrent_v2(CUcontext* pctx);
typedef CUresult CUDAAPI tcuCtxSetCurrent(CUcontext ctx);
typedef CUresult CUDAAPI tcuCtxGetCurrent(CUcontext* pctx);
typedef CUresult CUDAAPI tcuCtxGetDevice(CUdevice* device);
typedef CUresult CUDAAPI tcuCtxGetFlags(unsigned int* flags);
typedef CUresult CUDAAPI tcuCtxSynchronize(void);
typedef CUresult CUDAAPI tcuCtxSetLimit(CUlimit limit, size_t value);
typedef CUresult CUDAAPI tcuCtxGetLimit(size_t* pvalue, CUlimit limit);
typedef CUresult CUDAAPI tcuCtxGetCacheConfig(CUfunc_cache* pconfig);
typedef CUresult CUDAAPI tcuCtxSetCacheConfig(CUfunc_cache config);
typedef CUresult CUDAAPI tcuCtxGetSharedMemConfig(CUsharedconfig* pConfig);
typedef CUresult CUDAAPI tcuCtxSetSharedMemConfig(CUsharedconfig config);
typedef CUresult CUDAAPI tcuCtxGetApiVersion(CUcontext ctx, unsigned int* version);
typedef CUresult CUDAAPI tcuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
typedef CUresult CUDAAPI tcuCtxAttach(CUcontext* pctx, unsigned int flags);
typedef CUresult CUDAAPI tcuCtxDetach(CUcontext ctx);
typedef CUresult CUDAAPI tcuModuleLoad(CUmodule* module, const char* fname);
typedef CUresult CUDAAPI tcuModuleLoadData(CUmodule* module, const void* image);
typedef CUresult CUDAAPI tcuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues);
typedef CUresult CUDAAPI tcuModuleLoadFatBinary(CUmodule* module, const void* fatCubin);
typedef CUresult CUDAAPI tcuModuleUnload(CUmodule hmod);
typedef CUresult CUDAAPI tcuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
typedef CUresult CUDAAPI tcuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
typedef CUresult CUDAAPI tcuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name);
typedef CUresult CUDAAPI tcuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name);
typedef CUresult CUDAAPI tcuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
typedef CUresult CUDAAPI tcuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues);
typedef CUresult CUDAAPI tcuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues);
typedef CUresult CUDAAPI tcuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
typedef CUresult CUDAAPI tcuLinkDestroy(CUlinkState state);
typedef CUresult CUDAAPI tcuMemGetInfo_v2(size_t* free, size_t* total);
typedef CUresult CUDAAPI tcuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
typedef CUresult CUDAAPI tcuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
typedef CUresult CUDAAPI tcuMemFree_v2(CUdeviceptr dptr);
typedef CUresult CUDAAPI tcuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
typedef CUresult CUDAAPI tcuMemAllocHost_v2(void** pp, size_t bytesize);
typedef CUresult CUDAAPI tcuMemFreeHost(void* p);
typedef CUresult CUDAAPI tcuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
typedef CUresult CUDAAPI tcuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags);
typedef CUresult CUDAAPI tcuMemHostGetFlags(unsigned int* pFlags, void* p);
typedef CUresult CUDAAPI tcuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
typedef CUresult CUDAAPI tcuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId);
typedef CUresult CUDAAPI tcuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev);
typedef CUresult CUDAAPI tcuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event);
typedef CUresult CUDAAPI tcuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle);
typedef CUresult CUDAAPI tcuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr);
typedef CUresult CUDAAPI tcuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags);
typedef CUresult CUDAAPI tcuIpcCloseMemHandle(CUdeviceptr dptr);
typedef CUresult CUDAAPI tcuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags);
typedef CUresult CUDAAPI tcuMemHostUnregister(void* p);
typedef CUresult CUDAAPI tcuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult CUDAAPI tcuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy);
typedef CUresult CUDAAPI tcuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy);
typedef CUresult CUDAAPI tcuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy);
typedef CUresult CUDAAPI tcuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy);
typedef CUresult CUDAAPI tcuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream);
typedef CUresult CUDAAPI tcuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream);
typedef CUresult CUDAAPI tcuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
typedef CUresult CUDAAPI tcuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N);
typedef CUresult CUDAAPI tcuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
typedef CUresult CUDAAPI tcuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef CUresult CUDAAPI tcuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
typedef CUresult CUDAAPI tcuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
typedef CUresult CUDAAPI tcuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
typedef CUresult CUDAAPI tcuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
typedef CUresult CUDAAPI tcuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
typedef CUresult CUDAAPI tcuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
typedef CUresult CUDAAPI tcuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
typedef CUresult CUDAAPI tcuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
typedef CUresult CUDAAPI tcuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray);
typedef CUresult CUDAAPI tcuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
typedef CUresult CUDAAPI tcuArrayDestroy(CUarray hArray);
typedef CUresult CUDAAPI tcuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
typedef CUresult CUDAAPI tcuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
typedef CUresult CUDAAPI tcuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels);
typedef CUresult CUDAAPI tcuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
typedef CUresult CUDAAPI tcuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
typedef CUresult CUDAAPI tcuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
typedef CUresult CUDAAPI tcuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
typedef CUresult CUDAAPI tcuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
typedef CUresult CUDAAPI tcuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
typedef CUresult CUDAAPI tcuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);
typedef CUresult CUDAAPI tcuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr);
typedef CUresult CUDAAPI tcuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr);
typedef CUresult CUDAAPI tcuStreamCreate(CUstream* phStream, unsigned int Flags);
typedef CUresult CUDAAPI tcuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority);
typedef CUresult CUDAAPI tcuStreamGetPriority(CUstream hStream, int* priority);
typedef CUresult CUDAAPI tcuStreamGetFlags(CUstream hStream, unsigned int* flags);
typedef CUresult CUDAAPI tcuStreamGetCtx(CUstream hStream, CUcontext* pctx);
typedef CUresult CUDAAPI tcuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
typedef CUresult CUDAAPI tcuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags);
typedef CUresult CUDAAPI tcuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
typedef CUresult CUDAAPI tcuStreamQuery(CUstream hStream);
typedef CUresult CUDAAPI tcuStreamSynchronize(CUstream hStream);
typedef CUresult CUDAAPI tcuStreamDestroy_v2(CUstream hStream);
typedef CUresult CUDAAPI tcuEventCreate(CUevent* phEvent, unsigned int Flags);
typedef CUresult CUDAAPI tcuEventRecord(CUevent hEvent, CUstream hStream);
typedef CUresult CUDAAPI tcuEventQuery(CUevent hEvent);
typedef CUresult CUDAAPI tcuEventSynchronize(CUevent hEvent);
typedef CUresult CUDAAPI tcuEventDestroy_v2(CUevent hEvent);
typedef CUresult CUDAAPI tcuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);
typedef CUresult CUDAAPI tcuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
typedef CUresult CUDAAPI tcuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
typedef CUresult CUDAAPI tcuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
typedef CUresult CUDAAPI tcuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
typedef CUresult CUDAAPI tcuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags);
typedef CUresult CUDAAPI tcuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
typedef CUresult CUDAAPI tcuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
typedef CUresult CUDAAPI tcuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
typedef CUresult CUDAAPI tcuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);
typedef CUresult CUDAAPI tcuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);
typedef CUresult CUDAAPI tcuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams);
typedef CUresult CUDAAPI tcuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags);
typedef CUresult CUDAAPI tcuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
typedef CUresult CUDAAPI tcuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
typedef CUresult CUDAAPI tcuParamSetSize(CUfunction hfunc, unsigned int numbytes);
typedef CUresult CUDAAPI tcuParamSeti(CUfunction hfunc, int offset, unsigned int value);
typedef CUresult CUDAAPI tcuParamSetf(CUfunction hfunc, int offset, float value);
typedef CUresult CUDAAPI tcuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes);
typedef CUresult CUDAAPI tcuLaunch(CUfunction f);
typedef CUresult CUDAAPI tcuLaunchGrid(CUfunction f, int grid_width, int grid_height);
typedef CUresult CUDAAPI tcuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
typedef CUresult CUDAAPI tcuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
typedef CUresult CUDAAPI tcuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
typedef CUresult CUDAAPI tcuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
typedef CUresult CUDAAPI tcuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
typedef CUresult CUDAAPI tcuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
typedef CUresult CUDAAPI tcuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
typedef CUresult CUDAAPI tcuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
typedef CUresult CUDAAPI tcuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch);
typedef CUresult CUDAAPI tcuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
typedef CUresult CUDAAPI tcuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
typedef CUresult CUDAAPI tcuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
typedef CUresult CUDAAPI tcuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);
typedef CUresult CUDAAPI tcuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
typedef CUresult CUDAAPI tcuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
typedef CUresult CUDAAPI tcuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso);
typedef CUresult CUDAAPI tcuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor);
typedef CUresult CUDAAPI tcuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
typedef CUresult CUDAAPI tcuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetArray(CUarray* phArray, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim);
typedef CUresult CUDAAPI tcuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef);
typedef CUresult CUDAAPI tcuTexRefCreate(CUtexref* pTexRef);
typedef CUresult CUDAAPI tcuTexRefDestroy(CUtexref hTexRef);
typedef CUresult CUDAAPI tcuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
typedef CUresult CUDAAPI tcuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef);
typedef CUresult CUDAAPI tcuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
typedef CUresult CUDAAPI tcuTexObjectDestroy(CUtexObject texObject);
typedef CUresult CUDAAPI tcuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject);
typedef CUresult CUDAAPI tcuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject);
typedef CUresult CUDAAPI tcuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject);
typedef CUresult CUDAAPI tcuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc);
typedef CUresult CUDAAPI tcuSurfObjectDestroy(CUsurfObject surfObject);
typedef CUresult CUDAAPI tcuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject);
typedef CUresult CUDAAPI tcuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev);
typedef CUresult CUDAAPI tcuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
typedef CUresult CUDAAPI tcuCtxDisablePeerAccess(CUcontext peerContext);
typedef CUresult CUDAAPI tcuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);
typedef CUresult CUDAAPI tcuGraphicsUnregisterResource(CUgraphicsResource resource);
typedef CUresult CUDAAPI tcuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
typedef CUresult CUDAAPI tcuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource);
typedef CUresult CUDAAPI tcuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
typedef CUresult CUDAAPI tcuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags);
typedef CUresult CUDAAPI tcuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
typedef CUresult CUDAAPI tcuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
typedef CUresult CUDAAPI tcuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId);

typedef CUresult CUDAAPI tcuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags);
typedef CUresult CUDAAPI tcuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags);
typedef CUresult CUDAAPI tcuGLGetDevices_v2(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
typedef CUresult CUDAAPI tcuGLCtxCreate_v2(CUcontext* pCtx, unsigned int Flags, CUdevice device);
typedef CUresult CUDAAPI tcuGLInit(void);
typedef CUresult CUDAAPI tcuGLRegisterBufferObject(GLuint buffer);
typedef CUresult CUDAAPI tcuGLMapBufferObject_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer);
typedef CUresult CUDAAPI tcuGLUnmapBufferObject(GLuint buffer);
typedef CUresult CUDAAPI tcuGLUnregisterBufferObject(GLuint buffer);
typedef CUresult CUDAAPI tcuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags);
typedef CUresult CUDAAPI tcuGLMapBufferObjectAsync_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer, CUstream hStream);
typedef CUresult CUDAAPI tcuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream);

typedef const char* CUDAAPI tnvrtcGetErrorString(nvrtcResult result);
typedef nvrtcResult CUDAAPI tnvrtcVersion(int* major, int* minor);
typedef nvrtcResult CUDAAPI tnvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames);
typedef nvrtcResult CUDAAPI tnvrtcDestroyProgram(nvrtcProgram* prog);
typedef nvrtcResult CUDAAPI tnvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options);
typedef nvrtcResult CUDAAPI tnvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
typedef nvrtcResult CUDAAPI tnvrtcGetPTX(nvrtcProgram prog, char* ptx);
typedef nvrtcResult CUDAAPI tnvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);
typedef nvrtcResult CUDAAPI tnvrtcGetProgramLog(nvrtcProgram prog, char* log);
typedef nvrtcResult CUDAAPI tnvrtcAddNameExpression(nvrtcProgram prog, const char* name_expression);
typedef nvrtcResult CUDAAPI tnvrtcGetLoweredName(nvrtcProgram prog, const char* name_expression, const char** lowered_name);

typedef size_t CUDAAPI tcudnnGetVersion(void);
typedef size_t CUDAAPI tcudnnGetCudartVersion(void);
typedef const char* CUDAAPI tcudnnGetErrorString(cudnnStatus_t status);
typedef cudnnStatus_t CUDAAPI tcudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t* rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t* tag);
typedef cudnnStatus_t CUDAAPI tcudnnGetProperty(libraryPropertyType type, int* value);
typedef cudnnStatus_t CUDAAPI tcudnnCreate(cudnnHandle_t* handle);
typedef cudnnStatus_t CUDAAPI tcudnnDestroy(cudnnHandle_t handle);
typedef cudnnStatus_t CUDAAPI tcudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
typedef cudnnStatus_t CUDAAPI tcudnnGetStream(cudnnHandle_t handle, cudaStream_t* streamId);
typedef cudnnStatus_t CUDAAPI tcudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w);
typedef cudnnStatus_t CUDAAPI tcudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride);
typedef cudnnStatus_t CUDAAPI tcudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t* dataType, int* n, int* c, int* h, int* w, int* nStride, int* cStride, int* hStride, int* wStride);
typedef cudnnStatus_t CUDAAPI tcudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]);
typedef cudnnStatus_t CUDAAPI tcudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims, const int dimA[]);
typedef cudnnStatus_t CUDAAPI tcudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested, cudnnDataType_t* dataType, int* nbDims, int dimA[], int strideA[]);
typedef cudnnStatus_t CUDAAPI tcudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t* size);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);
typedef cudnnStatus_t CUDAAPI tcudnnTransformTensor(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnAddTensor(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t aDesc, const void* A, const void* beta, const cudnnTensorDescriptor_t cDesc, void* C);
typedef cudnnStatus_t CUDAAPI tcudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* opTensorDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt);
typedef cudnnStatus_t CUDAAPI tcudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t* opTensorOp, cudnnDataType_t* opTensorCompType, cudnnNanPropagation_t* opTensorNanOpt);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc);
typedef cudnnStatus_t CUDAAPI tcudnnOpTensor(cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc, const void* alpha1, const cudnnTensorDescriptor_t aDesc, const void* A, const void* alpha2, const cudnnTensorDescriptor_t bDesc, const void* B, const void* beta, const cudnnTensorDescriptor_t cDesc, void* C);
typedef cudnnStatus_t CUDAAPI tcudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t* reduceTensorDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp, cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt, cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType);
typedef cudnnStatus_t CUDAAPI tcudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t* reduceTensorOp, cudnnDataType_t* reduceTensorCompType, cudnnNanPropagation_t* reduceTensorNanOpt, cudnnReduceTensorIndices_t* reduceTensorIndices, cudnnIndicesType_t* reduceTensorIndicesType);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc);
typedef cudnnStatus_t CUDAAPI tcudnnGetReductionIndicesSize(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetReductionWorkspaceSize(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnReduceTensor(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, void* indices, size_t indicesSizeInBytes, void* workspace, size_t workspaceSizeInBytes, const void* alpha, const cudnnTensorDescriptor_t aDesc, const void* A, const void* beta, const cudnnTensorDescriptor_t cDesc, void* C);
typedef cudnnStatus_t CUDAAPI tcudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void* y, const void* valuePtr);
typedef cudnnStatus_t CUDAAPI tcudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void* y, const void* alpha);
typedef cudnnStatus_t CUDAAPI tcudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int k, int c, int h, int w);
typedef cudnnStatus_t CUDAAPI tcudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* k, int* c, int* h, int* w);
typedef cudnnStatus_t CUDAAPI tcudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int filterDimA[]);
typedef cudnnStatus_t CUDAAPI tcudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested, cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* nbDims, int filterDimA[]);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);
typedef cudnnStatus_t CUDAAPI tcudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t* mathType);
typedef cudnnStatus_t CUDAAPI tcudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int* groupCount);
typedef cudnnStatus_t CUDAAPI tcudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w, cudnnConvolutionMode_t mode, cudnnDataType_t computeType);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc, int* pad_h, int* pad_w, int* u, int* v, int* dilation_h, int* dilation_w, cudnnConvolutionMode_t* mode, cudnnDataType_t* computeType);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc, const cudnnFilterDescriptor_t filterDesc, int* n, int* c, int* h, int* w);
typedef cudnnStatus_t CUDAAPI tcudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[], const int filterStrideA[], const int dilationA[], cudnnConvolutionMode_t mode, cudnnDataType_t computeType);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested, int* arrayLength, int padA[], int strideA[], int dilationA[], cudnnConvolutionMode_t* mode, cudnnDataType_t* computeType);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc, const cudnnFilterDescriptor_t filterDesc, int nbDims, int tensorOuputDimA[]);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int* count);
typedef cudnnStatus_t CUDAAPI tcudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults);
typedef cudnnStatus_t CUDAAPI tcudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void* y, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults, void* workSpace, size_t workSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionFwdAlgo_t* algo);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnConvolutionForward(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnConvolutionBiasActivationForward(cudnnHandle_t handle, const void* alpha1, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* alpha2, const cudnnTensorDescriptor_t zDesc, const void* z, const cudnnTensorDescriptor_t biasDesc, const void* bias, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnConvolutionBackwardBias(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, const cudnnTensorDescriptor_t dbDesc, void* db);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int* count);
typedef cudnnStatus_t CUDAAPI tcudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t* perfResults);
typedef cudnnStatus_t CUDAAPI tcudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t dyDesc, const void* y, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void* dw, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t* perfResults, void* workSpace, size_t workSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionBwdFilterAlgo_t* algo);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t* perfResults);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc, cudnnConvolutionBwdFilterAlgo_t algo, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnConvolutionBackwardFilter(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta, const cudnnFilterDescriptor_t dwDesc, void* dw);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int* count);
typedef cudnnStatus_t CUDAAPI tcudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t* perfResults);
typedef cudnnStatus_t CUDAAPI tcudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void* dx, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t* perfResults, void* workSpace, size_t workSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionBwdDataAlgo_t* algo);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t* perfResults);
typedef cudnnStatus_t CUDAAPI tcudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnConvolutionBackwardData(cudnnHandle_t handle, const void* alpha, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
typedef cudnnStatus_t CUDAAPI tcudnnIm2Col(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, void* colBuffer);
typedef cudnnStatus_t CUDAAPI tcudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnSoftmaxBackward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void* alpha, const cudnnTensorDescriptor_t yDesc, const void* y, const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
typedef cudnnStatus_t CUDAAPI tcudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride);
typedef cudnnStatus_t CUDAAPI tcudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t* mode, cudnnNanPropagation_t* maxpoolingNanOpt, int* windowHeight, int* windowWidth, int* verticalPadding, int* horizontalPadding, int* verticalStride, int* horizontalStride);
typedef cudnnStatus_t CUDAAPI tcudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode, const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims, const int windowDimA[], const int paddingA[], const int strideA[]);
typedef cudnnStatus_t CUDAAPI tcudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested, cudnnPoolingMode_t* mode, cudnnNanPropagation_t* maxpoolingNanOpt, int* nbDims, int windowDimA[], int paddingA[], int strideA[]);
typedef cudnnStatus_t CUDAAPI tcudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int nbDims, int outputTensorDimA[]);
typedef cudnnStatus_t CUDAAPI tcudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int* n, int* c, int* h, int* w);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc);
typedef cudnnStatus_t CUDAAPI tcudnnPoolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnPoolingBackward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void* alpha, const cudnnTensorDescriptor_t yDesc, const void* y, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
typedef cudnnStatus_t CUDAAPI tcudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef);
typedef cudnnStatus_t CUDAAPI tcudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t* mode, cudnnNanPropagation_t* reluNanOpt, double* coef);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc);
typedef cudnnStatus_t CUDAAPI tcudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnActivationBackward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void* alpha, const cudnnTensorDescriptor_t yDesc, const void* y, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
typedef cudnnStatus_t CUDAAPI tcudnnCreateLRNDescriptor(cudnnLRNDescriptor_t* normDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK);
typedef cudnnStatus_t CUDAAPI tcudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned* lrnN, double* lrnAlpha, double* lrnBeta, double* lrnK);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc);
typedef cudnnStatus_t CUDAAPI tcudnnLRNCrossChannelForward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnLRNCrossChannelBackward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode, const void* alpha, const cudnnTensorDescriptor_t yDesc, const void* y, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
typedef cudnnStatus_t CUDAAPI tcudnnDivisiveNormalizationForward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* means, void* temp, void* temp2, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnDivisiveNormalizationBackward(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* means, const void* dy, void* temp, void* temp2, const void* beta, const cudnnTensorDescriptor_t dXdMeansDesc, void* dx, void* dMeans);
typedef cudnnStatus_t CUDAAPI tcudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc, cudnnBatchNormMode_t mode);
typedef cudnnStatus_t CUDAAPI tcudnnBatchNormalizationForwardTraining(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc, void* y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias, double exponentialAverageFactor, void* resultRunningMean, void* resultRunningVariance, double epsilon, void* resultSaveMean, void* resultSaveInvVariance);
typedef cudnnStatus_t CUDAAPI tcudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc, void* y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias, const void* estimatedMean, const void* estimatedVariance, double epsilon);
typedef cudnnStatus_t CUDAAPI tcudnnBatchNormalizationBackward(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alphaDataDiff, const void* betaDataDiff, const void* alphaParamDiff, const void* betaParamDiff, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnTensorDescriptor_t dxDesc, void* dx, const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void* bnScale, void* dBnScaleResult, void* dBnBiasResult, double epsilon, const void* savedMean, const void* savedInvVariance);
typedef cudnnStatus_t CUDAAPI tcudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t* stDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType, cudnnDataType_t dataType, const int nbDims, const int dimA[]);
typedef cudnnStatus_t CUDAAPI tcudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void* theta, void* grid);
typedef cudnnStatus_t CUDAAPI tcudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void* dgrid, void* dtheta);
typedef cudnnStatus_t CUDAAPI tcudnnSpatialTfSamplerForward(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* grid, const void* beta, cudnnTensorDescriptor_t yDesc, void* y);
typedef cudnnStatus_t CUDAAPI tcudnnSpatialTfSamplerBackward(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx, const void* alphaDgrid, const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* grid, const void* betaDgrid, void* dgrid);
typedef cudnnStatus_t CUDAAPI tcudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* dropoutDesc);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc);
typedef cudnnStatus_t CUDAAPI tcudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed);
typedef cudnnStatus_t CUDAAPI tcudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed);
typedef cudnnStatus_t CUDAAPI tcudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float* dropout, void** states, unsigned long long* seed);
typedef cudnnStatus_t CUDAAPI tcudnnDropoutForward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t xdesc, const void* x, const cudnnTensorDescriptor_t ydesc, void* y, void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnDropoutBackward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t dydesc, const void* dy, const cudnnTensorDescriptor_t dxdesc, void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int* count);
typedef cudnnStatus_t CUDAAPI tcudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t* yDesc, void* y, const cudnnTensorDescriptor_t hyDesc, void* hy, const cudnnTensorDescriptor_t cyDesc, void* cy, const float findIntensity, const int requestedAlgoCount, int* returnedAlgoCount, cudnnAlgorithmPerformance_t* perfResults, void* workspace, size_t workSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int* count);
typedef cudnnStatus_t CUDAAPI tcudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t* yDesc, void* y, const cudnnTensorDescriptor_t hyDesc, void* hy, const cudnnTensorDescriptor_t cyDesc, void* cy, const float findIntensity, const int requestedAlgoCount, int* returnedAlgoCount, cudnnAlgorithmPerformance_t* perfResults, void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int* count);
typedef cudnnStatus_t CUDAAPI tcudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* yDesc, const void* y, const cudnnTensorDescriptor_t* dyDesc, const void* dy, const cudnnTensorDescriptor_t dhyDesc, const void* dhy, const cudnnTensorDescriptor_t dcyDesc, const void* dcy, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnTensorDescriptor_t* dxDesc, void* dx, const cudnnTensorDescriptor_t dhxDesc, void* dhx, const cudnnTensorDescriptor_t dcxDesc, void* dcx, const float findIntensity, const int requestedAlgoCount, int* returnedAlgoCount, cudnnAlgorithmPerformance_t* perfResults, void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int* count);
typedef cudnnStatus_t CUDAAPI tcudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t* yDesc, const void* y, const float findIntensity, const int requestedAlgoCount, int* returnedAlgoCount, cudnnAlgorithmPerformance_t* perfResults, const void* workspace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void* dw, const void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, const int minibatch, const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t* plan);
typedef cudnnStatus_t CUDAAPI tcudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan);
typedef cudnnStatus_t CUDAAPI tcudnnSetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize, const int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t dataType);
typedef cudnnStatus_t CUDAAPI tcudnnSetRNNProjectionLayers(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int recProjSize, const int outProjSize);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNProjectionLayers(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int* recProjSize, int* outProjSize);
typedef cudnnStatus_t CUDAAPI tcudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int* hiddenSize, int* numLayers, cudnnDropoutDescriptor_t* dropoutDesc, cudnnRNNInputMode_t* inputMode, cudnnDirectionMode_t* direction, cudnnRNNMode_t* mode, cudnnRNNAlgo_t* algo, cudnnDataType_t* dataType);
typedef cudnnStatus_t CUDAAPI tcudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t* mType);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNWorkspaceSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNTrainingReserveSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNParamsSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes, cudnnDataType_t dataType);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void* w, const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc, void** linLayerMat);
typedef cudnnStatus_t CUDAAPI tcudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void* w, const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc, void** linLayerBias);
typedef cudnnStatus_t CUDAAPI tcudnnRNNForwardInference(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t* yDesc, void* y, const cudnnTensorDescriptor_t hyDesc, void* hy, const cudnnTensorDescriptor_t cyDesc, void* cy, void* workspace, size_t workSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnRNNForwardTraining(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t* yDesc, void* y, const cudnnTensorDescriptor_t hyDesc, void* hy, const cudnnTensorDescriptor_t cyDesc, void* cy, void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnRNNBackwardData(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* yDesc, const void* y, const cudnnTensorDescriptor_t* dyDesc, const void* dy, const cudnnTensorDescriptor_t dhyDesc, const void* dhy, const cudnnTensorDescriptor_t dcyDesc, const void* dcy, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnTensorDescriptor_t* dxDesc, void* dx, const cudnnTensorDescriptor_t dhxDesc, void* dhx, const cudnnTensorDescriptor_t dcxDesc, void* dcx, void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnRNNBackwardWeights(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t* yDesc, const void* y, const void* workspace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void* dw, const void* reserveSpace, size_t reserveSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t* ctcLossDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType);
typedef cudnnStatus_t CUDAAPI tcudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t* compType);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc);
typedef cudnnStatus_t CUDAAPI tcudnnCTCLoss(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const void* probs, const int* labels, const int* labelLengths, const int* inputLengths, void* costs, const cudnnTensorDescriptor_t gradientsDesc, const void* gradients, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc, void* workspace, size_t workSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnGetCTCLossWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t gradientsDesc, const int* labels, const int* labelLengths, const int* inputLengths, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc, size_t* sizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t* algoDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm);
typedef cudnnStatus_t CUDAAPI tcudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t* algorithm);
typedef cudnnStatus_t CUDAAPI tcudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc);
typedef cudnnStatus_t CUDAAPI tcudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t* algoPerf, int numberToCreate);
typedef cudnnStatus_t CUDAAPI tcudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc, cudnnStatus_t status, float time, size_t memory);
typedef cudnnStatus_t CUDAAPI tcudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t* algoDesc, cudnnStatus_t* status, float* time, size_t* memory);
typedef cudnnStatus_t CUDAAPI tcudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t* algoPerf, int numberToDestroy);
typedef cudnnStatus_t CUDAAPI tcudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t* algoSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnSaveAlgorithm(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, void* algoSpace, size_t algoSpaceSizeInBytes);
typedef cudnnStatus_t CUDAAPI tcudnnRestoreAlgorithm(cudnnHandle_t handle, void* algoSpace, size_t algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t algoDesc);
typedef cudnnStatus_t CUDAAPI tcudnnSetCallback(unsigned mask, void* udata, cudnnCallback_t fptr);
typedef cudnnStatus_t CUDAAPI tcudnnGetCallback(unsigned* mask, void** udata, cudnnCallback_t* fptr);
typedef cudnnStatus_t CUDAAPI tcudnnSetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize, const int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t dataType);
typedef cudnnStatus_t CUDAAPI tcudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnDataType_t dataType);


/* Function declarations. */
extern tcuGetErrorString *cuGetErrorString;
extern tcuGetErrorName *cuGetErrorName;
extern tcuInit *cuInit;
extern tcuDriverGetVersion *cuDriverGetVersion;
extern tcuDeviceGet *cuDeviceGet;
extern tcuDeviceGetCount *cuDeviceGetCount;
extern tcuDeviceGetName *cuDeviceGetName;
extern tcuDeviceGetUuid *cuDeviceGetUuid;
extern tcuDeviceTotalMem_v2 *cuDeviceTotalMem_v2;
extern tcuDeviceGetAttribute *cuDeviceGetAttribute;
extern tcuDeviceGetProperties *cuDeviceGetProperties;
extern tcuDeviceComputeCapability *cuDeviceComputeCapability;
extern tcuDevicePrimaryCtxRetain *cuDevicePrimaryCtxRetain;
extern tcuDevicePrimaryCtxRelease *cuDevicePrimaryCtxRelease;
extern tcuDevicePrimaryCtxSetFlags *cuDevicePrimaryCtxSetFlags;
extern tcuDevicePrimaryCtxGetState *cuDevicePrimaryCtxGetState;
extern tcuDevicePrimaryCtxReset *cuDevicePrimaryCtxReset;
extern tcuCtxCreate_v2 *cuCtxCreate_v2;
extern tcuCtxDestroy_v2 *cuCtxDestroy_v2;
extern tcuCtxPushCurrent_v2 *cuCtxPushCurrent_v2;
extern tcuCtxPopCurrent_v2 *cuCtxPopCurrent_v2;
extern tcuCtxSetCurrent *cuCtxSetCurrent;
extern tcuCtxGetCurrent *cuCtxGetCurrent;
extern tcuCtxGetDevice *cuCtxGetDevice;
extern tcuCtxGetFlags *cuCtxGetFlags;
extern tcuCtxSynchronize *cuCtxSynchronize;
extern tcuCtxSetLimit *cuCtxSetLimit;
extern tcuCtxGetLimit *cuCtxGetLimit;
extern tcuCtxGetCacheConfig *cuCtxGetCacheConfig;
extern tcuCtxSetCacheConfig *cuCtxSetCacheConfig;
extern tcuCtxGetSharedMemConfig *cuCtxGetSharedMemConfig;
extern tcuCtxSetSharedMemConfig *cuCtxSetSharedMemConfig;
extern tcuCtxGetApiVersion *cuCtxGetApiVersion;
extern tcuCtxGetStreamPriorityRange *cuCtxGetStreamPriorityRange;
extern tcuCtxAttach *cuCtxAttach;
extern tcuCtxDetach *cuCtxDetach;
extern tcuModuleLoad *cuModuleLoad;
extern tcuModuleLoadData *cuModuleLoadData;
extern tcuModuleLoadDataEx *cuModuleLoadDataEx;
extern tcuModuleLoadFatBinary *cuModuleLoadFatBinary;
extern tcuModuleUnload *cuModuleUnload;
extern tcuModuleGetFunction *cuModuleGetFunction;
extern tcuModuleGetGlobal_v2 *cuModuleGetGlobal_v2;
extern tcuModuleGetTexRef *cuModuleGetTexRef;
extern tcuModuleGetSurfRef *cuModuleGetSurfRef;
extern tcuLinkCreate_v2 *cuLinkCreate_v2;
extern tcuLinkAddData_v2 *cuLinkAddData_v2;
extern tcuLinkAddFile_v2 *cuLinkAddFile_v2;
extern tcuLinkComplete *cuLinkComplete;
extern tcuLinkDestroy *cuLinkDestroy;
extern tcuMemGetInfo_v2 *cuMemGetInfo_v2;
extern tcuMemAlloc_v2 *cuMemAlloc_v2;
extern tcuMemAllocPitch_v2 *cuMemAllocPitch_v2;
extern tcuMemFree_v2 *cuMemFree_v2;
extern tcuMemGetAddressRange_v2 *cuMemGetAddressRange_v2;
extern tcuMemAllocHost_v2 *cuMemAllocHost_v2;
extern tcuMemFreeHost *cuMemFreeHost;
extern tcuMemHostAlloc *cuMemHostAlloc;
extern tcuMemHostGetDevicePointer_v2 *cuMemHostGetDevicePointer_v2;
extern tcuMemHostGetFlags *cuMemHostGetFlags;
extern tcuMemAllocManaged *cuMemAllocManaged;
extern tcuDeviceGetByPCIBusId *cuDeviceGetByPCIBusId;
extern tcuDeviceGetPCIBusId *cuDeviceGetPCIBusId;
extern tcuIpcGetEventHandle *cuIpcGetEventHandle;
extern tcuIpcOpenEventHandle *cuIpcOpenEventHandle;
extern tcuIpcGetMemHandle *cuIpcGetMemHandle;
extern tcuIpcOpenMemHandle *cuIpcOpenMemHandle;
extern tcuIpcCloseMemHandle *cuIpcCloseMemHandle;
extern tcuMemHostRegister_v2 *cuMemHostRegister_v2;
extern tcuMemHostUnregister *cuMemHostUnregister;
extern tcuMemcpy *cuMemcpy;
extern tcuMemcpyPeer *cuMemcpyPeer;
extern tcuMemcpyHtoD_v2 *cuMemcpyHtoD_v2;
extern tcuMemcpyDtoH_v2 *cuMemcpyDtoH_v2;
extern tcuMemcpyDtoD_v2 *cuMemcpyDtoD_v2;
extern tcuMemcpyDtoA_v2 *cuMemcpyDtoA_v2;
extern tcuMemcpyAtoD_v2 *cuMemcpyAtoD_v2;
extern tcuMemcpyHtoA_v2 *cuMemcpyHtoA_v2;
extern tcuMemcpyAtoH_v2 *cuMemcpyAtoH_v2;
extern tcuMemcpyAtoA_v2 *cuMemcpyAtoA_v2;
extern tcuMemcpy2D_v2 *cuMemcpy2D_v2;
extern tcuMemcpy2DUnaligned_v2 *cuMemcpy2DUnaligned_v2;
extern tcuMemcpy3D_v2 *cuMemcpy3D_v2;
extern tcuMemcpy3DPeer *cuMemcpy3DPeer;
extern tcuMemcpyAsync *cuMemcpyAsync;
extern tcuMemcpyPeerAsync *cuMemcpyPeerAsync;
extern tcuMemcpyHtoDAsync_v2 *cuMemcpyHtoDAsync_v2;
extern tcuMemcpyDtoHAsync_v2 *cuMemcpyDtoHAsync_v2;
extern tcuMemcpyDtoDAsync_v2 *cuMemcpyDtoDAsync_v2;
extern tcuMemcpyHtoAAsync_v2 *cuMemcpyHtoAAsync_v2;
extern tcuMemcpyAtoHAsync_v2 *cuMemcpyAtoHAsync_v2;
extern tcuMemcpy2DAsync_v2 *cuMemcpy2DAsync_v2;
extern tcuMemcpy3DAsync_v2 *cuMemcpy3DAsync_v2;
extern tcuMemcpy3DPeerAsync *cuMemcpy3DPeerAsync;
extern tcuMemsetD8_v2 *cuMemsetD8_v2;
extern tcuMemsetD16_v2 *cuMemsetD16_v2;
extern tcuMemsetD32_v2 *cuMemsetD32_v2;
extern tcuMemsetD2D8_v2 *cuMemsetD2D8_v2;
extern tcuMemsetD2D16_v2 *cuMemsetD2D16_v2;
extern tcuMemsetD2D32_v2 *cuMemsetD2D32_v2;
extern tcuMemsetD8Async *cuMemsetD8Async;
extern tcuMemsetD16Async *cuMemsetD16Async;
extern tcuMemsetD32Async *cuMemsetD32Async;
extern tcuMemsetD2D8Async *cuMemsetD2D8Async;
extern tcuMemsetD2D16Async *cuMemsetD2D16Async;
extern tcuMemsetD2D32Async *cuMemsetD2D32Async;
extern tcuArrayCreate_v2 *cuArrayCreate_v2;
extern tcuArrayGetDescriptor_v2 *cuArrayGetDescriptor_v2;
extern tcuArrayDestroy *cuArrayDestroy;
extern tcuArray3DCreate_v2 *cuArray3DCreate_v2;
extern tcuArray3DGetDescriptor_v2 *cuArray3DGetDescriptor_v2;
extern tcuMipmappedArrayCreate *cuMipmappedArrayCreate;
extern tcuMipmappedArrayGetLevel *cuMipmappedArrayGetLevel;
extern tcuMipmappedArrayDestroy *cuMipmappedArrayDestroy;
extern tcuPointerGetAttribute *cuPointerGetAttribute;
extern tcuMemPrefetchAsync *cuMemPrefetchAsync;
extern tcuMemAdvise *cuMemAdvise;
extern tcuMemRangeGetAttribute *cuMemRangeGetAttribute;
extern tcuMemRangeGetAttributes *cuMemRangeGetAttributes;
extern tcuPointerSetAttribute *cuPointerSetAttribute;
extern tcuPointerGetAttributes *cuPointerGetAttributes;
extern tcuStreamCreate *cuStreamCreate;
extern tcuStreamCreateWithPriority *cuStreamCreateWithPriority;
extern tcuStreamGetPriority *cuStreamGetPriority;
extern tcuStreamGetFlags *cuStreamGetFlags;
extern tcuStreamGetCtx *cuStreamGetCtx;
extern tcuStreamWaitEvent *cuStreamWaitEvent;
extern tcuStreamAddCallback *cuStreamAddCallback;
extern tcuStreamAttachMemAsync *cuStreamAttachMemAsync;
extern tcuStreamQuery *cuStreamQuery;
extern tcuStreamSynchronize *cuStreamSynchronize;
extern tcuStreamDestroy_v2 *cuStreamDestroy_v2;
extern tcuEventCreate *cuEventCreate;
extern tcuEventRecord *cuEventRecord;
extern tcuEventQuery *cuEventQuery;
extern tcuEventSynchronize *cuEventSynchronize;
extern tcuEventDestroy_v2 *cuEventDestroy_v2;
extern tcuEventElapsedTime *cuEventElapsedTime;
extern tcuStreamWaitValue32 *cuStreamWaitValue32;
extern tcuStreamWaitValue64 *cuStreamWaitValue64;
extern tcuStreamWriteValue32 *cuStreamWriteValue32;
extern tcuStreamWriteValue64 *cuStreamWriteValue64;
extern tcuStreamBatchMemOp *cuStreamBatchMemOp;
extern tcuFuncGetAttribute *cuFuncGetAttribute;
extern tcuFuncSetAttribute *cuFuncSetAttribute;
extern tcuFuncSetCacheConfig *cuFuncSetCacheConfig;
extern tcuFuncSetSharedMemConfig *cuFuncSetSharedMemConfig;
extern tcuLaunchKernel *cuLaunchKernel;
extern tcuLaunchCooperativeKernel *cuLaunchCooperativeKernel;
extern tcuLaunchCooperativeKernelMultiDevice *cuLaunchCooperativeKernelMultiDevice;
extern tcuFuncSetBlockShape *cuFuncSetBlockShape;
extern tcuFuncSetSharedSize *cuFuncSetSharedSize;
extern tcuParamSetSize *cuParamSetSize;
extern tcuParamSeti *cuParamSeti;
extern tcuParamSetf *cuParamSetf;
extern tcuParamSetv *cuParamSetv;
extern tcuLaunch *cuLaunch;
extern tcuLaunchGrid *cuLaunchGrid;
extern tcuLaunchGridAsync *cuLaunchGridAsync;
extern tcuParamSetTexRef *cuParamSetTexRef;
extern tcuOccupancyMaxActiveBlocksPerMultiprocessor *cuOccupancyMaxActiveBlocksPerMultiprocessor;
extern tcuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags *cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
extern tcuOccupancyMaxPotentialBlockSize *cuOccupancyMaxPotentialBlockSize;
extern tcuOccupancyMaxPotentialBlockSizeWithFlags *cuOccupancyMaxPotentialBlockSizeWithFlags;
extern tcuTexRefSetArray *cuTexRefSetArray;
extern tcuTexRefSetMipmappedArray *cuTexRefSetMipmappedArray;
extern tcuTexRefSetAddress_v2 *cuTexRefSetAddress_v2;
extern tcuTexRefSetAddress2D_v3 *cuTexRefSetAddress2D_v3;
extern tcuTexRefSetFormat *cuTexRefSetFormat;
extern tcuTexRefSetAddressMode *cuTexRefSetAddressMode;
extern tcuTexRefSetFilterMode *cuTexRefSetFilterMode;
extern tcuTexRefSetMipmapFilterMode *cuTexRefSetMipmapFilterMode;
extern tcuTexRefSetMipmapLevelBias *cuTexRefSetMipmapLevelBias;
extern tcuTexRefSetMipmapLevelClamp *cuTexRefSetMipmapLevelClamp;
extern tcuTexRefSetMaxAnisotropy *cuTexRefSetMaxAnisotropy;
extern tcuTexRefSetBorderColor *cuTexRefSetBorderColor;
extern tcuTexRefSetFlags *cuTexRefSetFlags;
extern tcuTexRefGetAddress_v2 *cuTexRefGetAddress_v2;
extern tcuTexRefGetArray *cuTexRefGetArray;
extern tcuTexRefGetMipmappedArray *cuTexRefGetMipmappedArray;
extern tcuTexRefGetAddressMode *cuTexRefGetAddressMode;
extern tcuTexRefGetFilterMode *cuTexRefGetFilterMode;
extern tcuTexRefGetFormat *cuTexRefGetFormat;
extern tcuTexRefGetMipmapFilterMode *cuTexRefGetMipmapFilterMode;
extern tcuTexRefGetMipmapLevelBias *cuTexRefGetMipmapLevelBias;
extern tcuTexRefGetMipmapLevelClamp *cuTexRefGetMipmapLevelClamp;
extern tcuTexRefGetMaxAnisotropy *cuTexRefGetMaxAnisotropy;
extern tcuTexRefGetBorderColor *cuTexRefGetBorderColor;
extern tcuTexRefGetFlags *cuTexRefGetFlags;
extern tcuTexRefCreate *cuTexRefCreate;
extern tcuTexRefDestroy *cuTexRefDestroy;
extern tcuSurfRefSetArray *cuSurfRefSetArray;
extern tcuSurfRefGetArray *cuSurfRefGetArray;
extern tcuTexObjectCreate *cuTexObjectCreate;
extern tcuTexObjectDestroy *cuTexObjectDestroy;
extern tcuTexObjectGetResourceDesc *cuTexObjectGetResourceDesc;
extern tcuTexObjectGetTextureDesc *cuTexObjectGetTextureDesc;
extern tcuTexObjectGetResourceViewDesc *cuTexObjectGetResourceViewDesc;
extern tcuSurfObjectCreate *cuSurfObjectCreate;
extern tcuSurfObjectDestroy *cuSurfObjectDestroy;
extern tcuSurfObjectGetResourceDesc *cuSurfObjectGetResourceDesc;
extern tcuDeviceCanAccessPeer *cuDeviceCanAccessPeer;
extern tcuCtxEnablePeerAccess *cuCtxEnablePeerAccess;
extern tcuCtxDisablePeerAccess *cuCtxDisablePeerAccess;
extern tcuDeviceGetP2PAttribute *cuDeviceGetP2PAttribute;
extern tcuGraphicsUnregisterResource *cuGraphicsUnregisterResource;
extern tcuGraphicsSubResourceGetMappedArray *cuGraphicsSubResourceGetMappedArray;
extern tcuGraphicsResourceGetMappedMipmappedArray *cuGraphicsResourceGetMappedMipmappedArray;
extern tcuGraphicsResourceGetMappedPointer_v2 *cuGraphicsResourceGetMappedPointer_v2;
extern tcuGraphicsResourceSetMapFlags_v2 *cuGraphicsResourceSetMapFlags_v2;
extern tcuGraphicsMapResources *cuGraphicsMapResources;
extern tcuGraphicsUnmapResources *cuGraphicsUnmapResources;
extern tcuGetExportTable *cuGetExportTable;

extern tcuGraphicsGLRegisterBuffer *cuGraphicsGLRegisterBuffer;
extern tcuGraphicsGLRegisterImage *cuGraphicsGLRegisterImage;
extern tcuGLGetDevices_v2 *cuGLGetDevices_v2;
extern tcuGLCtxCreate_v2 *cuGLCtxCreate_v2;
extern tcuGLInit *cuGLInit;
extern tcuGLRegisterBufferObject *cuGLRegisterBufferObject;
extern tcuGLMapBufferObject_v2 *cuGLMapBufferObject_v2;
extern tcuGLUnmapBufferObject *cuGLUnmapBufferObject;
extern tcuGLUnregisterBufferObject *cuGLUnregisterBufferObject;
extern tcuGLSetBufferObjectMapFlags *cuGLSetBufferObjectMapFlags;
extern tcuGLMapBufferObjectAsync_v2 *cuGLMapBufferObjectAsync_v2;
extern tcuGLUnmapBufferObjectAsync *cuGLUnmapBufferObjectAsync;

extern tnvrtcGetErrorString *nvrtcGetErrorString;
extern tnvrtcVersion *nvrtcVersion;
extern tnvrtcCreateProgram *nvrtcCreateProgram;
extern tnvrtcDestroyProgram *nvrtcDestroyProgram;
extern tnvrtcCompileProgram *nvrtcCompileProgram;
extern tnvrtcGetPTXSize *nvrtcGetPTXSize;
extern tnvrtcGetPTX *nvrtcGetPTX;
extern tnvrtcGetProgramLogSize *nvrtcGetProgramLogSize;
extern tnvrtcGetProgramLog *nvrtcGetProgramLog;
extern tnvrtcAddNameExpression *nvrtcAddNameExpression;
extern tnvrtcGetLoweredName *nvrtcGetLoweredName;

extern tcudnnGetVersion *cudnnGetVersion;
extern tcudnnGetCudartVersion *cudnnGetCudartVersion;
extern tcudnnGetErrorString *cudnnGetErrorString;
extern tcudnnQueryRuntimeError *cudnnQueryRuntimeError;
extern tcudnnGetProperty *cudnnGetProperty;
extern tcudnnCreate *cudnnCreate;
extern tcudnnDestroy *cudnnDestroy;
extern tcudnnSetStream *cudnnSetStream;
extern tcudnnGetStream *cudnnGetStream;
extern tcudnnCreateTensorDescriptor *cudnnCreateTensorDescriptor;
extern tcudnnSetTensor4dDescriptor *cudnnSetTensor4dDescriptor;
extern tcudnnSetTensor4dDescriptorEx *cudnnSetTensor4dDescriptorEx;
extern tcudnnGetTensor4dDescriptor *cudnnGetTensor4dDescriptor;
extern tcudnnSetTensorNdDescriptor *cudnnSetTensorNdDescriptor;
extern tcudnnSetTensorNdDescriptorEx *cudnnSetTensorNdDescriptorEx;
extern tcudnnGetTensorNdDescriptor *cudnnGetTensorNdDescriptor;
extern tcudnnGetTensorSizeInBytes *cudnnGetTensorSizeInBytes;
extern tcudnnDestroyTensorDescriptor *cudnnDestroyTensorDescriptor;
extern tcudnnTransformTensor *cudnnTransformTensor;
extern tcudnnAddTensor *cudnnAddTensor;
extern tcudnnCreateOpTensorDescriptor *cudnnCreateOpTensorDescriptor;
extern tcudnnSetOpTensorDescriptor *cudnnSetOpTensorDescriptor;
extern tcudnnGetOpTensorDescriptor *cudnnGetOpTensorDescriptor;
extern tcudnnDestroyOpTensorDescriptor *cudnnDestroyOpTensorDescriptor;
extern tcudnnOpTensor *cudnnOpTensor;
extern tcudnnCreateReduceTensorDescriptor *cudnnCreateReduceTensorDescriptor;
extern tcudnnSetReduceTensorDescriptor *cudnnSetReduceTensorDescriptor;
extern tcudnnGetReduceTensorDescriptor *cudnnGetReduceTensorDescriptor;
extern tcudnnDestroyReduceTensorDescriptor *cudnnDestroyReduceTensorDescriptor;
extern tcudnnGetReductionIndicesSize *cudnnGetReductionIndicesSize;
extern tcudnnGetReductionWorkspaceSize *cudnnGetReductionWorkspaceSize;
extern tcudnnReduceTensor *cudnnReduceTensor;
extern tcudnnSetTensor *cudnnSetTensor;
extern tcudnnScaleTensor *cudnnScaleTensor;
extern tcudnnCreateFilterDescriptor *cudnnCreateFilterDescriptor;
extern tcudnnSetFilter4dDescriptor *cudnnSetFilter4dDescriptor;
extern tcudnnGetFilter4dDescriptor *cudnnGetFilter4dDescriptor;
extern tcudnnSetFilterNdDescriptor *cudnnSetFilterNdDescriptor;
extern tcudnnGetFilterNdDescriptor *cudnnGetFilterNdDescriptor;
extern tcudnnDestroyFilterDescriptor *cudnnDestroyFilterDescriptor;
extern tcudnnCreateConvolutionDescriptor *cudnnCreateConvolutionDescriptor;
extern tcudnnSetConvolutionMathType *cudnnSetConvolutionMathType;
extern tcudnnGetConvolutionMathType *cudnnGetConvolutionMathType;
extern tcudnnSetConvolutionGroupCount *cudnnSetConvolutionGroupCount;
extern tcudnnGetConvolutionGroupCount *cudnnGetConvolutionGroupCount;
extern tcudnnSetConvolution2dDescriptor *cudnnSetConvolution2dDescriptor;
extern tcudnnGetConvolution2dDescriptor *cudnnGetConvolution2dDescriptor;
extern tcudnnGetConvolution2dForwardOutputDim *cudnnGetConvolution2dForwardOutputDim;
extern tcudnnSetConvolutionNdDescriptor *cudnnSetConvolutionNdDescriptor;
extern tcudnnGetConvolutionNdDescriptor *cudnnGetConvolutionNdDescriptor;
extern tcudnnGetConvolutionNdForwardOutputDim *cudnnGetConvolutionNdForwardOutputDim;
extern tcudnnDestroyConvolutionDescriptor *cudnnDestroyConvolutionDescriptor;
extern tcudnnGetConvolutionForwardAlgorithmMaxCount *cudnnGetConvolutionForwardAlgorithmMaxCount;
extern tcudnnFindConvolutionForwardAlgorithm *cudnnFindConvolutionForwardAlgorithm;
extern tcudnnFindConvolutionForwardAlgorithmEx *cudnnFindConvolutionForwardAlgorithmEx;
extern tcudnnGetConvolutionForwardAlgorithm *cudnnGetConvolutionForwardAlgorithm;
extern tcudnnGetConvolutionForwardAlgorithm_v7 *cudnnGetConvolutionForwardAlgorithm_v7;
extern tcudnnGetConvolutionForwardWorkspaceSize *cudnnGetConvolutionForwardWorkspaceSize;
extern tcudnnConvolutionForward *cudnnConvolutionForward;
extern tcudnnConvolutionBiasActivationForward *cudnnConvolutionBiasActivationForward;
extern tcudnnConvolutionBackwardBias *cudnnConvolutionBackwardBias;
extern tcudnnGetConvolutionBackwardFilterAlgorithmMaxCount *cudnnGetConvolutionBackwardFilterAlgorithmMaxCount;
extern tcudnnFindConvolutionBackwardFilterAlgorithm *cudnnFindConvolutionBackwardFilterAlgorithm;
extern tcudnnFindConvolutionBackwardFilterAlgorithmEx *cudnnFindConvolutionBackwardFilterAlgorithmEx;
extern tcudnnGetConvolutionBackwardFilterAlgorithm *cudnnGetConvolutionBackwardFilterAlgorithm;
extern tcudnnGetConvolutionBackwardFilterAlgorithm_v7 *cudnnGetConvolutionBackwardFilterAlgorithm_v7;
extern tcudnnGetConvolutionBackwardFilterWorkspaceSize *cudnnGetConvolutionBackwardFilterWorkspaceSize;
extern tcudnnConvolutionBackwardFilter *cudnnConvolutionBackwardFilter;
extern tcudnnGetConvolutionBackwardDataAlgorithmMaxCount *cudnnGetConvolutionBackwardDataAlgorithmMaxCount;
extern tcudnnFindConvolutionBackwardDataAlgorithm *cudnnFindConvolutionBackwardDataAlgorithm;
extern tcudnnFindConvolutionBackwardDataAlgorithmEx *cudnnFindConvolutionBackwardDataAlgorithmEx;
extern tcudnnGetConvolutionBackwardDataAlgorithm *cudnnGetConvolutionBackwardDataAlgorithm;
extern tcudnnGetConvolutionBackwardDataAlgorithm_v7 *cudnnGetConvolutionBackwardDataAlgorithm_v7;
extern tcudnnGetConvolutionBackwardDataWorkspaceSize *cudnnGetConvolutionBackwardDataWorkspaceSize;
extern tcudnnConvolutionBackwardData *cudnnConvolutionBackwardData;
extern tcudnnIm2Col *cudnnIm2Col;
extern tcudnnSoftmaxForward *cudnnSoftmaxForward;
extern tcudnnSoftmaxBackward *cudnnSoftmaxBackward;
extern tcudnnCreatePoolingDescriptor *cudnnCreatePoolingDescriptor;
extern tcudnnSetPooling2dDescriptor *cudnnSetPooling2dDescriptor;
extern tcudnnGetPooling2dDescriptor *cudnnGetPooling2dDescriptor;
extern tcudnnSetPoolingNdDescriptor *cudnnSetPoolingNdDescriptor;
extern tcudnnGetPoolingNdDescriptor *cudnnGetPoolingNdDescriptor;
extern tcudnnGetPoolingNdForwardOutputDim *cudnnGetPoolingNdForwardOutputDim;
extern tcudnnGetPooling2dForwardOutputDim *cudnnGetPooling2dForwardOutputDim;
extern tcudnnDestroyPoolingDescriptor *cudnnDestroyPoolingDescriptor;
extern tcudnnPoolingForward *cudnnPoolingForward;
extern tcudnnPoolingBackward *cudnnPoolingBackward;
extern tcudnnCreateActivationDescriptor *cudnnCreateActivationDescriptor;
extern tcudnnSetActivationDescriptor *cudnnSetActivationDescriptor;
extern tcudnnGetActivationDescriptor *cudnnGetActivationDescriptor;
extern tcudnnDestroyActivationDescriptor *cudnnDestroyActivationDescriptor;
extern tcudnnActivationForward *cudnnActivationForward;
extern tcudnnActivationBackward *cudnnActivationBackward;
extern tcudnnCreateLRNDescriptor *cudnnCreateLRNDescriptor;
extern tcudnnSetLRNDescriptor *cudnnSetLRNDescriptor;
extern tcudnnGetLRNDescriptor *cudnnGetLRNDescriptor;
extern tcudnnDestroyLRNDescriptor *cudnnDestroyLRNDescriptor;
extern tcudnnLRNCrossChannelForward *cudnnLRNCrossChannelForward;
extern tcudnnLRNCrossChannelBackward *cudnnLRNCrossChannelBackward;
extern tcudnnDivisiveNormalizationForward *cudnnDivisiveNormalizationForward;
extern tcudnnDivisiveNormalizationBackward *cudnnDivisiveNormalizationBackward;
extern tcudnnDeriveBNTensorDescriptor *cudnnDeriveBNTensorDescriptor;
extern tcudnnBatchNormalizationForwardTraining *cudnnBatchNormalizationForwardTraining;
extern tcudnnBatchNormalizationForwardInference *cudnnBatchNormalizationForwardInference;
extern tcudnnBatchNormalizationBackward *cudnnBatchNormalizationBackward;
extern tcudnnCreateSpatialTransformerDescriptor *cudnnCreateSpatialTransformerDescriptor;
extern tcudnnSetSpatialTransformerNdDescriptor *cudnnSetSpatialTransformerNdDescriptor;
extern tcudnnDestroySpatialTransformerDescriptor *cudnnDestroySpatialTransformerDescriptor;
extern tcudnnSpatialTfGridGeneratorForward *cudnnSpatialTfGridGeneratorForward;
extern tcudnnSpatialTfGridGeneratorBackward *cudnnSpatialTfGridGeneratorBackward;
extern tcudnnSpatialTfSamplerForward *cudnnSpatialTfSamplerForward;
extern tcudnnSpatialTfSamplerBackward *cudnnSpatialTfSamplerBackward;
extern tcudnnCreateDropoutDescriptor *cudnnCreateDropoutDescriptor;
extern tcudnnDestroyDropoutDescriptor *cudnnDestroyDropoutDescriptor;
extern tcudnnDropoutGetStatesSize *cudnnDropoutGetStatesSize;
extern tcudnnDropoutGetReserveSpaceSize *cudnnDropoutGetReserveSpaceSize;
extern tcudnnSetDropoutDescriptor *cudnnSetDropoutDescriptor;
extern tcudnnRestoreDropoutDescriptor *cudnnRestoreDropoutDescriptor;
extern tcudnnGetDropoutDescriptor *cudnnGetDropoutDescriptor;
extern tcudnnDropoutForward *cudnnDropoutForward;
extern tcudnnDropoutBackward *cudnnDropoutBackward;
extern tcudnnCreateRNNDescriptor *cudnnCreateRNNDescriptor;
extern tcudnnDestroyRNNDescriptor *cudnnDestroyRNNDescriptor;
extern tcudnnGetRNNForwardInferenceAlgorithmMaxCount *cudnnGetRNNForwardInferenceAlgorithmMaxCount;
extern tcudnnFindRNNForwardInferenceAlgorithmEx *cudnnFindRNNForwardInferenceAlgorithmEx;
extern tcudnnGetRNNForwardTrainingAlgorithmMaxCount *cudnnGetRNNForwardTrainingAlgorithmMaxCount;
extern tcudnnFindRNNForwardTrainingAlgorithmEx *cudnnFindRNNForwardTrainingAlgorithmEx;
extern tcudnnGetRNNBackwardDataAlgorithmMaxCount *cudnnGetRNNBackwardDataAlgorithmMaxCount;
extern tcudnnFindRNNBackwardDataAlgorithmEx *cudnnFindRNNBackwardDataAlgorithmEx;
extern tcudnnGetRNNBackwardWeightsAlgorithmMaxCount *cudnnGetRNNBackwardWeightsAlgorithmMaxCount;
extern tcudnnFindRNNBackwardWeightsAlgorithmEx *cudnnFindRNNBackwardWeightsAlgorithmEx;
extern tcudnnCreatePersistentRNNPlan *cudnnCreatePersistentRNNPlan;
extern tcudnnSetPersistentRNNPlan *cudnnSetPersistentRNNPlan;
extern tcudnnDestroyPersistentRNNPlan *cudnnDestroyPersistentRNNPlan;
extern tcudnnSetRNNDescriptor *cudnnSetRNNDescriptor;
extern tcudnnSetRNNProjectionLayers *cudnnSetRNNProjectionLayers;
extern tcudnnGetRNNProjectionLayers *cudnnGetRNNProjectionLayers;
extern tcudnnSetRNNAlgorithmDescriptor *cudnnSetRNNAlgorithmDescriptor;
extern tcudnnGetRNNDescriptor *cudnnGetRNNDescriptor;
extern tcudnnSetRNNMatrixMathType *cudnnSetRNNMatrixMathType;
extern tcudnnGetRNNMatrixMathType *cudnnGetRNNMatrixMathType;
extern tcudnnGetRNNWorkspaceSize *cudnnGetRNNWorkspaceSize;
extern tcudnnGetRNNTrainingReserveSize *cudnnGetRNNTrainingReserveSize;
extern tcudnnGetRNNParamsSize *cudnnGetRNNParamsSize;
extern tcudnnGetRNNLinLayerMatrixParams *cudnnGetRNNLinLayerMatrixParams;
extern tcudnnGetRNNLinLayerBiasParams *cudnnGetRNNLinLayerBiasParams;
extern tcudnnRNNForwardInference *cudnnRNNForwardInference;
extern tcudnnRNNForwardTraining *cudnnRNNForwardTraining;
extern tcudnnRNNBackwardData *cudnnRNNBackwardData;
extern tcudnnRNNBackwardWeights *cudnnRNNBackwardWeights;
extern tcudnnCreateCTCLossDescriptor *cudnnCreateCTCLossDescriptor;
extern tcudnnSetCTCLossDescriptor *cudnnSetCTCLossDescriptor;
extern tcudnnGetCTCLossDescriptor *cudnnGetCTCLossDescriptor;
extern tcudnnDestroyCTCLossDescriptor *cudnnDestroyCTCLossDescriptor;
extern tcudnnCTCLoss *cudnnCTCLoss;
extern tcudnnGetCTCLossWorkspaceSize *cudnnGetCTCLossWorkspaceSize;
extern tcudnnCreateAlgorithmDescriptor *cudnnCreateAlgorithmDescriptor;
extern tcudnnSetAlgorithmDescriptor *cudnnSetAlgorithmDescriptor;
extern tcudnnGetAlgorithmDescriptor *cudnnGetAlgorithmDescriptor;
extern tcudnnCopyAlgorithmDescriptor *cudnnCopyAlgorithmDescriptor;
extern tcudnnDestroyAlgorithmDescriptor *cudnnDestroyAlgorithmDescriptor;
extern tcudnnCreateAlgorithmPerformance *cudnnCreateAlgorithmPerformance;
extern tcudnnSetAlgorithmPerformance *cudnnSetAlgorithmPerformance;
extern tcudnnGetAlgorithmPerformance *cudnnGetAlgorithmPerformance;
extern tcudnnDestroyAlgorithmPerformance *cudnnDestroyAlgorithmPerformance;
extern tcudnnGetAlgorithmSpaceSize *cudnnGetAlgorithmSpaceSize;
extern tcudnnSaveAlgorithm *cudnnSaveAlgorithm;
extern tcudnnRestoreAlgorithm *cudnnRestoreAlgorithm;
extern tcudnnSetCallback *cudnnSetCallback;
extern tcudnnGetCallback *cudnnGetCallback;
extern tcudnnSetRNNDescriptor_v6 *cudnnSetRNNDescriptor_v6;
extern tcudnnSetRNNDescriptor_v5 *cudnnSetRNNDescriptor_v5;


enum {
  CUEW_SUCCESS = 0,
  CUEW_ERROR_OPEN_FAILED = -1,
  CUEW_ERROR_ATEXIT_FAILED = -2,
};

enum {
	CUEW_INIT_CUDA  = (1 << 0),
	CUEW_INIT_NVRTC = (1 << 1),
	CUEW_INIT_CUDNN = (1 << 2),
};

int cuewInit(cuuint32_t flags);
const char *cuewErrorString(CUresult result);
const char *cuewCompilerPath(void);
int cuewCompilerVersion(void);
int cuewNvrtcVersion(void);

#ifdef __cplusplus
}
#endif

#endif  /* __CUEW_H__ */
