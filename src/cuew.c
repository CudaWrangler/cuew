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

#include <cuew.h>
#include <assert.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  include <windows.h>

/* Utility macros. */

typedef HMODULE DynamicLibrary;

#  define dynamic_library_open(path)         LoadLibrary(path)
#  define dynamic_library_close(lib)         FreeLibrary(lib)
#  define dynamic_library_find(lib, symbol)  GetProcAddress(lib, symbol)
#else
#  include <dlfcn.h>

typedef void* DynamicLibrary;

#  define dynamic_library_open(path)         dlopen(path, RTLD_NOW)
#  define dynamic_library_close(lib)         dlclose(lib)
#  define dynamic_library_find(lib, symbol)  dlsym(lib, symbol)
#endif
#define DL_LIBRARY_FIND_CHECKED(name) \
        name = (t##name)dynamic_library_find(lib, #name);

#define DL_LIBRARY_FIND(name) \
        name = (t##name)dynamic_library_find(lib, #name); \
        assert(name);

static DynamicLibrary lib;
/* Function definitions. */
tcuInit cuInit;
tcuDriverGetVersion cuDriverGetVersion;
tcuDeviceGet cuDeviceGet;
tcuDeviceGetCount cuDeviceGetCount;
tcuDeviceGetName cuDeviceGetName;
tcuDeviceTotalMem_v2 cuDeviceTotalMem_v2;
tcuDeviceGetAttribute cuDeviceGetAttribute;
tcuDeviceGetProperties cuDeviceGetProperties;
tcuDeviceComputeCapability cuDeviceComputeCapability;
tcuCtxCreate_v2 cuCtxCreate_v2;
tcuCtxDestroy_v2 cuCtxDestroy_v2;
tcuCtxPushCurrent_v2 cuCtxPushCurrent_v2;
tcuCtxPopCurrent_v2 cuCtxPopCurrent_v2;
tcuCtxSetCurrent cuCtxSetCurrent;
tcuCtxGetCurrent cuCtxGetCurrent;
tcuCtxGetDevice cuCtxGetDevice;
tcuCtxSynchronize cuCtxSynchronize;
tcuCtxSetLimit cuCtxSetLimit;
tcuCtxGetLimit cuCtxGetLimit;
tcuCtxGetCacheConfig cuCtxGetCacheConfig;
tcuCtxSetCacheConfig cuCtxSetCacheConfig;
tcuCtxGetSharedMemConfig cuCtxGetSharedMemConfig;
tcuCtxSetSharedMemConfig cuCtxSetSharedMemConfig;
tcuCtxGetApiVersion cuCtxGetApiVersion;
tcuCtxAttach cuCtxAttach;
tcuCtxDetach cuCtxDetach;
tcuModuleLoad cuModuleLoad;
tcuModuleLoadData cuModuleLoadData;
tcuModuleLoadDataEx cuModuleLoadDataEx;
tcuModuleLoadFatBinary cuModuleLoadFatBinary;
tcuModuleUnload cuModuleUnload;
tcuModuleGetFunction cuModuleGetFunction;
tcuModuleGetGlobal_v2 cuModuleGetGlobal_v2;
tcuModuleGetTexRef cuModuleGetTexRef;
tcuModuleGetSurfRef cuModuleGetSurfRef;
tcuMemGetInfo_v2 cuMemGetInfo_v2;
tcuMemAlloc_v2 cuMemAlloc_v2;
tcuMemAllocPitch_v2 cuMemAllocPitch_v2;
tcuMemFree_v2 cuMemFree_v2;
tcuMemGetAddressRange_v2 cuMemGetAddressRange_v2;
tcuMemAllocHost_v2 cuMemAllocHost_v2;
tcuMemFreeHost cuMemFreeHost;
tcuMemHostAlloc cuMemHostAlloc;
tcuMemHostGetDevicePointer_v2 cuMemHostGetDevicePointer_v2;
tcuMemHostGetFlags cuMemHostGetFlags;
tcuDeviceGetByPCIBusId cuDeviceGetByPCIBusId;
tcuDeviceGetPCIBusId cuDeviceGetPCIBusId;
tcuIpcGetEventHandle cuIpcGetEventHandle;
tcuIpcOpenEventHandle cuIpcOpenEventHandle;
tcuIpcGetMemHandle cuIpcGetMemHandle;
tcuIpcOpenMemHandle cuIpcOpenMemHandle;
tcuIpcCloseMemHandle cuIpcCloseMemHandle;
tcuMemHostRegister cuMemHostRegister;
tcuMemHostUnregister cuMemHostUnregister;
tcuMemcpy cuMemcpy;
tcuMemcpyPeer cuMemcpyPeer;
tcuMemcpyHtoD_v2 cuMemcpyHtoD_v2;
tcuMemcpyDtoH_v2 cuMemcpyDtoH_v2;
tcuMemcpyDtoD_v2 cuMemcpyDtoD_v2;
tcuMemcpyDtoA_v2 cuMemcpyDtoA_v2;
tcuMemcpyAtoD_v2 cuMemcpyAtoD_v2;
tcuMemcpyHtoA_v2 cuMemcpyHtoA_v2;
tcuMemcpyAtoH_v2 cuMemcpyAtoH_v2;
tcuMemcpyAtoA_v2 cuMemcpyAtoA_v2;
tcuMemcpy2D_v2 cuMemcpy2D_v2;
tcuMemcpy2DUnaligned_v2 cuMemcpy2DUnaligned_v2;
tcuMemcpy3D_v2 cuMemcpy3D_v2;
tcuMemcpy3DPeer cuMemcpy3DPeer;
tcuMemcpyAsync cuMemcpyAsync;
tcuMemcpyPeerAsync cuMemcpyPeerAsync;
tcuMemcpyHtoDAsync_v2 cuMemcpyHtoDAsync_v2;
tcuMemcpyDtoHAsync_v2 cuMemcpyDtoHAsync_v2;
tcuMemcpyDtoDAsync_v2 cuMemcpyDtoDAsync_v2;
tcuMemcpyHtoAAsync_v2 cuMemcpyHtoAAsync_v2;
tcuMemcpyAtoHAsync_v2 cuMemcpyAtoHAsync_v2;
tcuMemcpy2DAsync_v2 cuMemcpy2DAsync_v2;
tcuMemcpy3DAsync_v2 cuMemcpy3DAsync_v2;
tcuMemcpy3DPeerAsync cuMemcpy3DPeerAsync;
tcuMemsetD8_v2 cuMemsetD8_v2;
tcuMemsetD16_v2 cuMemsetD16_v2;
tcuMemsetD32_v2 cuMemsetD32_v2;
tcuMemsetD2D8_v2 cuMemsetD2D8_v2;
tcuMemsetD2D16_v2 cuMemsetD2D16_v2;
tcuMemsetD2D32_v2 cuMemsetD2D32_v2;
tcuMemsetD8Async cuMemsetD8Async;
tcuMemsetD16Async cuMemsetD16Async;
tcuMemsetD32Async cuMemsetD32Async;
tcuMemsetD2D8Async cuMemsetD2D8Async;
tcuMemsetD2D16Async cuMemsetD2D16Async;
tcuMemsetD2D32Async cuMemsetD2D32Async;
tcuArrayCreate_v2 cuArrayCreate_v2;
tcuArrayGetDescriptor_v2 cuArrayGetDescriptor_v2;
tcuArrayDestroy cuArrayDestroy;
tcuArray3DCreate_v2 cuArray3DCreate_v2;
tcuArray3DGetDescriptor_v2 cuArray3DGetDescriptor_v2;
tcuMipmappedArrayCreate cuMipmappedArrayCreate;
tcuMipmappedArrayGetLevel cuMipmappedArrayGetLevel;
tcuMipmappedArrayDestroy cuMipmappedArrayDestroy;
tcuPointerGetAttribute cuPointerGetAttribute;
tcuStreamCreate cuStreamCreate;
tcuStreamWaitEvent cuStreamWaitEvent;
tcuStreamAddCallback cuStreamAddCallback;
tcuStreamQuery cuStreamQuery;
tcuStreamSynchronize cuStreamSynchronize;
tcuStreamDestroy_v2 cuStreamDestroy_v2;
tcuEventCreate cuEventCreate;
tcuEventRecord cuEventRecord;
tcuEventQuery cuEventQuery;
tcuEventSynchronize cuEventSynchronize;
tcuEventDestroy_v2 cuEventDestroy_v2;
tcuEventElapsedTime cuEventElapsedTime;
tcuFuncGetAttribute cuFuncGetAttribute;
tcuFuncSetCacheConfig cuFuncSetCacheConfig;
tcuFuncSetSharedMemConfig cuFuncSetSharedMemConfig;
tcuLaunchKernel cuLaunchKernel;
tcuFuncSetBlockShape cuFuncSetBlockShape;
tcuFuncSetSharedSize cuFuncSetSharedSize;
tcuParamSetSize cuParamSetSize;
tcuParamSeti cuParamSeti;
tcuParamSetf cuParamSetf;
tcuParamSetv cuParamSetv;
tcuLaunch cuLaunch;
tcuLaunchGrid cuLaunchGrid;
tcuLaunchGridAsync cuLaunchGridAsync;
tcuParamSetTexRef cuParamSetTexRef;
tcuTexRefSetArray cuTexRefSetArray;
tcuTexRefSetMipmappedArray cuTexRefSetMipmappedArray;
tcuTexRefSetAddress_v2 cuTexRefSetAddress_v2;
tcuTexRefSetAddress2D_v3 cuTexRefSetAddress2D_v3;
tcuTexRefSetFormat cuTexRefSetFormat;
tcuTexRefSetAddressMode cuTexRefSetAddressMode;
tcuTexRefSetFilterMode cuTexRefSetFilterMode;
tcuTexRefSetMipmapFilterMode cuTexRefSetMipmapFilterMode;
tcuTexRefSetMipmapLevelBias cuTexRefSetMipmapLevelBias;
tcuTexRefSetMipmapLevelClamp cuTexRefSetMipmapLevelClamp;
tcuTexRefSetMaxAnisotropy cuTexRefSetMaxAnisotropy;
tcuTexRefSetFlags cuTexRefSetFlags;
tcuTexRefGetAddress_v2 cuTexRefGetAddress_v2;
tcuTexRefGetArray cuTexRefGetArray;
tcuTexRefGetMipmappedArray cuTexRefGetMipmappedArray;
tcuTexRefGetAddressMode cuTexRefGetAddressMode;
tcuTexRefGetFilterMode cuTexRefGetFilterMode;
tcuTexRefGetFormat cuTexRefGetFormat;
tcuTexRefGetMipmapFilterMode cuTexRefGetMipmapFilterMode;
tcuTexRefGetMipmapLevelBias cuTexRefGetMipmapLevelBias;
tcuTexRefGetMipmapLevelClamp cuTexRefGetMipmapLevelClamp;
tcuTexRefGetMaxAnisotropy cuTexRefGetMaxAnisotropy;
tcuTexRefGetFlags cuTexRefGetFlags;
tcuTexRefCreate cuTexRefCreate;
tcuTexRefDestroy cuTexRefDestroy;
tcuSurfRefSetArray cuSurfRefSetArray;
tcuSurfRefGetArray cuSurfRefGetArray;
tcuTexObjectCreate cuTexObjectCreate;
tcuTexObjectDestroy cuTexObjectDestroy;
tcuTexObjectGetResourceDesc cuTexObjectGetResourceDesc;
tcuTexObjectGetTextureDesc cuTexObjectGetTextureDesc;
tcuTexObjectGetResourceViewDesc cuTexObjectGetResourceViewDesc;
tcuSurfObjectCreate cuSurfObjectCreate;
tcuSurfObjectDestroy cuSurfObjectDestroy;
tcuSurfObjectGetResourceDesc cuSurfObjectGetResourceDesc;
tcuDeviceCanAccessPeer cuDeviceCanAccessPeer;
tcuCtxEnablePeerAccess cuCtxEnablePeerAccess;
tcuCtxDisablePeerAccess cuCtxDisablePeerAccess;
tcuGraphicsUnregisterResource cuGraphicsUnregisterResource;
tcuGraphicsSubResourceGetMappedArray cuGraphicsSubResourceGetMappedArray;
tcuGraphicsResourceGetMappedMipmappedArray cuGraphicsResourceGetMappedMipmappedArray;
tcuGraphicsResourceGetMappedPointer_v2 cuGraphicsResourceGetMappedPointer_v2;
tcuGraphicsResourceSetMapFlags cuGraphicsResourceSetMapFlags;
tcuGraphicsMapResources cuGraphicsMapResources;
tcuGraphicsUnmapResources cuGraphicsUnmapResources;
tcuGetExportTable cuGetExportTable;

tcuGraphicsGLRegisterBuffer cuGraphicsGLRegisterBuffer;
tcuGraphicsGLRegisterImage cuGraphicsGLRegisterImage;
tcuGLGetDevices cuGLGetDevices;
tcuGLCtxCreate_v2 cuGLCtxCreate_v2;
tcuGLInit cuGLInit;
tcuGLRegisterBufferObject cuGLRegisterBufferObject;
tcuGLMapBufferObject_v2 cuGLMapBufferObject_v2;
tcuGLUnmapBufferObject cuGLUnmapBufferObject;
tcuGLUnregisterBufferObject cuGLUnregisterBufferObject;
tcuGLSetBufferObjectMapFlags cuGLSetBufferObjectMapFlags;
tcuGLMapBufferObjectAsync_v2 cuGLMapBufferObjectAsync_v2;
tcuGLUnmapBufferObjectAsync cuGLUnmapBufferObjectAsync;


static void cuewExit(void) {
  if(lib != NULL) {
    /*  Ignore errors. */
    dynamic_library_close(lib);
    lib = NULL;
  }
}

/* Implementation function. */
int cuewInit(void) {
  /* Library paths. */
#ifdef _WIN32
  /* Expected in c:/windows/system or similar, no path needed. */
  const char *path = "nvcuda.dll";
#elif defined(__APPLE__)
  /* Default installation path. */
  const char *path = "/usr/local/cuda/lib/libcuda.dylib";
#else
  const char *path = "libcuda.so";
#endif
  static int initialized = 0;
  static int result = 0;
  int error, driver_version;

  if (initialized) {
    return result;
  }

  initialized = 1;

  error = atexit(cuewExit);
  if (error) {
    return 0;
  }

  /* Load library. */
  lib = dynamic_library_open(path);

  if (lib == NULL) {
    return 0;
  }

  /* Detect driver version. */
  driver_version = 1000;

  DL_LIBRARY_FIND_CHECKED(cuDriverGetVersion);
  if (cuDriverGetVersion) {
    cuDriverGetVersion(&driver_version);
  }

  /* We require version 4.0. */
  if (driver_version < 4000) {
    return 0;
  }
  /* Fetch all function pointers. */
  DL_LIBRARY_FIND(cuInit);
  DL_LIBRARY_FIND(cuDriverGetVersion);
  DL_LIBRARY_FIND(cuDeviceGet);
  DL_LIBRARY_FIND(cuDeviceGetCount);
  DL_LIBRARY_FIND(cuDeviceGetName);
  DL_LIBRARY_FIND(cuDeviceTotalMem_v2);
  DL_LIBRARY_FIND(cuDeviceGetAttribute);
  DL_LIBRARY_FIND(cuDeviceGetProperties);
  DL_LIBRARY_FIND(cuDeviceComputeCapability);
  DL_LIBRARY_FIND(cuCtxCreate_v2);
  DL_LIBRARY_FIND(cuCtxDestroy_v2);
  DL_LIBRARY_FIND(cuCtxPushCurrent_v2);
  DL_LIBRARY_FIND(cuCtxPopCurrent_v2);
  DL_LIBRARY_FIND(cuCtxSetCurrent);
  DL_LIBRARY_FIND(cuCtxGetCurrent);
  DL_LIBRARY_FIND(cuCtxGetDevice);
  DL_LIBRARY_FIND(cuCtxSynchronize);
  DL_LIBRARY_FIND(cuCtxSetLimit);
  DL_LIBRARY_FIND(cuCtxGetLimit);
  DL_LIBRARY_FIND(cuCtxGetCacheConfig);
  DL_LIBRARY_FIND(cuCtxSetCacheConfig);
  DL_LIBRARY_FIND(cuCtxGetSharedMemConfig);
  DL_LIBRARY_FIND(cuCtxSetSharedMemConfig);
  DL_LIBRARY_FIND(cuCtxGetApiVersion);
  DL_LIBRARY_FIND(cuCtxAttach);
  DL_LIBRARY_FIND(cuCtxDetach);
  DL_LIBRARY_FIND(cuModuleLoad);
  DL_LIBRARY_FIND(cuModuleLoadData);
  DL_LIBRARY_FIND(cuModuleLoadDataEx);
  DL_LIBRARY_FIND(cuModuleLoadFatBinary);
  DL_LIBRARY_FIND(cuModuleUnload);
  DL_LIBRARY_FIND(cuModuleGetFunction);
  DL_LIBRARY_FIND(cuModuleGetGlobal_v2);
  DL_LIBRARY_FIND(cuModuleGetTexRef);
  DL_LIBRARY_FIND(cuModuleGetSurfRef);
  DL_LIBRARY_FIND(cuMemGetInfo_v2);
  DL_LIBRARY_FIND(cuMemAlloc_v2);
  DL_LIBRARY_FIND(cuMemAllocPitch_v2);
  DL_LIBRARY_FIND(cuMemFree_v2);
  DL_LIBRARY_FIND(cuMemGetAddressRange_v2);
  DL_LIBRARY_FIND(cuMemAllocHost_v2);
  DL_LIBRARY_FIND(cuMemFreeHost);
  DL_LIBRARY_FIND(cuMemHostAlloc);
  DL_LIBRARY_FIND(cuMemHostGetDevicePointer_v2);
  DL_LIBRARY_FIND(cuMemHostGetFlags);
  DL_LIBRARY_FIND(cuDeviceGetByPCIBusId);
  DL_LIBRARY_FIND(cuDeviceGetPCIBusId);
  DL_LIBRARY_FIND(cuIpcGetEventHandle);
  DL_LIBRARY_FIND(cuIpcOpenEventHandle);
  DL_LIBRARY_FIND(cuIpcGetMemHandle);
  DL_LIBRARY_FIND(cuIpcOpenMemHandle);
  DL_LIBRARY_FIND(cuIpcCloseMemHandle);
  DL_LIBRARY_FIND(cuMemHostRegister);
  DL_LIBRARY_FIND(cuMemHostUnregister);
  DL_LIBRARY_FIND(cuMemcpy);
  DL_LIBRARY_FIND(cuMemcpyPeer);
  DL_LIBRARY_FIND(cuMemcpyHtoD_v2);
  DL_LIBRARY_FIND(cuMemcpyDtoH_v2);
  DL_LIBRARY_FIND(cuMemcpyDtoD_v2);
  DL_LIBRARY_FIND(cuMemcpyDtoA_v2);
  DL_LIBRARY_FIND(cuMemcpyAtoD_v2);
  DL_LIBRARY_FIND(cuMemcpyHtoA_v2);
  DL_LIBRARY_FIND(cuMemcpyAtoH_v2);
  DL_LIBRARY_FIND(cuMemcpyAtoA_v2);
  DL_LIBRARY_FIND(cuMemcpy2D_v2);
  DL_LIBRARY_FIND(cuMemcpy2DUnaligned_v2);
  DL_LIBRARY_FIND(cuMemcpy3D_v2);
  DL_LIBRARY_FIND(cuMemcpy3DPeer);
  DL_LIBRARY_FIND(cuMemcpyAsync);
  DL_LIBRARY_FIND(cuMemcpyPeerAsync);
  DL_LIBRARY_FIND(cuMemcpyHtoDAsync_v2);
  DL_LIBRARY_FIND(cuMemcpyDtoHAsync_v2);
  DL_LIBRARY_FIND(cuMemcpyDtoDAsync_v2);
  DL_LIBRARY_FIND(cuMemcpyHtoAAsync_v2);
  DL_LIBRARY_FIND(cuMemcpyAtoHAsync_v2);
  DL_LIBRARY_FIND(cuMemcpy2DAsync_v2);
  DL_LIBRARY_FIND(cuMemcpy3DAsync_v2);
  DL_LIBRARY_FIND(cuMemcpy3DPeerAsync);
  DL_LIBRARY_FIND(cuMemsetD8_v2);
  DL_LIBRARY_FIND(cuMemsetD16_v2);
  DL_LIBRARY_FIND(cuMemsetD32_v2);
  DL_LIBRARY_FIND(cuMemsetD2D8_v2);
  DL_LIBRARY_FIND(cuMemsetD2D16_v2);
  DL_LIBRARY_FIND(cuMemsetD2D32_v2);
  DL_LIBRARY_FIND(cuMemsetD8Async);
  DL_LIBRARY_FIND(cuMemsetD16Async);
  DL_LIBRARY_FIND(cuMemsetD32Async);
  DL_LIBRARY_FIND(cuMemsetD2D8Async);
  DL_LIBRARY_FIND(cuMemsetD2D16Async);
  DL_LIBRARY_FIND(cuMemsetD2D32Async);
  DL_LIBRARY_FIND(cuArrayCreate_v2);
  DL_LIBRARY_FIND(cuArrayGetDescriptor_v2);
  DL_LIBRARY_FIND(cuArrayDestroy);
  DL_LIBRARY_FIND(cuArray3DCreate_v2);
  DL_LIBRARY_FIND(cuArray3DGetDescriptor_v2);
  DL_LIBRARY_FIND(cuMipmappedArrayCreate);
  DL_LIBRARY_FIND(cuMipmappedArrayGetLevel);
  DL_LIBRARY_FIND(cuMipmappedArrayDestroy);
  DL_LIBRARY_FIND(cuPointerGetAttribute);
  DL_LIBRARY_FIND(cuStreamCreate);
  DL_LIBRARY_FIND(cuStreamWaitEvent);
  DL_LIBRARY_FIND(cuStreamAddCallback);
  DL_LIBRARY_FIND(cuStreamQuery);
  DL_LIBRARY_FIND(cuStreamSynchronize);
  DL_LIBRARY_FIND(cuStreamDestroy_v2);
  DL_LIBRARY_FIND(cuEventCreate);
  DL_LIBRARY_FIND(cuEventRecord);
  DL_LIBRARY_FIND(cuEventQuery);
  DL_LIBRARY_FIND(cuEventSynchronize);
  DL_LIBRARY_FIND(cuEventDestroy_v2);
  DL_LIBRARY_FIND(cuEventElapsedTime);
  DL_LIBRARY_FIND(cuFuncGetAttribute);
  DL_LIBRARY_FIND(cuFuncSetCacheConfig);
  DL_LIBRARY_FIND(cuFuncSetSharedMemConfig);
  DL_LIBRARY_FIND(cuLaunchKernel);
  DL_LIBRARY_FIND(cuFuncSetBlockShape);
  DL_LIBRARY_FIND(cuFuncSetSharedSize);
  DL_LIBRARY_FIND(cuParamSetSize);
  DL_LIBRARY_FIND(cuParamSeti);
  DL_LIBRARY_FIND(cuParamSetf);
  DL_LIBRARY_FIND(cuParamSetv);
  DL_LIBRARY_FIND(cuLaunch);
  DL_LIBRARY_FIND(cuLaunchGrid);
  DL_LIBRARY_FIND(cuLaunchGridAsync);
  DL_LIBRARY_FIND(cuParamSetTexRef);
  DL_LIBRARY_FIND(cuTexRefSetArray);
  DL_LIBRARY_FIND(cuTexRefSetMipmappedArray);
  DL_LIBRARY_FIND(cuTexRefSetAddress_v2);
  DL_LIBRARY_FIND(cuTexRefSetAddress2D_v3);
  DL_LIBRARY_FIND(cuTexRefSetFormat);
  DL_LIBRARY_FIND(cuTexRefSetAddressMode);
  DL_LIBRARY_FIND(cuTexRefSetFilterMode);
  DL_LIBRARY_FIND(cuTexRefSetMipmapFilterMode);
  DL_LIBRARY_FIND(cuTexRefSetMipmapLevelBias);
  DL_LIBRARY_FIND(cuTexRefSetMipmapLevelClamp);
  DL_LIBRARY_FIND(cuTexRefSetMaxAnisotropy);
  DL_LIBRARY_FIND(cuTexRefSetFlags);
  DL_LIBRARY_FIND(cuTexRefGetAddress_v2);
  DL_LIBRARY_FIND(cuTexRefGetArray);
  DL_LIBRARY_FIND(cuTexRefGetMipmappedArray);
  DL_LIBRARY_FIND(cuTexRefGetAddressMode);
  DL_LIBRARY_FIND(cuTexRefGetFilterMode);
  DL_LIBRARY_FIND(cuTexRefGetFormat);
  DL_LIBRARY_FIND(cuTexRefGetMipmapFilterMode);
  DL_LIBRARY_FIND(cuTexRefGetMipmapLevelBias);
  DL_LIBRARY_FIND(cuTexRefGetMipmapLevelClamp);
  DL_LIBRARY_FIND(cuTexRefGetMaxAnisotropy);
  DL_LIBRARY_FIND(cuTexRefGetFlags);
  DL_LIBRARY_FIND(cuTexRefCreate);
  DL_LIBRARY_FIND(cuTexRefDestroy);
  DL_LIBRARY_FIND(cuSurfRefSetArray);
  DL_LIBRARY_FIND(cuSurfRefGetArray);
  DL_LIBRARY_FIND(cuTexObjectCreate);
  DL_LIBRARY_FIND(cuTexObjectDestroy);
  DL_LIBRARY_FIND(cuTexObjectGetResourceDesc);
  DL_LIBRARY_FIND(cuTexObjectGetTextureDesc);
  DL_LIBRARY_FIND(cuTexObjectGetResourceViewDesc);
  DL_LIBRARY_FIND(cuSurfObjectCreate);
  DL_LIBRARY_FIND(cuSurfObjectDestroy);
  DL_LIBRARY_FIND(cuSurfObjectGetResourceDesc);
  DL_LIBRARY_FIND(cuDeviceCanAccessPeer);
  DL_LIBRARY_FIND(cuCtxEnablePeerAccess);
  DL_LIBRARY_FIND(cuCtxDisablePeerAccess);
  DL_LIBRARY_FIND(cuGraphicsUnregisterResource);
  DL_LIBRARY_FIND(cuGraphicsSubResourceGetMappedArray);
  DL_LIBRARY_FIND(cuGraphicsResourceGetMappedMipmappedArray);
  DL_LIBRARY_FIND(cuGraphicsResourceGetMappedPointer_v2);
  DL_LIBRARY_FIND(cuGraphicsResourceSetMapFlags);
  DL_LIBRARY_FIND(cuGraphicsMapResources);
  DL_LIBRARY_FIND(cuGraphicsUnmapResources);
  DL_LIBRARY_FIND(cuGetExportTable);

  DL_LIBRARY_FIND(cuGraphicsGLRegisterBuffer);
  DL_LIBRARY_FIND(cuGraphicsGLRegisterImage);
  DL_LIBRARY_FIND(cuGLGetDevices);
  DL_LIBRARY_FIND(cuGLCtxCreate_v2);
  DL_LIBRARY_FIND(cuGLInit);
  DL_LIBRARY_FIND(cuGLRegisterBufferObject);
  DL_LIBRARY_FIND(cuGLMapBufferObject_v2);
  DL_LIBRARY_FIND(cuGLUnmapBufferObject);
  DL_LIBRARY_FIND(cuGLUnregisterBufferObject);
  DL_LIBRARY_FIND(cuGLSetBufferObjectMapFlags);
  DL_LIBRARY_FIND(cuGLMapBufferObjectAsync_v2);
  DL_LIBRARY_FIND(cuGLUnmapBufferObjectAsync);


  result = 1;
  return result;
}

const char *cuewErrorString(CUresult result) {
  switch(result) {
    case CUDA_SUCCESS: return "No errors";
    case CUDA_ERROR_INVALID_VALUE: return "Invalid value";
    case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory";
    case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized";
    case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized";
    case CUDA_ERROR_PROFILER_DISABLED: return "PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "PROFILER_ALREADY_STOPPED";
    case CUDA_ERROR_NO_DEVICE: return "No CUDA-capable device available";
    case CUDA_ERROR_INVALID_DEVICE: return "Invalid device";
    case CUDA_ERROR_INVALID_IMAGE: return "Invalid kernel image";
    case CUDA_ERROR_INVALID_CONTEXT: return "Invalid context";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current";
    case CUDA_ERROR_MAP_FAILED: return "Map failed";
    case CUDA_ERROR_UNMAP_FAILED: return "Unmap failed";
    case CUDA_ERROR_ARRAY_IS_MAPPED: return "Array is mapped";
    case CUDA_ERROR_ALREADY_MAPPED: return "Already mapped";
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No binary for GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED: return "Already acquired";
    case CUDA_ERROR_NOT_MAPPED: return "Not mapped";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "Mapped resource not available for access as an array";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "Mapped resource not available for access as a pointer";
    case CUDA_ERROR_ECC_UNCORRECTABLE: return "Uncorrectable ECC error detected";
    case CUDA_ERROR_UNSUPPORTED_LIMIT: return "CUlimit not supported by device";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "CONTEXT_ALREADY_IN_USE";
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: return "PEER_ACCESS_UNSUPPORTED";
    case CUDA_ERROR_INVALID_SOURCE: return "Invalid source";
    case CUDA_ERROR_FILE_NOT_FOUND: return "File not found";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Link to a shared object failed to resolve";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "Shared object initialization failed";
    case CUDA_ERROR_OPERATING_SYSTEM: return "OPERATING_SYSTEM";
    case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle";
    case CUDA_ERROR_NOT_FOUND: return "Not found";
    case CUDA_ERROR_NOT_READY: return "CUDA not ready";
    case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources";
    case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "PEER_ACCESS_ALREADY_ENABLED";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "PEER_ACCESS_NOT_ENABLED";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "PRIMARY_CONTEXT_ACTIVE";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "CONTEXT_IS_DESTROYED";
    case CUDA_ERROR_ASSERT: return "ASSERT";
    case CUDA_ERROR_TOO_MANY_PEERS: return "TOO_MANY_PEERS";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return "HOST_MEMORY_ALREADY_REGISTERED";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: return "HOST_MEMORY_NOT_REGISTERED";
    case CUDA_ERROR_NOT_PERMITTED: return "NOT_PERMITTED";
    case CUDA_ERROR_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case CUDA_ERROR_UNKNOWN: return "Unknown error";
    default: return "Unknown CUDA error value";
  }
}
