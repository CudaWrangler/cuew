/*
 * Copyright 2011-2013 Blender Foundation
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

#include "cuew.h"
#include <assert.h>

#ifndef _WIN32
#  include <dlfcn.h>
#endif

/* function defininitions */

tcuInit *cuInit;
tcuDriverGetVersion *cuDriverGetVersion;
tcuDeviceGet *cuDeviceGet;
tcuDeviceGetCount *cuDeviceGetCount;
tcuDeviceGetName *cuDeviceGetName;
tcuDeviceComputeCapability *cuDeviceComputeCapability;
tcuDeviceTotalMem *cuDeviceTotalMem;
tcuDeviceGetProperties *cuDeviceGetProperties;
tcuDeviceGetAttribute *cuDeviceGetAttribute;
tcuCtxCreate *cuCtxCreate;
tcuCtxDestroy *cuCtxDestroy;
tcuCtxAttach *cuCtxAttach;
tcuCtxDetach *cuCtxDetach;
tcuCtxPushCurrent *cuCtxPushCurrent;
tcuCtxPopCurrent *cuCtxPopCurrent;
tcuCtxGetDevice *cuCtxGetDevice;
tcuCtxSynchronize *cuCtxSynchronize;
tcuModuleLoad *cuModuleLoad;
tcuModuleLoadData *cuModuleLoadData;
tcuModuleLoadDataEx *cuModuleLoadDataEx;
tcuModuleLoadFatBinary *cuModuleLoadFatBinary;
tcuModuleUnload *cuModuleUnload;
tcuModuleGetFunction *cuModuleGetFunction;
tcuModuleGetGlobal *cuModuleGetGlobal;
tcuModuleGetTexRef *cuModuleGetTexRef;
tcuModuleGetSurfRef *cuModuleGetSurfRef;
tcuMemGetInfo *cuMemGetInfo;
tcuMemAlloc *cuMemAlloc;
tcuMemAllocPitch *cuMemAllocPitch;
tcuMemFree *cuMemFree;
tcuMemGetAddressRange *cuMemGetAddressRange;
tcuMemAllocHost *cuMemAllocHost;
tcuMemFreeHost *cuMemFreeHost;
tcuMemHostAlloc *cuMemHostAlloc;
tcuMemHostGetDevicePointer *cuMemHostGetDevicePointer;
tcuMemHostGetFlags *cuMemHostGetFlags;
tcuMemcpyHtoD *cuMemcpyHtoD;
tcuMemcpyDtoH *cuMemcpyDtoH;
tcuMemcpyDtoD *cuMemcpyDtoD;
tcuMemcpyDtoA *cuMemcpyDtoA;
tcuMemcpyAtoD *cuMemcpyAtoD;
tcuMemcpyHtoA *cuMemcpyHtoA;
tcuMemcpyAtoH *cuMemcpyAtoH;
tcuMemcpyAtoA *cuMemcpyAtoA;
tcuMemcpy2D *cuMemcpy2D;
tcuMemcpy2DUnaligned *cuMemcpy2DUnaligned;
tcuMemcpy3D *cuMemcpy3D;
tcuMemcpyHtoDAsync *cuMemcpyHtoDAsync;
tcuMemcpyDtoHAsync *cuMemcpyDtoHAsync;
tcuMemcpyDtoDAsync *cuMemcpyDtoDAsync;
tcuMemcpyHtoAAsync *cuMemcpyHtoAAsync;
tcuMemcpyAtoHAsync *cuMemcpyAtoHAsync;
tcuMemcpy2DAsync *cuMemcpy2DAsync;
tcuMemcpy3DAsync *cuMemcpy3DAsync;
tcuMemsetD8 *cuMemsetD8;
tcuMemsetD16 *cuMemsetD16;
tcuMemsetD32 *cuMemsetD32;
tcuMemsetD2D8 *cuMemsetD2D8;
tcuMemsetD2D16 *cuMemsetD2D16;
tcuMemsetD2D32 *cuMemsetD2D32;
tcuFuncSetBlockShape *cuFuncSetBlockShape;
tcuFuncSetSharedSize *cuFuncSetSharedSize;
tcuFuncGetAttribute *cuFuncGetAttribute;
tcuFuncSetCacheConfig *cuFuncSetCacheConfig;
tcuArrayCreate *cuArrayCreate;
tcuArrayGetDescriptor *cuArrayGetDescriptor;
tcuArrayDestroy *cuArrayDestroy;
tcuArray3DCreate *cuArray3DCreate;
tcuArray3DGetDescriptor *cuArray3DGetDescriptor;
tcuTexRefCreate *cuTexRefCreate;
tcuTexRefDestroy *cuTexRefDestroy;
tcuTexRefSetArray *cuTexRefSetArray;
tcuTexRefSetAddress *cuTexRefSetAddress;
tcuTexRefSetAddress2D *cuTexRefSetAddress2D;
tcuTexRefSetFormat *cuTexRefSetFormat;
tcuTexRefSetAddressMode *cuTexRefSetAddressMode;
tcuTexRefSetFilterMode *cuTexRefSetFilterMode;
tcuTexRefSetFlags *cuTexRefSetFlags;
tcuTexRefGetAddress *cuTexRefGetAddress;
tcuTexRefGetArray *cuTexRefGetArray;
tcuTexRefGetAddressMode *cuTexRefGetAddressMode;
tcuTexRefGetFilterMode *cuTexRefGetFilterMode;
tcuTexRefGetFormat *cuTexRefGetFormat;
tcuTexRefGetFlags *cuTexRefGetFlags;
tcuSurfRefSetArray *cuSurfRefSetArray;
tcuSurfRefGetArray *cuSurfRefGetArray;
tcuParamSetSize *cuParamSetSize;
tcuParamSeti *cuParamSeti;
tcuParamSetf *cuParamSetf;
tcuParamSetv *cuParamSetv;
tcuParamSetTexRef *cuParamSetTexRef;
tcuLaunch *cuLaunch;
tcuLaunchGrid *cuLaunchGrid;
tcuLaunchGridAsync *cuLaunchGridAsync;
tcuEventCreate *cuEventCreate;
tcuEventRecord *cuEventRecord;
tcuEventQuery *cuEventQuery;
tcuEventSynchronize *cuEventSynchronize;
tcuEventDestroy *cuEventDestroy;
tcuEventElapsedTime *cuEventElapsedTime;
tcuStreamCreate *cuStreamCreate;
tcuStreamQuery *cuStreamQuery;
tcuStreamSynchronize *cuStreamSynchronize;
tcuStreamDestroy *cuStreamDestroy;
tcuGraphicsUnregisterResource *cuGraphicsUnregisterResource;
tcuGraphicsSubResourceGetMappedArray *cuGraphicsSubResourceGetMappedArray;
tcuGraphicsResourceGetMappedPointer *cuGraphicsResourceGetMappedPointer;
tcuGraphicsResourceSetMapFlags *cuGraphicsResourceSetMapFlags;
tcuGraphicsMapResources *cuGraphicsMapResources;
tcuGraphicsUnmapResources *cuGraphicsUnmapResources;
tcuGetExportTable *cuGetExportTable;
tcuCtxSetLimit *cuCtxSetLimit;
tcuCtxGetLimit *cuCtxGetLimit;
tcuGLCtxCreate *cuGLCtxCreate;
tcuGraphicsGLRegisterBuffer *cuGraphicsGLRegisterBuffer;
tcuGraphicsGLRegisterImage *cuGraphicsGLRegisterImage;
tcuCtxSetCurrent *cuCtxSetCurrent;
tcuLaunchKernel *cuLaunchKernel;

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRALEAN
#  include <windows.h>

/* utility macros */

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

#define CUDA_LIBRARY_FIND_CHECKED(name) \
	name = (t##name*)dynamic_library_find(lib, #name);

#define CUDA_LIBRARY_FIND(name) \
	name = (t##name*)dynamic_library_find(lib, #name); \
	assert(name);

#define CUDA_LIBRARY_FIND_V2(name) \
	name = (t##name*)dynamic_library_find(lib, #name "_v2"); \
	assert(name);

static DynamicLibrary lib;

static void cuewExit(void)
{
	if(lib != NULL) {
		//  Ignore errors
		dynamic_library_close(lib);
		lib = NULL;
	}
}

/* initialization function */

int cuewInit(void)
{
	/* library paths */
#ifdef _WIN32
	/* expected in c:/windows/system or similar, no path needed */
	const char *path = "nvcuda.dll";
#elif defined(__APPLE__)
	/* default installation path */
	const char *path = "/usr/local/cuda/lib/libcuda.dylib";
#else
	const char *path = "libcuda.so";
#endif

	static int initialized = 0;
	static int result = 0;
	int error, driver_version;

	if(initialized) {
		return result;
	}

	initialized = 1;

	error = atexit(cuewExit);
	if (error) {
		return 0;
	}

	/* load library */
	lib = dynamic_library_open(path);

	if(lib == NULL)
		return 0;

	/* detect driver version */
	driver_version = 1000;

	CUDA_LIBRARY_FIND_CHECKED(cuDriverGetVersion);
	if(cuDriverGetVersion)
		cuDriverGetVersion(&driver_version);

	/* we require version 4.0 */
	if(driver_version < 4000)
		return 0;

	/* fetch all function pointers */
	CUDA_LIBRARY_FIND(cuInit);
	CUDA_LIBRARY_FIND(cuDeviceGet);
	CUDA_LIBRARY_FIND(cuDeviceGetCount);
	CUDA_LIBRARY_FIND(cuDeviceGetName);
	CUDA_LIBRARY_FIND(cuDeviceComputeCapability);
	CUDA_LIBRARY_FIND(cuDeviceTotalMem);
	CUDA_LIBRARY_FIND(cuDeviceGetProperties);
	CUDA_LIBRARY_FIND(cuDeviceGetAttribute);
	CUDA_LIBRARY_FIND(cuCtxCreate);
	CUDA_LIBRARY_FIND(cuCtxDestroy);
	CUDA_LIBRARY_FIND(cuCtxAttach);
	CUDA_LIBRARY_FIND(cuCtxDetach);
	CUDA_LIBRARY_FIND(cuCtxPushCurrent);
	CUDA_LIBRARY_FIND(cuCtxPopCurrent);
	CUDA_LIBRARY_FIND(cuCtxGetDevice);
	CUDA_LIBRARY_FIND(cuCtxSynchronize);
	CUDA_LIBRARY_FIND(cuModuleLoad);
	CUDA_LIBRARY_FIND(cuModuleLoadData);
	CUDA_LIBRARY_FIND(cuModuleUnload);
	CUDA_LIBRARY_FIND(cuModuleGetFunction);
	CUDA_LIBRARY_FIND(cuModuleGetGlobal);
	CUDA_LIBRARY_FIND(cuModuleGetTexRef);
	CUDA_LIBRARY_FIND(cuMemGetInfo);
	CUDA_LIBRARY_FIND(cuMemAlloc);
	CUDA_LIBRARY_FIND(cuMemAllocPitch);
	CUDA_LIBRARY_FIND(cuMemFree);
	CUDA_LIBRARY_FIND(cuMemGetAddressRange);
	CUDA_LIBRARY_FIND(cuMemAllocHost);
	CUDA_LIBRARY_FIND(cuMemFreeHost);
	CUDA_LIBRARY_FIND(cuMemHostAlloc);
	CUDA_LIBRARY_FIND(cuMemHostGetDevicePointer);
	CUDA_LIBRARY_FIND(cuMemcpyHtoD);
	CUDA_LIBRARY_FIND(cuMemcpyDtoH);
	CUDA_LIBRARY_FIND(cuMemcpyDtoD);
	CUDA_LIBRARY_FIND(cuMemcpyDtoA);
	CUDA_LIBRARY_FIND(cuMemcpyAtoD);
	CUDA_LIBRARY_FIND(cuMemcpyHtoA);
	CUDA_LIBRARY_FIND(cuMemcpyAtoH);
	CUDA_LIBRARY_FIND(cuMemcpyAtoA);
	CUDA_LIBRARY_FIND(cuMemcpy2D);
	CUDA_LIBRARY_FIND(cuMemcpy2DUnaligned);
	CUDA_LIBRARY_FIND(cuMemcpy3D);
	CUDA_LIBRARY_FIND(cuMemcpyHtoDAsync);
	CUDA_LIBRARY_FIND(cuMemcpyDtoHAsync);
	CUDA_LIBRARY_FIND(cuMemcpyHtoAAsync);
	CUDA_LIBRARY_FIND(cuMemcpyAtoHAsync);
	CUDA_LIBRARY_FIND(cuMemcpy2DAsync);
	CUDA_LIBRARY_FIND(cuMemcpy3DAsync);
	CUDA_LIBRARY_FIND(cuMemsetD8);
	CUDA_LIBRARY_FIND(cuMemsetD16);
	CUDA_LIBRARY_FIND(cuMemsetD32);
	CUDA_LIBRARY_FIND(cuMemsetD2D8);
	CUDA_LIBRARY_FIND(cuMemsetD2D16);
	CUDA_LIBRARY_FIND(cuMemsetD2D32);
	CUDA_LIBRARY_FIND(cuFuncSetBlockShape);
	CUDA_LIBRARY_FIND(cuFuncSetSharedSize);
	CUDA_LIBRARY_FIND(cuFuncGetAttribute);
	CUDA_LIBRARY_FIND(cuArrayCreate);
	CUDA_LIBRARY_FIND(cuArrayGetDescriptor);
	CUDA_LIBRARY_FIND(cuArrayDestroy);
	CUDA_LIBRARY_FIND(cuArray3DCreate);
	CUDA_LIBRARY_FIND(cuArray3DGetDescriptor);
	CUDA_LIBRARY_FIND(cuTexRefCreate);
	CUDA_LIBRARY_FIND(cuTexRefDestroy);
	CUDA_LIBRARY_FIND(cuTexRefSetArray);
	CUDA_LIBRARY_FIND(cuTexRefSetAddress);
	CUDA_LIBRARY_FIND(cuTexRefSetAddress2D);
	CUDA_LIBRARY_FIND(cuTexRefSetFormat);
	CUDA_LIBRARY_FIND(cuTexRefSetAddressMode);
	CUDA_LIBRARY_FIND(cuTexRefSetFilterMode);
	CUDA_LIBRARY_FIND(cuTexRefSetFlags);
	CUDA_LIBRARY_FIND(cuTexRefGetAddress);
	CUDA_LIBRARY_FIND(cuTexRefGetArray);
	CUDA_LIBRARY_FIND(cuTexRefGetAddressMode);
	CUDA_LIBRARY_FIND(cuTexRefGetFilterMode);
	CUDA_LIBRARY_FIND(cuTexRefGetFormat);
	CUDA_LIBRARY_FIND(cuTexRefGetFlags);
	CUDA_LIBRARY_FIND(cuParamSetSize);
	CUDA_LIBRARY_FIND(cuParamSeti);
	CUDA_LIBRARY_FIND(cuParamSetf);
	CUDA_LIBRARY_FIND(cuParamSetv);
	CUDA_LIBRARY_FIND(cuParamSetTexRef);
	CUDA_LIBRARY_FIND(cuLaunch);
	CUDA_LIBRARY_FIND(cuLaunchGrid);
	CUDA_LIBRARY_FIND(cuLaunchGridAsync);
	CUDA_LIBRARY_FIND(cuEventCreate);
	CUDA_LIBRARY_FIND(cuEventRecord);
	CUDA_LIBRARY_FIND(cuEventQuery);
	CUDA_LIBRARY_FIND(cuEventSynchronize);
	CUDA_LIBRARY_FIND(cuEventDestroy);
	CUDA_LIBRARY_FIND(cuEventElapsedTime);
	CUDA_LIBRARY_FIND(cuStreamCreate);
	CUDA_LIBRARY_FIND(cuStreamQuery);
	CUDA_LIBRARY_FIND(cuStreamSynchronize);
	CUDA_LIBRARY_FIND(cuStreamDestroy);

	/* cuda 2.1 */
	CUDA_LIBRARY_FIND(cuModuleLoadDataEx);
	CUDA_LIBRARY_FIND(cuModuleLoadFatBinary);
	CUDA_LIBRARY_FIND(cuGLCtxCreate);
	CUDA_LIBRARY_FIND(cuGraphicsGLRegisterBuffer);
	CUDA_LIBRARY_FIND(cuGraphicsGLRegisterImage);

	/* cuda 2.3 */
	CUDA_LIBRARY_FIND(cuMemHostGetFlags);
	CUDA_LIBRARY_FIND(cuGraphicsGLRegisterBuffer);
	CUDA_LIBRARY_FIND(cuGraphicsGLRegisterImage);

	/* cuda 3.0 */
	CUDA_LIBRARY_FIND(cuMemcpyDtoDAsync);
	CUDA_LIBRARY_FIND(cuFuncSetCacheConfig);
	CUDA_LIBRARY_FIND(cuGraphicsUnregisterResource);
	CUDA_LIBRARY_FIND(cuGraphicsSubResourceGetMappedArray);
	CUDA_LIBRARY_FIND(cuGraphicsResourceGetMappedPointer);
	CUDA_LIBRARY_FIND(cuGraphicsResourceSetMapFlags);
	CUDA_LIBRARY_FIND(cuGraphicsMapResources);
	CUDA_LIBRARY_FIND(cuGraphicsUnmapResources);
	CUDA_LIBRARY_FIND(cuGetExportTable);

	/* cuda 3.1 */
	CUDA_LIBRARY_FIND(cuModuleGetSurfRef);
	CUDA_LIBRARY_FIND(cuSurfRefSetArray);
	CUDA_LIBRARY_FIND(cuSurfRefGetArray);
	CUDA_LIBRARY_FIND(cuCtxSetLimit);
	CUDA_LIBRARY_FIND(cuCtxGetLimit);

	/* functions which changed 3.1 -> 3.2 for 64 bit stuff, the cuda library
	 * has both the old ones for compatibility and new ones with _v2 postfix,
	 * we load the _v2 ones here. */
	CUDA_LIBRARY_FIND_V2(cuDeviceTotalMem);
	CUDA_LIBRARY_FIND_V2(cuCtxCreate);
	CUDA_LIBRARY_FIND_V2(cuModuleGetGlobal);
	CUDA_LIBRARY_FIND_V2(cuMemGetInfo);
	CUDA_LIBRARY_FIND_V2(cuMemAlloc);
	CUDA_LIBRARY_FIND_V2(cuMemAllocPitch);
	CUDA_LIBRARY_FIND_V2(cuMemFree);
	CUDA_LIBRARY_FIND_V2(cuMemGetAddressRange);
	CUDA_LIBRARY_FIND_V2(cuMemAllocHost);
	CUDA_LIBRARY_FIND_V2(cuMemHostGetDevicePointer);
	CUDA_LIBRARY_FIND_V2(cuMemcpyHtoD);
	CUDA_LIBRARY_FIND_V2(cuMemcpyDtoH);
	CUDA_LIBRARY_FIND_V2(cuMemcpyDtoD);
	CUDA_LIBRARY_FIND_V2(cuMemcpyDtoA);
	CUDA_LIBRARY_FIND_V2(cuMemcpyAtoD);
	CUDA_LIBRARY_FIND_V2(cuMemcpyHtoA);
	CUDA_LIBRARY_FIND_V2(cuMemcpyAtoH);
	CUDA_LIBRARY_FIND_V2(cuMemcpyAtoA);
	CUDA_LIBRARY_FIND_V2(cuMemcpyHtoAAsync);
	CUDA_LIBRARY_FIND_V2(cuMemcpyAtoHAsync);
	CUDA_LIBRARY_FIND_V2(cuMemcpy2D);
	CUDA_LIBRARY_FIND_V2(cuMemcpy2DUnaligned);
	CUDA_LIBRARY_FIND_V2(cuMemcpy3D);
	CUDA_LIBRARY_FIND_V2(cuMemcpyHtoDAsync);
	CUDA_LIBRARY_FIND_V2(cuMemcpyDtoHAsync);
	CUDA_LIBRARY_FIND_V2(cuMemcpyDtoDAsync);
	CUDA_LIBRARY_FIND_V2(cuMemcpy2DAsync);
	CUDA_LIBRARY_FIND_V2(cuMemcpy3DAsync);
	CUDA_LIBRARY_FIND_V2(cuMemsetD8);
	CUDA_LIBRARY_FIND_V2(cuMemsetD16);
	CUDA_LIBRARY_FIND_V2(cuMemsetD32);
	CUDA_LIBRARY_FIND_V2(cuMemsetD2D8);
	CUDA_LIBRARY_FIND_V2(cuMemsetD2D16);
	CUDA_LIBRARY_FIND_V2(cuMemsetD2D32);
	CUDA_LIBRARY_FIND_V2(cuArrayCreate);
	CUDA_LIBRARY_FIND_V2(cuArrayGetDescriptor);
	CUDA_LIBRARY_FIND_V2(cuArray3DCreate);
	CUDA_LIBRARY_FIND_V2(cuArray3DGetDescriptor);
	CUDA_LIBRARY_FIND_V2(cuTexRefSetAddress);
	CUDA_LIBRARY_FIND_V2(cuTexRefSetAddress2D);
	CUDA_LIBRARY_FIND_V2(cuTexRefGetAddress);
	CUDA_LIBRARY_FIND_V2(cuGraphicsResourceGetMappedPointer);
	CUDA_LIBRARY_FIND_V2(cuGLCtxCreate);

	/* cuda 4.0 */
	CUDA_LIBRARY_FIND(cuCtxSetCurrent);
	CUDA_LIBRARY_FIND(cuLaunchKernel);

	result = 1;

	return result;
}

const char *cuewErrorString(CUresult result)
{
	switch(result) {
	case CUDA_SUCCESS: return "No errors";
	case CUDA_ERROR_INVALID_VALUE: return "Invalid value";
	case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory";
	case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized";
	case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized";

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

	case CUDA_ERROR_INVALID_SOURCE: return "Invalid source";
	case CUDA_ERROR_FILE_NOT_FOUND: return "File not found";
	case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Link to a shared object failed to resolve";
	case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "Shared object initialization failed";

	case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle";

	case CUDA_ERROR_NOT_FOUND: return "Not found";

	case CUDA_ERROR_NOT_READY: return "CUDA not ready";

	case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed";
	case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources";
	case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout";
	case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing";

	case CUDA_ERROR_UNKNOWN: return "Unknown error";

	default: return "Unknown CUDA error value";
	}
}
