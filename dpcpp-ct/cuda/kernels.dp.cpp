/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cassert>
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"

/* DPCT_ORIG __device__ inline uint64_t llitoulli(int64_t l)*/
inline uint64_t llitoulli(int64_t l)
{
	uint64_t u;
        /*
        DPCT1053:173: Migration of device assembly code is not supported.
        */
        asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
        return u;
}

/* DPCT_ORIG __device__ inline int64_t ullitolli(uint64_t u)*/
inline int64_t ullitolli(uint64_t u)
{
	int64_t l;
        /*
        DPCT1053:174: Migration of device assembly code is not supported.
        */
        asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
        return l;
}

/*
DPCT1023:33: The SYCL sub-group does not support mask options for
dpct::select_from_sub_group.
*/
#define WARPMINIMUMEXCHANGE(tgx, v0, k0, mask)                                 \
 {                                                                             \
  float v1 = v0;                                                               \
  int k1 = k0;                                                                 \
  int otgx = tgx ^ mask;                                                       \
  /* DPCT_ORIG 		float v2    = __shfl_sync(0xffffffff, v0, otgx); \*/          \
  float v2 = dpct::select_from_sub_group(item_ct1.get_sub_group(), v0, otgx);  \
  /* DPCT_ORIG 		int k2      = __shfl_sync(0xffffffff, k0, otgx); \*/          \
  int k2 = dpct::select_from_sub_group(item_ct1.get_sub_group(), k0, otgx);    \
  int flag = ((v1 < v2) ^ (tgx > otgx)) && (v1 != v2);                         \
  k0 = flag ? k1 : k2;                                                         \
  v0 = flag ? v1 : v2;                                                         \
 }

#define WARPMINIMUM2(tgx, v0, k0) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 1) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 2) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 4) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 8) \
	WARPMINIMUMEXCHANGE(tgx, v0, k0, 16)

/*
DPCT1023:17: The SYCL sub-group does not support mask options for
dpct::select_from_sub_group.
*/
/*
DPCT1078:15: Consider replacing memory_order::acq_rel with memory_order::seq_cst
for correctness if strong memory order restrictions are needed.
*/
/*
DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
#define REDUCEINTEGERSUM(value, pAccumulator)                                  \
 /* DPCT_ORIG 	if (threadIdx.x == 0) \*/                                       \
 if (item_ct1.get_local_id(2) == 0)                                            \
 {                                                                             \
  *pAccumulator = 0;                                                           \
 }                                                                             \
 /* DPCT_ORIG 	__threadfence(); \*/                                            \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);  \
 /* DPCT_ORIG 	__syncthreads(); \*/                                            \
 item_ct1.barrier();                                                           \
 /* DPCT_ORIG 	if (__any_sync(0xffffffff, value != 0)) \*/                     \
 if (sycl::any_of_group(                                                       \
         item_ct1.get_sub_group(),                                             \
         (0xffffffff &                                                         \
          (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&          \
             value != 0))                                                      \
 {                                                                             \
  /* DPCT_ORIG 		uint32_t tgx            = threadIdx.x & cData.warpmask;       \
   * \*/                                                                       \
  uint32_t tgx = item_ct1.get_local_id(2) & cData.warpmask;                    \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 1); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 1);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 2); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 2);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 4); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 4);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 8); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 8);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 16); \*/                                                     \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 16);  \
  if (tgx == 0)                                                                \
  {                                                                            \
   /* DPCT_ORIG 			atomicAdd(pAccumulator, value); \*/                         \
   dpct::atomic_fetch_add(pAccumulator, value);                                \
  }                                                                            \
 }                                                                             \
 /* DPCT_ORIG 	__threadfence(); \*/                                            \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);  \
 /* DPCT_ORIG 	__syncthreads(); \*/                                            \
 item_ct1.barrier();                                                           \
 value = *pAccumulator;                                                        \
 /* DPCT_ORIG 	__syncthreads();*/                                              \
 item_ct1.barrier();

#define ATOMICADDI32(pAccumulator, value)                                      \
 atomicAdd(pAccumulator, (int)((value)))
#define ATOMICSUBI32(pAccumulator, value) atomicAdd(pAccumulator, -(value))
/*
DPCT1058:207: "atomicAdd" is not migrated because it is not called in the code.
*/
#define ATOMICADDF32(pAccumulator, value) atomicAdd(pAccumulator, (value))
/*
DPCT1058:206: "atomicAdd" is not migrated because it is not called in the code.
*/
#define ATOMICSUBF32(pAccumulator, value) atomicAdd(pAccumulator, -(value))

/*
DPCT1023:4: The SYCL sub-group does not support mask options for
dpct::select_from_sub_group.
*/
/*
DPCT1078:1: Consider replacing memory_order::acq_rel with memory_order::seq_cst
for correctness if strong memory order restrictions are needed.
*/
/*
DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
performance if there is no access to global memory.
*/
#define REDUCEFLOATSUM(value, pAccumulator)                                    \
 /* DPCT_ORIG 	if (threadIdx.x == 0) \*/                                       \
 if (item_ct1.get_local_id(2) == 0)                                            \
 {                                                                             \
  *pAccumulator = 0;                                                           \
 }                                                                             \
 /* DPCT_ORIG 	__threadfence(); \*/                                            \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);  \
 /* DPCT_ORIG 	__syncthreads(); \*/                                            \
 item_ct1.barrier();                                                           \
 /* DPCT_ORIG 	if (__any_sync(0xffffffff, value != 0.0f)) \*/                  \
 if (sycl::any_of_group(                                                       \
         item_ct1.get_sub_group(),                                             \
         (0xffffffff &                                                         \
          (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&          \
             value != 0.0f))                                                   \
 {                                                                             \
  /* DPCT_ORIG 		uint32_t tgx            = threadIdx.x & cData.warpmask;       \
   * \*/                                                                       \
  uint32_t tgx = item_ct1.get_local_id(2) & cData.warpmask;                    \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 1); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 1);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 2); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 2);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 4); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 4);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 8); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 8);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 16); \*/                                                     \
  value +=                                                                     \
      dpct::select_from_sub_group(item_ct1.get_sub_group(), value, tgx ^ 16);  \
  if (tgx == 0)                                                                \
  {                                                                            \
   /* DPCT_ORIG 			atomicAdd(pAccumulator, value); \*/                         \
   dpct::atomic_fetch_add(pAccumulator, value);                                \
  }                                                                            \
 }                                                                             \
 /* DPCT_ORIG 	__threadfence(); \*/                                            \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);  \
 /* DPCT_ORIG 	__syncthreads(); \*/                                            \
 item_ct1.barrier();                                                           \
 value = (float)(*pAccumulator);                                               \
 /* DPCT_ORIG 	__syncthreads();*/                                              \
 item_ct1.barrier();

/* DPCT_ORIG static __constant__ GpuData cData;*/
static dpct::constant_memory<GpuData, 0> cData;
static GpuData cpuData;

void SetKernelsGpuData(GpuData *pData) try {
/* DPCT_ORIG 	cudaError_t status;*/
        int status;
/* DPCT_ORIG 	status = cudaMemcpyToSymbol(cData, pData, sizeof(GpuData));*/
        /*
        DPCT1003:176: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(cData.get_ptr(), pData, sizeof(GpuData))
                      .wait(),
                  0);
        /*
        DPCT1001:175: The statement could not be removed.
        */
        RTERROR(status, "SetKernelsGpuData copy to cData failed");
        memcpy(&cpuData, pData, sizeof(GpuData));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void GetKernelsGpuData(GpuData *pData) try {
/* DPCT_ORIG 	cudaError_t status;*/
        int status;
/* DPCT_ORIG 	status = cudaMemcpyFromSymbol(pData, cData, sizeof(GpuData));*/
        /*
        DPCT1003:178: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(pData, cData.get_ptr(), sizeof(GpuData))
                      .wait(),
                  0);
        /*
        DPCT1001:177: The statement could not be removed.
        */
        RTERROR(status, "GetKernelsGpuData copy From cData failed");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Kernel files
/* DPCT_ORIG #include "calcenergy.cu"*/
#include "calcenergy.dp.cpp"
/* DPCT_ORIG #include "calcMergeEneGra.cu"*/
#include "calcMergeEneGra.dp.cpp"
/* DPCT_ORIG #include "auxiliary_genetic.cu"*/
#include "auxiliary_genetic.dp.cpp"
/* DPCT_ORIG #include "kernel1.cu"*/
#include "kernel1.dp.cpp"
/* DPCT_ORIG #include "kernel2.cu"*/
#include "kernel2.dp.cpp"
/* DPCT_ORIG #include "kernel3.cu"*/
#include "kernel3.dp.cpp"
/* DPCT_ORIG #include "kernel4.cu"*/
#include "kernel4.dp.cpp"
/* DPCT_ORIG #include "kernel_ad.cu"*/
#include "kernel_ad.dp.cpp"
/* DPCT_ORIG #include "kernel_adam.cu"*/
#include "kernel_adam.dp.cpp"
