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

#include <cstdint>
#include <cassert>
#include "defines.h"
#include "calcenergy.h"
#include "GpuData.h"

#define WARPMINIMUMEXCHANGE(tgx, v0, k0, mask)                          \
{                                                                       \
  float v1 = v0;                                                        \
  int k1 = k0;                                                          \
  int otgx = tgx ^ mask;                                                \
  float v2 = sycl::ext::oneapi::experimental::this_sub_group().shuffle(v0, otgx); \
  int k2 = sycl::ext::oneapi::experimental::this_sub_group().shuffle(k0, otgx); \
  int flag = ((v1 < v2) ^ (tgx > otgx)) && (v1 != v2);                  \
  k0 = flag ? k1 : k2;                                                  \
  v0 = flag ? v1 : v2;                                                  \
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
 if (item.get_local_id(2) == 0)                                            \
 {                                                                             \
  *pAccumulator = 0;                                                           \
 }                                                                             \
 /* DPCT_ORIG 	__threadfence(); \*/                                            \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);  \
 /* DPCT_ORIG 	__syncthreads(); \*/                                            \
 item.barrier();                                                           \
 /* DPCT_ORIG 	if (__any_sync(0xffffffff, value != 0)) \*/                     \
 if (sycl::any_of_group(                                                       \
         item.get_sub_group(),                                             \
         (0xffffffff &                                                         \
          (0x1 << item.get_sub_group().get_local_linear_id())) &&          \
             value != 0))                                                      \
 {                                                                             \
  /* DPCT_ORIG 		uint32_t tgx            = threadIdx.x & cData.warpmask;       \
   * \*/                                                                       \
  uint32_t tgx = item.get_local_id(2) & cData.warpmask;                    \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 1); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item.get_sub_group(), value, tgx ^ 1);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 2); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item.get_sub_group(), value, tgx ^ 2);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 4); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item.get_sub_group(), value, tgx ^ 4);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 8); \*/                                                      \
  value +=                                                                     \
      dpct::select_from_sub_group(item.get_sub_group(), value, tgx ^ 8);   \
  /* DPCT_ORIG 		value                  += __shfl_sync(0xffffffff,             \
   * value, tgx ^ 16); \*/                                                     \
  value +=                                                                     \
      dpct::select_from_sub_group(item.get_sub_group(), value, tgx ^ 16);  \
  if (tgx == 0)                                                                \
  {                                                                            \
   /* DPCT_ORIG 			atomicAdd(pAccumulator, value); \*/                         \
   dpct::atomic_fetch_add(pAccumulator, value);                                \
  }                                                                            \
 }                                                                             \
 /* DPCT_ORIG 	__threadfence(); \*/                                            \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);  \
 /* DPCT_ORIG 	__syncthreads(); \*/                                            \
 item.barrier();                                                           \
 value = *pAccumulator;                                                        \
 /* DPCT_ORIG 	__syncthreads();*/                                              \
 item.barrier();

#define REDUCEFLOATSUM(value, pAccumulator)                             \
 if (sycl::ext::oneapi::experimental::this_nd_item().get_local_id(0) == 0) \
 {                                                                      \
  *pAccumulator = 0;                                                    \
 }                                                                      \
                                                                        \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device); \
 sycl::ext::oneapi::experimental::this_nd_item().barrier();             \
                                                                        \
 sycl::sub_group sg = sycl::ext::oneapi::experimental::this_nd_item().get_sub_group(); \
 if (sycl::any_of_group(sg,                                             \
                        (0xffffffff &                                   \
                         (0x1 << sg.get_local_linear_id())) &&          \
                        value != 0.0f))                                 \
 {                                                                      \
  uint32_t tgx = sycl::ext::oneapi::experimental::this_nd_item().get_local_id(2) & cData.warpmask; \
  value += dpct::select_from_sub_group(sg, value, tgx ^ 1);             \
  value += dpct::select_from_sub_group(sg, value, tgx ^ 2);             \
  value += dpct::select_from_sub_group(sg, value, tgx ^ 4);             \
  value += dpct::select_from_sub_group(sg, value, tgx ^ 8);             \
  if (tgx == 0)                                                         \
  {                                                                     \
   dpct::atomic_fetch_add(pAccumulator, value);                         \
  }                                                                     \
 }                                                                      \
                                                                        \
 sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device); \
 sycl::ext::oneapi::experimental::this_nd_item().barrier();             \
 value = (float)(*pAccumulator);                                        \
 sycl::ext::oneapi::experimental::this_nd_item().barrier();

static sycl::ext::oneapi::experimental::device_global<GpuData> cData;
static GpuData cpuData;

void SetKernelsGpuData(GpuData *pData) {
  get_sycl_queue()->memcpy(cData.get(), pData, sizeof(GpuData)).wait();
  memcpy(&cpuData, pData, sizeof(GpuData));
}

void GetKernelsGpuData(GpuData *pData) {
  get_sycl_queue()->memcpy(pData, cData.get(), sizeof(GpuData)).wait();
}

// Kernel files
#include "calcenergy.cpp"
#include "calcMergeEneGra.cpp"
#include "auxiliary_genetic.cpp"
#include "kernel1.cpp"
#include "kernel2.cpp"
#include "kernel3.cpp"
#include "kernel4.cpp"
#include "kernel_ad.cpp"
#include "kernel_adam.cpp"
