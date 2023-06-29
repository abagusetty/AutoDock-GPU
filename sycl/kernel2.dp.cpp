/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian
Genetic Algorithm Copyright (C) 2017 TU Darmstadt, Embedded Systems and
Applications Group, Germany. All rights reserved. For some of the code,
Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research
Institute.

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

__attribute__((always_inline)) void
gpu_sum_evals_kernel(sycl::nd_item<1> item,
                     GpuData& cData)
// The GPU global function sums the evaluation counter states
// which are stored in evals_of_new_entities array foreach entity,
// calculates the sums for each run and stores it in evals_of_runs array.
// The number of blocks which should be started equals to num_of_runs,
// since each block performs the summation for one run.
{
  sycl::group thread_block = item.get_group();

  auto sSum_evals_multptr = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
  int* sSum_evals = sSum_evals_multptr.get();

  int partsum_evals = 0;
  int *pEvals_of_new_entities = cData.get().pMem_evals_of_new_entities + item.get_group(0) * cData.get().dockpars.pop_size;
  for (int entity_counter = item.get_local_id(0);
       entity_counter < cData.get().dockpars.pop_size;
       entity_counter += item.get_local_range(0))
  {
    partsum_evals += pEvals_of_new_entities[entity_counter];
  }

  // Perform warp-wise reduction
  sycl::reduce_over_group(item.get_sub_group(), , sycl::plus<int>());
  //REDUCEINTEGERSUM(partsum_evals, &sSum_evals);
  if (item.get_local_id(0) == 0)
  {
    cData.get().pMem_gpu_evals_of_runs[item.get_group(0)] += (*sSum_evals);
  }
}


void gpu_sum_evals(uint32_t blocks, uint32_t threadsPerBlock)
{
    get_sycl_queue()->submit([&](sycl::handler &cgh) {
        extern dpct::constant_memory<GpuData, 0> cData;
        cData.get().init();
        auto cData_ptr = cData.get().get_ptr();

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(blocks * threadsPerBlock),
                              sycl::range<1>(threadsPerBlock)),
            [=](sycl::nd_item<1> item) {
                gpu_sum_evals_kernel(item, *cData_ptr);
            });
    });
}
