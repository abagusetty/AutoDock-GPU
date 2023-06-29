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

void
gpu_calc_initpop_kernel(float* pMem_conformations_current,
                        float* pMem_energies_current,
                        sycl::nd_item<3> item,
                        GpuData cData)
{
  sycl::group thread_block = item.get_group();

  using tile_float3_t = float3[MAX_NUM_OF_ATOMS];
  tile_float3_t& calc_coords = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_float3_t>(thread_block);
  float& sFloatAccumulator = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[1]>(thread_block);
  
  float  energy = 0.0f;
  int run_id = item.get_group(0) / cData.dockpars.pop_size;
  float *pGenotype = pMem_conformations_current + item.get_group(0) * GENOTYPE_LENGTH_IN_GLOBMEM;

  // =============================================================
  gpu_calc_energy(pGenotype, energy, run_id, calc_coords,
                  &sFloatAccumulator, item, cData);
  // =============================================================

  // Write out final energy
  if (item.get_local_id(0) == 0)
  {
    pMem_energies_current[item.get_group(0)] = energy;
    cData.pMem_evals_of_new_entities[item.get_group(0)] = 1;
  }
}

void gpu_calc_initpop(
                      uint32_t blocks,
                      uint32_t threadsPerBlock,
                      float*   pConformations_current,
                      float*   pEnergies_current
                     )
{
  get_sycl_queue()->submit([&](sycl::handler &cgh) {
      extern dpct::constant_memory<GpuData, 0> cData;
      cData.init();
      auto cData_ptr_ct1 = cData.get_ptr();

      cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks) * sycl::range<1>(threadsPerBlock),
                          sycl::range<1>(threadsPerBlock)),
        [=](sycl::nd_item<1> item) {
          gpu_calc_initpop_kernel(pConformations_current, pEnergies_current,
                                  item, *cData_ptr_ct1);
        });
    });
}

