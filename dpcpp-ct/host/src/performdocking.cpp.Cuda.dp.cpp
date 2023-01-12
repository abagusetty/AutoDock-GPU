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


// If defined, will set the maximum Cuda printf FIFO buffer to 8 GB (default: commented out)
// This is not needed unless debugging Cuda kernels via printf statements
// #define SET_CUDA_PRINTF_BUFFER

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <vector>

#include "autostop.hpp"
#include "performdocking.h"
#include "correct_grad_axisangle.h"
#include "GpuData.h"
#include <cmath>

#include <time.h>

// CUDA kernels
void SetKernelsGpuData(GpuData* pData);

void GetKernelsGpuData(GpuData* pData);

void gpu_calc_initpop(
                      uint32_t blocks,
                      uint32_t threadsPerBlock,
                      float*   pConformations_current,
                      float*   pEnergies_current
                     );

void gpu_sum_evals(
                   uint32_t blocks,
                   uint32_t threadsPerBlock
                  );

void gpu_gen_and_eval_newpops(
                              uint32_t blocks,
                              uint32_t threadsPerBlock,
                              float*   pMem_conformations_current,
                              float*   pMem_energies_current,
                              float*   pMem_conformations_next,
                              float*   pMem_energies_next
                             );

void gpu_gradient_minAD(
                        uint32_t blocks,
                        uint32_t threads,
                        float*   pMem_conformations_next,
                        float*   pMem_energies_next
                       );

void gpu_gradient_minAdam(
                          uint32_t blocks,
                          uint32_t threads,
                          float*  pMem_conformations_next,
                          float*  pMem_energies_next
                         );

void gpu_perform_LS(
                    uint32_t blocks,
                    uint32_t threads,
                    float*   pMem_conformations_next,
                    float*   pMem_energies_next
                   );

template <typename Clock, typename Duration1, typename Duration2>
double elapsed_seconds(
                       std::chrono::time_point<Clock, Duration1> start,
                       std::chrono::time_point<Clock, Duration2> end
                      )
{
	using FloatingPointSeconds = std::chrono::duration<double, std::ratio<1>>;
	return std::chrono::duration_cast<FloatingPointSeconds>(end - start).count();
}

std::vector<int> get_gpu_pool() try {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
        int gpuCount=0;
/* DPCT_ORIG 	cudaError_t status;*/
        int status;
/* DPCT_ORIG 	status = cudaGetDeviceCount(&gpuCount);*/
        /*
        DPCT1003:65: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (gpuCount = dpct::dev_mgr::instance().device_count(), 0);
        /*
        DPCT1001:63: The statement could not be removed.
        */
        RTERROR(status, "cudaGetDeviceCount failed");
        std::vector<int> result;
/* DPCT_ORIG 	cudaDeviceProp props;*/
        dpct::device_info props;
        for(unsigned int i=0; i<gpuCount; i++){
/* DPCT_ORIG
 * RTERROR(cudaGetDeviceProperties(&props,i),"cudaGetDeviceProperties
 * failed");*/
                /*
                DPCT1003:67: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                RTERROR(
                    (dpct::dev_mgr::instance().get_device(i).get_device_info(
                         props),
                     0),
                    "cudaGetDeviceProperties failed");
/* DPCT_ORIG 		if(props.major>=3) result.push_back(i);*/
                /*
                DPCT1005:68: The SYCL device version is different from CUDA
                Compute Compatibility. You may need to rewrite this code.
                */
                if (props.get_major_version() >= 3) result.push_back(i);
        }
	if (result.size() == 0)
	{
		printf("No CUDA devices with compute capability >= 3.0 found, exiting.\n");
/* DPCT_ORIG 		cudaDeviceReset();*/
                dpct::get_current_device().reset();
                exit(-1);
	}
	return result;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void setup_gpu_for_docking(GpuData &cData, GpuTempData &tData) try {
        if(cData.devnum<-1) return; // device already setup
	auto const t0 = std::chrono::steady_clock::now();

	// Initialize CUDA
	int gpuCount=0;
/* DPCT_ORIG 	cudaError_t status = cudaGetDeviceCount(&gpuCount);*/
        /*
        DPCT1003:85: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        int status = (gpuCount = dpct::dev_mgr::instance().device_count(), 0);
        /*
        DPCT1001:69: The statement could not be removed.
        */
        RTERROR(status, "cudaGetDeviceCount failed");
        if (gpuCount == 0)
	{
		printf("No CUDA-capable devices found, exiting.\n");
/* DPCT_ORIG 		cudaDeviceReset();*/
                dpct::get_current_device().reset();
                exit(-1);
	}
	if (cData.devnum>=gpuCount){
		printf("Error: Requested device %i does not exist (only %i devices available).\n",cData.devnum+1,gpuCount);
		exit(-1);
	}
	if (cData.devnum<0)
/* DPCT_ORIG 		status = cudaFree(NULL); */
                /*
                DPCT1003:86: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                status =
                    (sycl::free(NULL, dpct::get_default_queue()),
                     0); // Trick driver into creating context on current device
        else
/* DPCT_ORIG 		status = cudaSetDevice(cData.devnum);*/
                /*
                DPCT1093:87: The "cData.devnum" may not be the best XPU device.
                Adjust the selected device if needed.
                */
                /*
                DPCT1003:88: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                status = (dpct::select_device(cData.devnum), 0);
        // Now that we have a device, gather some information
	size_t freemem, totalmem;
/* DPCT_ORIG 	cudaDeviceProp props;*/
        dpct::device_info props;
/* DPCT_ORIG 	RTERROR(cudaGetDevice(&(cData.devnum)),"cudaGetDevice
 * failed");*/
        RTERROR(cData.devnum = dpct::dev_mgr::instance().current_device_id(),
                "cudaGetDevice failed");
/* DPCT_ORIG
 * RTERROR(cudaGetDeviceProperties(&props,cData.devnum),"cudaGetDeviceProperties
 * failed");*/
        /*
        DPCT1003:89: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        RTERROR((dpct::dev_mgr::instance()
                     .get_device(cData.devnum)
                     .get_device_info(props),
                 0),
                "cudaGetDeviceProperties failed");
/* DPCT_ORIG 	tData.device_name = (char*) malloc(strlen(props.name)+32); */
        tData.device_name = (char *)malloc(
            strlen(props.get_name()) + 32); // make sure array is large enough
                                            // to hold device number text too
/* DPCT_ORIG 	strcpy(tData.device_name, props.name);*/
        strcpy(tData.device_name, props.get_name());
/* DPCT_ORIG 	if(gpuCount>1) snprintf(&tData.device_name[strlen(props.name)],
 * strlen(props.name)+32, " (#%d / %d)",cData.devnum+1,gpuCount);*/
        if (gpuCount > 1) snprintf(&tData.device_name[strlen(props.get_name())],
                                   strlen(props.get_name()) + 32, " (#%d / %d)",
                                   cData.devnum + 1, gpuCount);
        printf("Cuda device:                              %s\n",tData.device_name);
/* DPCT_ORIG 	RTERROR(cudaMemGetInfo(&freemem,&totalmem), "cudaGetMemInfo
 * failed");*/
        /*
        DPCT1003:90: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        RTERROR(
            (dpct::get_current_device().get_memory_info(freemem, totalmem), 0),
            "cudaGetMemInfo failed");
        printf("Available memory on device:               %lu MB (total: %lu MB)\n",(freemem>>20),(totalmem>>20));
	cData.devid=cData.devnum;
	cData.devnum=-2;
#ifdef SET_CUDA_PRINTF_BUFFER
	status = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 200000000ull);
	RTERROR(status, "cudaDeviceSetLimit failed");
#endif
	auto const t1 = std::chrono::steady_clock::now();
	printf("\nCUDA Setup time %fs\n", elapsed_seconds(t0 ,t1));

	// Allocate kernel constant GPU memory
/* DPCT_ORIG 	status = cudaMalloc((void**)&cData.pKerconst_interintra,
 * sizeof(kernelconstant_interintra));*/
        /*
        DPCT1003:91: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (cData.pKerconst_interintra =
                      sycl::malloc_device<kernelconstant_interintra>(
                          1, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:70: The statement could not be removed.
        */
        RTERROR(status,
                "cData.pKerconst_interintra: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&cData.pKerconst_intracontrib,
 * sizeof(kernelconstant_intracontrib));*/
        /*
        DPCT1003:92: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (cData.pKerconst_intracontrib =
                      sycl::malloc_device<kernelconstant_intracontrib>(
                          1, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:71: The statement could not be removed.
        */
        RTERROR(
            status,
            "cData.pKerconst_intracontrib: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&cData.pKerconst_intra,
 * sizeof(kernelconstant_intra));*/
        /*
        DPCT1003:93: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status =
            (cData.pKerconst_intra = sycl::malloc_device<kernelconstant_intra>(
                 1, dpct::get_default_queue()),
             0);
        /*
        DPCT1001:72: The statement could not be removed.
        */
        RTERROR(status, "cData.pKerconst_intra: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&cData.pKerconst_rotlist,
 * sizeof(kernelconstant_rotlist));*/
        /*
        DPCT1003:94: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (cData.pKerconst_rotlist =
                      sycl::malloc_device<kernelconstant_rotlist>(
                          1, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:73: The statement could not be removed.
        */
        RTERROR(status, "cData.pKerconst_rotlist: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&cData.pKerconst_conform,
 * sizeof(kernelconstant_conform));*/
        /*
        DPCT1003:95: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (cData.pKerconst_conform =
                      sycl::malloc_device<kernelconstant_conform>(
                          1, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:74: The statement could not be removed.
        */
        RTERROR(status, "cData.pKerconst_conform: failed to allocate GPU memory.\n");

        // Allocate mem data
/* DPCT_ORIG 	status = cudaMalloc((void**)&cData.pMem_rotbonds_const,
 * 2*MAX_NUM_OF_ROTBONDS*sizeof(int));*/
        /*
        DPCT1003:96: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (cData.pMem_rotbonds_const = sycl::malloc_device<int>(
                      2 * MAX_NUM_OF_ROTBONDS, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:75: The statement could not be removed.
        */
        RTERROR(status, "cData.pMem_rotbonds_const: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&cData.pMem_rotbonds_atoms_const,
 * MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS*sizeof(int));*/
        /*
        DPCT1003:97: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (cData.pMem_rotbonds_atoms_const = sycl::malloc_device<int>(
                      MAX_NUM_OF_ATOMS * MAX_NUM_OF_ROTBONDS,
                      dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:76: The statement could not be removed.
        */
        RTERROR(status, "cData.pMem_rotbonds_atoms_const: failed to allocate "
                        "GPU memory.\n");
/* DPCT_ORIG 	status =
 * cudaMalloc((void**)&cData.pMem_num_rotating_atoms_per_rotbond_const,
 * MAX_NUM_OF_ROTBONDS*sizeof(int));*/
        /*
        DPCT1003:98: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (cData.pMem_num_rotating_atoms_per_rotbond_const =
                      sycl::malloc_device<int>(MAX_NUM_OF_ROTBONDS,
                                               dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:77: The statement could not be removed.
        */
        RTERROR(status, "cData.pMem_num_rotiating_atoms_per_rotbond_const: "
                        "failed to allocate GPU memory.\n");

        // Allocate temporary data - JL TODO - Are these sizes correct?
	if(cData.preallocated_gridsize>0){
/* DPCT_ORIG 		status = cudaMalloc((void**)&(tData.pMem_fgrids),
 * cData.preallocated_gridsize*sizeof(float));*/
                /*
                DPCT1003:100: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                status = (tData.pMem_fgrids = sycl::malloc_device<float>(
                              cData.preallocated_gridsize,
                              dpct::get_default_queue()),
                          0);
                /*
                DPCT1001:99: The statement could not be removed.
                */
                RTERROR(status, "pMem_fgrids: failed to allocate GPU memory.\n");
        }
	size_t size_populations = MAX_NUM_OF_RUNS * MAX_POPSIZE * GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float);
/* DPCT_ORIG 	status = cudaMalloc((void**)&(tData.pMem_conformations1),
 * size_populations);*/
        /*
        DPCT1003:101: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (tData.pMem_conformations1 = (float *)sycl::malloc_device(
                      size_populations, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:78: The statement could not be removed.
        */
        RTERROR(status, "pMem_conformations1: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&(tData.pMem_conformations2),
 * size_populations);*/
        /*
        DPCT1003:102: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (tData.pMem_conformations2 = (float *)sycl::malloc_device(
                      size_populations, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:79: The statement could not be removed.
        */
        RTERROR(status, "pMem_conformations2: failed to allocate GPU memory.\n");
        size_t size_energies = MAX_POPSIZE * MAX_NUM_OF_RUNS * sizeof(float);
/* DPCT_ORIG 	status = cudaMalloc((void**)&(tData.pMem_energies1),
 * size_energies);*/
        /*
        DPCT1003:103: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (tData.pMem_energies1 = (float *)sycl::malloc_device(
                      size_energies, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:80: The statement could not be removed.
        */
        RTERROR(status, "pMem_energies1: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&(tData.pMem_energies2),
 * size_energies);*/
        /*
        DPCT1003:104: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (tData.pMem_energies2 = (float *)sycl::malloc_device(
                      size_energies, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:81: The statement could not be removed.
        */
        RTERROR(status, "pMem_energies2: failed to allocate GPU memory.\n");
/* DPCT_ORIG 	status = cudaMalloc((void**)&(tData.pMem_evals_of_new_entities),
 * MAX_POPSIZE*MAX_NUM_OF_RUNS*sizeof(int));*/
        /*
        DPCT1003:105: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (tData.pMem_evals_of_new_entities = sycl::malloc_device<int>(
                      MAX_POPSIZE * MAX_NUM_OF_RUNS, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:82: The statement could not be removed.
        */
        RTERROR(status,
                "pMem_evals_of_new_Entities: failed to allocate GPU memory.\n");
        size_t size_evals_of_runs = MAX_NUM_OF_RUNS*sizeof(int);
#if defined (MAPPED_COPY)
/* DPCT_ORIG 	status =
 * cudaMallocManaged((void**)&(tData.pMem_gpu_evals_of_runs),
 * size_evals_of_runs, cudaMemAttachGlobal);*/
        /*
        DPCT1003:106: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (tData.pMem_gpu_evals_of_runs = (int *)sycl::malloc_shared(
                      size_evals_of_runs, dpct::get_default_queue()),
                  0);
#else
	status = cudaMalloc((void**)&(tData.pMem_gpu_evals_of_runs), size_evals_of_runs);
#endif
        /*
        DPCT1001:83: The statement could not be removed.
        */
        RTERROR(status, "pMem_gpu_evals_of_runs: failed to allocate GPU memory.\n");
        size_t blocksPerGridForEachEntity = MAX_POPSIZE * MAX_NUM_OF_RUNS;
	size_t size_prng_seeds = blocksPerGridForEachEntity * NUM_OF_THREADS_PER_BLOCK * sizeof(unsigned int);
/* DPCT_ORIG 	status = cudaMalloc((void**)&(tData.pMem_prng_states),
 * size_prng_seeds);*/
        /*
        DPCT1003:107: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (tData.pMem_prng_states = (uint32_t *)sycl::malloc_device(
                      size_prng_seeds, dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:84: The statement could not be removed.
        */
        RTERROR(status, "pMem_prng_states: failed to allocate GPU memory.\n");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
void finish_gpu_from_docking(GpuData &cData, GpuTempData &tData) try {
        if(cData.devnum>-2) return; // device not set up

/* DPCT_ORIG 	cudaError_t status;*/
        int status;
        // Release all CUDA objects
	// Constant objects
/* DPCT_ORIG 	status = cudaFree(cData.pKerconst_interintra);*/
        /*
        DPCT1003:123: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status =
            (sycl::free(cData.pKerconst_interintra, dpct::get_default_queue()),
             0);
        /*
        DPCT1001:108: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing cData.pKerconst_interintra\n");
/* DPCT_ORIG 	status = cudaFree(cData.pKerconst_intracontrib);*/
        /*
        DPCT1003:124: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(cData.pKerconst_intracontrib,
                             dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:109: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing cData.pKerconst_intracontrib\n");
/* DPCT_ORIG 	status = cudaFree(cData.pKerconst_intra);*/
        /*
        DPCT1003:125: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(cData.pKerconst_intra, dpct::get_default_queue()), 0);
        /*
        DPCT1001:110: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing cData.pKerconst_intra\n");
/* DPCT_ORIG 	status = cudaFree(cData.pKerconst_rotlist);*/
        /*
        DPCT1003:126: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(cData.pKerconst_rotlist, dpct::get_default_queue()), 0);
        /*
        DPCT1001:111: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing cData.pKerconst_rotlist\n");
/* DPCT_ORIG 	status = cudaFree(cData.pKerconst_conform);*/
        /*
        DPCT1003:127: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(cData.pKerconst_conform, dpct::get_default_queue()), 0);
        /*
        DPCT1001:112: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing cData.pKerconst_conform\n");
/* DPCT_ORIG 	status = cudaFree(cData.pMem_rotbonds_const);*/
        /*
        DPCT1003:128: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(cData.pMem_rotbonds_const, dpct::get_default_queue()), 0);
        /*
        DPCT1001:113: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing cData.pMem_rotbonds_const");
/* DPCT_ORIG 	status = cudaFree(cData.pMem_rotbonds_atoms_const);*/
        /*
        DPCT1003:129: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(cData.pMem_rotbonds_atoms_const,
                             dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:114: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing cData.pMem_rotbonds_atoms_const");
/* DPCT_ORIG 	status =
 * cudaFree(cData.pMem_num_rotating_atoms_per_rotbond_const);*/
        /*
        DPCT1003:130: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(cData.pMem_num_rotating_atoms_per_rotbond_const,
                             dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:115: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing "
                        "cData.pMem_num_rotating_atoms_per_rotbond_const");

        // Non-constant
	if(tData.pMem_fgrids){
/* DPCT_ORIG 		status = cudaFree(tData.pMem_fgrids);*/
                /*
                DPCT1003:132: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                status = (sycl::free(tData.pMem_fgrids, dpct::get_default_queue()), 0);
                /*
                DPCT1001:131: The statement could not be removed.
                */
                RTERROR(status, "cudaFree: error freeing pMem_fgrids");
        }
/* DPCT_ORIG 	status = cudaFree(tData.pMem_conformations1);*/
        /*
        DPCT1003:133: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(tData.pMem_conformations1, dpct::get_default_queue()), 0);
        /*
        DPCT1001:116: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing pMem_conformations1");
/* DPCT_ORIG 	status = cudaFree(tData.pMem_conformations2);*/
        /*
        DPCT1003:134: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(tData.pMem_conformations2, dpct::get_default_queue()), 0);
        /*
        DPCT1001:117: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing pMem_conformations2");
/* DPCT_ORIG 	status = cudaFree(tData.pMem_energies1);*/
        /*
        DPCT1003:135: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(tData.pMem_energies1, dpct::get_default_queue()), 0);
        /*
        DPCT1001:118: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing pMem_energies1");
/* DPCT_ORIG 	status = cudaFree(tData.pMem_energies2);*/
        /*
        DPCT1003:136: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(tData.pMem_energies2, dpct::get_default_queue()), 0);
        /*
        DPCT1001:119: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing pMem_energies2");
/* DPCT_ORIG 	status = cudaFree(tData.pMem_evals_of_new_entities);*/
        /*
        DPCT1003:137: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(tData.pMem_evals_of_new_entities,
                             dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:120: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing pMem_evals_of_new_entities");
/* DPCT_ORIG 	status = cudaFree(tData.pMem_gpu_evals_of_runs);*/
        /*
        DPCT1003:138: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(tData.pMem_gpu_evals_of_runs,
                             dpct::get_default_queue()),
                  0);
        /*
        DPCT1001:121: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing pMem_gpu_evals_of_runs");
/* DPCT_ORIG 	status = cudaFree(tData.pMem_prng_states);*/
        /*
        DPCT1003:139: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (sycl::free(tData.pMem_prng_states, dpct::get_default_queue()), 0);
        /*
        DPCT1001:122: The statement could not be removed.
        */
        RTERROR(status, "cudaFree: error freeing pMem_prng_states");
        free(tData.device_name);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int docking_with_gpu(const Gridinfo *mygrid, Dockpars *mypars,
                     const Liganddata *myligand_init,
                     const Liganddata *myxrayligand, Profile &profile,
                     const int *argc, char **argv, SimulationState &sim_state,
                     GpuData &cData, GpuTempData &tData, std::string *output)
    /* The function performs the docking algorithm and generates the
    corresponding result files. parameter mygrid: describes the grid filled with
    get_gridinfo() parameter mypars: describes the docking parameters filled
    with get_commandpars() parameter myligand_init: describes the ligands filled
    with parse_liganddata() parameter myxrayligand: describes the xray ligand
                    filled with get_xrayliganddata()
    parameters argc and argv:
                    are the corresponding command line arguments parameter
    */
    try {
        char* outbuf;
	if(output!=NULL) outbuf = (char*)malloc(256*sizeof(char));

	auto const t1 = std::chrono::steady_clock::now();
/* DPCT_ORIG 	cudaError_t status = cudaSetDevice(cData.devid); */
        /*
        DPCT1093:154: The "cData.devid" may not be the best XPU device. Adjust
        the selected device if needed.
        */
        /*
        DPCT1003:155: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        int status = (dpct::select_device(cData.devid),
                      0); // make sure we're on the correct device

        Liganddata myligand_reference;

	float* cpu_init_populations;
	float* cpu_final_populations;
	unsigned int* cpu_prng_seeds;

	size_t size_populations;
	size_t size_energies;
	size_t size_prng_seeds;
	size_t size_evals_of_runs;

	int threadsPerBlock;
	int blocksPerGridForEachEntity;
	int blocksPerGridForEachRun;
	int blocksPerGridForEachLSEntity = 0;
	int blocksPerGridForEachGradMinimizerEntity = 0;

	int generation_cnt;
	int i;
	double progress;

	int curr_progress_cnt;
	int new_progress_cnt;

	clock_t clock_start_docking;
	clock_t	clock_stop_docking;

	// setting number of blocks and threads
	threadsPerBlock = NUM_OF_THREADS_PER_BLOCK;
	blocksPerGridForEachEntity = mypars->pop_size * mypars->num_of_runs;
	blocksPerGridForEachRun = mypars->num_of_runs;

	// allocating CPU memory for initial populations
	size_populations = mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float);
	sim_state.cpu_populations.resize(size_populations);
	memset(sim_state.cpu_populations.data(), 0, size_populations);

	// allocating CPU memory for results
	size_energies = mypars->pop_size * mypars->num_of_runs * sizeof(float);
	sim_state.cpu_energies.resize(size_energies);
	cpu_init_populations = sim_state.cpu_populations.data();
	cpu_final_populations = sim_state.cpu_populations.data();

	// generating initial populations and random orientation angles of reference ligand
	// (ligand will be moved to origo and scaled as well)
	myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, cpu_init_populations, &myligand_reference, mygrid);

	// allocating memory in CPU for pseudorandom number generator seeds and
	// generating them (seed for each thread during GA)
	size_prng_seeds = blocksPerGridForEachEntity * threadsPerBlock * sizeof(unsigned int);
	cpu_prng_seeds = (unsigned int*) malloc(size_prng_seeds);

	LocalRNG r(mypars->seed);
//	para_printf("RNG seed is %u\n", mypars->seed);

	for (i=0; i<blocksPerGridForEachEntity*threadsPerBlock; i++)
		cpu_prng_seeds[i] = r.random_uint();

	// allocating memory in CPU for evaluation counters
	size_evals_of_runs = mypars->num_of_runs*sizeof(int);
	sim_state.cpu_evals_of_runs.resize(size_evals_of_runs);
	memset(sim_state.cpu_evals_of_runs.data(), 0, size_evals_of_runs);

	// preparing the constant data fields for the GPU
	kernelconstant_interintra*	KerConst_interintra = new kernelconstant_interintra;
	kernelconstant_intracontrib*	KerConst_intracontrib = new kernelconstant_intracontrib;
	kernelconstant_intra*		KerConst_intra = new kernelconstant_intra;
	kernelconstant_rotlist*		KerConst_rotlist = new kernelconstant_rotlist;
	kernelconstant_conform*		KerConst_conform = new kernelconstant_conform;
	kernelconstant_grads*		KerConst_grads = new kernelconstant_grads;

	if (prepare_const_fields_for_gpu(&myligand_reference, mypars,
	                                 KerConst_interintra,
	                                 KerConst_intracontrib,
	                                 KerConst_intra,
	                                 KerConst_rotlist,
	                                 KerConst_conform,
	                                 KerConst_grads) == 1) {
		return 1;
	}

	// Upload kernel constant data - JL FIXME - Can these be moved once?
/* DPCT_ORIG 	status = cudaMemcpy(cData.pKerconst_interintra,
 * KerConst_interintra, sizeof(kernelconstant_interintra),
 * cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:156: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(cData.pKerconst_interintra, KerConst_interintra,
                              sizeof(kernelconstant_interintra))
                      .wait(),
                  0);
        /*
        DPCT1001:140: The statement could not be removed.
        */
        RTERROR(
            status,
            "cData.pKerconst_interintra: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	status = cudaMemcpy(cData.pKerconst_intracontrib,
 * KerConst_intracontrib, sizeof(kernelconstant_intracontrib),
 * cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:157: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status =
            (dpct::get_default_queue()
                 .memcpy(cData.pKerconst_intracontrib, KerConst_intracontrib,
                         sizeof(kernelconstant_intracontrib))
                 .wait(),
             0);
        /*
        DPCT1001:141: The statement could not be removed.
        */
        RTERROR(
            status,
            "cData.pKerconst_intracontrib: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	status = cudaMemcpy(cData.pKerconst_intra, KerConst_intra,
 * sizeof(kernelconstant_intra), cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:158: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(cData.pKerconst_intra, KerConst_intra,
                              sizeof(kernelconstant_intra))
                      .wait(),
                  0);
        /*
        DPCT1001:142: The statement could not be removed.
        */
        RTERROR(status, "cData.pKerconst_intra: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	status = cudaMemcpy(cData.pKerconst_rotlist, KerConst_rotlist,
 * sizeof(kernelconstant_rotlist), cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:159: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(cData.pKerconst_rotlist, KerConst_rotlist,
                              sizeof(kernelconstant_rotlist))
                      .wait(),
                  0);
        /*
        DPCT1001:143: The statement could not be removed.
        */
        RTERROR(status, "cData.pKerconst_rotlist: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	status = cudaMemcpy(cData.pKerconst_conform, KerConst_conform,
 * sizeof(kernelconstant_conform), cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:160: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(cData.pKerconst_conform, KerConst_conform,
                              sizeof(kernelconstant_conform))
                      .wait(),
                  0);
        /*
        DPCT1001:144: The statement could not be removed.
        */
        RTERROR(status, "cData.pKerconst_conform: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	cudaMemcpy(cData.pMem_rotbonds_const, KerConst_grads->rotbonds,
 * sizeof(KerConst_grads->rotbonds), cudaMemcpyHostToDevice);*/
        dpct::get_default_queue()
            .memcpy(cData.pMem_rotbonds_const, KerConst_grads->rotbonds,
                    sizeof(KerConst_grads->rotbonds))
            .wait();
        /*
        DPCT1001:145: The statement could not be removed.
        */
        RTERROR(status,
                "cData.pMem_rotbonds_const: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	cudaMemcpy(cData.pMem_rotbonds_atoms_const,
 * KerConst_grads->rotbonds_atoms, sizeof(KerConst_grads->rotbonds_atoms),
 * cudaMemcpyHostToDevice);*/
        dpct::get_default_queue()
            .memcpy(cData.pMem_rotbonds_atoms_const,
                    KerConst_grads->rotbonds_atoms,
                    sizeof(KerConst_grads->rotbonds_atoms))
            .wait();
        /*
        DPCT1001:146: The statement could not be removed.
        */
        RTERROR(status, "cData.pMem_rotbonds_atoms_const: failed to upload to "
                        "GPU memory.\n");
/* DPCT_ORIG 	cudaMemcpy(cData.pMem_num_rotating_atoms_per_rotbond_const,
 * KerConst_grads->num_rotating_atoms_per_rotbond,
 * sizeof(KerConst_grads->num_rotating_atoms_per_rotbond),
 * cudaMemcpyHostToDevice);*/
        dpct::get_default_queue()
            .memcpy(cData.pMem_num_rotating_atoms_per_rotbond_const,
                    KerConst_grads->num_rotating_atoms_per_rotbond,
                    sizeof(KerConst_grads->num_rotating_atoms_per_rotbond))
            .wait();
        /*
        DPCT1001:147: The statement could not be removed.
        */
        RTERROR(status, "cData.pMem_num_rotating_atoms_per_rotbond_const "
                        "failed to upload to GPU memory.\n");

        // allocating GPU memory for grids, populations, energies,
	// evaluation counters and random number generator states
	if(cData.preallocated_gridsize==0){
/* DPCT_ORIG 		status = cudaMalloc((void**)&(tData.pMem_fgrids),
 * mygrid->grids.size()*sizeof(float));*/
                /*
                DPCT1003:162: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                status = (tData.pMem_fgrids = sycl::malloc_device<float>(
                              mygrid->grids.size(), dpct::get_default_queue()),
                          0);
                /*
                DPCT1001:161: The statement could not be removed.
                */
                RTERROR(status, "pMem_fgrids: failed to allocate GPU memory.\n");
        }
	// Flippable pointers
	float* pMem_conformations_current = tData.pMem_conformations1;
	float* pMem_conformations_next = tData.pMem_conformations2;
	float* pMem_energies_current = tData.pMem_energies1;
	float* pMem_energies_next = tData.pMem_energies2;

	// Set constant pointers
	cData.pMem_fgrids = tData.pMem_fgrids;
	cData.pMem_evals_of_new_entities = tData.pMem_evals_of_new_entities;
	cData.pMem_gpu_evals_of_runs = tData.pMem_gpu_evals_of_runs;
	cData.pMem_prng_states = tData.pMem_prng_states;

	// Set CUDA constants
	cData.warpmask = 31;
	cData.warpbits = 5;

	// Upload data
/* DPCT_ORIG 	status = cudaMemcpy(tData.pMem_fgrids, mygrid->grids.data(),
 * mygrid->grids.size()*sizeof(float), cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:163: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(tData.pMem_fgrids, mygrid->grids.data(),
                              mygrid->grids.size() * sizeof(float))
                      .wait(),
                  0);
        /*
        DPCT1001:148: The statement could not be removed.
        */
        RTERROR(status, "pMem_fgrids: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	status = cudaMemcpy(pMem_conformations_current,
 * cpu_init_populations, size_populations, cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:164: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(pMem_conformations_current, cpu_init_populations,
                              size_populations)
                      .wait(),
                  0);
        /*
        DPCT1001:149: The statement could not be removed.
        */
        RTERROR(
            status,
            "pMem_conformations_current: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	status = cudaMemcpy(tData.pMem_gpu_evals_of_runs,
 * sim_state.cpu_evals_of_runs.data(), size_evals_of_runs,
 * cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:165: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status =
            (dpct::get_default_queue()
                 .memcpy(tData.pMem_gpu_evals_of_runs,
                         sim_state.cpu_evals_of_runs.data(), size_evals_of_runs)
                 .wait(),
             0);
        /*
        DPCT1001:150: The statement could not be removed.
        */
        RTERROR(status, "pMem_gpu_evals_of_runs: failed to upload to GPU memory.\n");
/* DPCT_ORIG 	status = cudaMemcpy(tData.pMem_prng_states, cpu_prng_seeds,
 * size_prng_seeds, cudaMemcpyHostToDevice);*/
        /*
        DPCT1003:166: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(tData.pMem_prng_states, cpu_prng_seeds,
                              size_prng_seeds)
                      .wait(),
                  0);
        /*
        DPCT1001:151: The statement could not be removed.
        */
        RTERROR(status, "pMem_prng_states: failed to upload to GPU memory.\n");

        //preparing parameter struct
	cData.dockpars.num_of_atoms                 = myligand_reference.num_of_atoms;
	cData.dockpars.true_ligand_atoms            = ((int)  myligand_reference.true_ligand_atoms);
	cData.dockpars.num_of_atypes                = myligand_reference.num_of_atypes;
	cData.dockpars.num_of_map_atypes            = mygrid->num_of_map_atypes;
	cData.dockpars.num_of_intraE_contributors   = ((int) myligand_reference.num_of_intraE_contributors);
	cData.dockpars.gridsize_x                   = mygrid->size_xyz[0];
	cData.dockpars.gridsize_y                   = mygrid->size_xyz[1];
	cData.dockpars.gridsize_z                   = mygrid->size_xyz[2];
	cData.dockpars.gridsize_x_times_y           = cData.dockpars.gridsize_x * cData.dockpars.gridsize_y;
	cData.dockpars.gridsize_x_times_y_times_z   = cData.dockpars.gridsize_x * cData.dockpars.gridsize_y * cData.dockpars.gridsize_z;
	cData.dockpars.grid_spacing                 = ((float) mygrid->spacing);
	cData.dockpars.rotbondlist_length           = ((int) NUM_OF_THREADS_PER_BLOCK*(myligand_reference.num_of_rotcyc));
	cData.dockpars.coeff_elec                   = ((float) mypars->coeffs.scaled_AD4_coeff_elec);
	cData.dockpars.elec_min_distance            = ((float) mypars->elec_min_distance);
	cData.dockpars.coeff_desolv                 = ((float) mypars->coeffs.AD4_coeff_desolv);
	cData.dockpars.pop_size                     = mypars->pop_size;
	cData.dockpars.num_of_genes                 = myligand_reference.num_of_rotbonds + 6;
	// Notice: dockpars.tournament_rate, dockpars.crossover_rate, dockpars.mutation_rate
	// were scaled down to [0,1] in host to reduce number of operations in device
	cData.dockpars.tournament_rate              = mypars->tournament_rate/100.0f;
	cData.dockpars.crossover_rate               = mypars->crossover_rate/100.0f;
	cData.dockpars.mutation_rate                = mypars->mutation_rate/100.f;
	cData.dockpars.abs_max_dang                 = mypars->abs_max_dang;
	cData.dockpars.abs_max_dmov                 = mypars->abs_max_dmov;
	cData.dockpars.qasp                         = mypars->qasp;
	cData.dockpars.smooth                       = mypars->smooth;
	cData.dockpars.lsearch_rate                 = mypars->lsearch_rate;
	cData.dockpars.adam_beta1                   = mypars->adam_beta1;
	cData.dockpars.adam_beta2                   = mypars->adam_beta2;
	cData.dockpars.adam_epsilon                 = mypars->adam_epsilon;

	if (cData.dockpars.lsearch_rate != 0.0f)
	{
		cData.dockpars.num_of_lsentities        = (unsigned int) (mypars->lsearch_rate/100.0*mypars->pop_size + 0.5);
		cData.dockpars.rho_lower_bound          = mypars->rho_lower_bound;
		cData.dockpars.base_dmov_mul_sqrt3      = mypars->base_dmov_mul_sqrt3;
		cData.dockpars.base_dang_mul_sqrt3      = mypars->base_dang_mul_sqrt3;
		cData.dockpars.cons_limit               = (unsigned int) mypars->cons_limit;
		cData.dockpars.max_num_of_iters         = (unsigned int) mypars->max_num_of_iters;

		// The number of entities that undergo Solis-Wets minimization,
		blocksPerGridForEachLSEntity = cData.dockpars.num_of_lsentities * mypars->num_of_runs;

		// The number of entities that undergo any gradient-based minimization,
		// by default, it is the same as the number of entities that undergo the Solis-Wets minimizer
		blocksPerGridForEachGradMinimizerEntity = cData.dockpars.num_of_lsentities * mypars->num_of_runs;

		// Enable only for debugging.
		// Only one entity per reach run, undergoes gradient minimization
		//blocksPerGridForEachGradMinimizerEntity = mypars->num_of_runs;
	}

	unsigned long min_as_evals = 0; // no minimum w/o heuristics
	if(mypars->use_heuristics){
		unsigned long heur_evals;
		unsigned long nev=mypars->num_of_energy_evals;
		if(strcmp(mypars->ls_method,"sw")==0){
//			heur_evals = (unsigned long)ceil(1000 * pow(2.0,1.3 * myligand_init->num_of_rotbonds + 3.5));
/* DPCT_ORIG 			heur_evals = (unsigned long)ceil(1000 *
 * pow(2.0,1.3 * myligand_init->num_of_rotbonds + 4.0));*/
                        heur_evals = (unsigned long)ceil(
                            1000 *
                            pow(2.0,
                                1.3 * myligand_init->num_of_rotbonds + 4.0));
                } else{
			if(strcmp(mypars->ls_method,"ad")==0){
//			heur_evals = (unsigned long)ceil(1000 * pow(2.0,0.4 * myligand_init->num_of_rotbonds + 7.0));
//			heur_evals = (unsigned long)ceil(1000 * pow(2.0,0.5 * myligand_init->num_of_rotbonds + 6.0));
			heur_evals = (unsigned long)ceil(64000 * pow(2.0, (0.5 - 0.2 * myligand_init->num_of_rotbonds/(20.0f + myligand_init->num_of_rotbonds)) * myligand_init->num_of_rotbonds));
			} else{
				para_printf("\nError: LS method \"%s\" is not supported by heuristics.\n       Please choose Solis-Wets (sw), Adadelta (ad),\n       or switch off the heuristics.\n",mypars->ls_method);
				exit(-1);
			}
		}
		if(heur_evals<500000) heur_evals=500000;
		heur_evals *= 50.0f/mypars->num_of_runs;
		// e*hm/(hm+e) = 0.95*e => hm/(hm+e) = 0.95
		// => 0.95*hm + 0.95*e = hm => 0.95*e = 0.05 * hm
		// => e = 1/19*hm
		// at hm = 50 M => e0 = 2.63 M where e becomes less than 95% (about 11 torsions)
		mypars->num_of_energy_evals = (unsigned long)ceil(heur_evals*(float)mypars->heuristics_max/(mypars->heuristics_max+heur_evals));
		para_printf("    Using heuristics: (capped) number of evaluations set to %lu\n",mypars->num_of_energy_evals);
		if (mypars->nev_provided && (mypars->num_of_energy_evals>nev)){
			para_printf("    Overriding heuristics, setting number of evaluations to --nev = %lu instead.\n",nev);
			mypars->num_of_energy_evals = nev;
			profile.capped = true;
		}
		float cap_fraction = (float)mypars->num_of_energy_evals/heur_evals;
		float a = 27.0/26.0; // 10% at cap_fraction of 50%
//		float a = 12.0/11.0; // 20% at cap_fraction of 50%
		float min_frac = a/(1+cap_fraction*cap_fraction*(a/(a-1.0f)-1.0f))+1.0f-a;
		min_as_evals = (unsigned long)ceil(mypars->num_of_energy_evals*min_frac)*mypars->num_of_runs;
		if(cap_fraction<0.5f){
			para_printf("    Warning: The set number of evals is %.2f%% of the uncapped heuristics estimate of %lu evals.\n",cap_fraction*100.0f,heur_evals);
			para_printf("             This means this docking may not be able to converge. Increasing ");
			if (mypars->nev_provided && (mypars->num_of_energy_evals>nev))
				para_printf("--nev");
			else
				para_printf("--heurmax");
			para_printf(" may improve\n             convergence but will also increase runtime.\n");
			if(mypars->autostop) para_printf("             AutoStop will not stop before %.2f%% (%lu) of the set number of evaluations.\n",min_frac*100.0f,min_as_evals/mypars->num_of_runs);
		}
	}
	
	char method_chosen[64]; // 64 chars will be enough for this message as mypars->ls_method is 4 chars at the longest
	if(strcmp(mypars->ls_method, "sw") == 0){
		strcpy(method_chosen,"Solis-Wets (sw)");
	}
	else if(strcmp(mypars->ls_method, "ad") == 0){
		strcpy(method_chosen,"ADADELTA (ad)");
	}
	else if(strcmp(mypars->ls_method, "adam") == 0){
		strcpy(method_chosen,"ADAM (adam)");
	}
	else{
		para_printf("\nError: LS method %s is not (yet) supported in the Cuda version.\n",mypars->ls_method);
		exit(-1);
	}
	para_printf("    Local-search chosen method is: %s\n", (cData.dockpars.lsearch_rate == 0.0f)? "GA" : method_chosen);

	if((mypars->initial_sw_generations>0) && (strcmp(mypars->ls_method, "sw") != 0))
		para_printf("    Using Solis-Wets (sw) for the first %d generations.\n",mypars->initial_sw_generations);

	// Get profile for timing
	profile.adadelta=(strcmp(mypars->ls_method, "ad")==0);
	profile.n_evals = mypars->num_of_energy_evals;
	profile.num_atoms = myligand_reference.num_of_atoms;
	profile.num_rotbonds = myligand_init->num_of_rotbonds;

	/*
	para_printf("dockpars.num_of_intraE_contributors:%u\n", dockpars.num_of_intraE_contributors);
	para_printf("dockpars.rotbondlist_length:%u\n", dockpars.rotbondlist_length);
	*/

        /*
        DPCT1008:167: clock function is not defined in the SYCL. This is a
        hardware-specific feature. Consult with your hardware vendor to find a
        replacement.
        */
        clock_start_docking = clock();

        SetKernelsGpuData(&cData);

#ifdef DOCK_DEBUG
	para_printf("\n");
	// Main while-loop iterarion counter
	unsigned int ite_cnt = 0;
#endif

	// Kernel1
	uint32_t kernel1_gxsize = blocksPerGridForEachEntity;
	uint32_t kernel1_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	para_printf("%-25s %10s %8lu %10s %4u\n", "K_INIT", "gSize: ", kernel1_gxsize, "lSize: ", kernel1_lxsize); fflush(stdout);
#endif
	// End of Kernel1

	// Kernel2
	uint32_t kernel2_gxsize = blocksPerGridForEachRun;
	uint32_t kernel2_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	para_printf("%-25s %10s %8lu %10s %4u\n", "K_EVAL", "gSize: ", kernel2_gxsize, "lSize: ",  kernel2_lxsize); fflush(stdout);
#endif
	// End of Kernel2

	// Kernel4
	uint32_t kernel4_gxsize = blocksPerGridForEachEntity;
	uint32_t kernel4_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	para_printf("%-25s %10s %8u %10s %4u\n", "K_GA_GENERATION", "gSize: ",  kernel4_gxsize, "lSize: ", kernel4_lxsize); fflush(stdout);
#endif
	// End of Kernel4

	uint32_t kernel3_gxsize = 0;
	uint32_t kernel3_lxsize = threadsPerBlock;
/*
	uint32_t kernel5_gxsize = 0;
	uint32_t kernel5_lxsize = threadsPerBlock;
	uint32_t kernel6_gxsize = 0;
	uint32_t kernel6_lxsize = threadsPerBlock;
*/
	uint32_t kernel7_gxsize = 0;
	uint32_t kernel7_lxsize = threadsPerBlock;
	uint32_t kernel8_gxsize = 0;
	uint32_t kernel8_lxsize = threadsPerBlock;
	if (cData.dockpars.lsearch_rate != 0.0f) {

		if ((strcmp(mypars->ls_method, "sw") == 0) || (mypars->initial_sw_generations>0)) {
			// Kernel3
			kernel3_gxsize = blocksPerGridForEachLSEntity;
			#ifdef DOCK_DEBUG
			para_printf("%-25s %10s %8u %10s %4u\n", "K_LS_SOLISWETS", "gSize: ", kernel3_gxsize, "lSize: ", kernel3_lxsize); fflush(stdout);
			#endif
			// End of Kernel3
		}
/* SD and Fire are not currently supported by the Cuda version
		if (strcmp(mypars->ls_method, "sd") == 0) {
			// Kernel5
			kernel5_gxsize = blocksPerGridForEachGradMinimizerEntity;
			#ifdef DOCK_DEBUG
			para_printf("%-25s %10s %8u %10s %4u\n", "K_LS_GRAD_SDESCENT", "gSize: ", kernel5_gxsize, "lSize: ", kernel5_lxsize); fflush(stdout);
			#endif
			// End of Kernel5
		}
		if (strcmp(mypars->ls_method, "fire") == 0) {
			// Kernel6
			kernel6_gxsize = blocksPerGridForEachGradMinimizerEntity;
			#ifdef DOCK_DEBUG
			para_printf("%-25s %10s %8u %10s %4u\n", "K_LS_GRAD_FIRE", "gSize: ", kernel6_gxsize, "lSize: ", kernel6_lxsize); fflush(stdout);
			#endif
			// End of Kernel6
		}
*/
		if (strcmp(mypars->ls_method, "ad") == 0) {
			// Kernel7
			kernel7_gxsize = blocksPerGridForEachGradMinimizerEntity;
			#ifdef DOCK_DEBUG
			para_printf("%-25s %10s %8u %10s %4u\n", "K_LS_GRAD_ADADELTA", "gSize: ", kernel7_gxsize, "lSize: ", kernel7_lxsize); fflush(stdout);
			#endif
			// End of Kernel7
		}
		if (strcmp(mypars->ls_method, "adam") == 0) {
			// Kernel8
			kernel8_gxsize = blocksPerGridForEachGradMinimizerEntity;
			#ifdef DOCK_DEBUG
			para_printf("%-25s %10s %8u %10s %4u\n", "K_LS_GRAD_ADADELTA", "gSize: ", kernel7_gxsize, "lSize: ", kernel7_lxsize); fflush(stdout);
			#endif
			// End of Kernel8
		}
	} // End if (dockpars.lsearch_rate != 0.0f)

	// Kernel1
	#ifdef DOCK_DEBUG
		para_printf("\nExecution starts:\n\n");
		para_printf("%-25s", "\tK_INIT");fflush(stdout);
		cudaDeviceSynchronize();
	#endif
	gpu_calc_initpop(kernel1_gxsize, kernel1_lxsize, pMem_conformations_current, pMem_energies_current);
	#ifdef DOCK_DEBUG
		cudaDeviceSynchronize();
		para_printf("%15s" ," ... Finished\n");fflush(stdout);
	#endif
	// End of Kernel1

	// Kernel2
	#ifdef DOCK_DEBUG
		para_printf("%-25s", "\tK_EVAL");fflush(stdout);
	#endif
	gpu_sum_evals(kernel2_gxsize, kernel2_lxsize);
	#ifdef DOCK_DEBUG
		cudaDeviceSynchronize();
		para_printf("%15s" ," ... Finished\n");fflush(stdout);
	#endif
	// End of Kernel2
	// ===============================================================================

	#if 0
	generation_cnt = 1;
	#endif
	generation_cnt = 0;
	unsigned long total_evals;

	auto const t2 = std::chrono::steady_clock::now();
	para_printf("\nRest of Setup time %fs\n", elapsed_seconds(t1 ,t2));

	//print progress bar
	AutoStop autostop(mypars->pop_size, mypars->num_of_runs, mypars->stopstd, mypars->as_frequency, output);
#ifndef DOCK_DEBUG
	if (mypars->autostop)
	{
		autostop.print_intro(mypars->num_of_generations, mypars->num_of_energy_evals);
	}
	else
	{
		para_printf("\nExecuting docking runs:\n");
		para_printf("        20%%        40%%       60%%       80%%       100%%\n");
		para_printf("---------+---------+---------+---------+---------+\n");
	}
#endif
	curr_progress_cnt = 0;

#if defined (MAPPED_COPY)
	while ((progress = check_progress(tData.pMem_gpu_evals_of_runs, generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
#else
	while ((progress = check_progress(sim_state.cpu_evals_of_runs.data(), generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs, total_evals)) < 100.0)
#endif
	{
		if (mypars->autostop) {
			if (generation_cnt % mypars->as_frequency == 0) {
/* DPCT_ORIG 				status =
 * cudaMemcpy(sim_state.cpu_energies.data(), pMem_energies_current,
 * size_energies, cudaMemcpyDeviceToHost);*/
                                /*
                                DPCT1003:169: Migrated API does not return error
                                code. (*, 0) is inserted. You may need to
                                rewrite this code.
                                */
                                status =
                                    (dpct::get_default_queue()
                                         .memcpy(sim_state.cpu_energies.data(),
                                                 pMem_energies_current,
                                                 size_energies)
                                         .wait(),
                                     0);
                                /*
                                DPCT1001:168: The statement could not be
                                removed.
                                */
                                RTERROR(status, "cudaMemcpy: couldn't download pMem_energies_current");
                                if (autostop.check_if_satisfactory(generation_cnt, sim_state.cpu_energies.data(), total_evals))
					if (total_evals>min_as_evals)
						break; // Exit loop when all conditions are satisfied
			}
		}
		else
		{
#ifdef DOCK_DEBUG
			ite_cnt++;
			para_printf("\nLGA iteration # %u\n", ite_cnt);
			fflush(stdout);
#endif
			//update progress bar (bar length is 50)
			new_progress_cnt = (int) (progress/2.0+0.5);
			if (new_progress_cnt > 50)
				new_progress_cnt = 50;
			while (curr_progress_cnt < new_progress_cnt) {
				curr_progress_cnt++;
#ifndef DOCK_DEBUG
				para_printf("*");
#endif
				fflush(stdout);
			}
		}
		// Kernel4
		#ifdef DOCK_DEBUG
			para_printf("%-25s", "\tK_GA_GENERATION");fflush(stdout);
		#endif

		//runKernel1D(command_queue,kernel4,kernel4_gxsize,kernel4_lxsize,&time_start_kernel,&time_end_kernel);
		gpu_gen_and_eval_newpops(kernel4_gxsize, kernel4_lxsize, pMem_conformations_current, pMem_energies_current, pMem_conformations_next, pMem_energies_next);
		#ifdef DOCK_DEBUG
			para_printf("%15s", " ... Finished\n");fflush(stdout);
		#endif
		// End of Kernel4
		if (cData.dockpars.lsearch_rate != 0.0f) {
			if ((strcmp(mypars->ls_method, "sw") == 0) || ((strcmp(mypars->ls_method, "ad") == 0) && (generation_cnt<mypars->initial_sw_generations))) {
				// Kernel3
				#ifdef DOCK_DEBUG
					para_printf("%-25s", "\tK_LS_SOLISWETS");fflush(stdout);
				#endif
				//runKernel1D(command_queue,kernel3,kernel3_gxsize,kernel3_lxsize,&time_start_kernel,&time_end_kernel);
				gpu_perform_LS(kernel3_gxsize, kernel3_lxsize, pMem_conformations_next, pMem_energies_next);                
				#ifdef DOCK_DEBUG
					para_printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel3
			} else if (strcmp(mypars->ls_method, "sd") == 0) {
				// Kernel5
				#ifdef DOCK_DEBUG
					para_printf("%-25s", "\tK_LS_GRAD_SDESCENT");fflush(stdout);
				#endif
				//runKernel1D(command_queue,kernel5,kernel5_gxsize,kernel5_lxsize,&time_start_kernel,&time_end_kernel);
				#ifdef DOCK_DEBUG
					para_printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel5
			} else if (strcmp(mypars->ls_method, "fire") == 0) {
				// Kernel6
				#ifdef DOCK_DEBUG
					para_printf("%-25s", "\tK_LS_GRAD_FIRE");fflush(stdout);
				#endif
				//runKernel1D(command_queue,kernel6,kernel6_gxsize,kernel6_lxsize,&time_start_kernel,&time_end_kernel);
				#ifdef DOCK_DEBUG
					para_printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel6
			} else if (strcmp(mypars->ls_method, "ad") == 0) {
				// Kernel7
				#ifdef DOCK_DEBUG
					para_printf("%-25s", "\tK_LS_GRAD_ADADELTA");fflush(stdout);
				#endif
				// runKernel1D(command_queue,kernel7,kernel7_gxsize,kernel7_lxsize,&time_start_kernel,&time_end_kernel);
				gpu_gradient_minAD(kernel7_gxsize, kernel7_lxsize, pMem_conformations_next, pMem_energies_next);
				#ifdef DOCK_DEBUG
					para_printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel7
			} else if (strcmp(mypars->ls_method, "adam") == 0) {
				// Kernel8
				#ifdef DOCK_DEBUG
					para_printf("%-25s", "\tK_LS_GRAD_ADAM");fflush(stdout);
				#endif
				// runKernel1D(command_queue,kernel8,kernel8_gxsize,kernel8_lxsize,&time_start_kernel,&time_end_kernel);
				gpu_gradient_minAdam(kernel8_gxsize, kernel8_lxsize, pMem_conformations_next, pMem_energies_next);
				#ifdef DOCK_DEBUG
					para_printf("%15s" ," ... Finished\n");fflush(stdout);
				#endif
				// End of Kernel8
			}
		} // End if (dockpars.lsearch_rate != 0.0f)
		// -------- Replacing with memory maps! ------------
		// -------- Replacing with memory maps! ------------
		// Kernel2
		#ifdef DOCK_DEBUG
			para_printf("%-25s", "\tK_EVAL");fflush(stdout);
		#endif
		//runKernel1D(command_queue,kernel2,kernel2_gxsize,kernel2_lxsize,&time_start_kernel,&time_end_kernel);
		gpu_sum_evals(kernel2_gxsize, kernel2_lxsize);

		#ifdef DOCK_DEBUG
			para_printf("%15s" ," ... Finished\n");fflush(stdout);
		#endif
		// End of Kernel2
		// ===============================================================================
#if not defined (MAPPED_COPY)
		cudaMemcpy(sim_state.cpu_evals_of_runs.data(), tData.pMem_gpu_evals_of_runs, size_evals_of_runs, cudaMemcpyDeviceToHost);
#endif
		generation_cnt++;
		// ----------------------------------------------------------------------
		// ORIGINAL APPROACH: switching conformation and energy pointers (Probably the best approach, restored)
		// CURRENT APPROACH:  copy data from one buffer to another, pointers are kept the same
		// IMPROVED CURRENT APPROACH
		// Kernel arguments are changed on every iteration
		// No copy from dev glob memory to dev glob memory occurs
		// Use generation_cnt as it evolves with the main loop
		// No need to use tempfloat
		// No performance improvement wrt to "CURRENT APPROACH"

		// Kernel args exchange regions they point to
		// But never two args point to the same region of dev memory
		// NO ALIASING -> use restrict in Kernel

		// Flip conformation and energy pointers
		float* pTemp;
		pTemp = pMem_conformations_current;
		pMem_conformations_current = pMem_conformations_next;
		pMem_conformations_next = pTemp;
		pTemp = pMem_energies_current;
		pMem_energies_current = pMem_energies_next;
		pMem_energies_next = pTemp;

		// ----------------------------------------------------------------------
		#ifdef DOCK_DEBUG
			para_printf("\tProgress %.3f %%\n", progress);
			fflush(stdout);
		#endif
	} // End of while-loop

	// Profiler
	profile.nev_at_stop = total_evals/mypars->num_of_runs;
	profile.autostopped = autostop.did_stop();

        /*
        DPCT1008:170: clock function is not defined in the SYCL. This is a
        hardware-specific feature. Consult with your hardware vendor to find a
        replacement.
        */
        clock_stop_docking = clock();
        if (mypars->autostop==0)
	{
		//update progress bar (bar length is 50)mem_num_of_rotatingatoms_per_rotbond_const
		while (curr_progress_cnt < 50) {
			curr_progress_cnt++;
			para_printf("*");
			fflush(stdout);
		}
		para_printf("\n");
	}

	auto const t3 = std::chrono::steady_clock::now();

	// ===============================================================================
	// Modification based on:
	// http://www.cc.gatech.edu/~vetter/keeneland/tutorial-2012-02-20/08-opencl.pdf
	// ===============================================================================
	//processing results
/* DPCT_ORIG 	status = cudaMemcpy(cpu_final_populations,
 * pMem_conformations_current, size_populations, cudaMemcpyDeviceToHost);*/
        /*
        DPCT1003:171: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(cpu_final_populations, pMem_conformations_current,
                              size_populations)
                      .wait(),
                  0);
        /*
        DPCT1001:152: The statement could not be removed.
        */
        RTERROR(
            status,
            "cudaMemcpy: couldn't copy pMem_conformations_current to host.\n");
/* DPCT_ORIG 	status = cudaMemcpy(sim_state.cpu_energies.data(),
 * pMem_energies_current, size_energies, cudaMemcpyDeviceToHost);*/
        /*
        DPCT1003:172: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        status = (dpct::get_default_queue()
                      .memcpy(sim_state.cpu_energies.data(),
                              pMem_energies_current, size_energies)
                      .wait(),
                  0);
        /*
        DPCT1001:153: The statement could not be removed.
        */
        RTERROR(status, "cudaMemcpy: couldn't copy pMem_energies_current to host.\n");

        // Final autostop statistics output
	if (mypars->autostop) autostop.output_final_stddev(generation_cnt, sim_state.cpu_energies.data(), total_evals);
	para_printf("\nDocking time %fs\n", elapsed_seconds(t2, t3));
#if defined (DOCK_DEBUG)
	for (int cnt_pop=0;cnt_pop<size_populations/sizeof(float);cnt_pop++)
		para_printf("total_num_pop: %u, cpu_final_populations[%u]: %f\n",(unsigned int)(size_populations/sizeof(float)),cnt_pop,cpu_final_populations[cnt_pop]);
	for (int cnt_pop=0;cnt_pop<size_energies/sizeof(float);cnt_pop++)
		para_printf("total_num_energies: %u, cpu_energies[%u]: %f\n",    (unsigned int)(size_energies/sizeof(float)),cnt_pop,sim_state.cpu_energies[cnt_pop]);
#endif

	// Assign simulation results to sim_state
	sim_state.myligand_reference = myligand_reference;
	sim_state.generation_cnt = generation_cnt;
	sim_state.sec_per_run = ELAPSEDSECS(clock_stop_docking, clock_start_docking)/mypars->num_of_runs;
	sim_state.total_evals = total_evals;

	free(cpu_prng_seeds);

	delete KerConst_interintra;
	delete KerConst_intracontrib;
	delete KerConst_intra;
	delete KerConst_rotlist;
	delete KerConst_conform;
	delete KerConst_grads;

	auto const t4 = std::chrono::steady_clock::now();
	para_printf("\nShutdown time %fs\n", elapsed_seconds(t3, t4));
	if(output!=NULL) free(outbuf);

	return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

double check_progress(
                      int* evals_of_runs,
                      int generation_cnt,
                      int max_num_of_evals,
                      int max_num_of_gens,
                      int num_of_runs,
                      unsigned long& total_evals
                     )
// The function checks if the stop condition of the docking is satisfied, returns 0 if no, and returns 1 if yes. The fitst
// parameter points to the array which stores the number of evaluations performed for each run. The second parameter stores
// the generations used. The other parameters describe the maximum number of energy evaluations, the maximum number of
// generations, and the number of runs, respectively. The stop condition is satisfied, if the generations used is higher
// than the maximal value, or if the average number of evaluations used is higher than the maximal value.
{
	// Stops if the sum of evals of every run reached the sum of the total number of evals

	int i;
	double evals_progress;
	double gens_progress;

	// calculating progress according to number of runs
	total_evals = 0;
	for (i=0; i<num_of_runs; i++)
		total_evals += evals_of_runs[i];

	evals_progress = (double)total_evals/((double) num_of_runs)/max_num_of_evals*100.0;

	// calculating progress according to number of generations
	gens_progress = ((double) generation_cnt)/((double) max_num_of_gens)*100.0; //std::cout<< "gens_progress: " << gens_progress <<std::endl;

	if (evals_progress > gens_progress)
		return evals_progress;
	else
		return gens_progress;
}
