//
// Created by max on 04-10-21.
//

#include "GPU_helpers.h"

#define CUDA_CHECK_RETURN(value) {										\
	cudaError_t _m_cudaStat = value;									\
	if (_m_cudaStat != cudaSuccess) {									\
		fprintf(stderr, " CUDA Error %s at line %d in file %s\n",	    \
             cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
        printf("CUDA Error %s at line %d in file %s\n",		            \
			 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		if(value == 2) exit(2);                                         \
		exit(1);														\
	} }

bool test_GPU(LogContext &logC){
    INIT_LOG(&logC.log_file,logC.mpi_rank);
    int Amount_of_GPUS_detected = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&Amount_of_GPUS_detected));
    log(LOG_INFO) << "amount of gpus detected: " << Amount_of_GPUS_detected << LOG_ENDL;
    cudaDeviceProp prop{};
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, 0));
    log(LOG_INFO) << "      PCI device id:                " << prop.pciBusID << LOG_ENDL;
    log(LOG_INFO) << "      Device name:                  " << prop.name << LOG_ENDL;
    log(LOG_INFO) << "      Clock Rate (KHz):             " << prop.clockRate << LOG_ENDL;
    log(LOG_INFO) << "      Memory Clock Rate (KHz):      " << prop.memoryClockRate << LOG_ENDL;
    log(LOG_INFO) << "      Memory Bus Width (bits):      " << prop.memoryBusWidth << LOG_ENDL;
    log(LOG_INFO) << "      Peak Memory Bandwidth (GB/s): " << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << LOG_ENDL;
    log(LOG_INFO) << "      Total global memory (Gbytes): " << (prop.totalGlobalMem / 1000000000) << LOG_ENDL;
    log(LOG_INFO) << "      Compute cabability :          " << prop.major << "." << prop.minor << LOG_ENDL;
    log(LOG_INFO) << "      Number of multiprocessors :   " << prop.multiProcessorCount << LOG_ENDL;
    return Amount_of_GPUS_detected;
}
