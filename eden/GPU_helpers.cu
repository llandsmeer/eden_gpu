//
// Created by max on 04-10-21.
//

#include <stdio.h>
#include <iostream>

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


bool GPU_checker(){
    int Amount_of_GPUS_detected = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&Amount_of_GPUS_detected));
    std::cout << "amount of gpus detected: " << Amount_of_GPUS_detected << std::endl;
    //todo
    cudaDeviceProp prop;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, 0));
    std::cout << "      PCI device id:                " << prop.pciBusID << "\n";
    std::cout << "      Device name:                  " << prop.name << "\n";
    std::cout << "      Clock Rate (KHz):             " << prop.clockRate << "\n";
    std::cout << "      Memory Clock Rate (KHz):      " << prop.memoryClockRate << "\n";
    std::cout << "      Memory Bus Width (bits):      " << prop.memoryBusWidth << "\n";
    std::cout << "      Peak Memory Bandwidth (GB/s): " << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << "\n";
    std::cout << "      Total global memory (Gbytes): " << (prop.totalGlobalMem / 1000000000) << "\n";
    std::cout << "      Compute cabability :          " << prop.major << "." << prop.minor << "\n";
    std::cout << "      Number of multiprocessors :   " << prop.multiProcessorCount << " \n";
    return true;
}

void test(){
    GPU_checker();
}
