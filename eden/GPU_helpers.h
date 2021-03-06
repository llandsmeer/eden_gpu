//
// Created by max on 04-10-21.
//

#ifndef EDEN_GPU_GPU_helpers_H
#define EDEN_GPU_GPU_helpers_H

#include "EngineConfig.h"

#ifdef USE_GPU
void test();
#endif

void setup_gpu(EngineConfig &engine_config){
    if(engine_config.backend == backend_kind_gpu){
        printf("GPU backend selected, checking if it's available\n");
    }
#ifdef USE_GPU
    test();
#else
    printf("No GPU support");
#endif
}

#endif //EDEN_GPU_GPU_helpers_H
