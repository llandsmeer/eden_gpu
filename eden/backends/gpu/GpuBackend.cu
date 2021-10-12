//
// Created by max on 12-10-21.
//

#include "GpuBackend.h"

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


void GpuBackend::execute_work_gpu(EngineConfig &engine_config, SimulatorConfig &config, int step, double time) {
    const float dt = engine_config.dt;

    for (size_t idx = 0; idx < tabs.consecutive_kernels.size(); idx++) {

        if(config.debug){
            printf("consecutive items %lld start\n", (long long)idx);
            // if(my_mpi.rank != 0) continue;
            // continue;
            fflush(stdout);
        }

        RawTables::ConsecutiveIterationCallbacks & cic = tabs.consecutive_kernels.at(idx);
        ((GPUIterationCallback)cic.callback) (
                          cic.start_item,
                          cic.n_items,
                          time,
                          dt,
                          m_global_constants,
                          m_global_const_f32_index,
                          m_global_tables_const_f32_sizes,
                          m_global_tables_const_f32_arrays,
                          m_global_table_const_f32_index,
                          m_global_tables_const_i64_sizes,
                          m_global_tables_const_i64_arrays,
                          m_global_table_const_i64_index,
                          m_global_tables_state_f32_sizes,
                          m_global_tables_stateNow_f32,
                          m_global_tables_stateNext_f32,
                          m_global_table_state_f32_index,
                          m_global_tables_state_i64_sizes,
                          m_global_tables_stateNow_i64,
                          m_global_tables_stateNext_i64,
                          m_global_table_state_i64_index,
                          m_global_state_now,
                          m_global_state_next,
                          m_global_state_f32_index,
                          step
            );

        if(config.debug){
            printf("consecutive items %lld end\n", (long long)idx);
            fflush(stdout);
        }
    }

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return;
}

bool GpuBackend::copy_data_to_device() {
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_constants,                       tabs.global_constants.size()*sizeof(tabs.global_constants[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_const_f32_index,                 tabs.global_constants.size() * sizeof(tabs.global_constants[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_const_f32_index,           tabs.global_const_f32_index.size() * sizeof(tabs.global_const_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_const_i64_index,           tabs.global_table_const_f32_index.size() * sizeof(tabs.global_table_const_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_state_f32_index,           tabs.global_table_const_i64_index.size() * sizeof(tabs.global_table_const_i64_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_state_i64_index,           tabs.global_table_state_f32_index.size() * sizeof(tabs.global_table_state_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_state_f32_index,                 tabs.global_table_state_i64_index.size() * sizeof(tabs.global_table_state_i64_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_state_now,                       tabs.global_state_f32_index.size() * sizeof(tabs.global_state_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_state_next,                      state->state_one.size() * sizeof(state->state_one[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNow_f32,             state->state_two.size() * sizeof(state->state_two[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNow_i64,             state->global_tables_stateOne_f32_arrays.size() * sizeof(state->global_tables_stateOne_f32_arrays[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNext_f32,            state->global_tables_stateOne_i64_arrays.size() * sizeof(state->global_tables_stateOne_i64_arrays[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNext_i64,            state->global_tables_stateTwo_f32_arrays.size() * sizeof(state->global_tables_stateTwo_f32_arrays[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_f32_arrays,         state->global_tables_stateTwo_i64_arrays.size() * sizeof(state->global_tables_stateTwo_i64_arrays[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_i64_arrays,         state->global_tables_const_f32_arrays.size() * sizeof(state->global_tables_const_f32_arrays[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_f32_sizes,          state->global_tables_const_i64_arrays.size() * sizeof(state->global_tables_const_i64_arrays[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_i64_sizes,          state->global_tables_const_f32_sizes.size() * sizeof(state->global_tables_const_f32_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_state_f32_sizes,          state->global_tables_const_i64_sizes.size() * sizeof(state->global_tables_const_i64_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_state_i64_sizes,          state->global_tables_state_f32_sizes.size() * sizeof(state->global_tables_state_f32_sizes[0])));

    CUDA_CHECK_RETURN(cudaMemcpy(m_global_constants                 ,  tabs.global_constants.data()                     ,tabs.global_constants.size()*sizeof(tabs.global_constants[0])                                         ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_const_f32_index           ,  tabs.global_const_f32_index.data()               ,tabs.global_constants.size() * sizeof(tabs.global_constants[0])                                       ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_const_f32_index     ,  tabs.global_table_const_f32_index.data()         ,tabs.global_const_f32_index.size() * sizeof(tabs.global_const_f32_index[0])                           ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_const_i64_index     ,  tabs.global_table_const_i64_index.data()         ,tabs.global_table_const_f32_index.size() * sizeof(tabs.global_table_const_f32_index[0])               ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_state_f32_index     ,  tabs.global_table_state_f32_index.data()         ,tabs.global_table_const_i64_index.size() * sizeof(tabs.global_table_const_i64_index[0])               ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_state_i64_index     ,  tabs.global_table_state_i64_index.data()         ,tabs.global_table_state_f32_index.size() * sizeof(tabs.global_table_state_f32_index[0])               ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_state_f32_index           ,  tabs.global_state_f32_index.data()               ,tabs.global_table_state_i64_index.size() * sizeof(tabs.global_table_state_i64_index[0])               ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_state_now                 ,  state->state_one.data()                          ,tabs.global_state_f32_index.size() * sizeof(tabs.global_state_f32_index[0])                           ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_state_next                ,  state->state_two.data()                          ,state->state_one.size() * sizeof(state->state_one[0])                                                 ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNow_f32       ,  state->global_tables_stateOne_f32_arrays.data()  ,state->state_two.size() * sizeof(state->state_two[0])                                                 ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNow_i64       ,  state->global_tables_stateOne_i64_arrays.data()  ,state->global_tables_stateOne_f32_arrays.size() * sizeof(state->global_tables_stateOne_f32_arrays[0]) ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNext_f32      ,  state->global_tables_stateTwo_f32_arrays.data()  ,state->global_tables_stateOne_i64_arrays.size() * sizeof(state->global_tables_stateOne_i64_arrays[0]) ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNext_i64      ,  state->global_tables_stateTwo_i64_arrays.data()  ,state->global_tables_stateTwo_f32_arrays.size() * sizeof(state->global_tables_stateTwo_f32_arrays[0]) ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_f32_arrays   ,  state->global_tables_const_f32_arrays.data()     ,state->global_tables_stateTwo_i64_arrays.size() * sizeof(state->global_tables_stateTwo_i64_arrays[0]) ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_i64_arrays   ,  state->global_tables_const_i64_arrays.data()     ,state->global_tables_const_f32_arrays.size() * sizeof(state->global_tables_const_f32_arrays[0])       ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_f32_sizes    ,  state->global_tables_const_f32_sizes.data()      ,state->global_tables_const_i64_arrays.size() * sizeof(state->global_tables_const_i64_arrays[0])       ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_i64_sizes    ,  state->global_tables_const_i64_sizes.data()      ,state->global_tables_const_f32_sizes.size() * sizeof(state->global_tables_const_f32_sizes[0])         ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_state_f32_sizes    ,  state->global_tables_state_f32_sizes.data()      ,state->global_tables_const_i64_sizes.size() * sizeof(state->global_tables_const_i64_sizes[0])         ,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_state_i64_sizes    ,  state->global_tables_state_i64_sizes.data()      ,state->global_tables_state_f32_sizes.size() * sizeof(state->global_tables_state_f32_sizes[0])         ,cudaMemcpyHostToDevice));
    return true;
}