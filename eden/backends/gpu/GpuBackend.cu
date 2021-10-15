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

    // alloc simple
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_constants, tabs.global_constants.size()*sizeof(tabs.global_constants[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_const_f32_index, tabs.global_const_f32_index.size()*sizeof(tabs.global_const_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_const_f32_index, tabs.global_table_const_f32_index.size()*sizeof(tabs.global_table_const_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_const_i64_index, tabs.global_table_const_i64_index.size()*sizeof(tabs.global_table_const_i64_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_state_f32_index, tabs.global_table_state_f32_index.size()*sizeof(tabs.global_table_state_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_table_state_i64_index, tabs.global_table_state_i64_index.size()*sizeof(tabs.global_table_state_i64_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_state_f32_index, tabs.global_state_f32_index.size()*sizeof(tabs.global_state_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_state_now, state->state_one.size()*sizeof(state->state_one[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_state_next, state->state_two.size()*sizeof(state->state_two[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_f32_sizes, state->global_tables_const_f32_sizes.size()*sizeof(state->global_tables_const_f32_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_i64_sizes, state->global_tables_const_i64_sizes.size()*sizeof(state->global_tables_const_i64_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_state_f32_sizes, state->global_tables_state_f32_sizes.size()*sizeof(state->global_tables_state_f32_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_state_i64_sizes, state->global_tables_state_i64_sizes.size()*sizeof(state->global_tables_state_i64_sizes[0])));

    // copy simple
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_constants,                tabs.global_constants.data(),                   tabs.global_constants.size()*sizeof(tabs.global_constants[0]),                                  cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_const_f32_index,          tabs.global_const_f32_index.data(),             tabs.global_const_f32_index.size()*sizeof(tabs.global_const_f32_index[0]),                      cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_const_f32_index,    tabs.global_table_const_f32_index.data(),       tabs.global_table_const_f32_index.size()*sizeof(tabs.global_table_const_f32_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_const_i64_index,    tabs.global_table_const_i64_index.data(),       tabs.global_table_const_i64_index.size()*sizeof(tabs.global_table_const_i64_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_state_f32_index,    tabs.global_table_state_f32_index.data(),       tabs.global_table_state_f32_index.size()*sizeof(tabs.global_table_state_f32_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_table_state_i64_index,    tabs.global_table_state_i64_index.data(),       tabs.global_table_state_i64_index.size()*sizeof(tabs.global_table_state_i64_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_state_f32_index,          tabs.global_state_f32_index.data(),             tabs.global_state_f32_index.size()*sizeof(tabs.global_state_f32_index[0]),                      cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_state_now,                state->state_one.data(),                        state->state_one.size()*sizeof(state->state_one[0]),                                            cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_state_next,               state->state_two.data(),                        state->state_two.size()*sizeof(state->state_two[0]),                                            cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_f32_sizes,   state->global_tables_const_f32_sizes.data(),    state->global_tables_const_f32_sizes.size()*sizeof(state->global_tables_const_f32_sizes[0]),    cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_i64_sizes,   state->global_tables_const_i64_sizes.data(),    state->global_tables_const_i64_sizes.size()*sizeof(state->global_tables_const_i64_sizes[0]),    cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_state_f32_sizes,   state->global_tables_state_f32_sizes.data(),    state->global_tables_state_f32_sizes.size()*sizeof(state->global_tables_state_f32_sizes[0]),    cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_state_i64_sizes,   state->global_tables_state_i64_sizes.data(),    state->global_tables_state_i64_sizes.size()*sizeof(state->global_tables_state_i64_sizes[0]),    cudaMemcpyHostToDevice));

    /* double pointers */

    std::vector<float*> temp_f32;

    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNow_f32, state->global_tables_stateOne_f32_arrays.size()*sizeof(state->global_tables_stateOne_f32_arrays[0]))); // state->global_tables_state_f32_sizes.data()
    for (size_t i = 0; i < state->global_tables_stateOne_f32_arrays.size(); i++) {
        size_t size = state->global_tables_state_f32_sizes[i];
        float * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateOne_f32_arrays[i], size*sizeof(float), cudaMemcpyHostToDevice));
        temp_f32.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNow_f32, temp_f32.data(), temp_f32.size()*sizeof(float*), cudaMemcpyHostToDevice));
    temp_f32.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNext_f32, state->global_tables_stateTwo_f32_arrays.size()*sizeof(state->global_tables_stateTwo_f32_arrays[0]))); // state->global_tables_state_f32_sizes.data()
    for (size_t i = 0; i < state->global_tables_stateTwo_f32_arrays.size(); i++) {
        size_t size = state->global_tables_state_f32_sizes[i];
        float * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateTwo_f32_arrays[i], size*sizeof(float), cudaMemcpyHostToDevice));
        temp_f32.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNext_f32, temp_f32.data(), temp_f32.size()*sizeof(float*), cudaMemcpyHostToDevice));
    temp_f32.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_f32_arrays, state->global_tables_const_f32_arrays.size()*sizeof(state->global_tables_const_f32_arrays[0]))); // state->global_tables_const_f32_sizes.data()
    for (size_t i = 0; i < state->global_tables_const_f32_arrays.size(); i++) {
        size_t size = state->global_tables_const_f32_sizes[i];
        float * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_const_f32_arrays[i], size*sizeof(float), cudaMemcpyHostToDevice));
        temp_f32.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_f32_arrays, temp_f32.data(), temp_f32.size()*sizeof(float*), cudaMemcpyHostToDevice));
    temp_f32.clear();

    std::vector<long long*> temp_i64;
    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNow_i64, state->global_tables_stateOne_i64_arrays.size()*sizeof(state->global_tables_stateOne_i64_arrays[0]))); // state->global_tables_state_i64_sizes.data(),
    for (size_t i = 0; i < state->global_tables_stateOne_i64_arrays.size(); i++) {
        size_t size = state->global_tables_state_i64_sizes[i];
        long long * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(long long)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateOne_i64_arrays[i], size*sizeof(long long), cudaMemcpyHostToDevice));
        temp_i64.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNow_i64, temp_i64.data(), temp_i64.size()*sizeof(long long*), cudaMemcpyHostToDevice));
    temp_i64.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_stateNext_i64, state->global_tables_stateTwo_i64_arrays.size()*sizeof(state->global_tables_stateTwo_i64_arrays[0]))); // state->global_tables_state_i64_sizes.data(),
    for (size_t i = 0; i < state->global_tables_stateTwo_i64_arrays.size(); i++) {
        size_t size = state->global_tables_state_i64_sizes[i];
        long long * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(long long)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateTwo_i64_arrays[i], size*sizeof(long long), cudaMemcpyHostToDevice));
        temp_i64.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_stateNext_i64, temp_i64.data(), temp_i64.size()*sizeof(long long*), cudaMemcpyHostToDevice));
    temp_i64.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_global_tables_const_i64_arrays, state->global_tables_const_i64_arrays.size()*sizeof(state->global_tables_const_i64_arrays[0]))); // state->global_tables_state_i64_sizes.data()
    for (size_t i = 0; i < state->global_tables_const_i64_arrays.size(); i++) {
        size_t size = state->global_tables_const_i64_sizes[i];
        printf("Allocating subarray %d of size %lld\n", i, (long long)size);
        long long * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(long long)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_const_i64_arrays[i], size*sizeof(long long), cudaMemcpyHostToDevice));
        temp_i64.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_global_tables_const_i64_arrays, temp_i64.data(), temp_i64.size()*sizeof(long long*), cudaMemcpyHostToDevice));
    temp_i64.clear();

    return true;
}

float * GpuBackend::global_state_now() const {
    CUDA_CHECK_RETURN(cudaMemcpy(
                state->state_one.data(),
                m_global_state_now,
                state->state_one.size()*sizeof(state->state_one[0]),
                cudaMemcpyDeviceToHost));
    return state->state_one.data();
}

Table_F32 * GpuBackend::global_tables_stateNow_f32 () const {
    // here be dragons
    // XXX TODO: remove temp allocation - we can just keep the temp_f32 vector from allocation
    // XXX TODO: call this function only when using MPI
    // Also, state->global_tables_stateOne_f32_arrays points to state->state_one in some way
    // leading to overwrites in certain cases, but not others (?)
    std::vector<float*> temp(state->global_tables_stateTwo_f32_arrays.size(), 0);
    CUDA_CHECK_RETURN(cudaMemcpy(
                temp.data(),
                m_global_tables_stateNow_f32,
                temp.size()*sizeof(float*),
                cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < temp.size(); i++) {
        size_t size = state->global_tables_state_f32_sizes[i];
        CUDA_CHECK_RETURN(cudaMemcpy(
                    state->global_tables_stateTwo_f32_arrays[i],
                    temp[i],
                    size*sizeof(float),
                    cudaMemcpyDeviceToHost));
    }
    return state->global_tables_stateTwo_f32_arrays.data();
}
