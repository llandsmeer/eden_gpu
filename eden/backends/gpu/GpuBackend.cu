//
// Created by max on 12-10-21.
//

#include "GpuBackend.h"

//change the iteration Callback to also contain some GPU specifics.
extern "C" {
typedef void ( *GPUIterationCallback )(
        long long start, long long n_items,
        double time, float dt, const float *__restrict__ global_constants, const long long *__restrict__ /*XXX*/ global_const_f32_index,
        const long long *__restrict__ global_const_table_f32_sizes, const Table_F32 *__restrict__ global_const_table_f32_arrays, long long *__restrict__ /*XXX*/ global_table_const_f32_index,
        const long long *__restrict__ global_const_table_i64_sizes, const Table_I64 *__restrict__ global_const_table_i64_arrays, long long *__restrict__ /*XXX*/ global_table_const_i64_index,
        const long long *__restrict__ global_state_table_f32_sizes, const Table_F32 *__restrict__ global_state_table_f32_arrays, Table_F32 *__restrict__ global_stateNext_table_f32_arrays,
        long long *__restrict__ /*XXX*/ global_table_state_f32_index,
        const long long *__restrict__ global_state_table_i64_sizes, Table_I64 *__restrict__ global_state_table_i64_arrays, Table_I64 *__restrict__ global_stateNext_table_i64_arrays,
        long long *__restrict__ /*XXX*/ global_table_state_i64_index,
        const float *__restrict__ global_state, float *__restrict__ global_stateNext, long long *__restrict__ global_state_f32_index,
        long long step, int threads_per_block, cudaStream_t *streams_calculate);
}

//Todo find a way to pass the stream to the calculate kernel
cudaStream_t streams_copy;
cudaStream_t streams_calculate;

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

void GpuBackend::synchronize_gpu() {
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
void GpuBackend::gpu_init(){

    //create the Statebuffers
    state = new StateBuffers(tabs);

    //create the copy back host pointers
    m_host_state_now               = state->state_one.data();
    m_host_state_next              = state->state_two.data();
    m_host_tables_stateNow_f32     = state->global_tables_stateOne_f32_arrays.data();
    m_host_tables_stateNow_i64     = state->global_tables_stateOne_i64_arrays.data();
    m_host_tables_stateNext_f32    = state->global_tables_stateTwo_f32_arrays.data();
    m_host_tables_stateNext_i64    = state->global_tables_stateTwo_i64_arrays.data();
    m_host_tables_state_f32_sizes  = state->global_tables_state_f32_sizes.data();
    m_host_tables_state_i64_sizes  = state->global_tables_state_i64_sizes.data();

    m_print_state_now              = state->state_print.data();
    m_print_tables_stateNow_f32    = state->global_tables_statePrint_f32_arrays.data();

    //Create the Streams
    CUDA_CHECK_RETURN(cudaStreamCreate(&streams_copy));
    CUDA_CHECK_RETURN(cudaStreamCreate(&streams_calculate));

    // alloc simple
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_constants, tabs.global_constants.size()*sizeof(tabs.global_constants[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_const_f32_index, tabs.global_const_f32_index.size()*sizeof(tabs.global_const_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_table_const_f32_index, tabs.global_table_const_f32_index.size()*sizeof(tabs.global_table_const_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_table_const_i64_index, tabs.global_table_const_i64_index.size()*sizeof(tabs.global_table_const_i64_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_table_state_f32_index, tabs.global_table_state_f32_index.size()*sizeof(tabs.global_table_state_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_table_state_i64_index, tabs.global_table_state_i64_index.size()*sizeof(tabs.global_table_state_i64_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_state_f32_index, tabs.global_state_f32_index.size()*sizeof(tabs.global_state_f32_index[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_state_now, state->state_one.size()*sizeof(state->state_one[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_state_next, state->state_two.size()*sizeof(state->state_two[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_const_f32_sizes, state->global_tables_const_f32_sizes.size()*sizeof(state->global_tables_const_f32_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_const_i64_sizes, state->global_tables_const_i64_sizes.size()*sizeof(state->global_tables_const_i64_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_state_f32_sizes, state->global_tables_state_f32_sizes.size()*sizeof(state->global_tables_state_f32_sizes[0])));
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_state_i64_sizes, state->global_tables_state_i64_sizes.size()*sizeof(state->global_tables_state_i64_sizes[0])));

    // copy simple
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_constants,                tabs.global_constants.data(),                   tabs.global_constants.size()*sizeof(tabs.global_constants[0]),                                  cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_const_f32_index,          tabs.global_const_f32_index.data(),             tabs.global_const_f32_index.size()*sizeof(tabs.global_const_f32_index[0]),                      cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_table_const_f32_index,    tabs.global_table_const_f32_index.data(),       tabs.global_table_const_f32_index.size()*sizeof(tabs.global_table_const_f32_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_table_const_i64_index,    tabs.global_table_const_i64_index.data(),       tabs.global_table_const_i64_index.size()*sizeof(tabs.global_table_const_i64_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_table_state_f32_index,    tabs.global_table_state_f32_index.data(),       tabs.global_table_state_f32_index.size()*sizeof(tabs.global_table_state_f32_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_table_state_i64_index,    tabs.global_table_state_i64_index.data(),       tabs.global_table_state_i64_index.size()*sizeof(tabs.global_table_state_i64_index[0]),          cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_state_f32_index,          tabs.global_state_f32_index.data(),             tabs.global_state_f32_index.size()*sizeof(tabs.global_state_f32_index[0]),                      cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_state_now,                state->state_one.data(),                        state->state_one.size()*sizeof(state->state_one[0]),                                            cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_state_next,               state->state_two.data(),                        state->state_two.size()*sizeof(state->state_two[0]),                                            cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_const_f32_sizes,   state->global_tables_const_f32_sizes.data(),    state->global_tables_const_f32_sizes.size()*sizeof(state->global_tables_const_f32_sizes[0]),    cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_const_i64_sizes,   state->global_tables_const_i64_sizes.data(),    state->global_tables_const_i64_sizes.size()*sizeof(state->global_tables_const_i64_sizes[0]),    cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_state_f32_sizes,   state->global_tables_state_f32_sizes.data(),    state->global_tables_state_f32_sizes.size()*sizeof(state->global_tables_state_f32_sizes[0]),    cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_state_i64_sizes,   state->global_tables_state_i64_sizes.data(),    state->global_tables_state_i64_sizes.size()*sizeof(state->global_tables_state_i64_sizes[0]),    cudaMemcpyHostToDevice));

    /* double pointers */
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_stateNow_f32, state->global_tables_stateOne_f32_arrays.size()*sizeof(state->global_tables_stateOne_f32_arrays[0]))); // state->global_tables_state_f32_sizes.data()
    std::vector<float*> temp_f32;
    for (size_t i = 0; i < state->global_tables_stateOne_f32_arrays.size(); i++) {
        size_t size = state->global_tables_state_f32_sizes[i];
        float * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateOne_f32_arrays[i], size*sizeof(float), cudaMemcpyHostToDevice));
        temp_f32.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_stateNow_f32, temp_f32.data(), temp_f32.size()*sizeof(float*), cudaMemcpyHostToDevice));
    temp_f32.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_stateNext_f32, state->global_tables_stateOne_f32_arrays.size()*sizeof(state->global_tables_stateOne_f32_arrays[0]))); // state->global_tables_state_f32_sizes.data()
    for (size_t i = 0; i < state->global_tables_stateOne_f32_arrays.size(); i++) {
        size_t size = state->global_tables_state_f32_sizes[i];
        float * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateOne_f32_arrays[i], size*sizeof(float), cudaMemcpyHostToDevice));
        temp_f32.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_stateNext_f32, temp_f32.data(), temp_f32.size()*sizeof(float*), cudaMemcpyHostToDevice));
    temp_f32.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_const_f32_arrays, state->global_tables_const_f32_arrays.size()*sizeof(state->global_tables_const_f32_arrays[0]))); // state->global_tables_const_f32_sizes.data()
    for (size_t i = 0; i < state->global_tables_const_f32_arrays.size(); i++) {
        size_t size = state->global_tables_const_f32_sizes[i];
        float * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_const_f32_arrays[i], size*sizeof(float), cudaMemcpyHostToDevice));
        temp_f32.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_const_f32_arrays, temp_f32.data(), temp_f32.size()*sizeof(float*), cudaMemcpyHostToDevice));
    temp_f32.clear();

    std::vector<long long*> temp_i64;
    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_stateNow_i64, state->global_tables_stateOne_i64_arrays.size()*sizeof(state->global_tables_stateOne_i64_arrays[0]))); // state->global_tables_state_i64_sizes.data(),
    for (size_t i = 0; i < state->global_tables_stateOne_i64_arrays.size(); i++) {
        size_t size = state->global_tables_state_i64_sizes[i];
        long long * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(long long)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateOne_i64_arrays[i], size*sizeof(long long), cudaMemcpyHostToDevice));
        temp_i64.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_stateNow_i64, temp_i64.data(), temp_i64.size()*sizeof(long long*), cudaMemcpyHostToDevice));
    temp_i64.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_stateNext_i64, state->global_tables_stateTwo_i64_arrays.size()*sizeof(state->global_tables_stateTwo_i64_arrays[0]))); // state->global_tables_state_i64_sizes.data(),
    for (size_t i = 0; i < state->global_tables_stateTwo_i64_arrays.size(); i++) {
        size_t size = state->global_tables_state_i64_sizes[i];
        long long * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(long long)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_stateTwo_i64_arrays[i], size*sizeof(long long), cudaMemcpyHostToDevice));
        temp_i64.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_stateNext_i64, temp_i64.data(), temp_i64.size()*sizeof(long long*), cudaMemcpyHostToDevice));
    temp_i64.clear();

    CUDA_CHECK_RETURN(cudaMalloc(&m_gpu_tables_const_i64_arrays, state->global_tables_const_i64_arrays.size()*sizeof(state->global_tables_const_i64_arrays[0]))); // state->global_tables_state_i64_sizes.data()
    for (size_t i = 0; i < state->global_tables_const_i64_arrays.size(); i++) {
        size_t size = state->global_tables_const_i64_sizes[i];
        long long * item_ptr;
        CUDA_CHECK_RETURN(cudaMalloc(&item_ptr, size*sizeof(long long)));
        CUDA_CHECK_RETURN(cudaMemcpy(item_ptr, state->global_tables_const_i64_arrays[i], size*sizeof(long long), cudaMemcpyHostToDevice));
        temp_i64.push_back(item_ptr);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(m_gpu_tables_const_i64_arrays, temp_i64.data(), temp_i64.size()*sizeof(long long*), cudaMemcpyHostToDevice));
    temp_i64.clear();
}


void GpuBackend::execute_work_gpu(EngineConfig &engine_config, SimulatorConfig &config, int step, double time, int threads_per_block) {

    //Start by updating the MPI buffers if neccesary
#ifdef USE_MPI
    //only needed if there are potentially incoming messages.
//    if(engine_config.my_mpi.world_size > 1) {
    //TABLE_F32

        std::vector<Table_F32> temp32b(state->global_tables_stateTwo_f32_arrays.size(), nullptr);
        CUDA_CHECK_RETURN(cudaMemcpy( temp32b.data(),
                                      m_gpu_tables_stateNow_f32,
                                      temp32b.size()*sizeof(Table_F32),
                                      cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < temp32b.size(); i++) {
            size_t size = m_host_tables_state_f32_sizes[i];
            if(config.debug) {
                printf("32 in - pointer %zu %zu %p \n", size, i, temp32b[i]);
            }
            if(size) {
                if(config.debug) {
                    for (size_t j = 0; j < size; j++)
                        printf(" %f\n", m_host_tables_stateNow_f32[i][j]);
                }
                CUDA_CHECK_RETURN(cudaMemcpy(
                        temp32b[i],
                        m_host_tables_stateNow_f32[i],
                        size * sizeof(float),
                        cudaMemcpyHostToDevice));
            }
        }

        //Table_I64
        std::vector<Table_I64> temp64b(state->global_tables_stateTwo_i64_arrays.size(), nullptr);
        CUDA_CHECK_RETURN(cudaMemcpy( temp64b.data(),
                                      m_gpu_tables_stateNow_i64,
                                      temp64b.size()*sizeof(Table_I64),
                                      cudaMemcpyDeviceToHost));


        for (size_t i = 0; i < temp64b.size(); i++) {
            size_t size = m_host_tables_state_i64_sizes[i];
            if(config.debug) {
                printf("64 in - pointer %zu %zu %p \n", size, i, temp64b[i]);
            }
            if(size) {
                if(config.debug) {
                    for (size_t j = 0; j < size; j++)
                        printf(" %lld\n", m_host_tables_stateNow_i64[i][j]);
                }
                CUDA_CHECK_RETURN(cudaMemcpy(
                        temp64b[i],
                        m_host_tables_stateNow_i64[i],
                        size * sizeof(long long int),
                        cudaMemcpyHostToDevice));
            }
        }
//    }
#endif

    const float dt = engine_config.dt;
    for (size_t idx = 0; idx < tabs.consecutive_kernels.size(); idx++) {
        RawTables::ConsecutiveIterationCallbacks & cic = tabs.consecutive_kernels.at(idx);
        if(config.debug){
            printf("consecutive item %lld (start %ld length %ld) start\n", (long long)idx, (long)cic.start_item, (long)cic.n_items);
            // if(my_mpi.rank != 0) continue;
            // continue;
            fflush(stdout);
        }
        ((GPUIterationCallback)cic.callback) (
                          cic.start_item,
                          cic.n_items,
                          time,
                          dt,
                          m_gpu_constants,
                          m_gpu_const_f32_index,
                          m_gpu_tables_const_f32_sizes,
                          m_gpu_tables_const_f32_arrays,
                          m_gpu_table_const_f32_index,
                          m_gpu_tables_const_i64_sizes,
                          m_gpu_tables_const_i64_arrays,
                          m_gpu_table_const_i64_index,
                          m_gpu_tables_state_f32_sizes,
                          m_gpu_tables_stateNow_f32,
                          m_gpu_tables_stateNext_f32,
                          m_gpu_table_state_f32_index,
                          m_gpu_tables_state_i64_sizes,
                          m_gpu_tables_stateNow_i64,
                          m_gpu_tables_stateNext_i64,
                          m_gpu_table_state_i64_index,
                          m_gpu_state_now,
                          m_gpu_state_next,
                          m_gpu_state_f32_index,
                          step,
                          threads_per_block,
                          &streams_calculate
            );

        if(config.debug){
            printf("consecutive items %lld end\n", (long long)idx);
            fflush(stdout);
        }
    }

    //copy back for printing of stuff.
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
            m_host_state_now,
            m_gpu_state_now,
            state->state_one.size()*sizeof(state->state_one[0]),
            cudaMemcpyDeviceToHost,streams_copy));


#ifdef USE_MPI
    //to copy back to the host for MPI communication
    synchronize_gpu();
    //TABLE_F32
    //todo store this vector
    std::vector<Table_F32> temp32(state->global_tables_stateTwo_f32_arrays.size(), nullptr);
    CUDA_CHECK_RETURN(cudaMemcpy( temp32.data(),
                                  m_gpu_tables_stateNext_f32,
                                  temp32.size()*sizeof(Table_F32),
                                  cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < temp32.size(); i++) {
        size_t size = m_host_tables_state_f32_sizes[i];
        CUDA_CHECK_RETURN(cudaMemcpy(
                m_host_tables_stateNext_f32[i],
                temp32[i],
                size*sizeof(float),
                cudaMemcpyDeviceToHost));
        if(config.debug) {
            printf("32 - out pointer %zu %zu %p \n", size, i, temp32[i]);
            for (size_t j = 0; j < size; j++)
                printf(" %f\n", m_host_tables_stateNext_f32[i][j]);
        }
    }

    //moet hier na!!
    CUDA_CHECK_RETURN(cudaMemcpy(
            m_host_state_next,
            m_gpu_state_next,
            state->state_one.size()*sizeof(state->state_one[0]),
            cudaMemcpyDeviceToHost));
    if(config.debug) {
        for (size_t j = 0; j < state->state_one.size(); j++)
            printf(" %f\n", m_host_state_next[j]);
    }

    //Table_I64
    //todo store this vector
    std::vector<Table_I64> temp64(state->global_tables_stateTwo_i64_arrays.size(), nullptr);
    CUDA_CHECK_RETURN(cudaMemcpy( temp64.data(),
                                  m_gpu_tables_stateNext_i64,
                                  temp64.size()*sizeof(Table_I64),
                                  cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < temp64.size(); i++) {
        size_t size = m_host_tables_state_i64_sizes[i];
        if(config.debug) {
            printf("64 - out pointer %zu %zu %p \n",size,i,temp64[i]);
        }
        if(size) {
            CUDA_CHECK_RETURN(cudaMemcpy(
                    m_host_tables_stateNext_i64[i],
                    temp64[i],
                    size * sizeof(long long int),
                    cudaMemcpyDeviceToHost));
            if(config.debug) {
                for(size_t j = 0; j <size;j++) {
                    printf(" %lld\n", m_host_tables_stateNext_i64[i][j]);
                }
            }
        }
    }
#endif
}



