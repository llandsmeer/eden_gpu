//
// Created by max on 12-10-21.
//

#ifndef EDEN_GPU_GPUBACKEND_H
#define EDEN_GPU_GPUBACKEND_H

#include <cstring>
#include "../../AbstractBackend.h"

class GpuBackend : public AbstractBackend {
    using AbstractBackend::AbstractBackend;
public:

  /* Pure CPU implementation just refers to existing state buffers */
//    Init function
    ~GpuBackend() override{
        delete state;
    }
    void init() override{
#ifdef USE_GPU
       gpu_init();
#else
        printf("NOOOOOO!!\n");
        exit(2);
#endif
    };

    //    getters
//    \\todo
    float     * print_state_now()               const override { return m_print_state_now; };
    Table_F32 * print_tables_stateNow_f32()     const override { return m_print_tables_stateNow_f32;};

    float     * device_state_now             () const override { return m_gpu_state_now; }
    Table_F32 * device_tables_stateNow_f32   () const override { return m_gpu_tables_stateNow_f32; }
    Table_I64 * device_tables_stateNow_i64   () const override { return m_gpu_tables_stateNow_i64; }

    float     * host_state_now               () const override { return m_host_state_now; }
    Table_F32 * host_tables_stateNow_f32     () const override { return m_host_tables_stateNow_f32; }
    Table_I64 * host_tables_stateNow_i64     () const override { return m_host_tables_stateNow_i64; }
    long long * host_tables_state_i64_sizes  () const override { return m_host_tables_state_i64_sizes;} //todo

//    functionality
    void execute_work_items(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) override{
#ifdef USE_GPU
        execute_work_gpu(engine_config,config, step, time, engine_config.threads_per_block);
#else
        printf("NOOOOOO!!\n");
        exit(2);
#endif
    };

    void synchronize() const override {
#ifdef USE_GPU
        synchronize_gpu();
#else
        printf("NOOOOOO!!\n");
        exit(2);
#endif
    };

    void swap_buffers() override {
        //GPU buffers
        std::swap(m_gpu_state_now, m_gpu_state_next);
        std::swap(m_gpu_tables_stateNow_f32, m_gpu_tables_stateNext_f32);
        std::swap(m_gpu_tables_stateNow_i64, m_gpu_tables_stateNext_i64);

        //Host buffers
        std::swap(m_host_state_now, m_host_state_next);
        std::swap(m_host_tables_stateNow_f32, m_host_tables_stateNext_f32);
        std::swap(m_host_tables_stateNow_i64, m_host_tables_stateNext_i64);
    }

    void populate_print_buffer() override{
        //swap next because beginning of loop
//        m_print_state_now = m_host_state_next;
//        m_print_tables_stateNow_f32 = m_host_tables_stateNext_f32;

//        std::swap(m_host_state_next, m_print_state_now);
//        std::swap(m_host_tables_stateNext_f32, m_print_tables_stateNow_f32);

        //Here we should print the m_global_tables_stateNow_f32 into m_print_tables_stateNow_f32
        for (size_t i = 0; i < state->global_tables_stateOne_f32_arrays.size(); i++) {
            const size_t size = state->global_tables_state_f32_sizes[i];
            std::memcpy(m_print_tables_stateNow_f32[i], m_host_tables_stateNext_f32[i], size*sizeof(float));
        }

        //this should copy the state now into the print buffer. so it can be printed without worries :D
        std::memcpy(m_print_state_now, m_host_state_next, state->state_one.size()*sizeof(state->state_one[0]));
    };

    void dump_iteration(SimulatorConfig & config, bool initializing, double time, long long step) override {
        if( config.dump_raw_state_scalar || config.dump_raw_state_table ){
            if( !initializing ){
                printf("State: t = %g %s\n", time, Scales<Time>::native.name);
            } else {
                printf("State: t = %g %s, initialization step %lld\n", time, Scales<Time>::native.name, step);
            }
        }
        if( config.dump_raw_state_scalar ){
            // print state, separated by work item
            for( size_t i = 0, itm = 1; i < state->state_one.size(); i++ ){
                printf("%g \t", host_state_now()[i]);
                while( itm < tabs.global_state_f32_index.size() && (i + 1) == (size_t)tabs.global_state_f32_index[itm] ){
                    printf("| ");
                    itm++;
                }
            }
            printf("\n");
        }
        if( config.dump_raw_state_table ) state->dump_raw_state_table(&tabs);
    }

#ifdef USE_GPU
    void execute_work_gpu(EngineConfig & engine_config, SimulatorConfig & config, int step, double time, int threads_per_block);
    void gpu_init();
    static void synchronize_gpu();
#endif

private:

    //    Print Variables
    float     * m_print_state_now                    = nullptr;
    Table_F32 * m_print_tables_stateNow_f32          = nullptr;

// HOST pointers
//very important running variables
    float     * m_host_state_now = nullptr;
    float     * m_host_state_next = nullptr;
    Table_F32 * m_host_tables_stateNow_f32 = nullptr;
    Table_I64 * m_host_tables_stateNow_i64 = nullptr;
    Table_F32 * m_host_tables_stateNext_f32= nullptr;
    Table_I64 * m_host_tables_stateNext_i64 = nullptr;
    long long * m_host_tables_state_f32_sizes  = nullptr;
    long long * m_host_tables_state_i64_sizes = nullptr;

// GPU Pointers
//very important running variables
    float     * m_gpu_state_now = nullptr;
    float     * m_gpu_state_next = nullptr;
    Table_F32 * m_gpu_tables_stateNow_f32 = nullptr;
    Table_I64 * m_gpu_tables_stateNow_i64 = nullptr;
    Table_F32 * m_gpu_tables_stateNext_f32= nullptr;
    Table_I64 * m_gpu_tables_stateNext_i64 = nullptr;
//     Constants
    float         * m_gpu_constants               = nullptr;
    long long int * m_gpu_const_f32_index         = nullptr;
    long long int * m_gpu_table_const_f32_index   = nullptr;
    long long int * m_gpu_table_const_i64_index   = nullptr;
    long long int * m_gpu_table_state_f32_index   = nullptr;
    long long int * m_gpu_table_state_i64_index   = nullptr;
    long long int * m_gpu_state_f32_index         = nullptr;
    Table_F32     * m_gpu_tables_const_f32_arrays = nullptr;
    Table_I64     * m_gpu_tables_const_i64_arrays = nullptr;
    long long     * m_gpu_tables_const_f32_sizes  = nullptr;
    long long     * m_gpu_tables_const_i64_sizes  = nullptr;
    long long     * m_gpu_tables_state_f32_sizes  = nullptr;
    long long     * m_gpu_tables_state_i64_sizes  = nullptr;

};


#endif //EDEN_CPU_CPUBACKEND_H