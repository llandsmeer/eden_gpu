//
// Created by max on 12-10-21.
//

#ifndef EDEN_GPU_GPUBACKEND_H
#define EDEN_GPU_GPUBACKEND_H

#include "../../AbstractBackend.h"

class GpuBackend : public AbstractBackend {
    using AbstractBackend::AbstractBackend;
public:
    /* Pure CPU implementation just refers to existing state buffers */
//    Init function

    ~GpuBackend() override{
        delete state;
    }
    void init() override {
        //create the Statebuffers
        state = new StateBuffers(tabs);

        //create the copy back host pointers
        m_Host_state_now = state->state_one.data();
        m_Host_state_next = state->state_two.data();

        //do GPU stuff
#ifdef USE_GPU
        copy_data_to_device();
#else
        //should never happen
#endif
    }

//    getters
#ifdef USE_GPU
    float * global_state_now() const override ;
    Table_F32 * global_tables_stateNow_f32 () const override ;
#else
    float * global_state_now() const override  { return 0; }
    Table_F32 * global_tables_stateNow_f32 () const override  { return 0; }
#endif

    float * global_state_next() const  { return m_global_state_next; }
    Table_I64 * global_tables_stateNow_i64 () const override { return m_global_tables_stateNow_i64; }
    Table_F32 * global_tables_stateNext_f32() const  { return m_global_tables_stateNext_f32; }
    Table_I64 * global_tables_stateNext_i64() const  { return m_global_tables_stateNext_i64; }

    Table_F32 * global_tables_const_f32_arrays() const  { return state->global_tables_const_f32_arrays.data(); }
    Table_I64 * global_tables_const_i64_arrays() const  { return state->global_tables_const_i64_arrays.data(); }
    long long * global_tables_const_f32_sizes() const  { return state->global_tables_const_f32_sizes.data(); }
    long long * global_tables_const_i64_sizes() const  { return state->global_tables_const_i64_sizes.data(); }
    long long * global_tables_state_f32_sizes() const  { return state->global_tables_state_f32_sizes.data(); }
    long long * global_tables_state_i64_sizes() const override { return state->global_tables_state_i64_sizes.data(); }


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
        std::swap(m_global_state_now, m_global_state_next);
        std::swap(m_global_tables_stateNow_f32, m_global_tables_stateNext_f32);
        std::swap(m_global_tables_stateNow_i64, m_global_tables_stateNext_i64);

        //CPU buffers to print stuff
        std::swap(m_Host_state_now, m_Host_state_next);
    }
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
                printf("%g \t", global_state_next()[i]);
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
    bool copy_data_to_device();
    static void synchronize_gpu();
#endif

private:
    float * m_global_constants;
    long long int* m_global_const_f32_index;
    long long int* m_global_table_const_f32_index;
    long long int* m_global_table_const_i64_index;
    long long int* m_global_table_state_f32_index;
    long long int* m_global_table_state_i64_index;
    long long int* m_global_state_f32_index;

    float * m_Host_state_now = nullptr;
    float * m_Host_state_next = nullptr;

    float * m_global_state_now = nullptr;
    float * m_global_state_next = nullptr;
    Table_F32 * m_global_tables_stateNow_f32 = nullptr;
    Table_I64 * m_global_tables_stateNow_i64 = nullptr;
    Table_F32 * m_global_tables_stateNext_f32= nullptr;
    Table_I64 * m_global_tables_stateNext_i64 = nullptr;

    Table_F32 * m_global_tables_const_f32_arrays = nullptr;
    Table_I64 * m_global_tables_const_i64_arrays = nullptr;
    long long * m_global_tables_const_f32_sizes = nullptr;
    long long * m_global_tables_const_i64_sizes = nullptr;
    long long * m_global_tables_state_f32_sizes = nullptr;
    long long * m_global_tables_state_i64_sizes = nullptr;

};


#endif //EDEN_CPU_CPUBACKEND_H
