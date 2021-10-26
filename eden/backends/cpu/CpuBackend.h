//
// Created by max on 12-10-21.
//

#ifndef EDEN_CPU_CPUBACKEND_H
#define EDEN_CPU_CPUBACKEND_H

#include <cstring>
#include "../../AbstractBackend.h"


class CpuBackend : public AbstractBackend {
    using AbstractBackend::AbstractBackend;
public:

//    destructor, constructor can stay pure virtual
    ~CpuBackend() override{
        delete state;
    }

//    Init function
    void init() override {
        //create the Statebuffers
        state = new StateBuffers(tabs);

        m_print_state_now                = state->state_print.data();
        m_print_tables_stateNow_f32      = state->global_tables_statePrint_f32_arrays.data();

        m_global_state_now               = state->state_one.data();
        m_global_state_next              = state->state_two.data();
        m_global_tables_stateNow_f32     = state->global_tables_stateOne_f32_arrays.data();
        m_global_tables_stateNow_i64     = state->global_tables_stateOne_i64_arrays.data();
        m_global_tables_stateNext_f32    = state->global_tables_stateTwo_f32_arrays.data();
        m_global_tables_stateNext_i64    = state->global_tables_stateTwo_i64_arrays.data();

        m_global_constants               = tabs.global_constants.data();
        m_global_const_f32_index         = tabs.global_const_f32_index.data();
        m_global_table_const_f32_index   = tabs.global_table_const_f32_index.data();
        m_global_table_const_i64_index   = tabs.global_table_const_i64_index.data();
        m_global_table_state_f32_index   = tabs.global_table_state_f32_index.data();
        m_global_table_state_i64_index   = tabs.global_table_state_i64_index.data();
        m_global_state_f32_index         = tabs.global_state_f32_index.data();

        m_global_tables_const_f32_arrays = state->global_tables_const_f32_arrays.data();
        m_global_tables_const_i64_arrays = state->global_tables_const_i64_arrays.data();
        m_global_tables_const_f32_sizes  = state->global_tables_const_f32_sizes.data();
        m_global_tables_const_i64_sizes  = state->global_tables_const_i64_sizes.data();
        m_global_tables_state_f32_sizes  = state->global_tables_state_f32_sizes.data();
        m_global_tables_state_i64_sizes  = state->global_tables_state_i64_sizes.data();
    }

//    getters all is CPU state so nothing to worry about.
    float     * print_state_now()             const override  { return m_print_state_now; }
    Table_F32 * print_tables_stateNow_f32()   const override  { return m_print_tables_stateNow_f32; }

    float     * device_state_now             () const override { return m_global_state_now; }
    Table_F32 * device_tables_stateNow_f32   () const override { return m_global_tables_stateNow_f32; }
    Table_I64 * device_tables_stateNow_i64   () const override { return m_global_tables_stateNow_i64; }

    float     * host_state_now               () const override { return m_global_state_now; }
    Table_F32 * host_tables_stateNow_f32     () const override { return m_global_tables_stateNow_f32; }
    Table_I64 * host_tables_stateNow_i64     () const override { return m_global_tables_stateNow_i64; }
    long long * host_tables_state_i64_sizes  () const override { return m_global_tables_state_i64_sizes; }

//    functionality
    void execute_work_items(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) override {
        execute_work_items_one_by_one(engine_config, config, step, time);
//        execute_work_items_as_consecutives(engine_config, config, step, time);
    }
    void synchronize() const override{
       //nothing to be done yet
    }
    void swap_buffers() override {
        std::swap(m_global_state_now, m_global_state_next);
        std::swap(m_global_tables_stateNow_f32, m_global_tables_stateNext_f32);
        std::swap(m_global_tables_stateNow_i64, m_global_tables_stateNext_i64);

    }

    //this happens on
    void populate_print_buffer() override{

        //Here we should print the m_global_tables_stateNow_f32 into m_print_tables_stateNow_f32
        for (size_t i = 0; i < state->global_tables_stateOne_f32_arrays.size(); i++) {
            const size_t size = state->global_tables_state_f32_sizes[i];
            std::memcpy(m_print_tables_stateNow_f32[i], m_global_tables_stateNext_f32[i], size*sizeof(float));
        }

        //this should copy the state now into the print buffer. so it can be printed without worries :D
        std::memcpy(m_print_state_now, m_global_state_next, state->state_one.size()*sizeof(state->state_one[0]));
    };

//    debug option
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
                printf("%g \t", m_global_state_next[i]);
                while( itm < tabs.global_state_f32_index.size() && (i + 1) == (size_t)tabs.global_state_f32_index[itm] ){
                    printf("| ");
                    itm++;
                }
            }
            printf("\n");
        }
        if( config.dump_raw_state_table ) state->dump_raw_state_table(&tabs);
    }

//    actuall working functions
    void execute_work_items_one_by_one(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) {
        //prepare for parallel iteration
        const float dt = engine_config.dt;
        // Execute all work items
        //#pragma omp parallel for schedule(runtime)
        for( long long item = 0; item < engine_config.work_items; item++ ){
            if(config.debug){
                printf("item %lld start\n", item);
                // if(my_mpi.rank != 0) continue;
                // continue;
                fflush(stdout);
            }

            tabs.callbacks[item]( time,
                                  dt,
                                  m_global_constants,
                                  m_global_const_f32_index[item],
                                  m_global_tables_const_f32_sizes,
                                  m_global_tables_const_f32_arrays,
                                  m_global_table_const_f32_index[item],
                                  m_global_tables_const_i64_sizes,
                                  m_global_tables_const_i64_arrays,
                                  m_global_table_const_i64_index[item],
                                  m_global_tables_state_f32_sizes,
                                  m_global_tables_stateNow_f32,
                                  m_global_tables_stateNext_f32,
                                  m_global_table_state_f32_index[item],
                                  m_global_tables_state_i64_sizes,
                                  m_global_tables_stateNow_i64,
                                  m_global_tables_stateNext_i64,
                                  m_global_table_state_i64_index[item],
                                  m_global_state_now,
                                  m_global_state_next,
                                  m_global_state_f32_index[item],
                                  step
            );
            if(config.debug){
                printf("item %lld end\n", item);
                fflush(stdout);
            }
        }
    }
    void execute_work_items_as_consecutives(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) {
        //prepare for parallel iteration
        const float dt = engine_config.dt;
        // Execute all work items
        for (size_t idx = 0; idx < tabs.consecutive_kernels.size(); idx++) {
            if(config.debug){
                printf("consecutive items %lld start\n", (long long)idx);
                // if(my_mpi.rank != 0) continue;
                // continue;
                fflush(stdout);
            }
            RawTables::ConsecutiveIterationCallbacks & cic = tabs.consecutive_kernels.at(idx);
//#pragma omp parallel for schedule(runtime)
            for (size_t item = cic.start_item; item < cic.start_item + cic.n_items; item++) {
                cic.callback( time,
                              dt,
                              m_global_constants,
                              m_global_const_f32_index[item],
                              m_global_tables_const_f32_sizes,
                              m_global_tables_const_f32_arrays,
                              m_global_table_const_f32_index[item],
                              m_global_tables_const_i64_sizes,
                              m_global_tables_const_i64_arrays,
                              m_global_table_const_i64_index[item],
                              m_global_tables_state_f32_sizes,
                              m_global_tables_stateNow_f32,
                              m_global_tables_stateNext_f32,
                              m_global_table_state_f32_index[item],
                              m_global_tables_state_i64_sizes,
                              m_global_tables_stateNow_i64,
                              m_global_tables_stateNext_i64,
                              m_global_table_state_i64_index[item],
                              m_global_state_now,
                              m_global_state_next,
                              m_global_state_f32_index[item],
                              step
                );

            }
            if(config.debug){
                printf("consecutive items %lld end\n", (long long)idx);
                fflush(stdout);
            }
        }
    }

private:
//    Print Variables
    float     * m_print_state_now                    = nullptr;
    Table_F32 * m_print_tables_stateNow_f32          = nullptr;

//    State Variables
    float     * m_global_state_now                   = nullptr;
    float     * m_global_state_next                  = nullptr;
    Table_F32 * m_global_tables_stateNow_f32         = nullptr;
    Table_I64 * m_global_tables_stateNow_i64         = nullptr;
    Table_F32 * m_global_tables_stateNext_f32        = nullptr;
    Table_I64 * m_global_tables_stateNext_i64        = nullptr;
//     Constants
    float         * m_global_constants               = nullptr;
    long long int * m_global_const_f32_index         = nullptr;
    long long int * m_global_table_const_f32_index   = nullptr;
    long long int * m_global_table_const_i64_index   = nullptr;
    long long int * m_global_table_state_f32_index   = nullptr;
    long long int * m_global_table_state_i64_index   = nullptr;
    long long int * m_global_state_f32_index         = nullptr;
    Table_F32     * m_global_tables_const_f32_arrays = nullptr;
    Table_I64     * m_global_tables_const_i64_arrays = nullptr;
    long long     * m_global_tables_const_f32_sizes  = nullptr;
    long long     * m_global_tables_const_i64_sizes  = nullptr;
    long long     * m_global_tables_state_f32_sizes  = nullptr;
    long long     * m_global_tables_state_i64_sizes  = nullptr;
};


#endif //EDEN_CPU_CPUBACKEND_H
