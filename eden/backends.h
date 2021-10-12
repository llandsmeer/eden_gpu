//description of the Backends implementations.
#ifndef BACKENDS_H
#define BACKENDS_H

#include "Common.h"
#include "NeuroML.h"
#include "RawTables.h"
#include "StateBuffers.h"
#include "SimulatorConfig.h"
#include "EngineConfig.h"

class AbstractBackend {

public:
    StateBuffers * state;
    RawTables tabs;
    AbstractBackend():state(nullptr){}

    virtual ~AbstractBackend() {};
    virtual void init() = 0;
    virtual float * global_state_now() const = 0;
    virtual float * global_state_next() const = 0;
    virtual Table_F32 * global_tables_stateNow_f32 () const = 0;
    virtual Table_I64 * global_tables_stateNow_i64 () const = 0;
    virtual Table_F32 * global_tables_stateNext_f32() const = 0;
    virtual Table_I64 * global_tables_stateNext_i64() const = 0;
    virtual Table_F32 * global_tables_const_f32_arrays() const = 0;
    virtual Table_I64 * global_tables_const_i64_arrays() const = 0;
    virtual long long * global_tables_const_f32_sizes() const = 0;
    virtual long long * global_tables_const_i64_sizes() const = 0;
    virtual long long * global_tables_state_f32_sizes() const = 0;
    virtual long long * global_tables_state_i64_sizes() const = 0;
    virtual void execute_work_items(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) = 0;
    virtual void swap_buffers() = 0;
    virtual void dump_iteration(SimulatorConfig & config, bool initializing, double time, long long step) = 0;
};


class CpuBackend : public AbstractBackend {
    using AbstractBackend::AbstractBackend;
public:
    /* Pure CPU implementation just refers to existing state buffers */
//    Init function
    ~CpuBackend(){
        delete state;
    }

    void init() override {
        //create the Statebuffers
        state = new StateBuffers(tabs);
        //create the pointers //todo trow away :D
        m_global_state_now = state->state_one.data();
        m_global_state_next = state->state_two.data();
        m_global_tables_stateNow_f32 = state->global_tables_stateOne_f32_arrays.data();
        m_global_tables_stateNow_i64 = state->global_tables_stateOne_i64_arrays.data();
        m_global_tables_stateNext_f32 = state->global_tables_stateTwo_f32_arrays.data();
        m_global_tables_stateNext_i64 = state->global_tables_stateTwo_i64_arrays.data();
    }
//    getters
    float * global_state_now() const override { return m_global_state_now; }
    float * global_state_next() const override { return m_global_state_next; }
    Table_F32 * global_tables_stateNow_f32 () const override { return m_global_tables_stateNow_f32; }
    Table_I64 * global_tables_stateNow_i64 () const override { return m_global_tables_stateNow_i64; }
    Table_F32 * global_tables_stateNext_f32() const override { return m_global_tables_stateNext_f32; }
    Table_I64 * global_tables_stateNext_i64() const override { return m_global_tables_stateNext_i64; }
    Table_F32 * global_tables_const_f32_arrays() const override { return state->global_tables_const_f32_arrays.data(); }
    Table_I64 * global_tables_const_i64_arrays() const override { return state->global_tables_const_i64_arrays.data(); }
    long long * global_tables_const_f32_sizes() const override { return state->global_tables_const_f32_sizes.data(); }
    long long * global_tables_const_i64_sizes() const override { return state->global_tables_const_i64_sizes.data(); }
    long long * global_tables_state_f32_sizes() const override { return state->global_tables_state_f32_sizes.data(); }
    long long * global_tables_state_i64_sizes() const override { return state->global_tables_state_i64_sizes.data(); }

//    functionality
    void execute_work_items(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) override {
        //execute_work_items_one_by_one(engine_config, config, step, time);
        execute_work_items_as_consecutives(engine_config, config, step, time);
    }

    void execute_work_items_one_by_one(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) {
        //prepare for parallel iteration
        const float dt = engine_config.dt;
        // Execute all work items
        #pragma omp parallel for schedule(runtime)
        for( long long item = 0; item < engine_config.work_items; item++ ){
            if(config.debug){
                printf("item %lld start\n", item);
                // if(my_mpi.rank != 0) continue;
                // continue;
                fflush(stdout);
            }
            tabs.callbacks[item]( time, dt,
                                   tabs.global_constants.data(),      tabs.global_const_f32_index[item], global_tables_const_f32_sizes(),            global_tables_const_f32_arrays(),      tabs.global_table_const_f32_index[item],
                                   global_tables_const_i64_sizes(),    global_tables_const_i64_arrays(),   tabs.global_table_const_i64_index[item],
                                   global_tables_state_f32_sizes(),    global_tables_stateNow_f32(),       global_tables_stateNext_f32(),              tabs.global_table_state_f32_index[item],
                                   global_tables_state_i64_sizes(),    global_tables_stateNow_i64(),       global_tables_stateNext_i64(),              tabs.global_table_state_i64_index[item],
                                   global_state_now(),                 global_state_next(),                tabs.global_state_f32_index[item],
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
            #pragma omp parallel for schedule(runtime)
            for (size_t item = cic.start_item; item < cic.start_item + cic.n_items; item++) {
               cic.callback( time, dt,
                                       tabs.global_constants.data(),      tabs.global_const_f32_index[item], global_tables_const_f32_sizes(),            global_tables_const_f32_arrays(),      tabs.global_table_const_f32_index[item],
                                       global_tables_const_i64_sizes(),    global_tables_const_i64_arrays(),   tabs.global_table_const_i64_index[item],
                                       global_tables_state_f32_sizes(),    global_tables_stateNow_f32(),       global_tables_stateNext_f32(),              tabs.global_table_state_f32_index[item],
                                       global_tables_state_i64_sizes(),    global_tables_stateNow_i64(),       global_tables_stateNext_i64(),              tabs.global_table_state_i64_index[item],
                                       global_state_now(),                 global_state_next(),                tabs.global_state_f32_index[item],
                                       step
                );
            }
            if(config.debug){
                printf("consecutive items %lld end\n", (long long)idx);
                fflush(stdout);
            }
        }
    }


    void swap_buffers() override {
        std::swap(m_global_state_now, m_global_state_next);
        std::swap(m_global_tables_stateNow_f32, m_global_tables_stateNext_f32);
        std::swap(m_global_tables_stateNow_i64, m_global_tables_stateNext_i64);
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

private:
    float * m_global_state_now = nullptr;
    float * m_global_state_next = nullptr;
    Table_F32 * m_global_tables_stateNow_f32 = nullptr;
    Table_I64 * m_global_tables_stateNow_i64 = nullptr;
    Table_F32 * m_global_tables_stateNext_f32= nullptr;
    Table_I64 * m_global_tables_stateNext_i64 = nullptr;

};

#endif
