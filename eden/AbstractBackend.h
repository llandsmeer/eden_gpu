//description of the Backends implementations.
#ifndef BACKENDS_H
#define BACKENDS_H

//dependencies
#include "NeuroML.h"    //needed for implementations
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
    virtual void execute_work_items(EngineConfig & engine_config, SimulatorConfig & config, int step, double time) = 0;
    virtual void swap_buffers() = 0;
    virtual void dump_iteration(SimulatorConfig & config, bool initializing, double time, long long step) = 0;
    virtual float * global_state_now() const = 0;
    virtual Table_F32 * global_tables_stateNow_f32() const = 0;
};
#endif
