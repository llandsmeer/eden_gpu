/*
       ###########    ###############              ###########    ######     #########
     ###############   ##################        ###############    #####  ##############
   #####          ####      #####    ######    #####          ####     ######       ######
  ####                    ####          ####  ####                     ####           #####
  ###############         ###            ###  ###############          ####            ####
  ############           ####            ###  ############             ###             ####
  ####                   ####            ###  ####                     ###             ####
  ####              #    ####           ####  ####              #     ####             ####
   #####          ###     ####         ####    #####          ###     ####            ####
    ################        ##############      ################     ####            ####
      ##########              ##########          ##########        ####            #####
*/

/*
Extensible Dynamics Engine for Networks
Parallel simulation engine for ODE-based models
*/

#include "Common.h"
#include "NeuroML.h"

#include <math.h>
#include <limits.h>
#include <errno.h>

#include <map>
#include <set>
#include <chrono>

#include "MMMallocator.h"

#if defined (__linux__) || defined(__APPLE__)
#include <dlfcn.h> // for dynamic loading
#endif

#ifdef _WIN32
// for dynamic loading and other OS specific stuff
// #include <windows.h> // loaded through Common.h at the moment, TODO break out in Windows specific header
#endif

// do not specify alignment for the pointers, in the generic interface
// cannot specify __restrict__ because it is quietly dropped by compilers ( ! ) when the type is allocated with new
// causing a type mismatch when operator delete(T * __restrict__) is called (then why didn't they drop __restrict__ from there too ??)
// just hope the type "mismatch" won't cause a crash in practice
typedef float * Table_F32;
typedef long long * Table_I64;

#include "mpi_setup.h"

// assume standard C calling convention, which is probably the only cross-module one in most architectures
// bother if problems arise LATER
#include "IterationCallback.h"
#include "AppendToVector.h"
#include "string_helpers.h"
#include "FixedWidthNumberPrinter.h"
#include "GeomHelp_Base.h"
#include "RawTables.h"
#include "TableEntry.h"
#include "TypePun.h"
#include "EngineConfig.h"
#include "SimulatorConfig.h"
#include "GenerateModel.h"
#include "StateBuffers.h"
#include "parse_command_line_args.h"
#include "print_eden_cli_header.h"
#include "MpiBuffers.h"
#include "TrajectoryLogger.h"

int main(int argc, char **argv){
    SimulatorConfig config;
    Model model; // TODO move to SimulatorConfig
    RunMetaData metadata;
    EngineConfig engine_config;
    RawTables tabs;

    // first of all, set stdout,stderr to Unbuffered, for live output
    // this action must happen before any output is written !
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    print_eden_cli_header();
    setup_mpi(argc, argv);
    parse_command_line_args(argc, argv, config, model, metadata.config_time_sec);

    //-------------------> create data structures and code based on the model
    timeval init_start, init_end;
    gettimeofday(&init_start, NULL);

    printf("Initializing model...\n");
    if(!GenerateModel(model, config, engine_config, tabs)){
        printf("NeuroML model could not be created\n");
        exit(1);
    }

    TrajectoryLogger trajectory_logger(engine_config);

    // prepare engine for crunching

    printf("Allocating state buffers...\n");

    StateBuffers state(tabs);

    // ************** WARNING ****************
    // dump_raw_state_table() and mpi.init_communicate() read these pointers
    // from the StateBuffers object itself that means that when we go to
    // the GPU we either need to copy back to the vectors, use a custom allocator
    // or just raise a warning if we try to use dump_raw_state_table() or MPI in
    // combination with GPU, or replace the StateBuffer reference with this pointer list
    float *global_state_now = state.state_one.data();
    float *global_state_next = state.state_two.data();
    Table_F32 *global_tables_stateNow_f32  = state.global_tables_stateOne_f32_arrays.data();
    Table_I64 *global_tables_stateNow_i64  = state.global_tables_stateOne_i64_arrays.data();
    Table_F32 *global_tables_stateNext_f32 = state.global_tables_stateTwo_f32_arrays.data();
    Table_I64 *global_tables_stateNext_i64 = state.global_tables_stateTwo_i64_arrays.data();

    Table_F32 * global_tables_const_f32_arrays = state.global_tables_const_f32_arrays.data();
    Table_I64 * global_tables_const_i64_arrays = state.global_tables_const_i64_arrays.data();
    long long * global_tables_const_f32_sizes = state.global_tables_const_f32_sizes.data();
    long long * global_tables_const_i64_sizes = state.global_tables_const_i64_sizes.data();
    long long * global_tables_state_f32_sizes = state.global_tables_state_f32_sizes.data();
    long long * global_tables_state_i64_sizes = state.global_tables_state_i64_sizes.data();

    gettimeofday(&init_end, NULL);
    metadata.init_time_sec = TimevalDeltaSec(init_start, init_end);

    if(config.dump_raw_layout){
        state.dump_raw_layout(tabs);
    }
    // MPI_Finalize();
    // exit(1);

#ifdef USE_MPI
    printf("Allocating comm buffers...\n");
    MpiBuffers mpi_buffers(engine_config);
#endif

    printf("Starting simulation loop...\n");

    // perform the crunching
    timeval run_start, run_end;
    gettimeofday(&run_start, NULL);

    double time = engine_config.t_initial;
    // need multiple initialization steps, to make sure the dependency chains of all state variables are resolved
    for( long long step = -3; time <= engine_config.t_final; step++ ){

        bool initializing = step <= 0;

        #ifdef USE_MPI
        mpi_buffers.init_communicate(engine_config, state, config);
        #endif

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
                tabs.global_constants.data(),   tabs.global_const_f32_index      [item], global_tables_const_f32_sizes, global_tables_const_f32_arrays,      tabs.global_table_const_f32_index[item],
                global_tables_const_i64_sizes,  global_tables_const_i64_arrays,      tabs.global_table_const_i64_index[item],
                global_tables_state_f32_sizes,  global_tables_stateNow_f32,          global_tables_stateNext_f32, tabs.global_table_state_f32_index[item],
                global_tables_state_i64_sizes,  global_tables_stateNow_i64,          global_tables_stateNext_i64, tabs.global_table_state_i64_index[item],
                global_state_now, global_state_next , tabs.global_state_f32_index      [item],
                step
            );
            if(config.debug){
                printf("item %lld end\n", item);
                fflush(stdout);
            }
        }

        if( !initializing ){
            // output what needs to be output
            trajectory_logger.write_output_logs(engine_config, time, global_state_now, /* needed on mpi???: */global_tables_stateNow_f32);
        }

        // output state dump
        if( config.dump_raw_state_scalar || config.dump_raw_state_table ){
            if( !initializing ){
                printf("State: t = %g %s\n", time, Scales<Time>::native.name);
            }
            else{
                printf("State: t = %g %s, initialization step %lld\n", time, Scales<Time>::native.name, step);
            }
        }
        if( config.dump_raw_state_scalar ){
            // print state, separated by work item
            for( size_t i = 0, itm = 1; i < state.state_one.size(); i++ ){
                printf("%g \t", global_state_next[i]);
                while( itm < tabs.global_state_f32_index.size() && (i + 1) == (size_t)tabs.global_state_f32_index[itm] ){
                    printf("| ");
                    itm++;
                }
            }
            printf("\n");
        }
        if( config.dump_raw_state_table ){
            state.dump_raw_state_table(tabs);
        }

        #ifdef USE_MPI
        mpi_buffers.finish_communicate();
        #endif

        //prepare for next parallel iteration
        if( !initializing ){
            time += engine_config.dt;
        }

        // or a modulo-based cyclic queue LATER? will logging be so much of an issue? if so let the logger clone the state instead of duplicating the entire buffers
        std::swap(global_state_now, global_state_next);
        std::swap(global_tables_stateNow_f32, global_tables_stateNext_f32);
        std::swap(global_tables_stateNow_i64, global_tables_stateNext_i64);
    }
    // done!


    gettimeofday(&run_end, NULL);
    metadata.run_time_sec = TimevalDeltaSec(run_start, run_end);


    printf("Config: %.3lf Setup: %.3lf Run: %.3lf \n", metadata.config_time_sec, metadata.init_time_sec, metadata.run_time_sec );
    #ifdef __linux__
    //get memory usage information too
    long long memResidentPeak = metadata.peak_resident_memory_bytes = getPeakResidentSetBytes();
    long long memResidentEnd = metadata.end_resident_memory_bytes = getCurrentResidentSetBytes();
    long long memHeap = getCurrentHeapBytes();
    printf("Peak: %lld Now: %lld Heap: %lld\n", memResidentPeak, memResidentEnd, memHeap );
    #endif
    //-------------------> release sim data structures, though it's not absolutely necessary at this point

#ifdef USE_MPI
    // this is necessary, so stdio files are actually flushed
    MPI_Finalize();
#endif

    return 0;
}
