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
#include "MpiBuffers.h"
#include "TrajectoryLogger.h"
#include "Timer.h"
#include "backends.h"

#include "parse_command_line_args.h"
#include "print_eden_cli_header.h"
#include "print_runtime_usage.h"

int main(int argc, char **argv){
    SimulatorConfig config;
    Model model;
    RunMetaData metadata;
    EngineConfig engine_config;
    RawTables tabs;
    setvbuf(stdout, NULL, _IONBF, 0); // first of all, set stdout,stderr to Unbuffered, for live output
    setvbuf(stderr, NULL, _IONBF, 0); // this action must happen before any output is written !
    print_eden_cli_header();
    setup_mpi(argc, argv);
    parse_command_line_args(argc, argv, config, model, metadata.config_time_sec);

    printf("Initializing model...\n");
    Timer init_timer;
    if(!GenerateModel(model, config, engine_config, tabs)){
        printf("NeuroML model could not be created\n"); exit(1);
    }
    TrajectoryLogger trajectory_logger(engine_config);

    printf("Allocating state buffers...\n");
    StateBuffers state(tabs);
    CpuBackend backend(tabs, state);
    backend.init();
    metadata.init_time_sec = init_timer.delta();
    if(config.dump_raw_layout) state.dump_raw_layout(tabs);
    MpiBuffers mpi_buffers(engine_config);

    printf("Starting simulation loop...\n");
    Timer run_timer;
    double time = engine_config.t_initial;
    // need multiple initialization steps, to make sure the dependency chains of all state variables are resolved
    for( long long step = -3; time <= engine_config.t_final; step++ ){
        bool initializing = step <= 0;
        mpi_buffers.init_communicate(engine_config, state, config); // need to copy between backend & state when using mpi
        backend.execute_work_items(engine_config, config, step, time);
        if( !initializing ){
            trajectory_logger.write_output_logs( engine_config, time,
                    backend.global_state_now(), /* needed on mpi???: */backend.global_tables_stateNow_f32());
        }
        backend.dump_iteration(config, initializing, time, step);
        mpi_buffers.finish_communicate();
        if( !initializing ) time += engine_config.dt;
        backend.swap_buffers();
    }

    metadata.run_time_sec = run_timer.delta();
    print_runtime_usage(metadata);
}
