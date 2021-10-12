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

//standard libs
#include <cmath>

#if defined (__linux__) || defined(__APPLE__)
#include <dlfcn.h> // for dynamic loading
#endif

#ifdef _WIN32
// for dynamic loading and other OS specific stuff
// #include <windows.h> // loaded through Common.h at the moment, TODO break out in Windows specific header
#endif

// Local includes
#include "Common.h"
#include "NeuroML.h"
#include "MMMallocator.h"
#include "StringHelpers.h"

//clean
#include "GPU_helpers.h"
#include "Timer.h"

// mess to clean up
#include "IterationCallback.h"
#include "FixedWidthNumberPrinter.h"
#include "GeomHelp_Base.h"
#include "TypePun.h"

#include "EngineConfig.h"
#include "SimulatorConfig.h"
#include "GenerateModel.h"
#include "Mpi_helpers.h"
#include "TrajectoryLogger.h"
#include "backends.h"
#include "parse_command_line_args.h"
#include "print_eden_cli_header.h"

int main(int argc, char **argv){

//-----> Starting the simulator
    {
        setvbuf(stdout, NULL, _IONBF, 0); // first of all, set stdout,stderr to Unbuffered, for live output
        setvbuf(stderr, NULL, _IONBF, 0); // this action must happen before any output is written !
        print_eden_cli_header();          // To print the CLI header
    }

//-----> declaration of all used variables
    RunMetaData metadata;             // Struct for keeping track of non backend specific meta data
    SimulatorConfig config;           // Struct for Configuration of the simulator
    Model model;                      // Struct for Keeping track of the model
    EngineConfig engine_config;       // Struct for Configuration of the Engine

    AbstractBackend *backend = nullptr;             // Class to handle all backend calls
    TrajectoryLogger *trajectory_logger = nullptr;  // Class to handle all output generation
    MpiBuffers *mpi_buffers = nullptr;              // Class to handle all MPI communication

//-----> Find and check the specific engine_config
    setup_mpi(argc, argv);              //check if everything works fine
    setup_gpu(engine_config);           //same for gpu

//-----> Check the command line input with options
    parse_command_line_args(argc, argv, config, model, metadata.config_time_sec);

//-----> Init the backend
    printf("Initializing backend...\n");
    {
        if (engine_config.backend == backend_kind_cpu) {
            backend = new CpuBackend();
        } else {
            printf("No valid backed selected");
            exit(10);
        }
    }

//----> Initialize the backend
    printf("Initializing model...\n");
    {
        Timer init_timer;
        if (!GenerateModel(model, config, engine_config, backend->tabs)) {
            printf("NeuroML model could not be created\n");
            exit(1);
        }
        trajectory_logger = new TrajectoryLogger(engine_config); //To log results
        printf("Allocating state buffers...\n");
        backend->init();

        //just some timer functions to time this meta data --> encorperate this into meta data class
        metadata.init_time_sec = init_timer.delta();

        //debug mem layout
        if(config.dump_raw_layout) backend->state->dump_raw_layout(backend->tabs);

        //Initialize MPI
         mpi_buffers = new MpiBuffers(engine_config);
    }

//----> Simulations loop
    printf("Starting simulation loop...\n");
    {
        Timer run_timer;
        double time = engine_config.t_initial;
        // need multiple initialization steps, to make sure the dependency chains of all state variables are resolved
        for (long long step = -3; time <= engine_config.t_final; step++) {

            // we don't need to keep setting this variable i think however if statement is worse.
            bool initializing = step <= 0;

            //init mpi communication --> empty call if no mpi compilation
            mpi_buffers->init_communicate(engine_config, backend->state, config); // need to copy between backend & state when using mpi

            //execute the actual work items
            backend->execute_work_items(engine_config, config, (int)step, time);

            //dont check on initializing check on step < 0
            if (!initializing) {
                trajectory_logger->write_output_logs(engine_config, time,
                                                    backend->global_state_now(), /* needed on mpi???: */backend->global_tables_stateNow_f32());
            }

            //dump to CMD CLI
            backend->dump_iteration(config, initializing, time, step);

            //waith for all the MPI communication to be done.
            mpi_buffers->finish_communicate();

            // check on step
            if (!initializing) time += engine_config.dt;

            //swap the double buffering idea.
            backend->swap_buffers();
        }
        metadata.run_time_sec = run_timer.delta();
    }

//----> Print meta overeview
    metadata.print();

//-----> Terminating program
    delete backend;
    delete trajectory_logger;
    delete mpi_buffers;
}
