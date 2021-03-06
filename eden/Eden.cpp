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


//Fixen van MPI meuk voor output logger

//standard libs

// Local includes
#include "Common.h"
#include "NeuroML.h"
#include "MMMallocator.h"
#include "GPU_helpers.h"
#include "backends/cpu/CpuBackend.h"
#include "backends/gpu/GpuBackend.h"
#include "GenerateModel.h"
#include "EngineConfig.h"
#include "SimulatorConfig.h"
#include "Mpi_helpers.h"
#include "TrajectoryLogger.h"
#include "parse_command_line_args.h"
#include "../thirdparty/miniLogger/miniLogger.h"

int main(int argc, char **argv){
    INIT_LOG();

//-----> Starting the simulator
    print_eden_cli_header();          // To print the CLI header

//-----> declaration of all used variables
    RunMetaData metadata;             // Struct for keeping track of non backend specific meta data
    SimulatorConfig config;           // Struct for Configuration of the simulator
    Model model;                      // Struct for Keeping track of the model
    EngineConfig engine_config;       // Struct for Configuration of the Engine

    AbstractBackend *backend = nullptr;             // Class to handle all backend calls
    TrajectoryLogger *trajectory_logger = nullptr;  // Class to handle all output generation
    MpiBuffers *mpi_buffers = nullptr;              // Class to handle all MPI communication

//-----> Check the command line input with options
    log(LOG_MES) << "Parse command lines and Build model"<< LOG_ENDL;
    parse_command_line_args(argc, argv, engine_config, config, model, metadata.config_time_sec);

//-----> Find and check the specific engine_config
    setup_mpi(argc, argv, &engine_config);         //check if everything works fine, sorry if you use legacy cmd line args
    if (engine_config.backend == backend_kind_gpu) {
        setup_gpu(engine_config);                   //same for gpu
    }

//-----> Init the backend
    log(LOG_MES) << "Initializing backend... "<< LOG_ENDL;
    {
        if (engine_config.backend == backend_kind_gpu) {
            log(LOG_INFO) << "USING BACKEND GPU" << LOG_ENDL;
        } else {
            log(LOG_INFO) << "USING BACKEND CPU" << LOG_ENDL;
        }
        if (engine_config.backend == backend_kind_cpu) {
            backend = new CpuBackend();
        } else if (engine_config.backend == backend_kind_gpu) {
            backend = new GpuBackend();
        } else {
            log(LOG_ERR)<< "No valid backed selected" <<LOG_ENDL;
            exit(10);
        }
    }

//----> Initialize the backend
    log(LOG_MES) << "Initializing model... "<< LOG_ENDL;
    {
        Timer init_timer;
        if (!GenerateModel(model, config, engine_config, backend->tabs)) {
            log(LOG_ERR) << "NeuroML model could not be created\n" << LOG_ENDL;
            exit(1);
        }
        trajectory_logger = new TrajectoryLogger(engine_config); //To log results

        log(LOG_INFO) << "Allocating state buffers..." << LOG_ENDL;
        backend->init();

        //just some timer functions to time this meta data --> encorperate this into meta data class
        metadata.init_time_sec = init_timer.delta();

        //debug mem layout
        if(config.dump_raw_layout) backend->state->dump_raw_layout(backend->tabs);
        if(config.dump_array_locations) backend->state->dump_array_locations(backend->tabs);

        //Initialize MPI
         mpi_buffers = new MpiBuffers(engine_config);
    }

//----> Simulations loop
    log(LOG_MES) << "Starting simulation loop..."<< LOG_ENDL;
    {
        Timer run_timer;
        double time = engine_config.t_initial;
        // need multiple initialization steps, to make sure the dependency chains of all state variables are resolved
        for (long long step = -3; time <= engine_config.t_final; step++) {

//            Start and check the output logger
            if(step > 1){
                backend->populate_print_buffer();
                auto sn_f32 = engine_config.use_mpi ? backend->print_tables_stateNow_f32() : nullptr;
                trajectory_logger->write_output_logs(engine_config, time - engine_config.dt,
                                                     backend->print_state_now(),
                                                    /* needed on mpi: */sn_f32);
            }

            //init mpi communication --> empty call if no mpi compilation
            mpi_buffers->init_communicate(engine_config, backend, config); // need to copy between backend & state when using mpi

            //execute the actual work items
            backend->execute_work_items(engine_config, config, (int)step, time);

            //dump to CMD CLI
            backend->dump_iteration(config, (step <= 0), time, step);

            //waith for all the MPI communication to be done.
            mpi_buffers->finish_communicate(engine_config);

            // check on step
            if (step > 0) time += engine_config.dt;

            //synchronize the backend.
            backend->synchronize();

            //swap the double buffering idea.
            backend->swap_buffers();

//            getchar();
        }

        //----> fix the last printing to the outputfile one can just select the global_state_now for this.
        backend->populate_print_buffer();
        auto sn_f32 = engine_config.use_mpi ? backend->print_tables_stateNow_f32() : 0;
        trajectory_logger->write_output_logs(engine_config, time-engine_config.dt,
                                             backend->print_state_now(),
                /* needed on mpi: */sn_f32);

        metadata.run_time_sec = run_timer.delta();
    }




//----> Print meta overeview
    metadata.print();

//-----> Terminating program
    delete backend;
    delete trajectory_logger;
    delete mpi_buffers;
}
