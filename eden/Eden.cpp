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

/* todo:
 *      gebruik std::async ipv thread ! :D  research if it's better
 *      input flag's nakijken
 *
 *  */

//
#include <thread>

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


int main(int argc, char **argv){
//-----> declaration of all used variables
    RunMetaData metadata;             // Struct for keeping track of non backend specific meta data
    SimulatorConfig config;           // Struct for Configuration of the simulator
    Model model;                      // Struct for Keeping track of the model
    EngineConfig engine_config;       // Struct for Configuration of the Engine

    AbstractBackend *backend = nullptr;             // Class to handle all backend calls
    TrajectoryLogger *trajectory_logger = nullptr;  // Class to handle all output generation
    MpiBuffers *mpi_buffers = nullptr;              // Class to handle all MPI communication

//----> SETUP MPI first. This will also set the log file.
    setup_mpi(argc, argv, &engine_config);         //check if everything works fine, sorry if you use legacy cmd line args

//-----> Starting the simulator
    print_eden_cli_header(engine_config.log_context);          // To print the CLI header

//-----> Check the command line input with options
    parse_command_line_args(argc, argv, engine_config, config, model, metadata.config_time_sec);

//-----> Initialize the logger
    {
        if(engine_config.log_to_file && !engine_config.log_context.log_file.is_open()) {
            char tmps[555];
            engine_config.log_context.mpi_rank = engine_config.my_mpi.rank;
            sprintf(tmps, "log_rank_%d.gen.txt", engine_config.log_context.mpi_rank);
            engine_config.log_context.log_file.open(tmps);
        }
    }
    INIT_LOG(&engine_config.log_context.log_file,engine_config.log_context.mpi_rank);
    log(LOG_INFO) << "Hello from processor " << engine_config.my_mpi.processor_name <<", rank " << engine_config.my_mpi.rank << " out of " << engine_config.my_mpi.world_size << LOG_ENDL;

//-----> Init the backend
    log(LOG_MES) << "Initializing backend... "<< LOG_ENDL;
    {
        if (engine_config.backend == backend_kind_gpu) {
#ifdef USE_GPU
            if(!test_GPU(engine_config.log_context)){
                engine_config.backend = backend_kind_cpu;
                log(LOG_ERR) << "NO GPU FOUND ~ USING BACKEND CPU" << LOG_ENDL;
            }else{
                log(LOG_INFO) << "USING BACKEND GPU" << LOG_ENDL;
            }
#else
            log(LOG_ERR) << "NOT COMPILED WITH GPU SUPPORT ~ USING BACKEND CPU" << LOG_ENDL;
#endif
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

//----> Initialize the model
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
    std::thread Write_Output_Thread;

    log(LOG_MES) << "Starting simulation loop..."<< LOG_ENDL;
    {

        Timer run_timer;
        size_t total_steps = ceil((engine_config.t_final -engine_config.t_initial)/engine_config.dt);
        double time = engine_config.t_initial;

        // need multiple initialization steps, to make sure the dependency chains of all state variables are resolved
        for (long long step = -3; time <= engine_config.t_final; step++) {

            // Start and check the output logger
            if(step > 1){
                if (Write_Output_Thread.joinable()) Write_Output_Thread.join();
                backend->populate_print_buffer();
                auto sn_f32 = engine_config.use_mpi ? backend->print_tables_stateNow_f32() : nullptr;
//             threading

                Write_Output_Thread = std::thread(
                        &TrajectoryLogger::write_output_logs,
                        trajectory_logger,
                        &engine_config,
                        time - engine_config.dt,
                        backend->print_state_now(),
                        sn_f32 );

//                Write_Output_Thread = std::thread(
//                        &OutputMonitor::PrintTimestep,
//                        sim.output_monitor,
//                        NetworkState_l[0].hostVs_print,
//                        NetworkState_l[0].hostYs_print,
//                        NetworkState_l[0].hostCalc_print,
//                        NetworkState_l[0].hostCurrents_print,
//                        sim.timestep,
//                        step,
//                        nCE,
//                        nCO,
//                        nCh,
//                        nGA,
//                        (int) (sim.time / sim.timestep),
//                        &runMetaData.OutputWriteTime_ms);


//              no treading:
//                trajectory_logger->write_output_logs(engine_config, time - engine_config.dt,
//                                 backend->print_state_now(),
//                                 sn_f32);
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

            //some progress output
            if(step > 0 && !(step % (total_steps/10)))
                log(LOG_INFO) << "Progress: " <<  (float)step/(float)total_steps*100 << " %" << LOG_ENDL;
        }

        //----> fix the last printing to the outputfile one can just select the global_state_now for this.
        if (Write_Output_Thread.joinable()) Write_Output_Thread.join();  //join the last launched thread :D
        backend->populate_print_buffer();
        auto sn_f32 = engine_config.use_mpi ? backend->print_tables_stateNow_f32() : 0;
        trajectory_logger->write_output_logs(&engine_config, time-engine_config.dt,
                                             backend->print_state_now(),
                /* needed on mpi: */sn_f32);

        metadata.run_time_sec = run_timer.delta();
    }

//----> Print meta overeview
    log(LOG_MES) << "Stopping simulation loop..." << LOG_ENDL;
    log(LOG_TIME) << "Timing:" << LOG_ENDL;
    log(LOG_TIME) << "   initTime   " << metadata.init_time_sec << LOG_ENDL;
    log(LOG_TIME) << "   configTime " << metadata.config_time_sec << LOG_ENDL;
    log(LOG_TIME) << "   save_time  " << metadata.save_time_sec << LOG_ENDL;
    log(LOG_TIME) << "   runTime    " << metadata.run_time_sec << LOG_ENDL;
    log(LOG_TIME) << "MEMORY:" << LOG_ENDL;
    log(LOG_TIME) << "   peak resident memory in bytes:    " << getPeakResidentSetBytes() << LOG_ENDL;

//-----> Terminating program
    engine_config.log_context.log_file.close();
    delete backend;
    delete trajectory_logger;
    delete mpi_buffers;
}
