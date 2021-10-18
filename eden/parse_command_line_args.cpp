#include "parse_command_line_args.h"
#include "../thirdparty/miniLogger/miniLogger.h"

void print_eden_cli_header() {
    INIT_LOG();
    log(LOG_OVERWRITE) << "       ###########    ###############              ###########    ######     #########       "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "     ###############   ##################        ###############    #####  ##############    "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "   #####          ####      #####    ######    #####          ####     ######       ######   "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "  ####                    ####          ####  ####                     ####           #####  "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "  ###############         ###            ###  ###############          ####            ####  "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "  ############           ####            ###  ############             ###             ####  "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "  ####                   ####            ###  ####                     ###             ####  "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "  ####              #    ####           ####  ####              #     ####             ####  "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "   #####          ###     ####         ####    #####          ###     ####            ####   "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "    ################        ##############      ################     ####            ####    "<< LOG_ENDL;
    log(LOG_OVERWRITE) << "      ##########              ##########          ##########        ####            #####    "<< LOG_ENDL << LOG_ENDL;
    log(LOG_OVERWRITE) << "--- Extensible Dynamics Engine for Networks ---" << LOG_ENDL << LOG_ENDL;
#ifndef BUILD_STAMP
#define BUILD_STAMP __DATE__
#endif
    log(LOG_INFO) << "Build version " <<  BUILD_STAMP  << LOG_ENDL;
}

void parse_command_line_args(int argc, char ** argv, EngineConfig & engine_config, SimulatorConfig & config, Model & model, double & config_time_sec) {
	INIT_LOG();
    timeval config_start, config_end;
	gettimeofday(&config_start, NULL);
	bool model_selected = false;
	//-------------------> Read config options for run
	for (int i = 1; i < argc; i++){
		std::string arg(argv[i]);
		
		// model options
		if(arg == "nml"){
			//read JSON
			if(i == argc - 1){
			    log(LOG_ERR) << "\"cmdline: NeuroML filename missing\"" << LOG_ENDL;
				exit(1);
			}
			
			timeval nml_start, nml_end;
			gettimeofday(&nml_start, NULL);
			if(!( ReadNeuroML(argv[i+1], model, true) )){
                log(LOG_ERR) << "cmdline: could not make sense of NeuroML file" << LOG_ENDL;
				exit(1);
			}
			gettimeofday(&nml_end, NULL);
            log(LOG_DEBUG) << "cmdline: Parsed "<<  argv[i+1]  << " in " << TimevalDeltaSec(nml_start, nml_end) << " seconds" << LOG_ENDL;
			model_selected = true;
			i++;
		}
		// model overrides
		else if(arg == "rng_seed"){
			if(i == argc - 1){
                log(LOG_ERR) << "cmdline: "<<  arg.c_str() << "value missing" << LOG_ENDL;
				exit(1);
			}
			const std::string sSeed = argv[i+1];
			long long seed;
			if( sscanf( sSeed.c_str(), "%lld", &seed ) == 1 ){
				config.override_random_seed = true;
				config.override_random_seed_value = seed;
			}
			else{
                log(LOG_ERR) <<"cmdline: "<< arg.c_str() <<" must be a reasonably-sized integer, not " << sSeed.c_str() << LOG_ENDL;
				exit(1);
			}
			
			i++; // used following token too
		}
		// solver options
		else if(arg == "cable_solver"){
			//read JSON
			if(i == argc - 1){
			    log(LOG_ERR) <<"cmdline: "<< arg.c_str() <<" type missing" << LOG_ENDL;
				exit(1);
			}
			const std::string soltype = argv[i+1];
			if( soltype == "fwd_euler" ){
                log(LOG_WARN) << "Cable solver set to Forward Euler: make sure system is stable\n" << LOG_ENDL;
				config.cable_solver = SimulatorConfig::CABLE_FWD_EULER;
			}
			else if( soltype == "bwd_euler" ){
				config.cable_solver = SimulatorConfig::CABLE_BWD_EULER;
			}
			else if( soltype == "auto" ){
				config.cable_solver = SimulatorConfig::CABLE_SOLVER_AUTO;
			}
			else{
			    log(LOG_ERR) <<"cmdline: unknown  " << arg.c_str() << "  type " << soltype.c_str() << " choices are auto, fwd_euler, bwd_euler" << LOG_ENDL;
				exit(1);
			}
			i++; // used following token too
		}
		// debugging options
		else if(arg == "verbose"){
			config.verbose = true;
		}
		else if(arg == "full_dump"){
			config.dump_raw_state_scalar = true;
			config.dump_raw_state_table = true;
			config.dump_raw_layout = true;
		}
		else if(arg == "dump_state_scalar"){
			config.dump_raw_state_scalar = true;
		}
		else if(arg == "dump_raw_layout"){
			config.dump_raw_layout = true;
		}
		else if(arg == "debug"){
			config.debug = true;
			config.debug_netcode = true;
		}
		else if(arg == "debug_netcode"){
			config.debug_netcode = true;
		}
        else if(arg == "debug_gpu_kernels") {
            config.debug_gpu_kernels = true;
        }
		else if(arg == "-S"){
			config.output_assembly = true;
		}
		// optimization, compiler options
		else if(arg == "icc"){
			config.use_icc = true;
		}
		else if(arg == "gcc"){
			config.use_icc = false;
		}
        else if(arg == "single-kernels") {
            config.skip_combining_consecutive_kernels = true;
        }
        else if(arg == "syscall-guard") {
            config.syscall_guard_callback = true;
        }
        else if(arg == "dump_array_locations") {
            config.dump_array_locations = true;
        }
        else if(arg == "gpu") {
            engine_config.backend = backend_kind_gpu;
        }
		else{
			//unknown, skip it
            log(LOG_WARN) << "cmdline: skipping unknow token " << argv[i] << LOG_ENDL;
		}
	}
	// handle model missing
	if(!model_selected){
		log(LOG_ERR) << "NeuroML model not selected (select one with nml <file> in command line)" << LOG_ENDL;
		exit(2);
	}
	gettimeofday(&config_end, NULL);
	config_time_sec = TimevalDeltaSec(config_start, config_end);
}
