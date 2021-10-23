void parse_command_line_args(int argc, char ** argv, SimulatorConfig & config, Model & model, double & config_time_sec) {
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
				printf("cmdline: NeuroML filename missing\n");
				exit(1);
			}
			
			timeval nml_start, nml_end;
			gettimeofday(&nml_start, NULL);
			if(!( ReadNeuroML(argv[i+1], model, true) )){
				printf("cmdline: could not make sense of NeuroML file\n");
				exit(1);
			}
			gettimeofday(&nml_end, NULL);
			printf("cmdline: Parsed %s in %lf seconds\n", argv[i+1], TimevalDeltaSec(nml_start, nml_end));
			
			model_selected = true;
			
			i++;
		}
		// model overrides
		else if(arg == "rng_seed"){
			if(i == argc - 1){
				printf( "cmdline: %s value missing\n", arg.c_str() );
				exit(1);
			}
			const std::string sSeed = argv[i+1];
			long long seed;
			if( sscanf( sSeed.c_str(), "%lld", &seed ) == 1 ){
				config.override_random_seed = true;
				config.override_random_seed_value = seed;
			}
			else{
				printf( "cmdline: %s must be a reasonably-sized integer, not %s\n", arg.c_str(), sSeed.c_str() );
				exit(1);
			}
			
			i++; // used following token too
		}
		// solver options
		else if(arg == "cable_solver"){
			//read JSON
			if(i == argc - 1){
				printf( "cmdline: %s type missing\n", arg.c_str() );
				exit(1);
			}
			const std::string soltype = argv[i+1];
			if( soltype == "fwd_euler" ){
				printf("Cable solver set to Forward Euler: make sure system is stable\n");
				config.cable_solver = SimulatorConfig::CABLE_FWD_EULER;
			}
			else if( soltype == "bwd_euler" ){
				config.cable_solver = SimulatorConfig::CABLE_BWD_EULER;
			}
			else if( soltype == "auto" ){
				config.cable_solver = SimulatorConfig::CABLE_SOLVER_AUTO;
			}
			else{
				printf( "cmdline: unknown %s type %s; "
					"choices are auto, fwd_euler, bwd_euler\n", arg.c_str(),  soltype.c_str() );
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
        else if(arg == "dump_array_locations") {
            config.dump_array_locations = true;
        }
		else if(arg == "debug"){
			config.debug = true;
			config.debug_netcode = true;
		}
		else if(arg == "debug_netcode"){
			config.debug_netcode = true;
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
		else{
			//unknown, skip it
			printf("cmdline: skipping unknown token \"%s\"\n", argv[i]);
		}
	}
	
	// handle model missing
	if(!model_selected){
		printf("error: NeuroML model not selected (select one with nml <file> in command line)\n");
		exit(2);
	}
	gettimeofday(&config_end, NULL);
	config_time_sec = TimevalDeltaSec(config_start, config_end);
}
