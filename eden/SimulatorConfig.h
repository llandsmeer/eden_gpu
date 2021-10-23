#ifndef EDEN_SIMULATOR_CONFIG
#define EDEN_SIMULATOR_CONFIG

extern "C" {
// general options for the simulator
struct SimulatorConfig{
	
	bool override_random_seed;
	long long override_random_seed_value;
	
	bool verbose;// TODO implement in more places	
	
	bool debug;
	bool debug_netcode;
	
	bool dump_raw_state_scalar;
	bool dump_raw_state_table;
	bool dump_raw_layout;
	bool dump_array_locations = false;
	
	bool use_icc;
	bool tweak_lmvec;
	bool output_assembly;

    bool skip_combining_consecutive_kernels = false;
    bool syscall_guard_callback = false;
	
	// TODO knobs:
	// vector vs.hardcoded sequence for bwd euler, also heuristic
	enum CableEquationSolver{
		CABLE_SOLVER_AUTO,
		CABLE_FWD_EULER,
		CABLE_BWD_EULER,
	};
	CableEquationSolver cable_solver;
	
	SimulatorConfig(){
		verbose = false;
		debug = false;
		debug_netcode = false;
		
		dump_raw_state_scalar = false;
		dump_raw_state_table = false;
		dump_raw_layout = false;
		
		use_icc = false;
		tweak_lmvec = false;
		output_assembly = false;
		
		cable_solver = CABLE_SOLVER_AUTO;
		
		override_random_seed = false;
	}
};
}

#endif
