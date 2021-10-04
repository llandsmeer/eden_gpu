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
#include "mpi_setup.h"

#if defined (__linux__) || defined(__APPLE__)
#include <dlfcn.h> // for dynamic loading
#endif

#ifdef _WIN32
// for dynamic loading and other OS specific stuff
// #include <windows.h> // loaded through Common.h at the moment, TODO break out in Windows specific header 
#endif

extern "C" {	

// do not specify alignment for the pointers, in the generic interface
// cannot specify __restrict__ because it is quietly dropped by compilers ( ! ) when the type is allocated with new
// causing a type mismatch when operator delete(T * __restrict__) is called (then why didn't they drop __restrict__ from there too ??)
// just hope the type "mismatch" won't cause a crash in practice
typedef float * Table_F32;
typedef long long * Table_I64;

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
std::vector<FILE *> open_trajectory_files(EngineConfig & engine_config) {
	// open the logs, one for each logger
	std::vector<FILE *> trajectory_open_files;
	for(auto logger : engine_config.trajectory_loggers){
		#ifdef USE_MPI
		assert( my_mpi.rank == 0);
		#endif
		const char *path = logger.logfile_path.c_str();
		FILE *fout = fopen( path, "wt");
		if(!fout){
			auto errcode = errno;// NB: keep errno right away before it's overwritten
			printf("Could not open trajectory log \"%s\" : %s\n", path, strerror(errcode) );
			exit(1);
		}
		trajectory_open_files.push_back(fout);
	}
    return trajectory_open_files;
}

void print_eden_cli_header() {
	//print logo ^_^
	printf("--- Extensible Dynamics Engine for Networks ---\n");
	#ifndef BUILD_STAMP
	#define BUILD_STAMP __DATE__
	#endif
	printf("Build version " BUILD_STAMP "\n");
}

void setup_mpi(int & argc, char ** & argv) {
#ifdef USE_MPI
	// first of first of all, replace argc and argv
	// Modern implementations may keep MPI args from appearing anyway; non-modern ones still need this
	MPI_Init(&argc, &argv);
	// Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &my_mpi.world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi.rank);
	char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

	if( 1 || config.verbose){
		printf("Hello from processor %s, rank %d out of %d processors\n", processor_name, my_mpi.rank, my_mpi.world_size);
		
		if( my_mpi.rank != 0 ){
			char tmps[555];
			sprintf(tmps, "log_node_%d.gen.txt", my_mpi.rank);
			freopen(tmps,"w",stdout);
			stderr = stdout;
		}
	}
#endif
}

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

    //-------------------> crunch the numbers
    // set up printing in logfiles
    const int column_width = 16;
    char tmps_column[ column_width + 5 ];
    FixedWidthNumberPrinter column_fmt(column_width, '\t', 0);

    std::vector<FILE *> trajectory_open_files = open_trajectory_files(engine_config);

    // prepare engine for crunching

    printf("Allocating state buffers...\n");
    // allocate at least two state vectors, to iterate in parallel
    RawTables::Table_F32 state_one = tabs.global_initial_state; // eliminate redundancy LATER
    RawTables::Table_F32 state_two(tabs.global_initial_state.size(), NAN);

	auto	 		tables_state_f32_one = tabs.global_tables_state_f32_arrays;
	decltype(		tables_state_f32_one )  			tables_state_f32_two;
					tables_state_f32_two.reserve(		tables_state_f32_one.size());
	for( auto tab : tables_state_f32_one )      		tables_state_f32_two.emplace_back( tab.size(), NAN );
	
	auto     		tables_state_i64_one = tabs.global_tables_state_i64_arrays;
	decltype(		tables_state_i64_one)              	tables_state_i64_two;
					tables_state_i64_two.reserve(      	tables_state_i64_one.size() );
	for( auto tab : tables_state_i64_one )      		tables_state_i64_two.emplace_back( tab.size(), 0 );
	// now things need to be done a little differently, since for example trigger(and lazy?) variables of Next ought to be zero for results to make sense

	float *global_state_now = &state_one[0];
	float *global_state_next = &state_two[0];
	
	//also allocate pointer and size vectors, to use instead of silly std::vectors
	auto GetSizePtrTables = []( auto &tablist, auto &pointers, auto &sizes ){
		
		pointers.resize( tablist.size() );
		sizes.resize( tablist.size() );
		for(size_t i = 0; i < tablist.size(); i++){
			pointers[i] = tablist.at(i).data();
			sizes[i] = (long long)tablist.at(i).size();
		}
	};
	std::vector <long long> global_tables_const_f32_sizes; std::vector <Table_F32> global_tables_const_f32_arrays;
	std::vector <long long> global_tables_const_i64_sizes; std::vector <Table_I64> global_tables_const_i64_arrays;
	
	std::vector <long long> global_tables_state_f32_sizes; std::vector <Table_F32> global_tables_stateOne_f32_arrays; std::vector <Table_F32> global_tables_stateTwo_f32_arrays;
	std::vector <long long> global_tables_state_i64_sizes; std::vector <Table_I64> global_tables_stateOne_i64_arrays; std::vector <Table_I64> global_tables_stateTwo_i64_arrays;
	
	GetSizePtrTables(tabs.global_tables_const_f32_arrays, global_tables_const_f32_arrays, global_tables_const_f32_sizes);
	GetSizePtrTables(tabs.global_tables_const_i64_arrays, global_tables_const_i64_arrays, global_tables_const_i64_sizes);
	
	GetSizePtrTables(tables_state_f32_one, global_tables_stateOne_f32_arrays, global_tables_state_f32_sizes);
	GetSizePtrTables(tables_state_i64_one, global_tables_stateOne_i64_arrays, global_tables_state_i64_sizes);
	GetSizePtrTables(tables_state_f32_two, global_tables_stateTwo_f32_arrays, global_tables_state_f32_sizes);
	GetSizePtrTables(tables_state_i64_two, global_tables_stateTwo_i64_arrays, global_tables_state_i64_sizes);
	
	// also, set up the references to the flat vectors
	global_tables_const_f32_arrays[tabs.global_const_tabref] = tabs.global_constants.data();
	global_tables_const_f32_sizes [tabs.global_const_tabref] = tabs.global_constants.size();
	
	global_tables_stateOne_f32_arrays[tabs.global_state_tabref] = state_one.data();
	global_tables_stateTwo_f32_arrays[tabs.global_state_tabref] = state_two.data();
	global_tables_state_f32_sizes    [tabs.global_state_tabref] = state_one.size();
	
	Table_F32 *global_tables_stateNow_f32  = global_tables_stateOne_f32_arrays.data();
	Table_I64 *global_tables_stateNow_i64  = global_tables_stateOne_i64_arrays.data();
	Table_F32 *global_tables_stateNext_f32 = global_tables_stateTwo_f32_arrays.data();
	Table_I64 *global_tables_stateNext_i64 = global_tables_stateTwo_i64_arrays.data();
	
	gettimeofday(&init_end, NULL);
	metadata.init_time_sec = TimevalDeltaSec(init_start, init_end);
	
	if(config.dump_raw_layout){
		
		printf("Constants:\n");
		for(auto val : tabs.global_constants ) printf("%g \t", val); 
		printf("\n");
		
		printf("ConstIdx:\n");
		for(auto val : tabs.global_const_f32_index ) printf("%lld \t", val);
		printf("\n");
		printf("StateIdx:\n");
		for(auto val : tabs.global_state_f32_index ) printf("%lld \t", val);
		printf("\n");
		
		auto PrintTables = [](const auto &index, const auto &arrays){
			size_t next_tabchunk = 0;
			for(size_t i = 0; i < arrays.size() ;i++){
				
				if(next_tabchunk < index.size() && i == (size_t)index.at(next_tabchunk) ){
					printf("%zd", i);
					while( next_tabchunk < index.size() && i == (size_t)index.at(next_tabchunk) ) next_tabchunk++;
				}
				printf(" \t");
				printf(" %16p \t", arrays.at(i).data());
				for(auto val : arrays.at(i) ) printf("%s \t", (presentable_string(val)).c_str());
				printf("\n");
			}
		} ;
		auto PrintRawTables = [](const auto &index, const auto &arrays, const auto &sizes){
			size_t next_tabchunk = 0;
			for(size_t i = 0; i < arrays.size() ;i++){
				
				if(next_tabchunk < index.size() && i == (size_t)index.at(next_tabchunk) ){
					printf("%zd", i);
					while( next_tabchunk < index.size() && i == (size_t)index.at(next_tabchunk) ) next_tabchunk++;
				}
				printf(" \t");
				printf(" %16p \t", arrays.at(i));
				for( std::size_t j = 0; j < (size_t)sizes.at(i); j++ ){
					auto val = arrays.at(i)[j];
					printf("%s \t", (presentable_string(val)).c_str());
				}
				printf("\n");
			}
		} ;
		
		
		printf("TabConstF32: %zd %zd\n", tabs.global_table_const_f32_index.size(), tabs.global_tables_const_f32_arrays.size() );
		PrintTables(tabs.global_table_const_f32_index, tabs.global_tables_const_f32_arrays);
		printf("TabConstI64: %zd %zd\n", tabs.global_table_const_i64_index.size(), tabs.global_tables_const_i64_arrays.size() );
		PrintTables(tabs.global_table_const_i64_index, tabs.global_tables_const_i64_arrays);
		printf("TabStateF32: %zd %zd\n", tabs.global_table_state_f32_index.size(), tabs.global_tables_state_f32_arrays.size() );
		PrintTables(tabs.global_table_state_f32_index, tabs.global_tables_state_f32_arrays);
		printf("TabStateI64: %zd %zd\n", tabs.global_table_state_i64_index.size(), tabs.global_tables_state_i64_arrays.size() );
		PrintTables(tabs.global_table_state_i64_index, tabs.global_tables_state_i64_arrays);
		
		printf("RawStateI64:\n");
		PrintRawTables(tabs.global_table_state_i64_index, global_tables_stateOne_i64_arrays, global_tables_state_i64_sizes);
		
		printf("CallIdx:\n");
		for(auto val : tabs.callbacks ) printf("%p \t", val);
		printf("\n");
		
		
		printf("Initial state:\n");
		printf("TabStateOneF32:\n");
		PrintTables(tabs.global_table_state_f32_index, tables_state_f32_one);
		printf("TabStateOneI64:\n");
		PrintTables(tabs.global_table_state_i64_index, tables_state_i64_one);
		printf("TabStateTwoF32:\n");
		PrintTables(tabs.global_table_state_f32_index, tables_state_f32_two);
		printf("TabStateTwoI64:\n");
		PrintTables(tabs.global_table_state_i64_index, tables_state_i64_two);
		printf("Initial scalar state:\n");
		for(auto val : tabs.global_initial_state ) printf("%g \t", val);
		printf("\n");
		
	}
	// MPI_Finalize();
	// exit(1);
	
	#ifdef USE_MPI
	
	printf("Allocating comm buffers...\n");
	typedef std::vector<float> SendRecvBuf;
	std::vector<int> send_off_to_node;
	std::vector< SendRecvBuf > send_bufs;
	std::vector<int> recv_off_to_node;
	std::vector< SendRecvBuf > recv_bufs;
	
	for( const auto &keyval : engine_config.sendlist_impls ){
		send_off_to_node.push_back( keyval.first );
		send_bufs.emplace_back();
		// allocate as they come, why not
	}
	
	for( const auto &keyval : engine_config.recvlist_impls ){
		recv_off_to_node.push_back( keyval.first );
		recv_bufs.emplace_back();
		// allocate as they come, why not
	}
	std::vector<MPI_Request> send_requests( send_off_to_node.size(), MPI_REQUEST_NULL );
	std::vector<MPI_Request> recv_requests( recv_off_to_node.size(), MPI_REQUEST_NULL );
	
	// recv's have to be probed before recv'ing
	std::vector<bool> received_probes( recv_off_to_node.size(), false);	
	std::vector<bool> received_sends( recv_off_to_node.size(), false);
	
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
		auto NetMessage_ToString = []( size_t buf_value_len, const auto &buf ){
			std::string str;
			for( size_t i = 0; i < buf_value_len; i++ ){
				str += presentable_string( buf[i] ) + " ";
			}
			str += "| ";
			for( size_t i = buf_value_len; i < buf.size(); i++ ){
				str += presentable_string( EncodeF32ToI32( buf[i] ) ) + " ";
			}
			return str;
		};
		// Send info needed by other nodes
		// TODO try parallelizing buffer fill, see if it improves latency
		for( size_t idx = 0; idx < send_off_to_node.size(); idx++ ){
			auto other_rank = send_off_to_node.at(idx);
			const auto &sendlist_impl = engine_config.sendlist_impls.at(other_rank);
			
			auto &buf = send_bufs[idx];
			auto &req = send_requests[idx];
			
			// get the continuous_time values
			size_t vpeer_buf_idx = 0;
			size_t vpeer_buf_len = sendlist_impl.vpeer_positions_in_globstate.size();
			
			size_t daw_buf_idx = vpeer_buf_idx + vpeer_buf_len;
			size_t daw_buf_len = sendlist_impl.daw_columns.size();
			
			size_t buf_value_len = vpeer_buf_len + daw_buf_len;
			
			buf.resize(buf_value_len); // std::vector won't reallocate due to size reduction
			// otherwise implement appending the spikes manually
			
			// NB make sure these buffers are synchronized with CPU memory LATER
			for( size_t i = 0; i < sendlist_impl.vpeer_positions_in_globstate.size(); i++ ){
				size_t off = sendlist_impl.vpeer_positions_in_globstate[i];
				
				buf[ vpeer_buf_idx + i ] = global_state_now[off];
			}
			
			for( size_t i = 0; i < sendlist_impl.daw_columns.size(); i++ ){
				assert( my_mpi.rank != 0 && other_rank == 0 );
				auto &col = sendlist_impl.daw_columns[i];
				size_t off = col.entry;
				// also apply scaling, so receiving node won't bother
				
				buf[ daw_buf_idx + i ] = global_state_now[ off ] * col.scaleFactor ;
			}
			
			size_t spikebuf_off = sendlist_impl.spike_mirror_buffer;
			// get the spikes into the buffer (variable size)
			Table_I64 SpikeTable = global_tables_stateNow_i64[spikebuf_off];
			long long SpikeTable_size = global_tables_state_i64_sizes[spikebuf_off];
			
			for( int i = 0; i < SpikeTable_size; i++ ){
			
				// TODO packed bool buffers
				if( SpikeTable[i] ){
					// add index	
					buf.push_back(  EncodeI32ToF32(i) );
					// clear trigger flag for the timestep after the next one
					SpikeTable[i] = 0;
				}
			}
			if( config.debug_netcode ){
				Say("Send %d : %s", other_rank, NetMessage_ToString( buf_value_len, buf).c_str());
			}
			MPI_Isend( buf.data(), buf.size(), MPI_FLOAT, other_rank, MYMPI_TAG_BUF_SEND, MPI_COMM_WORLD, &req );
		}
		
		// Recv info needed by this node
		auto PostRecv = []( int other_rank, std::vector<float> &buf, MPI_Request &recv_req ){
			MPI_Irecv( buf.data(), buf.size(), MPI_FLOAT, other_rank, MYMPI_TAG_BUF_SEND, MPI_COMM_WORLD, &recv_req );
		};
		auto ReceiveList = [ &engine_config, &global_tables_stateNow_f32, &global_tables_stateNow_i64 ]( const EngineConfig::RecvList_Impl &recvlist_impl, std::vector<float> &buf ){
			
			// copy the continuous-time values
			size_t value_buf_idx = 0;
			float *value_buf = global_tables_stateNow_f32[ recvlist_impl.value_mirror_buffer ];
			// NB make sure these buffers are synchronized with CPU memory LATER
			for( ptrdiff_t i = 0; i < recvlist_impl.value_mirror_size; i++ ){
				value_buf[i] = buf[ value_buf_idx + i ];
				
				// global_state_now[ off + i] = buf[ value_buf_idx + i ];
			}
			
			// and deliver the spikes to trigger buffers
			for( int i = recvlist_impl.value_mirror_size; i < (int)buf.size(); i++ ){
				int spike_pos = EncodeF32ToI32( buf[i] );
				for( auto tabent_packed : recvlist_impl.spike_destinations[spike_pos] ){
					auto tabent = GetDecodedTableEntryId( tabent_packed );
					// TODO packed bool buffers
					global_tables_stateNow_i64[tabent.table][tabent.entry] = 1;
				}
			}
			
			// all done with message
		};
		// TODO min_delay option when no gap junctions exist
		// Also wait for recvs to finish
		// Spin it all, to probe for multimple incoming messages
		
		bool all_received = true;
		do{
			all_received = true; // at least for the empty set examined before the loop
			// TODO also try parallelizing this, perhaps?
			for( size_t idx = 0; idx < recv_off_to_node.size(); idx++ ){
				auto other_rank = recv_off_to_node.at(idx);
				const auto &recvlist_impl = engine_config.recvlist_impls.at(other_rank);
				
				if( received_sends[idx] ) continue;
				
				// otherwise it's pending
				all_received = false;
				
				
				auto &buf = recv_bufs[idx];
				auto &req = recv_requests[idx];
				
				if( received_probes[idx] ){
					// check if recv is done
					int flag = 0;
					MPI_Status status;
					MPI_Test( &req, &flag, &status);
					if( flag ){
						// received, yay !
						// Say("Recv %d.%zd", other_rank,  recvlist_impl.value_mirror_size);
						if( config.debug_netcode ){
							Say("Recv %d : %s", other_rank, NetMessage_ToString( recvlist_impl.value_mirror_size, buf).c_str());
						}
						ReceiveList( recvlist_impl, buf );
						received_sends[idx] = true;
					}
				}
				else{
					// check if probe is ready
					int flag = 0;
					MPI_Status status;
					MPI_Iprobe( other_rank, MYMPI_TAG_BUF_SEND, MPI_COMM_WORLD, &flag, &status);
					if( flag ){
						int buf_size;
						MPI_Get_count( &status, MPI_FLOAT, &buf_size );
						buf.resize( buf_size );
						PostRecv( other_rank, buf, req );
						received_probes[idx] = true;
					}
				}
				
			}
			// MPI_Finalize();
			// 	exit(1);
		} while( !all_received );
		// and clear the progress flags
		received_probes.assign( received_probes.size(), false );
		received_sends .assign( received_sends .size(), false );
		
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
				tabs.global_constants        .data(), tabs.global_const_f32_index      [item],
				global_tables_const_f32_sizes.data(), global_tables_const_f32_arrays.data(), tabs.global_table_const_f32_index[item],
				global_tables_const_i64_sizes.data(), global_tables_const_i64_arrays.data(), tabs.global_table_const_i64_index[item],
				global_tables_state_f32_sizes.data(), global_tables_stateNow_f32, global_tables_stateNext_f32, tabs.global_table_state_f32_index[item],
				global_tables_state_i64_sizes.data(), global_tables_stateNow_i64, global_tables_stateNext_i64, tabs.global_table_state_i64_index[item],
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
			for(size_t i = 0; i < engine_config.trajectory_loggers.size(); i++){
				#ifdef USE_MPI
				assert(my_mpi.rank == 0);
				#endif
				const auto &logger = engine_config.trajectory_loggers[i];
				FILE *& fout = trajectory_open_files[i];
				
				const ScaleEntry seconds = {"sec",  0, 1.0};
				const double time_scale_factor = Scales<Time>::native.ConvertTo(1, seconds);
				
				float time_val = time * time_scale_factor;
				column_fmt.write( time_val, tmps_column );
				fprintf(fout, "%s", tmps_column);
				
				auto GetColumnValue = [
				#ifdef USE_MPI
					&engine_config,
					&global_tables_stateNow_f32,
				#endif
					&global_state_now
				]( const EngineConfig::TrajectoryLogger::LogColumn &column ){
					
					switch( column.type ){
						case EngineConfig::TrajectoryLogger::LogColumn::Type::TOPLEVEL_STATE :{
							if(column.value_type == EngineConfig::TrajectoryLogger::LogColumn::ValueType::F32){
								#ifdef USE_MPI
								if( column.on_node >= 0 && column.on_node != my_mpi.rank ){
									size_t table = engine_config.recvlist_impls.at(column.on_node).value_mirror_buffer;
									
									// scaling is done on remote node
									return global_tables_stateNow_f32[table][column.entry];
								}
								#endif
								// otherwise a local one
								return (float) (global_state_now[column.entry] * column.scaleFactor);
							}
							else if(column.value_type == EngineConfig::TrajectoryLogger::LogColumn::ValueType::I64){
								printf("but i have no flat i64 states lol\n");
								exit(2);
							}
							else{
								printf("internal error: unknown value type\n");
								exit(2);
							}
							
							break;
						}
						case EngineConfig::TrajectoryLogger::LogColumn::Type::TABLE_STATE :
						default:
							printf("internal error: unknown log type\n");
							exit(2);
					}
				};
				for( const auto &column : logger.columns ){
					char tmps_column[ column_width + 5 ];
					
					float col_val = GetColumnValue(column);
					column_fmt.write( col_val, tmps_column );
					fprintf( fout, "\t%s", tmps_column );
					// fprintf( fout, "\t%f", col_val );
					
				}
				fprintf(fout, "\n");
			}
		}
		
		// output state dump
		if(
			config.dump_raw_state_scalar
			|| config.dump_raw_state_table
		){
			if( !initializing ){
				printf("State: t = %g %s\n", time, Scales<Time>::native.name);
			}
			else{
				printf("State: t = %g %s, initialization step %lld\n", time, Scales<Time>::native.name, step);
			}
		}
		if(config.dump_raw_state_scalar){
			// print state, separated by work item
			for( size_t i = 0, itm = 1; i < state_one.size(); i++ ){
				printf("%g \t", global_state_next[i]);
				while( itm < tabs.global_state_f32_index.size() && (i + 1) == (size_t)tabs.global_state_f32_index[itm] ){
					printf("| ");
					itm++;
				}
			}
			printf("\n");
		}
		if(config.dump_raw_state_table){
			
			auto PrintVeryRawTables = [](const auto &index, const auto &arrays, const auto &sizes){
				size_t next_tabchunk = 0;
				for(size_t i = 0; i < sizes.size() ;i++){
					
					if(next_tabchunk < index.size() && i == (size_t)index.at(next_tabchunk) ){
						printf("%zd", i);
						while( next_tabchunk < index.size() && i == (size_t)index.at(next_tabchunk) ) next_tabchunk++;
					}
					printf(" \t");
					printf(" %16p \t", arrays[i]);
					for( std::size_t j = 0; j < (size_t)sizes.at(i); j++ ){
						auto val = arrays[i][j];
						printf("%s \t", (presentable_string(val)).c_str());
					}
					printf("\n");
				}
			} ;
			
			printf("RawStateF32:\n");
			PrintVeryRawTables(tabs.global_table_state_f32_index, global_tables_stateNow_f32, global_tables_state_f32_sizes);
			printf("RawStateI64:\n");
			PrintVeryRawTables(tabs.global_table_state_i64_index, global_tables_stateNow_i64, global_tables_state_i64_sizes);
			printf("RawStateNextF32:\n");
			PrintVeryRawTables(tabs.global_table_state_f32_index, global_tables_stateNext_f32, global_tables_state_f32_sizes);
			printf("RawStateNextI64:\n");
			PrintVeryRawTables(tabs.global_table_state_i64_index, global_tables_stateNext_i64, global_tables_state_i64_sizes);
		}
		
		#ifdef USE_MPI
		// wait for sends, to finish the iteration
		MPI_Waitall( send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE );
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
	
	// close loggers
	for( auto &fout : trajectory_open_files ){
		fclose(fout);
		fout = NULL;
	}
	
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
