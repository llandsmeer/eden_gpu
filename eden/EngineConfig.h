#ifndef ENGINECONFIG_H
#define ENGINECONFIG_H

#include "TableEntry.h"

extern "C" {

typedef uint32_t backend_kind;
#define backend_kind_nil 0
#define backend_kind_cpu 1
#define backend_kind_gpu 2

// and more information that is needed for the engine
struct EngineConfig{
	
	struct TrajectoryLogger {
		struct LogColumn{
			enum Type{
				NONE,
				TOPLEVEL_STATE,
				TABLE_STATE
			};
			enum ValueType{
				F32,
				I64
			};
			
			Type type;
			ValueType value_type;
			
			size_t entry;
			// size_t table;
			double scaleFactor; // in case of float values
			
			#ifdef USE_MPI
			int on_node;
			LogColumn(){
				on_node = -1;
			}
			#endif
		};
		
		std::string logfile_path;
		std::vector<LogColumn> columns;
		
	};

	long long work_items;
	double t_initial; // in engine time units
	double t_final;
	float dt; // in engine time units
    backend_kind backend = backend_kind_cpu;

	std::vector<TrajectoryLogger> trajectory_loggers;
	
	// for inter-node communication
	struct SendList_Impl{
		
		// std::vector<int> vpeer_positions_in_buffer;
		std::vector<size_t> vpeer_positions_in_globstate; // TODO packed table entries
		
		std::vector< TrajectoryLogger::LogColumn > daw_columns;
		// std::vector<int> daw_positions_in_buffer;
		// std::vector<size_t> daw_positions_in_globstate; // TODO packed table entries
		
		size_t spike_mirror_buffer;
	};
	struct RecvList_Impl{
		size_t value_mirror_buffer;
		ptrdiff_t value_mirror_size;
		// refs to mirror_buffer in table accesses, as well as off-table trajectory loggers, are resolved
		
		std::vector< std::vector<TabEntryRef_Packed> > spike_destinations;
	};
	std::map< int, SendList_Impl > sendlist_impls;
	std::map< int, RecvList_Impl > recvlist_impls;
	
	//spikes are triggered in buffer, they're automatically gathered
	
};
}

#endif