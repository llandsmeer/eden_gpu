#ifdef USE_MPI
#include <mpi.h>
// MPI context, just a few globals like world size, rank etc.
struct MpiContext{
	int world_size;
	int rank;
};
const int MYMPI_TAG_BUF_SEND = 99;
MpiContext my_mpi;


// TODO use logging machinery in whole codebase
FILE *fLog = stdout;
void Say( const char *format, ... ){
	va_list args;
	va_start(args, format);
	
	std::string new_format = "rank "+std::to_string(my_mpi.rank)+" : " + format + "\n";
	vfprintf(fLog, new_format.c_str(), args);
	fflush(stdout);
	
	va_end (args);
}
#endif

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

	if(1){
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

