// MPI context, just a few globals like world size, rank etc.
#ifdef USE_MPI
#include <mpi.h>
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
