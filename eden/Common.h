/*
 *  Common includes for all source files
 *
 */

#ifndef COMMON_H
#define COMMON_H

//standard includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <float.h>
#include <stddef.h>
#include <cinttypes>

//for model representation and more
#include <climits>  //for INT_MAX
#include <cmath> // apparently including <algorithm> undefines ::isfinite() function, on ICC
#include <vector>
#include <functional>
#include <algorithm>
#include <utility>
#include <cstring>
#include <string>
#include <cstdarg> //for advanced whining
#include <unistd.h> //for time & memory usage measurement

// The logger
#include "../thirdparty/miniLogger/miniLogger.h"

//passable this way as function argument
struct LogContext{
    uint mpi_rank = 0;
    std::ofstream log_file;
};
#define LOG_DEFAULT LOG_DEBUG
#define INIT_LOG(FiLe,PiD) miniLogger log(LOG_DEFAULT,std::cout,FiLe, __FUNCTION__,PiD)


#if defined (__linux__) || defined(__APPLE__)
#include <dlfcn.h> // for dynamic loading
#endif

#ifdef __APPLE__
// use these lines for non-OSX LATER, if you're targeting an iPad or something you know how to handle the specifics
// #include <TargetConditionals.h>
// #if defined(TARGET_OS_OSX) && (TARGET_OS_OSX)
// ...
// #endif
#endif

#ifdef _WIN32
// for dynamic loading and other OS specific stuff
// #include <windows.h> // loaded through Common.h at the moment, TODO break out in Windows specific header

// Minimum supported version: Windows XP, aka 5.1
#define WINVER       0x0501
#define _WIN32_WINNT 0x0501

#define WIN32_LEAN_AND_MEAN // because all sorts of useful keywords like PURE are defined with the full headers
#define NOMINMAX
#include <windows.h>
// and more namespace pollution
#undef IN
#undef OUT
#undef INOUT
#endif

// Non-standard helper routines
#define pow10(powah) pow(10,(powah))
#define stricmp strcasecmp

//---->> Append to vector helpers
template< typename Container >
static void AppendToVector(Container &append_to, const Container &append_this){
    append_to.insert(append_to.end(), append_this.begin(), append_this.end());
};
template<
        typename CAppendTo, typename CAppendThis,
        typename std::enable_if< !std::is_same< CAppendTo, CAppendThis >::value, int >::type = 0
>
static void AppendToVector(CAppendTo &append_to, const CAppendThis &append_this){
    auto new_size = append_to.size() + append_this.size() ;
    append_to.reserve( new_size );
    for( size_t i = 0; i < append_this.size(); i++ ){
        append_to.push_back( append_this[i] );
    }
};

// do not specify alignment for the pointers, in the generic interface
// cannot specify __restrict__ because it is quietly dropped by compilers ( ! ) when the type is allocated with new
// causing a type mismatch when operator delete(T * __restrict__) is called (then why didn't they drop __restrict__ from there too ??)
// just hope the type "mismatch" won't cause a crash in practice
typedef float * Table_F32;
typedef long long * Table_I64;


//------------------> OS-independent utilities
// implementations can be found in Utils.cpp

// May be missing if not suported by platform, though
double TimevalDeltaSec(const timeval &start, const timeval &end);
struct Timer {
    timeval start;
    Timer() {
        gettimeofday(&start, 0);
    }
    double delta() {
        timeval end;
        gettimeofday(&end, 0);
        return TimevalDeltaSec(start, end);
    }
};

// Memory measurements on Linux
// TODO support on more platforms !
#ifdef 	__linux__
//Memory consumption getters return 0 for not available
int64_t getCurrentResidentSetBytes();
int64_t getPeakResidentSetBytes();
int64_t getCurrentHeapBytes();
#endif

// Struct to save meta data into
struct RunMetaData{
	double config_time_sec;
	double init_time_sec;
	double run_time_sec;
	double save_time_sec;
	
	int64_t peak_resident_memory_bytes; //0 for unknown
	int64_t end_resident_memory_bytes; //0 for unknown
    RunMetaData(){
		config_time_sec = NAN;
		init_time_sec = NAN;
		run_time_sec = NAN;
		save_time_sec = NAN;
		
		peak_resident_memory_bytes = 0;
		end_resident_memory_bytes = 0;
	}

	void print() {
        printf("Config: %.3lf Setup: %.3lf Run: %.3lf \n", config_time_sec, init_time_sec, run_time_sec);
#ifdef __linux__
        //get memory usage information too
        long long memResidentPeak = peak_resident_memory_bytes = getPeakResidentSetBytes();
        long long memResidentEnd = end_resident_memory_bytes = getCurrentResidentSetBytes();
        long long memHeap = getCurrentHeapBytes();
        printf("Peak: %lld Now: %lld Heap: %lld\n", memResidentPeak, memResidentEnd, memHeap);
#endif
    }
};

// A very fast and chaotic RNG
// straight from Wikipedia
class XorShiftMul{
private:
		uint64_t state[1];
public:
	// NB if xorshiftmul's state becomes 0 it outputs zero forever :D
	// force it to not happen, with the highest bit set
	// In this case, certain combinations of shift factors ensure a complete cycle through all non-zero values.
	// It has been proven through linear algebra, somehow.
	XorShiftMul(uint64_t _seed){state[0] = _seed | (1LL << 63);}
	uint64_t Get(){
		uint64_t x = state[0];
		x ^= x >> 12; // a
		x ^= x << 25; // b
		x ^= x >> 27; // c
		state[0] = x;
		return x * 0x2545F4914F6CDD1D;
	}
};

// Tokenize a string, as with String.split() in string-capable languages
std::vector<std::string> string_split(const std::string& str, const std::string& delim);
bool GetLineColumnFromFile(const char *filename, ptrdiff_t file_byte_offset, long long &line, long long &column);

//A more structured way to complain 
//Accepts varargs just because string manipulation is clunky in C/C++, even clunkier than varargs
void ReportErrorInFile_Base(FILE *error_log, const char *filename, ptrdiff_t file_byte_offset, const char *format, va_list args);
void ReportErrorInFile(FILE *error_log, const char *filename, ptrdiff_t file_byte_offset, const char *format, ...);

//------------------> Utilities end

//------------------> Windows specific util routines
#ifdef _WIN32
// TODO replace printf with a wrapper, on UTF16 targets
std::string DescribeErrorCode_Windows(DWORD error_code);
#endif
//------------------> end Windows specific util routines

#endif