#ifndef EDEN_ITERATION_CALLBACK_H
#define EDEN_ITERATION_CALLBACK_H

#include "Common.h"

extern "C" {
// assume standard C calling convention, which is probably the only cross-module one in most architectures
// bother if problems arise LATER
typedef void ( *IterationCallback)(
	float time,
	float dt,
	const float *__restrict__ constants, long long const_local_index,
	const long long *__restrict__ const_table_f32_sizes, const Table_F32 *__restrict__ const_table_f32_arrays, long long table_cf32_local_index,
	const long long *__restrict__ const_table_i64_sizes, const Table_I64 *__restrict__ const_table_i64_arrays, long long table_ci64_local_index,
	const long long *__restrict__ state_table_f32_sizes, const Table_F32 *__restrict__ state_table_f32_arrays, Table_F32 *__restrict__ stateNext_table_f32_arrays, long long table_sf32_local_index,
	const long long *__restrict__ state_table_i64_sizes,       Table_I64 *__restrict__ state_table_i64_arrays, Table_I64 *__restrict__ stateNext_table_i64_arrays, long long table_si64_local_index,
	const float *__restrict__ state, float *__restrict__ stateNext, long long state_local_index,
	long long step
	
);

typedef void ( * GPUIterationCallback )(
        long long start, long long n_items,
        float time, float dt, const float *__restrict__ global_constants, const long long * __restrict__ /*XXX*/ global_const_f32_index,
        const long long *__restrict__ global_const_table_f32_sizes, const Table_F32 *__restrict__ global_const_table_f32_arrays, long long * __restrict__ /*XXX*/ global_table_const_f32_index,
        const long long *__restrict__ global_const_table_i64_sizes, const Table_I64 *__restrict__ global_const_table_i64_arrays, long long * __restrict__ /*XXX*/ global_table_const_i64_index,
        const long long *__restrict__ global_state_table_f32_sizes, const Table_F32 *__restrict__ global_state_table_f32_arrays, Table_F32 *__restrict__ global_stateNext_table_f32_arrays, long long * __restrict__ /*XXX*/ global_table_state_f32_index,
        const long long *__restrict__ global_state_table_i64_sizes,       Table_I64 *__restrict__ global_state_table_i64_arrays, Table_I64 *__restrict__ global_stateNext_table_i64_arrays, long long * __restrict__ /*XXX*/ global_table_state_i64_index,
        const float *__restrict__ global_state, float *__restrict__ global_stateNext, long long * __restrict__ global_state_f32_index,
        long long step, int threads_per_block);

//void UndefinedCallback(
//	float time,
//	float dt,
//	const float *__restrict__ constants, long long const_local_index,
//	const long long *__restrict__ const_table_f32_sizes, const Table_F32 *__restrict__ const_table_f32_arrays, long long table_cf32_local_index,
//	const long long *__restrict__ const_table_i64_sizes, const Table_I64 *__restrict__ const_table_i64_arrays, long long table_ci64_local_index,
//	const long long *__restrict__ state_table_f32_sizes, const Table_F32 *__restrict__ state_table_f32_arrays, Table_F32 *__restrict__ stateNext_table_f32_arrays, long long table_sf32_local_index,
//	const long long *__restrict__ state_table_i64_sizes,       Table_I64 *__restrict__ state_table_i64_arrays, Table_I64 *__restrict__ stateNext_table_i64_arrays, long long table_si64_local_index,
//	const float *__restrict__ state, float *__restrict__ stateNext, long long state_local_index,
//	long long step
//){
//	// what else? LATER
//	abort();
//}


}

#endif
