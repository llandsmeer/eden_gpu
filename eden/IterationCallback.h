#ifndef EDEN_ITERATION_CALLBACK_H
#define EDEN_ITERATION_CALLBACK_H

extern "C" {
// assume standard C calling convention, which is probably the only cross-module one in most architectures
// bother if problems arise LATER
typedef void ( *IterationCallback)(
	double time,
	float dt,
	const float *__restrict__ constants, long long const_local_index,
	const long long *__restrict__ const_table_f32_sizes, const Table_F32 *__restrict__ const_table_f32_arrays, long long table_cf32_local_index,
	const long long *__restrict__ const_table_i64_sizes, const Table_I64 *__restrict__ const_table_i64_arrays, long long table_ci64_local_index,
	const long long *__restrict__ state_table_f32_sizes, const Table_F32 *__restrict__ state_table_f32_arrays, Table_F32 *__restrict__ stateNext_table_f32_arrays, long long table_sf32_local_index,
	const long long *__restrict__ state_table_i64_sizes,       Table_I64 *__restrict__ state_table_i64_arrays, Table_I64 *__restrict__ stateNext_table_i64_arrays, long long table_si64_local_index,
	const float *__restrict__ state, float *__restrict__ stateNext, long long state_local_index,
	long long step
	
);
void UndefinedCallback(
	double time,
	float dt,
	const float *__restrict__ constants, long long const_local_index,
	const long long *__restrict__ const_table_f32_sizes, const Table_F32 *__restrict__ const_table_f32_arrays, long long table_cf32_local_index,
	const long long *__restrict__ const_table_i64_sizes, const Table_I64 *__restrict__ const_table_i64_arrays, long long table_ci64_local_index,
	const long long *__restrict__ state_table_f32_sizes, const Table_F32 *__restrict__ state_table_f32_arrays, Table_F32 *__restrict__ stateNext_table_f32_arrays, long long table_sf32_local_index,
	const long long *__restrict__ state_table_i64_sizes,       Table_I64 *__restrict__ state_table_i64_arrays, Table_I64 *__restrict__ stateNext_table_i64_arrays, long long table_si64_local_index,
	const float *__restrict__ state, float *__restrict__ stateNext, long long state_local_index,
	long long step
){
	// what else? LATER
	abort();
}
}

#endif
