#ifndef RAWTABLES_H
#define RAWTABLES_H

#include <vector>
#include "MMMallocator.h"
#include "IterationCallback.h"

extern "C" {
// initial states, internal constants, connectivity matrices, iteration function pointers and everything
// so crunching can commence
struct RawTables{
    const static size_t ALIGNMENT = 32;

    typedef std::vector< float, _mm_Mallocator<float, ALIGNMENT> > Table_F32;
    typedef std::vector< long long, _mm_Mallocator<long long, ALIGNMENT> > Table_I64;

    struct ConsecutiveIterationCallbacks{
        size_t start_item;
        size_t n_items;
        IterationCallback callback;
    };

    // TODO aligned vectors, e.g. std::vector<T, boost::alignment::aligned_allocator<T, 16>>
    Table_F32 global_initial_state; // sum of states of work items
    Table_F32 global_constants; // sum of states of work items
    Table_I64 index_constants; // TODO if ever

    std::vector<long long> global_state_f32_index; // for each work unit TODO explain they are not completely "global"
    std::vector<long long> global_const_f32_index; // for each work unit

    //the tables TODO aligned
    std::vector<long long> global_table_const_f32_index; // for each work unit
    std::vector<long long> global_table_const_i64_index; // for each work unit
    std::vector<long long> global_table_state_f32_index; // for each work unit
    std::vector<long long> global_table_state_i64_index; // for each work unit
    //using std::vector due to a-priori-unknown size of tables; will use a pointer-based allocator to pass to callbacks as vectors, and compaction LATER
    std::vector<Table_F32> global_tables_const_f32_arrays; //the backing store for each table
    std::vector<Table_I64> global_tables_const_i64_arrays; //the backing store for each table
    std::vector<Table_F32> global_tables_state_f32_arrays; //the backing store for each table
    std::vector<Table_I64> global_tables_state_i64_arrays; //the backing store for each table

    std::vector<IterationCallback> callbacks; // for each work unit
    std::vector<ConsecutiveIterationCallbacks> consecutive_kernels;

    // some special-purpose tables

    // These are to access the singular, flat state & const vectors. They are not filled in otherwise.
    long long global_const_tabref;
    long long global_state_tabref;

    // the data loggers use this to access the variables, for remapping to be possible in MPI case - why not in eng.config then?
    // long long logger_table_const_i64_to_state_f32_index;

    // TODO what about the MPI mirrors ?

    RawTables(){
        global_const_tabref = -1;
        global_state_tabref = -1;
    }

    void create_consecutive_kernels_vector() {
        consecutive_kernels.clear();
        if (callbacks.size() == 0) return;
        ConsecutiveIterationCallbacks cic;
        cic.start_item = 0;
        cic.n_items = 1;
        cic.callback = callbacks.at(0);
        for (size_t idx = 1; idx < callbacks.size(); idx++) {
            if (callbacks.at(idx) == cic.callback) {
                cic.n_items += 1;
            } else {
                consecutive_kernels.push_back(cic);
                cic.start_item = idx;
                cic.n_items = 1;
                cic.callback = callbacks.at(idx);
            }
        }
        consecutive_kernels.push_back(cic);
    }
};
}
#endif
