#ifndef STATEBUFFERS_H
#define STATEBUFFERS_H


#include <vector>
#include "Common.h"
#include "StringHelpers.h"
#include "RawTables.h"


struct StateBuffers {
    // allocate at least two state vectors, to iterate in parallel
    RawTables::Table_F32 state_one;
    RawTables::Table_F32 state_two;

    std::vector<RawTables::Table_F32>  tables_state_f32_one;
    std::vector<RawTables::Table_F32>  tables_state_f32_two;
    std::vector<RawTables::Table_I64>  tables_state_i64_one;
    std::vector<RawTables::Table_I64>  tables_state_i64_two;

    //also allocate pointer and size vectors, to use instead of silly std::vectors
    std::vector <long long> global_tables_const_f32_sizes;
    std::vector <Table_F32> global_tables_const_f32_arrays;
    std::vector <long long> global_tables_const_i64_sizes;
    std::vector <Table_I64> global_tables_const_i64_arrays;

    std::vector <long long> global_tables_state_f32_sizes;
    std::vector <Table_F32> global_tables_stateOne_f32_arrays;
    std::vector <Table_F32> global_tables_stateTwo_f32_arrays;
    std::vector <long long> global_tables_state_i64_sizes;
    std::vector <Table_I64> global_tables_stateOne_i64_arrays;
    std::vector <Table_I64> global_tables_stateTwo_i64_arrays;

    StateBuffers(RawTables & tabs) :
                state_one(tabs.global_initial_state),
                state_two(tabs.global_initial_state.size(), NAN),
                tables_state_f32_one(tabs.global_tables_state_f32_arrays),
                tables_state_i64_one(tabs.global_tables_state_i64_arrays)
    {
        auto GetSizePtrTables = []( auto &tablist, auto &pointers, auto &sizes ){
            pointers.resize( tablist.size() );
            sizes.resize( tablist.size() );
            for(size_t i = 0; i < tablist.size(); i++){
                pointers[i] = tablist.at(i).data();
                sizes[i] = (long long)tablist.at(i).size();
            }
        };

        tables_state_f32_two.reserve( tables_state_f32_one.size());
        for( auto tab : tables_state_f32_one ) tables_state_f32_two.emplace_back( tab.size(), NAN );
        tables_state_i64_two.reserve( tables_state_i64_one.size() );
        for( auto tab : tables_state_i64_one ) tables_state_i64_two.emplace_back( tab.size(), 0 );

        // now things need to be done a little differently, since for example trigger(and lazy?) variables of Next ought to be zero for results to make sense

        GetSizePtrTables(tabs.global_tables_const_f32_arrays, global_tables_const_f32_arrays, global_tables_const_f32_sizes);
        GetSizePtrTables(tabs.global_tables_const_i64_arrays, global_tables_const_i64_arrays, global_tables_const_i64_sizes);
        //
        GetSizePtrTables(tables_state_f32_one, global_tables_stateOne_f32_arrays, global_tables_state_f32_sizes);
        GetSizePtrTables(tables_state_i64_one, global_tables_stateOne_i64_arrays, global_tables_state_i64_sizes);
        GetSizePtrTables(tables_state_f32_two, global_tables_stateTwo_f32_arrays, global_tables_state_f32_sizes);
        GetSizePtrTables(tables_state_i64_two, global_tables_stateTwo_i64_arrays, global_tables_state_i64_sizes);

        // also, set up the references to the flat vectors
        global_tables_const_f32_arrays[tabs.global_const_tabref] = tabs.global_constants.data();
        global_tables_const_f32_sizes [tabs.global_const_tabref] = tabs.global_constants.size();
        //
        global_tables_stateOne_f32_arrays[tabs.global_state_tabref] = state_one.data();
        global_tables_stateTwo_f32_arrays[tabs.global_state_tabref] = state_two.data();
        global_tables_state_f32_sizes    [tabs.global_state_tabref] = state_one.size();
    }

    void dump_array_locations(RawTables & tabs) {
        printf("ARRAY_LOC constants %p %lu %lu\n", tabs.global_constants.data(), tabs.global_constants.size()*sizeof(tabs.global_constants[0]), sizeof(tabs.global_constants[0]));
        printf("ARRAY_LOC const_f32_index %p %lu %lu\n", tabs.global_const_f32_index.data(), tabs.global_const_f32_index.size()*sizeof(tabs.global_const_f32_index[0]), sizeof(tabs.global_const_f32_index[0]));
        printf("ARRAY_LOC table_const_f32_index %p %lu %lu\n", tabs.global_table_const_f32_index.data(), tabs.global_table_const_f32_index.size()*sizeof(tabs.global_table_const_f32_index[0]), sizeof(tabs.global_table_const_f32_index[0]));
        printf("ARRAY_LOC table_const_i64_index %p %lu %lu\n", tabs.global_table_const_i64_index.data(), tabs.global_table_const_i64_index.size()*sizeof(tabs.global_table_const_i64_index[0]), sizeof(tabs.global_table_const_i64_index[0]));
        printf("ARRAY_LOC table_state_f32_index %p %lu %lu\n", tabs.global_table_state_f32_index.data(), tabs.global_table_state_f32_index.size()*sizeof(tabs.global_table_state_f32_index[0]), sizeof(tabs.global_table_state_f32_index[0]));
        printf("ARRAY_LOC table_state_i64_index %p %lu %lu\n", tabs.global_table_state_i64_index.data(), tabs.global_table_state_i64_index.size()*sizeof(tabs.global_table_state_i64_index[0]), sizeof(tabs.global_table_state_i64_index[0]));
        printf("ARRAY_LOC state_f32_index %p %lu %lu\n", tabs.global_state_f32_index.data(), tabs.global_state_f32_index.size()*sizeof(tabs.global_state_f32_index[0]), sizeof(tabs.global_state_f32_index[0]));
        printf("ARRAY_LOC state_now %p %lu %lu\n", state_one.data(), state_one.size()*sizeof(state_one[0]), sizeof(state_one[0]));
        printf("ARRAY_LOC state_next %p %lu %lu\n", state_two.data(), state_two.size()*sizeof(state_two[0]), sizeof(state_two[0]));

        printf("ARRAY_LOC tables_const_f32_sizes %p %lu %lu\n", global_tables_const_f32_sizes.data(), global_tables_const_f32_sizes.size()*sizeof(global_tables_const_f32_sizes[0]), sizeof(global_tables_const_f32_sizes[0]));
        printf("ARRAY_LOC tables_const_i64_sizes %p %lu %lu\n", global_tables_const_i64_sizes.data(), global_tables_const_i64_sizes.size()*sizeof(global_tables_const_i64_sizes[0]), sizeof(global_tables_const_i64_sizes[0]));
        printf("ARRAY_LOC tables_state_f32_sizes %p %lu %lu\n", global_tables_state_f32_sizes.data(), global_tables_state_f32_sizes.size()*sizeof(global_tables_state_f32_sizes[0]), sizeof(global_tables_state_f32_sizes[0]));
        printf("ARRAY_LOC tables_state_i64_sizes %p %lu %lu\n", global_tables_state_i64_sizes.data(), global_tables_state_i64_sizes.size()*sizeof(global_tables_state_i64_sizes[0]), sizeof(global_tables_state_i64_sizes[0]));

        printf("ARRAY_LOC tables_stateNow_f32 %p %lu %lu\n", global_tables_stateOne_f32_arrays.data(), global_tables_stateOne_f32_arrays.size()*sizeof(global_tables_stateOne_f32_arrays[0]), sizeof(global_tables_stateOne_f32_arrays[0]));
        printf("ARRAY_LOC tables_stateNow_i64 %p %lu %lu\n", global_tables_stateOne_i64_arrays.data(), global_tables_stateOne_i64_arrays.size()*sizeof(global_tables_stateOne_i64_arrays[0]), sizeof(global_tables_stateOne_i64_arrays[0]));
        printf("ARRAY_LOC tables_stateNext_f32 %p %lu %lu\n", global_tables_stateTwo_f32_arrays.data(), global_tables_stateTwo_f32_arrays.size()*sizeof(global_tables_stateTwo_f32_arrays[0]), sizeof(global_tables_stateTwo_f32_arrays[0]));
        printf("ARRAY_LOC tables_stateNext_i64 %p %lu %lu\n", global_tables_stateTwo_i64_arrays.data(), global_tables_stateTwo_i64_arrays.size()*sizeof(global_tables_stateTwo_i64_arrays[0]), sizeof(global_tables_stateTwo_i64_arrays[0]));
        printf("ARRAY_LOC tables_const_f32_arrays %p %lu %lu\n", global_tables_const_f32_arrays.data(), global_tables_const_f32_arrays.size()*sizeof(global_tables_const_f32_arrays[0]), sizeof(global_tables_const_f32_arrays[0]));
        printf("ARRAY_LOC tables_const_i64_arrays %p %lu %lu\n", global_tables_const_i64_arrays.data(), global_tables_const_i64_arrays.size()*sizeof(global_tables_const_i64_arrays[0]), sizeof(global_tables_const_i64_arrays[0]));

        for (size_t i = 0; i < global_tables_stateOne_f32_arrays.size(); i++) {
            size_t size = global_tables_state_f32_sizes[i];
            if (size == 0) continue;
            printf("ARRAY_LOC table_stateOne_f32_arrays[%lu] %p %lu %lu\n", i, global_tables_stateOne_f32_arrays[i], size*sizeof(float), sizeof(float));
        }

        for (size_t i = 0; i < global_tables_stateTwo_f32_arrays.size(); i++) {
            size_t size = global_tables_state_f32_sizes[i];
            if (size == 0) continue;
            printf("ARRAY_LOC table_stateTwo_f32_arrays[%lu] %p %lu %lu\n", i, global_tables_stateTwo_f32_arrays[i], size*sizeof(float), sizeof(float));
        }

        for (size_t i = 0; i < global_tables_const_f32_arrays.size(); i++) {
            size_t size = global_tables_const_f32_sizes[i];
            if (size == 0) continue;
            printf("ARRAY_LOC table_arrays[%lu] %p %lu %lu\n", i, global_tables_const_f32_arrays[i], size*sizeof(float), sizeof(float));
        }

        for (size_t i = 0; i < global_tables_stateOne_i64_arrays.size(); i++) {
            size_t size = global_tables_state_i64_sizes[i];
            if (size == 0) continue;
            printf("ARRAY_LOC table_arrays[%lu] %p %lu %lu\n", i, global_tables_stateOne_i64_arrays[i], size*sizeof(long long), sizeof(long long));
        }

        for (size_t i = 0; i < global_tables_stateTwo_i64_arrays.size(); i++) {
            size_t size = global_tables_state_i64_sizes[i];
            if (size == 0) continue;
            printf("ARRAY_LOC table_stateTwo_i64_arrays[%lu] %p %lu %lu\n", i, global_tables_stateTwo_i64_arrays[i], size*sizeof(long long), sizeof(long long));
        }

        for (size_t i = 0; i < global_tables_const_i64_arrays.size(); i++) {
            size_t size = global_tables_const_i64_sizes[i];
            if (size == 0) continue;
            printf("ARRAY_LOC table_const_i64_arrays[%lu] %p %lu %lu\n", i, global_tables_const_i64_arrays[i], size*sizeof(long long), sizeof(long long));
        }

        fflush(stdout);
    }

    void dump_raw_layout(RawTables & tabs) {
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

    void dump_raw_state_table(RawTables * tabs) {
        Table_F32 *global_tables_stateNow_f32  = global_tables_stateOne_f32_arrays.data();
        Table_I64 *global_tables_stateNow_i64  = global_tables_stateOne_i64_arrays.data();
        Table_F32 *global_tables_stateNext_f32 = global_tables_stateTwo_f32_arrays.data();
        Table_I64 *global_tables_stateNext_i64 = global_tables_stateTwo_i64_arrays.data();

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
        PrintVeryRawTables(tabs->global_table_state_f32_index, global_tables_stateNow_f32, global_tables_state_f32_sizes);
        printf("RawStateI64:\n");
        PrintVeryRawTables(tabs->global_table_state_i64_index, global_tables_stateNow_i64, global_tables_state_i64_sizes);
        printf("RawStateNextF32:\n");
        PrintVeryRawTables(tabs->global_table_state_f32_index, global_tables_stateNext_f32, global_tables_state_f32_sizes);
        printf("RawStateNextI64:\n");
        PrintVeryRawTables(tabs->global_table_state_i64_index, global_tables_stateNext_i64, global_tables_state_i64_sizes);
    }
};

#endif
