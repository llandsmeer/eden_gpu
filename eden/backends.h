struct AbstractBackend {
    RawTables * tabs;
    StateBuffers * state;
    AbstractBackend(RawTables & tabs, StateBuffers & state) {
        this->tabs = &tabs;
        this->state = &state;
    }
    void init() {}
    virtual float * global_state_now() const = 0;
    virtual float * global_state_next() const = 0;
    virtual Table_F32 * global_tables_stateNow_f32 () const = 0;
    virtual Table_I64 * global_tables_stateNow_i64 () const = 0;
    virtual Table_F32 * global_tables_stateNext_f32() const = 0;
    virtual Table_I64 * global_tables_stateNext_i64() const = 0;
    virtual Table_F32 * global_tables_const_f32_arrays() const = 0;
    virtual Table_I64 * global_tables_const_i64_arrays() const = 0;
    virtual long long * global_tables_const_f32_sizes() const = 0;
    virtual long long * global_tables_const_i64_sizes() const = 0;
    virtual long long * global_tables_state_f32_sizes() const = 0;
    virtual long long * global_tables_state_i64_sizes() const = 0;
    virtual void execute_work_items(EngineConfig & engine_config, SimulatorConfig & config, int step, float time) = 0;
    virtual void swap_buffers() = 0;
};

// NEEED DOUBLE BUFFERING STIL!!!!
struct CpuBackend : AbstractBackend {
    using AbstractBackend::AbstractBackend;
    /* Pure CPU implementation just refers to existing state buffers */
    void init() {
        m_global_state_now = state->state_one.data();
        m_global_state_next = state->state_two.data();
        m_global_tables_stateNow_f32 = state->global_tables_stateOne_f32_arrays.data();
        m_global_tables_stateNow_i64 = state->global_tables_stateOne_i64_arrays.data();
        m_global_tables_stateNext_f32 = state->global_tables_stateTwo_f32_arrays.data();
        m_global_tables_stateNext_i64 = state->global_tables_stateTwo_i64_arrays.data();

    }
    // double buffered
    float * m_global_state_now = 0;
    float * global_state_now() const { return m_global_state_now; }
    float * m_global_state_next = 0;
    float * global_state_next() const { return m_global_state_next; }
    Table_F32 * m_global_tables_stateNow_f32 = 0;
    Table_F32 * global_tables_stateNow_f32 () const { return m_global_tables_stateNow_f32; }
    Table_I64 * m_global_tables_stateNow_i64 = 0;
    Table_I64 * global_tables_stateNow_i64 () const { return m_global_tables_stateNow_i64; }
    Table_F32 * m_global_tables_stateNext_f32= 0;
    Table_F32 * global_tables_stateNext_f32() const { return m_global_tables_stateNext_f32; }
    Table_I64 * m_global_tables_stateNext_i64= 0;
    Table_I64 * global_tables_stateNext_i64() const { return m_global_tables_stateNext_i64; }
    // non buffered
    Table_F32 * global_tables_const_f32_arrays() const { return state->global_tables_const_f32_arrays.data(); }
    Table_I64 * global_tables_const_i64_arrays() const { return state->global_tables_const_i64_arrays.data(); }
    long long * global_tables_const_f32_sizes() const { return state->global_tables_const_f32_sizes.data(); }
    long long * global_tables_const_i64_sizes() const { return state->global_tables_const_i64_sizes.data(); }
    long long * global_tables_state_f32_sizes() const { return state->global_tables_state_f32_sizes.data(); }
    long long * global_tables_state_i64_sizes() const { return state->global_tables_state_i64_sizes.data(); }

    void execute_work_items(EngineConfig & engine_config, SimulatorConfig & config, int step, float time) {
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
            tabs->callbacks[item]( time, dt,
                tabs->global_constants.data(),      tabs->global_const_f32_index[item], global_tables_const_f32_sizes(),            global_tables_const_f32_arrays(),      tabs->global_table_const_f32_index[item],
                global_tables_const_i64_sizes(),    global_tables_const_i64_arrays(),   tabs->global_table_const_i64_index[item],
                global_tables_state_f32_sizes(),    global_tables_stateNow_f32(),       global_tables_stateNext_f32(),              tabs->global_table_state_f32_index[item],
                global_tables_state_i64_sizes(),    global_tables_stateNow_i64(),       global_tables_stateNext_i64(),              tabs->global_table_state_i64_index[item],
                global_state_now(),                 global_state_next(),                tabs->global_state_f32_index[item],
                step
            );
            if(config.debug){
                printf("item %lld end\n", item);
                fflush(stdout);
            }
        }
    }

    void swap_buffers() {
        std::swap(m_global_state_now, m_global_state_next);
        std::swap(m_global_tables_stateNow_f32, m_global_tables_stateNext_f32);
        std::swap(m_global_tables_stateNow_i64, m_global_tables_stateNext_i64);
    }


    void dump_iteration(SimulatorConfig & config, bool initializing, double time, long long step) {
        if( config.dump_raw_state_scalar || config.dump_raw_state_table ){
            if( !initializing ){
                printf("State: t = %g %s\n", time, Scales<Time>::native.name);
            } else {
                printf("State: t = %g %s, initialization step %lld\n", time, Scales<Time>::native.name, step);
            }
        }
        if( config.dump_raw_state_scalar ){
            // print state, separated by work item
            for( size_t i = 0, itm = 1; i < state->state_one.size(); i++ ){
                printf("%g \t", global_state_next()[i]);
                while( itm < tabs->global_state_f32_index.size() && (i + 1) == (size_t)tabs->global_state_f32_index[itm] ){
                    printf("| ");
                    itm++;
                }
            }
            printf("\n");
        }
        if( config.dump_raw_state_table ) state->dump_raw_state_table(tabs);
    }

};

struct GpuBackend : AbstractBackend {
    // ************** WARNING ****************
    // dump_raw_state_table() and mpi.init_communicate() read these pointers
    // from the StateBuffers object itself that means that when we go to
    // the GPU we either need to copy back to the vectors, use a custom allocator
    // or just raise a warning if we try to use dump_raw_state_table() or MPI in
    // combination with GPU, or replace the StateBuffer reference with this pointer list
    // Also backends save a pointer to tabs & state, so descruct CpuBackend before tabs & state
    using AbstractBackend::AbstractBackend;
    void init () {
    }
    float * d_global_state_now = 0;
    float * global_state_now() const { return d_global_state_now; }
    float * d_global_state_next = 0;
    float * global_state_next() const { return d_global_state_next; }
    Table_F32 * d_global_tables_stateNow_f32  = 0;
    Table_F32 * global_tables_stateNow_f32 () const { return d_global_tables_stateNow_f32; }
    Table_I64 * d_global_tables_stateNow_i64  = 0;
    Table_I64 * global_tables_stateNow_i64 () const { return d_global_tables_stateNow_i64; }
    Table_F32 * d_global_tables_stateNext_f32 = 0;
    Table_F32 * global_tables_stateNext_f32() const { return d_global_tables_stateNext_f32; }
    Table_I64 * d_global_tables_stateNext_i64 = 0;
    Table_I64 * global_tables_stateNext_i64() const { return d_global_tables_stateNext_i64; }
    Table_F32 * d_global_tables_const_f32_arrays = 0;
    Table_F32 * global_tables_const_f32_arrays() const { return d_global_tables_const_f32_arrays; }
    Table_I64 * d_global_tables_const_i64_arrays = 0;
    Table_I64 * global_tables_const_i64_arrays() const { return d_global_tables_const_i64_arrays; }
    long long * d_global_tables_const_f32_sizes = 0;
    long long * global_tables_const_f32_sizes() const { return d_global_tables_const_f32_sizes; }
    long long * d_global_tables_const_i64_sizes = 0;
    long long * global_tables_const_i64_sizes() const { return d_global_tables_const_i64_sizes; }
    long long * d_global_tables_state_f32_sizes = 0;
    long long * global_tables_state_f32_sizes() const { return d_global_tables_state_f32_sizes; }
    long long * d_global_tables_state_i64_sizes = 0;
    long long * global_tables_state_i64_sizes() const { return d_global_tables_state_i64_sizes; }
    void swap_buffers() {
        std::swap(d_global_state_now, d_global_state_next);
        std::swap(d_global_tables_stateNow_f32, d_global_tables_stateNext_f32);
        std::swap(d_global_tables_stateNow_i64, d_global_tables_stateNext_i64);
    }
};

