#ifndef EDEN_TRAJECTORYLOGGER_H
#define EDEN_TRAJECTORYLOGGER_H

constexpr int column_width = 16;
struct FixedWidthNumberPrinter{
    int column_size;
    int delimiter_size;
    char delimiter_char;
    char format[50];

    int getNumberSize() const {
        return column_size - delimiter_size;// for separator
    }

    FixedWidthNumberPrinter( int _csize, char _delcha = ' ', int _dellen = 1 ){
        column_size = _csize;
        delimiter_size = _dellen;
        delimiter_char = _delcha;
        assert( column_size > delimiter_size );

        const int number_size = getNumberSize();
        const int digits = column_size - 3 - 5; // "+1.", "e+308"
        sprintf(format, "%%+%d.%dg", number_size, digits );
    }

    // should have length of column_size + terminator
    void write( float val, char *buf) const {
        const int number_size = getNumberSize();
        snprintf( buf, number_size + 1, format, val );
        // Also add some spaces, to delimit columns
        for( int i = 0; i < delimiter_size; i++ ){
            buf[number_size + i] = delimiter_char;
        }
        buf[number_size+1] = '\0';
    }
};
struct TrajectoryLogger {
    std::vector<FILE *> trajectory_open_files;

    //-------------------> crunch the numbers
    // set up printing in logfiles
    char tmps_column[ column_width + 5 ];
    FixedWidthNumberPrinter column_fmt;

    void open_trajectory_files(EngineConfig & engine_config) {
        // open the logs, one for each logger
        for(auto logger : engine_config.trajectory_loggers){
            #ifdef USE_MPI
            assert( engine_config.my_mpi.rank == 0);
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
    }

    TrajectoryLogger(EngineConfig & engine_config) : column_fmt(column_width, '\t', 0) {
        open_trajectory_files(engine_config);
    }

    void write_output_logs(EngineConfig & engine_config, double time, float * global_state_now, /* for MPI??: */Table_F32 * global_tables_stateNow_f32) {
        for(size_t i = 0; i < engine_config.trajectory_loggers.size(); i++){
            #ifdef USE_MPI
            assert(engine_config.my_mpi.rank == 0);
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
                            if( column.on_node >= 0 && column.on_node != engine_config.my_mpi.rank ){
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
                float col_val = GetColumnValue(column);
                column_fmt.write( col_val, tmps_column );
                fprintf( fout, "\t%s", tmps_column );
                // fprintf( fout, "\t%f", col_val );
            }
            fprintf(fout, "\n");
        }
    }

    void close () {
        // close loggers
        for( auto &fout : trajectory_open_files ){
            fclose(fout);
            fout = NULL;
        }
        trajectory_open_files.clear();
    }

    ~TrajectoryLogger() {
        close();
    }
};

#endif