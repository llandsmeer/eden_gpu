//
// Created by max on 02-11-21.
//

#include "Mpi_helpers.h"

void setup_mpi(int & argc, char ** & argv, EngineConfig* Engine) {
    // first of first of all, replace argc and argv
    // Modern implementations may keep MPI args from appearing anyway; non-modern ones still need this
    MPI_CHECK_RETURN(MPI_Init(&argc, &argv));
    // Get the number of processes
    MPI_CHECK_RETURN(MPI_Comm_size(MPI_COMM_WORLD, &Engine->my_mpi.world_size));
    // Get the rank of the process
    MPI_CHECK_RETURN(MPI_Comm_rank(MPI_COMM_WORLD, &Engine->my_mpi.rank));

    // Get processor name
    int name_len;
    MPI_CHECK_RETURN(MPI_Get_processor_name(Engine->my_mpi.processor_name, &name_len));
    INIT_LOG(nullptr,Engine->my_mpi.rank);
}
void MpiBuffers::init(EngineConfig & engine_config) {

    logP = new miniLogger(LOG_DEFAULT,
                          std::cout,
                          &engine_config.log_context.log_file,
                          __FUNCTION__,
                          engine_config.log_context.mpi_rank);
    (*logP)(LOG_INFO) << "Allocating MPI buffers..." << LOG_ENDL;

    actually_using_mpi = true;

    for( const auto &keyval : engine_config.sendlist_impls ){
        send_off_to_node.push_back( keyval.first );
        send_bufs.emplace_back();
        // allocate as they come, why not
    }
    for( const auto &keyval : engine_config.recvlist_impls ){
        recv_off_to_node.push_back( keyval.first );
        recv_bufs.emplace_back();
        // allocate as they come, why not
    }
}
void MpiBuffers::init_communicate(EngineConfig &engine_config, AbstractBackend *backend, SimulatorConfig &config) {
    if (!engine_config.use_mpi) return;

    float     * global_state_now                = backend->host_state_now();
    Table_F32 * global_tables_stateNow_f32      = backend->host_tables_stateNow_f32();
    Table_I64 * global_tables_stateNow_i64      = backend->host_tables_stateNow_i64();
    long long * global_tables_state_i64_sizes   = backend->host_tables_state_i64_sizes();

    auto NetMessage_ToString = []( size_t buf_value_len, const auto &buf ){
        std::string str;
        for( size_t i = 0; i < buf_value_len; i++ ){
            str += presentable_string( buf[i] ) + " ";
        }
        str += "| ";
        for( size_t i = buf_value_len; i < buf.size(); i++ ){
            str += presentable_string( EncodeF32ToI32( buf[i] ) ) + " ";
        }
        return str;
    };

    // Send info needed by other nodes
    // TODO try parallelizing buffer fill, see if it improves latency
    for( size_t idx = 0; idx < send_off_to_node.size(); idx++ ){
        auto other_rank = send_off_to_node.at(idx);
        const auto &sendlist_impl = engine_config.sendlist_impls.at(other_rank);

        auto &buf = send_bufs[idx];
        auto &req = send_requests[idx];

        // get the continuous_time values
        size_t vpeer_buf_idx = 0;
        size_t vpeer_buf_len = sendlist_impl.vpeer_positions_in_globstate.size();

        size_t daw_buf_idx = vpeer_buf_idx + vpeer_buf_len;
        size_t daw_buf_len = sendlist_impl.daw_columns.size();

        size_t buf_value_len = vpeer_buf_len + daw_buf_len;

        buf.resize(buf_value_len); // std::vector won't reallocate due to size reduction
        // otherwise implement appending the spikes manually

        // NB make sure these buffers are synchronized with CPU memory LATER
        for( size_t i = 0; i < sendlist_impl.vpeer_positions_in_globstate.size(); i++ ){
            size_t off = sendlist_impl.vpeer_positions_in_globstate[i];
            buf[ vpeer_buf_idx + i ] = global_state_now[off];
        }

        for( size_t i = 0; i < sendlist_impl.daw_columns.size(); i++ ){
            assert( engine_config.my_mpi.rank != 0 && other_rank == 0 );
            auto &col = sendlist_impl.daw_columns[i];
            size_t off = col.entry;
            // also apply scaling, so receiving node won't bother
            buf[ daw_buf_idx + i ] = global_state_now[ off ] * col.scaleFactor ;
        }

        size_t spikebuf_off = sendlist_impl.spike_mirror_buffer;
        // get the spikes into the buffer (variable size)
        Table_I64 SpikeTable      = global_tables_stateNow_i64[spikebuf_off];
        long long SpikeTable_size = global_tables_state_i64_sizes[spikebuf_off];

        for( int i = 0; i < SpikeTable_size; i++ ){

            // TODO packed bool buffers
            if( SpikeTable[i] ){
                // add index
                buf.push_back(  EncodeI32ToF32(i) );
                // clear trigger flag for the timestep after the next one
                SpikeTable[i] = 0;
            }
        }
        if( config.debug_netcode ){
            (*logP)(LOG_DEBUG) << NetMessage_ToString( buf_value_len, buf).c_str() << LOG_ENDL;
        }
        MPI_CHECK_RETURN(MPI_Isend( buf.data(), buf.size(), MPI_FLOAT, other_rank, MYMPI_TAG_BUF_SEND, MPI_COMM_WORLD, &req ));
    }


    // Recv info needed by this node
    auto PostRecv = [&config]( int other_rank, std::vector<float> &buf, MPI_Request &recv_req ){
        MPI_CHECK_RETURN(MPI_Irecv( buf.data(), buf.size(), MPI_FLOAT, other_rank, MYMPI_TAG_BUF_SEND, MPI_COMM_WORLD, &recv_req ));
    };
    auto ReceiveList = [ &engine_config, &global_tables_stateNow_f32, &global_tables_stateNow_i64 ]( const EngineConfig::RecvList_Impl &recvlist_impl, std::vector<float> &buf ){

        // copy the continuous-time values
        size_t value_buf_idx = 0;
        float *value_buf = global_tables_stateNow_f32[ recvlist_impl.value_mirror_buffer ];
        // NB make sure these buffers are synchronized with CPU memory LATER
        for( ptrdiff_t i = 0; i < recvlist_impl.value_mirror_size; i++ ){
            value_buf[i] = buf[ value_buf_idx + i ];

            // global_state_now[ off + i] = buf[ value_buf_idx + i ];
        }

        // and deliver the spikes to trigger buffers
        for( int i = recvlist_impl.value_mirror_size; i < (int)buf.size(); i++ ){
            int spike_pos = EncodeF32ToI32( buf[i] );
            for( auto tabent_packed : recvlist_impl.spike_destinations[spike_pos] ){
                auto tabent = GetDecodedTableEntryId( tabent_packed );
                // TODO packed bool buffers
                global_tables_stateNow_i64[tabent.table][tabent.entry] = 1;
            }
        }

        // all done with message
    };
    // TODO min_delay option when no gap junctions exist
    // Also wait for recvs to finish
    // Spin it all, to probe for multimple incoming messages
    bool all_received = true;
    do{
        all_received = true; // at least for the empty set examined before the loop
        // TODO also try parallelizing this, perhaps?
        for( size_t idx = 0; idx < recv_off_to_node.size(); idx++ ){
            auto other_rank = recv_off_to_node.at(idx);
            const auto &recvlist_impl = engine_config.recvlist_impls.at(other_rank);

            if( received_sends[idx] ) continue;

            // otherwise it's pending
            all_received = false;

            auto &buf = recv_bufs[idx];
            auto &req = recv_requests[idx];

            if( received_probes[idx] ){
                // check if recv is done
                int flag = 0;
                MPI_Status status;
                MPI_CHECK_RETURN(MPI_Test( &req, &flag, &status));
                if( flag ){
                    // received, yay !
                    if( config.debug_netcode ){
                        (*logP)(LOG_DEBUG) << NetMessage_ToString( recvlist_impl.value_mirror_size, buf).c_str() << LOG_ENDL;
                    }
                    ReceiveList( recvlist_impl, buf );
                    received_sends[idx] = true;
                }
            }
            else{
                // check if probe is ready
                int flag = 0;
                MPI_Status status;
                MPI_CHECK_RETURN(MPI_Iprobe( other_rank, MYMPI_TAG_BUF_SEND, MPI_COMM_WORLD, &flag, &status));
                if( flag ){
                    int buf_size;
                    MPI_CHECK_RETURN(MPI_Get_count( &status, MPI_FLOAT, &buf_size ));
                    buf.resize( buf_size );
                    PostRecv( other_rank, buf, req );
                    received_probes[idx] = true;
                }
            }

        }
    } while( !all_received );
    // and clear the progress flags
    received_probes.assign( received_probes.size(), false );
    received_sends .assign( received_sends .size(), false );
}
void MpiBuffers::finish_communicate(EngineConfig & engine_config) {
    if (!engine_config.use_mpi) return;
    // wait for sends, to finish the iteration
    MPI_CHECK_RETURN(MPI_Waitall( send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE ));
}