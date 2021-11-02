#ifndef EDEN_MPI_HELPERS_H
#define EDEN_MPI_HELPERS_H

#include "EngineConfig.h"
#include "StateBuffers.h"
#include "eden/backends/AbstractBackend.h"

#ifdef USE_MPI

#define MPI_CHECK_RETURN(error_code) {                                           \
    if (error_code != MPI_SUCCESS) {                                             \
        char error_string[BUFSIZ];                                               \
        int length_of_error_string;                                              \
        int world_rank;                                                          \
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);                              \
        MPI_Error_string(error_code, error_string, &length_of_error_string);     \
        fprintf(stderr, "%3d: %s\n", world_rank, error_string);                  \
        exit(1);                                                                 \
    }}

#include "TypePun.h"
#include <mpi.h>

void setup_mpi(int & argc, char ** & argv, EngineConfig* Engine);

struct MpiBuffers {
private:
    bool actually_using_mpi = false;
    miniLogger *logP;

public:
    typedef std::vector<float> SendRecvBuf;
    std::vector<int> send_off_to_node;
    std::vector< SendRecvBuf > send_bufs;
    std::vector<int> recv_off_to_node;
    std::vector< SendRecvBuf > recv_bufs;
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;

    // recv's have to be probed before recv'ing
    std::vector<bool> received_probes;
    std::vector<bool> received_sends;

    MpiBuffers(EngineConfig & engine_config) :
        send_requests( engine_config.sendlist_impls.size(), MPI_REQUEST_NULL ),
        recv_requests( engine_config.recvlist_impls.size(), MPI_REQUEST_NULL ),
        received_probes( engine_config.recvlist_impls.size(), false),
        received_sends( engine_config.recvlist_impls.size(), false)
    {
        if (!engine_config.use_mpi) return;
        init(engine_config);
    }
    ~MpiBuffers () {
        // this is necessary, so stdio files are actually flushed
        if (actually_using_mpi) {
            MPI_CHECK_RETURN(MPI_Finalize());
        }
    }

    void init(EngineConfig & engine_config);
    void init_communicate(EngineConfig & engine_config, AbstractBackend * backend, SimulatorConfig & config);
    void finish_communicate(EngineConfig & engine_config);
};
#else
static void setup_mpi(int & argc, char ** & argv, EngineConfig* Engine){}
struct MpiBuffers {
    explicit MpiBuffers(EngineConfig & engine_config) {}
    void init_communicate(EngineConfig & engine_config, AbstractBackend * backend, SimulatorConfig & config) {}
    void finish_communicate(EngineConfig & engine_config) {}
};
#endif
#endif
