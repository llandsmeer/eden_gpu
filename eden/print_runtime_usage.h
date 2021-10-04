void print_runtime_usage(RunMetaData & metadata) {
    printf("Config: %.3lf Setup: %.3lf Run: %.3lf \n", metadata.config_time_sec, metadata.init_time_sec, metadata.run_time_sec );
    #ifdef __linux__
    //get memory usage information too
    long long memResidentPeak = metadata.peak_resident_memory_bytes = getPeakResidentSetBytes();
    long long memResidentEnd = metadata.end_resident_memory_bytes = getCurrentResidentSetBytes();
    long long memHeap = getCurrentHeapBytes();
    printf("Peak: %lld Now: %lld Heap: %lld\n", memResidentPeak, memResidentEnd, memHeap );
    #endif
}

