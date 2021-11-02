# TODO
#   -- add docker build
#   -- add test cases
#   -- add help support
#   -- check the following:
#              -- include(CheckIPOSupported)
#              -- check_ipo_supported(RESULT supported OUTPUT error)
#              -- add_executable(example Example.cpp)
#              -- if( supported )
#              --     message(STATUS "IPO / LTO enabled")
#              --     set_property(TARGET example PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
#              -- else()
#              --     message(STATUS "IPO / LTO not supported: <${error}>")
#              -- endif()

cmake_minimum_required(VERSION 3.12)
include(CMakeDependentOption)

project(simHH LANGUAGES CUDA CXX C)
find_package(OpenMP REQUIRED)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unused-result -Wno-unused-function -lm -g -fopenmp")

option(USE_MPI "build with MPI support" OFF)

# ---> Default resources minimal to creat an build of eden
set(DEFAULT_SOURCES_LIB
        eden/Eden.cpp
        eden/Utils.cpp
        eden/NeuroML.cpp
        eden/neuroml/LEMS_Expr.cpp
        eden/GenerateModel.cpp
        eden/parse_command_line_args.cpp
    )

BISON_TARGET(MyParser eden/neuroml/LEMS_Expr.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/LEMS_Expr.tab.h)
FLEX_TARGET(MyScanner eden/neuroml/LEMS_Expr.lex  ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp COMPILE_FLAGS -8 DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/LEMS_Expr.yy.h)
ADD_FLEX_BISON_DEPENDENCY(MyScanner MyParser)

SET(RESOURCE_COMPILER xxd)
SET(OUTPUT_FILE ${CMAKE_BINARY_DIR}/LEMS_CoreComponents.gen.cpp)
ADD_CUSTOM_COMMAND(
        OUTPUT ${OUTPUT_FILE}
        COMMAND cd ${CMAKE_SOURCE_DIR} && ${RESOURCE_COMPILER} -i eden/neuroml/LEMS_CoreComponents.inc.xml ${OUTPUT_FILE}
        COMMENT "Compiling ${INPUT_FILE} to binary")

add_library(cJSON ${CMAKE_SOURCE_DIR}/thirdparty/cJSON-1.7.1/cJSON.c)
add_library(pugixml ${CMAKE_SOURCE_DIR}/thirdparty/pugixml-1.9/pugixml.cpp)
# <---

# ---> GPU resources to build with a GPU backend if applicable.
include(FindCUDA)
find_package(CUDA)
if(CUDA_FOUND)
    message("STATUS [Eden] CUDA enabled")
    enable_language(CUDA)
    add_definitions(-DUSE_GPU)
    set(CMAKE_CUDA_STANDARD 14)
    set(CUDA_SOURCES_LIB
                eden/GPU_helpers.cu
                eden/backends/gpu/GpuBackend.cu
            )
else ()
    message("STATUS [Eden] Could not find CUDA support. Disabling CUDA sources.")
endif(CUDA_FOUND)

# <---
# ---> MPI resources to build with MPI communication if applicable.
#if(USE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        message("STATUS [Eden] MPI enabled")
        add_definitions(-DUSE_MPI)

        # Check if openMPI is build cuda aware.
        EXECUTE_PROCESS (COMMAND "ompi_info" --parsable --all COMMAND "grep" "mpi_built_with_cuda_support:value" OUTPUT_VARIABLE OMPI_CUDA_OUTPUT)
        IF (DEFINED OMPI_CUDA_OUTPUT)

            # The output to stdout may have an endline, so we have to do a
            # string search, not just a comparison.  It's possible that
            # neither "true" nor "false" are found, so we test both.
            STRING (FIND "${OMPI_CUDA_OUTPUT}" "mca:mpi:base:param:mpi_built_with_cuda_support:value:true"  OMPI_INFO_OUTPUT_FOUND_TRUE)
            STRING (FIND "${OMPI_CUDA_OUTPUT}" "mca:mpi:base:param:mpi_built_with_cuda_support:value:false" OMPI_INFO_OUTPUT_FOUND_FALSE)

            IF (NOT (OMPI_INFO_OUTPUT_FOUND_TRUE EQUAL -1))
                MESSAGE (STATUS  "[Eden]  \"ompi_info\" explicitly claims that your MPI implementation is CUDA aware.")
                SET (OMPI_INFO_OUTPUT_FOUND_FALSE ON)

            ELSEIF (NOT (OMPI_INFO_OUTPUT_FOUND_FALSE EQUAL -1))
                MESSAGE (STATUS "[Eden]  \"ompi_info\" explicitly claims that your MPI implementation is NOT CUDA aware.  You may want to use a different OpenMPI installation that is CUDA aware.  For details: https://www.open-mpi.org/faq/?category=runcuda")
            ELSE ()
                MESSAGE (STATUS "[Eden]  \"ompi_info\" doesn' know  whether your MPI implementation is CUDA aware. output: \"${OMPI_CUDA_OUTPUT}\".")
            ENDIF ()
        ELSE ()
            MESSAGE (STATUS "[Eden] the ompi_info call did not return any sane information or did not run at all, assuming unaware, plz fix")
        ENDIF ()

        if (NOT (OMPI_INFO_OUTPUT_FOUND_TRUE EQUAL -1))
            add_definitions(-DMPI_CUDA_AWARE)
        endif()

        # Check on Cray systems if MPICH is RDMA enabled aka cuda aware
        if($ENV{MPICH_RDMA_ENABLED_CUDA})
            MESSAGE (STATUS "[Eden] Cray tells us that MPI is CUDA aware")
            add_definitions(-DMPI_CUDA_AWARE)
        endif()

    #    add sources here:
        set(MPI_SOURCES_LIB

                )
    else ()
        message(STATUS "[Eden] Could not find MPI support. Disabling MPI sources.")
    endif(MPI_FOUND)
#endif()
# <---

if(MPI_FOUND)
    if(CUDA_FOUND)
        set(MPI_CXX_LIB mpi_cxx)
    endif(CUDA_FOUND)
endif(MPI_FOUND)

add_executable(eden
        ${DEFAULT_SOURCES_LIB}
        ${MPI_SOURCES_LIB}
        ${CUDA_SOURCES_LIB}
        ${BISON_MyParser_OUTPUTS}
        ${FLEX_MyScanner_OUTPUTS}
        ${OUTPUT_FILE}
    )
target_include_directories(eden PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/eden/neuroml
    ${CMAKE_BINARY_DIR})

target_link_libraries(eden PRIVATE cJSON pugixml ${CMAKE_DL_LIBS} ${MPI_CXX_LIB})

if(CUDA_FOUND)
    set_target_properties(eden PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(eden PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            >)
endif(CUDA_FOUND)

if(MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(eden PRIVATE ${MPI_C_LIBRARIES})
    target_link_libraries(eden PRIVATE ${MPI_CUDA_LIBRARIES})
endif(MPI_FOUND)

# que ??
install(TARGETS eden 
    TYPE RUNTIME)
