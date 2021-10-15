cmake_minimum_required(VERSION 3.10)
project(eden)
find_package(OpenMP REQUIRED)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unused-result -lm -g -fopenmp")

set(DEFAULT_SOURCES_LIB
        eden/Eden.cpp
        eden/Utils.cpp
        eden/NeuroML.cpp
        eden/neuroml/LEMS_Expr.cpp
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

add_executable(eden
        ${DEFAULT_SOURCES_LIB}
        ${BISON_MyParser_OUTPUTS}
        ${FLEX_MyScanner_OUTPUTS}
        ${OUTPUT_FILE}
    )
target_include_directories(eden PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/eden/neuroml
    ${CMAKE_BINARY_DIR})
target_link_libraries(eden cJSON pugixml ${CMAKE_DL_LIBS})

install(TARGETS eden 
    TYPE RUNTIME)