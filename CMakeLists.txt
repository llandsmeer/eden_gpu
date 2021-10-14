cmake_minimum_required(VERSION 3.10)
project(eden)
find_package(OpenMP REQUIRED)
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unused-result -lm -g -fopenmp")

include_directories(eden/neuroml)

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
SET(OUTPUT_FILE LEMS_CoreComponents.gen.cpp)
ADD_CUSTOM_COMMAND(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${RESOURCE_COMPILER} -i ${CMAKE_SOURCE_DIR}/eden/neuroml/LEMS_CoreComponents.inc.xml ${OUTPUT_FILE}
        COMMENT "Compiling ${INPUT_FILE} to binary")

add_library(cJSON thirdparty/cJSON-1.7.1/cJSON.c)
add_library(pugixml thirdparty/pugixml-1.9/pugixml.cpp)


add_executable(eden
        ${DEFAULT_SOURCES_LIB}
        ${BISON_MyParser_OUTPUTS}
        ${FLEX_MyScanner_OUTPUTS}
        ${OUTPUT_FILE}
    )

target_link_libraries(eden cJSON pugixml ${CMAKE_DL_LIBS})

# MAKEFILE

#TOOLCHAIN = cc
#Assuming TOOLCHAIN = gcc...
#g++ -c eden/Eden.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -fopenmp -o obj/eden.debug.gcc.cpu.o
#g++ -c eden/Utils.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -o obj/Utils.debug.gcc.cpu.o
#g++ -c eden/NeuroML.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -o obj/NeuroML.debug.gcc.cpu.o -I.
#bison --defines=./obj/LEMS_Expr.tab.h --output=./obj/LEMS_Expr.tab.cpp  eden/neuroml/LEMS_Expr.y
#g++ -c ./obj/LEMS_Expr.tab.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -I ./eden/neuroml/ -o obj/LEMS_Expr.tab.debug.gcc.cpu.o
#flex -8 --outfile=./obj/LEMS_Expr.yy.cpp --header-file=./obj/LEMS_Expr.yy.h eden/neuroml/LEMS_Expr.lex
#g++ -c ./obj/LEMS_Expr.yy.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -I ./eden/neuroml/ -o obj/LEMS_Expr.yy.debug.gcc.cpu.o
#g++ -c eden/neuroml/LEMS_Expr.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -I ./eden/neuroml/ -I ./obj -o obj/LEMS_Expr.debug.gcc.cpu.o
#ar rcs obj/LEMS_Expr.debug.gcc.cpu.a obj/LEMS_Expr.debug.gcc.cpu.o obj/LEMS_Expr.yy.debug.gcc.cpu.o obj/LEMS_Expr.tab.debug.gcc.cpu.o
#xxd -i eden/neuroml/LEMS_CoreComponents.inc.xml ./obj/LEMS_CoreComponents.gen.cpp
#g++ -c ./obj/LEMS_CoreComponents.gen.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -o obj/LEMS_CoreComponents.debug.gcc.cpu.o
#g++ -c thirdparty/pugixml-1.9/pugixml.cpp -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -o obj/pugixml-1.9.debug.gcc.cpu.o
#g++ obj/eden.debug.gcc.cpu.o obj/Utils.debug.gcc.cpu.o obj/NeuroML.debug.gcc.cpu.o obj/LEMS_Expr.debug.gcc.cpu.a obj/LEMS_CoreComponents.debug.gcc.cpu.o obj/pugixml-1.9.debug.gcc.cpu.o  -ldl -ldl -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"2021-10-13\"   -g -fopenmp -I ./eden -I . -std=c++14 -fopenmp -o bin/eden.debug.gcc.cpu.x
#true || true # /usr/bin/ld bin/eden.debug.gcc.cpu.x -headerpad_max_install_names -o bin/eden.debug.gcc.cpu.x


