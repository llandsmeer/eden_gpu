# line intentionally left blank, TODO anything to declare

#Select build type. Override by vars passed to Makefile from shell (ie make target VARIABLE=value )
BUILD_STAMP ?= $(shell date +"%Y-%m-%d")
BUILD ?= debug
PLATFORM ?= cpu

PROJ_BASE	?= .

# where to generate build output, in general?
OUT_DIR ?= $(PROJ_BASE)

# i.e. final artifacts
BIN_DIR ?= $(OUT_DIR)/bin

# i.e. intermediate artifacts
OBJ_DIR ?= $(OUT_DIR)/obj

# Main product's source code
SRC_EDEN := $(PROJ_BASE)/eden
# the distinction may be useful LATER, with multiple build targets
SRC_COMMON := $(PROJ_BASE)/eden
# Third-party components
SRC_THIRDPARTY := $(PROJ_BASE)/thirdparty
# Testing infrastructure for the various targets
TESTING_DIR := $(PROJ_BASE)/testing

# Third-party component config
PUGIXML_NAME := pugixml-1.9
SRC_PUGIXML := $(SRC_THIRDPARTY)/$(PUGIXML_NAME)

CJSON_NAME := cJSON-1.7.1
SRC_CJSON := $(SRC_THIRDPARTY)/$(CJSON_NAME)

# Pick a toolchain
TOOLCHAIN ?= $(CC)# maybe fon't try to take a hint from CC LATER
$(info TOOLCHAIN = $(TOOLCHAIN))

# default toolchain is GCC
ifeq "$(TOOLCHAIN)" "cc"
$(info Assuming TOOLCHAIN = gcc...)

TOOLCHAIN := gcc
endif
# other options: USE_MPI


# (internal variable) is a compiler selected?
COMPILER_SET := ko

# TODO automatic test of the ICC build
ifeq "$(TOOLCHAIN)" "icc"

ifdef USE_MPI
LD  := mpiicpc
CC  := mpiicc
CXX := mpiicpc
endif

ifndef USE_MPI
LD  := icpc
CC  := icc
CXX := icpc
endif

COMPILER_SET := ok
endif

ifeq ($(TOOLCHAIN), gcc)

ifdef USE_MPI
LD  := ld
CC  := mpicc
CXX := mpic++
endif

ifndef USE_MPI
LD  := ld
CC  := gcc
CXX := g++
endif

COMPILER_SET := ok
endif

# for special custom toolchains
ifdef TOOLCHAIN_OVERRIDE

$( info Overriding toolchain selection, make sure CXX, CFLAGS, LD etc. are configured )

COMPILER_SET := ok
endif

ifneq ($(COMPILER_SET), ok)
$(error Only gcc and icc toolchains are currently allowed, but TOOLCHAIN=$(TOOLCHAIN))
endif

# Compiler flags
# TODO more optimization flags
CFLAGS_basic := -Wall -Werror -Wno-unused-result -lm -DBUILD_STAMP=\"$(BUILD_STAMP)\"
CFLAGS_release := ${CFLAGS_basic} -O3
CFLAGS_debug := ${CFLAGS_basic} -g

CFLAGS_omp_gcc := -fopenmp
CFLAGS_omp_icc :=  -openmp
CFLAGS_omp :=  ${CFLAGS_omp_${TOOLCHAIN}}

CFLAGS_cpu = ${CFLAGS_omp}

CFLAGS ?= ${CFLAGS_${BUILD}} ${CFLAGS_${PLATFORM}} -I ${SRC_COMMON} -I ${PROJ_BASE}

# TODO temporary till targets are better specified in makefile
ifdef USE_MPI
CFLAGS += -DUSE_MPI
endif

CXXFLAGS := ${CFLAGS} -std=c++14

# Final targets
TARGETS := eden
# Other auxiliary modules
MODULES := cJSON pugixml

DOT_O := .${BUILD}.${TOOLCHAIN}.${PLATFORM}.o
DOT_A := .${BUILD}.${TOOLCHAIN}.${PLATFORM}.a
DOT_X := .${BUILD}.${TOOLCHAIN}.${PLATFORM}.x

all: clean ${TARGETS} test

# executable targets

# TODO add a build without OpenMP, to debug OpenMP errors
eden:  ${BIN_DIR}/eden${DOT_X}
${BIN_DIR}/eden${DOT_X}: ${OBJ_DIR}/eden${DOT_O} ${OBJ_DIR}/Utils${DOT_O} \
		${OBJ_DIR}/NeuroML${DOT_O} ${OBJ_DIR}/LEMS_Expr${DOT_A} ${OBJ_DIR}/LEMS_CoreComponents${DOT_O} \
		${OBJ_DIR}/${PUGIXML_NAME}${DOT_O} # third-party libs
	$(CXX) $^ -ldl $(CXXFLAGS) $(CFLAGS_omp) -o $@
${OBJ_DIR}/eden${DOT_O}: ${SRC_EDEN}/Eden.cpp ${SRC_EDEN}/NeuroML.h ${SRC_EDEN}/neuroml/LEMS_Expr.h ${SRC_COMMON}/Common.h  ${SRC_COMMON}/MMMallocator.h
	$(CXX) -c $< $(CXXFLAGS) $(CFLAGS_omp) -o $@

# own helper libraries
${OBJ_DIR}/Utils${DOT_O}: ${SRC_COMMON}/Utils.cpp ${SRC_COMMON}/Common.h
	$(CXX) -c $< $(CXXFLAGS) -o $@

${OBJ_DIR}/NeuroML${DOT_O}: ${SRC_EDEN}/NeuroML.cpp ${SRC_EDEN}/NeuroML.h ${SRC_EDEN}/neuroml/LEMS_Expr.h ${SRC_COMMON}/Common.h  ${SRC_PUGIXML}/pugixml.hpp ${SRC_PUGIXML}/pugiconfig.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

${OBJ_DIR}/LEMS_Expr${DOT_A}: ${OBJ_DIR}/LEMS_Expr${DOT_O} ${OBJ_DIR}/LEMS_Expr.yy${DOT_O} ${OBJ_DIR}/LEMS_Expr.tab${DOT_O} 
	ar rcs $@ $^
${OBJ_DIR}/LEMS_Expr${DOT_O}: ${SRC_EDEN}/neuroml/LEMS_Expr.cpp  ${SRC_EDEN}/neuroml/LEMS_Expr.h ${OBJ_DIR}/LEMS_Expr.yy${DOT_O}
	$(CXX) -c $< $(CXXFLAGS) -I ${SRC_EDEN}/neuroml/ -I ${OBJ_DIR} -o $@
${OBJ_DIR}/LEMS_Expr.tab${DOT_O}: ${SRC_EDEN}/neuroml/LEMS_Expr.y ${SRC_EDEN}/neuroml/LEMS_Expr.h
	bison --defines=${OBJ_DIR}/LEMS_Expr.tab.h --output=${OBJ_DIR}/LEMS_Expr.tab.cpp  $<
	$(CXX) -c ${OBJ_DIR}/LEMS_Expr.tab.cpp $(CXXFLAGS) -I ${SRC_EDEN}/neuroml/ -o $@
${OBJ_DIR}/LEMS_Expr.yy${DOT_O}: ${SRC_EDEN}/neuroml/LEMS_Expr.lex ${OBJ_DIR}/LEMS_Expr.tab${DOT_O}
	flex -8 --outfile=${OBJ_DIR}/LEMS_Expr.yy.cpp --header-file=${OBJ_DIR}/LEMS_Expr.yy.h $<
	$(CXX) -c ${OBJ_DIR}/LEMS_Expr.yy.cpp $(CXXFLAGS) -I ${SRC_EDEN}/neuroml/ -o $@
	
# an embedded data file
${OBJ_DIR}/LEMS_CoreComponents${DOT_O}: ${SRC_EDEN}/neuroml/LEMS_CoreComponents.inc.xml
#	note that this technique is arch-independent
	# $(LD) --relocatable --format=binary --output=$@.tmp.o $<
#	to place contents in .rodata
	# objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents $@.tmp.o $@
	# use above when xxd's 6x inflation of embedded file (byte -> "0x00, ") becomes a problem
	xxd -i $< ${OBJ_DIR}/LEMS_CoreComponents.gen.cpp
	$(CXX) -c ${OBJ_DIR}/LEMS_CoreComponents.gen.cpp $(CXXFLAGS) -o $@

# Basic LEMS testing
test_lems: ${BIN_DIR}/LEMS_Expr_Test${DOT_X}
	$< "(1+2 +-++3 * test1) < test2"
	$< "(766+cos(9)+nana*+5-nana+abs(H(mana))) > 7"
${BIN_DIR}/LEMS_Expr_Test${DOT_X}: ${OBJ_DIR}/LEMS_Expr_Test${DOT_O} ${OBJ_DIR}/LEMS_Expr${DOT_A}
	$(CXX) $^ $(CXXFLAGS)  -o $@
${OBJ_DIR}/LEMS_Expr_Test${DOT_O}: ${SRC_EDEN}/neuroml/LEMS_Expr_Test.cpp ${SRC_EDEN}/neuroml/LEMS_Expr.h ${OBJ_DIR}/LEMS_Expr.yy${DOT_O} 
	$(CXX) -c $< $(CXXFLAGS) -I ${SRC_EDEN}/neuroml/  -o $@


# external libraries
cJSON: ${OBJ_DIR}/${CJSON_NAME}${DOT_O}
${OBJ_DIR}/${CJSON_NAME}${DOT_O}: ${SRC_CJSON}/cJSON.c ${SRC_CJSON}/cJSON.h
	$(CC) -c $< $(CFLAGS) -o $@

pugixml: ${OBJ_DIR}/${PUGIXML_NAME}${DOT_O}
${OBJ_DIR}/${PUGIXML_NAME}${DOT_O}: ${SRC_PUGIXML}/pugixml.cpp ${SRC_PUGIXML}/pugixml.hpp ${SRC_PUGIXML}/pugiconfig.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@


# testing for EDEN and associated machinery

TESTBIN_EDEN := eden.${BUILD}.${TOOLCHAIN}.cpu.x 
TESTBIN_NML_PROJECTOR := nml_projector.${BUILD}.${TOOLCHAIN}.cpu.x 

nml_projector: ${BIN_DIR}/nml_projector${DOT_X}
${BIN_DIR}/nml_projector${DOT_X}: ${BIN_DIR}/nml_projector${DOT_O} ${OBJ_DIR}/Utils${DOT_O} \
		${OBJ_DIR}/${PUGIXML_NAME}${DOT_O} # third-party libs
	$(CXX) $^ $(CXXFLAGS) -o $@
${BIN_DIR}/nml_projector${DOT_O}: ${TESTING_DIR}/nml_projector.cpp ${SRC_COMMON}/Common.h \
		${SRC_PUGIXML}/pugixml.hpp ${SRC_PUGIXML}/pugiconfig.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

test:
	make -f testing/docker/Makefile test

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.yy.* $(OBJ_DIR)/*.tab.* $(OBJ_DIR)/*.a  $(OBJ_DIR)/*.gen.*
	rm -f $(BIN_DIR)/*.x
	find $(TESTING_DIR)/sandbox/. ! -name 'README.txt' ! -name '.' -type d -exec rm -rf {} +

.PHONY: all test clean ${TARGETS} ${MODULES} 
.PHONY: toolchain
