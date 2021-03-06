# line intentionally left blank, TODO anything to declare

# Select build type. Override by vars passed to Makefile from shell (ie make target VARIABLE=value )
# TODO auto build stamp does not work on Windows, just override it
BUILD_STAMP ?= $(shell "date" +"%Y-%m-%d")
BUILD ?= debug
PLATFORM ?= cpu

PROJ_BASE	?= .

# where to generate build output, in general?
OUT_DIR ?= $(PROJ_BASE)

# i.e. final artifacts
BIN_DIR ?= $(OUT_DIR)/bin

# i.e. intermediate artifacts
OBJ_DIR ?= $(OUT_DIR)/obj

# TODO WHEEL_DIR

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
TOOLCHAIN ?= $(CC)# maybe don't try to take a hint from CC LATER
$(info TOOLCHAIN = $(TOOLCHAIN))

# default toolchain is GCC
ifeq "$(TOOLCHAIN)" "cc"
$(info Assuming TOOLCHAIN = gcc...)

TOOLCHAIN := gcc
endif
# other options: USE_MPI

ifndef TARGET
# Auto-detect target OS/arch/etc, somehow
# should work on recent gcc, icc, llvm
# Hopefully, nobody will care to cross-compile using MSVC
	ifneq "$(TOOLCHAIN)" "nvcc"
		TARGET = $(shell $(TOOLCHAIN) -dumpmachine )
	endif
endif

# Negative logic to allow shorting with a positive result in shell
MAYBE_TARGET_LINUX ?= false
MAYBE_NOT_TARGET_LINUX ?= true
ifneq (,$(findstring linux,$(TARGET)))
	LIBS := $(LIBS) -ldl
	MAYBE_TARGET_LINUX := true
	MAYBE_NOT_TARGET_LINUX := false
endif

MAYBE_TARGET_MAC ?= false
MAYBE_NOT_TARGET_MAC ?= true
ifneq (,$(findstring darwin,$(TARGET)))
	# LIBS := $(LIBS) -ldl
	MAYBE_TARGET_MAC := true
	MAYBE_NOT_TARGET_MAC := false
endif

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

ifeq ($(TOOLCHAIN), nvcc)

ifdef USE_MPI
LD  := ld
CC  := mpicc
CXX := mpic++
endif

ifndef USE_MPI
LD  := ld
CC  := nvcc
CXX := nvcc
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

# only used for building wheels on Mac, assume Homebrew by default
TOOLCHAIN_LIBS_PATH ?= /usr/local/

# Compiler flags
# TODO more optimization flags
ifneq "$(TOOLCHAIN)" "nvcc"
	CFLAGS_basic := -Wall -Werror -Wno-unused-result -Wno-unused-function -lm -DBUILD_STAMP=\"$(BUILD_STAMP)\" ${CFLAGS_extra}
	CFLAGS_release := ${CFLAGS_basic} -O3
	CFLAGS_debug := ${CFLAGS_basic} -g

	CFLAGS_omp_gcc := -fopenmp
	CFLAGS_omp_icc :=  -openmp
	CFLAGS_omp :=  ${CFLAGS_omp_${TOOLCHAIN}}

	CFLAGS_cpu = ${CFLAGS_omp}

	CFLAGS ?= ${CFLAGS_${BUILD}} ${CFLAGS_${PLATFORM}} -I ${SRC_COMMON} -I ${PROJ_BASE}
endif

LIBS ?=  
ifneq (,$(findstring linux,$(TARGET)))
	LIBS := $(LIBS) -ldl
endif
ifneq (,$(findstring darwin,$(TARGET)))
	LIBS := $(LIBS) -ldl
endif
ifneq (,$(findstring mingw,$(TARGET)))
	LIBS := $(LIBS)
	EXE_EXTENSION := .exe
	EXE_EXTENSION_DIST := .exe
endif

# TODO temporary till targets are better specified in makefile
ifdef USE_MPI
	CFLAGS += -DUSE_MPI
endif

ifeq "$(TOOLCHAIN)" "nvcc"
	CFLAGS += -DUSE_GPU
endif

CXXFLAGS := ${CFLAGS} -std=c++14

# Final targets
TARGETS := eden
# Other auxiliary modules
MODULES := cJSON pugixml

EXE_EXTENSION ?= .x
# and one more time for dist-able executable binaries, practically omit extension on unices
# TODO abolish .x since it's not desirable
EXE_EXTENSION_dist ?= 

DOT_O := .${BUILD}.${TOOLCHAIN}.${PLATFORM}.o
DOT_A := .${BUILD}.${TOOLCHAIN}.${PLATFORM}.a
DOT_X := .${BUILD}.${TOOLCHAIN}.${PLATFORM}$(EXE_EXTENSION)

all: clean ${TARGETS} test

# executable targets

# TODO add a build without OpenMP, to debug OpenMP errors
eden:  ${BIN_DIR}/eden${DOT_X}
${BIN_DIR}/eden${DOT_X}: \
		${OBJ_DIR}/eden${DOT_O}\
		${OBJ_DIR}/Utils${DOT_O} \
		${OBJ_DIR}/NeuroML${DOT_O} \
		${OBJ_DIR}/LEMS_Expr${DOT_A} \
		${OBJ_DIR}/LEMS_CoreComponents${DOT_O} \
		${OBJ_DIR}/${PUGIXML_NAME}${DOT_O} \
		${OBJ_DIR}/parse_command_line_args${DOT_O}\
		${OBJ_DIR}/GenerateModel${DOT_O}

ifeq "$(CXX)" "nvcc"
	$(CXX) -std=c++14  -c ${SRC_EDEN}/backends/gpu/GpuBackend.cu -o ${OBJ_DIR}/GpuBackend.o $(CXXFLAGS)
	$(CXX) -std=c++14  -c ${SRC_EDEN}/GPU_helpers.cu -o ${OBJ_DIR}/GPU_helpers.o $(CXXFLAGS)
	$(CXX) $^ ${OBJ_DIR}/GPU_helpers.o  ${OBJ_DIR}/GpuBackend.o $(LIBS) $(CXXFLAGS) $(CFLAGS_omp) -o $@
else
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(CFLAGS_omp) -o $@
endif

	$(MAYBE_NOT_TARGET_MAC) || true # /usr/bin/ld $@ -headerpad_max_install_names -o $@

${OBJ_DIR}/eden${DOT_O}: ${SRC_EDEN}/Eden.cpp ${SRC_EDEN}/NeuroML.h ${SRC_EDEN}/neuroml/LEMS_Expr.h ${SRC_COMMON}/Common.h  ${SRC_COMMON}/MMMallocator.h
	$(CXX) -c $< $(CXXFLAGS) $(CFLAGS_omp) -o $@


# own helper libraries
${OBJ_DIR}/Utils${DOT_O}: ${SRC_COMMON}/Utils.cpp ${SRC_COMMON}/Common.h
	$(CXX) -c $< $(CXXFLAGS) -o $@ 

${OBJ_DIR}/NeuroML${DOT_O}: ${SRC_EDEN}/NeuroML.cpp ${SRC_EDEN}/NeuroML.h ${SRC_EDEN}/neuroml/LEMS_Expr.h ${SRC_COMMON}/Common.h  ${SRC_PUGIXML}/pugixml.hpp ${SRC_PUGIXML}/pugiconfig.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@ -I.

${OBJ_DIR}/parse_command_line_args${DOT_O}: ${SRC_EDEN}/parse_command_line_args.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@ -I.

${OBJ_DIR}/GenerateModel${DOT_O}: ${SRC_EDEN}/GenerateModel.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@ -I.


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


EXTRA_WHEEL_PACKAGE_TAGS ?=
# Python wheel with embedded executable of EDEN
# building a wheel out of tree simply doesn't work, without warning. Copy the package tree in a temporary location and build there (cwd must also be on location of setup.py or the files won't be added) 
wheel: eden
	rm -rf $(TESTING_DIR)/sandbox/python_package/
	cp -r $(TESTING_DIR)/python_package $(TESTING_DIR)/sandbox/
	"mkdir" -p $(TESTING_DIR)/sandbox/python_package/bin/
	mv $(TESTING_DIR)/sandbox/python_package/eden_tools $(TESTING_DIR)/sandbox/python_package/eden_simulator
	cp ${BIN_DIR}/eden${DOT_X} $(TESTING_DIR)/sandbox/python_package/bin/eden$(EXE_EXTENSION_DIST)
	$(MAYBE_NOT_TARGET_MAC) || bash $(TESTING_DIR)/mac/bundle-dylibs.bash $(TESTING_DIR)/sandbox/python_package/bin/eden$(EXE_EXTENSION_DIST) "$(TOOLCHAIN_LIBS_PATH)"
	cd $(TESTING_DIR)/sandbox/python_package && python3 setup_wheel.py --package-version ${WHEEL_VERSION}  bdist_wheel  $(EXTRA_WHEEL_PACKAGE_TAGS)
	$(MAYBE_NOT_TARGET_LINUX) || python3 -m auditwheel repair --plat ${WHEEL_TARGET_PLAT} --only-plat --wheel-dir $(TESTING_DIR)/sandbox/python_package/dist/ $(TESTING_DIR)/sandbox/python_package/dist/eden_simulator-${WHEEL_VERSION}-py3-none-${WHEEL_PLAT}.whl

# external libraries
cJSON: ${OBJ_DIR}/${CJSON_NAME}${DOT_O}
${OBJ_DIR}/${CJSON_NAME}${DOT_O}: ${SRC_CJSON}/cJSON.c ${SRC_CJSON}/cJSON.h
	$(CC) -c $< $(CFLAGS) -o $@

pugixml: ${OBJ_DIR}/${PUGIXML_NAME}${DOT_O}
${OBJ_DIR}/${PUGIXML_NAME}${DOT_O}: ${SRC_PUGIXML}/pugixml.cpp ${SRC_PUGIXML}/pugixml.hpp ${SRC_PUGIXML}/pugiconfig.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@


# testing for EDEN and associated machinery

TESTBIN_EDEN := eden.${BUILD}.${TOOLCHAIN}.cpu$(EXE_EXTENSION) 
TESTBIN_NML_PROJECTOR := nml_projector.${BUILD}.${TOOLCHAIN}.cpu$(EXE_EXTENSION)

nml_projector: ${BIN_DIR}/nml_projector${DOT_X}
${BIN_DIR}/nml_projector${DOT_X}: ${BIN_DIR}/nml_projector${DOT_O} ${OBJ_DIR}/Utils${DOT_O} \
		${OBJ_DIR}/${PUGIXML_NAME}${DOT_O} # third-party libs
	$(CXX) $^ $(CXXFLAGS) -o $@
${BIN_DIR}/nml_projector${DOT_O}: ${TESTING_DIR}/nml_projector.cpp ${SRC_COMMON}/Common.h \
		${SRC_PUGIXML}/pugixml.hpp ${SRC_PUGIXML}/pugiconfig.hpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

test:
	make -f testing/docker/Makefile test

run: eden
	bin/eden.$(BUILD).gcc.cpu.x nml examples/LEMS_NML2_Ex25_MultiComp.xml

run_gpu: eden
	bin/eden.$(BUILD).nvcc.cpu.x nml examples/LEMS_NML2_Ex25_MultiComp.xml

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.yy.* $(OBJ_DIR)/*.tab.* $(OBJ_DIR)/*.a  $(OBJ_DIR)/*.gen.*
	rm -f $(BIN_DIR)/*$(EXE_EXTENSION)
	"find" $(TESTING_DIR)/sandbox/. ! -name 'README.txt' ! -name '.' -type d -exec rm -rf {} +

.PHONY: all run run_gpu test clean ${TARGETS} ${MODULES}
.PHONY: toolchain
