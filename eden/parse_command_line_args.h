#ifndef EDEN_PARSE_COMMAND_LIINE_ARGS_H
#define EDEN_PARSE_COMMAND_LIINE_ARGS_H

#include "NeuroML.h"
#include "SimulatorConfig.h"
#include "EngineConfig.h"

void print_eden_cli_header(LogContext& logC);
void parse_command_line_args(int argc, char ** argv, EngineConfig & engineConfig, SimulatorConfig & config, Model & model, double & config_time_sec);

#endif
