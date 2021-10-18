#ifndef EDEN_PARSE_COMMAND_LIINE_ARGS_H
#define EDEN_PARSE_COMMAND_LIINE_ARGS_H

#include "NeuroML.h"
#include "SimulatorConfig.h"

void print_eden_cli_header();
void parse_command_line_args(int argc, char ** argv, SimulatorConfig & config, Model & model, double & config_time_sec);

#endif
