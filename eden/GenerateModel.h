//
// Created by max on 04-10-21.
//

#ifndef EDEN_GENERATEMODEL_H
#define EDEN_GENERATEMODEL_H

#include "RawTables.h"
#include "NeuroML.h"
#include "SimulatorConfig.h"
#include "EngineConfig.h"

bool GenerateModel(const Model &model, const SimulatorConfig &config, EngineConfig &engine_config, RawTables &tabs);

#endif
