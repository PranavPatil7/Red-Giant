/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "tuner_context.hpp"
#include "tuner.hpp"
#include "config.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define __hidden __attribute__ ((visibility("hidden")))

__hidden ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) {
  TunerContext* ctx = new TunerContext{nullptr, 0, 0, nRanks, nNodes, logFunction, nullptr};
  if (!ctx) 
    return ncclSystemError;

  const char* saveRequests = getenv("NCCL_TUNER_PLUGIN_SAVE_REQUESTS");
  if (saveRequests != nullptr && strcmp(saveRequests, "TRUE") == 0) {
    ctx->requests_ptr = std::make_unique<std::unordered_map<std::string, ConfigRequest>>();
  }

  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/ExamplePlugin: Initializing tuner for %zu nodes, %zu ranks", nNodes, nRanks);
  }

  // Try to load config file from environment variable or default location
  const char* configFile = getenv("NCCL_TUNER_CONFIG_FILE");
  if (!configFile) {
    configFile = "nccl_tuner.conf"; // default config file name
  }

  ncclResult_t result = loadConfig(ctx, configFile);
  if (result != ncclSuccess) {
    delete ctx;
    return result;
  }

  *context = ctx;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  TunerContext* ctx = (TunerContext*)context;
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  if (!ctx) return ncclInternalError;

  // Default channels
  // *nChannels = 1;

  ctx->addRequest(collType, nBytes, numPipeOps, regBuff);

  // Look for matching configuration
  for (int i = 0; i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];

    // Check if this config matches the current collective, size range, topology, pipeline ops, and regBuff
    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps) &&
        (config->regBuff == -1 || config->regBuff == regBuff)) {

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config matches. Applying algo=%s, proto=%s, channels=%d",
                         algorithmToString(config->algorithm), protocolToString(config->protocol), config->nChannels);
      }

      // Check bounds
      if (config->algorithm < numAlgo && config->protocol < numProto) {
        if (table[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Setting cost table[%s][%s] (%p) = 0.0 (was %.1f)",
                             algorithmToString(config->algorithm), protocolToString(config->protocol),
                             &table[config->algorithm][config->protocol], table[config->algorithm][config->protocol]);
          }
          table[config->algorithm][config->protocol] = 0.0; // Set low cost to prefer this configuration

          // Only override channels if not set to -1 (keep default)
          if (config->nChannels != -1) {
            *nChannels = config->nChannels;
          }

          if (ctx->logFunction) {
            if (config->nChannels == -1) {
              ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=default (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nNodes, config->nRanks);
            } else {
              ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=%d (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nChannels, config->nNodes, config->nRanks);
            }
          }
          return ncclSuccess;
        } else {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Algorithm/protocol combination [%s][%s] is marked as IGNORE",
                             algorithmToString(config->algorithm), protocolToString(config->protocol));
          }
        }
      }
    }
  }

  // If no specific config found, apply default behavior
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: No matching config found");
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo_v3(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int* nChannels) {
  TunerContext* ctx = (TunerContext*)context;
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  if (!ctx) return ncclInternalError;

  int regBuff = -1; // unused

  // Default channels
  // *nChannels = 1;

  ctx->addRequest(collType, nBytes, numPipeOps, regBuff);

  // Look for matching configuration
  for (int i = 0; i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];

    // Check if this config matches the current collective, size range, topology, pipeline ops,
    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps)) {

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config matches. Applying algo=%s, proto=%s, channels=%d",
                         algorithmToString(config->algorithm), protocolToString(config->protocol), config->nChannels);
      }

      // Check bounds
      if (config->algorithm < numAlgo && config->protocol < numProto) {
        if (table[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Setting cost table[%s][%s] (%p) = 0.0 (was %.1f)",
                             algorithmToString(config->algorithm), protocolToString(config->protocol),
                             &table[config->algorithm][config->protocol], table[config->algorithm][config->protocol]);
          }
          table[config->algorithm][config->protocol] = 0.0; // Set low cost to prefer this configuration

          // Only override channels if not set to -1 (keep default)
          if (config->nChannels != -1) {
            *nChannels = config->nChannels;
          }

          if (ctx->logFunction) {
            if (config->nChannels == -1) {
              ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=default (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nNodes, config->nRanks);
            } else {
              ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=%d (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nChannels, config->nNodes, config->nRanks);
            }
          }
          return ncclSuccess;
        } else {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Algorithm/protocol combination [%s][%s] is marked as IGNORE",
                             algorithmToString(config->algorithm), protocolToString(config->protocol));
          }
        }
      }
    }
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginDestroy(void* context) {
  if (context) {
    TunerContext* ctx = (TunerContext*)context;
    ctx->saveRequests();
    delete ctx;
  }
  return ncclSuccess;
}

#define PLUGIN_NAME "CSCS-ALPS-RCCL"

extern "C" const ncclTuner_v4_t ncclTunerPlugin_v4 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .destroy = pluginDestroy
};

extern "C" const ncclTuner_v3_t ncclTunerPlugin_v3 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo_v3,
  .destroy = pluginDestroy
};