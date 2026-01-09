#pragma once

#include "common.hpp"
#include "config.hpp"
#include "err.hpp"

#include <unordered_map>
#include <memory>
#include <fstream>


typedef struct {
  ncclFunc_t collType;
  size_t bytes;
  int numPipeOps;
  int regBuff;
} ConfigRequest;

class TunerContext {

public:

    std::unique_ptr<TuningConfig[]> configs;
    int numConfigs;
    int maxConfigs;         // Added to track allocated size
    size_t nRanks;
    size_t nNodes;
    ncclDebugLogger_t logFunction;
    std::unique_ptr<std::unordered_map<std::string, ConfigRequest>> requests_ptr = nullptr;


    void addRequest(ncclFunc_t collType, size_t bytes, int numPipeOps, int regBuff) {
        if (this->requests_ptr == nullptr) {
            return;
        }
        std::string key = this->reqToStr(collType, bytes, numPipeOps, regBuff);
        if (this->requests_ptr->find(key) == this->requests_ptr->end()) {
            (*this->requests_ptr)[key] = ConfigRequest{collType, bytes, numPipeOps, regBuff};
        }    
    }

    void saveRequests() {
        if (this->requests_ptr == nullptr) {
            return;
        }
        const char* configFile = getenv("NCCL_TUNER_TUNING_FILE");
        if (!configFile) {
            configFile = "tuning.csv"; // default config file name
        }
        std::ofstream outstream(configFile);
        if (!outstream) {
            if (this->logFunction) {
                this->logFunction(
                    NCCL_LOG_WARN,
                    NCCL_TUNING,
                    __FILE__,
                    __LINE__,
                    "TUNER: Could not save tuning requests"
                );
            }
            return;
        }
        // save header
        outstream << "collective,num_bytes,nNodes,nRanks,numPipeOps,regBuff\n";
        for (const auto& p : *(this->requests_ptr)) {
            auto v = p.second;
            outstream << collTypeToString(v.collType) << "," \
                << v.bytes << "," \
                << this->nNodes << "," << this->nRanks << "," \
                << v.numPipeOps << "," << v.regBuff << "\n";
        }
        outstream.close();

        if (this->logFunction) {
          this->logFunction(
              NCCL_LOG_WARN,
              NCCL_TUNING,
              __FILE__,
              __LINE__,
              "TUNER: Saved tuning requests at %s",
              configFile
          );
        }
    }

private:

    std::string reqToStr(ncclFunc_t collType, size_t bytes, int numPipeOps, int regBuff) {
        std::string name(collTypeToString(collType));
        name += std::to_string(bytes) + "_";
        name += std::to_string(numPipeOps) + "_";
        name += std::to_string(regBuff) + "_";
        return name;
    }

};

ncclResult_t loadConfig(TunerContext* ctx, const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Config file %s not found, using defaults", filename);
    }
    return ncclSuccess; // Not finding config file is not an error
  }

  // First pass: count valid configuration lines
  int configCount = countConfigLines(filename);
  if (configCount == 0) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: No valid configurations found in %s", filename);
    }
    fclose(file);
    return ncclSuccess;
  }

  // Allocate memory for configurations based on actual count
  ctx->configs = std::make_unique<TuningConfig[]>(configCount);
  if (!ctx->configs) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Failed to allocate memory for %d configurations", configCount);
    }
    fclose(file);
    return ncclSystemError;
  }

  ctx->maxConfigs = configCount;
  ctx->numConfigs = 0;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: Allocated memory for %d configurations", configCount);
  }

  // Reset file pointer to beginning
  fseek(file, 0, SEEK_SET);

  char line[MAX_LINE_LENGTH];

  while (fgets(line, sizeof(line), file) && ctx->numConfigs < ctx->maxConfigs) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Parse CSV format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
    char* token;
    char* tokens[CONFIG_FIELDS_MAX];
    int tokenCount = 0;

    // Make a copy of the line for tokenizing
    char lineCopy[MAX_LINE_LENGTH];
    strncpy(lineCopy, line, sizeof(lineCopy));
    lineCopy[sizeof(lineCopy) - 1] = '\0';

    // Tokenize by comma
    token = strtok(lineCopy, ",");
    while (token != NULL && tokenCount < CONFIG_FIELDS_MAX) {
      // Trim whitespace
      while (*token == ' ' || *token == '\t') token++;
      char* end = token + strlen(token) - 1;
      while (end > token && (*end == ' ' || *end == '\t')) {
        *end = '\0';
        end--;
      }
      tokens[tokenCount++] = token;
      token = strtok(NULL, ",");
    }

    // Validate field count: support required fields (8), with pipeOps (9), or with regBuff (10)
    if (tokenCount >= CONFIG_FIELDS_REQUIRED && tokenCount <= CONFIG_FIELDS_MAX) {
      TuningConfig* config = &ctx->configs[ctx->numConfigs];
      config->collType = parseCollType(tokens[CONFIG_FIELD_COLLTYPE]);
      config->minBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MINBYTES], NULL, 10);
      config->maxBytes = (size_t)strtoull(tokens[CONFIG_FIELD_MAXBYTES], NULL, 10);
      config->algorithm = parseAlgorithm(tokens[CONFIG_FIELD_ALGORITHM]);
      config->protocol = parseProtocol(tokens[CONFIG_FIELD_PROTOCOL]);
      config->nChannels = atoi(tokens[CONFIG_FIELD_CHANNELS]);
      config->nNodes = atoi(tokens[CONFIG_FIELD_NNODES]);
      config->nRanks = atoi(tokens[CONFIG_FIELD_NRANKS]);

      // numPipeOps is optional (9th field, index 8)
      if (tokenCount >= CONFIG_FIELDS_WITH_PIPEOPS) {
        config->numPipeOps = atoi(tokens[CONFIG_FIELD_PIPEOPS]);
      } else {
        config->numPipeOps = -1; // -1 means match any numPipeOps
      }

      // regBuff is optional (10th field, index 9)
      if (tokenCount >= CONFIG_FIELDS_WITH_REGBUFF) {
        config->regBuff = atoi(tokens[CONFIG_FIELD_REGBUFF]);
      } else {
        config->regBuff = -1; // -1 means match any regBuff value
      }

      ctx->numConfigs++;

      if (ctx->logFunction) {
        if (config->numPipeOps == -1 && config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks);
        } else if (config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps);
        } else if (config->numPipeOps == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->regBuff);
        } else {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
        }
      }
    }
  }

  fclose(file);
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: Loaded %d tuning configurations from %s", ctx->numConfigs, filename);
  }
  return ncclSuccess;
}