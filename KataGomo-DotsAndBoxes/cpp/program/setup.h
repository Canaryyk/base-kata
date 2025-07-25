#ifndef PROGRAM_SETUP_H_
#define PROGRAM_SETUP_H_

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../search/asyncbot.h"

//Some bits of initialization and main function logic shared between various programs
namespace Setup {

  void initializeSession(ConfigParser& cfg);

  enum setup_for_t {
    SETUP_FOR_GTP,
    SETUP_FOR_BENCHMARK,
    SETUP_FOR_MATCH,
    SETUP_FOR_ANALYSIS,
    SETUP_FOR_OTHER,
    SETUP_FOR_DISTRIBUTED
  };

  NNEvaluator* initializeNNEvaluator(
    const std::string& nnModelNames,
    const std::string& nnModelFiles,
    const std::string& expectedSha256,
    ConfigParser& cfg,
    Logger& logger,
    Rand& seedRand,
    int maxConcurrentEvals,
    int expectedConcurrentEvals,
    int defaultNNXLen,
    int defaultNNYLen,
    int defaultMaxBatchSize,
    bool defaultRequireExactNNLen,
    bool disableFP16,
    setup_for_t setupFor
  );

  std::vector<NNEvaluator*> initializeNNEvaluators(
    const std::vector<std::string>& nnModelNames,
    const std::vector<std::string>& nnModelFiles,
    const std::vector<std::string>& expectedSha256s,
    ConfigParser& cfg,
    Logger& logger,
    Rand& seedRand,
    int maxConcurrentEvals,
    int expectedConcurrentEvals,
    int defaultNNXLen,
    int defaultNNYLen,
    int defaultMaxBatchSize,
    bool defaultRequireExactNNLen,
    bool disableFP16,
    setup_for_t setupFor
  );

  constexpr int MAX_BOT_PARAMS_FROM_CFG = 4096;

  constexpr double DEFAULT_ANALYSIS_WIDE_ROOT_NOISE = 0.04;

  //Loads search parameters for bot from config, by bot idx.
  //Fails if no parameters are found.
  std::vector<SearchParams> loadParams(
    ConfigParser& cfg,
    setup_for_t setupFor
  );
  SearchParams loadSingleParams(
    ConfigParser& cfg,
    setup_for_t setupFor
  );

  Player parseReportAnalysisWinrates(
    ConfigParser& cfg, Player defaultPerspective
  );

  //Komi is just set to 7.5 and is not read in from cfg
  Rules loadSingleRules(
    ConfigParser& cfg
  );

  //Returns true if the user's config specified the size, false if it did not. If false, does not set defaultBoardXSizeRet or defaultBoardYSizeRet.
  bool loadDefaultBoardXYSize(
    ConfigParser& cfg,
    Logger& logger,
    int& defaultBoardXSizeRet,
    int& defaultBoardYSizeRet
  );

  std::string loadHomeDataDirOverride(
    ConfigParser& cfg
  );

  //Return config prefixes for GPU backends.
  std::vector<std::string> getBackendPrefixes();

  //Get sets of options that are mutually exclusive. Intended for use in configParser
  std::vector<std::pair<std::set<std::string>,std::set<std::string>>> getMutexKeySets();

}

#endif  // PROGRAM_SETUP_H_
