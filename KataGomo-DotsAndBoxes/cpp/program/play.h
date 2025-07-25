#ifndef PROGRAM_PLAY_H_
#define PROGRAM_PLAY_H_

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/multithread.h"
#include "../core/rand.h"
#include "../core/threadsafecounter.h"
#include "../core/threadsafequeue.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/sgf.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../search/search.h"
#include "../search/searchparams.h"
#include "../program/playsettings.h"

struct InitialPosition {
  Board board;
  BoardHistory hist;
  Player pla;

  InitialPosition();
  InitialPosition(const Board& board, const BoardHistory& hist, Player pla);
  ~InitialPosition();
};


struct OtherGameProperties {
  bool isSgfPos = false;
  bool isHintPos = false;
  bool allowPolicyInit = true;

  int hintTurn = -1;
  Hash128 hintPosHash;
  Loc hintLoc = Board::NULL_LOC;

  //Note: these two behave slightly differently than the ones in searchParams - as properties for the whole
  //game, they make the playouts *actually* vary instead of only making the neural net think they do.
  double playoutDoublingAdvantage = 0.0;
  Player playoutDoublingAdvantagePla = C_EMPTY;
};

//Object choosing random initial rules and board sizes for games. Threadsafe.
class GameInitializer {
 public:
  GameInitializer(ConfigParser& cfg, Logger& logger);
  GameInitializer(ConfigParser& cfg, Logger& logger, const std::string& randSeed);
  ~GameInitializer();

  GameInitializer(const GameInitializer&) = delete;
  GameInitializer& operator=(const GameInitializer&) = delete;

  //Initialize everything for a new game with random rules, unless initialPosition is provided, in which case it uses
  //those rules (possibly with noise to the komi given in that position)
  //Also, mutates params to randomize appropriate things like utilities, but does NOT fill in all the settings.
  //User should make sure the initial params provided makes sense as a mean or baseline.
  //Does NOT place handicap stones, users of this function need to place them manually
  void createGame(
    Board& board, Player& pla, BoardHistory& hist,
    SearchParams& params,
    const InitialPosition* initialPosition,
    const PlaySettings& playSettings,
    OtherGameProperties& otherGameProps,
    const Sgf::PositionSample* startPosSample
  );

  //A version that doesn't randomize params
  void createGame(
    Board& board, Player& pla, BoardHistory& hist,
    const InitialPosition* initialPosition,
    const PlaySettings& playSettings,
    OtherGameProperties& otherGameProps,
    const Sgf::PositionSample* startPosSample
  );

  Rules randomizeScoringAndTaxRules(Rules rules, Rand& randToUse) const;

  //Only sample the space of possible rules
  Rules createRules();
  bool isAllowedBSize(int xSize, int ySize);

  std::vector<int> getAllowedBSizes() const;
  int getMinBoardXSize() const;
  int getMinBoardYSize() const;
  int getMaxBoardXSize() const;
  int getMaxBoardYSize() const;

 private:
  void initShared(ConfigParser& cfg, Logger& logger);
  void createGameSharedUnsynchronized(
    Board& board, Player& pla, BoardHistory& hist,
    const InitialPosition* initialPosition,
    const PlaySettings& playSettings,
    OtherGameProperties& otherGameProps,
    const Sgf::PositionSample* startPosSample
  );
  Rules createRulesUnsynchronized();

  std::mutex createGameMutex;
  Rand rand;

  std::vector<std::string> allowedScoringRuleStrs;
  std::vector<int> allowedScoringRules;
  std::vector<int> allowedBSizes;
  std::vector<double> allowedBSizeRelProbs;

  double allowRectangleProb;

  double komiMean;
  double komiStdev;
  double komiBigStdevProb;
  double komiBigStdev;

  double noResultRandRadius;

  std::vector<Sgf::PositionSample> startPoses;
  std::vector<double> startPosCumProbs;
  double startPosesProb;

  std::vector<Sgf::PositionSample> hintPoses;
  std::vector<double> hintPosCumProbs;
  double hintPosesProb;

  int minBoardXSize;
  int minBoardYSize;
  int maxBoardXSize;
  int maxBoardYSize;
};


//Object for generating and servering evenly distributed pairings between different bots. Threadsafe.
class MatchPairer {
 public:
  //Holds pointers to the various nnEvals, but does NOT take ownership for freeing them.
  MatchPairer(
    ConfigParser& cfg,
    int numBots,
    const std::vector<std::string>& botNames,
    const std::vector<NNEvaluator*>& nnEvals,
    const std::vector<SearchParams>& baseParamss,
    bool forSelfPlay,
    bool forGateKeeper
  );
  MatchPairer(
    ConfigParser& cfg,
    int numBots,
    const std::vector<std::string>& botNames,
    const std::vector<NNEvaluator*>& nnEvals,
    const std::vector<SearchParams>& baseParamss,
    bool forSelfPlay,
    bool forGateKeeper,
    const std::vector<bool>& excludeBot
  );

  ~MatchPairer();

  struct BotSpec {
    int botIdx;
    std::string botName;
    NNEvaluator* nnEval;
    SearchParams baseParams;
  };

  MatchPairer(const MatchPairer&) = delete;
  MatchPairer& operator=(const MatchPairer&) = delete;

  //Get the total number of games that the matchpairer will generate
  int64_t getNumGamesTotalToGenerate() const;

  //Get next matchup and log stuff
  bool getMatchup(
    BotSpec& botSpecB, BotSpec& botSpecW, Logger& logger
  );

 private:
  int numBots;
  std::vector<std::string> botNames;
  std::vector<NNEvaluator*> nnEvals;
  std::vector<SearchParams> baseParamss;

  std::vector<bool> excludeBot;
  std::vector<int> secondaryBots;
  std::vector<int> blackPriority;
  std::vector<std::pair<int,int>> extraPairs;
  std::vector<std::pair<int,int>> nextMatchups;
  std::vector<std::pair<int,int>> nextMatchupsBuf;
  Rand rand;

  int matchRepFactor;
  int repsOfLastMatchup;

  int64_t numGamesStartedSoFar;
  int64_t numGamesTotal;
  int64_t logGamesEvery;

  std::mutex getMatchupMutex;

  std::pair<int,int> getMatchupPairUnsynchronized();
};


//Functions to run a single game or other things
namespace Play {

  //In the case where checkForNewNNEval is provided, will MODIFY the provided botSpecs with any new nneval!
  FinishedGameData* runGame(
    const Board& startBoard, Player pla, const BoardHistory& startHist, 
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    const std::string& searchRandSeed,
    bool clearBotBeforeSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, const std::function<bool()>& shouldStop,
    const WaitableFlag* shouldPause,
    const PlaySettings& playSettings, const OtherGameProperties& otherGameProps,
    Rand& gameRand,
    std::function<NNEvaluator*()> checkForNewNNEval,
    std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const std::vector<double>&, const Search*)> onEachMove
  );

  //In the case where checkForNewNNEval is provided, will MODIFY the provided botSpecs with any new nneval!
  FinishedGameData* runGame(
    const Board& startBoard, Player pla, const BoardHistory& startHist, 
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    Search* botB, Search* botW,
    bool clearBotBeforeSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, const std::function<bool()>& shouldStop,
    const WaitableFlag* shouldPause,
    const PlaySettings& playSettings, const OtherGameProperties& otherGameProps,
    Rand& gameRand,
    std::function<NNEvaluator*()> checkForNewNNEval,
    std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const std::vector<double>&, const Search*)> onEachMove
  );

}


//Class for running a game and enqueueing the result as training data.
//Wraps together most of the neural-net-independent parameters to spawn and run a full game.
class GameRunner {
  bool logSearchInfo;
  bool logMoves;
  int maxMovesPerGame;
  bool clearBotBeforeSearch;
  PlaySettings playSettings;
  GameInitializer* gameInit;

public:
  GameRunner(ConfigParser& cfg, PlaySettings playSettings, Logger& logger);
  GameRunner(ConfigParser& cfg, const std::string& gameInitRandSeed, PlaySettings fModes, Logger& logger);
  ~GameRunner();

  //Will return NULL if stopped before the game completes. The caller is responsible for freeing the data
  //if it isn't NULL.
  //afterInitialization can be used to run any post-initialization configuration on the search
  FinishedGameData* runGame(
    const std::string& seed,
    const MatchPairer::BotSpec& botSpecB,
    const MatchPairer::BotSpec& botSpecW,
    const Sgf::PositionSample* startPosSample,
    Logger& logger,
    const std::function<bool()>& shouldStop,
    const WaitableFlag* shouldPause,
    std::function<NNEvaluator*()> checkForNewNNEval,
    std::function<void(const MatchPairer::BotSpec&, Search*)> afterInitialization,
    std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const std::vector<double>&, const Search*)> onEachMove
  );

  const GameInitializer* getGameInitializer() const;

};


#endif  // PROGRAM_PLAY_H_
