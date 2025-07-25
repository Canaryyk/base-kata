#ifndef SEARCH_ASYNCBOT_H_
#define SEARCH_ASYNCBOT_H_

#include "../search/search.h"

class AsyncBot {
 public:
  AsyncBot(SearchParams params, NNEvaluator* nnEval, Logger* logger, const std::string& randSeed);
  ~AsyncBot();

  AsyncBot(const AsyncBot& other) = delete;
  AsyncBot& operator=(const AsyncBot& other) = delete;

  //Unless otherwise specified, functions in this class are NOT threadsafe, although they may spawn off asynchronous events.
  //Usage of this API should be single-threaded!

  const Board& getRootBoard() const;
  const BoardHistory& getRootHist() const;
  Player getRootPla() const;
  Player getPlayoutDoublingAdvantagePla() const;
  SearchParams getParams() const;

  //Get the search directly. If the asyncbot is doing anything asynchronous, the search MAY STILL BE RUNNING!
  const Search* getSearch() const;
  //Get the search, after stopping and waiting to terminate any existing search
  //Note that one still should NOT mind any threading issues using this search object and other asyncBot calls at the same time.
  Search* getSearchStopAndWait();

  //Setup, same as in search.h
  //Calling any of these will stop any ongoing search, waiting for a full stop.
  void setPosition(Player pla, const Board& board, const BoardHistory& history);
  void setPlayerAndClearHistory(Player pla);
  void setPlayerIfNew(Player pla);
  void setRootHintLoc(Loc loc);
  void setAvoidMoveUntilByLoc(const std::vector<int>& bVec, const std::vector<int>& wVec);
  void setParams(SearchParams params);
  void setParamsNoClearing(SearchParams params);
  void clearSearch();

  //Updates position and preserves the relevant subtree of search
  //Will stop any ongoing search, waiting for a full stop.
  //If the move is not legal for the current player, returns false and does nothing, else returns true
  bool makeMove(Loc moveLoc, Player movePla);
  bool isLegalTolerant(Loc moveLoc, Player movePla) const;
  bool isLegalStrict(Loc moveLoc, Player movePla) const;

  //Begin searching and produce a move.
  //Will stop any ongoing search, waiting for a full stop.
  //Asynchronously calls the provided function upon success, passing back the move and provided searchId.
  //The provided callback is expected to terminate quickly and should NOT call back into this API.
  //onSearchBegun is called when the search has initialized its tree, after which many asynchronous search query functions become safe
  void genMoveAsync(Player movePla, int searchId, const TimeControls& tc, const std::function<void(Loc,int)>& onMove);
  void genMoveAsync(Player movePla, int searchId, const TimeControls& tc, double searchFactor, const std::function<void(Loc,int)>& onMove);
  void genMoveAsync(Player movePla, int searchId, const TimeControls& tc, double searchFactor, const std::function<void(Loc,int)>& onMove, const std::function<void()>& onSearchBegun);

  //Same as genMove, but waits directly for the move and returns it here.
  Loc genMoveSynchronous(Player movePla, const TimeControls& tc);
  Loc genMoveSynchronous(Player movePla, const TimeControls& tc, double searchFactor);
  Loc genMoveSynchronous(Player movePla, const TimeControls& tc, double searchFactor, const std::function<void()>& onSearchBegun);

  //Begin pondering, returning immediately. Future genMoves may be faster if this is called.
  //Will not stop any ongoing searches.
  void ponder();
  void ponder(double searchFactor);

  //Terminate any existing searches, and then begin pondering while periodically calling the specified callback
  void analyzeAsync(
    Player movePla,
    double searchFactor,
    double callbackPeriod,
    double firstCallbackAfter,
    const std::function<void(const Search* search)>& callback
  );
  //Same as genMove but with periodic analyze callbacks
  void genMoveAsyncAnalyze(
    Player movePla,
    int searchId,
    const TimeControls& tc,
    double searchFactor,
    const std::function<void(Loc,int)>& onMove,
    double callbackPeriod,
    double firstCallbackAfter,
    const std::function<void(const Search* search)>& callback
  );
  void genMoveAsyncAnalyze(
    Player movePla,
    int searchId,
    const TimeControls& tc,
    double searchFactor,
    const std::function<void(Loc,int)>& onMove,
    double callbackPeriod,
    double firstCallbackAfter,
    const std::function<void(const Search* search)>& callback,
    const std::function<void()>& onSearchBegun
  );
  Loc genMoveSynchronousAnalyze(
    Player movePla,
    const TimeControls& tc,
    double searchFactor,
    double callbackPeriod,
    double firstCallbackAfter,
    const std::function<void(const Search* search)>& callback
  );
  Loc genMoveSynchronousAnalyze(
    Player movePla,
    const TimeControls& tc,
    double searchFactor,
    double callbackPeriod,
    double firstCallbackAfter,
    const std::function<void(const Search* search)>& callback,
    const std::function<void()>& onSearchBegun
  );

  //Signal an ongoing genMove or ponder to stop as soon as possible, and wait for the stop to happen.
  //Safe to call even if nothing is running.
  void stopAndWait();
  //Same, but does NOT wait for the stop. Also safe to call even if nothing is running.
  //Does not lock anything, so even safe to call from inside callbacks from this API.
  void stopWithoutWait();
  //Call this to permanently kill this bot and prevent future search.
  void setKilled();


 private:
  Search* search;

  std::mutex controlMutex;
  std::condition_variable threadWaitingToSearch;
  std::condition_variable userWaitingForStop;
  std::thread searchThread;

  bool isRunning;
  bool isPondering;
  bool isKilled;
  std::atomic<bool> shouldStopNow;
  int queuedSearchId;
  std::function<void(Loc,int)> queuedOnMove;
  TimeControls timeControls;
  double searchFactor;
  double analyzeCallbackPeriod;
  double analyzeFirstCallbackAfter;
  std::function<void(const Search* search)> analyzeCallback;
  std::function<void()> searchBegunCallback;

  void stopAndWaitAlreadyLocked(std::unique_lock<std::mutex>& lock);
  void waitForSearchToEnd();
  void waitForSearchToEndAlreadyLocked(std::unique_lock<std::mutex>& lock);

 public:
  //Only for internal use
  void internalSearchThreadLoop();
};


#endif  // SEARCH_ASYNCBOT_H_
