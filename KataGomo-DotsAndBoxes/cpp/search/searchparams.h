#ifndef SEARCH_SEARCHPARAMS_H_
#define SEARCH_SEARCHPARAMS_H_

#include "../core/global.h"
#include "../game/board.h"

struct SearchParams {
  //Utility function parameters
  double winLossUtilityFactor;     //Scaling for [-1,1] value for winning/losing
  double noResultUtilityForWhite; //Utility of having a no-result game (simple ko rules or nonterminating territory encore) 

  //Search tree exploration parameters
  double cpuctExploration;  //Constant factor on exploration, should also scale up linearly with magnitude of utility
  double cpuctExplorationLog; //Constant factor on log-scaling exploration, should also scale up linearly with magnitude of utility
  double cpuctExplorationBase; //Scale of number of visits at which log behavior starts having an effect

  double cpuctUtilityStdevPrior;
  double cpuctUtilityStdevPriorWeight;
  double cpuctUtilityStdevScale;

  double fpuReductionMax;   //Max amount to reduce fpu value for unexplore children
  double fpuLossProp; //Scale fpu this proportion of the way towards assuming a move is a loss.

  bool fpuParentWeightByVisitedPolicy; //For fpu, blend between parent average and parent nn value based on proportion of policy visited.
  double fpuParentWeightByVisitedPolicyPow; //If fpuParentWeightByVisitedPolicy, what power to raise the proportion of policy visited for blending.
  double fpuParentWeight; //For fpu, 0 = use parent average, 1 = use parent nn value, interpolates between.

  //Tree value aggregation parameters
  double valueWeightExponent; //Amount to apply a downweighting of children with very bad values relative to good ones
  bool useNoisePruning; //For computation of value, prune out weight that greatly exceeds what is justified by policy prior
  double noisePruneUtilityScale; //The scale of the utility difference at which useNoisePruning has effect
  double noisePruningCap; //Maximum amount of weight that noisePruning can remove

  //Uncertainty weighting
  bool useUncertainty; //Weight visits by uncertainty
  double uncertaintyCoeff; //The amount of visits weight that an uncertainty of 1 utility is.
  double uncertaintyExponent; //Visits weight scales inversely with this power of the uncertainty
  double uncertaintyMaxWeight; //Add minimum uncertainty so that the most weight a node can have is this

  //Graph search
  bool useGraphSearch; //Enable graph search instead of tree search?
  double graphSearchCatchUpLeakProb; //Chance to perform a visit to deepen a branch anyways despite being behind on visit count.
  //double graphSearchCatchUpProp; //When sufficiently far behind on visits on a transposition, catch up extra by adding up to this fraction of parents visits at once.

  //Root parameters
  bool rootNoiseEnabled;
  double rootDirichletNoiseTotalConcentration; //Same as alpha * board size, to match alphazero this might be 0.03 * 361, total number of balls in the urn
  double rootDirichletNoiseWeight; //Policy at root is this weight * noise + (1 - this weight) * nn policy

  double rootPolicyTemperature; //At the root node, scale policy probs by this power
  double rootPolicyTemperatureEarly; //At the root node, scale policy probs by this power, early in the game
  double rootFpuReductionMax; //Same as fpuReductionMax, but at root
  double rootFpuLossProp; //Same as fpuLossProp, but at root
  int rootNumSymmetriesToSample; //For the root node, sample this many random symmetries (WITHOUT replacement) and average the results together.
  bool rootSymmetryPruning; //For the root node, search only one copy of each symmetrically equivalent move.
  //We use the min of these two together, and also excess visits get pruned if the value turns out bad.
  double rootDesiredPerChildVisitsCoeff; //Funnel sqrt(this * policy prob * total visits) down any given child that receives any visits at all at the root

  //Parameters for choosing the move to play
  double chosenMoveTemperature; //Make move roughly proportional to visit count ** (1/chosenMoveTemperature)
  double chosenMoveTemperatureEarly; //Temperature at start of game
  double chosenMoveTemperatureHalflife; //Halflife of decay from early temperature to temperature for the rest of the game, scales for board sizes other than 19.
  double chosenMoveSubtract; //Try to subtract this many visits from every move prior to applying temperature
  double chosenMovePrune; //Outright prune moves that have fewer than this many visits

  bool useLcbForSelection; //Using LCB for move selection?
  double lcbStdevs; //How many stdevs a move needs to be better than another for LCB selection
  double minVisitPropForLCB; //Only use LCB override when a move has this proportion of visits as the top move
  bool useNonBuggyLcb; //LCB was very minorly buggy as of pre-v1.8. Set to true to fix.

  //Mild behavior hackery
  bool rootPruneUselessMoves; //Prune moves that are entirely useless moves that prolong the game.
  double wideRootNoise; //Explore at the root more widely

  double playoutDoublingAdvantage; //Play as if we have this many doublings of playouts vs the opponent
  Player playoutDoublingAdvantagePla; //Negate playoutDoublingAdvantage when making a move for the opponent of this player. If empty, opponent of the root player.

  float nnPolicyTemperature; //Scale neural net policy probabilities by this temperature, applies everywhere in the tree

  //Threading-related
  int nodeTableShardsPowerOfTwo; //Controls number of shards of node table for graph search transposition lookup
  double numVirtualLossesPerThread; //Number of virtual losses for one thread to add

  //Asyncbot
  int numThreads; //Number of threads
  int64_t maxVisits; //Max number of playouts from the root to think for, counting earlier playouts from tree reuse
  int64_t maxPlayouts; //Max number of playouts from the root to think for, not counting earlier playouts from tree reuse
  double maxTime; //Max number of seconds to think for

  //Same caps but when pondering
  int64_t maxVisitsPondering;
  int64_t maxPlayoutsPondering;
  double maxTimePondering;

  //Amount of time to reserve for lag when using a time control
  double lagBuffer;

  //Time control
  double treeReuseCarryOverTimeFactor; //Assume we gain this much "time" on the next move purely from % tree preserved * time spend on that tree.
  double overallocateTimeFactor; //Prefer to think this factor longer than recommended by base level time control
  double midgameTimeFactor; //Think this factor longer in the midgame, proportional to midgame weight
  double midgameTurnPeakTime; //The turn considered to have midgame weight 1.0, rising up from 0.0 in the opening, for 19x19
  double endgameTurnTimeDecay; //The scale of exponential decay of midgame weight back to 1.0, for 19x19
  double obviousMovesTimeFactor; //Think up to this factor longer on obvious moves, weighted by obviousness
  double obviousMovesPolicyEntropyTolerance; //What entropy does the policy need to be at most to be (1/e) obvious?
  double obviousMovesPolicySurpriseTolerance; //What logits of surprise does the search result need to be at most to be (1/e) obvious?

  double futileVisitsThreshold; //If a move would not be able to match this proportion of the max visits move in the time or visit or playout cap remaining, prune it.
  int64_t finishGameSearchDelayMicroseconds; //Avoid running "too fast" at the end of the game, to cost less CPU

  SearchParams();
  ~SearchParams();

  void printParams(std::ostream& out);

  //Params to use for testing, with some more recent values representative of more real use (as of Jan 2019)
  static SearchParams forTestsV1();
  //Params to use for testing, with some more recent values representative of more real use (as of Mar 2022)
  static SearchParams forTestsV2();

  static void failIfParamsDifferOnUnchangeableParameter(const SearchParams& initial, const SearchParams& dynamic);
};

#endif  // SEARCH_SEARCHPARAMS_H_
