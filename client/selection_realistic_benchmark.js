/**
 * SBFLT-21: Realistic Top-N Accuracy Test
 * AC: "Capability-ranked clients selected in top-N for 90% of rounds"
 *
 * The naive test (see tests/test_client_selection.js) uses STATIC scores
 * that never change between report-time and selection-time, which trivially
 * gives 100% accuracy and doesn't actually test the AC. In a real deployment,
 * two things can cause the "true" top-N at selection time to differ from
 * what was ranked at report time:
 *
 *   1. STALENESS: a client's capability score was reported some time ago
 *      (network conditions, CPU load, battery state changed since).
 *   2. DROPOUT: a client that reported a high score may have closed the
 *      tab / gone offline before the round actually starts.
 *
 * This script simulates both effects across many rounds and reports what
 * fraction of rounds the capability-ranked selection still matches the
 * TRUE (at-selection-time) top-N — this is the real evidence for the 90% AC.
 *
 * Run with: node client/selection_realistic_benchmark.js
 */

import { ClientSelector } from './client_selection.js';

/**
 * Simulate score drift between report time and "true" selection time.
 * driftStdDev controls how much a score can wander (e.g. 0.05 = small drift).
 */
function driftScore(originalScore, driftStdDev) {
  // Simple Gaussian-ish noise via sum of uniforms (Irwin-Hall approx)
  const noise = ((Math.random() + Math.random() + Math.random()) / 3 - 0.5) * 2 * driftStdDev;
  return Math.min(Math.max(originalScore + noise, 0), 1);
}

function runRealisticAccuracyTest({
  poolSize = 50,
  topN = 10,
  rounds = 500,
  driftStdDev = 0.08,   // how much scores can drift between report & round start
  dropoutRate = 0.05    // probability a given client goes offline before round starts
} = {}) {
  let matchingRounds = 0;
  const roundDetails = [];

  for (let r = 0; r < rounds; r++) {
    const selector = new ClientSelector();

    // Each client reports a score at "report time"
    const reportedScores = new Map();
    for (let i = 0; i < poolSize; i++) {
      const score = Number(Math.random().toFixed(4));
      reportedScores.set(`client-${i}`, score);
      selector.registerClient({
        clientId: `client-${i}`,
        timestamp: Date.now(),
        rawSignals: {},
        capabilityScore: score,
        scoreBreakdown: {}
      });
    }

    // System selects top-N based on the REPORTED (possibly now-stale) scores
    const selected = selector.selectTopN(topN, r);
    const selectedIds = new Set(selected.map(c => c.clientId));

    // Simulate what happens between report time and actual round start:
    // scores drift, and some clients drop out entirely.
    const trueScoresAtRoundStart = new Map();
    for (const [clientId, reportedScore] of reportedScores.entries()) {
      const droppedOut = Math.random() < dropoutRate;
      if (droppedOut) continue; // not available at round start
      trueScoresAtRoundStart.set(clientId, driftScore(reportedScore, driftStdDev));
    }

    // The TRUE top-N is computed from who's actually available with their
    // actual (drifted) capability at round start.
    const trueRanked = Array.from(trueScoresAtRoundStart.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topN)
      .map(([clientId]) => clientId);
    const trueTopNIds = new Set(trueRanked);

    // "Matches" = the clients the system selected are still (mostly) the
    // true top-N once reality (drift + dropout) is accounted for.
    // We use overlap fraction >= 0.9 (i.e. at least 90% of the selected
    // clients are still genuinely top-tier) as the pass criterion for a round,
    // since exact-set-equality is an unrealistically strict bar for any
    // system relying on periodically-reported, not-live, scores.
    const overlap = [...selectedIds].filter(id => trueTopNIds.has(id)).length;
    const overlapFraction = overlap / topN;
    const roundPasses = overlapFraction >= 0.9;

    if (roundPasses) matchingRounds++;
    roundDetails.push({ round: r, overlapFraction });
  }

  return {
    poolSize,
    topN,
    rounds,
    driftStdDev,
    dropoutRate,
    matchingRounds,
    accuracy: Number((matchingRounds / rounds).toFixed(4)),
    meetsAC: matchingRounds / rounds >= 0.9
  };
}

// Only auto-run when this file is executed directly (not when imported),
// so importing runRealisticAccuracyTest elsewhere doesn't trigger a run.
if (import.meta.url === `file://${process.argv[1]}`) {
  const result = runRealisticAccuracyTest();
  console.log('SBFLT-21 Realistic Top-N Accuracy Test (AC3)');
  console.log('==============================================');
  console.log(`Pool size: ${result.poolSize}, Top-N: ${result.topN}, Rounds: ${result.rounds}`);
  console.log(`Score drift std-dev: ${result.driftStdDev}, Dropout rate: ${result.dropoutRate}`);
  console.log(`Rounds meeting >=90% overlap with true top-N: ${result.matchingRounds}/${result.rounds}`);
  console.log(`Accuracy: ${(result.accuracy * 100).toFixed(2)}%`);
  console.log(`Meets AC3 (>=90%) at these conditions: ${result.meetsAC}`);
  console.log('\nNOTE: accuracy will drop if driftStdDev or dropoutRate increase.');
  console.log('Try re-running with harsher parameters to find the breaking point -');
  console.log('that breaking point IS your real answer to "why 90%" in evaluation.');
}

export { runRealisticAccuracyTest, driftScore };
