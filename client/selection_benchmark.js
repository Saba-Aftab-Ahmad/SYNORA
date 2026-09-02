/**
 * SBFLT-21: Benchmark — capability-ranked selection vs random baseline
 * AC: "Round completion time compared against random selection baseline"
 *
 * This simulates many FL rounds against a pool of clients with known
 * capability scores, and estimates round completion time as the time of
 * the SLOWEST selected client (since FL rounds wait for stragglers).
 *
 * Run in Node with: node client/selection_benchmark.js
 * (Uses a mock capability pool instead of real navigator APIs, since this
 * script runs outside the browser.)
 */

import { ClientSelector } from './client_selection.js';

/**
 * Simulate a client's "round completion time" as inversely proportional
 * to its capability score, plus some random jitter (network variance etc).
 * Higher capabilityScore -> faster round.
 */
function simulateRoundTime(capabilityScore) {
  const baseTimeMs = 10000; // 10s for a maximally capable client
  const slowdownFactor = 1 / Math.max(capabilityScore, 0.05);
  const jitter = 0.85 + Math.random() * 0.3; // +-15% jitter
  return baseTimeMs * slowdownFactor * jitter;
}

function buildMockClientPool(count) {
  const selector = new ClientSelector();
  for (let i = 0; i < count; i++) {
    const score = Math.random(); // uniform 0..1 capability spread
    selector.registerClient({
      clientId: `client-${i}`,
      timestamp: Date.now(),
      rawSignals: {}, // not needed for this simulation
      capabilityScore: Number(score.toFixed(4)),
      scoreBreakdown: {}
    });
  }
  return selector;
}

function runBenchmark({ poolSize = 50, topN = 10, rounds = 200 } = {}) {
  const rankedResults = [];
  const randomResults = [];

  for (let r = 0; r < rounds; r++) {
    const selector = buildMockClientPool(poolSize);

    const rankedSelected = selector.selectTopN(topN, r);
    const rankedRoundTime = Math.max(
      ...rankedSelected.map(c => simulateRoundTime(c.capabilityScore))
    );
    rankedResults.push(rankedRoundTime);

    const randomSelected = selector.selectRandomN(topN, r);
    const randomRoundTime = Math.max(
      ...randomSelected.map(c => simulateRoundTime(c.capabilityScore))
    );
    randomResults.push(randomRoundTime);
  }

  const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

  const rankedAvg = avg(rankedResults);
  const randomAvg = avg(randomResults);
  const improvementPct = ((randomAvg - rankedAvg) / randomAvg) * 100;

  return {
    poolSize,
    topN,
    rounds,
    rankedAvgMs: Number(rankedAvg.toFixed(1)),
    randomAvgMs: Number(randomAvg.toFixed(1)),
    improvementPct: Number(improvementPct.toFixed(2))
  };
}

// Run when executed directly
const results = runBenchmark();
console.log('SBFLT-21 Selection Benchmark');
console.log('=============================');
console.log(`Pool size: ${results.poolSize}, Top-N: ${results.topN}, Rounds simulated: ${results.rounds}`);
console.log(`Capability-ranked avg round time: ${results.rankedAvgMs} ms`);
console.log(`Random baseline avg round time:   ${results.randomAvgMs} ms`);
console.log(`Improvement over random: ${results.improvementPct}%`);

export { runBenchmark, simulateRoundTime, buildMockClientPool };
