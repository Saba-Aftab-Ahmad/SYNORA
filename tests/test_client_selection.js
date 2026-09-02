/**
 * Unit Tests for SBFLT-21: Resource-Aware Client Selection
 * Run with: node tests/test_client_selection.js
 */

import { ClientSelector } from '../client/client_selection.js';

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`✅ PASS: ${message}`);
    passed++;
  } else {
    console.log(`❌ FAIL: ${message}`);
    failed++;
  }
}

function makeReport(clientId, capabilityScore) {
  return {
    clientId,
    timestamp: Date.now(),
    rawSignals: {},
    capabilityScore,
    scoreBreakdown: {}
  };
}

function test_register_and_count() {
  console.log('\nTest 1: Register clients and count');
  const selector = new ClientSelector();
  selector.registerClient(makeReport('c1', 0.5));
  selector.registerClient(makeReport('c2', 0.8));
  assert(selector.getRegisteredClientCount() === 2, 'Registered client count is 2');
}

function test_top_n_prioritizes_higher_capability() {
  console.log('\nTest 2: selectTopN prioritises higher-capability clients');
  const selector = new ClientSelector();
  selector.registerClient(makeReport('low', 0.1));
  selector.registerClient(makeReport('mid', 0.5));
  selector.registerClient(makeReport('high', 0.9));

  const selected = selector.selectTopN(2);
  const ids = selected.map(c => c.clientId);

  assert(ids.includes('high') && ids.includes('mid'), 'Top-2 selects the two highest-scoring clients');
  assert(!ids.includes('low'), 'Top-2 excludes the lowest-scoring client');
}

function test_top_n_handles_n_larger_than_pool() {
  console.log('\nTest 3: selectTopN handles N larger than available pool');
  const selector = new ClientSelector();
  selector.registerClient(makeReport('c1', 0.5));
  const selected = selector.selectTopN(5);
  assert(selected.length === 1, 'Returns all available clients when N exceeds pool size');
}

function test_no_clients_throws() {
  console.log('\nTest 4: selectTopN throws when no clients registered');
  const selector = new ClientSelector();
  let threw = false;
  try {
    selector.selectTopN(3);
  } catch (e) {
    threw = true;
  }
  assert(threw, 'Throws an error when no clients are registered');
}

function test_selection_logging() {
  console.log('\nTest 5: Selection decisions are logged with capability scores');
  const selector = new ClientSelector();
  selector.registerClient(makeReport('c1', 0.3));
  selector.registerClient(makeReport('c2', 0.7));
  selector.selectTopN(1, 1);

  const log = selector.getSelectionLog();
  assert(log.length === 1, 'One log entry created after one selection call');
  assert(
    log[0].strategy === 'capability-ranked' && log[0].selectedScores.length === 1,
    'Log entry records strategy and selected capability scores'
  );
}

function test_random_selection_logged_separately() {
  console.log('\nTest 6: Random baseline selection is logged with correct strategy label');
  const selector = new ClientSelector();
  selector.registerClient(makeReport('c1', 0.3));
  selector.registerClient(makeReport('c2', 0.7));
  selector.selectRandomN(1, 1);

  const log = selector.getSelectionLog();
  assert(log[0].strategy === 'random', 'Random selection logs with strategy "random"');
}

function test_top_n_accuracy_metric() {
  console.log('\nTest 7: computeTopNAccuracy reports 100% when selection always matches true ranking');
  const selector = new ClientSelector();
  selector.registerClient(makeReport('a', 0.9));
  selector.registerClient(makeReport('b', 0.2));
  selector.registerClient(makeReport('c', 0.5));

  selector.selectTopN(2, 1);
  selector.selectTopN(2, 2);

  const result = selector.computeTopNAccuracy();
  assert(result.accuracy === 1, 'Accuracy is 1.0 (100%) when capability ranking is deterministic');
}

function runAllTests() {
  console.log('='.repeat(60));
  console.log('Running SBFLT-21 Client Selection Tests');
  console.log('='.repeat(60));

  test_register_and_count();
  test_top_n_prioritizes_higher_capability();
  test_top_n_handles_n_larger_than_pool();
  test_no_clients_throws();
  test_selection_logging();
  test_random_selection_logged_separately();
  test_top_n_accuracy_metric();

  console.log('\n' + '='.repeat(60));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log('='.repeat(60));

  process.exitCode = failed > 0 ? 1 : 0;
}

runAllTests();
