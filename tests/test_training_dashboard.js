/**
 * Unit Tests for SBFLT-22: Real-time Training Metrics Dashboard
 */

import { TrainingMetricsDashboard } from '../client/training_dashboard.js';
import { MetricsStreamManager, MAX_UPDATE_LATENCY_MS } from '../client/metrics_stream.js';

function setupDom(containerId) {
  document.body.innerHTML = `<div id="${containerId}"></div>`;
}

// Test 1: Dashboard creation
function test_dashboard_creation() {
  console.log('\nTest 1: Dashboard creation');
  try {
    setupDom('dash1');
    const dashboard = new TrainingMetricsDashboard('dash1');
    if (dashboard.getRounds().length === 0) {
      console.log('✅ PASS: Dashboard created with empty state');
    } else {
      console.log('❌ FAIL: Dashboard not empty on init');
    }
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 2: Add round metrics
function test_add_round_metrics() {
  console.log('\nTest 2: Add round metrics');
  try {
    setupDom('dash2');
    const dashboard = new TrainingMetricsDashboard('dash2');
    dashboard.addRound(1, 0.85, 0.42);

    const latest = dashboard.getLatestRound();
    if (latest.round === 1 && latest.accuracy === 0.85 && latest.loss === 0.42) {
      console.log('✅ PASS: Round metrics stored correctly');
    } else {
      console.log('❌ FAIL: Stored metrics incorrect');
    }
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 3: Multiple rounds accumulate (AC #3)
function test_multiple_rounds_accumulate() {
  console.log('\nTest 3: All completed rounds reflected accurately');
  try {
    setupDom('dash3');
    const dashboard = new TrainingMetricsDashboard('dash3');
    dashboard.addRound(1, 0.70, 0.60);
    dashboard.addRound(2, 0.78, 0.50);
    dashboard.addRound(3, 0.83, 0.40);

    const rounds = dashboard.getRounds();
    if (rounds.length === 3 && rounds[2].round === 3) {
      console.log(`✅ PASS: ${rounds.length} rounds accumulated in order`);
    } else {
      console.log('❌ FAIL: Rounds did not accumulate correctly');
    }
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 4: DOM renders without navigation (AC #4)
function test_render_updates_dom_in_place() {
  console.log('\nTest 4: Dashboard updates DOM in place (no navigation)');
  try {
    setupDom('dash4');
    const dashboard = new TrainingMetricsDashboard('dash4');
    const urlBefore = window.location.href;

    dashboard.addRound(1, 0.9, 0.3);

    const tbody = document.getElementById('dash4-tbody');
    const rowExists = tbody.innerHTML.includes('90.00%');
    const urlUnchanged = window.location.href === urlBefore;

    if (rowExists && urlUnchanged) {
      console.log('✅ PASS: DOM updated in place, no page navigation');
    } else {
      console.log('❌ FAIL: DOM not updated correctly or navigation occurred');
    }
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 5: Input validation
function test_invalid_metrics_rejected() {
  console.log('\nTest 5: Invalid metrics rejected');
  try {
    setupDom('dash5');
    const dashboard = new TrainingMetricsDashboard('dash5');
    let threw = false;
    try {
      dashboard.addRound(1, 1.5, 0.3); // invalid accuracy > 1
    } catch (e) {
      threw = true;
    }
    console.log(threw ? '✅ PASS: Invalid accuracy rejected' : '❌ FAIL: Invalid accuracy accepted');
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 6: Real-time update via MetricsStreamManager (AC #1, #2)
function test_stream_updates_dashboard() {
  console.log('\nTest 6: Stream manager updates dashboard live');
  try {
    setupDom('dash6');
    const dashboard = new TrainingMetricsDashboard('dash6');
    const stream = new MetricsStreamManager(dashboard);

    stream.pushRoundManually(1, 0.88, 0.25, Date.now());

    const latest = dashboard.getLatestRound();
    if (latest && latest.round === 1 && latest.accuracy === 0.88) {
      console.log('✅ PASS: Dashboard updated via stream manager');
    } else {
      console.log('❌ FAIL: Dashboard not updated via stream');
    }
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 7: Update latency under 5s budget (AC #2)
function test_update_latency_under_budget() {
  console.log('\nTest 7: Update latency within 5s budget');
  try {
    setupDom('dash7');
    const dashboard = new TrainingMetricsDashboard('dash7');
    const stream = new MetricsStreamManager(dashboard);

    const completedAt = Date.now();
    stream.pushRoundManually(1, 0.8, 0.35, completedAt);

    const latency = stream.getLastLatencyMs();
    if (latency !== null && latency < MAX_UPDATE_LATENCY_MS) {
      console.log(`✅ PASS: Latency = ${latency}ms (budget ${MAX_UPDATE_LATENCY_MS}ms)`);
    } else {
      console.log(`❌ FAIL: Latency ${latency}ms exceeds budget`);
    }
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

function runAllTests() {
  console.log('='.repeat(60));
  console.log('Running SBFLT-22 Dashboard Tests');
  console.log('='.repeat(60));

  test_dashboard_creation();
  test_add_round_metrics();
  test_multiple_rounds_accumulate();
  test_render_updates_dom_in_place();
  test_invalid_metrics_rejected();
  test_stream_updates_dashboard();
  test_update_latency_under_budget();

  console.log('\n' + '='.repeat(60));
  console.log('All tests completed');
  console.log('='.repeat(60));
}

export { runAllTests };

// Auto-run when executed directly with: node test_training_dashboard.js
runAllTests();
