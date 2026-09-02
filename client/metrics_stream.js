/**
 * SBFLT-22 Subtask 2: Real-time Metrics Stream Manager
 * Listens for round-completion events from the federated trainer and
 * pushes them into the dashboard within 5 seconds - AC #2
 */

const MAX_UPDATE_LATENCY_MS = 5000;

class MetricsStreamManager {
  /**
   * @param {TrainingMetricsDashboard} dashboard
   */
  constructor(dashboard) {
    this.dashboard = dashboard;
    this.eventTarget = new EventTarget();
    this._boundHandler = this._onRoundComplete.bind(this);
    this.lastLatencyMs = null;
    this.latencyBreaches = 0;
  }

  /**
   * Attach to any object that dispatches a 'roundComplete' event
   * (e.g. the federated trainer's EventTarget/EventEmitter).
   * Expected event.detail: { round, accuracy, loss, completedAt }
   */
  attachToTrainer(trainerEventSource) {
    trainerEventSource.addEventListener(
      'roundComplete',
      this._boundHandler
    );
    this._source = trainerEventSource;
  }

  detach() {
    if (this._source) {
      this._source.removeEventListener(
        'roundComplete',
        this._boundHandler
      );
    }
  }

  /**
   * Handle an incoming round-complete event.
   * Updates the dashboard immediately and measures latency between
   * round completion and the dashboard update - AC #2
   */
  _onRoundComplete(event) {
    const receivedAt = Date.now();
    const { round, accuracy, loss, completedAt } = event.detail;

    this.dashboard.addRound(round, accuracy, loss);

    const updateFinishedAt = Date.now();
    const latency = updateFinishedAt - (completedAt || receivedAt);
    this.lastLatencyMs = latency;

    if (latency > MAX_UPDATE_LATENCY_MS) {
      this.latencyBreaches += 1;
      console.warn(
        `⚠️ Dashboard update took ${latency}ms (> ${MAX_UPDATE_LATENCY_MS}ms budget) for round ${round}`
      );
    }

    this.eventTarget.dispatchEvent(
      new CustomEvent('dashboardUpdated', {
        detail: { round, latency }
      })
    );
  }

  /**
   * Manually trigger an update (useful for testing or polling fallback).
   */
  pushRoundManually(round, accuracy, loss, completedAt = Date.now()) {
    this._onRoundComplete({
      detail: { round, accuracy, loss, completedAt }
    });
  }

  getLastLatencyMs() {
    return this.lastLatencyMs;
  }

  getLatencyBreachCount() {
    return this.latencyBreaches;
  }
}

export { MetricsStreamManager, MAX_UPDATE_LATENCY_MS };
