/**
 * SBFLT-22 Subtask 1: Training Metrics Dashboard UI
 * Displays live accuracy and loss per federated learning round.
 * Renders in-place (no page navigation required) - AC #4
 */

class TrainingMetricsDashboard {
  /**
   * @param {string} containerId - id of the DOM element to render into
   */
  constructor(containerId) {
    this.containerId = containerId;
    this.container = document.getElementById(containerId);
    this.rounds = []; // { round, accuracy, loss, timestamp }

    if (!this.container) {
      console.warn(
        `Dashboard container "#${containerId}" not found in DOM`
      );
    }

    this._renderShell();
  }

  /**
   * Render the static shell (title + empty table) once.
   * Subsequent updates only touch the table body - AC #4 (no navigation).
   */
  _renderShell() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="metrics-dashboard">
        <h3>Federated Training Metrics</h3>
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Round</th>
              <th>Accuracy</th>
              <th>Loss</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody id="${this.containerId}-tbody"></tbody>
        </table>
        <p class="metrics-empty" id="${this.containerId}-empty">
          Waiting for first round to complete...
        </p>
      </div>
    `;
  }

  /**
   * Add metrics for a completed round and re-render.
   * All completed rounds accumulate and stay visible - AC #3
   */
  addRound(roundNumber, accuracy, loss) {
    if (typeof roundNumber !== 'number' || roundNumber < 0) {
      throw new Error('roundNumber must be a non-negative number');
    }
    if (typeof accuracy !== 'number' || accuracy < 0 || accuracy > 1) {
      throw new Error('accuracy must be a number between 0 and 1');
    }
    if (typeof loss !== 'number' || loss < 0) {
      throw new Error('loss must be a non-negative number');
    }

    const existingIndex = this.rounds.findIndex(
      r => r.round === roundNumber
    );
    const entry = {
      round: roundNumber,
      accuracy,
      loss,
      timestamp: Date.now()
    };

    if (existingIndex >= 0) {
      this.rounds[existingIndex] = entry;
    } else {
      this.rounds.push(entry);
      this.rounds.sort((a, b) => a.round - b.round);
    }

    this.render();
    return entry;
  }

  /**
   * Re-render only the table body (cheap DOM update, keeps user on the
   * same board/view) - AC #4
   */
  render() {
    if (!this.container) return;

    const tbody = document.getElementById(`${this.containerId}-tbody`);
    const emptyMsg = document.getElementById(`${this.containerId}-empty`);
    if (!tbody) return;

    if (this.rounds.length === 0) {
      tbody.innerHTML = '';
      if (emptyMsg) emptyMsg.style.display = 'block';
      return;
    }

    if (emptyMsg) emptyMsg.style.display = 'none';

    tbody.innerHTML = this.rounds
      .map(r => `
        <tr data-round="${r.round}">
          <td>${r.round}</td>
          <td>${(r.accuracy * 100).toFixed(2)}%</td>
          <td>${r.loss.toFixed(4)}</td>
          <td>${new Date(r.timestamp).toLocaleTimeString()}</td>
        </tr>
      `)
      .join('');
  }

  getRounds() {
    return [...this.rounds];
  }

  getLatestRound() {
    return this.rounds.length
      ? this.rounds[this.rounds.length - 1]
      : null;
  }

  clear() {
    this.rounds = [];
    this.render();
  }
}

export { TrainingMetricsDashboard };
