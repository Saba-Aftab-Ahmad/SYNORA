/**
 * SBFLT-21: Resource-Aware Client Selection
 * Selects top-N clients per FL round based on reported capability scores,
 * instead of random sampling. Also logs selection decisions for later
 * analysis and benchmark comparison (AC: "Selection decisions logged with
 * capability scores for analysis", "Round completion time compared against
 * random selection baseline").
 */

class ClientSelector {
  constructor() {
    /**
     * clientId -> latest capability report
     * { clientId, timestamp, rawSignals, capabilityScore, scoreBreakdown }
     */
    this.registeredClients = new Map();

    /**
     * Array of selection log entries, one per round, for analysis /
     * export (AC: "Selection decisions logged with capability scores").
     */
    this.selectionLog = [];
  }

  /**
   * Register or update a client's capability report.
   * Called whenever CapabilityScorer.getCapabilityReport() produces a fresh reading.
   *
   * Validates the report before accepting it (AC1 fix): a client cannot be
   * registered without a real, numeric capabilityScore in [0, 1]. This closes
   * the gap where malformed/incomplete reports were silently accepted.
   */
  registerClient(capabilityReport) {
    if (!capabilityReport || typeof capabilityReport.clientId !== 'string' || !capabilityReport.clientId) {
      throw new Error('registerClient: capabilityReport must include a non-empty clientId');
    }
    if (
      typeof capabilityReport.capabilityScore !== 'number' ||
      Number.isNaN(capabilityReport.capabilityScore) ||
      capabilityReport.capabilityScore < 0 ||
      capabilityReport.capabilityScore > 1
    ) {
      throw new Error(
        `registerClient: capabilityScore for '${capabilityReport.clientId}' must be a number in [0, 1], got ${capabilityReport.capabilityScore}`
      );
    }
    if (typeof capabilityReport.timestamp !== 'number') {
      throw new Error(`registerClient: capabilityReport for '${capabilityReport.clientId}' must include a numeric timestamp`);
    }

    this.registeredClients.set(capabilityReport.clientId, capabilityReport);
  }

  /**
   * Remove clients whose report is older than maxAgeMs (staleness handling).
   * Real deployments should call this before each round, since a client's
   * hardware/network conditions can change between reports (AC3 relies on
   * this — see selection_realistic_benchmark.js).
   */
  pruneStaleClients(maxAgeMs, nowMs = Date.now()) {
    for (const [clientId, report] of this.registeredClients.entries()) {
      if (nowMs - report.timestamp > maxAgeMs) {
        this.registeredClients.delete(clientId);
      }
    }
  }

  removeClient(clientId) {
    this.registeredClients.delete(clientId);
  }

  getRegisteredClientCount() {
    return this.registeredClients.size;
  }

  /**
   * Capability-ranked selection: pick the top-N clients by capabilityScore.
   * AC: "Selection prioritises higher-capability clients"
   * AC: "Capability-ranked clients selected in top-N for 90% of rounds"
   *     -> this method always ranks by capability; the 90% figure is
   *        something you verify empirically across many simulated rounds
   *        (see benchmark script), allowing for ties / clients dropping out.
   */
  selectTopN(n, roundNumber = null) {
    const all = Array.from(this.registeredClients.values());

    if (all.length === 0) {
      throw new Error('No registered clients available for selection');
    }

    const ranked = [...all].sort(
      (a, b) => b.capabilityScore - a.capabilityScore
    );

    const selected = ranked.slice(0, Math.min(n, ranked.length));

    this._logSelection({
      roundNumber,
      strategy: 'capability-ranked',
      requestedN: n,
      selectedClientIds: selected.map(c => c.clientId),
      selectedScores: selected.map(c => c.capabilityScore),
      allScoresSnapshot: ranked.map(c => ({
        clientId: c.clientId,
        capabilityScore: c.capabilityScore
      }))
    });

    return selected;
  }

  /**
   * Random baseline selection, used only for benchmarking comparison
   * against the capability-ranked strategy (AC: "Round completion time
   * compared against random selection baseline"). Not used in production
   * rounds — this exists so you can run a controlled A/B comparison.
   */
  selectRandomN(n, roundNumber = null) {
    const all = Array.from(this.registeredClients.values());

    if (all.length === 0) {
      throw new Error('No registered clients available for selection');
    }

    const shuffled = [...all].sort(() => Math.random() - 0.5);
    const selected = shuffled.slice(0, Math.min(n, shuffled.length));

    this._logSelection({
      roundNumber,
      strategy: 'random',
      requestedN: n,
      selectedClientIds: selected.map(c => c.clientId),
      selectedScores: selected.map(c => c.capabilityScore),
      allScoresSnapshot: all.map(c => ({
        clientId: c.clientId,
        capabilityScore: c.capabilityScore
      }))
    });

    return selected;
  }

  /**
   * Internal: append a structured log entry for a selection decision.
   */
  _logSelection(entry) {
    this.selectionLog.push({
      ...entry,
      timestamp: Date.now()
    });
  }

  /**
   * Return the full selection log (for export to CSV/JSON for analysis).
   */
  getSelectionLog() {
    return this.selectionLog;
  }

  /**
   * Export selection log as JSON string, downloadable in-browser.
   */
  exportSelectionLogAsJSON() {
    return JSON.stringify(this.selectionLog, null, 2);
  }

  /**
   * Analysis helper: what fraction of capability-ranked rounds actually
   * selected the true top-N clients available at that moment (accounts for
   * clients that may have disconnected between report and selection).
   * Useful for verifying AC: "top-N for 90% of rounds".
   */
  computeTopNAccuracy() {
    const rankedRounds = this.selectionLog.filter(
      e => e.strategy === 'capability-ranked'
    );

    if (rankedRounds.length === 0) return null;

    let correctRounds = 0;
    for (const round of rankedRounds) {
      const sortedSnapshot = [...round.allScoresSnapshot].sort(
        (a, b) => b.capabilityScore - a.capabilityScore
      );
      const expectedTopIds = sortedSnapshot
        .slice(0, round.requestedN)
        .map(c => c.clientId)
        .sort();
      const actualIds = [...round.selectedClientIds].sort();

      const matches =
        expectedTopIds.length === actualIds.length &&
        expectedTopIds.every((id, i) => id === actualIds[i]);

      if (matches) correctRounds++;
    }

    return {
      totalRounds: rankedRounds.length,
      correctRounds,
      accuracy: Number((correctRounds / rankedRounds.length).toFixed(4))
    };
  }
}

export { ClientSelector };
