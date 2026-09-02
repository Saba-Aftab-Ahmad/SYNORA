/**
 * SBFLT-21 Subtask: Client Capability Scoring
 * Measures browser/device hardware capability and produces a normalized score.
 *
 * Signals used (all available in-browser without special permissions):
 *  - navigator.hardwareConcurrency  -> CPU logical cores
 *  - navigator.deviceMemory         -> approx RAM in GB (Chrome-based browsers only)
 *  - connection.downlink / effectiveType -> network speed estimate
 *  - a short synthetic compute benchmark (measures actual JS execution speed,
 *    which correlates with real training throughput better than raw core count)
 */

class CapabilityScorer {
  /**
   * Run a short CPU benchmark (blocking, ~50-150ms) and return operations/ms.
   * This is a more honest signal than hardwareConcurrency alone, since two
   * devices with the same core count can have very different single-core speed.
   */
  static runComputeBenchmark(iterations = 2_000_000) {
    const start = performance.now();
    let acc = 0;
    for (let i = 0; i < iterations; i++) {
      acc += Math.sqrt(i) * Math.sin(i);
    }
    const elapsedMs = performance.now() - start;
    // Guard against elapsedMs being 0 on very fast machines
    const opsPerMs = iterations / Math.max(elapsedMs, 1);
    return { opsPerMs, elapsedMs, _antiDCE: acc };
  }

  /**
   * Collect raw hardware signals available to the browser.
   * Falls back gracefully when an API is unsupported (e.g. Firefox/Safari
   * don't expose deviceMemory).
   */
  static collectRawSignals() {
    const nav = typeof navigator !== 'undefined' ? navigator : {};
    const conn = nav.connection || nav.mozConnection || nav.webkitConnection || {};

    return {
      cores: nav.hardwareConcurrency || 2,          // default conservative guess
      memoryGB: nav.deviceMemory || 4,               // default conservative guess
      downlinkMbps: conn.downlink || 5,               // default conservative guess
      effectiveType: conn.effectiveType || '4g',
      compute: this.runComputeBenchmark()
    };
  }

  /**
   * Normalize a raw value into [0, 1] given expected min/max bounds.
   */
  static normalize(value, min, max) {
    if (max === min) return 1;
    const clamped = Math.min(Math.max(value, min), max);
    return (clamped - min) / (max - min);
  }

  /**
   * Combine raw signals into a single capability score in [0, 1].
   * Weights are configurable; defaults favor compute speed since that's
   * the most direct predictor of training round duration.
   */
  static computeScore(rawSignals, weights = null) {
    const w = weights || {
      cores: 0.2,
      memory: 0.2,
      network: 0.15,
      compute: 0.45
    };

    const coresNorm = this.normalize(rawSignals.cores, 1, 16);
    const memoryNorm = this.normalize(rawSignals.memoryGB, 0.5, 16);
    const networkNorm = this.normalize(rawSignals.downlinkMbps, 0.5, 50);
    // opsPerMs typically ranges ~500 - 20000 depending on device; adjust
    // bounds after collecting real data from your test devices.
    const computeNorm = this.normalize(rawSignals.compute.opsPerMs, 500, 20000);

    const score =
      w.cores * coresNorm +
      w.memory * memoryNorm +
      w.network * networkNorm +
      w.compute * computeNorm;

    return {
      score: Number(score.toFixed(4)),
      breakdown: {
        coresNorm: Number(coresNorm.toFixed(4)),
        memoryNorm: Number(memoryNorm.toFixed(4)),
        networkNorm: Number(networkNorm.toFixed(4)),
        computeNorm: Number(computeNorm.toFixed(4))
      },
      weights: w
    };
  }

  /**
   * Convenience: collect signals + score in one call. This is what a client
   * calls once on registration (and optionally re-runs periodically, since
   * network conditions can change mid-session).
   */
  static getCapabilityReport(clientId) {
    const raw = this.collectRawSignals();
    const scored = this.computeScore(raw);

    return {
      clientId,
      timestamp: Date.now(),
      rawSignals: {
        cores: raw.cores,
        memoryGB: raw.memoryGB,
        downlinkMbps: raw.downlinkMbps,
        effectiveType: raw.effectiveType,
        computeOpsPerMs: Number(raw.compute.opsPerMs.toFixed(2))
      },
      capabilityScore: scored.score,
      scoreBreakdown: scored.breakdown
    };
  }
}

export { CapabilityScorer };
