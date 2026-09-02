/**
 * federatedTrainer.js
 * US-06 – GPU-accelerated federated training hook (TensorFlow.js)
 *
 * Wraps a standard TF.js training loop to:
 *   1. Detect & apply the best backend before training starts
 *   2. Run a CPU baseline benchmark once (if GPU is available) to
 *      measure the speed-up ratio (AC: ≥ 2× improvement with GPU)
 *   3. Emit live metrics via an event emitter so the React dashboard
 *      can subscribe without coupling to training internals
 *
 * Usage
 * ─────
 *   import * as tf from '@tensorflow/tfjs';
 *   import '@tensorflow/tfjs-backend-webgpu';
 *   import { FederatedTrainer } from './federatedTrainer';
 *
 *   const trainer = new FederatedTrainer(tf, { onMetrics: updateDashboard });
 *   await trainer.init();
 *   await trainer.train(model, dataset, { epochs: 5 });
 */

import { detectAndApplyBackend } from '../../client/acceleration/backendDetector.js';

// ── Constants ──────────────────────────────────────────────────────────────

/** Minimum acceptable GPU speed-up ratio (AC requirement) */
const MIN_SPEEDUP_RATIO = 2.0;

/** Small synthetic dataset used for the speed benchmark */
const BENCH_SAMPLES = 512;
const BENCH_FEATURES = 64;
const BENCH_EPOCHS = 3;

// ── Benchmark helper ───────────────────────────────────────────────────────

/**
 * Runs a tiny training loop on the given backend and returns ms/epoch.
 *
 * @param {object} tf
 * @param {string} backend - 'webgpu' | 'webnn' | 'cpu'
 * @returns {Promise<number>} average milliseconds per epoch
 */
async function benchmarkBackend(tf, backend) {
  // Switch backend temporarily
  await tf.setBackend(backend);
  await tf.ready();

  const xs = tf.randomNormal([BENCH_SAMPLES, BENCH_FEATURES]);
  const ys = tf.randomUniform([BENCH_SAMPLES, 1]);

  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [BENCH_FEATURES], units: 32, activation: 'relu' }),
      tf.layers.dense({ units: 1 }),
    ],
  });
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const t0 = performance.now();
  await model.fit(xs, ys, { epochs: BENCH_EPOCHS, verbose: 0 });
  const elapsed = performance.now() - t0;

  // Cleanup
  model.dispose();
  xs.dispose();
  ys.dispose();

  return elapsed / BENCH_EPOCHS;
}

// ── FederatedTrainer ───────────────────────────────────────────────────────

export class FederatedTrainer {
  /**
   * @param {object} tf - TensorFlow.js module
   * @param {object} [options]
   * @param {function} [options.onMetrics]  - Called with TrainingMetrics on each epoch end
   * @param {function} [options.onStatus]   - Called with a status string during init
   */
  constructor(tf, options = {}) {
    this._tf = tf;
    this._onMetrics = options.onMetrics ?? (() => { });
    this._onStatus = options.onStatus ?? (() => { });

    /** @type {import('../../client/acceleration/backendDetector.js').BackendDetectionResult | null} */
    this.backendInfo = null;

    /** @type {{ gpuMsPerEpoch: number, cpuMsPerEpoch: number, ratio: number } | null} */
    this.benchmark = null;

    this._initialized = false;
  }

  // ── Public ──────────────────────────────────────────────────────────────

  /**
   * Detects backend and (if GPU) runs the speed benchmark.
   * Must be called before train().
   *
   * @returns {Promise<void>}
   */
  async init() {
    this._onStatus('Detecting hardware backend…');
    this.backendInfo = await detectAndApplyBackend(this._tf);
    this._onStatus(`Backend: ${this.backendInfo.label}`);

    if (this.backendInfo.isGPU) {
      this._onStatus('Running speed benchmark…');
      await this._runBenchmark();

      const ratio = this.benchmark?.ratio ?? 0;
      if (ratio < MIN_SPEEDUP_RATIO) {
        console.warn(
          `[FederatedTrainer] GPU speed-up ratio (${ratio.toFixed(2)}×) ` +
          `is below the required ${MIN_SPEEDUP_RATIO}×. ` +
          `This may indicate the model is too small to saturate the GPU.`
        );
      }

      // Restore GPU backend after benchmark (benchmark switches to cpu temporarily)
      await this._tf.setBackend(this.backendInfo.backend);
      await this._tf.ready();
    }

    this._initialized = true;
    this._onStatus('Ready');
  }

  /**
   * Trains the given compiled model on the dataset.
   *
   * @param {object} model   - A compiled tf.LayersModel
   * @param {object} dataset - Object with xs and ys tensors (or tf.data.Dataset)
   * @param {object} [opts]
   * @param {number} [opts.epochs=10]
   * @param {number} [opts.batchSize=32]
   * @returns {Promise<tf.History>}
   */
  async train(model, dataset, opts = {}) {
    if (!this._initialized) {
      throw new Error('[FederatedTrainer] Call init() before train().');
    }

    const epochs = opts.epochs ?? 10;
    const batchSize = opts.batchSize ?? 32;

    return model.fit(dataset.xs, dataset.ys, {
      epochs,
      batchSize,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          /** @type {TrainingMetrics} */
          const metrics = {
            epoch: epoch + 1,
            totalEpochs: epochs,
            loss: logs.loss,
            acc: logs.acc ?? logs.accuracy ?? null,
            backend: this.backendInfo.backend,
            backendLabel: this.backendInfo.label,
            isGPU: this.backendInfo.isGPU,
            speedupRatio: this.benchmark?.ratio ?? null,
            timestamp: Date.now(),
          };
          this._onMetrics(metrics);
        },
      },
    });
  }

  // ── Private ─────────────────────────────────────────────────────────────

  async _runBenchmark() {
    const tf = this._tf;
    const gpuBackend = this.backendInfo.backend; // 'webgpu' or 'webnn'

    const gpuMsPerEpoch = await benchmarkBackend(tf, gpuBackend);
    const cpuMsPerEpoch = await benchmarkBackend(tf, 'cpu');
    const ratio = cpuMsPerEpoch / gpuMsPerEpoch;

    this.benchmark = { gpuMsPerEpoch, cpuMsPerEpoch, ratio };

    this._onStatus(
      `Benchmark complete — GPU ${ratio.toFixed(1)}× faster than CPU ` +
      `(${gpuMsPerEpoch.toFixed(0)} ms/epoch vs ${cpuMsPerEpoch.toFixed(0)} ms/epoch)`
    );
  }
}

/**
 * @typedef {Object} TrainingMetrics
 * @property {number}      epoch
 * @property {number}      totalEpochs
 * @property {number}      loss
 * @property {number|null} acc
 * @property {string}      backend
 * @property {string}      backendLabel
 * @property {boolean}     isGPU
 * @property {number|null} speedupRatio
 * @property {number}      timestamp
 */