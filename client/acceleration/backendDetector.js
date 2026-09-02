/**
 * backendDetector.js
 * US-06 – Detect and apply WebGPU / WebNN acceleration
 *
 * Automatically selects the best available backend for TensorFlow.js
 * at runtime and reports the result for dashboard display.
 *
 * Priority order: webgpu → webnn → cpu
 */

// ── Types ─────────────────────────────────────────────────────────────────

/**
 * @typedef {'webgpu' | 'webnn' | 'cpu'} BackendName
 *
 * @typedef {Object} BackendDetectionResult
 * @property {BackendName}  backend        - Active backend name
 * @property {boolean}      isGPU          - True when a hardware-accelerated backend is active
 * @property {boolean}      webgpuAvailable
 * @property {boolean}      webnnAvailable
 * @property {string}       label          - Human-readable label for dashboard display
 * @property {string}       reason         - Why this backend was chosen
 * @property {number}       detectedAt     - Unix timestamp (ms)
 */

// ── Internal helpers ───────────────────────────────────────────────────────

/**
 * Probes WebGPU availability without importing TF.js.
 * Returns true only when navigator.gpu can actually create an adapter
 * (the API exists but may still be unavailable on some systems).
 *
 * @returns {Promise<boolean>}
 */
async function probeWebGPU() {
  try {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/**
 * Probes WebNN availability.
 * The API is still experimental; we guard every access carefully.
 *
 * @returns {Promise<boolean>}
 */
async function probeWebNN() {
  try {
    if (typeof navigator === 'undefined' || !('ml' in navigator)) return false;
    // navigator.ml.createContext() throws or returns null when unsupported
    const ctx = await navigator.ml.createContext({ deviceType: 'gpu' });
    return ctx !== null;
  } catch {
    return false;
  }
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Detects available hardware backends and sets the optimal one on the
 * provided TensorFlow.js instance.
 *
 * @param {object} tf - A TensorFlow.js module (import * as tf from '@tensorflow/tfjs')
 * @returns {Promise<BackendDetectionResult>}
 *
 * @example
 * import * as tf from '@tensorflow/tfjs';
 * import '@tensorflow/tfjs-backend-webgpu';
 * import { detectAndApplyBackend } from './backendDetector';
 *
 * const result = await detectAndApplyBackend(tf);
 * console.log(result.label); // "WebGPU (GPU)"
 */
export async function detectAndApplyBackend(tf) {
  const webgpuAvailable = await probeWebGPU();
  const webnnAvailable  = await probeWebNN();

  /** @type {BackendDetectionResult} */
  let result;

  if (webgpuAvailable) {
    await tf.setBackend('webgpu');
    await tf.ready();
    result = {
      backend: 'webgpu',
      isGPU: true,
      webgpuAvailable,
      webnnAvailable,
      label: 'WebGPU (GPU)',
      reason: 'WebGPU adapter found; hardware-accelerated backend activated.',
      detectedAt: Date.now(),
    };
  } else if (webnnAvailable) {
    await tf.setBackend('webnn');
    await tf.ready();
    result = {
      backend: 'webnn',
      isGPU: true,
      webgpuAvailable,
      webnnAvailable,
      label: 'WebNN (GPU)',
      reason: 'WebNN context available; using neural-network hardware acceleration.',
      detectedAt: Date.now(),
    };
  } else {
    // Graceful fallback – AC: "Graceful fallback to CPU confirmed when GPU unavailable"
    await tf.setBackend('cpu');
    await tf.ready();
    result = {
      backend: 'cpu',
      isGPU: false,
      webgpuAvailable,
      webnnAvailable,
      label: 'CPU (fallback)',
      reason: 'No WebGPU or WebNN support detected; falling back to CPU backend.',
      detectedAt: Date.now(),
    };
  }

  // Verify the backend TF.js actually set matches what we requested
  const active = tf.getBackend();
  if (active !== result.backend) {
    console.warn(
      `[BackendDetector] Requested "${result.backend}" but TF.js reports "${active}". ` +
      `Updating result to reflect active backend.`
    );
    result.backend = active;
    result.isGPU   = active !== 'cpu';
    result.label   = active.toUpperCase();
    result.reason  += ` (TF.js overrode selection to "${active}")`;
  }

  return result;
}

/**
 * Returns a snapshot of what backends are available in this browser,
 * without changing the active backend.
 *
 * Useful for dashboard info panels and telemetry.
 *
 * @returns {Promise<{ webgpu: boolean, webnn: boolean }>}
 */
export async function getAvailableBackends() {
  const [webgpu, webnn] = await Promise.all([probeWebGPU(), probeWebNN()]);
  return { webgpu, webnn };
}