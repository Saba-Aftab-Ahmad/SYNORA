/**
 * SBFLT-26 Subtask 1: Client-Side Payload Privacy Guard
 * Ensures only model weight tensors leave the browser — never raw text.
 */

class PrivacyGuard {
  /**
   * Inspect a payload before it is allowed to leave the client.
   * Returns { safe: boolean, reason?: string }
   */
  static inspectPayload(payload) {
    if (payload === null || typeof payload !== 'object') {
      return { safe: false, reason: 'Payload must be a structured object' };
    }

    // 1. Only an allow-listed set of top-level keys is permitted.
    const allowedKeys = new Set(['weights', 'shapes', 'modelId', 'round', 'clientId']);
    for (const key of Object.keys(payload)) {
      if (!allowedKeys.has(key)) {
        return { safe: false, reason: `Disallowed field in payload: ${key}` };
      }
    }

    // 2. "weights" must exist and be numeric arrays / typed arrays only.
    if (!payload.weights || !Array.isArray(payload.weights)) {
      return { safe: false, reason: 'weights field missing or not an array' };
    }

    for (const tensor of payload.weights) {
      if (!this._isNumericTensor(tensor)) {
        return { safe: false, reason: 'Non-numeric data detected in weights (possible raw text leak)' };
      }
    }

    // 3. Scan the whole payload recursively for any string values that
    //    look like natural-language / raw training text (defense in depth,
    //    in case a future field accidentally includes strings).
    const suspiciousString = this._findSuspiciousString(payload);
    if (suspiciousString) {
      return { safe: false, reason: `Raw text-like content detected: "${suspiciousString.slice(0, 20)}..."` };
    }

    return { safe: true };
  }

  static _isNumericTensor(value) {
    if (ArrayBuffer.isView(value)) return true; // Float32Array etc.
    if (Array.isArray(value)) {
      return value.every(v => typeof v === 'number' || this._isNumericTensor(v));
    }
    return typeof value === 'number';
  }

  static _findSuspiciousString(obj, path = '') {
    if (typeof obj === 'string') {
      // Flag strings that contain letters/spaces (natural language-like),
      // but allow short identifiers (modelId, clientId).
      const isIdentifierField = path.endsWith('modelId') || path.endsWith('clientId');
      if (!isIdentifierField && /[a-zA-Z]{2,}\s+[a-zA-Z]{2,}/.test(obj)) {
        return obj;
      }
      return null;
    }
    if (Array.isArray(obj)) {
      for (const item of obj) {
        const hit = this._findSuspiciousString(item, path);
        if (hit) return hit;
      }
      return null;
    }
    if (obj !== null && typeof obj === 'object') {
      for (const [key, val] of Object.entries(obj)) {
        const hit = this._findSuspiciousString(val, `${path}.${key}`);
        if (hit) return hit;
      }
    }
    return null;
  }
}

export { PrivacyGuard };
