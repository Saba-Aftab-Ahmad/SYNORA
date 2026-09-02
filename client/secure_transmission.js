/**
 * SBFLT-26 Subtask 2: Secure Transmission
 * Enforces HTTPS and runs the PrivacyGuard check automatically
 * before every transmission to the coordination server.
 */

import { PrivacyGuard } from './privacy_guard.js';

class SecureTransmission {
  /**
   * Send a client update payload to the server.
   * Automatically runs the privacy inspection first, and refuses
   * to send over a non-HTTPS endpoint.
   */
  static async sendUpdate(endpointUrl, payload) {
    // 1. Enforce HTTPS
    if (!this._isHttps(endpointUrl)) {
      return {
        success: false,
        status: null,
        message: `Blocked: endpoint is not HTTPS (${endpointUrl})`
      };
    }

    // 2. Automatic payload inspection (runs before every transmission)
    const inspection = PrivacyGuard.inspectPayload(payload);
    if (!inspection.safe) {
      return {
        success: false,
        status: null,
        message: `Blocked by PrivacyGuard: ${inspection.reason}`
      };
    }

    // 3. Send
    try {
      const response = await fetch(endpointUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.status === 400) {
        return {
          success: false,
          status: 400,
          message: 'Server rejected payload as malformed'
        };
      }

      return {
        success: response.ok,
        status: response.status,
        message: response.ok ? 'Update sent successfully' : `Server error: ${response.status}`
      };
    } catch (error) {
      return { success: false, status: null, message: `Network error: ${error.message}` };
    }
  }

  static _isHttps(url) {
    try {
      return new URL(url).protocol === 'https:';
    } catch {
      return false;
    }
  }
}

export { SecureTransmission };
