/**
 * US-09 Subtask 1: Fetch global model from server
 * Retrieves latest aggregated model weights before local training
 */

// Base URL for the Synora coordination server.
// Change to https://your-server.com in production.
const SERVER_BASE_URL = 'http://localhost:5000';

class ModelReceiver {
  /**
   * Fetch the current global model weights from the server.
   * Calls GET /model/global — returns JSON with version + weights array.
   */
  static async fetchGlobalModel() {
    try {
      const startTime = Date.now();

      const response = await fetch(`${SERVER_BASE_URL}/model/global`);

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const modelJSON = await response.json();

      const elapsed = (Date.now() - startTime) / 1000;
      console.log(`✅ Global model fetched in ${elapsed.toFixed(2)}s (version ${modelJSON.version})`);

      return {
        success: true,
        modelJSON,
        elapsedSeconds: elapsed
      };
    } catch (error) {
      console.error(`❌ Error fetching global model: ${error.message}`);
      return { success: false, error: error.message };
    }
  }

  /**
   * Check server for a new model version before each training round.
   * Calls GET /model/version — returns { version: number }.
   */
  static async checkForNewVersion(currentVersion) {
    try {
      const response = await fetch(`${SERVER_BASE_URL}/model/version`);
      const data = await response.json();

      return {
        hasNewVersion: data.version !== currentVersion,
        latestVersion: data.version
      };
    } catch (error) {
      console.error(`❌ Error checking model version: ${error.message}`);
      return { hasNewVersion: false, latestVersion: currentVersion };
    }
  }

  /**
   * Submit local weight update to the server after training.
   * Calls POST /model/update — enforced by PrivacyGuard before sending.
   *
   * @param {string} clientId - Registered client UUID
   * @param {number} round    - Current round number
   * @param {Array}  weights  - Serialised weight layers
   */
  static async submitWeightUpdate(clientId, round, weights) {
    try {
      const response = await fetch(`${SERVER_BASE_URL}/model/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: clientId, round, weights })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || `Server error ${response.status}`);
      }

      console.log(`✅ Weight update accepted for round ${round}`);
      return { success: true, data };
    } catch (error) {
      console.error(`❌ Error submitting weight update: ${error.message}`);
      return { success: false, error: error.message };
    }
  }
}

export { ModelReceiver };