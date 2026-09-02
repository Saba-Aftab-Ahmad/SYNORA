/**
 * US-09 Subtask 2: Load received weights into local TFJS model
 * Ensures no shape mismatch, ready-to-train without extra config
 */

import * as tf from '@tensorflow/tfjs';

class ModelWeightLoader {
  /**
   * Validate that incoming weights match the local model's expected shapes
   */
  static validateShapes(localModel, incomingWeights) {
    const localWeights = localModel.getWeights();

    if (localWeights.length !== incomingWeights.length) {
      return {
        valid: false,
        reason: `Weight count mismatch: local=${localWeights.length}, incoming=${incomingWeights.length}`
      };
    }

    for (let i = 0; i < localWeights.length; i++) {
      const localShape = localWeights[i].shape.join(',');
      const incomingShape = incomingWeights[i].shape.join(',');

      if (localShape !== incomingShape) {
        return {
          valid: false,
          reason: `Shape mismatch at layer ${i}: local=${localShape}, incoming=${incomingShape}`
        };
      }
    }

    return { valid: true };
  }

  /**
   * Apply global weights onto local model safely
   */
  static async applyGlobalWeights(localModel, globalModel) {
    try {
      const incomingWeights = globalModel.getWeights();

      const validation = this.validateShapes(localModel, incomingWeights);
      if (!validation.valid) {
        throw new Error(validation.reason);
      }

      localModel.setWeights(incomingWeights);

      console.log('✅ Global weights applied to local model');
      return { success: true };
    } catch (error) {
      console.error(
        `❌ Error applying global weights: ${error.message}`
      );
      return { success: false, error: error.message };
    }
  }

  /**
   * Confirm model is ready for training (compiled + weights set)
   */
  static isReadyForTraining(model) {
    const compiled = !!model.optimizer;
    const hasWeights = model.getWeights().length > 0;
    return compiled && hasWeights;
  }
}

export { ModelWeightLoader };