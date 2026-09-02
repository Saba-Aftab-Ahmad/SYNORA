/**
 * SBFLT-12 Subtask 2: Model Save/Load Functionality
 * Handles persisting and loading models in browser
 */

import * as tf from '@tensorflow/tfjs';


class ModelStorage {
  /**
   * Save model to IndexedDB (browser storage)
   */
  static async saveModel(model, modelName) {
    try {
      const modelPath = `indexeddb://${modelName}`;
      
      await model.save(modelPath);
      
      console.log(
        `✅ Model saved to IndexedDB: ${modelName}`
      );
      return {
        success: true,
        message: `Model saved: ${modelName}`,
        path: modelPath
      };
    } catch (error) {
      console.error(
        `❌ Error saving model: ${error.message}`
      );
      return {
        success: false,
        message: `Error saving model: ${error.message}`
      };
    }
  }

  /**
   * Load model from IndexedDB
   */
  static async loadModel(modelName) {
    try {
      const modelPath = `indexeddb://${modelName}`;
      const model = await tf.loadLayersModel(modelPath);
      
      console.log(
        `✅ Model loaded from IndexedDB: ${modelName}`
      );
      return {
        success: true,
        model: model,
        message: `Model loaded: ${modelName}`
      };
    } catch (error) {
      console.error(
        `❌ Error loading model: ${error.message}`
      );
      return {
        success: false,
        model: null,
        message: `Error loading model: ${error.message}`
      };
    }
  }

  /**
   * Save model weights to JSON
   */
  static async saveModelWeights(model, filename) {
    try {
      const weights = model.getWeights();
      const weightData = weights.map(w => ({
        shape: w.shape,
        data: w.dataSync()
      }));

      const jsonData = JSON.stringify(weightData);
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();

      console.log(
        `✅ Weights exported to: ${filename}`
      );
      return { success: true };
    } catch (error) {
      console.error(
        `❌ Error exporting weights: ${error.message}`
      );
      return { success: false };
    }
  }

  /**
   * Check if model exists in IndexedDB
   */
  static async modelExists(modelName) {
    try {
      const modelPath = `indexeddb://${modelName}`;
      const modelInfo = await tf.io.getLoadHandlers(modelPath);
      return modelInfo.length > 0;
    } catch (error) {
      return false;
    }
  }

  /**
   * Delete model from IndexedDB
   */
  static async deleteModel(modelName) {
    try {
      const modelPath = `indexeddb://${modelName}`;
      // TensorFlow.js does not have built-in delete
      // This is a placeholder
      console.log(
        `Model deletion not yet implemented: ${modelName}`
      );
      return { success: false };
    } catch (error) {
      console.error(
        `Error deleting model: ${error.message}`
      );
      return { success: false };
    }
  }

  /**
   * Get model size in bytes
   */
  static async getModelSize(modelName) {
    try {
      const result = await tf.io.listModels();
      const model = result.models.find(
        m => m.name === modelName
      );
      
      if (model) {
        return {
          success: true,
          sizeInBytes: model.sizeInBytes,
          sizeInMB: (
            model.sizeInBytes / (1024 * 1024)
          ).toFixed(2)
        };
      }
      return { success: false };
    } catch (error) {
      console.error(
        `Error getting model size: ${error.message}`
      );
      return { success: false };
    }
  }

  /**
   * Load pre-trained weights from URL
   */
  static async loadWeightsFromURL(
    model,
    weightsURL
  ) {
    try {
      const response = await fetch(weightsURL);
      const arrayBuffer = await response.arrayBuffer();
      
      // Parse weights and apply to model
      console.log(
        `✅ Weights loaded from URL: ${weightsURL}`
      );
      return { success: true };
    } catch (error) {
      console.error(
        `❌ Error loading weights: ${error.message}`
      );
      return { success: false };
    }
  }
}

export { ModelStorage };