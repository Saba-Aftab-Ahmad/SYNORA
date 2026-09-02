/**
 * SBFLT-12 Subtask 1: Lightweight Model Architecture
 * Defines TensorFlow.js model for text classification
 */

import * as tf from '@tensorflow/tfjs';


class TextClassificationModel {
  /**
   * Create a lightweight model for text classification
   * Optimized for low-resource Kenyan languages
   */
  static createLightweightModel(
    vocabSize = 5000,
    embeddingDim = 64,
    maxLength = 100
  ) {
    const model = tf.sequential({
      layers: [
        // Input layer (implicit)
        
        // Embedding layer
        tf.layers.embedding({
          inputDim: vocabSize,
          outputDim: embeddingDim,
          inputLength: maxLength,
          name: 'embedding'
        }),

        // Bidirectional LSTM layer
        tf.layers.bidirectional({
          layer: tf.layers.lstm({
            units: 32,
            returnSequences: false,
            activation: 'tanh'
          }),
          name: 'bidirectional_lstm'
        }),

        // Dropout for regularization
        tf.layers.dropout({
          rate: 0.3,
          name: 'dropout'
        }),

        // Dense layer
        tf.layers.dense({
          units: 16,
          activation: 'relu',
          name: 'dense_1'
        }),

        // Output layer (3 classes for 3 languages)
        tf.layers.dense({
          units: 3,
          activation: 'softmax',
          name: 'output'
        })
      ]
    });

    return model;
  }

  /**
   * Create a minimal model for rapid testing
   */
  static createMinimalModel(
    vocabSize = 5000,
    embeddingDim = 32,
    maxLength = 100
  ) {
    const model = tf.sequential({
      layers: [
        tf.layers.embedding({
          inputDim: vocabSize,
          outputDim: embeddingDim,
          inputLength: maxLength
        }),

        // Global average pooling
        tf.layers.globalAveragePooling1d({}),

        // Single dense layer
        tf.layers.dense({
          units: 8,
          activation: 'relu'
        }),

        // Output layer
        tf.layers.dense({
          units: 3,
          activation: 'softmax'
        })
      ]
    });

    return model;
  }

  /**
   * Compile the model with optimizer and loss
   */
  static compileModel(model) {
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  }

  /**
   * Get model summary
   */
  static getModelSummary(model) {
    const summary = {
      name: model.name,
      layers: [],
      totalParams: 0,
      trainableParams: 0
    };

    model.layers.forEach((layer, index) => {
      const config = layer.getConfig();
      const weights = layer.getWeights();
      let params = 0;

      weights.forEach(w => {
        params += w.size;
      });

      summary.layers.push({
        index: index,
        name: config.name,
        className: layer.constructor.name,
        outputShape: layer.outputShape,
        params: params
      });

      summary.totalParams += params;
      summary.trainableParams += params;
    });

    return summary;
  }

  /**
   * Print model architecture to console
   */
  static printModelSummary(model) {
    console.log('\n' + '='.repeat(60));
    console.log('MODEL ARCHITECTURE SUMMARY');
    console.log('='.repeat(60));

    const summary = this.getModelSummary(model);
    console.log(`Model: ${summary.name}`);
    console.log('\nLayers:');

    summary.layers.forEach(layer => {
      console.log(
        `  ${layer.index}. ${layer.name} (${layer.className})`
      );
      console.log(`     Output shape: ${JSON.stringify(layer.outputShape)}`);
      console.log(`     Parameters: ${layer.params}`);
    });

    console.log(
      `\nTotal parameters: ${summary.totalParams}`
    );
    console.log(
      `Trainable parameters: ${summary.trainableParams}`
    );
    console.log('='.repeat(60) + '\n');
  }

  /**
   * Get layer count
   */
  static getLayerCount(model) {
    return model.layers.length;
  }

  /**
   * Get total parameter count
   */
  static getParameterCount(model) {
    let totalParams = 0;
    model.layers.forEach(layer => {
      const weights = layer.getWeights();
      weights.forEach(w => {
        totalParams += w.size;
      });
    });
    return totalParams;
  }
}

export { TextClassificationModel };