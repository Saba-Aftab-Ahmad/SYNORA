/**
 * SBFLT-13 Subtask 1: TensorFlow.js Training Loop
 * Implements local training with epoch management
 */

import * as tf from '@tensorflow/tfjs';


class BrowserTrainingLoop {
  /**
   * Initialize training loop for a given model
   */
  constructor(model, learningRate = 0.01) {
    this.model = model;
    this.learningRate = learningRate;
    this.trainingHistory = {
      epochs: [],
      losses: [],
      accuracies: []
    };
    this.currentEpoch = 0;
  }

  /**
   * Run a single epoch of training
   */
  async trainEpoch(xTrain, yTrain, batchSize = 32) {
    const numSamples = xTrain.shape[0];
    const numBatches = Math.ceil(numSamples / batchSize);

    let epochLoss = 0;
    let epochAccuracy = 0;

    for (let batch = 0; batch < numBatches; batch++) {
      const start = batch * batchSize;
      const end = Math.min(start + batchSize, numSamples);

      const batchX = xTrain.slice([start, 0], [end - start, -1]);
      const batchY = yTrain.slice([start, 0], [end - start, -1]);

      // Forward pass + backward pass using model.optimizer
      // tf.losses.categoricalCrossentropy does not exist in TFjs v4 —
      // use model.trainOnBatch() which handles both loss + gradient update
      // internally via the compiled optimizer (set in model_architecture.js).
      const batchLossTensor = await this.model.trainOnBatch(batchX, batchY);

      // trainOnBatch returns a scalar number or array [loss, ...metrics]
      const lossValue = Array.isArray(batchLossTensor)
        ? [batchLossTensor[0]]
        : [batchLossTensor];

      epochLoss += lossValue[0];

      batchX.dispose();
      batchY.dispose();

      console.log(
        `Batch ${batch + 1}/${numBatches} - Loss: ${lossValue[0].toFixed(4)}`
      );
    }

    this.currentEpoch += 1;
    const avgLoss = epochLoss / numBatches;
    this.trainingHistory.epochs.push(this.currentEpoch);
    this.trainingHistory.losses.push(avgLoss);

    console.log(
      `Epoch ${this.currentEpoch} completed - Avg Loss: ${avgLoss.toFixed(4)}`
    );

    return avgLoss;
  }

  /**
   * Run multiple epochs of training
   */
  async trainMultipleEpochs(xTrain, yTrain, numEpochs = 5, batchSize = 32) {
    console.log(
      `Starting training for ${numEpochs} epochs...`
    );

    for (let epoch = 0; epoch < numEpochs; epoch++) {
      console.log(
        `\n--- Epoch ${epoch + 1}/${numEpochs} ---`
      );
      await this.trainEpoch(xTrain, yTrain, batchSize);
    }

    console.log('\nTraining completed!');
    return this.trainingHistory;
  }

  /**
   * Get training history
   */
  getHistory() {
    return this.trainingHistory;
  }

  /**
   * Reset training state
   */
  reset() {
    this.trainingHistory = {
      epochs: [],
      losses: [],
      accuracies: []
    };
    this.currentEpoch = 0;
  }
}

export { BrowserTrainingLoop };

/**
 * Loss Calculation Module
 * Computes categorical cross-entropy loss
 */

class LossCalculator {
  /**
   * Compute categorical cross-entropy loss
   */
  static categoricalCrossEntropy(yTrue, yPred) {
    return tf.tidy(() => {
      const epsilon = 1e-7;
      const clipped = tf.clipByValue(yPred, epsilon, 1 - epsilon);
      const loss = tf.neg(
        tf.sum(tf.mul(yTrue, tf.log(clipped)), -1)
      );
      return loss.mean();
    });
  }

  /**
   * Compute accuracy for classification
   */
  static categoricalAccuracy(yTrue, yPred) {
    return tf.tidy(() => {
      const trueLabels = tf.argMax(yTrue, -1);
      const predLabels = tf.argMax(yPred, -1);
      const correct = tf.equal(trueLabels, predLabels);
      const accuracy = tf.mean(
        tf.cast(correct, 'float32')
      );
      return accuracy;
    });
  }
}

/**
 * Gradient Computation Module
 */

class GradientComputer {
  /**
   * Compute gradients using TensorFlow.js
   */
  static computeGradients(model, xBatch, yBatch) {
    return tf.tidy(() => {
      const f = () => {
        const predictions = model.predict(xBatch);
        const loss = LossCalculator.categoricalCrossEntropy(
          yBatch,
          predictions
        );
        return loss;
      };

      const { value: loss, grads } = tf.variableGrads(f);

      return {
        loss: loss,
        gradients: grads
      };
    });
  }

  /**
   * Apply gradients to model weights
   */
  static applyGradients(model, gradients, learningRate) {
    const weights = model.getWeights();
    const newWeights = [];

    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      const g = gradients[model.getWeights()[i].name];

      if (g) {
        const updated = tf.tidy(() => {
          return tf.sub(
            w,
            tf.mul(g, learningRate)
          );
        });
        newWeights.push(updated);
      } else {
        newWeights.push(w);
      }
    }

    model.setWeights(newWeights);
  }
}

export { LossCalculator, GradientComputer };

/**
 * Stochastic Gradient Descent Optimizer
 */

class SGDOptimizer {
  /**
   * Initialize SGD optimizer
   */
  constructor(learningRate = 0.01, momentum = 0.0) {
    this.learningRate = learningRate;
    this.momentum = momentum;
    this.velocities = {};
  }

  /**
   * Update model weights using SGD
   */
  updateWeights(model, gradients) {
    const weights = model.getWeights();
    const newWeights = [];

    weights.forEach((w, index) => {
      tf.tidy(() => {
        const weightName = `weight_${index}`;
        const gradient = gradients[index];

        if (!this.velocities[weightName]) {
          this.velocities[weightName] = tf.zerosLike(w);
        }

        let velocity = this.velocities[weightName];

        // Apply momentum
        if (this.momentum > 0) {
          velocity = tf.tidy(() => {
            return tf.add(
              tf.mul(velocity, this.momentum),
              tf.mul(gradient, this.learningRate)
            );
          });
        } else {
          velocity = tf.mul(gradient, this.learningRate);
        }

        // Update weights
        const updated = tf.sub(w, velocity);
        newWeights.push(updated);

        this.velocities[weightName] = velocity;
      });
    });

    model.setWeights(newWeights);
  }

  /**
   * Set learning rate
   */
  setLearningRate(rate) {
    this.learningRate = rate;
  }

  /**
   * Get current learning rate
   */
  getLearningRate() {
    return this.learningRate;
  }

  /**
   * Reset optimizer state
   */
  reset() {
    this.velocities = {};
  }
}

export { SGDOptimizer };