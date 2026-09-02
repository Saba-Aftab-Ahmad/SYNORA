/**
 * Unit Tests for SBFLT-13: Browser Training Loop
 */

import * as tf from '@tensorflow/tfjs';
import { BrowserTrainingLoop } from '../client/training_loop.js';
import { LossCalculator, GradientComputer } from '../client/training_loop.js';
import { SGDOptimizer } from '../client/training_loop.js';


// Helper function to create a simple model
function createSimpleModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [10],
        units: 16,
        activation: 'relu'
      }),
      tf.layers.dense({
        units: 8,
        activation: 'relu'
      }),
      tf.layers.dense({
        units: 3,
        activation: 'softmax'
      })
    ]
  });
  return model;
}

// Helper function to create dummy training data
function createTrainingData(numSamples = 100) {
  const xTrain = tf.randomNormal([numSamples, 10]);
  const yTrain = tf.oneHot(
    tf.randomUniform([numSamples], 0, 3, 'int32'),
    3
  );
  return { xTrain, yTrain };
}


// Test 1: Training Loop Initialization
function test_training_loop_initialization() {
  console.log('\nTest 1: Training loop initialization');
  const model = createSimpleModel();
  const loop = new BrowserTrainingLoop(model, 0.01);

  if (loop.currentEpoch === 0) {
    console.log('✅ PASS: Epoch counter initialized to 0');
  } else {
    console.log('❌ FAIL: Epoch counter not initialized');
  }

  model.dispose();
}

// Test 2: Single Epoch Training
async function test_single_epoch_training() {
  console.log('\nTest 2: Single epoch training');
  const model = createSimpleModel();
  const loop = new BrowserTrainingLoop(model, 0.01);
  const { xTrain, yTrain } = createTrainingData(50);

  try {
    const loss = await loop.trainEpoch(xTrain, yTrain, 16);
    if (!isNaN(loss) && loss > 0) {
      console.log(
        `✅ PASS: Epoch trained, loss = ${loss.toFixed(4)}`
      );
    } else {
      console.log('❌ FAIL: Loss value invalid');
    }
  } catch (e) {
    console.log(`❌ FAIL: Training error: ${e.message}`);
  }

  model.dispose();
  xTrain.dispose();
  yTrain.dispose();
}

// Test 3: Multiple Epoch Training
async function test_multiple_epochs() {
  console.log('\nTest 3: Multiple epochs training');
  const model = createSimpleModel();
  const loop = new BrowserTrainingLoop(model, 0.01);
  const { xTrain, yTrain } = createTrainingData(50);

  try {
    const history = await loop.trainMultipleEpochs(
      xTrain,
      yTrain,
      3,
      16
    );
    if (history.epochs.length === 3) {
      console.log(
        '✅ PASS: 3 epochs completed, history recorded'
      );
    } else {
      console.log('❌ FAIL: History not recorded properly');
    }
  } catch (e) {
    console.log(`❌ FAIL: Training error: ${e.message}`);
  }

  model.dispose();
  xTrain.dispose();
  yTrain.dispose();
}

// Test 4: Loss Calculation
function test_loss_calculation() {
  console.log('\nTest 4: Loss calculation');
  const yTrue = tf.tensor2d([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ]);
  const yPred = tf.tensor2d([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
  ]);

  try {
    const loss = LossCalculator.categoricalCrossEntropy(
      yTrue,
      yPred
    );
    const lossValue = loss.dataSync()[0];
    if (!isNaN(lossValue) && lossValue < 1) {
      console.log(
        `✅ PASS: Loss calculated = ${lossValue.toFixed(4)}`
      );
    } else {
      console.log('❌ FAIL: Loss value invalid');
    }
    loss.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }

  yTrue.dispose();
  yPred.dispose();
}

// Test 5: Accuracy Calculation
function test_accuracy_calculation() {
  console.log('\nTest 5: Accuracy calculation');
  const yTrue = tf.tensor2d([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ]);
  const yPred = tf.tensor2d([
    [0.9, 0.05, 0.05],
    [0.05, 0.9, 0.05],
    [0.05, 0.05, 0.9]
  ]);

  try {
    const accuracy = LossCalculator.categoricalAccuracy(
      yTrue,
      yPred
    );
    const accValue = accuracy.dataSync()[0];
    if (accValue === 1.0) {
      console.log(
        `✅ PASS: Accuracy calculated = ${(accValue * 100).toFixed(1)}%`
      );
    } else {
      console.log('❌ FAIL: Accuracy calculation wrong');
    }
    accuracy.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }

  yTrue.dispose();
  yPred.dispose();
}

// Test 6: SGD Optimizer
function test_sgd_optimizer() {
  console.log('\nTest 6: SGD optimizer');
  const optimizer = new SGDOptimizer(0.01, 0.9);

  if (optimizer.getLearningRate() === 0.01) {
    console.log('✅ PASS: Optimizer initialized with correct LR');
  } else {
    console.log('❌ FAIL: Learning rate not set');
  }

  optimizer.setLearningRate(0.001);
  if (optimizer.getLearningRate() === 0.001) {
    console.log(
      '✅ PASS: Learning rate updated successfully'
    );
  } else {
    console.log('❌ FAIL: Learning rate update failed');
  }
}


// Run all tests
async function runAllTests() {
  console.log('='.repeat(50));
  console.log('Running SBFLT-13 Training Loop Tests');
  console.log('='.repeat(50));

  test_training_loop_initialization();
  await test_single_epoch_training();
  await test_multiple_epochs();
  test_loss_calculation();
  test_accuracy_calculation();
  test_sgd_optimizer();

  console.log('\n' + '='.repeat(50));
  console.log('All tests completed');
  console.log('='.repeat(50));
}

export { runAllTests };
// Auto-run when executed directly with: node test_training_loop.js
runAllTests();
