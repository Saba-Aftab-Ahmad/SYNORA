/**
 * Unit Tests for SBFLT-12: Model Loading
 */

import * as tf from '@tensorflow/tfjs';
import { TextClassificationModel } from '../client/model_architecture.js';
import { ModelStorage } from '../client/model_storage.js';


// Test 1: Model Creation
function test_model_creation() {
  console.log('\nTest 1: Model creation');
  
  try {
    const model = TextClassificationModel.createLightweightModel(
      5000, 64, 100
    );
    
    if (model && model.layers) {
      console.log('✅ PASS: Model created successfully');
      console.log(
        `    Layers: ${model.layers.length}`
      );
      model.dispose();
    } else {
      console.log('❌ FAIL: Model not created');
    }
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 2: Model Layer Count
function test_model_layer_count() {
  console.log('\nTest 2: Model layer count');
  
  try {
    const model = TextClassificationModel.createLightweightModel();
    const layerCount = TextClassificationModel.getLayerCount(model);
    
    if (layerCount > 0) {
      console.log(
        `✅ PASS: Layer count = ${layerCount}`
      );
    } else {
      console.log('❌ FAIL: Layer count invalid');
    }
    model.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 3: Model Parameter Count
function test_model_parameter_count() {
  console.log('\nTest 3: Model parameter count');
  
  try {
    const model = TextClassificationModel.createLightweightModel();
    const paramCount = TextClassificationModel.getParameterCount(model);
    
    if (paramCount > 0 && paramCount < 1000000) {
      console.log(
        `✅ PASS: Parameters = ${paramCount} (lightweight)`
      );
    } else {
      console.log(
        `❌ FAIL: Parameter count suspicious: ${paramCount}`
      );
    }
    model.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 4: Model Compilation
function test_model_compilation() {
  console.log('\nTest 4: Model compilation');
  
  try {
    let model = TextClassificationModel.createLightweightModel();
    model = TextClassificationModel.compileModel(model);
    
    if (model.optimizer && model.loss) {
      console.log('✅ PASS: Model compiled with optimizer and loss');
    } else {
      console.log('❌ FAIL: Model compilation failed');
    }
    model.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 5: Model Summary
function test_model_summary() {
  console.log('\nTest 5: Model summary');
  
  try {
    const model = TextClassificationModel.createLightweightModel();
    const summary = TextClassificationModel.getModelSummary(model);
    
    if (summary.layers.length > 0 && summary.totalParams > 0) {
      console.log('✅ PASS: Model summary generated');
      console.log(
        `    Layers: ${summary.layers.length}, ` +
        `Params: ${summary.totalParams}`
      );
    } else {
      console.log('❌ FAIL: Summary generation failed');
    }
    model.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 6: Minimal Model
function test_minimal_model() {
  console.log('\nTest 6: Minimal model creation');
  
  try {
    const minimalModel = TextClassificationModel.createMinimalModel();
    const paramCount = TextClassificationModel.getParameterCount(minimalModel);
    
    if (paramCount < 350000) {
      console.log(
        `✅ PASS: Minimal model created (${paramCount} params)`
      );
    } else {
      console.log('❌ FAIL: Model too large');
    }
    minimalModel.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Test 7: Model Prediction Shape
function test_prediction_shape() {
  console.log('\nTest 7: Model prediction shape');
  
  try {
    const model = TextClassificationModel.createLightweightModel();
    TextClassificationModel.compileModel(model);
    
    // Create dummy input
    const input = tf.randomUniform([1, 100], 0, 5000, 'int32');
    const output = model.predict(input);
    
    const outputShape = output.shape;
    if (outputShape[0] === 1 && outputShape[1] === 3) {
      console.log(
        `✅ PASS: Output shape = ${JSON.stringify(outputShape)}`
      );
    } else {
      console.log(
        `❌ FAIL: Output shape ${JSON.stringify(outputShape)} unexpected`
      );
    }
    
    input.dispose();
    output.dispose();
    model.dispose();
  } catch (e) {
    console.log(`❌ FAIL: ${e.message}`);
  }
}

// Run all tests
function runAllTests() {
  console.log('='.repeat(60));
  console.log('Running SBFLT-12 Model Loading Tests');
  console.log('='.repeat(60));

  test_model_creation();
  test_model_layer_count();
  test_model_parameter_count();
  test_model_compilation();
  test_model_summary();
  test_minimal_model();
  test_prediction_shape();

  console.log('\n' + '='.repeat(60));
  console.log('All tests completed');
  console.log('='.repeat(60));
}

export { runAllTests };
// Auto-run when executed directly with: node test_model_loading.js
runAllTests();
