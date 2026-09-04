/**
 * Synora FL Model — Browser Side
 * TensorFlow.js model for Kenyan language text classification
 * US-04, US-05, US-09
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ||
    "https://synora-coordination-server.onrender.com";

/**
 * Simple text classification model for browser training
 * Lightweight architecture for fast convergence
 */
export async function createModel(vocabSize = 5000, numClasses = 3) {
    // Dynamic import — TF.js sirf browser mein load ho
    const tf = await import("@tensorflow/tfjs");

    const model = tf.sequential({
        layers: [
            tf.layers.embedding({
                inputDim: vocabSize,
                outputDim: 32,
                inputLength: 100,
                name: "embedding"
            }),
            tf.layers.globalAveragePooling1d({ name: "pooling" }),
            tf.layers.dense({
                units: 16,
                activation: "relu",
                name: "dense_1"
            }),
            tf.layers.dropout({ rate: 0.3, name: "dropout" }),
            tf.layers.dense({
                units: numClasses,
                activation: "softmax",
                name: "output"
            })
        ]
    });

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    return model;
}

/**
 * US-09: Server se global model weights load karo browser mein
 */
export async function loadGlobalModel() {
    const tf = await import("@tensorflow/tfjs");

    const response = await fetch(`${BACKEND_URL}/global-model`);
    const serverData = await response.json();

    const model = await createModel();

    // Agar server pe weights hain to apply karo
    if (serverData.weights && serverData.weights.length > 0) {
        try {
            const tensors = serverData.weights.map(layer => {
                if (Array.isArray(layer)) {
                    // 2D weight array (e.g. Dense layer)
                    return tf.tensor(layer);
                } else {
                    // 1D bias array
                    return tf.tensor1d(layer);
                }
            });
            model.setWeights(tensors);
            tensors.forEach(t => t.dispose());
            console.log(
                `Global model loaded — version ${serverData.version}`
            );
        } catch (err) {
            console.warn("Could not load server weights, using random init:", err);
        }
    } else {
        console.log("No server weights yet — using random initialization");
    }

    return { model, version: serverData.version || 0 };
}

/**
 * US-05: Browser mein local training run karo
 */
export async function trainLocally(model, xData, yData, epochs = 3) {
    const tf = await import("@tensorflow/tfjs");

    // Dummy data for demo — real implementation mein
    // yeh actual CSV partition se aayega
    const numSamples = 50;
    const seqLength = 100;
    const numClasses = 3;

    const xTrain = xData ||
        tf.randomUniform([numSamples, seqLength], 0, 5000, "int32");
    const yTrain = yData ||
        tf.oneHot(
            tf.floor(tf.randomUniform([numSamples], 0, numClasses)),
            numClasses
        );

    const history = await model.fit(xTrain, yTrain, {
        epochs,
        batchSize: 16,
        validationSplit: 0.1,
        verbose: 0
    });

    // Cleanup
    if (!xData) xTrain.dispose();
    if (!yData) yTrain.dispose();

    const finalEpoch = history.history;
    const accuracy = finalEpoch.acc?.[epochs - 1] ||
        finalEpoch.accuracy?.[epochs - 1] || 0;
    const loss = finalEpoch.loss?.[epochs - 1] || 0;

    return { accuracy, loss, history: finalEpoch };
}

/**
 * Model weights extract karo server ko bhejna ke liye
 * Weights ko plain arrays mein convert karo (JSON serializable)
 */
export async function extractWeights(model) {
    const weights = model.getWeights();
    const extracted = await Promise.all(
        weights.map(async (tensor) => {
            const data = await tensor.data();
            const shape = tensor.shape;
            if (shape.length === 2) {
                // 2D — convert to nested array
                const result = [];
                for (let i = 0; i < shape[0]; i++) {
                    result.push(Array.from(data.slice(i * shape[1], (i + 1) * shape[1])));
                }
                return result;
            }
            // 1D
            return Array.from(data);
        })
    );
    return extracted;
}

/**
 * US-10: Local weights server ko bhejo FedAvg ke liye
 */
export async function submitWeightsToServer(
    model,
    clientId,
    roundNum,
    datasetSize,
    metrics
) {
    const weights = await extractWeights(model);

    const response = await fetch(`${BACKEND_URL}/submit-update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            client_id: clientId,
            weights: weights,
            dataset_size: datasetSize,
            metrics: metrics,
            round: roundNum
        })
    });

    if (!response.ok) {
        throw new Error(`Weight submission failed: ${response.status}`);
    }

    return response.json();
}