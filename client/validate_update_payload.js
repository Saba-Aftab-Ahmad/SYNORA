/**
 * SBFLT-26 supporting piece: Server-side payload validation.
 * Needed for AC3 ("Server returns 400 error on malformed payload test").
 * This is server-side, so share with whoever owns the coordination server
 * (e.g. SBFLT-18 / US-10 FedAvg aggregation task) if it isn't already there.
 */

function validateUpdatePayload(req, res, next) {
  const payload = req.body;

  const allowedKeys = new Set(['weights', 'shapes', 'modelId', 'round', 'clientId']);
  const isNumericTensor = (v) =>
    Array.isArray(v) ? v.every(isNumericTensor) : typeof v === 'number';

  if (!payload || typeof payload !== 'object') {
    return res.status(400).json({ error: 'Malformed payload: not a JSON object' });
  }

  for (const key of Object.keys(payload)) {
    if (!allowedKeys.has(key)) {
      return res.status(400).json({ error: `Malformed payload: unexpected field "${key}"` });
    }
  }

  if (!Array.isArray(payload.weights) || !payload.weights.every(isNumericTensor)) {
    return res.status(400).json({ error: 'Malformed payload: weights must be numeric tensors' });
  }

  next();
}

export { validateUpdatePayload };

/* Example usage in the Express app:
 *
 * import { validateUpdatePayload } from './validate_update_payload.js';
 * app.post('/update', validateUpdatePayload, (req, res) => {
 *   // safe to process req.body.weights here
 * });
 */
