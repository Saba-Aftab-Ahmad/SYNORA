/**
 * Unit Tests for SBFLT-26: Client-Side Data Privacy Enforcement
 * Plain Node.js assertions — run with: node tests/test_privacy_guard.js
 */

import assert from 'node:assert';
import { PrivacyGuard } from '../client/privacy_guard.js';

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`PASS: ${name}`);
    passed++;
  } catch (e) {
    console.log(`FAIL: ${name} -> ${e.message}`);
    failed++;
  }
}

test('accepts a valid weights-only payload', () => {
  const payload = { weights: [[0.1, 0.2, -0.3]], shapes: [[1, 3]], modelId: 'm1', round: 2 };
  const result = PrivacyGuard.inspectPayload(payload);
  assert.strictEqual(result.safe, true);
});

test('rejects payload containing raw text field', () => {
  const payload = { weights: [[0.1, 0.2]], rawText: 'the quick brown fox' };
  const result = PrivacyGuard.inspectPayload(payload);
  assert.strictEqual(result.safe, false);
});

test('rejects weights array containing a string', () => {
  const payload = { weights: [[0.1, 'hello world', 0.3]] };
  const result = PrivacyGuard.inspectPayload(payload);
  assert.strictEqual(result.safe, false);
});

test('rejects natural-language string hidden in a nested field', () => {
  const payload = { weights: [[0.1, 0.2]], modelId: 'm1', shapes: [[1, 2]], round: 'this is a sentence' };
  const result = PrivacyGuard.inspectPayload(payload);
  assert.strictEqual(result.safe, false);
});

test('rejects non-object payload', () => {
  const result = PrivacyGuard.inspectPayload('not an object');
  assert.strictEqual(result.safe, false);
});

test('rejects payload missing weights field', () => {
  const result = PrivacyGuard.inspectPayload({ modelId: 'm1' });
  assert.strictEqual(result.safe, false);
});

test('accepts short identifier strings without flagging them', () => {
  const payload = { weights: [[0.1]], modelId: 'client-042', clientId: 'abc123' };
  const result = PrivacyGuard.inspectPayload(payload);
  assert.strictEqual(result.safe, true);
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
