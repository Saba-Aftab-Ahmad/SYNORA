/**
 * Unit Tests for SBFLT-26: Secure Transmission
 * Covers AC 3, 4, 5 (which the first test file did not).
 * Run with: node tests/test_secure_transmission.js
 */

import assert from 'node:assert';
import { SecureTransmission } from '../client/secure_transmission.js';

let passed = 0;
let failed = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`PASS: ${name}`);
    passed++;
  } catch (e) {
    console.log(`FAIL: ${name} -> ${e.message}`);
    failed++;
  }
}

function mockFetch(status) {
  global.fetch = async () => ({ status, ok: status >= 200 && status < 300 });
}

const validPayload = { weights: [[0.1, 0.2]], modelId: 'm1', round: 1 };

await test('AC5: blocks transmission to a non-HTTPS endpoint', async () => {
  let fetchCalled = false;
  global.fetch = async () => { fetchCalled = true; return { status: 200, ok: true }; };
  const result = await SecureTransmission.sendUpdate('http://insecure-server.com/update', validPayload);
  assert.strictEqual(result.success, false);
  assert.strictEqual(fetchCalled, false, 'fetch should never be called for non-HTTPS endpoint');
});

await test('AC5: allows transmission to an HTTPS endpoint', async () => {
  mockFetch(200);
  const result = await SecureTransmission.sendUpdate('https://server.com/update', validPayload);
  assert.strictEqual(result.success, true);
});

await test('AC4: automatic inspection blocks a raw-text payload before fetch is called', async () => {
  let fetchCalled = false;
  global.fetch = async () => { fetchCalled = true; return { status: 200, ok: true }; };
  const badPayload = { weights: [[0.1]], note: 'this is raw participant text' };
  const result = await SecureTransmission.sendUpdate('https://server.com/update', badPayload);
  assert.strictEqual(result.success, false);
  assert.strictEqual(fetchCalled, false, 'fetch should never be called when PrivacyGuard rejects payload');
});

await test('AC3: surfaces server 400 response for a malformed payload as a failure', async () => {
  mockFetch(400);
  const result = await SecureTransmission.sendUpdate('https://server.com/update', validPayload);
  assert.strictEqual(result.success, false);
  assert.strictEqual(result.status, 400);
});

await test('handles a network-level failure gracefully', async () => {
  global.fetch = async () => { throw new Error('network down'); };
  const result = await SecureTransmission.sendUpdate('https://server.com/update', validPayload);
  assert.strictEqual(result.success, false);
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
