import { ClientSelector } from '../client/client_selection.js';

let passed = 0, failed = 0;
function assert(cond, msg) {
  if (cond) { console.log(`✅ PASS: ${msg}`); passed++; }
  else { console.log(`❌ FAIL: ${msg}`); failed++; }
}

const s = new ClientSelector();

// Missing capabilityScore should now throw
try {
  s.registerClient({ clientId: 'bad1', timestamp: Date.now() });
  assert(false, 'Rejects report with missing capabilityScore');
} catch (e) {
  assert(true, 'Rejects report with missing capabilityScore');
}

// Out-of-range score should throw
try {
  s.registerClient({ clientId: 'bad2', timestamp: Date.now(), capabilityScore: 1.5 });
  assert(false, 'Rejects capabilityScore out of [0,1] range');
} catch (e) {
  assert(true, 'Rejects capabilityScore out of [0,1] range');
}

// Missing clientId should throw
try {
  s.registerClient({ timestamp: Date.now(), capabilityScore: 0.5 });
  assert(false, 'Rejects report with missing clientId');
} catch (e) {
  assert(true, 'Rejects report with missing clientId');
}

// Valid report should succeed
try {
  s.registerClient({ clientId: 'good', timestamp: Date.now(), capabilityScore: 0.5 });
  assert(s.getRegisteredClientCount() === 1, 'Accepts valid report and registers exactly 1 client');
} catch (e) {
  assert(false, 'Accepts valid report and registers exactly 1 client');
}

// pruneStaleClients removes old reports
s.registerClient({ clientId: 'old', timestamp: Date.now() - 100000, capabilityScore: 0.9 });
s.pruneStaleClients(5000, Date.now());
assert(!s.registeredClients.has('old') && s.registeredClients.has('good'), 'pruneStaleClients removes stale, keeps fresh');

console.log(`\nResults: ${passed} passed, ${failed} failed`);
