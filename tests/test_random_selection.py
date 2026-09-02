"""
US-12: Test Random Client Selection Strategy
"""
import requests
import json
from collections import Counter

BASE_URL = "http://127.0.0.1:5000"

print("=" * 60)
print("US-12: Testing Random Client Selection")
print("=" * 60)

# ── Setup: Register 5 clients ─────────────────────────────
print("\n Setup: Registering 5 clients...")
client_ids = []
for i in range(1, 6):
    r = requests.post(f"{BASE_URL}/register",
        json={"client_name": f"client_{i}"})
    if r.status_code == 200:
        client_ids.append(r.json()["client_id"])
        print(f"  client_{i}:  Registered")
    else:
        print(f"  client_{i}: already registered")

# ── Test 1: Set selection count to 2 ──────────────────────
print("\n  Test 1: Set selection count to 2")
r = requests.post(f"{BASE_URL}/config/selection",
    json={"selection_count": 2})
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 2: Single round selection ────────────────────────
print("\n Test 2: Single round selection")
r = requests.get(f"{BASE_URL}/select/clients")
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 3: Verify selected count matches config ───────────
print("\n Test 3: Verify selected count = 2")
data = r.json()
selected_count = data.get("selected_count", 0)
print(f"  Expected: 2")
print(f"  Actual:   {selected_count}")
print(f"  Result:   {' PASS' if selected_count == 2 else '❌ FAIL'}")

# ── Test 4: Run 20 rounds ─────────────────────────────────
print("\n Test 4: Running 20 rounds to check distribution...")
for i in range(19):  # Already ran 1 round above
    requests.get(f"{BASE_URL}/select/clients")

# ── Test 5: Check selection history ───────────────────────
print("\n Test 5: Selection history after 20 rounds")
r = requests.get(f"{BASE_URL}/select/history")
print(f"Status Code: {r.status_code}")
data = r.json()
print(f"\nStats:")
print(f"  Rounds completed: {data['stats']['rounds_completed']}")
print(f"  Selection count:  {data['stats']['selection_count']}")

# ── Test 6: Check frequency distribution ──────────────────
print(f"\n Test 6: Frequency Distribution")
dist = data.get("frequency_distribution", {})
for client_id, info in dist.items():
    bar = "█" * int(info["percentage"] / 2)
    print(f"  {client_id[:8]}...: "
          f"{info['selected_count']:2} times "
          f"({info['percentage']:5.1f}%) {bar}")

# ── Test 7: Check bias ────────────────────────────────────
print(f"\n Test 7: Bias Check")
bias = data.get("bias_check", {})
print(f"  Bias detected:    {bias.get('bias_detected', 'N/A')}")
print(f"  Expected %:       {bias.get('expected_percentage', 'N/A')}%")
print(f"  Rounds completed: {bias.get('rounds_completed', 'N/A')}")
if not bias.get("bias_detected"):
    print("  Result:  No consistent bias — selection is approximately uniform!")
else:
    print(f"  Biased clients: {bias.get('biased_clients')}")

# ── Test 8: Invalid selection count ───────────────────────
print("\n Test 8: Invalid selection count (0)")
r = requests.post(f"{BASE_URL}/config/selection",
    json={"selection_count": 0})
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

print("\n" + "=" * 60)
print(" All US-12 Tests Complete!")
print("=" * 60)