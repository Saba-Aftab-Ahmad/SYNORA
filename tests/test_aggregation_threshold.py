"""
US-11: Test Aggregation Threshold Configuration
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

print("=" * 60)
print("US-11: Testing Aggregation Threshold Configuration")
print("=" * 60)

# ── Test 1: Get default threshold ─────────────────────────
print("\n Test 1: Get default threshold configuration")
r = requests.get(f"{BASE_URL}/config/threshold")
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 2: Set threshold to 3 ────────────────────────────
print("\n  Test 2: Set threshold to 3 clients")
r = requests.post(f"{BASE_URL}/config/threshold",
    json={"min_clients": 3})
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 3: Check aggregation below threshold ──────────────
print("\n Test 3: Check aggregation (0 clients, threshold=3)")
r = requests.get(f"{BASE_URL}/aggregate/check")
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 4: Register 2 clients ────────────────────────────
print("\n Test 4: Register 2 clients")
for name in ["client_amna", "client_2"]:
    r = requests.post(f"{BASE_URL}/register",
        json={"client_name": name})
    print(f"  {name}: Status {r.status_code}")

# ── Test 5: Check aggregation still below threshold ────────
print("\n Test 5: Check aggregation (2 clients, threshold=3)")
r = requests.get(f"{BASE_URL}/aggregate/check")
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 6: Register 3rd client ───────────────────────────
print("\n Test 6: Register 3rd client")
r = requests.post(f"{BASE_URL}/register",
    json={"client_name": "client_3"})
print(f"Status Code: {r.status_code}")

# ── Test 7: Check aggregation at threshold ─────────────────
print("\n Test 7: Check aggregation (3 clients, threshold=3)")
r = requests.get(f"{BASE_URL}/aggregate/check")
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 8: Set threshold below minimum ───────────────────
print("\n Test 8: Set invalid threshold (0 clients)")
r = requests.post(f"{BASE_URL}/config/threshold",
    json={"min_clients": 0})
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

print("\n" + "=" * 60)
print("All US-11 Tests Complete!")
print("=" * 60)