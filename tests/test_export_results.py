"""
US-13: Test Export Experiment Results
"""
import requests
import json
import csv
import os

BASE_URL = "http://127.0.0.1:5000"

print("=" * 60)
print("US-13: Testing Export Experiment Results")
print("=" * 60)

# ── Setup: Register clients ────────────────────────────────
print("\n Setup: Registering 3 clients...")
client_ids = []
for i in range(1, 4):
    r = requests.post(f"{BASE_URL}/register",
        json={"client_name": f"export_client_{i}"})
    if r.status_code == 200:
        client_ids.append(r.json()["client_id"])
        print(f"  export_client_{i}:  Registered")

# ── Test 1: Set experiment config ─────────────────────────
print("\n  Test 1: Set experiment configuration")
r = requests.post(f"{BASE_URL}/experiment/config",
    json={
        "min_clients": 2,
        "selection_count": 2,
        "max_rounds": 5,
        "dataset": "kenyan-low-resource",
        "language_pairs": ["dav_swa", "kln_swa", "luo_swa"]
    })
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 2: Log 5 rounds ───────────────────────────────────
print("\n Test 2: Logging 5 experiment rounds...")
import random
for i in range(1, 6):
    accuracy = round(0.5 + (i * 0.08) + random.uniform(-0.02, 0.02), 4)
    loss = round(1.0 - (i * 0.15) + random.uniform(-0.02, 0.02), 4)
    selected = random.sample(client_ids, min(2, len(client_ids)))
    r = requests.post(f"{BASE_URL}/experiment/log",
        json={
            "round": i,
            "accuracy": accuracy,
            "loss": loss,
            "participating_clients": selected
        })
    print(f"  Round {i}: accuracy={accuracy} loss={loss} → Status {r.status_code}")

# ── Test 3: Get summary ────────────────────────────────────
print("\n Test 3: Get experiment summary")
r = requests.get(f"{BASE_URL}/experiment/summary")
print(f"Status Code: {r.status_code}")
print(f"Response: {json.dumps(r.json(), indent=2)}")

# ── Test 4: Export JSON ────────────────────────────────────
print("\n Test 4: Export results as JSON")
r = requests.get(f"{BASE_URL}/experiment/export/json")
print(f"Status Code: {r.status_code}")
if r.status_code == 200:
    json_file = "experiment_results.json"
    with open(json_file, 'wb') as f:
        f.write(r.content)
    print(f"   JSON file saved: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"   JSON opens correctly")
    print(f"   Total rounds in file: {data['total_rounds']}")
    print(f"   Config present: {bool(data['configuration'])}")
    print(f"   Rounds data present: {bool(data['rounds'])}")

# ── Test 5: Export CSV ─────────────────────────────────────
print("\n Test 5: Export results as CSV")
r = requests.get(f"{BASE_URL}/experiment/export/csv")
print(f"Status Code: {r.status_code}")
if r.status_code == 200:
    csv_file = "experiment_results.csv"
    with open(csv_file, 'wb') as f:
        f.write(r.content)
    print(f"   CSV file saved: {csv_file}")
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"   CSV opens correctly")
    print(f"   Total rows: {len(rows)}")
    print(f"   Columns: {list(rows[0].keys())}")

# ── Test 6: Verify no data corruption ─────────────────────
print("\n Test 6: Verify data integrity")
with open("experiment_results.json", 'r') as f:
    json_data = json.load(f)
with open("experiment_results.csv", 'r') as f:
    csv_rows = list(csv.DictReader(f))

json_rounds = len(json_data['rounds'])
csv_rounds = len(csv_rows)
print(f"  JSON rounds: {json_rounds}")
print(f"  CSV rounds:  {csv_rounds}")
print(f"  Match: {' PASS' if json_rounds == csv_rounds else '❌ FAIL'}")

# Verify accuracy values match
json_acc = [r['accuracy'] for r in json_data['rounds']]
csv_acc = [float(r['accuracy']) for r in csv_rows]
match = all(abs(j-c) < 0.0001 for j,c in zip(json_acc, csv_acc))
print(f"  Accuracy values match: {' PASS - No corruption!' if match else '❌ FAIL'}")

# ── Test 7: Verify pandas compatibility ───────────────────
print("\n🐼 Test 7: Verify pandas compatibility")
try:
    import pandas as pd
    df = pd.read_csv("experiment_results.csv")
    print(f"   Pandas reads CSV successfully")
    print(f"   Shape: {df.shape} (rows x columns)")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n  Data Preview:")
    print(df[['round', 'accuracy', 'loss', 'client_count']].to_string())
except ImportError:
    print("  Installing pandas...")
    os.system("pip install pandas")
    import pandas as pd
    df = pd.read_csv("experiment_results.csv")
    print(f"   Pandas reads CSV: {df.shape}")

print("\n" + "=" * 60)
print(" All US-13 Export Tests Complete!")
print("=" * 60)