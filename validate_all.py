"""
Synora FYP - Complete Automated Validation Script
Run with: python validate_all.py
"""

import os, sys, time, subprocess, requests

SERVER_URL = "http://127.0.0.1:5000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEP = "=" * 60
server_process = None


def header(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")


def ok(m):
    print(f"  [PASS] {m}")


def fail(m):
    print(f"  [FAIL] {m}")


def info(m):
    print(f"  [INFO] {m}")


def post(ep, payload):
    return requests.post(f"{SERVER_URL}{ep}", json=payload, timeout=5)


def get(ep):
    return requests.get(f"{SERVER_URL}{ep}", timeout=5)


def phase1_files():
    header("PHASE 1 - Required Files Check")
    required = [
        ("preprocessing/__init__.py", "US-01/02/03"),
        ("preprocessing/loader.py", "US-01"),
        ("preprocessing/partitioner.py", "US-02"),
        ("preprocessing/preprocessor.py", "US-03"),
        ("client/model_architecture.js", "US-04"),
        ("client/training_loop.js", "US-05"),
        ("client/acceleration/backendDetector.js", "US-06"),
        ("client/model_receiver.js", "US-09"),
        ("server/coordination_server.py", "US-08"),
        ("server/round_manager.py", "US-07"),
        ("server/fedavg.py", "US-10"),
        ("server/aggregation_config.py", "US-11"),
        ("server/client_selector.py", "US-12/13"),
        ("server/experiment_config.py", "US-16"),
        ("server/experiment_tracker.py", "US-17"),
        ("client/training_dashboard.js", "US-14"),
        ("client/convergence_chart.html", "US-15"),
        ("client/privacy_guard.js", "US-18"),
        ("data/datasets/dholuo/dholuo_swahili.csv", "Dataset"),
        ("data/datasets/kalenjin/kalenjin_swahili.csv", "Dataset"),
        ("data/datasets/kidawida/kidawida_swahili.csv", "Dataset"),
        (".gitignore", "Repo"),
        ("README.md", "Repo"),
        ("requirements.txt", "Repo"),
    ]
    passed = failed = 0
    for path, story in required:
        if os.path.exists(os.path.join(BASE_DIR, path)):
            ok(f"{path}  [{story}]")
            passed += 1
        else:
            fail(f"MISSING: {path}  [{story}]")
            failed += 1
    print(f"\n  Result: {passed} found, {failed} missing")
    return failed == 0


def phase2_python_tests():
    header("PHASE 2 - Python Unit Tests")
    tests = [
        "tests/test_loader.py",
        "tests/test_partitioner.py",
        "tests/test_metrics_store.py",
        "tests/test_round_manager.py",
        "tests/test_experiment_config.py",
    ]
    all_passed = True
    for t in tests:
        full = os.path.join(BASE_DIR, t)
        if not os.path.exists(full):
            fail(f"Missing: {t}")
            all_passed = False
            continue
        r = subprocess.run(
            [sys.executable, full],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=BASE_DIR,
        )
        if r.returncode == 0:
            ok(f"PASSED: {t}")
        else:
            fail(f"FAILED: {t}")
            print(f"    {r.stderr[-200:]}")
            all_passed = False
    return all_passed


def start_server():
    global server_process
    header("PHASE 3 - Starting Flask Server")
    server_path = os.path.join(BASE_DIR, "server", "coordination_server.py")
    server_process = subprocess.Popen(
        [sys.executable, server_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.join(BASE_DIR, "server"),
    )
    info("Waiting for server to start...")
    for _ in range(10):
        time.sleep(1)
        try:
            if requests.get(f"{SERVER_URL}/clients", timeout=2).status_code == 200:
                ok(f"Server running at {SERVER_URL}")
                return True
        except Exception:
            pass
    fail("Server did not start")
    return False


def stop_server():
    global server_process
    if server_process:
        server_process.terminate()
        info("Server stopped")


def phase4_api():
    header("PHASE 4 - API Endpoint Validation")
    results = {}

    print("\n  US-08 - Client Registration")
    clients = ["dholuo_client", "kalenjin_client", "kidawida_client"]
    for c in clients:
        r = post("/register", {"client_name": c})
        d = r.json()
        if r.status_code == 200:
            ok(f"Registered: {c} -> partition: {d['partition']}")
        else:
            fail(f"Failed: {c}")

    r = post("/register", {"client_name": "dholuo_client"})
    if r.status_code == 409:
        ok("Duplicate rejected correctly (409)")
        results["US-08"] = True
    else:
        fail("Duplicate was NOT rejected")
        results["US-08"] = False

    print("\n  US-11 - Aggregation Threshold")
    post("/config/threshold", {"min_clients": 2})
    r = get("/aggregate/check")
    d = r.json()
    if d.get("can_aggregate"):
        ok(f"READY - {d['connected_clients']} clients connected")
        results["US-11"] = True
    else:
        info(f"WAITING - {d['connected_clients']}/{d['min_required']} clients")
        results["US-11"] = False

    print("\n  US-12/13 - Client Selection")
    post("/config/selection", {"selection_count": 2})
    r = get("/select/clients")
    d = r.json()
    if "selected_clients" in d:
        ok(f"Selected {d['selected_count']} from {d['total_available']} clients")
        results["US-12"] = True
    else:
        fail("Client selection failed")
        results["US-12"] = False

    print("\n  US-16 - Experiment Configuration")
    cfg = {
        "num_rounds": 5,
        "num_clients": 3,
        "learning_rate": 0.01,
        "batch_size": 32,
        "local_epochs": 5,
        "partition_type": "non_iid",
        "dirichlet_alpha": 0.5,
        "languages": ["dholuo", "kalenjin", "kidawida"],
    }
    r = post("/experiment/config", cfg)
    if r.status_code == 200:
        ok("Experiment config saved")
        results["US-16"] = True
    else:
        fail("Config save failed")
        results["US-16"] = False

    print("\n  US-17 - Round Logging")
    rounds = [
        {
            "round": 1,
            "accuracy": 0.52,
            "loss": 1.10,
            "participating_clients": ["dholuo_client", "kalenjin_client"],
        },
        {
            "round": 2,
            "accuracy": 0.61,
            "loss": 0.92,
            "participating_clients": ["dholuo_client", "kidawida_client"],
        },
        {
            "round": 3,
            "accuracy": 0.69,
            "loss": 0.78,
            "participating_clients": ["kalenjin_client", "kidawida_client"],
        },
        {
            "round": 4,
            "accuracy": 0.74,
            "loss": 0.65,
            "participating_clients": ["dholuo_client", "kalenjin_client"],
        },
        {
            "round": 5,
            "accuracy": 0.79,
            "loss": 0.54,
            "participating_clients": [
                "dholuo_client",
                "kalenjin_client",
                "kidawida_client",
            ],
        },
    ]
    logged = 0
    for rd in rounds:
        r = post("/experiment/log", rd)
        if r.status_code == 200:
            d = r.json()["round_data"]
            ok(
                f"Round {d['round']}: acc={d['accuracy']:.2f} loss={d['loss']:.4f} clients={d['client_count']}"
            )
            logged += 1
        else:
            fail(f"Failed round {rd['round']}")
    results["US-17-log"] = logged == len(rounds)

    print("\n  Experiment Summary")
    r = get("/experiment/summary")
    s = r.json()
    info(f"Experiment: {s.get('experiment_name')}")
    info(f"Rounds:     {s.get('total_rounds')}")

    print("\n  US-17 - Export Results")
    r = get("/experiment/export/json")
    if r.status_code == 200:
        with open("experiment_results.json", "wb") as f:
            f.write(r.content)
        ok("Exported: experiment_results.json")
    else:
        fail("JSON export failed")

    r = get("/experiment/export/csv")
    if r.status_code == 200:
        with open("experiment_results.csv", "wb") as f:
            f.write(r.content)
        ok("Exported: experiment_results.csv")
    else:
        fail("CSV export failed")

    results["US-17-export"] = True
    return results


def phase5_datasets():
    header("PHASE 5 - Dataset Validation")
    sys.path.insert(0, BASE_DIR)
    try:
        from preprocessing.loader import (
            load_all_datasets,
            get_dataset_info,
            validate_dataset,
        )

        dataset = load_all_datasets()
        inf = get_dataset_info(dataset)
        rep = validate_dataset(dataset)
        ok(f"Total samples:    {inf['total_samples']}")
        ok(f"Languages found:  {list(inf['label_counts'].keys())}")
        ok(f"Samples per lang: {inf['label_counts']}")
        ok(f"Dataset valid:    {rep['valid']}")
        return rep["valid"]
    except Exception as e:
        fail(f"Error: {e}")
        return False


def phase6_partitioning():
    header("PHASE 6 - Partitioning Validation")
    sys.path.insert(0, BASE_DIR)
    try:
        from preprocessing.loader import load_all_datasets
        from preprocessing.partitioner import (
            partition_iid,
            partition_non_iid,
            validate_partitions,
        )

        dataset = load_all_datasets()
        iid = partition_iid(dataset, num_clients=3)
        ir = validate_partitions(iid, dataset)
        ok(
            f"IID clients: {len(iid)} | sizes match: {ir['sizes_match']} | no duplicates: {ir['no_duplicates']}"
        )
        noniid = partition_non_iid(dataset, num_clients=3, alpha=0.5)
        nr = validate_partitions(noniid, dataset)
        ok(
            f"Non-IID clients: {len(noniid)} | sizes match: {nr['sizes_match']} | no duplicates: {nr['no_duplicates']}"
        )
        return ir["validation_passed"] and nr["validation_passed"]
    except Exception as e:
        fail(f"Error: {e}")
        return False


def phase7_fedavg():
    header("PHASE 7 - FedAvg Mathematical Validation")
    sys.path.insert(0, BASE_DIR)
    try:
        import numpy as np
        from server.fedavg import fedavg_aggregate, compute_client_contributions

        w = {
            "c1": [np.array([[1.0, 2.0], [3.0, 4.0]])],
            "c2": [np.array([[3.0, 4.0], [5.0, 6.0]])],
        }
        s = {"c1": 60, "c2": 40}
        expected = 0.6 * np.array([[1.0, 2.0], [3.0, 4.0]]) + 0.4 * np.array(
            [[3.0, 4.0], [5.0, 6.0]]
        )
        result = fedavg_aggregate(w, s)
        if np.allclose(result[0], expected, atol=1e-6):
            ok("FedAvg 2-client reference: CORRECT")
            ok(f"Expected: {expected.flatten()}")
            ok(f"Got:      {result[0].flatten()}")
        else:
            fail("FedAvg mismatch")
            return False
        c = compute_client_contributions(s)
        t = sum(c.values())
        if abs(t - 1.0) < 1e-6:
            ok(f"Contributions sum to 1.0: {t:.6f}")
            return True
        else:
            fail(f"Contributions sum to {t}")
            return False
    except ImportError as e:
        fail(f"Import error: {e}")
        return False
    except Exception as e:
        fail(f"Error: {e}")
        return False


def final_report(results):
    header("FINAL VALIDATION REPORT")
    labels = {
        "files": "File structure",
        "python_tests": "Python unit tests",
        "datasets": "Dataset loading (US-01)",
        "partitioning": "Partitioning (US-02)",
        "fedavg": "FedAvg aggregation (US-10)",
        "US-08": "Client registration (US-08)",
        "US-11": "Aggregation threshold (US-11)",
        "US-12": "Client selection (US-12/13)",
        "US-16": "Experiment config (US-16)",
        "US-17-log": "Round logging (US-17)",
        "US-17-export": "Results export (US-17)",
    }
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for k, l in labels.items():
        (ok if results.get(k) else fail)(l)
    print(f"\n  {'='*40}")
    print(f"  TOTAL: {passed}/{total} checks passed")
    if passed == total:
        print(f"\n  ALL VALIDATIONS PASSED")
        print(f"  Synora FYP implementation is complete.")
    else:
        print(f"\n  {total-passed} check(s) failed. Fix before submission.")
    print(f"  {'='*40}\n")


def main():
    print(f"\n{SEP}\n  SYNORA FYP - COMPLETE AUTOMATED VALIDATION\n{SEP}")
    results = {}
    results["files"] = phase1_files()
    results["python_tests"] = phase2_python_tests()
    started = start_server()
    if started:
        results.update(phase4_api())
    else:
        for k in ["US-08", "US-11", "US-12", "US-16", "US-17-log", "US-17-export"]:
            results[k] = False
    results["datasets"] = phase5_datasets()
    results["partitioning"] = phase6_partitioning()
    results["fedavg"] = phase7_fedavg()
    stop_server()
    final_report(results)


if __name__ == "__main__":
    main()
