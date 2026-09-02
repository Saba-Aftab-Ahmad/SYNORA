import requests
import json

BASE_URL = "http://127.0.0.1:5000"

print("=" * 50)
print("Testing Client Registration")
print("=" * 50)

# Test 1 - Successful Registration
print("\n Test 1: Register new client")
response = requests.post(f"{BASE_URL}/register", 
    json={"client_name": "client_amna"})
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2 - Duplicate Registration
print("\n❌ Test 2: Duplicate registration")
response = requests.post(f"{BASE_URL}/register", 
    json={"client_name": "client_amna"})
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 3 - Register second client
print("\n Test 3: Register second client")
response = requests.post(f"{BASE_URL}/register", 
    json={"client_name": "client_2"})
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 4 - View all clients
print("\n Test 4: View all registered clients")
response = requests.get(f"{BASE_URL}/clients")
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")