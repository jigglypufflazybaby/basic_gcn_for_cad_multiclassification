import os
import json
import random

# Directory to store generated JSON files
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

def generate_random_cad_graph():
    """Generate a random CAD graph representation."""
    return {
        "nodes": [
            {"id": i, "type": random.choice(["extrude", "cut", "chamfer", "fillet"]), "params": {"size": random.uniform(1, 10)}}
            for i in range(random.randint(5, 15))
        ],
        "edges": [
            {"from": random.randint(0, 4), "to": random.randint(5, 14), "relation": "dependency"}
            for _ in range(random.randint(3, 10))
        ]
    }

# Number of files to generate
num_files = 10

for i in range(num_files):
    file_path = os.path.join(output_dir, f"cad_graph_{i+1}.json")
    with open(file_path, "w") as f:
        json.dump(generate_random_cad_graph(), f, indent=4)
    print(f"Generated: {file_path}")
