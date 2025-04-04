import os
import json
import random
import numpy as np
from pathlib import Path
from copy import deepcopy

# Set paths
input_dir = Path("data/raw")
output_dir = Path("data/cad_graph")
output_dir.mkdir(parents=True, exist_ok=True)

# Number of augmented versions per original file
NUM_AUGMENTATIONS = 5

def augment_graph(graph):
    augmented = deepcopy(graph)

    # 1. Slight noise to node features
    for node in augmented['nodes']:
        for key, val in node.items():
            if isinstance(val, (int, float)):
                node[key] = val + np.random.normal(0, 0.1)  # small Gaussian noise

    # 2. Randomly swap some edges
    if len(augmented['edges']) >= 2:
        random.shuffle(augmented['edges'])
        num_swaps = random.randint(1, len(augmented['edges']) // 2)
        for _ in range(num_swaps):
            i = random.randint(0, len(augmented['edges']) - 1)
            edge = augmented['edges'][i]

            # Handle dict format with 'from' and 'to'
            if isinstance(edge, dict) and 'from' in edge and 'to' in edge:
                src, tgt = edge['from'], edge['to']
            elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                src, tgt = edge
            else:
                raise ValueError(f"Unexpected edge format: {edge}")

            new_src = (src + random.randint(1, 3)) % len(augmented['nodes'])
            new_tgt = (tgt + random.randint(1, 3)) % len(augmented['nodes'])
            augmented['edges'][i] = [new_src, new_tgt]

    return augmented

# Loop through original files and create augmented data
file_list = list(input_dir.glob("*.json"))

for file in file_list:
    with open(file, 'r') as f:
        original_graph = json.load(f)

    base_name = file.stem
    for i in range(NUM_AUGMENTATIONS):
        aug_graph = augment_graph(original_graph)
        aug_path = output_dir / f"{base_name}_aug_{i}.json"
        with open(aug_path, 'w') as out_f:
            json.dump(aug_graph, out_f, indent=2)

print(f"✅ Augmentation complete. Generated {len(file_list) * NUM_AUGMENTATIONS} new files in {output_dir}")
