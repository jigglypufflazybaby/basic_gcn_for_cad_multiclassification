import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class CADGraphDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        with open(file_path, 'r') as f:
            graph_data = json.load(f)

        print(f"Loaded JSON keys: {graph_data.keys()}")

        nodes = graph_data.get('nodes', [])
        num_nodes = len(nodes)

        if isinstance(nodes, list) and len(nodes) > 0 and isinstance(nodes[0], dict):
            feature_keys = [key for key in nodes[0] if key != 'id']

            def convert_feature(value):
                if isinstance(value, (int, float)):
                    return value
                elif isinstance(value, str):
                    return hash(value) % 1000
                else:
                    return 0

            node_features = torch.tensor([[convert_feature(node[k]) for k in feature_keys] for node in nodes], dtype=torch.float)
        else:
            print(f"Warning: Unexpected or empty node format in {file_path}.")
            node_features = torch.zeros((num_nodes, 1), dtype=torch.float)

        edge_list = graph_data.get('edges', [])
        valid_edges = [e for e in edge_list if e['from'] < num_nodes and e['to'] < num_nodes]

        # === edge_index ===
        if valid_edges:
            edge_index = torch.tensor([[e['from'], e['to']] for e in valid_edges], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # === edge_attr ===
        relation_map = {"connectivity": 0, "dependency": 1, "symmetric": 2}
        edge_attr = [relation_map.get(e.get("relation", "connectivity"), 0) for e in valid_edges]
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).unsqueeze(1)  # [num_edges, 1]

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

