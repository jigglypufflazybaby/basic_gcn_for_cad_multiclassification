import os
import json
import torch
from torch_geometric.data import InMemoryDataset, Data

class CADGraphJSONDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CADGraphJSONDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # List all JSON files in the raw directory
        raw_dir = os.path.join(self.root, 'raw')
        return [f for f in os.listdir(raw_dir) if f.endswith('.json')]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # Download logic if needed (not implemented)
        pass
    
    def process(self):
        data_list = []
        raw_dir = os.path.join(self.root, 'raw')
        for filename in self.raw_file_names:
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, 'r') as f:
                graph_dict = json.load(f)
            # Convert JSON lists into tensors.
            # Assume graph_dict has keys: 'x', 'edge_index', 'y'
            x = torch.tensor(graph_dict['x'], dtype=torch.float)
            # 'edge_index' is expected as a list of two lists; convert to tensor of shape [2, num_edges]
            edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
            y = torch.tensor(graph_dict['y'], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
