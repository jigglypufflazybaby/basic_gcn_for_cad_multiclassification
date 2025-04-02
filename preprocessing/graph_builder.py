import networkx as nx
import pickle

def load_cad_data(file_path):
    """Load CAD data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def cad_to_graph(cad_data):
    """Convert CAD construction history into a graph."""
    G = nx.DiGraph()
    for step in cad_data['history']:
        G.add_node(step['id'], operation=step['operation'], params=step['params'])
        if 'parent_ids' in step:
            for parent in step['parent_ids']:
                G.add_edge(parent, step['id'])
    return G

def save_graph(graph, output_path):
    """Save graph to a file."""
    nx.write_gpickle(graph, output_path)

# Example usage
# cad_data = load_cad_data('data/raw/sample.pkl')
# graph = cad_to_graph(cad_data)
# save_graph(graph, 'data/processed/sample_graph.pkl')
