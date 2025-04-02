# Configuration file for hyperparameters and settings

config = {
    'in_channels': 128,        # Dimensionality of input node features
    'hidden_channels': 256,    # Hidden layer dimension for the GNN
    'out_channels': 10,        # Number of construction groups (multi-label outputs)
    'num_layers': 3,           # Number of GNN layers
    'dropout': 0.5,            # Dropout rate for the classifier
    'learning_rate': 0.001,    # Learning rate for Adam optimizer
    'epochs': 100,             # Total number of training epochs
    'batch_size': 16,          # Batch size for training
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Device setting
}
