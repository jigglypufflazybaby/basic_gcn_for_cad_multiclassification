{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a1e5c7e2",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis\n",
    "\n",
    "This notebook is for exploring the CAD graph dataset and visualizing model predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df8a3f99",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from dataset import CADGraphDataset\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load a small sample from the dataset\n",
    "dataset = CADGraphDataset(root='data/cad_graph')\n",
    "print(f'Total graphs: {len(dataset)}')\n",
    "\n",
    "# Visualize node features of the first graph\n",
    "data = dataset[0]\n",
    "print(f'Graph has {data.num_nodes} nodes and {data.num_edges} edges')\n",
    "\n",
    "# Plot the first two feature dimensions of all nodes\n",
    "features = data.x.numpy()\n",
    "plt.scatter(features[:, 0], features[:, 1], c='blue', alpha=0.6)\n",
    "plt.title('Node Feature Scatter Plot')\n",
    "plt.xlabel('Feature 1')\n",
    "plt.ylabel('Feature 2')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
