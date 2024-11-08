import torch
from torch.nn import Linear
import torch.nn.functional as F
from data_utils import load_data
import argparse
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
 
# Argument parsing for input parameters, dataset name, number of GNN layers, and optional topology configurations
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="name of the dataset", type=str)
parser.add_argument("--k", help="number of GNN layers", type=int, default=2)
parser.add_argument("--use_neighbors", help="use neighbors in GNN", action='store_true')
parser.add_argument("--topology_weight", help="weight for neighbor influence", type=float, default=1.0)
args = parser.parse_args()
 
# Load dataset
graph_dataset = load_data(args.dataset)
print(f'Dataset: {args.dataset}')
print(f'Number of GNN layers: {args.k}')
print('======================')
print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset.num_features}')
print(f'Number of classes: {graph_dataset.num_classes}')
 
graph_data = graph_dataset[0]  # Get the first graph object
print(f'Graph info: {graph_data}')
print('===========================================================================================================')
 
# Stats about the graph
print(f'Number of nodes: {graph_data.num_nodes}')
print(f'Number of edges: {graph_data.num_edges}')
print(f'Number of training nodes: {graph_data.train_mask.sum()}')
print(f'Number of validation nodes: {graph_data.val_mask.sum()}')
print(f'Number of test nodes: {graph_data.test_mask.sum()}')
 
# Define GNN model
class CustomGNN(torch.nn.Module):
    # Initializing the model with specified dimensions, number of layers, and neighbor settings
    def __init__(self, input_dim, output_dim, hidden_dim, layer_count, include_neighbors=True, neighbor_weight=1.0):
        super(CustomGNN, self).__init__()
        self.layer_count = layer_count
        self.include_neighbors = include_neighbors
        self.neighbor_weight = neighbor_weight
        self.dropout = torch.nn.Dropout(p=0.4)

        # Define the list of GNN layers in the model
        self.layers = torch.nn.ModuleList()
        self.layers.append(Linear(input_dim, hidden_dim))
       
        # Initializing additional hidden layers as per the specified number of layers
        for layer_index in range(layer_count - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))

        # Output layer that maps hidden features to the output classes    
        self.output_layer = Linear(hidden_dim, output_dim)

    # Forward propagation for the GNN model
    def forward(self, features, edge_indices):
        for layer in self.layers:
            if self.include_neighbors:
                # Neighbor aggregation: Summing neighbor features to the central node
                row, col = edge_indices
                neighborhood_sum = torch.zeros_like(features)
                neighborhood_sum.index_add_(0, col, features[row])
                features = layer(features + self.neighbor_weight * neighborhood_sum)  # Apply topology weight here
            else:
                features = layer(features)

            # Activation and dropout for each hidden layer
            features = F.relu(features)
            features = self.dropout(features)
        features = self.output_layer(features)
        return F.log_softmax(features, dim=1)
 
# Instantiating model with updated parameters
gnn_model = CustomGNN(graph_dataset.num_features, 
                      graph_dataset.num_classes, 
                      hidden_dim=32, 
                      layer_count=args.k, 
                      include_neighbors=args.use_neighbors, 
                      neighbor_weight=args.topology_weight)

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.0001, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
 
# Lists to store losses and F1 scores for plotting
training_losses, validation_losses = [], []
training_f1_scores, validation_f1_scores = [], []
 
# Variables to track the best scores
best_val_f1_score = 0
lowest_val_loss = float('inf')
optimal_epoch_f1 = 0
optimal_epoch_loss = 0
 
# Defining training and evaluation functions
def train_step():
    gnn_model.train()
    optimizer.zero_grad()
    predictions = gnn_model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(predictions[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())
    training_f1_scores.append(evaluate_f1(graph_data.train_mask))
    return loss.item()
 
# Defining function to compute the Macro F1-score 
def evaluate_f1(mask):
    gnn_model.eval()
    with torch.no_grad():
        predictions = gnn_model(graph_data.x, graph_data.edge_index)
        predicted_labels = predictions[mask].argmax(dim=1)
        f1 = f1_score(graph_data.y[mask].cpu(), predicted_labels.cpu(), average='macro')
    return f1
 
def test_step(mask):
    gnn_model.eval()
    predictions = gnn_model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(predictions[mask], graph_data.y[mask]).item()
    f1 = evaluate_f1(mask)
    return loss, f1
 
# Training loop with logging for each epoch
for epoch in range(1, 2001):
    loss = train_step()
    val_loss, val_f1 = test_step(graph_data.val_mask)
   
    # Updating best validation scores
    if val_f1 > best_val_f1_score:
        best_val_f1_score = val_f1
        optimal_epoch_f1 = epoch
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        optimal_epoch_loss = epoch
   
    # Logging every 20 epochs
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train F1: {training_f1_scores[-1]:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
 
    # Storing validation losses and F1-scores for plotting
    validation_losses.append(val_loss)
    validation_f1_scores.append(val_f1)
 
# Printing test accuracy and F1-score after training
test_loss, test_f1_score = test_step(graph_data.test_mask)
print(f'Test Loss: {test_loss:.4f}, Test F1-score: {test_f1_score:.4f}')
print(f'Best Validation F1-score: {best_val_f1_score:.4f} at Epoch {optimal_epoch_f1}')
print(f'Lowest Validation Loss: {lowest_val_loss:.4f} at Epoch {optimal_epoch_loss}')
 
# Plot Epoch vs Loss
plt.figure()
plt.plot(range(1, 2001), training_losses, label='Train Loss')
plt.plot(range(1, 2001), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Epoch vs Loss')
plt.savefig('epoch_vs_loss.png')
plt.show()
 
# Plot Epoch vs F1-Score
plt.figure()
plt.plot(range(1, 2001), training_f1_scores, label='Train Macro F1')
plt.plot(range(1, 2001), validation_f1_scores, label='Validation Macro F1')
plt.xlabel('Epoch')
plt.ylabel('Macro F1 Score')
plt.legend()
plt.title('Epoch vs F1-Score')
plt.savefig('epoch_vs_f1.png')
plt.show()
