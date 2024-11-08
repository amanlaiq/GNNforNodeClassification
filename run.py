import torch
from torch.nn import Linear
import torch.nn.functional as F
from data_utils import load_data
import argparse
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
 
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="name of the dataset", type=str)
parser.add_argument("--k", help="number of GNN layers", type=int, default=2)
parser.add_argument("--use_neighbors", help="use neighbors in GNN", action='store_true')
parser.add_argument("--topology_weight", help="weight for neighbor influence", type=float, default=1.0)
args = parser.parse_args()
 
# Load dataset
dataset = load_data(args.dataset)
print(f'Dataset: {args.dataset}')
print(f'Number of GNN layers: {args.k}')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
 
data = dataset[0]  # Get the first graph object
print(f'Graph info: {data}')
print('===========================================================================================================')
 
# Stats about the graph
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of test nodes: {data.test_mask.sum()}')
 
# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, use_neighbors=True, topology_weight=1.0):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.use_neighbors = use_neighbors
        self.topology_weight = topology_weight
        self.dropout = torch.nn.Dropout(p=0.4)
        self.layers = torch.nn.ModuleList()
        self.layers.append(Linear(in_channels, hidden_channels))
       
        for _ in range(num_layers - 1):
            self.layers.append(Linear(hidden_channels, hidden_channels))
        self.out_layer = Linear(hidden_channels, out_channels)
 
    def forward(self, x, edge_index):
        for layer in self.layers:
            if self.use_neighbors:
                row, col = edge_index
                neighbor_sum = torch.zeros_like(x)
                neighbor_sum.index_add_(0, col, x[row])
                x = layer(x + self.topology_weight * neighbor_sum)  # Apply topology weight here
            else:
                x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)
 
# Instantiate model with updated parameters
model = GNN(dataset.num_features, dataset.num_classes, hidden_channels=32, num_layers=args.k, use_neighbors=args.use_neighbors, topology_weight=args.topology_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()
 
# Lists to store losses and F1 scores for plotting
train_losses, val_losses = [], []
train_f1_scores, val_f1_scores = [], []
 
# Variables to track the best scores
best_val_f1 = 0
best_val_loss = float('inf')
best_epoch_f1 = 0
best_epoch_loss = 0
 
# Define training and evaluation functions
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    train_f1_scores.append(compute_f1(data.train_mask))
    return loss.item()
 
def compute_f1(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average='macro')
    return f1
 
def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask]).item()
    f1 = compute_f1(mask)
    return loss, f1
 
# Training loop with logging for each epoch
for epoch in range(1, 2001):
    loss = train()
    val_loss, val_f1 = test(data.val_mask)
   
    # Update best validation scores
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch_f1 = epoch
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch_loss = epoch
   
    # Log every 20 epochs
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train F1: {train_f1_scores[-1]:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
 
    # Store validation losses and F1-scores for plotting
    val_losses.append(val_loss)
    val_f1_scores.append(val_f1)
 
# Print test accuracy and F1-score after training
test_loss, test_f1 = test(data.test_mask)
print(f'Test Loss: {test_loss:.4f}, Test F1-score: {test_f1:.4f}')
print(f'Best Validation F1-score: {best_val_f1:.4f} at Epoch {best_epoch_f1}')
print(f'Lowest Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch_loss}')
 
# Plot Epoch vs Loss
plt.figure()
plt.plot(range(1, 2001), train_losses, label='Train Loss')
plt.plot(range(1, 2001), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Epoch vs Loss')
plt.savefig('epoch_vs_loss.png')
plt.show()
 
# Plot Epoch vs F1-Score
plt.figure()
plt.plot(range(1, 2001), train_f1_scores, label='Train Macro F1')
plt.plot(range(1, 2001), val_f1_scores, label='Validation Macro F1')
plt.xlabel('Epoch')
plt.ylabel('Macro F1 Score')
plt.legend()
plt.title('Epoch vs F1-Score')
plt.savefig('epoch_vs_f1.png')
plt.show()
