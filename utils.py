import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
import datetime

from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader



class PointCloudDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]  
    def __popItem__(self, idx):
        return  

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def plotPCbatch_2d(pcArray1, pcArray2, show=True, save=False, name=None, fig_count=9, sizex=12, sizey=3):
    pc1 = pcArray1[:fig_count]
    pc2 = pcArray2[:fig_count]
    
    fig = plt.figure(figsize=(sizex, sizey))

    for i in range(fig_count * 2):
        ax = fig.add_subplot(2, fig_count, i + 1)
        
        if i < fig_count:
            ax.scatter(pc1[i, :, 0], pc1[i, :, 1], c='b', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc2[i - fig_count, :, 0], pc2[i - fig_count, :, 1], c='r', marker='.', alpha=0.8, s=8)

        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(0.25, 0.75)
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    if save and name:
        fig.savefig(name + '.png')
        plt.close(fig)

    if show:
        plt.show()
    else:
        return fig

def plotPCbatch(pcArray1, pcArray2, show = True, save = False, name=None, fig_count=9 , sizex = 12, sizey=3):
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*2):

        ax = fig.add_subplot(2,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)

        ax.set_xlim3d(0.25, 0.75)
        ax.set_ylim3d(0.25, 0.75)
        ax.set_zlim3d(0.25, 0.75)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig

def plot_pointcloud_comparison(pc_gt, pc_pred, show=True, save=False, name=None, fig_count=1, fixed_axes=False):
    """
    Plot 2D point clouds: top row is ground truth, bottom row is predictions.
    
    Parameters:
        pc_gt (Tensor): [B, N, 2] ground truth point clouds
        pc_pred (Tensor): [B, N, 2] predicted point clouds
        show (bool): whether to display the plot
        save (bool): whether to save the plot as an image
        name (str): filename to save (only used if save=True)
        fig_count (int): number of samples to display
        fixed_axes (bool): whether to use fixed axis limits
    """
    fig_count = min(fig_count, len(pc_gt), len(pc_pred))

    pc_pred = pc_pred.detach()
    
    fig, axes = plt.subplots(2, fig_count, figsize=(fig_count * 10, 16))
    if fig_count == 1:
        axes = axes.reshape(2, 1)

    for i in range(fig_count):
        # Ground truth
        ax = axes[0, i]
        ax.scatter(pc_gt[i, :, 0], pc_gt[i, :, 1], c='blue', s=5)
        ax.set_title(f"GT #{i}")
        ax.axis('off')
        if fixed_axes:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

        # Prediction
        ax = axes[1, i]
        ax.scatter(pc_pred[i, :, 0], pc_pred[i, :, 1], c='red', s=5)
        ax.set_title(f"Pred #{i}")
        ax.axis('off')
        if fixed_axes:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

    plt.tight_layout()

    if save and name:
        plt.savefig(f"{name}.png")
        plt.close(fig)

    if show:
        plt.show()
    else:
        return fig

def chamfer_distance(x, y, batch_reduction="mean", point_reduction="mean"):
    """
    Compute the chamfer distance between two point clouds x and y
    
    Parameters:
    -----------
    x: torch.Tensor
        First point cloud, shape (batch_size, num_points_x, dim)
    y: torch.Tensor
        Second point cloud, shape (batch_size, num_points_y, dim)
    batch_reduction: str
        Reduction operation to apply across the batch dimension: 'mean', 'sum' or None
    point_reduction: str
        Reduction operation to apply across the point dimension: 'mean', 'sum' or None
        
    Returns:
    --------
    dist_chamfer: torch.Tensor
        Chamfer distance between the point clouds
    """
    # Check input dimensions
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("Input point clouds must be 3-dimensional tensors")
    
    # Extract batch size and number of points
    batch_size, num_points_x, dim = x.shape
    _, num_points_y, _ = y.shape
    
    # Reshape to compute pairwise distances efficiently
    x_expanded = x.unsqueeze(2)  # [batch_size, num_points_x, 1, dim]
    y_expanded = y.unsqueeze(1)  # [batch_size, 1, num_points_y, dim]
    
    # Compute squared distances between each pair of points
    # |x-y|^2 = |x|^2 + |y|^2 - 2*x·y
    dist = ((x_expanded - y_expanded) ** 2).sum(dim=3)  # [batch_size, num_points_x, num_points_y]
    
    # For each point in x, find the distance to the closest point in y
    # Use values, not tuples from min()
    x_to_y = torch.min(dist, dim=2).values  # [batch_size, num_points_x]
    
    # For each point in y, find the distance to the closest point in x
    y_to_x = torch.min(dist, dim=1).values  # [batch_size, num_points_y]
    
    # Apply point reduction
    if point_reduction == "mean":
        x_to_y = x_to_y.mean(dim=1)  # [batch_size]
        y_to_x = y_to_x.mean(dim=1)  # [batch_size]
    elif point_reduction == "sum":
        x_to_y = x_to_y.sum(dim=1)  # [batch_size]
        y_to_x = y_to_x.sum(dim=1)  # [batch_size]
    elif point_reduction is None:
        pass
    else:
        raise ValueError(f"Invalid point_reduction: {point_reduction}")
    
    # Combine the two directional distances
    chamfer_dist = x_to_y + y_to_x  # [batch_size] or [batch_size, num_points]
    
    # Apply batch reduction
    if batch_reduction == "mean":
        chamfer_dist = chamfer_dist.mean()
    elif batch_reduction == "sum":
        chamfer_dist = chamfer_dist.sum()
    elif batch_reduction is None:
        pass
    else:
        raise ValueError(f"Invalid batch_reduction: {batch_reduction}")
    
    return chamfer_dist

def visualize_latent_space_pca(z, labels=None, title="Latent Space (PCA)", class_names=None, noise_std=1e-3):
    """
    Visualize latent space with PCA instead of t-SNE.

    Args:
        z: Latent space embeddings (Tensor)
        labels: Class labels (Tensor)
        title: Plot title
        class_names: Dictionary mapping label indices to class names
    """
    # Convert tensor to numpy
    z_np = z.detach().cpu().numpy()

    # Normalize if needed (centering and scaling)
    z_np = (z_np - np.mean(z_np, axis=0)) / (np.std(z_np, axis=0) + 1e-8)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_np)

    plt.figure(figsize=(12, 8))
    if labels is not None:
        labels_np = labels.cpu().numpy().flatten()
        unique_labels = np.unique(labels_np)
        num_classes = len(unique_labels)

        # Choose colormap
        cmap = plt.get_cmap("tab20b", num_classes) if num_classes <= 20 else plt.get_cmap("nipy_spectral", num_classes)
        norm = mcolors.Normalize(vmin=labels_np.min(), vmax=labels_np.max())

        # Iterate and plot each point as a character
        for i, label in enumerate(labels_np):
            plt.scatter(
                z_2d[i, 0], z_2d[i, 1], 
                marker=f"${class_names[int(label)]}$",  # Use class character as marker
                s=10,  # Adjust size for visibility
                color=cmap(norm(label)),  # Assign color
                alpha=0.8
            )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Set title and axis labels
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"latent_visualisation{timestamp}.png")

def evaluate_dataset(model, loader, dataset_name):
    model.eval()
    latent_representations = []  # Store the latent vectors
    labels = []  # Store corresponding labels (only for visualization)
    counter = 0
    print(f"\nProcessing {dataset_name} set:")
    with torch.no_grad():
        for batch in loader:            
            # Get the latent representation for the entire batch
            z = model.encode(batch)
            latent_representations.append(z)  # Assuming z[0] has shape [batch_size, latent_dim]
            
            if batch.y is not None:
                for i in range(batch.y.shape[0]):
                    labels.append(batch.y[i].unsqueeze(0))  # Append the label for the i-th graph in the batch
            counter += 1 

        latent_representations = torch.cat(latent_representations, dim=0)

        labels = torch.cat(labels, dim=0) if labels else None
        print("size of labels", labels.size)
        
        return latent_representations, labels

def evaluate_model(model, loaders):
    train_loader, test_loader = loaders

    print("Evaluating model...")

    #Evaluate all datasets
    train_samples, train_labels = evaluate_dataset(model, train_loader, "Training")
    test_samples, test_labels = evaluate_dataset(model, test_loader, "Test")

    test_labels = torch.as_tensor([25 for i in range(10)])

    # Concatenate all datasets
    all_samples = torch.cat([train_samples, test_samples], dim=0)

    train_labels = torch.full((train_samples.shape[0],), 0) if train_labels is None else train_labels
    test_labels = torch.full((test_samples.shape[0],), 2) if test_labels is None else test_labels

    print("train labels: ", train_labels.shape)
    print("test labels: ", test_labels.shape)
    all_labels = torch.cat([train_labels, test_labels], dim=0)

    print(all_samples.shape)
    print(all_labels.shape)

    #Visualize all datasets together
    visualize_latent_space_pca(
        all_samples.cpu(),
        all_labels.cpu(),
        title="Latent Space Visualization",
        class_names= ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    )

def get_datasets(batch_size=1):

    train_pc = torch.load("data/preprocessed_train_point_cloud.pt", weights_only=False)
    val_pc = torch.load("data/preprocessed_val_point_cloud.pt", weights_only=False)
    test_pc = torch.load("data/preprocessed_test_point_cloud.pt", weights_only=False)

    merged_pc = test_pc + val_pc

    test_pc = merged_pc

    train_loader = DataLoader(train_pc, shuffle=True, num_workers=0, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_pc, shuffle=True, num_workers=0, batch_size=batch_size, pin_memory=True)
    
    return train_loader, test_loader

###################################################################