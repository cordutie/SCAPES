import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

class LatentSpaceExplorer:
    """
    A visualizer for the 1024-D context embeddings in the AtomSequenceDataset.
    Color-codes dots based on their source audio file and samples evenly per file.
    """
    def __init__(self, dataset, max_samples_per_file=50):
        self.dataset = dataset
        self.max_samples_per_file = max_samples_per_file
        
        # Identify available keys
        self.requested_keys = getattr(dataset, 'requested_keys', [])
        self.has_clap = "clap_context_win" in self.requested_keys
        self.has_ctx = "ctx_emb_context_win" in self.requested_keys
        
        if not self.has_clap and not self.has_ctx:
            raise ValueError("Dataset does not request 'clap_context_win' or 'ctx_emb_context_win'. Nothing to visualize!")
            
        self.embeddings = {
            "clap": [],
            "ctx": []
        }
        self.labels = [] # Will store the filename for each point
        
        self._gather_data()

    def _gather_data(self):
        """Groups dataset by file, samples evenly, and gathers embeddings."""
        
        # 1. Group all global dataset indices by filename
        file_to_indices = {}
        for idx, (fname, _) in enumerate(self.dataset.all_indices):
            if fname not in file_to_indices:
                file_to_indices[fname] = []
            file_to_indices[fname].append(idx)
            
        # 2. Sample up to `max_samples_per_file` for each file
        selected_indices = []
        for fname, indices in file_to_indices.items():
            if len(indices) > self.max_samples_per_file:
                sampled = np.random.choice(indices, self.max_samples_per_file, replace=False).tolist()
            else:
                sampled = indices
            selected_indices.extend(sampled)
            
        # Shuffle so the progress bar time estimate is smooth (mixes big and small files)
        np.random.shuffle(selected_indices)
        
        print(f"Gathering {len(selected_indices)} total samples (up to {self.max_samples_per_file} per file)...")
        
        # 3. Load the actual tensors for the selected indices
        for idx in tqdm(selected_indices, desc="Extracting Embeddings"):
            sample = self.dataset[idx]
            
            # Grab the filename label
            self.labels.append(sample["label"])
            
            if self.has_clap:
                clap_emb = sample["clap_context_win"].view(-1, 1024).mean(dim=0).cpu().numpy()
                self.embeddings["clap"].append(clap_emb)
                
            if self.has_ctx:
                ctx_emb = sample["ctx_emb_context_win"].view(-1, 1024).mean(dim=0).cpu().numpy()
                self.embeddings["ctx"].append(ctx_emb)
                
        # Convert to numpy arrays
        if self.has_clap:
            self.embeddings["clap"] = np.vstack(self.embeddings["clap"])
        if self.has_ctx:
            self.embeddings["ctx"] = np.vstack(self.embeddings["ctx"])

    def _reduce_dimensions(self, data, method="pca"):
        """Applies PCA or t-SNE to reduce data to 2 dimensions."""
        if method.lower() == "pca":
            reducer = PCA(n_components=2)
        elif method.lower() in ["tsne", "t-sne"]:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)        
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        return reducer.fit_transform(data)

    def plot(self, method="pca", show_legend=True):
        """Plots the reduced embeddings, strictly color-coded by source file."""
        method_name = method.upper()
        print(f"\nComputing {method_name}... this might take a moment.")
        
        # --- 1. Prepare the STRICT Color Mapping ---
        unique_labels = sorted(list(set(self.labels))) # Sort to keep colors consistent across runs
        
        if len(unique_labels) <= 10:
            cmap = plt.get_cmap("tab10")
        elif len(unique_labels) <= 20:
            cmap = plt.get_cmap("tab20")
        else:
            cmap = plt.get_cmap("hsv")
            
        # Build a strict dictionary of Label -> Exact RGBA Color
        label_to_color = {}
        for i, lbl in enumerate(unique_labels):
            if len(unique_labels) <= 20:
                label_to_color[lbl] = cmap(i) # Discrete colormaps take integers
            else:
                # Continuous colormaps take a float between 0.0 and 1.0
                label_to_color[lbl] = cmap(i / max(1, len(unique_labels) - 1))
                
        # Generate the exact color for every single point in the dataset
        point_colors = [label_to_color[lbl] for lbl in self.labels]
        
        # --- 2. Setup Plot Layout ---
        num_plots = sum([self.has_clap, self.has_ctx])
        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
        if num_plots == 1:
            axes = [axes]
            
        ax_idx = 0
        
        # --- 3. Plot CLAP ---
        if self.has_clap:
            reduced_clap = self._reduce_dimensions(self.embeddings["clap"], method=method)
            axes[ax_idx].scatter(
                reduced_clap[:, 0], reduced_clap[:, 1], 
                c=point_colors, alpha=0.7, s=15, edgecolors='none' # <--- Now taking explicit colors!
            )
            axes[ax_idx].set_title(f"CLAP Embeddings ({method_name})", fontsize=14)
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1
            
        # --- 4. Plot GlobalEncoder ---
        if self.has_ctx:
            reduced_ctx = self._reduce_dimensions(self.embeddings["ctx"], method=method)
            axes[ax_idx].scatter(
                reduced_ctx[:, 0], reduced_ctx[:, 1], 
                c=point_colors, alpha=0.7, s=15, edgecolors='none' # <--- Now taking explicit colors!
            )
            axes[ax_idx].set_title(f"GlobalEncoder Context ({method_name})", fontsize=14)
            axes[ax_idx].grid(True, alpha=0.3)
            
        # --- 5. Add Legend ---
        if show_legend and len(unique_labels) <= 30:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=label_to_color[lbl], # <--- Pulling from the exact same dictionary
                       markersize=8, label=lbl)
                for lbl in unique_labels
            ]
            axes[-1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
        elif show_legend:
            print(f"Note: Suppressed legend because there are too many unique files ({len(unique_labels)}).")
            
        plt.tight_layout()
        plt.show()