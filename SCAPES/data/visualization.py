# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from tqdm import tqdm

# class LatentSpaceExplorer:
#     """
#     A visualizer for the 1024-D context embeddings in the AtomSequenceDataset.
#     Color-codes dots based on their source audio file and samples evenly per file.
#     """
#     def __init__(self, dataset, max_samples_per_file=50):
#         self.dataset = dataset
#         self.max_samples_per_file = max_samples_per_file
        
#         # Identify available keys
#         self.requested_keys = getattr(dataset, 'requested_keys', [])
#         self.has_semantic = "target_semantic" in self.requested_keys
        
#         if not self.has_semantic:
#             raise ValueError("Dataset does not request 'target_semantic'. Nothing to visualize!")
            
#         self.embeddings = {
#             "semantic": []
#         }
#         self.labels = [] # Will store the filename for each point
        
#         self._gather_data()

#     def _gather_data(self):
#         """Groups dataset by file, samples evenly, and gathers embeddings."""
        
#         # 1. Group all global dataset indices by filename
#         file_to_indices = {}
#         for idx, (fname, _) in enumerate(self.dataset.all_indices):
#             if fname not in file_to_indices:
#                 file_to_indices[fname] = []
#             file_to_indices[fname].append(idx)
            
#         # 2. Sample up to `max_samples_per_file` for each file
#         selected_indices = []
#         for fname, indices in file_to_indices.items():
#             if len(indices) > self.max_samples_per_file:
#                 sampled = np.random.choice(indices, self.max_samples_per_file, replace=False).tolist()
#             else:
#                 sampled = indices
#             selected_indices.extend(sampled)
            
#         # Shuffle so the progress bar time estimate is smooth (mixes big and small files)
#         np.random.shuffle(selected_indices)
        
#         print(f"Gathering {len(selected_indices)} total samples (up to {self.max_samples_per_file} per file)...")
        
#         # 3. Load the actual tensors for the selected indices
#         for idx in tqdm(selected_indices, desc="Extracting Embeddings"):
#             sample = self.dataset[idx]
            
#             # Grab the filename label
#             self.labels.append(sample["label"])
            
#             if self.has_semantic:
#                 semantic_emb = sample["target_semantic"].view(-1, 1024).mean(dim=0).cpu().numpy()
#                 self.embeddings["semantic"].append(semantic_emb)
                
#         # Convert to numpy arrays
#         if self.has_semantic:
#             self.embeddings["semantic"] = np.vstack(self.embeddings["semantic"])

#     def _reduce_dimensions(self, data, method="pca"):
#         """Applies PCA or t-SNE to reduce data to 2 dimensions."""
#         if method.lower() == "pca":
#             reducer = PCA(n_components=2)
#         elif method.lower() in ["tsne", "t-sne"]:
#             reducer = TSNE(n_components=2, perplexity=30, random_state=42)        
#         else:
#             raise ValueError("Method must be 'pca' or 'tsne'")
            
#         return reducer.fit_transform(data)

#     def plot(self, method="pca", show_legend=True):
#         """Plots the reduced embeddings, strictly color-coded by source file."""
#         method_name = method.upper()
#         print(f"\nComputing {method_name}... this might take a moment.")
        
#         # --- 1. Prepare the STRICT Color Mapping ---
#         unique_labels = sorted(list(set(self.labels))) # Sort to keep colors consistent across runs
        
#         if len(unique_labels) <= 10:
#             cmap = plt.get_cmap("tab10")
#         elif len(unique_labels) <= 20:
#             cmap = plt.get_cmap("tab20")
#         else:
#             cmap = plt.get_cmap("hsv")
            
#         # Build a strict dictionary of Label -> Exact RGBA Color
#         label_to_color = {}
#         for i, lbl in enumerate(unique_labels):
#             if len(unique_labels) <= 20:
#                 label_to_color[lbl] = cmap(i) # Discrete colormaps take integers
#             else:
#                 # Continuous colormaps take a float between 0.0 and 1.0
#                 label_to_color[lbl] = cmap(i / max(1, len(unique_labels) - 1))
                
#         # Generate the exact color for every single point in the dataset
#         point_colors = [label_to_color[lbl] for lbl in self.labels]
        
#         # --- 2. Setup Plot Layout ---
#         num_plots = 1 if self.has_semantic else 0
#         fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
#         if num_plots == 1:
#             axes = [axes]
            
#         ax_idx = 0
        
#         # --- 3. Plot Target Semantic ---
#         if self.has_semantic:
#             reduced_semantic = self._reduce_dimensions(self.embeddings["semantic"], method=method)
#             axes[ax_idx].scatter(
#                 reduced_semantic[:, 0], reduced_semantic[:, 1], 
#                 c=point_colors, alpha=0.7, s=15, edgecolors='none' # <--- Now taking explicit colors!
#             )
#             axes[ax_idx].set_title(f"Target Semantic ({method_name})", fontsize=14)
#             axes[ax_idx].grid(True, alpha=0.3)
            
#         # --- 5. Add Legend ---
#         if show_legend and len(unique_labels) <= 30:
#             from matplotlib.lines import Line2D
#             legend_elements = [
#                 Line2D([0], [0], marker='o', color='w', 
#                        markerfacecolor=label_to_color[lbl], # <--- Pulling from the exact same dictionary
#                        markersize=8, label=lbl)
#                 for lbl in unique_labels
#             ]
#             axes[-1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
#         elif show_legend:
#             print(f"Note: Suppressed legend because there are too many unique files ({len(unique_labels)}).")
            
#         plt.tight_layout()
#         plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


class LatentSpaceExplorer:
    """
    Visualizes semantic + structure signals from AtomSequenceDataset.

    STRUCTURE MODEL:
        [F, T] per sample
        → expanded into F independent samples of dimension T
    """

    def __init__(
        self,
        dataset,
        max_samples_per_file=50,
    ):
        self.dataset = dataset
        self.max_samples_per_file = max_samples_per_file

        self.requested_keys = getattr(dataset, "requested_keys", [])

        self.has_semantic = "target_semantic" in self.requested_keys
        self.has_structure = "target_structure" in self.requested_keys

        if not self.has_semantic and not self.has_structure:
            raise ValueError(
                "Dataset must request target_semantic or target_structure."
            )

        self.embeddings = {}
        self.labels = []

        # IMPORTANT: separate labels for structure expansion
        self.structure_labels = []

        if self.has_semantic:
            self.embeddings["semantic"] = []

        if self.has_structure:
            self.embeddings["structure"] = []

        self._gather_data()

    # ==========================================================
    # DATA GATHERING
    # ==========================================================

    def _gather_data(self):

        file_to_indices = {}

        for idx, (fname, _) in enumerate(self.dataset.all_indices):
            file_to_indices.setdefault(fname, []).append(idx)

        selected_indices = []

        for fname, indices in file_to_indices.items():

            if len(indices) > self.max_samples_per_file:
                sampled = np.random.choice(
                    indices,
                    self.max_samples_per_file,
                    replace=False
                ).tolist()
            else:
                sampled = indices

            selected_indices.extend(sampled)

        np.random.shuffle(selected_indices)

        print(f"Gathering {len(selected_indices)} samples...")

        for idx in tqdm(selected_indices, desc="Extracting embeddings"):
            # print(f"[IDX {idx}] label={sample['label']}")

            sample = self.dataset[idx]

            # ======================================================
            # LABEL (semantic is sample-level)
            # ======================================================
            self.labels.append(sample["label"])

            # ======================================================
            # SEMANTIC (1 sample → 1 embedding)
            # ======================================================
            if self.has_semantic:
                semantic = sample["target_semantic"]

                emb = (
                    semantic
                    .view(-1, semantic.shape[-1])
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )

                self.embeddings["semantic"].append(emb)

            # ======================================================
            # STRUCTURE (EXPANDED: F samples per item)
            # ======================================================
            if self.has_structure:

                structure = sample["target_structure"]  # [F, T]

                if structure.ndim != 1:
                    for i in range(structure.shape[1]):

                        vec = structure[:,i].cpu().numpy()

                        self.embeddings["structure"].append(vec)

                        # IMPORTANT: duplicate label per feature-vector
                        self.structure_labels.append(sample["label"])
                else:
                    # Handle case where structure is already 1D (e.g., mean-pooled)
                    vec = structure.cpu().numpy()
                    self.embeddings["structure"].append(vec)
                    self.structure_labels.append(sample["label"])
                    
        print(f"Collected {len(self.embeddings.get('semantic', []))} semantic and {len(self.embeddings.get('structure', []))} structure samples.")

        # Convert to numpy arrays
        for key in self.embeddings:
            self.embeddings[key] = np.vstack(self.embeddings[key])

    # ==========================================================
    # DIM REDUCTION
    # ==========================================================

    def _reduce_dimensions(self, data, method="pca"):

        if method.lower() == "pca":
            reducer = PCA(n_components=2)

        elif method.lower() in ["tsne", "t-sne"]:
            reducer = TSNE(
                n_components=2,
                perplexity=30,
                random_state=42
            )

        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        return reducer.fit_transform(data)

    # ==========================================================
    # COLOR MAPPING
    # ==========================================================

    def _build_color_mapping(self, labels):

        unique_labels = sorted(set(labels))

        cmap = (
            plt.get_cmap("tab10")
            if len(unique_labels) <= 10 else
            plt.get_cmap("tab20")
            if len(unique_labels) <= 20 else
            plt.get_cmap("hsv")
        )

        label_to_color = {}

        for i, lbl in enumerate(unique_labels):
            if len(unique_labels) <= 20:
                label_to_color[lbl] = cmap(i)
            else:
                label_to_color[lbl] = cmap(
                    i / max(1, len(unique_labels) - 1)
                )

        point_colors = [label_to_color[l] for l in labels]

        return unique_labels, label_to_color, point_colors

    # ==========================================================
    # PLOTS
    # ==========================================================

    def plot_semantic(self, method="pca", show_legend=True):

        if not self.has_semantic:
            raise ValueError("No semantic embeddings available.")

        reduced = self._reduce_dimensions(
            self.embeddings["semantic"],
            method=method
        )

        self._plot(
            reduced,
            self.labels,
            title=f"Semantic ({method.upper()})",
            show_legend=show_legend
        )

    def plot_structure(self, method="pca", show_legend=True):

        if not self.has_structure:
            raise ValueError("No structure embeddings available.")

        reduced = self._reduce_dimensions(
            self.embeddings["structure"],
            method=method
        )

        self._plot(
            reduced,
            self.structure_labels,
            title=f"Structure ({method.upper()})",
            show_legend=show_legend
        )

    # ==========================================================
    # INTERNAL PLOT
    # ==========================================================

    def _plot(self, reduced, labels, title, show_legend=True):

        _, label_to_color, colors = self._build_color_mapping(labels)

        plt.figure(figsize=(8, 6))

        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=colors,
            s=15,
            alpha=0.7,
            edgecolors="none"
        )

        plt.title(title)
        plt.grid(True, alpha=0.3)

        if show_legend and len(set(labels)) <= 30:

            from matplotlib.lines import Line2D

            unique_labels = sorted(set(labels))

            legend_elements = [
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=label_to_color[lbl],
                    markersize=8,
                    label=lbl
                )
                for lbl in unique_labels
            ]

            plt.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
                fontsize=9
            )

        elif show_legend:
            print(f"Legend suppressed ({len(set(labels))} labels).")

        plt.tight_layout()
        plt.show()