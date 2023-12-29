import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# Load the weights
baseline_weights_1 = torch.load("ckpt/aishell_sub4/avg_20.pt")
baseline_weights_2 = torch.load("ckpt/aishell_sub6/avg_20.pt")
baseline_weights_3 = torch.load("ckpt/aishell_sub8/avg_20.pt")
proposed_weights = torch.load("ckpt/aishell_hydra468_nopos/avg_20.pt")

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import kde


# Extract weights from specific layers
def extract_weights(weights_dict, layer_name):
    for name, param in weights_dict.items():
        if name == layer_name:
            param=param.detach().cpu().numpy()
            param=param.reshape(param.shape[0], -1)
            return param[400:450, :]

baseline_weights_1_layers1 = []
baseline_weights_1_layers2 = []
baseline_weights_2_layers1 = []
baseline_weights_2_layers2 = []
baseline_weights_3_layers1 = []
baseline_weights_3_layers2 = []
proposed_weights_layers1 = []
proposed_weights_layers2 = []

# Extract weights for all 12 layers
for i in range(12):
    layer_name1 = f"encoder.encoders.{i}.conv_module.pointwise_conv1.weight"
    layer_name2 = f"encoder.encoders.{i}.feed_forward.w_1.weight"

    baseline_weights_1_layers1.append(extract_weights(baseline_weights_1, layer_name1))
    baseline_weights_1_layers2.append(extract_weights(baseline_weights_1, layer_name2))

    baseline_weights_2_layers1.append(extract_weights(baseline_weights_2, layer_name1))
    baseline_weights_2_layers2.append(extract_weights(baseline_weights_2, layer_name2))

    baseline_weights_3_layers1.append(extract_weights(baseline_weights_3, layer_name1))
    baseline_weights_3_layers2.append(extract_weights(baseline_weights_3, layer_name2))

    proposed_weights_layers1.append(extract_weights(proposed_weights, layer_name1))
    proposed_weights_layers2.append(extract_weights(proposed_weights, layer_name2))

# Concatenate weights of all 12 layers
baseline_weights_1_concat_layer1 = np.concatenate(baseline_weights_1_layers1, axis=0)
baseline_weights_1_concat_layer2 = np.concatenate(baseline_weights_1_layers2, axis=0)

baseline_weights_2_concat_layer1 = np.concatenate(baseline_weights_2_layers1, axis=0)
baseline_weights_2_concat_layer2 = np.concatenate(baseline_weights_2_layers2, axis=0)

baseline_weights_3_concat_layer1 = np.concatenate(baseline_weights_3_layers1, axis=0)
baseline_weights_3_concat_layer2 = np.concatenate(baseline_weights_3_layers2, axis=0)

proposed_weights_concat_layer1 = np.concatenate(proposed_weights_layers1, axis=0)
proposed_weights_concat_layer2 = np.concatenate(proposed_weights_layers2, axis=0)


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Patch

from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

from matplotlib.lines import Line2D

def plot_weights(weights_list, labels, layer_name, title, filename):
    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    reduced_weights = [tsne.fit_transform(w.reshape(w.shape[0], -1)) for w in weights_list]

    # Plot scatter plot for each set of weights
    plt.figure(figsize=(10, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    linewidths = [1, 3, 5, 7]  

    for i, (w, color, label, lw) in enumerate(zip(reduced_weights, colors, labels, linewidths)):
        plt.scatter(w[:, 0], w[:, 1], c=color, alpha=0.5, label=label)

        # Add outer contour for each set of points with different linewidths
        hull = ConvexHull(w)
        for simplex in hull.simplices:
            plt.plot(w[simplex, 0], w[simplex, 1], color=color, linewidth=lw)

    # Add labels and title
    #plt.xlabel("t-SNE 1")
    #plt.ylabel("t-SNE 2")
    #plt.title(f"{title} (Layer: {layer_name})")

    # Create custom legend with corresponding colors and linewidths
    legend_elements = [Line2D([0], [0], color=colors[i], linewidth=linewidths[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=legend_elements, fontsize=20)

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

# Plot for layer 1
weights_list_layer1 = [baseline_weights_1_concat_layer1, baseline_weights_2_concat_layer1, baseline_weights_3_concat_layer1, proposed_weights_concat_layer1]
labels_layer1 = ["baseline (1/4sub)", "baseline (1/6sub)", "baseline (1/8sub)", "HydraFormer"]
plot_weights(weights_list_layer1, labels_layer1, "Concatenated Layer 1", "Weights Distribution", "concat_layer1_weights20.png")

# Plot for concatenated layer 2
weights_list_layer2 = [baseline_weights_1_concat_layer2, baseline_weights_2_concat_layer2, baseline_weights_3_concat_layer2, proposed_weights_concat_layer2]
labels_layer2 = ["baseline (1/4sub)", "baseline (1/6sub)", "baseline (1/8sub)", "HydraFormer"]
plot_weights(weights_list_layer2, labels_layer2, "Concatenated Layer 2", "Weights Distribution", "concat_layer2_weights20.png")