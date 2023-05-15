from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import json

path = 'activity/0a09f8fc-ff87-4210-b682-d2ae38af33eb_pt.json'
# Load your IMU embeddings
with open(path, 'r') as f:
    data = json.load(f)

window_ids = list(data.keys())
embeddings = [data[window_id][0] for window_id in window_ids]
labels = [data[window_id][1] for window_id in window_ids]
print("Shape of embeddings: ", np.array(embeddings).shape)
print("Shape of labels: ", np.array(labels).shape)

# Convert the embeddings and labels to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)
embeddings = embeddings.reshape(embeddings.shape[0], -1)


# Initialize the t-SNE model with desired hyperparameters
tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, verbose=1, random_state=42)

# Fit the model to the embeddings
embeddings_tsne = tsne.fit_transform(embeddings)

# Get unique labels
unique_labels = np.unique(labels)

# Create a color map for the unique labels
label_colors = plt.cm.get_cmap('tab10', len(unique_labels))

num_shades = len(unique_labels)
# Adjust brightness or alpha value for distinct shades
color_values = np.linspace(0.3, 1, num_shades)

# Visualize the embeddings in a scatter plot with color-coded labels
for i, label in enumerate(unique_labels):
    mask = labels == label
    plt.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], s=5, c=label_colors(i % len(unique_labels)), label=label)

# Add legend
plt.legend()

# Show the plot
plt.show()
plt.savefig("viz/{}.png".format(path.split("_")[-1].split(".")[0]))
