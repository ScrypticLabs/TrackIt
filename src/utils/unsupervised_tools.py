import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Utility function to visualize the outputs of PCA and t-SNE
def scatter_plot(x, colors):
    """
    x: ndarray of shape (N, D) where D = 2 to visualize in two-dimensions
    colors: ndarray of shape (N,) where each value is a label for the respective sample
    """
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))+1
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
#     ax.axis('off')
#     ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def scatter_plot_3D(x, colors):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))+1
    palette = np.array(sns.color_palette("hls", num_classes))

    sc = ax.scatter3D(x[:,0], x[:,1], x[:, 2], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.show()

def pca(X, components=2):
    """
    X: ndarray of shape (N, D) where D = latent dimension 
    """
    pca = PCA(n_components=components)
    X_prime = pca.fit_transform(X)
    return X_prime # shape is (N, components)

def tsne(X, RS=123):
    """
    X: ndarray of shape (N, D) where D = latent dimension 
    """
    X_prime = TSNE(random_state=RS).fit_transform(X)
    return X_prime # shape is (N, 2)