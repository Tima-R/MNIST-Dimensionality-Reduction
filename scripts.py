
# 1. load_dataset
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()




# 2. display_samples          
import matplotlib.pyplot as plt

# Number of images to be displayed
num_images = 10

# To plot the images
plt.figure(figsize=(10,10)) # To create the plotting area, 10*10" canvas for image
for i in range(num_images):
  # Reshape the data to 28*28 pixels
  img = x_train[i].reshape(28,28) # To reshape the data to 28x28 pixels since MNIST images are 28x28 grayscale images

  # To displey the image
  plt.subplot(1, num_images, i+1) # To create a subplot for each image
  plt.imshow(img, cmap='gray') # To display the images in grayscale
  plt.axis('off') # To hide the axes

plt.show()




# 3. preprocess_data

# Reshape and scale the data
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0




# 4. apply_pca  
from sklearn.decomposition import PCA

# Number of components
n_components = 100 # To reduce the number of components from 784 for each image to 100 to capture the most variance in the data

# To create the PCA instance
pca = PCA(n_components=n_components)

# To fit the PCA on training data
pca.fit(x_train) # To compute the principal components based solely on the training data

# To apply the mapping to both training and test data (transforming the both training and test data)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)




# 5. visualize_pca_components
import matplotlib.pyplot as plt

fig, axes = plt.subplots(10, 10, figsize=(9,9),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
  ax.imshow(pca.components_[i].reshape(28,28), cmap='gray')

plt.show()



# 6.check_variance   
import numpy as np

# Variance explained
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Culmulative Explained Variance')
plt.show()



# 7. cluster_kmeans 
# Clustering with K-Means
from sklearn.cluster import KMeans

# I select 10 clusters for MNIST
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)

# To fit the model to the PCA-reduced data
kmeans.fit(x_train_pca)

# To predict the cluster labels for the train dataset
y_train_pred = kmeans.predict(x_train_pca)




# 8. classify_logistic       
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# To create the logistic regression model
logisticRegr = LogisticRegression(max_iter=1000)

# To train the model with the PCA-reduced data
logisticRegr.fit(x_train_pca, y_train)

# To predict labels for the test set
y_test_pred = logisticRegr.predict(x_test_pca)

# To calculate the accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Classification accuracy with PCA-reduced data: {accuracy}")




# 9. visualize_2d_pca  
pca_2d = PCA(n_components=2)
x_train_pca_2d = pca_2d.fit_transform(x_train)

# To plot the first 2 principals components
plt.figure(figsize=(8, 6))

# Using scatter plot to visualize the first two components
scatter = plt.scatter(x_train_pca_2d[:, 0],
                      x_train_pca_2d[:, 1],
                      c=y_train, alpha=0.5,
                      cmap='tab10')

# Adding a legend for each digit
plt.legend(*scatter.legend_elements(), title="Digits")

# Label the axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of MNIST Dataset')

plt.show()




# 10. visualize_tsne
# Using the PCA-reduced data as input to t-SNE
x_pca_reduced = pca.transform(x_train)

# Creating a t-sne model
tsne = TSNE(n_components=2, perplexity=30,
            learning_rate=200,
            random_state=42)

# Fitting and transforming the data
x_train_tsne = tsne.fit_transform(x_pca_reduced)

# Plotting the results of t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1],
                      c=y_train, alpha=0.6,
                      cmap='tab10')
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title('t-SNE visualization of MNIST data')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()




# 11. visualize_umap
!pip install umap-learn

import umap

# Create a UMAP instance
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

# Fit and transform the data
# You can use the raw data, but using PCA-reduced data can speed up the process and may sometimes improve the results.
x_train_umap = umap_model.fit_transform(x_train_pca)

# To plot the result of UMAP
plt.figure(figsize=(12, 10))
scatter = plt.scatter(x_train_umap[:, 0], x_train_umap[:, 1], c=y_train, alpha=0.6, cmap='tab10')
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title('UMAP visualization of MNIST data')
plt.xlabel('UMAP feature 1')
plt.ylabel('UMAP feature 2')
plt.show()
