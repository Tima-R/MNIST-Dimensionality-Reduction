**MNIST Dataset Analysis Overview:**

This project provides a comprehensive analysis of the MNIST dataset, utilizing various machine learning techniques to explore, visualize, and model handwritten digit data. The MNIST dataset consists of 70,000 images of handwritten digits, commonly used for training image processing systems. This repository encompasses the entire workflow of data handling, from initial loading and preprocessing to complex model application and visualization.

Getting Started
Prerequisites

Installation
1. **Clone the repository**:
    ```
    git clone https://github.com/Tima-R/MNIST-Dimensionality-Reduction.git
    ```
## Environment Setup
The project was developed and run using Google Colab, ensuring access to high computational power necessary for training deep learning models. Below are the key dependencies required:

- **Python 3.10.12**
- **Numpy**
- **Matplotlib**
- **scikit-learn**
- **UMAP-learn**
- **TSNE** 
- **TensorFlow**

**Key Components:**

**Data Loading and Preprocessing:** The dataset is loaded directly from TensorFlow's Keras API, with preprocessing steps including reshaping and normalization of image data to facilitate more efficient machine learning processing.

![train-set](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/1b7033ff-1e4e-4f10-b5b3-aa779257f36f)

**Principal Component Analysis (PCA):** PCA is used to reduce the dimensionality of the dataset while retaining the most significant features. This transformation simplifies the dataset, reducing computational requirements and potentially improving model performance.

![pca-reduced-set](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/908c5c37-1e40-4276-9f15-6b32c1b209ce)

**Variance Explained:** to check how much variance each principal component explains, to understand the effectiveness of the PCA.

![image](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/c20ef013-2928-4006-867e-89b636760420)

**Visualization of PCA Components:** The major PCA components are visualized to understand the variance and patterns within the data that PCA captures.

![first-two-components](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/534ba997-06e7-42c7-b91d-de791c73f577)

**K-Means Clustering**: To explore underlying patterns in the dataset, k-means clustering is applied, allowing for the examination of how digits are grouped together based on their visual similarity.

**Logistic Regression Classification:** A logistic regression model is trained on the PCA-reduced data, achieving a classification accuracy of 91.99% on the test set. This demonstrates the efficacy of using PCA for dimension reduction before classification.

  ```
  Classification accuracy with PCA-reduced data: 0.9199
  ```

**Advanced Visualizations:** The project employs several advanced visualization techniques to explore the multi-dimensional nature of the MNIST data.

**2D PCA Visualization:** Projects the high-dimensional data into two dimensions using PCA, providing insights into the data separability.

![first-two-components](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/1fa9a299-c5d1-412c-be4e-738a1496e3eb)

**t-SNE Visualization:** t-SNE further reduces dimensionality for visualization, highlighting clusters and patterns that are not immediately obvious with PCA.

![t-SNE](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/b0ce99d2-dd0d-49b5-adbf-c9ddfcf35f05)

**UMAP Visualization:** UMAP is used as an alternative to t-SNE for creating a visually appealing representation of high-dimensional data, offering a different perspective on the structure of the dataset.

 ![umap](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/b875801e-056b-4fff-a8f4-610161afbad3)


**Results:**

The project highlights the power of dimensionality reduction techniques and their impact on classification performance, with the logistic regression model achieving an accuracy of 91.99% on PCA-reduced data.
This repository serves as a complete guide for handling image recognition tasks with machine learning, providing scripts for each step of the process, from data preprocessing to detailed visualizations and modeling.

