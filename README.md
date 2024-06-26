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

*The PCA cumulative variance plot shows a steep initial rise, indicating that the first few components capture a significant portion of the variance (about 80% to 85% by the 50th component), suggesting these hold the majority of the information. The curve gradually flattens, with diminishing returns on additional components past 100, making around 60-80 components optimal for capturing up to 90% of the variance while maintaining efficient dimensionality reduction.*

**Visualization of PCA Components:** The major PCA components are visualized to understand the variance and patterns within the data that PCA captures.

**K-Means Clustering**: To explore underlying patterns in the dataset, k-means clustering is applied, allowing for the examination of how digits are grouped together based on their visual similarity.

**Logistic Regression Classification:** A logistic regression model is trained on the PCA-reduced data, achieving a classification accuracy of 91.99% on the test set. This demonstrates the efficacy of using PCA for dimension reduction before classification.

  ```
  Classification accuracy with PCA-reduced data: 0.9199
  ```
*The classification accuracy of 91.99% indicates that the logistic regression model, trained on PCA-reduced data from the MNIST dataset, correctly predicts the handwritten digits with a high level of precision, correctly identifying the digits in 91.99% of the cases in the test set. This performance suggests that even with reduced dimensionality, the essential features for digit recognition are retained effectively.*

**Advanced Visualizations:** The project employs several advanced visualization techniques to explore the multi-dimensional nature of the MNIST data.

**2D PCA Visualization:** Projects the high-dimensional data into two dimensions using PCA, providing insights into the data separability.

![first-two-components](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/128e96c3-4a2c-47ae-b896-438455036039)

*The image displays a scatter plot of the MNIST dataset reduced to two principal components using PCA. Each point represents a digit (0-9), color-coded by class, illustrating how PCA projects the high-dimensional data into a two-dimensional space where similar digits cluster together, albeit with some overlap. The plot reveals the inherent data structure and variability, with some digits forming distinct groups while others spread out, suggesting varying degrees of separability between different classes in this reduced space.*

**t-SNE Visualization:** t-SNE further reduces dimensionality for visualization, highlighting clusters and patterns that are not immediately obvious with PCA.

![t-SNE](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/b0ce99d2-dd0d-49b5-adbf-c9ddfcf35f05)

*The t-SNE visualization of the MNIST dataset shows distinct clusters of different digit classes, with each cluster represented by a unique color, highlighting the effective separation of similar digits into cohesive groups. This method reveals clear boundaries and grouping that are not as evident in PCA, indicating t-SNE's strength in preserving local data structures and revealing the intrinsic clustering of complex, high-dimensional data.*

**UMAP Visualization:** UMAP is used as an alternative to t-SNE for creating a visually appealing representation of high-dimensional data, offering a different perspective on the structure of the dataset.

 ![umap](https://github.com/Tima-R/MNIST-Dimensionality-Reduction/assets/116596345/b875801e-056b-4fff-a8f4-610161afbad3)

*The UMAP visualization effectively separates the MNIST digits into distinct clusters, each represented by unique colors, illustrating UMAP's ability to preserve both local and global data structures across different classes. This method shows even clearer separation and more compact clusters compared to PCA and t-SNE, highlighting its utility in accurately representing the intrinsic geometric structure of high-dimensional data.* 


**Results:**

Throughout this project, we've successfully demonstrated various dimensionality reduction and visualization techniques on the MNIST dataset, including PCA, t-SNE, and UMAP. Each method provided unique insights: PCA highlighted the general variance and was useful for initial data reduction; t-SNE excelled in revealing local data structures and forming distinct clusters of the digit classes; UMAP further refined these visualizations by preserving both local and global structures, showcasing the most definitive separation and clustering of the data. The logistic regression model, trained on PCA-reduced data, achieved a notable accuracy of 91.99%, indicating that significant information was maintained despite the dimensionality reduction. This project underscores the effectiveness of these techniques in extracting and visualizing the underlying patterns in complex datasets, making it easier to interpret the data and apply machine learning models efficiently.
