- Although most of machine learning applications are based on supervised learning, most of problems we encountered are unsupervised learning.
- We did done the most common unsupervised learning in chapter 8: dimensionality reduction. 
- There are other applications of unsupervised learning as well:
    - Clustering: The goal is to group instances into *clusters*. Clustering is great for data analysis, customer segmentation, recommender system, search engines, image segmentation, semi-supervised learning, dimensionality reduction, etc.
    - Anomaly detection (also called outliers detection): The goal is to learn what a "normal" instance looks like, and use it to detect abnormal instances. These instances are called *anomalies* or *outliers*, while the normal one is called *inliers*. This is great for fraud detection (an example is Spam or Ham dataset), detecting defective item in manufacturing, identifying new trends in time series, and remove outliers from dataset prior to training, which can greatly increase the performance of the model.
    - Density estimation: This is the task for estimating of *probability density function* (PDF) of the random process that generates the dataset. Density estimation can be used to detecting outliers, instances which lie in a low-density area has a high chance to be an outlier. This is also helpful for data analysis and visualization.

# Clustering Algorithms: k-means and DBSCAN

- Suppose you are enjoying the sun on the beach, and you see lots of shell on the shore. They are not perfectly identical, yet similar enough for you to conclude that they probably belong to the same species (or at least the same genus). You may need an expert to tell exactly what species it is, but you don't need one to group multiple similar-looking objects together.
- This is called clustering: It is the task of identifying similar instances and assign them together to *clusters*, which are groups of similar objects.
- Just like in classification, each instance will be assigned to a group. However, unlike classification, clustering doesn't have labels, thus is an unsupervised task. 
- Look at the image in the learning notebook:
    - The left plot is the original Iris dataset with the corresponding label for each instance. This is a classification task, so we can apply random forest, logistic regression, SVM, etc for it.
    - The right plot is the instances without labels. Because of the lack of labels, you can't use a classification algorithm anymore. This is where clustering algorithms steps in: Most of them can easily detect the lower-left cluster. That is quite easy to conclude with our own eyes, but the fact that the upper right is composed of two distinct clusters.
    - That said, here we only use two features out of 4 given features, so if we add 2 remaining in, clustering then can work on all available features and gives us a reasonably good result. For example, using a Gaussian mixture model, we only incorrectly assign 5 out of 150 instances in the dataset to the wrong cluster.
- Clustering has a wide range of applications, including:
    - Customer segmentation: You can cluster your customers based on their behaviour when using our your products (typically a website or an app). Then you can set specific marketing campaigns and products changing aimed to specific segments. For example, customer segmentation can be used in *recommender systems* to suggest content that other users in the same cluster may be interested in.
    - Data analysis: When you analyze a new dataset, it can be helpful to run a simple clustering algorithm and learn each cluster separately.
    - Dimensionality reduction: Once a dataset has been clustered, it is possible to measure the *affinity* of each instance with regard to each cluster. Affinity is any measure of how well an instance fits into a cluster. Each instance's features vector $\textbf{x}$ can then be replaced by the vector of its cluster affinities. If there are k clusters, then this vector is k-dimensional. The new vector is typically much lower-dimensional than the original feature vector, but still preserves enough information for further processing.
    - Feature engineering: The cluster affinities can also be used as extra features. For example, we used k-means in chapter 2 to add geographical cluster affinities features to the California housing dataset, and they helped us get better performance.
    - Anomaly detection (also called outlier detection): Any instance that has a low affinity to all the clusters is likely to be an anomaly. For example, if you have clustered the users of your websites based on their behavior, you can detect users with unusual behavior, such as an unusual number of requests per second.
    - Semi-supervised learning: If you have only a few labels, you could perform clustering and propagate the labels to all the instances in the same cluster. This technique can greatly increase the amount of labels available to an subsequent algorithm, thus greatly increase its performance.
    - Search engines: Clustering can help users find similar results grouped together and suggest related queries. By clustering each customer to a cluster, it also help personalized the search result. Clustering can help utilizing search by image. For example, when a user provides an image, we assign it to a cluster and return all images in that cluster.
    - Image segmentation: By clustering pixels according to their color, then replacing each pixel's color with the mean color of its cluster, it is possible to considerably reduce the number of different colors in an image. Image segmentation is used in many object detection and tracking systems, as it is easier to detect the contour of each object.
    - Document clustering: In natural language processing, clustering can be used to group similar documents together. This can used in news website, where you can group multiple articles under the same topic.
- There is no universal definition of a cluster: it all depends on the context, and different algorithms capture different types of clusters:
    - Some algorithms look for instances centered around a particular point, called a *centroid*.
    - Other algorithms look for continuous regions of densely packed instances, while these clusters can take on any shape.
    - Some algorithms are hierarchical: They look for clusters of clusters.
    - And more.

## k-means

- Look at the image in the learning notebook. You can easily see there are five blobs of instances.
- We must specify the number of cluster *k* in order to run the algorithm. In this example, we know that *k=5*, but in general it won't be that easy. We will talk about it later