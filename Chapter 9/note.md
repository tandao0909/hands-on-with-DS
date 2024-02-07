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
- Each instance then be assigned to one of the five clusters. 
- In the context of clustering, an instance's label is the index of the cluster to which the algorithms assign this instance. You shouldn't be confused with the class labels in the classification tasks, which are used as targets (remember that clustering is an unsupervised task). You can see the code in the learning notebook.
- Most instances are assigned to the correct cluster and only a few instances are missed, especially those along the edge of the decision boundaries between two top-left cluster and the middle cluster. 
- In fact, *k-means* does not perform very well when the blobs have different diameters. 
- This is because *k-means* only cares about the distance from an instance to the centroid when trying to assign an instance to that cluster.
- Instead of assigning each instance to a single cluster, which is called *hard clustering* , we can give each instance a score per cluster, which is called *soft clustering*.
- This score can be the distance between the instance and the centroid or a similarity score (or affinity), such as the Gaussian radial basis function we used previously in chapter 2.
- For example, if you apply a k-means algorithm to a dataset, then we obtain a new dataset with k features: This can be a very efficient nonlinear dimensionality reduction technique.
- Similarly, you can also use these features as extra features, as we did in chapter 2.

### The K-means algorithm

- At the end, how does the algorithm work?
- Suppose, you have been given the centroids. Then you can easily label to each instance by assigning it to the cluster which is the closest.
- Conversely, if you were give n all the instances labels, you can find the centroid of each cluster by computing the mean of all instances in that cluster.
- But you don't have neither labels or centroids, how do you start with?
- Here is the break down of the k-means algorithm:
    - First, you initialize the centroids randomly. This means we pick k instances randomly from the dataset and using their location as centroids.
    - Next, now you have the centroids, you can label each instances.
    - Now you have a new label for each instance, you can average all the instance with the same label and obtain a new centroid.
    - We continue to update the centroids, update the labels, until the centroids stop moving.
    - This algorithm is guaranteed to converge in a finite number of steps (usually quite small). This is because the sum of all squared distance between the instances and their closest centroids is strictly decrease, and since it is non negative, the algorithm must converge.
- You can see each step of the algorithm in the image in the learning notebook:
    - The centroids are initialized randomly (top left).
    - Then the instances are labeled (top right).
    - Then the centroids are updated (center left).
    - The instances are relabeled (center right).
    - So on.
- As you can see, after only 3 iterations, the algorithm has reached a state seems very close to the optimal value.
- The computational complexity is generally linear to the number of instances *m*, the number of cluster *k*, and the number of features *n*. However, this is only true if the dataset has a clustering structure. If this assumption doesn't hold, then the computational complexity increases exponentially with the number of instances. In practice, this is rarely the case, thus k-means is generally one of the fastest clustering algorithms.
- Although the algorithm is guaranteed to converge, it may not converge to the right solution (i.e. it may converge the local optimum): whether it does or not depends on the centroids initialization.
- The next image in the learning notebook show two suboptimal solutions that the algorithm can converge to if you are not lucky with the random initialization step.
- There are a few way to mitigate this risk by improving the centroid initialization step.

### Centroid initialization methods

- If you happen to know approximately where the the centroids should be (i.e. you ran another clustering algorithm earlier), then you can set the `init` hyperparameter to a NumPy array containing the list of centroids, and set n_init to 1.
- Another solution is to run the algorithm several times with different random initializations and keep the best solution.
- The number of random initializations is controlled by the `n_init` hyperparameter: By default, it is 10, which means the whole algorithm described previously run 10 times when you call `fit()`, and Scikit-learn keeps the best result.
- But how does we know what is the best model? We use a performance metric!
- That metric is called the mode's *inertia*, which is the sum of squared distances between the instances and their closest centroids.
- The `KMeans` in Scikit-learn run the model `n_init` times and choose the model with the lowest inertia.
- Instead of initialize the centroids randomly, it's preferable to initialize them using the following algorithm, proposed in [a 2006 paper](https://courses.cs.duke.edu/spring07/cps296.2/papers/kMeansPlusPlus.pdf) by David Arthur and Sergei Vassilvitskii:
    - Take one centroid $\textbf{c}^{(1)}$, chosen uniformly at random form the dataset.
    - Take a new centroid $\textbf{c}^{(i)}$, choosing an instance $\textbf{x}^{(i)}$ with probability
    $$\frac{D(\textbf{x}^{(i)})^2}{\sum\limits_{j=1}^mD(\textbf{x}^{(j)})^2}$$ 
    <br>
    where $D(\textbf{x}^{(i)})^2$ is the distance between the instance $\textbf{x}^{(i)}$ and the closest centroid that was already chosen. This probability distribution ensures that instances farther away from already chosen centroids are much more likely to be selected as centroids.

    - Repeat the previous step until all *k* centroids have been chosen.
- The rest of K-Means++ is the same as regular K-Means.
- With this initialization, the K-Means++ algorithm is much less likely to converge to a suboptimal solution, so it is possible to reduce `n_init` considerably. Most of the time, this largely compensates for the additional complexity of the initialization process.

### Accelerated k-means and mini-batch k-means

- There is another improvement to the k-means algorithm was proposed in [a 2003 paper](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf) by Charles Elkan, named accelerated k-means.
- On large datasets with many clusters, the algorithm can be accelerated by avoiding many unnecessary distance calculations.
- This can be achieved by exploiting the triangle inequality (i.e. the straight line is the shortest distance between two points) and by keeping track of lower and upper bounds for distance between instances and centroids.
- However, Elkan's algorithm doesn't always speed up training, and sometimes it can slow down training considerably; it depends on the dataset.
- Another important variant of K-means was proposed in [a 2010 paper](https://dl.acm.org/doi/abs/10.1145/1772690.1772862) by David Sculley, named mini-batch k-means.
- Instead of using the full dataset at each iteration, the algorithm is capable of using mini-batches, moving the centroids just slightly at each iterations.
- This speeds up the algorithm (typically three to four times) and made it possible to cluster huge datasets that does not fit in the memory.
- Although the mini-batch k-means algorithm is much faster than the original k-means algorithm, its inertia is generally a bit worse.
- You can see in the learning notebook:
    - The plot on the left compares the difference between the inertias of mini-batch k-means and regular k-means models trained on the previous five-blobs dataset using various number of cluster k. The difference between two models is small, but visible.
    - In the plot on the right, you can see that overall mini-batch k-means is a bit slower than regular k-means.