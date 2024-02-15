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

### Finding the optimal number of clusters

- So far, we set the number of clusters k to 5 because we know from the start there are 5 clusters and we see that there are 5 cluster when plotting the dataset.
- However, in practice, it wont't be easy to find the right number of cluster k, and the result may be quite bad if we set it wrong.
- As you can see in the learning notebook, if we set the number of cluster to either 3 or 8, the result is fairly bad.
- You may be thinking you just pick the model with the lowest inertia. Unlucky for us, it's not that simple. 
- The inertia for $k=3$ is about 653.2, which is much higher than $k=5$ (211.6). But for $k=8$, the inertia is just 119.2.
- The inertia is not a good performance metric when trying to choose k, because it will keep getting smaller as we increase k. In fact, the more clusters there are, the closest each instances will be to its closest centroids, hence the lower the inertia will be.
- In the learning notebook,we plot the inertia as a function of k. When we do this, the curve usually contains an inflexion point called the *elbow*.
- As you can see, the inertia drops very quickly when we increase k up to 4, but then it decreases much more slowly as we keep increasing k. This curve has roughly the shape of an arm, and there's an elbow at $k=4$. 
- So if we don't know any better, we might think 4 is a good choice: Any lower value would be dramatic, while any higher value would not help much, and we might just be splitting a perfectly good cluster in half for no good reason.
- This technique for choosing the right number of clusters is rather coarse. A more precise (but also more computational expensive) approach is to use the *silhouette score*, which is the mean *silhouette coefficient* over all the instances.
- The silhouette coefficient is equal to $(b-a)/\max(a,b)$, where a is the mean distance to all other instances in the same cluster (i.e., the mean intra-cluster distance) and b is the mean nearest-cluster distance (i.e. the mean distance to the instances of the next closest cluster, defines as the one minimizes *b*, excluding the instance's own cluster). 
- The silhouette coefficient can range from -1 to 1:
    - A coefficient closes to 1 means that the instance is well inside its own cluster and far away from other cluster.
    - A coefficient closes to 0 means that the instance is close to a cluster boundary.
    - A coefficient closes to -1 means that the instance is probably assigned to the wrong cluster.
- You can see an image compares the silhouette score of different number of clusters in the learning notebook:
    - This visualization is much richer than the previous one.
    - We can confirm that $k=4$ is a very good choice. 
    - However, $k=5$ is also a pretty good choice, and much better than $k=6$ or 7.
    - We can have this precise comment if we just rely on the inertias.
- An even more informative way is plotting every instance's silhouette, sorted by the clusters they are assigned to and by the value of the coefficient. This is called a *silhouette diagram*.
- Each silhouette contains:
    - One knife shape per cluster.
    - The shape's height indicated the number of instances in the associated cluster.
    - The shape's represents the sorted silhouette coefficients of the instance in the cluster (wider is better).
    - The vertical dashed lines represent the mean silhouette score for each number of clusters.
- When most of the instances in a cluster have a lower silhouette coefficient than the vertical line (i.e. if many of the instances stop short of the dashed line, ending to the left of it), then the cluster is rather bad, since this means are mostly too close to other cluster.
- In the learning notebook, we plot the silhouette diagram for $k = 3, 4, 5, 6$:
    - If $k = 3$ or 6, we get bad clusters.
    - But when $k=4$ or 5, the clusters actually look promising: Most instances extend beyond the dashed line, to the right and closer to 1.0.
    - When $k=4$, the cluster at index 1 is rather big.
    - When $k=5$, all clusters have similar sizes.
    - So even though $k=4$ yields a better overall silhouette scores, it seems like a better idea to use $k=5$, as we can have clusters of similar sizes. 

## Limits of K-Means

- Despite many of its merits, most notably being fast and scalable, k-means is definitely not perfect.
- As we saw, you may need to rerun the model several times to avoid the suboptimal solutions, plus you need to specify the number fo clusters, which can be a very difficult task.
- Moreover, k-means doesn't behave well when the clusters have varying sizes, different densities, or nonspherical shape.
- There is some examples of how k-means clusters a dataset including three ellipsoidal clusters of different dimensions, densities and orientations in the learning notebook.
- As you can see, neither of our solutions is good:
    - The solution on the left is better, but it still chops off about 25% instances of the middle cluster and assign them to the cluster on the right.
    - The solution on the right is just terrible, even when its inertia is lower.
- So depending on the dataset, different clustering algorithm may perform better. On these types of elliptical clusters, Gaussian mixture models work great.
- It is important to scale the input features before you run k-means, or the clusters may be very stretched and k-means will perform poorly. Scaling the features doesn't guarantee that all the clusters will be nice and spherical, but it generally helps k-means.

## Using Clustering for Image Segmentation

- *Image segmentation* is the task of partitioning an image into multiple segments (i.e. pixels that have some relationships together). There are several variants:
    - *Color segmentation*: Pixels with a similar color get assigned to the same segment. This is sufficient in many applications. For example, if you want to analyze satellite images to measure how much total forest area there is in a region, color segmentation may be just fine.
    - *Semantic segmentation*: All pixels that are part of the same object type get assigned to the same segment. For example, in a self-driving car's vision system, all pixels that are part of a pedestrians' images might be assigned to the "pedestrians" segment (there would be one segment containing all the pedestrians).
    - *Instance segmentation*: All pixels that are part of the same individual object are assigned to the same segment. In this case, there would a segment for each pedestrian.
- The state of the art in sematic or instance segmentation today is achieved using complex architecture based on convolutional neural networks (will be discussed in chapter 14).
- In this chapter, we will approach the (much simpler) color segmentation problem using k-means.
- We will load an image about a ladybug.
- This image is represented as a 3D array:
    - The first dimension is the height.
    - The second dimension is the width.
    - The third dimension is the number of color channels, in this case red, green and blue (RGB).
- In other words, for each pixel, there is a 3D vector containing the intensities of red, green, blue as unsigned 8-bit integers range from 0 to 255.
- Some images may have fewer channels (e.g. greyscale images, which only have one), and some images may have more channels (e.g. images with an additional *alpha channel* for transparency, or satellite images, which often contains channels for additional light frequencies, like infrared).
- You can experiment with various number of clusters in this small problem, as we did in the learning notebook.
- When you use fewer than eight clusters, notice that the ladybug's flashy red color fails to get a cluster of its own: It gets merged with colors from the environment. This is because k-means prefers clusters of similar sizes. The ladybug is small - much smaller compared to the rest of the image - so even though its color is flashy, k-means fails to dedicate a cluster to it.

## Using Clustering for Semi-Supervised Learning

- Another way to use clustering is in semi-supervised learning, when we have plenty of unlabeled instances and very few labeled instances.
- This part focuses a lot in the learning notebook. We did the following:
    - Import a digits dataset. Pretend only the first 50 instances and the test set are labeled.
    - Train only on these 50 instances.
    - Train a k-means to clustering 50 clusters, with each cluster we find the digit closest to its centroid.
    - Then we propagate to the whole training set.
    - We trip off outliers.
- More detail in the learning notebook.
- Scikit-learn also offers two classes that can propagate labels automatically: `LabelSpreading` and `LabelPropagation` in the `sklearn.semi_supervised` package. Both classes construct a similarity matrix between all the instances, and iteratively propagate labels from labeled instances to similar unlabeled instances.
- There's also a very different class named `SelfTrainingClassifier` in the same package: You give it a base classifier (such as a `RandomForestClassifier`) and it trains on the labeled instances, then uses it to predict label for unlabeled samples. It then updates  the training set with the labels it is most confident about, and repeats this process of training and labeling until it cannot add labels anymore.
- These techniques is not silver bullet, but it can occasionally give your model a little boost.
- To further improve your model and your training set, the next step could be to do a few rounds of *active learning*, which is when a human expert interacts with the learning algorithm, providing labels for specific instances when the algorithm asks for them.
- There are many different strategies for active learning, but one of the most common ones is called *uncertainty sampling*. Here is the breakdown of it:
    - The model is trained on the labeled instances gathered so far, and this model is used to make predictions on all the unlabeled instances.
    - The instances for which the model is the most uncertain (i.e., where its estimated probability is lowest) are given to the expert for labeling.
    - You iterate this process until the performance improvements stops being worth the labeling effort.
- Other active learning strategies include labeling the instances that would result in the largest model change or the largest drop in the model's validation error, or the instances that different models disagree on (e.g., an SVM and a random forest).

# DBSCAN

- The *density-based spatial clustering application with noise* (DBSCAN) algorithm defines cluster as continuous regions of high density. Here is how it works:
    - For each instance, the algorithm counts how many instances are located within a small distance $\varepsilon$ (epsilon) from it. This region is called the instances's *$\varepsilon$-neighborhood*.
    - If an instance has at least `min_samples` instances in its $\varepsilon$-neighborhood (including itself), then it is called a *core instance*. In other words, core instances are those located in dense regions.
    - All instances in the neighborhood of a core instance belong to the same cluster. This neighborhood may include other core instances; therefore, a long sequence of neighboring core instances form a single cluster.
    - Any instance that is not a core instance and does not have one in its neighborhood is considered an anomaly.
- Most of this part will be in the learning notebook. Visit it for more detail.
- This algorithm works well if all the clusters are well separated by low-density regions.
- In short, DBSCAN is a very simple yet powerful algorithm capable of identifying any number of clusters of any shape. It is robust to outliers, and only have two hyperparameters to tweak (`min_samples` and `eps`)
- However, if the density varies across the clusters or if there's no sufficiently low-density region around some cluster, DBSCAN will be struggle to capture all the clusters properly.
- Moreover, its computational complexity is $O(m^2n)$, hence it doesn't scale well with large datasets.
- You may want to try *hierarchial DBSCAN* (HDBSCAN), which is implemented in the [scikit-learn-contrib project](https://github.com/scikit-learn-contrib/hdbscan), as it is usually better than DBSCAN at finding clusters of varying densities.

## Other clustering algorithms

Scikit-learn offers several more clustering algorithms that you should take a look at (in the `sklearn.cluster` package). We will cover just a fraction of them:
- *Agglomerative clustering*: A hierarchy of cluster built from the bottom up. Think of bubbles, small bubbles float on water and float, eventually merge together and form one big group of bubbles. Similarly, at each iteration, agglomerative clustering merge the nearest pairs of clusters (start with individual instances). If you draw a tree with a branch for every pair of cluster you merge, you would get a binary tree of clusters, where its leaf nodes are the individual instances. This approach can capture clusters of various shape; it also produces a flexible and informative cluster tree instead of forcing us to choose a specific cluster size, and it can be used with any pairwise distance metric. It can scale well with the number of instances if you provide a connectivity matrix, which is a sparse $m \times m$ matrix that indicates which pairs of instances are neighbors (e.g., returned by `sklearn.neighbors.kneighbors_graph()`). Without a connectivity matrix, the algorithm doesn't scale well to large datasets.
- *Balanced Iterative Reducing and Clustering using Hierarchy* (BIRCH) is an algorithm designed specifically for very large datasets, and it can be faster than batch k-means, with similar results, as long as the number of features is not large (<20). During training, it builds a tree structure containing just enough information to quick ly assign new instance to a cluster, without having store all instances in the tree. This approach allows it to use limited memory while handling huge datasets.
- *Mean-shift*: Works by first placing a circle centered at each instance; then for each circle, it computes the mean of all the instances located within it, and it shifts the circle such that the circle is centered on the mean. The algorithm then repeats until all the circle stops moving (i.e. until each of them is centered on the mean of the instances it contains). Mean-shift shifts the circles in the direction of higher density, until each of them reached a local density maximum. Finally, all the instances whose circles have settled in the same place (or close enough) get assigned to the same cluster. Mean-shift has some of the features of DBSCAN, like how it can find any number of clusters of any shape, it has very few hyperparameters (just one - the radius of the circle, called the *bandwidth*), and it relies on local density estimation. But different from DBSCAN, mean-shift tends to chop off clusters into pieces when they have internal density variations. Unfortunately, its computational complexity is $O(m^2n)$, makes it not suited for large datasets.
- *Affinity propagation*: In this algorithm, instances repeatedly exchange messages between one another until every instance has selected another instance (or itself) to represent it. These elected instances are called *exemplars*. Each exemplar  and all the instances that elected it form a cluster. In real-life politics, you typically want to vote for a candidate whose opinions are similar to yours, but you also want them to win the election, so you might choose a candidate you don't full agree with, but is more popular. You typically measures popularity by the number of polls. Affinity propagation typically works in a similar way, and it tends to choose examples located near the center of clusters, similar to k-means. but unlike k-means, you don't have to set the number of cluster prior to training, it is determined during training instead. Moreover, affinity propagation can deal nicely with clusters of different sizes. Sadly, its computation complexity is $O(m^2)$, so it doesn't suit for large datasets.
- *Spectral clustering*: This algorithm takes a similarity matrix between the instances and creates a low-dimensional embedding from it (e.g. it reduces the matrix's dimensionality), then it uses another clustering algorithm in this low-dimensional space (Scikit-learn's implementation uses k-means). Spectral clustering can capture complex cluster structures, and it can slo be used to cut graphs (e.g., to identify clusters of friends on a social network). It does not scale well with the number of instances, and it does not behave well when the cluster have ver different sizes.

# Gaussian Mixtures

- A *Gaussian Mixture Model* (GMM) can be used for density estimation, anomaly detection and clustering.
- It is a probabilistic model that assumes the instances were generated from a mixture of Gaussian distribution whose parameters are unknown. All the instances generated from a single Gaussian distribution form a cluster that looks like an ellipsoid. Each cluster can have a different ellipsoidal shape, size, density, and orientation.
- However, when you observe an instance, you know it was generated from one of the Gaussian distribution, but you have no idea which one, and what the parameters of these distributions are.
- There are several GMM variants. In the simplest variant, which is implemented in the `GaussianMixture` class, You must know in advance the number of Gaussian distributions $k$.
- The dataset then is assumed tp have been generated through the following probabilistic process:
    - For each instance, a cluster is chosen randomly from among $k$ clusters. The probability of choosing the $j^{th}$ cluster is the cluster weight $\phi^{(j)}$. The index of the cluster chosen for the $i^{th}$ instance in noted $z^{i}$.
    - If the $i^{th}$ instance is assigned to the $j^{th}$ cluster (i.e., $z^{i}=j$), then the location $x^{i}$ of this instance is sampled randomly from the Gaussian distribution with mean $\mu^{j}$ and covariance matrix $\Sigma^{(j)}$. This is noted as $x^{(i)} ~\sim \mathcal{N}\left(\mu^{(j)}, \Sigma^{(j)}\right)$.
- So what can you do with such model? Given the dataset $\textbf{X}$, you typically want to start by estimating the weight $\phi$ and all the distributions parameters $\mu^{(1)}$ to $\mu^{(k)}$ and $\Sigma^{(1)}$ to $\Sigma^{(k)}$.
- We have implemented a Gaussian Mixture model in the learning notebook. This model is pretty good:
    - We created 3 clusters: The first two contain 500 instances each, while the third one only has 250 instances. So the true cluster weights are 0.4, 0.4 and 0.2, the order can be switched, and that is roughly what the algorithm found.
    - Similarly, the true means and covariance is quite close to those found by the algorithm.
- But how does this model work? It relies on the *expectation-maximization* (EM) algorithm, which has many similarities with the k-means algorithm:
    - First, it initializes the cluster parameters randomly.
    - Next, it assigns instances to clusters (this is called the *expectation* step) and updating the cluster (this is called the *maximization* step).
    - The algorithm repeats these two step until converge or reach the maximum number of iteration.
- In the context of clustering, you can think of EM as a generalization to k-means that not only find the cluster centers ($\mu^{(1)}$ to $\mu^{(k)}$), but also their sizes, shape and orientation ($\Sigma^{(1)}$ to $\Sigma^{(k)}$), as well as their relative weights $\phi^{(1)}$ to $\phi^{(k)}$.
- Unlike k-means, EM uses soft assignments, not hard assignments. For each instances, during the expectation step, the algorithm estimates the probability that it belongs to each cluster (based ot the current cluster parameters). Then during the maximization step, each cluster is updated using *all* the instances in the dataset, with each instances weighted by the estimated probability that it belongs to that cluster.
- These probabilities are called the *responsibilities* of the clusters for the instances.
- During the maximization step, each cluster's update will mostly be impacted by the instances it is most responsible for.
- Unfortunately, just like k-means, EM can end up converging to poor solutions, so it needs to be run several times, keeping only the best solution. This is why we set `n_init=10`. Be mindful: `n_init` is set to 1 by default.
- After having an estimation of the location, size, shape, orientation and relative weight of each cluster, the model can easily assign each instance to the most likely cluster (hard clustering) or estimate the probability that it belongs to a particular cluster (soft clustering).
- Gaussian mixture model is a *generative* model, meaning you can sample new instances from it (note that they are ordered by the cluster index).
- We can also estimate the density of the model at any given location. This is achieved using the `score_samples()` method: for each instance it is given, this method estimates the log of the *probability density function* (PDF) at that location. The greater the score, the higher the density.
- If you compute the exponential of these scores, you obtain the value of PDF at the location of the given instances. These are not probabilities, but probabilities densities: they can take on any positive value, not just a value between 0 and 1. To estimate the probability that an instance will fall within a particular region, you would hve to take the integral the PDF over that region (if you do so over the entire space of possible instances locations, the result will be 1).
- In the learning notebook, you can see a pretty good solution found by the Gaussian mixture model.
- This task won't be that easy in real-life, because data is not always Gaussian and low-dimensional (our dataset is 2D). We also gave the algorithm the right number of clusters.
- When there are too many dimensions, or too many clusters, or too few instances, EM can struggle to converge to the optimal solution. 
- You may need to reduce the difficulty of the task by limiting the number of parameters that the model has to learn.
- One way to do this is to limit the range of shapes and orientations that the clusters can have. This can be achieved by imposing constraints on the covariance matrices.
- The computational complexity of training a `GaussianMixture` model depends on the number of instances *m*, the number of dimensions *n*, and the constraints on the covariance matrices:
    - If `covariance_type` is `"spherical"` or `"diag"`, it is $O(kmn)$, assuming the dataset has a clustering structure.
    - If `covariance_type` is `"spherical"` or `"diag"`, it is $O(kmn^2 + kn^3)$, so it wont't scale to large numbers of features.

## Using Gaussian Mixture for Anomaly Detection

- Using Gaussian mixture model for anomaly detection is quite simple: any instances located in a low-density region can be considered an anomaly. You just need to define the threshold you want to use.
- For example, for a manufacturing that tries to detect defect products, the ratio of defective products is usually well known. Say it is 1%. You can then set the density threshold to be the value that results in having 1% of the instances located in ares below that threshold density.
- If you get too many false positive, you can lower the threshold. If you get too many false negative, you can increase the threshold. This is the precision/recall trade-off we mentioned in chapter 3.
- A closely related task is *novelty detection*: it differs from anomaly detection in that the algorithm is assumed to be trained on a "clean" dataset, uncontaminated by outliers, whereas anomaly detection does not make this assumption. In fact, outlier detection is often used to clean up a dataset.
- Gaussian mixture models will try to fit all the data, including the outliers; so if you have to many of them will bias the model viewpoint about "normality", and some outliers may wrongly be considered as normal.
- If this happen, you can try to fit the model once, use it to detect and remove the most extreme outliers, then refit the model on the cleaned-up dataset. Another approach is to use robust covariance estimation methods (see the `EllipticEnvelope` class).

## Selecting the Number of Clusters

- Just like k-means, the `GaussianMixture` algorithm needs to now the number of clusters prior to training.
- With k-means, you can use inertia or the silhouette score to find the optimal number of clusters. However, they are not suitable for Gaussian mixture, as they are not reliable when the clusters are not spherical or have different sizes.
- You should try to find the model that minimizes a *theoretical information criterion*, such as the *Bayesian information criterion* (BIC) or the *Akaike information criterion* (AIC):
    $$BIC = \log(m)p-2\log(\hat{\mathcal{L}})$$
    $$AIC = 2p-2\log(\hat{\mathcal{L}})$$
    where:
    - *m* is the number of instances, as always.
    - *p* is the number of parameters learned by the model.
    - $\hat{L}$ is the maximum value of the *likelihood function* of the model.
- Both BIC and AIC penalize models that have more parameters to learn (e.g. more clusters) and rewards models that fit the data well. They often end up selecting the same model. However, when they differ, the model selected by BIC tends to be simpler (fewer parameters) than the one selected by AIC, but tends to not fit the data quite as well (this is especially true for larger dataset).

### Likelihood Function

- The terms "probability" and "likelihood" are often used interchangeably in everyday languages, bu they have very different meanings in statistics.
- Given a statistic model with some parameters $\theta$, "probability" is used to describe how plausible  a future outcome $\textbf{x}$ is (knowing the parameter values $\theta$), while the word "likelihood" is used to describe how plausible a particular set of parameter values $\theta$ is, after the outcome $\textbf{x}$ is known.
- In other words, (which may be too much simplistic) we can think of likelihood is when we try to estimate the past, and probability is when we try to guess the future.
- Consider a 1D mixture model of two Gaussian distributions centered at -4 and 1.
- For simplicity, this toy model has a single parameter $\theta$ that controls the standard deviations of both distributions.
- The top left contour plot shows the entire $f(x, \theta)$ as a function of both $x$ and $\theta$.
- To estimate the probability distribution of a future outcome $x$, you need to set the model parameter $\theta$.
- For example, if you set $\theta = 1.3$ (the black horizontal line), you get the probability density function (PDF) $f(x; \theta=1.3)$ shown in the lower-left plot.
- Say you want to estimate the probability that $x$ will fall between -2 and 2. Then you must calculate the integral of the PDF on this range (i.e., the surface of the shaded region).
- But what if you observed a single instance $x=2.5$ (the blue vertical line in the upper-left plot)? In this case, you need to get the likelihood function $\mathcal{L}(\theta|x=2.5)= f(x=2.5|\theta)$, described in the upper-right plot.
- In short, the PDF is a function of $x$ (with $\theta$ fixed), while the likelihood function is a function of $\theta$ (with $x$ fixed).
- It's worth noticed that the likelihood function is not a probability distribution: if you integrate a possibility distribution over all possible values of $x$, you always get 1, but if you integrate the likelihood function over all possible values of $\theta$, the result can be any positive value.
- Given a dataset X, a common task is to estimate the most likely value for the model parameters.
- To do this, you must find the value that maximizes the likelihood function, given X. In the given example, if you have observed a single instance $x=2.5$, the *maximum likelihood estimate* (MLE) of $\theta$ is $\hat{\theta} \approx 1.66$.
- If a prior probability distribution $g$ over $\theta$, we cna take it into account by maximizing $\mathcal{L}\{\theta|x\}g(\theta)$ instead of $\mathcal{L}\{\theta|x\}$.
- This is called *maximum a-posteriori* (MAP) estimation. Since MAP constrains the parameter values, you can think of it as a regularized version of MLE.
- Note that optimizing the likelihood function is equivalent to optimizing its logarithm (plotted in the lower-left plot).
- It turns out that it is generally easier to work with the log of the likelihood function. For example, if you observed several independent instances $x^{(1)}$ to $x^{(m)}$, then you need to find the value of $\theta$ that maximizes the product of individual likelihoods functions.
- But we can do that in a simpler way, to maximize the sum (not the product) of the log likelihood functions, thanks to the property $\log(ab) = \log(a)+\log(b)$ of the log function.
- Once you have estimated $\hat{\theta}$, which is the value of $\theta$ that maximizes the likelihood function, then you can compute $\hat{\mathcal{L}} = \mathcal{L}(\hat{\theta}, X)$, which in turn will be used to calculate the AIC and BIC. You can think of $\hat{\mathcal{L}}$ as a measure of how well the model fits the dataset.
