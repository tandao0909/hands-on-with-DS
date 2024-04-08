Autoencoders are auto-encoders, which means they learn about the internal structure of the dataset, without the label. More technically, they can learn about the dense representations if the dataset, which is called *latent representations* or *codings*, without any supervision. These codings typically have much lower dimensionality than the original one (in some sense, they are similar to embeddings). Therefore, they are useful for dimensionality reduction, especially for data visualization tasks (see chapter 8). They can act as *feature detectors*, and can be used in unsupervised pretraining step in deep neural networks (see chapter 11). Finally, some autoencoders are generative models, which means we can ask them to generate new data that are similar to the training data. For example, you can train an autoencoder on a dataset of faces, and it would generate some very convincing faces for you.

*Generate adversarial networks* (GANs) are also deep neural networks capable of generating new data. In fact, they can create pictures of faces so convincing that it's hard to tell the person is actually not real. You can checkout yourself at https://thispersondoesnotexist.com, a website that generate fake faces using a GAN architecture named StyleGAN. You can also checkout the list of non-existent Airbnb listings at https://thisrentaldoesnotexist.com. GANs are now widely used in super resolution (increasing the resolution of an image), colorization, image editing (e.g., replace backgrounds of an image), augmenting data (for training another model), predicting the next frames in a video, generating other type of data (mostly sequence, such as text, video, audio, time series), identifying the weaknesses of other models.

Another competitor has joined recently is *diffusion models*. In 2021, they managed to generate more diverse and higher-quality than GANs, while being much easier to train and has become the norm. However, diffusion models are much slower to run.

Autoencoders, GANs, and diffusions models are all unsupervised models, capable of learning latent representations, and can be used to generate new data. However, the way they work are very different:
- Autoencoders: They simply learn to copy its input to its output. However, it's not so trivial that the model can just pass the data all the way to the last layer. As you will see, we will add lots of constraints to the model in training. For example, we can add some noises and train it to recover the original input, or limit the size of the latent representations. These constraints restrict the model from trivially copy the input to the output, force them to represent data efficiently. In short, the codings are byproduct of the autoencoder trying to learn the identity function under some constraints.
- GANs are composed of two deep neural networks: One is a *discriminator*, trying to find which data is counterfeit data, while the other is a *generator*, trying to create more and more convincing fake data. This architecture is very original in deep learning in that the generator are the discriminator compete between each other: You can think of the generator is the criminal trying to create fake money, while the discriminator is the police trying to tell which money is fake. *Adversarial learning* (training competing neural networks) is considered one of the most important innovations in 2010s. In fact, Yann LeCun even said that it was "the most interesting idea in the last 10 years in machine learning" in 2016.
- A *denoising diffusion probabilistic model* (DDPM): is trained to remove a tiny bit of noise form an image. If you feed it an image full of Gaussian noise and repeatedly run the diffusion model on that image, a high-quality image will slowly emerge, similar to the training images, but not identical.


# Efficient Data Representations

Think of how we memoize a complex thing: Do you remember every detail of it? Or trying to find a pattern, and just remember that pattern? If you have a great memory, you can choose the first option, but the fact that the thing you try to memorize so complex will eventually force you to choose the second one. This fact also pushes the autoencoder to find and exploit the pattern in training data, which hopefully can be generalize to new data as well.

The relationship between memory, perception and pattern matching was famously studied by William Chase and Herbert Simon in the early 1970s. They observed that expert chess player can remember the position of most chess pieces in just 5 seconds, which is impossible for most people. However, this is only the case if the pieces were placed in a realistic positions (in an actual game), not randomly. Chess experts do not have better memory than you and me, they just see chess patterns more easily, thanks to their experience with the game. Noticing patterns help them to store data efficiently.

Just like the chess players in the memory experiment, an autoencoder look at the training dataset, convert them to an efficient latent representation, and when asked, spits out something that (hopefully) similar to the training set. An autoencoder is always composed of two parts: an *encoder* (or *recognition network*) to convert to the latent representations, and a *decoder* (or *generation network*) to revert to the original size.
![The chess memory experiment (left) and a simple autoencoder (right)](image.png)

As you can see, an autoencoder typically an architecture of an MLP (discussed in chapter 10), expect the number of output neuron must be equal to the number of inputs. The outputs are called the *reconstruction*, as the autoencoder try to reconstruct the inputs. The cost function contains a *reconstruction loss*, indicates how many information was lost in the model.

Because the internal representation has a lower dimensionality than the input data (2D compared to 3D), the autoencoder is said to be *undercomplete*. An undercomplete autoencoder cannot trivially copy its inputs to its encodings, yet it must find a way to output a copy of its inputs. Therefore, the model is forced to find patterns (of features) in the input data, keep in mind the most important one, and drop the rest.

# Performing PCA with an Undercomplete Linear Autoencoder

If you constraints the autoencoder to only use linear activations (no activation function) and use the mean square error (MSE), then the encoder will try to find the most informative linear combination of feature, or in other word, perform principal analysis (PCA, see chapter 8).

The code in the learning notebook creates a very simple autoencoder to perform PCA on a 3D dataset, projecting it to 2D space. You can notice some key things:
- The encoder and decoder are all simple `Sequential` models, as we did in previous chapters. We just need to stack two of them together to obtain the autoencoder.
- The number of units in the last layer of the decoder must be equal to the number of inputs (3 in this case).
- To perform PCA, we do not use any activation function and use the mean square error loss function. This is because PCA, at its heart, is a linear transformation.

Note when we train the model, we use `X_train` both as the input and the target. As you can see in the learning notebook, if you ask the encoder to predict the `X_train` (i.e., ask it to show its codings), the encoder will show a projection of `X_train` into a 2D space that preserves as much information as possible, which is variance in this case. Hence in the end, we will approximate PCA.

You can think of this model, and autoencoders in general, as a self-supervised learning model. This is because the labels are created automatically, simply equal to the target in the autoencoder's case.

# Stacked Autoencoders

Just like any other neural network architecture, we can make autoencoders more powerful simply by stacking several of them together, create a deep neural network. However, we can't simply stack one autoencoder on another, but stack the encoders and decoders separately instead.

That said, you need to be careful, as making the autoencoder too deep make it too powerful, which means the encoder can simply map each data point to an arbitrary number, and the decoder just learn the reserve mapping, without the model having to learn any meaningful codings. This will make the model works perfectly with the training set, but not able to generalize well.

The architecture is typically symmetrical with regard to the central hidden layer, which is the coding layer. You can think of fit as a sandwich. For example, the architecture below is trained on the MNIST dataset, starts with the encoder of 784 units, then 100 units, then the central coding layer of 30 units, then we reserve the order in the decoder with 100 units and 784 units in the output layer:
![Stacked autoencoder](image-1.png)

## Implementing a Stacked Autoencoder Using Keras

You can implement a deep stacked autoencoder similarly to a regular MLP. You can find the implementation in the learning notebook. There's nothing special in this model compared to the previous one: you need to make sure the number of output neurons is the same as the number of inputs, and feeding the same dataset as both the inputs and the targets. Just remember the symmetry, and you're good to go.

## Visualizing the Reconstructions

One way to check if the outputs are similar to the inputs is just draw them out and judge them by our eyes. You can look at the learning notebook for an example.

The reconstruction is a bit lossy, but recognizable. If you wish to make them better, you can try making the autoencoder deeper, making the codings larger, or training it for longer. However, if you make the model too powerful, the model can end up not just remember the training set, without learning any useful representations. For now, we'll stick with this model.

## Visualizing the Fashion MNIST Dataset

For visualization, stacked autoencoders don't yield good result compared to other dimensionality reduction algorithms discussed in chapter 8, but one big advantage of it is the ability to handle big datasets with many instances and many features. Therefore, a common pipeline for data visualization is using autoencoders to reduce the dataset to a reasonable dimension, and use better (but slower) algorithms to reduce it to 2D or 3D.

The learning notebook shows us one example: I used the trained autoencoder to reduce the MNIST dataset to 30 dimension, then use t-SNE to reduce to 2D. 

So, autoencoders can be used for dimensionality reduction. Let's see how it deal with unsupervised pretraining.

## Unsupervised Pretraining Using Stacked Autoencoders

As we discussed in chapter 11, one way to tackling complex task with little training data is finding a pretrained network with similar task to yours and reuse the lower layers. Your network will take advantage of the feature detector and have a good performance, albeit having little training data.

Similarly, if you don't have many labeled data at hand, but lots of unlabeled one, you can train an autoencoder on the unlabeled data and reuse its encoder. The idea is we can reuse the encoder's latent representation as a feature detector, so if we build a new neural network using these lower layers, we can achieve a high performance score, with little training data.

When training the network, which reuses part of the autoencoder, if you really don't have much labeled training data, you should freeze the reused layers (at least the lower ones).

There's nothing special about the implementation. You train the stacked autoencoder on all the data (including the unlabeled and labeled ones), reuse the encoder's layers, add a few more layers on top based on your tasks, and trained this model on the labeled training data.

Having plenty of unlabeled data but little labeled data is common: a simple script can crawl millions of images from the internet, but having people manually labeled is time-consuming and expensive. You usually end up having millions of unlabeled data, but just thousands, or even just hundreds, of labeled data.

## Tying Weights

When autoencoder is neatly symmetrical, you may think: Hey, why don't we just reuse the weights? The matrices have same shape, right? Well, that's actually a great idea! Doing so would halve the number of weights, result in smaller model, faster training time, and reducing the risk of overfitting.

If the autoencoder has N layers (we don't count the input layer), then N is even. Layer 1 is the first hidden layer, layer N/2 is the coding layer, and layer N is the output layer. If we called $\textbf{W}_i$ is the weight of the i-th layer, then what we want to do is $W_{i} = W_{N-i}$, where $L = 1, \dots, N/2 - 1$.

To tie weights between layers using Keras, you must define a custom layer. This custom layer acts as a regular `Dense` layer, but has the weights tied to another `Dense` layer, transposed (using `transpose_b` is equivalent to transposing the second argument, but more efficient). It still has its own bias vector, though.

We build the model as usual, just need to tie the layers in the decoder to the appropriate layer in the encoder. This model had a loss score roughly as good as the previous model, but using only half amount of parameters.

## Training One Autoencoder at a Time

Rather than training the whole stack of autoencoder as we did, we can instead train a (simple) separate autoencoder individually, and insert them together to have a deep stack. It's easier to see it visually:
![Training one autoencoder at a time](image-2.png)

This technique is not used as much these days, but there's still papers that talk about "greedy layerwise training", so it's good to know what it means.

During the first phase of training, we train the outermost layers as an autoencoder, which consists of the input layer, output layer and the first hidden layer. After that, we use the encoder to create the encoded training set, and use it to train a second autoencoder. That's the second phase. Finally, we build a big sandwich using all these autoencoders, by stacking the encoders in that order, and then the decoders in the reverse order. This gives us the final stacked autoencoder, see the learning notebook for an implementation. We could easily train more autoencoders this way, result in a deeper autoencoder.

As mentioned earlier, one of the triggers of the tsunami of deep learning was the discovery in 2006 by [Geoffrey Hinton et al.](https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf) that deep neural network can be pretrained in a unsupervised manner, using this greedy layerwise approach. They used restricted Boltzmann machines (RBMs, see https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) for this task, but later in 2007, [Yoshua Bengio et al.](https://proceedings.neurips.cc/paper_files/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf) showed that autoencoders work just as well. For several years, this was the only way to train a deep nets, until many of the techniques introduced in chapter 11 made it possible to train a deep net in one shot.

# Convolutional Autoencoders

If you work with images, you'll find that the autoencoders we've dealt so far will not work so great (unless the images are very small). As yu saw in chapter 14, convolutional neural networks are better suited than dense network for image tasks. So if you want to build an autoencoder for images, you should build an [convolutional autoencoder](https://people.idsia.ch/~ciresan/data/icann2011.pdf) instead.

The encoder is a regular CNN with convolutional and pooling layers. Its job is to reduce the spatial dimension (height and width), while increasing the depth (number of feature maps). The decoder must do the reserve, increase the spatial dimension back to the original size, while reduce the depth, and for this, you can use transpose convolutional layers (which are equivalent pairs of upsampling and convolutional layers)

It's also possible to create autoencoders with other architecture types, such as RNNs (see the learning notebook for an example)

Up until now, we have looked at various kinds of autoencoders (basic, stacked, convolutional, sequence-to-sequence) and how to train them (either in one shot or layer-by-layer). We also saw some applications, such as data visualization and unsupervised pretraining.

In order ot force the autoencoder to learn interesting features, we have limited the size of the coding layer, making it undercomplete. There are actually many other different kinds of constraints that can be used, including one that allow the coding layer just be as large as the inputs, or even larger, result in an *overcomplete autoencoder*. We'll look at a few more kinds of overcomplete autoencoders: denoising autoencoders, sparse autoencoders, and variational autoencoders.

# Denoising Autoencoders

Another way to force the autoencoder to learn useful feature is add some random noises to the inputs, training it to recover the original, noise-free data. This idea has been around since the 1980s (e.g., it is mentioned in Yann LeCun's 1987 master thesis). In a [2008 paper](), Pascal Vincent et al. showed that autoencoders could also be used for feature extraction. In a [2010 paper](), Vincent et al. introduced *stacked denoising autoencoders*.

The noise can be pure Gaussian noises added to the inputs, or it can be randomly switched-off inputs, just like in dropout (introduced in chapter 11).
![Denoising autoencoders, with Gaussian noise (left) or dropout (right)](image-3.png)

The implementation is straightforward: it's a regular stacked autoencoder with an additional `Dropout` layer applied to teh encoder's inputs (or you could use a `GaussianNoise` layer instead). Note that the `Dropout` layer (and the `GaussianNoise` layer) only activate during training.

You can look at the learning notebook to see precisely what the model saw during training: half of the pixels are turned off. You can also see their corresponding predictions. Notice how the autoencoder makes up details that are not actually appear in the original image, such as the top of the shirt (in the fourth image, bottom row). As you can see, not only we can use autoencoders for data visualization and unsupervised pretraining, we can also use them to remove noises from the inputs, in a simple and efficient manner.

# Sparse Autoencoders

Another kind of constraints that often leads to good feature extraction is *sparsity*: by adding an appropriate term to the cost function, we can force the autoencoder to only use a limited number of active neurons in the coding layer, for each instance. For example, it may be pushed to only have about 5% of neurons are significantly activate. This makes the model to represent each input as a combination of a small number of activations. As a result, each neuron in the coding layer typically ends up presenting a useful feature (if you only say a few words a month, you will probably try to make them worth listening to).

A simple approach is using the sigmoid activation function in the coding layer (to constraint the codings to value between 0 and 1), using a large coding layer (300 neurons in example), and adding an $\ell_1$ regularization term to the coding layer's activation.

This can be achieved using an `ActivityRegularization` layer, which just returns its inputs, but as a side effect also adds to the training loss the sum of the absolute values of it inputs times the weights (which is `1e-4` in our example). This only affects during training. Equivalently, you could remove the `ActivityRegularization` layer and set `activity_regularization=tf.keras.regularizers.l1(1e-4)` in the previous layer. This regularizer will encourage the coding layer to be close to 0, but since it will also be penalized if does not reconstruct the inputs properly, it needs to keep some nonzero neurons. Using the $\ell_1$ instead of $\ell_2$ help the coding layer to push all the most unimportant neurons close to 0, instead of pushing all of the neurons equally.

Another approach, which often yields better result, is to measure the actual sparsity of the coding layer at each training iteration, and penalize the model if it differs from the desired sparsity. We do so by taking average activation of each neuron in the coding layer, over a given training batch. Te batch size must not be too small, or else the mean will not be descriptive.

Once we have mean activation of each neuron, we want to penalize the neurons that are too active, or not active enough, by adding a sparsity loss to the cost function. For example, a neuron has an average activation of 0.3, but the target sparsity is 0.1, then it must be penalized to be less active. One approach could be taking the mean squared error, which is $(0.3 - 0.1)^2 = 0.04$, to the cost function. That's good, but there's a better way: using the Kullback-Leibler (KL) divergence (briefly discussed in chapter 4), which has a stronger gradients compared to the MSE.

Given two *discrete* probability distribution *P* and *Q*, the KL divergence between these distributions, noted $D_{KL}(P||Q) $ can be computed as:
    $$D_{KL}(P||Q) = \sum_i P(i) \log\left(\frac{P(i)}{Q(i)} \right) $$
    where i is the number of possible outcome of distribution *P*.

In our case, there are only out possible outcome, active or not, so if we assume the target possibility of a neuron to be active is *p*, while the actual possibility is *q*, estimated by the mean activation over the training batch, then KL divergence can be simplified as:
    $$D_{KL}(p||q) = p \log\left(\frac{p}{q} \right) + (1-p) \log\left(\frac{1-p}{1-q} \right)$$
    Looks very similar to binary cross entropy, don't you think so?

Once we calculate the equation above for each neuron, we can sum the results up, and add it to the cost function. In order to control the relative importance of the sparsity loss and the reconstruction loss, you can optionally multiple the sparsity loss by a sparsity weight hyperparameter. If this number is too high, the neurons' sparsity will be close to your desire, but the reconstruction would be terrible. If this number is too low, the model will ignore the sparsity objective, which means it wouldn't learn any interesting features.

Enough with the theory! We need to build a custom regularizer to apply KL divergence regularization, and use it for the `activity_regularizer` argument in the coding layer. See the implementation in the learning notebook.

# Variational Autoencoders

An important category of autoencoders was introduced in 2013 by [Diederik Kingma and Max Welling](https://arxiv.org/abs/1312.6114) and quickly became the one of the most popular variants: *variational autoencoders* (VAEs).

VAEs has some key differences compare to other variants we discussed so far:
- They are *probabilistic autoencoders*, meaning that their outputs are partly determined by chance, even after training (as opposed to denoising autoencoders, which use randomness only during training).
- Most importantly, they are *generative models*, which means they can created new instances that look like instances in the training set.

Both these properties make them rather similar to RBMs, but they are easier to train, and the sampling process is much faster (with RBMs, you need to wait until the network stabilize to a "thermal equilibrium" before you can sample a new instance). As the name suggests, variational autoencoders rely on variational Bayesian inference, which is an efficient of carrying out approximate Bayesian inference. Recall that Bayesian inference means updating a probability distribution based on new data, using equation derived from Bayes' theorem. The original distribution is called the *prior*, while the updated distribution is called the *posterior*. In our case, we want to find a good approximation of the data distribution. Once we have that, we can sample from it.

Let's see how VAEs work. Its structure is similar to an autoencoder: consists of an encoder and a decoder. The difference lies on the coding layer: instead of simply output a coding for a given input, the encoder produces a *mean coding* $\mu$ and a standard deviation $\sigma$. The actual coding is then sampled randomly is then sampled randomly from a Gaussian distribution with mean $\mu$ and standard deviation $\sigma$. After that, the decoder decode the sampling normally, given an output (hopefully) resembles the training instances.
![A variational autoencoder (left) and an instance going through it (right)](image-4.png)

You can think of the coding is a set of instances, each of them are sampled from a Gaussian distribution. The job of the last hidden layer in the encoder is to output 2 vectors: a vector consists of the means, and a vector consists of the standard deviations of these distributions.

As we can see from the diagram, even though the inputs may have a very convoluted (i.e., complex) distribution, a variational autoencoder tends to produce codings that look as they were sampled from a simple Gaussian distribution (variational autoencoders are more general, they are not limited to Gaussian distribution): during training, the cost function (discussed later) will gradually push the codings within the coding space (also called the *latent space*) to end up looking like a cloud of Gaussian points. One great consequence is after training, you can sample a random coding from the Gaussian distribution, pass it through the decoder, and there you have a new instance.

Now, let's look at the cost function. It's composed of two parts. The first part is the usual reconstruction loss we have used until now to push the autoencoder to reproduce it inputs. We can calculate it using MSE, as we did earlier. The second is the *latent loss* that pushes the autoencoder to have codings that look like they are sampled from the Gaussian distribution: it is the KL divergence between the target distribution (e.g., the Gaussian distribution) and the actual distribution.

The math is a bit more complex than the sparse autoencoder, in particular because of the Gaussian noise, which limit the amount of information that can be transmitted to the coding layer. This pushes the autoencoder to learn useful features. Luckily, the equation simplifies a lot, so the latent loss can be computed using:
    $$\textbf{L} = -\frac{1}{2}\sum_{i=1}^n\left(1 + \log(\sigma_i^2) - \sigma_i^2 - \mu_i^2 \right)$$
where $\textbf{L}$ is the latent loss, $n$ is the codings' dimensionality, and $\mu_i$ and $\sigma_i$ are the mean and the standard deviation of the i-th component of the codings. The vectors $\mu$ and $\sigma$ are created by the encoder, as shown in the left part of the figure above.

A common tweak to the variational autoencoders' architecture is to let the encoder output $\gamma = \log(\sigma^2) $ rathe than $\sigma$. The latent loss then be computed as:
    $$\textbf{L} = \frac{1}{2} \sum_{i=1}^n \left(1 + \gamma_i -\exp(\gamma_i) -\mu_i^2 \right) $$
This approach is more numerically stable and speeds up training.

Let's start the implementation! First we need a custom layer to sample the codings using the $\mu$ and $\gamma$. This `Sampling` layer takes two inputs: `mean` ($\mu$) and `log_var` ($\gamma$). It uses the function `tf.random.normal()` to sample a random vector (of the same shape as $\gamma$) from the Gaussian distribution, with has mean 0 and standard deviation 1. Then it multiples it by $\exp(\gamma/2)$ (which is equals to $\sigma$), pluses it with $\mu$, and return the result. This samples a codings vector from the Gaussian distribution with mean $\mu$ and standard deviation $\sigma$.

Now we can create the encoder, using the functional API, as the model is not entirely sequential. Note that the `Dense` layers that output `coding_means` ($\mu$) and `coding_log_var` ($\gamma$) have the same inputs (i.e., the outputs of the second hidden layers). We then pass both of them to the `Sampling` layer. Finally, the `variational_encoder` has three outputs. Only the `codings` are required for the decoder, but we add `coding_means`, and `coding_log_var` as well, in case we want to inspect their values.

For the decoder, we can use the sequential API instead, as it's just a stack of `Dense` layer. Here I use functional API for consistency.

Now, we can build the whole model using the encoder and decoder as we did earlier. We will ignore the first two outputs, as we just need the codings. Lastly, add the latent loss to the reconstruction loss, and we can compile and train the model.

We use the improved equation for the latent loss for each instance in the batch, summing over the last axis. Then we compute the mean loss over the whole batch, and divide the result by 784 to have the appropriate scale compared to the reconstruction loss. Indeed, the variational autoencoders' reconstruction loss is supposed to computed a sum of reconstruction loss in all pixels, but Keras computes the mean, which is the sum divided by 784 instead. So the reconstruction loss is 784 times smaller than we need it to be. We could write a custom loss to compute the sum instead of the loss, but it's way easier to just divide the latent loss by 784 (then the final loss will be 784 times smaller than it should be, but then we could just an optimizer with a larger learning rate).

# Generating Fashion MNIST Images

Now we can use this variational decoder to create new images that look like instances form the Fashion MNIST dataset. All we need to do is sample random codings from a Gaussian distribution and decode them.

The majority of the images look fairly convincing, if not a bit too fuzzy. The rest are not good, but don't be too harsh on the decoder, it has only a few minutes to learn.

Another perk of variational autoencoders is the ability to perform *sematic interpolation*: instead of interpolating between two images at the pixel level, which would look like two images were just overlaid, we can interpolate at the codings level. For example, let's take a few codings along an arbitrary line in latent space and decode them. We'll get a sequence of images that go from sweaters to boots.

# Generative Adversarial Networks

Generative Adversarial Networks were proposed in a [2014 paper](https://arxiv.org/abs/1406.2661) by Ian Goodfellow et al., and although the idea got researchers excited almost instantly, it took a few years to overcome some of the difficulties of training GANs. Like many great ides, it's simple in the hindsight: make neural networks compete against each other in the hope that this competition will push them to excel.

A GAN is composed of two parts:
- *Generator*: Takes a random distribution as input (typically Gaussian) and outputs some data - typically an image. You can think of the random inputs as the latent representations (i.e., codings) of the image to be generated from. So, we can think of the generator as the decoder in the variational autoencoder, and it can be used in the same way: feeding it some Gaussian noise, and it'll output a new image. However, it's trained differently, as you'll see shortly.
- *Discriminator*: Takes either a real image from the training set or a fake image created from the generator, and tell whether the image is real or fake.
![A generative adversarial network](image-5.png)

During training, the generator and the discriminator has opposite goals: the discriminator try to tell the real images from fake images, but the generator try to create convincing images enough to trick the discriminator. Because GAN is composed of two neural networks with different objectives, it can't be trained normally. Each training iteration is divided into two phases:
- In the first phase, we train the discriminator. A batch of real images is sampled from the training set and is completed with an equal number of fake images generated by the generator. The labels are set to 1 for real images and 0 for fake images, and the discriminator is trained in this labeled batch for one step, using the binary cross-entropy loss. Importantly, backpropagation only affects the discriminator's weights in this phase.
- In the second phase, we train the generator. We first use it to produce another batch of fake images, and once again the discriminator is used to tell whether the images are fake or real. This time, we do not add another real images to the batch, and set all the labels to 1 (real). In other words, we want the generator to produce fake images that the discriminator (wrongly) believe that the images are real! Crucially, the discriminator's weights are frozen at this step.

The generator never see any real image, yet it still able to create convincing images! All it gets is the gradients flowing back through the discriminator. Luckily, the better the discriminator is, the more information about the real images is contained in these secondhand gradients, so the generator can make huge progress.

Now we'll build a simple GAN model for the Fashion MNIST dataset. First, we need to build the discriminator and the generator. The generator is similar to an autoencoder's decoder, and the discriminator is a regular binary classifier: it takes an image and ends with a `Dense` layer with one neuron that outputs whether the image is real of fake. Of course the activation of that last layer is the sigmoid activation function. For the second phase of each training iteration, we also need the full GAN model, which composed of the generator followed by the discriminator.

Next, we need to compile the model. As the discriminator is a binary classifier, we can naturally use the binary cross-entropy loss. The GAN model is also a binary classifier, so we can use the binary cross-entropy too. However, the generator will only be trained in the GAN model, so we don't need to compile it separately. Importantly, the discriminator is not trained during the second phase, so we need to make it non-trainable before compiling the GAN model.

The `trainable` attribute is taken into account by Keras only when compiling a model, so when we run the code, the `discriminator` will be trainable in the first line, so you can call its `fit()` and `train_on_batch()` methods. However, when you call these methods on the `gan` model, the `discriminator` is now non-trainable, so it will not be affected when you fit the `gan` model.

Since the training loop is unusual, we can use the convenient `fit()` method. We need to write a custom training loop instead. Don't worry, you just need to define the dataset properly, the heavy-lifting part will be handled by the `train_on_batch()` method. For this, we need to define a Dataset that iterate through the images (also shuffle, batch and prefetch it).

We wrap the whole training process in a `train_gan()` function. As discussed earlier, we need to work with two pashes:
- In phase one, we feed the generator Gaussian noise to let it produce some fake images, and we complete the batch by concatenating an equal number of real images. We set the target `y1` to 0 for fake images and 1 for real images. Then we train the discriminator on this batch. Notice that the generator is untouched in this phase.
- In phase two, we feed the GAN some Gaussian noise. Its generator will start by producing fake images, then the discriminator will try to guess whether the images are fake or real. In this phase, we are trying to improve the generators, which means we want the discriminator to fail: this is why the target `y2` are all set to 1, although the images are fake. The gradients will tweak the generator such that it would create images the discriminator thinks are real. The discriminator is not trainable in this phase, so only the generator is improved.

If you display the images created by the generator after training, you will see that the images look like (very) lossy version of Fashion MNIST images.

Unfortunately, the images never really got much better than that, and if you look the images created during training, you can find some epochs where the GAN seems to forget what it has learned. Well, as mentioned earlier, it turns out that training a GAN can be very difficult.