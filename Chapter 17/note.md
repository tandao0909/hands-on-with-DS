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