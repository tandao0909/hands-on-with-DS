1. Here are some main tasks that an autoencoder can be used for:
- Unsupervised pretraining
- Dimensionality reduction, possibly data visualization (but not recommended)
- Generative models
- Feature extraction
- Anomaly detection (autoencoder will not be great at reconstructing anomalies).
2. 
- The encoder in the autoencoder can be a good feature detector, which can be reused in the lower layers of the classifier.
- I can self-supervised the autoencoder, feeding the training data (both labeled and unlabeled) both as inputs and outputs, then reuse the generator in the lower layer, stack some more layer on top of it (this is up to you), and use the labeled training data to fine-tune the classifier. If we really don't have many labeled training data, then consider freezing the reused layers.
3. 
- If the autoencoder perfectly reconstructs the inputs, that's definitely overfitting. And of course, the model is not good. It's probably the case that the generator is way to powerful, that it just arbitrarily map each training instances to a single number, and the decoder learn the reverse mapping. This mapping is not useful at all: we want the autoencoder to find pattern, not remember everything.
- We can evaluate the performance of the autoencoder by feeding an image, and measuring the reconstruction loss (which is the loss between the original image and the reconstructed image), possibly using MSE, MAE. A model with a very low reconstruction loss probably is overfitting, but the one with a vert high reconstruction loss is also not good.
- Another way to measure the performance of an autoencoder is using the task you need it for. For example, if you use t and pretrained model for another classifier, you can also evaluate the classifier's performance.
4. 
- An undercomplete model is a model with the coding layer smaller than the input. An overcomplete model is the opposite: a model with the coding layer larger than than the input.
- The main risk of of an excessively undercomplete autoencoder is it's too weak to even be able to reconstruct the inputs properly.
- The main risk of an overcomplete autoencoder is it can be way too powerful that it can just maps the inputs to the outputs, without learning anything useful.
5.
- A stacked autoencoder can be thought as a symmetrical neural network, where the symmetrical center is the coding layer. We then make the weight of one layer in the decoder equals to its symmetrical counterpart in the encoder, transposed.
- Tying weights allows us to use only half the amount of weights, making converge faster with less training data, and reducing the risk of overfitting.
6.
- A generative model is a model capable of generating new data similar (in a semantical meaning) to training data.
- For example, one trained on the Fashion MNIST model can output a grayscale image resembles the fashion items in the Fashion MNIST dataset. Some generative models can be parametrized - for example, to generate some kind of inputs. 
- A type of generative autoencoder is the variational autoencoder.
7.
- GAN, short for Generative Adversarial Networks, is composed of two neural network. These two compete each other: an generator try to generate fake image to fool the discriminator, while the discriminator tries not to be fooled. The discriminator is a binary classifier, and the goal of the generator is maximize the discriminator's error.
- Some applications for GANs: The generator can be used as a generative model, specifically for advanced image processing tasks, such as super resolution, filling holes, image extending, predicting the next frames of a video, replacing the background of an image, turning a simple sketch to a realistic image. The discriminator can used to spot weaknesses in other models and strengthen them. Together, they can bes used to augment a dataset (to train other models) and to generate other types of data.
8. Training GANs is difficult, since the complex dynamic between the generator and the discriminator:
- Mode collapse: we can easily stuck in a state where the discriminator is way easier to be fooled in a specific class by the generator. Then the generator, followed by the discriminator, will start to only focus in that class. This is bad, since we want the model to good at every class, not only one of them.
- Unstable training: the training process can start out stable, then suddenly oscillating or diverge, without any transparent reason.
- Sensitive to initial arguments: Slightly different initial parameters may lead to wildly different training process.
9.
- Diffusion models are good at generating diverse and high-quality images. It works by removing a bit of noise at every step, so if we feed it a completely full of noise image, we can use it to remove noise slowly, and have a great generated image at the end.
- Their main limitation is the long inference time: Since we must do one step at a time, the process is very long. Additional, this happens on my machine, diffusion model takes much longer to train (3 times) compared to GAN for each epoch.