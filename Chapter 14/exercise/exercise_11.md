This file is the note file for the exercise 11.

# Neural Style Transfer

- This tutorial uses deep learning to compose one image in the style of another image.
- This is known as *neural style transfer* and the technique is outlined in [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.
- Modern approaches train a model to generalize the stylized image directly (similar to [Cycle GAN]). This approach is much faster (up to 1000x).
- For a simple application of style transfer with a pretrained model from [TensorFlow Hub](), give the [Fast style transfer for arbitrary styles]() tutorial a try, which uses an [arbitrary image stylization model]().
- For an example of style transfer using [TensorFLow Lite](), refer to [Artistic style transfer with TensorFlow Lite]().
- Neural style transfer is an optimization technique used to take two images - a *content* image and a *style reference* image (such as an artwork by a famous painter) - and blend them together so that the output image looks like the content image, but painted in the style of the style reference image.
- This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image.
- These statistics are extracted from the images using a convolutional network.

## Define content and style representations

- We will use the intermediate layers of the model to get the *content* and *style* representation of the image.
- Starting from the network's input layer, the first few layer activations represent low-level features such as edges or textures.
- As we step through the network, the final few layers represents higher-level features - object parts like *wheels* or *eyes*.
- In this case, we will use the VGG19 network architecture (you can look at the note.md of this chapter for more information), a pretrained image classification network, pretrained on the ImageNet dataset.
- These intermediate layers are necessary to define the representation of content and style from the images.
- For an input image, try to match the corresponding style and content target representations at these intermediate layers.
- But why these intermediate outputs within our pretrained image classification network allow us to define style and content representations?
- At a high level, in order for a network to preform image classification (which this network has been trained to do), it must understand the image. This requires taking the raw images as input pixels and building an internal representation that converts the raw image pixels into a complex understanding of the features present within the image.
- This is also the reason why convolutional neural networks are able to generalize well: they are able to capture the invariances and defining features within classes (e.g., ducks or chickens) that are agnostic to background noises and other nuisances.
- Thus, somewhere between where the raw pixels is fed to the model and the output classification label, the model serves as a complex feature extractor.
- By accessing intermediate layers of the model, we can describe the content and style of the input images.

## Calculate style

- The content of an image is represented by the values of the intermediate feature maps.
- It turns out, the style of an image can be described by the means and correlations across different feature maps.
- Calculate a Gram matrix that includes this information by taking the outer product of the feature vector with itself at each location, and averaging that outer product over all locations.
- Suppose we flatten the feature map into a 1D vector (the naming is weird, but I just try to explain the equation provided by TensorFlow), then the value of Gram matrix can be calculated as:
    $$G^l_{cd} = \frac{ \sum_{k} F^l_{ik}(x) F^l_{ik}(x)}{L}$$
    where:
    - Here we performs an operation similar to a normal matrix multiplication.
    - $G^l_{cd}$ is the value of the element at row $c$, column $d$ in the Gram matrix of this layer.
    - $F^l_{ik}(x)$ is the activated output of the $i$ filter at position $k$ in layer $l$.
    - $L$ is the number of neurons in a feature map.
- In other words, we calculate an inner product between the vectorized feature map $i$ and $j$ in layer $l$.
- TensorFlow explains this in a more technical way (in my opinion): we don't need to vectorize the feature map, just perform an element-wise multiplication between every feature map and sum these results up. That is also the reason why we use `tf.linalg.einsum()`, to take advantage of how Einstein summation works.

## Run Gradient Descent

- With this style and content extractor, we can now implement the style transfer algorithm. We do this by calculating the mean squared error for your image's output relative to each target, then take the weighted sum of these losses.

## Total Variation Loss

- One downside to this basic implementation is that it produces a lot of high frequency artifacts.
- Decrease these using an explicit regularization term on the high frequency components of the image. In style transfer, this is often called the *total variation loss*.
- The code shows how the high frequency components have increased.
- Also, this high frequency component is basically an edge-detector. You can get similar output from the Sobel edge detector.
- The regularization loss associated with this is the sum of the squares of the values.
- Keras includes a standard implementation for us in `tf.image.total_variation()` function.