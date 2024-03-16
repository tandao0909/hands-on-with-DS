- Although IBM's Deep Blue supercomputer beat the chess world champion Garry Kasparov back in 1996, not until recently that computers were able to reliably perform seemingly trivial task, such as detecting a kitten in a picture or recognizing spoken words.
- The reason why these tasks are so trivial to us human is because evolution helps us a lot. Billions of years allows us to have such specialized visual, auditory, and other sensor modules in our brains
- These automatically takes place in the process of inferring the information in the real world, before it reaches out conscious.
- However, this is not trivial at all, and we must look at how our sensory modules works to understand it.
- *Convolutional neural networks* emerged from the study of the brain visual cortex, and they have been used in computer image recognition since the 1980s.
- Over the last 10 years, thanks to the increase in computational power, the amount of available training data, and the tricks represented in chapter 11 in training deep nets, CNNs have been able to achieve superhuman performance o some complex visual tasks: image search services, self-driving cars, automatic video classification systems, and more.
- Moreover, CNNs are not restricted to visual perception: they are also success at many other tasks, including voice recognition and natural language processing.
- We will only focus on visual application in this chapter.

# The Architecture of the Visual Cortex

- David H. Hubel and Torsten Wiesel performed a series of experiments on cats in [1958](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1357023/pdf/jphysiol01301-0020.pdf) and [1959](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1363130/pdf/jphysiol01298-0128.pdf) (and a [few years later on monkeys](https://physoc.onlinelibrary.wiley.com/doi/epdf/10.1113/jphysiol.1968.sp008455)), giving crucial insights into the structure of the visual cortex.
- In particular, they showed that many neurons in the visual cortex have a small *local receptive field*, meaning they react only to a visual stimuli located in a limited region of the visual filed.
- The receptive fields of different neurons may overlap, and together they tile the whole visual field.
- Moreover, the authors showed that some neurons react only to images of horizontal lines, while others mau only react to dashed lines (two neurons may have the same receptive field but react to different line orientation).
- They also noticed that some neurons have larger receptive fields, and they react to more complex patterns that are combinations of the lower-level patterns.
- These observations led to the idea that the higher-level neurons are based on the outputs of neighboring lower-level patterns.
- This powerful architecture is able to detect all sorts of complex patterns in any area of visual field.
- These studies of the visual cortex inspired the [neocognition](https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf), introduced in 1980, which gradually evolved into what we now call convolutional neural networks.
- An important milestone was a [1998 paper](https://homl.info/75) by Yann LeCun et al. that introduced the famous *LeNet-5* architecture, which became widely used by banks to recognize hand-written digits on checks.
- This architecture has some building blocks that you already know, such as fully connected layers and the sigmoid activation function, but it also introduced two new building blocks: *convolutional layers* and *pooling layers*.
- Why don't we use a deep neural networks with fully connected layers for image recognition tasks?
    - Although this works fine for small images (e.g., MNIST), it won't be great if we talk about larger images, because of the huge number of parameters it requires.
    - For example, a $100 \times 100$-pixel image has 10,000 pixels, and if the first layer has just 1,000 neurons (which already severely restricts the amount of information transmitted to the next layer), this means a total of 10 million connections. And that's just the first layer.
    - CNNs solve this problem using partially connected layers and weight sharing
- A convolutional is a mathematical operation that slides one function over another and measure the integral of their pointwise multiplication. It has deep connections with the Fourier transform and the Laplace transform and is heavily used in signal processing.
- Convolutional layers actually use cross-correlations, which are very similar to convolutions (see [Wikipedia page](https://en.wikipedia.org/wiki/Convolution) for more details).

# Convolutional Layers

- The most important building blocks of a CNN is, of course the convolutional layers: neurons in the first convolutional layer are not connected to every single pixel in the input image, but only to pixels in their receptive fields.
- In turn, each neuron in the second convolutional layer is connected only to neurons located within a small rectangle in the first layer.
- This architecture allows the network to concatenate on small low-level in the first hidden layer, then assemble them into larger higher-level features in the next hidden layer, and so on.
- This hierarchical structure is common in real-world images, which is one of the reasons why CNNs work so well for image recognition.
- All the multilayer neural networks we've looked at so far had layers composed of a long line of neurons, and we had to flatten input images to 1D before feeding them to the neural network. In a CNN, each layer is represented in 2D, which make it easier to match neurons with their corresponding inputs region.
- A neuron located in row $i$, column $j$ of a given layer is connected to the outputs of the neurons in the previous layer located in rows $i - f_h$ to $i + f_h$, columns $j - f_w$ to $j + f_w$, where $2 \times f_h + 1$ and $2 \times f_w + 1$ are the height and width of the receptive field. This is also the reason why the height and the width of the receptive field are odd.
![An image about how we project neurons](image.png)
> Note that the author use $f_h$ and $f_w$ where should be $2 \times f_h + 1$ and $2 \times f_w + 1$.
- In order for a layer to have the same height and width as the previous layer, it's common to add zeros around the inputs, as shown in the diagram. This is called *zero padding*.
- We also can connect a large input layer to a much smaller layer by spacing out the receptive fields. This greatly reduce the model's computational complexity.
- The horizontal of vertical step size from one receptive field to the next is called the *stride*.
- A neuron located in row $i$, column $j$ of a given layer is connected to the outputs of the neurons in the previous layer located in rows $i \times s_h - f_h$ to $i \times s_h + f_h$, columns $j \times s_w - f_w$ to $j \times s_w + f_w$, where $s_h$ and $s_2$ are the vertical and horizontal strides.
![An image about how we project neurons with stride](image-1.png)

## Filters

- A neuron's weights can be represented as a small image the size of the receptive field.
- For example, the figure shows two possible sets of weights, called *filters* (or *convolution kernels*, or just *kernels*):
    - The first one is represented as a black square with a vertical white line in the middle (it's a $7 \times 7$ matrix full of 0s expect for the central column, which is full of 1s); neurons using these weights will ignore everything in their receptive field except for the central vertical line (since all inputs will be multiplied by zeros, expect for the ones in the central vertical line).
    - The second filter is a black square with a horizontal white line in the middle. Neurons using these weights will ignore everything in their receptive field except for the central horizontal line.
    - Now, if all neurons in a layer use the same vertical line filter (and the same bias term), and you feed the network th input image shown below, the layer will output the top-left image. Notice how the vertical white lines get enhanced while the rest gets blurred.
    - Similarly, the upper-right image is what you get if all neurons use the same horizontal line filter; notice that the horizontal white lines get enhanced while the rest is blurred out.
![Applying two different filters to get two feature maps](image-2.png)
- Thus, a layer full of neurons using the same filter outputs a *feature map*, which highlights the areas in an image that activate the filter the most.
- You don't have to find the appropriate feature maps yourself: instead, during training the convolutional layer will automatically learn the most useful filters for its task, and the layers above will learn to combine them into more complex patterns.

## Stacking Multiple Feature Maps

- Up until now, for simplicity we have seen the output of each convolutional layer as a 2D layer, but in reality, a convolutional layer has multiple filters (you decide how many) and outputs one feature map per filter, so it's more accurately represented in 3D.
- It has one neuron per pixel in each feature map, and all neurons within a given feature share the same parameter (i.e., the same kernels and bias term). Neurons in different feature maps use different parameters.
- A neuron's receptive field is the same as described earlier, but it extends across all the features maps of the previous layer.
- In short, a convolutional layer simultaneously applies multiple trainable filters of its inputs, making it capable of detecting multiple features anywhere in its inputs.
![Two convolutional layers with multiple filters each (kernels), processing a color image with three color channels; each convolutional layer outputs one feature map per filter](image-3.png)
- The fact that all neurons in a feature map share the same parameters dramatically reduces the number fo parameters in the model. Once the CNN has learned to recognize a pattern in one location, it can recognize that pattern in any other location.
- In contrast, once a fully connected neural network has learned to recognize a pattern in one location, it can only recognize it in that particular location.
- Input images are also composed of multiple sub-layers: one per *color channel*.
- As mentioned in chapter 9, there are typically three: red, green, blue (RGB). Grayscale image have just one channel, but some images may have many more - for example, satellite images that capture extra light frequencies (such as infrared).
- Specifically, a neuron located in row $i$, column $j$ of the feature map $k$ in a given convolutional layer $l$ is connected to the outputs of the neurons in the previous layer $l-1$, located in rows $i \times s_h - f_h$ to $i \times s_h + f_h$ and columns $j \times s_w - f_w$ to $j \times s_w + f_w$, across all feature maps (in layer $l-1$).
- Note that, within a layer, all neurons located in the same row i and column j but in different feature maps are connected to the outputs of the exact same neurons in the previous layer.
- The following equation explains how to compute the output of a give neuron in a convolutional layer:
    $$z_{i,j,k}=b_k+\sum_{k'=0}^{f_{n'}-1}\sum_{u=0}^{2f_w-1}\sum_{v=0}^{2f_h-1} x_{i', j', k'} \times w_{u,v,k'k,k} \text{ with } 
    \begin{cases}
    i' = i \times s_h - f_h + u \\
    j' = j \times s_w - f_w + v \\
    \end{cases}$$
- In this equation:
    - $z_{i,j,k}$ is the output of the neuron located at row $i$, column $j$, in the feature map $k$ of the convolutional layer (layer $l$).
    - $f_{n'}$ is the number of feature in the $l-1$ layer, $2f_w-1$ and $2f_h-1$ is the height and width of the receptive field.
    - $s_h$ and $s_w$ are the vertical and the horizontal stride.
    - $x_{i', j', k'}$ is the value of the neuron at the i' row, j' column, k' feature map of the convolutional layer (layer $l-1$).
    - $w_{u, v, k', k}$ is the connection weight between any neuron in the feature map $k$ and its input located at $u$ row, $v$ column (relative to the receptive field), and the feature map $k'$.
    - $b_k$ is the bias term of the feature map $k$ in layer $l$. You can think of is as a knob tweak the overall brightness of the feature map $k$.
- The indices make it ugly, but everything this equation does is calculate the weighted sum of all inputs, plus the bias term.

## Implementing Convolutional Layers with Keras

- In the learning notebook, let's load and process a couple of sample images, using Scikit-learn's `load_sample_image()` function and Keras's `CenterCrop` and `Rescaling` layers (all of which were introduced in chapter 13).
- If you look at the shape of the `images` tensor, it's a 4D tensor: `[2, 70, 120, 3]`!
    - There are two sample images, which explains the first dimension.
    - Then each image is $70 \times 120$, since that's the size we specified when creating the `CenterCrop` layer (the original size were $427 \times 640$). This explains the second and third dimensions.
    - Lastly, each pixel holds one value per color channel, and there are three of them - red, green, and blue - which explains the last dimension.
- Now let's create a 2D convolutional layer and feed it these images to see what comes out.
- For this, Keras provides a `Convolution2D`, alias `Conv2D`. Under the hood, this layer relies on TensorFlow's `tf.nn.conv2d()` operation.
- Now we create a convolutional layer with 32 filters, each of size $7 \times 7$ (using `kernel_size=7`, which is equivalent to using `kernel_size=(7, 7)`), and apply this layer to out small batch of two images.
- When we talk about a 2D convolutional layer, "2D" refers to the number of *spatial* dimensions (height and width), but as you can see, the layer takes 4D inputs: as we saw, the two additional dimensions are the batch size (first dimension) and the channels (last dimension).
- We then look at the shape of the outputs. The outputs's shape is now `[2, 64, 114, 32]`, so it's similar to the input shapes, and there're two main differences.
- First, there are 32 channels instead of 3. This is because we set `filters=32`, so we get 32 output feature maps: instead of the intensity of read, green, and blue at each location, we now have the intensity of each feature at each location.
- Second, the height and width both have shrunk by 6 pixels. This is due to the fact that the `Conv2D` layer does not use any zero-padding by default, which means that we lose a few pixels on the side of the output feature maps, depending on the size of the filters. In this case, since the kernel size is 7, we lose 6 pixels horizontally and 6 pixels vertically (i.e., 3 pixels on each side).
- The default option is weirdly named `padding="valid"`, which actually means no zero-padding at all!
- This name comes from the fact that in this case, every neuron's receptive field lies strictly within valid positions inside the input (it does not go out of bounds). This is not a Keras naming quirk: everyone is using this nomenclature.
- If we instead set `padding="same"`, then the inputs are padded with enough zeros on all sides to ensure that the output feature maps end up with the same size as the input (hence the name of this option).
- These two padding option are illustrated in the figure below. For simplicity, only the horizontal dimension is shown here, but the same logic applies to the vertical dimension as well.
![The two padding options, when $strides=1$](image-4.png)
- If the stride is greater than 1 (in any direction), then the output size will not equal to the input size, even if `padding="same"`. For example, if you set `strides=2` (or equivalently `strides=(2, 2)`), then the output feature maps will be $35 \times 60$: halved both vertically and horizontally.
- The following figure shows what happens when `strides=2`, with both padding options:
![With strides greater than 1, the output is much smaller even when using
"same" padding (and "valid" padding may ignore some inputs)](image-5.png)
- Here is how the output size is computed:
    - With `padding="valid"`, if the width of the input is $i_w$, then the output width is equal to $(i_h - f_h + s_h) / s_h$, rounded down. Recall that $f_h$ is the receptive's width, and $s_h$ is the horizontal stride. Any remainder in the division corresponds to ignored columns on the right side of the input image. the same logic can be applied to compute height,a nd any ignored rows is at the bottom of the image.
    - With `padding="same"`, the output width is equal to $i_h / s_h$, rounded up. To make this possible, the appropriate number of zero columns are padded to the left and the right of the input image (an equal number of possible, or just one more on either side). Assuming the output width is $o_w$, then the number of padded zero columns is $(o_w-1)\times s_h + f_h - i_h$. Again, the same logic can be used to compute the output height and the number of padded rows.
- Now, let's look at the layer's weight (which were noted $w_{u, v, k, k'}$ and $b_k$) in the previous equation. Just like a `Dense` layer, a `Conv2D` layer holds all the layer's weights, including the kernels and biases.
- The kernels are initialized randomly, while the biases are initialized at zero.
- These weights are accessible via the `weights` attribute, or as NumPy arrays via the `get_weights()` method.
- The `kennels` array is 4D, and its shape is [*kernel_height, kernel_width, input_channels, output_channels*].
- The `biasses` array is 1D, has shape [*output channel*]. The number of output channel is equal to the number of output feature maps, which is equal to the number fo filters.
- Most importantly, note that the height and width of the input images do not appear in the kernel's shape: this is because all the neurons in the output feature maps share the same weights, as explained earlier.
- Which means you can feed images of any size to this layer, as long as they are at least the size of the kernels (i.e., the receptive field), and  have the right number of channels (three in this case).
- Lastly, you would like to specify an activation function (such as ReLU) when creating a `Conv2D` layer, and also specify the corresponding kernel initializer (such as He initializer). This has the same reason as `Dense` layers: a convolutional layer performs a linear operation, so if you stacked many of them without any activation function, it's just equivalent to a single convolutional layer, and they wouldn't be able to learn anything complex.
- As you can see, convolutional layers have quite a few hyperparameters: `filters`, `kernel_sizes`, `padding`, `strides`, `activation`, `kernel_initializer`, etc. You can check the documentation for the full list of them.
- As always, you can use cross-validation ot find the best hyperparameter values, but this is very resource-consuming.
- We will discuss common CNN architectures later in this chapter, to give you some ideas of which hyperparameter values work best in practice.

## Memory Requirements

- Another challenges with CNNs is that the convolutional layers require a huge amount of RAM.
- This is especially true during training, because the reserve pass of backpropagation requires all the intermediate values computes during the forward pass.
- For example, consider a convolutional layer with 200 $5 \times 5$ filters, with stride 1 and `"same"` padding. If the input is a $150 \times 100$ RGB images (three channels), then the number of parameters is $(5 \times 5 \times 3 + 1) \times 200 = 15,200$ (the + 1 corresponds to the bias terms), which is fairly small compared to a fully connected layer (to produce the same size outputs, a fully connected layer would need $200 \times 150 \times 100$ neurons, each connected to all $150 \times 100 \times 3$ inputs, which results in $200 \times 150 \times 100 \times (150 \times 100 \times 3 + 1) \approx 135$ billion parameters).
- However, each of the 200 feature maps contains $150 \times 100$ neurons, and each of these neurons needs to compute a weighted sum of its $5 \times 5 \times 3 = 75$ inputs: that's a total of 225 million float multiplications. Not as bad as fully connected layer, but still quite computationally intensive.
- Moreover, if the feature maps are represented using 32-bit floats, then the convolutional layer's output will occupy $200 \times 150 \times 200 \times 32 = 96$ million bits (12 MB) of RAM. And that's one instance - if a training batch contains 100 instances, then this layer will use up 1.2 GB of RAM!
- During inference (i.e., when making a prediction for a new instance), the RAM occupied by one layer can be released as soon as the next layer has been computed, so you only need as much RAM as required by two consecutive layers.
- But during training, everything computed during the forward pass needs to be preserved for the reserve pass, so the amount of RAM is (at least) the total amount of RAM required by all layers.
- If training crashes because of an out-of-memory error, you can try reducing the mini-batch size. Alternatively, you can try reducing dimensionality using a stride, removing a few layers, using 16-bit floats instead of 32-bit floats, or distributing the CNN across multiple devices (you'll see how to do this in chapter 19).