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

# Pooling Layers

- Pooling layers are quite similar to convolutional layers.
- Their goal is to *subsample* (i.e., shrink) the input image in order to reduce the computational load, the memory usage, and the number of parameters (thereby limiting the risk of overfitting).
- Just like in convolutional layers, each neuron in a pooling layer is connected to the outputs of a limited number of neurons in the previous layer, located within a small rectangular receptive field.
- You must define its size, the stride, and the padding type, just like before.
- However, a pooling neuron has no weights; all it does is aggregate the input using an aggregation function such as the max or mean.
- For example, let's consider the *max pooling layer*, which is the most common type of pooling layer.
- In this case, we use a $2 \times 2$ *pooling kernel*, with a stride of 2 and no padding.
- Only the max input value in each receptive field makes it to the next layer, while the other inputs are dropped.
- For example, in the lower-left receptive field, the input values are 1, 5, 3, 2, so only the max value, 5, is propagated to the next layer.
- Because of the stride of 2, the output images has half the height and half the width of the input image (rounded down since we use no padding).
![Max pooling layer (2 × 2 pooling kernel, stride 2, no padding)](image-6.png)
- A pooling layer typically works on every input channel independently, so the output depth (i.e., the number of channels) is the same as the input depth.
- Other than reducing computations, memory usage, and the number of parameters, a max pooling layer also introduces some level of *invariance* to small translations, as shown below:
![Invariance to small translations](image-7.png)
- Here we assume that the bright pixels have lower values than dark pixels, and we consider three images (A, B, C) going through a max pooling layer with $2 \times 2$ kernel and stride 2. Images B and C are the same as image A, but shifted by one and two pixels to the right.
- As you can see, the outputs of the max pooling layer for images A and B are identical. This is what translation invariance means. For image C, the output is different: it is shifted one pixel to the right (but there is still 50% invariance).
- By inserting a max pooling layer every few layers in a CNN, it is possible to get some level of translation invariance at a larger scale.
- Moreover, max pooling offers a small amount of rotational invariance and a slight scale invariance. Such invariance (even if it is limited) can be useful in case where the prediction should not depend on these details, such as in classification tasks.
- However, max pooling has some downsides, too:
    - It's obviously very destructive: even with a tiny $2 \times 2$ kernel and a stride of 2, the output image would end up being two time smaller in both directions (its ares will be four times smaller), simply dropping 75% of the input values.
    - And in some application, invariance is not desirable. Take semantic segmentation (the task of classifying each pixel in an image according to the object that pixel belongs to) for example, obviously, if the input image is translated by one pixel to the right, the output should also be translated by one pixel to the right. The goal in this is *equivariance*, not variance: a small change to the input should correspond to a small change in the output.

## Implementing Pooling Layers with Keras

- The code in the learning notebook creates a `MaxPooling2D` layer, alias `MaxPool2D`, using a $2 \times 2$ kernel.
- The strides default to the kernel size, so this layer uses a stride of 2 (horizontally and vertically).
- By default, it use `"valid"` padding (i.e., no padding at all).
- To create an *average pooling layer*, just use `AveragePooling2D`, alias `AvgPool2D`, instead of `MaxPool2D`.
- This layer works exactly like a max pooling layer, except it computes the mean rather than the max.
- Average pooling layers used ti be very popular, but people mostly use max pooling layers now, as they generally perform better. This may come as a surprise, as computing the mean generally loses less information than computing the max.
- But on the other hand, max pooling preserves only the strongest features, getting rid of all the meaningless ones, so the next layers get a cleaner signal to work with.
- Moreover, max pooling offers stronger translation invariance than average pooling, and it requires slightly less compute.
- Note that max pooling and average pooling can be performed along the depth dimension instead of the spatial dimensions, although it's not as common. This can allow the CNN to learn to be invariant to various features.
- For example, it could learn multiple filters each detecting a different rotation of the same pattern, and the depthwise max pooing layer would ensure that the output is the same, regardless of the rotation.
- The CNN could similarly learn to be invariant to anything: brightness, skew, color, and so on.
- Keras doesn't include a depthwise max pooling layer, but you can find a custom layer do just this in the learning notebook.
- This layer reshapes its inputs to split the channels into groups of the desired size (`pool_size`), then it uses `tf.reduce_max()` to compute the max of each group.
- This implementation assumes that the stride is equal to the pool size, which is generally what you want.
- Alternatively, you could use TensorFLow's `tf.nn.max_pool()` operation, and wrap it in a `Lambda` layer to use it inside a Keras model, but sadly, this operation does not implement depthwise pooling for the GPU, only the CPU.
- One last type of pooling layer that you will often see in modern architecture is the *global average pooling layer*.
- It works very differently: it computes the mean of each entire feature map (it's like an average pooling layer using a pooling kernel the same spatial dimensions as the inputs).
- This means it just outputs a single number per feature map and per instance.
- Although it is of course very destructive (most of the information in the feature map is lost), it can be useful just before the output layer, as you'll see later in this chapter.
- To use this layer, simply define a `GlobalAveragePooling2D`, alias `GlobalAvgPool2D`, instance.
- We can define a `Lambda` layer equivalently to this layer, which computes the mean over the spatial dimensions (height and width).
- For example, if you apply this layer to the input images, we get the mean intensity of red, green and blue of each image.

# CNN Architectures

- Typically CNN architectures stack a few convolutional layers (each one generally followed by a ReLU layer), then a pooling layer, then another few convolutional layers (+ReLU), then another pooling layer, and so on.
- The image gets smaller and smaller as it progresses though the network, but it also typically gets deeper and deeper (i.e., with more feature maps), thanks to the convolutional layers.
- At the top of the stack, a regular feed-forward neural network is added, composed of a few fully connected layers (+ReLUs), and the final layer outputs the prediction (e.g., a softmax layer that outputs estimated class probabilities).
- A common mistake is to use convolutional kernels that are too large.
- For example, instead of using a convolutional layer with a $5 \times 5$ kernel, stack two layers with $3 \times 3$ kernels: it will use fewer parameters and require fewer computations, and it will usually performs better.
- One exception is for the first convolutional layer: it can typically have a larger kernel (e.g., $5 \times 5$), usually with a stride of 2 or more. This will reduce the spatial dimension of the image without losing too much information and since the input image only has three channels in general, it will not be too costly.
- Let's go through an implementation example in the learning notebook:
    - We use the `functools.partial()` function (introduced in chapter 11) to define `DefaultConv2D`, which acts just like `Conv2D` but with different default arguments: a small kernel size of 3, `"same"` padding, the ReLU activation function, and its corresponding He initializer.
    - Next, we create the `Sequential` model. Its first layer is a `DefaultConv2D` with 64 fairly large filters ($7 \times 7$). It uses the default stride of 1 because the input images are not very large. It also sets `input_shape=[28, 28, 1]`, because the images are $28 \times 28$ pixels, with a single color channel (i.e., grayscale). When you load the Fashion MNIST dataset, make sure each image has this shape: you may need to use `np.reshape()` or `np.expanddims()` to add the channel dimension. Alternatively, you could use a `Reshape` layer as the first layer in the model.
    - We then add a max pooling layer that uses the default pool size of 2, so it divides each spatial dimension by a factor of 2.
    - Then we repeat the same structure twice: two convolutional layers followed by a max pooling layer. For larger images, we could repeat this structure several more times. The number of repetitions is a hyperparameter you can tune.
    - Note that the number of filters doubles as we climb up the CNN toward the output layer (it is first 64, the 128, then 256): it makes sense for it to grow, since the number of lower-level features is often fairly low (e.g., small circles, horizontal lines), but there are many different ways to combine them into higher-level features. It is a common practice to double the number fo filters after each pooling layer: since a pooling layer divides each spatial dimension by a factor of 2, we can afford to double the number of feature maps in the next layer without fear of exploding the number of parameters, memory usage, or computational load.
    - next is the fully connected network, composed of two hidden dense layers and a dense output layer. Since it's a classification task with 10 classes, the output layer has 10 units, and it uses the softmax activation function. Note that we must flatten the inputs just before the fist hidden layer, since it expects a 1D array of features for each instance. We also add two dropout layers, with a dropout rate of 50% each, to regularize the model.
- Over the years, variants of this fundamental architecture have been developed, leading to amazing advances in the field. A good measure of this progress is the error rate in competitions such as the LLSVRC [ImageNet challenge](https://image-net.org/).
- In this competition, the top-five error-rate for image classification - that is, the number of test images for which the system's top five predictions did not include the correct answer - fell form over 26% to less than 2.3% in just six years.
- The images are fairly large (e.g., 256 pixels high) and there are 1,000 classes, some of which are really subtle (try distinguish 120 dog breeds).
- We first look at the classical LeNet-5 architecture (1998), then several winners of the ILSVRC challenge: AlexNet (2012), GoogLeNet (2014), ResNet(2015), and SENet (2017). Along the way, we will also look at a few more architectures including Xception, ResNeXt, DenseNet, MobileNet, CSPNet, and EfficientNet.

## LeNet-5

- The LeNet-5 architecture is perhaps the most widely known CNN architecture.
- As mentioned earlier, it was created by Yann LeCun in 1998 and has been widely used for hand written digit recognition (MNIST)

Layer  | Type            | Maps | Size     | Kernel size | Stride | Activation
-------|-----------------|------|----------|-------------|--------|-----------
 Out   | Fully connected | -    | 10       | -           | -      | RBF
 F6    | Fully connected | -    | 84       | -           | -      | tanh
 C5    | Convolution     | 120  | 1 × 1    | 5 × 5       | 1      | tanh
 S4    | Avg pooling     | 16   | 5 × 5    | 2 × 2       | 2      | tanh
 C3    | Convolution     | 16   | 10 × 10  | 5 × 5       | 1      | tanh
 S2    | Avg pooling     | 6    | 14 × 14  | 2 × 2       | 2      | tanh
 C1    | Convolution     | 6    | 28 × 28  | 5 × 5       | 1      | tanh
 In    | Input           | 1    | 32 × 32  | -           | -      | -   

- As you can see, this look pretty similar to out Fashion MNIST model: a stack of convolutional layers. and pooling layers, followed by a dense network.
- The main difference with more modern classification CNNs is, perhaps, the activation functions: today, we would use ReLU instead of tanh and softmax instead of RBF.
- There are some other minor differences that don't really matter very much, but in case you are interested, they are listed in this [chapter's notebook](https://colab.research.google.com/github/ageron/handson-ml3/blob/main/14_deep_computer_vision_with_cnns.ipynb) written by the author.

## AlexNet

- The [AlexNet CNN architecture](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) won the 2012 ILSVRC challenge by a large margin: it achieved a top-five error rate of 17%, while the second best competitor achieved only 26%.
- AlexNet was developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton.
- It is similar to LeNet, only much larger and much deeper, and it was the first to stack convolutional layers directly on top of one another, instead of stacking a pooling layer on top of each convolutional layer.

| Layer | Type             | Maps | Size       | Kernel size | Stride | Padding | Activation |
|-------|------------------|------|------------|-------------|--------|---------|------------|
| Out   | Fully connected | -    | 1,000      | -           | -      | -       | Softmax    |
| F10   | Fully connected | -    | 4,096      | -           | -      | -       | ReLU       |
| F9    | Fully connected | -    | 4,096      | -           | -      | -       | ReLU       |
| S8    | Max pooling     | 256  | 6 × 6      | 3 × 3       | 2      | valid   | -          |
| C7    | Convolution     | 256  | 13 × 13    | 3 × 3       | 1      | same    | ReLU       |
| C6    | Convolution     | 384  | 13 × 13    | 3 × 3       | 1      | same    | ReLU       |
| C5    | Convolution     | 384  | 13 × 13    | 3 × 3       | 1      | same    | ReLU       |
| S4    | Max pooling     | 256  | 13 × 13    | 3 × 3       | 2      | valid   | -          |
| C3    | Convolution     | 256  | 27 × 27    | 5 × 5       | 1      | same    | ReLU       |
| S2    | Max pooling     | 96   | 27 × 27    | 3 × 3       | 2      | valid   | -          |
| C1    | Convolution     | 96   | 55 × 55    | 11 × 11     | 4      | valid   | ReLU       |
| In    | Input           | 3 (RGB) | 227 × 227 | -           | -      | -       | -          |

- To reduce overfitting, the authors used two regularization techniques:
    - First, they applied dropout (introduced in chapter 11) with a 50% dropout rate during training to the outputs of layers F9 and F10.
    - Second, they performed data augmentation by randomly shifting the training images by various offset, flipping them horizontally, and changing the lighting conditions.
- AlexNet also uses a competitive normalization step immediately after the ReLU step of layers C1 and C3, called *local response normalization* (LRN): the most strongly activated neurons inhibit other neurons located at the same position in neighboring feature maps. Such competitive activation has been observed in biological neurons.
- This encourages different feature maps to specialize, pushing them apart and forcing them to explore a wider range of features, ultimately improving generalization.
- The following equation shows how to apply LRN:
$$b_i = a_i \left(k + \alpha\sum_{j=j-low}^{j-high}a_j^2\right)^{-\beta} \text{ with } 
\begin{cases}
\text{j-high} = \min(i + \frac{r}{2}, f_n - 1) \\
\text{j-low} = \max(0, i - \frac{r}{2})
\end{cases}
$$
- In this equation:
    - $b_i$ is the normalized output of the neuron located in feature map $i$, at some row $u$ and column $v$ (note that in this equation we only consider neurons located at this row and column, so $u$ and $v$ are not shown).
    - $a_i$ is the activation of that neuron after the ReLU step, but before normalization.
    - $k, \alpha, \beta$ and $r$ are hyperparameters. $k$ is called the *bias*, and $r$ is called the *depth radius*.
    - $f_n$ is the number of feature maps.
- For example, if $r=2$ and a neuron has as strong activation, it will dominate the activation of the neurons located in the feature maps immediately below and above it.
- In AlexNet, the hyperparameters are set as: $r=5, \alpha=0.0001, \beta=0.75$ and $k=2$. You can implement this step by using the `tf.nn.local_response_normalization()` function (which you can wrap in a `Lambda` layer if you want to use it a model).
- A variant of AlexNet called the [ZF Net]() was developed by Matthew Zeiler and Rob Fergus nd won the 2013 ILSVRC challenge. It is just AlexNet with a few tweaked hyperparameters (number of feature maps, kernel sides, stride, etc.).

### Data Augmentation

- Data augmentation artificially increases the size of the training set by generating many realistic variants of each training instance. This reduces overfitting, makes it a regularization technique.
- The generated instances should be as realistic as possible: ideally, given an image from the augmented training, a human should be be able to tell whether it was augmented or not.
- Simply adding white noises does not help; the modifications should be learnable (while white noises are not).
- For example, you can slightly shift, rotate, and resize every picture in the training set by various amounts and add the resulting pictures to the training set.
![Generating new training instances from existing ones](image-8.png)
- To do this, you can use Keras's data augmentation layers, introduced in chapter 13 (e.g., `RandomCrop`, `RandomRotation`, etc.).
- This forces the model to be more tolerant of variations in the position orientation, and size of the objects in the pictures.
- To produce a model that's more tolerant of different lighting conditions you can similarly generate many images with various contrasts.
- In general, you can also flip the image horizontally, (expect for text, and other asymmetrical objects).
- By combining these transformations, you can greatly increase your training set size.
- Data augmentation is also useful when you have unbalanced dataset: you can use it to generate more samples of the less frequent classes. This is called *synthetic minority oversampling technique* or SMOTE for short.

## GoogLeNet

- The [GoogLeNet architecture](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) was developed by Christian Szegedy et al. from Google Research, nad it won the ILSVRC 2014 challenge by pushing the top-five error rate below 7%.
- This great performance came in large part from the fact that the network was much deeper than previous CNNs.
- This was made possible by sub-networks called *inception modules*, which allow GoogLeNet to use parameters much more efficiently than previous architectures: GoogLeNet actually has 10 times fewer parameters than AlexNet (roughly 6 million instead of 60 million).
- This figure shows the architecture of an inception module.
![alt text](image-9.png)
- The notation "$3 \times 3 + 1$(S)" means that the layer uses a $3\times 3$ kernel, stride 1, and `"same"` padding.
- The input signal is first fed to four different layers in parallel.
- All convolutional layers use the ReLU activation function.
- Note that the top convolutional layers use different kernel sizes ($1\times 1$, $3 \times 3$, and $5 \times 5$), allowing them to capture patterns at different scales.
- Also note that every single layer uses a stride of 1 and `"same"` padding (even the max pooling layer), so their outputs all have the same height and width as their inputs.
- This make it possible to concatenate all the outputs along the depth dimension in the final *depth concatenation layer* (i.e., to stack the feature map from all four top convolutional layers).
- It can be implemented using Keras's `Concatenate` layer, using the default `axis=-1`.
- The reason why inception modules have convolutional layers with $1\times 1$ kernel are:
    - They are configured to output fewer feature maps than their inputs, so they serve as *bottleneck layers*, meaning they reduce dimensionality. This cuts the computational cost and the number of parameters, speeding up training and improving generalization.
    - Although they cannot capture spatial patterns, they capture patterns along the depth dimension (i.e., across channels).
    - Each pair of convolutional layers ([$1\times 1$, $3\times 3$] and [$1\times 1$, $5\times 5$]) acts like a single powerful convolutional layer. A single convolutional layer is equivalent to sweeping a dense layer across the image (at each location, it only looks at a small receptive field), and these pairs of convolutional layers are equivalent to sweeping two-layer neural networks across the images.
- In short, you can think of the whole inception module as a convolutional layer on steroids, able to output feature maps that capture complex patterns at various scales.
- Now you can look at the architecture of the GoogLeNet CNN:
![GoogLeNet architecture](image-10.png)
- The number of feature maps output by each convolutional layer and each pooling layer is shown before the kernel size.
- The architecture is so deep that it has to be represented in three columns, but GoogLeNet is actually one tall stack, including nine inception modules (the boxes with the spinning tops).
- The six numbers in the inception modules represent the number of feature maps output by each convolutional layer in the module (in the same other as in the figure described the inception module).
- Note that all the convolutional layers use the ReLU activation function.
- Let's go through this network:
    - The first two layers divide the image's height and width by 4 (so its area is divided by 16), to reduce the computational load. The first layer uses a large kernel size, $7 \times 7$, so that much of the information is preserved.
    - Then the local response normalization layer ensures that the previous layers learn a wide variety of features (as discussed earlier).
    - Two convolutional layer follow, where the first acts as a bottleneck layer. As mentioned, you can think of this pair as a single smarter convolutional layer.
    - Again, a local response normalization layer ensures that the previous layers capture a wide variety of patterns.
    - Next, a max pooling layer reduces the image height and width by 2, again to speed up computations.
    - Then comes the CNN's backbone: a tall stack of nine inception modules, interleaved with a couple of max pooling layers to reduce dimensionality and speed up the net.
    - Next, the global average pooling layer outputs the mean of each feature map: this drops any remaining spatial information, which is fine because there is not much spatial information left at that point. Indeed, GoogLeNet input images are typically expected to be $224 \times 224$ pixels, so after 5 max pooling layers, each dividing the height and width by 2, the feature maps are down to $7 \times 7$. Moreover, this is a classification task, not localization, so it doesn't matter where the object is. Thanks to the dimensionality reduction brought to this layer, there is no need to have several fully connected layers at the top of the CNN (like in AlexNet), and this considerably reduces the number of parameters in the network and limits the ris of overfitting.
    - The last layers are self-explanatory: dropout for regularization, then a fully connected layer with 1,000 units (since there are 1,000 classes) and a softmax activation function to output estimated class probabilities.
- The original GoogLeNet architecture included two auxiliary classifiers plugged on top of the third and sixth inception modules.
- They were both composed of one averaging pooling layer, one convolutional layer, two fully connected layers, and a softmax activation layer.
- During training, their loss (scaled down by 70%) was added ot the overall loss.
- The goal was to fight the vanishing gradient problem and regularize the network, bu tit was later shown that their effects was relatively minor.
- Several variants of the GoogLeNet architecture were later proposed by Google researchers, including Inception-v3 and Inception-v4, using slightly different inception modules to reach even better performance.

## VGGNet

- The runner-up in the ILSVRC 2014 challenge was [VGGNet](https://arxiv.org/abs/1409.1556).
- Karen Simonyan and Andrew Zisserman, from the Visual Geometry Group (VGG) research lab at Oxford University, developed a very simple and classical architecture:
    - it had 2 or 3 convolutional layers and a pooling layer
    - then again 2 or 3 convolutional layers and a pooling layer
    - so on, reaching a total of 16 or 19 convolutional layers, depending on the VGG variant.
    - plus a final dense network with 2 hidden layers and the output layer.
- It used small $3\times filters$, but it had many of them.

## ResNet

- Kaiming He et al. won the ILSVRC 2015 challenge using a [Residual Network (ResNet)](https://arxiv.org/abs/1512.03385) aht delivered an astounding top-five error rate under 3.6%.
- The wining variant used an extremely deep CNN composed of 152 layers (other variants had 34, 50, and 101 layers).
- It confirmed the general trend: computer vision models were getting deeper and deeper, with fewer and fewer parameters.
- The key of being able to train such a deep network is to use *skip connections* (also called *shortcut connections*): the signal feeding to a layer is also added to the output of a a layer located higher up the stack.
- When training a neural network, the goal is to make it model a target function $h(\textbf{x})$.
- If you add the input $\textbf{x}$ to the output of the network (i.e., you add a skip connection), then the network will be forced to model $f(\textbf{x}) = h(\textbf{x}) - \textbf{x}$ rather than $h(\textbf{x})$. This is called *residual learning*.
![Residual learning](image-11.png)
- When you initialize a regular neural network, its weights are close to zero, so the network just outputs values close to zero.
- If you add a skip connection, the resulting network just outputs a copy of its inputs; in other words, it initially models the identity function
- If the target function is close to the identity function (which is often the case), this will speed up training considerably.
- Moreover, if you add many skip connections, the network can start making progress even if several layers have not started learning yet.
![Regular deep neural network (left) and deep residual network (right)](image-12.png)
- Thanks to skip connections, the gradients now flow in 2 main ways, instead of 1, hence the signal can make its way across the whole network.
- The deep residual network can be seen as a stack of *residual units* (RUs), where each residual unit is a small neural network with a skip connection.
- Now, we look at ResNet's architecture, notice how simple it is:
    - It starts and ends exactly like GoogLeNet (except without a dropout layer).
    - In between is a very deeps stack of residual units.
    - Each residual units is composed of two convolutional layers (and no pooling layer!), with batch normalization (BN) and ReLU activation, using $3 \times 3$ kernel nad preserving spatial dimensions (stride 1, padding `"same"`).
![ResNet architecture](image-13.png)
- Note that the number of feature map is doubled once after a few residual units, at the same time as their height and width are halved (using a convolutional layer with stride 2).
- When this happens, the inputs cannot be added directly to the outputs of the residual units because they don't have the same shape (for example, see the figure below).
![Skip connection when changing feature map size and depth](image-14.png)
- To solve this problem, the inputs are passed through a $1 \times 1$ convolutional layer with stride 2 and the right number of feature maps.
- Different variations of the architecture exist, with different number of layers.
- ResNet34 is a ResNet with 34 layers (only counting the convolutional layers and the fully connected layer) containing 3 RUs that output 64 features maps, 4 RUs with 128 feature maps, 6 RUs with 256 feature maps, and 3 RUs with 512 maps. We will implement this architecture later in this chapter.
- Google's [Inception-v4](https://arxiv.org/abs/1602.07261) architecture merged the ideas of GoogLeNet and ResNet and achieve a top-five error rate of close to 3% on ImageNet classification.
- ResNet deeper than that, such as ResNet-152, use slightly different residual units.
- Instead of two $3 \times 3$ convolutional layers with, say, 256 feature maps, they use three convolutional layers: first a $1\times 1$ convolutional layer with just 64 feature maps (4 times less), which acts a bottleneck layer, then a $3 \times 3$ layer with 64 feature maps and finally another $1\times 1$ convolutional layer with 256 features maps that restores the original depth.
- ResNet-152 contains 3 such RUs that outputs 256 maps, then 8 RUs with 512 maps, then a whopping 36 RUs with 1,024 maps, and finally 3 RUs with 2,048 maps.

## Xception

- Another variant of the GoogLeNet architecture is worth noting: [Xception](https://arxiv.org/abs/1610.02357), which stands for *Extreme Inception*, was proposed in 2016 by François Chollet (the author of Keras).
- It significantly outperformed Inception-v3 on a huge vision task (350 million images and 17,000 classes).
- Just like Inception-v4, it merges the ideas of GoogLeNet and ResNet, but it replaces the inception modules with a special type of layer named a *depthwise separable convolution layer* (or *separable convolution layer* for short).
- These layers had been used before in some CNN architectures, but they were not as central as in the Xception architecture.
- While a regular convolutional layer uses filters that try to simultaneously capture spatial patterns (e.g., an oval) and cross-channel patterns (e.g., mouth + nose + eyes = face), a separable convolutional layer makes the strong assumption that spatial patterns and cross-channel patterns can be modeled separably.
![Depthwise separable convolutional layer](image-15.png)
- Thus, it's composed of two parts: the first part applies a single spatial filter to each input feature map, then the second part looks exclusively for cross-channel patterns - it is just a regular convolutional layer with $1\times 1$ filters.
- Since separable convolutional layers only have one spatial filter per input channel, you should avoid using them after layers that have too few channels, such as the input layer (well, that's what the above figure represents, but it's just for illustration purposes).
- For this reason, Xception architecture starts with 2 regular convolutional layers, but then the rest of the architecture uses only separable convolutions (34 in all), plus a few max pooling layers and the usual final layers (a global average pooling layer and a dense output layer).
- You might wonder why Xception is considered a variant of GoogLeNet, since it has no inception module at all.
- Well, as discussed earlier, an inception module contains convolutional layers with $1\times 1$ filters: they look exclusively for cross-channel patterns. However, the convolutional layers that look both for spatial and cross-channel patterns jointly.
- So you can think of an inception module as an intermediate between a regular convolutional layer (which considers spatial patterns and cross-channel jointly) and a separable convolutional layer (which considers them separately).
- In practice, it seems that separable convolutional layers often perform better.
- Separable convolutional layers use fewer parameters, less memory, and fewer computations than regular convolutional layers, and they often perform better.
- Consider using them, except after layers with few channels (such as the input layer).
- In Keras, just use `SeparableConv2D` instead of `Conv2D`: it's a drop-in replacement.
- Keras also offers a `DepthwiseConv2D` layer that implements the first part of a depthwise separable convolutional layer (i.e., applying one spatial filter per input feature map).

## SENet

- The winning architecture in the ILSVRC 2017 challenge was the [Squeeze-and-Excitation (SENet)]().
- This architecture extends existing architectures such as inception networks and ResNets, and boosts their performance.
- This allows SENet to win the competition with an astonishing 2.25% top-five error rate!
- The extended versions of inceptions networks and ResNets are called *SE-inception* and *SE-ResNet*, respectively.
- The boost comes from the fact that a SENet adds a small neural network, called an *SE block*, to every inception module or residual unit in the original architecture, as shown below:
![SE-Inception module (left) and SE-ResNet unit (right)](image-16.png)
- An SE block analyzes the output of the unit it is attached to, focusing exclusively on the depth dimension (it does not look for any spatial patterns), and it learns which features are usually most active together.
- It then uses this information to recalibrate the feature maps, as shown below:
![An SE block performs feature map recalibration](image-17.png)
- For example, an SE block may learn that mouths, noses, and eyes usually appear together in pictures: if you see a mouth and a nose, you should expect to see eyes as well.
- So, if the block sees a strong activation in the mouth and nose feature maps, but only mild activation in the eye feature map, it will boost the eye feature map (more accurately, it will reduce irrelevant feature maps).
- If the eyes was somewhat confused with something else, this feature map recalibration will help solve the ambiguity.
- An SE block is composed of just three layers: a global average pooling layer, a hidden layer using the ReLU activation function, and a dense output layer using the sigmoid activation function.
![SE block architecture](image-18.png)
- As earlier, the global average pooling layer computes the mean activation for each feature map.
- For example, if it inputs contains 256 feature maps, it will output 256 number representing the overall level fo responses for each filter.
- The next layer is where the "squeeze" happens: this layer has significantly fewer than 256 neurons - typically 16 times fewer than the number of feature maps (e.g., 16 neurons) - so the 256 numbers got compressed into a small vector (e.g., 16 dimensions).
- This is low-dimensional vector representation (i.e., an embedding) of the distribution of the feature responses.
- This bottleneck step forces the SE block to learn a general representation of the feature combinations (we will see this principle in action again when we discuss autoencoders in chapter 17).
- Finally, the output layer takes the embedding and outputs a recalibration vector containing one number per feature map (e.g., 256), each between 0 and 1.
- The feature maps are then multiplied by this recalibration vector, so irrelevant features (with a low recalibration weight) get scaled down, while relevant features (with a recalibration weight closes to 1) are left alone.

## Other Noteworthy Architectures

- There are many other CNN architecture to talk about. We will have a brief overview of some of the most noteworthy in this part.
- [*ResNeXt*](https://arxiv.org/abs/1611.05431): ResNeXt improves the residual units in ResNet.
- Whereas the residual units in the best ResNet models just contains 3 convolutional layers each, the ResNeXt residual units are composed of many parallel stacks (e.g., 32 stacks), with 3 convolutional layers each.
- However, the first two layers in each stack only use a few filters (eg., just four), so the overall number of parameters remains the same as in the ResNet.
- Then the outputs of all stacks are added together, and the result is passed to the next residual units (along with the skip connections).
- [*DenseNet*](https://arxiv.org/abs/1608.06993): A DenseNet is composed of several dense blocks, each made up of a few densely connected convolutional layers. This architecture achieved excellent accuracy while using comparatively few parameters.
- But what does "densely connected" mean? The output of each layer is fed as input to every layer after it within the same block. For example, layer 4 in a block takes as input the depthwise concatenation of the outputs of layers 1, 2, and 3 in that block.
- Dense blocks are separated by a few transition layers.
- [*MobileNet*](https://arxiv.org/abs/1704.04861): MobileNet are streamlined models designed to be lightweight and fast, making them popular in mobile and web applications.
- They are based on depthwise separable convolutional layers, like Xception.
- The authors proposed several variants, trading a bit of accuracy for faster and smaller models.
- [*CSPNet*](https://arxiv.org/abs/1911.11929): A Cross Stage Partial Network (CSPNet) is similar to a DenseNet, but part of each dense block's input is concatenated directly ot that block's output, without going through the block.
- [*EfficientNet*](https://arxiv.org/abs/1905.11946): EfficientNet is arguably the most important model in this list.
- The authors proposed a method to scale any CNN efficiently, by jointly increasing the depth (number of layers), width (number of filters per layer), and resolution (size of the input image) in a principled way. This is called *compound scaling*.
- They used neural architecture search to find a good architecture for a scaled version of ImageNet (with smaller and fewer images), and then used compound scaling to create larger and larger version of this architecture.
- When EfficientNet came out, they vastly outperformed all existing models, across all compute budgets, and they remain among the best models out there today.
- Understanding EfficientNet's compound scaling method is helpful to gain a deeper understanding of CNNs, especially if you ever need to scale a CNN architecture.
- It based on a logarithmic measure of the compute budget, noted *$\phi$*: if your compute budget doubles, then *$\phi$* increases by 1.
- In other words, the number of floating-points operations available for training is proportional to $2^\phi$.
- Your CNN architecture's depth, width, and resolution should scale as $\alpha^\phi$, $\beta^\phi$, and $\gamma^\phi$.
- The factors $\alpha$, $\beta$, and $\gamma$ must be greater than 1, and $\alpha + \beta^2 + \gamma^2$ should be close to 2. The optimal values for these factors depend on the CNN's architecture.
- To find the optimal values for the EfficientNet architecture, the authors started with a small baseline model (EfficientNetB0), fixed $\phi=1$, ad simply ran a gird search: they found $\alpha=1.2$, $\beta=1.1$, and $\gamma=1.1$. They then used these factors to create several larger architectures, named EfficientNetB1 to EfficientNetB7, for increasing value of $\phi$.

## Choosing the Right CNN Architecture  

- With so many CNN architectures, how do you choose which one is best for your project?
- Again, it depends to on the task at hand, specifically at the constraints and your trade-off between them: Accuracy, model size (e.g., for deployment to a mobile device), inference speed on CPU, on GPU.
- You can find the list of pretrained models available Keras [here](https://keras.io/api/applications/)
![Pretrained models available in Keras](image-19.png)
- For each model, the table shows the Keras class name to use (in the `tf.keras.applications` package), the model's size in MB, the top-1 and top-5 validation accuracy on the ImageNet dataset, the number of parameters (millions), depth, and the inference times on CPU and GPU in ms, using batches of 32 images on reasonably powerful hardware.

# Implementing a ResNet-34 CNN Using Keras

- Most CNN architectures described so far can be implemented pretty naturally using Keras (although generally you would load a pretrained network instead, as you will see later).
- To illustrate this process, let's implement a ResNet-34 from scratch with Keras.
- First, we create a `ResidualUnit` class:  
    - In the constructor, we create all the layers we need: the main layers on the right, and the skip layers on the left (only needed if the strides is greater than 1).
    - In the `call()` method, we let the inputs go through the main layers and the skip layers (if any), and we add both outputs and apply the activation function.
- Now we can build a ResNet-34 using the `Sequential` model, since it's just a really long sequence of layers - we can treat each residual units as a single layer since we have the `ResidualUnit` class.
- The only tricky part in the implementation is the loop that adds the `ResidualUnit` layers to the model: as explained earlier, the first 3 RUs have 64 filters, then the next 4 RUs have 128 filters, and so on. At each iteration, we must set the stride to 1, or else we set it to 2; then we add the `ResidualUnit`, and finally we update `prev_filters`.
- Isn't it amazing that in about 40 lines of code, we can build the model that won the ILSVRC 2015 challenge? This demonstrates both the elegance of the ResNet model and the expressiveness of Keras API.
- Implementing the other CNN architectures is a bit longer, but not much harder.

# Using Pretrained Models from Keras

- In general, you won't have to implement standard models like GoogLeNet or ResNet manually, since pretrained networks are readily available with a single line of code in the `tf.keras.applications` package.
- For example, you can load the ResNet-50 model, pretrained on ImageNet, with the following line of code: <br>
`model = tf.keras.applications.ResNet50(weights="imagenet")`
- This will create a ResNet-50 model and download weights pretrained on the ImageNet dataset.
- To use it, you first ensure that the images have the right size. A ResNet-50 model expects $224 times 224$-pixel images (other models may expect other sizes, such as $299 \times 299$), so you should use Keras's `Resizing` layer (introduced in chapter 13) to resize two sample images (after cropping them to the target aspect ratio).
- The pretrained model assumes that the images are preprocessed in a specific way. In some cases, they may expect that the inputs to be scaled from 0 to 1, or from -1 to 1, and so on.
- Each model provides a `preprocess_input()` function that you can use to preprocess your images.
- These functions assumes that the original pixel values range from 0 to 255, which is the case here.
- You can use the pretrained model to make predictions, just like a regular one.
- As usual, the prediction result is a matrix with one row per image and one column per class (in this case, there are 1,000 classes).
- If you want to display the top K predictions, including the class name and the estimated probability of each predicted class, use the `decode_predictions()` function.
- For each image, it returns an array containing the top K predictions, where each prediction is represented as an array containing the class identifier, its name, and the corresponding confidence score.
- The correct class are palace and dahlia, so the model is correct for the first image but wrong for the second. However, that's because dahlia is nto one of the 1,000 ImageNet classes.
- As you can see, it is very easy to create a pretty good image classifier using a pretrained model.

# Pretrained Models for Transfer Learning

- But what if you want to use an image classifier that are not part of ImageNet? In this case, you may still benefit from the pretrained models by using them to perform transfer learning.
- Also, you may want to build an image classifier but do not have enough data to train it from scratch, then it is still a good idea to reuse the lower layers of a pretrained model, as we discussed in chapter 11.
- For example, let's train a model to classify pictures of flowers, reusing a pretrained Xception model.
- First, we load the flowers dataset using TensorFlow Datasets.
- Note that you can get information about the dataset by setting `with_info=True`. Here, we get the dataset size and the names of the classes.
- Unfortunately, there is only a `"train"` dataset, no test set or validation set, so we need to split the training set. Therefore, we will call `tfds.load()` again, this time taking the first 10% of the dataset for testing, the next 15% for validation, and the remaining 75% for training.
- All three datasets contain individual images.
- We need to batch them, but for this we first need to ensure they all have the same size, or else batching will not work. We can use a `Resizing` layer for this.
- We must also call the `tf.keras.applications.xception.preprocess_input()` function to preprocess the images appropriately for the Xception model.
- Lastly, we'll also add shuffling and prefetching to the training dataset.
- Since the dataset is not very large, a bit of data augmentation will certainly help.
- We create a data augmentation model: during training, it will randomly flip the images horizontally, rotate them a little bit, and tweak the contrast.
- The `tf.keras.preprocessing.image.ImageDataGenerator` class make it easy to load images from disk and augment them in various ways: you can shift each image, rotate it, rescale it, flip it horizontally or vertically, shear it, or apply any transformation function you want to it. This is very convenient for simple projects.
- However, a tf.data pipeline is not much more complicated, and it's generally faster.
- Moreover, if you have a GPU and you include the preprocessing or data augmentation layers inside your model, they will benefit from GPU acceleration during training.
- Now, let's load an Xception model, pretrained on ImageNet. We exclude the top of the network by setting `include_top=False`. This excludes the global average pooling layer and the dense output layer.
- We then add our own global average pooling layer (feeding it the output of the base model), followed by a dense output layer with one unit per class, using the softmax activation function.
- Finally, we wrap all this in a Keras `Model`.
- As explained in chapter 11, it's usually a good idea to freeze the weights of the pretrained layers, at least at the beginning of training.
- Since our model uses the base model's layers directly, rather than the `base_model` object itself, setting `base_model.trainable=False` would have no effect.
- Finally, we can compile the model and start training.
- If available, use the GPU. We still can train without one, but it will be painfully slow.
- After training the model for a few epochs, its validation accuracy should reach a bit over 80% and then stop improving.
- This means that the top layers are now pretty well trained, and we're ready to unfreeze some of the base model's top layers, then continue training.
- For example, we unfreeze layers 56 and above (that's the start of residual units 7 out of 14, as you can see if you list the layer names).
- Remember to compile the model whenever you freeze or unfreeze layers.
- Also make sure to use a much lower learning rate to avoid damaging the pretrained weights.
- This model should reach around 92% accuracy on the test set, in just a few minutes of training (with a GPU).
- If you tune the hyperparameters, lower the learning rate, and train for a bit longer, you should be able to reach 95% to 95%.

# Classification and Localization

- Localizing an object in a picture can be expressed as a regression task, as discussed in chapter 10: to predict a bounding box around the object, a common approach is to predict the horizontal and vertical coordinates of the object's center, as well as its height and width. This means we have four numbers to predict.
- It does not require much change to the model; we just need to add a second output layer with four units (typically on top of the global average pooling layer), and it can be trained using the MSE loss.
- But now we have a problem: the flower dataset does not have bounding boxes around the flowers. So we need to add them ourselves.
- This is often one of the hardest and most costly parts of a machine learning project: getting the labels.
- It's a good idea to spend time looking for the right tools.
- To annotate images with bounding boxes, you may want to use an open source image labeling tooling like VGG Image Annotator, LabelImg, OpenLabeler, or ImgLab, or perhaps a commercial tool like LabelBox or Supervisely.
- You may also want to consider crowdsourcing platforms such as Amazon Mechanical Turk if you have a very large number of images to annotate.
- However, it is quite a lot of work to set up a crowdsourcing platform, prepare the form to be sent to the worker, supervise them, and ensure that the quality of the bounding boxes they produce is good, so make sure it is worth the effort.
- Adriana Kovashka et al. wrote a very practical [paper](https://arxiv.org/abs/1611.02145) about crowdsourcing in computer vision.
- If there are just a few hundred, ro even a couple of thousand images to label, and you don't plan to do this frequently, it may be preferable to do it yourself: with the right tools, it will only take a few days, and you'll also gain a better understanding of your dataset and task.
- Now, let's suppose you've obtained the bounding boxes for every image in the flowers dataset (for now, we assume there is a single bounding box per image).
- You then need to create a dataset whose items will be batches of preprocessed images along with their class labels and their bounding boxes. Each item should be a tuple of the form `(images, (class_labels, bounding_boxes))`.
- The bounding boxes should be normalized so that the horizontal and vertical coordinates, as well as the height and width, all range from 0 to 1.
- Also, it's common to predict the square root of the height and width, instead of the height and width directly: this way, a 10-pixel error for a large bounding box will not be penalized as much as a 10-pixel error for a small bounding box.
- The MSE often works fairly well as a cost function to train the model, but it is not a great metric to evaluate how well the model can predict bounding boxes.
- The most common metric for this task is the *intersection over union* (IoU): the area of overlap between the predicted bounding box and the target bounding box, divided by the area of their union.
![IoU metric for bounding boxes](image-20.png)
- In Keras, it is implemented by the `tf.Keras.metrics.MeanIoU` class.
- Classifying and localizing a single object is fine now, but what about multiple objects (as is often the case in the flower dataset)?

# Object Detection

- The task of classifying and localizing multiple objects in an image is called *object detection*.
- Until a few years ago, a common approach was to take a CNN that was trained to classify and locate a single object roughly centred in the image, then slide this CNN across the image and make predictions at each step.
- The CNN was generally trained to predict not only class probabilities and a bounding box, but also an *objectness score*: this is the estimated probability that the image does indeed contain an object centered near the middle.
- This is a binary classification output; it can be produced by a dense output layer with a single unit, using the sigmoid activation function and trained using the binary cross-entropy loss.
- Instead of an objectness score, sometimes a "no-object" class was added, but itn general it doesn't work as well: the questions "Is an object present" and "what type of object is it?" are best answered separately.
- The sliding-CNN approach is illustrated below:
![Detecting multiple objects by sliding a CNN across the image](image-21.png)
- In this example, the image was chopped into a $5 \times 7$ grid, and we see a CNN - the thick black rectangle - sliding across all $3\times 3$ regions and make predictions at each step.
- In this figure, the CNN has already made predictions for three of these $3\times 3$ regions:
    - When looking at the top-left $3\times 3$ region (centered on the red-shaded grid cell located in the second row and second column), it detect the leftmost rose. Notice that the predicted bounding box exceeds the boundary of this $3 \times 3$ region. This is completely fine: even though the CNN can't see the bottom part of the rose, it was able to make a reasonable guess as to where it might be. It also predicted class probabilities, giving a high probability to the "rose" class. Lastly, it predicted a fairly high objectness score, since the center of the bounding box lies within the central gird cell (in this figure, the objectness score is represented by hte thickness of the bounding box).
    - When looking at the next $3 \times 3$ region, one grid cell to the right (centered on the shaded blue square), it did not detect any flower centered in that region, so it predict a very low objectness score; therefore, the predicted bounding box and class probabilities can be safely ignored. You can see that the predicted bounding box was not good anyway.
    - Finally, when looking at the next $3 \times 3$ region, again one grid cell to the right(centered on the shaded green cell), it detected the rose at the top, although not perfectly: this rose is not well centered within this region, so the predicted objectness score was not very high.
- Sliding the CNN across the whole image this way would give you a total of 15 predicted bounding boxes, organized in a $3 \times 5$ grid, with each bounding box accompanied by its estimated class probabilities and objectness score.
- Since objects can have varying sizes, you may then want to slide the CNN again across larger $4 \times 4$ regions as well, to get even more bounding boxes.
- This technique is straightforward, but as you can see, it will often detect the same object multiple times, at slightly different positions. Some postprocessing is required to get rid of all the unnecessary bounding boxes.
- A common approach is called *non-max suppression*. Here is how it works:
    - First, get rid of all the bounding boxes for which the objectness score is below some threshold: since the CNN believes there's no object at that location, the bounding box is useless.
    - Find all the remaining bounding box with the highest objectness score, and get rid of all the other remaining bounding boxes that overlap with it (e.g., with an IoU greater than 60%). For example, in the figure above, the bounding box that touches this same rose overlap a lot with the max bounding box, so we will get rid of it (although in this example, it would have been removed in the previous step).
    - Repeat step 2 until there is no more bounding boxes to get rid of.
- This simple approach to object detection works great, but it requires running the CNN many times (15 times in our case), so it is quite slow.
- Fortunately, there is a much faster way to slide a CNN across an image: using a *fully convolutional network* (FCN).

## Fully Convolutional Network

- The idea of FCNs was first introduced in a [2015 paper]() by Jonathan Long et al., for semantic segmentation (the task of classifying every pixel in an image according to the class of the object it belongs to).
- The author pointed out that you can replace the dense layers at the top of a CNN with convolutional layers.
- To understand this, let's look at an example: suppose a dense layer with 200 neurons sits on top of a convolutional layer that outputs 100 feature maps, each of size $7 \times 7$ (this is the feature map size, not the kernel size).
- Each neuron will compute a weighted sum of all $100 \times 7 \times 7$ activations form the convolutional layer (plus a bias term).
- Now let's see what happens if we replace the dense layer with a convolutional layer using 200 filters, each of size $7 \times 7$, and with `"valid"` padding.
- This layer will output 200 feature maps, each $1 \times 1$ (since the kernel is exactly the size of the input feature maps and we are using `"valid"` padding).
- In other words, it will output 200 numbers, just like the dense layer's output was a tensor of shape *[batch sie, 200]*, while the convolutional layer will output a tensor of shape *[batch size, 1, 1, 200]*.
- To convert a dense layer to a convolutional layer, the number of filters in the convolutional layer must be equal to the the number of units in the dense layer, the filter size must be equal ot the size of the input feature maps, and you must use `"valid"` padding. The stride may be set to 1 or more, as you will see shortly.
- Why is this important? While a dense layer expects a specific input size (since it has one weight per input feature), a convolutional layer will happily process images of any size (however, it does expects its inputs to have a specific number of channels, since each kernel contains a different set of weights of each input channel).
- Since an FCN contains only convolutional layers (and pooling layers, which have the same property), it can be trained and execute on images of any size.
- For example, suppose we've already trained a CNN for flower classification and localization. It was trained on $224 \times 224$, and it outputs 10 numbers:
    - Outputs 0 to 4 are sent through the softmax activation function, and this gives the class probabilities (one per class).
    - Output 5 is sent through the sigmoid activation function, and this gives the objectness score.
    - Outputs 6 and 7 represents the bounding box's center coordinates; they also go through a sigmoid activation function to ensure they range from 0 to 1.
    - Lastly, outputs 8 and 9 represent the bounding box's height and width; they do not go through any activation function to allow the bounding boxes to extend beyond the borders of the image.
- We can now convert the CNN's dense layers to convolutional layers. In fact, we don'even need to retrain it; we can just copy the weights from the dense layers to the convolutional layers.
- Alternatively, we could have converted the CNN into an FCN before training.
- Now suppose the last convolutional layer before the output layer (also called the bottleneck) layer outputs $7 \times 7$ feature maps when the network is fed a $224 \times 224$ image (see the left side of the figure below)
- If we feed the FCN a $448 \times 448$ image (see the right side of the figure below), the bottleneck layer will now output $14 \times 14$ feature maps.
![The same fully convolutional network processing a small image (left) and a large one (right)](image-22.png)
- Since the dense output was replaced by a convolutional layer using 10 filters of size $7 \times 7$, with `"valid"` padding and stride 1, the output will be composed of 10 feature maps, each of size $8 \times 8$ (since $14 - 7 + 1 = 8$).
- In other words, the FCN will process the whole image only once, and it will output an $8 \times 8$ where each cell contains 10 numbers (5 class probabilities, 1 objectness score, and 4 bounding box coordinates).
- It's exactly like taking the original CNN and sliding a $7 \times 7$ window across this grid; there will be $8 \times 8 = 64$ possible locations for the window, hence 64 predictions.
- However, the FCN approach is much more efficient, since the network only looks at the image once. In fact, *You Only Look Once* (YOLO) is the name of a very popular object detection architecture, which we'll look at next.

## You Only Look Once

- YOLO is a fast and accurate object detection architecture proposed by Joseph Redmon et al. in a [2015 paper](https://arxiv.org/abs/1506.02640).
- It is so fast that it can run in real time on a video, as seen in Redmon's [demo](https://www.youtube.com/watch?v=MPU2HistivI).
- YOLO's architecture is quite similar to the one we just discussed, but with a few important differences. You can find a more detailed explanation [here](https://hackernoon.com/understanding-yolo-f5a74bbc7967).
- For each grid, YOLO only considers objects whose bounding box center lies within that cell. The bounding box coordinates are relative to that cell, where (0, 0) means the top-left corner of the cell and (1, 1) means the bottom right. However, the bounding box's height and width may extend beyond the cell.
- It outputs two bounding boxes for each grid cell (instead of just one), which allows the model to handle cases where two objects are so close to each other that their bounding box centers lie within the same cell. Each bounding box also comes with its own objectness score.
- YOLO also outputs a class probability distribution for each grid cell, predicting 20 class probabilities per grid cell since YOLO was trained on the PASCAL VOC dataset, which contains 20 classes.
- This produces a coarse *class probability map*. Note that the model predicts one class probability distribution per grid cell, not per bounding box.
- However, it's possible to estimate class probabilities for each bounding box during postprocessing, by measuring how well each bounding box matches each class in the class probability map.
- For example, imagine a picture of a person standing in front of a car. There will be two bounding boxes: one large horizontal one for the car, and a smaller vertical one for the person.
- These bounding boxes may have their centers within the same grid cell. So how can we tell which class should be assigned to each bounding box? Across the image, there will a region where the "car" class is dominant, and inside it there will be a smaller region where the "person" class is dominant.
- Hopefully, the car's bounding box will roughly match the "car" region, while the person's bounding box will roughly match the "person" region: this will allow the correct class to be assigned to each bounding box.
- YOLO was originally developed using Darknet, an open source deep learning framework initially developed in C by Joseph Redmon, but it was soon ported to TensorFlow, Keras, PyTorch, and more.
- It was continuously improved over years, with YOLOv2, YOLOv3, and YOLO9000 (again by Joseph Redmon et al.), YOLOv4 (by Alexey Bochkovskiy et al.), YOLOv5 (by Glenn Jocher), and PP-YOLO (by Xiang Long et al.).
- Each version brought some impressive improvements in speed and accuracy, using a variety of techniques:
    - For example, YOLOv3 boosted accuracy in part thanks to *anchor priors*, exploiting the fact that som bounding box shapes are more likely than others, depending on the class (e.g., people tend to have vertical bounding boxes, while cars usually don't).
    - They also increased the number of bounding boxes per gird cell, they trained on different datasets with many more classes (up to 9,000 classes organized in a hierarchy in the case of YOLO9000)
    - They added skip connections to recover some of the spatial resolution that is lost in the CNN (we will discuss this shortly in semantic segmentation).
    - And much more.
- There are many variants of these models too, such as YOLOv4-tiny, which is optimized to be trained on less powerful machines and can run extremely fast (at over 1,000 frames per second!) but with a slightly lower *mean average precision* (mAP).
- Many object detection models are available on TensorFlow Hub, often with pretrained weights, such as YOLOv5, [SSD](https://arxiv.org/abs/1512.02325), [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [EfficientDet](https://arxiv.org/abs/1911.09070).
- SSD and EfficientDet are "look once" detection models, similar to YOLO.
- EfficientDet is base on the EfficientNet convolutional architecture.
- Faster R-CNN is more complex: the image first goes through a CNN, then the output is passed to a *region proposal network* (RPN) that proposes bounding boxes that are most likely to contain an object; a classifier is then run for each bounding box, based on the cropped output of the CNN.
- The author's recommendation place to learn these models is the TensorFlow Hub's [object detection tutorial](https://www.tensorflow.org/hub/tutorials/tf2_object_detection).
- So far, we've only considered detecting objects in single images. But what videos? Objects must not only be detected in each frame, they must also be tracked over time.

# Object Tracking

- Object tracking is a challenge task: objects move, they may grow or shrink as they get closer or further away from the camera, their appearance may change as they turn around or move to different lighting conditions or backgrounds, they may be temporarily occluded by other objects, and so on.
- One of the most popular object tracking systems is [DeepSORT](https://arxiv.org/abs/1703.07402).
- It is based on a combination of classical algorithms and deep learning:
    - It uses *Kalman filters* to estimate the most likely current position of an object given prior detections, and assuming that objects tend to move at a constant speed.
    - It uses a deep learning model to measure the resemblance between new detections and existing tracked objects.
    - Lastly, it uses the *Hungarian algorithm* to map new detections to existing tracked objects (or new tracked objects): this algorithm efficiently finds the combination of mappings that minimizes the distance between the detections nad the predicted position of tracked objects, while also minimizing the appearance discrepancy.
- For example, imagine a blue ball that just bounced off a red ball traveling in the opposite direction.
- Based on the previous positions of the balls, the Kalman filter will predict that the balls will go through each other: in fact, it assumes that the objects move at a constant speed, so it will not expect the bounce.
- If the Hungarian algorithm only considers positions, it would happily map the new detections to the wrong balls, as if they had just gone through each other and swapped colors.
- But thanks to the resemblance measure, the Hungarian algorithm will notice the problem. Assuming the balls are not too similar, the algorithm will map the new detections to the correct balls.
- There are a few DeepSORT implementations available on GitHub, including a TensorFlow [implementation](https://github.com/theAIGuysCode/yolov4-deepsort) of YOLOv4 + DeepSORT.
- So far, we have located objects using bounding boxes.
- This is often sufficient, but sometimes you need to locate objects with much more precision - for example, to remove the background behind a person during a video conference call.

# Semantic Segmentation

- In *semantic segmentation*, each pixel is classified according to the class of the object it belongs to (e.g., road, car, pedestrian, building, etc.), as shown below:
![Semantic segmentation](image-23.png)
- Note that different objects of the same class are not distinguished.
- For example, all the bicycles on the right side of the segmented image ends up as one big blob of pixels.
- The main difficulty in this task is that when the images go through a regular CNN, they gradually loses their spatial resolution (due to the layers with strides greater than 1); so, a regular CNN may end up knowing that there's a person somewhere in the bottom left of the image, but it will not be much more precise than that.
- Just like for object detection, there are many different approaches to tackle this problem, some are quite complex.
- However, a fairly simple solution was proposed in the 2015 paper by Jonathan Long et al. we mentioned earlier on fully convolutional networks.
- The authors starts by taking a pretrained CNN and turning it into an FCN.
- The CNN applies an overall stride of 32 to the input image (i.e., if you add up all the strides greater than 1), meaning the last output features maps are 32 times smaller than the input image. 
- This is clearly to coarse, so they add a single *upsampling layer* that multiplies the resolution by 32.
- There are several solutions available for upsampling (increasing the size of an image), such as bilinear interpolation, but that only works reasonably well up to x4 or x8.
- Instead, they used a *transposed convolutional layer*: this is equivalent to first stretching the image by inserting empty rows and columns (full of zeros), then performing a regular convolution.
![Upsampling using a transposed convolutional layer](image-24.png)
- Alternatively some people prefer to think of it as a regular convolutional layer that uses fractional strides (e.g., the stride is 1/2 in this case).
- The transposed convolutional layer cna be initialized to perform something close to linear interpolation, but since it is a trainable layer, it will learn to do better during training. 
- In Keras, you can use the `Conv2DTranspose` layer.
- In a transposed convolutional layer, the stride define how much the input will be stretched, not the size of the filter steps, so the larger the stride, the larger the output (unlike for convolutional layers or pooling layers).
- Using transposed convolutional layers for upsampling is OK, but still too imprecise.
- To do better, Long et al. added skip connections from lower layers: for example, they upsampled the output image by a factor of 2 (instead of 32), and they added the output of a lower layer that had this double resolution.
- Then they upsampled the result by a factor of 16, leading to total upsampling factor of 32.
![Skip layers recover some spatial resolution from lower layers](image-25.png)
- This recovered some of the spatial resolution that was lost in earlier pooling layers.
- In their bets architecture, they used a second similar skip connection to recover even finer details from an even lower layer.
- In short, the output of the original CNN goes through the following extra steps: upsample x2, add the output of a lower layer (of the appropriate scale), upsample x2, add the output of an even lower layer, and finally upsample x8.
- It is even possible to scale up beyond the size of the original image: this can be used to increase the solution of an image, which is a technique called *super-resolution*.
- *Instance segmentation* is similar to semantic segmentation, but instead of merging all objects of the same class into one big lump, each object is distinguished from each others (e.g., it identifies each individual bicycle).
- For example, the *Mask R-CNN* architecture, proposed in a [2017 paper]() by Kaiming He et al., extends the Faster R-CNN model by additionally producing a pixel mask for each bounding box.
- So, not only you get a bounding box around each object, with a set of estimated class probabilities, but you also get a pixel mask that locates pixels in the bounding box that belong to the object.
- This model is available on TensorFlow Hub, pretrained on the COCO 2017 dataset.
- This field is moving fast, though so if you want to try the latest and greatest models, please check out the state-of-the-art section [https://paperswithcode.com](https://paperswithcode.com).
- As you can see, the field of deep computer vision is vast and fast-paced, with all sorts of architectures popping up every year.
- Almost all of them are based on convolutional neural networks, but since 2020, another neural networks architecture has entered the computer vision space: transformers (which we will discussed in chapter 16).
- The progress made over the last decade has been astounding, and researchers are now focusing on harder and harder problems problems, some of them are:
    - *adversarial learning*: attempts to make the network more resistant to images designed to fool it.
    - *explainability*: understanding  why the network makes specific classification.
    - realistic *image generation*: we will come back to this in chapter 17.
    - *single-shot learning*: also named one-shot learning, a system that can recognize an object after it has seen it just once.
    - *zero-shot learning*: a system that can recognize how to interact with an object it has never seen before.
    - predicting the next frames in a video
    - combing text and images tasks.