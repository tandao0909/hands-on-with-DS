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
- For example, it could learn multiple filters each detecting a different rotation of the same pattern, and the depth-wise max pooing layer would ensure that the output is the same, regardless of the rotation.
- The CNN could similarly learn to be invariant to anything: brightness, skew, color, and so on.
- Keras doesn't include a depth-wise max pooling layer, but you can find a custom layer do just this in the learning notebook.
- This layer reshapes its inputs to split the channels into groups of the desired size (`pool_size`), then it uses `tf.reduce_max()` to compute the max of each group.
- This implementation assumes that the stride is equal to the pool size, which is generally what you want.
- Alternatively, you could use TensorFLow's `tf.nn.max_pool()` operation, and wrap it in a `Lambda` layer to use it inside a Keras model, but sadly, this operation does not implement depth-wise pooling for the GPU, only the CPU.
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