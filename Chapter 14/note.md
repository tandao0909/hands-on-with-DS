- Although IBM's Deep Blue supercomputer beat the chess world champion Garry Kasparov back in 1996, not until recently that computers were able to reliably perform seemingly trivial task, such as detecting a kitten in a picture or recognizing spoken words.
- The reason why these tasks are so trivial to us human is because evolution helps us a lot. Billions of years allows us to have such specialized visual, auditory, and other sensor modules in our brains
- These automatically takes place in the process of inferring the information in the real world, before it reaches out conscious.
- However, this is not trivial at all, and we must look at how our sensory modules works to understand it.
- *Convolutional neural networks* emerged from the study of the brain visual cortex, and they have been used in computer image recognition since the 1980s.
- Over the last 10 years, thanks to the increase in computational power, the amount of available training data, and the tricks represented in chapter 11 in training deep nets, CNNs have been able to achieve superhuman performance o some complex visual tasks: image search services, self-driving cars, automatic video classification systems, and more.
- Moreover, CNNs are not restricted to visual perception: they are also success at many other tasks, including voice recognition and natural language processing.
- We will only focus on visual application in this chapter.

# The Architecture of the Visual Cortex

- David H. Hubel and Torsten Wiesel performed a series of experiments on cats in [1958]() and [1959]() (and a [few years later on monkeys]()), giving crucial insights into the structure of the visual cortex.
- In particular, they showed that many neurons in the visual cortex have a small *local receptive field*, meaning they react only to a visual stimuli located in a limited region of the visual filed.
- The receptive fields of different neurons may overlap, and together they tile the whole visual field.
- Moreover, the authors showed that some neurons react only to images of horizontal lines, while others mau only react to dashed lines (two neurons may have the same receptive field but react to different line orientation).
- They also noticed that some neurons have larger receptive fields, and they react to more complex patterns that are combinations of the lower-level patterns.
- These observations led to the idea that the higher-level neurons are based on the outputs of neighboring lower-level patterns.
- This powerful architecture is able to detect all sorts of complex patterns in any area of visual field.
- These studies of the visual cortex inspired the [neocognition](), introduced in 1980, which gradually evolved into what we now call convolutional neural networks.
- An important milestone was a [1998 paper]() by Yann LeCun et al. that introduced the famous *LeNet-5* architecture, which became widely used by banks to recognize hand-written digits on checks.
- This architecture has some building blocks that you already know, such as fully connected layers and the sigmoid activation function, but it also introduced two new building blocks: *convolutional layers* and *pooling layers*.
- Why don't we use a deep neural networks with fully connected layers for image recognition tasks?
    - Although this works fine for small images (e.g., MNIST), it won't be great if we talk about larger images, because of the huge number of parameters it requires.
    - For example, a $100 \times 100$-pixel image has 10,000 pixels, and if the first layer has just 1,000 neurons (which already severely restricts the amount of information transmitted to the next layer), this means a total of 10 million connections. And that's just the first layer.
    - CNNs solve this problem using partially connected layers and weight sharing.

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
![](image-2.png)
- Thus, a layer full of neurons using the same filter outputs a *feature map*, which highlights the areas in an image that activate the filter the most.
- You don't have to find the appropriate feature maps yourself: instead, during training the convolutional layer will automatically learn the most useful filters for its task, and the layers above will learn to combine them into more complex patterns.