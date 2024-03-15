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