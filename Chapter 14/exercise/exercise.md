1. Some of the advantages of a CNN over a fully connected DNN for image classification:
- Because consecutive layers are just partially connected and it heavily reused weights, a CNN uses dramatically fewer parameters (up to some order of magnitudes), hence speed up the training process and reduce the RAM needed, reduce the risk of overfitting, and requires much less training data.
- Because for each feature map, every neuron would use the same matrix weight, this allows a feature can be learned across the whole image. In contrast, if a DNN detects a feature, it can only detect that feature in that specific location. Since images typically made up of many repetitive features, CNNs are able to generalize much better than DNN for image processing task such as image classification, using much less training data.
- Finally, a CNN don't know how nearby pixels are organized, which makes it excel at detecting small features. However, a DNN knows the whole image at once, so it assumes small features in a area is related to areas far way (which is just due to sheer luck). Lower layers typically identifies low-level features, while higher-level features are combined by many lower-level features. This works great with most natural images, giving CNNs an advantage over DNN.
2.
a. Here is a breakdown of how I find the number of parameters:
- The first convolutional layer has $3 \times 3$ kernel and 3 input channels (red, green and blue), each feature map has the kernel size times the input feature maps (the input in this first layer) plus 1, which is the bias term. So that's $3 \times 3 \times 3 + 1 = 28$ parameters per feature map. There are 100 feature maps in this layer, so there are $28 \times 100 = 2,800$ parameters in this first layer.
- The second convolutional layer has $3 \times 3$ kernel and 100 input feature maps, so in each feature maps, we need $3 \times 3 \times 3 \times 100 + 1 = 901$. This layer has 200 feature maps, hence there are $901 \times 200 = 180,200$ parameters in this second layer.
- The third convolutional layer has $3 \times 3$ kernel and 200 feature maps, so in each feature maps, we need $3 \times 3 \times 200 + 1 = 1,801$. This layer has 400 feature maps, hence there are $1,801 \times 400 = 720,400$ parameters.
- So there are a total of $903,400$ parameters in the CNN.
b. Here is a breakdown of how I find the number of RAM required to make a single prediction for a single instance:
- Now, we compute what is the size of the feature map in each layer. We used a stride of 2, and the `"same"` padding, so the feature map's size is divided by 2 after each layer, rounding up if necessary. The input images is $200\times 300$ pixels, so the first layer's feature map are $100 \times 150$, the second's are $50 \times 75$, and the last are $25 \times 38$.
- Since 32 bits is 4 bytes and the first layer has 100 feature maps, this first layer takes $4 \times 100 \times 150 \times 100 = 6,000,000$ bytes, which is 6 MB.
- Similarly, the second layer has 200 feature maps, so this layer takes $4 \times 50 \times 75 times 200 = 3,000,000$ bytes, which is 3 MB, and the third layer has 400 features maps, so this layer takes $4 \times 25 \times 38 times 400 = 1,520,000$ bytes, which is 1,52 MB.
- However, if the neural network is optimized, we just need to hold at most 2 layers in the memory (when we compute one layer, we need the whole input layer to be stored in the memory, but when the computation is completed, we can release the memory of the input layer), we need at most $6 + 3 = 9$ MB.
- But we need to store the whole CNN's parameters in the memory as well. As we did earlier, there are $903,400$ parameters in total, all of them are 32-bit floats, so we need an additional of $903,400 \times 4 = 3,613,600$ bytes, or about 3.6 MB.
- In total, we need (at least) $12,613,600$ bytes, or about 12,6 MB, of RAM to make prediction for a single instance.
c. Here is the breakdown of how to find the memory needed when training on a mini-batch of 50 images (now we count in megabytes, instead of bytes, for the sake of brevity):
- When training, all the intermediate values computed in each layer must be reserved for the reserve pass. 
- So we need to compute the memory needed by all layers when training an instance, and multiply it by 50, which is $(6 + 3 + 1.52) \times 50 = 526$ MB.
- Add the memory needed for the images $50 \times 200 \times 300 \times 3 = 9,000,000$ bytes, which is 9 MB, and for the model's parameters, which is about 3.6 MB (computed earlier).
- Yes, we need memory for the gradients, but we now neglect this, because this is already complex enough, and the memory required will be decreased when the gradients flow down the network during the reverse pass.
- So in the end, we need $526 + 9 + 3.6 = 565.6$ MB of RAM, and this is the optimistic bare minimum.
3. Here are five things to try if the GPU runs out of memory while training a CNN:
- Reduce the mini-batch size.
- Reduce the dimensionality using a larger stride.
- Remove one or more layers.
- Using 16-bit instead of 32-bit, or even 8-bit and 4-bit!
- Distributing computation across many devices.
4. We would like to use a max pooling layer to reduce the dimensionality of the dataset, without having anymore parameters, while a convolutional layer would have parameters.
5.
- A local response normalization layer makes the neurons with strong activations inhibits other neurons at the same location in neighboring feature maps.
- This encourages different feature maps to specialize and pushes them apart, forcing them to explore a wider range of features.
- It is typically used in the lower layers to have a large pool of low-level features that the upper layers can build upon.
6. Here I name the main innovations in a specific architecture, compared to other architectures before it:
- AlexNet: the first one to stack a convolutional layer on top of another convolutional layer. It also introduced the local response normalization layer. It is much larger and deeper than LeNet-5, too.
- GoogLeNet: introduce the inception module, allows it to have a much deeper CNN, with less parameters.
- ResNet: introduce the use of skip connections, allows it to go beyond 100 layers.
- SENet: the SE block, which is a two-layer dense network, after every inception module in the inception model or after every residual unit in a ResNet to recalibrate the relative importance of feature maps.
- Xception: introduce the depthwise separable convolution layer, which look at spatial patterns and depthwise patterns differently.
- DifferentNet: the compound scaling method, allows the model to scale efficiently to a larger computing budget.
7. 
- The fully convolutional network are neural networks composed exclusively of convolutional and pooling layers. They can efficient process images of any width and height (at least about the minimum size). They are most useful for object detection and semantic segmentation because they only need to look at the image once (instead of having to run the CNN multiple times on different parts of the image).
- If you have some dense layers on top, you can try convert them to CNN to create an FCN: just replace the lowest dense layer with a convolutional layer with a kernel size equal to the size of the layer's input, with one filter per neuron in the dense layer, and using `"valid"` padding.
- Generally the stride should be 1, but you can ste it to a higher value if you want.
- The activation function should be the same as the dense layer.
- The other dense layers should be converted the same way, but using $1\times 1$ filter.
- We actually can convert a whole trained CNN this way by reshaping the dense layer's weight matrices.
8. The main technical difficulty of sematic segmentation are:
- A lot of spatial information is lost in the CNN as the signal flows through each layer, especially pooling layers and convolutional layers that have strides larger than 1.
- This spatial information need to be restored somehow to accurately predict the class of each pixel.