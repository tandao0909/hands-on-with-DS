- Up until now, we've used only TensorFlow's high-level API, Keras, but it already got us pretty far, as we have:
    - built various neural network architecture, including regression and classification nets, Wide & Deep nets, and self-normalizing nets
    - used all sort of techniques, such as batch normalization, dropout, and learning rate schedules.
- In fact, 95% of the use cases you will encounter will not require anything other than Keras (tf.data, in chapter 13).
- But here, we'll take a deep dive into TensorFlow and take a look at its lower-level [Python API](https://www.tensorflow.org/api_docs/python/tf).
- This will be useful when you need extra control to write custom loss functions, custom metrics, layers, models, initializers, regularizers, weight constraints, and more.
- You may even need to fully control the training loop itself; for example, to apply special transformations or constraints to the gradients (beyond just clipping them) or to use multiple optimizers for different parts of the network.
- We will also look at how to boost your custom models and training algorithms using TensorFlow's automatic graph generation feature.

## A Quick Tour of TensorFlow

- As you know, TensorFlow is a powerful library for numerical computation, particularly well suited and fine-tuned for large-scale machine learning (but you can use it for anything else that requires heavy computations).
- TensorFlow is used for all sort of machine learning tasks, such as image classification, natural language processing, recommender systems, and time series forecasting.
- Here's a summary of what TensorFlow offers:
    - Its core is very similar to NumPy, but with GPU support.
    - It supports distributed computing (across multiple devices and servers).
    - It included a kind of just-in-time (JIT) complier that allows it to optimize computations for speed and memory usages. It works by extracting the *computation graph* from a Python function, optimizing it (e.g., by pruning unused nodes), and running it efficiently (e.g., by automatically running independent operations in parallel).
    - Computation graphs can be exported to a portable format, so you can train a TensorFlow model in one environment (e.g., using Python in Linux) and run it in another (e.g., using Java on an Android device).
    - It implements reverse-mode autodiff (see chapter 10) and provides some excellent optimizers, such as RMSProp and Nadam (see chapter 11), so you can easily minimize all sorts of loss functions.
- TensorFlow offers many more features built on top of these core features:
    - The most important is of course Keras.
    - Data loading and preprocessing operations (tf.data, tf.io, etc.)
    - Signal processing operations (tf.signal)
    - And more (see the book for a overview of TensorFlow's Python API).
- We will cover many of the packages and functions of the TensorFlow API, but it's also impossible to cover them all, so you should really take some time to browse though the API, as it is quite rich and well documented.
- At the lowest level, each TensorFlow operation is implemented using highly efficient C++ code.
- Many operations have multiple implementations called *kernels*: each kernel is dedicated to a specific device type, such as CPUs, GPUs, or even TPUs (*tensor precessing units*).
- As you may know, GPUs can dramatically speed up computations by splitting into many smaller chunks and running them in parallel across many GPU threads. TPUs are even faster: they are custom ASIC chips built specifically for deep learning operations.
- You can find the TensorFlow's architecture in the book.
- Most of the time, your code will use the high-level APIs (especially Keras and tf.data), but when you need more flexibility, you will use the lower-level Python API, handling tensors directly.
- In any case, TensorFlow's execution engine will take care of running the operations efficiently, even across multiple devices and machines if you tell it to.
- TensorFLow runs not only on Windows, Linux and macOS, bu also on mobile devices (using *TensorFlow Lite*), including both iOS and Android.
- There are APIs for other languages as well: There are C++, Java, Swift APIs. There is even a JavaScript implementation called *TensorFlow.js* that makes it possible to run you models directly in your browser.
- There's more to TensorFlow than the library itself. 
- TensorFlow is at the center of an extensive ecosystem of libraries:
    - First, there's TensorBoard for visualization (see chapter 10).
    - Next, there's [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx), which is a set of libraries built by Google to productionize TensorFlow projects: it includes tool for data validation, preprocessing, model analysis, and serving (with TF Serving, see chapter 19).
    - Google's *TensorFlow Hub* provides a way to easily download and reuse pretrained neural networks.
    - TensorFlow's *model garden* offers many neural network architectures, some of them pretrained.
    - Check out the [TensorFlow Resources] and [the awesome-tensorflow project](ttps://github.com/jtoy/awesome-tensorflow) for more TensorFLow-based projects.
    - On GitHub, you can find hundreds of TensorFlow projects, so it's often easy to find existing code for whatever you are trying to do.
- More and more ML papers are released along with their implementations, and sometimes even with pretrained models. Checkout [https://paperswithcode.com](https://paperswithcode.com) to easily find them.
- Last but not least, TensorFlow has a core team, as well as a large community to improving it.
- To ask technical questions, you should use [https://stackoverflow.com](https://stackoverflow.com) and tag your question with *tensorflow* and *python*.
- For general discussions, join the [TensorFlow Forum](https://discuss.tensorflow.org).

## Using TensorFlow like NumPy

- TensorFlow's API revolves around *tensors*, which flows from operation to operation, hence the name *TensorFlow*.
- A tensor is very similar to NumPy's `ndarray`: it is usually a multidimensional array, but it can hold a scalar (a single number).
- These tensors will be important when we create custom functions, custom metrics, custom layers, and more.

### Tensors and Operations

- You can create a tensor using `tf.constant()`.
- Just like a `ndarray`, a `tf.Tensor` has a shape and a data type (`dtype`).
- Indexing works much like NumPy.
- Most importantly, all kinds of tensor operations are available.
- Note that writing `t + 10` is equivalent to calling `tf.add(t, 10)`. In fact, Python calls the magic method `t.__add__(10)`, which just calls `tf.add(t, 10)`.
- Other operators, like - and *, are also supported.
- The @ operator was added in Python 3.5, for matrix multiplication: it is equivalent to calling the *tf.matmul()* function.
- Many functions and class have aliases. For example, `tf.add()` and `tf.math.add()` are the same function. This allows TensorFlow to have concise names for the most common operations while preserving well-organized packages.
- Note that `tf.math.log()` is commonly used, but does not have a `tf.log()` alias, as it might be confused with logging.
- A tensor can also hold a scalar value. IN this case, the shape is empty.
- The Keras API has its own low-level API, located in `tf.keras.backend`.
- This package is usually imported as `K`, for conciseness.
- It used to include functions like `K.square()`, `K.exp()`, and `K.sqrt()`, which you may run across in already existing code: this was useful to write to portable code back when Keras supported multiple backends, but now Keras is TensorFlow-only, you should call TensorFlow's low-level API directly.
- You can find all the basic math operations that you need (`tf.add()`, `tf.multiply()`, `tf.square()`, `tf.exp()`, `tf.sqrt()`, etc.) and most operations that you can find in NumPy (e.g., `tf.reshape()`, `tf.squeeze()`, `tf.tile()`).
- Some functions have a different name than in NumPy; for instance, `tf.reduce_mean()`, `tf.reduce_sum()`,`tf.reduce_max()`, and `tf.math.log()` are the equivalent of `np.mean()`, `np.sum()`, `np.max()`, and `np.log()`.
- When the name differs, there is usually a good reason for it.
- For example, in TensorFlow, you must write `tf.transpose(t)`; you cannot just write `t.T` TensorFlow, a new tensor os created with its own copy of the transposed data, while in NumPy, `t.T` is just a transposed view on the same data.
- Similarly, the `tf.reduce_sum()` operation is named this way because its GPU kernel (i.e., GPU implementation) uses a reduce algorithm that does not guarantee the order in which the elements are added: because 32-bit floats have limited precision, the result may change ever so slightly every time you call this operation. The same is true for `tf.reduce_mean()` (but of course `tf.reduce_max()` is deterministic).

### Tensors and NumPy

- Tensors play nice with NumPy: you can create a tensor from a NumPy array, and vice versa.
- You can even apply TensorFlow operations to NumPy arrays and NumPy operations to tensors.
- Notice that NumPy uses 64-bits precision by default, while TensorFlow uses 32-bit. This is because 32-bits precisions is generally more than enough for neural networks, plus it runs faster and uses less RAM.
- So when you create a tensor from a NumPy array, make sure to set `dtype=tf.float32`.

### Type Conversions

- Type conversions can significantly hurt performance, and they can easily go unnoticed when they are done automatically.
- To avoid this, TensorFlow does not perform any type conversions automatically: it just raises an exception if you try to execute if you try to execute an operation on tensors with incompatible types.
- For example, you cannot add a float tensor and an integer tensor, and you cannot even add a 32-bit float and a 64-bit float.
- Of course, you can use `tf.cast()` when you really need to convert types.

### Variables

- The `tf.Tensor` values we've seen so far are immutable: we cannot modify them.
- This means we cannot use regular tensors to implement weights in a neural network, since they need to be tweaked by backpropagation.
- Plus, other parameters may also need to change over time (e.g., a momentum optimizer keeps track of past gradients).
- What we need is a `tf.Variable`.
- A `tf.Variable` acts much like a `tf.Tensor`: you can perform the same operations with it, it plays nicely with NumPy, and is picky with types as well.
- But, it can be modified in place using the `assign()` method (or `assign_add()` or `assign_sub()`, which increment or decrement the variable by the given value).
- You can also modify individual cells (or slices), by using the cell's (or slice's) `assign()` method or by using the `scatter_update()` or `scatter_nd_update()` method.
- Direct assignment will not work.
- In practice, you will rarely have to create variables manually; Keras provides an `add_weight()` method that will take care of it for you, as you will see shortly.
- Moreover, model parameters will generally be updated directly by the optimizers, so you will rarely need to update variables manually.

### Other Data Structures

- TensorFlow supports several other data structures. Here I list all of them.

#### String tensors

- Are regular tensors of type `tf.string`. These represents byte strings, not Unicode strings, so of you create a string tensor using a Unicode string (e.g., a regular Python 3 string like `"café"`), then it will get encoded to UTF-8 automatically (e.g., `b"af\xc3\xa9"`).
- Alteratively, you represent Unicode strings using tensors of type `tf.int32`, where each item represents a Unicode point (e.g., `[99, 97, 102, 233]`).
- It's important to note that a `tf.string` is atomic, meaning that its length odes not appear in the tensor's shape. Once you convert it to a Unicode tensor (i.e., a tensors of type `tf.int32` holding Unicode points), the length will appear in the shape.
- The `tf.strings` package contains operations for byte strings and Unicode strings, such as `length()` to count the number of bits in a byte string (or the number of code points if you set `unit="UTF8_CHAR"`), `unicode_decode()` to convert a Unicode string tensor (i.e., int32 tensor) to a byte string tensor, and `unicode_decode()` to do the reverse.
- You can also manipulate tensors containing multiple strings.

#### Ragged tensors

- A special kind of tensors, represents lists of tensors, all of the same rank and data type, but varying sizes.
- The dimensions along which the tensor sizes vary are called the *ragged dimensions*.
- In all ragged tensors, the first dimensions is always a regular dimension (also called a *uniform dimension*).
- A slice of a ragged tensor is also a ragged tensor.
- The `tf'ragged` package contains operations for ragged tensors.
- The way we concatenate two ragged tensors in unusual, check out the learning notebook.
- If you call the `to_tensor()` method, the ragged tensor gets converted to a regular tensor by padding shorter tensors with zero values to get tensors of equal lengths. You can change the default value using the `default_value` argument.
- Many TensorFlow operations support ragged tensors. Check out the documentation of `tf.RaggedTensor` class for the full list.

#### Sparse Tensors

- Tensor can also efficiently represent *sparse tensors* (i.e., tensors containing mostly zeros).
- Just create a `tf.SparseTensor`, specifying the indices and values of the nonzero elements and the tensor's shape. The indices must be listed in "reading order" (from left ro right, top to bottom).
- If you're unsure, just use `tf.sparse.reorder()`.
- You can convert a sparse tensor to a dense tensor (i.e., a regular tensor) using `tf.sparse_to_dense()`.
- Note that sparse tensors do not support as much operations as dense tensors. For example, you can multiple a sparse tensor with a scalar to get a new sparse tensor, but you cannot add a scalar value to a sparse tensor, as this would not create a sparse tensor.

#### Tensor Arrays

- A `tf.TensorArray` represents a list of tensors. This can be handy in dynamic models containing loops, to accumulate results and later compute some statistics.
- You can write or read tensors at any location in the array.
- By default, reading an item also replaces it with a tensor of the same shape but full of zeros. You can set `clear_after_read` to `False` if you don't want this.
- When you write to te array, you must assign the output back to the array as well, as shown in the learning notebook. If you don't, although your code will work in eager mode, it will break in graph mode.
- By default, a `TensorArray` has a fixed size that is set upon creation. Alternatively, you can set `size=0` and `dynamic_size=True` to let the array grow automatically when needed.
- However, this will hinder performance, so if you know the `size` in advance, better use a fixed-size array.
- You must also specify the `dtype`, and all elements must have the same shape as the first one written to the array.
- You can stack all the items into a regular tensor by calling the `stack()` method.

#### Sets

- TensorFlow supports sets of integers or strings (but not floats).
- It represents sets using regular tensors.
- Note that the tensor must have at least two dimensions, and the sets must be in the last dimension.
- The `tf.sets` package contains several functions to manipulate sets. 
- For example, we can create two sets and compute their union (the result is however a sparse tensor, so we call `to_dense()` to display it).
- You can also compute the union of multiple pairs of sets simultaneously. If some sets are shorter than others, you must pad them with a padding value, such as 0.
- If you prefer using a different padding value, such as -1, then you must set `default_value=-1` (or your preferred value) when calling `to_dense()`.
- The default `default_value` is 0, so when dealing with string sets, you must set this parameter (e.g., to an empty string).
- Other available functions in `tf.sets` are `difference()`, `intersection()`, and `size()`, which are self-explanatory.
- If you want to check whether a set contains some given values, you can compute the intersection of that set and the values. 
- If you want to add some values to a set, you can compute the union of the set and the values.

#### Queues

- TensorFlow implements several types of queues in `tf.queue` package.
- they used to be very important when implementing efficient data loading and preprocessing pipelines, but the tf.data API had essentially made them useless (expect perhaps in some rare cases) because it is much simpler to use and provides all the tools you need to build efficient pipelines. For the sake of completeness, we will also talk about them.
- The simplest kind of queue is the first-in, first-out (FIFO) queue. To build it, you need to specify the maximum number of records it can contain.
- Moreover, each record is a tuple of tensors, so you must specify the type of each tensor, and optionally their shapes.
- It is also possible to enqueue and dequeue multiple records at once using `enqueue_many()` and `dequeue_many()` (to use `dequeue_many()`, you must specify the `shapes` argument when you create the queue).
- Other queue types include:
    - `PaddingFIFOQueue`: Same as `FIFOQueue`, but its `dequeue_many()` method supports dequeueing multiple records of different shapes. It automatically pads the shortest records to ensure all the records in the batch have the same shape.
    - `PriorityQueue`: A queue that dequeues records in a prioritized order. The priority must be 64-bit integer included as the first element as the first element of each record. Surprisingly, records with lower priority will be dequeued first. Records with the same priority will be dequeued in FIFO order.
    - `RandomShuffleQueue`: A queue whose records are dequeued in random order. This was useful to implement a shuffle buffer before tf.data existed.
- If a queue is already full and you try enqueue another record, the `enqueue()` method will freeze until a record is dequeued by another thread.
- Similarly, if a queue is empty and you try dequeue a record, the `dequeue()` method will freeze until a record is pushed to the queue by another thread.

# Customizing Models and Training Algorithms

## Custom Loss Functions

- Suppose you want to train a regression model, but your training set is a bit noisy.
- Of course, you start by trying to clean the dataset up by removing or fixing the outliers, but that turns out to be insufficient; the dataset is still noisy.
- Which loss function should you use?
    - The mean squared error might penalize large errors too much and make your model too pessimistic, make it imprecise.
    - The mean absolute error would not penalize outliers as much, but training might take a long time to converge, and the trained model might not be ver precise.
    - This is a good time to use Huber loss, instead of good old MSE.
- The Huber loss is available in Keras (just use `tf.keras.losses.Huber`), but let's pretend it's not there.
- To implement it, just create a function that takes the labels and the model's predictions as arguments, and uses TensorFlow operations to compute a tensor containing all the losses (one per sample).
- For better performance, you should use a vectorized implementation, as in the learning notebook.
- Moreover, if you want to benefit from TensorFlow's graph optimization features, you should use only TensorFlow operations.
- You can now use you custom Huber loss function when you compile the Keras model, and train as usual.
- For each batch, Keras will call the loss function to compute the loss, then it will use reverse-mode autodiff to compute the gradients of the loss with regard to all model's trainable parameters, and finally it will perform a gradient descent step.
- Moreover, it will keep track of the total loss since the beginning of the epoch, and it will display the mean loss.

## Saving and Loading Models that contain Custom Components

- Saving a model containing a custom loss function works fine, but when you load it, you'll need to provide dictionary that maps the function name to the actual function.
- More generally, if you load a model containing custom objects, you need to map the names to the objects using a dictionary.
- If you decorate the `huber_fn()` function with `@tf.keras.utils.register_keras_serializable()`, it will automatically be available to the `load_model()` function: there's no need to include it in the `custom_object` dictionary.
- If the object is a class, it must implement the `get_config()` method.
- With our current implementation, any value in the threshold between -1 and 1 is considered "small". But what if you want a different threshold?
- A solution is to create a function that create a configured loss function.
- Unfortunately, when you save the model, the `threshold` will not be saved.
- This means that you will have to specify the `threshold` value when loading the model. 
- Note that we use `"huber_fn"`, which is the name of the loss function we want Keras to use. We don't have to give Keras the name of the function that created it.
- You can solve this by creating a subclass of the `tf.keras.losses.Loss` class, and then implementing its `get_config()` method.
- Let's walk through the implementation of this class:
    - The constructor accepts `**kwargs` and passes them to the parent constructor, which handles standard hyperparameters: the `name` of the loss and the `reduction` algorithm to use to aggregate the individual instance losses. By default, this is `"AUTO"`, which is equivalent to `"SUM_OVER_BATCH_SIZE"`: the loss wil be the sum of the instances losses, weighted by the sample weights, if any, and divided by the batch size (not by the sum of the weights, so this is not the weighted mean). Other possible values are `"SUM"` and `"NONE"`.
    - The `call()` method takes the labels and predictions, computes all the instance losses, and returns them.
    - The `get_config()` method returns a dictionary mapping each hyperparameters name to its value. It first calls the parent class's `get_config()` method, then adds the new hyperparameters to this dictionary.
- You can then use any instance of this class when you compile the model.
- When you save the model, the threshold will be saved along with it; and when you load the model, you just need to map the class itself.
- when you save the model, Keras calls the loss instance's `get_config()` method and save the config in the SavedModel format. When you load the model, it calls the `from_config()` class method on the `HuberLoss` class: this method is implemented by the base class (`Loss`) and creates an instance of this class, passing `**config` to the constructor.

## Custom Activation Functions, Initializers, Regularizers, and Constraints

- Most Keras functionalities, such as losses, regularizers, constraints, initializers, metrics, activation functions, layers, and even full models, can be customized in very much the same way.
- Most of the time, you will just need to write a simple function with the appropriate inputs and outputs.
- You can see the implementation in the learning notebook, along with their equivalent in TensorFlow.
- As you can see, the arguments depend on the type of custom function. 
- These custom functions can be used normally, as shown in the learning notebook.
- The activation function will be applied to the output of the `Dense` layer, and its result will be passed on to the next layer.
- The layer's weight will be initialized using the value returned by the initializer.
- At each training, the weights will be passed to the regularization function to compute the regularization loss, which will be added to the main loss to get the final loss used for training.
- Finally, the constraint function will be called after each training step, and the layer's weights will be replaced by the constrained weights.
- If a function has a hyperparameter that needs to be saved along with the model, then you will want to subclass the appropriate class, such as `tf.keras.regularizers.Regularizer`, `tf.keras.constraints.Constraint`, `tf.keras.initializers.Initializer` or `tf.keras.layers.Layer` (for any layer, including activation function).
- Much as you did for the custom loss, there is a simple class for $\ell_1$ regularization that saves its `factor` hyperparameter you can find in the learning notebook. This time you do not need to call parent constructor or the `get_config()` method as they are not defined by the parent class.
- Note that you must implement the `call()` method for losses, layers (including activation functions), and models, or the `__call__()` method for regularizers, initializers, and constraints.

## Custom Metrics

- Losses and metrics are conceptually not the same thing:
    - Losses (e.g., cross entropy) are used by gradient descent to *train* a model, so they must be differentiable (at least at the points where they evaluated), and their gradients should not be zero everywhere. Plus, it's OK if they are not easily interpretable by humans.
    - In contrast, metrics (e.g., accuracy) are used to *evaluate* a model: they must be more easily interpretable, and they can be non-differentiable or have zero gradients everywhere.
- That said, in most cases, defining a custom metric function is exactly the same as defining the custom cost function.
- In fact, we could even use the Huber loss function we created earlier as a metric (though the Huber loss is seldom used as a metric, MSE and MAE is more preferred); it would work just fine (and saving would also work the same way, in this case only saving the name of the function, `"huber_fn"`, not the threshold).
- For each batch during training, Keras will compute this metric and keep tracks of its mean since the beginning of the epoch. Most of the time, this is exactly what you want. But not always!
- Suppose you train a binary classifier and want to measure its precision.
- As you saw in chapter 3, precision is the number of true positives divided by the number of positive predictions (including both true positives and false positives).
- Suppose the model made five positive predictions the first batch, four of which were correct: that's 80% precision.
- Then, suppose the model made three positive predictions in the second batch, but they were all incorrect: that's 0% precision for the second batch.
- If you compute the mean of these two precisions, you get 40%.
- But that's not the model's precision over these two batches. In fact, there were a total of four true positives (4 + 0) out of eight positive predictions (5 + 3), so the overall precision is 50%, not 40%.
- What we need is an object that keeps track of the number of true positives and the number of false positives and use them to compute the precision based on these numbers when requested. That is precisely what `tf.keras.metrics.Precision` class does.
- In our example, we created a `Precision` object, then we used it like a function, passing it the labels and predictions for the first batch, then the second batch (you can optionally pass sample weights as well, if you want).
- We used the same number of true and false positives as in the example we just discussed.
- After the first batch, it returns a precision of 80%; then after the second batch, it returns 50% (which is the overall precision so far, not the second batch's precision).
- This is called a *streaming metric* (or *stateful metric*), as it is gradually updated, batch after batch.
- At any point, we can call the `result()` method to get the current value of the metric.
- We can also look at its variables (tracking the number of true and false positives) by using the `variables` attribute using the `real_states()` method.
- If you need to define your own custom streaming metric, create a subclass of the `tf.keras.metrics.Metric` class.
- We implemented a basic example that keeps track of the total Huber loss and the number of instances seen so far.
- When we asked for the result, it returns the ratio, which is just the mean Huber loss.
- We will walk through the implementation in the learning notebook:
    - The constructor used the `add_weight()` method to create the variables needed to keep track of the metric's state over multiple batches - in this case, the sum of all the Huber losses (`total`) and the number of instances seen so far (`count`). You could just create the variables manually if you preferred.
    Keras keeps track of any `tf.Variable` that is set as an attribute (and more generally, any "trackable" object, such as layers or models).
    - The `update_state()` method is called when you use an instance of this class as a function (as we did with the `Precision` object). It updates the variable, given the labels and predictions for one batch (and sample weights, but we ignore it for our case).
    - The `result()` method computes and return the final result, in this case the mean Huber metric over all instances. When you use the metric as a function, the `update_state()` method gets called first, then the `result()` method is called, and the output is returned.
    - The `get_config()` method is implemented to ensure the `threshold` gets saved along with the model.
    - The default implementation of the `reset_state()` method resets all variables to 0.0 (but you can override it if needed).
- Keras will take care of variable persistence seamlessly; we don't have to do anything.
- When you define a metric using a simple function, Keras automatically calls it for each batch, and it keeps track of the mean during each epoch.
- So the only benefit of our `HuberMetric` class is that the `threshold` will be saved.
- But of course, some metrics, like precision, cannot simply be averaged over batches: in these cases, we have no other option than to implement a streaming metric.

## Custom Layers

- You may occasionally want to build an architecture that contains an exotic layer for which TensorFlow does not provide a default implementation.
- Or you may want to build a very repetitive architecture, in which a particular block of layers is repeated many times, and it would be convenient to treat each block as a single layer.
- For such cases, you want to build a custom layer.
- There are some layers that have no weights, such as `tf.keras.layers.Flatten` or `tf.keras.layers.ReLU`. If you want to create one, the simplest option is to write a function and wrap it in a `tf.keras.layers.Lambda` layer.
- This custom layer can then be used like any other layer, using the Sequential API, the functional API, or the subclassing API.
- You can also use it as an activation function, or you could use `activation=tf.exp`.
- The exponential layer is sometimes used in the output layer of a regression model when the values to predict have very different scales (e.g., 0.0001, 10, 1000).
- In fact, the exponential function is one of the standard activation functions in Keras, so you can just use `activation="exponential"`.
- To build a custom stateful layer (i.e., a layer with weights), you need to create a subclass of the `tf.keras.layers.Layer` class.
- We'll walk through the implementation of our custom (simplified) `Dense` layer in the learning notebook:
    - The constructor takes all the hyperparameters as arguments (in this example, `units` and `activation`), and importantly, it also takes a `**kwargs` argument. It calls the parent constructor, passing it the `kwargs`: this takes care of standard arguments such as `input_shape`, `trainable` and `name`.Then it saves the hyperparameters as attributes, converting the `activation` argument to the appropriate activation function using `tf.keras.activations.get()` function (which accepts functions, standard strings like `"relu"` or `"swish"`, or simply `None`).
    - The `build()` method creates the layer's variables by calling the `add_weight()` method for each weight. The `build()` method is called the first time the layer is used. At this point, Keras will know the shape of this layer's inputs, and it will pass it to the `build()` method, which is often necessary to create some of the weights. For example, we need to know the number of neurons in the previous layer in order to create the connection weights matrix (i.e., the `"kernel"`): this corresponds to the size of the last dimension of the inputs. At the end of the `build()` method (and only at the end), you must call the parent's `build()` method: this tells Keras that the layer is built (it just sets `self.built = True`).
    - The `call()` method performs the desired operations. In this case, we compute the matrix multiplication of the inputs X and the layer's kernel, we add the bias vector, and we apply the activation function to the result, and this gives us the output of the layer.
    - The `get_config()` method is just like in the previous custom classes. Note that we save the activation function's full configuration by calling `tf.keras.activations.serialize()`.
- You can `MyDense` layer just like any other layer.
- Keras automatically infers the output shape, expect when the layer is dynamic (as you will see shortly). In this (rare) case, you need to implement the `compute_output_shape()` method, which must return a `TensorShape` object.
- To create a layer with multiple inputs (e.g., `Concatenate`), the argument to the `call()` method should be a tuple containing all the inputs. To create a layer with multiple outputs, the `call()` method should return the list of outputs. An example can be found in the learning notebook.
- This layer may now be used like nay other layers, but of course only using functional and subclassing APIs, not the sequential API (which only accepts layers with one input and one output).
- If your layer needs to behave differently during training and during testing (e.g., if it uses `Dropout` or `BatchNormalization` layers), then you must add a `t.raining` argument to the `call()` method and use this argument ot decide what to do.
- For instance, you see in the learning notebook an example of a layer that adds Gaussian noise during training, but does nothing during testing (Keras has a layer that does the same thing, `tf.keras.layers.GaussianNoise`).

## Custom Models

- We already looked at how to create a custom model class in chapter 10, when we discussed the subclassing API.
- It's straightforward: subclass the `tf.keras.models.Model` class, create layers and variables in the constructor, and implement the `call()` method to do whatever you want the model to do.
- Suppose we want to build a model which does the following:
    - The inputs go through a first dense layer
    - Then through a *residual block* composed of two dense layers and an addition operation (as you will se in chapter 14, a residual block adds its inputs to its outputs).
    - Then through this same residual block three more times.
    - Then through a second residual block.
    - Finally, go through a dense output layer.
- To implement this model, it is best to first define a `ResidualLayer` class, since we are going to create a couple of identical blocks (and you may want to reuse it in another model).
- You can see the implementation in the learning notebook.
- This layer is a bit special, as it contains other layers.
- This is handled transparently by Keras: it automatically detects that the `hidden` attribute contains trackable objects (layers in this case), so their variables is automatically added to this layer's list of variables.
- Next, we use the subclassing API to define the model: We create the layers in the constructor and use them in the `call()` method.
- This model can then be used like any other model (compile, fit, evaluate, make predictions).
- If you want to save the model using the `save()` method nd load it using the `tf.keras.models.load_model()` function, you must implement the `get_config` method (as we did earlier) in both the `ResidualBlock` and `ResidualRegressor`.
- Alternatively, you can save and load the weights using the `save_weights()` and `load_weights()` methods.
- The `Model` class is a subclass of the `Layer` class, so models can be defined and used exactly like layers.
- A model has some extra functionalities, including its `compile()`, `fit()`, `evaluate()` and `predict()` methods (and a few variants), plus the `get_layer()` method (which can return any of the model's layers by name or by index) and the `save()` method (and support for `tf.keras.models.load_model()` and `tf.keras.models.clone_model()`).
- If models provide more functionality than layers, why not just defines every layer a model? Well, technically you could, but it usually cleaner to distinguish the internal components of your model (i.e., layers or reusable blocks of layers) form the model itself (i.e., the object you will train).

## Losses and Metrics Based on Model Internals

- The custom losses and metrics we defined earlier were all based on the labels and the predictions (and optionally sample weights).
- You may want to define losses based on other parts of your model, such as the weights or activations of its hidden layers.
- This may be useful if you want to have some regularization or monitor some internal aspects of your model.
- To define a custom loss based on model internals, compute it based on any part the model you want, then pass the result to the `add_loss()` method.
- For example, we'll build a custom regression MLP model composed of a stack of five hidden layers plus an output layer.
- This custom model will also have an auxiliary output on top of the upper hidden layer.
- The loss associated with this auxiliary output will be called the *reconstruction loss* (see chapter 17): it is the mean squared difference between the reconstructions and its inputs.
- By adding this loss, we encourage the model to preserve as much information as possible through the hidden layers, even if that piece of informations is not directly useful for the regression task.
- It is also possible to add a custom metric using the `add_metric()` method.
- We go through the implementation of this custom model with a custom reconstruction loss and a corresponding metric:
    - The constructor creates all five hidden layers and the output layer. It also implements a streaming metric that keep tracks of the reconstruction error.
    - The `build()` method builds a layer that predicts the reconstructed result. Note that we have to implement it here, as the number of neurons in the reconstructed layer is the number of inputs, and we only know the number of inputs if the `build()` method are called.
    - The `call()` method run the inputs through all the hidden layers and reconstruct using the reconstruction layer.
    - It then calculates the mean of the reconstruction loss over the whole input batch, and adds it to the model's loss using the `add_loss()` method. Note that we multiply the reconstruction loss by 0.05, which is a hyperparameter we can tune. This ensures the reconstruction loss doesn't dominate the main loss.
    - It also adds this error as a metric using the `add_metric()` method. These two lines in the `if` can be simplified to `self.add_metric(reconstruction_error)`: Keras will automatically keeps track of the mean for you.
    - Finally, the `call()` method passes the output of the hidden layers to the output layer and return its output.
- Both the total loss and the reconstruction loss will go down during training.

## Computing Gradients Using Autodiff

- A separate notebook about the details of how TensorFlow implements autodiff can be found in the same directory.
- To understand how to use auto diff to compute gradients automatically, let's consider a simple toy function:
    $$f(w1, w2)=3 \times w1^2 + 2\times w1 \times w2$$
- If you know calculus, you can analytically find that the partial derivate of this function with regard to $w1$ is $6 w1 + 2 w2$ and with regard to $w2$ is $2 w1$.
- For example, at the point $(w1, w2) = (5, 3)$, these partial derivates are 36 and 10, respectively, so the gradient vector at this point is (36, 10).
- But if this is a neural network, the function would be much more complex, typically with tens of thousands of parameters, and finding the partial derivates analytically by hand would be an impractical task.
- One solution could be computing an approximation of each partial derivate by measuring how much the function changes if you tweak the corresponding parameter by a tiny amount.
- This works well and easy to implement, but it's just an approximation, and importantly you need to call $f()$ at least once per hyperparameter (not twice, since we can cache $f(w1, w2)$).
- Having to call $f()$ at least once for each parameter makes this approach intractable for large neural networks.
- That's why we should use reverse-mode autodiff instead. TensorFlow makes this pretty simple with `tf.GradientTape()`
- We first define two variables `w1` and `w2`, then we create a `tf.GradientTape` context that will automatically record every operation that involves a variable, and finally we ask this tape to compute the gradients of the result `z` with regard to both variables `[w1, w2]`.
- Not only the results that Tensorflow computed is accurate (the precision is only limited by the floating-point errors), but the `gradient()`method only goes through the recorded computations once (in reverse order to the forward pass), no matter how many variables there are, so it is incredibly efficient.
-   In order to save memory, only put the strict minimum inside the `tf.GradientTape()` block.
- Alternatively, pause recording by creating a `with tape.stop_recording()` block inside the `tf.GradientTape()` block.
- The tape is automatically erased immediately after after you call its `gradient()` method, so you will get exception if you try to call `gradient` twice.
- If you need to call `gradient()` more than once, you must make the tape persistence and delete it each time you're done to free recourses. Python's garbage collector will delete the tape for you if the tape goes out of scope, for example when a function that used it returns.
- By default, the tape only tracks operations involving variables, so if you try to compute the gradient of `z` with regard to anything other than a variable, the result will be `None`.
- However, you can force the tape to watch any tensors you like, to record every operation involves them. You can then compute the gradients with regard to these tensors, as if they were variables.
- This can be useful in some cases, like if you want to implement a regularization loss that penalties activations that vary a lot wen the inputs vary little: the loss will be based on the gradient of the activations with regard to the inputs. Since the inputs are not variables, you must tell the tape to watch them.
- Most of the time, a gradient tape is used to compute the gradients of a single value (usually the loss) with regard to a set of values (usually the model parameters).
- This is where reverse-mode autodiff shines: it just need to do one forward pass and one reverse pass to calculate all the gradients at once.
- If you try to compute the gradients of a vector, for example, a vector of multiple losses, then TensorFlow will compute the gradients of the vector's sum.
- If you want to get the individual gradients (e.g., the gradients of each loss with regard to the model parameters), you must the tape's `jacobian()` method: it will perform reverse-mode autodiff once for each loss in the vector (all in parallel by default).
- You can also compute second-order partial derivate (the Hessians, i.e., the partial derivates of the partial derivates), but this is rarely need in practice.
- In some cases, you may want to stop gradients from backpropagating through some parts of your neural network.
- To do this, you must use the `tf.stop_gradient()` function.
- This function return its inputs during the forward pass, but it does not let gradients through backpropagation (it acts like a constant).
- Finally, you may occasionally run into some numerical issues when computing gradients.
- For example, if you compute the gradients of the squared root function at $x=10^{-50}$, the result will be infinite. In reality, the slope at that point is not infinite, but it's more than what 32-bit floats can handle.
- To solve this, it's recommended to add a tiny value to $x$ (such as $10^{-6}$) when computing its squared root.
- The exponential function is also another problem, as it grows extremely fast.
- For example, the softplus function we defined earlier is not numerically stable. If you compute it at 1000.0, you will get infinity instead of the true value (about 1000).
- But we can rewrite it in a numerically stable form:
    $$\begin{align*}
    \text{softplus}(x) &= \log(1 + \exp(x)) \\
                       &= \log(1 + \exp(x)) - \log(\exp(x)) + \log(\exp(x)) \\
                       &= \log\left(\frac{1+\exp(x)}{\exp(x)}\right) + x \\
                       &= \log\left(\frac{1}{\exp(x)} + 1\right) + x \\
                       &= \log(\exp(-x) + 1) + x \\
                       &= \text{softplus}(-x) + x \\
                       &= \text{softplus}(-|x|) + \max(0, x) \\
    \end{align*}$$
- The last equal sign comes when you realize it is trivial when $x<0$ and is the second to last equal sign when $x \geq 0$.
- In some rare cases, a numerically stable function may still have numerically unstable gradients.
- In such cases, you will have to tell TensorFlow which equation to use for the gradients, instead of letting it use autodiff.
- For this, you must use the `$tf.custom_gradient` decorator when defining the function, and return both the function's actual result and a function that compute the gradients.
- Using differential calculus, you can find that the derivate of softplus is:
    $$\frac{\exp(x)}{1+\exp(x)}$$
- But this form is not stable: for large value of x, it ends up computing infinity / infinity, which returns NaN.
- However, using a bit of algebra, you can verify that it's equal to $1-\displaystyle\frac{1}{1+\exp(x)}$, which is stable. The `my_softplus_gradients` function uses this equation to compute the gradients.
- Note that this function will receive the gradients what was propagated as the inputs, and according to the chain rule (which is just $(f(g(x)))' = g'(x)f'(g(x))$), we have to multiply them to the function's gradients.


## Custom Training Loops

- In some cases, the `fit()` method may not be flexible enough for what you need to do.
- For example, in the [Wide & Deep paper](https://arxiv.org/abs/1606.07792) we mentioned in chapter 10, the authors used two different optimizer: one for the wide path and the other for the deep path.
- Since the `fit()` method only use one optimizer, which is specified when we compile the model, reimplementing this paper requires you to write your own custom training loop.
- You may also want to write custom training loops simply to feel more confident that they do precisely what you intend them to do (perhaps you are unsure about some details of the `fit()` method).
- It can sometimes feel safer to make everything explicit.
- However, remember that writing a custom training loop will make your code longer, more error-prone, and harder to maintain.
- Unless you're learning or really need that extra flexibility, you should prefer using the `fit()` method rather than implementing your own training loop, especially when you work in a team.
- First, we build a simple model. We don't need to compile it, since we will handle the training loop manually.
- Next, we create a function that randomly sample a batch of instances from the training set. We will discuss the tf.data API, which offers a much better alternative.
- We also define a function to display the training status, including the number of steps, the total number of steps, the mean loss form the start of the epoch and other metrics.
- Now, we define some hyperparameters: The numbers of epochs, the batch size, the optimizer, the loss function, and the metrics. We'll use the `Mean`metric to keep track of the mean loss.
- Here, I will go through the implementation of the custom training loop in the learning notebook:
    - We created two nested loops: One loops through the epochs and other through the batches within an epoch.
    - In each batch, we use the function defined above to randomly sampled a batch from the training set.
    - We find the gradients using `tf.GradientTape()`. Inside the tape, we make a prediction using the model as a function, and we compute the loss, which is the mean loss plus all the model's losses (which is only one regularization error each layer, in our case). Since `mean_squared_error()` returns one loss per instance, we need to perform a `reduce_mean` to have the mean loss of this batch. This is also where you apply sample weights, if you want to do it. The regularization losses are already reduced to a single scalar value each, so we just need to sum them up (we used `f.add_n()`, which multiple tensors of the same shape and same data type).
    - Next, we ask the tape what the gradients of the loss with regard to all the trainable gradients are - not all variables - and perform a gradient descent step to all the trainable variables using the optimizer.
    - We then update the mean loss and the metrics of the function in this current epoch so far. We also display a status bar to check the progress of training.
    - At the end of each epoch, we reset the states of all the mean loss and metrics.
- If you want to apply gradient clipping, set the optimizer's `clipnorm` or `clipvalue` hyperparameter.
- If you want to apply any other transformation to the gradients, simply do so  before calling the `apply_gradients()` method.
- If you want to add weight constraints to your model (e.g., by setting `kernel_constraints` or `bias_constraints` when creating a layer), you should update the training loop too apply these constraints juts after the `apply_gradients()` method.
- Don't forget to set `training=True` when calling the model in the training loop, especially if your model behaves differently during training and testing (e.g., if you model uses `BatchNormalization` or `Dropout`).
- If it's a custom model, make sure to propagate the `training` argument to all the layers that your model calls.
- As you can see, lots of things you need to get right and it's easy to make a mistake. But the advantage is the full control you get.