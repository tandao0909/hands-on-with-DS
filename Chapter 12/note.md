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