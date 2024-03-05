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