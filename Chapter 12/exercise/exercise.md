1. 
- TensorFlow is an open-source library for numerical computation, and especially well-suited for large-scale machine learning.
- Its main features are:
    - Its core is similar to NumPy.
    - Support for GPU.
    - Support for distributed computing.
    - Computation graph analysis.
    - Optimization capabilities
    - A portable graph format allows you to train a model on one environment and run it on another.
    - An optimization API based on reverse-mode autodiff.
    - Several powerful APIs include tf.keras, tf.data, tf.image, tf.signal and so on.
- Some other popular deep learning libraries: PyTorch (by Facebook), PyTensor (a successor of Theano), Apache MXNet (being archived), Microsoft Cognitive Toolkit (stop supported), Chainer (stop supported).
2. While TensorFlow provides most of functionalities by NumPy, it's not a drop-in replacement. Here are the main differences between these two:
- First, some operations have different name to their alternatives. For example, `tf.reduce_sum()` versus `np.sum()`.
- Second, some operations are fundamentally different. For example, `tf.transpose()` exactly returns a transposed copy of a tensor, while the `T` attribute return a transposed view without copying the original matrix.
- Finally, tensors in TensorFlow are immutable by default, while arrays are mutable in NumPy. You can use `tf.Variable` if you need a mutable tensor.
3. Both `tf.range(10)` and `tf.constant(np.arange(10))` will return a one-dimensional array containing integers from 0 to 9. However, as TensorFlow uses 32-bit integers and NumPy uses 64-bit integers by default, the former will use 32-bit integers and the latter will use 64-bit integers.
4. Beyond regular tensors, here are 6 other data structures available in TensorFlow:
- Ragged tensor
- Tensor arrays
- Queues
- Sparse tensors
- Sets
- String tensors

The last two are actually represented as regular tensors, but TensorFlow provides special function to deal with them, in `tf.sets` and `tf.strings`, respectively.
5. 
- In general, use a Python function as a custom loss function is good enough.
- But if you have to support some hyperparameters (and other states), then you should subclass the `tf.keras.losses.Loss` class and implement `__init__()` and `call()` method.
- If you want the hyperparameters get saved along with the model, then you must also implement the `get_config()` method.
6. The answer is similar to the exercise 5: 
- In general, use a Python function as a custom loss metric is good enough.
- But if you have to support some hyperparameters (and other states), then you should subclass the `tf.keras.metrics.Metric` class.
- Moreover, if computing the metric over the whole epoch is not equivalent to computing the mean metric across the batches (such as precisions and recalls), then you should subclass `tf.keras.metrics.Mean` instead and implement the `__init__()`, `update_states()` and `result()` methods to keep track of a running metric during each epoch.
- You should implement the `reset_states()` method if you want it to do something other than reset all variables to 0.0.
- If you want the hyperparameters get saved along with the model, then you must also implement the `get_config()` method.
7. Here is the applicable situation for each of them:
- A custom layer: When you want to use it as a internal structure (i.e., layers or reusable blocks of layers) to use in a model. This should subclass the `tf.keras.layers.Layer`.
- A custom model: When you to use it as a trainable object. This should subclass the `tf.keras.models.Model`.
8.
- Writing your own custom training loop is a fairly advanced work, therefore you should only do it when you really need to.
- Keras already provides several tools to customize training loop without having to write one: callbacks custom optimizers, custom regularizers, custom constraints, etc.
- You should use these instead of writing your own one, as writing your own custom training loop is more error-prone, harder to maintain and to collaborate with other people.
- However, in some cases, writing a custom one may be useful, such as:
    - When you want to use different optimizers for different part of your model, like in the Wide&Deep model.
    - When you want to debug the process of training.
    - When you want to learn.
9. 
- Custom Keras components should be convertible to TF functions, which means they should stick to TF operations as much as possible and respect the rules written in the last part of this chapter (the TF function rules).
- However, it can contain arbitrary Python code, though it will hurt performance.
- If you want to do that, you just need to wrap the Python function in `tf.py_function()`, or set `dynamic=True` when creating the model, or set `run_eagerly=True` when calling the `compile()` method.
10. The rules can be found in the last part of this chapter (the TF function rules).
11. 
- You would need to create a dynamic Keras model can be useful for debugging and understanding the underlying process of your model, as it will not compile any custom component to a TF function, and you can use the Python debugger to debug your code.
- You can also contain any arbitrary Python code in your model, including your calls to external libraries.
- If you want to do that, you just need to set `dynamic=True` when creating the model, or set `run_eagerly=True` when calling the `compile()` method.
- Making a model dynamic prevents Keras from using any of TensorFlow's graph features, so it will hinder performance, slow down your training and inference time. You also cannot export the computation graph in a portable format, which limit your model's portability.