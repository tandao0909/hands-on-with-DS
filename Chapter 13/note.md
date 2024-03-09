- In chapter 2, you knew that loading and preprocessing data is an important part of any machine learning project.
- You used Pandas to load and explore the (modified) California housing dataset - which was stored in a CSV file - and you applied ScikitLearn's transformers for preprocessing.
- These tools are quite convenient, and you will probably be using them often, especially when exploring and experimenting with data.
- However, when training TensorFlow models on large datasets, you may prefer to use TensorFlow's own data loading and preprocessing API, called *tf.data*.
- It is capable of loading and preprocessing data extremely efficiently, reading from multiple files in parallel using multithreading and queuing, shuffling and batching samples, and more.
- Plus, it can do all of this on the fly - it loads and preprocesses the next batch of data across multiple CPU cores, while your GPUs or TPUs are busy training the current batch of data.
- The tf.data API lets you handle datasets that don't fit in memory, and it allows you to make full use of your hardware resources, thereby speeding up training.
- Off the shell, the tf.data API can read from text files (such as CSV files), binary files with fixed-size records, and binary files that use TensorFlow's TFRecord format, which supports records of varying sizes.
- TFRecord is a flexible and efficient binary format usually containing protocol buffers (an open source binary format).
- The tf.data API also has support for reading from SQL databases.
- Moreover, many open source extensions are available to read from all sort of data sources, such as Google's BigQuery service.
- Keras also comes with powerful yet easy-to-use preprocessing layers that can be embedded in your models: this way, when you deploy a model to production, it will be able to ingest raw data directly, without you having to add any additional preprocessing code.
- This eliminates the risk of mismatch between the preprocessing code used during training and the preprocessing code used in production, which would likely cause *training/serving skew*.
- If you deploy your model in multiple apps coded in different programming languages, you won't have to reimplement the same preprocessing code multiple times, which also reduces the risk of mismatch.
- As you will see, both APIs can be used jointly - for example, to benefit from the efficient data loading offered by tf.data and the convenience of the Keras preprocessing layers.

# The tf.data API

- The whole tf.data API revolves around the concept of a `tf.data.Dataset`: this represents a sequence of data items.
- Usually you will use datasets that gradually read data from disk, but for simplicity let's create a dataset from a simple data tensor using `tf.data.Dataset.from_tensor_slices()`.
- The `from_tensor_slices()` function takes a tensor and creates a `tf.data.Dataset` object whose elements are all the slices of X along the first dimension, so this dataset contains 10 items : tensor 1, 2, ..., 9.
- In this case, we could use `tf.data.Dataset.range()`, expect the datatype is 64-bit integer instead of 32-bit integer.
- The tf.data API is a streaming API: you can very efficiently iterate through a dataset's items, but the API is not designed for indexing or slicing.
- A dataset may also contain tuples of tensors, or dictionaries of name/tensor pairs, or even nested tuples and dictionaries of tensors. When slicing a tuple, a dictionary, or a nested structure, the dataset will only slice the tensor it contains, while preserving the tuple/dictionary structure.