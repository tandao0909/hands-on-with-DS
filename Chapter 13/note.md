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

## Chaining Transforms

- Once you have a dataset, you can apply all sorts of transformations to it by calling its transformation methods.
- Each method returns a new dataset, so you can chain transforms.
- In the learning notebook, we first call the `repeat()` method in the original dataset, and it returns a new dataset that repeats the items of the original dataset three times. Of course, this will copy all the data in memory three times.
- If you call this method with no argument, the new dataset will repeat the source dataset forever, so the code that iterates over the dataset will have to decide when to stop.
- Then we call the `batch()` method on this new dataset, and again, this creates a new dataset. This one will group the items of the previous dataset in batches of seven items.
- Finally, we iterate over the final dataset using a manual for loop.
- The `batch()` method had to output a final batch of size 2 instead of 7, but you can call `batch()` with `drop_remainder=True` if you want it to drop this final batch, such that all batches have the exact same size.
- The dataset method do not modify datasets - they crate new ones. So make sure to keep a reference to these new datasets (e.g., with `dataset=...`) or else nothing will happen.
- You can also transform the items using the `map()` method.
- This `map()` method is the one you will call to apply any preprocessing to your data.
- Sometimes this process can include extensive computations, such as reshaping or rotating an image, so it's better to divide the work to multiple threads. This can be done by setting the `num_parallel_calls` argument to the number of threads to run, or to `tf.data.AUTOTUNE`. 
- Note that the function passed to the `map()` method must be convertible to TF functions.
- You can simply filter the dataset using the `filter()` method.
- If you want to look at a few first items from the dataset, you can use the `take()` method.

## Shuffling the Data

- As we discussed in chapter 4, gradient descent works best when the instances in the training set are independent and identically distributed (IID).
- A simple way to ensure this is to shuffle the instances, using the `shuffle()` method.
- It will create a new dataset start by filling up a buffer with the first items from the original dataset.
- Then, whenever be asked for a new item, the buffer return aa instance randomly and replace it with a new instance from the dataset, until the source dataset is exhausted.
- At that point, it will just return instances randomly until it is empty.
- You must specify the size fo the buffer, or else shuffling will not be so effective.  
    > Just imagine you have a sorted deck of cards on your left. Now take the top three and shuffle them in your hand, pick one randomly and put it to the right, keep two in your hand. Now take another card from the left, shuffle the three cards in your hand and pick one of them randomly, put it to the right. When you are done going through the whole deck like this, you will have a deck of card on your right. Do you think it would be perfectly shuffled?
- Just don't go exceed the amount of RAM you have, even though if you have plenty of them, there's no point to go beyond the dataset's size.
- You can provide a random seed if you want the same random order every time you rerun the program.
- If you call `repeat()` on a shuffled dataset, by default it will generate a new order at every iteration.
- This is generally a good idea, but if you prefer to reuse the same order at each iteration (e.g., for tests or debugging), you can set `reshuffle_each_iteration=False` when calling `shuffle()`.
- For a large dataset, this simple shuffling-buffer approach may not be sufficient, since te buffer will be small compared to the dataset.
- One solution is to shuffle the source data itself (for example, run the `shuf` command on Linux to shuffle text files). This will improve shuffling a lot.
- Even if the source data is shuffled, you will usually want to shuffle it some more, or else the same order will be repeated at each epoch, and the model may end up being biased (e.g., due to some incorrect patterns showed up by chance in the source data's order).
- To shuffle the instances some more, one solution is to split the source data into multiple files, then read them in a random order during training.
- However, instances located in the same file will still end up close to each other.
- To avoid this, you can pick multiple files randomly and read them simultaneously, interleaving their records.
- Then on top of that, you can add a shuffling buffer using the `shuffle()` method.