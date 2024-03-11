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

### Interleaving Lines from Multiple Files

- First, suppose you've loaded the California housing dataset, shuffled it (unless it was already shuffled), and split it into a training set, a validation set, and a test set.
- The you split each set into many CSV files that each look like this (each row contains eight input features plus the target median house value).
- Let's also suppose that `train_filepaths` contains the list of training filepaths (and you also have `valid_filepaths` and `test_filepaths`).
- Alternatively, you could use file patterns; for example `train_filepaths=housing/datasets/my_train_*.csv`
- By default, the `list_files()` function return a dataset that shuffles the filepaths. In general, this is a good thing, but you can set `shuffle=False` if you don't want that happens for some reason.
- Next, you can call the `interleave()` method to read from five files at a time and interleave their lines. You can skip the first line of each file - which is the header row - by using the `skip()` method.
- The `interleave()` method will do the following:
    - Create a dataset that will pull five filepaths from the `filepath_dataset`
    - For each one, it will call the function you gave it (a lambda in this example) to create a new dataset (in this case, a `TextLineDataset`).
    - At this stage, there are seven datasets at all: the filepath dataset, the interleave dataset and the five `TextLineDataset` created internally by the interleave dataset.
    - When you iterate over the interleave dataset, it will cycle through these five `TextLineDataset`s and reading one line at a time form each of them. In other words, it reads one line, moves to the next file, reads one line, repeat. This will continue until one dataset is out of item.
    - Then it will fetch the next five filepaths from the `filepath_dataset` and interleave them the same way, and so on until it runs out of filepaths.
- For interleaving to work the best, it is preferable to have files of identical lengths; otherwise the ends of all the longer files will not be interleave.
- By default, `interleave()` method does not use parallelism; it just reads one line at a time from each file, sequentially.
- If you want it to actually read files in parallel, you can set the `interleave()` method's `num_parallel_calls` argument to the number of threads you want (recall that the `map()` method also has this argument). You can even set it to `tf.data.AUTOTUNE` to make TensorFlow choose the number fo threads dynamically based on the number available CPUs.
- It is possible to pass a list of filepaths to the `TextLineDataset` constructor: it will go through each file in order, line by line.
- If you also set the `num_parallel_reads` argument to a number greater than one, then the dataset will read that exact number of files in parallel and interleave their lines (without having to call the `interleave()` method). However, it will not shuffle the files, nor it will skip the header rows.

## Preprocessing the Data

- Now, we have a housing dataset that returns each instance as a tensor containing a byte string, we will define a function to do a bit of preprocessing, including parsing the string and scaling the data.
- We'll walk through the implementation of this function in the learning notebook:
    - First, assume we have the mean and standard deviation of each feature in the training set beforehand. `X_mean` and `X_std` are two 1D tensor (or NumPy array) composed of 8 floats, one for each input feature. This can be achieved by using a `StandardScaler()` from Scikit-learn. Later, we will do this using a Keras preprocessing layer instead.
    - Now we define the `parse_csv_line()` function that takes in a line in a CSV file. To help with that, we uses `tf.io.decode_csv()`, which takes in the line to parse and the default value for each column in the CSV line. The `default` record not only tells TensorFlow the default values, but also the number of columns and their data types. In our example, we tell it that all features columns are floats and the missing value must default to zero, but we provide an empty array of type `tf.float32` as the default value for the last column (the target): this tells TensorFlow that this column contains float, but there is no default value, so TensorFlow will throw an exception if it encounters a missing value.
    - The `tf.io.decode_csv()` function returns a list of scalar tensors (one per column), but we need to return a 1D tensor array instead. So we call `tf.stack()` to stack these scalar tensors into a 1D tensor, expect the last one. We then do the same for the target value, makes it an 1D tensor array with a single value, rather than a scalar tensor. Finally we return the input features and the target.
    - The `preprocess()` function calls the `parse_csv_line()` function, rescaled the input features by using the mean and the standard deviation arrays we mentioned earlier and return a tuple containing the rescaled input features and the target.
