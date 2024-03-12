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

## Putting Everything Together + Prefetching

- Now we link everything we discussed so far and define a helper function that efficiently loads a dataset from multiple CSV files, preprocesses, shuffles and returns it in batches.
- Note that we use the `prefetch()` method on the very last line. What is that?
- By calling `prefetch(1)` at the end of the last line tells the dassett tries its best to always be one batch ahead.
- In other words, while our training algorithm is working on one batch, the dataset will already working in parallel to load the data from disk and preprocess to make the next batch.
- If we also ensure loading and preprocessing are multithreaded (by setting `num_parallel_calls` when calling `map()` and `interleave()` methods), then we can utilize multiple CPU cores and hopefully make preparing one batch of data shorter than running a training step on GPU: This way, the GPU will almost be 100% utilized, expect for the data transfer time from the CPU to the GPU, and training will be much faster.
- If the dataset is small enough to fit in the memory, you can significantly speed up training by using the dataset's `cache()` method to cache its content to RAM.
- You should generally do this after loading and preprocessing the data, but before shuffling, repeating, batching, and prefetching. This way, each instance will be read and preprocessed once (instead of one per epoch), but the data will still be reshuffled differently at each epoch, and the next will still be prepared in advance.
- We have now learned how to build efficient input pipelines to load and preprocess data form multiple text files.
- We have discussed the most common dataset methods, but there a few more you may want to look at, such as `concatenate()`, `zip()`, `window()`, `reduce()`, `shard()`, `flat_map()`, `apply()`, `unbatch()` and `padded_batch()`.
- There are also a few more class methods, such as `from_generator()` and `from_tensors()`, which create a new dataset from a Python generator or a list of tensors, respectively.
- Also note that there are experimental features available in `tf.data.experimental`, many of which will likely make it to the core API in future releases (e.g., check out the `CsvDataset` class, as well as the `make_csv_dataset()` method which takes care of inferring the data type of each column).

## Using the Dataset with Keras

- Now, we can use the custom `csv_reader_dataset()` function we wrote earlier to create dataset for the training set, also for the validation set and the test set. The training set will be shuffled at each epoch (note that the validation set and the test set will also be shuffled, though we don't really need that).
- Now you can simply build and train a Keras model using these datasets. When you call the `fit()` method, you pass `train_set` instead of `X_train, y_train` and pass `validation_data=valid_set` instead of `validation_data=(X_valid, y_valid)`.
- The `fit()` method will take care of repeating the training dataset once per epoch, using a different random order at each epoch.
- Similarly, you can pass a dataset to the `evaluate()` and `predict()` methods.
- Note that the set we pass to the `predict()` method will typically doesn't have labels. If it does, as in our case, Keras will ignore them. 
- Note that in all of these cases, you can still use NumPy arrays instead of datasets if you prefer (but of course they must be loaded and preprocessed first).
- You can build your own custom training loop (as we did in chapter 12) by iterate over the whole training set.
- You can even create a separate TF function that trains the model for the whole epoch. This can really speed up training.
- In Keras, the `steps_per_execution` argument of the `compile()` method lets you define the number of batches that the `fit()` method will process during each call to the `tf.function` it uses for training.
- The default is just 1, so if you set it to 50, you will often see a significant improvement in performance. However, the `on_batch_*()` method of Keras callbacks will only be called every 50 batches.
- So far, we've been using CSV files, which are common, simple and convenient but not really efficient, and do not support large or complex data structures (such as images or audio) very well. The next part will look at TFRecords.
- If you're fine with CSV files (or whatever formats you are using), you do not have to use TFRecords. As the saying goes, if it ain't broke, don't fix it! 
- That said, TFRecords are useful hwn the bottleneck during training is loading and parsing data.

# The TFRecord Format

- The TFRecord format is the TensorFlow's preferred format for storing large amounts of data and reading it efficiently.
- It is a very simple binary file format that just contains a sequence of binary records of varying sizes. Each record is comprised of a length, a RCR checksum to verify that the length was not corrupted, then the actual data, and finally a CRC checksum for the data.
- You can create a TFRecord file using the `tf.io.TFRecordWriter` class.
- You can then use a `tf.data.TFRecordDataset` to read one or more TFRecord files.
- By default, a `TFRecordDataset` will read one file at a time, but you can make it read multiple files in parallel and interleave their records by passing the constructor a list of filepaths and setting `num_parallel_reads` to a number greater than one.
- Alternatively, you can obtain the same result by using `lits_files()` and `interleave()` as we did earlier to read multiple CSV files.

## Compressed TFRecord Files

- It can sometimes be useful to compress your TFRecord files, especially if they need to be loaded via a network connection.
- You can create a compressed TFRecord file by setting the `options` argument to a `tf.io.TFRecordOptions` object.
- When reading a compressed TFRecord file, you need to specify the compression type.

## A Brief Introduction To Protocol Buffers

- Even though each record can use any binary format you want, TFRecord files usually contain serialized protocol buffers (also called *protobufs*).
- This is a portable, extensible, and efficient binary format developed at Google back in 2001 and mode open source in 2008; protobufs are now widely used, in particular in gRPC, Google's remote procedure call system.
- They are defined using a simple language that looks like this:
```proto
syntax = "proto3";
message Person {
    string name = 1;
    int32 id = 2;
    repeated string email = 3;
}
```
- The protobuf objects are meant to be serialized and transmitted, hence they are called *messages*.
- This protobuf definition says we are using version 3 of the protobuf format, and it specifies that each `Person` object may (optionally) have a `name` of type string, an `id` of type int32, and zero or more `email` fields, each of type string.
- The number 1, 2 and 3 are the field identifiers: they will be used in each record's binary representation.
- Once you have a definition in a `.proto` file, you can compile it. This requires `protoc`, the protobuf complier, to generate access classes in Python (or some other languages).
- Note that the protobuf definition will generally use in TensorFlow have already precompiled for you, and their Python classes are part of the TensorFlow library, so you will not have to use `protoc`.
- In short, we import the `Person` class generated by `protoc`, we create an instance and play with it, visualizing it and reading and writing some fields, then we serialize it using the `SerializeToString()` method. This is the binary data that is ready to be saved or transmitted over the network.
- When reading or receiving this binary data, we na parse it using the `ParseFromString()` method, and we get a copy of the object that was serialized.
- You could save the serialized `Person` object to a TFRecord file, then load and parse it: everything would work fine.
- However, `ParseFromString()` is not a TensorFlow operation, so you couldn't use it in a preprocessing function in a tf.data pipeline (expect by wrapping it in a `tf.py_function()` operation, which make the code slower and less portable).
- However, you could use the `tf.io.decode_proto()` function, which can parse any protobuf you want, provided you gave it the protobuf definition.
- That said, in practical, you will generally want to use the predefined protobufs for which TensorFlow provides dedicated parsing operations.

## TensorFlow Protobufs

- The main protobuf typically used in a TFRecord file is the `Example` protobuf, which presents one instance in a dataset.
- It contains a list of named features, where each feature can either be a list of byte strings, a list of floats, or a list of integers.
- Here is the protobuf definition (from TensorFlow's source code):
```proto
syntax="proto3";
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features {map<string, Feature> feature = 1; };
message Example { Features feature = 1; };
```
- The definitions of `BytesList`, `FloatList` and `Int64List` are straightforward. Note that `[packed = true]` is used for repeated numerical fields, for a more efficient encoding.
- A `Feature` contains either a `BytesList`, a `FloatList` or an `Int64List`.
- A `Features` (with an `s`) contains a dictionary that map a feature name to its corresponding features value.
- Finally, an `Example` contains exactly one `Features`.
- You can find the implementation in TensorFlow representing the same person as earlier in the learning notebook.
- Now we have an `Example` protobuf, we can serialize it by calling its `SerializeToString()` method, then write the resulting data to a TFRecord file.
- In the learning notebook, we pretend we have five contacts. In real life, you typically would create a convention script to read from your current format (such as CSV files), create an `Example` protobuf for each instance, serialize them, and save them to several TFRecord files, ideally shuffling them in the process.
- This requires a bit of work, so again make sure it is really necessary (i.e., your pipeline really needs reduce that extra I/O time).

## Loading and Parsing Examples

- To load the serialized `Example` protobufs, we will use a `tf.data.TFRecordDataset` once again, and we will parse each `Example` using `tf.io.parse_string_example()`.
- It requires at least two arguments: a string scalar tensor containing the serialized data, and a description of each feature.
- The description is a dictionary that maps each feature name to either a `tf.io.FixedLenFeature` descriptor indicating the feature's shape, type, and default value, or a `tf.io.VarLenFeature` descriptor indicating only the type if the length of the list may vary (such as for the `"emails"` feature).
- The code in the learning notebook defines a description dictionary, then creates a `TFRecordDataset` and applies a custom preprocessing function to parse each serialized `Example` protobuf that this dataset contains.
- The fixed-length features are parsed as regular tensors, but the variable-length features are parsed as sparse tensors. You can convert a sparse tensor by using `tf.sparse.to_dense()`, but it is simpler to just access its values in this case.
- Instead of parsing examples one by one using `tf.io.parse_single_example()`, you may want to parse them batch by batch using `tf.io.parse_example()`.

## Extra Material - Storing Images and Tensors in TFRecords

- A `BytesList` can contain any binary data you want, including any serialized object.
- For example, you can use `tf.io.encode_jpeg()` to encode an image using the JPEG format and this binary data in a `BytesList`
- Later, when your code reads the TFRecord, it wil start by parsing the `Example`, then it will need to call `tf.io.decode_jpeg()` to parse the data and get the original image (or you can use `tf.io.decode_image()`, which can decode any BMP, GIF, JPEG or PNG image).
- You can also store any tensor you want in a `BytesList` by serializing the tensor using `tf.io.serialize_tensor()` then putting the resulting byte string in a `BytesList` feature.
- Later, when you parse the `TFRecord`, you can parse this data using `tf.io.parse_tensor()`.

## Handling Lists of Lists Using the SequenceExample Protobuf

- As you have seen, the `Example` protobuf is quite flexible, so it would be sufficient for most use cases.
- However, it's a big problem when you try to deal with lists of lists.
- For example, assume you want to deal with text documents.
- Each document can be represented as a list of sentences, where each sentence is represented as a list of words.
- Perhaps each document may has a list of comments, where each comment is also represented as a list of words.
- There may be some contextual data too, such as the document's author, title, and publication date.
- These use cases is more suited for `SequenceExample` protobuf. Here is its definition:
```proto
message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
    Features context = 1;
    FeatureLists feature_lists = 2;
};
```
- A `SequenceExample` contains a `Features` object for the contextual data and a `FeatureLists` object that contains one or more named `FeatureList` objects (e.g., a `FeatureList` named `"content"` and another named `"comments"`).
- Each `FeatureList` contains a list of `Feature` objects, each of which may be a list of byte strings, a list of 64-bit integers or a list of floats.
- Building a `SequenceExample`, serializing it and parsing it is similar to building, serializing, and parsing an `Example`, but you must use `tf.io.parse_single_sequence_example()` to parse a single `SequenceExample` or `tf.io.parse_sequence_example()` to parse a batch.
- Both functions return a tuple containing the context features (as a dictionary) and the feature lists (also as a dictionary).
- If the feature lists contain sequences of varying sizes, you may want to convert them to a ragged tensor using `tf.RaggedTensor.from_parse()`.

## The Normalization Layer

- As we saw in chapter 10, Keras provides a `Normalization` layer that we can use to standardize the input features.
- We can either specify the mean and the variance of each input feature or - more simply - passing the training set to the layer's `adapt()` method before fitting the model, so the layer can measure the means and the variances of the input features on its own before training.
- The data sample passed to the `adapt()` method must be big enough to be representative of your dataset, but it does not have to be the full training set: for the `Normalization` layer, a few hundred instances randomly sampled from the training set is sufficient to have a good estimation about the future means and variances.
- Since we included the `Normalization` layer inside the model, we can now deploy our model to production without having to worry about normalization again: the model will just handle it for you.
- This approach completely eliminates the risk of preprocessing mismatch, which happens when people try to maintain different preprocessing code for training and production but update one and forget the other. The production models then ends up getting data preprocessed in way it doesn't expect. If you're lucky, they scream out to you. If you're not, then the model's accuracy just silently degrades.
- Including the preprocessing layer directly in the model is nice and straightforward, but it will slow down training (only very slightly in the case of the `Normalization` layer).
- This is because preprocessing happened on the fly during training, it happens once per epoch.
- We can do better by only normalizing the whole training set just once before training.
- We do this by treating the `Normalization` layer as a separate layer (like we did with Scikit-learn's `StandardScaler`).
- Then we trained the model one the scaled data, without a `Normalization` layer this time.
- This would speed up training quite a bit. But now the model won't preprocess its input in production.
- To fix this, we create a new model that wraps both the adapted `Normalization` layer and the model we just trained can deploy this model to the production instead.
- This new model can then preprocess its inputs and making predictions.
- Now we have the bets of both worlds: training is fast as we just preprocess once before training, and the final model can preprocess its inputs on the fly without risking any of preprocessing mismatch.
- Moreover, the Keras preprocessing layers play nice with the tf.data API.
- For example, you could apply an adapted `Normalization` layer to the input features of each batch batch in a dataset.
- Lastly, if you ever need more features than what the Keras preprocessing layers offer, you can write your own custom layer, like we did in chapter 12.
- Fr example, you can look at the learning notebook for an example of a custom `Normalization` layer, if we pretend it doesn't exist.

## The Discretization Layer

- This layer's goal is to transform a numerical feature into a categorical feature by mapping value ranges (called bins) to categories.
- This is sometimes useful for features with multimodal distributions, or with features that have a highly nonlinear relationships with the target.
- For example, you can see in the learning notebook the code maps a numerical `age` feature ot three categories, less than 18, 18 to less than 50, and 50 or more.
- In this code, we provided the desired bin boundaries. If you prefer, you can instead provide the number of bins you want, then call the layer's `adapt()` method to let it find the appropriate boundaries based on the value percentiles.
- For example, if we set `num_bins=3`, then the bin boundaries will be located a the values just below the 33rd and 66th percentiles.
- Categorical identifiers such as these should generally not be passed directly to a neural network, as their values cannot be meaningfully compared (if we set red = 1, green = 2, it does not mean green is two times more than red!).
- Instead, they should be encoded, for example using one-hot encoding.