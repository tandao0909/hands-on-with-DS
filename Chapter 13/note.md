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

## The CategoryEncoding Layer

- When there are only a few categories (e.g., less than a dozen or two), then one-hot encoding if often a good option.
- To do this, Keras provides the `CategoryEncoding` layer. The usage can be seen in the learning notebook.
- If you try to encode more than one categorical feature at a time (which only makes sense if they all use the same categories), the `CategoryEncoding` class will perform *multi-hot encoding* by default: the output tensor will contain a 1 for each category present in any input feature.
- You want to count how many times each category occurred instead of if it occurred or not, you can set `output_mode="count"` when creating the `CategoryEncoding` layer, in which case the output tensor will contain contain the number of occurrences of each category.
- In our example, the output would be the same expect for the second row, which would become `[0., 0., 2.]`.
- Note that both multi-hot encoding and count encoding lose information, since it's not possible to know which feature each active category came from.
- For example, both `[1., 0.]` and `[0., 1.]` are encoded as `[1., 1., 0.]`.
- If you want to avoid this, then you need to one-hot encode each feature separately and concatenate the outputs. This way, `[1., 0.]` get encoded as `[0., 1., 0., 0., 0., 0.]` and `[0., 1.]` get encoded as `[0., 0., 0., 0., 1., 0.,]`. The learning notebook shows three ways to do this.
- In our examples, the first three columns correspond to the first feature, and the last three correspond to the second feature. This allows the model to distinguish the two features.
- However, it also increases the number of features fed to the model, and thereby requires more model parameters.
- There's not a clear win between a single multi-hot encoding or a per-feature one-hot encoding will work best: it depends on the task, and you may need to test both options.

## The StringLookup Layer

- We create a `StringLookup` layer, then we adapt it to the data: it finds that there are three distinct categories.
- Then we use the layer to encode a few cities.
- They are encoded as integers by default.
- Unknown categories get mapped to 0, as is the case for "Montreal" in our example.
- The known categories are numbered starting at 1, from the most frequent category to the least frequent.
- Conveniently, if you set `output_mode="one_hot"` when creating the `StringLookup` layer, it will output a one-hot vector for each category, instead of an integer.
- Keras also includes an `IntegerLookup` layer that acts very much like the `StringLookup` layer but take integers as input, rather than strings.
- If the training set is very large, it helps to adapt the layer to just a small subset of the training set.
- In this case, the layer's `adapt()` method may miss some of the rarer categories.
- By default, it would map them all to the category 0, make them indistinguishable by the model.
- A solution (while still adapting the layer only to a subset of the training set) is setting `num_oov_indices` to an integer greater than 1.
- This is the number of out-of vocabulary (OOV) buckets to use: each unknown category will get mapped pseudorandomly to one of the OOV buckets. This allows the model to distinguish at least some of the rare categories.
- Since there are five OOV buckets, the first known category's ID is now 5 (`"Paris"`). But `"Foo"`, `"Bar"`, and `"Baz"` are unknown, so they each get mapped to one of the OOV buckets.
- `"Bar"` gets its own dedicated bucket (with ID 3), but sadly `"Foo"` and `"Baz"` happen to be mapped to the same bucket (with ID 4), so they remain indistinguishable by the model. This is called a *hashing collision*.
- The only way to reduce the risk of collision is to increase the number of OOV buckets. However, this will also increase the total number of categories, which will require more RAM and extra model parameters once the categories are one-hot encoded. So, don't increase this number too much.

## The Hashing Layer

- The ideas of mapping categories pseudorandomly to buckets is called the *hashing trick*. Keras provides a dedicated layer which does just that: the `Hashing` layer.
- For each category, this layer computes a hash, modulo the number of buckets (or "bins").
- The mapping is entirely pseudorandom, but stable across runs and platforms (i.e., the same category will always be mapped to the same integer, as long as the number of bins is unchanged).
- The benefit of this layer is that it does not need to be adapted at all, which may sometimes be useful, especially in an out-of-core setting (when the dataset is too large to fit in the memory).
- However, we once again get a hashing collision: "Tokyo" and "Montreal" are mapped to the same ID, making them indistinguishable by the model.
- That's why it's usually preferable to stick to the `StringLookup` layer.

## Encoding Categorical Features Using Embeddings

- An embedding is a dense representation of some higher-dimensional data, such as a category, or a word in a vocabulary.
- If there are 50,000 possible categories, then one-hot encoding would produce a 50,000-dimensional sparse vector (i.e., containing mostly zeros).
- In contrast, an embedding would be a comparatively small dense vector; for example, with just 100 dimensions.
- In deep learning, embeddings are usually initialized randomly, and then they are trained using gradient descent, along with the other model parameters.
- For example, the `"NEAR BAY"` category in the California housing dataset could be represented initially by a random vector such as `[0.131, 0.890]`, while the `"NEAR OCEAN"` category might be presented by another random vector such as `[0.632, 0.791]`.
- In this example, we use 2D embeddings, but the number of dimensions is a hyperparameter you can tweak.
- Since these embeddings are trainable, they will gradually improve during training; and as they represent fairly similar categories in this case, gradient descent will certainly end up pushing them close together, while it tends to move them away from the `"INLAND"` category's embedding.
- In fact, the better the representation, the easier it will be for the neural network ti make accurate predictions, so training tends to make embeddings useful representations of the categories.
- This is called *representation learning*, which will be discussed in chapter 17.
- Keras provides an `Embedding` layer, which wraps an `embedding matrix`: this matrix has one row per category and one column per embedding dimension.
- By default, it is initialized randomly.
- To convert a category 1D to an embedding, the `Embedding` layer just looks up and returns the row that corresponds to that category. You can find an example in the learning notebook.
- An `Embedding` layer is initialized randomly, so it doesn't make sense to sue it outside of a model as  a standalone preprocessing layer, unless you initialize with pretrained weights.
- If you want to embed a categorical text attribute, you can simply chain a `StringLookup` layer and an `Embedding` layer.
- Note that the number of rows in the embedding matrix needs to be equal to the vocabulary size: that's the total number of categories, including the known categories plus the OOV buckets (just one by default). The `vocabulary_size()` method of the `StringLookup` class conveniently returns this number.
- In this example, we use 2D embeddings, but as a rule of thumb embeddings typically have 10 to 300 dimensions, depending on the task, the vocabulary size, and the size of training set. This is a hyperparameter you need to tune.
- Putting everything together, we can now a Keras model that can process a categorical text feature along with regular numerical features and learn an embedding for each category (as well for each OOV bucket).
- You can find the model's implementation in the learning notebook:
    - The model takes two inputs: `num_input`, which contains eight numerical features per instance, plus `cat_input`, which contains a single categorical text input per instance.
    - The model uses the `lookup_and_embed` we created earlier to encode each ocean-proximity category as the corresponding trainable embedding.
    - Next, it concatenates the the numerical inputs and the embeddings using the `concatenate()` function to produce the complete encoded inputs, which are fed to a neural network.
    - We could add any kind of neural network at this point, but for simplicity, we just use a single dense output layer, and then we create the Keras `Model` with the inputs and output we've just defined.
    - Finally, we compile the model and train it by passing both the numerical and categorical inputs.
- Since the `Input` layers are named `"num"` and `"cat"`, we could also have passed the training data to the `fit()` method using a dictionary instead of a tuple: `{"num": X_train_num, "cat": X_train_cat}`.
- Alternatively, we could have pass a `tf.data.Dataset` containing batches, each represented as `((X_batch_num, X_batch_cat), y_train)` or as `({"num": X_batch_num, "cat": X_batch_cat}, y_batch)`. The same goes for validation data.
- One-hot encoding followed by a `Dense` layer (with no activation function and no biases) is equivalent to an `Embedding` layer.
- However, the `Embedding` layer uses way fewer computations, as it avoids many multiplications by zero - the performance difference becomes clear once the size of the embedding matrix grows big.
- The `Dense` layer's weight matrix plays the role of the embedding matrix.
- For example, using one-hot vectors of size 20 and a `Dense` layer with 10 units is equivalent to using an `Embedding` layer with `input_dim=10` and `output_dim=20`.
- As a result, it'd be wasteful to use more embedding dimensions than the number of units in the layer that follows the `Embedding` layer.

### Word Embeddings

- Not only will embeddings generally be useful representations for the task at hand, but quite often these same embeddings can be reused for other tasks.
- The most common examples in real world is *text embeddings* (e.e., embeddings of individual words): when you are working on a natural language precessing task, you are often better off reusing pretrained word embeddings than training your own.
- The idea of using vectors to represent words is born back to the 1960s, and many sophisticated techniques have been used to generate useful vectors, including using neural networks. But things really took off in 2013, when Tomáš Mikolov and other Google researchers published a [paper](https://arxiv.org/abs/1310.4546) describing an efficient technique to learn word embeddings using neural networks, significantly outperforming pervious attempts.
- This allowed them to learn embedding on a very large corpus of text: they trained a neural network to predict the words ner any given word and received outstanding word embeddings.
- For example, synonyms had very close embeddings, and semantically related words such as *France*, *Spain*, and *Italy* ended up clustered together.
- It's not just about proximity, word embeddings were also organized along meaningful axes in the embedding space.
- Here is a famous example: if you compute *King - Man + Woman* (adding and subtracting the embedding vectors of these words), then the result will be very close to the embedding vector of the word *Queen*. In other words, the word embeddings encode the concept of gender!
- Similarly, you can compute *Madrid - Spain + France*, and the result is close to *Paris*, which seems to show that the notion of capital city was also encoded in the embeddings.
- Unfortunately, word embeddings sometimes capture our worst biases.
- For example, even though they correctly learn that *Man is to King as Woman is to Queen*, they also seem to learn that *Man is to Doctor as Woman is to Nurse*, which is a sexist bias! To be fair , this particular is probably exaggerated, as was pointed out in a [2019 paper](https://arxiv.org/abs/1905.09866) by Malvina Nissim et al.
- Nevertheless, ensuing fairness in deep learning algorithms is an important and active research area.

## Text Preprocessing

- Keras provides a `TextVectorization` layer for basic text preprocessing.
- Much like the `StringLookup` layer, you must either pass it a vocabulary upon creation, or let it learn the vocabulary from some training data using the `adapt()` method.
- The two sentences "Be good!" and "Question: be or be?" were encoded as `[2, 1, 0, 0]` and `[6, 2, 1, 2]`, respectively.
- The vocabulary was learned from the four sentences in the training data: "be" = 2, "to" = 3, etc. To construct the vocabulary, the `adapt()` method first converted the training sentences to lowercase and removed punctuation, which is why "Be", "be", and "be?" are all get encoded as "be" = 2.
- Next, the sentences were split on whitespace, and the resulting words were sorted by descending frequency, producing the final vocabulary. When encoding sentences, unknown words get encoded as 0s.
- Lastly, since the first sentence is shorter than the second, it was padded with 0s.
- The `TextVectorization` layer has many options:
    - For example, you can preserve the case and punctuation if you want, by setting `standardize=None`, or you can pass any standardize function you want to the `standardize` argument.
    - You can prevent splitting by setting `splitting=None`, or you can pass your own splitting function.
    - You can set the `output_sequence_length` argument to ensure that the output sequences all get cropped or padded to the desired length, or you can set `ragged=True` to get a ragged tensor instead of a regular one.
    - Please check out the documentation for more options.
- The word IDs must be encoded, typically using an `Embedding` layer: we will do this in chapter 16.
- Alternatively, you can set the `TextVectorization` layer's `output_mode` argument to `"multi_hot"` or `"count"` to get the corresponding encodings.
- However, simply counting words is usually not ideal: words like "to" and "the" are so frequent that they hardly matter at all, whereas, rarer words such as "computer" are much more informative.
- So, rather than setting `output_mode` to `"multi_hot"` or `"count"`, it is usually preferable to set it to `"tf_idf"`, which stands for *term-frequency $\times inverse-document-frequency$* (TF-IDF).
- This is similar to the count encoding, but words that occur frequently in the training data are down-weighted, and conversely, rare words get up-weighted.
- There are many TF-IDF variants, but the way the `TextVectorization` layer implements it is by multiplying each word count by a weight equal to 
    $$\log\left(1 + \frac{d}{f+1}\right)$$
    where $d$ is the total number of sentences (a.k.a., documents) in the training data and $f$ counts how many of these training sentences contain the given word.
- For example, there are $d = 4$ sentences in the training data, and the word "be" appears in $f = 3$ of these. Since the word "be" occurs twice in the sentence "Question: be or be?", it gets encoded as $2\times \log(1+4/(1+3))\approx 1.3862944$. The word "question" appears only once, but since it's a less common word, its encoding is nearly as high, at $1 \times \log(1+4/(1+1))\approx 1.0986123$. Note that the average weight is used for unknown words.
- This approach to text encoding is straightforward to use and can yield pretty good result for basic natural language processing tasks, but it has several important limitations: it only works with languages that separates words with spaces, it doesn't distinguish between homonyms (e.g., "to bear" versus "the bear"), it gives no hint to your model that words like "interest" and "interested" are related, etc.
- And if you use multi-hot, count, or TF-IDF encoding, then the order of the words is lost.
- One other option is to use the [TensorFlow Text library](https://tensorflow.org/text), which provides more advanced text preprocessing features than the `TextVectorization` layer.
- For example, it includes several subword tokenizers capable of splitting the text into tokens smaller than words, which makes it possible for the model to more easily detect that "evolution" and "evolutionary" have something in common.