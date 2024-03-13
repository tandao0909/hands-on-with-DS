1. Here are some reasons why you want to use tf.data API:
- The compatibility with TensorFlow.
- It helps us load and preprocess data fairly simple.
- It offers many features, including: loading data from various file types (such as text or binary files), reading ghe data in parallel across many threads, transforming it, interleaving the records, shuffling the data, batching it, prefetching it.
2. Some benefits of splitting a large dataset into multiple files:
- We can shuffle the data on a coarser level before shuffling it at a finer level using a shuffling buffer.
- We can handle big dataset that cannot fit in a single machine.
- It's easier to work with thousands of small files than a single large one, for example when splitting data across many subsets.
- We can also distribute the data across many servers, allows many parts of the data to be downloaded simultaneously, increase the network bandwidth.
3. 
- You can use TensorBoard to visualize profiling data: if the GPU is not fully utilized, then you I/O pipeline may be the bottleneck.
- Here are some ways to fix it:
    - Make sure the CPU reads and prepossesses the data across multiple threads, and ensures it prefetches a few batches.
    - If this is not enough, make sure your preprocessing code is optimized.
    - You can also try to saving the data across many TFRecords files, and if necessary perform some of the preprocessing ahead of time so that it does not need to be done on the fly during training (TF transform can help with this).
    - If your budget allows, consider buying a machine with more CPU and RAM, and ensure that the GPU bandwidth is high enough.
4.
- You can save absolutely any binary format in a TFRecord file, as TFRecord can contain a sequence of binary records of varying sizes.
- However, in practice, most TFRecord files contain sequence of serialized protocol buffers.
- This makes it benefit from the advantages of protocol buffers: they can be read easily across multiple platforms and languages and their definition can be updated in a backward-compatible way.
5. 
- The `Example` protobuf format introduces a convention to save data, without having to define one yourself. 
- TensorFlow provides some operations to parse them: the `tf.io.parse_single_example()` and `tf.io.parse_single_example()` functions.
- That said, you can define your own protocol buffer, compile it using the `protoc` compiler, set the `--descriptor_set_out` and `--include_imports` arguments to export the protobuf definition and use the `tf.io.decode_proto()` function to parse the serialized protobuf.
- It's more complicated,a nd it requires deploying the descriptor along with the model, but it can be done.
6.
- You want to active compression when you want to download TFRecord files by the training script, as compressing makes the files smaller, thus download faster.
- But if the dataset is already in the same machine as the training script, you shouldn't compress it to cut down the decompression time for the CPU.
7. 
- If you preprocess directly when writing the data files:
    - Pros: The training script will run much faster, since it will not have to perform preprocessing on the fly. In some cases, the preprocessed data will be much smaller than the original data, save you some RAM and speed up download. It may also be helpful to materialize the preprocessed data, for example to inspect or archive it.
    - Cons: First, you will not be able to experiment with various prepossessing logics, as you need to create a new preprocessed dataset for each variant. Second, if you want to apply data augmentation, you have to materialize many variants of your dataset, which use up a lot of disk space and take a lot of time to generate. Lastly, the model will expect preprocessed data, so you must include the preprocessing steps in the application before it calls the model. There's a risk of code duplication and preprocessing mismatch in this case.
- If you process within the tf.data pipeline:
    - Pros: It's much easier to tweak the preprocessing logic and apply data augmentation. tf.data also makes it easy to build highly efficient preprocessing pipeline (by using multithreading and prefetching).
    - Cons: It will slow down training. Moreover, the instance will now be preprocessed once each epoch, instead of once per training, unless the dataset fits in the RAM and you cache it using the dataset's `cache()` method.
    - The model will still expect preprocessed data, but if you preprocess it using layers, then just reuse these layers in your production model by adding them before training, to avoid code duplication and preprocessing mismatch.
- If you use preprocessing layers within you model:
    - Pros: You only have to write the code once for both of your training and production model. If your model have to be deployed to multiple platforms, then you don't have to write the preprocessing code multiple times. You won't have the risk of using the wrong preprocessing logic for your model, since it will be part of the model
    - Cons: It will slow down training, and each instance is preprocessed once per epoch instead of only once.
8. A few common ways to encode categorical integer features:
- If the feature has an order, such as the rating (e.g., "bad", "average", "good"), then you can use ordinal encoding: sort the categories in their order and map each category to its rank (e.g., "bad" maps to 0, "average" maps to 1, "good" maps to 2).
- However, most categorical features does't have that order, such as jobs and countries. In this case, you can use one-hot encoding, or embeddings if there are many categories.
- With Keras, the `StringLookup` layer can be used for ordinal encoding (using the default `output_mode="int"`), or one-hot encoding (using `output_mode="one_hot"`).
- It can also perform multi-hot encoding (using `output_mode="multi_hot"`) if you want to encode multiple categorical text features together, assuming they share the same categories and it doesn't matter which feature contributed which category.
- For trainable, you must first use `StringLookup` layer to produce an ordinal encoding, then use an `Embedding` layer.
About text:
- The `TextVectorization` is easy to use and it can work well for simple tasks, and you can use TF Text for more advanced features.
- However, you would probably want to use a pretrained language models, which you can obtain using tools like TF Hub or Hugging Face's Transformer library. We will discuss in more detail in chapter 16.