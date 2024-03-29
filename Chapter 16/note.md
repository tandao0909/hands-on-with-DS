- When Alan Turing imagined his famous [Turing test](https://web-archive.southampton.ac.uk/cogprints.org/499/1/turing.html) in 1950, he proposed a way to evaluate a machine's ability to match human intelligence.
- He could have tested for many things, such as the ability to recognize human in a picture, solve a mathematic problem, escape a maze, create a song, but, interestingly, he chose a linguistic task.
- More specifically, he devised a chatbot capable of fooling its interlocutor into thinking it was a human.
- This test does have it weaknesses: a set of hardcode rules can fool unsuspecting or naive humans (e.g., a human could give vague predefined answers in response to some keywords, it could pretend that it is joking or drunk to get a pass on its weirdest answers, ot it could escape difficult questions by answering them with its own questions), and many aspects of human intelligence are utterly ignored (e.g., the ability to interpret nonverbal communication such as facial expressions, or to learn a manual task).
- The test however highlight the fact that mastering language is arguably *Homo sapiens*'s greatest cognitive ability.
- Building a machine that can master written and spoken language is the ultimate goal of NLP research, so in practice researchers focus on more specific tasks, such as text classification, translation, summarization, question answering, and many more.
- A common approach for natural language tasks is to use recurrent neural networks.
- We will therefore continue to explore RNNs (introduced in chapter 15), starting with a *character RNN*, or *char-RNN*, trained to predict the net character in a sentence. This will alow us to generate some original text.
- We first use a *stateless RNN* (which learns on random portions of text at each iteration, without any information about the rets of the text), then we build a *stateful RNN* (which preserves the hidden state between training iterations and continues reading where it left off, allowing it to learn longer patterns).
- Next, we build an RNN to perform sentiment analysis (e.g., reading movie reviews and extracting the rater's    feeling about the movie), this time treating sentences as sequences of words, rather than characters.
- Then we will show how RNNs can be use to build an encoder-decoder architecture capable of performing neural machine translation (NMT), translating English to Spanish.
- In the second part of this chapter, we will explore *attention mechanisms*.
- As the name suggests, these are neural network components that learn ot select the part of inputs that the rest of the model should focus on at each time step.
- First, we will boost the performance of an RNN-based encoder-decoder architecture using attention.
- Next, we drop RNNs altogether and use a very successful attention-only architecture, called the *transformer*, to build a translation model.
- We will then discuss some of the most important advances in NLP in the last few years, including incredibly powerful language models such as GPT or BERT, both based on transformers.
- Lastly, we'll see hwo to get started with the excellent Transformers library by Hugging Face.

# Generating Shakespearean Text Using a Character RNN

- In a famous [2015 blog post]() titled "The Unreasonable Effectiveness of Recurrent Neural Networks", Andrej Karpathy showed how to train an RNN to predict the next character in a sentence.
- This *char-RNN* can then be used to generate novel text, one character at a time.

## Creating the Training Dataset

- First, using Keras's `tf.keras.utils.get_file()` function to download all of Shakespeare's work. We download it from 's [char-rnn project]().
- Next, we use `tf.keras.layers.TextVectorization` layer (introduced in Chapter 13) to encode this text. We set `split="character"` to get character-level encoding rather than the default word-level encoding, and we use `standardize="lower"` to convert the text to lowercase, which simplify the task.
- Each character is mapped to an integer, starting at 2. The `TextVectorization` layer reserved the value 0 for the padding tokens, and reserved 1 for unknown characters.
- We don't need neither of these two tokens, so we subtract 2 form the character IDs and compute the number of distinct characters and the total number of characters.
- Next, like we did in chapter 15, we cna turn this very long sequence into a dataset of windows that we can then use to train a sequence-to-sequence RNN.
- The target will be similar to the inputs, but shifted by one time step into "future".
- For example, one sample in the dataset may be a sequence of character IDs representing the text "to be or not to b" (without the final "e"), and the corresponding target is a sequence of character IDs representing the text "o be or not to be" (with the final "e", but without the leading "t").
- This function starts much like the `to_windows()` custom utility function we created in Chapter 15:
    - It takes a sequence as input (i.e., the encoded text), and creates a dataset containing all the windows of the desired length.
    - It increases the length by one, since we need the next character for the target.
    - Then it shuffles the windows (optionally), batches them, splits them into input/output pairs, and activates prefetching.
- The figure below summarizes the dataset preparation steps: it shows of length 11, and a batch size of 3. The start index of each window is indicated next to it.
![Preparing a dataset of shuffled windows](image.png)
- Now we're ready to create the training set, the validation set, and the test set. We will use roughly 90% of the text for training, 5% for validation, and 5% for testing.
- We set the window length to 100, but you can try tuning it: it's easier and faster to train RNNs on shorter input sequences, but the RNN will not be able to learn any pattern longer than `length`, so don't make it too small.

## Building and Training the Char-RNN model

- Since out dataset is reasonably large, and modeling language is quite a difficult task, we need more than a simple RNN with a few recurrent neurons.
- We build and train a model with one `GRU` layer composed of 128 units. You can try tweaking the number of layers and units later, if needed.
- We'll walk through the implementation in the learning notebook:
    - We use an `Embedding` layer as the first layer, to encode the character IDs (embeddings was introduced in chapter 13). The `Embedding` layer's number of input dimensions is the number of distinct character IDs, and the number of output dimensions is a hyperparameter you can tune, we'll set it to 16 for now.
    - Whereas the inputs of the `Embedding` layer will be 2D tensors of shape [*batch size, window length*], the output of the `Embedding` layer will be a 3D tensor of shape [*batch size, window length, embedding size*].
    - We use a `Dense` layer for the output layer: it must have 39 units (`n_tokens`) because there are 39 distinct characters in the text, and we want to output a probability for each possible character (at each time step). The 39 output probabilities should sum up to 1 at each time step, so we apply the softmax activation function to the outputs of the `Dense` layer.
    - Lastly, we compile the model using the `sparse_categorical_crossentropy` loss function and a Nadam optimizer, and we train the model for several epochs, using a `ModelCheckpoint` callback to save the best model (in term of validation accuracy) as training progress.
- This model does no handle text preprocessing, so we wrap it in a final model containing the `tf.keras.layers.TextVectorization` layer as the first layer, plus a `tf.keras.layers.Lambda` layer to subtract 2 from the character IDs since we're not using the padding and unknown tokens for now.

## Generating Fake Shakespearean Text

- To generate new text using the Char-RNN model, we could feed it some text, make the model predict the most likely next letter, add it to the end of the text, them give the extended text to the model to guess the next letter, and so on. This is called *greedy decoding*.
- But in practice, this often leads to the same words being repeated over and over again.
- Instead, we can sample the next character randomly, with a probability equal to the estimated probability, using TensorFlow's `tf.random.categorical()` function. This will generate more diverse and interesting text.
- The `categorical()` function randomly samples random class indices, given the class log probabilities (logits).
- To have more control over the diversity of the generated text, we can divide the logits by a number called the *temperature*, which we can tweak.
- A temperature close to zero favors high-probability characters, while a high temperature gives all characters an equal probability.
- Lower temperatures are typically preferred when generating fairly rigid and precise text, such as mathematical equations, while higher temperatures are preferred when generating more diverse and creative text.
- You can look at how things unroll in the learning notebook.
- To generate more convincing text, a common technique is to sample only from the top k characters, or only from a the smallest set of top characters whose total probability exceeds some threshold (this is called *nucleus sampling*).
- Alternatively, you could try using *beam search*, which we will discuss later in this chapter, or using more `GRU` layers and more neurons per layer, training for longer, and adding some regularization if needed.
- Also note that the model is currently incapable of learning patterns longer than `length`, which is just 100 characters.
- You could try making this window larger, but it will also make training harder, and even LSTM and GRU cells cannot handle very long sequences.
- An alterative approach is to use a stateful RNN.

## Stateful RNN

- Until now, we have only used *stateless RNNs*: at each training iteration, the model starts with a hidden sate full of zeros, then it updates this state at each time step, and after the last time step, it throws away its hidden state, as it is no longer needed.
- What if we instructed the RNN to preserve this final state after processing a training batch and use it as the initial state for the next training batch?
- This way the model could learn long-term patterns despite only backpropagating through short sequences. This is called a *stateful RNN*.
- First, note that a stateful RNN only makes sense if each input sequence in a batch starts exactly where the corresponding sequence in the previous batch left off.
- So the first thing we need to do to build a stateful RNN is to use sequential and non-overlapping input sequences (rather than shuffled and overlapping sequences we used to train stateless RNNs).
- When creating the `tf.data.Dataset`, we must therefore use `shift=length` (instead of `shift=1`) when calling the `window()` method.
- Moreover, we must not call the `shuffle()` method.
- Unfortunately, batching is much harder when preparing a dataset for a stateful RNN than it is for a stateless RNN.
- In fact, if we were to call `batch(32)`, then 32 consecutive windows would be put in the same batch, and the following batch would not continue each of these windows where it left off.
- The first batch would contain windows 1 to 32 and the second batch would contain windows 33 to 64, so if you consider the first window of each batch, for example, (i.e., windows 1 and 33), you can see that they are not consecutive.
- The simplest solution to this problem is to just use a batch size of 1. The `to_dataset_for_stateful_rnn()` function in the learning notebook does just that. The figure below summarizes the main steps of this function:
![Preparing a dataset of consecutive sequence fragments for a stateful RNN](image-1.png)
- Batching is harder, but not impossible.
- For example, we could chop Shakespeare's text into 32 texts of equal length, create one dataset of consecutive input sequences for each of them, and finally use `tf.data.Dataset.zip(datasets).map(lambda *windows: tf.stack(windows))` to create proper consecutive batches, where the n-th input sequence in a batch starts off exactly where the n-th input sequence ended in the previous batch. See the learning notebook for the code.
- Now, let's create the stateful RNN. We need to set the the `stateful` argument to `True` when creating each recurrent layer.
- Because the stateful RNN needs to know the batch size (since it will preserve a state for each input sequence in the batch), we must set the `batch_input_shape` argument in the first layer.
- Note that we can leave teh second dimension unspecified, since the input sequences could have any length.
- At the end of each epoch, we need to reset the state before we go back to the beginning of the text. We can use a custom Keras callback for this.
- Then we can compile and train the model using our callbacks.
- After the model is trained, it will only be able to make predictions for batches of the same size as the were used during training. To avoid this restriction, create an identical *stateless* model, and copy the stateful model's weights to this model.
- Interestingly, although a Char-RNN model is just trained to predict the next character, this seemingly simple actually requires it to learn some higher-level tasks as well.
- For example, to find the next character after "Great movie, I really", it's helpful to understand that the sentence is positive, so what follows is more likely to be the letter "l" (for "loved") rather than "h" (for "hated").
- In fact, a [2017 paper](https://arxiv.org/abs/1704.01444) by Alec Radford and other OpenAI researchers describes how the authors trained a big Char-RNN-like model on a large dataset, and found that one the the neurons acted as an excellent sentiment analysis classifier.
- Although the model was trained without any labels, the *sentiment neuron* - as they called it - reached state-of-the-art performance on sentiment analysis benchmarks. This foreshowed and motivated unsupervised pretraining in NLP.
- But before we explore unsupervised learning, let us talk about word-level models and how to use them in a supervised fashion for sentiment analysis.

# Sentiment Analysis

- In real-life projects, one of the most common applications of NLP is text classification, especially sentiment analysis.
- If image classification on the MNIST dataset is the "Hello world!" of computer vision, then sentiment analysis on the IMDb dataset is the "Hello world!" of natural language processing.
- The IMDb dataset consists of 50,000 movie reviews in English (25,000 for training, 25,000 for testing) extracted from the famous [Internet Movie Database](https://imdb.com), along with a simple binary target for each review indicating whether it is negative (0) or positive (1).
- We load the IMDb dataset using the TensorFlow Dataset library (introduced in chapter 13).
- We'll use the first 90% of the training set for training, and the remaining 10% for validation.
- If you inspect a few reviews, then you'll realize that some reviews are easy to classify. For example, the first review include the words "terrible movie" in the very first sentence.
- But in many cases, things are not that simple. For example, the third review starts off positively, even though it's ultimately a negative review.
- To build a model for this task, we need to preprocess this text, but this time chop into words instead of characters.
- For this, we can use the `tf.keras.layers.TextVectorization` layer again.
- Note that it uses spaces to identify word boundaries, which will not work well in some languages.
- For example, Chinese writing does not use spaces between words, Vietnamese uses spaces even within words, and German often attaches multiple words together, without spaces. Even in English, spaces are not always the best way to tokenize text: think of "San Francisco" or "#ILoveDeepLearning" or "state-of-the-art".
- Fortunately, there are solutions to address these issues. In a [2016 paper](https://arxiv.org/abs/1508.07909), Rico Sennrich et al. explored several methods to tokenize and detokenize text at the subword level.
- This way, even if your model encounters a rare word it has never been seen before, it can still reasonably guess what it means.
- For example, even if the model never saw the word "smartest" during training, if it learned the word "smart" and it also learned that the suffix "est" means "the most", it can infer the meaning of "smartest".
- One of the techniques the authors evaluated is *byte pair encoding* (BPE). BPE works by splitting the whole training set into individual characters (including spaces), then repeatedly merging the adjacent pairs until the vocabulary reaches the desired size.
- A [2019 paper](https://arxiv.org/abs/1804.10959) by Taku Kudo at Google further improved subword tokenization, often removing the need for language-specific preprocessing prior to tokenization.
- Moreover, the paper proposed a novel regularization technique called *subword regularization*, which improves accuracy and robustness by introducing some randomness in tokenization during training: for example, "New England" may be tokenized as "New" + "England", or "New" + "Eng" + "land", or simply "New England" (just one token).
- Google's [SentencePiece](https://github.com/google/sentencepiece) provides an open source implementation, which is described in a [paper](https://arxiv.org/abs/1808.06226) by Taku Kudo and John Richardson.
- The [TensorFlow Text](https://medium.com/tensorflow/introducing-tf-text-438c8552bd5e) library also implements various tokenization strategies, including [WordPiece](https://arxiv.org/abs/1609.08144) (a variant of BPE), and last and not least, the [Tokenizers library by Hugging Face](https://huggingface.co/docs/tokenizers/index) implements a wide range of extremely fast tokenizers.
- However, for the IDMb task in English, using spaces for token boundaries should be good enough.
- We will create a `TextVectorization` layer and adapt it to the training set. The vocabulary will be limited to 1,000 tokens, including the most frequent 998 words plus a padding token and a token for unknown words, since it's unlikely that very rare words will be important for this task, and limiting the vocabulary size will reduce the number of parameters the model needs to learn.
- Now we can create the model and train it.
- The first layer is the `TextVectorization` layer we just prepared, followed by an `Embedding` layer that will convert word IDs into embeddings.
- The embedding matrix needs to have one row per token in the vocabulary (`vocab_size`) and one column per embedding dimension (this example uses 128 dimensions, but this is a hyperparameter you could tune).
- Next, we use a `GRU` layer and a `Dense` layer with a single neuron and the sigmoid activation function, since this is a binary classification task: the model's output will be the estimated probability that the review expresses a positive sentiment regrading the movie.
- We then compile the model, and we fit it on the dataset we prepared earlier for a couple of epochs (or you can train for longer to get better results).
- Sadly, if you run this code, you will generally find that the model fails to learn anything at all: the accuracy remains close to 50%, not better than random chance. Why?
- The reviews have different lengths, so when the `TextVectorization` layer converts them to sequence of token IDs it pads the shorter sequence using the the padding token (with ID 0) to make them as long as the longest sequence in the batch.
- As a result, most sequences end with many padding tokens - often dozens or even hundreds of them.
- Even though we're using a `GRU` layer, which is much better than a `SimpleRNN` layer, its short-term memory is still not great, so when it goes through many padding tokens, it ends up forgetting what the reviews was about.
- One solution is to feed the model with batches of equal-length sentences (which also speeds up training). Another solution is to make the RNN ignore the padding tokens, which can be done using masking.

## Masking

- Making the model ignore padding tokens is trivial using Keras: simply add `mask_zero=True` when creating the `Embedding` layer.
- This means that padding tokens (whose IDs is 0) will be ignored by all downstream layers.
- Thw way this works is that the `Embedding` layer creates a *mask tensor* equal to `tf.math.not_equal(inputs, 0)`: it is a Boolean tensor with the same shape as the inputs and it is equal to `False` anywhere the token IDs are 0, and `True` otherwise.
- This mask tensor is then automatically propagated by the model to the next layer.
- If that layer's `call()` method has a `mask` argument, then it automatically receives the mask. This allows the layer to ignore the appropriate time steps.
- Each layer will handle differently, but in general, they simply ignore masked times steps (i.e., time steps for which the mask is `False`).
- For example, when a recurrent layer encounters a masked time step, it simply copies the output from the previous time step.
- Next, if the layer's `supports_masking` attribute is `True`, the mask is automatically propagated to the next layer. 
- It keeps propagating this way for as long as the layers have `supports_masking=True`.
- As an example, a recurrent layer's `supports_masking` attribute is `True` when `return_sequences=True`, but it's `False` when `return_sequences=False`, since there's no need for a mask anymore in this case.
- So if you have a model with several recurrent layers with `return_sequences=True`, followed by a recurrent layer with `return_sequences=False`, then the mask will automatically propagate to the last recurrent layer: that layer will use the mask to ignore masked steps, but it will not propagate the mask to the next layer.
- Similarly, if you set `mask_zero=True` when creating the `Embedding` layer in the sentiment analysis model we just built, the `GRU` layer will receive and use the mask, but it will not pass the mask to the next layer, since it don't set `return_sequences=True`.
- Some layers need to update the mask before propagating it to the next layer: they do so by implementing the `compute_mask()` method, which takes two arguments: the input and the previous mask. It then computes the updated mask and return it. The default implementation of `compute_mask()` just return the previous mask unchanged.
- May Keras layers support masking: `SimpleRNN`, `GRU`, `LSTM`, `Bidirectional`, `Dense`, `TimeDistributed`, `Add`, and a few more (all in the `tf.keras.layers` package).
- However, convolutional layers (including `Conv1D`) do not support masking - it's not obvious how they would do so anyway.
- If the mask propagates all the way to the output, then it gets applied to the losses as well, so the masked time steps will not contribute to the loss (their loss will be 0). This assumes that the model output sequences, which is not the case in our sentiment analysis model.
- The `LSTM` and `GRU` layers have an optimized implementation for GPUs, based on Nvidia's cuDNN library.
- However, this implementation only supports masking if all the padding tokens are at the end of the sequences.
- It also requires you to use the default value for several hyperparameters: `activation`, `recurrent_activation`, `recurrent_dropout`, `unroll`, `use_bias`, and `reset_after`. If that's not the case, then these layers will fall back to the (much slower) default GPU implementation.
- If you want to implement you own custom layer with masking support, you should add a `mask` argument to the `call()` method, and obviously make the method uses the mask.
- Additionally, if the mask must be propagated to the next layers, then you should set `self.support_masking=True` in the constructor.
- If the mask must be updated before it's propagated, then you must implement the `compute_mask()` method.
- If your model does not start with an `Embedding` layer, you can use the `tf.keras.layers.Masking`  layer instead: by default, it sets the mask to `tf.math.reduce_any(tf.math.not_equal(X, 0), axis=-1)`, meaning that time steps where the last dimension (the innermost vector) is full of zero will be masked out in subsequent layers.
- Using masking layers and automatic mask propagation works best for simple models.
- It will not work well for more complex models, such as when you need to mix `Conv1D` layers with recurrent layers.
- In such cases, you will need to explicitly compute the mask and pass it ot the appropriate layers, using either the functional API or the subclassing API.
- For example, the model in the learning notebook is equivalent ot the previous model, except it is built using the functional API, and handles masking manually. It also adds a bit if dropout since the previous model was overfitting slightly.
- One last approach is to feed the model with ragged tensors.
- In practice, all you need to do is set `ragged=True` when creating the `textVectorization` layer, so that the input sequences are represented as ragged tensors.
- Keras's recurrent layers have built-in support for ragged tensors, so there's nothing else you need to do. There's no need to pass `mask_zero=True` or handle masks explicitly - it's all implemented for you.
- Whichever masking approach you prefer, after training this model for a few epochs, it will become quite good at judging whether a review is positive or not.
- If you use the `tf.keras.callbacks.TensorBoard()` callback, you can visualize the embeddings in TensorBoard as they are being learned: it's fascinating to watch words like "awesome" and "amazing" gradually cluster on one side of the embedding space, while words like "awful" and "terrible" cluster on the other side.
- Some words are not as positive as you might expect (at least with this model), such as the word "good", presumably because many negative reviews contains the phrases "not good".