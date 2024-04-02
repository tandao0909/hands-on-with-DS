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

- In a famous [2015 blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) titled "The Unreasonable Effectiveness of Recurrent Neural Networks", Andrej Karpathy showed how to train an RNN to predict the next character in a sentence.
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

## Reusing Pretrained Embeddings and Language Models

- It's impressive that the model is able to learn useful word embeddings based on just 250,000 movie reviews. Just imagine how good the embeddings would be if we had billions of reviews to train on.
- Unfortunately, we don't, but perhaps we can reuse word embeddings trained on some other very large text corpus (e.g., Amazon reviews, available on TensorFlow Datasets), even if it is not composed of movie reviews?
- After all, the word "amazing" generally has the same meaning whether you use it to talk about movies or anything else.
- Moreover, perhaps embeddings would be useful for sentiment analysis even if they were trained on another task: since words like "awesome" and "amazing" have a similar meaning, they will likely cluster in the embedding space, even with the tasks such as predicting the next word in a sentence.
- If all positive words and negative words form clusters, then this will be helpful for sentiment analysis.
- So instead of training word embeddings from scratch, we could just download and use pretrained word embeddings, such as Google's [Word2Vec embeddings](https://arxiv.org/abs/1310.4546), Stanford's [GloVe embeddings](https://nlp.stanford.edu/projects/glove/), or Facebook's [FastText embeddings](https://fasttext.cc/).
- Using pretrained word embeddings was popular for several years, but this approach has its limits.
- In particular, a word has a single representation, no matter the context. For example, the word "right" is encoded the same way in "left and right" and "right and wrong", even though it means two very different things.
- To address this limitation, a [2018 paper](https://arxiv.org/abs/1802.05365) by Matthew Peters introduced *Embeddings from Language Models* (ELMo): these are contextualized word embeddings learned from the internal states of a deep bidirectional language model.
- Instead of just using pretrained embeddings in your model, you reuse part of a pretrained language model.
- At roughly the same time, the [Universal Language Model Fine-Tuning(ULMFiT) paper](https://arxiv.org/abs/1801.06146) by Jeremy Howard and Sebastian Ruder demonstrated the effectiveness of unsupervised pretraining for NLP tasks: the authors trained an LSTM language model on a huge text corpus using self-supervised learning (i.e., generating the labels automatically from the data), then they fine-tuned it on various tasks.
- Their model outperformed the state-of-the-art on six text classification tasks by a large margin (reducing the error rate by 18-25% in most cases).
- Moreover, the authors showed a pretrained model fine-tuned on just 100 labeled examples could achieve the same performance as one trained from scratch on 10,000 examples.
- Before the ULMFiT paper, using pretrained models was only the norm in computer vision; in the context of NLP, pretraining was limited to word embeddings.
- This paper marked the beginning of a new era in NLP: today, reusing pretrained language models is the norm.
- For example, let us build a classifier based on the Universal Sentence Encoder, a model architecture introduced in a [2018 paper](https://arxiv.org/abs/1803.11175) by a team of Google researchers.
- This model is based on the transformer architecture, which we will look at later in this chapter, and it is available on TensorFlow Hub.
- Note that the model is quite large - close to 1GB in size - so it take a while to download.
- By default, TensorFlow Hub models are saved to a temporary directory, and they get downloaded again and again every time you run your program. To avoid this, you must set the `TFHUB_CACHE_DIR` environment variable to a directory of your choice: the modules will then be saved there, and only downloaded once.
- Note that the last part of the TensorFlow Hub module URL specifies that we want version 4 of the model. This versioning ensures that if a new module version is released on TF Hub, it will nto break your model.
- Conveniently, if you just type this URL into your web browser, you will get the documentation for this module.
- Also note that we set `trainable=True` when creating the `hub.KerasLayer`.
- This way, the pretrained Universal Sentence Encoder is fine-tuned during training: some of its weights are tweaked via backpropagation.
- Not all TensorFlow Hub modules are fine-tunable, so make sure to check the documentation for each pretrained model you're working with.
- After training, this model would reach a validation accuracy of over 90%.
- That's actually really good: if you try to perform the task yourself, you will probably do only marginally better since many reviews contain both positive and negative comments. Classifying these ambiguous reviews is like flipping a coin.

## An Encoder-Decoder Network for Neural Machine Translation

- Let's begin with a simple [NMT model](https://arxiv.org/abs/1409.3215) that will translate English sentences to Spanish:
![A simple machine translation model](image-2.png)
- In short, the architecture is as follows: English sentences as fed as inputs to the encoder, and the decoder outputs the Spanish translations.
- Note that the Spanish translations are also used as inputs to the decoder during training, but shifted back by one step.
- In other words, during training, the decoder is given as input the word it *should* have output at the previous step, regardless of what it actually output.
- This is called *teacher forcing*, a technique that significantly speeds up training and improves the model's performance.
- For the very first word, the decoder is given the start-of-sequence (SOS) token, and the decoder is expected to end the sentence with an end-of-sequence (EOS) token.
- Each word is initially represented by its ID (e.g., `854` for the word "soccer").
- Next, an `Embedding` layer returns the word embedding. These word embeddings are then fed to the encoder and the decoder.
- At each step, the decoder outputs a score for each word in the output vocabulary (i.e., Spanish), then the softmax activation function turns these cores into probabilities.
- For example, at the first time step, the word "Me" may have a probability of 7%, "Yo" may have a probability of 1%, and so on. Thw word with the highest probability is the output.
- This is very much like a regular multiclass classification task, and in fact, you can train the model using the `"sparse_categorical_crossentropy"` loss, much like we did in the char-RNN model.
- Note that at inference time (after training), you will not have the target sentence to feed the decoder.
- Instead, you need to feed it the word that it has juts output at the previous step, as shown below. This will require an embedding lookup that is not shown in the diagram.
![At inference time, the decoder is fed as input the word it just output at the previous time step](image-3.png)
- In a [2015 paper](https://arxiv.org/abs/1506.03099), Samy Bengio et al. proposed gradually switching from feeding the decoder the previous *target* token to feeding it the previous *output* token during training.
- Now we can build and train this model. First, we need to download a dataset of English/Spanish pairs. The dataset come from the contributors of [Tatoeba project](https://tatoeba.org/en/). You can the original zip file in the website [https://www.manythings.org/anki/](https://www.manythings.org/anki/).
- Each line contains an English sentence and the corresponding Spanish translation, separated by a tab.
- We'll start by removing the Spanish characters "¡" and "¿", which the `TextVectorization` layer doesn't handle, then we will parse the sentence pairs and shuffle them.
- Finally, we split them into two separate lists, one per language.
- Next, let's create two `TextVectorization` layers - one per language - and adapt them to the text.
- There are a few things to note here:
    - We limit the vocabulary size to 1,000, which is quite small. That's because the training set is not very large, and because using a small value will speed up training. State-of-the-art translation models typically use a much larger vocabulary (e.g., 30,000), a much larger training set (gigabytes), and a much larger model (hundreds or even thousands of megabytes). For example, [Opus-MT models](https://huggingface.co/Helsinki-NLP/opus-mt-en-mt) by the University of Helsinki, or the [M2M-100 model](https://huggingface.co/docs/transformers/model_doc/m2m_100) by Facebook.
    - Since all sentences in the dataset have a maximum of 50 words, we set `output_sequence_length` to 50: this way the input sequences will automatically be padded with zeros until they are all 50 tokens long. If there was any sentence longer than 50 tokens in the training set, it would be cropped to 50 tokens.
    - For the Spanish text, we add "startofseq" and "endofseq" to each sentence when adapting the `TextVectorization` layer: we will use these words as SOS and EOS tokens. You could any other words, along as they are not actual Spanish words.
- Let's see the first 10 tokens in both vocabularies. They start with the padding token, the SOS and EOS tokens (only in the Spanish vocabulary), then the actual words, sorted by decreasing frequency.
- Next, we create the training set and the validation set (you could create a test set if you want to).
- We will use the first 100,000 sentence pairs for training, and the rest for validation. The decoder's inputs are the Spanish sentences plus an SOS token prefix. The targets are the Spanish sentences suffix with an EOS token.
- Now we can build our translation model. We will use the functional API for that, since the model is not sequential.
- It requires two text inputs, so let's start with that.
- Next, we need to encode these sentences using the `TextVectorization` layers was prepared earlier, followed by an `Embedding` layer for each language, with `mask_zero=True` to ensure masking is handled automatically. You can tune the embedding size, as always.
- Now let's create the encoder and passed it the embedded inputs.
- To keep things simple, we just used a single `LSTM` layer, but you could stack several of them.
- We also set `return_state=True` to get a reference to the layer's final state.
- Since we're using an `LSTM` layer, there are actually two states: the short-term state and the long-term state.
- The layer return these states separately, which is why we had to write `*encoder_state` to group both states in a list.
    > In Python, if you run `a, *b = [1, 2, 3, 4]`, then `a` equals 1 and b equals to `[2, 3, 4]`.
- Now we can use this (double) state as the initial state of the decoder.
- Next, pass the decoder's outputs through a `Dense` layer with the softmax activation function to get the word probabilities for each step.
- And that's everything. We just need to cerate the Keras `Model`, compile and train it.
- After training, we can use the model to translate new English sentences to Spanish.
- But it's not as simple as calling `model.predict()`, because the decoder expects as input the word that was predicted at the previous time step.
- However, to keep things simple, we can just call the model multiple times, predicting one extra word at each round.
- In the learning notebook, I create a function to do just that. The function simply keeps predicting one word at a time, gradually completing the translation, and it stops once it reaches the EOS token. It feeds the whole sentence again when trying to predict the next word, so it's not optimized, but our sentences are short.
- If you feed it (very) short sentences, it does indeed works! But if you try longer sentences, well, it turns out really struggles.
- You can try to increase the training set size and add more `LSTM` layers in both the encoder and the decoder. But this approach has a limit, so let's look at more sophisticated techniques.

## Optimizing the Output Layer

- When the output vocabulary is large, outputting a probability for each and every possible word can be quite slow.
- If the target vocabulary contained, say 50,000 Spanish words instead of 1,000, then the decoder would output 50,000-dimensional vectors, and compute the softmax function over such a large vector would be very computationally intensive.
- To avoid this, one solution is to look only the logits output by the model for the correct word and for a random sample of incorrect words, then compute an approximation of the loss based only on these logits.
- This *sampled softmax* technique was [introduced in 2015](https://arxiv.org/abs/1412.2007) by Sébastien Jean et al.
- In TensorFlow, you can use the `tf.nn.sampled_softmax_loss()` function for this  during training and use the normal softmax function at inference time (sampled softmax cannot be used at inference time because it requires knowing the target).
- Another thing you can do to speed up training - which is compatible with sampled softmax - is to tie the weights of the output layer to the transpose of the decoder's embedding matrix (you will see how to tie weights in chapter 17).
- This significantly reduces the number of model parameters, which speeds up training and may sometimes improve the model's accuracy as well, especially if you don't have a lot of training data.
- The embedding matrix is equivalent to one-hoe encoding followed by a linear layer with no bias term and no activation function that maps the one-hot vector to the embedding space.
- The output layer does the reverse.
- So, if the model can find an embedding matrix whose transpose is close to its inverse (such a matrix is called an *orthogonal matrix*), then there's no need to learn a separate set of weights for the output layer.

## Bidirectional RNNs

- At each time step, a regular recurrent layer only look at past and present inputs before generating its output.
- In other words, it is *causal*, meaning it cannot look into the future.
- This type of RNN makes sense when forecasting time series, or in the decoder of a sequence-to-sequence (seq2seq) model.
- But for tasks like text classification, or in the encoder of seq2seq model, it is often preferable ot look ahead at the next word before encoding a given word.
- For example, consider the phrases "the right arm", "the right person", or "the right to vote": to properly encode the word "right", you need to look ahead.
- One solution is to run two recurrent layers on the same inputs, one reading the words from left to right and the other reading them from right to left, then combine their outputs at each time step, typically by concatenating them. This is what a *bidirectional recurrent layer* does.
![A bidirectional recurrent layer](image-4.png)
- To implement a bidirectional recurrent layer in Keras, just wrap a recurrent layer in a `tf.keras.layers.Bidirectional` layer. For example, you can find an example of a `Bidirectional` layer being used as the encoder in a translation model in the learning notebook.
- The `Bidirectional` model will create a clone of the `LSTM` layer, but in the reverse direction, and it will run both and concatenate their outputs at each time step.
- So although the `LSTM` has 128 units, for example, the `Bidirectional` layer will output 20 values per time step.
- There's just one problem. This layer will now return four states instead of two: the final short-term and long-term states of the forward `LSTM` layer, and the final short-term and long-term states of the backward `LSTM` layer.
- We cannot use this quadruple states directly as the initial state of the decoder's `LSTM` layer, since it expects just two states (short-term and long-term).
- We cannot make the decoder bidirectional, since it must remain causal: otherwise it would cheat during training and it would not work.
- Instead, we can concatenate the two short-term states, and also concatenate the two long-term states. This can be seen as double the length of the time series.
- Let's switch our focus on another technique aim at improving the performance of a translation model at inference time: beam search.

## Beam Search

- Suppose we have trained an encoder-decoder model, use it to translate the sentence "I like soccer" to Spanish.
- We'are hoping that it will output the proper translation "me gusta el fútbol", but unfortunately it outputs "me gustan los jugadores", which means "I like the players".
- Looking at the training set, you notice many sentences such as "I like cars", which translates to "me gustan los autos", so it's absurd to the model to output "me gustan los" after seeing "I like".
- Unfortunately, in this case it was a mistake since "soccer" is singular.
- The model couldn't go back and fix it, so it tried to complete the sentence as best as it could, thi case using the word "jugadores".
- How can we will the model a chance to go back and fix mistakes it made earlier?
- One of most common solutions is *beam search*: it keeps track of a short list of the k most promising sentences (say, the top three), and at each decoder step it tries to extend by one word, keeping only the k most likely sentences. The parameter *k* is called the *beam width*.
- For example, suppose you train to train the model to translate the sentence "I like soccer" using beam search with a beam width of 3 (see the figure below).
- At the first decoder step, the model will output an estimated probability for each possible first word in the translated sentence.
- Suppose the top three are "me" (75% estimated probability), "a" (3%), and "como" (1%). That's our short list so far.
- Next, we use the model to find the next word for each sentence. For the first sentence ("me"), perhaps the model outputs a probability of 36% for the word "gustan", 32% for the word "gusta", 16% for the word "encanta", and so on.
- Note that these are actually *conditional* probabilities, given that the sentences starts with "me".
- For the second sentence ("a"), the model might output a conditional probability of 50% for the word "mi", and so on.
- Assuming the vocabulary has 1,000 words, we will end up with 1,000 probabilities per sentence.
- Next, we compute the probabilities of each of the 3,000 two-word sentences we considered ($3 \times 1,000$).
- We do this by multiplying the estimated conditional probability of each word by the estimated probability of the sentence it completes, ultimately apply the Bayesian rule.
- For example, the estimated probability of the sentence "me" was 75%, while the estimated conditional probability of the word "gustan" (given the first word is "me") was 36%, so the estimated probability of the sentence "me gustan" is $75\% \times 36\% = 27\%$.
- After computing the probabilities of all 3,00 two-word sentences, we keep only the top 3. In this example, they all start with the word "me": "me gustan" (27%), "me gusta" (24%), and "me encanta" (12%).
- Right now, the sentence "me gustan" is winning, but "me gusta" has not been eliminated.
![Beam search, with a beam width of 3](image-5.png)
- Then we repeat the same process: we use the model to predict the next word in each of these three sentences, and we compute the probabilities of all 3,000 three-word sentences we considered.
- Perhaps the top three are now "me gustan los" (10%), "me gusta el" (8%), and "me gusta mucho" (2%).
- At the next step, we may get "me gusta el fútbol" (6%), "me gusta mucho el" (1%), and "me gusta el deporte" (0.2%).
- Notice that "me gustan" was eliminated, and the correct translation is now ahead.
- We boosted our encoder-decoyer model's performance without any extra training, simply by using it more wisely.
- The TensorFlow Addons library includes a full seq2seq API that lets you build encoder-decoder models with attention, including beam search and more. However, its documentation is currently rather limited.
- You can implement beam search yourself! Look at the learning notebook ofr an example.
- With all of this, you can get reasonably good translations for fairly short sentences.
- Unfortunately, this model will (still) be really bad at translating long sentences.
- Once again, the problem comes from the limited short-term memory of RNNs.
- *Attention mechanisms* are the game-changing innovation that addressed this problem.

## Attention Mechanisms

- Consider the path from the word "soccer" to its translation "fútbol" back in the figure below: it is quite long!
- This means a representation of this word (along with all the other words) needs to be carried over many steps before it is actually used. How can we make this path shorter?  
- This was the core idea in a landmark [2014 paper](https://arxiv.org/abs/1409.0473) by Dzmitry Bahdanau et al., where the author introduced a technique that allowed the decoder to focus on the the appropriate words (as encoded by the encoder) at each time step.
- For example, at the time step where the decoder needs to output the word "fútbol", it will focus its attention on the word "soccer".
- This means the path from an input word to its translation is now much shorter, so the short-term memory limitations of RNNs have much less impact.
- Attention mechanisms revolutionized neural machine translation (and deep learning in general), allowing a significant improvement in the state of the art, especially for long sentences (e.g., over 30 words).
- The most common metric used in NMT is the *bilingual evaluation understudy* (BLEU) score, which compares each translation produced by the model with several good translations produced by humans: it count the number of *n*-grams (sequences of *n* words) that appear in any target translations and adjusts the score to take into account the frequency of the produced *n*-grams in the target translations. Further reading can be found [here](https://viblo.asia/p/tim-hieu-ve-bleu-va-wer-metric-cho-1-so-tac-vu-trong-nlp-Eb85oA16Z2G), note that this is written in Vietnamese.
- The figure below shows our encoder-decoder model with an added attention mechanism.
- One the left is our friends encoder and decoder.
- Instead of just sending the encoder's final hidden state to the decoder, as well as the previous target word at each step (which is still done, although it is not shown in the figure), we now send all of the encoder's outputs to the decoder as well.
- Since the decoder cannot deal with all these encoder outputs at once, they need to be aggregated: at each time step, the decoder's memory cell computes a weighted sum of all the encoder outputs. This determined which words it will focus on at this time step.
- The weight $\alpha_{(t,i)}$ is the weight of the i-th encoder output at the t-th decoder time step.
- For example, if the weight $\alpha_{(3,2)}$ is much larger than the weight $\alpha_{(3,0)}$ and $\alpha_{(3,1)}$, then the decoder will pay much more attention to the encoder's output for word #2 ("soccer") than to the other two outputs, at least at this time step.
- The rest of the decoder works just like earlier: at each time step, the memory cell receives the inputs we just discussed, plus the hidden state from the previous time step, and finally (although it is not represented in the diagram) it receives the target word from the previous time step (or at inference time, the output from the previous time step).
![Neural machine translation using an encoder–decoder network with an attention model](image-6.png)
- But how do we find these weights?
- Well, they are generated by a small neural network called an *alignment model* (or an *attention layer*), which is trained jointly with the rest of the encoder-decoder model. This alignment model is illustrated on the right-hand side of the figure above.
- It starts with a `Dense` layer composed of a single neuron for each of the encoder's outputs, along with the decoder previous hidden state (e.g., $\textbf{h}_{(2)}$).
- This layer outputs a score (or energy) for each encoder output (e..g, $e_{(3,2)}$): this score measures how well each output is aligned with the decoder's previous hidden state.
- For example, in the figure above, the model has already output "me gusta el" (meaning "I like"), so it's now expecting a noun: the word "soccer" is the one that best aligns with the current state, so it gets a high score.
- Finally, all the scores go through a softmax layer to get the final weight for each encoder output (e.g, $\alpha_{(3, 2)}$). All the weights for a given decoder time step add up to 1.
- This particular attention mechanism is called *Bahdanau attention* (named after the 2014 paper's first author).
- Since it concatenates the encoder output with the decoder's previous hidden state, it is sometimes called *concatenative attention* (or *additive attention*).
- If the input sequence is *n* words long, and assuming the output sentence is about as long, then this model will need to compute about $n^2$ weights.
- Fortunately this quadratic computational complexity is still tractable because even long sentences don't have thousands of words.
- Another common attention mechanism, know as *Luong attention* or *multiplicative attention*, was proposed shortly after, in [2015](https://arxiv.org/abs/1508.04025) by Minh-Thang Luong et al.
- Because the goal of the alignment model is to measure the similarity between one of the encoder's outputs and the decoder's previous hidden state, the authors proposed to simple compute the dot product of these two vectors, as this is often a fairly good measure, and modern hardware cna compute it very efficiently.
- For this to be possible, both vectors must have the same dimensionality.
- The dot product gives a score, and all the scores (at a given decoder time step) go though a softmax layer to give the final weights, just like in Bahdanau attention.
- Another simplification Luong et al. proposed was to use the decoder's hidden state at the current time step rather than at the previous time step (i.e., $\textbf{(t)}$ rather than $\textbf{h}_{(t-1)}$), then to use the output of the attention mechanism (noted $\tilde{\textbf{h}}_{(t)}$) directly to compute the decoder's predictions, rather than using to compute the decoder's current hidden state.
- The researchers also proposed a variant of the dot product mechanism where the encoder outputs first go through  a fully connected layer (without the bias term) before the dot products are computed. This is called the "general" dot product approach.
- The researchers compared both dot product approaches with the concatenative mechanism (adding a rescaling parameter vector **v**), and they observed that the dot product variants performed better than concatenative attention.
- For this reason, concatenative attention is much less used now.
- The equations for these three attention mechanisms are summarized below:
    $$\tilde{\textbf{h}}_{(t)}=\sum_i \alpha_{(t,i)}\textbf{y}_{(i)} $$
    $$\text{with} \alpha_{(t,i)} = \frac{\exp(e_{(t,i)})}{\displaystyle\sum_{i'} \exp(e_{t,i'})} $$
    $$\text{and} e_{(t,i)} = \begin{cases} 
    \textbf{h}_{(i)}^\intercal \textbf{y}_{(i)} \text{ dot} \\
    \textbf{h}_{(i)}^\intercal \textbf{W}\textbf{y}_{(i)} \text{ general} \\
    \textbf{v}^\intercal \tanh(\textbf{W}[\textbf{h}_{(t)}; \textbf{y}_{(i)}]) \text{ concat} \\
    \end{cases} $$
- Keras provides a `tf.keras.layers.Attention` layer for Luong attention, and an `AdditiveAttention` layer for Bahdanau attention.
- Let's add Luong attention to our encoder-decoder model. Since we will need to pass all the encoder's outputs to the `Attention` layer, we first need to set `return_sequences=True` when creating the encoder.
- Next, we need to create the attention layer and pass it the decoder's states and the encoder's outputs.
- However, to access the decoder's states at each step we would need to write a custom memory cell.
- For simplicity, let's use the decoder's outputs instead of its states: in practice, this works well too, and it's much easier to code.
- Then we just pass the attention layer's outputs directly to the output layer, as suggested in the Luong attention paper.
- That's it! If you train the model, you'll find it can deal with much longer sentences.
- In short, the attention layer provides a way to focus the attention of the model on part of the inputs.
- But there's another way to think of this layer: it acts as a differentiable memory retrieval mechanism.
- For example, let's suppose the encoder analyzed the input sentence "I like soccer", and it managed to understand that the word "I" is the subject and the word "like" is the verb, so it encoded this information in its outputs for these words.
- Now suppose the decoder has already translated the subject, and it thinks that it should translate the verb next.
- For this, it needs to fetch the verb from the input sentence.
- This is analogous to a dictionary lookup: it's as if the encoder has created a dictionary {"subject": "They", "verb": "played", ...} and the decoder wanted to look up the value that corresponds to the key "verb".
- However, the model des not have discrete tokens to represent the keys (like "subject" or "verb"); instead, it has vectorized representations of these concepts that it learned during training, so the query it will use for lookup will not perfectly match any key in the dictionary.
- No problem! The solution is just compute a similarity measure between the query and each key in the dictionary, and then use the softmax function to convert these similarity scores to weights that add up to 1.
- If the key that represents the verb is by far the most similar to the query, then that key's weights will be close to 1.
- A good analogy is you type in the Google search bar some words, that's the query, then Google receives these words, asks the database which information best matches theses words, that's the key, and the database returns the values, which's of course the value. Now replace you with the decoder, Google with the encoder, the database with the internal understanding of the encoder.
- Next, the attention layer computes a weighted sum of the corresponding values: if the weight of "verb" key is closed to 1, then weighted sum will be very close to the representation of the word "played".
- This is why the Keras's `Attention` and `AdditiveAttention` layers both expect a list of inputs, containing two or three items: the *queries*, the *keys*, and optionally the *values*.
- If you do not pass any values, then they are automatically equal to the keys.
- So looking at the previous code example, the decoder outputs are the queries, and the encoder outputs are both the keys and the values. For each decoder output (i.e., each query), the attention layer returns a weighted sum of the encoder outputs (i.e., the keys/values) that are most similar to the decoder output.
- The bottom line is that an attention mechanism is a trainable memory retrieval system.
- It's so powerful that you can build state-of-the-art models using only attention mechanisms. Enter the transformer architecture.

## Attention Is All You Need: The Original Transformer Architecture

- In a groundbreaking [2017 paper](https://arxiv.org/abs/1706.03762), a  team of Google researchers suggested that "Attention Is All You Need".
- They created an architecture called the *transformer*, which significantly improved the state-of-the-art in NMT without using any recurrent or convolutional layers, just attention mechanisms (plus embedding layers, dense layers, normalization layers, and a few other bits and pieces).
- Because the model is not recurrent, it doesn't suffer as much from the unstable gradient problems as RNNs, it can be trained in fewer steps, it's easier to parallelize across multiple GPUs, and it can better capture long-range patterns than RNNs.
- The original 2017 transformer architecture is represented below:
![The original 2017 transformer architecture](image-7.png)
- In short, the left part of the figure is the encoder, and the right part is the decoder.
- Each embedding layer outputs a 3D tensor of shape [*batch size*, *sequence length*, *embedding size*].
- After that, the tensors are gradually transformed as they flow through the transformer but their shape remains the same.
- If you use the transformer for NMT, then during training you must feed the English sentences to the encoder and the corresponding Spanish translations to the decoder, with an extra SOS token inserted at the start of each sequence.
- At inference time, you must call then transformer multiple times, producing the translations one word at a time and feeding the partial translations to the decoder at each round, just like we did earlier in the `translate()` function.
- The encoder's role is to gradually transform the inputs - word representations of the English sentence - until each word's representation perfectly captures the meaning of the word, in the context of the sentence.
- For example, if you feed the encoder with the sentence "I like soccer", then the word "like" will start off with a rather vague representation, since this word could mean different things in different contexts: think of "I like soccer" versus "It's like that". 
- But after going through the encoder, the word's representation should capture the correct meaning of "like" in the given sentence (i.e., to be fond of), as well as any other information that may be required for translation (e.g., it's a verb).
- The decoder's role is to gradually transform each word representation in the translated sentence into a word representation of the next word in the translation.
- For example, if the sentence to translate is "I like soccer", and the decoder's input sentence is "<SOS> me gusta el fútbol", then after going through the decoder, the word representation of the word "el" will end up transformed into a representation of the word "fútbol".
- Similarly, the representation of the word "fútbol" will be transformed into a representation of the EOS token.
- After going through the decoder, each word representation goes through a final `Dense` layer with a softmax activation function, which will hopefully output a high probability for the correct next word and a low probability for all other words.
- The predicted sentence should be "me gusta el fútbol <EOS>".
- That was the big picture; now let's walk through the figure below in more detail.
- First, notice that both the encoder and the decoder contain modules that are stacked N times. In the paper, N = 6. The final outputs of the whole encoder stack are fed to the decoder at each of these N levels.
- Zooming in, you can see that you are already similar with most components: there are two embedding layers; several skip connections, each of them followed by a layer normalization; several feedforward modules that are composed two dense layers each (the first one using the ReLU activation function, the second with no activation function); and finally the output layer is a dense layer using the softmax activation function.
- You can also sprinkle a bit of dropout after the attention layers and the feedforward modules, if needed.
- Since all of these layers are time-distributed, each word is treated independently from all the others.
- But how can we translate a sentence by looking at the words completely separately? Well, we can't, so that's where the new components come in:
    - The encoder's *multi-head attention* layer updates each word representation by attending to (paying attention to) all other words in the same sentence. That's where the vague representation of the word "like" become a richer and more accurate representation, capturing its precise meaning in the given sentence. We'll discuss exactly how this works later.
    - The decoder's *masked multi-head attention* layer does the same thing, but when it processes a word, it doesn't attend to words located after it: it's a causal layer. For example, when it processes the word "gusta", it only attends to the words "<SOS> me gusta", and it ignores the words "el fútbol" (or else that would be cheating).
    - The decoder's upper *multi-head attention layer* is where the decoder pays attention ot the words in the English sentence. This is called *cross-attention*, not *self-attention* in this case. For example, the decoder will probably pay close attention to the word "soccer" when it processes the word "soccer" and transforms its representation into a representation of the word "fútbol".
    - The *positional encodings* are dense vectors (much like word embeddings) that represent the position of each word in the sentence. The n-th positional encoding is added to the word embedding of the n-th word in each sentence. This is needed because all layers in the transformer architecture ignore word positions: without positional encodings, you could shuffle the output sequences in the same way, and it would just shuffle the output sequence the same way. Obviously, the order of words matters, which is why we need to give positional encodings to the word to the transformer somehow: adding positional encodings to the word representations is a good way to achieve this.
- The first two arrows going into each multi-head attention layer in the figure below represents the keys and values, and the third arrow represents the queries.
- In the self-attention layers, all three are equal to the word representations output by the previous layer, while in the decoder's upper attention layer, the keys and values are equal to the encoder's final word representations, and the queries are equal to the word representations output by the previous layer.

### Positional encodings

- A positional encoding is a dense vector that encodes the position of a word within a sentence: the i-th positional encoding is added to the word embedding of the i-th word in the sentence.
- The easiest way to implement this is to use an `Embedding` layer and make it encode all the positions from 0 to the maximum sequences length in the batch, then add the result to the word embeddings.
- The rules of broadcasting will ensure that the positional encodings get applied to every input sequence.
- Note that the implementation in the learning notebook assumes that the embeddings are represented as regular tensors, not ragged tensors.
- The encoder and the decoder share the same `Embedding` layer for the positional encodings, since they have the same embedding size (this is often the case).
- Instead of using trainable positional encodings, the authors of the transformer paper chose to use fixed positional encodings, based on the sine and cosine functions at different frequencies.
- The positional encoding matrix $\textbf{P}$ is defined in the equation below and represented at the top of the figure below (transposed), where $P_{(p,i)}$ is the i-th component of the encoding for the word located at the p-th position in the sentence.
    $$P_{(p,i)} = \begin{cases}
    \sin (p/10000^{i/d})  \text{ if i is even} \\
    \cos (p/10000^{(i-1)/d})  \text{ if i is odd} \\ 
    \end{cases} $$
- In the equation, $d$ is the dimensionality of the embedding space. 
- This solution can give the same performance as trainable positional encodings, and it can extend to arbitrarily long sentences without adding any parameters. However, when there is a huge amount of pretraining data, trainable positional encodings are usually favored.
- After these positional encodings are added to the word embeddings, the rest of the model has access to the absolute position of each word in the sentence because there is a unique positional encoding for each position.
- For example, the positional encoding for the word located at the 22nd position in a sentence is presented by the vertical dashed line at the top left of the figure in the learning notebook, and you can see it's unique to that position.
- Moreover, the choice of oscillating functions (sine and cosine) makes it possible for the model to learn relative positions as well.
- For example, words located 38 words apart (e.g., at positions p = 22 and p = 60) always have the same positional encoding values in the encoding dimensions i = 100 and i = 101, as you can see in the figure.
- In other words, there're components (e.g., the 100-th and 101-st) in the embedding vector, which are exact some steps apart (e.g., 38), are equal to each other, no matter where they are. The only relevant information is the relative positions.
- This explains why we need both the sine and cosine for each frequency: if we only used the sine (the blue wave at i = 100), the model would not be able to distinguish positions p = 22 and p = 35 (marked by a cross).
- There is no `PositionalEncoding` layer in TensorFlow, but it is not too hard to create one.
- For efficiency reasons, we pre-compute the positional encoding matrix in the constructor.
- The `call()` method just truncates this encoding matrix to the max length of the input sequences, and it adds them to the inputs.
- We also set `supports_masking=True` to propagate the input's automatic mask to the next layer.

### Multi-head attention
- To understand how a multi-head attention layer works, we must first understand the *scaled dot-product attention* layer, which it is based on.
- Its equation is shown below, in a vectorized form. It's the same as Luong attention, except for a scaling factor.
    $$\text{Attention} (\textbf{Q}, \textbf{K}, \textbf{V}) = \text{softmax} \left(\frac{\textbf{Q}\textbf{V}^\intercal}{\sqrt{d_{\text{keys}}}} \right)\textbf{K} $$
- In this equation:
    - $\textbf{Q}$ is a matrix containing one row per *query*. Its shape is [$n_{\text{queries}}, d_{\text{keys}} $], where $n_{\text{queries}}$ is the number of queries and $d_{\text{keys}}$ is the number of dimensions of each query and each key, which means each key and each query must have the same dimensionality.
    - $\textbf{K}$ is a matrix containing one row per *key*. Its shape is [$n_{\text{keys}}, d_{\text{keys}}$], where $n_{\text{keys}}$ is the number of keys and values.
    - $\textbf{V}$ is a matrix containing one row per *value*. Its shape is [$n_{\text{keys}}, d_{\text{values}}$], where $d_{\text{values}}$ is the number of dimensions of each value.
    - The shape of $\textbf{Q}\textbf{V}^\intercal$ is [$n_{\text{queries}}, n_{\text{keys}}$]: it contains one similarity score for each query/key pair. To prevent this matrix from being huge, the input sequences must not be too long (we will discuss how to overcome this limitation later in this chapter).
    - The output of the softmax function has the same shape, but all rows sum up to 1.
    - The final output has a shape [$n_{\text{queries}}, d_{\text{values}}$]: there is one row per query, where each row represents the query result (a weighted sum of the values).
    - The scaling factor $1 / (\sqrt{d_{\text{keys}}})$ scales down the similarity scores to avoid saturating the softmax function, which would lead to tiny gradients.
    - It is possible to mask out some key/value pairs by adding a very large negative value to the corresponding similarity scores, just before computing the softmax. This is useful in the masked multi-head attention layer.
- If you set `use_scale=True` when creating a `tf.keras.layers.Attention` layer, then it will create an additional parameter that lets the layer learn how to properly downscale the similarity scores.
- The scaled dot-product attention used in the transformer model is almost the same, except it always scales the similarity by the same factor, $1 / (\sqrt{d_{\text{keys}}})$.
- Note that the `Attention` layer's inputs are just like $\textbf{Q}$, $\textbf{K}$, and $\textbf{V}$, except with an extra batch dimension (the first dimension).
- Internally, the layer computes all the attention scores for all sentences in the batch with just one call to `tf.matmul(queries, keys)`: this makes it extremely efficient.
- In fact, in TensorFlow, if `A` and `B` are tensors with more than two dimensions - say, of shape [2, 3, 4, 5] and [2, 3, 5, 6], respectively - then, `tf.matmul(A, B)` will treat these tensors as $2 \times 3$ arrays where each cell contains a matrix, and it will multiply the corresponding matrices: the matrix at the i-th row and j-th column in `A` will be multiplied by the matrix at the i-th row and j-th column in `B`. Since the product of a $4 \times 5$ matrix with a $5 \times 6$ matrix is a $4 \times 6$ matrix `tf.matmul(A, B)` will return an array of shape [2, 3, 4, 6].
- Now we're ready to look at the multi-head attention layer. Its architecture is shown below:
![Multi-head attention layer architecture](image-8.png)
- As you can see, it is just a bunch of scaled dot-product attention layers, each preceded by a linear transformation of the values, keys, and queriers (i.e., a time-distributed dense layer with no activation function).
- All the outputs are simply concatenated, and they go through a final linear transformation (again, time-distributed).
- But what's the intuition behind this architecture? Well, consider once again the word "like" in the sentence "I like soccer".
- The encoder was smart enough to encoder the fact that it is a verb. But the word representation also include its position in the text, thanks to the positional encodings, and it probably includes many other features that are useful for its translation, such as the fact that it is in the present tense.
- In short, the word representation encodes many different characteristics of the word.
- If we just used a single scaled dot-product attention layer we would only be able to query all of these characteristics in one shot.
- This is why the multi-head attention layer applies *multiple* different linear transformation of the values, keys, and queries: this allows the model to apply many different projections of the word representation into different subspaces, each focusing on a subset of the word's characteristics.
- Perhaps one of the linear layers will project the word representation into a subspace where all that remains is the information that the word is a verb, another linear layer will extract just the fact that it is present tense, and so on.
- Then the scaled dot-product attention layers implement the lookup phase, and finally we concatenate all the results and project them back to the original space.
- Keras includes a `tf.keras.layers.MultiHeadAttention` layer, so we now have everything we need to build the rest of the transformer.
- Let's start with the full encoder, which is exactly as the architecture shown in the figure above, except we use a stack of two blocks (`N = 2`) instead of six, since we don't have a huge training set, and we add a bit of dropout as well.
- The code can be found in the learning notebook. It's straightforward, except for masking.
- The `MultiHeadAttention` layer does not support automatic masking ([yet](https://github.com/keras-team/keras/issues/16248)), so we must handle it manually.
- The `MultiHeadAttention` layer accepts an `attention_mask` argument, which is a Boolean tensor of shape [*batch size*, * max query length*, *max value length*]: for every token in every query sequence, this mask indicates which tokens in the corresponding value sequence should be attended to.
- We want to tell the `MultiHeadAttention` layer to ignore all the padding tokens in the values. 
- So we first compute the padding tokens using `tf.math.not_equal(encoder_input_ids, 0)`. This returns a Boolean tensor of shape [*batch size*, *max sequence length*].
- We then insert a second axis  using `[:, tf.newaxis]`, to get a mask of shape [*batch size*, 1, *max sequence length*].
- This allows us to use this mask as the `attention_mask` when calling the `MultiHeadAttention` layer: thanks to broadcasting, the same mask will be used for all tokens in each query. This way, the padding tokens in the values will be ignored correctly.
- However, the layer will compute outputs for every single token, including the padding tokens.
- We need to mask the outputs that correspond to these padding tokens.
- Recall that we use `mask_zero` in the `Embedding` layers, and we set `support_masking` to `True` in the `PositionalEncoding` layer, so the automatic mask was propagated all the way to the `MultiHeadAttention` layer's inputs (`encoder_in`).
- We can use this to our advantage in the skip connection: indeed, the `Add` layer supports automatic masking, so when we add `Z` and `skip` (which is initially equal to `encoder_in`), the outputs get automatically masked correctly.
- Now on the decoder! Once again, masking is going to be he only tricky part.
- The first multi-head attention layer is a self-attention layer, like in the encoder, but it is a *masked* multi-head attention layer, meaning it is causal: it should ignore all tokens in the future. So we need two masks: a padding mask and a casual mask.
- The padding mask is exactly the same as the one we created for the encoder, except it's based on the decoder's inputs rather than the encoder's.
- The casual mask is created by using the `tf.linalg.band_part()` function, which takes a tensor and return a copy with all the values outside a diagonal band set to zero.
- With these arguments, we get a square matrix of size `batch_max_len_dec` (the max length of the input sequences in the batch), with 1s in the lower-left triangle and 0s in the upper right.
- If you use this mask as the attention mask, you will get exactly what we want: the first query token will only attend to the first value token the second will only attend to the first two, the third will only attend to the first three, and so on. 
- In other words, query tokens cannot attend to any value token in the future.
- For the first attention layer, we use `causal_mask & decoder_pad_mask` to mask both the padding tokens and future tokens.
- The causal mask only has two dimensions: it's missing the batch dimension, but that's okay since broadcasting ensures that it gets copied across all the instances in the batch.
- For the second attention layer, there's nothing special.
- The only thing to note is that we are using `encoder_pad_mask`, not `decoder_pad_mask`, because this attention layer uses the encoder's final outputs as its values.
- Now we just need to add the final output layer, create the model, compile and train it. That's a full transformer from scratch, and trained it for automatic translation. This is getting quite advanced!
- The Keras team has created a new [Keras NLP project](https://github.com/keras-team/keras-nlp), including an API to build a transformer more easily. Also check out the new [Keras CV project for computer vision](https://github.com/keras-team/keras-cv)

## An Avalanche of Transformer Models

- The year 2018 has been called the "ImageNet moment for NLP". Since then, progress ahs been astounding, with larger and larger transformer-based architectures trained on immense datasets.
- First, the [GPT paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) by Alec Radford and other OpenAI researchers once again demonstrated the effectiveness of unsupervised pretraining, like the ELMo and ULMFiT paper before it, but this time using a transformer-like architecture.
- The authors pretrained a large but fairly simple architecture composed of a stack of 12 transformers modules using only masked multi-head attention layers, like in the original transformer's decoder.
- They trained it on a very large dataset, using the same autoregressive technique we used for our Shakespearean char-RNN: just predict the next token. This is a form of semi-supervised learning.
- Then they fine-tuned it on various language tasks using only minor adaptions for each task.
- The task was quite diverse: they include text classification, *entailment* (whether sentence A imposes, involves, or implies sentence B as a necessary consequence), similarity (e.g., "Nice weather today" is very similar to "It is sunny"), and question answering (given a few paragraphs of text giving some context, the model must answer some multiple-choice questions).
- Then Google's [BERT paper](https://arxiv.org/abs/1810.04805) came out: it also demonstrated the effectiveness of self-supervised pretraining on a large corpus, using a similar architecture to GPT but with non-masked multi-head attention layers only, like in the original transformer's encoder.
- This means the model is naturally bidirectional; hence the B in BERT (*Bidirectional Encoder Reorientations from Transformers*).
- Most importantly, the authors proposed two pretraining tasks that explain most of the model's strength.
- *Masked language model (MLM)*: Each word in a sentence has a 15% probability of being masked, and the model is trained to predict the masked words.
- For example, if the original sentence is "She had fun at the birthday party", then the model may be given the sentence "She <mask> fun at the <mask> party" and it must predict the words "had" and "birthday" (the other outputs will be ignored).
- To be more precise, each selected word has an 80% chance of being masked, a 10% chance of being replaced by a random word (to reduce the discrepancy between pretraining and fine-tuning, since the model will not see <mask> token during fine-tuning), and a 10% chance of being left alone (to bias the model toward the correct answer).
- *Next sentence prediction (NSP)*: The model is trained to predict whether two sentences are consecutive or not.  
- For example, it should predict that "The dog sleeps" and "It snores loudly" are consecutive sentences, while "The dog sleeps" and "The Earth orbits the Sun" are not consecutive.
- Later research showed that NSP was not as important as was initially thought, so it was dropped in most later architectures.
- The model is trained on these two tasks simultaneously, see the figure below.
- For the NSP task, the authors inserted a class token (<CLS>) at the start of every input, and the corresponding output token represents the model's prediction: sentence B follows sentence A, or it does not.
- The two input sentences are concatenated, separated only by a special token (<SEP>), and they are fed as input to the model.
- To help the model knows which sentence each input token belongs to, a *segment embedding* is added on top of each token's positional embeddings: there are just two possible segment embeddings, one for sentence A and one for sentence B.
- For the MLM task, some input words are masked (as we just saw) and the model tries to predict what those words were.
- The loss is only computed on the NSP prediction and the masked tokens, not on the unmasked ones.
![BERT training and fine-tuning process](image-9.png)
- After this unsupervised pretraining phase on a very large corpus of text, the model is then fine-tuned on many different tasks, changing very little for each task.
- For example, for text classification such as sentiment analysis, all output tokens are ignored except the first one, corresponding to the class token, and a new output layer replaces the previous one, which was just a binary classification layer for NSP.
- In February 2019, just a few months after BERT was published, Alec Radford, Jeffrey Wu, and other OpenAI researchers published the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), which proposed a very similar architecture to GPT, but larger still (with over 1.5 billion parameters).
- The researchers showed that the new and improved GPT model could perform *zero-shot learning* (ZSL), meaning it could achieve good performance on many tasks without fine-tuning.
- This was just the start of the race toward larger and larger models: Google's [Switch Transformers](https://arxiv.org/abs/2101.03961) (introduced in January 2021) used 1 trillion parameters, and soon much larger models came out, such as Wu Dao 2.0 model by the Beijing Academy of Artificial Intelligence (BAII), announced in June 2021.
- An unfortunate consequence of this trend toward gigantic models is that only well-funded organizations can afford to train such models: it can easily cost hundreds of thousands of dollars or more. And the energy required to train a single model corresponds to an American household's electricity consumption for several years; it's not eco-friendly at all.
- Many of these models are just too big to even be used on regular hardware: they wouldn't fit in RAM, and they would be horribly slow.
- Lastly, some are so costly that they are not released publicly.
- Luckily, ingenious researchers are finding new ways to downsize transformers and make them more data-efficient.
- For example, the [DistilBERT model](https://arxiv.org/abs/1910.01108), introduced in October 2019 by Victor Sanh et al. from Hugging Face, is a small and fast transformer model based on BERT.
- It's available on Hugging Face's excellent model hub, along with thousands of others - we'll look at an example later in this chapter.
- DistilBERT was trained using *distillation* (hence the name): this means transferring knowledge from a teacher model to a student one, which is usually much smaller than the teacher model.
- This is typically done by using the teacher's predicted probabilities for each training instance as targets for the student.
- Surprisingly, distillation often works better than training the student from scratch on the same dataset as the teacher! Indeed, the student benefits from the teacher more nuanced labels.
- Many more transformer architectures came out after BERT, almost on a monthly basis, often improving on the state-of-the-art across all the NLP tasks: XLNet (June 2019), RoBERTa (July 2019), StructBert(August 2019), ALBERT (September 2019), T5 (October 2019), ELECTRA (March 2020), GPT3 (May 2020), DeBERTa (June 2020), Switch Transformers (January 2021), Wu Dao 2.0 (June 2021), Gopher (December 2021), GPT-NeoX-20B (February 2022), Chinchilla (March 2022), OPT (May 2022), and the list goes on and on.
- Each of these models brought new ideas and techniques, which is summarized [here](https://www.topbots.com/leading-nlp-language-models-2020/), but the author particularly like the [T5 paper](https://arxiv.org/abs/1910.10683) by Google researchers: it frames all NLP tasks as text-to-text, using an encoder-decoder transformer.
- For example, to translate "I like soccer" to Spanish, you can just call the model with the input sentence "translate English to Spanish: I like soccer" and it outputs "me gusta el fútbol". To summarize a paragraph, you just enter "summarize:" followed by the paragraph, and it outputs the summary. For classification, you only need to change the prefix to "classify:" and the model outputs the class name, as text.
- This simplifies using the model, adn it also makes it possible to pretrain it on even more tasks.
- Last but not least, in April 2022, Google researchers used a new large scale training platform called *Pathways* (which we will briefly discuss in chapter 19) to train a humongous language model named the [Pathways Language Model (PaLM)](https://arxiv.org/abs/2204.02311), with a whopping 540 billion parameters, using over 6,000 TPUs.
- Other than its incredible size, this model is a standard transformer, using only decoders (i.e., with masked multi-head attention layers), with just a few tweaks (see the paper for details).
- This model achieved incredible performance on all sorts of NLP tasks, particularly in natural language understanding (NLU).
- It's capable of impressive feats, such as explaining jokes, giving detailed step-by-step answers to questions, and even coding.
- This is in part due to the model's size, but also thanks to a technique called [Chain of thought prompting](https://arxiv.org/abs/2201.11903), which was introduced a couple months earlier by another team of Google researchers.
- In question answering tasks, regular prompting typically includes a few examples of questions and answers, such as: "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? A: 11." The prompt then continues with the actual question, such as "Q: John takes care of 10 dogs. Each dogs takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?", and the model's job is to append the answer: in this case, "35".
- But with chain of thought prompting, the example answer include all the reasoning steps that lead to the conclusion.
- For example, instead of "A: 11", the prompt contains "A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11." This encourages the model to give a detailed answer to the actual question, such as "John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. So that is 10 x .5 = 5 hours a day. 5 hours a day x 7 days a week = 35 hours a week. The answer is 35 hours a week." This is an actual example from the paper!
- Not only does the model give the right answer much more frequently than using regular prompting - we're encouraging the model to think things through - but it also provides all the reasoning steps, which can be useful to better understand the rationale behind a model's answer.
- Transformers have been taken over NLP, but they didn't stop there: they soon expanded to computer vision as well.