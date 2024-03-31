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