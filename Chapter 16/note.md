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
- In fact, a [2017 paper]() by Alec Radford and other OpenAI researchers describes how the authors trained a big Char-RNN-like model on a large dataset, and found that one the the neurons acted as an excellent sentiment analysis classifier.
- Although the model was trained without any labels, the *sentiment neuron* - as they called it - reached state-of-the-art performance on sentiment analysis benchmarks. This foreshowed and motivated unsupervised pretraining in NLP.
- But before we explore unsupervised learning, let us talk about word-level models and how to use them in a supervised fashion for sentiment analysis.