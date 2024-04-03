1. Here are some pros and cons of using a stateful RNN versus a stateless RNN:
- Definition: A stateless RNN drops the hidden state completely after a training iteration, while a stateful one preserves it.
- Pros: Preserving the hidden state across training iterations allow stateful RNN to learn patterns longer than the length of the window context, while a stateless one can't.
- Cons: Preparing a dataset for a stateful RNN is much harder: you must ensure each input sequence in a batch starts exactly where the corresponding sequence in the previous batch left off. Moreover, stateful RNN requires the dataset to be independent and identically distributed (IID).
- Gradient descent doesn't very much fond of non-IID dataset.
2. Here are some reasons why people use encoder-decoder RNNs rather than plain sequence-to-sequence RNNs for automatic translation:
- Because the last word may affect the whole meaning of the sentence, hence we need to know the last word before making any prediction.
- Using encoder-decoder RNNs allows the model to capture the whole sentence in its hidden state, which allows it to make a better translation.
3. Here is how to deal with variable-length input sequences:
- You may not know the length of each sequence, but you may know the upper bound of all the sequences' length. In this case, you just need to use padding to ensure all the sequence have the same length, and using making to ensure that the RNN ignore padding tokens.
- You may want to batch sequences of similar length for better performance.
- Ragged tensors can hold sequences of variable lengths, and Keras now supports them, which simplify handling variable-length input sequences (it still does not handle ragged tensors as the targets on the GPU, though).
About variable-length output sequence:
- Regarding variable-length output sequence, if you know in advance the length of the output sequences (e.g., it equals to the length of the input sequence), then you just need to configure the loss function to ignore tokens that comes after the end of the sequences.
- Similarly, the code that will use the model should also ignore the tokens beyond the end of the sequence.
- But generally, the length of the output sequence is unknown, so you may need to train the model to output an end-of-sequence token at the end of the sequence.
4. 
- Beam search is an algorithm about making a sequence prediction, used to improve the performance of a trained Encoder-Decoder model, for example, in a neural machine translation system.
- The algorithm works by keeping track of the top *k* most promising output sequences so far (e.g., the top three), and at each decoder step, it tries to extend the output sequence by one word, then it keeps only the top *k* most likely sequences.
- The parameter *k* is called the *beam width*: the larger it is, the more CPU and RAM is required, and the more accurate the model will be.
- Instead of greedily choosing the most likely next word at each step to extend a single sentence (which may be just locally optimal), beam search allows the model to explore several promising sentences simultaneously.  - 
- Moreover, this technique lends itself well to parallelization.
- You can implement beam search by writing a custom cell.
- Alternatively, TensorFlow Addons's seq2seq API provides an implementation.
5. 
- Attention mechanism is technique initially used in Encoder-Decoder models to give the decoder more direct access to the input sequence, allowing it to deal with longer input sequences.
- At each decoder time step, the current decoder's state and the full outputs of the encoder are processed by an alignment model that outputs an alignment score for each input time step.
- This score indicates which part of the inputs are the most relevant to the current decoder time step.
- The weighted sum of the encoder input is then fed to the decoder, which produces the next decoder state and the output for this time step.
- The purpose of this architecture is it lets the decoder focuses on the word it need to be attention to, which allows the model to process longer input sequences.
- Another benefit is that the alignment score allows us to debug and interpret the model easier: when the model makes a mistake, you can look at which part of the input it was paying attention to, and this can help diagnose the issue.
6.
- The most important layer in the transformer architecture is the Multi-Head Attention layer. The original Transformer contains 18 of them, including 6 Masked Multi-Head Attention layer.
- Its purpose is allow the model to identify which words is the most aligned with each other, consider many contextual meaning space, and improve each word's representation using these contextual clues.
7. Here are some words about sampled softmax:
- Sampled softmax is a variant of softmax, where instead of taking into account all classes, it just calculates the logits of the target class and a random sample of wrong classes.
- This can speed up training considerably when there are too many classes, for example, 1,000 classes in the ImageNet dataset.
- At inference time, you can't use sampled softmax (as you don't know the target class), so you must use the regular softmax.