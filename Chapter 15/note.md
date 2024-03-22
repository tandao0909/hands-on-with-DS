- Predicting the (near) future is something we do everyday, from knowing which dishes would we eat by smelling from the kitchen, to finishing our friend's sentence.
- In this chapter, we will talk about recurrent neural networks (RNNs) - a class of networks that can predict the future (up to a point).
- RNNs can analyze time series data, such as the number of daily active user on a website, the hourly temperature in your city, your home's daily power consumption, the trajectories of nearby cars, and more.
- Once an RNN learned past patterns in the data, it is able to use its knowledge to forecast the future, assuming of course that past patterns still hold in the future.
- More generally, RNNs can work on sequences of arbitrary lengths, rather than on fixed-sized inputs.
- For example, they can take sentences, documents, or audio samples as input, making them extremely useful fo natural language applications such as automatic translation or speech-to-text.
- However, RNNs are not the only type of neural networks capable of handling sequential data. For small sequences, a regular dense network can do the trick, and for very long sequences, such as audio samples or text, convolutional neural networks can actually work quite well too.

# Recurrent Neurons and Layers

- Up to now, we have focused on feedforward neural networks, where the activations flow only in one direction, from the input layer to the output layer.
- A recurrent neural network looks very much like a feedforward neural network, except it also has connections pointing backwards.
- Let's look at the simplest possible RNN, composed of one neuron receiving inputs, producing an output, and sending that output back to itself, as shown in the left of the figure below:
![A recurrent neuron (left) unrolled through time (right)](image.png)
- At each *time step t* (also called a *frame*), this *recurrent neuron* receives the inputs $\textbf{x}_{(t)}$ as well as its own output from the previous time step, $\hat{y}_{(t-1)}$.
- Since there is no previous output at the first time step, it is generally set to 0.
- We can represent this tiny network against the time axis, as shown in the right part. This is called *unrolling the network through time* (it's the same recurrent neuron represented once per time step).
- You can easily create a layer of recurrent neurons.
- At each time step $t$, every neuron receives both the input vector $\textbf{x}_{(t)}$ and the output vector from the previous time step $\hat{\textbf{y}}$, as shown below:
![A layer of recurrent neurons (left) unrolled through time (right)](image-1.png)
- Note that both the inputs and outputs are now vectors (when there was just a single neuron, the output was just a scalar).
- Each recurrent neuron has two sets of weights: one for the inputs $\textbf{x}_{(t)}$ and the other for the outputs of the previous time step $\hat{\textbf{y}}_{(t-1)}$.
- Let's call these weight vector $\textbf{w}_x$ and $\textbf{w}_y$. If we consider the whole recurrent layer instead of just one recurrent neuron, we can then place all the weight vectors in two weight matrices: $\textbf{W}_x$ and $\textbf{W}_{\hat{y}}$.
- The output vector of the whole recurrent layer can then be computed pretty much as we did up until now:
    $$\hat{\textbf{y}} = \varphi(\textbf{W}_x^T\textbf{x}_{(t)}+\textbf{W}^T_{\hat{y}}\hat{\textbf{y}}_{(t-1)}+\textbf{b})$$
    where:
    - $\textbf{b}$ is the bias vector.
    - $\varphi()$ is the activation function
- Just as with feedforward neural networks, we can compute a recurrent layer's output in one shot for an entire mini-batch by placing all the inputs at time step $t$ into an input matrix $\textbf{X}_{(t)}$:
    $$\begin{align*}
    \hat{\textbf{Y}} &= \varphi(\textbf{X}_{(t)}\textbf{W}_x+\hat{\textbf{Y}}_{(t-1)}\textbf{W}_{\hat{y}}+\textbf{b}) \\
                     &= \varphi([\textbf{X}_{(t)} \hat{\textbf{Y}}_{(t-1)}]\textbf{W}+\textbf{b}) \text{ with }
                    \textbf{W}= \begin{bmatrix}
                    \textbf{W}_x\\
                    \textbf{W}_{\hat{y}}\\
                     \end{bmatrix}
    \end{align*}$$
- In this equation:
    - $\hat{\textbf{Y}}_{(t)}$ is an $m \times n_{\text{neurons}}$ matrix containing the layer's outputs at time step $t$ for each instance in the mini-batch, where $m$ is the number of instances in the mini-batch and $n_{\text{neurons}}$ is the number of neurons.
    - $\textbf{X}_{(t)}$ is an $m \times n_{\text{inputs}}$ matrix containing the inputs for all instance, where $n_{\text{neurons}}$ is the number of input features.
    - $\textbf{W}_x$ is an $n_{\text{inputs}} \times n_{\text{neurons}}$ matrix containing the connection weights for the inputs at the current time step.
    - $\textbf{W}_{\hat{y}}$ is an $n_{\text{neurons}} \times n_{\text{neurons}}$ matrix containing the connection weights for the outputs of the previous time step.
    - $\textbf{b}$ is a vector of size $n_{\text{neurons}}$ containing each neuron's bias term.
    - The weight matrices $\textbf{W}_x$ and $\textbf{W}_{\hat{y}}$ are often concatenated vertically into a single weight matrix $\textbf{W}$ of shape $(n_{\text{inputs}} + n_{\text{neurons}}) \times n_{\text{neurons}}$.
    - The notation $[\textbf{X}_{(t)} \hat{\textbf{Y}}_{(t-1)}]$ represents the horizontal concatenation of the matrices $\textbf{X}_{(t)}$ and $\hat{\textbf{Y}}_{(t-1)}$.
- Notice that $\hat{\textbf{Y}}_{(t)}$ is a function of $\textbf{X}_{(t)}$ and $\hat{\textbf{Y}}_{(t-1)}$, which is a function of $\textbf{X}_{(t-1)}$ $\hat{\textbf{Y}}_{(t-2)}$, which is a function of $\textbf{X}_{(t-2)}$ and $\hat{\textbf{Y}}_{(t-3)}$, and so on. This makes $\hat{\textbf{Y}}_{(t)}$ a function of all the input since time $t=0$ (that is, $\textbf{X}_{(0)}, \textbf{X}_{(1)}, \dots, \textbf{X}_{(t)}$).
- At the first time step, $t=0$, there are no previous inputs, so they are typically assumed to be zeros.

## Memory Cells

- Since the output of a recurrent neuron at time step $t$ is a function of all the inputs from previous time step, we can say it as a form of *memory*.
- A part of the neural network that preserves some states across time steps is called a *memory cell* (or simply a *cell*).
- A single recurrent neuron, or a layer of recurrent neurons, is a very basic cell, capable of learning only short patterns (typically about 10 steps long, but this varies depending on the task).
- Later in this chapter, we will look at some more complex and powerful types of cells capable of learning longer patterns (roughly 10 times longer, but again, this depends on the task).
- A cell's state at time step $t$, denoted $\textbf{h}_{(t)}$ (the "h" stands for "hidden"), is a function of some inputs at that time step and its state at the previous time step: $\textbf{h}_{(t)} = f(\textbf{x}_{(t)}, \textbf{h}_{(t-1)})$.
- Its output at time step $t$, denoted $\hat{\textbf{y}}_{(t)}$, si also a function of the previous state and the current inputs.
- In the case of the basic cells we have discussed so far, the output is just equal to the state, but in more complex cells this is not always he case, as shown below:
![A cell’s hidden state and its output may be different](image-2.png)

## Input and Output Sequences

- An RNN can simultaneously take a sequence of inputs and produce a sequence of outputs (see the top-left network in the figure below).
- This type of *sequence-to-sequence network* is useful to forecast time series, such as your home's daily power consumption: you feed it the data over the last $N$ days, and you train it to output the power consumption shifted by one day into the future (i.e., from $N-1$ days ago to tomorrow).
- Alteratively, you could feed the network a sequence of inputs and ignore all outputs expect for the last one (see the top-right network). This is a *sequence-to-vector network*.
- For example, you could the network a sequence of words corresponding to a movie review, and the network would putout a sentiment score (e.g., from 0 [hate] to 1 [love]).
- Conversely, you could feed the network the same input vector over and over again at each time step and let it putout a sequence (see the bottom-left network). This is a *vector-to-sequence network*.
- For example, the input could be an image (or the output of a CNN), and the output could be a caption for that image.
- Lastly, you could have a sequence-to-vector network, called an *encoder*, followed by a vector-to-sequence network, called a *decoder* (see the bottom-right network).
- For example, this could be used for translating a sentence form one language to another. You would feed the network a sentence in one language, the encoder would convert this sentence into a single vector representation, and then the decoder would decode this vector into a sentence in another language.
- This two-step model, called an [encoder-decoder](https://aclanthology.org/D13-1176.pdf), works much better than trying to translate on the fly with a single sequence-to-sequence RNN (like the one represented at the top left): the last word of a sentence can affect the first word of the translation, so you need to wait until you have seen the whole sentence before translating it.
- We will go through the implementation of an encoder-decoder in chapter 16 (as you'll see, it is a bit more complex than what this figure suggests).
![Sequence-to-sequence (top left), sequence-to-vector (top right), vector-to-sequence (bottom left), and encoder–decoder (bottom right) networks](image-3.png)

# Training RNNs

- To train an RNN, the trick is to unroll it through time (like we just did) and then use regular backpropagation (see the figure below). This strategy is called *backpropagation through time* (BPTT):
![Backpropagation through time](image-4.png)
- Just like in regular backpropagation, there is a first forward pass through the unrolled network (represented by the dashed arrows).
- Then the output is evaluated using a loss function $\textbf{L}(\textbf{Y}_{(0)}, \textbf{Y}_{(1)}, \dots, \textbf{Y}_{(T)}, \hat{\textbf{Y}}_{(0)}, \hat{\textbf{Y}}_{(1)}, \dots, \hat{\textbf{Y}}_{(T)})$, where $\textbf{Y}_{(i)}$ is the i-th target value, $\hat{\textbf{Y}}_{(i)}$ is the i-th prediction, and $T$ is the max time step.
- Note that this loss function can ignore some outputs: For example, in a sequence-to-vector, every output is ignored, except for the last one; in the figure below, the loss function take into account only the last three outputs.
- The gradients of that loss function are then propagated through the unrolled network, represented by the solid arrows.
- In this example, since the output $\hat{\textbf{Y}}_{(0)}$ and $\hat{\textbf{Y}}_{(0)}$ are not used to compute the loss, the gradients do not flow backward through them; they only flow through $\hat{\textbf{Y}}_{(2)}$, $\hat{\textbf{Y}}_{(3)}$, and $\hat{\textbf{Y}}_{(4)}$.
- MOreover, since the same parameter $\textbf{W}$ and $\textbf{b}$ are used at each time step, their gradients will be tweaked multiple times during backpropagation.
- Once the backward phase is complete and all the gradients have been computed, BPTT can perform a gradient descent step to update the parameters (this is the same as regular backpropagation).
- Luckily, Keras takes care of all this complexity for us, as we'll see. But before that, let's load a time series and start analyzing it using classical tools to better understand what we're dealing with, as well as some baseline metrics.

## Forecasting a Time Series

- Assume we need to build a model to forecast the number of passengers that will ride on bus and rail the next day. We have accessed to daily ridership data since 2001. You can look up for it [online](https://data.cityofchicago.org/Transportation/CTA-Ridership-Daily-Boarding-Totals/6iiy-9s97).
- We start, of course, by loading and cleaning up the data. We load the CSV file. set short column names, sort he rows by date, remove the redundant `total` column, and drop duplicate rows.
- Look at the `df.head()` result: On January 1st, 2001, there are 297,192 people boarded a bus, and 125,455 boarded a train in Chicago.
- The `day_type` column contains `W` for **W**eekdays, `A` for S**a**turday, and `U` for S**u**nday or holidays.
- We plot the bus and rial ridership figures over a few months in 2019, to see what it looks like.
- Note that Pandas includes both the start and end month in the range, so this plots the data from the 1st fo March all the way up to 31st of May.
- This is a *time series*: data with values at different time steps, usually at regular intervals.
- More specifically, since there are multiple values per time step, it would be a *multivariate time series*. 
- If we only look at the `bus` column, it would be a *univariate time series*, with a single value per time step.
- Predicting future value (ie., forecasting) is the most typical task when dealing with times series, and this is what we will focus on in this chapter.
- Other tasks include imputation (filling in missing past values), classification, anomaly detection, and more.
- Looking at the plot, we can see that a similar pattern is clearly repeated every week. This is called a weekly *seasonality*.
- In fact, it's so strong in this case that forecasting tomorrow's ridership by just copying the values from the a week earlier will yield reasonably good results. This is called *naive forecasting*: simply copying a past value to make our forecast.
- Naive forecasting is often a great baseline, and it can even be tricky to beat in some cases.
- In general, naive forecasting means copying the latest known value (e.g., forecasting that tomorrow will be the same as today). However, in our case, copying the value from the previous week works better, due to strong weekly seasonality.
- To visualize these naive forecasts, let's overlay the tow time series (for bus and rail) as well as the time series lagged by one week (i.e., shifted toward the right) using dotted lines.
- We'll also plot the difference between the two (i.e., the value at time $t$ minus the value at time $t-7$); this is called *differencing*.
- If you look at the plot, you can notice how closely the lagged time series track the actual time series.
- When a time series is correlated with a lagged version of itself, we say that the time series is *autocorrelated*.
- As you can see, most of the differences are fairly small, except at the end of May. If you check the `day_type` column, you'll see that there was a long weekend back then, as the Monday was the Memorial Day holiday.
- We could use this column to improve our forecasts, but for now let's juts measure the mean absolute error over the three-month period we're arbitrarily focusing on - March April, and May 2019 - to get a rough idea.
- Our naive forecasts get an MAE of about 43,916 bus riders, and about 42,143 rail riders.
- It's hard to tell at a glance how good or bad this is, so let's put the forecast errors into perspective dividing them by the target values.
- What we just computed is called the *mean absolute percentage error* (MAPE): it looks like our naive forecasts give us a MAPE of roughly 8.3% for bus and 9.0% for rail.
- It's interesting to note that the MAE for the rail forecasts looks slightly better than the MAE for the bus forecasts, while the opposite is true for MAPE.
- That's because the bus ridership is larger than the rail ridership, so naturally the forecast errors are also larger, but when we put the errors into perspectives, it turns out that the bus forecasts are actually slightly better than the rial forecasts.
- The MAE, MAPE, and MSE are among the most common metrics you can use to evaluate your forecasts. As always, choose the right metric depend on the task.
- For example, if your project suffers quadratically more from large errors than small ones, then the MSE may be preferable, as it strongly penalizes large errors.
- Looking at the time series, these doesn't appear to be any significant monthly seasonality, but let's check whether there's any yearly seasonality.
- We'll look at the data from 2001 to 2019. to reduce the risk of data snooping, we'll ignore recent data for now.
- Let's also plot a 12-month rolling average for each series to visualize long-term trends.
- Yep! There's definitely some yearly seasonality as well, although it is noisier than the weekly seasonality, and more visible for the rail series than the bus series: we see peaks and troughs at roughly the same dates each year.
- Let's check what we get if we plot the 12-month difference.
- Notice how differencing not only removed the yearly seasonality, but it also removed the long-term trends.
- For example, the linear downward trend present in the time series form 2016 to 2019 became a roughly constant negative value in the differenced time series.
- In fact, differencing is a common technique used to remove trend and seasonality form a time series: it's easier to study a *stationary* time series, meaning one whose statistical properties remain constant over time, without any seasonality or trends.
- Once you're able to make accurate forecasts on the differenced time series, it's easy to turn them into forecasts for the actual time series by just adding back the past values that were previously subtracted.
- You may be thinking that we're only trying to predict tomorrow's ridership, so the long-term patterns matter much less than the short-term ones.
- That's right, but still, we may be able to perform slightly better by taking long-term patterns into account.
- For example, daily bus ridership dropped by about 2,500 in October 2017, which represents about 570 fewer passengers each week, so if we were at the end of October 2017, it would make sense to forecast tomorrow's ridership by copying the value from last week, minus 570.
- Accounting for the trend will make your forecasts a bit more accurate on average.