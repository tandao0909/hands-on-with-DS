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

## The ARMA Model Family

- We'll start with the *autoregressive moving average* (ARMA) model, developed by Herman Wold in the 1930s: it computes its forecasts using a simple weighted sum of lagged values and and corrects these forecasts by adding a moving average, very much like we just discussed.
- Specifically, the moving average component is computed using a weighted sum of the last few forecast errors.
- The following equation show show the model makes its forecast:
    $$\hat{y}_{(t)} = \sum_{i=1}^p \alpha_i y_{(t-1)} + \sum_{i=1}^q \theta_i \epsilon_{(t-i)} \text{ with } \epsilon_{(t)} = y_{(t)} - \hat{y}_{(t)} $$
- In this equation:
    - $\hat{y}_{(t)} $ is the model's forecast for time step $t$.
    - $y_{(t)} $ is the time series' value at time step $t$.
    - The first sum is the weighted sum of the past $p$ values of the time series, using the learned weights $\alpha_i$. The number $p$ is a hyperparameter, determines how far back into the past the model should look. This sum is the *autoregressive* component of the model: it performs regression based on past values.
    - The second sum is the weighted sum over the past $q$ forecast errors $\epsilon_{(t)} $, using the learned weights $\theta_i$. The number $q$ is a hyperparameter. This sum is the moving average component of the model.
- Importantly, this model assumes that the time series is stationary. If its is not, then differencing may help.
- Using differencing over a single time step will produce an approximation of the derivate of the time series: indeed, it will give the slope of the series at each times step.
- This means that it will eliminate any linear trend, transforming it into a constant value. For example, if you apply one-step differencing to the series [3, 5, 7, 9, 11] , you get the differenced series [2, 2, 2, 2].
- If the original series has a quadratic trend instead of linear trend, then a single round of differencing will not be enough.
- For example, the series [1, 4, 9, 16, 25, 36] becomes [3, 5, 7, 9, 11] after one round of differencing, but if you run differencing for a second round, then you get [2, 2, 2, 2].
- So, running two rounds of differencing will eliminate quadratic trends.
- More generally, running $d$ consecutive rounds of differencing computes an approximation of the d-th order derivate of the time series, so it will eliminate polynomial trends up to degree d. This hyperparameter d is called the *order of integration*.
- Differencing is the central contribution of the *autoregressive integrated moving average* (ARIMA) model, introduced in 1970 by George Box and Gwilym Jenkins in their book *Time Series Analysis* (Wiley).
- This model runs d rounds of differencing to make the time series more stationary, then it applies a regular ARMA model.
- When making forecasts, it uses this ARMA model then it adds back the terms that were subtracted by differencing.
- One last member of the ARMA family is the *seasonal ARIMA* (SARIMA) model: it models the time series the same way as ARIMA, but it additionally models a seasonal component for a given frequency (e.g., weekly), using the exact same ARIMA approach.
- It has a total of seven hyperparameters: the same p, d, and q hyperparameters as ARIMA, plus additional P, D, and Q hyperparameters to model the seasonal pattern, and lastly the period of the seasonal pattern, noted s.
- The hyperparameters P, D, and Q are just like p, d, and q, but they are used to model the time series at t - s, t - 2s, t - 3s, etc.
- Now, let's see how to fit a SARIMA model to the rail time series, and use it to make a forecast for tomorrow's ridership.
- We'll pretend today is the last day of May 2019, and we want to forecast the rail ridership for "tomorrow", the 1st of June, 2019.
- For this, we can use the `statsmodels` library, which contains many different statistical models, including the ARMA model and its variants, implemented by the `ARIMA` class.
- You can find the implementation in the learning notebook:
    - We start by importing the `ARIMA` class, then we take the rail ridership data from the start of 2019 up to "today", and use `asfreq("D")` to set the time series' frequency to daily: this doesn't change the data at all in our case, as it's already daily, but without this the `ARIMA` class would have to guess the frequency, and it would display a warning.
    - Next, we create an `ARIMA` instance, passing it all the data until today, and we set the model hyperparameters: `order=(1, 0, 0)` means that $p=1,d=0,q=0$, and `seasonal_order=(0, 1, 1, 7)` means that $P=0, D=1, Q=1$, and $s=7$. Notice that the the `statsmodels` API differs a bit from Scikit-Learn's API, since we pass the data to the model at construction time, instead of passing it to the `fit()` method.
    - Next, we fit the model, and use it to make a forecast for "tomorrow", the 1st of June, 2019.
- The forecast is 427,758=9 passengers, when in fact there were 379,044. Well, we're 12.85% off - that's pretty bad.
- It's actually slightly worse than naive forecasting, which forecasts 426,932, off by 12.63%. But perhaps we just unlucky that day?
- To check this, we can run the same code in a loop to make forecasts for every day in March, April, and May, and compute the MAE over that period. You can find the code in the learning notebook.
- And yes, that's much better! We achieve the MAE of about 32,041, which is significantly lower than the MAE we got with naive forecasting (42,143). So although the model is not perfect, it still beats naive forecasting by a large margin, on average.
- There are several methods to pick a good hyperparameters for the SARIMA model, but the simplest to understand and to get started with is the brute-force approach: just run a grid search.
- For each model you want evaluate, (i.e., each hyperparameter combination), you can run the loop example, changing only the hyperparameter values.
- Good $p, q, P,$ and $Q$ values are usually fairly small (typically 0 to 2, sometimes up to 5 or 6), and $d$ and $D$ are typically 0 or 1, sometimes 2. As for $s$, it's just the main seasonal pattern's period: in our case it's 7 as there's a strong weekly seasonality. The model with the lowest MAE wins.
- Of course, you can replace the MAE with another metric if it better matches your business objective.
- There are other more principled approaches to selecting good hyperparameters, based on analyzing the *autocorrelation function* (ACF) and *partial autocorrelation function* (PACF) or minimizing the AIC or BIC metrics (introduced in chapter 9) to penalize models that use too many parameters and reduce the risk of overfitting the data, but gird search is a good place to start.
- For more detail on the ACF-PACF approach, check this [post]() by Jason Brownlee.

## Preparing the Data for Machine Learning Models

- Now that we have two baselines, naive forecasting and SARIMA, let's try to use the machine learning models we've covered so far to forecast this time series, starting with a basic linear model.
- Our goal will be to forecast tomorrow's ridership based on the ridership of the past 8 weeks of data (56 days).
- The inputs to our model will therefore be sequences (usually a single sequence per day once the model is in production), each containing 56 values from time step $t-55$ to $t$.
- For each input sequence, the model will output a single value: the forecast for time step $t+1$.
- We'll use every 56-day window from the past as training data, and the target for each window will be the value immediately following it.
- Keras actually has a nice utility function called `tf.keras.utils.timeseries_dataset_from_array()` to help us prepare the training set.
- It takes a time series as input and it builds a `tf.data.Dataset` (introduced in chapter 13) containing all the windows of the desired length, as well as their corresponding targets.
- You can see an example that takes in a time series containing the numbers 0 to 5 and creates a dataset containing all the windows of length 3, with their corresponding targets, grouped into batches of size 2 in the learning notebook.
- Each sample in the dataset is a window of length 3, along with its corresponding target (i.e., the value immediately after the window). The windows are [0, 1, 2], [1, 2, 3] and [2, 3, 4], and their respective targets are 3, 4, and 5.
- Since there are three windows in total, which is not a multiple of the batch size, the last batch only contains one window instead of two.
- Another way to get the same result is to use the `window()` method of `tf.dat.Dataset` class.
- It's more complex, but it gives full control, which will come in handy later in this chapter, so let's see how it works. The `window()` method returns a dataset of window datasets.
- In this example, the dataset contains six windows, each shifted by one step compared to the previous one, and the last three windows are smaller because they've reached the end of the series. In general, you'll want to get rid of these smaller windows by passing `drop_remainder=True` to the `window()` method.
- The `window()` method returns a *nested dataset*, analogues to a list of lists. This is useful when you want to transform each window by calling its dataset methods (e.g., to shuffle them or batch them).
- However, we cannot use a nested dataset directly for training directly for training, as out model expect tensors as inputs, not datasets.
- Therefore, we must call the `flat_map()` method: it converts a nested dataset into a *flat dataset* (one that contains tensors, not datasets).
- For example, suppose {1, 2, 3} represents a dataset containing the sequence of tensors 1, 2, and 3. If you flatten the nested dataset {{1, 2}, {3, 4, 5, 6}}, you get back the flat dataset {1, 2, 3, 4, 5, 6}.
- Moreover, the `flat_map()` method takes a function as an argument, which allows you to transform the nested dataset {{1, 2}, {3, 4, 5, 6}} into the flat dataset {[1, 2], [3, 4], [5, 6]}: it's dataset containing 3 tensors, each of size 2.
- Since each window dataset contains exactly four items, calling `batch(4)` on a window produce a single tensor of size 4. 
- We also create a helper function function to make it easier ot extract windows from a dataset.
- Then we split each window into inputs and targets, using the `map()` method. We can also group the resulting windows into batches of size 2.
- As you can see, we now have the same output as got earlier with the `timeseries_dataset_from_array()` function (with a bit more effort, but it will be worthwhile soon).
- Now, before we start training, we need to split our data into a training period, a validation period, and a test period. We will focus on the rail ridership for now.
- We will also scale it down by a factor of one million, to ensure the values are near the 0-1 range; this plays nicely with the default weight initialization and learning rate.
- When dealing with time series, you generally want to split across time.
- However, in some cases you may be able to split along other dimensions, which will give you a longer time period to train on.
- For example, if you have data about financial health of 10,000 companies form 2001 to 2019, you might be able to split this data across the different companies.
- It's very likely that many of these companies will be strongly correlated, though (e.g., the whole economic sectors may go up or down jointly), and if you have correlated companies across the training set and the test set, your test set will not be as useful, as its measure of generalization error will be optimistically biased.
- Next, we use `tf.keras.utils.timeseries_dataset_from_array()` to create datasets for training and validation.
- Since gradient descent excepts the instances in the training set to be independent and identically distributed (IID), as we saw in chapter 4, we must set the argument `shuffle=True` to shuffle the training windows (but not their contents).

## Forecasting Using a Linear Model

- Let's try a basic linear model first. We will use the Huber loss, which usually works better than minimizing the MAE directly, as discussed in chapter 10. We'll also us early stopping.
- This model reaches a validation MAE of about 37,555. That's better than naive forecasting, but worse than the SARIMA model.

## Forecasting Using a Simple RNN

- Let's try the most basic RNN, containing a single recurrent layer with just one recurrent neuron.
- All recurrent layers in Keras expect 3D inputs of shape [*batch size, time steps, dimensionality*], where *dimensionality* is 1 for univariate time series and more for multivariate time series.
- Recall that the `input_shape` argument ignores the first dimension (i.e., the batch size), and since recurrent layers can take input sequences of any length, we can set eh second dimension to `None`, which means "any size".
- Lastly, since we're dealing with a univariate time series, we need the last dimension's size to be 1.
- This is why we specified the input shape `[None, 1]`: it means "univariate sequences of any length".
- Note that the datasets actually contain inputs of shape [*batch size, time steps*], so we're missing the last dimension, but Keras will automatically add it for us in this case.
- This model works exactly as we saw earlier: the initial state $h_{(\text{init})}$ is set to 0, and it is passed to a single recurrent neuron, along with the value of the first time step, $x_{(0)}$.
- The neuron computes a weighted sum of these values plus the bias term, and it applies the activation function to the result, using the hyperbolic tangent function by default. The result is the first output, $y_{(0)} $.
- In a simple RNN, this output is also the new state $h_0$. This new state is passed to the same recurrent neuron along with the next input value, $x_{(1)} $, and the process is repeated until the last time step.
- At the end, the layer just output the last value: in our case the sequences are 56 steps long, so the last value is $y_{55}$. All of this is performed simultaneously for every sequence in the batch, of which there 32 in this case.
- By default, recurrent layer in Keras only return the final output. To make them return one output per time step, you must set `return_sequence=True`, as you will see.
- So that's out first recurrent model! It's a sequence-to-vector model. Since there's a single output neuron, the output vector has a size of 1.
- Now if you compile, train, and evaluate this model just ike the previous model, you will find that it's no good at all: its validation MAE is grater than 100,000!
- That was to be expected, for two reasons:
    - The model only has a single recurrent neuron, so the only data it cna use to make a prediction is the input value at the current time step and the output value from the previous time step. That's not much to go on! In other words, the RNN's memory is extremely limited: it's just a single number, its previous input. And the whole model only has three parameters, two weights plus a bias term, as there's just one recurrent neuron with only two input values. That's no where close to enough for this time series. In contrast, our previous model could look at all 56 previous values at once, and it had a total of 57 parameters.
    - The time series contains values from 0 to about 1.4, but since the default activation function is tanh, the recurrent layer can only output values between -1 and 1. There's no way it can predict values between 1.0 and 1.4.
- Let's deal with both of these issues: we will create a model with a larger recurrent layer, containing 32 recurrent neurons, and we will add a dense output layer on top of it with a single output neuron and no activation function.
- The recurrent layer will be able to carry much more information from one time step to the next, and the dense output layer will project the final output from 32 dimensions down to 1, without any value range constraints.
- Now if you compile, fit, and evaluate this model just like the previous one, you will find that its validation MAE reaches 30,420. That's the bets model we've trained so far, and it even beats the SARIMA model.
- We've only normalized the time series, without removing trend and seasonality, and yet the model still performs well. This is connivent, as it makes it possible to quickly search for promising models without worrying too much about preprocessing.
- However, to get the best performance, you may want to try making the time series more stationary; for example, using differencing.

## Forecasting Using a Deep RNN

- It's quite common to stack multiple layers of cells, as shown below. This gives you a *deep RNN*.
![A deep RNN (left) unrolled through time (right)](image-5.png)
- Implementing a deep RNN with Keras is straightforward: just stack recurrent layers. You can look at the learning notebook for an example.
- We use three `SimpleRNN` layers (but we could use other types of recurrent layers, such as an `LSTM` layer or a `GRU` layer, which we will discuss shortly).
- The first two are sequence-to-sequence layers, and the last one is a sequence-to-vector layer.
- Finally, the `Dense` layer produces the model's forecast (you can think of it as a vector-to-vector layer).
- This model is similar to the figure above, except the outputs $\hat{\textbf{Y}}_{(0)}$ to $\hat{\textbf{Y}}_{(t-1)}$ are ignored, and there's a dense layer on top of $\hat{\textbf{Y}}_{(t)}$, which outputs teh actual forecast.
- Make sure to set `return_sequence=True` for all recurrent layers (except the last one, if you only care about the last output).
- If you forget to set this parameter for one recurrent layers, it will output a 2D array containing only the output of the last time step, instead of a 3D array containing outputs for all time steps. The next recurrent layer will complain that you are not feeding it sequences in the expected 3D format.
- If you train and evaluate this model, you will that it reaches an MAE of about 31,625. That's better tah both baselines, but it doesn't beat our "shallower" RNN. It looks like this RNN is a bit too large for our task.

## Forecasting Multivariate time Series

- A great quality of neural networks is their flexibility: in particular, they can deal with multivariate time series with almost no change to their architecture.
- For example, let's try to forecast the rail time series using both the bus and rail data as input. In fact, let's also throw in the day type!
- Since we know in advance whether tomorrow is going to be a weekday, a weekend, or a holiday, we can shift the day type series one day into the future, so that the model is given tomorrow's day type as input.
- Now `df_multivariate` is a DataFrame with five columns: the bus and rail data, plus three columns containing the one-hot encoding of the next day's type (recall that there are three possible day types `W`, `A`, and `U`).
- Next, we can proceed much like we did earlier:
    - First, we split the data into three periods, for training, validation, and testing.
    - Then, we create the dataset.
    - Finally, we create the RNN.
- Notice that the only difference from the `univariate_model` RNN we built earlier is the input shape: at each time step, the model now receives five inputs instead of one.
- This model reaches a validation MAE of 22,625! We're making great progress.
- In fact, it's not hard to make the RNN forecast both the bus and rail ridership. You just need to change the targets when creating the datasets, setting them to `multivariate_train["rail"][seq_length:]` for the training set,and `multivariate_valid["rail"][seq_length:]` for the validation set.
- You also need to add an extra neuron in the output `Dense` layer, since it must now make two forecasts: one for tomorrow's bus ridership, and the other for rail.
- As we discussed in chapter 10, using a single model for multiple related tasks often result in better performance than using a separate model for each task, since features learned for one task may be useful for the other tasks, and also prevents the model from overfitting (it's a form of regularization).
- However, it depends on the task, and in this particular case the multitask RNN that forecasts both the bus and the rail ridership doesn't perform as well as dedicated models that forecast one or the other (using all five columns as inputs). Still, it reaches a validation MAE of 23,947 for rail and 26,5, which is pretty good.