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

## Forecasting Several Time Steps Ahead

- So far, we have only predicted the value at the next time step, but we could just as easily have predicted the value several steps ahead by changing the targets appropriately (e.g., ot predict the ridership 2 weeks from now, we could just change the targets to be the value 14 days ahead instead of 1 day ahead). But what if we want to predict the next 14 values?
- The first option is to take the `univariate_model` RNN we trained earlier for the rail time series, make it predict the next value, and add that value to the inputs, acting as if the predicted value has actually occurred; we would then use the model again to predict the following value, and so on.
- In the code (in the learning notebook), we take the rail ridership of the first 56 days of the validation period, and we convert the data to a NumPy array of shape [1, 56, 1] (recall that recurrent layers except 3D inputs).
- Then we repeatedly use the model ot forecast the next value, and we append each forecast to the input series, along the time axis (`axis=1`). You can find the resulting plot in the learning notebook.
- If the model makes an error at one time step, then the forecasts for the following time steps are impacted as well: the errors tend to accumulate.
- So it's preferable to sue this technique only for a small number of steps.
- The second option is to train an RNN to predict the next 14 values in one shot.
- We can still use a sequence-to-vector model, but it will output 14 values instead of 1.
- However, we first need to change the targets to be vectors containing the next 14 values.
- To do this, we can use `tf.keras.utils.timeseries_dataset_from_array()` again, but this time asking it to create datasets without targets (`targets=None`) and with longer sequences, of length `seq_length` + 14.
- Then we can use the dataset's `map()` method to apply a custom function to each batch of sequences, splitting them into inputs and targets.
- In this example, we use the multivariate time series as input (using all five columns), and we forecast the rail ridership for the next 14 days.
- And now, we just need the output layer to have 14 units instead of 1.
- After training this model, you can predict the next 14 values at once by reshaping the inputs, like an example in the learning notebook.
- This approach works quite well. Its forecast for the next day are obviously better than its forecasts for 14 days into the future, but it doesn't accumulate errors like the previous approach did.

## Forecasting Using a Sequence-to-Sequence Model

- Instead of training the model to forecast the next 14 values only at the very last time step, we can train it to forecast the next 14 values at each and every time step.
- In other words, we can turn this sequence-to-vector RNN into a sequence-to-sequence RNN.
- The advantage of this technique is that the loss will contain a term for the output of the RNN at each and every time step, not just the output at the last time step.
- This means there will be many more error gradients flowing through the model, and they won't have ot flow through time as much since they will come from the output of each time step, not just the last one. This will both stabilize and speed up training.
- To be clear, at time step 0, the model will output a vector containing the forecasts for time step 1 to 14, then at time step 1, the model will forecast time steps 2 to 15, and so on.
- In other words, the targets are sequences of consecutive windows, shifted by one time step at each time step.
- The target is not a vector anymore, but a sequence of the same length as the inputs, containing a 14-dimensional vector at each step.
- Preparing the dataset is not trivial, since each instance has a window as input and a sequence of window as output.
- One way to do this is to use the `to_windows()` utility we created earlier, twice in a row, to get the windows of consecutive windows.
- For example, you can find the code that turns the series of numbers 0 to 6 into a dataset containing sequences of 4 consecutive windows, each of length 3.
- And we can use the map() method to split these windows of windows into inputs and targets.
- Now, the dataset contains sequences of length 4 as inputs, and the targets are sequences containing the next two steps, for each time step.
- For example, the first input sequence is [0, 1, 2, 3], and its corresponding targets are [[1, 2], [2, 3], [3, 4], [4, 5]], which are the next two values ofr each time step.
- More specifically, you work from the innermost dimension: build a window for each time step first, which means create the input and output at each time step, then create the batch, in this case is the the whole time interval.
- It may be surprising that the targets contain values that appear in the inputs. But you don't have to worry about data snooping in this case: at each time step, an RNN only knows about past time steps; it cannot look ahead. It is said to be a *causal* model.
- Let's create another little utility function to prepare the datasets for our sequence-to-sequence model. It will also take care of shuffling (optional) and batching. We'll use this function to create the dataset.
- Lastly, we can build the sequence-to-sequence model.
- It is almost identical ot our previous model: the only difference is that we set `return_sequences=True` in the `SimpleRNN` layer. This way, it will output a sequence of vectors (each of size 32), instead of outputting a single vector at the last time step.
- The `Dense` layer is smart enough ot handle sequences as input: it will be applied at each time step, taking a 32-dimensional vector as input and outputting a 14-dimensional vector.
- In fact, another way to get the exact same result is to use a `Conv1D` layer with a kernel size of 1: `Conv1D(14, kernel_size=1)`.
- Keras offers a `TimeDistributed` layer that lets you apply any vector-to-vector layer to every vector in the input sequences, at every time steps. It does this efficiently, by reshaping the inputs so that each time step is treated as a separate instance, then it reshapes the layer's outputs to recover the time dimension.
- In our case, we don't need it since the `Dense` layer already supports sequences as inputs.
- The training code is the same as usual. During training, all the model's output are used, but after training only the output of the very last time step matters, and the rest can be ignored.
- If you evaluate this model's forecasts for $t + 1$, you will find a validation MAE of 25,839. For $t+2$ it's 29,614, and the performance continues to drop gradually as the model tries to forecast further into the future. At $t+14$, the MAE is 34,613.
- You can combine both approaches ot forecasting multiple time steps ahead: for example, you can train a model to get forecasts for the following 14 days, then take its output and append it to the inputs, then run the model again to get forecasts for the following 14 days, and possibly repeat the process.
- Simple RNNs can be quite good at forecasting time series or handling other kinds of sequences, but hey do not performs as well on long time series or sequences. The next part explains why and see what can we do about it.

# Handling Long Sequences

- To train an RNN on long sequences, we must run it over many time steps, making the unrolled RNN a very deep network.
- Juts like any deep neural network, it may suffer from the unstable gradients problem, discussed in chapter 11: it may take forever to train, or training may be unstable.
- Moreover, when an RNN processes a long sequence, it will gradually forget the first input in the sequence.
- We'll deal with both problems, staring with the unstable gradients problem.

## Fighting the Unstable Gradients Problem

- Many of the tricks we used in deep networks to alleviate the unstable gradients problem can also be used for RNNs: good parameter initialization, faster optimizers, dropout, and so on.
- However, nonsaturating activation functions (e.g., ReLU) may not help as much here. In fact, they may actually lead the RNN to be even more unstable during training.
- But why? Well, suppose gradient descent updates the weight in a way that increases the outputs ever so slightly at the first time step.
- Because the same weights are used at very time step, the outputs at the second time step may also be slightly increased, and those at the third, and so on until the outputs explode - and a nonsaturating activation function doesn't prevent that.
- You can reduce this risk by using a smaller learning rate, or you can use a saturating activation function like the hyperbolic tangent (this explains why it's the default).
- In much the same way, the gradients themselves can explode. If you notice that training is unstable, you may want to monitor the size of the gradients (e.g., using TensorBoard) and perhaps use gradient clipping.
- Moreover, batch normalization cannot be used as efficiently with RNNs as with deep feedforward nets. In fact, you cannot use it between time steps, only between recurrent layers.
- To be more precise, it is technically possible to add a BN layer to a memory cell (as you will see shortly) so that it will be applied at each time step (both on the inputs for that time step and on the hidden state form the previous step).
- However, the same BN layer will be used at each time step, with the same parameters, regardless of the actual scale and offset of the inputs and hidden state.
- In practice, this does not yield good results, as was demonstrated in a [2015 paper](https://arxiv.org/abs/1510.01378) by César Laurent et al.: The authors found that BN was slightly beneficial only when it was applied to the layer's inputs, not to the hidden states.
- In other words, it was slightly better than nothing when applied between recurrent layers (i.e., vertically in the figure represents the unrolled deep RNN network), but not within recurrent layers (i.e., horizontally).
- In Keras, you can apply BN between layers simply by adding a `BatchNormalization` layer before each recurrent layer, but it will slow down training, and it may not help much.
- Another form of moralization often works better with RNNs: *layer normalization*. This idea was introduced by Jimmy Lei Ba et al. in a [2016 paper](https://arxiv.org/abs/1607.06450).
- It is very similar to batch normalization, but instead of normalizing across the batch dimension, layer normalization normalizes across the feature dimension. A great visual explanations can be found in this [post](https://www.pinecone.io/learn/batch-layer-normalization/).
- One advantage is that it can compute the required statistics on the fly, at each time step, independently for each instance.
- This also means that it behaves the same way during training and testing (as opposed to BN), and it does not need to use exponential moving average to estimate the feature statistics across all instances in the training set, like BN does.
- Like BN, layer normalization learns a scale and an offset parameter for each input.
- In an RNN, it is typically used right after the linear combination of the inputs and the hidden states.
- Let's use Keras to implement layer normalization within a single memory cell. To do this, we need to define a custom memory cell, which is just like a regular layer, expect its `call()` method takes two arguments: the `inputs` at the current time step and the hidden `states` from the previous time step.
- Note that the `status` argument is a list containing one or more tensors. In the case of a simple RNN, it contains a single tensor equal to the outputs of the previous time step, but other cells may have multiple state tensors (e.g., an `LSTM` cell has a long-term state and a short-term state, as we'll see shortly).
- A cell must also have a `state_size` attribute and an `output_size` attribute. In a simple RNN, both are simply equal to the number of units.
- The code in the learning notebook implements a custom memory cell that will behave like a `SimpleRNNCell`, except it will also apply layer normalization at each time step. Let's walk through this code:
    - Our `LNSimpleRNNCell` class inherits from the `tf.keras.layers.Layer` class, just like any custom layer.
    - The constructor takes the number of units and the desired activation function and sets the `state_size` and `output_size` attributes, then creates a `SimpleRNNCell` with no activation function (because we want to perform layer normalization after the linear operation but before the activation function). Then the constructor creates the `LayerNormalization` layer, and finally it fetches the desires activation function.
    - The `call()` method starts by applying the `simpleRNNCell`, which computes a linear combination of the current inputs and the previous hidden states, and it returns the result twice (indeed, in a `simpleRNNCell`, the outputs are just equal to the hidden states: in other words, `new_states[0]` is equal to `outputs`, so we can safely ignore `new_states` in the rest of the `call()` method). Next, the `call()` method applies layer normalization, and passes to the activation function. Finally, it returns the outputs twice: once as the outputs, and once as the new hidden states. To use this custom cell, all we need to do is create a `tf.keras.layers.RNN` layer, passing it a cell instance.
- It would have been simpler to inherit from `SimpleRNNCell` instead so that we wouldn't have to create an internal `SimpleRNNCell` or handle the `output_size` and `state_size` attributes. The goal here was to show how to create a custom RNN cell from scratch.
- Similarly, you can create a custom cell to apply dropout between each time step. But there's a simpler way: most recurrent layers and cells provided by Keras have `dropout` and `recurrent_dropout` hyperparameters: the former defines the dropout rate to apply to the inputs, and the latter defines the dropout rate for the hidden layers, between each time step. So, you don't really need to create a custom cell to apply dropout at each time step in an RNN.
- When forecasting time series, it is often useful to have same error bars along with your predictions.
- For this, one approach is to use MC dropout, introduced in chapter 11: use `recurrent_dropout` during training, then keep dropout active at inference time by calling `model(X, training=True)`. Repeat this several times to get multiple slightly different forecasts, then compute the mean and standard deviation of these predictions for each time step.

## Tackling the Short-Term Memory Problem

- Due to the transformation that the data goes through when traversing an RNN, some information is lost at each time step.
- After a while, the RNN's state contains virtually no trace of the first inputs. In other words, it forget the first input.
- To tackle this problem, various type of cells with long-term memory have been introduced.
- They have proven so successful that the basic cells are not used much anymore. 
- Let's look at the most popular of these long-term memory cells: the LSTM cell.

# LSTM 

- The *long short-term memory* (LSTM) cell was proposed in [1997](https://ieeexplore.ieee.org/abstract/document/6795963) by Sepp Hochreiter and Jürgen Schmidhuber and gradually improved over the years by several researchers, such as [Alex Graves](https://www.cs.toronto.edu/~graves/), [Haşim Sak](https://arxiv.org/abs/1402.1128), and [Wojciech Zaremba](https://arxiv.org/abs/1409.2329).
- If you consider LSTM cell as a black box, it can be used very much like a basic cell, expect it will perform much better; training will converge faster, and it will detect long-term patterns in the data.
- In Keras, you can simply use the `LSTM` layer instead of the `SimpleRNN` layer.
- Alteratively, you could use the general-purpose `tf.keras.layers.RNN` layer, giving it an `LSTMCell` as an argument.
- However, the `LSTM` layer uses an optimized implementation when running on a GPU, so in general, it is preferable to use it (the `RNN` layer is mostly useful when you define custom cells, as we did earlier).
- So how does an LSTM cell work? Its architecture is shown below:
![An LSTM cell](image-6.png)
- If you don't look at what's inside the box, the LSTM cell looks exactly like a regular cell, expect that its state is split into two vectors: $\textbf{h}_{(t)}$ and $\textbf{c}_{(t)}$ ("c" stands for "cell").
- You can think of $\textbf{h}_{(t)} $ as the short-term state and $\textbf{c}_{(t)} $ as the long-term state.
- The key idea is that the network can learn what to store in the long-term state, what to throw away, and what to read from it.
- As the long-term state $\textbf{c}_{(t-1)}$ traverses the network from left to right, you can see: it first goes through a *forget gate*, dropping some memories, then it adds some new memories via the addition operation (which adds the memories that were selected by an *input gate*).
- The result $\textbf{c}_{(t)}$ is sent straight out, without any further transformation.
- So, at each time step, some memories are dropped and some memories are added.
- Moreover, after the addition operation, the long-term state is copied and passed through the tanh function, and then the result is filtered by the *output gate*.
- This produce the short-term state $\textbf{h}_{(t)}$, which is equal to the cell's output for this time step, $\textbf{y}_{(t)}$.
- Now, we will look at where the new memories come from and how the gates work.
- The current input vector $\textbf{x}_{(t)}$ and the previous short-term state $\textbf{h}_{(t-1)}$ are fed to four different fully connected layers. They all serves a different purposes:
    - The main layer is the one that outputs $\textbf{g}_{(t)}$. It has the usual role of analyzing the current inputs $\textbf{x}_{(t)}$ and the previous (short-term) state $\textbf{h}_{(t-1)}$. In a basic cell, there is nothing other than this layer, and its output goes straight out to $\textbf{y}_{(t)}$ and $\textbf{h}_{(t)}$. But in an LSTM cell, this layer's output does not go straight out; instead is most important parts are stored in the long-term state (while the rest is dropped).
    - The three other layers are *gate controllers*. Since they use the logistic activation function, the outputs range from 0 to 1. The gate controllers' outputs are fed to element-wise multiplication operations: if they outputs 0s they close the gate, and if they output 1s they open it.
    - The *forget gate* (controlled by $\textbf{f}_{(t)}$) controls which parts of the long-term state should be erased.
    - The *input gate* (controlled by $\textbf{i}_{(t)}$) controls which parts of $\textbf{g}_{(t)}$ should be added to the long-term state.
    - The *output gate* (controlled by $\textbf{o}_{(t)}$) controls which parts of the long-term state should be read and output at this time step, both to $\textbf{h}_{(t)}$ and to $\textbf{y}_{(t)}$.
- In short, an LSTM cell can learn to recognize an important input (that's the role of the input gate), store it in the long-term state, preserve it for as long as needed (that's the role of the forget gate), and extract it whenever it is needed.
- This explains why these cells have been amazingly successful at capturing long-term patterns in time series, long texts, audio recordings, and more.
- These following equations explain hwo to compute the cell's long-term state, its short-term state, and its output at each time step for a single instance. The equations for a whole mini-batch are very similar:
    $$\textbf{i}_{(t)} = \sigma(\textbf{W}_{xi}^T\textbf{x}_{(t)} + \textbf{W}_{hi}^T\textbf{h}_{(t-1)} + \textbf{b}_i) $$
    $$\textbf{f}_{(t)} = \sigma(\textbf{W}_{xf}^T\textbf{x}_{(t)} + \textbf{W}_{hf}^T\textbf{h}_{(t-1)} + \textbf{b}_f) $$
    $$\textbf{o}_{(t)} = \sigma(\textbf{W}_{xo}^T\textbf{x}_{(t)} + \textbf{W}_{ho}^T\textbf{h}_{(t-1)} + \textbf{b}_o) $$
    $$\textbf{g}_{(t)} = \tanh(\textbf{W}_{xg}^T\textbf{x}_{(t)} + \textbf{W}_{hg}^T\textbf{h}_{(t-1)} + \textbf{b}_g) $$
    $$\textbf{c}_{(t)} = \textbf{f}_{(t)} \otimes \textbf{c}_{(t-1)} + \textbf{i}_{(t)}\otimes\textbf{g}_{(t)}$$
    $$\textbf{y}_{(t)} = \textbf{h}_{(t)} = \textbf{o}_{(t)} \otimes \tanh(\textbf{c}_{(t)})$$
- In this equation:
    - $\textbf{W}_{xi}$, $\textbf{W}_{xf}$, $\textbf{W}_{xo}$, and $\textbf{W}_{xg}$ are the weight matrices of each of the four layers for their connection to the input vector $\textbf{x}_{(t)}$.
    - $\textbf{W}_{hi}$, $\textbf{W}_{hf}$, $\textbf{W}_{ho}$, and $\textbf{W}_{hg}$ are the weight matrices of each of the four layers for their connection to the previous short-term state vector $\textbf{h}_{(t-1)}$.
    - $\textbf{b}_i$, $\textbf{b}_f$, $\textbf{b}_o$, and $\textbf{b}_g$ are the bias terms for each of the four layers. Note that TensorFlow initializes $\textbf{b}_f$ to a vector full of 1s instead of 0s. This prevents forgetting everything at teh start of training.
- There are several variants of LSTM cell. One particular popular variant is the GRU cell.
# GRUs

- The *gated recurrent unit* (GRU) cell was proposed by Kyunghyun Cho et al. in a [2014](https://arxiv.org/abs/1406.1078) paper that also introduce the same encoder-decoder network we discussed earlier.
![GRU cell](image-7.png)
- The GRU cell is a simplified version of the LSTM cell, and it seems to perform just [as well](https://arxiv.org/abs/1503.04069) (which explains its growing popularity). These are the main simplifications:
    - Both state vectors are merged into a single vector $\textbf{h}_{(t)}$.
    - A single gate controller $\textbf{z}_{(t)}$ controls both the forget gate and the input gate. If the gate controller outputs a 1, the forget gate is open (= 1) and the input gate si closed (1 - 1 = 0). If it outputs a 0, the opposite happens.
    - In other words, whenever a memory must be stored, the location where it will be stored is erased first. This is actually a frequent variant to the LSTM cell in and of itself.
    - $\textbf{r}_{(t)}$ is similar to $\textbf{i}_{(t)}$ in the LSTM cell.
    - There is no output gate; the full state is output at every time step. However, there is a new gate controller $\textbf{r}_{(t)}$ that controls which part of the previous state will be shown to the main layer ($\textbf{g}_{(t)}$).
- The following equations summarizes how to compute the cell's state at each time step for a single instance:
    $$\textbf{z}_{(t)} = \sigma(\textbf{W}_{xz}^T\textbf{x}_{(t)} + \textbf{W}_{hz}^T\textbf{h}_{(t-1)} + \textbf{b}_z) $$
    $$\textbf{r}_{(t)} = \sigma(\textbf{W}_{xr}^T\textbf{x}_{(t)} + \textbf{W}_{hr}^T\textbf{h}_{(t-1)} + \textbf{b}_r) $$
    $$\textbf{g}_{(t)} = \tanh(\textbf{W}_{xg}^T\textbf{x}_{(t)} + \textbf{W}_{hg}^T(\textbf{r}_{(t)} \otimes \textbf{h}_{(t-1)}) + \textbf{b}_g) $$
    $$\textbf{h}_{(t)} = \textbf{z}_{(t)} \otimes \textbf{h}_{(t-1)} + (1 - \textbf{z}_{(t)}) \otimes \textbf{g}_{(t-1)}$$
- Keras provides a `tf.keras.layers.GRU` layer: using it is just a matter of replacing `SimpleRNN` or `LSTM` with `GRU`.
- It also provides a `tf.keras.layers.GRUCell`, in case you want to implement a custom cell based on the GRU cell.
- LSTM and GRU cells are one of the main reasons behind the success of RNNs.
- Yet while they can tackle much longer sequences than simple RNNs, they still have a fairly limited short-term memory, and they have trouble learning long-term patterns in sequences of 100 time steps or more, such as audio samples, long time series, or long sentences.
- One way to solve this is to shorten the input sequences; for example, using 1D convolutional layers.

## Using 1D Convolutional Layers to Process Sequences

- In chapter 14, we saw that a 2D convolutional layer works by sliding several fairly small kernels (or filters) across an image, producing multiple 2D feature maps (one per kernel).
- Similarly, a 1D convolutional layer slide several kernels across a sequence, producing a 1D feature map per kernel.
- Each kernel will learn to detect a single very short sequential pattern (no longer than the kernel size). If you use 10 kernels, then the layer's output will be composed of 10 1D sequences (all of the same length), or equivalently you can view this output as a single 10D sequences.
- This means that you can build a neural network composed of a mix of recurrent layers and 1D convolutional layers (or even 1D pooling layers).
- If you use a 1D convolutional layer with a stride of 1 and `"same"` padding, then the output sequence will have the same length as the input sequence.
- But if you use `"valid"` padding or a stride greater than 1, then the output sequence will be shorter than the input sequence, so make sure you adjust the targets accordingly.
- For example, the model in the learning notebook is the same as earlier, except it starts with a 1D convolutional layer that downsamples the input sequence by a factor of 2, using a stride of 2.
- The kernel size is larger than the stride, so all inputs will be used to compute the layer's output, and therefore the model can learn to preserve the useful information, dropping only the unimportant details.
- By shortening the sequences, the convolutional layer may help the `GRU` layers detect longer patterns, so we can afford to double the input sequence length to 112 days.
- Note that we must also crop off the first three time steps in the targets: indeed, the kernel's size is 4, so the first output of the convolutional layer will be based on the input time steps 0 to 3, and the first forecasts will be for time steps 4 to 17 (instead of time steps 1 to 14).
- Moreover, we must downsample the targets by a factor of 2, because of the stride.
- If you train and evaluate this model, you will find that it outperforms the previous model (by a small margin).
- In fact, we actually can use only 1D convolutional layers and drop the recurrent layers entirely.

## WaveNet

- In a [2016 paper](https://arxiv.org/abs/1609.03499), Aaron van den Oord and other DeepMind researchers introduced a novel architecture named WaveNet.
- They stacked 1D convolutional layers, doubling the dilation rate (how spread each neuron's inputs are) at very layer: the first convolutional layer gets a glimpse of just two time steps at a time, while the next one see four time steps (its receptive field is four time steps long), the next one sees eight time steps, and so on.
- This way, the lower layers learn short-term patterns, while the higher layers learn long-term patterns.
- Thanks to doubling dilation rate, the network can process extremely large sequences very efficiently.
- The authors of the paper actually stacked 10 convolutional layers with dilation rates of 1, 2, 4, ..., 256, 512, and then they stacked another group of 10 identical group of 10 layers.
- They justified this architecture by pointing out that a single stack of 10 convolutional layers with these dilation rates will act like a super efficient convolutional layer with a kernel of size 1,024, except way faster, more powerful, and using significantly fewer hyperparameters.
- They also left-padded the input sequences with a number of zeros equal to the dilation rate before every layer, to preserve the same sequence length throughout the network.
- You cna find an implementation of WaveNet in the learning notebook.
- This `Sequential` model starts with an explicit input layer - this is simpler than trying to set `input_shape` only to the first layer.
- The it continues with a 1D convolutional layer using `"causal"` padding, which is like `"same"` padding except that the zeros are appended only at the start of the input sequences instead of on both sides. This ensures that the convolutional layer does not peek into the future when making predictions.
- Then we add similar pairs of layers using growing dilation rates: 1, 2, 4, and 8, and again 1, 2, 4, and 8.
- Finally, we add the output layer: a convolutional layer with 14 filters of size 1 and without any activation function. As we saw earlier, such a convolutional layer is equivalent to a `Dense` layer with 14 units.
- Thanks to causal padding, every convolutional layer outputs sequence of the same length as its inputs sequences, so the targets we use during training can be the full 112-day sequences: no need to crop or downsample them.
- The models we've discussed in this section offer similar performance for the ridership forecasting task, but they may vary significantly depending on the task and the amount of available data.
- In the WaveNet paper the authors achieved state-of-the-art performance on various audio tasks (hence the name of the architecture), including text-to-speech tasks, producing incredibly realistic voices across several languages.
- They also used the model to generate music, one audio sample at a time. This feat is even more impressive when you realize that a single record of audio can contain ten of thousands of time steps - even LSTMs and GRUs can't handle such long sequences.
- If you evaluate our best Chicago ridership models on the test period, starting in 2020, you will find that they perform much worse than expected! This is because when the COVID-19 pandemic started, public transportation was greatly affected.
- As mentioned earlier, these will only work well if the patterns they learned from the past continue in the future.
- In any case, before deploying a model to production, verify that it works well on recent data.
- Once it's in production, make sure to monitor its performance regularly.