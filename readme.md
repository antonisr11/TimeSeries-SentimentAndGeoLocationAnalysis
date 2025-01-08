## Problem description

In this project, we were asked to apply methods of exploratory analytical data analysis to a dataset that was collected from Twitter.

For the development of the system, we relied on two datasets, which contain data related to information about the tweets, their location of origin, the time, and more information about each individual user.

After we collected the data, we applied preprocessing and processing methods, and we applied time series analysis methods.

## Datasets

The datasets we used are:

1. [Tweet data from Cheng-Caverlee-Lee](https://archive.org/details/twitter_cikm_2010)
2. [USA states GeoLocation from Kate Gallo](https://www.kaggle.com/datasets/pompelmo/usa-states-geojson)
3. [United States Cities from Sergej Nuss](https://www.kaggle.com/datasets/sergejnuss/united-states-cities-database)

Due to large output files, the data folder isn't uploaded here.

## Preprocess

Developing this application, we preprocessed all the data, and we exported 2 pkl files using the pickle library so that the data can be loaded quickly without having to preprocess the same data again (the function that does this is already implemented and is called exportPickleData(), it's just not used in main).

By using pkl files, we can achieve both faster data loading and lower RAM requirements.

Although the data has gone through a preprocess before becoming pkl files, in some parts of the code, additional pre-processing of the data is required. For instance, in the contents of the tweets before the construction of the geolocation map that we will analyze below.

## Data structure

At this point, we will explain the contents of the data folder. Inside the data folder, we will find three sub-folders:

1. staticData: contains the pkl files that our program will load so that it does not load and preprocess the data from scratch.
2. rawData: contains the data we got from various datasets. In this folder, we will find tweetData with data from Twitter and USData with information about the United States of America.
3. generatedData: contains the data generated by the program. When the program starts, it deletes all the files inside this folder, and any plot or file produced by the program goes into this folder.

## Tweet count

![1](https://github.com/user-attachments/assets/630cfd0d-9daa-4527-a5a9-79f6c1941ae4)

We notice that since the beginning of the graph and for a very long period of time, there have not been many tweets, so we will assume that those are outsiders, and we will deal only with the data that is from 01/03/2009 and onwards.

![2](https://github.com/user-attachments/assets/bd32704c-7f5a-454a-b037-c5e0e1dc948e)

> Note: We notice that the tweets are unevenly distributed in the graph as there are large ups and downs without following any pattern. As a result, we must mention that the time series analysis will not have optimal results as the data has unpredictable fluctuations.

## Autocorrelation

We create the autocorrelation function to see if the data are stationary.

![3](https://github.com/user-attachments/assets/e5e318aa-3590-4859-98c1-85e8c289ec3f)

We notice that the first nodes are not in the blue area, so we know that our data have a trend and/or seasonality (meaning that the data are not stationary).

> Note: At this point, we must mention that we expect increased MSE (Mean Square Error) from a model based on time series analysis on non-stationary data.

## Seasonal Decomposition

![4](https://github.com/user-attachments/assets/15ca5ad8-cb7b-42f1-a61e-ee83c931fdfa)

## Exponential Smoothing Models

Continuing our analysis, we want to compare and find the exponential model with the smallest MSE. Exponential models can optionally take two variables: the smoothing level and the smoothing slope.

- Practically the smoothing level variable controls how much weight is given to recent observations compared to previous observations when predicting future values. It's a way to measure the variation in the data and create a smoother trend.
  
  A lower level of smoothing gives greater or equal weight to both recent and older observations, and this makes the forecast less sensitive to short-term fluctuations while also providing a more stable, long-term trend.

- As for the smoothing slope variable, the value determines how much weight is given to changes over time. It helps to record whether the trend is up or down and at what rate. A high value means that the trend will be more sensitive to changes in the growth rate. This allows the forecast to respond more quickly to sudden changes in the trend.

### Experiments

To find the values that will have the smallest error, we do a variety of experiments. We try values ranging from 0.1 to 0.9 for both the smoothing level and the gradient smoothing.

During these experiments, we created four models:

1. BEST: This model is a Triple Exponential Smoothing model with additive trend and multiple seasonality. We have named it BEST because, as we will see below, it is the best compared to the other models.
2. SES: This model is Simple Exponential Smoothing.
3. DES: This model is Double Exponential Smoothing with an additive trend.
4. TES: This model is Triple Exponential Smoothing with additive trend and additive seasonality.

After the models are trained on 80% of the data and output the predictions, we calculate the MSE of each of the four models based on the remaining 20%.

![5](https://github.com/user-attachments/assets/b74dfe38-e524-4422-9f08-3daa4b172ecb)

### Forecast

Then we keep the values of the smoothing level and slope that minimize the MSE of the BEST model (specifically, 0.5 and 0.2). Based on these values, we show in a diagram the predictions of each model.

![6](https://github.com/user-attachments/assets/da412f52-c10a-40f4-b709-a73f2ee5952c)

We can see that BEST is indeed closer to the actual data line compared to the others. Also, the shape of the line in each model reveals the differences between those models. For example, DES seems to be better than SES because its forecat line takes trend into account. Furthermore, TES also takes seasonality into account (we can see that the brown line is not just a straight line); therefore, the use of TES leads to a smaller MSE.

### Compare MSEs

![7](https://github.com/user-attachments/assets/b20333ed-32b0-4b3c-89b5-c2e950e17bf4)

In this graph, we see that BEST has the smallest error compared to the rest. But as mentioned above, the mean squared error is very large (we see that the scale is in $10^8$), which indicates that time series analysis for this data should not be the preferred method.

## Grouping data and repeating experiments

We repeat this process (i.e., the plot of the data, the plot of the autocorrelation, the plot of the trend, seasonality and residuals (seasonal decomposition), finding the optimal smoothing level and slope, the plot of the forecast, and finally the plot of the four squares errors) a total of four times:

1. For all data from 01/03/2009 onwards
2. For all data from 01/03/2009 onwards, grouped by 2 days (average binning). In practice, the data changes as we group them by two days and replace them with the resulting average. This is a de-noising process.
3. For all data from 01/03/2009 and then grouped by 7 days (we follow the same average method as above).
4. For all data from 03/01/2009 onwards, grouped by 14 days (we follow the same average method as above).

## Compare BEST's MSEs

By doing this process, we found that grouping by 14 days has the lowest MSE, so it is the best (in comparison with the other 3 groups).

## Sentiment Analysis

Regarding the implementation of the sentiment analysis, we initially removed all mentions from the messages as the names of the users might have influenced the final sentiment of the tweet. Then we chose a transformer-based model, Roberta, which was fine-tuned on Twitter data specifically for sentiment analysis. We emphasized the speed of the model because the number of tweets we had was tremendous.

![8](https://github.com/user-attachments/assets/e19fa706-5b50-4f68-ad2c-30d1a86243d1)

## Geolocation Analysis

Finally, we did a geographic analysis, and we used a library to create a map. We chose a library that will nicely display the states to the user (it generates and opens an HTML file so for the user to be able to interact with the map) and the number of tweets posted by each state (any tweet not from a state of America is ignored).

![9](https://github.com/user-attachments/assets/d21162da-3248-47d0-a2ec-47386bb64a30)
