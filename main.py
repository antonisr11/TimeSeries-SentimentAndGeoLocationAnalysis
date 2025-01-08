import os
import pickle
import time
import webbrowser
import pandas as pd
import re
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from transformers import pipeline
import tensorflow
import torch
import geopandas as gpd
from shapely.geometry import Point
import folium
import warnings
warnings.filterwarnings("ignore")

def exportPickleData():
    def loadTweetData(fileName):
        regex = re.compile(r'^(\d+)\t(\d+)\s(.+)\t(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})$', re.MULTILINE)

        with open(fileName, encoding='UTF-8') as f:
            fileContent = f.read()
        matches = regex.findall(fileContent)

        data = [{'userId': match[0],
                 'messageId': match[1],
                 'message': match[2],
                 'dateTime': match[3] + ' ' + match[4]} for match in matches]

        return pd.DataFrame(data)

    def loadTrainingUsersData(fileName):
        regex = re.compile(r'^(\d+)\t(.+)$', re.MULTILINE)

        with open(fileName, encoding='UTF-8') as f:
            fileContent = f.read()
        matches = regex.findall(fileContent)

        data = [{'userId': match[0],
                 'city': match[1],
                 'latitude': None,
                 'longitude': None
                 }
                for match in matches]

        return pd.DataFrame(data)

    def loadTestUsersData(fileName):
        regex = re.compile(r'^(\d+)\tUT: (-?\d+\.\d+),(-?\d+\.\d+)$', re.MULTILINE)

        with open(fileName, encoding='cp437') as f:
            fileContent = f.read()
        matches = regex.findall(fileContent)

        data = [{'userId': match[0],
                 'city': None,
                 'latitude': match[1],
                 'longitude': match[2]
                 }
                for match in matches]

        return pd.DataFrame(data)

    dataFromTest = loadTweetData("data/rawData/TweetData/test_set_tweets.txt")
    dataFromTrain = loadTweetData("data/rawData/TweetData/training_set_tweets.txt")
    tweetDataLocal = pd.concat([dataFromTest, dataFromTrain], ignore_index=True)

    usersFromTest = loadTestUsersData("data/rawData/TweetData/test_set_users.txt")
    usersFromTrain = loadTrainingUsersData("data/rawData/TweetData/training_set_users.txt")
    usersDataLocal = pd.concat([usersFromTest, usersFromTrain], ignore_index=True)

    # convert column dateTime (format yyyy-MM-dd HH:MM:SS) to datetime
    tweetDataLocal['dateTime'] = pd.to_datetime(tweetDataLocal['dateTime'], format='%Y-%m-%d %H:%M:%S')

    with open('data/staticData/usersData.pkl', 'wb') as usersDataFile:
        pickle.dump(usersDataLocal, usersDataFile)

    with open('data/staticData/tweetData.pkl', 'wb') as tweetDataFile:
        pickle.dump(tweetDataLocal, tweetDataFile)

    return usersDataLocal, tweetDataLocal


def pickleLoadData():
    print("Loading data from pickle files...")
    usersDataLocal = pd.read_pickle("data/staticData/usersData.pkl")
    tweetDataLocal = pd.read_pickle("data/staticData/tweetData.pkl")
    return usersDataLocal, tweetDataLocal


def plotRaw(df, title):
    print("Plotting raw " + title)

    plt.plot(df['date'], df['count'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'data/generatedData/raw - {title}.png')
    plt.show()
    return


def plotSeasonalDecomposed(df, title):
    print("Plotting " + title + " with their trend, seasonality and residuals")

    decomposition = seasonal_decompose(df['count'], model='multiplicative')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.title('Seasonal Decomposition - ' + title)
    plt.subplot(411)
    plt.plot(df['count'], label='All data')
    plt.legend(loc='upper left')

    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')

    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='upper left')

    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='upper left')

    plt.ylim(-50, 50)
    plt.tight_layout()
    plt.savefig(f'data/generatedData/seasonal decomposed - {title}.png')
    plt.show()
    return


def calculateSmoothing(df, smoothing_level: float = None, smoothing_slope: float = None, train_ratio: float = 0.8):
    train_size = int(len(df) * train_ratio)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    forecast_length = len(test_data)

    # Fit models
    best_model = ExponentialSmoothing(train_data['count'], trend='add', seasonal='mul').fit(smoothing_level=smoothing_level,
                                                                                       smoothing_slope=smoothing_slope)
    ses_model = SimpleExpSmoothing(train_data['count']).fit(smoothing_level=smoothing_level)
    des_model = ExponentialSmoothing(train_data['count'], trend='add').fit(smoothing_level=smoothing_level,
                                                                           smoothing_slope=smoothing_slope)
    tes_model = ExponentialSmoothing(train_data['count'], trend='add', seasonal='add').fit(
        smoothing_level=smoothing_level, smoothing_slope=smoothing_slope)

    # Make forecasts
    best_predictions = best_model.forecast(forecast_length)
    ses_predictions = ses_model.forecast(forecast_length)
    des_predictions = des_model.forecast(forecast_length)
    tes_predictions = tes_model.forecast(forecast_length)

    # Calculate mean squared error for each model
    best_mse = mean_squared_error(test_data['count'], best_predictions)
    ses_mse = mean_squared_error(test_data['count'], ses_predictions)
    des_mse = mean_squared_error(test_data['count'], des_predictions)
    tes_mse = mean_squared_error(test_data['count'], tes_predictions)

    return train_data, test_data, best_predictions, ses_predictions, des_predictions, tes_predictions, best_mse, ses_mse, des_mse, tes_mse


def plotSmoothingAndMSEComparison(df, title, smoothing_level: float = None, smoothing_slope: float = None, train_ratio: float = 0.8):
    print("Plotting smoothing and MSE comparison for " + title)

    train_data, test_data, best_predictions, ses_predictions, des_predictions, tes_predictions, best_mse, ses_mse, des_mse, tes_mse = calculateSmoothing(
        df, smoothing_level, smoothing_slope, train_ratio)

    # Plot forecasts and actual values
    plt.plot(train_data['count'], label='Training')
    plt.plot(test_data['count'], label='Actual')
    plt.plot(best_predictions, label='Forecast (BEST)')
    plt.plot(ses_predictions, label='Forecast (SES)')
    plt.plot(des_predictions, label='Forecast (DES)')
    plt.plot(tes_predictions, label='Forecast (TES)')

    plt.legend(loc='upper left')
    plt.title('Forecast for '+title)
    plt.tight_layout()
    plt.savefig(f'data/generatedData/forecast - {title}.png')
    plt.show()


    # Plot MSEs of models
    plt.bar(["BEST", "SES", "DES", "TES"], [round(best_mse, 4), round(ses_mse, 4), round(des_mse, 4), round(tes_mse, 4)])
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE of each model in ' + title)
    plt.tight_layout()
    plt.savefig(f'data/generatedData/forecast mses - {title}.png')
    plt.show()
    return


def plotMSEComparisonForSeveralSmoothingArguments(df, title, smoothingLevelList: list, smoothingSlopeList: list, train_ratio: float = 0.8, countOfMSEs:int = 5):
    print("Calculating several MSEs with several smoothing arguments for " + title)

    if smoothingLevelList is None or len(smoothingLevelList) == 0:
        raise ValueError("smoothing_level cannot be None or empty")
    if smoothingSlopeList is None or len(smoothingSlopeList) == 0:
        raise ValueError("smoothing_slope cannot be None or empty")

    # Create a dataframe to store the MSEs of each model
    mseDf = pd.DataFrame(columns=['levelAndScope', 'best_mse', 'ses_mse', 'des_mse', 'tes_mse'])

    total_iterations = len(smoothingLevelList) * len(smoothingSlopeList)
    progress_bar = tqdm(total=total_iterations, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    for i in range(len(smoothingLevelList)):
        for j in range(len(smoothingSlopeList)):
            train_data, test_data, predictions, ses_predictions, des_predictions, tes_predictions, mse, ses_mse, des_mse, tes_mse = calculateSmoothing(df, smoothingLevelList[i], smoothingSlopeList[j], train_ratio)
            mseDf = pd.concat([mseDf, pd.DataFrame([[f"{smoothingLevelList[i]}, {smoothingSlopeList[j]}", mse, ses_mse, des_mse, tes_mse]], columns=['levelAndScope', 'best_mse', 'ses_mse', 'des_mse', 'tes_mse'])])
            progress_bar.update(1)
    progress_bar.close()

    time.sleep(1)
    print("Calculating MSEs is done. Plotting the results for " + title)

    # Sort the dataframe by best_mse
    mseDf = mseDf.sort_values(by=['best_mse'])

    # Make the plot
    plt.bar(mseDf['levelAndScope'].head(countOfMSEs), mseDf['best_mse'].head(countOfMSEs), label='Best MSE')
    plt.bar(mseDf['levelAndScope'].head(countOfMSEs), mseDf['ses_mse'].head(countOfMSEs), label='Simple MSE',
            bottom=mseDf['best_mse'].head(countOfMSEs))
    plt.bar(mseDf['levelAndScope'].head(countOfMSEs), mseDf['des_mse'].head(countOfMSEs), label='Double MSE',
            bottom=mseDf['best_mse'].head(countOfMSEs) + mseDf['ses_mse'].head(countOfMSEs))
    plt.bar(mseDf['levelAndScope'].head(countOfMSEs), mseDf['tes_mse'].head(countOfMSEs), label='Triple MSE',
            bottom=mseDf['best_mse'].head(countOfMSEs) + mseDf['ses_mse'].head(countOfMSEs) + mseDf['des_mse'].head(
                countOfMSEs))

    plt.xlabel('Level, Scope')
    plt.ylabel('MSE')
    plt.title('MSE by Level and Scope for 4 models in ' + title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'data/generatedData/mses comparison - {title}.png')
    plt.show()

    minLevel, minScope = mseDf['levelAndScope'].head(1).values[0].split(',')
    return float(minLevel), float(minScope)


def plotAutoCorrelation(df, title):
    print("Plotting autocorrelation for " + title)

    plot_acf(pd.Series(df['count']))
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function in ' + title)
    plt.tight_layout()
    plt.savefig(f'data/generatedData/autocorrelation - {title}.png')
    plt.show()
    return


def addSentimentAnalysis(tweetsDf):
    print("Performing sentiment analysis to tweets")

    # remove @user from tweets in column 'message'
    tweetsDf['message'] = tweetsDf['message'].str.replace('@[^\s]+', '', regex=True)

    sentiment_pipeline = pipeline(model="cardiffnlp/roberta-base-tweet-sentiment-en")
    for index, row in tqdm(tweetsDf.iterrows(), total=tweetsDf.shape[0], bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        tweetsDf.at[index, 'sentiment'] = sentiment_pipeline(row['message'])[0]['label']

    return tweetsDf


def plotAll(df: pd.DataFrame, title: str, countOfDecimals: int = 10):
    plotRaw(df, title)
    time.sleep(1)
    plotAutoCorrelation(df, title)
    time.sleep(1)
    plotSeasonalDecomposed(df, title)
    time.sleep(1)
    decimalRange = [i / countOfDecimals for i in range(1, countOfDecimals)]
    bestLevel, bestSlope = plotMSEComparisonForSeveralSmoothingArguments(df, title, smoothingLevelList=decimalRange, smoothingSlopeList=decimalRange)
    time.sleep(1)
    plotSmoothingAndMSEComparison(df, title, smoothing_level=bestLevel, smoothing_slope=bestSlope)
    return

def plotSentimentAnalysis(tweetsDf:pd.DataFrame, title:str):
    print("Plotting sentiment analysis for " + title)

    plt.pie(tweetsDf['sentiment'].value_counts(), labels=tweetsDf['sentiment'].value_counts().index, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title("Sentiment Analysis in " + title)
    plt.tight_layout()
    plt.savefig(f'data/generatedData/sentiment analysis - {title}.png')
    plt.show()
    return

def plotGeolocationMap(usersDf:pd.DataFrame):
    def findStateId():
        nonlocal usersDf

        print("\tFinding stateId for each user")

        # append column 'state' in usersDf
        usersDf['stateId'] = None

        gdf_states = gpd.read_file('data/rawData/USData/us-states.json')
        usCities = pd.read_csv('data/rawData/USData/uscities.csv')
        for index, row in tqdm(usersDf.iterrows(), total=usersDf.shape[0],
                               bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

            stateId = None
            if str(row['city']) == 'nan':
                # We have longitude and latitude but not state id
                point_geom = Point(row['longitude'], row['latitude'])

                # Perform a spatial join to find the intersecting state
                intersecting_states = gdf_states[gdf_states.geometry.contains(point_geom)]

                if not intersecting_states.empty:
                    stateId = intersecting_states['id'].iloc[0]
            else:
                # State is available but not with state id
                for city in str(row['city']).split(','):
                    city.strip()

                    if city in usCities['city'].values:
                        try:
                            stateId = usCities.loc[usCities['city'] == city, 'state_id'].item()
                        except ValueError:
                            # In this case we have several cities with the same name in different states, so we will ignore this city
                            pass
                        break
            usersDf.at[index, 'stateId'] = stateId
        return usersDf

    print("Plotting geolocation map...")

    # Convert Latitude Longitude to states
    usersDf = findStateId()

    # drop users with none stateId
    usersDf = usersDf.dropna(subset=['stateId'])

    state_counts = usersDf['stateId'].value_counts().reset_index()
    state_counts.columns = ['stateId', 'count']

    m = folium.Map(location=[48, -102], zoom_start=3)

    folium.Choropleth(
        geo_data="data/rawData/USData/us-states.json",
        name="choropleth",
        data=state_counts,
        columns=state_counts.columns,
        key_on="feature.id",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Tweets Count",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save('data/generatedData/map.html')
    webbrowser.open('file://' + os.path.abspath('data/generatedData/map.html'))
    return


if __name__ == '__main__':
    # delete all files in data/generatedData
    for filename in os.listdir('data/generatedData'):
        os.remove(f'data/generatedData/{filename}')

    # Load data

    usersData, tweetData = pickleLoadData()

    # Count all tweets by day
    aggregatedByDayTweetData = pd.DataFrame(columns=['date', 'count'])
    aggregatedByDayTweetData['count'] = tweetData.groupby([pd.Grouper(key='dateTime', freq='D')])['messageId'].count()
    aggregatedByDayTweetData['date'] = pd.to_datetime(aggregatedByDayTweetData.index.date)
    aggregatedByDayTweetData.set_index('date', inplace=True, drop=False)

    time.sleep(1)
    print(" --- All tweets by day --- ")
    # Plot all data
    plotRaw(aggregatedByDayTweetData, 'all tweets')


    # Filter tweets and take those from 2009-03-01
    aggregatedByDayTweetDataFrom2009 = aggregatedByDayTweetData[
        aggregatedByDayTweetData['date'] >= pd.to_datetime('2009-03-01')]

    time.sleep(1)
    print(" --- All tweets from 2009-03-01 --- ")
    # Plot data from 2009-03-01
    plotAll(aggregatedByDayTweetDataFrom2009, 'tweets from 2009-03-01')


    time.sleep(1)
    print(" --- All tweets from 2009-03-01 aggregated by 2 days --- ")
    # Aggregate data by 2 days
    aggregatedByDayTweetDataFrom2009By2Days = aggregatedByDayTweetDataFrom2009.copy()
    aggregatedByDayTweetDataFrom2009By2Days['count'] = aggregatedByDayTweetDataFrom2009By2Days.groupby(aggregatedByDayTweetDataFrom2009By2Days['date'].dt.floor('2D'))['count'].transform('mean')
    # Plot data from 2009-03-01 aggregated by 2 days
    plotAll(aggregatedByDayTweetDataFrom2009By2Days, 'tweets aggregated by 2 days')


    time.sleep(1)
    print(" --- All tweets from 2009-03-01 aggregated by 7 days --- ")
    # Aggregate data by 7 days
    aggregatedByDayTweetDataFrom2009ByWeek = aggregatedByDayTweetDataFrom2009.copy()
    aggregatedByDayTweetDataFrom2009ByWeek['count'] = aggregatedByDayTweetDataFrom2009ByWeek.groupby(aggregatedByDayTweetDataFrom2009ByWeek['date'].dt.floor('7D'))['count'].transform('mean')
    # Plot data from 2009-03-01 aggregated by 7 days
    plotAll(aggregatedByDayTweetDataFrom2009ByWeek, 'tweets aggregated by week')


    time.sleep(1)
    print(" --- All tweets from 2009-03-01 aggregated by 14 days --- ")
    # Aggregate data by 14 days
    aggregatedByDayTweetDataFrom2009By14Days = aggregatedByDayTweetDataFrom2009.copy()
    aggregatedByDayTweetDataFrom2009By14Days['count'] = aggregatedByDayTweetDataFrom2009By14Days.groupby(aggregatedByDayTweetDataFrom2009By14Days['date'].dt.floor('14D'))['count'].transform('mean')
    # Plot data from 2009-03-01 aggregated by 14 days
    plotAll(aggregatedByDayTweetDataFrom2009By14Days, 'tweets aggregated by 2 weeks')


    time.sleep(1)
    print(" --- Calculate date with most tweets --- ")
    # Calculate date with most tweets
    dateWithMostTweets = str(aggregatedByDayTweetDataFrom2009[aggregatedByDayTweetDataFrom2009['count'] == aggregatedByDayTweetDataFrom2009['count'].max()]['date'].values[0])[0:10]
    print('Date with most tweets: ' + dateWithMostTweets)

    # Do sentiment analysis on tweets from date with most tweets
    tweetDataFromDateWithMostTweets = tweetData[tweetData['dateTime'].astype(str).str[:10] == dateWithMostTweets]
    tweetDataWithSentimentFromDateWithMostTweets = addSentimentAnalysis(tweetDataFromDateWithMostTweets)

    time.sleep(1)
    print(" --- Sentiment analysis of tweets from date with most tweets --- ")
    # Plot sentiment analysis
    plotSentimentAnalysis(tweetDataWithSentimentFromDateWithMostTweets, 'Sentiment analysis of tweets from date ' + dateWithMostTweets)


    time.sleep(1)
    print(" --- Constructing a map from users who tweet in the day with the most tweets --- ")
    # Plot geolocation map
    usersIDsWhoTweetedInDateWithMostTweets = tweetDataFromDateWithMostTweets['userId'].unique()
    usersWhoTweetedInDateWithMostTweets = usersData[usersData['userId'].isin(usersIDsWhoTweetedInDateWithMostTweets)]
    plotGeolocationMap(usersWhoTweetedInDateWithMostTweets)