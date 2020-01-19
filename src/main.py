import pandas as pd
import statsmodels.api as sm
import numpy as np
from models import ELO

# This is the main function you are requested to define.
# required_predictions: a list of matches that you should predict
# data_loader: a NbaDataLoader class instance.
#     Contains getSeason, getGame and getPlayers methods. Please refer to simulate.py for details about
#     the provided methods.
# (return): You should return, for each match requested in required_predictions, two numbers:
#     The difference between the scores: homeScore - awayScore
#     And the sum of the scores: homeScore - awayScore
def predictOLD(required_predictions, data_loader):
    # Load games data for the 2011 season.
    # Seasons from 2009 onwards are available, including POST seasons, such as 2011POST
    games2011 = data_loader.getSeason('2011')
    print(f'Loaded {len(games2011)} 2011 games')
    print(f'First entry in 2011 season is\n{games2011.iloc[[0]]}')

    # Loading a season that is ahead of the cutoff training time returns no results.
    # In this case, the default cutoff time is in 2019, so loading 2020 data returns no results.
    # You can change the cutoff time by passing to simulate.py
    #     --cutoff YYYY-MM-DD
    games2020 = data_loader.getSeason('2020')
    print(f'Loaded {len(games2020)} 2020 games')

    # You can load an individual match's data
    aGame = data_loader.getGame(games2011.loc[100, 'gameId'])
    print(f'Game 100 in the 2011 database is\n{aGame}')

    # You can also load the full players data for an entire season
    full2011seasonPlayers = data_loader.getPlayers('2011')
    print(f'Loaded {len(full2011seasonPlayers)} 2011 season player rows')

    print(f'Required predictions are\n{required_predictions}')

    # You should fill in the sum and diff fields of the required predictions
    for index, match in required_predictions.iterrows():
        # Here we are using the values from the baseline error model, so we should get the baseline score
        required_predictions.at[index, 'predictedSum'] = 200
        required_predictions.at[index, 'predictedDiff'] = 0
    print('finished')


def get_multi_season_game_data(data_loader, first_year, last_year):
    data = [pd.DataFrame(data_loader.getSeason(str(season))) for season in range(first_year, last_year + 1)]
    data = pd.concat(data, axis=0)
    data.dropna(axis=0, inplace=True)
    data.dateTime=pd.to_datetime(data.dateTime)
    data.sort_values('dateTime', inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data

def predict(required_predictions, data_loader, first_season=2009, last_season=2020):

    print('Loading training data')
    train_data = get_multi_season_game_data(data_loader, first_season, last_season)
    first_match_date = train_data.dateTime.dt.date.min().toordinal()
    first_monday = first_match_date - (first_match_date - 1) % 7
    train_data['week'] = train_data['dateTime'].dt.date.apply(lambda x: (x.toordinal() - first_monday)//7)

    model = ELO(K=10, home_advantage=0, use_margin=False, lag=3, reset_after_season=True)
    model.add_data(train_data)
    model.evolve(weeks='current_season') #calculate ELO over all weeks available
    model.fit()

    current_season = train_data['season'].max()
    data_season = train_data[train_data['season'] == current_season]
    first_match_date = data_season.dateTime.dt.date.min().toordinal()
    first_monday = first_match_date - (first_match_date - 1) % 7
    data_season['week'] = train_data['dateTime'].dt.date.apply(lambda x: (x.toordinal() - first_monday)//7)
    print(f"Calculating predictions for season {current_season}, week {data_season['week'].max() + 1}")
    model.predict(required_predictions)

    return required_predictions
