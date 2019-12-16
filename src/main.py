import pandas as pd
import statsmodels.api as sm
import numpy as np
import math

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


## Elo model's probability of home team winning
def home_win_probability(home_elo, away_elo,
                         home_advantage = 100 # source: 538
                        ):
    return 1 / (1 + math.pow(10, -(home_advantage+home_elo - away_elo) / 400))


## Get new Elo ratings home and away teams after a game
def get_updated_elo(
    home_elo, away_elo,
    point_diff, ## > 0 if home team wins, < 0 if away team wins, cannot be 0 (will throw error)
    K,  ## model hyperparameter
):

    P_home_win = home_win_probability(home_elo, away_elo)
    P_away_win = 1 - P_home_win

    # When home team wins
    if point_diff > 0 :
        elo_diff = home_elo-away_elo
        M = (point_diff+3)**0.8/(7.5+0.006*elo_diff)# multiplier, source 538
        home_elo += M * K * P_away_win
        away_elo -= M * K * P_home_win

    # When away team wins
    elif point_diff < 0 :
        elo_diff = away_elo-home_elo
        M = (-point_diff+3)**0.8/(7.5+0.006*elo_diff)# multiplier, source 538
        home_elo -= M * K * P_away_win
        away_elo += M * K * P_home_win

    else : raise ValueError(f"point_diff should be non-zero number. Got {point_diff}")

    return home_elo, away_elo


## Iterate through games updating each teams Elo rating
def get_elos_over_time(data,  ## dataframe of games, must be in order of occurence
                       starting_elo_dict={},  ## dictionary of elo scores by team at the beginning of the data period
                       default_elo=0,  ## elo initally given to a team not in starting_elo_dict
                       K=20,  ## model hyperparameter; higher number means individuals game affects Elo more
                       ):
    elo_dict = starting_elo_dict.copy()
    data['homeElo'] = np.nan
    data['awayElo'] = np.nan

    ## Iterate over rows of the dataframe (i.e. over games)
    for i, row in data.iterrows():
        home_team = row['homeTeam']
        away_team = row['awayTeam']
        home_elo = elo_dict.get(home_team, default_elo)
        away_elo = elo_dict.get(away_team, default_elo)

        ## Put the team's current ELO in the dataframe (this is the teams ELO *before* the match)
        data.loc[i, 'homeElo'] = home_elo
        data.loc[i, 'awayElo'] = away_elo

        ## Calculate the new elo scores and update elo_dict with them
        point_diff = row['pointsDiff']
        home_elo, away_elo = get_updated_elo(home_elo, away_elo, point_diff, K)
        elo_dict[home_team] = home_elo
        elo_dict[away_team] = away_elo

    return elo_dict

def fit_scores(train_data):
    train_data['EloDifference'] = train_data['homeElo'] - train_data['awayElo']
    train_data['EloSum'] = train_data['homeElo'] + train_data['awayElo']

    #print('Fitting linear model from Elo difference and sum to points difference')
    X = train_data[['EloDifference', 'EloSum']]
    X = sm.add_constant(X)
    y = train_data['pointsDiff']
    diff_model = sm.OLS(y, X).fit()

    #print('Fitting linear model from Elo difference and sum to points sum')
    y = train_data['pointsSum']
    sum_model = sm.OLS(y, X).fit()

    return diff_model, sum_model

def predict_scores(required_predictions, elo_dict, diff_model, sum_model):

    #print('Generating predictions')
    #     required_predictions = pd.DataFrame(required_predictions)
    tmp = required_predictions[['homeTeam', 'awayTeam']].copy()
    tmp['homeElo'] = [elo_dict.get(team,0) for team in tmp['homeTeam']]
    tmp['awayElo'] = [elo_dict.get(team,0) for team in tmp['awayTeam']]
    tmp['EloDifference'] = tmp.eval('homeElo - awayElo')
    tmp['EloSum'] = tmp.eval('homeElo + awayElo')
    X = tmp[['EloDifference', 'EloSum']]
    X = sm.add_constant(X)
    tmp['predictedDiff'] = diff_model.predict(X)
    tmp['predictedSum'] = sum_model.predict(X)

    required_predictions['predictedDiff'] = tmp['predictedDiff']
    required_predictions['predictedSum'] = tmp['predictedSum']

    #print('Finished')


def predict(required_predictions, data_loader, first_season=2020, last_season=2020):

    print('Loading training data')
    train_data = get_multi_season_game_data(data_loader, first_season, last_season)

    print('Getting Elo ratings over time on train data')
    elo_dict = get_elos_over_time(train_data,
                                  starting_elo_dict={},
                                  K=10)
    print(elo_dict)

    print('Fitting model')
    diff_model, sum_model = fit_scores(train_data)
    predict_scores(required_predictions, elo_dict, diff_model, sum_model)
    print(required_predictions)

    return required_predictions
