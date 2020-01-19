from collections import defaultdict
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from scipy.optimize import minimize


def home_win_probability(home_elo, away_elo,
                         K, home_advantage # source: 538
                        ):
    return 1 / (1 + math.pow(K, -(home_advantage+home_elo - away_elo) / 400))


## Get new Elo ratings home and away teams after a game
def get_updated_elo(
    home_elo, away_elo,
    point_diff, ## > 0 if home team wins, < 0 if away team wins, cannot be 0 (will throw error)
    K,  ## model hyperparameter
    home_advantage,
    use_margin
):

    P_home_win = home_win_probability(home_elo, away_elo, K, home_advantage)
    P_away_win = 1 - P_home_win

    # When home team wins
    if point_diff > 0 :
        elo_diff = home_elo-away_elo
        M = (point_diff+3)**0.8/(7.5+0.006*elo_diff) if use_margin else 1 # multiplier, source 538
        home_elo += M * K * P_away_win
        away_elo -= M * K * P_home_win

    # When away team wins
    elif point_diff < 0 :
        elo_diff = away_elo-home_elo
        M = (-point_diff+3)**0.8/(7.5+0.006*elo_diff) if use_margin else 1 # multiplier, source 538
        home_elo -= M * K * P_away_win
        away_elo += M * K * P_home_win

    else : raise ValueError(f"point_diff should be non-zero number. Got {point_diff}")

    return home_elo, away_elo


## Iterate through games updating each teams Elo rating
def get_elos_over_time(data,  ## dataframe of games, must be in order of occurence
                       starting_elo_dict={},  ## dictionary of elo scores by team at the beginning of the data period
                       default_elo=0,  ## elo initally given to a team not in starting_elo_dict
                       K=10,
                       home_advantage=100,
                       use_margin=True## model hyperparameter; higher number means individuals game affects Elo more
                       ):
    elo_dict = starting_elo_dict.copy()

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
        home_elo, away_elo = get_updated_elo(home_elo, away_elo, point_diff, K, home_advantage, use_margin)
        elo_dict[home_team] = home_elo
        elo_dict[away_team] = away_elo

    return elo_dict

def fit_scores(train_data, family):
    train_data['EloDifference'] = train_data['homeElo'] - train_data['awayElo']
    train_data['EloSum'] = train_data['homeElo'] + train_data['awayElo']
    X = train_data[['EloDifference', 'EloSum']]
    X = sm.add_constant(X.values)

    # y = train_data['pointsDiff']
    # diff_model = sm.OLS(y, X).fit()
    # y = train_data['pointsSum']
    # sum_model = sm.OLS(y, X).fit()

    y = train_data['homeScore']
    home_model = sm.GLM(y, X, family=family).fit()
    y = train_data['awayScore']
    away_model = sm.GLM(y, X, family=family).fit()

    #return diff_model, sum_model
    return home_model, away_model

#def predict_scores(required_predictions, elo_dict, diff_model, sum_model):
def predict_scores(required_predictions, elo_dict, home_model, away_model):
    tmp = required_predictions[['homeTeam', 'awayTeam']].copy()
    tmp['homeElo'] = [elo_dict.get(team,0) for team in tmp['homeTeam']]
    tmp['awayElo'] = [elo_dict.get(team,0) for team in tmp['awayTeam']]
    tmp['EloDifference'] = tmp.eval('homeElo - awayElo')
    tmp['EloSum'] = tmp.eval('homeElo + awayElo')
    X = tmp[['EloDifference', 'EloSum']]
    X = sm.add_constant(X.values)
    #tmp['predictedDiff'] = diff_model.predict(X)
    #tmp['predictedSum'] = sum_model.predict(X)
    tmp['predictedDiff'] = home_model.predict(X) - away_model.predict(X)
    tmp['predictedSum'] = home_model.predict(X) + away_model.predict(X)


    required_predictions['predictedDiff'] = tmp['predictedDiff']
    required_predictions['predictedSum'] = tmp['predictedSum']

class ELO():
    def __init__(self, K=10, home_advantage=0, use_margin=False, lag = -1, reset_after_season=False,
                 family=sm.families.Gaussian()):
        """
        reset_evolve     resetting ELO dictionary in the beginning of a new seasons
        """
        self.data = None
        self.ELO = defaultdict(float)
        self.K = K
        self.home_advantage = home_advantage
        self.use_margin = use_margin
        self.reset_after_season = reset_after_season
        self.family = family

        if not (lag == -1 or lag > 0):
            raise Exception('Lag parameter must be either -1 or positive integer')
        self.lag = lag

    def add_data(self, data):
        self.data = pd.concat([self.data, data], sort=False)

    def evolve(self, weeks='last'):
        if 'homeElo' not in self.data.columns:
            self.data.loc[:,'homeElo'] = np.nan
            self.data.loc[:,'awayElo'] = np.nan

        if weeks == 'last':
            last_week = self.data[self.data['week'] == self.data['week'].max()]
            previous_weeks = self.data[self.data['week'] < self.data['week'].max()]
            self.ELO = get_elos_over_time(last_week,
                                          starting_elo_dict=self.ELO,
                                          K=self.K, home_advantage=self.home_advantage, use_margin=self.use_margin)
            self.data = pd.concat([previous_weeks, last_week], sort=False)

        if weeks == 'current_season':
            evolve_data = self.data[self.data['season'] == self.data['season'].max()]
            keep_data = self.data[self.data['season'] != self.data['season'].max()]
            self.ELO = get_elos_over_time(evolve_data,
                                          starting_elo_dict={},
                                          K=self.K, home_advantage=self.home_advantage, use_margin=self.use_margin)
            self.data = pd.concat([evolve_data, keep_data], sort=False)

        if weeks == 'all':
            self.ELO = get_elos_over_time(self.data,
                                          starting_elo_dict={},
                                          K=self.K, home_advantage=self.home_advantage, use_margin=self.use_margin)



    def reset_evolve(self):
        if self.reset_after_season:
            self.ELO = defaultdict(float)

    def fit(self):
        if self.lag == -1:
            current_season = self.data['season'].max()
            fit_data = self.data[self.data['season'] == current_season]
        else:
            weeks = sorted(self.data['week'].unique(), reverse=True)[0:self.lag]
            fit_data = self.data[self.data['week'].isin(weeks)]
        self.diff_model, self.sum_model = fit_scores(fit_data, self.family)
        self.diff_model, self.sum_model = fit_scores(fit_data, self.family)

    def predict(self, required_predictions):
        predict_scores(required_predictions, self.ELO, self.diff_model, self.sum_model)


class Powers():
    def __init__(self, lag = -1):
        self.data = None
        if not (lag == -1 or lag > 0):
            raise Exception('Lag parameter must be either -1 or positive integer')
        self.lag = lag

    @staticmethod
    def loss(X, M, powers):
        pred_home = powers[:,0].reshape(-1,1) - powers[:,3].reshape(1,-1)
        pred_away = powers[:,2].reshape(-1,1) - powers[:,1].reshape(1,-1)
        loss = np.sum(((pred_home - X[:,:,0])**2 + (pred_away - X[:,:,1])**2)*M)
        return loss

    @staticmethod
    def predict_match(home_team, away_team, teams, powers):
        if not(home_team in teams and away_team in teams):
            return 200, 100
        else:
            id_home = teams.index(home_team)
            id_away = teams.index(away_team)
            pred_home = powers[id_home][0] - powers[id_away][3] #home attack - away defense
            pred_away = powers[id_away][2] - powers[id_home][1] #away attack - home defense
            pred_diff = pred_home - pred_away
            pred_sum = pred_home + pred_away
            return pred_diff, pred_sum

    def add_data(self, data):
        self.data = pd.concat([self.data, data], sort=False)
        self.teams = list(set(self.data.homeTeam.unique()).union(set(self.data.awayTeam.unique())))
        self.n = len(self.teams)

    def evolve(self, **kwargs):
        pass

    def reset_evolve(self):
        pass

    def fit(self):
        if self.lag == -1:
            current_season = self.data['season'].max()
            fit_data = self.data[self.data['season'] == current_season]
        else:
            weeks = sorted(self.data['week'].unique(), reverse=True)[0:self.lag]
            fit_data = self.data[self.data['week'].isin(weeks)]

        X = np.zeros([self.n,self.n, 2])
        M = np.zeros([self.n,self.n])
        for _, row in fit_data.iterrows():
            id_home = self.teams.index(row['homeTeam'])
            id_away = self.teams.index(row['awayTeam'])
            #first approach: if more matches between the same teams, use score of the later match
            X[id_home, id_away, 0] = (row['pointsDiff'] + row['pointsSum'])/2
            X[id_home, id_away, 1] = (-row['pointsDiff'] + row['pointsSum'])/2
            M[id_home, id_away] = 1

        loss_fun = lambda x: self.loss(X, M, x.reshape(-1,4))
        x0 = np.zeros([self.n,4])
        x0[:,0] = 100; x0[:,2] = 100
        self.powers = minimize(loss_fun,
                          x0.reshape(-1),
                          method='Nelder-Mead'
                         ).x.reshape(-1,4)

    def predict(self, required_predictions):
        required_predictions['predictedDiff'], required_predictions['predictedSum'] = zip(*required_predictions.apply(
                                                        lambda x: self.predict_match(x['homeTeam'],x['awayTeam'],self.teams,self.powers),
                                                        axis = 1)
                                                       )
