import json
import re
import urllib.parse
from typing import Callable
import requests
import pandas
import time

class SimulationSettings:
    env: str = 'prod'
    cutoff: str
    cutoffend: str = None
    resultpath: str = None
    predict: Callable

def _getRequest(url):
    r = requests.get(url)
    if r.status_code < 200 or r.status_code > 299:
        raise Exception(f'Could not obtain data from url {url}. Server responded with status code {r.status_code}')
    return r.json()

class NbaDataLoader:
    def __init__(self, settings: SimulationSettings):
        self.settings = settings
        self.playerColumns = [
            'gameId',
            'name',
            'dateTime',
            'team',
            'season',
            'blocks',
            'injuryBodyPart',
            'injuryStatus',
            'minutes',
            'points',
            'position',
            'rebounds',
            'steals'
        ]

    # Obtain the games of a season.
    # Seasons are strings such as '2009' or '2010POST'
    # The earliest available season is '2009'
    def getSeason(self, season: str):
        data = _getRequest(f'https://{self.settings.env}api.nbadatachallenge.com/data/seasons/{urllib.parse.quote(season)}')
        result = []
        for d in data:
            dateTime = d['dateTime']
            if (dateTime is not None) and dateTime < self.settings.cutoff:
                result.append(d)
        if len(result) == 0:
            return pandas.DataFrame()
        result = pandas.DataFrame(result, columns=['gameId', 'dateTime', 'homeTeam', 'awayTeam', 'homeBlocks', 'homeMinutes', 'homeRebounds', 'homeScore', 'homeSteals', 'quarter0home', 'quarter1home', 'quarter2home', 'quarter3home', 'awayBlocks', 'awayMinutes', 'awayRebounds', 'awayScore', 'awaySteals', 'quarter0away', 'quarter1away', 'quarter2away', 'quarter3away', 'season', 'status'])
        result['pointsSum'] = result.eval('homeScore + awayScore')
        result['pointsDiff'] = result.eval('homeScore - awayScore')
        result = result[['gameId', 'dateTime', 'homeTeam', 'awayTeam', 'pointsDiff', 'pointsSum', 'homeBlocks', 'homeMinutes', 'homeRebounds', 'homeScore', 'homeSteals', 'quarter0home', 'quarter1home', 'quarter2home', 'quarter3home', 'awayBlocks', 'awayMinutes', 'awayRebounds', 'awayScore', 'awaySteals', 'quarter0away', 'quarter1away', 'quarter2away', 'quarter3away', 'season', 'status']]
        return result

    # Obtain a single game data
    # The gameId is a numerical game identifier.
    # You can find the gameId from the results of getSeason
    def getGame(self, gameId: int):
        data = _getRequest(f'https://{self.settings.env}api.nbadatachallenge.com/data/games/{urllib.parse.quote(str(gameId))}')
        result = []
        for d in data:
            dateTime = d['dateTime']
            if dateTime < self.settings.cutoff:
                result.append(d)
        return pandas.DataFrame(result, columns=self.playerColumns)

    # Obtain full player data about all the games in a season.
    def getPlayers(self, season: str):
        data = _getRequest(f'https://{self.settings.env}api.nbadatachallenge.com/data/gameplayersfull/{urllib.parse.quote(season)}')
        result = []
        for d in data:
            dateTime = d['dateTime']
            if dateTime < self.settings.cutoff:
                result.append(d)
        return pandas.DataFrame(result, columns=self.playerColumns)

def _loadPredictions(settings: SimulationSettings):
    url = f'https://{settings.env}api.nbadatachallenge.com/data/predictions/{urllib.parse.quote(settings.cutoff)}'
    if settings.cutoffend is not None:
        url += f'/{urllib.parse.quote(settings.cutoffend)}'
    return _getRequest(url)

def _sanitizeResult(results, predictions):
    if len(results) != len(predictions):
        raise Exception(f'User returned {len(results)} predictions, but expecting {len(predictions)} predictions')
    sanitized = []
    for result in results:
        if not isinstance(result['gameId'], int):
            raise Exception(f'gameId field in the prediction must be an int')
        if not (isinstance(result['predictedSum'], float) or isinstance(result['predictedSum'], int)):
            raise Exception(f'predictedSum field in the prediction must be a float or int')
        if not (isinstance(result['predictedDiff'], float) or isinstance(result['predictedDiff'], int)):
            raise Exception(f'predictedDiff field in the prediction must be a float or int')
        sanitized.append({
            'gameId': result['gameId'],
            'predictedSum': result['predictedSum'],
            'predictedDiff': result['predictedDiff']
        })
    return sanitized

def _findByGameId(results, gameId):
    for r in results:
        if r['gameId'] == gameId:
            return r
    return None

def _getField(doc, field):
    if doc is None:
        return 'None'
    if field in doc:
        return doc[field]
    else:
        return 'None'

def _computeSum(a, b):
    if a is None or b is None:
        return None
    return a + b

def _computeDiff(a, b):
    if a is None or b is None:
        return None
    return a - b

def _displayPredictionsAndResults(results, actual):
    totalScore = 0
    baselineSum = 200
    baselineDiff = 0
    for game in actual:
        gameId = game['gameId']
        resultGame = _findByGameId(results, gameId)
        homeScore = _getField(game, 'homeScore')
        awayScore = _getField(game, 'awayScore')
        actualSum = _computeSum(homeScore, awayScore)
        actualDiff = _computeDiff(homeScore, awayScore)
        predictedSum = _getField(resultGame, 'predictedSum')
        predictedDiff = _getField(resultGame, 'predictedDiff')
        totalScore -= single_game_error(predictedDiff, predictedSum, actualDiff, actualSum)
        totalScore += single_game_error(baselineDiff, baselineSum, actualDiff, actualSum)
        print(f'Game {gameId}. Actual results: home {homeScore} - away {awayScore}. '
            f'Actual: sum {actualSum} - diff {actualDiff}. '
            f'Predicted results: sum {predictedSum} - predictedDiff {predictedDiff}')
    print(f'Total score is {totalScore}, number of games is {len(actual)}, score per game is {totalScore/len(actual)}')

def single_game_error(predictedDiff, predictedSum, actualDiff, actualSum):
    return abs(predictedDiff - actualDiff) + abs(predictedSum - actualSum)


def score_predictions(predictions):
    x1 = predictions['predictedDiff']
    x2 = predictions['predictedSum']
    y1 = predictions['pointsDiff']
    y2 = predictions['pointsSum']

    ## baseline model
    x1_baseline = 0  ## no information about who will win
    x2_baseline = 200  ## avergae points total between 2009 and 2016 seasons

    predictions['error'] = single_game_error(x1, x2, y1, y2)
    predictions['baseline_error'] = single_game_error(x1_baseline, x2_baseline, y1, y2)

    predictions['score'] = predictions.eval('baseline_error - error')

    return predictions.score.sum()

def runSimulation(settings: SimulationSettings):
    startTime = time.time()
    if not re.match('^\d\d\d\d-\d\d-\d\d$', settings.cutoff):
        print(f'--cutoff argument value is not valid. Expected a YYYY-MM-DD format')
        return

    print(f'Loading prediction matches starting from {settings.cutoff}')
    predictionsFull = _loadPredictions(settings)
    predictions = []
    for prediction in predictionsFull:
        predictions.append({
            'date': prediction['date'],
            'homeTeam': prediction['homeTeam'],
            'awayTeam': prediction['awayTeam'],
            'gameId': prediction['gameId']
        })

    predictions = pandas.DataFrame(predictions, columns=['gameId', 'date', 'homeTeam', 'awayTeam', 'predictedSum', 'predictedDiff'])

    dataLoader = NbaDataLoader(settings)

    print('Starting call to user defined function')
    settings.predict(predictions, dataLoader)
    print('User defined function completed')
    result = predictions.to_dict('records')
    result = _sanitizeResult(result, predictionsFull)
    _displayPredictionsAndResults(result, predictionsFull)

    if settings.resultpath is not None:
        print('Writing result...')
        resultfile = open(settings.resultpath, 'w')
        resultfile.write(json.dumps(result))
        resultfile.close()

    elapsedSeconds = time.time() - startTime
    print(f'Completed in {elapsedSeconds} seconds')
    return predictions
