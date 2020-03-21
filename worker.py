import strategy as s
import portfolio as p

from constants import *
    
def calculate_all(data: tpFrame, strat: s.Strategy, params: dict):
    name, res = strat.calculate_fromDict(data, params)
    name, res = p.NaivePortfolio.apply_strategy(data, name, res)
    trades = p.NaivePortfolio.get_trades(origin = res)
    metrics = p.NaivePortfolio.get_metrics(res, name, trades)
    return name, metrics