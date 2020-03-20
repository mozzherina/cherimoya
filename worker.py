import strategy as s
import portfolio as p

from constants import *

def mycalc(data: tpFrame, strat: s.Strategy, params:dict):
    return params
    
def calc_all(data: tpFrame, strat: s.Strategy, params: dict):
    name, res = strat.calculate_fromDict(data, params)
    return name, res

#port.apply_strategy(data, *
#return port.get_metrics(res, name)
