import pandas as pd
import numpy as np
import logging
import operator

from typing import Any, Dict, List, Union, Optional
from typeguard import typechecked
from functools import partial
from itertools import product
from abc import ABC, abstractmethod
from collections.abc import Iterable

from constants import *


@typechecked
class Indicator:
    """
    Class with Indicators definitions.
    
    By Indicator we understand ANY function, 
    applied to a Series and returning Series as a result.
    Shoul satisfy tpIndicator = Callable[[tpSeries], tpSeries]
    
    TODO: Import standard financial libraries.
    """
    
    @staticmethod
    def sma(prices: tpSeries, **kwargs) -> tpSeries:
        """
        Simple Moving Average. 
        
        Expects 'window' in parameters dictionary - 
        number of previous bars for calucating average.
        
        Parameters:
        prices - Series of prices for function application
        
        Returns:
        Series of resulting calculations
        """
    
        if 'window' not in kwargs:
            raise KeyError('Parameters list must contain a window value')
        return prices.rolling(kwargs['window']).mean().fillna(0)
    

@typechecked
class Policy:
    """
    Class defines different policies of applying the Indicator.
    """
    
    @classmethod
    def cross(cls, fst:tpSeries, snd:tpSeries, 
              ret_val:int = 1, op = operator.eq) -> tpSeries:
        """
        Generate Series, where ret_val is where fst Series cross snd Series, 
        otherwise NaN 
        
        Parameters:
        fst - first Series of Data
        snd - second Series of Data
        ret_val - value to be returned
        op - operator for comparison, equal by default
        
        Returns:
        Series with ret_val and NaN
        """
        
        sign = np.sign(fst - snd)
        change = op(sign, sign.shift())
        return ret_val * change.where(change)
    
    @classmethod
    def up_cross_indicator(cls, prices: tpSeries, indicator: tpIndicator, 
                           ret_val:int, **kwargs) -> tpSeries:
        """
        Generate Series, where ret_val is where prices up cross indicator, 
        otherwise NaN 
        
        Parameters:
        prices - Series of Data
        indicator - indicator, applied to the prices
        ret_val - value to be returned
        
        Returns:
        Series with ret_val and NaN
        """
        
        return cls.cross(prices, indicator(prices, **kwargs), ret_val, operator.lt)
    
    @classmethod
    def down_cross_indicator(cls, prices: tpSeries, indicator: tpIndicator, 
                             ret_val:int, **kwargs) -> tpSeries:
        """
        Generate Series, where ret_val is where prices down cross indicator, 
        otherwise NaN 
        
        Parameters:
        prices - Series of Data
        indicator - indicator, applied to the prices
        ret_val - value to be returned
        
        Returns:
        Series with ret_val and NaN
        """
        
        return cls.cross(prices, indicator(prices, **kwargs), ret_val, operator.gt)
    
    @classmethod
    def propagate(cls, signal: tpSeries, ret_val: int, **kwargs) -> tpFrame:
        """
        Propagate signal for several bars. 
        Number of bars is a 'window' parameter in kwargs.
        ------------------
        | index | signal |
        |-------|--------|
        |   0   |   1    |
        |   1   |  NaN   |
        |   2   |  NaN   |
        |   3   |  NaN   |
        ------------------
        ret_val = 0, window = 2
        ---------------------------
        | index | signal | sindex |
        |-------|--------|--------|
        |   0   |   NaN  |   NaT  |
        |   1   |   NaN  |   NaT  |
        |   2   |    0   |    0   |
        |   3   |   NaN  |    1   |
        ---------------------------
        
        Parameters:
        signal - Series of original signal
        ret_val - value for returning
        
        Returns:
        DataFrame[signal, sindex] - shifted signal and index to what it belongs
        """
        
        if 'window' not in kwargs:
            raise KeyError('Parameters list must contain window value')   
        w = kwargs['window']
        if w < 0:
            raise ValueError('Value of window must be >= 0')   

        # buy-and-hold strategy
        if w == 0:
            # returns only NaN == never exit
            return signal.mask(abs(signal) == 1, np.NaN)
        else: # w > 0
            result = pd.DataFrame()
            result['signal'] = signal.copy()
            result['signal'] = result.signal.mask(abs(result.signal) == 1, ret_val)
            result['sindex'] = signal.index
            result = result.shift(w)
            return result
        
        
@typechecked
class StrategyFrame(ABC):
    """
    StrategyFrame class joins enters' and exits' policies
    into one combined signal, while ignoring some enters when 
    in position.
    """
    
    def __init__(self, long: tpSeries = None, short: tpSeries = None, 
                 exLong: Union[tpSeries, tpFrame, partial] = None, 
                 exShort: Union[tpSeries, tpFrame, partial] = None, 
                 enter_on_next: bool = False) -> None:
        """
        long - Series for entering longs
        short - Series for entering shorts
        exLong - Series or Frame for exiting longs
        exShort - Series or Frame for exiting shorts    
        enter_on_next - if True, cannot enter on the same bar with exit, 
                        only on the next one
        """
        
        self._long = long
        self._short = short
        self._exLong = exLong
        self._exShort = exShort
        self._onNext = 1 if enter_on_next else 0
    
    @property
    def long(self) -> tpSeries:
        return self._long
    
    @long.setter
    def long(self, value: tpSeries):
        self._long = value

    @property
    def short(self) -> tpSeries:
        return self._short
    
    @short.setter
    def short(self, value: tpSeries):
        self._short = value
            
    @property
    def exLong(self) -> Union[tpSeries, tpFrame, partial]:
        return self._exLong
     
    @exLong.setter
    def exLong(self, value: Union[tpSeries, tpFrame, partial]):
        self._exLong = value
    
    @property
    def exShort(self) -> Union[tpSeries, tpFrame, partial]:
        return self._exShort
     
    @exShort.setter
    def exShort(self, value: Union[tpSeries, tpFrame, partial]):
        self._exShort = value
    
    def _check_properties(self) -> (bool, bool):
        """
        Check if enough properties were specified
        
        Parameters:
        At least one must be specified:
            long - Policy for entering longs
            short - Policy for entering shorts
        Must be specified, if `long` is declared:
            exLong - Policy for exiting longs
        Must be specified, if `short` is declared:
            exShort - Policy for exiting shorts    
            
        Returns:
        (fst, snd)
        fst - True, if long and exLong was specified
        snd - True, if short and exShort was specified
        """
        
        fst = False
        snd = False
        
        if (self._long is None) and (self._short is None):
            logging.error("At least long or short Policy must be declared.")
        
        if self._long is not None:
            if self._exLong is None:
                logging.error("Policy for exit from longs must be declared.")
            elif type(self._exLong) == partial:
                if type(self._exLong(signal = self._long)) == partial:
                    logging.error("Not enough parameters for exLong Policy.")
                else:
                    self._exLong = self._exLong(signal = self._long)  
                    fst = True
            else:
                fst = True
        
        if self._short is not None:
            if self._exShort is None:
                logging.error("Policy for exit from shorts must be declared.")
            elif type(self._exShort) == partial:
                if type(self._exShort(signal = self._short)) == partial:
                    logging.error("Not enough parameters for exShort Policy.")
                else:
                    self._exShort = self._exShort(signal = self._short)  
                    snd = True
            else:
                snd = True
        
        return fst, snd
    
    def _get_signal(self, data: Union[tpSeries, tpFrame]) -> Signal:
        """
        Returns signal value from the data
        
        Parameters:
        data - Series or DataFrame with `signal` column
        
        Returns:
        Value of signal, {-1, 0, 1}
        """
        
        if type(data) == tpFrame:
            data = data['signal']
        return data.unique()[-1]        
    
    def _next_index(self, exit: Union[tpSeries, tpFrame], 
                    prev_idx: tpDateTime) -> Optional[tpDateTime]:
        """
        Find next valuable after prev_idx index in exit.
        Take care about tpSeries and tpFrame data types.
        
        Parameters:
        exit - Series or DataFrame for exiting positions
        prev_idx - previous valuable index
        
        Returns:
        Next valuable index in exit data
        """
        
        if type(exit) == tpSeries:
            idx = exit[prev_idx:].index
            if len(idx) > 0:
                return exit[idx[1:]].first_valid_index()
            else:
                return None
        else: # type(exit) == tpFrame
            idx = exit.iloc[:, 1].loc[lambda s: s == prev_idx]
            return idx.index[0] if len(idx.index) > 0 else None
    
    def _combine2(self, enter: tpSeries, exit: Union[tpSeries, tpFrame]) -> tpSeries:
        """
        Combines enter and corresponding exit signal into one.
        
        Parameters:
        Both generated by Policy class.
        enter - Series of enters
        exit - Series or DataFrame of exits
        
        Returns:
        Series of signal
        """
        
        entPoints = set()
        exPoints = set()
        signal = self._get_signal(exit)
        
        idx = enter.first_valid_index()
        while idx is not None:
            entPoints.add(idx)
            idx = self._next_index(exit, idx)
            if idx is not None:
                exPoints.add(idx)
                idx = enter[idx:][self._onNext:].first_valid_index()
    
        result = enter.mask(~enter.index.isin(entPoints), np.NaN)
        result = result.mask(result.index.isin(exPoints - entPoints), signal)
        return result
    
    def _find_min(self, fst: Optional[tpDateTime], 
                  snd: Optional[tpDateTime]) -> (Optional[tpDateTime], Optional[bool]):
        """
        Find min of two arguments, handling None as max
        
        Returns:
        (min, bool)
        True - if the first one was smaller
        False - if the second one was smaller
        None - otherwise
        """
        
        if (fst is None) & (snd is None):
            return (None, None)
        if fst is None: return (snd, False) 
        if snd is None: return (fst, True)
        result = min(fst, snd)
        return (result, result == fst)
    
    def _combine4(self, long: tpSeries, exLong: Union[tpSeries, tpFrame], 
                  short: tpSeries, exShort: Union[tpSeries, tpFrame]) -> tpSeries:
        """
        Combines Policies into one consistent signal.
        
        Parameters:
        All generated by Policy class.
        long - Series of long enters
        exLong - Series or DataFrame of long exits
        short - Series of short enters
        exShort - Series or DataFrame of short exits
        
        Returns:
        Series of signal
        """
        
        entLongPoints = set()
        entShortPoints = set()
        exLongPoints = set()
        exShortPoints = set()
        
        # saving Signals for futher using
        signal_exit_long = self._get_signal(exLong)
        signal_short = self._get_signal(short)
        signal_exit_short = self._get_signal(exShort)
        
        idx, in_long = self._find_min(long.first_valid_index(), 
                                      short.first_valid_index())
        while idx is not None:
            if in_long:
                entLongPoints.add(idx)
                idx = self._next_index(exLong, idx)
                if idx is not None:
                    exLongPoints.add(idx)
                    idx, in_long = self._find_min(long[idx:][self._onNext:].first_valid_index(), 
                                                  short[idx:][self._onNext:].first_valid_index())
            else:
                entShortPoints.add(idx)
                idx = self._next_index(exShort, idx)
                if idx is not None:
                    exShortPoints.add(idx)
                    idx, in_long = self._find_min(long[idx:][self._onNext:].first_valid_index(), 
                                                  short[idx:][self._onNext:].first_valid_index())
                
        result = long.mask(~long.index.isin(entLongPoints), np.NaN)
        result = result.mask(result.index.isin(entShortPoints), signal_short)
        enters = entLongPoints.union(entShortPoints)
        result = result.mask(result.index.isin(exLongPoints - enters), signal_exit_long)
        result = result.mask(result.index.isin(exShortPoints - enters), signal_exit_short)
        
        return result
    
    def signal(self) -> tpSeries:
        """
        Combines Policies into one consistent signal.
        
        Returns:
        Series of a signal data.
        """
        
        has_long, has_short = self._check_properties()
        if has_long and has_short:
            return self._combine4(self._long, self._exLong, 
                                  self._short, self._exShort)
        elif has_long:
            return self._combine2(self._long, self._exLong)
        else: # if has_short
            return self._combine2(self._short, self._exShort)
        
        
@typechecked
class Strategy(Iterable):
    """
    Strategy is a class providing an interface for
    all subsequent (inherited) strategy handling objects.
    Responsible for iteration for parameters' tuning.

    The goal of a (derived) Strategy object is to generate Signal
    objects for particular symbols based on the inputs of Bars 
    (OHLCV) generated by a DataHandler object.
    """

    def __init__(self, name: str, 
                 data: tpFrame,
                 long: tpPolicy = None, exLong: tpPolicy = None, 
                 short: tpPolicy = None, exShort: tpPolicy = None,
                 enter_on_next: bool = False,
                 modes: List[Mode] = [MODE_ALL, MODE_LONGS, MODE_SHORTS], 
                 long_params: Dict[str, Any] = {}, 
                 exLong_params: Dict[str, Any] = {}, 
                 short_params: Dict[str, Any] = {}, 
                 exShort_params: Dict[str, Any] = {}) -> None:
        """
        Parameters:
        name - name of the strategy
        data - DataFrame with bars
        long - Policy for long enters
        exLong - Policy for long exits
        short - Policy for short enters
        exShort - Policy for short exits
        enter_on_next - if True, cannot enter on the same bar with exit, 
                        only on the next one
        modes - list of availiable modes (all, longs or shorts only)
        long_params - dict with parameters for `long`
        exLong_params - dict with parameters for `exLong`
        short_params - dict with parameters for `short`
        exShort_params - dict with parameters for `exShort`
        """
        
        self._name = {'Strategy name': name}
        self._data = data
        self._long = long
        self._exLong = exLong
        self._short = short
        self._exShort = exShort
        self._onNext = enter_on_next
        self._long_params = long_params
        self._exLong_params = exLong_params
        self._short_params = short_params
        self._exShort_params = exShort_params
        
        #check whether enough parameters were given
        self._has_long, self._has_short = self._check_properties()
        if self._has_long or self._has_short:
            # set up Policies and returns a dict with all iterable parameters
            self._params = self._set_up_params(modes)
            logging.debug("Number of variable parameters is %s" %len(self._params))
            # all variations of possible parameters
            self._iter_params = self.product_dict(**self._params)
    
    def reset(self):
        """
        Reset parameters for a new iteration
        """
        self._iter_params = self.product_dict(**self._params)
    
    def _check_properties(self) -> (bool, bool):
        """
        Check if enough properties were specified
        
        Parameters:
        At least one must be specified:
            long - Policy for entering longs
            short - Policy for entering shorts
        Must be specified, if `long` is declared:
            exLong - Policy for exiting longs
        Must be specified, if `short` is declared:
            exShort - Policy for exiting shorts    
            
        Returns:
        (fst, snd)
        fst - True, if long and exLong was specified
        snd - True, if short and exShort was specified
        """
        
        fst = False
        snd = False
        
        if (self._long is None) and (self._short is None):
            logging.error("At least long or short Policy must be declared.")
        
        if self._long is not None: 
            if self._exLong is None:
                logging.error("Policy for exit from longs must be declared.")
            else:
                fst = True
            
        if self._short is not None:
            if self._exShort is None:
                logging.error("Policy for exit from shorts must be declared.")
            else:
                snd = True
        
        return fst, snd
    
    def _set_up_params(self, modes: List[Mode]) -> dict:
        """
        Responsible for setting all Policies
        
        Parameters:
        modes - list of availiable modes (all, longs or shorts only)
        
        Returns:
        dict of all iterable parameters
        """ 
        
        result = {'modes': modes}

        if self._has_long:
            self._long, self._long_params, params = self._set_up_policy(
                                self._long, self._long_params, 
                                LONG_PREFIX, STR_LONG, STR_LPAR)
            result.update(params)
            self._exLong, self._exLong_params, params = self._set_up_policy(
                                self._exLong, self._exLong_params, 
                                EXLONG_PREFIX, STR_ELONG, STR_ELPAR)
            result.update(params)
            
        if self._has_short:
            self._short, self._short_params, params = self._set_up_policy(
                                self._short, self._short_params, 
                                SHORT_PREFIX, STR_SHORT, STR_SPAR)
            result.update(params)
            self._exShort, self._exShort_params, params = self._set_up_policy(
                                self._exShort, self._exShort_params, 
                                EXSHORT_PREFIX, STR_ESHORT, STR_ESPAR)
            result.update(params)

        return result
    
    def _set_up_policy(self, ls: tpPolicy, params: Dict[str, Any], prefix: str,
                      dic_strat: str, dic_params: str) -> (partial, dict, dict):
        """
        Set up enter or exit Policy, while parsing parameters and 
        setting them for iteration or calling
        
        Parameters:
        ls - Policy method, which returns tpSeries or tpFrame
        params - parameters for calling Policy method
        prefix - prefix for combining dictionary
        dic_strat - string for dictionary representation
        dic_params - string for dictionary representation
        
        Returns:
        partial application of Policy
        dict of defaults for iterable parameters
        dict for parameters iteration 'param_name' -> [param_variations]
        """
        
        def_dict = params.get('defaults')
        iter_params = {} # dict of lists for parameters iteration
        def_params = {} # dict of defaults for iterable parameters
        basic_params = {} # dict of parameters, that will not change
        
        for k, v in params.items():
            if (type(v) == list) or (type(v) == range):
                if type(v) == range: 
                    v = list(v)
                iter_params[prefix + k] = v
                if k in def_dict:
                    def_params[k] = def_dict[k]
                else:
                    def_params[k] = v[0]
            elif type(v) != dict:
                basic_params[k] = v
                
        self._name[dic_strat] = str(ls)
        self._name[dic_params] = str(basic_params)
        
        basic_params = self._update_price(basic_params)
        return partial(ls, **basic_params), def_params, iter_params
    
    def _update_price(self, params: dict) -> dict:
        if 'prices' in params:
            name = params['prices']
            params['prices'] = self._data[name]
        return params   
          
    def product_dict(self, **kwargs):
        """
        Provides iteration over dictionary.
        Same as itertools.product, but for dictionary, not lists.
        """
        
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))
        
    def calculate_all(self, m:Mode = MODE_ALL,  
                      long_params: Dict[str, Any] = {}, 
                      exLong_params: Dict[str, Any] = {}, 
                      short_params: Dict[str, Any] = {}, 
                      exShort_params: Dict[str, Any] = {}) -> (str, tpSeries):
        """
        Provides the mechanisms to calculate all signals at once 
        for the historic data.
        
        Parameters:
        
        m - Mode: all, shorts or longs only
        long_params - parameters for long Policy, if not using defaults
        exLong_params - parameters for exLong Policy, if not using defaults
        short_params - parameters for short Policy, if not using defaults
        exShort_params - parameters for exShort Policy, if not using defaults
        
        Returns: 
        String representation of the Strategy
        Series of signals {-1, 0, 1}, where
        -1 is for enter short, or keeping short position
         0 is for exit position, if any
         1 is for enter long, or keeping long position
        """ 
        
        name = {'Strategy': self._name['Strategy name'], STR_MODE: m}
        
        frame = StrategyFrame(enter_on_next = self._onNext)
        
        if self._has_long and (m != MODE_SHORTS):
            if not long_params:
                long_params = self._long_params.copy()
            if not exLong_params:
                exLong_params = self._exLong_params.copy()
                
            if long_params:
                name[STR_LPAR] = '{}'.format(long_params)
                long_params = self._update_price(long_params)
            frame.long = self._long(**long_params)
            
            if exLong_params:
                name[STR_ELPAR] = '{}'.format(exLong_params)
                exLong_params = self._update_price(exLong_params)
            frame.exLong = partial(self._exLong, **exLong_params)         
                        
        if self._has_short and (m != MODE_LONGS):
            if not short_params:
                short_params = self._short_params.copy()
            if not exShort_params:
                exShort_params = self._exShort_params.copy()
            
            if short_params:
                name[STR_SPAR] = '{}'.format(short_params)
                short_params = self._update_price(short_params)
            frame.short = self._short(**short_params) 
            
            if exShort_params:
                name[STR_ESPAR] = '{}'.format(exShort_params)
                exShort_params = self._update_price(exShort_params)
            frame.exShort = partial(self._exShort, **exShort_params)
                        
        return self._dict_format(name), frame.signal()               
    
    def _dict_format(self, name: dict) -> str:
        s = []
        for k, v in name.items():
            s.append('{}: {}'.format(k, v))
        return '\n'.join(s)
    
    def __str__(self):    
        return self._dict_format(self._name)
    
    def __iter__(self):
        return self   
                               
    def __next__(self):
        value = next(self._iter_params)
        return self.calculate_all(m = value['modes'],  
                  long_params = {k[PREFIX_LEN:]:v for (k, v) in value.items() if k.startswith(LONG_PREFIX)}, 
                  exLong_params = {k[PREFIX_LEN:]:v for (k, v) in value.items() if k.startswith(EXLONG_PREFIX)}, 
                  short_params = {k[PREFIX_LEN:]:v for (k, v) in value.items() if k.startswith(SHORT_PREFIX)}, 
                  exShort_params = {k[PREFIX_LEN:]:v for (k, v) in value.items() if k.startswith(EXSHORT_PREFIX)})