import pandas as pd
import numpy as np
import logging
import time

from typing import Any, Union, Optional, Dict
from typeguard import typechecked
from functools import partial
from abc import ABC, abstractmethod

from data import CSVDataHandler
from strategy import Indicator, Policy, Strategy

from constants import *


class Portfolio(ABC):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar".
    """

    '''
    @abstractmethod
    def update_time(self, event):
        """
        Acts on a MarketEvent to update current positions.
        """
        raise NotImplementedError("Should implement update_signal()")
    
    @abstractmethod
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        """
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        raise NotImplementedError("Should implement update_fill()")
    '''
        
    @abstractmethod
    def apply_strategy(self, symbol, strategy):
        """
        Applies strategy to the symbol in vector format.
        """
        raise NotImplementedError("Should implement apply_strategy()")
        
    @abstractmethod
    def get_trades(self, symbol, strategy_name):
        """
        Returns a DataFrame with all trades
        """
        raise NotImplementedError("Should implement get_trades()")
        
    @abstractmethod
    def get_metrics(self, symbol, strategy_name):
        """
        Returns a DataFrame with calculated metrics
        """
        raise NotImplementedError("Should implement get_metrics()")
        
        
@typechecked
class NaivePortfolio(Portfolio):
    """
    The NaivePortfolio object is designed to send orders to
    a brokerage object blindly, i.e. without any risk management. 
    It is used to test simple strategies.
    """
    
    def __init__(self, bars, capital=10000.0, quantity = 1, 
                 events = None, symbols = None, calc_fee = True, 
                 fees = {'base':0.000354, 'min':0.5}):
        """
        Initialises the portfolio with bars and an event queue. 
        Also includes a starting datetime index and initial capital 
        (USD unless otherwise stated).

        Parameters:
        bars - The DataHandler object with current market data.
        capital - The starting capital in USD for each ticker.
        quantity - The number of shares to buy/sell. If =0, then go on maximum sum.
        events - The Event Queue object.
        symbols - List of tickers for processing. If =None, then take all in DataHandler.
        calc_fee - Whether or not fee is calculated.
        fees - Dictionary for fees calculations (base, min)
        """
        
        self.strategy_name = 'events'
        self.open_trade = {}
        self.bars_held = {}
        
        self._bars = bars
        self._capital = capital
        self._quantity = quantity
        self._events = events
        self._symbols = symbols
        self._calc_fee = calc_fee
        self._fees = fees
        
        self.visualizer = StockTradingGraph()
               
        self.positions = {} # (symbol, strategy) -> DataFrame during time
        self.trades = {} # (symbol, strategy) -> DataFrame of trades, summary
        self.metrics = {} # (symbol, strategy) -> DataFrame of calculated metrics
        
        # if in event mode, prepare dataframes for positions and trades
        '''
        if events is not None:
            if symbols is None: 
                self.symbols = self.bars.symbol_data.keys()
            for s in self.symbols:
                self.positions[(s, self.strategy_name)] = self._create_positions_df(s)
                self.trades[(s, self.strategy_name)] = pd.DataFrame(columns=['entry_date'])
                self.trades[(s, self.strategy_name)].set_index('entry_date', inplace = True)
        '''
    
    @staticmethod
    def apply_strategy(data: tpFrame, name: str, signal: tpSeries, 
                       capital:float = 10000.0, quantity:int = 1, 
                       calc_fee:bool = False, 
                       fees:dict = {'base':0.000354, 'min':0.5}) -> (str, tpFrame):
        """
        Function is responsible for applying given strategy and calculating 
        dataframe with results
        
        Parameters:
        data - Dataframe with prices for Strategy application
        name - short name of the Strategy (for current signal)
        signal - signal of the Strategy
        capital - The starting capital in USD for each ticker.
        quantity - The number of shares to buy/sell. Should be > 0.
        calc_fee - Whether or not fee is calculated.
        fees - Dictionary for fees calculations (base, min)
        
        Returns:
        Dataframe with results
        """
        
        df = data.copy()[['open', 'high', 'low', 'close']]
        df['signal'] = signal
        # make continuous signal: 1, 1, 1, 0, 0, 0 - without NaNs
        df['signal'].fillna(method='pad', inplace=True)
        if quantity <= 0:
            logging.debug("Can not go for max sum in vector mode, let quantity = 1")
            quantity = 1
        # volume of the position, + for long, - for short
        df['position'] = quantity * df.signal.shift()
        df['position'].fillna(0, inplace=True)
        # number of bars held so far, + and - also
        df['bars'] = df.signal.groupby((df.signal != df.signal.shift()).cumsum()).cumsum().shift()
        df['bars'].fillna(0, inplace=True)
        # count prices of open positions
        df.loc[df.bars == 1, 'open_long'] = df.position * df.open
        df.loc[df.bars == 1, 'open_short'] = 0
        df.loc[df.bars == -1, 'open_short'] = df.position * df.open
        df.loc[df.bars == -1, 'open_long'] = 0
        df.loc[df.bars == 0, ['open_long', 'open_short']] = [[0, 0]]
        # propagate open prices while not exit
        df['open_long'].fillna(method='pad', inplace=True)
        df['open_short'].fillna(method='pad', inplace=True)
        # and propagate for one more bar for exit position
        df.loc[(df.bars >= 0) & (df.bars.shift() < 0), 'open_short'] = df.open_short.shift()
        df.loc[(df.bars <= 0) & (df.bars.shift() > 0), 'open_long'] = df.open_long.shift()
        # calculations for drawback and equity
        df['hl_cost'] = 0.0
        df.loc[df.position > 0, 'hl_cost'] = df.position * df.low    
        df.loc[df.position < 0, 'hl_cost'] = df.position * df.high
        df['close_cost'] = df.position * df.close 
        # close position
        df.loc[(df.position != df.position.shift()) & (df.position.shift() != 0), 'close_trade'] = df.position.shift() * df.open 
        df['close_trade'].fillna(0, inplace=True)
        # fees calculations
        df['fee'] = 0.0
        if calc_fee:
            df.loc[df.bars == 1, 'fee'] = np.maximum(df.open_long * fees['base'], fees['min'])
            df.loc[df.bars == -1, 'fee'] = np.maximum(abs(df.open_short) * fees['base'], fees['min'])
            df.loc[df.close_trade != 0, 'fee'] += np.maximum(abs(df.close_trade) * fees['base'], fees['min'])
        df['fee_total'] = df.fee.cumsum()
        # profit calculations
        df['profit'] = 0.0
        df.loc[df.open_long > 0, 'profit'] = df.close_trade + df.close_cost - df.open_long + df.open_short
        df.loc[df.open_short < 0, 'profit'] = df.close_trade + df.close_cost - df.open_short - df.open_long
        # fixed profit
        df['profit_fix'] = 0.0
        df.loc[df.close_trade > 0, 'profit_fix'] = df.close_trade - df.open_long
        df.loc[df.close_trade < 0, 'profit_fix'] = df.close_trade - df.open_short
        df['profit_total'] = df.profit_fix.cumsum()
        # total capital and equity calculations
        df['total'] = capital - df.fee_total + df.profit_total
        df['equity'] = df.total + df.profit - df.profit_fix
        #df = df.apply(pd.to_numeric, downcast='float')
        return name, df

    @staticmethod
    def get_trades(origin: tpFrame, file:str = None) -> tpFrame:
        """
        Function is responsible for extracting trades from the given  
        dataframe in the WealthLab format
        
        Parameters:
        origin - DataFrame after strategy application
        file - file name for saving results
        
        Returns:
        Dataframe with trades
        """
        
        df = pd.DataFrame()
        # entering positions
        df['entry_date'] = origin[abs(origin.bars) == 1].index
        df.set_index('entry_date', inplace = True)
        df['quantity'] = abs(origin[abs(origin.bars) == 1].position)
        df.loc[origin[origin.bars == 1].index, 'position'] = 'LONG'
        df.loc[origin[origin.bars == -1].index, 'position'] = 'SHORT'
        if not df.loc[df.position == 'SHORT'].empty:
            df.loc[df.position == 'SHORT', 'entry_price'] = abs(origin.open_short)
        if not df.loc[df.position == 'LONG'].empty:
            df.loc[df.position == 'LONG', 'entry_price'] = origin.open_long
        # exiting positions
        ex_date = origin[origin.close_trade != 0].index.tolist()
        ex_price = abs(origin[origin.close_trade != 0].close_trade).tolist()
        profit = origin[origin.profit_fix != 0].profit_fix.tolist()
        # if the last trade was not closed
        if (len(ex_date) < df.shape[0]): 
            ex_date.append(np.nan)
            ex_price.append(np.nan)
        if (len(profit) < df.shape[0]): 
            profit.append(np.nan)
        df['exit_date'] = ex_date
        df['exit_price'] = ex_price
        # other parameters calculations
        df['profit, $'] = np.around(profit, decimals = 2)
        df['profit, %'] = np.around(profit / df.entry_price * 100, decimals = 2)
        df['bars_held'] = (abs(origin.loc[(origin.position != origin.position.shift(-1)) & (origin.position != 0), 'bars']) + 1).tolist()
        df['quantity'] = df['quantity'].astype('int32', copy = False)
        df['bars_held'] = df['bars_held'].astype('int32', copy = False)

        if file is not None:
            df.to_csv(file)    
        return df
        
    @staticmethod
    def get_metrics(origin:tpFrame, strategy_name: str, 
                    trades: tpFrame, metrics: tpFrame = None, 
                    file:str = None) -> tpFrame:
        """
        Function is responsible for calculating metrics, e.g. Profit factor
        
        Parameters:
        origin - DataFrame after strategy application
        strategy_name - Name of the stratedy for a header
        trades - DataFrame with extracted trades
        metrics - DataFrame of previously calculated metrics
        file - file name for saving results
        
        Returns:
        Dataframe with metrics
        """
        
        if metrics is None:
            metrics = pd.DataFrame(columns = ['Feature', strategy_name])
            metrics.set_index('Feature', inplace = True)
        
        tr_copy = trades.copy()
        if (pd.isnull(tr_copy.exit_date[-1])): 
            tr_copy.iloc[-1, tr_copy.columns.get_loc('exit_date')] = origin.index[-1]
            tr_copy.iloc[-1, tr_copy.columns.get_loc('exit_price')] = origin.close[-1]
            profit = origin.profit[-1]
            tr_copy.iloc[-1, tr_copy.columns.get_loc('profit, $')] = np.around(profit, decimals = 2)
            tr_copy.iloc[-1, tr_copy.columns.get_loc('profit, %')] = np.around(profit / tr_copy.entry_price[-1] * 100, decimals = 2)
        wins = tr_copy.loc[tr_copy['profit, $'] > 0]
        losses = tr_copy.loc[tr_copy['profit, $'] < 0]      
        
        metrics.loc['Net Profit', strategy_name] = np.around(tr_copy['profit, $'].sum(), decimals = 2)
        metrics.loc['Total Commission', strategy_name] =  origin.fee_total.max()
        num_trades = len(tr_copy)
        metrics.loc['Number of Trades', strategy_name] = num_trades
        metrics.loc['Average Profit', strategy_name] = np.around(metrics.loc['Net Profit', strategy_name] / num_trades, decimals = 2)
        metrics.loc['Average Profit, %', strategy_name] = np.around(tr_copy['profit, %'].mean(), decimals = 2)
        metrics.loc['Average Bars Held', strategy_name] = np.around(tr_copy['bars_held'].mean(), decimals = 2)
            
        num_win_trades = len(wins)
        metrics.loc['Winning Trades', strategy_name] = num_win_trades
        metrics.loc['Win Rate, %', strategy_name] = np.around(metrics.loc['Winning Trades', strategy_name] / num_trades * 100, decimals = 2)
        metrics.loc['Gross Profit', strategy_name] = sum(wins['profit, $'])
        metrics.loc['Avg Profit', strategy_name] = np.around(metrics.loc['Gross Profit', strategy_name] / num_win_trades, decimals = 2)
        metrics.loc['Avg Profit, %', strategy_name] = np.around(wins['profit, %'].mean(), decimals = 2)
        metrics.loc['Avg Winning Bars Held', strategy_name] = np.around(wins['bars_held'].mean(), decimals = 2)
        tr_copy.loc[:, 'consec'] = np.where(tr_copy['profit, $'] > 0, 1, 0)
        tr_copy.loc[:, 'consec'] = tr_copy.consec.groupby((tr_copy.consec == 0).cumsum()).cumcount()
        metrics.loc['Max Consecutive Winners', strategy_name] = tr_copy.consec.max()

        num_los_trades = len(losses)
        metrics.loc['Losing Trades', strategy_name] = num_los_trades
        metrics.loc['Loss Rate, %', strategy_name] = np.around(metrics.loc['Losing Trades', strategy_name] / num_trades * 100, decimals = 2)
        metrics.loc['Gross Loss', strategy_name] = sum(losses['profit, $'])
        metrics.loc['Avg Loss', strategy_name] = np.around(metrics.loc['Gross Loss', strategy_name] / num_los_trades, decimals = 2)
        metrics.loc['Avg Loss, %', strategy_name] = np.around(losses['profit, %'].mean(), decimals = 2)
        metrics.loc['Avg Losing Bars Held', strategy_name] = np.around(losses['bars_held'].mean(), decimals = 2)
        tr_copy.loc[:, 'consec'] = np.where(tr_copy['profit, $'] < 0, 1, 0)
        tr_copy.loc[:, 'consec'] = tr_copy.consec.groupby((tr_copy.consec == 0).cumsum()).cumcount()
        metrics.loc['Max Consecutive Losses', strategy_name] = tr_copy.consec.max()

        drawdown = origin.equity.cummax() - origin.equity
        metrics.loc['Maximum Drawdown', strategy_name] = np.around(drawdown.max(), decimals = 2)
        metrics.loc['Maximum Drawdown Date', strategy_name] = drawdown.idxmax()
            
        metrics.loc['Profit Factor', strategy_name] = np.around(
            metrics.loc['Gross Profit', strategy_name] / np.absolute(metrics.loc['Gross Loss', strategy_name]), 
            decimals = 2)
        metrics.loc['Recovery Factor', strategy_name] = np.around(
            metrics.loc['Net Profit', strategy_name] / np.absolute(drawdown.max()), 
            decimals = 2)
        metrics.loc['Payoff Ratio', strategy_name] = np.around(
            metrics.loc['Avg Profit', strategy_name] / np.absolute(metrics.loc['Avg Loss', strategy_name]), 
            decimals = 2)
        
        if file is not None:
            metrics.to_csv(file)    
        return metrics