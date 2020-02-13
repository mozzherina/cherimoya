import pandas as pd
import numpy as np
import logging
import os, os.path
import datetime
import pytz

from abc import ABC, abstractmethod
from pytz import timezone
from queue import Queue, Empty
from typing import Any, Dict

from constants import *
from event import MarketEvent


class DataHandler(ABC):
    """
    DataHandler is an abstract base class providing an interface for
    all inherited data handlers (both live and historic).
    """
    
    @abstractmethod
    def all_bars(self, symbol:str) -> tpFrame:
        """
        Returns all bars for a given symbol.
        
        Parameters:
        symbol - a Ticker name 
        
        Returns: 
        DataFrame for the symbol
        datetime -> OHLCV
        """
        raise NotImplementedError("Should implement all_bars(symbol)")
    
    @abstractmethod
    def latest_bars(self, symbol:str, n:int = 1) -> (tpFrame, bool):
        """
        Returns the DataFrame with last N bars for a given symbol 
        (or fewer if less bars are available) 
        
        Parameters:
        symbol - a Ticker name 
        n - a number of bars
        
        Returns: 
        DataFrame for the symbol
        datetime -> OHLCV
        True - if enough bars was available
        False - if fewer bars was available
        """
        raise NotImplementedError("Should implement latest_bars(symbol, n)")

    @abstractmethod
    def update_bars(self) -> bool:
        """
        Updates a datetime, thus shift to the next latest bar.
        
        Returns: 
        True - if shift was successful
        False - if no more bars exist
        """
        raise NotImplementedError("Should implement update_bars()")
        

class CSVDataHandler(DataHandler):
    """
    CSVDataHandler is designed to read CSV files for
    each requested symbol from disk and implement the DataHandler
    interface. 
    """

    def __init__(self, csv_dir: str, system_tf: Timeframe, 
                 symbol_dict: Dict[str, Dict[str, Any]], events: Queue = None) -> None:
        """
        Initialises the data handler by requesting the location 
        of the CSV files and a list of symbols.

        It is assumed that all files are in the form 'symbol<...>.csv', 
        where symbol is in the symbol_dict dictionary: symbol -> parameters.

        Parameters:
        csv_dir - Directory path to the CSV files (the 'data' folder).
        system_tf - Timeframe for a system 'heartbeat' and resampling (e.g. 5 minutes, day).
        symbol_dict - A dictionary of symbols with parameters.
        events - The Event Queue. If None (as by default), then only all_bars() is available.
        """
        
        self._csv_dir = csv_dir
        self._system_tf = system_tf
        self._symbol_dict = symbol_dict
        self._events = events
        
        self._symbol_data = {} # Dictionary: symbol -> DataFrame of bars
        # {'APPL':0} -> {'AAPL':1} -> ... {'APPL':None}
        self._latest_idx = {} # Dictionary: symbol -> index of the 'last' bar
        
        self.continue_backtest = True # whether there are more bars

        self._convert_csv_files()
    
    def _read_csv(self, file: str, joint: bool) -> tpFrame:
        """
        Calls pandas read_csv function with suitable parameters.
        
        Parameters:
        file - full file name.
        joint - if True, date and time are in the same column.
        
        Returns:
        DataFrame
        """
        
        n = []
        if joint:
            n=['datetime','open','high','low','close','volume']
        else:
            n=['date', 'time','open','high','low','close','volume']
        return pd.io.parsers.read_csv(file, header=None, skiprows=1, names=n)
    
    def _load_csv(self, symbol: str, joint_dt: bool, exact_name:str = None) -> tpFrame:
        """
        Load data from the files
        
        Parameters:
        symbol - the Ticker name.
        joint_dt - if True, Date and Time are in the same column.
        exact_name - exact file name; if None, then looking for all files 'symbol[...].csv'.
        
        Returns: 
        DataFrame for the symbol
        """
        
        df = pd.DataFrame()
        if exact_name:
            # load data from the exact file
            try:
                file = os.path.join(self._csv_dir, self._symbol_dict[symbol]['file'])
                df = self._read_csv(file, joint_dt)
            except KeyError:
                logging.error("No file name is given")
                raise
        else: # combine all files like 'symbol...csv'
            frames = []
            for root, dirs, files in os.walk(self._csv_dir):
                for file in files:
                    if file.startswith(symbol) and 'checkpoint' not in file:
                        frames.append(self._read_csv(os.path.join(root, file), joint_dt))
            df = pd.concat(frames)        
        return df
    
    def _convert_files(self, symbol: str, dt_format: str, joint_dt: bool, reindex: bool) -> tpFrame:
        """
        Import files, downloaded from sample sources.
        
        Parameters:
        symbol - the Ticker name.
        dt_format - string of a DateTime format.
        joint_dt - if True, Date and Time are in the same column.
        reindex - if True, then revert index from the last to the first.
        
        Returns: 
        DataFrame for the symbol
        """
        
        df = pd.DataFrame()
        try:
            df = self._load_csv(symbol, joint_dt, exact_name = self._symbol_dict[symbol]['exn'])
        except KeyError:
            logging.debug("No exact name is given. Take all files %s<...>.csv" % symbol)
            df = self._load_csv(symbol, joint_dt)
       
        # set date + time as index
        try:
            if not joint_dt:
                df['datetime'] = df['date'].astype(str) + df['time'].astype(str)
            df['datetime'] = pd.to_datetime(df['datetime'], format=dt_format)
        except ValueError:
            logging.debug("No time column is given")
            # '%Y-%m-%d %H:%M:%S' extract only '%Y-%m-%d'
            df['datetime'] = pd.to_datetime(df['date'], format=dt_format[0:dt_format.index('d') + 1])
        df = df.set_index('datetime')
        
        if not joint_dt:
            df.drop('date', axis = 1, inplace=True)
            df.drop('time', axis = 1, inplace=True)
        
        # reindex from the last to the first
        if reindex: 
            df = df.reindex(index=df.index[::-1])
            
        return df
    
    def _resample_symbol_data(self, df: tpFrame) -> tpFrame:
        """
        Resample the DataFrame according to the selected timeframe.
        If timeframe is less, than already given to the system, 
        does nothing.
        
        Parameters:
        df - Dataframe.
        
        Returns: 
        modified Dataframe.
        """
        
        df_new = pd.DataFrame({'open': df.open.resample(self._system_tf, label='right', closed='right').first().dropna(),
                               'high': df.high.resample(self._system_tf, label='right', closed='right').max().dropna(),
                               'low': df.low.resample(self._system_tf, label='right', closed='right').min().dropna(),
                               'close': df.close.resample(self._system_tf, label='right', closed='right').last().dropna(),
                               'volume': df.volume.resample(self._system_tf, label='right', closed='right').sum(
                                   min_count = 1).dropna().astype(float)
                              })
        #df_new = df_new.apply(pd.to_numeric, downcast='float')
        return df_new   
    
    def _convert_csv_files(self) -> bool:
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames under a symbol in the _symbol_data 
        dictionary.
        
        Returns:
        True - if all tickers were loaded
        False - otherwise
        """    
        
        result = True
        for s in self._symbol_dict.keys():
            df = None
            if self._symbol_dict[s]['src'] is 'av': # file is from Alpha Vantage
                df = self._convert_files(s, AV_DATETIME, True, True)
            elif self._symbol_dict[s]['src'] is 'finam': # file is from Finam
                df = self._convert_files(s, FINAM_DATETIME, False, False)
            # elif ... another source
            
            if df is None:
                result = False
                logging.error("Error converting files for the symbol %s" % s)
            else:
                df = self._resample_symbol_data(df)
                # if timeframe is minutes or hours then set up localization
                if ('T' or 'H') in self._system_tf:
                    df = df.tz_localize(timezone(self._symbol_dict[s]['tz']))

                self._symbol_data[s] = df
                self._latest_idx[s] = 0
                # save maximum index for this symbol
                self._symbol_dict[s]['len'] = len(list(df.index))
        return result
    
    def all_bars(self, symbol):
        return self._symbol_data[symbol]
  
    def latest_bars(self, symbol:str, n:int = 1) -> (tpFrame, bool):
        """
        Returns the DataFrame with last n bars for a given symbol 
        (or fewer if less bars are available) 
        """  
        
        try:
            bars = self._symbol_data[symbol]
            idx = self._latest_idx[symbol]
            df = bars.iloc[idx-n:idx]
            return df, (df.shape[0] == n)       
        except KeyError:
            logging.error("Symbol %s is not available in the data set." % s)            
      
    def _next_datetime(self) -> tpDateTime:
        """
        Return next minimum datetime from the data.
        """
        dt = None
        for s in self._symbol_dict.keys():
            if self._latest_idx[s] is not None:
                idx = self._latest_idx[s]
                idx_dt = self._symbol_data[s].index[idx]
                if dt is None or (idx_dt < dt): 
                    dt = idx_dt
        return dt
    
    def update_bars(self) -> bool:
        """
        Updates a datetime, thus shift to the next latest bar.
        """
        
        for s in self._symbol_dict.keys():
            if self._latest_idx[s] == self._symbol_dict[s]['len']:
                self._latest_idx[s] = None
                  
        self._system_dt = self._next_datetime()
        
        # find symbols, which have updated time in the data
        result = False
        upd_symbols = []
        for s in self._symbol_dict.keys():
            if self._latest_idx[s] is not None:
                result = True
                if self._system_dt in self._symbol_data[s].index:
                    self._latest_idx[s] += 1  
                    upd_symbols.append(s)
                    
        if self._events is not None:
            self._events.put(MarketEvent(upd_symbols, self._system_dt))               
        
        self.continue_backtest = result
        return result