import pandas as pd
import numpy as np
import os, os.path
import datetime
import pytz

from abc import ABC, abstractmethod
from pytz import timezone

from event import MarketEvent


class DataHandler(ABC):
    """
    DataHandler is an abstract base class providing an interface for
    all inherited data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars for each symbol requested. 
    """
    
    @abstractmethod
    def get_all_bars(self, symbol):
        """
        Returns a dataframe for the symbol.
        """
        raise NotImplementedError("Should implement get_all_bars()")
    
    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the dataframe with last N bars 
        for the symbol, or fewer if less bars are available, 
        AND number of returned bars.
        
        Return: (df, n)
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Updates latest_idx structure for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")
        

class CSVDataHandler(DataHandler):
    """
    CSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide the DataHandler
    interface. 
    """

    def __init__(self, csv_dir, system_tf, symbol_dict, events = None):
        """
        Initialises the data handler by requesting the location 
        of the CSV files and a list of symbols.

        It will be assumed that all files are in the form 'symbol...csv', 
        where symbol is from the dictionary symbol -> parameters.

        Parameters:
        csv_dir - Directory path to the CSV files ("data" folder).
        system_tf - Timeframe for system "heartbeat" and resampling.
        symbol_dict - A dictionary of symbols with parameters.
        events - The Event Queue. If None (as by default), then only get_all_bars()
        """
        
        self.csv_dir = csv_dir
        self.system_tf = system_tf
        self.symbol_dict = symbol_dict
        self.events = events
        
        self.symbol_data = {}
        self.latest_idx = {} # {'APPL':0} -> .. {'APPL':None}
        self.continue_backtest = True       

        self._open_convert_csv_files()
    
    def _resample_symbol_data(self, df):
        """
        Resample dataframe according to the selected timeframe.
        If timeframe is less, than already given to the system, 
        does nothing.
        
        Parameters:
        df - Dataframe.
        Return: new Dataframe
        """
        
        df_new = pd.DataFrame({'open': df.open.resample(self.system_tf, label='right', closed='right').first().dropna(),
                               'high': df.high.resample(self.system_tf, label='right', closed='right').max().dropna(),
                               'low': df.low.resample(self.system_tf, label='right', closed='right').min().dropna(),
                               'close': df.close.resample(self.system_tf, label='right', closed='right').last().dropna(),
                               'volume': df.volume.resample(self.system_tf, label='right', closed='right').sum(
                                   min_count = 1).dropna().astype(float)
                              })
        #df_new = df_new.apply(pd.to_numeric, downcast='float')
        return df_new
    
    def _read_csv(self, file, joint = None):
        n = []
        if joint:
            n=['datetime','open','high','low','close','volume']
        else:
            n=['date', 'time','open','high','low','close','volume']
        return pd.io.parsers.read_csv(file, header=None, skiprows=1, names=n)
    
    def _load_csv(self, symbol, joint, exact = False):
        df = pd.DataFrame()
        if exact:
            # load data from the exact file
            try:
                file = os.path.join(self.csv_dir, self.symbol_dict[symbol]['file'])
                df = self._read_csv(file, joint=joint)
            except KeyError:
                print("No file name is given")
                raise
        else: # combine all files like 'symbol...csv'
            frames = []
            for root, dirs, files in os.walk(self.csv_dir):
                for file in files:
                    if file.startswith(symbol) and 'checkpoint' not in file:
                        frames.append(self._read_csv(os.path.join(root, file), joint=joint))
            df = pd.concat(frames)        
        return df
    
    def _convert_av_files(self, symbol):
        """
        Import files, downloaded from Alpha Vantage.
        
        Parameters:
        symbol - Ticker name.
        """
        df = pd.DataFrame()
        try:
             df = self._load_csv(symbol, True, self.symbol_dict[symbol]['exn'])
        except KeyError:
            print("No exact name is given")
            df = self._load_csv(symbol, True)
       
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('datetime')
        df = df.reindex(index=df.index[::-1])
        return df
    
    def _convert_finam_files(self, symbol):
        """
        Import files, downloaded from Finam.
        
        Parameters:
        symbol - Ticker name.
        """
        df = pd.DataFrame()
        try:
             df = self._load_csv(symbol, False, self.symbol_dict[symbol]['exn'])
        except KeyError:
            print("No exact name is given")
            df = self._load_csv(symbol, False)
       
        # set date + time as index
        try:
            df['datetime'] = df['date'].astype(str) + df['time'].astype(str)
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S')
        except ValueError:
            print("No time is given")
            df['datetime'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.set_index('datetime')
        # drop columns
        df.drop('date', axis = 1, inplace=True)
        df.drop('time', axis = 1, inplace=True)
        return df
    
    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames under a symbol dictionary.
        """        
        for s in self.symbol_dict.keys():
            df = None
            if self.symbol_dict[s]['src'] is 'av':
                df = self._convert_av_files(s)
            elif self.symbol_dict[s]['src'] is 'finam':            
                df = self._convert_finam_files(s)
            #elif ... another source
            
            df = self._resample_symbol_data(df)
            if self.system_tf not in ['D', 'M']:
                df = df.tz_localize(timezone(self.symbol_dict[s]['tz']))
            
            self.symbol_data[s] = df
            self.latest_idx[s] = 0
            self.symbol_dict[s]['len'] = len(list(df.index))
    
    def get_all_bars(self, symbol):
        return self.symbol_data[symbol]

    def _next_datetime(self):
        """
        Return next minimum datetime from the data.
        """
        dt = None
        for s in self.symbol_dict.keys():
            if self.latest_idx[s] is not None:
                idx = self.latest_idx[s]
                idx_dt = self.symbol_data[s].index[idx]
                if dt is None or idx_dt < dt: 
                    dt = idx_dt
        return dt
            
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars for the symbol or N-k if less available.
        
        Parameters:
        symbol - Ticker name.
        N - number of bars to return, 1 by default.
        """
        try:
            bars_list = self.symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the data set.")
        else:
            idx = self.latest_idx[symbol]
            df = bars_list.iloc[idx-N:idx]
            n = df.shape[0]
            return df, n          
            
    def update_bars(self):
        """
        Increase bar index according to the new system_dt.
        Check whether is still enough data.
        """
        for s in self.symbol_dict.keys():
            if self.latest_idx[s] == self.symbol_dict[s]['len']:
                self.latest_idx[s] = None
                  
        self.system_dt = self._next_datetime()
        
        cont = False
        new_data = []
        for s in self.symbol_dict.keys():
            if self.latest_idx[s] is not None:
                cont = True
                if self.system_dt in self.symbol_data[s].index:
                    self.latest_idx[s] += 1  
                    new_data.append(s)
                    
        if self.events is not None:
            self.events.put(MarketEvent(new_data, self.system_dt))            
        self.continue_backtest = cont