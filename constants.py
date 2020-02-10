import pandas as pd

tpSeries = pd.core.series.Series
tpFrame = pd.core.frame.DataFrame

Timeframe = str
TIMEFRAME_MIN5 = '5T'
TIMEFRAME_MIN15 = '15T'
TIMEFRAME_HOUR = 'H' 
TIMEFRAME_DAY = 'D'
TIMEFRAME_MONTH = 'M'

FINAM_DATETIME = '%Y%m%d%H%M%S'
AV_DATETIME = '%Y-%m-%d %H:%M:%S'

Direction = str
DIRECTION_EXIT = 'EXIT'
DIRECTION_LONG = 'LONG'
DIRECTION_SHORT = 'SHORT'

Mode = str
MODE_ALL = 'all'
MODE_LONGS = 'longs'
MODE_SHORTS = 'shorts'

Signal = int
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_EXIT = 0
