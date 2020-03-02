import pandas as pd
from typing import Callable, Union

tpSeries = pd.core.series.Series
tpFrame = pd.core.frame.DataFrame
tpDateTime = pd.Timestamp
tpIndicator = Callable[[tpSeries], tpSeries]
tpPolicy = Callable[..., Union[tpSeries, tpFrame]]

Timeframe = str
TIMEFRAME_MIN5 = '5T'
TIMEFRAME_MIN15 = '15T'
TIMEFRAME_HOUR = 'H' 
TIMEFRAME_DAY = 'D'
TIMEFRAME_MONTH = 'M'

FINAM_DATETIME = '%Y%m%d%H%M%S'
AV_DATETIME = '%Y-%m-%d %H:%M:%S'
STD_DATE = '%Y-%m-%d'
STD_DATETIME = '%Y-%m-%d %H:%M'

Direction = str
DIRECTION_EXIT = 'EXIT'
DIRECTION_LONG = 'LONG'
DIRECTION_SHORT = 'SHORT'

Mode = str
MODE_ALL = 'all'
MODE_LONGS = 'longs'
MODE_SHORTS = 'shorts'

Signal = Union[int, float]
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_EXIT = 0

Event = str
EVENT_MARKET = 'MARKET'

BAR_SIZE = 5
BAR_SIZE_NARROW = 1
MARKER_SIZE = 15
FIGURE_SIZE = (15, 11)
