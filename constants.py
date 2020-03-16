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

PREFIX_LEN = 4
LONG_PREFIX = 'lng_'
EXLONG_PREFIX = 'eln_'
SHORT_PREFIX = 'sht_'
EXSHORT_PREFIX = 'esh_'

STR_NAME = 'Strategy name'
STR_MODE = 'Runnning mode'
STR_LONG = 'Long function'
STR_LPAR = 'Long parameters'
STR_ELONG = 'Exit long function'
STR_ELPAR = 'Exit long parameters'
STR_SHORT = 'Short function'
STR_SPAR = 'Short parameters'
STR_ESHORT = 'Exit short function'
STR_ESPAR = 'Exit short parameters'