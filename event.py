from abc import ABC, abstractmethod
from typing import List

from constants import *

class Event(ABC):
    """
    Event is base class providing an interface for all subsequent 
    (inherited) events, that will trigger further events in the 
    trading infrastructure.   
    """
    
    def __init__(self, event_type: Event) -> None:
        self._event_type = event_type
    
    @property
    def event_type(self) -> Event:
        return self._event_type
    
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError("Should implement __str__()")


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update 
    (new bars become available)
    """

    def __init__(self, symbols: List[str], datetime: tpDateTime) -> None:
        """
        Parameters:
        symbols - List of ticker symbols that have new data.
        datetime - The timestamp at which the signal was generated.
        """
        super().__init__(EVENT_MARKET)
        self._symbols = symbols
        self._datetime = datetime
    
    def __str__(self) -> str:
        return "MARKET {} at {}".format(self._symbols, self._datetime)