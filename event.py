from abc import ABC


class Event(ABC):
    """
    Event is base class providing an interface for all subsequent 
    (inherited) events, that will trigger further events in the 
    trading infrastructure.   
    """
    pass

class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with 
    corresponding bars.
    """

    def __init__(self, sym_list, datetime):
        """
        Initialises the MarketEvent.
        
        Parameters:
        sym_list - List of ticker symbols with new data.
        datetime - The timestamp at which the signal was generated.
        """
        self.type = 'MARKET'
        self.sym_list = sym_list
        self.datetime = datetime
    
    def __str__(self):
        return "MARKET {} at {}".format(
                    self.sym_list, self.datetime)
        
        
class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    
    def __init__(self, strategy, symbol, datetime, signal_type):
        """
        Initialises the SignalEvent.

        Parameters:
        symbol - The ticker symbol.
        strategy - Name of the strategy, which generated
        datetime - The timestamp at which the signal was generated.
        signal_type - 'LONG' or 'SHORT' or 'EXIT'
        """
        
        self.type = 'SIGNAL'
        self.strategy = strategy
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        
    def __str__(self):
        return "SIGNAL from {} at {} for {} do {}".format(
                    self.strategy, self.datetime, self.symbol, self.signal_type)

class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (i.e. ticker), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self, symbol, order_type, quantity, direction, maxsum_com = None):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), its 
        quantity (integral) and its direction ('BUY' or 'SELL').

        Parameters:
        symbol - The ticker symbol.
        order_type - 'MKT' or 'LMT' for Market or Limit.
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL' or 'EXIT LONG/SHORT'.
        maxsum_com - maximum sum with comission for order
        """
        
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.maxsum_com = maxsum_com # None if sell already bought

    def __str__(self):
        q = 'on all %s' % self.maxsum_com if self.quantity == 0 else 'exactly %s' % self.quantity 
        return "ORDER: Symbol={}, Type={}, Quantity={}, Direction={}".format(
                    self.symbol, self.order_type, q, self.direction)

class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """

    def __init__(self, datetime, symbol, quantity, 
                 direction, fill_cost, fee=None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional 
        commission.

        Parameters:
        datetime - Date and time when the order was filled.
        symbol - The ticker symbol.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL' or 'EXIT ...')
        fill_cost - The holdings value in dollars.
        commission - An optional commission sent from broker.
        """
        
        self.type = 'FILL'
        self.datetime = datetime
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.fee = fee
    
    def __str__(self):
        return "FILL: Symbol={}, Quantity={}, Direction={}, Time={}, Price={}".format(
                    self.symbol, self.quantity, self.direction, self.datetime, self.fill_cost)