import bt

data = bt.get('aapl,msft,c,gs,ge', start='2010-01-01')

# data returns a data frame of n observations and t columns (one per stock)

# se obtiene la media movil tomando las primeras 50 observaciones y sacando el promedio simple, por eso 
# se tienen 49 datos faltantes al inicio
sma = data.rolling(50).mean()

# merged solo unes los datasets en el eje vertical, es decir, aumenta el número de columnas. 
merged = bt.merge(data, sma)

class SelectWhere(bt.Algo):

    """
    Selects securities based on an indicator DataFrame.

    Selects securities where the value is True on the current date (target.now).

    Args:
        * signal (DataFrame): DataFrame containing the signal (boolean DataFrame)

    Sets:
        * selected

    """
    def __init__(self, signal):
        self.signal = signal

    def __call__(self, target):
        # get signal on target.now
        if target.now in self.signal.index:
            sig = self.signal.loc[target.now]

            # get indices where true as list
            selected = list(sig.index[sig])

            # save in temp - this will be used by the weighing algo
            target.temp['selected'] = selected

        # return True because we want to keep on moving down the stack
        return True

# first we create the Strategy
s = bt.Strategy('above50sma', [SelectWhere(data > sma),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])

# now we create the Backtest
t = bt.Backtest(s, data)

# and let's run it!
res = bt.run(t)
print(sma)