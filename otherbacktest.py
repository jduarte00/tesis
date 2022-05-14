import bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ffn
from hcaa_implementation import hcaa_alocation


tickers = {'equity':['ITOT', 'IVV', 'IJH', 'IJR', 'IUSG', 'IUSV', 'IJK', 'IJJ', 'IJS', 'IJT', 'OEF', 'IWC'],
'bond':['AGG', 'LQD', 'GOVT', 'MBB', 'MUB', 'TIP', 'SHY', 'IEF', 'TLT', 'HYG', 'FLOT', 'CMBS'],
}

prices = bt.data.get(tickers['equity'] + tickers['bond'], clean_tickers=False)

class WeightHCAA(bt.Algo):
    """
    mi descripción equis
    """

    # el lookback es cuanta información pasada se quiere utilizar para 
    #calcular el nuevo portfolio. 
    #notar que para que corra todo correctamente utilizando este parámetro hay que 
    # verificar que para cuando corra por primera vez ya se tenga suficiente info histórica.
    def __init__(self, lag = pd.DateOffset(days = 0), lookback = pd.DateOffset(months=3)):
        super(WeightHCAA, self).__init__()
        self.lag = lag
        self.lookback = lookback
    
    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp['weights'] = {}
            return True
        if len(selected) == 1:
            target.temp['weights'] = {selected[0]: 1.0}
            return True
        
        #notar que en lag hay que especificar un lag, que funciona para especificar si 
        # se quiere utilizar info del día de "hoy" o no .

        # target.universe trae un dataframe que contiene todas las columnas pero en las filas
        #, que son las observaciones diarias, trae solo hasta target.now
        dataset = target.universe[selected].dropna()
        #TODO
        # determinar K
        k = 17
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights =hcaa_alocation(dataset.values, k)
        
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        target.temp["weights"] = new_weights
        return True

        #ejemplo de target.temp["weights"]
        # {'ITOT': 0.8, 'AGG': 0.2}

        # ejemplo de target.temp["selected"]
        # ['ITOT', 'IVV', 'IJH', 'IJR', 'IUSG', 'IUSV', 'IJK', 'IJJ', 'IJS', 'IJT', 'OEF', 'IWC', 'AGG', 'LQD', ...]
        


aggressive = bt.Strategy('aggressive', algos = [
    bt.algos.RunQuarterly(run_on_first_date=False),
    bt.algos.SelectAll(),
    WeightHCAA(),
    bt.algos.Rebalance()
])

backtest_aggressive = bt.Backtest(aggressive,prices)

report = bt.run(backtest_aggressive)
report.display()
fig =report.plot()
fig.figure.savefig('image.png')