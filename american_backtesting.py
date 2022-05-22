import bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ffn
from hcaa_implementation import hcaa_alocation
import rie_estimator
import csestimator


prices = pd.read_csv('/home/dum/Desktop/data/sp_500_original_clean.csv', index_col = "Date", parse_dates=True)

def wrapper_function_cluster(X_matrix):
    return csestimator.get_shrinkage_est(X_matrix, alpha = 1)


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
        # TODO 
        # Aquí el dataset contiene los precios, no los retornos, y los estimadores de la matriz de correlación deben ser
        # calculados sobre los retornos, por lo que hay que convertir aún a precios esta parte. 
        dataset = target.universe[selected].dropna()
        #returns = dataset.pct_change().iloc[1:]
        returns = (np.log(dataset) - np.log(dataset.shift(1))).iloc[1:]
        #TODO
        # determinar K
        k = 18
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights =hcaa_alocation(mat_X =returns, n_clusters = k, custom_corr=rie_estimator.get_rie, inverse_data= False)
        #index, weights =hcaa_alocation(mat_X =dataset.values, n_clusters = k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        target.temp["weights"] = new_weights
        return True

        #ejemplo de target.temp["weights"]
        # {'ITOT': 0.8, 'AGG': 0.2}

        # ejemplo de target.temp["selected"]
        # ['ITOT', 'IVV', 'IJH', 'IJR', 'IUSG', 'IUSV', 'IJK', 'IJJ', 'IJS', 'IJT', 'OEF', 'IWC', 'AGG', 'LQD', ...]

class WeightHCAAclustering(bt.Algo):
    """
    mi descripción equis
    """

    # el lookback es cuanta información pasada se quiere utilizar para 
    #calcular el nuevo portfolio. 
    #notar que para que corra todo correctamente utilizando este parámetro hay que 
    # verificar que para cuando corra por primera vez ya se tenga suficiente info histórica.
    def __init__(self, lag = pd.DateOffset(days = 0), lookback = pd.DateOffset(months=3)):
        super(WeightHCAAclustering, self).__init__()
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
        #returns = dataset.pct_change().iloc[1:]
        returns = (np.log(dataset) - np.log(dataset.shift(1))).iloc[1:]
        #TODO
        # determinar K
        k = 18
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights =hcaa_alocation(mat_X =returns.values, n_clusters = k, custom_corr=wrapper_function_cluster, inverse_data= False)
        #index, weights =hcaa_alocation(mat_X =dataset.values, n_clusters = k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        target.temp["weights"] = new_weights
        return True

        #ejemplo de target.temp["weights"]
        # {'ITOT': 0.8, 'AGG': 0.2}

        # ejemplo de target.temp["selected"]
        # ['ITOT', 'IVV', 'IJH', 'IJR', 'IUSG', 'IUSV', 'IJK', 'IJJ', 'IJS', 'IJT', 'OEF', 'IWC', 'AGG', 'LQD', ...]

class WeightHCAAsimple(bt.Algo):
    """
    mi descripción equis
    """

    # el lookback es cuanta información pasada se quiere utilizar para 
    #calcular el nuevo portfolio. 
    #notar que para que corra todo correctamente utilizando este parámetro hay que 
    # verificar que para cuando corra por primera vez ya se tenga suficiente info histórica.
    def __init__(self, lag = pd.DateOffset(days = 0), lookback = pd.DateOffset(months=3)):
        super(WeightHCAAsimple, self).__init__()
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
        #returns = dataset.pct_change().iloc[1:]
        returns = (np.log(dataset) - np.log(dataset.shift(1))).iloc[1:]
        #TODO
        # determinar K
        k = 18
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights =hcaa_alocation(mat_X =returns.values, n_clusters = k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        target.temp["weights"] = new_weights
        return True

        #ejemplo de target.temp["weights"]
        # {'ITOT': 0.8, 'AGG': 0.2}

        # ejemplo de target.temp["selected"]
        # ['ITOT', 'IVV', 'IJH', 'IJR', 'IUSG', 'IUSV', 'IJK', 'IJJ', 'IJS', 'IJT', 'OEF', 'IWC', 'AGG', 'LQD', ...]
        


rie_testing = bt.Strategy('rie_testing', algos = [
    bt.algos.RunMonthly(run_on_first_date=False),
    bt.algos.SelectAll(),
    WeightHCAA(),
    bt.algos.Rebalance()
])

corr_testing = bt.Strategy('corr_testing', algos = [
    bt.algos.RunMonthly(run_on_first_date=False),
    bt.algos.SelectAll(),
    WeightHCAAsimple(),
    bt.algos.Rebalance()
])

clust_testing = bt.Strategy('clustering_testing', algos = [
    bt.algos.RunMonthly(run_on_first_date=False),
    bt.algos.SelectAll(),
    WeightHCAAclustering(),
    bt.algos.Rebalance()
])

equal_testing = bt.Strategy('equal_testing', algos = [
    bt.algos.RunYearly(run_on_first_date=False),
    bt.algos.SelectAll(),
     bt.algos.WeighMeanVar(),
    bt.algos.Rebalance()
])

backtest_rie = bt.Backtest(rie_testing,prices)
backtest_corr = bt.Backtest(corr_testing,prices)
backtest_clust = bt.Backtest(clust_testing,prices)
backtest_equal= bt.Backtest(equal_testing,prices)

report = bt.run( backtest_rie, backtest_corr, backtest_clust)
#report = bt.run(backtest_corr)
report.display()
fig =report.plot()
fig.figure.savefig('image.png')