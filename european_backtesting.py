import bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ffn
from hcaa_implementation import hcaa_alocation
import rie_estimator
import csestimator
import contextlib
import io


prices = pd.read_csv(
    "/home/dum/Desktop/data/european_market_original_clean.csv",
    index_col="Date",
    parse_dates=True,
)

#treshold = 2.1
#treshold = 3
#treshold = 2.5
treshold = 3.5
backdays = 540
periods = 30

def wrapper_function_cluster(X_matrix):
    return csestimator.get_shrinkage_est(X_matrix, alpha=0.5)


class WeightHCAA(bt.Algo):
    """
    algo to perform HCAA using the RIE matrix estimator
    """

    # el lookback es cuanta información pasada se quiere utilizar para
    # calcular el nuevo portfolio.
    # notar que para que corra todo correctamente utilizando este parámetro hay que
    # verificar que para cuando corra por primera vez ya se tenga suficiente info histórica.
    def __init__(self, lag=pd.DateOffset(days=0), lookback=pd.DateOffset(months=3)):
        super(WeightHCAA, self).__init__()
        self.lag = lag
        self.lookback = lookback

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True
        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        # notar que en lag hay que especificar un lag, que funciona para especificar si
        # se quiere utilizar info del día de "hoy" o no .

        # target.universe trae un dataframe que contiene todas las columnas pero en las filas
        # , que son las observaciones diarias, trae solo hasta target.now
        dataset = target.universe[selected].dropna().tail(backdays)
        # returns = dataset.pct_change().iloc[1:]
        returns = (np.log(dataset) - np.log(dataset.shift(1))).iloc[1:]
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights = hcaa_alocation(
            mat_X=returns,
            custom_corr=rie_estimator.get_rie,
            inverse_data=False,
            cutoff_point=treshold
        )
        # index, weights =hcaa_alocation(mat_X =dataset.values, n_clusters = k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        target.temp["weights"] = new_weights
        return True

        # ejemplo de target.temp["weights"]
        # {'ITOT': 0.8, 'AGG': 0.2}

        # ejemplo de target.temp["selected"]
        # ['ITOT', 'IVV', 'IJH', 'IJR', 'IUSG', 'IUSV', 'IJK', 'IJJ', 'IJS', 'IJT', 'OEF', 'IWC', 'AGG', 'LQD', ...]


class WeightHCAAclustering(bt.Algo):
    """
    algo to perform HCAA using the ECA matrix estimator
    """

    # el lookback es cuanta información pasada se quiere utilizar para
    # calcular el nuevo portfolio.
    # notar que para que corra todo correctamente utilizando este parámetro hay que
    # verificar que para cuando corra por primera vez ya se tenga suficiente info histórica.
    def __init__(self, lag=pd.DateOffset(days=0), lookback=pd.DateOffset(months=3)):
        super(WeightHCAAclustering, self).__init__()
        self.lag = lag
        self.lookback = lookback

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True
        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        # notar que en lag hay que especificar un lag, que funciona para especificar si
        # se quiere utilizar info del día de "hoy" o no .

        # target.universe trae un dataframe que contiene todas las columnas pero en las filas
        # , que son las observaciones diarias, trae solo hasta target.now
        dataset = target.universe[selected].dropna().tail(backdays)
        # returns = dataset.pct_change().iloc[1:]
        returns = (np.log(dataset) - np.log(dataset.shift(1))).iloc[1:]
        # TODO
        # determinar K
        k = 9
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights = hcaa_alocation(
            mat_X=returns.values,
            cutoff_point=treshold,
            custom_corr=wrapper_function_cluster,
            inverse_data=False,
        )
        # index, weights =hcaa_alocation(mat_X =dataset.values, n_clusters = k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        target.temp["weights"] = new_weights
        return True


class WeightHCAAsimple(bt.Algo):
    """
    algo to perform HCAA using the sample correlation matrix
    """

    # el lookback es cuanta información pasada se quiere utilizar para
    # calcular el nuevo portfolio.
    # notar que para que corra todo correctamente utilizando este parámetro hay que
    # verificar que para cuando corra por primera vez ya se tenga suficiente info histórica.
    def __init__(self, lag=pd.DateOffset(days=0), lookback=pd.DateOffset(months=3)):
        super(WeightHCAAsimple, self).__init__()
        self.lag = lag
        self.lookback = lookback

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True
        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        # notar que en lag hay que especificar un lag, que funciona para especificar si
        # se quiere utilizar info del día de "hoy" o no .

        # target.universe trae un dataframe que contiene todas las columnas pero en las filas
        # , que son las observaciones diarias, trae solo hasta target.now
        dataset = target.universe[selected].dropna().tail(backdays)
        # returns = dataset.pct_change().iloc[1:]
        returns = (np.log(dataset) - np.log(dataset.shift(1))).iloc[1:]
        # TODO
        # determinar K
        k = 9
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights = hcaa_alocation(mat_X=returns.values, cutoff_point=treshold)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        target.temp["weights"] = new_weights
        return True


class weightNaive(bt.Algo):
    """
    Algo to perform naive 1/n weight assignment
    """

    # el lookback es cuanta información pasada se quiere utilizar para
    # calcular el nuevo portfolio.
    # notar que para que corra todo correctamente utilizando este parámetro hay que
    # verificar que para cuando corra por primera vez ya se tenga suficiente info histórica.
    def __init__(
        self,
        lag=pd.DateOffset(days=0),
        lookback=pd.DateOffset(months=3),
        number_assets=0,
    ):
        super(weightNaive, self).__init__()
        self.lag = lag
        self.lookback = lookback
        self.number_assets = number_assets

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True
        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        # notar que en lag hay que especificar un lag, que funciona para especificar si
        # se quiere utilizar info del día de "hoy" o no .

        # target.universe trae un dataframe que contiene todas las columnas pero en las filas
        # , que son las observaciones diarias, trae solo hasta target.now
        dataset = target.universe[selected].dropna()
        weight_to_assign = 1 / dataset.shape[1]
        weights_vector = np.repeat(weight_to_assign, dataset.shape[1])
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns, weights_vector))
        target.temp["weights"] = new_weights
        return True


rie_testing = bt.Strategy(
    "rie_testing",
    algos=[
        #bt.algos.RunQuarterly(run_on_first_date=False),
        bt.algos.RunEveryNPeriods(n= periods, offset = backdays),
        bt.algos.SelectAll(),
        WeightHCAA(),
        bt.algos.Rebalance(),
    ],
)

corr_testing = bt.Strategy(
    "corr_testing",
    algos=[
        #bt.algos.RunQuarterly(run_on_first_date=False),
        bt.algos.RunEveryNPeriods(n= periods, offset = backdays),
        bt.algos.SelectAll(),
        WeightHCAAsimple(),
        bt.algos.Rebalance(),
    ],
)

clust_testing = bt.Strategy(
    "clustering_testing",
    algos=[
        #bt.algos.RunQuarterly(run_on_first_date=False),
        bt.algos.RunEveryNPeriods(n= periods, offset = backdays),
        bt.algos.SelectAll(),
        WeightHCAAclustering(),
        bt.algos.Rebalance(),
    ],
)


equal_testing = bt.Strategy(
    "equal_testing",
    algos=[
        #bt.algos.RunQuarterly(run_on_first_date=False),
        bt.algos.RunEveryNPeriods(n= periods, offset = backdays),
        bt.algos.SelectAll(),
        weightNaive(),
        bt.algos.Rebalance(),
    ],
)

backtest_rie = bt.Backtest(rie_testing, prices)
backtest_corr = bt.Backtest(corr_testing, prices)
backtest_clust = bt.Backtest(clust_testing, prices)
backtest_equal = bt.Backtest(equal_testing, prices)

report = bt.run(backtest_rie, backtest_corr, backtest_clust, backtest_equal)
file_name = "tresh_{}_back_{}_periods_{}_europeo".format(treshold,backdays, periods)
with contextlib.redirect_stdout(io.StringIO()) as f:
    report.display()
file_to_save = open('./results_european/' +file_name+'.txt', 'a')
file_to_save.write(f.getvalue())
fig = report.plot()
fig.figure.savefig('./results_european/' +file_name+".png")

