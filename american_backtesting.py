import contextlib
import io

import bt
import csestimator
import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rie_estimator
import scipy.spatial.distance as ssd
from hcaa_implementation import hcaa_alocation
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import calinski_harabasz_score
import matplotlib
import matplotlib.dates as mdates

#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
#plt.gcf().autofmt_xdate()


prices = pd.read_csv(
    "./sp_500_original_clean.csv",
    index_col="Date",
    parse_dates=True,
)

# treshold = 2.1
# treshold = 3                         
# treshold = 2.5
treshold = 3.7
backdays = prices.shape[1] * 2
periods = 30


def wrapper_function_cluster(X_matrix):
    return csestimator.get_shrinkage_est(X_matrix, alpha=0.5)


def get_optimal_k_eigen(corr_matrix, N, T):
    eigenvals = np.linalg.eigvals(corr_matrix)
    count = (eigenvals > 1 + 2 * np.sqrt(N / T) + N / T).sum()
    return count


def get_optimal_k_calenski(dataset, bottom_range, top_range, corr_function):
    corr_mat = corr_function(dataset.T)
    D_matrix = np.sqrt(2 * (1 - corr_mat))
    D_matrix = np.around(D_matrix, decimals=7)
    D_condensed = ssd.squareform(D_matrix)
    Z = linkage(D_condensed, "ward", optimal_ordering=True)
    indices = []
    for i in range(bottom_range, top_range):
        labels = fcluster(Z, i, criterion="maxclust")
        indices.append(calinski_harabasz_score(dataset.T, labels))
    # pd.Series(indices).plot()
    print(
        kneed.KneeLocator(
            range(bottom_range, top_range),
            indices,
            curve="convex",
            direction="decreasing",
        ).knee
    )
    return kneed.KneeLocator(
        range(bottom_range, top_range), indices, curve="convex", direction="decreasing"
    ).knee


def get_optimal_k_calenski_rie(dataset, bottom_range, top_range, corr_function):
    corr_mat = corr_function(dataset)
    print(corr_mat.shape)
    D_matrix = np.sqrt(2 * (1 - corr_mat))
    D_matrix = np.around(D_matrix, decimals=7)
    D_condensed = ssd.squareform(D_matrix)
    Z = linkage(D_condensed, "ward", optimal_ordering=True)
    indices = []
    for i in range(bottom_range, top_range):
        labels = fcluster(Z, i, criterion="maxclust")
        indices.append(calinski_harabasz_score(dataset.T, labels))
    # pd.Series(indices).plot()
    print(
        kneed.KneeLocator(
            range(bottom_range, top_range),
            indices,
            curve="convex",
            direction="decreasing",
        ).knee
    )
    return kneed.KneeLocator(
        range(bottom_range, top_range), indices, curve="convex", direction="decreasing"
    ).knee


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
        #k = get_optimal_k_eigen(rie_estimator.get_rie(returns), dataset.shape[0], dataset.shape[1])
        # usando calenski
        k = get_optimal_k_calenski_rie(returns, 2, 50, rie_estimator.get_rie)

        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights = hcaa_alocation(
            mat_X=returns,
            n_clusters=k,
            custom_corr=rie_estimator.get_rie,
            inverse_data=False,
        )
        # index, weights =hcaa_alocation(mat_X =dataset.values, n_clusters = k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        if "n_clusters" not in target.perm:
            target.perm["n_clusters"] = [[target.now, k]]
        else:
            target.perm["n_clusters"].append([target.now, k])
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
        # determinar K
        # k = get_optimal_k_eigen(wrapper_function_cluster(dataset), dataset.shape[0], dataset.shape[1])
        # usando calenski
        k = get_optimal_k_calenski_rie(returns, 2, 50, wrapper_function_cluster)
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights = hcaa_alocation(
            mat_X=returns.values,
            n_clusters=k,
            custom_corr=wrapper_function_cluster,
            inverse_data=False,
        )
        # index, weights =hcaa_alocation(mat_X =dataset.values, n_clusters = k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        if "n_clusters" not in target.perm:
            target.perm["n_clusters"] = [[target.now, k]]
        else:
            target.perm["n_clusters"].append([target.now, k])
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
        # k = get_optimal_k_eigen(np.corrcoef(dataset.values.T), dataset.shape[0], dataset.shape[1])
        k = get_optimal_k_calenski(returns, 2, 50, np.corrcoef)
        # llamar la funcion de HCAA mía sobre el dataset
        # regresar los pesos y los índices
        index, weights = hcaa_alocation(mat_X=returns.values, n_clusters=k)
        # con eso formar el dict y guardarlo en target.temp["weights"]
        new_weights = dict(zip(dataset.columns[index], weights))
        if "n_clusters" not in target.perm:
            target.perm["n_clusters"] = [[target.now, k]]
        else:
            target.perm["n_clusters"].append([target.now, k])
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
        bt.algos.RunEveryNPeriods(n=periods, offset=backdays),
        bt.algos.SelectAll(),
        WeightHCAA(),
        bt.algos.Rebalance(),
    ],
)

corr_testing = bt.Strategy(
    "corr_testing",
    algos=[
        bt.algos.RunEveryNPeriods(n=periods, offset=backdays),
        bt.algos.SelectAll(),
        WeightHCAAsimple(),
        bt.algos.Rebalance(),
    ],
)

clust_testing = bt.Strategy(
    "clustering_testing",
    algos=[
        bt.algos.RunEveryNPeriods(n=periods, offset=backdays),
        bt.algos.SelectAll(),
        WeightHCAAclustering(),
        bt.algos.Rebalance(),
    ],
)

equal_testing = bt.Strategy(
    "equal_testing",
    algos=[
        bt.algos.RunEveryNPeriods(n=periods, offset=backdays),
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
file_name = f"tec_calenski_back_{backdays}_periods_{periods}"
with contextlib.redirect_stdout(io.StringIO()) as f:
    report.display()
file_to_save = open("./results_american/" + file_name + ".txt", "w")
file_to_save.write(f.getvalue())
fig = report.plot()
fig.figure.savefig("./results_american/" + file_name + ".png")
report.prices.to_csv(f"./results_american/prices_{file_name}.csv")
report.prices.to_returns().to_csv(f"./results_american/returns_{file_name}.csv")
report.get_weights().to_csv(f"./results_american/weights_{file_name}.csv")

report.get_weights('rie_testing').to_csv(f"./results_american/weights_{file_name}_rie.csv")
report.get_weights('corr_testing').to_csv(f"./results_american/weights_{file_name}_corr.csv")
report.get_weights('clustering_testing').to_csv(f"./results_american/weights_{file_name}_clust.csv")
report.get_weights('equal_testing').to_csv(f"./results_american/weights_{file_name}_equal.csv")

pd.DataFrame(np.array(report.backtests["rie_testing"].strategy.perm["n_clusters"]), columns = ['date', 'n_clusters']).set_index('date').to_csv(f"./results_american/clusters_rie_{file_name}.csv")
pd.DataFrame(np.array(report.backtests["corr_testing"].strategy.perm["n_clusters"]), columns = ['date', 'n_clusters']).set_index('date').to_csv(f"./results_american/clusters_corr_{file_name}.csv")
pd.DataFrame(np.array(report.backtests["clustering_testing"].strategy.perm["n_clusters"]), columns = ['date', 'n_clusters']).set_index('date').to_csv(f"./results_american/clusters_clust_{file_name}.csv")