import numpy as np
import pandas as pd


class PostStratification:
    """
    Пост-стратификация по категориальным ковариатам.

    Взвешивает оценки по стратам согласно популяционным долям, устраняя случайный
    дисбаланс групп по категориям и снижая дисперсию оценки среднего.
    """

    def __init__(self, population_weights: dict):
        """
        :param population_weights: истинные доли страт, например {'iOS': 0.3, 'Android': 0.7}
        """
        assert abs(sum(population_weights.values()) - 1.0) < 1e-5, "Веса страт должны давать 1.0"
        self.population_weights = population_weights

    def calculate_stratified_metrics(self, target: np.ndarray, strata: np.ndarray) -> tuple:
        """Возвращает (стратифицированное среднее, дисперсия среднего)."""
        df = pd.DataFrame({'target': target, 'stratum': strata})

        stratified_mean = 0.0
        stratified_var_of_mean = 0.0

        stats = df.groupby('stratum')['target'].agg(['mean', 'var', 'count'])

        for strat_val, row in stats.iterrows():
            weight = self.population_weights.get(strat_val)
            if weight is None:
                continue

            strat_var = row['var'] if not np.isnan(row['var']) else 0.0
            n = row['count']

            stratified_mean += weight * row['mean']
            if n > 0:
                stratified_var_of_mean += (weight ** 2) * (strat_var / n)

        return stratified_mean, stratified_var_of_mean