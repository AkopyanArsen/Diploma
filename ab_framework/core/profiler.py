import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


class DataProfiler:
    """
    Анализатор статистических свойств данных.
    Вычисляет метрики, необходимые для работы Decision Engine.
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.target_col = config.get('target_metric')
        self.pre_col = config.get('pre_experiment_metric')
        self.categorical_cols = config.get('categorical_covariates', [])



    @staticmethod
    def _hill_estimator(x: np.ndarray, top_fraction: float = 0.10) -> float:
        """
        Оценка Хилла для индекса хвоста распределения.

        Используются только положительные значения из верхних ``top_fraction``
        доли выборки (по умолчанию топ-10%). Формула:

            alpha_hat = 1 / mean(log(x_i / x_min))   для x_i > x_min

        где x_min = квантиль (1 - top_fraction) выборки.

        Возвращает +inf, если данных недостаточно или вариации нет.
        При alpha < 2 → теоретическая дисперсия бесконечна (тяжёлый хвост
        Парето-типа); при alpha ≥ 2 дисперсия конечна.
        """
        x_pos = x[x > 0]
        if len(x_pos) < 10:
            return np.inf

        x_min = np.quantile(x_pos, 1.0 - top_fraction)
        tail_vals = x_pos[x_pos > x_min]

        if len(tail_vals) < 2 or x_min <= 0:
            return np.inf

        log_ratios = np.log(tail_vals / x_min)
        mean_log = np.mean(log_ratios)
        if mean_log <= 0:
            return np.inf

        return 1.0 / mean_log



    @staticmethod
    def _between_group_variance_fraction(
        target: np.ndarray,
        strata: np.ndarray,
    ) -> float:
        """
        Доля межгрупповой (between-group) дисперсии в общей дисперсии метрики.

        Формула разложения дисперсии:
            Var_total = Var_between + Var_within

            Var_between = sum_k [ w_k * (mu_k - mu_global)^2 ]

        где w_k — доля страты k в выборке.

        Возвращает значение от 0 до 1. Если > 0.10 → стратификация выгодна.
        """
        total_var = np.var(target, ddof=1)
        if total_var <= 0:
            return 0.0

        global_mean = np.mean(target)
        labels, counts = np.unique(strata, return_counts=True)
        weights = counts / counts.sum()

        between_var = 0.0
        for lbl, w in zip(labels, weights):
            mask = strata == lbl
            if mask.sum() < 1:
                continue
            mu_k = np.mean(target[mask])
            between_var += w * (mu_k - global_mean) ** 2

        return between_var / total_var



    def get_profile(self) -> dict:
        """
        Возвращает словарь с мета-информацией о датасете.
        """
        profile = {}

        target_data = self.df[self.target_col].values

        profile['sample_size'] = len(self.df)
        profile['mean'] = np.mean(target_data)
        profile['variance'] = np.var(target_data, ddof=1)

        profile['skewness'] = skew(target_data, bias=False)
        profile['kurtosis'] = kurtosis(target_data, bias=False)
        profile['has_heavy_tails'] = profile['kurtosis'] > 10.0

        # Hill estimator: alpha < 2 → теоретическая дисперсия бесконечна
        alpha = self._hill_estimator(target_data, top_fraction=0.10)
        profile['tail_index_alpha'] = float(alpha)
        profile['is_variance_infinite'] = alpha < 2.0

        if self.pre_col and self.pre_col in self.df.columns:
            pre_data = self.df[self.pre_col].values
            correlation_matrix = np.corrcoef(pre_data, target_data)
            rho = correlation_matrix[0, 1]
            profile['rho'] = rho
        
            profile['is_cuped_applicable'] = abs(rho) >= 0.25
        else:
            profile['rho'] = 0.0
            profile['is_cuped_applicable'] = False

        strat_fractions = {}
        for col in self.categorical_cols:
            if col in self.df.columns:
                frac = self._between_group_variance_fraction(
                    target_data, self.df[col].values
                )
                strat_fractions[col] = float(frac)

        profile['between_group_var_fractions'] = strat_fractions
        profile['is_stratification_beneficial'] = any(
            v > 0.10 for v in strat_fractions.values()
        ) if strat_fractions else False

        return profile