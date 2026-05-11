import numpy as np


class DeltaMethod:
    """
    Корректная оценка дисперсии ratio-метрики вида E[Y] / E[X] (ratio-of-means).

    Наивный t-test по per-user значениям Y_i/X_i систематически занижает дисперсию:
    он игнорирует вклад изменчивости знаменателя X. Дельта-метод учитывает это
    через линейную аппроксимацию Тейлора.
    """

    @staticmethod
    def calculate_ratio_stats(numerator: np.ndarray, denominator: np.ndarray) -> tuple:
        """
        :param numerator:   числитель (выручка, клики и т.п.)
        :param denominator: знаменатель (сессии, показы и т.п.)
        :return: (ratio_of_means, variance_of_estimator)
        """
        mask = denominator > 0
        y = numerator[mask]
        x = denominator[mask]
        n = len(y)

        mu_y = np.mean(y)
        mu_x = np.mean(x)
        var_y = np.var(y, ddof=1)
        var_x = np.var(x, ddof=1)
        cov_yx = np.cov(y, x)[0, 1]

        ratio_metric = mu_y / mu_x


        delta_variance = (
            var_y / mu_x**2
            + var_x * mu_y**2 / mu_x**4
            - 2 * cov_yx * mu_y / mu_x**3
        )

        return ratio_metric, delta_variance / n