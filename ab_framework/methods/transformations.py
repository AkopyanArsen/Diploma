import numpy as np


class Winsorizer:
    """
    Ограничение экстремальных значений по заданным перцентилям.
    Снижает дисперсию за счёт «обрезки» хвостов, но вносит смещение в оценку среднего.
    Применяется как предобработка перед variance reduction, а не самостоятельный метод.
    """

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, x: np.ndarray):
        self.lower_bound = np.percentile(x, self.lower_quantile * 100)
        self.upper_bound = np.percentile(x, self.upper_quantile * 100)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower_bound, self.upper_bound)


class LogTransformer:
    """
    Логарифмическое преобразование.
    Применяется при alpha < 2 по оценке Хилла — тяжёлый Pareto-хвост с потенциально
    бесконечной теоретической дисперсией.
    """

    def transform(self, x: np.ndarray) -> np.ndarray:
        if np.any(x < 0):
            x = x - np.min(x)
        return np.log1p(x)