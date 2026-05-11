import numpy as np


class StandardCUPED:
    """
    Классический метод CUPED (Controlled-experiment Using Pre-Experiment Data).
    Снижает дисперсию целевой метрики за счёт линейной регрессии на исторические данные.
    """

    def __init__(self):
        self.theta = 0.0
        self.covariate_mean = 0.0

    def fit_transform(self, target: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        covariance = np.cov(target, covariate)[0, 1]
        variance_covariate = np.var(covariate, ddof=1)

        if variance_covariate == 0:
            return target

        self.theta = covariance / variance_covariate
        self.covariate_mean = np.mean(covariate)

        return target - self.theta * (covariate - self.covariate_mean)