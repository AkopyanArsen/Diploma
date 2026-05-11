"""
Генераторы синтетических данных для бенчмарка методов ускорения A/B-тестов.

В этом модуле два слоя:

1. ``FintechDataGenerator`` — исторический генератор (используется в ноутбуках
   01 и 02). Сохранён для обратной совместимости.

2. Семейство ``Scenario``-классов — основа для бенчмарка:
   * :class:`EcommerceScenario`     — денежная метрика с pre-периодом, rho≈0.5
   * :class:`RatioScenario`         — ratio-метрика ARPU = revenue / n_sessions
   * :class:`LowCorrelationScenario`— денежная метрика с rho≈0.18
                                      (случай, где CUPED почти не помогает)
   * :class:`SegmentedScenario`     — данные с явными сегментами по устройству
                                      (большая between-group variance —
                                       выгодно стратифицировать)

Все сценарии имеют единый интерфейс::

    df = scenario.generate(n_users, effect_size, seed=...)

Корреляция pre/post в денежных сценариях управляется параметром ``target_rho``
через Cholesky-разложение в log-пространстве (формула обратного
log-normal-преобразования корреляции).
"""

from __future__ import annotations

import numpy as np
import pandas as pd



class FintechDataGenerator:
    """
    Генератор синтетических данных для симуляции A/B-тестов в финтехе.
    Моделирует метрики с тяжёлыми хвостами (Mixture: Log-Normal + Pareto)
    и позволяет контролировать корреляцию между pre- и post-периодами через
    параметр ``correlation_noise``.
    """

    def __init__(self, n_users=100000, whale_frac=0.05, effect_size=0.02, correlation_noise=0.2):
        self.n_users = n_users
        self.whale_frac = whale_frac
        self.effect_size = effect_size
        self.correlation_noise = correlation_noise

    def _generate_mixture_base(self, size):
        is_whale = np.random.binomial(1, self.whale_frac, size)
        regular_spends = np.random.lognormal(mean=6.0, sigma=0.8, size=size)
        pareto_xm = np.percentile(regular_spends, 95)
        pareto_alpha = 1.16
        whale_spends = (np.random.pareto(pareto_alpha, size) + 1) * pareto_xm
        return np.where(is_whale == 1, whale_spends, regular_spends)

    def generate_dataset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        user_ids = np.arange(1, self.n_users + 1)
        device_types = np.random.choice(
            ['iOS', 'Android', 'Web'],
            size=self.n_users,
            p=[0.4, 0.5, 0.1],
        )
        age_groups = np.random.choice(
            ['18-24', '25-34', '35-44', '45+'],
            size=self.n_users,
            p=[0.2, 0.4, 0.25, 0.15],
        )

        is_treated = np.random.binomial(1, 0.5, self.n_users)
        base_spending = self._generate_mixture_base(self.n_users)

        noise_pre = np.random.lognormal(mean=0, sigma=self.correlation_noise, size=self.n_users)
        noise_target = np.random.lognormal(mean=0, sigma=self.correlation_noise, size=self.n_users)

        pre_experiment_metric = base_spending * noise_pre
        target_metric_base = base_spending * noise_target

        treatment_multiplier = 1 + (self.effect_size * is_treated)
        target_metric = target_metric_base * treatment_multiplier

        df = pd.DataFrame({
            'user_id': user_ids,
            'device_type': device_types,
            'age_group': age_groups,
            'is_treated': is_treated,
            'pre_experiment_metric': np.round(pre_experiment_metric, 2),
            'target_metric': np.round(target_metric, 2),
        })
        return df



def _correlated_lognormals(n, mu, sigma, target_rho, rng):
    """
    Генерирует две лог-нормальные выборки одинаковой длины с заданной
    Пирсоновской корреляцией ``target_rho`` (в исходной, не лог-, шкале).

    Использует обратную формулу для корреляции lognormal-переменных:

        rho_LN = (exp(rho_N * s1*s2) - 1) / sqrt((exp(s1^2)-1)*(exp(s2^2)-1))

    отсюда при s1=s2=sigma::

        rho_N = ln(1 + rho_LN * (exp(sigma^2) - 1)) / sigma^2

    Затем нормальные пары получаются через разложение Холецкого.
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if abs(target_rho) > 1e-9:
        denom = np.exp(sigma ** 2) - 1.0
        argument = 1.0 + target_rho * denom
        if argument <= 0:
            raise ValueError(
                f"Невозможно достичь target_rho={target_rho} при sigma={sigma}: "
                "значение слишком отрицательное для lognormal-преобразования."
            )
        rho_n = float(np.log(argument) / (sigma ** 2))
        rho_n = float(np.clip(rho_n, -0.999, 0.999))
    else:
        rho_n = 0.0

    z = rng.standard_normal(size=(2, n))
    L = np.array([[1.0, 0.0], [rho_n, np.sqrt(max(0.0, 1.0 - rho_n ** 2))]])
    y = L @ z
    x_pre = np.exp(mu + sigma * y[0])
    x_post = np.exp(mu + sigma * y[1])
    return x_pre, x_post



class _BaseScenario:

    name: str = "base"

    def generate(self, n_users: int, effect_size: float, seed: int | None = None) -> pd.DataFrame:
        raise NotImplementedError



class EcommerceScenario(_BaseScenario):
    """
    Метрика — ``revenue`` (доход на пользователя за период наблюдения).
    Распределение — смесь lognormal (масса) + Pareto-хвост (киты), чтобы
    воспроизвести характерные тяжёлые хвосты в денежных метриках.

    Для пользователя есть pre-период ``pre_revenue`` с заданной (по умолчанию
    реалистичной) корреляцией ``target_rho ≈ 0.5`` к ``revenue``.

    Также возвращаются категориальные ковариаты ``device_type`` и ``age_group``,
    чтобы проверить пост-стратификацию.
    """

    name = "ecommerce"

    def __init__(self, whale_frac: float = 0.01, target_rho: float = 0.5,
                 mu: float = 6.0, sigma: float = 0.8, pareto_alpha: float = 2.5,
                 pareto_xm_quantile: float = 0.90):

        self.whale_frac = whale_frac
        self.target_rho = target_rho
        self.mu = mu
        self.sigma = sigma
        self.pareto_alpha = pareto_alpha
        self.pareto_xm_quantile = pareto_xm_quantile

    def generate(self, n_users, effect_size, seed=None):
        rng = np.random.default_rng(seed)

        pre_base, post_base = _correlated_lognormals(
            n_users, mu=self.mu, sigma=self.sigma,
            target_rho=self.target_rho, rng=rng,
        )

        pareto_xm = float(np.quantile(pre_base, self.pareto_xm_quantile))
        is_whale_pre = rng.binomial(1, self.whale_frac, n_users).astype(bool)
        is_whale_post = rng.binomial(1, self.whale_frac, n_users).astype(bool)
        whale_shock_pre = (rng.pareto(self.pareto_alpha, n_users) + 1.0) * pareto_xm
        whale_shock_post = (rng.pareto(self.pareto_alpha, n_users) + 1.0) * pareto_xm

        pre_revenue = pre_base + np.where(is_whale_pre, whale_shock_pre, 0.0)
        post_revenue_base = post_base + np.where(is_whale_post, whale_shock_post, 0.0)

        group = rng.binomial(1, 0.5, n_users)
        treatment_multiplier = 1.0 + effect_size * group
        revenue = post_revenue_base * treatment_multiplier

        device_type = rng.choice(
            ['iOS', 'Android', 'Web'], size=n_users, p=[0.4, 0.5, 0.1],
        )
        age_group = rng.choice(
            ['18-24', '25-34', '35-44', '45+'],
            size=n_users, p=[0.2, 0.4, 0.25, 0.15],
        )

        return pd.DataFrame({
            'user_id': np.arange(1, n_users + 1),
            'group': group.astype(int),
            'pre_revenue': np.round(pre_revenue, 2),
            'revenue': np.round(revenue, 2),
            'device_type': device_type,
            'age_group': age_group,
        })



class RatioScenario(_BaseScenario):
    """
    Ratio-метрика ARPU = revenue / n_sessions.

    На каждого пользователя генерируется число сессий ``n_sessions``
    (lognormal, округлённое до целого) и связанный с ним ``revenue``.
    Связь revenue ~ n_sessions реалистичная: больше сессий — больше дохода,
    плюс мультипликативный per-session шум.

    Effect_size трактуется как мультипликативный сдвиг ARPU в treatment-группе.
    """

    name = "ratio"

    def __init__(self, sessions_mu: float = 1.5, sessions_sigma: float = 0.6,
                 rps_mu: float = 2.5, rps_sigma: float = 0.5):
        self.sessions_mu = sessions_mu
        self.sessions_sigma = sessions_sigma
        self.rps_mu = rps_mu
        self.rps_sigma = rps_sigma

    def generate(self, n_users, effect_size, seed=None):
        rng = np.random.default_rng(seed)

        raw_sessions = rng.lognormal(mean=self.sessions_mu, sigma=self.sessions_sigma, size=n_users)
        n_sessions = np.maximum(1, np.round(raw_sessions)).astype(int)

        revenue_per_session = rng.lognormal(mean=self.rps_mu, sigma=self.rps_sigma, size=n_users)

        group = rng.binomial(1, 0.5, n_users)
        multiplier = 1.0 + effect_size * group

        revenue = n_sessions * revenue_per_session * multiplier

        device_type = rng.choice(
            ['iOS', 'Android', 'Web'], size=n_users, p=[0.4, 0.5, 0.1],
        )

        return pd.DataFrame({
            'user_id': np.arange(1, n_users + 1),
            'group': group.astype(int),
            'revenue': np.round(revenue, 2),
            'n_sessions': n_sessions,
            'device_type': device_type,
        })



class LowCorrelationScenario(EcommerceScenario):
    """
    Тот же e-commerce-сценарий, но с очень слабой корреляцией pre/post
    (итоговая rho ≈ 0.17, target_rho=0.18 с учётом whale-шоков).
    Используется чтобы продемонстрировать, когда CUPED неэффективен
    и лучше применять стратификацию.
    """

    name = "low_correlation"

    def __init__(self, whale_frac: float = 0.01, target_rho: float = 0.18,
                 mu: float = 6.0, sigma: float = 0.8, pareto_alpha: float = 2.5,
                 pareto_xm_quantile: float = 0.90):
        super().__init__(
            whale_frac=whale_frac, target_rho=target_rho,
            mu=mu, sigma=sigma, pareto_alpha=pareto_alpha,
            pareto_xm_quantile=pareto_xm_quantile,
        )




class SegmentedScenario(_BaseScenario):
    """
    Сегментированный сценарий — без pre-периода.

    Три сегмента (``device_type``: iOS, Android, Web) с заметно разными
    средними и дисперсиями дохода. Это ровно тот случай, где пост-стратификация
    должна быть наиболее выгодной (большая between-group variance).
    """

    name = "segmented"


    _segment_params = {
        'iOS':     {'p': 0.30, 'mu': 7.0, 'sigma': 0.5},   
        'Android': {'p': 0.50, 'mu': 5.5, 'sigma': 1.2}, 
        'Web':     {'p': 0.20, 'mu': 4.0, 'sigma': 1.5},   
    }

    def __init__(self, segment_params: dict | None = None):
        if segment_params is not None:
            self._segment_params = segment_params

    def generate(self, n_users, effect_size, seed=None):
        rng = np.random.default_rng(seed)

        devices = list(self._segment_params.keys())
        probs = [self._segment_params[d]['p'] for d in devices]
        device_type = rng.choice(devices, size=n_users, p=probs)

        revenue_base = np.empty(n_users)
        for dev, params in self._segment_params.items():
            mask = device_type == dev
            n_seg = int(mask.sum())
            if n_seg == 0:
                continue
            revenue_base[mask] = rng.lognormal(
                mean=params['mu'], sigma=params['sigma'], size=n_seg,
            )

        group = rng.binomial(1, 0.5, n_users)
        multiplier = 1.0 + effect_size * group
        revenue = revenue_base * multiplier

        return pd.DataFrame({
            'user_id': np.arange(1, n_users + 1),
            'group': group.astype(int),
            'revenue': np.round(revenue, 2),
            'device_type': device_type,
        })




def all_scenarios() -> dict[str, _BaseScenario]:
    """Реестр сценариев для использования в Benchmarker."""
    return {
        'ecommerce':       EcommerceScenario(),
        'ratio':           RatioScenario(),
        'low_correlation': LowCorrelationScenario(),
        'segmented':       SegmentedScenario(),
    }
