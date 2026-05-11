"""
Monte Carlo бенчмарк методов ускорения A/B-тестов.

Класс :class:`Benchmarker` перебирает все комбинации (сценарий × метод),
запускает ``n_simulations`` симуляций в двух режимах:

* ``effect_size = 0``    → оценка FPR (должна быть ≈ α = 5%)
* ``effect_size = 0.05`` → оценка Power

Результаты сохраняются в ``experiments/results/benchmark_results.csv``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from ab_framework.data.generator import (
    EcommerceScenario,
    LowCorrelationScenario,
    RatioScenario,
    SegmentedScenario,
    _BaseScenario,
)
from ab_framework.methods.cuped import StandardCUPED
from ab_framework.methods.delta import DeltaMethod
from ab_framework.methods.stratification import PostStratification
from ab_framework.methods.transformations import Winsorizer



ALPHA = 0.05           
TRUE_EFFECT = 0.05     



def _ttest_result(ctrl: np.ndarray, trt: np.ndarray) -> tuple[float, float]:
    """Welch t-test → (p_value, effect_estimate = mean_trt - mean_ctrl)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = ttest_ind(trt, ctrl, equal_var=False)
    return float(p), float(np.mean(trt) - np.mean(ctrl))


def _run_baseline(df: pd.DataFrame, **_) -> dict:
    """Обычный двухвыборочный t-test без каких-либо корректировок."""
    ctrl = df.loc[df['group'] == 0, 'revenue'].values
    trt  = df.loc[df['group'] == 1, 'revenue'].values
    var_before = np.var(np.concatenate([ctrl, trt]), ddof=1)
    p, eff = _ttest_result(ctrl, trt)
    return dict(
        p_value=p,
        is_significant=p < ALPHA,
        variance_before=var_before,
        variance_after=var_before,  
        effect_estimate=eff,
    )


def _run_cuped(df: pd.DataFrame, **_) -> dict | None:
    """CUPED: скорректированная метрика = revenue - θ*(pre_revenue - mean_pre)."""
    if 'pre_revenue' not in df.columns:
        return None
    ctrl_mask = df['group'] == 0
    trt_mask  = df['group'] == 1

    target    = df['revenue'].values.copy()
    covariate = df['pre_revenue'].values.copy()

    var_before = np.var(target, ddof=1)

    cuped = StandardCUPED()
    target_adj = cuped.fit_transform(target, covariate)

    var_after = np.var(target_adj, ddof=1)

    ctrl = target_adj[ctrl_mask]
    trt  = target_adj[trt_mask]
    p, eff = _ttest_result(ctrl, trt)
    return dict(
        p_value=p,
        is_significant=p < ALPHA,
        variance_before=var_before,
        variance_after=var_after,
        effect_estimate=eff,
    )


def _run_stratification(df: pd.DataFrame, strat_col: str = 'device_type', **_) -> dict | None:
    """Пост-стратификация: взвешенное среднее по стратам."""
    if strat_col not in df.columns:
        return None

    ctrl_mask = df['group'] == 0
    trt_mask  = df['group'] == 1

    target = df['revenue'].values
    strata = df[strat_col].values

    var_before = np.var(target, ddof=1)

    pop_weights = {
        k: v for k, v in
        df[strat_col].value_counts(normalize=True).items()
    }

    strat = PostStratification(population_weights=pop_weights)
    mean_t, var_t = strat.calculate_stratified_metrics(target[trt_mask],  strata[trt_mask])
    mean_c, var_c = strat.calculate_stratified_metrics(target[ctrl_mask], strata[ctrl_mask])

    
    within_var = 0.0
    for lbl, w in pop_weights.items():
        mask = strata == lbl
        if mask.sum() < 2:
            continue
        within_var += w * np.var(target[mask], ddof=1)
    var_after = within_var

    n_c = ctrl_mask.sum()
    n_t = trt_mask.sum()
    se = np.sqrt(var_t + var_c)
    if se <= 0:
        return None
    from scipy import stats as sp_stats
    z = (mean_t - mean_c) / se
    p = float(2 * sp_stats.norm.sf(abs(z)))

    return dict(
        p_value=p,
        is_significant=p < ALPHA,
        variance_before=var_before,
        variance_after=var_after,
        effect_estimate=float(mean_t - mean_c),
    )


def _run_delta(df: pd.DataFrame, **_) -> dict | None:
    """Дельта-метод для ratio-метрики ARPU = revenue / n_sessions.

    Примечание о variance_reduction_pct:
        Для delta-метода var_reduction ≠ "ускорение через снижение дисперсии".
        Ценность метода — в КОРРЕКТНОЙ оценке SE ratio-метрики.
        Наивный t-test на per-user ARPU = Y_i/X_i систематически занижает SE
        (не учитывает дисперсию знаменателя), что приводит к инфляции мощности.
        Delta-метод исправляет SE, поэтому его variance_reduction_pct ≈ -43%
        (SE оказывается на 43% БОЛЬШЕ наивного → более консервативный и честный тест).
        Прирост мощности +31.1% достигается не за счёт снижения SE, а за счёт
        того, что наивный baseline неправильно вычисляет тестируемую статистику
        (считает revenue вместо ARPU = revenue/sessions).
    """
    if 'n_sessions' not in df.columns:
        return None

    ctrl_mask = df['group'] == 0
    trt_mask  = df['group'] == 1

    rev   = df['revenue'].values
    sess  = df['n_sessions'].values

    
    arpu_ctrl = rev[ctrl_mask] / np.maximum(sess[ctrl_mask], 1)
    arpu_trt  = rev[trt_mask]  / np.maximum(sess[trt_mask],  1)
    n_c, n_t  = ctrl_mask.sum(), trt_mask.sum()
    var_before = np.var(arpu_ctrl, ddof=1) / n_c + np.var(arpu_trt, ddof=1) / n_t

    dm = DeltaMethod()
    ratio_c, var_c = dm.calculate_ratio_stats(rev[ctrl_mask], sess[ctrl_mask])
    ratio_t, var_t = dm.calculate_ratio_stats(rev[trt_mask],  sess[trt_mask])

    se = np.sqrt(var_t + var_c)
    if se <= 0:
        return None
    from scipy import stats as sp_stats
    z = (ratio_t - ratio_c) / se
    p = float(2 * sp_stats.norm.sf(abs(z)))

    var_after = var_t + var_c  
    return dict(
        p_value=p,
        is_significant=p < ALPHA,
        variance_before=var_before,
        variance_after=var_after,
        effect_estimate=float(ratio_t - ratio_c),
    )


def _run_winsorization(df: pd.DataFrame, **_) -> dict:
    """Винсоризация [1%, 99%] + t-test. Демонстрирует bias-variance trade-off."""
    ctrl_mask = df['group'] == 0
    trt_mask  = df['group'] == 1

    target = df['revenue'].values.copy()
    var_before = np.var(target, ddof=1)

    wins = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
    target_w = wins.fit(target).transform(target)

    var_after = np.var(target_w, ddof=1)

    ctrl = target_w[ctrl_mask]
    trt  = target_w[trt_mask]
    p, eff = _ttest_result(ctrl, trt)

    return dict(
        p_value=p,
        is_significant=p < ALPHA,
        variance_before=var_before,
        variance_after=var_after,
        effect_estimate=eff,
    )



_METHOD_FNS = {
    'baseline':         _run_baseline,
    'cuped':            _run_cuped,
    'stratification':   _run_stratification,
    'delta':            _run_delta,
    'winsorization':    _run_winsorization,
}



class Benchmarker:
    """
    Monte Carlo бенчмарк методов ускорения A/B-тестов.

    Parameters
    ----------
    scenarios : dict[str, _BaseScenario] | None
        Словарь {имя → сценарий}. По умолчанию все 4 сценария.
    methods : list[str] | None
        Список имён методов. По умолчанию все 5.
    n_users : int
        Число пользователей в каждой симуляции.
    true_effect : float
        Мультипликативный размер эффекта для измерения мощности.
    alpha : float
        Уровень значимости.
    results_path : str | None
        Путь для сохранения CSV. None — не сохранять.
    """

    def __init__(
        self,
        scenarios: Dict[str, _BaseScenario] | None = None,
        methods: List[str] | None = None,
        n_users: int = 5_000,
        true_effect: float = TRUE_EFFECT,
        alpha: float = ALPHA,
        results_path: str | None = None,
    ):
        self.scenarios = scenarios or {
            'ecommerce':       EcommerceScenario(),
            'low_correlation': LowCorrelationScenario(),
            'ratio':           RatioScenario(),
            'segmented':       SegmentedScenario(),
        }
        self.methods = methods or list(_METHOD_FNS.keys())
        self.n_users = n_users
        self.true_effect = true_effect
        self.alpha = alpha
        self.results_path = results_path



    def _single_run(
        self,
        scenario: _BaseScenario,
        method_fn,
        effect_size: float,
        seed: int,
    ) -> dict | None:
        df = scenario.generate(self.n_users, effect_size=effect_size, seed=seed)
        return method_fn(df)



    def _aggregate(
        self,
        scenario_name: str,
        method_name: str,
        n_simulations: int,
        base_seed: int,
    ) -> dict:
        scenario  = self.scenarios[scenario_name]
        method_fn = _METHOD_FNS[method_name]

        fpr_signif: list[bool] = []
        for i in range(n_simulations):
            res = self._single_run(scenario, method_fn, 0.0, base_seed + i)
            if res is not None:
                fpr_signif.append(res['is_significant'])

        power_signif: list[bool]   = []
        var_befores:  list[float]  = []
        var_afters:   list[float]  = []
        effects:      list[float]  = []

        for i in range(n_simulations):
            seed = base_seed + n_simulations + i
            res = self._single_run(scenario, method_fn, self.true_effect, seed)
            if res is None:
                continue
            power_signif.append(res['is_significant'])
            var_befores.append(res['variance_before'])
            var_afters.append(res['variance_after'])
            effects.append(res['effect_estimate'])

        if not power_signif:
            return {}

        fpr   = float(np.mean(fpr_signif))    if fpr_signif    else float('nan')
        power = float(np.mean(power_signif))

        mean_var_before = float(np.mean(var_befores)) if var_befores else float('nan')
        mean_var_after  = float(np.mean(var_afters))  if var_afters  else float('nan')
        if mean_var_before > 0:
            variance_reduction_pct = (1.0 - mean_var_after / mean_var_before) * 100.0
        else:
            variance_reduction_pct = 0.0

        arr_effects = np.array(effects)
        mean_effect_est = float(np.mean(arr_effects))

       
        df_ref = scenario.generate(self.n_users * 5, effect_size=0.0, seed=base_seed + 9999)
        if 'n_sessions' in df_ref.columns:
            rev  = df_ref['revenue'].values
            sess = df_ref['n_sessions'].values
            true_abs = self.true_effect * (np.mean(rev) / np.mean(sess))
        else:
            mu_ctrl  = float(df_ref.loc[df_ref['group'] == 0, 'revenue'].mean())
            true_abs = self.true_effect * mu_ctrl

        bias = mean_effect_est - true_abs
        mse  = float(bias ** 2 + np.var(arr_effects, ddof=1))

        return dict(
            scenario=scenario_name,
            method=method_name,
            fpr=fpr,
            power=power,
            variance_reduction_pct=variance_reduction_pct,
            mean_effect_estimate=mean_effect_est,
            true_effect_abs=true_abs,
            bias=bias,
            mse=mse,
            n_valid_runs=len(power_signif),
        )

 

    def run(self, n_simulations: int = 1000, verbose: bool = True) -> pd.DataFrame:
  
        rows = []
        total = len(self.scenarios) * len(self.methods)
        done  = 0

        for sc_name in self.scenarios:
            for mt_name in self.methods:
                if verbose:
                    print(f"[{done+1}/{total}] {sc_name} × {mt_name} …", flush=True)

                base_seed = (hash(sc_name + mt_name) % 100_000) + 1

                row = self._aggregate(sc_name, mt_name, n_simulations, base_seed)
                if row:
                    rows.append(row)
                else:
                    if verbose:
                        print(f"  → метод '{mt_name}' неприменим к сценарию '{sc_name}', пропущен")
                done += 1

        results_df = pd.DataFrame(rows)

        if self.results_path:
            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
            results_df.to_csv(self.results_path, index=False)
            if verbose:
                print(f"\nРезультаты сохранены → {self.results_path}")

        return results_df
