"""
Главный пайплайн A/B-тестирования.

Оркестрирует четыре шага:
  1. Валидация данных (DataContract)
  2. Профилирование (DataProfiler)
  3. Выбор метода (DecisionEngine)
  4. Применение метода и оценка (t-test / delta method)

Пример использования с EcommerceScenario::

    from ab_framework.data.generator import EcommerceScenario
    from ab_framework.pipeline.runner import ABTestRunner

    df = EcommerceScenario().generate(n_users=10_000, effect_size=0.05, seed=42)
    config = {
        'target_metric':          'revenue',
        'group_column':           'group',
        'pre_experiment_metric':  'pre_revenue',
        'categorical_covariates': ['device_type', 'age_group'],
        'metric_type':            'continuous',
    }
    results = ABTestRunner(df, config).run()
    print(results['is_significant'], results['p_value'])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from scipy.stats import ttest_ind

from ab_framework.data.contract import DataContract
from ab_framework.core.profiler import DataProfiler
from ab_framework.core.engine import DecisionEngine
from ab_framework.methods.cuped import StandardCUPED
from ab_framework.methods.stratification import PostStratification
from ab_framework.methods.transformations import Winsorizer, LogTransformer
from ab_framework.methods.delta import DeltaMethod


class ABTestRunner:
    """
    Главный пайплайн фреймворка.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датасет (один из Scenario.generate() или пользовательский).
    config : dict
        Конфигурация теста. Обязательные ключи:
          - ``target_metric``  — имя колонки с метрикой
          - ``group_column``   — имя колонки с группами (0/1 или 'control'/'treatment')
        Опциональные ключи:
          - ``pre_experiment_metric``  — ковариата для CUPED (имя колонки)
          - ``categorical_covariates`` — список колонок для стратификации
          - ``denominator_metric``     — знаменатель для ratio-метрики (delta method)
          - ``metric_type``            — 'continuous' (по умолчанию) или 'ratio'
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.raw_df = df
        self.config = config
        self.target_col       = config.get('target_metric')
        self.denominator_col  = config.get('denominator_metric')
        self.pre_col          = config.get('pre_experiment_metric')
        self.categorical_cols = config.get('categorical_covariates', [])



    def run(self) -> dict:
        """
        Запускает полный цикл анализа.

        Returns
        -------
        dict с ключами:
          ``p_value``, ``is_significant``, ``effect_estimate``,
          ``control_mean``, ``treatment_mean``,
          ``variance_before``, ``variance_after``,
          ``execution_plan`` (текстовое объяснение),
          ``profile`` (словарь DataProfiler),
          ``plan`` (словарь DecisionEngine),
          ``method_used`` (итоговый метод оценки как строка)
        """
        contract = DataContract(self.raw_df, self.config)
        df_clean = contract.validate_and_transform()
        group_col = 'group_id'   

        control_mask   = df_clean[group_col] == 0
        treatment_mask = df_clean[group_col] == 1

        if control_mask.sum() == 0 or treatment_mask.sum() == 0:
            raise ValueError("Одна из групп пустая после валидации.")

        profiler = DataProfiler(df_clean, self.config)
        profile  = profiler.get_profile()

        engine      = DecisionEngine(profile, self.config)
        plan        = engine.build_execution_plan()
        explanation = engine.explain_decision(plan)

        target          = df_clean[self.target_col].values.copy().astype(float)
        variance_before = float(np.var(target, ddof=1))

        if plan['transformation'] == 'log_transform':
            transformer = LogTransformer()
            target = transformer.transform(target)
        elif plan['transformation'] == 'winsorization':
            transformer = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
            target = transformer.fit(target).transform(target)

        strat_result = None   # заполняется только для post_stratification

        if plan['variance_reduction'] == 'standard_cuped' and self.pre_col:
            covariate = df_clean[self.pre_col].values.copy().astype(float)
            cuped  = StandardCUPED()
            target = cuped.fit_transform(target, covariate)

        elif plan['variance_reduction'] == 'post_stratification' and self.categorical_cols:
            strat_col   = self.categorical_cols[0]
            strata_data = df_clean[strat_col].values
            pop_weights = df_clean[strat_col].value_counts(normalize=True).to_dict()

            stratifier = PostStratification(population_weights=pop_weights)
            mean_t, var_t = stratifier.calculate_stratified_metrics(
                target[treatment_mask], strata_data[treatment_mask])
            mean_c, var_c = stratifier.calculate_stratified_metrics(
                target[control_mask],   strata_data[control_mask])
            strat_result = dict(mean_t=mean_t, var_t=var_t, mean_c=mean_c, var_c=var_c)

        variance_after = float(np.var(target, ddof=1))

        target_ctrl = target[control_mask]
        target_trt  = target[treatment_mask]

        results: dict = {
            'execution_plan':    explanation,
            'profile':           profile,
            'plan':              plan,
            'control_size':      int(control_mask.sum()),
            'treatment_size':    int(treatment_mask.sum()),
            'control_mean':      float(np.mean(target_ctrl)),
            'treatment_mean':    float(np.mean(target_trt)),
            'variance_before':   variance_before,
            'variance_after':    variance_after,
            'variance_reduction_pct': (1 - variance_after / variance_before) * 100
                                       if variance_before > 0 else 0.0,
        }

        if plan['evaluation_method'] == 't_test' and strat_result is not None:
            mean_t = strat_result['mean_t']
            mean_c = strat_result['mean_c']
            se = np.sqrt(strat_result['var_t'] + strat_result['var_c'])
            if se > 0:
                t_stat = (mean_t - mean_c) / se
                dof    = len(target_ctrl) + len(target_trt) - 2
                p_value = float(2 * sp_stats.t.sf(abs(t_stat), dof))
            else:
                p_value = 1.0
            results.update({
                'p_value':         p_value,
                'is_significant':  p_value < 0.05,
                'effect_estimate': float(mean_t - mean_c),
                'method_used':     'post_stratification + t-test',
            })

        elif plan['evaluation_method'] == 'delta_method':
            if self.denominator_col and self.denominator_col in df_clean.columns:
                den_ctrl = df_clean.loc[control_mask,   self.denominator_col].values
                den_trt  = df_clean.loc[treatment_mask, self.denominator_col].values
                dm = DeltaMethod()
                ratio_c, var_c = dm.calculate_ratio_stats(target_ctrl, den_ctrl)
                ratio_t, var_t = dm.calculate_ratio_stats(target_trt,  den_trt)
                se = np.sqrt(var_t + var_c)
                z_stat  = (ratio_t - ratio_c) / se if se > 0 else 0.0
                p_value = float(2 * sp_stats.norm.sf(abs(z_stat)))
                results.update({
                    'p_value':         p_value,
                    'is_significant':  p_value < 0.05,
                    'effect_estimate': float(ratio_t - ratio_c),
                    'control_ratio':   float(ratio_c),
                    'treatment_ratio': float(ratio_t),
                    'method_used':     'delta method',
                })
            else:
                import warnings
                warnings.warn(
                    "metric_type='ratio' но 'denominator_metric' не задан или не найден. "
                    "Используется Welch t-test — результат может быть некорректным.",
                    UserWarning, stacklevel=2,
                )
                t_stat, p_value = ttest_ind(target_trt, target_ctrl, equal_var=False)
                results.update({
                    'p_value':         float(p_value),
                    'is_significant':  float(p_value) < 0.05,
                    'effect_estimate': float(np.mean(target_trt) - np.mean(target_ctrl)),
                    'method_used':     'welch t-test (fallback, denominator missing)',
                })

        else:
            t_stat, p_value = ttest_ind(target_trt, target_ctrl, equal_var=False)
            vr_label = plan['variance_reduction'] or 'none'
            results.update({
                'p_value':         float(p_value),
                'is_significant':  float(p_value) < 0.05,
                'effect_estimate': float(np.mean(target_trt) - np.mean(target_ctrl)),
                'method_used':     f'{vr_label} + welch t-test',
            })

        return results
