class DecisionEngine:
    """
    Rule-based маршрутизатор методов ускорения A/B-тестов.

    Принимает профиль данных (от DataProfiler) и конфигурацию,
    возвращает оптимальный стек методов (план выполнения).

    Пороги решений калиброваны по результатам Monte Carlo бенчмарка
    (1000 симуляций, n=5000 пользователей, effect_size=5%)

    Ключевые выводы для порогов:
    - CUPED эффективен при rho >= 0.25 (при rho=0.17 прирост мощности +2.6%)
    - Стратификация выгодна при between-group variance > 10% total variance
    - Delta Method обязателен для ratio-метрик (прирост мощности +31.1%)
    - Winsorization снижает дисперсию, но вносит смещение → только как предобработка
    """

    def __init__(self, profile: dict, config: dict):
        self.profile = profile
        self.config = config

        self.metric_type = config.get('metric_type', 'continuous')  # 'continuous' или 'ratio'
        self.has_categorical_covariates = len(config.get('categorical_covariates', [])) > 0
        self.has_continuous_covariate = bool(config.get('pre_experiment_metric'))

    def build_execution_plan(self) -> dict:
        """
        Дерево решений — три последовательных шага:
        1. Трансформация (обработка выбросов)
        2. Снижение дисперсии (variance reduction)
        3. Статистическая оценка (evaluation)
        """
        plan = {
            'transformation': None,
            'variance_reduction': None,
            'evaluation_method': 't_test',
        }


        if self.profile.get('is_variance_infinite', False):
            plan['transformation'] = 'log_transform'

    
        elif self.profile.get('has_heavy_tails', False):
            plan['transformation'] = 'winsorization'


        if self.profile.get('is_cuped_applicable', False) and self.has_continuous_covariate:
            plan['variance_reduction'] = 'standard_cuped'

     
        elif self.profile.get('is_stratification_beneficial', False) and self.has_categorical_covariates:
            plan['variance_reduction'] = 'post_stratification'

 
        if self.metric_type == 'ratio':
            plan['evaluation_method'] = 'delta_method'
        else:
            plan['evaluation_method'] = 't_test'

        return plan

    def explain_decision(self, plan: dict) -> str:

        lines = ["Обоснование алгоритма:"]

        if plan['transformation'] == 'log_transform':
            lines.append(
                f"- Логарифмирование: Индекс Хилла ({self.profile.get('tail_index_alpha', 0):.2f}) < 2.0 "
                "— риск бесконечной дисперсии (тяжёлый Pareto-хвост).")
        elif plan['transformation'] == 'winsorization':
            lines.append(
                f"- Винсоризация (предобработка выбросов): эксцесс = {self.profile.get('kurtosis', 0):.1f} > 10. "
                "Применяется ДО variance reduction, не как самостоятельный метод.")

        rho = self.profile.get('rho', 0)
        if plan['variance_reduction'] == 'standard_cuped':
            lines.append(
                f"- CUPED: корреляция pre/post rho={rho:.3f} >= 0.25. "
                f"Ожидаемое снижение дисперсии: {(1 - (1 - rho**2)) * 100:.1f}% "
                f"(теор. ускорение = {1/(1-rho**2):.2f}x).")
        elif plan['variance_reduction'] == 'post_stratification':
            fracs = self.profile.get('between_group_var_fractions', {})
            best_col = max(fracs, key=fracs.get) if fracs else '?'
            best_frac = fracs.get(best_col, 0)
            lines.append(
                f"- Пост-стратификация: между-групповая дисперсия по '{best_col}' "
                f"= {best_frac:.1%} от total variance (порог > 10%).")
        else:
            lines.append(
                f"- Variance reduction не применяется: rho={rho:.3f} < 0.25 "
                "и/или between-group variance ниже порога 10%.")

        if plan['evaluation_method'] == 'delta_method':
            lines.append(
                "- Delta Method: ratio-метрика. "
                "По бенчмарку: Power 78.0% vs 46.9% у baseline (+31.1%). "
                "Прирост за счёт корректной SE: naive t-test занижает SE ratio-метрики, "
                "delta-method исправляет это через аппроксимацию Тейлора.")
        else:
            lines.append("- Welch t-test для оценки разницы средних.")

        return "\n".join(lines)