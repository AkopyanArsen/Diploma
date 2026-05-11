import pandas as pd


class DataContract:
    """
    Валидация и стандартизация входного датасета.

    Проверяет наличие обязательных колонок, удаляет строки с NaN в ключевых полях
    и нормализует метку группы к бинарному формату {0, 1} — результат всегда в колонке
    ``group_id``, независимо от исходного формата (строки, числа).
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.raw_df = df.copy()
        self.config = config
        self.target_col = config.get('target_metric')
        self.group_col = config.get('group_column')
        self.pre_col = config.get('pre_experiment_metric')
        self.covariates = config.get('covariates', [])

    def validate_and_transform(self) -> pd.DataFrame:
        """Возвращает очищенный DataFrame с колонкой ``group_id`` ∈ {0, 1}."""
        if not self.target_col or self.target_col not in self.raw_df.columns:
            raise ValueError(f"Целевая метрика '{self.target_col}' не найдена в датасете.")
        if not self.group_col or self.group_col not in self.raw_df.columns:
            raise ValueError(f"Колонка групп '{self.group_col}' не найдена в датасете.")

        cols_to_check = [self.target_col, self.group_col]
        if self.pre_col:
            cols_to_check.append(self.pre_col)

        initial_len = len(self.raw_df)
        self.raw_df = self.raw_df.dropna(subset=cols_to_check)
        dropped = initial_len - len(self.raw_df)
        if dropped > 0:
            print(f"[DataContract] Предупреждение: удалено {dropped} строк с NaN.")

        unique_groups = self.raw_df[self.group_col].unique()
        if len(unique_groups) != 2:
            raise ValueError(f"В колонке групп должно быть ровно 2 варианта. Найдено: {unique_groups}")

        if self.raw_df[self.group_col].dtype == 'object':
            group_mapping = {sorted(unique_groups)[0]: 0, sorted(unique_groups)[1]: 1}
            self.raw_df['group_id'] = self.raw_df[self.group_col].map(group_mapping)
        else:
            self.raw_df['group_id'] = self.raw_df[self.group_col]

        return self.raw_df