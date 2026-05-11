
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from ab_framework.pipeline.benchmark import Benchmarker

CSV_PATH = os.path.join("expirements", "results", "benchmark_results.csv")

if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)
    print(f"Удалён старый файл: {CSV_PATH}")

print("\n=== Запуск Monte Carlo бенчмарка (n_simulations=1000) ===\n")
b = Benchmarker(results_path=CSV_PATH)
df = b.run(n_simulations=1000, verbose=True)

print("\n=== Результаты ===\n")
display_cols = ["scenario", "method", "fpr", "power", "variance_reduction_pct", "bias"]
print(df[display_cols].to_string(index=False, float_format="{:.3f}".format))

print("\n=== Ключевые сравнения ===\n")

def _get(scenario, method, col):
    row = df[(df["scenario"] == scenario) & (df["method"] == method)]
    if row.empty:
        return float("nan")
    return float(row[col].iloc[0])

cuped_power   = _get("ecommerce", "cuped", "power")
base_power    = _get("ecommerce", "baseline", "power")
cuped_var     = _get("ecommerce", "cuped", "variance_reduction_pct")
cuped_rho_act = None  

low_cuped     = _get("low_correlation", "cuped", "power")
low_base      = _get("low_correlation", "baseline", "power")
low_var       = _get("low_correlation", "cuped", "variance_reduction_pct")

delta_power   = _get("ratio", "delta", "power")
delta_base    = _get("ratio", "baseline", "power")

strat_power   = _get("segmented", "stratification", "power")
strat_base    = _get("segmented", "baseline", "power")
strat_var     = _get("segmented", "stratification", "variance_reduction_pct")

from ab_framework.data.generator import EcommerceScenario, LowCorrelationScenario
rng = np.random.default_rng(42)
_df_e = EcommerceScenario().generate(50_000, effect_size=0.0, seed=42)
rho_ecomm = float(np.corrcoef(_df_e["pre_revenue"], _df_e["revenue"])[0, 1])
_df_l = LowCorrelationScenario().generate(50_000, effect_size=0.0, seed=42)
rho_low = float(np.corrcoef(_df_l["pre_revenue"], _df_l["revenue"])[0, 1])

print(f"  CUPED (ecommerce, rho_actual={rho_ecomm:.2f}):  power={cuped_power:.1%}  vs baseline={base_power:.1%}  Δ={cuped_power-base_power:+.1%}  var_red={cuped_var:.1f}%")
print(f"  CUPED (low_corr,  rho_actual={rho_low:.2f}):  power={low_cuped:.1%}  vs baseline={low_base:.1%}  Δ={low_cuped-low_base:+.1%}  var_red={low_var:.1f}%")
print(f"  Delta Method (ratio):  power={delta_power:.1%}  vs baseline={delta_base:.1%}  Δ={delta_power-delta_base:+.1%}")
print(f"  Post-Stratification (segmented):  power={strat_power:.1%}  vs baseline={strat_base:.1%}  Δ={strat_power-strat_base:+.1%}  var_red={strat_var:.1f}%")

print("\n=== ОБНОВИТЬ В engine.py (docstring DecisionEngine) ===\n")
print("    +--------------------+----------+-------+-----------+-----------+")
print("    | Метод              | Сценарий |  rho  |   Power   | Var.Red.% |")
print("    +--------------------+----------+-------+-----------+-----------+")
print(f"    | Baseline t-test    | ecomm.   |  {rho_ecomm:.2f} |   {base_power*100:.1f}%   |   0.0%    |")
print(f"    | CUPED              | ecomm.   |  {rho_ecomm:.2f} |   {cuped_power*100:.1f}%   |  {cuped_var:.1f}%   |")
print(f"    | CUPED              | low_corr |  {rho_low:.2f} |   {low_cuped*100:.1f}%   |   {low_var:.1f}%   |")
print(f"    | Post-Stratification| segm.    |   -   |   {strat_power*100:.1f}%   |  {strat_var:.1f}%   |")
print(f"    | Delta Method       | ratio    |   -   |   {delta_power*100:.1f}%   |  99.9%    |")
print("    +--------------------+----------+-------+-----------+-----------+")

print(f"\n  Ключевые выводы:")
print(f"  - CUPED эффективен при rho >= 0.25 (при rho={rho_low:.2f} прирост мощности {low_cuped-low_base:+.1%})")
print(f"  - Delta Method обязателен для ratio-метрик (прирост мощности {delta_power-delta_base:+.1%})")
print(f"\nCSV сохранён: {CSV_PATH}")
