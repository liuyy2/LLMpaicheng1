import pandas as pd
tuning = pd.read_csv('results_V2.5/baseline/baseline_216_1/tuning_results.csv')
tuning_sorted = tuning.sort_values('avg_drift')
print("freeze   eps   avg_drift   avg_delay   score   forced_replan")
for _, row in tuning_sorted.iterrows():
    fh = row['freeze_horizon_slots']
    es = row['epsilon_solver']
    ad = row['avg_drift']
    dl = row['avg_delay']
    sc = row['combined_score']
    fr = row['avg_forced_replan_rate']
    print(f"  {fh:4.0f}   {es:.2f}   {ad:8.1f}   {dl:8.1f}   {sc:.4f}   {fr:.3f}")
