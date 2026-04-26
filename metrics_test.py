import pandas as pd
from scipy.stats import ttest_rel

sgd = pd.read_csv("Metrics_SGD.csv")
nb  = pd.read_csv("Metrics_Baseline.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1"]
rows = []
for project in sgd["dataset"].unique():
    sgd_p = sgd[sgd["dataset"] == project]
    nb_p  = nb[nb["dataset"] == project]
    sgd_p = sgd_p.sort_values("run_id").reset_index(drop=True)
    nb_p  = nb_p.sort_values("run_id").reset_index(drop=True) #sort them so both align
    for metric in metrics:
        x = sgd_p[metric]
        y = nb_p[metric]
        t, p = ttest_rel(x, y) #work out p value from t test
        p = p / 2 if t > 0 else 1 - p / 2
        print(project, metric, "p =", format(p, ".3g"), "reject" if p <= 0.05 else "accept") #if less thna sig level reject h0 if not accept h0

        rows.append({
            "dataset": project,
            "metric": metric,
            "p_value": p, #put all the data in
            "decision": "reject" if p <= 0.05 else "accept"
        })

results = pd.DataFrame(rows)
results.to_csv("results_stats_test.csv", index=False)
avg_sgd = sgd.groupby("dataset")[metrics].mean().reset_index()
avg_nb  = nb.groupby("dataset")[metrics].mean().reset_index()

avg_sgd[metrics] = avg_sgd[metrics].round(3)
avg_nb[metrics]  = avg_nb[metrics].round(3) #round the metrics by 3 dp

avg_sgd.to_csv("avg_metrics_SGD.csv", index=False)
avg_nb.to_csv("avg_metrics_Baseline.csv", index=False)