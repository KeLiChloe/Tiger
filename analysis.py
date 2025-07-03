import pickle
import pandas as pd
# Load the newly uploaded pickle file

latest_file_path = "exp_result_dict_cool.pkl"
with open(latest_file_path, "rb") as f:
    latest_exp_result_dict = pickle.load(f)
    
algo_run_counts = {algo: len(latest_exp_result_dict[algo]) for algo in ['gmm', 'policy_tree', 'oast']}
print("Number of runs for each algorithm:")
for algo, count in algo_run_counts.items():
    print(f"{algo}: {count} runs")

# Extract profits into a DataFrame
n_runs_latest = len(latest_exp_result_dict["gmm"])
latest_profit_comparison = []

for i in range(n_runs_latest):
    gmm_profit = latest_exp_result_dict["gmm"][i]["profit_at_manager_picked_M"]
    policy_tree_profit = latest_exp_result_dict["policy_tree"][i]["profit_at_manager_picked_M"]
    oast_profit = latest_exp_result_dict["oast"][i]["profit_at_manager_picked_M"]

    latest_profit_comparison.append({
        "gmm": gmm_profit,
        "policy_tree": policy_tree_profit,
        "oast": oast_profit
    })

# Convert to DataFrame
latest_profit_df = pd.DataFrame(latest_profit_comparison)

# Count comparisons
oast_vs_gmm = (latest_profit_df["oast"] > latest_profit_df["gmm"]).sum()
policy_tree_vs_gmm = (latest_profit_df["policy_tree"] > latest_profit_df["gmm"]).sum()
either_tree_vs_gmm = ((latest_profit_df["policy_tree"] > latest_profit_df["gmm"]) | (latest_profit_df["oast"] > latest_profit_df["gmm"])).sum()

oast_vs_policy_tree = (latest_profit_df["oast"] > latest_profit_df["policy_tree"]).sum()

# Average profits
latest_avg_profits = latest_profit_df.mean()

# Display counts of comparisons
print("\nCounts of Comparisons:")
print(f"OAST vs GMM: {oast_vs_gmm}")
print(f"Policy Tree vs GMM: {policy_tree_vs_gmm}")
print(f"OAST vs Policy Tree: {oast_vs_policy_tree}")
print(f"Either Tree vs GMM: {either_tree_vs_gmm}")

# Display average profits
print("\nAverage Profits:")
print(latest_avg_profits)
