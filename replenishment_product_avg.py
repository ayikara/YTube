import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_productwise_avg_spread(spread_df, txn_df):
    # Constants
    THREE_SEC_NS = 3 * 10**9
    FIVE_SEC_NS = 5 * 10**9
    INTERVAL_NS = int(0.1 * 1e9)
    TOTAL_POINTS = 83

    # Initialize dictionaries to accumulate sums and counts per product
    sum_spreads_dict = defaultdict(lambda: np.zeros(TOTAL_POINTS, dtype=np.float64))
    count_spreads_dict = defaultdict(lambda: np.zeros(TOTAL_POINTS, dtype=np.int32))

    # Sort input data
    spread_df = spread_df.sort_values(["Product code", "timestamp"]).reset_index(drop=True)
    txn_df = txn_df.sort_values(["Product code", "timestamp"]).reset_index(drop=True)

    # Group spread data by product for fast access
    spread_grouped = spread_df.groupby("Product code")

    for product, txn_group in txn_df.groupby("Product code"):
        if product not in spread_grouped.groups:
            continue  # skip if no spread data for this product

        # Get spread data for the current product
        product_spread_df = spread_grouped.get_group(product).reset_index(drop=True)
        spread_times = product_spread_df["timestamp"].values
        spread_values = product_spread_df["bid ask spread"].values

        # Loop over each transaction for the product
        for _, txn in txn_group.iterrows():
            txn_time = txn["timestamp"]

            # 81-point time window from -3s to +5s
            times_range = np.arange(txn_time - THREE_SEC_NS, txn_time + FIVE_SEC_NS + 1, INTERVAL_NS)

            for j, ts in enumerate(times_range):
                idx = np.searchsorted(spread_times, ts, side="right") - 1
                if 0 <= idx < len(spread_values):
                    val = spread_values[idx]
                    sum_spreads_dict[product][j] += val
                    count_spreads_dict[product][j] += 1

            # One point just before transaction
            idx_before = np.searchsorted(spread_times, txn_time, side="right") - 1
            if 0 <= idx_before < len(spread_values):
                sum_spreads_dict[product][81] += spread_values[idx_before]
                count_spreads_dict[product][81] += 1

            # One point just after transaction
            idx_after = np.searchsorted(spread_times, txn_time, side="left")
            if 0 <= idx_after < len(spread_values):
                sum_spreads_dict[product][82] += spread_values[idx_after]
                count_spreads_dict[product][82] += 1

    # Final result: DataFrame with one row per product, 83 columns per row
    result_rows = []
    products = sorted(sum_spreads_dict.keys())
    column_labels = [f"idx_{i}" for i in range(TOTAL_POINTS)]

    for product in products:
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_spread = sum_spreads_dict[product] / count_spreads_dict[product]
            avg_spread = np.where(count_spreads_dict[product] > 0, avg_spread, np.nan)

        row = {"Product code": product}
        row.update({f"idx_{i}": avg_spread[i] for i in range(TOTAL_POINTS)})
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows, columns=["Product code"] + column_labels)

    return result_df
