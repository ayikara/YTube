import pandas as pd
import numpy as np

def calculate_avg_spread_over_transactions(spread_df, txn_df):
    # Ensure timestamps are sorted
    spread_df = spread_df.sort_values("timestamp").reset_index(drop=True)
    txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)

    # Constants
    THREE_SEC_NS = 3 * 10**9
    FIVE_SEC_NS = 5 * 10**9
    INTERVAL_NS = int(0.1 * 1e9)  # 0.1 seconds in nanoseconds
    TOTAL_POINTS = 83  # 81 from ± window, 2 additional (before & after)

    # Prepare spread time/value arrays
    spread_times = spread_df["timestamp"].values
    spread_values = spread_df["spread"].values

    # Initialize accumulators
    sum_spreads = np.zeros(TOTAL_POINTS, dtype=np.float64)
    count_spreads = np.zeros(TOTAL_POINTS, dtype=np.int32)

    for i, txn in txn_df.iterrows():
        txn_time = txn["timestamp"]

        # Range of timestamps: from (T - 3s) to (T + 5s) at 0.1s intervals (81 points)
        times_range = np.arange(txn_time - THREE_SEC_NS, txn_time + FIVE_SEC_NS + 1, INTERVAL_NS)

        snapshot = []

        # 81 values from times_range
        for j, ts in enumerate(times_range):
            idx = np.searchsorted(spread_times, ts, side="right") - 1
            if 0 <= idx < len(spread_values):
                val = spread_values[idx]
                sum_spreads[j] += val
                count_spreads[j] += 1

        # One just before transaction time
        idx_before = np.searchsorted(spread_times, txn_time, side="right") - 1
        if 0 <= idx_before < len(spread_values):
            sum_spreads[81] += spread_values[idx_before]
            count_spreads[81] += 1

        # One just after transaction time
        idx_after = np.searchsorted(spread_times, txn_time, side="left")
        if 0 <= idx_after < len(spread_values):
            sum_spreads[82] += spread_values[idx_after]
            count_spreads[82] += 1

    # Avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        avg_spreads = sum_spreads / count_spreads
        avg_spreads = np.where(count_spreads > 0, avg_spreads, np.nan)

    # Create result dataframe
    column_labels = [f"idx_{i}" for i in range(TOTAL_POINTS)]
    result_df = pd.DataFrame([avg_spreads], columns=column_labels)

    return result_df
