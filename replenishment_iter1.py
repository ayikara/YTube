import pandas as pd
import numpy as np

def get_spread_snapshots(spread_df, txn_df):
    # Ensure timestamps are sorted
    spread_df = spread_df.sort_values("timestamp").reset_index(drop=True)
    txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)

    # Convert seconds to nanoseconds
    THREE_SEC_NS = 3 * 10**9
    FIVE_SEC_NS = 5 * 10**9
    INTERVAL_NS = int(0.1 * 1e9)  # 0.1 seconds in nanoseconds

    # Use numpy array for faster search
    spread_times = spread_df["timestamp"].values
    spread_values = spread_df["spread"].values

    result = []

    for i, txn in txn_df.iterrows():
        txn_time = txn["timestamp"]
        
        # Time range: from (T - 3s) to (T + 5s) in 0.1s interval
        times_range = np.arange(txn_time - THREE_SEC_NS, txn_time + FIVE_SEC_NS + 1, INTERVAL_NS)
        
        # Total of 81 time points
        snapshot = []
        
        for ts in times_range:
            idx = np.searchsorted(spread_times, ts, side="right") - 1
            if 0 <= idx < len(spread_values):
                snapshot.append(spread_values[idx])
            else:
                snapshot.append(np.nan)  # If out of bounds, set as NaN

        # Capture bid-ask just before and after transaction time
        idx_before = np.searchsorted(spread_times, txn_time, side="right") - 1
        idx_after = np.searchsorted(spread_times, txn_time, side="left")

        spread_before = spread_values[idx_before] if 0 <= idx_before < len(spread_values) else np.nan
        spread_after = spread_values[idx_after] if 0 <= idx_after < len(spread_values) else np.nan

        # Append the 2 additional spreads
        snapshot.append(spread_before)
        snapshot.append(spread_after)

        result.append({
            "txn_index": i,
            "txn_time": txn_time,
            "snapshot_spread": snapshot  # length should be 83
        })

    return pd.DataFrame(result)
