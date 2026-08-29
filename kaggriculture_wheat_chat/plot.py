from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", default="artifacts/training_log.csv")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    out = Path(args.csv).with_suffix("")

    plots = [
        ("final_money", "Episode vs Final Money", "final_money.png"),
        ("return", "Episode vs Average Reward Return", "return.png"),
        ("wheat_sold", "Episode vs Wheat Sold", "wheat_sold.png"),
        ("workers_hired", "Episode vs Worker Usage", "workers.png"),
        ("land_purchases", "Episode vs Land Purchases", "land.png"),
        ("invalid_actions", "Episode vs Invalid Actions", "invalid_actions.png"),
    ]
    for col, title, filename in plots:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 4.5))
        plt.plot(df["episode"], df[col])
        plt.xlabel("Episode")
        plt.ylabel(col)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out.parent / filename, dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
