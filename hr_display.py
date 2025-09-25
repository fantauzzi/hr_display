#!/usr/bin/env python3
"""
hr_display.py

Plot heart rate monitor data from a CSV file located in the `data` directory.
Reads zone thresholds from config.yaml using OmegaConf.
"""

import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  # use mpl.colormaps
from omegaconf import OmegaConf


def load_config(config_path: pathlib.Path) -> list[int]:
    """Load zone thresholds (in bpm) from config.yaml."""
    cfg = OmegaConf.load(config_path)
    return list(map(int, cfg.zone_thresholds_bpm))


def load_data(filepath: pathlib.Path) -> pd.DataFrame:
    """Load heart rate data from a CSV file."""
    return pd.read_csv(filepath, skipinitialspace=True)


def compute_zone_times(df: pd.DataFrame, thresholds: list[int]) -> list[float]:
    """
    Compute time spent in each heart rate zone.
    Returns list of minutes per zone.
    """
    # Remove rows where Heartrate is 0 (missing values)
    hr = df['Heartrate'][df['Heartrate'] != 0]

    # Each row = 1 second
    seconds_per_sample = 1.0

    # Define zone boundaries
    bins = [0] + thresholds + [float('inf')]
    zone_times: list[float] = []

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (hr >= low) & (hr < high)
        total_seconds = mask.sum() * seconds_per_sample
        zone_times.append(total_seconds / 60.0)  # minutes

    return zone_times


def compute_percentages(values: list[float]) -> list[int]:
    """Compute integer percentages that add up to 100."""
    total = sum(values)
    if total == 0:
        return [0] * len(values)

    raw = [v / total * 100 for v in values]
    rounded = [int(round(x)) for x in raw]

    diff = 100 - sum(rounded)
    if rounded:
        rounded[-1] += diff
    return rounded


def plot_data(df: pd.DataFrame, filename: str, thresholds: list[int]) -> None:
    """
    Plot heart rate data (left, ~2/3 width) and time in zones histogram (right, ~1/3 width).
    - Elapsed Time shown in minutes.
    - Heart rate values of 0 are skipped.
    - Grid with thin lines is displayed.
    - Histogram uses tab20b colors and shows percentages above bars.
    - Mark maximum HR with red dot and label (first occurrence only).
    """
    # Remove rows where Heartrate is 0 (missing values)
    df = df[df['Heartrate'] != 0].copy()

    # Convert elapsed time from ms to minutes
    df['Minutes'] = df['Elapsed Time'] / 60000.0

    # Compute zone times
    zone_times = compute_zone_times(df, thresholds)
    num_zones = len(zone_times)
    zone_labels = [f'Z{i}' for i in range(num_zones)]
    percentages = compute_percentages(zone_times)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]}
    )

    # Left: HR vs time (blue line)
    ax1.plot(df['Minutes'], df['Heartrate'], color='blue', zorder=2)
    ax1.set_xlabel('Elapsed Time (minutes)')
    ax1.set_ylabel('Heart Rate (bpm)')
    ax1.set_title(filename)
    ax1.grid(True, linewidth=0.5, zorder=1)

    # Mark maximum HR (first occurrence), behind the line
    if not df.empty:
        max_hr = df['Heartrate'].max()
        max_idx = df['Heartrate'].idxmax()
        max_time = df.loc[max_idx, 'Minutes']
        ax1.scatter(max_time, max_hr, color='red', zorder=1)
        ax1.text(max_time, max_hr, f' {max_hr}', color='red', va='bottom')

    # Right: histogram of time in zones (tab20b colors)
    cmap = mpl.colormaps['tab20b']
    bar_colors = [cmap(i % cmap.N) for i in range(num_zones)]
    bars = ax2.bar(zone_labels, zone_times, color=bar_colors)

    ax2.set_xlabel('Heart Rate Zones')
    ax2.set_ylabel('Time (minutes)')
    ax2.set_title('Time in Zones')
    ax2.grid(True, axis='y', linewidth=0.5)

    # Add percentages above bars
    ymax = max(zone_times) if zone_times else 0
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * 0.02,
            f'{pct}%',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    # Add headroom so text doesn’t overlap top
    ax2.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main entry point of the script."""
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <filename.csv>')
        sys.exit(1)

    data_dir = pathlib.Path('data')
    config_path = pathlib.Path('config.yaml')

    filename = sys.argv[1]
    filepath = data_dir / filename

    if not filepath.exists():
        print(f'Error: file {filepath} does not exist.')
        sys.exit(1)

    if not config_path.exists():
        print(f'Error: config file {config_path} does not exist.')
        sys.exit(1)

    thresholds = load_config(config_path)
    df = load_data(filepath)
    plot_data(df, filename, thresholds)


if __name__ == '__main__':
    main()