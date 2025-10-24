#!/usr/bin/env python3
"""
hr_display.py

Plot heart rate monitor data from one or more files located in the `data` directory.
Reads zone thresholds from config.yaml using OmegaConf.

Usage:
- With argument: hr_display.py <filename>
- Without argument: hr_display.py   (plots all files in `data`, alphabetically descending, saves to PDF in `reports`)
"""

import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  # use mpl.colormaps
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from datetime import datetime


def load_config(config_path: pathlib.Path) -> list[int]:
    """Load zone thresholds (in bpm) from config.yaml."""
    cfg = OmegaConf.load(config_path)
    return list(map(int, cfg.zone_thresholds_bpm))


def load_data(filepath: pathlib.Path) -> pd.DataFrame:
    """Load heart rate data from a CSV-like file."""
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


def plot_data(df: pd.DataFrame, filename: str, thresholds: list[int],
              y_limits: tuple[int, int] | None = None) -> plt.Figure:
    """
    Create a figure with heart rate data (left, ~2/3 width) and time in zones histogram (right, ~1/3 width).
    - Elapsed Time shown in minutes.
    - Heart rate values of 0 are skipped.
    - Grid with thin lines is displayed.
    - Histogram uses Set2 colors and shows percentages above bars.
    - Mark maximum HR with red dot and label (first occurrence only).
    - Zone bands are shown on HR plot with the same colors as the histogram.
    - Apply y_limits if provided (do not alter them for zone bands).
    - Line plot title includes filename and total duration (MM:SS).
    Returns the matplotlib Figure.
    """
    # Remove rows where Heartrate is 0 (missing values)
    df = df[df['Heartrate'] != 0].copy()

    # Convert elapsed time from ms to minutes
    df['Minutes'] = df['Elapsed Time'] / 60000.0

    # Compute total duration
    if not df.empty and 'Elapsed Time' in df.columns:
        total_ms = df['Elapsed Time'].iloc[-1]
        total_seconds = int(total_ms / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        duration_str = f"{minutes:02d}:{seconds:02d}"
    else:
        duration_str = "00:00"

    # Compute zone times
    zone_times = compute_zone_times(df, thresholds)
    num_zones = len(zone_times)
    zone_labels = [f'Z{i}' for i in range(num_zones)]
    percentages = compute_percentages(zone_times)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]}
    )

    # Define colormap and colors (highly distinct)
    cmap = mpl.colormaps['Set2']
    bar_colors = [cmap(i % cmap.N) for i in range(num_zones)]

    # Left: HR vs time (blue line)
    ax1.plot(df['Minutes'], df['Heartrate'], color='blue', zorder=2)
    ax1.set_xlabel('Elapsed Time (minutes)')
    ax1.set_ylabel('Heart Rate (bpm)')
    ax1.set_title(f"{filename} {duration_str}")
    ax1.grid(True, linewidth=0.5, zorder=1)

    if y_limits:
        ymin, ymax = y_limits
        # add a little headroom to the top
        ax1.set_ylim(ymin, ymax * 1.02)

    # --- Add horizontal colored bands for HR zones (AFTER y-limits are known) ---
    bins = [0] + thresholds + [float('inf')]
    current_ylim = ax1.get_ylim()
    top_limit = current_ylim[1]
    for (low, high), color in zip(zip(bins[:-1], bins[1:]), bar_colors):
        band_top = high if high != float('inf') else top_limit
        ax1.axhspan(low, band_top, facecolor=color, alpha=0.25, zorder=0)
    # ---------------------------------------------------------------------------

    # Mark maximum HR (first occurrence)
    if not df.empty:
        max_hr = df['Heartrate'].max()
        max_idx = df['Heartrate'].idxmax()
        max_time = df.loc[max_idx, 'Minutes']
        ax1.scatter(max_time, max_hr, color='red', zorder=3)
        ax1.text(max_time, max_hr, f' {max_hr}', color='red', va='bottom')

    # Right: histogram of time in zones (Set2 colors)
    bars = ax2.bar(zone_labels, zone_times, color=bar_colors)

    ax2.set_xlabel('Heart Rate Zones')
    ax2.set_ylabel('Time (minutes)')
    ax2.set_title('Time in Zones')
    ax2.grid(True, axis='y', linewidth=0.5)

    # Add percentages above bars
    ymax_hist = max(zone_times) if zone_times else 0
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax_hist * 0.02,
            f'{pct}%',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    # Add headroom so text doesn’t overlap top
    ax2.set_ylim(0, ymax_hist * 1.15 if ymax_hist > 0 else 1)

    plt.tight_layout()
    return fig


def main() -> None:
    """Main entry point of the script."""
    data_dir = pathlib.Path('data')
    reports_dir = pathlib.Path('reports')
    config_path = pathlib.Path('config.yaml')

    if not config_path.exists():
        print(f'Error: config file {config_path} does not exist.')
        sys.exit(1)

    thresholds = load_config(config_path)

    if len(sys.argv) == 2:
        # Single file mode (interactive display)
        filename = sys.argv[1]
        filepath = data_dir / filename

        if not filepath.exists():
            print(f'Error: file {filepath} does not exist.')
            sys.exit(1)

        df = load_data(filepath)
        fig = plot_data(df, filename, thresholds)
        plt.show()
    elif len(sys.argv) == 1:
        # Multi-file mode: all files in data dir, sorted alphabetically descending, save to PDF
        files = sorted(
            [f for f in data_dir.iterdir() if f.is_file()],
            key=lambda f: f.name.lower(),
            reverse=True
        )

        if not files:
            print(f'No files found in {data_dir}')
            sys.exit(1)

        # Determine global y-limits across all files
        y_min, y_max = None, None
        dataframes = []
        for filepath in files:
            df = load_data(filepath)
            df = df[df['Heartrate'] != 0]
            if not df.empty:
                current_min, current_max = df['Heartrate'].min(), df['Heartrate'].max()
                y_min = current_min if y_min is None else min(y_min, current_min)
                y_max = current_max if y_max is None else max(y_max, current_max)
            dataframes.append((filepath, df))

        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        pdf_path = reports_dir / f'{timestamp}.pdf'

        with PdfPages(pdf_path) as pdf:
            for filepath, df in dataframes:
                fig = plot_data(df, filepath.name, thresholds, (y_min, y_max))
                pdf.savefig(fig)
                plt.close(fig)

        print(f'Report saved to {pdf_path}')
    else:
        print(f'Usage: {sys.argv[0]} [<filename>]')
        sys.exit(1)


if __name__ == '__main__':
    main()
