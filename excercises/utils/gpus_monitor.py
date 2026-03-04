import datetime
import functools
import logging
import os
import signal
import subprocess
import time
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sb

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Define and creat the dir path where gpu monitoring data and this module's logs will be saved


GPUS_MONITOR_DIR = os.getenv(
    "GPUS_MONITOR_DIR",
    f"gpus-monitor/{os.environ['SLURM_NODEID']}-{os.environ['SLURMD_NODENAME']}",
)
GPUS_MONITOR_PREFIX_PATH = os.getenv("GPUS_MONITOR_PREFIX_PATH", "")
OUTDIR = os.path.join(GPUS_MONITOR_PREFIX_PATH, GPUS_MONITOR_DIR)
os.makedirs(OUTDIR, exist_ok=True)

SAMPLING_INTERVAL_SECONDS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s |  %(levelname)s | %(name)s : %(message)s",
    handlers=[logging.FileHandler(os.path.join(OUTDIR, "gpus_monitor.logs"))],
    force=True,
)

logger = logging.getLogger(__name__)

running = True
query_fields = [
    "timestamp",
    "index",
    "name",
    "power.draw",
    "memory.used",
    "utilization.gpu",
]
cols = [
    "script_timestamp",
    "nvidia_timestamp",
    "gpu_index",
    "gpu_name",
    "power_draw_watts",
    "memory_used_MiB",
    "utilization_percent",
]
gpus_metrics = pd.DataFrame(columns=cols)


def label_rotate(ax: "Axes", rotation: float, axis: Literal["x", "y"]) -> "Axes":
    if axis == "x":
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
    if axis == "y":
        for label in ax.get_yticklabels():
            label.set_rotation(rotation)
    return ax


def plot(data, x, y, hue, title, ylabel, png_name):

    plt.figure(figsize=(10, 8))
    # Ensure gpu_index is string for legend clarity
    data = data.copy()
    data[hue] = data[hue].astype(str)
    # Plot each group
    for gpu_idx, group in data.groupby(hue):
        plt.plot(group[x], group[y], label=f"{hue} {gpu_idx}")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.legend(title=hue)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=30))  # Limit xticks
    plt.xticks(rotation=30, fontsize=10)
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, png_name), format="png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def sn_plot(data, x, y, hue, title, ylabel, png_name):
    ax = sb.lineplot(data=data, x=x, y=y, hue=hue)
    f = ax.get_figure()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time")
    ax = label_rotate(ax, 45, "x")
    ax = label_rotate(ax, 45, "y")

    logger.info("Saving %s", os.path.join(OUTDIR, png_name))
    f.savefig(
        os.path.join(OUTDIR, png_name), format="png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def get_power_kwhrs(watt: list[float]) -> float:
    """Calculate total energy consumption in kWh from power draw data.

    Args:
        data (pd.DataFrame): DataFrame containing 'power_draw_watts' column.
    Returns:
        float: Total energy consumption in kWh.
    """

    total_energy_kwh = 0.0
    for i in range(1, len(watt)):
        # Energy (kWh) = Power (W) * Time (h)
        # Time between samples is 5 seconds = 5/3600 hours
        total_energy_kwh += (
            ((watt[i] + watt[i - 1]) / 2) * (SAMPLING_INTERVAL_SECONDS / 3600) / 1000
        )

    return total_energy_kwh


def handle_sigterm(signum, frame):
    global running
    global gpus_metrics
    logger.info("Received SIGTERM. Preparing to stop monitoring...")

    logger.info("Ploting metrics...")

    gpus_metrics["script_timestamp_time"] = gpus_metrics["script_timestamp"].apply(
        lambda x: x.time().strftime("%H:%M:%S")
    )
    # gpus_metrics["script_timestamp"] = gpus_metrics["script_timestamp"].apply(
    #    lambda x: x - gpus_metrics.loc[0, "script_timestamp"]
    # )
    gpus_metrics["power_draw_watts"] = gpus_metrics["power_draw_watts"].apply(
        lambda x: float(x)
    )
    gpus_metrics["utilization_percent"] = gpus_metrics["utilization_percent"].apply(
        lambda x: int(x)
    )
    gpus_metrics["memory_used_GiB"] = gpus_metrics["memory_used_MiB"].apply(
        lambda x: int(x) / 1024
    )
    gpus_metrics["gpu_index"] = gpus_metrics.gpu_index.apply(lambda x: str(x))
    # gpus_metrics["ts_hour"] = gpus_metrics.script_timestamp.apply(lambda x: x.time())
    # gpus_metrics = gpus_metrics.sort_values(by="script_timestamp")

    x = "script_timestamp_time"
    hue = "gpu_index"

    # Calculate energy consumption per GPU and get average
    total_kwhrs = 0.0
    num_gpus = 0
    for gpu_idx, group in gpus_metrics.groupby("gpu_index"):
        gpu_kwhrs = get_power_kwhrs(group["power_draw_watts"].tolist())
        total_kwhrs += gpu_kwhrs
        num_gpus += 1

    avg_kwhrs = total_kwhrs / num_gpus if num_gpus > 0 else 0.0

    # print Watt energy consumption
    y = "power_draw_watts"
    png_name = "watt.png"
    title = f"GPU Power consumption - Avg: {avg_kwhrs:.4f} kWh/GPU, Total: {total_kwhrs:.4f} kWh"
    ylabel = "Watt"

    plot(
        data=gpus_metrics,
        x=x,
        y=y,
        hue=hue,
        title=title,
        ylabel=ylabel,
        png_name=png_name,
    )
    # print memory
    y = "memory_used_GiB"
    png_name = "memory.png"
    title = "GPU memory used"
    ylabel = "GiB"
    plot(
        data=gpus_metrics,
        x=x,
        y=y,
        hue=hue,
        title=title,
        ylabel=ylabel,
        png_name=png_name,
    )

    # print utilization
    y = "utilization_percent"
    png_name = "utilization.png"
    title = "GPU utilization percentage"
    ylabel = "%"
    plot(
        data=gpus_metrics,
        x=x,
        y=y,
        hue=hue,
        title=title,
        ylabel=ylabel,
        png_name=png_name,
    )

    exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


def logtimeit(func):
    """Decorator that times the execution of a function and prints the elapsed time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timing
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Compute elapsed time
        logger.info(
            f"⏱️ Function '{func.__name__}' executed in {elapsed_time:.6f} seconds"
        )
        return result

    return wrapper


@logtimeit
def get_gpu_metrics() -> list[list[str]]:
    query = ",".join(query_fields)
    cmd = f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits"

    try:
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        lines = output.strip().split("\n")
        gpu_data = []

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            gpu_data.append(parts)
        return gpu_data

    except subprocess.CalledProcessError as e:
        print("Error _with_nsysrunning nvidia-smi:", e)
        return []


def monitor_gpus(
    interval_seconds: int = SAMPLING_INTERVAL_SECONDS,
    duration_seconds: int = 10 * 3600,
    output_file="gpu_log.csv",
):
    # writer = csv.writer(csvfile)
    global running
    global gpus_metrics
    while running:
        timestamp = datetime.datetime.now()
        metrics = get_gpu_metrics()
        for i, _ in enumerate(metrics):
            # Insert custom timestamp from script at the start
            # writer.writerow([timestamp] + gpu)
            metrics[i].insert(0, timestamp)
        gpus_metrics = pd.concat(
            [gpus_metrics, pd.DataFrame(columns=cols, data=metrics)], ignore_index=True
        )

        gpus_metrics.to_csv(output_file)

        time.sleep(interval_seconds)

    return


if __name__ == "__main__":
    # Change values as needed

    monitor_gpus(
        interval_seconds=SAMPLING_INTERVAL_SECONDS,
        output_file=os.path.join(OUTDIR, "gpu_usage_log.csv"),
    )
