import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_words_from_stable_whisper(json_path: str) -> list[dict]:
    """Extract word data from stable_whisper JSON output."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    for segment in data.get("segments", []):
        for word_data in segment.get("words", []):
            words.append({
                "word": word_data.get("word", "").strip(),
                "start_time": word_data.get("start"),
                "end_time": word_data.get("end"),
                "probability": word_data.get("probability"),
            })

    return words


def compute_moving_average(probs: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average with given window size."""
    moving_avg = np.full(len(probs), np.nan)
    for i in range(len(probs)):
        start = max(0, i - window + 1)
        moving_avg[i] = np.mean(probs[start:i + 1])
    return moving_avg


def compute_cusum(moving_avg: np.ndarray, window: int = 100, target: float = 0.25) -> np.ndarray:
    """Compute one-sided negative CUSUM against a fixed target."""
    cusum = np.full(len(moving_avg), np.nan)

    cumulative = 0
    for i in range(window, len(moving_avg)):
        cumulative = min(0, cumulative + (moving_avg[i] - target))
        cusum[i] = cumulative
    return cusum


def compute_gradient(moving_avg: np.ndarray) -> np.ndarray:
    """Compute gradient (rate of change) of moving average."""
    gradient = np.full(len(moving_avg), np.nan)
    gradient[1:] = np.diff(moving_avg)
    return gradient


def detect_degradation_rolling_avg(probs: np.ndarray, window: int = 100, target: float = 0.25) -> int:
    """Find index where rolling avg drops below target."""
    moving_avg = compute_moving_average(probs, window)

    for i in range(window, len(moving_avg)):
        if moving_avg[i] < target:
            return i
    return -1


def detect_degradation_cusum(
    probs: np.ndarray, window: int = 100, threshold: float = 50.0, target: float = 0.25
) -> int:
    """CUSUM change point detection for detecting mean shift."""
    moving_avg = compute_moving_average(probs, window)

    cusum_neg = 0

    for i in range(window, len(moving_avg)):
        diff = moving_avg[i] - target
        cusum_neg = min(0, cusum_neg + diff)

        if abs(cusum_neg) > threshold:
            return i
    return -1


def detect_degradation_gradient(probs: np.ndarray, window: int = 100) -> int:
    """Find the steepest drop in rolling average."""
    moving_avg = compute_moving_average(probs, window)
    gradient = np.diff(moving_avg[window:])  # Gradient of moving average

    min_gradient_idx = np.argmin(gradient)
    return min_gradient_idx + window


def detect_degradation_sustained(probs: np.ndarray, window: int = 100, threshold: float = 0.25, n_points: int = 100) -> int:
    """Find index where moving average stays below threshold for N consecutive points."""
    moving_avg = compute_moving_average(probs, window)

    consecutive_below = 0
    for i in range(window, len(moving_avg)):
        if moving_avg[i] < threshold:
            consecutive_below += 1
            if consecutive_below >= n_points:
                return i - n_points + 1  # Return start of sustained drop
        else:
            consecutive_below = 0
    return -1


def detect_degradation(probs: np.ndarray, window: int = 100, target: float = 0.25) -> dict:
    """Run all detection methods and return results."""
    results = {
        "rolling_avg_method": detect_degradation_rolling_avg(probs, window=window, target=target),
        "cusum_method": detect_degradation_cusum(probs, window=window, target=target),
        "gradient_method": detect_degradation_gradient(probs, window=window),
        "sustained_method": detect_degradation_sustained(probs, window=window, threshold=target, n_points=200),
    }

    # Recommended: use the earliest detection (most conservative)
    valid_results = [v for v in results.values() if v > 0]
    results["recommended"] = min(valid_results) if valid_results else -1

    return results


def plot_probability_and_moving_avg(df: pd.DataFrame, output_path: str, window: int = 100) -> None:
    """Plot probability and moving average."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df['probability'], alpha=0.3, label='Probability', color='blue')
    ax.plot(df.index, df['moving_avg'], label=f'Moving Average (window={window})', color='red', linewidth=2)

    ax.set_xlabel('Index')
    ax.set_ylabel('Probability')
    ax.set_title('Probability and Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")
    plt.show()


def plot_cusum(df: pd.DataFrame, output_path: str) -> None:
    """Plot CUSUM values."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df['cusum'], label='CUSUM', color='purple', linewidth=1.5)
    ax.axhline(y=-50, color='red', linestyle='--', label='Threshold (-50)')

    ax.set_xlabel('Index')
    ax.set_ylabel('CUSUM')
    ax.set_title('CUSUM (Cumulative Sum of Deviations from Baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")
    plt.show()


def export_to_excel(words: list[dict], output_path: str, window: int = 100, target: float = 0.25) -> None:
    """Export words to Excel with moving average, cusum, gradient, sorted by start time."""
    df = pd.DataFrame(words)
    df = df.sort_values(by="start_time").reset_index(drop=True)

    # Add calculated columns
    probs = df["probability"].values
    moving_avg = compute_moving_average(probs, window=window)
    df["moving_avg"] = moving_avg
    df["cusum"] = compute_cusum(moving_avg, window=window, target=target)
    df["gradient"] = compute_gradient(moving_avg)

    # Detect degradation and print results
    results = detect_degradation(probs, window=window, target=target)
    print("\n=== Degradation Detection Results ===")
    print(f"Rolling Average Method (below {target}): index {results['rolling_avg_method']}")
    print(f"CUSUM Method (target={target}, threshold=50): index {results['cusum_method']}")
    print(f"Gradient Method (steepest drop): index {results['gradient_method']}")
    print(f"Sustained Method (below {target} for 200 pts): index {results['sustained_method']}")
    print(f"Recommended (earliest): index {results['recommended']}")

    # Create detection results summary dataframe
    results_df = pd.DataFrame([
        {"Method": "Gradient", "Index": results['gradient_method'], "Description": "Steepest single drop"},
        {"Method": "Sustained", "Index": results['sustained_method'], "Description": f"Below {target} for 200 consecutive points"},
        {"Method": "Rolling Avg", "Index": results['rolling_avg_method'], "Description": f"Moving avg dropped below {target}"},
        {"Method": "CUSUM", "Index": results['cusum_method'], "Description": f"Cumulative deviation from {target} exceeded 50"},
        {"Method": "Recommended", "Index": results['recommended'], "Description": "Earliest detection (most conservative)"},
    ])

    # Write to Excel with two sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Words', index=False)
        results_df.to_excel(writer, sheet_name='Detection Results', index=False)

    print(f"\nExported {len(df)} words to {output_path}")
    print("Sheets: 'Words' (data), 'Detection Results' (summary)")

    # Generate plots
    base_name = output_path.rsplit('.', 1)[0]
    plot_probability_and_moving_avg(df, f"{base_name}_probability.png", window=window)
    plot_cusum(df, f"{base_name}_cusum.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze stable_whisper JSON output for transcription quality degradation")
    parser.add_argument("json_file", help="Path to stable_whisper JSON file")
    parser.add_argument("--ma-window", type=int, default=100, help="Moving average window size (default: 100)")
    parser.add_argument("--target", type=float, default=0.25, help="Target probability threshold for degradation detection (default: 0.25)")
    args = parser.parse_args()

    json_path = args.json_file
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{base_name}_words.xlsx")

    words = extract_words_from_stable_whisper(json_path)
    export_to_excel(words, output_path, window=args.ma_window, target=args.target)


if __name__ == "__main__":
    main()
