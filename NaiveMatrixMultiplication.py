import csv
import time
import numpy as np
import os
import psutil

"""
Matrix Multiplication Benchmark Script with CPU and Memory Tracking
"""

# --- Parameters ---
matrix_sizes = [10, 100, 1_000, 10_000]  # [10, 100, 1_000, 10_000]
iterations = 100
pause_every = 20  # Pause every 20 iterations
pause_duration = 10  # Pause duration in seconds
csv_file = "results/python_results.csv"
language = "Python"
matrix_dir = "matrices"

process = psutil.Process(os.getpid())  # Current Python process

def read_matrix_from_binary(filename, size):
    """Reads a square matrix from a binary file saved with numpy.tofile()."""
    try:
        matrix = np.fromfile(filename, dtype=np.int32).reshape((size, size))
        print(f"✅ Loaded matrix from '{filename}' ({size}x{size})")
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Couldn't open file '{filename}'")
    except ValueError:
        raise ValueError(f"❌ Could not reshape file '{filename}' into {size}x{size}")

def naive_matrix_multiplication(matrix_a, matrix_b, n):
    """Performs naive matrix multiplication of two square matrices and returns elapsed time."""
    try:
        C = [[0 for _ in range(n)] for _ in range(n)]
        start = time.perf_counter()
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += matrix_a[i][k] * matrix_b[k][j]
        end = time.perf_counter()
        return end - start
    except RuntimeError:
        raise RuntimeError("❌ Couldn't perform multiplication")

def warm_up(matrix_a, matrix_b, matrix_size, iterations=5, pause=2):
    """Performs warm-up multiplications for the largest matrix size."""
    print(f"\n=== Warm-up: {iterations} iterations for size {matrix_size}x{matrix_size} ===")
    for i in range(1, iterations + 1):
        naive_matrix_multiplication(matrix_a, matrix_b, matrix_size)
        print(f"✅ Warm-up iteration {i} completed")
        time.sleep(pause)

def save_results_to_csv(results):
    """Saves the matrix multiplication results to a CSV file."""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f"\nSaving all results to {csv_file}...")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Size", "Matrix A File", "Matrix B File",
            "Mean Time (s)", "Median Time (s)", "Std Time (s)",
            "Mean CPU (%)", "Median CPU (%)", "Std CPU (%)",
            "Mean Memory (MB)", "Median Memory (MB)", "Std Memory (MB)",
            "Language"
        ])
        writer.writerows(results)
    print("✅ All results saved successfully")

# --- Main Execution ---
if __name__ == "__main__":
    results = []

    matrix_a_warm_up = read_matrix_from_binary(
        os.path.join(matrix_dir, f"A_{max(matrix_sizes)}.bin"), max(matrix_sizes))
    matrix_b_warm_up = read_matrix_from_binary(
        os.path.join(matrix_dir, f"B_{max(matrix_sizes)}.bin"), max(matrix_sizes))
    warm_up(matrix_a_warm_up, matrix_b_warm_up, max(matrix_sizes))

    for size in matrix_sizes:
        file_a = os.path.join(matrix_dir, f"A_{size}.bin")
        file_b = os.path.join(matrix_dir, f"B_{size}.bin")

        print(f"\n=== Processing matrices of size {size}x{size} ===")
        matrix_a = read_matrix_from_binary(file_a, size)
        matrix_b = read_matrix_from_binary(file_b, size)

        # Run multiple iterations and collect execution times, CPU, and memory
        times = []
        cpu_usages = []
        memory_usages = []

        print(f"\n=== Multiplicating matrices of size {size}x{size} ===")
        print(f"Running {iterations} iterations with a {pause_duration}s pause every {pause_every} iterations...")

        for i in range(1, iterations + 1):
            process.cpu_percent(interval=None)
            start_mem = process.memory_info().rss
            start_time = time.perf_counter()

            naive_matrix_multiplication(matrix_a, matrix_b, size)

            elapsed = time.perf_counter() - start_time
            end_mem = process.memory_info().rss
            cpu = process.cpu_percent(interval=None)  # Instant CPU % after operation
            mem = (end_mem - start_mem) / (1024 * 1024)  # Memory usage delta in MB

            times.append(elapsed)
            cpu_usages.append(cpu)
            memory_usages.append(mem)

            if i % pause_every == 0 and i != iterations:
                print(f"💤 Pausing for {pause_duration} seconds to cool off the CPU...")
                time.sleep(pause_duration)

        # Calculate statistics
        mean_time = np.mean(times)
        median_time = np.median(times)
        std_time = np.std(times)

        mean_cpu = np.mean(cpu_usages)
        median_cpu = np.median(cpu_usages)
        std_cpu = np.std(cpu_usages)

        mean_mem = np.mean(memory_usages)
        median_mem = np.median(memory_usages)
        std_mem = np.std(memory_usages)

        print(f"✅ Stats for size {size}: mean_time={mean_time:.6f}s, mean_cpu={mean_cpu:.2f}%, mean_mem={mean_mem:.2f}MB")

        # Append results
        results.append((
            size, file_a, file_b,
            mean_time, median_time, std_time,  # <--- Std Time (s) now
            mean_cpu, median_cpu, std_cpu,
            mean_mem, median_mem, std_mem,
            language
        ))

    save_results_to_csv(results)
    print("\n✅ Process completed successfully")
