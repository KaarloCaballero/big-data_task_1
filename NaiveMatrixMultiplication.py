import csv
import time
import numpy as np
import os

"""
Matrix Multiplication Benchmark Script

This script performs naive matrix multiplication on pre-generated square matrices 
stored as binary files. For each matrix size, it executes the multiplication 100 times,
pausing for 10 seconds every 20 iterations to cool off the CPU. It measures execution 
times with high-resolution perf_counter and calculates mean, median, and standard deviation. 
Results are saved into a CSV file.
"""

# --- Parameters ---
matrix_sizes = [10, 100]
iterations = 100
pause_every = 20  # Pause every 20 iterations
pause_duration = 10  # Pause duration in seconds
csv_file = "results/python_results.csv"
language = "Python"
matrix_dir = "matrices"


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
    C = [[0 for _ in range(n)] for _ in range(n)]
    start = time.perf_counter()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += matrix_a[i][k] * matrix_b[k][j]
    end = time.perf_counter()
    return end - start

def warm_up(matrix_a, matrix_b, n, iterations=3, pause=2):
    """
    Performs warm-up multiplications for the largest matrix size to stabilize CPU/cache/threads.
    
    Parameters:
        matrix_a (np.ndarray): First matrix
        matrix_b (np.ndarray): Second matrix
        n (int): Matrix size
        iterations (int): Number of warm-up iterations
        pause (float): Pause in seconds between warm-up iterations
    """
    print(f"\n=== Warm-up: {iterations} iterations for size {n}x{n} ===")
    for i in range(1, iterations + 1):
        naive_matrix_multiplication(matrix_a, matrix_b, n)
        print(f"💨 Warm-up iteration {i} completed")
        time.sleep(pause)


def save_results_to_csv(results):
    """Saves the matrix multiplication results to a CSV file."""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f"\nSaving all results to {csv_file}...")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Size", "Matrix A File", "Matrix B File",
            "Mean Time (s)", "Median Time (s)", "Std Dev (s)", "Language"
        ])
        writer.writerows(results)
    print("✅ All results saved successfully")


# --- Main Execution ---
if __name__ == "__main__":
    results = []

    for size in matrix_sizes:
        file_a = os.path.join(matrix_dir, f"A_{size}.bin")
        file_b = os.path.join(matrix_dir, f"B_{size}.bin")

        print(f"\n=== Processing matrices of size {size}x{size} ===")
        matrix_a = read_matrix_from_binary(file_a, size)
        matrix_b = read_matrix_from_binary(file_b, size)

        # Run multiple iterations and collect execution times
        times = []
        print(f"Running {iterations} iterations with a {pause_duration}s pause every {pause_every} iterations...")

        for i in range(1, iterations + 1):
            elapsed = naive_matrix_multiplication(matrix_a, matrix_b, size)
            times.append(elapsed)

            # Pause every 'pause_every' iterations except the last one
            if i % pause_every == 0 and i != iterations:
                print(f"💤 Pausing for {pause_duration} seconds to cool off the CPU...")
                time.sleep(pause_duration)

        # Calculate statistics
        mean_time = np.mean(times)
        median_time = np.median(times)
        std_time = np.std(times)

        print(f"✅ Stats for size {size}: mean={mean_time:.6f}, median={median_time:.6f}, std={std_time:.6f}")

        # Append results with size as the first column
        results.append((size, file_a, file_b, mean_time, median_time, std_time, language))

    # Save results to CSV
    save_results_to_csv(results)
    print("✅ Process completed successfully")
