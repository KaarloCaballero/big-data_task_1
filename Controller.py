import os
import numpy as np
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time

# Global constants
MATRIX_SIZES = [10, 100, 1_000, 10_000]
RESULTS_DIR = "results"
GRAPHS_DIR = "graphs"
MATRIX_DIR = "matrices"
SEED = 1


#------------------- FUNCTIONS ------------------
def generate_matrices(matrix_sizes, seed, matrix_dir):
    print("\n=== Generating matrices ===")
    rng = np.random.default_rng(seed)  # ✅ Usar el generador correctamente
    
    os.makedirs(matrix_dir, exist_ok=True)  # Asegura que el directorio exista
    
    for size in matrix_sizes:
        # ✅ Generar enteros aleatorios entre 0 y 9 (inclusive)
        MATRIX_A = rng.integers(0, 10, size=(size, size), dtype=np.int32)
        MATRIX_B = rng.integers(0, 10, size=(size, size), dtype=np.int32)

        # Archivos de salida
        matrix_a_file = os.path.join(matrix_dir, f"A_{size}.bin")
        matrix_b_file = os.path.join(matrix_dir, f"B_{size}.bin")

        # ✅ Guardar como binario
        MATRIX_A.tofile(matrix_a_file)
        MATRIX_B.tofile(matrix_b_file)

        #print(f"Saved {matrix_a_file} and {matrix_b_file}")
        print(f"✅ Generated matrices of size {size}x{size}...")

    print("✅ All matrices generated and saved as binary files.")

def run_file(cmd, lang):
    print(f"\n=== Running {lang} script ===")
    subprocess.run(cmd, shell=True, check=True)

def execute_naive_matrix_multiplication():
    file_name = "NaiveMatrixMultiplication"
    run_file(fr"C:/Python313/python.exe c:/Users/valko/VSCodeProjects/big-data_task_1/{file_name}.py", "Python")
    run_file(fr"java -Xmx4G {file_name}", "Java")
    run_file(fr".\c_{file_name}.exe", "C")
    run_file(r".\naive_matrix\target\release\naive_matrix.exe", "Rust") 
    return 0

def generate_time_graph(df, lang, graphs_dir):
    plt.figure(figsize=(8, 6))
    # Replace zeros with a small positive value for log plotting
    y_mean = df['Mean Time (s)'].replace(0, 1e-12)
    y_median = df['Median Time (s)'].replace(0, 1e-12)
    y_std = df['Std Time (s)'].replace(0, 1e-12)

    plt.plot(df['Size'], y_mean, marker='o', label='Mean')
    plt.plot(df['Size'], y_median, marker='s', label='Median')
    plt.plot(df['Size'], y_std, marker='^', label='Std')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.title(f"{lang} Benchmark - Time")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f"{lang.lower()}_time.png"))
    plt.close()

def generate_cpu_graph(df, lang, graphs_dir):
    plt.figure(figsize=(8, 6))
    y_mean = df['Mean CPU (%)'].replace(0, 1e-12)
    y_median = df['Median CPU (%)'].replace(0, 1e-12)
    y_std = df['Std CPU (%)'].replace(0, 1e-12)

    plt.plot(df['Size'], y_mean, marker='o', label='Mean')
    plt.plot(df['Size'], y_median, marker='s', label='Median')
    plt.plot(df['Size'], y_std, marker='^', label='Std')
    plt.xlabel("Matrix Size")
    plt.ylabel("CPU Usage (%)")
    plt.title(f"{lang} Benchmark - CPU Usage")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(bottom=0)  # y-axis starts at 0
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f"{lang.lower()}_cpu.png"))
    plt.close()

def generate_memory_graph(df, lang, graphs_dir):
    plt.figure(figsize=(8, 6))
    y_mean = df['Mean Memory (MB)'].replace(0, 1e-12)
    y_median = df['Median Memory (MB)'].replace(0, 1e-12)
    y_std = df['Std Memory (MB)'].replace(0, 1e-12)

    plt.plot(df['Size'], y_mean, marker='o', label='Mean')
    plt.plot(df['Size'], y_median, marker='s', label='Median')
    plt.plot(df['Size'], y_std, marker='^', label='Std')
    plt.xlabel("Matrix Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title(f"{lang} Benchmark - Memory Usage")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(bottom=0)  # y-axis starts at 0
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f"{lang.lower()}_memory.png"))
    plt.close()

def generate_comparison_graphs(dfs, graphs_dir):
    metrics = ['Mean Time (s)', 'Median Time (s)', 'Mean CPU (%)', 'Median CPU (%)', 'Mean Memory (MB)', 'Median Memory (MB)']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        for lang, df in dfs.items():
            y_values = df[metric].replace(0, 1e-12)
            plt.plot(df['Size'], y_values, marker='o', label=lang)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Matrix Size")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison Across Languages")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(graphs_dir, f"{safe_metric.lower()}_comparison_graph.png"))
        plt.close()

    print(f"[OK] All graphs saved in '{graphs_dir}'")


def plot_results(csv_files=None, graphs_dir="graphs", check_interval=1):
    if csv_files is None:
        csv_files = {
            "Python": "results/python_results.csv",
            "Java": "results/java_results.csv",
            "C": "results/c_results.csv",
            "Rust": "results/rust_results.csv"
        }

    # --- Wait until all CSVs exist ---
    print("⏳ Waiting for all CSV files to be generated...")
    while not all(os.path.exists(f) for f in csv_files.values()):
        time.sleep(check_interval)
    print("✅ All CSV files found!")

    os.makedirs(graphs_dir, exist_ok=True)

    # --- Read each CSV into a DataFrame ---
    dfs = {lang: pd.read_csv(path) for lang, path in csv_files.items()}

    # --- Per-language graphs ---
    for lang, df in dfs.items():
        generate_time_graph(df, lang, graphs_dir)
        generate_cpu_graph(df, lang, graphs_dir)
        generate_memory_graph(df, lang, graphs_dir)

    # --- Comparison graphs ---
    generate_comparison_graphs(dfs, graphs_dir)





#------------------- DIRECTORY SET UP ------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)


#------------------- GENERATE MATRICES ------------------
generate_matrices(MATRIX_SIZES, SEED, MATRIX_DIR)


#------------------- RUN EXECUTABLES ------------------
execute_naive_matrix_multiplication()


#------------------- PLOT RESULTS ------------------
plot_results()
