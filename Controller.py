import os
import numpy as np
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Global constants
MATRIX_SIZES = [10, 100] #[10, 100, 1_000, 10_000]
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
    #run_file(fr"C:/Python313/python.exe c:/Users/valko/VSCodeProjects/big-data_task_1/{file_name}.py", "Python")
    #run_file(fr"java -Xmx4G {file_name}", "Java")
    run_file(fr".\c_{file_name}.exe", "C")
    #run_file(fr".\rust_{file_name}.exe", "Rust")
    return 0

def plot_results(csv_file="results/python_results.csv", graphs_dir="graphs"):
    """
    Reads benchmark results from a CSV file and generates plots.
    
    - One graph per language: shows Mean, Median, and Std over matrix sizes.
    - Three comparison graphs: Mean, Median, and Std across all languages.
    
    Parameters:
        csv_file (str): Path to the CSV file with benchmark results.
        graphs_dir (str): Directory where graphs will be saved.
    """
    # Ensure output directory exists
    os.makedirs(graphs_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_file)

    # Get list of languages
    languages = df['Language'].unique()

    # --- One graph per language ---
    for lang in languages:
        df_lang = df[df['Language'] == lang]
        plt.figure(figsize=(8, 6))
        plt.plot(df_lang['Size'], df_lang['Mean Time (s)'], marker='o', label='Mean')
        plt.plot(df_lang['Size'], df_lang['Median Time (s)'], marker='s', label='Median')
        plt.plot(df_lang['Size'], df_lang['Std Dev (s)'], marker='^', label='Std Dev')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Matrix Size")
        plt.ylabel("Time (s)")
        plt.title(f"{lang} Benchmark")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{lang}_benchmark.png"))
        plt.close()

    # --- Comparison graphs across languages ---
    metrics = ['Mean Time (s)', 'Median Time (s)', 'Std Dev (s)']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        for lang in languages:
            df_lang = df[df['Language'] == lang]
            plt.plot(df_lang['Size'], df_lang[metric], marker='o', label=lang)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Matrix Size")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison Across Languages")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(graphs_dir, f"comparison_{safe_metric}.png"))
        plt.close()

    print(f"✅ All graphs saved in '{graphs_dir}'")




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
