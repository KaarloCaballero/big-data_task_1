import numpy as np
import os
import subprocess
import time

def generate_matrices(matrix_sizes, seed, matrix_dir):
    print("\n=== Generating matrices ===")
    rng = np.random.default_rng(seed)  # ✅ Usar el generador correctamente
    
    os.makedirs(matrix_dir, exist_ok=True)  # Asegura que el directorio exista
    
    for size in matrix_sizes:
        print(f"\nGenerating matrices of size {size}x{size}...")

        # ✅ Generar enteros aleatorios entre 0 y 9 (inclusive)
        MATRIX_A = rng.integers(0, 10, size=(size, size), dtype=np.int32)
        MATRIX_B = rng.integers(0, 10, size=(size, size), dtype=np.int32)

        # Archivos de salida
        matrix_a_file = os.path.join(matrix_dir, f"A_{size}.bin")
        matrix_b_file = os.path.join(matrix_dir, f"B_{size}.bin")

        # ✅ Guardar como binario
        MATRIX_A.tofile(matrix_a_file)
        MATRIX_B.tofile(matrix_b_file)

        print(f"Saved {matrix_a_file} and {matrix_b_file}")

    print("\n✅ All matrices generated and saved as binary files.")

def run_file(cmd, lang):
    print(f"\n=== Running {lang} experiment ===")
    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    end = time.time()
    print(f"{lang} finished in {end - start:.2f} seconds")

def save_results_in_csv():
    for lang in ["C", "Rust", "Java", "Python"]:
    src = f"{lang.lower()}_results.csv"
    if os.path.exists(src):
        os.rename(src, os.path.join(RESULTS_DIR, src))

        #& C:/Python313/python.exe c:/Users/valko/VSCodeProjects/big-data_task_1/NaiveMatrixMultiplication.py

print("\n✅ All experiments completed! Results saved in:", RESULTS_DIR)

def execute_naive_matrix_multiplication():
    run_file(f"./matrix_mul_c {MATRIX_A_FILE} {MATRIX_B_FILE}", "C")
    run_file(f"target/release/matrix_mul_rust {MATRIX_A_FILE} {MATRIX_B_FILE}", "Rust")
    run_file(f"java MatrixMul {MATRIX_A_FILE} {MATRIX_B_FILE}", "Java")
    run_file(f"python matrix_mul.py {MATRIX_A_FILE} {MATRIX_B_FILE}", "Python")
    return 0
