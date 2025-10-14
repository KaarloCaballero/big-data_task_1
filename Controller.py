import subprocess
import os
import time
import controller_functions

# Global constants
MATRIX_SIZES = [10, 100, 1_000, 10_000]
RESULTS_DIR = "results"
GRAPHS_DIR = "graphs"
MATRIX_DIR = "matrices"
SEED = 1

# Directory setup
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)

# Generate matrices
controller_functions.generate_matrices(MATRIX_SIZES, SEED, MATRIX_DIR)

# 

def run_cmd(cmd, lang):
    print(f"\n=== Running {lang} experiment ===")
    start = time.time()
    subprocess.run(cmd, shell=True, check=True)
    end = time.time()
    print(f"{lang} finished in {end - start:.2f} seconds")

# 1. Compile (only once)
print("\n=== Compiling binaries ===")
subprocess.run("gcc -O3 -fopenmp matrix_mul.c -o matrix_mul_c", shell=True, check=True)
subprocess.run("cargo build --release", shell=True, check=True)
subprocess.run("javac MatrixMul.java", shell=True, check=True)

# 2. Execute all experiments using binary matrices
# Pass the matrix file names as command-line arguments to each program
run_cmd(f"./matrix_mul_c {MATRIX_A_FILE} {MATRIX_B_FILE}", "C")
run_cmd(f"target/release/matrix_mul_rust {MATRIX_A_FILE} {MATRIX_B_FILE}", "Rust")
run_cmd(f"java MatrixMul {MATRIX_A_FILE} {MATRIX_B_FILE}", "Java")
run_cmd(f"python matrix_mul.py {MATRIX_A_FILE} {MATRIX_B_FILE}", "Python")

# 3. Move outputs into results directory
for lang in ["C", "Rust", "Java", "Python"]:
    src = f"{lang.lower()}_results.csv"
    if os.path.exists(src):
        os.rename(src, os.path.join(RESULTS_DIR, src))

print("\n✅ All experiments completed! Results saved in:", RESULTS_DIR)



