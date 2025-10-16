import controller_functions
import os

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
#controller_functions.generate_matrices(MATRIX_SIZES, SEED, MATRIX_DIR)

# Run executables
controller_functions.run_file()

# Plot the results

