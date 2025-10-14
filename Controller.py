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
controller_functions.generate_matrices(MATRIX_SIZES, SEED, MATRIX_DIR)

# Compile codes
#controller_functions.compile_codes()

# Execute multiplications
#controller_functions.execute_naive_matrix_multiplication()

# Plot the results

