import numpy as np
import os


def generate_matrices(matrix_sizes, seed, matrix_dir):
    np.random.seed(seed)
    
    for size in matrix_sizes:
        print(f"Generating matrices of size {size}x{size}...")

        # Generate random integers 0-9
        MATRIX_A = np.random.Generator(0, 10, size=(size, size), dtype=np.float64)
        MATRIX_B = np.random.Generator(0, 10, size=(size, size), dtype=np.float64)

        # File names
        matrix_a_file = os.path.join(matrix_dir, f"A_{size}.bin")
        matrix_b_file = os.path.join(matrix_dir, f"B_{size}.bin")

        # Save as raw binary float64
        MATRIX_A.tofile(matrix_a_file)
        MATRIX_B.tofile(matrix_b_file)

        print(f"Saved {matrix_a_file} and {matrix_b_file}")

    print("\n✅ All matrices generated and saved as binary files.")