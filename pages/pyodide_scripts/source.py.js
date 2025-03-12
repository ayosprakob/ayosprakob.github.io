window.SVD_script = `

"""Computes the Singular Value Decomposition (SVD) of a nested list."""

import numpy as np

# The variable passed from JS ::::::::::::::::::::::::::
nested_list = __matrix_input__

matrix = np.array(nested_list)
U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

print("initial matrix:")
print(matrix)

print("\\nU:")
print(U)

print("\\nS:")
print(S)

print("\\nVt:")
print(Vt)

`