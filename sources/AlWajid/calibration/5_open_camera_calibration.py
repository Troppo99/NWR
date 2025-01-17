import numpy as np

with np.load("main/Al-Wajid/calibration/camera_calibration.npz") as X:
    camera_matrix, dist_coeffs = X["camera_matrix"], X["dist_coeffs"]

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
