import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


print("hera_output/geo_bottleneck-dim0.npy")
print(np.load("hera_output/geom_bottleneck-dim0.npy"))

print("hera_output/geo_bottleneck-dim1.npy")
print(np.load("hera_output/geom_bottleneck-dim1.npy"))

print("hera_output/wasserstein-dim0.npy")
print(np.load("hera_output/wasserstein-dim0.npy"))

print("hera_output/wasserstein-dim1.npy")
print(np.load("hera_output/wasserstein-dim1.npy"))