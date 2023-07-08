import LBVH
import numpy as np

arr = np.array([
  [[0.0, 0.0], [1.0, 1.0]],
  [[0.5, 0.5], [1.5, 1.5]],
  [[1.4, 1.4], [2.4, 2.4]]
], dtype=float)
print(LBVH.find_intersections(arr))

arr = np.array([
  [0.0, 0.0, 1.0, 1.0],
  [0.5, 0.5, 1.5, 1.5],
  [1.4, 1.4, 2.4, 2.4]
], dtype=float)
print(LBVH.find_intersections(arr))

arr = np.random.rand(100000, 2, 3)
arr[:, 1, :] = arr[:, 0, :] + 0.05
print(len(LBVH.find_intersections(arr)))
