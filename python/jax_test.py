import LBVH
import jax.numpy as np
from jax import random

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

key = random.PRNGKey(42)
mins = random.uniform(key, (100000, 3))
maxs = mins + 0.05;
arr = np.concatenate([mins, maxs], 1)
print(len(LBVH.find_intersections(arr)))
