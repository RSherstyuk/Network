import numpy as np

data = np.array([
  [-2, -1],  
  [25, 6],    
  [17, 4],    
  [-15, -6],  
])
all_y_trues = np.array([
  1, 
  0, 
  0, 
  1, 
])

num_samples = 1000
X = np.tile(data, (num_samples // len(data) + 1, 1))[:num_samples]
y = np.tile(all_y_trues, (num_samples // len(all_y_trues) + 1))[:num_samples]


