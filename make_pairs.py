import numpy as np
import os
import random
random.seed(1337)

root = "../dtd/images/"
for i in range(4000//50):
    left = []
    right = []
    all_paths = [os.listdir('root'+d) for d in os.listdir(root)]
    pairs = []
    pair_type = random.choice(['same','diff'])
    if pair_type == 'same':
            left = []
            right = []
            category = random.choice(range(47))
        for i in range(50):
            left_sample = random.choice(all_paths[category])
            all_paths[category].remove(left_sample)
            right_sample = random.choice(all_paths[category])
            all_paths[category].remove(right_sample)
            left.append(left_sample)
            right.append(right_sample)
