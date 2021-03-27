import custom_cav
from glob import glob
import numpy as np
import pickle
import tensorflow as tf

# save_dict = pickle.load(pkl_file)
# mixed8:damselfly_concept5:[1.0, 1.0, 1.0, 1.0, 0.9629629629629629, 0.9629629629629629, 1.0, 0.9629629629629629, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9629629629629629, 0.9629629629629629, 1.0]

cavs = []
accuracies = []
concept = 'damselfly_concept5'
cav_paths = f'./ACE/cavs/{concept}-random500_*-mixed8-linear-0.01.pkl'
for cav_path in glob(cav_paths):

    with tf.io.gfile.GFile(cav_path, 'rb') as pkl_file:
      save_dict = pickle.load(pkl_file)
    
    print(save_dict['concepts'])
    cav_instance = custom_cav.CAV.load_cav(cav_path)
    # print(cav_instance.accuracies)
    accuracies.append(cav_instance.accuracies['overall'])
    print(len(cav_instance.cavs))
    print(len(cav_instance.cavs[0]))
    print(len(cav_instance.cavs[1]))
    sdf

# expect 0.9907407407407407±0.01603750747748963
print(np.mean(accuracies))
print(np.std(accuracies))
print(accuracies)
