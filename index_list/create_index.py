#!/usr/bin/env python

import numpy as np

###
### Variables to modify
###
save_file_name = "/sps/t2k/eleblevec/updated_watchmal/index_lists/taille_100_000/train55k_val70k_test100k.npz" # you have to include .npz
nb_events = 99000
keys = ['train_idxs', 'val_idxs', 'test_idxs'] # ['train_idxs', 'val_idxs', 'test_idxs']
indexs = [0, 55_000, 70_000, nb_events-1] # [train_first_index, val_first_index, test_first_index, num_events=test_last_index-1]


###
### No modifications needed below
###
rng = np.random.default_rng()
a = rng.permutation(range(nb_events))

res = [a[indexs[i]:indexs[i+1]] for i in range(len(indexs) - 1)]
assert len(keys) == len(res), f"Lenght of keys and res should match. Got {len(keys)} for keys vs {len(res)} for res"
    
res_dict = {}
for name, ind_list in zip(keys, res):
    res_dict[name] = ind_list

np.savez(save_file_name, **res_dict)

print("Done")
print(f"Index list saved at : {save_file_name}")
