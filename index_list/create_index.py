#!/usr/bin/env python

import numpy as np

###
### Variables to modify
###
save_file_name = "/sps/t2k/eleblevec/updated_watchmal/WatChMaL/index_list/demo_index/test_new_create_index.npz" # you have to include .npz
keys = ['train_idxs', 'val_idxs', 'test_idxs'] # ['train_idxs', 'val_idxs', 'test_idxs']

# indexs[-1] should always be < len(datasets)
indexs = [0, 10_000, 30_000, 38800] # [train_first_index, val_first_index, test_first_index]


###
### No modifications needed below
###

rng = np.random.default_rng()
a = rng.permutation(range(indexs[0], indexs[-1]))

res = [a[( indexs[i] - indexs[0] ):( indexs[i+1] - indexs[0] )] for i in range(len(indexs)-1)]
assert len(keys) == len(res), f"Lenght of keys and res should match. Got {len(keys)} for keys vs {len(res)} for res"
    
res_dict = {}
for name, ind_list in zip(keys, res):
    res_dict[name] = ind_list

np.savez(save_file_name, **res_dict)

print("Done")
print(f"Index list saved at : {save_file_name}")
