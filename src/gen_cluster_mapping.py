import numpy as np

mapping = np.load('data/mapping.npy', allow_pickle=True)
new_mapping = {}

count = 0
for ind, i in enumerate(mapping):
	if ind in [0, 2, 40]:
	    new_mapping[i] = 0

	if ind in [4,6,9,16,20,34,36]:
	    new_mapping[i] = 1

	if ind in [32, 48]:
	    new_mapping[i] = 2

	if ind in [17, 37, 44]:
	    new_mapping[i] = 3

	if ind in [1,18]:
	    new_mapping[i] = 4

	if ind in [8, 30]:
	    new_mapping[i] = 5

	if ind in [10, 13]:
	    new_mapping[i] = 6

	if ind in [24, 35]:
	    new_mapping[i] = 7

	if ind in [3, 5, 7, 11, 12, 14, 15, 19, 21, 22,23, 25, 26, 27, 28, 29, 31, 33, 38, 33 , 40, 41, 45, 46, 47]:
	    new_mapping[i] = ind+8
	    count += 1
np.save("clustered_mapping.npy", new_mapping, allow_pickle=True)
