import numpy as np

arr = np.uint8(np.random.rand(7,7)*1.2)

print arr

ind_x, ind_y = np.nonzero(arr)

for n in range(len(ind_x)):
    rand_x = np.int8(np.random.rand() * 3) -1
    rand_y = np.int8(np.random.rand() * 3) -1
    while rand_x == 0 and rand_y == 0:
        rand_x = np.int8(np.random.rand() * 3) -1
        rand_y = np.int8(np.random.rand() * 3) -1
    new_x = ind_x[n]+rand_x
    new_y = ind_y[n]+rand_y
    if new_x < 0 or new_x >= arr.shape[0]:
        new_x = ind_x[n]-rand_x
    if new_y < 0 or new_y >= arr.shape[1]:
        new_y = ind_y[n]-rand_y

    arr[new_x, new_y] = 2

print arr
