import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
import numpy as np
def check_equal(a, b):
    return ((np.sum(a==b)==a.shape[0]) and a.shape[0]==b.shape[0])
    #tolerance = 0.001
    #return np.allclose(a, b, atol=tolerance)
'''

data1 = np.loadtxt('./sorted_mc.txt', dtype=int)

data2 = np.loadtxt('./sorted_mortonCode.txt', dtype=int) # his

are_equal = np.array_equal(data1, data2)
print(f'mortonCodes equal: {are_equal}')

data1 = np.loadtxt('./lbvhconinfo.txt', dtype=int)

data2 = np.loadtxt('./lbvh_coninfo.txt', dtype=int) #his

are_equal = np.array_equal(data1, data2)
print(f'lbvh_coninfo equal: {are_equal}')

bvh_aabb = np.loadtxt('./lbvhaabb.txt', dtype=float)

data1 = np.loadtxt('./my_ele_aabb.txt', dtype=float)

data2 = np.loadtxt('./elements.txt', dtype=float) #his

are_equal = np.array_equal(data1, data2)
print(f'elements equal: {are_equal}')

'''
df = pd.read_csv('./lbvh.csv', delimiter='\s+')
df2 = pd.read_csv('./data.csv', delimiter='\s+')
#df = pd.read_csv('./data.csv', delimiter='\s+')
left_array = df['left'].values
right_array = df['right'].values
primitiveIdx_array = df['primitiveIdx'].values
aabb_min_x_array = df['aabb_min_x'].values
aabb_min_y_array = df['aabb_min_y'].values
aabb_min_z_array = df['aabb_min_z'].values
aabb_max_x_array = df['aabb_max_x'].values
aabb_max_y_array = df['aabb_max_y'].values
aabb_max_z_array = df['aabb_max_z'].values

left_array2 = df2['left'].values
right_array2 = df2['right'].values
primitiveIdx_array2 = df2['primitiveIdx'].values
aabb_min_x_array2 = df2['aabb_min_x'].values
aabb_min_y_array2 = df2['aabb_min_y'].values
aabb_min_z_array2 = df2['aabb_min_z'].values
aabb_max_x_array2 = df2['aabb_max_x'].values
aabb_max_y_array2 = df2['aabb_max_y'].values
aabb_max_z_array2 = df2['aabb_max_z'].values

print(f'left :{check_equal(left_array, left_array2)}')
print(f'right :{check_equal(right_array, right_array2)}')
print(f'primitiveIdx :{check_equal(primitiveIdx_array, primitiveIdx_array2)}')
print(f'aabb_min_x :{check_equal(aabb_min_x_array, aabb_min_x_array2)}')
print(f'aabb_min_y :{check_equal(aabb_min_y_array, aabb_min_y_array2)}')
print(f'aabb_min_z :{check_equal(aabb_min_z_array, aabb_min_z_array2)}')
print(f'aabb_max_x :{check_equal(aabb_max_x_array, aabb_max_x_array2)}')
print(f'aabb_max_y :{check_equal(aabb_max_y_array, aabb_max_y_array2)}')
print(f'aabb_max_z :{check_equal(aabb_max_z_array, aabb_max_z_array2)}')

