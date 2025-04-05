import numpy as np

# 加载文件
'''
file1 = np.load('/home/king/Isaac/LEAP_Hand_Sim/leapsim/cache/custom_grasp_cache_grasp_50k_s11.npy')
file2 = np.load('/home/king/Isaac/LEAP_Hand_Sim/leapsim/cache/leap_hand_in_palm_cube_grasp_50k_s11.npy')
'''
file1 = np.load('/home/king/Isaac/LEAP_Hand_Sim/leapsim/runs/pretrained/nn/custom_grasp_cache_grasp_50k_s0964.npy')
#file2 = np.load('/home/king/Isaac/LEAP_Hand_Sim/leapsim/runs/pretrained/nn/custom_grasp_cache_grasp_50k_s090.npy')
file2 = np.load('/home/king/Isaac/LEAP_Hand_Sim/leapsim/runs/pretrained/nn/16384/leap_hand_in_palm_cube_grasp_50k_s09.npy')


# 比较文件内容
if np.array_equal(file1, file2):
    print("两个文件内容完全相同。")
else:
    print("两个文件内容不同。")