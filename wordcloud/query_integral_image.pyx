# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
import array
import numpy as np


def query_integral_image(unsigned int[:,:] integral_image, int size_x, int
                         size_y, random_state):
    cdef int x = integral_image.shape[0]
    cdef int y = integral_image.shape[1]
    cdef int area, i, j
    cdef int hits = 0

    # count how many possible locations
    # 所有可能的矩形起始位置
    for i in xrange(x - size_x):
        # 遍历给定水平位置下可能的垂直位置。
        for j in xrange(y - size_y):
            # 矩形区域的面积使用积分图像计算，从而能够高效地计算矩形区域内像素值的和。
            area = integral_image[i, j] + integral_image[i + size_x, j + size_y]
            area -= integral_image[i + size_x, j] + integral_image[i, j + size_y]
            # 语句检查矩形内像素值的和是否为零，如果是，则将计数器 hits 增加。
            if not area:
                hits += 1
    if not hits:
        # no room left
        return None


    # pick a location at random  随便选一个地点
    # 这段代码是在给定的积分图像（integral_image）中计算大小为 (size_x, size_y)
    # 的非重叠矩形区域中随机选择一个所有像素值都为零（黑色）的矩形的起始位置，并返回这个起始位置。
    cdef int goal = random_state.randint(0, hits) # 首先，goal 变量被赋值为 random_state.randint(0, hits)，hits 在这里代表所有像素值为零的矩形的数量。
    hits = 0 # 另一个计数器 hits 被设置为 0，用于遍历所有可能的矩形起始位置。
    # 遍历所有可能的水平起始位置，
    for i in xrange(x - size_x):
        # 遍历给定水平位置下可能的垂直位置。
        for j in xrange(y - size_y):
            area = integral_image[i, j] + integral_image[i + size_x, j + size_y]
            area -= integral_image[i + size_x, j] + integral_image[i, j + size_y]
            if not area: # 语句检查矩形内像素值的和是否为零
                hits += 1 # 如果是，则将计数器 hits 增加
                if hits == goal:
                    return i, j
