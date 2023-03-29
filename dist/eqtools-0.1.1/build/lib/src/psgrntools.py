'''
Written by kfhe, at 12/19/2022
'''

import numpy as np


def pnt_dist(nr, r1, r2, samplratio):
    '''
    Object : 获取水平面非均匀采样情况下，点位的分散距离
    Input  :
        nr         : the discreted number in the given horizontal range
        r1         : the start point of the horizontal range
        r2         : the end point of the horizontal range
        samplratio : the ratio of the maximum interval and the minmum interval
    Output :
        samppnts    : the discreted points
    '''
    dr = 2*(r2-r1)/(nr-1.0)/(1.0 + samplratio)
    samppnts = []
    samppnts.append(r1)
    for i in range(2, nr+1):
        dract = dr*(1.0 + (samplratio-1.0)*(i-2.0)/(nr-2.0))
        samppnts.append(samppnts[-1] + dract)
    samppnts = np.array(samppnts)
    return samppnts


if __name__ == '__main__':
    samppnts = pnt_dist(201, 0, 1500, 12.0)
    print(samppnts)