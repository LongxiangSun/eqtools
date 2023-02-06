# %% import libs
import pandas as pd
import numpy as np
import re
import io


def ReadGMTLines(linefile, comment='#', names=['X', 'Y'], encoding='utf-8', readZ=False):
    '''
    Input    : 
        linefile   : GMT line segments file
    Output   :
        lineList   : A list for line list
    Added by kfhe, at 01/08/2023
    '''

    with open(linefile, 'rt', encoding=encoding) as fin:
        linestr = fin.read()

    # remove comment content
    comment = '#'
    line_trap, count = re.subn('(?<!.)(' + comment + '.*(\\r\\n|\\n)){1,}?', r'', linestr)
    # Split > to different string
    line_segs = re.split('>.*?(?:\\r\\n|\\n)', line_trap, flags=re.S)
    # Extract value following the Mark Z
    if readZ:
        zvals = np.array([float(izval) for izval in re.findall('>.*?-Z ?([-0-9.]+)', line_trap, flags=re.S)])
    # remove null string
    line_segs = [seg for seg in line_segs if seg]
    
    # line segments
    segments = []
    for seg in line_segs:
        segments.append(pd.read_csv(io.StringIO(seg), sep=r'\s+', names=names))
    # Remove empty DataFrame
    if segments[0].empty:
        segments = segments[1:]
    
    # All Done
    if readZ:
        return segments, zvals
    else:
        return segments


if __name__ == '__main__':
    import os
    dirname = r'tests'
    faultfile = r'readgmtlines_test.txt' # 'Haiyuan_Relative_fault.dat'
    linefile = os.path.join(dirname, faultfile)

    segments = ReadGMTLines(linefile)
# %%