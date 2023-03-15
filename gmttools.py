# %% import libs
import pandas as pd
import numpy as np
import re
import io


def ReadGMTLines(gmtfile, comment='#', names=['X', 'Y'], encoding='utf-8', readZ=False):
    '''
    Input    : 
        linefile   : GMT line segments file
    Output   :
        lineList   : A list for line list
    Added by kfhe, at 01/08/2023
    '''

    with open(gmtfile, 'rt', encoding=encoding) as fin:
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


def WriteLines2GMT(segs, zval, gmtfile=None, csimode=False, coordtrunc=3, ztrunc=1):
    '''
    Args   :
        * segs       : List of pandas.DataFrame
        * zval       : List of zvals
    
    Kwargs :
        * gmtfile    : outfile in gmt format; if None print to Screen.
        * csimode    : Keep.
        * coordtrunc : Precision of coordinate
        * ztrunc     : Precision of zval
    '''
    ndim = segs[0].shape[1]
    coordpat = ''
    if gmtfile:
        fout = open(gmtfile, 'wt')
    else:
        fout = None
    for i in range(ndim):
        coordpat += '{' + f'{i:d}' + ':.' + f'{coordtrunc:d}'+ 'f} '
    # coordpat = '{0:.3f} {1:.3f} {2:.3f}'
    zpat = '> -Z{0:.' + f'{ztrunc:d}'+ 'f}'
    for iseg, iz in zip(segs, zval):
        print(zpat.format(iz), file=fout)
        iseg = iseg if iseg.__class__ in (np.ndarray,) else iseg.values
        for ipnt in iseg:
            print(coordpat.format(*ipnt), file=fout)
    
    if gmtfile:
        fout.close()
    
    # All Done
    return


if __name__ == '__main__':
    import os
    dirname = r'tests'
    faultfile = r'readgmtlines_test.txt' # 'Haiyuan_Relative_fault.dat'
    linefile = os.path.join(dirname, faultfile)

    segments = ReadGMTLines(linefile)
# %%