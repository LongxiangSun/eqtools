# coding: utf-8

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import integrate
from numba import jit
from pandas import IndexSlice as idx
from .pscmp_visco_interseismic import *

#----------------------------Calculating Viscoelaxtion driven by afterslip------------#
from scipy.interpolate import interp1d
from numba import complex128,float64,jit, njit

@jit(nopython=True, cache=True)
def cv_cum(t, tobs, disp):
    '''
    Interpolate the viscoelastic relaxation corresponding to the computed node to the observed time.
    
    Args     :
        * t         : Calculation time node of viscoelastic relaxation
        * tobs      : Observation time
        * disp      : the viscoelaxation relaxation at the time nodes corresponding to "t"
    
    Return   :
        * CV        : viscoelastic deformation at time "tobs" caused by coseismic slip 
    '''
    return np.interp(t, tobs, disp)


@jit(nopython=True, cache=True)
def as_cum(t, disp0, tau_as=0.25, alpha=1.0):
    '''
    The deformation due cumulative afterslip is calculated according to the exponential decay function.
    
    Args    :
        * t         : Observation time
        * disp0     : Elastic deformation calculated by Coulomb stress due to coseismic slip

    Kwargs  :
        * tau_as    : characteristic time
        * alpha     : scaling factor for the coulomb stress distribution in each patch

    Return  :
        * AS        : Deformation due to afterslip
    '''
    return alpha*disp0*(1-np.exp(-t/tau_as))

    
@jit(nopython=True, cache=True)
def visco_decay(t, tobs, disp):
    '''
    According to the Green's function of viscoelastic time series, the deformation at the desired time is interpolated

    Args   :
        * t         : Calculation time node of viscoelastic relaxation
        * tobs      : Observation time
        * disp      : the viscoelaxation relaxation at the time nodes corresponding to "t"
    
    Return :
        * DispAtIntptime
    '''
    return np.interp(t, tobs, disp)


@jit(nopython=True, cache=True)
def delta_v(t, tau_as=0.25, alpha=1.0):
    '''
    Calculating the cumulative afterslip at time t
    '''
    return alpha/tau_as*np.exp(-t/tau_as)


@jit(nopython=True, cache=True)
def v_func(t, tobs, disp, tau_as, alpha, T=5):
    '''
    Kernel function for viscoelastic relaxation due to afterslip 
    '''
    return delta_v(t, tau_as, alpha)*visco_decay(T-t, tobs, disp)


def do_integrate(func, ti, args):
    '''
    Calculating AV
    Args    :
        * func     : v_func
        * ti       : Observation time
        * args     : Arguments except t in func.
    
    Return  :
        * AV       : Viscoelastion relaxation due to afterslip.
    '''
    d, err = integrate.quad(func, 0, ti, args=args, epsabs=1e-10, epsrel=1e-15)
    return d, err


def calAS_AV_dir(pscmpts, obsdt, alpha, tau, tunit='D', unit='m', direction='E', onlyAV=True):
    '''
    单站的余滑导致的余滑和余滑导致的黏弹性松弛的计算
    Args     :
        * pscmpts         : 源自extract_pscmp_ts函数, 重新提取到单站上的数据
                            aspscmpts = aspscmpts.loc[idx[:, loc], :]
        * obsdt            : pd.Timedelta对象
        * alpha           : 余滑尺度因子
        * tau             : 余滑松弛因子
    
    Kwargs   :
        * tunit           : tau和pscmpts以及obst - eqdate的求取积分时的默认单位
        * unit            : 欲求取的形变单位，源自extract_pscmp_ts时设置
        * direction       : 欲求取的形变方向
    
    Return   :
        * av_asdisp       : 
    '''
    t, av_asdisp = [], []
    pscmptunit = pscmpts.iloc[0].tunit
    dir_inunit = f'{direction}({unit})'
    obsdt_intunit = obsdt/pd.Timedelta(f'1{tunit}')
    pscmpdt_intunit = pscmpts.index.get_level_values(0).unique()*pd.Timedelta('1{0}'.format(pscmptunit))/pd.Timedelta('1{0}'.format(tunit))

    if onlyAV:
        disp = (pscmpts[dir_inunit]-pscmpts[dir_inunit][0]).values
    else:
        disp = pscmpts[dir_inunit].values

    for ti in obsdt_intunit.values:
        d, _ = do_integrate(v_func, ti, args=(pscmpdt_intunit.values, disp, tau, alpha, ti))
        t.append(ti)
        av_asdisp.append(d)
    t = np.array(t)
    av_asdisp = np.array(av_asdisp)

    # All done 
    return av_asdisp


def asvisco_interg_parallel(rest, pscmpt, dispENU, alpha, tau, k):
    '''
    并行计算余滑导致的黏弹性松弛效应
    '''
    dispE = dispENU[:, 0]
    dispN = dispENU[:, 1]
    dispU = dispENU[:, 2]
    t, as_avdisp = [], []
    for ti in rest: #np.arange(0, 172, 4)
        dE, _ = do_integrate(v_func, ti, args=(pscmpt, dispE, tau, alpha, ti))
        dN, _ = do_integrate(v_func, ti, args=(pscmpt, dispN, tau, alpha, ti))
        dU, _ = do_integrate(v_func, ti, args=(pscmpt, dispU, tau, alpha, ti))
        t.append(ti)
        as_avdisp.append([dE, dN, dU])
    t = np.array(t)
    as_avdisp = np.array(as_avdisp)
    return k, as_avdisp


def calAS_AV(pscmpts, obsdate, eqdate, alpha, tau, unit='m', intp_tunit='Y', onlyAV=True, mcpu=4):
    '''
    计算余滑导致的黏弹性松弛和余滑本身造成的累积形变

    Args    :
        * pscmpts          : 累积余滑分布导致的黏弹性松弛时间函数，提取自pscmp/pylith
                             ，此时应该用未在时间域上插值的pscmpts
        * obsdate          : 观测时间序列. 需要插值的时间节点位置，默认为pd.Timestamp形式的时间序列/pd.DatetimeIndex
                             ,也可是pd.DatetimeIndex的参数
        * alpha            : 余滑幅度值
        * tau              : 余滑松弛因子, tau的单位需要和intp_tunit一致
        * intp_tunit       : 计算时间插值所在单位，默认为Y,可选为D；pscmpts的默认单位在其tunit字段中储存
    
    Kwargs  :
        * onlyAV           : 默认onlyAV=True，表示仅计算余滑导致的黏弹性，而不包括余滑本身，否则两者都包括
        * mcpu             : 并行计算cpu个数，默认为4，None则等于mp.cpu_count()
        * unit             : pscmpts中形变单位，默认为m
    '''
    import multiprocessing as mp
    from tqdm import tqdm
    idx = pd.IndexSlice

    disp_cols = ['{0}({1})'.format(idir, unit) for idir in 'E N U'.split()]
    dispENU = pscmpts[disp_cols]
    simtunit = pscmpts.iloc[0].tunit
    if onlyAV:
        dispENU[disp_cols] -= dispENU.loc[idx[0.0, :], disp_cols].droplevel(0)
    date_index = dispENU.index.get_level_values(0).unique()
    caldtinyr = date_index.to_numpy()
    site_index = dispENU.index.get_level_values(1).unique()

    obsdate = pd.DatetimeIndex(obsdate)
    obsdt = pd.TimedeltaIndex([iobs-eqdate for iobs in obsdate])
    obsdt = (obsdt.days + obsdt.seconds/24.0/3600.0)/365.2425 if intp_tunit == 'Y' else (obsdt.days + obsdt.seconds/24.0/3600.0)
    # print(obsdt)

    pool = mp.Pool(mcpu) if mcpu is not None else mp.cpu_count()
    rows, cols = obsdt.shape[0], site_index.shape[0]
    pscmp_obsts = np.empty((rows, cols, 3), dtype=np.float_)
    pscmp_obsts[:, :, :] = np.nan

    tsim = dispENU.index.get_level_values(0).unique()
    tsim = tsim*pd.Timedelta('1{0}'.format(simtunit))/pd.Timedelta('1{0}'.format(intp_tunit))
    # print(tsim)
    # Join the parallel Pool
    jobs = [pool.apply_async(asvisco_interg_parallel, args=(obsdt.values, tsim.values, dispENU.loc[idx[:, site_index[k]], :].values, alpha, tau, k)) 
                            for k in range(cols)]
        
    pool.close()
    # result_list_tqdm = []
    for job in tqdm(jobs):
        # result_list_tqdm.append(job.get())
        k, as_avdisp = job.get() 
        pscmp_obsts[:, k, :] = as_avdisp
    pool.join()

    # 输出索引index
    obsindex = pd.MultiIndex.from_product((obsdt, site_index), names=['date', 'index'])
    obsdata = pd.DataFrame(pscmp_obsts.reshape((-1, 3)), index=obsindex, columns=disp_cols)
    lonlat = pscmpts.loc[idx[0.0, :], ['Lon[deg]', 'Lat[deg]']]
    lonlat.index = site_index
    obsdata = pd.merge(lonlat, obsdata, left_index=True, right_on=obsdata.index.get_level_values(1))
    obsdata.drop('key_0', axis=1, inplace=True)
    obsdata['tunit'] = intp_tunit

    # All Done
    return obsdata
#-------------------------------------------------------------------------------------#


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # 1.生成待计算时间序列
    nts, nobs = gentsseq_pscmp([95], [100], 2, cycles=9)
    # 2. 生成以上时间序列在pscmp结果文件中对应文件名
    filenames = fmt_pscmp(nobs, truncation=0, screen=False, onlyfilename=True)
    # 3. 用于csi中fault反演的数据集坐标
    sitecoords = np.loadtxt(r'd:\2022Menyuan\Interseismic_InSAR\3DDisp_External\3DDisp\3DDisp2InterInv\InvCode\sites4pscmp.dat').reshape(-1, 2)
    # 4. 提取pscmp生成的时间序列
    data = extract_dispfromPscmpV2(r'e:\Haiyuan_ViscoInterseismic\PSGRN_PSCMP\Maxwell_20km\pscmp_gps_lc1_0e19_regular_LLL\snapshot_10090_year.dat', allsites=False, sitecoords=sitecoords)
    # dirname = r'e:\Haiyuan_ViscoInterseismic\PSGRN_PSCMP\Maxwell_25km\compar_analysis\pscmp_gps_lc1e19u_regular_T1000'
    # pscmpts = extract_pscmp_ts(dirname, filenames=filenames, allsites=False, sitecoords=sitecoords)
    # # 5. 计算对应于每个站点的格林函数
    # vscale = 4e-2 # m/yr
    # viscoG = calviscoGfromPscmp(pscmpts, T=100)/4.0

    dirname = r'e:\Maduo_psgrn_pscmp\Maxwell_32km_crust_mantle\afterslip_driven_visco\pscmp1_0\pscmp_gps_lc2_0e18_regular'
    from glob import glob
    filenames = glob(os.path.join(dirname, 'snapshot_*.dat'))
    filenames = [os.path.basename(ifile) for ifile in filenames]
    sitecoords = np.array([[98.782, 34.648], [98.573, 34.718], [99.196, 34.271]])
    pscmpts = extract_pscmp_ts(dirname, filenames=filenames, allsites=False, sitecoords=sitecoords)
    eqdate = pd.Timestamp('2021-05-21')
    obsdate = pd.DatetimeIndex(['2021-05-26', '2021-06-19', '2021-07-13', '2021-07-25', '2021-08-06', '2021-08-12', '2021-08-18', '2021-08-24'])
    obsdate = pd.TimedeltaIndex(np.arange(0, 170), unit='D') + eqdate
    obsdata = interp_pscmp_ts(pscmpts, obsdate, eqdate=eqdate, mcpu=6)

    import matplotlib.pyplot as plt
    # AS + AV
    as_av = calAS_AV(pscmpts, obsdate, eqdate, alpha=0.091, tau=45.0, unit='m', intp_tunit='D', onlyAV=False, mcpu=4)
    plt.plot(as_av.index.get_level_values(0).unique(), as_av.loc[idx[:, 0], ['E(m)']])
    # AV
    as_av = calAS_AV(pscmpts, obsdate, eqdate, alpha=0.091, tau=45.0, unit='m', intp_tunit='D', onlyAV=True, mcpu=4)
    plt.plot(as_av.index.get_level_values(0).unique(), as_av.loc[idx[:, 0], ['E(m)']])
    plt.show()