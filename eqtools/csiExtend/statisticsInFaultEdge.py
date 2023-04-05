# External libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp2d, interp1d
import sys
import os
import copy

# CSI routines
from csi import TriangularPatches
from csi import SourceInv as csiSourceInv


class StatisticsInFault(csiSourceInv):
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None,
                 source=None, outdir='output_stats'):
        super().__init__(name, utmzone, ellps, lon0, lat0)

        if source is None:
            self.source = TriangularPatches('source', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            self.source = source
        
        self.outdir = outdir
    
    def setSource(self, source):
        self.source = source 

    def setOutdir(self, outdir):
        self.outdir = outdir
    
    def getfaultEdgeInVertices(self, topscale=0.01, bottomscale=0.01):
        '''
        这一步完全是fault.findFaultEdge_Corner该做的工作，后面转过去
        '''
        import copy
        fault = self.source
        edge, corner = fault.findFaultEdge_Corner(topscale=topscale, bottomscale=bottomscale)

        # Top Check
        right_edge = copy.deepcopy(edge['right'])
        corner_vinds = fault.Faces[corner['right_top']].flatten()
        edge_vinds = fault.Faces[right_edge].flatten()
        if np.intersect1d(edge_vinds, corner_vinds).size <= 0:
            tmp = corner['right_top']
            corner['right_top'] = corner['left_top']
            corner['left_top'] = tmp
        #
        #  Bottom Check
        corner_vinds = fault.Faces[corner['right_bottom']].flatten()
        edge_vinds = fault.Faces[right_edge].flatten()
        if np.intersect1d(edge_vinds, corner_vinds).size <= 0:
            tmp = corner['right_bottom']
            corner['right_bottom'] = corner['left_bottom']
            corner['left_bottom'] = tmp
        
        # Get right edge
        right_edge = copy.deepcopy(edge['right'])
        if corner['right_top'] is not None:
            right_edge.append(corner['right_top'])
        if corner['right_bottom'] is not None:
            right_edge.append(corner['right_bottom'])
        right_pnts = fault.Faces[right_edge]
        rpnt_inds = np.unique(np.sort(right_pnts.flatten()))
        # Get left edge
        left_edge = copy.deepcopy(edge['left'])
        if corner['left_top'] is not None:
            left_edge.append(corner['left_top'])
        if corner['left_bottom'] is not None:
            left_edge.append(corner['left_bottom'])
        left_pnts = fault.Faces[left_edge]
        lpnt_inds = np.unique(np.sort(left_pnts.flatten()))
        # Get top edge
        top_edge = copy.deepcopy(edge['top'])
        if corner['left_top'] is not None:
            top_edge.append(corner['left_top'])
        if corner['right_top'] is not None:
            top_edge.append(corner['right_top'])
        top_pnts = fault.Faces[top_edge]
        tpnt_inds = np.unique(np.sort(top_pnts.flatten()))
        # Get bottom edge
        bottom_edge = copy.deepcopy(edge['bottom'])
        if corner['left_bottom'] is not None:
            bottom_edge.append(corner['left_bottom'])
        if corner['right_bottom'] is not None:
            bottom_edge.append(corner['right_bottom'])
        bottom_pnts = fault.Faces[bottom_edge]
        bpnt_inds = np.unique(np.sort(bottom_pnts.flatten()))
        
        # Left, Right, Top, Bottom
        # Left: In West, Right: In East
        if np.min(fault.Vertices[lpnt_inds, 0]) > np.min(fault.Vertices[rpnt_inds, 0]):
            rpnt_inds, lpnt_inds = lpnt_inds, rpnt_inds
            right_edge, left_edge = left_edge, right_edge

        top_edge = np.unique(top_edge)
        bottom_edge = np.unique(bottom_edge)
        left_edge = np.unique(left_edge)
        right_edge = np.unique(right_edge)
        self.edges = {
            'left': left_edge, 
            'right': right_edge, 
            'top': top_edge, 
            'bottom': bottom_edge}
        self.edgepntsInVertices = {
            'left': lpnt_inds, 
            'right': rpnt_inds, 
            'top': tpnt_inds, 
            'bottom': bpnt_inds}

        # All Done
        return

    def getSideEdgeLine(self, side='right', outfile=None, plot=True, threshold=0.2,
                        step=0, rotateAngle=None, write2file=True, sort=True, horz_angle=None,
                        xaxis_ticks=[0, 1], xaxis_scale=1.0, xaxis_zoffset=0.2,
                        yaxis_ticks=[0, 10, 20], tick_scale=0.2):
        '''
        Args     :
            *
        
        Kwargs   :
            * step : + km depend on the dip direction
            * axis_tick_dir : 在x-axis在倾向上向下的旋转坐标系下的角度
        '''
        outdir = self.outdir
        fault = self.source

        if horz_angle is None:
            horz_angle = 90

        side_edge = self.edges[side]
        strike_mean = np.rad2deg(np.mean(fault.getStrikes()[side_edge]))
        if rotateAngle is None:
            arr, cnt = np.unique(np.sort(fault.Faces[side_edge].flatten()), return_counts=True)
            pnts_co = arr[np.argwhere(cnt==2)].flatten() # Bug: 认为只有边上有共点
            pnt_verts = fault.Vertices[pnts_co]
            inds = np.argsort(pnt_verts[:, -1])
            pnt_verts = fault.Vertices[pnts_co[inds]]
            diffy = pnt_verts[1:, 1] - pnt_verts[0, 1]
            diffx = pnt_verts[1:, 0] - pnt_verts[0, 0]
            rotateAngle = -np.mean(np.rad2deg(np.arctan2(diffy, diffx)))
        pnt_inds = self.edgepntsInVertices[side]
        pnt_verts = fault.Vertices[pnt_inds]
        # 边上点的索引
        _, inds1, _ = np.intersect1d(pnt_inds, pnts_co, return_indices=True)

        # Rotation to perpendicular to fault strike
        coord_rot = (pnt_verts[:, 0] + pnt_verts[:, 1]*1.j)*np.exp(1.j*rotateAngle/180*np.pi)
        # # Plot for testing.
        # plt.scatter(pnt_verts[:, 0], pnt_verts[:, 1])
        # plt.show()
        # plt.scatter(coord_rot.real,coord_rot.imag)
        # plt.show()

        x_trans, y_trans = coord_rot.real, coord_rot.imag
        y_mean_inds1 = np.mean(y_trans[inds1])
        y_mean = np.mean(y_trans)


        flag = np.logical_and(y_trans-y_mean_inds1 > -threshold, y_trans-y_mean_inds1<threshold)

        x1, y1 = x_trans[flag], y_trans[flag]
        # 平均值在边线之上，则向下偏移，否则向上偏移
        sign = -np.sign(y_mean - y_mean_inds1)
        # 沿着水平角平移一段距离
        xy_offset = np.ones_like(x1)*step*sign
        step_x, step_y = xy_offset*np.cos(np.deg2rad(horz_angle)), xy_offset*np.sin(np.deg2rad(horz_angle))
        x1 += step_x
        y1 += step_y
        xy = (x1 + y1*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        xy = np.vstack((xy.real, xy.imag)).T

        x, y = xy[:, 0], xy[:, 1]
        z = pnt_verts[flag, -1]
        lon, lat = fault.xy2ll(x, y)
        lonlat = np.vstack((lon, lat, z)).T

        if sort:
            inds = np.argsort(lonlat[:, -1])
            lonlat = lonlat[inds, :]
            x1 = x1[inds]
            y1 = y1[inds]
            z = z[inds]
            x = x[inds]
            y = y[inds]

        
        self.sideedges = {}
        self.sideedges[side] = {
            'rotateAngle': rotateAngle,
            'step'       : step,
            'sign'       : sign,
            'x'          : x,
            'y'          : y,
            'x_trans'    : x1,
            'y_trans'    : y1,
            'depth'      : z,
            'llz'        :lonlat,
            'strike_mean': strike_mean
        }

        # 生成刻度线 Y-axis

        # Y-tick start
        depth = np.array(yaxis_ticks)
        fit = interp1d(z, x1, fill_value="extrapolate")
        x_pred = fit(depth)
        y_pred = np.ones_like(x_pred)*y1.mean()
        xy2 = (x_pred + y_pred*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x2, y2, z2 = xy2.real, xy2.imag, depth
        lon2, lat2 = fault.xy2ll(x2, y2)
        lonlat_st = np.vstack((lon2, lat2, z2)).T
        # Y-tick end
        xy_offset = np.ones_like(lon2)*tick_scale*sign
        step_x, step_y = xy_offset*np.cos(np.deg2rad(horz_angle)), xy_offset*np.sin(np.deg2rad(horz_angle))
        xy2 = ((x_pred+step_x) + (y_pred+step_y)*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x2, y2, z2 = xy2.real, xy2.imag, depth
        lon2, lat2 = fault.xy2ll(x2, y2)
        lonlat_ed = np.vstack((lon2, lat2, z2)).T

        # 生成刻度 X-axis  


        if outfile is None:
            outfile = 'fault_{}edge_{}.dat'.format(side, self.source.name)
            outfile_xaxis = 'fault_{}edge_{}_xaxis.dat'.format(side, self.source.name)
            outfile_yaxis = 'fault_{}edge_{}_yaxis.dat'.format(side, self.source.name)
        if write2file:
            lonlat = pd.DataFrame(lonlat, columns=['lon', 'lat', 'depth'])
            lonlat.to_csv(os.path.join(outdir, outfile), sep=' ', float_format='%.6f', index=False, header=False)
            # Write Y-axis
            with open(os.path.join(outdir, outfile_yaxis), 'wt') as fout:
                for ist, ied in zip(lonlat_st, lonlat_ed):
                    print('>', file=fout)
                    print('{0:.3f} {1:.3f} {2:.1f}'.format(*ist), file=fout)
                    print('{0:.3f} {1:.3f} {2:.1f}'.format(*ied), file=fout)

        if plot:
            plt.plot(xy[:, 0], xy[:, 1])
            plt.show()
        # All Done
        return lonlat

    def genXaxis(self, side='right', xaxis_ticks=[0.3, 4.3], xaxis_scale=1.0, xaxis_zoffset=0.5, 
                 horz_angle=None, vert_angle=None, tick_scale=0.2):
        '''
        xaxis_ticks : 以转换后平行于倾向为x轴的相对坐标
        axis_tick_dir: 以转换后平行于倾向为x轴的相对角度
        self.sideedges = {
            'rotateAngle': rotateAngle,
            'step'       : step,
            'sign'       : sign,
            'x'          : x,
            'y'          : y,
            'x_trans'    : x1,
            'y_trans'    : y1,
            'depth'      : z,
            'llz'        :lonlat,
            'strike_mean': strike_mean
        }
        '''
        outdir = self.outdir
        fault = self.source

        sideedges = self.sideedges[side]
        rotateAngle = sideedges['rotateAngle']
        z, x1 = sideedges['depth'], sideedges['x_trans']
        y1 = sideedges['y_trans']
        sign = sideedges['sign']

        if horz_angle is None:
            horz_angle = 90
        if vert_angle is None:
            vert_angle = 0

        # Rotation axis_angle
        # 地表块体的倾角
        dip_angle_0 = np.rad2deg(np.arctan2(z[1]-z[0], x1[1]-x1[0]))
        dip_angle = vert_angle - 90 + dip_angle_0

        # Offset 
        hdep = np.array([-xaxis_zoffset, -xaxis_zoffset])

        xy_offset = np.array(xaxis_ticks)*sign
        x_offset, y_offset = xy_offset*np.cos(np.deg2rad(horz_angle)), xy_offset*np.sin(np.deg2rad(horz_angle))
        x_st = x1[0] + x_offset
        y_st = y1.mean() + y_offset
        z_st = np.ones_like(y_st)*z[0]
        # Rotate across Y-axis
        zx2_vert = (z_st + x_st*1.j)*np.exp(1.j*np.deg2rad(dip_angle))

        x_st = zx2_vert.imag
        y_st = y_st
        z_st = hdep + zx2_vert.real
        z_ed = z_st + tick_scale*sign

        # Roation to back
        ## llz_st
        xy_st = (x_st + y_st*1.j)
        zx2_st = (z_st + xy_st.real*1.j)*np.exp(-1.j*np.deg2rad(dip_angle))
        xy2_st = (zx2_st.imag + xy_st.imag*1.j)*np.exp(-1.j*np.deg2rad(rotateAngle))

        xyz_st = np.vstack((xy2_st.real, xy2_st.imag, zx2_st.real)).T
        lon, lat = fault.xy2ll(xyz_st[:, 0], xyz_st[:, 1])
        llz_st = np.vstack((lon, lat, xyz_st[:, -1])).T
        ## llz_ed
        xy_ed = (x_st + y_st*1.j)
        zx2_ed = (z_ed + xy_ed.real*1.j)*np.exp(-1.j*np.deg2rad(dip_angle))
        xy2_ed = (zx2_ed.imag + xy_ed.imag*1.j)*np.exp(-1.j*np.deg2rad(rotateAngle))

        xyz_ed = np.vstack((xy2_ed.real, xy2_ed.imag, zx2_ed.real)).T
        lon, lat = fault.xy2ll(xyz_ed[:, 0], xyz_ed[:, 1])
        llz_ed = np.vstack((lon, lat, xyz_ed[:, -1])).T

        outfile_xaxis = 'fault_{}edge_{}_xaxis.dat'.format(side, self.source.name)
        with open(os.path.join(outdir, outfile_xaxis), 'wt') as fout:
            print('>', file=fout)
            for illz in llz_st:
                print('{0:.6f} {1:.6f} {2:.1f}'.format(*illz), file=fout)
            
            # ticks
            for ist, ied in zip(llz_st, llz_ed):
                print('>', file=fout)
                print('{0:.6f} {1:.6f} {2:.1f}'.format(*ist), file=fout)
                print('{0:.6f} {1:.6f} {2:.1f}'.format(*ied), file=fout)
        

        # Keep coordinate system
        sideedges['horz_angle'] = horz_angle
        sideedges['dip_angle'] = dip_angle
        sideedges['xaxis_zoffset'] = xaxis_zoffset
        sideedges['xaxis_ticks'] = xaxis_ticks
        sideedges['xaxis_xyz']  = xyz_st
        sideedges['xaxis_xyz1'] = np.vstack((x_st, y_st, z_st)).T

        # All Done
        return

    def statSlipAlongDepth(self, slip='total', interval=2.0):
        fault = self.source
        # Cal Norm slip
        slip_pd = pd.DataFrame(fault.slip, columns=['ss', 'ds', 'open'])
        slip_pd['total'] = np.sqrt(slip_pd.eval('ss**2 + ds**2'))
        if slip == 'total':
            slip = slip_pd.total.values
        elif slip == 'strikeslip':
            slip = slip_pd.ss.values
        elif slip == 'dipslip':
            slip = slip_pd.ds.values
        data = pd.DataFrame(np.hstack((slip.flatten()[:, None], np.array([ic[-1] for ic in fault.getcenters()])[:, None])), columns=['slip', 'z'])
        zcut = pd.cut(data.z.values.flatten(), np.arange(fault.top, data.z.values.max()+0.1, interval)) # ind.mid ind.left ind.right
        slip_dep = data.groupby(zcut).mean()
        slip_dep = slip_dep/slip_dep.max()[0]
        ind = pd.IntervalIndex(slip_dep.index)
        z_dep = ind.mid.values
        return slip_dep.slip.values, z_dep

    def StatInfoInSide(self, side='right', slip='total', interval=2.0, curvelist=[], histlist=[]):
        '''
        沿深度方向的信息统计
        '''
        outdir = self.outdir
        fault = self.source

        sideedges = self.sideedges[side]
        rotateAngle = sideedges['rotateAngle']
        z, x1 = sideedges['depth'], sideedges['x_trans']
        y1 = sideedges['y_trans']
        sign = sideedges['sign']
        horz_angle = sideedges['horz_angle']
        step0, steped = sideedges['xaxis_ticks']
        step = steped - step0

        # 统计滑动沿深度信息
        hslp, hdep = self.statSlipAlongDepth(slip=slip, interval=interval)
        print(hslp, hdep)

        fit = interp1d(z, x1)
        z_pred = hdep
        x_pred = fit(z_pred)
        y_pred = np.ones_like(x_pred)*y1.mean()
        xy_offset = (step0 + step*hslp)*sign*np.exp(1.j*np.deg2rad(horz_angle))
        x_pred += xy_offset.real
        y_pred += xy_offset.imag

        xy2 = (x_pred + y_pred*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x2, y2, z2 = xy2.real, xy2.imag, z_pred
        lon2, lat2 = fault.xy2ll(x2, y2)
        lonlat2 = np.vstack((lon2, lat2, z2)).T

        outfile_slp = 'fault_{}edge_{}_slip.dat'.format(side, self.source.name)
        lonlat2 = pd.DataFrame(lonlat2, columns=['lon', 'lat', 'depth'])
        lonlat2.to_csv(os.path.join(outdir, outfile_slp), index=False, header=False, float_format='%.6f')

        # All Done
        return # zcut, slip_dep
    
    def StatHistInSide(self, value=None, depth=None, bins=15, side='right', outfile='stat_hist.gmt'):

        outdir = self.outdir
        fault = self.source

        sideedges = self.sideedges[side]
        rotateAngle = sideedges['rotateAngle']
        z, x1 = sideedges['depth'], sideedges['x_trans']
        y1 = sideedges['y_trans']
        sign = sideedges['sign']
        horz_angle = sideedges['horz_angle']
        step0, steped = sideedges['xaxis_ticks']
        step = steped - step0

        acums, adeps = np.histogram(depth, bins=bins)
        acums = acums / acums.max() # Norm

        ## 生成柱状图
        adeps_hist = np.repeat(adeps, 2)[1:-1]
        acums_hist = np.repeat(acums, 2)

        fit = interp1d(z, x1)
        z_pred = adeps_hist
        x_pred = fit(z_pred)
        y_pred = np.ones_like(x_pred)*y1.mean()
        xy_offset = step0*sign*np.exp(1.j*np.deg2rad(horz_angle))
        x_pred += xy_offset.real
        y_pred += xy_offset.imag

        xy2 = (x_pred + y_pred*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x2, y2, z2 = xy2.real, xy2.imag, z_pred
        lon2, lat2 = fault.xy2ll(x2, y2)
        lonlat2 = np.vstack((lon2, lat2, z2)).T

        xy_offset = step*acums_hist*sign*np.exp(1.j*np.deg2rad(horz_angle))
        x_pred += xy_offset.real
        y_pred += xy_offset.imag
        xy3 = (x_pred + y_pred*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x3, y3, z3 = xy3.real, xy3.imag, adeps_hist
        lon3, lat3 = fault.xy2ll(x3, y3)
        lonlat3 = np.vstack((lon3, lat3, z3)).T

        # Write2file
        with open(os.path.join(outdir, outfile), 'wt') as fout:
            for i in range(acums.shape[0]):
                print('>', file=fout)
                st1, st2 = lonlat2[2*i:2*i+2]
                st4, st3 = lonlat3[2*i:2*i+2]
                for st in [st1, st2, st3, st4]:
                    print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
        
        # All Done
        return
    
    def StatCurveInSide(self, value, depth, side='right', zrange=None, method='mean',
                        zinterval=2, outfile='curve_stat.gmt'):
        '''
        Args      :
            * depth    : depth in km, +
        Kwargs    :
            * zrange   : [fault.top, fault.maxdepth] if None 
        '''
        fault = self.source

        zcut = pd.cut(depth, np.arange(fault.top, depth.max()+0.1, zinterval)) # ind.mid ind.left ind.right
        data = pd.DataFrame(value, columns=['value'])
        if method == 'mean':
            slip_dep = data.groupby(zcut).mean()
        elif method == 'sum':
            slip_dep = data.groupby(zcut).sum()
        slip_dep = slip_dep/slip_dep.max()[0]
        ind = pd.IntervalIndex(slip_dep.index)
        z_dep = ind.mid.values
        slip_dep = slip_dep.value.values
        # 以上是数据切割部分

        outdir = self.outdir
        fault = self.source

        sideedges = self.sideedges[side]
        rotateAngle = sideedges['rotateAngle']
        z, x1 = sideedges['depth'], sideedges['x_trans']
        y1 = sideedges['y_trans']
        sign = sideedges['sign']
        horz_angle = sideedges['horz_angle']
        step0, steped = sideedges['xaxis_ticks']
        step = steped - step0

        # 统计滑动沿深度信息
        hslp, hdep = slip_dep, z_dep

        fit = interp1d(z, x1)
        z_pred = hdep
        x_pred = fit(z_pred)
        y_pred = np.ones_like(x_pred)*y1.mean()
        xy_offset = (step0 + step*hslp)*sign*np.exp(1.j*np.deg2rad(horz_angle))
        x_pred += xy_offset.real
        y_pred += xy_offset.imag

        xy2 = (x_pred + y_pred*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x2, y2, z2 = xy2.real, xy2.imag, z_pred
        lon2, lat2 = fault.xy2ll(x2, y2)
        lonlat2 = np.vstack((lon2, lat2, z2)).T

        lonlat2 = pd.DataFrame(lonlat2, columns=['lon', 'lat', 'depth'])
        lonlat2.to_csv(os.path.join(outdir, outfile), index=False, header=False, float_format='%.6f')

        # All Done
        return 
    
    def statinTop(self, value=None, lonlat=None, hinterval=2.0, slip='total', statkind='curve', height_scale=1.0,
                  bins=15, method='mean', cutmethod='pdcut', discretizeInterval=0.2, depth_eps=0.25):
        '''
        所有的数据沿断层的统计
        statkind  : curve, hist, bar
            * curve  : 输出最大值位置的曲线
            * bar    : 竖线
            * hist   : 柱形图模式
        cutmethod : pdcut, hist; 柱形显示的分割方式
            * pdcut   : hinterval来设置区间
            * hist    : bins来设置水平区间
        '''
        outdir = self.outdir
        fault = self.source
        side = 'top'

        top_edge = self.edges[side]
        top_vert_inds = self.edgepntsInVertices[side]
        top_vert_inds = fault.Faces[top_edge]
        top_verts_finds = top_vert_inds.flatten()
        top_verts = fault.Vertices[top_verts_finds]
        # 顶部三角形按照x坐标排序
        sort_ind = np.argsort(np.mean(fault.Vertices[top_vert_inds, :], axis=1)[:, 0])
        top_edge_sort = np.array(top_edge)[sort_ind]

        top_slip = fault.slip[top_edge_sort, :]
        top_slip_norm = np.linalg.norm(top_slip, axis=1)

        # 深度判定flag
        top = fault.top
        flag = top_verts[:, -1] < top + depth_eps
        finds = top_verts_finds[flag]
        # 顶部三角形顶边顶点按照x坐标排序
        sort_ind = np.argsort(fault.Vertices[finds][:, 0])
        finds_sort = finds[sort_ind]

        strikes = fault.getStrikes()[top_edge_sort]
        dips = fault.getDips()[top_edge_sort]

        if vert_angle is None:
            vert_angle = 0

        if lonlat is not None:
            # Bug: 用经度判断位置
            top_cnts_ll = np.mean(fault.Vertices_ll[fault.Faces[top_edge_sort], :], axis=1)
            int_indx = np.searchsorted(top_cnts_ll[:, 0], lonlat[:, 0])
            print(int_indx.shape)

            strikes = fault.getStrikes()[top_edge_sort]
            dips = fault.getDips()[top_edge_sort]

            # Rotation axis_angle
            dips = vert_angle + dips

            # 计算坐标
            fault.discretize(every=discretizeInterval)
            dis_trace = fault.cumdistance(discretized=True)

            # Cal Hist
            x, y = fault.ll2xy(lonlat[:, 0], lonlat[:, 1])
            xy = np.vstack((x, y)).T
            xy_trace = np.vstack((fault.xi, fault.yi)).T
            dist_mat = np.linalg.norm(xy_trace[None, :, :] - xy[:, None,:], axis=2)
            ind_dist = np.argsort(dist_mat, axis=1)[:, 0]
            print(dist_mat.shape, ind_dist.shape)
            # 数据距离断层迹线起始点的距离
            disti = dis_trace[ind_dist]

            if value is None:
                acums, adists = np.histogram(disti, bins=bins)
                acums_hist = np.repeat(acums, 2)
                adists_hist = np.repeat(adists, 2)[1:-1]
                inds = np.searchsorted(dis_trace, adists_hist)
                # 这是重新插值repeat后的坐标
                xi, yi = fault.xi[inds], fault.yi[inds]
                zi = np.ones_like(xi)*fault.top
                value = acums_hist
            else:
                zcut = pd.cut(disti, np.arange(fault.top, disti.max()+0.1, hinterval)) # ind.mid ind.left ind.right
                data = pd.DataFrame(value, columns=['value'])
                if method == 'mean':
                    slip_stk = data.groupby(zcut).mean()
                elif method == 'sum':
                    slip_stk = data.groupby(zcut).sum()
                slip_stk = slip_stk/slip_stk.max()[0]
                ind = pd.IntervalIndex(slip_stk.index)
                x_st, x_ed = ind.left.values, ind.right.values
                x_bins = np.vstack((x_st, x_ed)).T.flatten()
                inds = np.searchsorted(dis_trace, x_bins)
                slip_stk = slip_stk.value.values
                xi, yi = fault.xi[inds], fault.yi[inds]
                zi = np.ones_like(xi)*fault.top
                value = np.repeat(slip_stk, 2)
                # 聚类后需要用聚类后的坐标来提取dips和strikes
                x_mid = ind.mid.values
                inds = np.searchsorted(dis_trace, x_mid)
                xm = fault.xi[inds]
                centers = np.asarray(fault.getcenters())
                inds2 = np.searchsorted(centers[top_edge_sort, 0], xm)
                strikes = np.repeat(strikes[inds2], 2)
                dips = np.repeat(dips[inds2], 2)

            top_x, top_z = value*np.cos(dips)*height_scale, value*np.sin(dips)*height_scale
            verts = np.vstack((xi, yi, zi)).T
            trans_verts = (verts[:, 0] + verts[:, 1]*1.j)*np.exp(1.j*strikes)
        else:
            # Rotation axis_angle
            dips = vert_angle + dips
            strikes = np.repeat(strikes, 2)
            dips = np.repeat(dips, 2)

            top_slps = np.repeat(top_slip_norm, 2)
            top_slps /= top_slps.max()
            top_x, top_z = top_slps*np.cos(dips)*height_scale, top_slps*np.sin(dips)*height_scale

            verts = fault.Vertices[finds_sort, :]
            trans_verts = (verts[:, 0] + verts[:, 1]*1.j)*np.exp(1.j*strikes)
        
        # All Done
        return top_x, top_z, trans_verts

    def StatHistinTop(self, value=None, lonlat=None, hinterval=2.0, slip='total', bins=15, side='top', depth_eps=0.25, 
                      zoffset=0.2, hight_scale=1.0, vert_angle=None, method='mean',
                      outfile='stat_histInTop.gmt', discretizeInterval=0.2):
        '''
        self.edges = {
            'left': left_edge, 
            'right': right_edge, 
            'top': top_edge, 
            'bottom': bottom_edge}
        self.edgepntsInVertices = {
            'left': lpnt_inds, 
            'right': rpnt_inds, 
            'top': tpnt_inds, 
            'bottom': bpnt_inds}
        '''
        outdir = self.outdir
        fault = self.source

        top_edge = self.edges[side]
        top_vert_inds = self.edgepntsInVertices[side]
        top_vert_inds = fault.Faces[top_edge]
        top_verts_finds = top_vert_inds.flatten()
        top_verts = fault.Vertices[top_verts_finds]
        # 顶部三角形按照x坐标排序
        sort_ind = np.argsort(np.mean(fault.Vertices[top_vert_inds, :], axis=1)[:, 0])
        top_edge_sort = np.array(top_edge)[sort_ind]

        top_slip = fault.slip[top_edge_sort, :]
        top_slip_norm = np.linalg.norm(top_slip, axis=1)

        # 深度判定flag
        top = fault.top
        flag = top_verts[:, -1] < top + depth_eps
        finds = top_verts_finds[flag]
        # 顶部三角形顶边顶点按照x坐标排序
        sort_ind = np.argsort(fault.Vertices[finds][:, 0])
        finds_sort = finds[sort_ind]

        strikes = fault.getStrikes()[top_edge_sort]
        dips = fault.getDips()[top_edge_sort]

        if vert_angle is None:
            vert_angle = 0

        if lonlat is not None:
            # Bug: 用经度判断位置
            top_cnts_ll = np.mean(fault.Vertices_ll[fault.Faces[top_edge_sort], :], axis=1)
            int_indx = np.searchsorted(top_cnts_ll[:, 0], lonlat[:, 0])
            print(int_indx.shape)

            strikes = fault.getStrikes()[top_edge_sort]
            dips = fault.getDips()[top_edge_sort]

            # Rotation axis_angle
            dips = vert_angle + dips

            # strikes = strikes[int_indx]
            # dips = dips[int_indx]
            # # Repeat for Hist
            # strikes = np.repeat(strikes, 2)
            # dips = np.repeat(dips, 2)
            # print(dips.shape, strikes.shape)

            # 计算坐标
            fault.discretize(every=discretizeInterval)
            dis_trace = fault.cumdistance(discretized=True)

            # Cal Hist
            x, y = fault.ll2xy(lonlat[:, 0], lonlat[:, 1])
            xy = np.vstack((x, y)).T
            xy_trace = np.vstack((fault.xi, fault.yi)).T
            dist_mat = np.linalg.norm(xy_trace[None, :, :] - xy[:, None,:], axis=2)
            ind_dist = np.argsort(dist_mat, axis=1)[:, 0]
            print(dist_mat.shape, ind_dist.shape)
            # 数据距离断层迹线起始点的距离
            disti = dis_trace[ind_dist]

            if value is None:
                acums, adists = np.histogram(disti, bins=bins)
                acums_hist = np.repeat(acums, 2)
                adists_hist = np.repeat(adists, 2)[1:-1]
                inds = np.searchsorted(dis_trace, adists_hist)
                # 这是重新插值repeat后的坐标
                xi, yi = fault.xi[inds], fault.yi[inds]
                zi = np.ones_like(xi)*fault.top
                value = acums_hist
            else:
                zcut = pd.cut(disti, np.arange(fault.top, disti.max()+0.1, hinterval)) # ind.mid ind.left ind.right
                data = pd.DataFrame(value, columns=['value'])
                if method == 'mean':
                    slip_stk = data.groupby(zcut).mean()
                elif method == 'sum':
                    slip_stk = data.groupby(zcut).sum()
                slip_stk = slip_stk/slip_stk.max()[0]
                ind = pd.IntervalIndex(slip_stk.index)
                x_st, x_ed = ind.left.values, ind.right.values
                x_bins = np.vstack((x_st, x_ed)).T.flatten()
                inds = np.searchsorted(dis_trace, x_bins)
                slip_stk = slip_stk.value.values
                xi, yi = fault.xi[inds], fault.yi[inds]
                zi = np.ones_like(xi)*fault.top
                value = np.repeat(slip_stk, 2)
                # 聚类后需要用聚类后的坐标来提取dips和strikes
                x_mid = ind.mid.values
                inds = np.searchsorted(dis_trace, x_mid)
                xm = fault.xi[inds]
                print(top_edge_sort)
                centers = np.asarray(fault.getcenters())
                inds2 = np.searchsorted(centers[top_edge_sort, 0], xm)
                strikes = np.repeat(strikes[inds2], 2)
                dips = np.repeat(dips[inds2], 2)

            top_x, top_z = value*np.cos(dips)*hight_scale, value*np.sin(dips)*hight_scale
            verts = np.vstack((xi, yi, zi)).T
            trans_verts = (verts[:, 0] + verts[:, 1]*1.j)*np.exp(1.j*strikes)
        else:
            # Rotation axis_angle
            dips = vert_angle + dips
            strikes = np.repeat(strikes, 2)
            dips = np.repeat(dips, 2)

            top_slps = np.repeat(top_slip_norm, 2)
            top_x, top_z = top_slps*np.cos(dips)*hight_scale, top_slps*np.sin(dips)*hight_scale

            verts = fault.Vertices[finds_sort, :]
            trans_verts = (verts[:, 0] + verts[:, 1]*1.j)*np.exp(1.j*strikes)

        # 起始位置偏移项
        step0 = zoffset
        step0_x, step0_z = step0*np.cos(dips), step0*np.sin(dips)

        # 底部顶点
        trans_verts -= step0_x
        z = verts[:, -1] - step0_z
        xy = trans_verts*np.exp(-1.j*strikes)
        x0, y0 = xy.real, xy.imag
        lon, lat = fault.xy2ll(x0, y0)
        lonlatz1 = np.vstack((lon, lat, z)).T

        # 顶部顶点
        trans_verts -= top_x
        z -= top_z
        xy = trans_verts*np.exp(-1.j*strikes)
        x0, y0 = xy.real, xy.imag
        lon, lat = fault.xy2ll(x0, y0)
        lonlatz2 = np.vstack((lon, lat, z)).T

        with open(os.path.join(outdir, outfile), 'wt') as fout: # slp_stats_hist_Tip
            for i in range(int(lon.shape[0]/2.0)):
                print('>', file=fout)
                st1, st2 = lonlatz1[2*i:2*i+2]
                st4, st3 = lonlatz2[2*i:2*i+2]
                for st in [st1, st2, st3, st4]:
                    print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
        
        if not hasattr(self, 'sideedges'):
            self.sideedges = {}
        self.sideedges[side] = {
            'top_edge_sort': top_edge_sort,
            'strikes_edge_sort': strikes,
            'dips_edge_sort': dips
        }
        
        # All Done 
        return

    def StatCurveinTop(self, value, lonlat, side='top', zrange=None, method='mean', depth_eps=0.25, 
                      zoffset=0.2, hight_scale=1.0, vert_angle=None, outfile='stat_CurveInTop.gmt', outkind='bar'):
        '''
        outkind  : bar/curve
        Bug: 如果断层反倾得话，vert_angle会导致想不同方向偏转
        '''
        outdir = self.outdir
        fault = self.source
        top_edge_sort = self.sideedges[side]['top_edge_sort']
        # 排序并且寻找临近子元的走向角和倾向角
        top_cnts_ll = np.mean(fault.Vertices_ll[fault.Faces[top_edge_sort], :], axis=1)
        int_indx = np.searchsorted(top_cnts_ll[:, 0], lonlat[:, 0])

        strikes = fault.getStrikes()[top_edge_sort]
        dips = fault.getDips()[top_edge_sort]

        if vert_angle is None:
            vert_angle = 0

        # Rotation axis_angle
        dips = vert_angle + dips

        strikes = strikes[int_indx]
        dips = dips[int_indx]

        slp_x, slp_z = value*np.cos(dips)*hight_scale, value*np.sin(dips)*hight_scale

        # 起始位置偏移项
        step0 = zoffset
        step0_x, step0_z = step0*np.cos(dips), step0*np.sin(dips)

        vx, vy = fault.ll2xy(slp_pan.lon.values, slp_pan.lat.values)
        vz = np.zeros_like(vx)
        verts = np.vstack((vx, vy, vz)).T
        trans_verts = (verts[:, 0] + verts[:, 1]*1.j)*np.exp(1.j*strikes)

        # 底部顶点
        trans_verts -= step0_x
        z = verts[:, -1] - step0_z
        xy = trans_verts*np.exp(-1.j*strikes)
        x0, y0 = xy.real, xy.imag
        lon, lat = fault.xy2ll(x0, y0)
        lonlatz1 = np.vstack((lon, lat, z)).T

        # 顶部顶点
        trans_verts -= slp_x
        z -= slp_z
        xy = trans_verts*np.exp(-1.j*strikes)
        x0, y0 = xy.real, xy.imag
        lon, lat = fault.xy2ll(x0, y0)
        lonlatz2 = np.vstack((lon, lat, z)).T

        if outkind == 'bar':
            with open(os.path.join(outdir, outfile), 'wt') as fout:
                for i in range(lonlatz1.shape[0]):
                    print('>', file=fout)
                    st1 = lonlatz1[i, :]
                    st3 = lonlatz2[i, :]
                    for st in [st1, st3]:
                        print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
        else:
            with open(os.path.join(outdir, outfile), 'wt') as fout:
                print('>', file=fout)
                for i in range(lonlatz2.shape[0]):
                    st = lonlatz2[i, :]
                    print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    from collections import OrderedDict

    # -----------------------------------Proj Information-------------------------------------#
    # center for local coordinates--M7.1 epicenter 
    lon0, lat0 = 101.31, 37.80

    # -------------------------Generate Triangular Fault Object-------------------------------#
    # Case 2:
    source = TriangularPatches('Menyuan_main', lon0=lon0, lat0=lat0)
    slipfile = r'slip_total_0.gmt'
    source.readPatchesFromFile(slipfile)
    # 设置地表迹线
    trace = os.path.join('..', 'Fault_Trace_Menyuan_Yanghongfeng_scale.txt')
    trace = pd.read_csv(trace, names=['lon', 'lat'], sep=r'\s+', comment='#')
    source.trace(trace.lon.values, trace.lat.values)

    statobj = StatisticsInFault('statinfault', lon0=lon0, lat0=lat0, source=source)
    statobj.getfaultEdgeInVertices()

    lonlat = statobj.getSideEdgeLine(plot=False, step=0.5, yaxis_ticks=[0, 5, 10, 15], horz_angle=120, tick_scale=0.5)
    # 为了让统计信息沿断层面，vert_angle通常不要设置，即不旋转倾向角
    statobj.genXaxis(xaxis_ticks=[0.8, 4.2], tick_scale=1.0, horz_angle=120, vert_angle=None, xaxis_zoffset=0.7)
    # slip in total
    statobj.StatInfoInSide(interval=1.5)

    data = pd.read_csv(r'd:\2022Menyuan\2022Menyuan\RelocatedAftershocks\FanLiping\Proj2Fault\seis_reloc_proj.gmt', sep=r'\s+')
    statobj.StatHistInSide(None, data.dep.values, bins=15, side='right')

    statobj.StatCurveInSide(np.ones_like(data.mag.values), data.dep.values, zrange=None, zinterval=1.5, method='sum')

    statobj.StatHistinTop(np.ones_like(data.mag.values), data[['lon', 'lat']].values, bins=15, hinterval=2, side='top', depth_eps=0.25, vert_angle=0,
                      zoffset=0.2, hight_scale=4.0, outfile='stat_histInTop_test.gmt', method='sum')

    # 现场勘查
    slp_file = r'c:\Users\kfhe\Desktop\Menyuan_Surface\surfslip_panjiawei.csv'
    slp_pan = pd.read_csv(slp_file, sep=r'\s+')

    statobj.StatCurveinTop(slp_pan.slip.values, slp_pan[['lon', 'lat']].values, side='top', zrange=None, method='mean', depth_eps=0.25, 
                      zoffset=0.2, hight_scale=1.0, vert_angle=0, outfile='curve_statInTop.gmt', outkind='curve')
    statobj.StatCurveinTop(slp_pan.slip.values, slp_pan.iloc[:, 2:].values, side='top', zrange=None, method='mean', depth_eps=0.25, 
                      zoffset=0.2, hight_scale=1.0, vert_angle=0, outfile='bar_statInTop.gmt', outkind='bar')