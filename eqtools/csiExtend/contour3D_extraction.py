import h5py
import numpy as np
#from mayavi import mlab
from pyproj import Proj
import scipy.linalg as slinalg
import pandas as pd
from pandas import Index as idx
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Patch3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl


def contour2DInterpTo3D(xyz, line2d, interp_method='linear'):
    '''
    Interpolate two - dimensional contour lines into three - dimensional ones.

    Args      :
        *
    
    Kwargs    :
        * interp_method: default = 'linear' for griddata

    Return    :
        * line3d in xyz mesh
    '''



    # All Done
    return 


def plot_contourinsurf(x, y, z, t, levels=3, dtype='tri', topo=None, N=None,
                        intptype='linear', intpidx=None, XYZinds=[0, 1, 2]):
    '''
    Calculate the contour lines of the mesh with variables in the nodes.
    Note     :
        * Get the contour lines, and interpolate Z in them, to swap the order of XYZ as desired 
            to ensure the maximum projected area
        * In essence, contour lines are established on the two-dimensional plane closest to the original surface, 
            and then plane contour lines are interpolated to the third spatial position. 
            Finally, coordinate conversion or back projection calculation can be carried out.
    
    Args     :
        * x, y, z    : coordinates of the nodes in the mesh.
        * t          : Values of the nodes, also the variables to be interpolated.
    
    Kwargs   :
        * levels     : int or array-like, optional; levels of contours
            Determines the number and positions of the contour lines / regions.
            If an int n, use MaxNLocator, which tries to automatically choose no more than n+1 
                "nice" contour levels between between minimum and maximum numeric values of Z.
            If array-like, draw contour lines at the specified levels. The values must be in increasing order.
        * dtype      : tri/rect
        * topo       : Topology of the elements for all nodes.
        * N          : Keep.
        * intptype   : linear/nearest et. for method used in the griddata
        * intpidx    : idxs is indes of nodes to be used to the interpolation calculation.
        * XYZinds    : 0 for x, 1 for y, 2 for z; Change the ourput order of (x, y, z); 
                        By default, the last dimension represents the dimension to be interpolated;
                        e.g., [2, 0, 1] is outputing (z, x, y)
    
    Return   :
        * line3d     : list of 3D contours
        * lec        : list of the values for contours
    '''

    from scipy.interpolate import griddata
    
    xyz = np.hstack((x.flatten()[:, None], y.flatten()[:, None], z.flatten()[:, None]))
    # 1. Select two dimensions to calculate and extract contour lines on the plane
    if dtype == 'rect':
        conts = plt.contour(x, y, t, levels=levels, alpha=0.)
        plt.close()
    else:
        import matplotlib.tri as mtri
        triang = mtri.Triangulation(x, y, topo)
        conts = plt.tricontour(triang, t, levels, alpha=0.)
        plt.close() 
    
    if intpidx is None:
        intpidx = np.arange(xyz.shape[0], dtype=np.int_)
        
    # Corresponding to different levels of line segment list, isoline extraction
    allsegs = conts.allsegs
    levels = conts.levels
    intpsegs = []
    for i in range(levels.size):
        intp_segis = []
        n = len(allsegs[i])
        if n>0:
            segis = allsegs[i]

            for segi in segis:
                cx = segi[:, 0]
                cy = segi[:, 1]
                # 2. The two - dimensional contour lines closest to the curved surface fault 
                # are interpolated into a third - dimensional coordinate.
                cz = griddata(xyz[intpidx, :2], xyz[intpidx, 2], (cx, cy), method=intptype)
                cxyz = np.hstack((cx[:, None], cy[:, None], cz[:, None]))
                intp_segis.append(cxyz)
            intpsegs.append(intp_segis)
        else:
            intpsegs.append([])

    # The isoline of the coseismic slip and the corresponding deformation value are obtained
    line3d = []
    lec = []
    for i in range(levels.size):
        segis = intpsegs[i]
        if segis:
            for line in segis:
                line3d.append(line[:, XYZinds])
                lec.append(levels[i])
    
    # All Done.
    # lec： 代表等值线各级值对应的颜色
    return line3d, lec
   

def extract_cline(XYZ, cotopo, mag_slip, invdist=None, triinds=None, XYZinds=[2, 0, 1], srange=[-1, 1.], levels=3, genColorbar=False):
    '''
    Step     :
        * 1. To interpolate the central dislocation of the trig element to the trig point;
        * 2. Some trigonometric elements are extracted according to the demand 
             for interpolation and used to calculate dislocation.
    
    Notes    : 
        * This step only applies if the slip value is at the center of the sub-element.

    Args     :
        * XYZ          : Node coordinates of nodes in the mesh
        * cotopo       : Topology of each sub-element  
        * invdist      : Inverse distance weight between nodes and the centre points of the sub-elements
        * mag_slip     : The variable to be calculated for the contour

    Kwargs   :
        * triinds      : [start, end], The index sequence range in cotopo of the subelement used in the calculation
        * XYZinds      : 0 for x, 1 for y, 2 for z; Change the order of XYZ in the calculation; 
                        By default, the last dimension represents the dimension to be interpolated;
                        e.g., [2, 0, 1] is interpolating y given (z, x)
        * srange       : Keep for generating colorbar
        * levels     : int or array-like, optional; levels of contours
            Determines the number and positions of the contour lines / regions.
            If an int n, use MaxNLocator, which tries to automatically choose no more than n+1 
                "nice" contour levels between between minimum and maximum numeric values of Z.
            If array-like, draw contour lines at the specified levels. The values must be in increasing order.
    
    Return   :
        * (line3d, lec)
            * line3d   : list of 3D contours
            * lec      : list of the values for contours
    '''
    
    if invdist is None:
        XYZ_topo = XYZ[cotopo, :]
        Xc = np.mean(XYZ_topo[:, :, 0], axis=1)
        Yc = np.mean(XYZ_topo[:, :, 1], axis=1)
        Zc = np.mean(XYZ_topo[:, :, 2], axis=1)
        # 计算三角点和三角形中心点之间的距离矩阵；单位为 m
        cXYZ = np.hstack((Xc[:, None], Yc[:, None], Zc[:, None]))
        dist = np.linalg.norm(XYZ[:, None, :] - cXYZ[None, :, :], axis=2)
        dist /= np.mean(dist)
        invdist = 1./dist

    invdist = invdist[:, triinds]
    cotopo = cotopo[triinds, :]
    mag_slip = mag_slip[triinds]
    cotri = []
    # Slip values in the nodes interpolated from the slip in the centers of the sub-elements.
    magtri = []
    for i in range(XYZ.shape[0]):
        cotmp = []
        for k, topo in enumerate(cotopo):
            if i in topo:
                cotmp.append(k)
        magtmp = np.dot(invdist[i, cotmp], mag_slip.flatten()[cotmp])/np.sum(np.asarray(invdist[i, cotmp]))
        magtri.append(magtmp)
        cotri.append(cotmp)

    magtri = np.array(magtri)
    # idxs is indes of nodes with no-nan values.
    idxs = np.where(np.isnan(magtri), False, True)

    # YZX ,插值X
    indx, indy, indz = XYZinds
    XYZinds = np.argsort(XYZinds)
    line3d, lec = plot_contourinsurf(XYZ[:, indx], XYZ[:, indy], 
                                         XYZ[:, indz], magtri[:], 
                                         dtype='tri', topo=cotopo, 
                                         intptype='linear', levels=levels, 
                                         XYZinds=XYZinds, intpidx=idxs)

    # Generates the corresponding RGBA color list
    if genColorbar:
        smin, smax = srange
        norm = mpl.colors.Normalize(smin, smax)
        m = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
        ec = m.to_rgba(lec)
    
    # All Done.
    if genColorbar:
        return line3d, lec, ec
    else:
        return line3d, lec


def calc_crossprod(x, y, z):
    '''
    计算三点的叉积
    '''
    vec1 = np.array([x[1] - x[0],
                    y[1] - y[0],
                    z[1] - z[0]]
                    )
    vec2 = np.array([x[2] - x[0],
                    y[2] - y[0],
                    z[2] - z[0]]
                    )
    return np.cross(vec1, vec2)


from csi import SourceInv
from csi import TriangularPatches
from scipy.interpolate import griddata
from eqtools.gmttools import ReadGMTLines


class Contour3DExtraction(SourceInv):
    '''
    '''
    
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, 
                 source=None, receiver=None, outputfile='contour3d.gmt'):
        super().__init__(name, utmzone, ellps, lon0, lat0)
        if source is None:
            self.source = TriangularPatches('source', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            self.source = source

        if receiver is None:
            self.receiver = TriangularPatches('receiver', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            self.receiver = receiver
        self.outputfile = outputfile
    

    def setReceiver(self, receiver):
        self.receiver = receiver
    
    def setSource(self, source):
        self.source = source
    
    def setOutputfile(self, outputfile):
        self.outputfile = outputfile
    
    def splitReceiver(self, times):
        receiver = self.receiver.duplicateFault()
        for _ in range(times):
            subpatches = []
            for ip in range(len(receiver.patch)):
                ipatch = receiver.patch[ip]
                p1, p2, p3, p4 = receiver.splitPatch(ipatch)
                subpatches.extend(np.array([p1, p2, p3, p4]))
            receiver.patch = subpatches
            # receiver.computeEquivRectangle()
            receiver.patch2ll()
            receiver.initializeslip()
            # Too slow
            # receiver.setVerticesFromPatches()
            # rXYZ = receiver.Vertices
        self.receiver_dense = receiver

        # All Done.
        return
    
    def write2file(self, data='receiver', ZUp=False, outputfile=None):
        if data == 'receiver':
            line3d = self.line3d_in_receiver
        elif data == 'source':
            line3d = self.line3d_in_source
        nsegs = len(line3d)
        lec = self.lec

        if outputfile is not None:
            self.outputfile = outputfile

        with open(self.outputfile, 'wt') as fout:
            for k in range(nsegs):
                print('> -Z{0:.2f}'.format(lec[k]), file=fout)
                lens = len(line3d[k])
                x, y, z = [line3d[k][:, i] for i in range(3)]
                lon, lat = self.putm(x*1000.0, y*1000.0, inverse=True)
                if ZUp:
                    zkm = z 
                else:
                    zkm = -z
                for s in range(lens):
                    print('{0:8.4f} {1:8.4f} {2:8.4f}'.format(lon[s], lat[s], zkm[s]), file=fout)
        
        # All Done
        return
    
    def calContour3d(self, subpatch_st=None, subpatch_ed=None, ZUp=False, write2file=True, outputfile=None):
        # receiver coordinates
        rXYZ = np.asarray(self.receiver.patch).reshape(-1, 3)
        receiver = self.receiver_dense
        # source coordinates
        sourcefault = self.source
        topo = sourcefault.Faces
        vertices = np.asarray(sourcefault.Vertices)
        cXYZ = np.asarray(sourcefault.getcenters())

        sourcefault.computetotalslip()
        slip_norm = sourcefault.totalslip

        # 计算三角点和三角形中心点之间的距离矩阵；单位为 km
        dist = np.linalg.norm(vertices[:, None, :] - cXYZ[None, :, :], axis=2)
        dist /= np.mean(dist)
        invdist = 1./dist
        # 节面子断层个数
        st = 0 if subpatch_st is None else subpatch_st
        ed = sourcefault.slip.shape[0] if subpatch_ed is None else subpatch_ed
        ntris_plane = [st, ed]
        subtris_node = np.cumsum(np.array(ntris_plane))

        # XYZinds表示前两个插值后面一个
        line3d, lec = extract_cline(vertices, topo, slip_norm, invdist=invdist, 
                                    triinds=list(range(subtris_node[0], subtris_node[1])), 
                                    XYZinds=[0, 2, 1], srange=[0, 4.0], levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        self.line3d_in_source = line3d
        self.lec = lec

        self.line3d_in_receiver = []

        if receiver.name != sourcefault.name:
            for k in range(len(lec)):
                lens = len(line3d[k])
                x, y, z = [line3d[k][:, i] for i in range(3)]
                y = griddata(rXYZ[:, [0, 2]], rXYZ[:, 1], (x, z), method='linear').flatten()
                self.line3d_in_receiver.append(np.vstack((x, y, z)).T)
        else:
            self.line3d_in_receiver = self.line3d_in_source
        
        if write2file:
            self.write2file(data='receiver', ZUp=ZUp, outputfile=outputfile)

        # All Done
        return line3d, lec
    
    def plot_contour3d(self, data='receiver', ZUp=False):
        if data == 'receiver':
            line3d = self.line3d_in_receiver
        elif data == 'source':
            line3d = self.line3d_in_source
        lec = self.lec

        # Plot Figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for iline, ic in zip(line3d, lec):
            x, y, z = iline[:, 0], iline[:,1], iline[:, 2]
            if ZUp:
                z *= -1
            ax.plot3D(x, y, z)
        # plt.show()

        # All Done
        return fig
    



if __name__ == '__main__':

    from csi import TriangularPatches

    # 投影信息
    lon0, lat0 = 101.31, 37.80

    # ------------源断层信息-----------------------#
    # # Case 1: Set source fault
    # projstr_trelis = '+proj=utm +lon_0={0:.1f} +lat_0={1:.1f}'.format(lon0, lat0)
    # # p_trelis = Proj(projstr_trelis)
    # vertexfile = 'fault_nodes.inp'
    # topofile = 'fault1.tri'
    # source = TriangularPatches('source', lon0=lon0, lat0=lat0)
    # source.readPatchesFromAbaqus(vertexfile, topofile, projstr=projstr_trelis)
    # slip = pd.read_csv('slip_0.dat', names='ss ds open'.split(), sep=r'\s+')
    # source.initializeslip(values=slip.values)

    # Case 2: Set source fault
    sourcefault = TriangularPatches('source', lon0=lon0, lat0=lat0)
    slipfile = r'output\slip_total_0.gmt'
    sourcefault.readPatchesFromFile(slipfile)

    # ------------接收断层信息-----------------------#
    receiver = TriangularPatches('receiver', lon0=lon0, lat0=lat0)
    slipfile = r'slip_total_1.gmt'
    receiver.readPatchesFromFile(slipfile)

    # 声明提取3d等值线类
    contour3d = Contour3DExtraction('contour3d', lon0=lon0, lat0=lat0, source=sourcefault, receiver=receiver)
    contour3d.splitReceiver(2)
    contour3d.setOutputfile('contour_coseismic_nplane0.gmt')
    # 计算等值线
    line3d, lec = contour3d.calContour3d()
    
    contour3d.plot_contour3d(data='receiver', ZUp=False)