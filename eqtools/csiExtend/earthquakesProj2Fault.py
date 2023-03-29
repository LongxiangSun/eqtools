from csi import seismiclocations
from csi import RectangularPatches as Rect
from csi import TriangularPatches as Tri
from csi import SourceInv

import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import os
import copy


class EqseqProj2Fault(SourceInv):
    '''
    '''
    
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, 
                 seismic=None, receiver=None, outputfile='earthquakes_projed.dat'):
        super().__init__(name, utmzone, ellps, lon0, lat0)
        if seismic is None:
            self.seismic = seismiclocations('seismic', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            self.seismic = seismic
        
        if receiver is None:
            self.receiver = Tri('receiver', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            self.receiver = receiver
        
        self.outputfile = outputfile
    
    def setReceiver(self, receiver):
        self.receiver = receiver
    
    def setSeismic(self, seismic):
        self.seismic = seismic
    
    def setSeismicFromFile(self, seisfile, header=0):
        '''
          Y  m  D  H  M  Sec          Lat   Lon      Dep    Mag
        '''
        seis_info = pd.read_csv(seisfile, sep=r'\s+', skiprows=header, comment='#')
        datetime = seis_info.apply(lambda x: '{0:4d}-{1:02d}-{2:02d}T{3:02d}:{4:02d}:{5:.3f}'.format(*[int(i) for i in x.values[0:5]], x.values[6]), axis=1)
        seis_info['datetime'] = pd.to_datetime(datetime, format='%Y-%m-%dT%H:%M:%S.%f')

        seis = self.seismic

        seis.time = seis_info.datetime.values
        seis.lon = seis_info.Lon.values
        seis.lat = seis_info.Lat.values
        seis.mag = seis_info.Mag.values
        seis.depth = seis_info.Dep.values
        seis.lonlat2xy()

        # All Done
        return
    
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

        # All Done
        return
    
    def write2file(self, outputfile=None, cotime=None):
        if outputfile is not None:
            self.outputfile = outputfile
        
        eqtime = self.seis_proj.time
        if cotime is not None:
            dt = eqtime - cotime
            dt_days = dt.apply(lambda x: x.days + x.seconds/3600./24.)
            # 投影信息输出
            self.seis_proj.write2file(outputfile, add_column=dt_days)
        else:
            self.seis_proj.write2file(outputfile)

        # All Done
        return 

    def calproj(self, write2file=False):

        seis = self.seismic
        receiver = self.receiver_dense

        # The patch number nearest to the fault is obtained successively
        ipatch = seis.getClosestFaultPatch(receiver)

        # Create a list of patch centers
        Centers = [receiver.getpatchgeometry(i, center=True)[:3] for i in ipatch]

        seis_llh = []
        for i in range(len(Centers)):
            lon, lat = receiver.putm(Centers[i][0]*1000, Centers[i][1]*1000, inverse=True)
            seis_llh.append([lon, lat, Centers[i][2]])

        seis_llh = pd.DataFrame(seis_llh, columns='lon lat dep'.split())

        seis_proj = copy.deepcopy(self.seismic)
        seis_proj.lon = seis_llh.lon.values
        seis_proj.lat = seis_llh.lat.values
        seis_proj.depth = seis_llh.dep.values
        seis_proj.lonlat2xy()

        self.seis_proj = seis_proj

        if write2file:
            self.write2file(self.outputfile)

        # All Done 
        return


if __name__ == '__main__':

    # 投影信息
    lon0, lat0 = 101.5, 37.5

    # Building the receiver fault
    slipfile = r'slip_total_0.gmt'
    outfile = os.path.join('.', 'seis_reloc_proj_test.gmt')
    receiver = Tri('Menyuan', utmzone=None, ellps='WGS84', verbose=True, lon0=lon0, lat0=lat0)
    receiver.readPatchesFromFile(slipfile, readpatchindex=False)

    # Build a seismiclocations object
    ## Case 1
    seisfile = r'relocated_seismic.txt'
    # seis = seismiclocations('seis_reloc', lon0=lon0, lat0=lat0)
    # seis.read_from_Hauksson(seisfile, header=4)
    # lon, lat = seis.lat, seis.lon
    # seis.lon = lon
    # seis.lat = lat
    # seis.lonlat2xy()

    # Define the projection object
    eqprojobj = EqseqProj2Fault('eqproj', utmzone=None, ellps='WGS84', 
                                lon0=lon0, lat0=lat0, receiver=receiver, outputfile=outfile)
    # Dense receiver
    eqprojobj.splitReceiver(3)

    # eqprojobj.setSeismic(seis)

    # Case 2
    eqprojobj.setSeismicFromFile(seisfile, header=3)

    # Select the time range and magnitude range, as well as the space range
    # minlon, maxlon, minlat, maxlat = [101.008153, 101.203804, 37.629900, 37.822300]
    # seis.selectbox(minlon, maxlon, minlat, maxlat, depth=100000., mindep=0.0)
    # seis.selecttime(start=[2001, 1, 1], end=[2101, 1, 1])
    eqprojobj.seismic.selectmagnitude(minimum=0, maximum=6.8)

    # Projection and output
    eqprojobj.calproj(write2file=True)