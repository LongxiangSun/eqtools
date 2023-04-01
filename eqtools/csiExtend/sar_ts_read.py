import numpy as np
import pandas as pd
import h5py
import os
from math import floor, ceil
from csi import insartimeseries as csiinsartimeseries
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

class mysarts(csiinsartimeseries):
    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True, lon0=None, 
                 lat0=None, h5filedict=None, dirname=None, downsample=1):
        
        # Base class init
        super(mysarts,self).__init__(name,
                                     utmzone=utmzone,
                                     ellps=ellps,
                                     lon0=lon0, 
                                     lat0=lat0,
                                     verbose=False) 
        
        self.h5filedict = {
            'avgSpatialCoh': 'geo_avgSpatialCoh.h5',
            'geoRadar': 'geo_geometryRadar.h5',
            'maskTempCoh': 'geo_maskTempCoh.h5',
            'temporalCoherence': 'geo_temporalCoherence.h5',
            'velocity': 'geo_velocity.h5',
            'timeseries': 'geo_timeseries_demErr.h5'
        }

        self.downsample = downsample

        if h5filedict is not None:
            self.h5filedict.update(h5filedict)
        
        if dirname is not None:
            self.dirname = dirname
        else:
            self.dirname = None
        
    
    def setdirname(self, dirname):
        self.dirname = dirname
    
    def updateh5filedict(self, h5filedict):
        self.h5filedict.update(h5filedict)
    
    def setGeometry(self, shadowMask=False, downsample=1, keeprowtimeseries=True):

        dirname = self.dirname
        filenames = self.h5filedict
        self.downsample = downsample

        # keys: ['azimuthAngle', 'height', 'incidenceAngle', 'latitude', 
        #        'longitude', 'shadowMask', 'slantRangeDistance']
        with h5py.File(os.path.join(dirname, filenames['geoRadar']), 'r+') as geoRadar:
            azimuthAngle = geoRadar['azimuthAngle'][:]
            incidenceAngle = geoRadar['incidenceAngle'][:]
            longitude = geoRadar['longitude'][:]
            latitude = geoRadar['latitude'][:]
            if shadowMask:
                shadowMask = geoRadar['shadowMask'][:]

        minlon, maxlon = np.nanmin(longitude), np.nanmax(longitude)
        minlat, maxlat = np.nanmin(latitude), np.nanmax(latitude)
        self.rawcoordrange = [minlon, maxlon, minlat, maxlat]
        
        ysize, xsize = longitude.shape
        lon = np.linspace(minlon, maxlon, xsize)
        lat = np.linspace(maxlat, minlat, ysize)
        meshlon, meshlat = np.meshgrid(lon, lat)

        if keeprowtimeseries:
            self.rawmeshlon = meshlon
            self.rawmeshlat = meshlat

        self.setLonLat(meshlon.flatten()[::downsample], meshlat.flatten()[::downsample], 
                        incidence=incidenceAngle.flatten()[::downsample], 
                        heading=azimuthAngle.flatten()[::downsample], dtype=np.float32)

        # All Done
        return

    def extractTimeSeries(self, mask=True, downsample=1, factor=1, keeprowtimeseries=True):

        dirname = self.dirname
        filenames = self.h5filedict
        self.factor = factor
        # Extract SAR time series
        # keys: ['bperp', 'date', 'timeseries']
        with h5py.File(os.path.join(dirname, filenames['timeseries']), 'r+') as ts:
            timeseries = ts['timeseries'][:]
            dateseries = pd.DatetimeIndex(pd.to_datetime(ts['date'][:], format='%Y%m%d'))
            # bperp = ts['bperp'][:]
            pydates = [ts.to_pydatetime() for ts in dateseries]

        if mask:
            with h5py.File(os.path.join(dirname, filenames['maskTempCoh']), 'r+') as maskTempCoh:
                mask = maskTempCoh['mask'][:]
            # mask sar image
            sar_ts = []
            for ts in timeseries:
                ts[mask == False] = np.nan
                sar_ts.append(ts.flatten()[::downsample]*factor)
        
        if keeprowtimeseries:
            self.rawtimeseries = timeseries

        self.initializeTimeSeries(time=pydates, dtype=np.float32)

        return self.setTimeSeries(sar_ts)

    def read_from_h5file(self, dirname=None, h5filedict=None, factor=1.0, mask=True, 
                         downssample=1, shadowMask=False, keeprowtimeseries=True):
        '''
        
        Args      :
            * dirname         :
            * sarfile_pattern : dict

        Kwargs   :
            * factor          : 形变缩放单位

        Return
            * None
        '''
        if dirname is not None:
            self.setdirname(dirname)
        self.setGeometry(shadowMask=shadowMask, downsample=downsample)
        self.extractTimeSeries(mask=mask, downsample=downsample, factor=factor, keeprowtimeseries=keeprowtimeseries)
    
        # All Done
        return
    
    def cutrawts(self):
        #----------根据预设数据范围提取数据--------------#
        cornerlon = self.rawcoordrange[0]
        cornerlat = self.rawcoordrange[-1]
        nx, ny = self.rawmeshlon.shape
        minlon, maxlon, minlat, maxlat = self.rawcoordrange
        dx = (maxlon - minlon)/(ny-1)
        dy = (-maxlat + minlat)/(nx-1)

        llow = floor((lonrange[0] - cornerlon)/dx) + 0
        rlow = ceil((lonrange[1] - cornerlon)/dx) + 0
        trow = floor((latrange[0] - cornerlat)/dy) + 0
        brow = ceil((latrange[1] - cornerlat)/dy) + 0
        # 限定下边界不能小于0
        if llow < 0:
            llow = 0
        if brow < 0:
            brow = 0


        # tdim, xdim, ydim = timeseries.shape
        # keepimgs = np.array([True]*tdim, dtype=bool)
        # 例如：移除7/16/2019年对应的索引序号
        # keepimgs[1] = False
        # 稀疏采样间隔
        interval = 1
        tsts = self.rawtimeseries[:, brow: trow:interval, llow: rlow:interval]
        # dateseries = dateseries[keepimgs]

        # masks = mask[brow: trow:interval, llow: rlow:interval]
        lon = self.rawmeshlon[brow: trow:interval, llow: rlow:interval]
        lat = self.rawmeshlat[brow: trow:interval, llow: rlow:interval]

        self.rawmeshlon = lon
        self.rawmeshlat = lat
        self.rawtimeseries = tsts

        # All Done
        return
    
    def plotinmpl(self, coordrange=None, faults=None, rawdownsample4plot=100, factor4plot=100, 
                  vmin=None, vmax=None, symmetry=True):
        endsar = self.rawtimeseries[-1, ::rawdownsample4plot, ::rawdownsample4plot] * factor4plot
        if coordrange is not None:
            extent = coordrange
        else:
            extent = self.rawcoordrange
        rvmax = vmax if vmax is not None else np.nanmax(endsar)
        rvmin = vmin if vmin is not None else np.nanmin(endsar)
        scale = factor4plot
        if symmetry:
            vmax = np.max([np.abs(rvmin), rvmax])
            vmin = -vmax
        else:
            vmax = rvmax
            vmin = rvmin
        cmap = mpl.cm.jet
        #设定每个图的colormap和colorbar所表示范围是一样的，即归一化
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.imshow(endsar, cmap=cmap, vmin=vmin, vmax=vmax,
                    origin='upper', extent=extent)
        if faults is not None:
            for ifault in faults:
                ax.plot(ifault.lon, ifault.lat, color='black', lw=0.5)
        plt.colorbar(mappable=mapper, ax=ax, aspect=20, shrink=0.45, label='Disp. (cm)')
        plt.show()


if __name__ == '__main__':

    ## 选取数据范围
    lonrange = [99.46, 102.762]
    latrange = [36.59, 38.942]

    # %% Building a SAR Timeseries Object
    # 5 m(距离向) * 15 m
    downsample = 50 # 10

    # center for local coordinates--M7.4 epicenter
    lon0 = 101.31
    lat0 = 37.80
    sarts_menyuan = mysarts('Menyuan', utmzone=None, lon0=lon0, lat0=lat0)
    sarts_menyuan.setdirname(os.path.join('..', '..', 'geo'))
    sarts_menyuan.read_from_h5file(factor=1.0, mask=True, downssample=100)

    # Image Display
    # Fault Trace
    main_rupture = pd.read_csv('../Main.txt', names=['lon', 'lat'], sep=r'\s+')
    tip_rupture = pd.read_csv('../Second.txt', names=['lon', 'lat'], sep=r'\s+')

    from csi import RectangularPatches as csiRect
    main_fault = csiRect('main', lon0=lon0, lat0=lat0)
    main_fault.trace(main_rupture.lon.values, main_rupture.lat.values)
    main_fault.discretize()
    sec_fault = csiRect('sec', lon0=lon0, lat0=lat0)
    sec_fault.trace(tip_rupture.lon.values, tip_rupture.lat.values)
    sec_fault.discretize()
    faults = [main_fault, sec_fault]


    # hypocenter: 98.38 34.86 Strike: 285
    hypolon = 101.428
    hypolat = 37.736
    strike = 110
    sarts_menyuan.getProfiles('hypo', loncenter=hypolon, latcenter=hypolat, length=60, azimuth=strike-90, width=5, verbose=True)
    sarts_menyuan.smoothProfiles('hypo', window=0.25, method='mean')
    sarts_menyuan.plotProfiles('Smoothed hypo', color='b')

    sarts_menyuan.plotinmpl(vmin=-2, vmax=2, rawdownsample4plot=1, faults=faults)

    sartmp = sarts_menyuan.timeseries[-1]
    name = 'hypo {}'.format(sartmp.name)
    # sartmp.smoothProfile(name, window=0.25, method='mean')
    # sartmp.plotprofile(name, norm=[-0.02, 0.02]) # Smoothed {}
    sartmp.plotprofile('Smoothed {}'.format(name), norm=[-0.02, 0.02])


    # %% 提取特定点位附近的insar点信号
    from csi import gps as csigps, gpstimeseries as csigpstimeseries

    samp_gps = csigps('Sampling_Points', utmzone=None, lon0=lon0, lat0=lat0)
    pnt_stat_filename = os.path.join('..', 'sampling_points_lonlat.dat')
    samp_gps.setStatFromFile(pnt_stat_filename, initVel=False, header=1)
    # samp_gps.initializeTimeSeries(time=sarts_menyuan.time, los=True, verbose=True)
    # 采样点大小需要斟酌， distance： x Km
    samp_pnts = sarts_menyuan.extractAroundGPS(samp_gps, distance=2, doprojection=False, reference=False)

    # %% 将采样剖面存储下来
    # sarts_menyuan.writeProfiles2Files('hypo', 'samp', )
