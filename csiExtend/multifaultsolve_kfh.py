from csi import multifaultsolve
import copy
import numpy as np
import pyproj as pp


def getsign(rake, bound=-1):
    '''
    bound: -1表示下界，1表示上界
    '''
    sign = np.ones_like(rake)
    if bound == -1:
        mark = (rake > -90) & (rake<90)
        sign[mark] *= -1.
    elif bound == 1:
        mark = (rake < -90) | (rake > 90)
        sign[mark] *= -1.
    return sign


def makeRakeBoundMat(rakeseq, rampmark=None):
    '''
    cnt_rake = cnt_subfault
    Output:
        Ax <= 0的A矩阵；
        x: [ss, ds]排列的未知参数
    Input:
        rakeseq: [(-45., 45.), [-180.， 180.],...] 逆时针为正， 上下界旋转方向为逆时针
        rake角之间的interval不能大于180.， 且不能提供90.的rake角作为边界
    Initial: 8/19/2020 10:00, by kefenghe
    Modify: 
    '''
    cnt_subfault = rakeseq.shape[0]
    # 定下界设计矩阵
    lowerrake = rakeseq[:, 0]
    slope = np.tan(lowerrake/180.*np.pi)
    # print(slope)
    lowerboundss = -slope*np.eye(cnt_subfault)
    lowerboundds = np.eye(cnt_subfault)
    lowerdesignMat = np.hstack((lowerboundss, lowerboundds))
    lowerdesignMat = getsign(lowerrake, -1)[:, None]*lowerdesignMat

    # 定上界设计矩阵
    upperrake = rakeseq[:, 1]
    slope = np.tan(upperrake/180.*np.pi)
    upperboundss = -slope*np.eye(cnt_subfault)
    upperboundds = np.eye(cnt_subfault)
    upperdesignMat = np.hstack((upperboundss, upperboundds))
    upperdesignMat = getsign(upperrake, 1)[:, None]*upperdesignMat

    rakedesignMat = np.vstack((lowerdesignMat, upperdesignMat))
    return rakedesignMat


class multifaultsolve_kfh(multifaultsolve):
    '''
    反演滑动分布和轨道参数
        1. 添加Laplace平滑约束
        2. 构造生成边界约束的新函数
        3. 写一个装配平滑矩阵和对应数据对象的函数
        4. 添加一个约束最小二乘反演函数
    '''
    
    def __init__(self, name, faults, verbose=True, extra_parameters=None):
        super(multifaultsolve_kfh, self).__init__(name,
                                                faults,
                                                verbose=verbose)
        self.lb = None
        self.ub = None

        self.rakedesignMat = None
        self.rakevec = None

        if extra_parameters is not None:
            self.ramp_switch = len(extra_parameters)
        else:
            self.ramp_switch = 0
        return
    

    def setrakebound(self, eachfault_rakelimit):
        '''
        ss ds
        ss ds
        (135, -135), (-45, 45)
        '''

        Ns = 0
        Nsd = 0
        Np = self.G.shape[1]
        for fault in self.faults:
            Ns += fault.slip.shape[0]
            Nsd += int(Ns*len(fault.slipdir))
        rakedesignMat = np.zeros((Nsd, Np))
        rakevec = np.zeros((Nsd,))
        st = 0
        for fault, rakelimit in zip(self.faults, eachfault_rakelimit):
            Nslocal = fault.slip.shape[0]
            st = self.fault_indexes[fault.name][0]
            # se = self.fault_indexes[fault.name][1]
            Nsdlocal = len(fault.slipdir)*Nslocal
            rake_bound = np.zeros((Nslocal, 2))
            rake_bound[:, 0] = rakelimit[0]
            rake_bound[:, 1] = rakelimit[1]
            rakedesignlocal = makeRakeBoundMat(rake_bound)
            rakedesignMat[st: st+Nsdlocal, st:st+Nsdlocal] = rakedesignlocal

        self.rakedesignMat = rakedesignMat
        self.rakevec = rakevec
        return

    def setbound(self, lb=None, ub=None, eachfaultlimit=None, extralimit=None):
        '''
        eachfaultlimit 和lb/ub二选一输入，同时输入以lb, ub为准
        Input:
            eachfaultlimit: 和faults等长的列表，每一个子元包含两个值
            extralimit: 二元列表，表示上下限
        '''

        Np = self.G.shape[1]
        lowerboundary = np.zeros((Np, ))
        upperboundary = np.zeros((Np, ))
        
        if lb is None and ub is None and eachfaultlimit is None:
            lowerboundary -= 1000.
            upperboundary += 1000.
        elif lb is not None and ub is not None:
            if lb.__class__ in (list, np.ndarray):
                lowerboundary = np.array(lb)
            elif lb.__class__ in (float, int):
                lowerboundary += lb
            
            if ub.__class__ in (list, np.ndarray):
                upperboundary = np.array(ub)
            elif lb.__class__ in (float, int):
                upperboundary += ub
        elif eachfaultlimit is not None:
            for fault, limit in zip(self.faults, eachfaultlimit):
                lb, ub = limit
                if extralimit is not None:
                    lbramp, ubramp = extralimit
                else:
                    lbramp, ubramp = -1000., 1000.
                
                st = self.fault_indexes[fault.name][0]
                se = self.fault_indexes[fault.name][1]
                Nslocal = int(fault.slip.shape[0]*len(fault.slipdir))
                lowerboundary[st:st+Nslocal] = lb
                lowerboundary[st+Nslocal: se] = lbramp
                upperboundary[st:st+Nslocal] = ub
                upperboundary[st+Nslocal: se] = ubramp

        self.ub = upperboundary # ub
        self.lb = lowerboundary # lb
        return


    def ConstrainedLeastSquareSoln(self, extra_parameters=None, bounds=None, penWeight=1., 
                                    iterations=1000, tolerance=None, maxfun=100000, D_lap=None, lap_bounds=None, method='mudpy'):
        '''
        lap_bounds for fault.patchType == 'rectangle'
        '''
        import scipy.linalg as scilin
        from scipy.optimize import lsq_linear as scilsql
        from scipy.linalg import block_diag as blkdiag
        import lsqlin
        
        # Get the faults
        faults = self.faults

        # Get the matrixes and vectors
        G = self.G
        Cd = self.Cd
        d = self.d

        # Nd = d.shape[0]
        Np = G.shape[1]
        Ns = 0
        # build Laplace
        for fault in faults:
            Ns += int(fault.slip.shape[0]*len(fault.slipdir))
        G_lap = np.zeros((Ns, Np))
        d_lap = np.zeros((Ns, ))

        # 行初始索引
        rst = 0
        rse = 0
        if D_lap is None:
            for fault in faults:
                st = self.fault_indexes[fault.name][0]
                rse += int(fault.slip.shape[0]*len(fault.slipdir))
                if fault.type is 'Fault':
                    if fault.patchType in ('rectangle'):
                        lap = fault.buildLaplacian(method=method, bounds=lap_bounds) # _kfh()
                    else:
                        lap = fault.buildLaplacian(method=method, bounds=lap_bounds)
                    lapsd = blkdiag(lap, lap)
                    Nsd = len(fault.slipdir)
                    se = st + Nsd*lap.shape[0]
                    G_lap[rst:rse, st:se] = lapsd
                    rst = rse
        else:
            # G_lap[:, :Ns] = D_lap
            G_lap = np.zeros((D_lap.shape[0], Np))
            G_lap[:, :Ns] = D_lap
            d_lap = np.zeros((G_lap.shape[0], ))
        self.G_lap = G_lap
        # penWeight = 1.
        G_lap2I = penWeight*G_lap
        # d = np.hstack((d, d_lap))

        Icovd = np.linalg.inv(Cd)
        W = np.linalg.cholesky(Icovd)
        self.dataweight = W
        d2I = np.vstack((np.dot(W, d)[:, None], d_lap[:, None])).flatten()

        G2I = np.vstack((np.dot(W, G), G_lap2I)) 
        # Compute 
        # mpost = np.dot( np.dot( scilin.inv(np.dot( G2I.T, G2I )), G2I.T ), d2I)

        # ----------------------------Inverse using lsqlin-----------------------------#
        # set the Rake Constraint
        # assert self.rakedesignMat is not None, "You should assumble the rake bound first"
        # rake_bound = np.vstack((np.array([[135., -135.]]*1628), np.array([[-45., 45.]]*646)))
        rakedesignMat = self.rakedesignMat
        rakevec = self.rakevec

        # Set the constraint of the upper/lower Bounds
        # assert self.lb is not None, "You should assumble the upper/lower bounds first"
        lb = self.lb
        ub = self.ub

        # Compute using lsqlin equivalent to the lsqlin in matlab
        opts = {'show_progress': False}
        ret = lsqlin.lsqlin(G2I, d2I, 0, rakedesignMat, rakevec, None, None, lb, ub, None, opts)
        mpost = ret['x']
        # Store mpost
        self.mpost = lsqlin.cvxopt_to_numpy_matrix(mpost)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distributem(self, verbose=False):
        '''
        After computing the m_post model, this routine distributes the m parameters to the faults.

        Kwargs:
            * verbose   : talk to me

        Returns:
            * None
        '''

        # Get the faults
        faults = self.faults

        # Loop over the faults
        for fault in faults:

            if verbose:
                print ("---------------------------------")
                print ("---------------------------------")
                print("Distribute the slip values to fault {}".format(fault.name))

            # Store the mpost
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            fault.mpost = self.mpost[st:se]

            # Transformation object
            if fault.type=='transformation':
                
                # Distribute simply
                fault.distributem()

            # Fault object
            if fault.type is "Fault":

                # Affect the indexes
                self.affectIndexParameters(fault)

                # put the slip values in slip
                st = 0
                if 's' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,0] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 'd' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,1] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 't' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,2] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 'c' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.coupling = fault.mpost[st:se]
                    st += fault.slip.shape[0]

                # check
                if hasattr(fault, 'NumberCustom'):
                    fault.custom = {} # Initialize dictionnary
                    # Get custom params for each dataset
                    for dset in fault.datanames:
                        if 'custom' in fault.G[dset].keys():
                            nc = fault.G[dset]['custom'].shape[1] # Get number of param for this dset
                            se = st + nc
                            fault.custom[dset] = fault.mpost[st:se]
                            st += nc

            # Pressure object
            elif fault.type is "Pressure":

                st = 0
                if fault.source in {"Mogi", "Yang"}:
                    se = st + 1
                    print(np.asscalar(fault.mpost[st:se]*fault.mu))
                    fault.deltapressure = np.asscalar(fault.mpost[st:se]*fault.mu)
                    st += 1
                elif fault.source is "pCDM":
                    se = st + 1
                    fault.DVx = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    se = st + 1
                    fault.DVy = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    se = st + 1
                    fault.DVz = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    print("Total potency scaled by", fault.scale)

                    if fault.DVtot is None:
                        fault.computeTotalpotency()
                elif fault.source is "CDM":
                    se = st + 1
                    print(np.asscalar(fault.mpost[st:se]*fault.mu))
                    fault.deltaopening = np.asscalar(fault.mpost[st:se])
                    st += 1

            # Get the polynomial/orbital/helmert values if they exist
            if fault.type in ('Fault', 'Pressure'):
                fault.polysol = {}
                fault.polysolindex = {}
                for dset in fault.datanames:
                    if dset in fault.poly.keys():
                        if (fault.poly[dset] is None):
                            fault.polysol[dset] = None
                        else:

                            if (fault.poly[dset].__class__ is not str) and (fault.poly[dset].__class__ is not list):
                                if (fault.poly[dset] > 0):
                                    se = st + fault.poly[dset]
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += fault.poly[dset]
                            elif (fault.poly[dset].__class__ is str):
                                if fault.poly[dset] is 'full':
                                    nh = fault.helmert[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                                if fault.poly[dset] in ('strain', 'strainnorotation', 'strainonly', 'strainnotranslation', 'translation', 'translationrotation'):
                                    nh = fault.strain[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                                # Added by kfhe, at 10/12/2021
                                if fault.poly[dset] is 'eulerrotation':
                                    nh = fault.eulerrot[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                                if fault.poly[dset] is 'internalstrain':
                                    nh = fault.intstrain[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                            elif (fault.poly[dset].__class__ is list):
                                nh = fault.transformation[dset]
                                se = st + nh
                                fault.polysol[dset] = fault.mpost[st:se]
                                fault.polysolindex[dset] = range(st,se)
                                st += nh

        # All done
        return
    # ----------------------------------------------------------------------
