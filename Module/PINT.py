import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cvxopt.solvers import qp
from cvxopt import matrix

class tuning_curves:
    def __init__(self, A, Tarr, sigma, iin, order = 0, mode = 'train', test_r = None, \
                 TEST_FRACTION = 0.3, status = 'parent'):

        self.Q, self.N, self.R = A.shape
        self.A = A
        self.Tarr = Tarr
        self.sigma = sigma
        self.iin = iin.reshape((-1,1))
        self.__gen_locs__()
        self.mode = mode
        self.P = order
        self.freq_threshold = 20

        self.fcount = 0
        self.fs = {}
        self.raw_decoders = {}
        self.cvx_decoders = {}
        self.status = status

        # Produce train and test tuning curve objects if the mode is cross-validation
        if (self.mode == 'cross-validate'):
            self.test_r = test_r
            rs = np.arange(self.R).astype(int)
            if (self.test_r is None):
                self.test_r = np.random.choice(rs, size = int(self.R*TEST_FRACTION))
            self.train_r = np.setdiff1d(rs, self.test_r)
            self.train_curves = tuning_curves(self.A[:,:,self.train_r], self.Tarr[self.train_r],
                                              self.sigma, self.iin, order = self.P, mode = 'train')
            self.test_curves = tuning_curves(self.A[:,:,self.test_r], self.Tarr[self.test_r],
                                             self.sigma, self.iin, order = self.P, mode = 'test')
            self.child = self.train_curves.child
            self.bad_neurons = self.train_curves.bad_neurons
            self.Ngood = self.train_curves.Ngood
        self.__gen__()
        if (self.mode != 'test'):
            self.__fresh_invert__()

    def __gen__(self):
        self.__gen_AA__()
        self.__gen_cov__()
        self.__gen_B__()
        self.__gen_W__()
        self.__gen_cs__()
        if((self.status == 'parent') & (self.mode == 'train')):
        	self.__gen_bad_ns__()
        self.metric = None

    def __gen_bad_ns__(self):
    	self.bad_neurons = np.zeros((self.N)).astype(bool)
    	for n in range(self.N):
    		a = self.A[:,n,:]
    		if (np.max(a) < self.freq_threshold):
    			self.bad_neurons[n] = True
    	print('creating child')
    	self.child = tuning_curves(
    		self.A[:,~self.bad_neurons, :], self.Tarr, self.sigma, self.iin, order = self.P, 
    		mode = self.mode, status = 'child')
    	self.Ngood = self.N - np.sum(self.bad_neurons)

    def __fresh_invert__(self):
        if (self.mode == 'cross-validate'):
            self.train_eop = self.train_curves.eop
            self.train_trace = self.train_curves.trace
            self.train_eigenerrors = self.train_curves.eigenerrors
            self.train_eigenfunctions = self.train_curves.eigenfunctions
            self.csinv = self.train_curves.csinv
            self.W = self.train_curves.W

            self.test_eop = np.eye(self.Q)
            self.test_eop += self.train_curves.W.T @ self.train_curves.csinv @ self.test_curves.cs @ \
                             self.train_curves.csinv @ self.train_curves.W
            self.test_eop -= self.train_curves.W.T @ self.train_curves.csinv @ self.test_curves.W
            self.test_eop -= self.test_curves.W.T @ self.train_curves.csinv @ self.train_curves.W
            self.test_trace = np.trace(self.test_eop)
            self.test_eigenerrors, self.test_eigenfunctions = np.linalg.eigh(self.test_eop)

        else:
            self.csinv = np.linalg.inv(self.cs)
            self.eop = np.eye(self.Q) - self.W.T @ self.csinv @ self.W
            self.trace = np.trace(self.eop)
            self.eigenerrors, self.eigenfunctions = np.linalg.eigh(self.eop)

            self.fmaxes = np.ndarray(self.Q)
            for i in range(100):
                f = self.eigenfunctions.T[i]
                f /= np.max(np.abs(f))
                d = self.csinv@self.W@f
                self.fmaxes[i] = (1/np.mean(np.abs(d)))


    def __gen_AA__(self):
        self.AA = np.ndarray((self.R*self.Q, self.N))
        for r in range(self.R):
            self.AA[r*self.Q:(r+1)*self.Q,:] = self.A[:,:,r]
        self.LL = np.ndarray((self.R*self.Q, self.N*(self.P + 1)))
        for p in range(self.P+1):
            Tp = np.eye(self.R*self.Q)
            for r in range(self.R):
                Tp[r*self.Q:(r+1)*self.Q, r*self.Q:(r+1)*self.Q] *= np.power(self.Tarr[r], p)
            self.LL[:,p*self.N: (p+1)*self.N] = Tp@self.AA

    def __gen_cov__(self):
        self.cov = 1/self.R*self.LL.T @ self.LL

    def __gen_B__(self):
        self.B = np.ndarray(((self.P+1)*self.N, (self.P + 1)*self.N))
        for p in range(2*self.P + 1):
            bn = self.Q*self.N*np.mean(np.power(self.Tarr, p))*np.eye(self.N)
            for i in range(self.P + 1):
                for j in range(self.P + 1):
                    if (i + j == p):
                        self.B[i*self.N:(i+1)*self.N, j*self.N:(j+1)*self.N] = bn

    def __gen_W__(self):
        self.G = np.ndarray((self.R * self.Q, self.Q))
        for r in range(self.R):
            self.G[r*self.Q:(r+1)*self.Q,:] = np.eye(self.Q)/self.R
        self.W = self.LL.T @ self.G

    def __gen_cs__(self):
        self.cs = self.cov + self.sigma**2 * self.B

    def update_sigma(self, sigma):
        self.sigma = sigma
        self.__gen_cs__()
        if (self.mode == 'cross-validate'):
            self.train_curves.update_sigma(sigma)
            self.test_curves.update_sigma(sigma)
        if (self.mode != 'test'):
            self.__fresh_invert__()

    def gen_lmargins(self, metric = None):
        if (metric is not None):
            self.metric = metric
        if (self.mode == 'cross-validate'):
            self.train_lmargins = np.ndarray((self.N))
            self.test_lmargins = np.ndarray((self.N))
        else:
            self.lmargins = np.ndarray((self.N))
        for n in tqdm(range(self.N)):
            les = self.__lesion__(n)
            if (self.mode == 'cross-validate'):
                self.train_lmargins[n], self.test_lmargins[n] = les
            else:
                self.lmargins[n] = les

    def update_metric(self, metric):
        self.metric = metric

    def get_fmax(self, f):
        if (self.P == 0):
            return self.get_fmax_LSAT(f)
        elif (self.P == 1):
            return self.get_fmax_LINT(f)
        else:
            print('get_fmax not implemented for order ' + str(self.P))

    def get_fmax_LSAT(self, f):
        f /= np.max(np.abs(f))
        d = self.csinv@self.W@f
        return 1/np.mean(np.abs(d))

    def get_fmax_LINT(self, f):
        f /= np.max(np.abs(f))
        d = self.csinv@self.W@f
        dd = np.ndarray((2, self.N))
        dd[0] = np.abs(d[:self.N] + np.max(self.Tarr)*d[self.N:])
        dd[1] = np.abs(d[:self.N] + np.min(self.Tarr)*d[self.N:])
        dmax = np.max(dd, axis = 0)
        return 1/np.mean(dmax)

    def __lesion__(self, j):
        Ap = self.A.copy()
        Ap[:,j,:] = 0
        if(self.mode == 'cross-validate'):
            les = tuning_curves(Ap, self.Tarr, self.sigma, self.iin, mode = self.mode, test_r = self.test_r, order = self.P)
        else:
            les = tuning_curves(Ap, self.Tarr, self.sigma, self.iin, mode = self.mode, order = self.P)
        if (self.metric is None):
            if (self.mode == 'cross-validate'):
                ltr = les.train_trace - self.train_trace
                lte = les.test_trace - self.test_trace
                result = [ltr, lte]
            else:
                result = les.trace - self.trace
        else:
            if (self.mode == 'cross-validate'):
                ltr = np.trace(self.metric.T@(les.train_eop - self.train_eop)@self.metric)
                lte = np.trace(self.metric.T@(les.test_eop - self.test_eop)@self.metric)
                result = [ltr, lte]
            else:
                result = np.trace(self.metric.T@(les.eop - self.eop)@self.metric)
        del les
        return result

    def __gen_locs__(self):
        self.intercepts = np.ndarray((self.N, self.R))
        self.gains = np.ndarray((self.N, self.R))
        for n in range(self.N):
            for r in range(self.R):
                self.intercepts[n,r], self.gains[n,r] = self.__intercept_gain__(n, r)
        self.mean_intercepts = np.mean(self.intercepts, axis = 1)
        self.mean_gains = np.mean(self.gains, axis = 1)

    def __intercept_gain__(self, n, r):
        thresh = 30
        curve = self.A[:,n,r]
        left = curve[-1] < curve[0]
        firing = np.where(curve > thresh)[0]
        if (len(firing) is 0):
            return [None, None]
        if (left):
            threshold = self.iin[firing[-1]]
        else:
            threshold = self.iin[firing[0]]
        gain = (curve[firing][-1] - curve[firing][0])/np.max([self.iin[firing][-1] - \
                                                              self.iin[firing][0], 0.001])
        return threshold, gain

    def pad_decoder(self, d):
    	print(d.shape)
    	dd = np.zeros((self.N*(self.P+1)))
    	for k in range(self.P + 1):
    	    dd[k*self.N : (k+1)*self.N][~self.bad_neurons] = d[k*self.Ngood:(k+1)*self.Ngood]
    	return dd


    def decode_raw(self, f_target):
        fstring = str(self.fcount)
        self.fs[fstring] = f_target
        d = self.child.csinv @ self.child.W @ f_target
        d = d.reshape(-1)
        d = self.pad_decoder(d)
        self.raw_decoders[fstring] = d
        self.cvx_decoders[fstring] = 'Not CVX Decoded'
        self.fcount += 1
        return d

    def decode_cvx(self, f_target, Ts = None):
        if (self.P == 0):
            return self.decode_cvx_LSAT(f_target)
        elif (self.P == 1):
            return self.decode_cvx_LINT(f_target, Ts = Ts)
        else:
            print('Not implemented for order ' + str(self.P))

    def decode_cvx_LSAT(self, f):
        N = self.Ngood
        P= self.child.cs
        q = -self.child.W @ f
        G = np.zeros((2*N, N))
        G[:N,:] = np.eye(N)
        G[N:, :] = -np.eye(N)
        h = np.ones(N*2)

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        solution = qp(P, q, G, h)
        d = np.array(solution['x']).reshape(-1)
        d = self.pad_decoder(d)

        fstring = str(self.fcount)
        self.fs[fstring] = f
        self.cvx_decoders[fstring] = d.reshape(-1)
        self.raw_decoders[fstring] = 'Not raw decoded'
        self.fcount += 1
        return d.reshape(-1)

    def decode_cvx_LINT(self, f, Ts = None):
        if (Ts is None):
            T1 = np.max(self.Tarr)
            T2 = np.min(self.Tarr)
        else:
            T1 = Ts[0]
            T2 = Ts[1]
        N = self.Ngood
        P= self.child.cs
        q = -self.child.W @ f
        G = np.zeros((4*N, 2*N))
        I = np.eye(N)
        G[:N,:N] = I
        G[:N,N:] = T1*I
        G[N:2*N, :N] = -I
        G[N:2*N, N:] = -T1*I
        G[2*N:3*N, :N] = I
        G[2*N:3*N, N:] = T2*I
        G[3*N:, :N] = -I
        G[3*N:, N:] = -T2*I

        h = np.ones(4*N)

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        solution = qp(P, q, G, h)
        d = np.array(solution['x']).reshape(-1)
        d = self.pad_decoder(d)

        fstring = str(self.fcount)
        self.fs[fstring] = f
        self.cvx_decoders[fstring] = d.reshape(-1)
        self.raw_decoders[fstring] = 'Not raw decoded'
        self.fcount += 1
        return d.reshape(-1)
