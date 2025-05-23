import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# Helper functions (Rcpp equivalents)
# ----------------------------

def id_low_search(V):
    """
    Given an extended vector V (with np.inf at the end),
    return an integer array id_low such that for each i, id_low[i]
    is the largest index j (starting at –1) with V[j+1] < V[i].
    """
    V = np.asarray(V)
    n = len(V)
    id_low = -np.ones(n, dtype=int)
    it = -1
    for i in range(n):
        while (it + 1 < n) and (V[it + 1] < V[i]):
            it += 1
        id_low[i] = it
    return id_low

def q_low_compute(id_low, Qcumsum):
    """
    For each index i with id_low[i] >= 0, assign q_low[i] = Qcumsum[i, id_low[i]].
    """
    n = len(id_low)
    q_low = np.zeros(n)
    for i in range(n):
        if id_low[i] >= 0:
            q_low[i] = Qcumsum[i, int(id_low[i])]
    return q_low

def LCP_construction_path_distance(V, id_low, q_low, qn, Hnew, HnewT):
    """
    This function mimics the C++ function LCP_construction_path_distance.
    V is a 1D array (extended with np.inf as its last element),
    id_low and q_low are 1D arrays (typically produced from the above functions),
    qn is a vector (typically of length n0, with n0 = len(V)-1),
    Hnew is an (m x n) matrix and HnewT is an (n x m) matrix.
    The function returns a dict with key 'alphas' (an m x n matrix) and 'vs'.
    """
    V = np.asarray(V)
    n = len(V)
    n0 = n - 1
    m = Hnew.shape[0]
    
    theta = np.zeros((m, n))
    # Build cumulative theta (using the recurrence in the C++ code)
    for i in range(1, n):
        theta[:, i] = theta[:, i-1].copy()
        if (id_low[i] - id_low[i-1]) > 0:
            # Add the columns of Hnew corresponding to indices (id_low[i-1]+1) to id_low[i] inclusive
            for i1 in range(int(id_low[i-1] + 1), int(id_low[i] + 1)):
                theta[:, i] += Hnew[:, i1]
    qn_new = theta[:, n0].copy()
    
    alphas = np.zeros((m, n))
    for i in range(m):
        # For the i-th test sample, compute normalized quantities.
        # (Note: In the original code qn and q_low are expected to be defined for the training indices.)
        # Here we use the first n0 entries.
        normalizers = qn[:n0] + HnewT[:n0, i]
        p1 = (q_low[:n0] + HnewT[:n0, i]) / normalizers
        p2 = q_low[:n0] / normalizers
        ttheta_i = theta[i, :].copy()          # length n
        ttheta_i0 = theta[i, :n0].copy()         # first n0 entries
        ttheta_i = ttheta_i / (qn_new[i] + 1.0)
        ttheta_i0 = ttheta_i0 / (qn_new[i] + 1.0)
        
        # Identify index sets as in the C++ code:
        idA1 = np.where(p1 < ttheta_i0)[0]
        idA2 = np.where(p2 >= ttheta_i0)[0]
        idA3 = np.where((p2 < ttheta_i0) & (p1 >= ttheta_i0))[0]
        pA1 = p1[idA1] if idA1.size > 0 else np.array([])
        pA2 = p2[idA2] if idA2.size > 0 else np.array([])
        pA3 = np.array([id_low[j] for j in idA3]) if idA3.size > 0 else np.array([])
        nA1, nA2, nA3 = len(pA1), len(pA2), len(pA3)
        pA1sort = np.sort(pA1) if nA1 > 0 else pA1
        pA2sort = np.sort(pA2) if nA2 > 0 else pA2
        pA3sort = np.sort(pA3) if nA3 > 0 else pA3
        
        itA1 = 0
        itA2 = 0
        itA3 = 0
        k = -1
        # The loop runs over k = 0,...,n-1:
        while k < (n - 1):
            k += 1
            if nA1 > 0 and itA1 < nA1:
                while itA1 < (nA1 - 1) and pA1sort[itA1] < ttheta_i[k]:
                    itA1 += 1
                if itA1 == (nA1 - 1) and pA1sort[itA1] < ttheta_i[k]:
                    itA1 += 1
            if nA2 > 0 and itA2 < nA2:
                while itA2 < (nA2 - 1) and pA2sort[itA2] < ttheta_i[k]:
                    itA2 += 1
                if itA2 == (nA2 - 1) and pA2sort[itA2] < ttheta_i[k]:
                    itA2 += 1
            if nA3 > 0 and itA3 < nA3:
                id_low_k = id_low[k]
                while itA3 < (nA3 - 1) and pA3sort[itA3] < id_low_k:
                    itA3 += 1
                if itA3 == (nA3 - 1) and pA3sort[itA3] < id_low_k:
                    itA3 += 1
            alphas[i, k] = (itA1 + itA2 + itA3) / (n0 + 1.0)
    
    return {'alphas': alphas, 'vs': V}

def LCP_construction_distance_loop(V, Qcumsum, H):
    """
    This function loops over training samples and for each constructs an “LCP”
    using the helper LCP_construction_path_distance.
    """
    V = np.asarray(V)
    n = len(V)
    alphas = np.zeros((n, n))
    id_low = id_low_search(V)
    q_low = q_low_compute(id_low, Qcumsum)
    for i in range(n):
        # Build new arrays excluding the i-th element.
        Vi = np.zeros(n)
        id_low_i = np.zeros(n)
        q_low_i = np.zeros(n - 1)
        qn_i = np.zeros(n - 1)
        Hnew = np.zeros((1, n - 1))
        HnewT = np.zeros((n - 1, 1))
        for i1 in range(n - 1):
            i2 = i1 if i1 < i else i1 + 1
            q_low_i[i1] = q_low[i2]
            id_low_i[i1] = id_low[i2]
            Vi[i1] = V[i2]
            if i <= id_low[i2]:
                q_low_i[i1] -= H[i2, i]
                id_low_i[i1] -= 1
            qn_i[i1] = Qcumsum[i2, n - 1] - H[i2, i]
            Hnew[0, i1] = H[i, i2]
            HnewT[i1, 0] = H[i2, i]
        Vi[n - 1] = np.inf
        id_low_i[n - 1] = n - 2
        ret_tmp = LCP_construction_path_distance(Vi, id_low_i, q_low_i, qn_i, Hnew, HnewT)
        alphas[i, :] = ret_tmp['alphas']
    return {'alphas': alphas, 'vs': V}

# ----------------------------
# Main functions (converted from R)
# ----------------------------

def LCP_alpha(vs, alphas, alpha):
    """
    Given a sorted vector vs (possibly extended with np.inf) and a corresponding array of alphas,
    return the last element of vs for which the corresponding alpha value is below the threshold.
    For matrix alphas (each row processed separately) a vector is returned.
    """
    vs = np.asarray(vs)
    alphas = np.asarray(alphas)
    if alphas.ndim == 1:
        idxes = np.where(alphas < alpha)[0]
        if len(idxes) > 0:
            return vs[idxes[-1]]
        else:
            return np.inf
    else:
        v_bounds = np.full(alphas.shape[0], np.inf)
        for i in range(alphas.shape[0]):
            idx = np.where(alphas[i, :] < alpha)[0]
            if len(idx) > 0:
                v_bounds[i] = vs[idx[-1]]
        return v_bounds

def LCP_construction_path(alpha_val, V, id_low, q_low, qn, Hnew, HnewT, type="distance",
                          neighbor_size=100, idx_boundary=None, size_boundary=None, distance_boundary=None):
    """
    This function wraps a call to LCP_construction_path_distance.
    In the “neighbor” type a loop is performed with extra boundary conditions.
    """
    if type == "neighbor":
        if idx_boundary is None or distance_boundary is None:
            raise ValueError("boundary conditions for nearest neighbor localizers missing.")
        m = Hnew.shape[0]
        n = len(V)
        n0 = n - 1
        Smat = np.zeros((m, n))
        deltaLCP = np.zeros(m)
        for i in range(m):
            # Compute a ranking of Hnew[i, :]. (Ties are broken arbitrarily.)
            # Here we mimic R’s rank by using argsort and adding one.
            ranks = np.empty_like(Hnew[i, :], dtype=int)
            order = np.argsort(Hnew[i, :], kind='mergesort')
            ranks[order] = np.arange(1, len(order)+1)
            # Use neighbor_size as a threshold
            hnew = np.where(ranks < neighbor_size, 1, 0)
            hnewT = HnewT[:, i].copy()
            qni = qn.copy()
            q_lowi = q_low.copy()
            for j in range(n - 1):
                if hnewT[j] < distance_boundary[j]:
                    hnewT[j] = 1
                    qni[j] = qni[j] - 1
                    if idx_boundary[j] <= id_low[i] + 1:
                        q_lowi[j] = q_lowi[j] - 1
                elif hnewT[j] > distance_boundary[j]:
                    hnewT[j] = 0
                else:
                    indicator = np.random.binomial(1, 1 / (size_boundary[j] + 1))
                    if indicator == 1:
                        hnewT[j] = 1
                        qni[j] = qni[j] - 1
                        if idx_boundary[j] <= id_low[i] + 1:
                            q_lowi[j] = q_lowi[j] - 1
                    else:
                        hnewT[j] = 0
            hnew = hnew.reshape(1, -1)
            hnewT = hnewT.reshape(-1, 1)
            ret = LCP_construction_path_distance(V, id_low, q_lowi, qni, hnew, hnewT)
            deltaLCP[i] = LCP_alpha(V, ret['alphas'][0, :], 1 - alpha_val)
            Smat[i, :] = ret['alphas'][0, :]
    else:
        ret = LCP_construction_path_distance(V, id_low, q_low, qn, Hnew, HnewT)
        deltaLCP = LCP_alpha(V, ret['alphas'], 1 - alpha_val)
        Smat = ret['alphas']
    return {'deltaLCP': deltaLCP, 'Smat': Smat}

def autoTune_distance(V, n, hs, D, alpha=0.05, delta=0.05, B=10, trace=True, lambda_=1):
    """
    Auto-tuning procedure (distance version).
    V is a 1D array, D a square distance matrix.
    """
    V = np.asarray(V)
    D = np.asarray(D)
    n0 = len(V)
    J = len(hs)
    n1 = min(n0, n + 1)
    B1 = 1 if n1 >= (2/3 * n0) else B

    C1 = np.zeros((B1, 2, J))
    C2 = np.zeros((n0, B, J))
    mu = np.zeros((n0, J))
    varphi = np.full((n0, B, J), np.inf)
    Ts = np.zeros((J, 3))
    
    # Sort V and adjust D accordingly
    orders = np.argsort(V)
    V = V[orders]
    D = D[np.ix_(orders, orders)]
    thr = np.max(V)
    
    print("###estimate means######")
    for b in range(B1):
        if trace:
            print(f"Iteration (mean estimation) {b+1}")
        I = np.sort(np.random.choice(np.arange(n0), size=n1, replace=False))
        Vb = V[I]
        Db = D[np.ix_(I, I)]
        def mean_loop(j):
            h = hs[j]
            H = np.exp(-Db / h)
            Qcumsumb = np.cumsum(H, axis=1)
            ret1 = LCP_construction_distance_loop(Vb, Qcumsumb, H)
            tmp = np.zeros(len(Vb))
            for i in range(len(Vb)):
                Vb0 = np.delete(Vb, i)
                Vb0_ext = np.concatenate([Vb0, [np.inf]])
                tmp[i] = LCP_alpha(Vb0_ext, ret1['alphas'][i, :], 1 - alpha)
            return np.mean(tmp > thr), np.mean(tmp[tmp <= thr])
        mean_results = Parallel(n_jobs=-1)(delayed(mean_loop)(j) for j in range(J))
        for j in range(J):
            C1[b, 0, j] = mean_results[j][0]
            C1[b, 1, j] = mean_results[j][1]
    
    print("###standard deviation estimation#####")
    if lambda_ > 0:
        for b in range(B):
            if trace:
                print(f"Iteration (sd estimation) {b+1}")
            I = np.sort(np.random.choice(np.arange(n0), size=n, replace=True))
            Vb = V[I]
            Db = D[np.ix_(I, I)]
            Dnew = D[:, I]
            DnewT = D[np.ix_(I, np.arange(n0))]
            Vb_ext = np.concatenate([Vb, [np.inf]])
            id_low = id_low_search(Vb_ext)
            def sd_loop(j):
                h = hs[j]
                H = np.exp(-Db / h)
                Hnew = np.exp(-Dnew / h)
                HnewT = np.exp(-DnewT / h)
                Qcumsumb = np.cumsum(H, axis=1)
                q_low = q_low_compute(id_low[:-1], Qcumsumb)
                qn = Qcumsumb[:, n - 1]
                ret = LCP_construction_path(alpha, Vb_ext, id_low, q_low, qn, Hnew, HnewT, type="distance")
                return LCP_alpha(Vb_ext, ret['Smat'], 1 - alpha)
            sd_results = Parallel(n_jobs=-1)(delayed(sd_loop)(j) for j in range(J))
            for j in range(J):
                C2[:, b, j] = sd_results[j]
        for j in range(J):
            for i in range(n0):
                tmp = C2[i, :, j]
                tmp1 = np.mean(tmp <= thr)
                if tmp1 > 0:
                    mu[i, j] = np.mean(tmp[tmp <= thr])
                for b in range(B):
                    if C2[i, b, j] < np.inf:
                        varphi[i, b, j] = (C2[i, b, j] - mu[i, j])**2
    
    Ts[:, 0] = hs
    Ts[:, 1] = np.mean(C1[:, 0, :], axis=0)
    Ts[:, 2] = np.mean(C1[:, 1, :], axis=0)
    intermediants = {'probs': Ts[:, 1], 'means': Ts[:, 2], 'sds': np.zeros(J)}
    if lambda_ > 0:
        for j in range(J):
            tmp = varphi[:, :, j]
            intermediants['sds'][j] = np.sqrt(np.mean(tmp[tmp < np.inf]))
            Ts[j, 2] = Ts[j, 2] + lambda_ * intermediants['sds'][j]
    idx = np.where(Ts[:, 1] <= delta)[0]
    if len(idx) == 0:
        h_final = None
    else:
        hs0 = np.array(hs)[idx]
        c0 = Ts[idx, 2]
        h_final = hs0[np.argmin(c0)]
    return {'Ts': Ts, 'h': h_final, 'intermediants': intermediants}

# ----------------------------
# Other utility functions
# ----------------------------

def dist(x, x1):
    """
    Compute distance between x and x1.
    If x is one-dimensional (or a column vector) returns absolute difference.
    Otherwise, computes the square root of the mean squared difference row‐wise.
    """
    x = np.asarray(x)
    x1 = np.asarray(x1)
    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        return np.abs(x - x1)
    else:
        diffs = x - x1
        return np.sqrt(np.mean(diffs**2, axis=1))

def LCPdefault_distance(utangent, uorthogonal, estimated_sds, xtrain, xcalibration, xtest, alpha=0.05):
    """
    Computes default distance matrices and a combined matrix Hcv.
    estimated_sds is expected to be a list [z, z0, z1].
    """
    z, z0, z1 = estimated_sds[0], estimated_sds[1], estimated_sds[2]
    n0 = xcalibration.shape[0]
    m = xtest.shape[0]
    n = len(z)
    Hdist1 = np.zeros((n0 + m, n0 + m))
    Hdist2 = np.zeros((n0 + m, n0 + m))
    Hdist3 = np.zeros((n0 + m, n0 + m))
    Hdist1cv = np.zeros((n, n))
    Hdist2cv = np.zeros((n, n))
    Hdist3cv = np.zeros((n, n))
    
    for j in range(n0):
        Hdist1[0:n0, j] = dist(z0, z0[j])
        Hdist1[n0:(n0 + m), j] = dist(z1, z0[j])
    for j in range(m):
        Hdist1[0:n0, j + n0] = dist(z0, z1[j])
        Hdist1[n0:(n0 + m), j + n0] = dist(z1, z1[j])
    for j in range(n):
        Hdist1cv[0:n, j] = dist(z, z[j])
    
    if utangent is not None:
        z0parallel = xcalibration.dot(utangent)
        z1parallel = xtest.dot(utangent)
        zparallel = xtrain.dot(utangent)
        if z0parallel.ndim == 1 or (z0parallel.ndim == 2 and z0parallel.shape[1] == 1):
            for j in range(n0):
                Hdist2[0:n0, j] = dist(z0parallel, z0parallel[j])
                Hdist2[n0:(n0 + m), j] = dist(z1parallel, z0parallel[j])
            for j in range(m):
                Hdist2[0:n0, j + n0] = dist(z0parallel, z1parallel[j])
                Hdist2[n0:(n0 + m), j + n0] = dist(z1parallel, z1parallel[j])
        else:
            for j in range(n0):
                Hdist2[0:n0, j] = dist(z0parallel, z0parallel[j, :])
                Hdist2[n0:(n0 + m), j] = dist(z1parallel, z0parallel[j, :])
            for j in range(m):
                Hdist2[0:n0, j + n0] = dist(z0parallel, z1parallel[j, :])
                Hdist2[n0:(n0 + m), j + n0] = dist(z1parallel, z1parallel[j, :])
        if zparallel.ndim == 1 or (zparallel.ndim == 2 and zparallel.shape[1] == 1):
            for j in range(n):
                Hdist2cv[0:n, j] = dist(zparallel, zparallel[j])
        else:
            for j in range(n):
                Hdist2cv[0:n, j] = dist(zparallel, zparallel[j, 0])
    if uorthogonal is not None:
        z0orthogonal = xcalibration.dot(uorthogonal)
        z1orthogonal = xtest.dot(uorthogonal)
        zorthogonal = xtrain.dot(uorthogonal)
        if z0orthogonal.ndim == 1 or (z0orthogonal.ndim == 2 and z0orthogonal.shape[1] == 1):
            for j in range(n0):
                Hdist3[0:n0, j] = dist(z0orthogonal, z0orthogonal[j])
                Hdist3[n0:(n0 + m), j] = dist(z1orthogonal, z0orthogonal[j])
            for j in range(m):
                Hdist3[0:n0, j + n0] = dist(z0orthogonal, z1orthogonal[j])
                Hdist3[n0:(n0 + m), j + n0] = dist(z1orthogonal, z1orthogonal[j])
        else:
            for j in range(n0):
                Hdist3[0:n0, j] = dist(z0orthogonal, z0orthogonal[j, :])
                Hdist3[n0:(n0 + m), j] = dist(z1orthogonal, z0orthogonal[j, :])
            for j in range(m):
                Hdist3[0:n0, j + n0] = dist(z0orthogonal, z1orthogonal[j, :])
                Hdist3[n0:(n0 + m), j + n0] = dist(z1orthogonal, z1orthogonal[j, :])
        if zorthogonal.ndim == 1 or (zorthogonal.ndim == 2 and zorthogonal.shape[1] == 1):
            for j in range(n):
                Hdist3cv[0:n, j] = dist(zorthogonal, zorthogonal[j])
        else:
            for j in range(n):
                Hdist3cv[0:n, j] = dist(zorthogonal, zorthogonal[j, :])
    
    # Compute scaling constants
    tmp_mask = np.not_equal(*np.indices(Hdist1cv.shape))
    s01 = np.mean(Hdist1cv[tmp_mask])
    s02 = np.mean(Hdist2cv[tmp_mask]) if utangent is not None else 0
    s03 = np.mean(Hdist3cv[tmp_mask]) if uorthogonal is not None else 0
    w = s02 / (s02 + s03) if (s02 + s03) != 0 else 0
    if uorthogonal is None:
        w = 0
    Hdist23 = (1 - w) * Hdist2 + w * Hdist3
    Hdist23cv = (1 - w) * Hdist2cv + w * Hdist3cv
    s023 = np.mean(Hdist23cv[tmp_mask])
    a = 1
    Hcv = a * Hdist1cv / s01 + Hdist23cv[0:n, 0:n] / s023
    tmp = a * Hdist1 / s01 + Hdist23 / s023
    H_out = tmp[0:n0, 0:n0]
    Hnew1 = tmp[n0:(n0 + m), 0:n0]
    HnewT1 = tmp[0:n0, n0:(n0 + m)]
    return {'Hcv': Hcv, 'H': H_out, 'Hnew': Hnew1, 'HnewT': HnewT1}

# ----------------------------
# Class implementation (R6-style)
# ----------------------------

class LCP:
    """
    A Python class mimicking the R6 class LCP.
    """
    def __init__(self, H, V, h=1, alpha=0.05, type="distance", invert_func=None):
        self.H = np.asarray(H)
        self.h = h
        self.V = np.concatenate([np.asarray(V), [np.inf]])
        self.n = len(self.V) - 1
        self.alpha = alpha
        self.type = type
        self.invert_func = invert_func if invert_func is not None else (lambda x: x)
        
        # placeholders for later values
        self.Hnew = None
        self.HnewT = None
        self.Hdistance = None
        self.Hrank = None
        self.idx_boundary = None
        self.distance_boundary = None
        self.size_boundary = None
        self.Qcumsum = None
        self.qlow0 = None
        self.qn0 = None
        self.id_low = None
        self.band_V = None
        self.band_Y = None
        self.Smat = None

    def lower_idx(self):
        self.id_low = id_low_search(self.V)

    def cumsum_unnormalized(self):
        if self.type == "distance":
            self.Hdistance = np.exp(-self.H / self.h)
            self.Qcumsum = np.cumsum(self.Hdistance, axis=1)
        elif self.type == "neighbor":
            # Compute ranks with a stable sort (ties broken arbitrarily)
            self.Hrank = np.zeros_like(self.H)
            for i in range(self.H.shape[0]):
                order = np.argsort(self.H[i, :], kind='mergesort')
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(order) + 1)
                self.Hrank[i, :] = ranks
            # For each row, record the indices where the rank equals h
            self.idx_boundary = np.array([np.where(self.Hrank[i, :] == self.h)[0] for i in range(self.Hrank.shape[0])])
            self.distance_boundary = np.array([np.sort(self.H[i, :])[int(self.h) - 1] for i in range(self.H.shape[0])])
            n = self.Hrank.shape[0]
            for i in range(n):
                tmp = self.Hrank[i, :]
                self.Hrank[i, :] = np.where(tmp <= self.h, 1, 0)
            self.Qcumsum = np.cumsum(self.Hrank, axis=1)
        else:
            raise ValueError("unsupported localizer type.")
        self.qlow0 = q_low_compute(self.id_low[:-1], self.Qcumsum)
        self.qn0 = self.Qcumsum[:, self.n - 1]

    def LCP_construction(self, Hnew, HnewT):
        if self.type == "distance":
            Hnew = np.exp(-np.asarray(Hnew) / self.h)
            HnewT = np.exp(-np.asarray(HnewT) / self.h)
        Hnew = np.atleast_2d(Hnew)
        HnewT = np.atleast_2d(HnewT)
        if self.type == "neighbor":
            ret = LCP_construction_path(self.alpha, self.V, self.id_low,
                                        self.qlow0, self.qn0,
                                        Hnew, HnewT, type=self.type,
                                        neighbor_size=self.h,
                                        idx_boundary=self.idx_boundary,
                                        size_boundary=self.size_boundary,
                                        distance_boundary=self.distance_boundary)
        else:
            ret = LCP_construction_path(self.alpha, self.V, self.id_low,
                                        self.qlow0, self.qn0,
                                        Hnew, HnewT, type=self.type)
        self.band_V = ret['deltaLCP']
        self.Smat = ret['Smat']

    def LCP_auto_tune(self, V0, H0, hs, B=5, delta=0.05, lambda_=0, trace=True):
        if self.type == "distance":
            ret = autoTune_distance(V0, self.n, hs, H0, alpha=self.alpha,
                                    delta=delta, B=B, trace=trace, lambda_=lambda_)
        else:
            raise ValueError("unsupported localizer for auto-tune.")
        return ret
