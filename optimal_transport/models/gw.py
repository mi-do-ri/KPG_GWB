from ._ot import _OT
from ..utils import Distance, SquaredEuclidean, KLDiv, softmax, JSDiv
from ..utils import list_to_array, Distance, SquaredEuclidean, unif, update_kl_loss, update_square_loss, update_square_loss_kpg, check_random_state
from sklearn.metrics import pairwise_distances_argmin_min
from ot.backend import get_backend

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
from numpy import linalg as LA
from sklearn.cluster import KMeans
import ot
from scipy import stats
from scipy.sparse import random
import cvxpy as cvx
import scipy.optimize._linprog as linprog

class KeypointGW(_OT):
    def __init__(
        self,
        # ys: np.ndarray,
        distance: Distance = SquaredEuclidean,
        similarity: Distance = JSDiv,
        p: np.ndarray = None,
        lambdas: list = None,
        loss_fun = 'square_loss',
        num_free_barycenters: Optional[int] = None,
        n_clusters: int = 3,
        sinkhorn_reg: float = 0.01, 
        temperature: float = 0.1, 
        div_term: float = 1e-10, 
        alpha: float = 0.95,
        stop_thr: float = 1e-5, 
        tol: float = 1.5e-5,
        max_iters: int = 100,
        sinkhorn_max_iters: int = 100,
        fused: bool = True,
        normalized = True,
        learning_rate: int =0.001
    ):
        super().__init__(distance)
        self.N = n_clusters + num_free_barycenters,
        # self.ys = ys
        self.sim_fn = similarity
        self.dist_fn = distance
        self.k = num_free_barycenters
        self.eps = sinkhorn_reg
        self.rho = temperature
        self.div_term = div_term
        self.stop_thr = stop_thr
        self.max_iters = max_iters
        self.sinkhorn_max_iters = sinkhorn_max_iters
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.fused = fused
        self.p = p
        self.lambdas = lambdas
        self.loss_fun = loss_fun
        self.tol = tol
        self.normalized = normalized
        self.lr = learning_rate

        self.Pa_: Optional[np.ndarray] = None
        self.Pb_: Optional[np.ndarray] = None
        self.z_: Optional[np.ndarray] = None
        
    def fit(
        self,
        xs: np.ndarray,
        ys: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        K: List[Tuple],
        **kwargs,
    ):
        z, p, L = self._init_barycenters(xs, ys, self.n_clusters, self.k, len(K))
        I, J = self._init_keypoint_inds(K)
        # print(L)

        Ms, Mt = self._init_masks(xs, z, xt, I, L, J)

        Cs = self.dist_fn(xs, xs)
        Cs = Cs / (Cs.max() + self.div_term)
        Ct = self.dist_fn(xt, xt)
        Ct = Ct / (Ct.max() + self.div_term)
        Cz = self.dist_fn(z, z)
        Cz = Cz / (Cz.max() + self.div_term)
        
        C = [Cs, Ct]
        C = list_to_array(*C)
        
        arr = [*C]
        ps = [a, b]
        
        if ps is not None:
            arr += list_to_array(*ps)
        else:
            ps = [unif(C.shape[0], type_as=C) for C in C]
        if p is not None:
            arr.append(list_to_array(p))
        else:
            p = unif(self.N, type_as=Cs[0])

        nx = get_backend(*arr)
        
        S = len(C)
        if self.lambdas is None:
            self.lambdas = [1. / S] * S
        
        for i in range(self.max_iters):      
            # print(i)   
            self.Cz_ = Cz   
            Gs, Rs_1, Rs = self._guide_matrix(Cs, Cz, I, L)
            Gt, Rs, Rs_2 = self._guide_matrix(Cz, Ct, L, J)

            Ps = self._update_plans(Cs, Cz, a, p, Gs, Ms)
            Pt = self._update_plans(Cz, Ct, p, b, Gt, Mt)
            # print(Pt)
            T = [Ps.T, Pt]
            R = [Rs_1, Rs, Rs_2]
            
            Cz = self._update_barycenters(Cs, Ct, I, J, L, p, a, b, self.lambdas, T, nx, self.alpha, self.lr)
            # print(Cz)

            err = nx.norm(Cz - self.Cz_)
            # print(err)
            if err <= self.tol:
                # print(f"Threshold reached at iteration {i}")
                break

        self.Pa_ = Ps
        self.Pb_ = Pt
        self.Cz_ = Cz
        self.L = L
        self.P_ = Ps.dot(Pt)
        
        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        m = xt.shape[0]
        assert self.P_ is not None, "Should run fit() before mapping"

        return np.dot(self.P_, xt) / (np.dot(self.P_, np.ones((m, 1))) + self.div_term)

    def _init_keypoint_inds(
        self,
        K: List[Tuple]
    ) -> Tuple[np.ndarray]:
        I = np.array([pair[0] for pair in K])
        J = np.array([pair[1] for pair in K])
        # L = np.arange(len(K))
        return I, J

    def _init_barycenters(
        self, 
        x: np.ndarray,
        y: np.ndarray,
        n_clusters: int,
        n_free_anchors: int,
        n_keypoints: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (n_keypoints + n_free_anchors) % n_clusters == 0: 
            T = (int)(n_keypoints + n_free_anchors) // n_clusters

            unique_labels = np.unique(y)
            selected_centers = []

            for label in unique_labels:
                # Lấy chỉ mục của các điểm thuộc class hiện tại
                indices = np.where(y == label)[0]

                # Tính trung bình của các điểm trong class
                center = np.mean(x[indices], axis=0)

                # Tính khoảng cách giữa mỗi điểm và trung tâm
                distances = pairwise_distances_argmin_min(x[indices], [center])[1]

                # Chọn 2 điểm có khoảng cách nhỏ nhất
                selected_indices = indices[np.argsort(distances)[:T]]
                selected_centers.extend(selected_indices)

            Z = x[selected_centers]
            L = [i * T for i in range(len(Z)//T)]
            # Z = np.vstack(np.array(Z))
            h = np.ones(len(Z)) / (len(Z))
        else:
            model = KMeans(n_clusters=n_keypoints + n_free_anchors)
            model.fit(x)
            Z = model.cluster_centers_
            # Z = pd.DataFrame(z)
            # L = np.arange(n_keypoints)
            L = [i for i in range(n_keypoints)]
            h = np.ones(len(Z)) / (len(Z))

        return Z, h, L
    
    def _init_masks(
        self,
        xs: np.ndarray, z: np.ndarray, xt: np.ndarray,
        I: np.ndarray, L: np.ndarray, J: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Ms = self._guide_mask(xs, z, I, L)
        Mt = self._guide_mask(z, xt, L, J)
        return Ms, Mt

    def _guide_mask(
        self,
        xs: np.ndarray, xt: np.ndarray,
        I: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        mask = np.ones((xs.shape[0], xt.shape[0]))
        mask[I, :] = 0
        mask[:, J] = 0
        mask[I, J] = 1
        return mask

    def _guide_matrix(
        self,
        Cs: np.ndarray, Ct: np.ndarray,
        I: np.ndarray, J: np.ndarray,
    ) -> np.ndarray:

        Cs_kp = Cs[:, I]
        Ct_kp = Ct[:, J]
        R1 = softmax(-2 * Cs_kp / self.rho)
        R2 = softmax(-2 * Ct_kp / self.rho)
        
        G = self.sim_fn(R1, R2, eps=self.div_term)
        return G, R1, R2

    def _init_matrix(
        self,
        Cx: np.ndarray, Cy: np.ndarray,
        p: np.ndarray, q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        def fx(a):
            return (a ** 2)
        def fy(b):
            return (b ** 2)
        def hx(a):
            return a
        def hy(b):
            return (b * 2)
        
        constCx = np.dot(np.dot(fx(Cx), p.reshape(-1, 1)),
                            np.ones(len(q)).reshape(1, -1))
        constCy = np.dot(np.ones(len(p)).reshape(-1, 1),
                            np.dot(q.reshape(1, -1), fy(Cy).T))
        
        constC = constCx + constCy
        hCx = hx(Cx)
        hCy = hy(Cy)
        
        return constC, hCx, hCy
    
    def _product(self, constC, hCx, hCy, T):
        A = -np.dot(np.dot(hCx, T), (hCy.T))
        tens = constC + A
        return tens
    
    def _gwloss(self, constC, hCx, hCy, T):
        tens = self._product(constC, hCx, hCy, T)
        return np.sum(tens * T)

    def _gwggrad(self, constC, hCx, hCy, T):
        return 2 * self._product(constC, hCx, hCy, T)
    
    def _update_barycenters(
        self, 
        Cs: np.ndarray,
        Ct: np.ndarray,
        I: np.ndarray, 
        J: np.ndarray,
        L: np.ndarray, 
        p: np.ndarray, 
        a: np.ndarray,
        b: np.ndarray, 
        lambdas,
        T, 
        nx: np.ndarray, 
        alpha, 
        lr
        # R: np.ndarray,
        # L: np.ndarray,
        # p: np.ndarray, 
        # lambdas: np.ndarray,
        # T: np.ndarray, 
        # Cs: np.ndarray,
        # nx: np.ndarray,
        # normalized: bool,
        # eps
    ) -> np.ndarray:
        if self.loss_fun == 'square_loss':
            # C = update_square_loss_kpg(R, L, p, lambdas, T, Cs, nx, normalized, eps)
            C = update_square_loss_kpg(Cs, Ct, I, J, L, p, a, b, self.lambdas, T, nx, self.alpha, self.lr)
            
        # elif self.loss_fun == 'kl_loss':
        #     C = update_kl_loss(p, lambdas, T, Cs, nx)
        return C

    def _update_plans(
        self,
        Cx: np.ndarray, Cy: np.ndarray,
        p: np.ndarray, q: np.ndarray,
        G: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        G = G / (G.max() + self.div_term)
        
        constC, hCx, hCy = self._init_matrix(Cx, Cy, p, q)
        G0 = p.reshape(-1, 1) * q.reshape(1, -1)
        
        def f(G):
            return self._gwloss(constC, hCx, hCy, G)
        
        def df(G):
            return self._gwggrad(constC, hCx, hCy, G)
        
        P, f_val = self._cg(p, q, Cx, Cy, constC, f, df, G0, mask, G)
        self.f_val = f_val
        
        return P
    
    def _cg(
        self,
        p: np.ndarray, q: np.ndarray,
        Cx: np.ndarray, Cy: np.ndarray,
        constC: np.ndarray, f, df, 
        G0: np.ndarray, mask: np.ndarray,
        G: np.ndarray
    ) -> np.ndarray:
        numItermax = 5
        loop = 1
        
        def cost(G0):
            if not self.fused:
                return f(G0)
            else:
                return self.alpha * f(G0) + (1.0 - self.alpha) * np.sum(mask * G * G0)
        
        def cost_mask(G0):
            return cost(mask * G0)
        
        if mask is None:
            f_val = cost(G0)
        else:
            f_val = cost_mask(G0)
            
        it = 0
        
        while loop:
            it += 1
            old_fval = f_val
            
            if mask is None:
                dfG = df(G0)
            else:
                dfG = df(mask * G0)
            
            if self.fused: 
                M = self.alpha * dfG + (1.0 - self.alpha) * mask * G
            else:
                M = dfG
                
            M += M.min()
            
            Gc = self._sinkhorn_log_domain(p, q, M, mask)
            
            deltaG = Gc - G0
            
            if mask is None:
                alpha1, f_val = self._solve_linesearch(cost, G0, deltaG, Cx, Cy, constC)
            else:
                alpha1, f_val = self._solve_linesearch(cost, mask*G0, mask*deltaG, Cx, Cy, constC)
                
            G0 += alpha1 * deltaG
            # print("G0: ", G0)
            # print("alpha1: ", alpha1)
            
            if it >= numItermax:
                loop = 0
            
            abs_delta_fval = abs(f_val - old_fval)
            relative_delta_fval = abs_delta_fval / abs(f_val)
            if relative_delta_fval < self.stop_thr or abs_delta_fval < self.stop_thr:
                loop = 0
        
        return G0, f_val
    
    def _solve_linesearch(
        self,
        cost, G: np.ndarray, deltaG: np.ndarray, 
        Cx: np.array, Cy: np.ndarray, 
        constC: np.ndarray
    ):
        dotx = np.dot(Cx, deltaG)
        dotxy = np.dot(dotx, Cy)
        a = -2 * np.sum(dotxy * deltaG)
        b = np.sum(constC * deltaG) - 2 * (np.sum(dotxy * G) + np.sum(np.dot(np.dot(Cx, G), Cy) * deltaG))
        c = cost(G)
        
        alpha = self._solve_1d_linesearch_quad(a, b, c)
        f_val = cost(G + alpha * deltaG)
        
        return alpha, f_val
    
    def _solve_1d_linesearch_quad(
        self,
        a, b, c
    ):
        f0 = c
        df0 = b
        f1 = a + f0 + df0

        if a > 0:  # convex
            minimum = min(1, max(0, -b/2.0 * a))
            return minimum
        else:  # non convex
            if f0 > f1:
                return 1
            else:
                return 0
    
    def _sinkhorn_log_domain(
        self,
        p: np.ndarray, q: np.ndarray,
        C: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            M =  (-C + np.expand_dims(u,1) + np.expand_dims(v,0)) / self.eps
            if mask is not None:
                M[mask==0] = -1e6
            return M

        def lse(A):
            "log-sum-exp"
            max_A = np.max(A, axis=1, keepdims=True)
            return np.log(np.exp(A-max_A).sum(1, keepdims=True) + self.div_term) + max_A  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        for i in range(self.sinkhorn_max_iters): #self.max_iters
            u1 = u  # useful to check the update
            u = self.eps * (np.log(p) - lse(M(u, v)).squeeze()) + u
            v = self.eps * (np.log(q) - lse(M(u, v).T).squeeze()) + v
            err = np.linalg.norm(u - u1)
            if err < self.stop_thr:
                print(f"Threshold of sinkhorn reached at iteration {i}")
                break

        U, V = u, v
        P = np.exp(M(U, V))  # P = diag(a) * K * diag(b)
        return P


class GW(_OT):
    def __init__(
        self, 
        distance: Distance = SquaredEuclidean,
        loss_fun = 'square_loss',
        max_iters = 10000,
        tol_rel = 1e-09,
        div_term: float = 1e-10
    ):
        super().__init__(distance)
        self.loss_fun = loss_fun,
        self.max_iters = max_iters,
        self.tol_rel = tol_rel
        self.div_term = div_term
    
    def fit(
        self,
        xs: np.ndarray,
        ys: Optional[np.ndarray],
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        K,
        **kwargs,
    ):
        Cs = self.dist_fn(xs, xs)
        Cs = Cs / (Cs.max() + self.div_term)
        Ct = self.dist_fn(xt, xt)
        Ct = Ct / (Ct.max() + self.div_term)
        
        self.P_ = ot.gromov.gromov_wasserstein(Cs, Ct, a, b, 'square_loss', verbose=False)
        # self.P = self.P_[0]
        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        m = xt.shape[0]
        assert self.P_ is not None, "Should run fit() before mapping"

        return np.dot(self.P_, xt) / (np.dot(self.P_, np.ones((m, 1))) + self.div_term)
        
        
class GW_Barycenter(_OT):
    def __init__(
        self, 
        distance: Distance = SquaredEuclidean,
        lambdas: list = None,
        N: int = None,
        loss_fun = 'square_loss',
        max_iters=1000,
        stop_criterion='barycenter',
        random_state=None, 
        log = True,
        div_term: float = 1e-10,
        p = None
    ):
        super().__init__(distance)
        self.lambdas = lambdas,
        self.N = N,
        self.loss_fun = loss_fun,
        self.max_iters = max_iters,
        self.stop_criterion = stop_criterion,
        self.random_state = random_state,
        self.log = log
        self.div_term = div_term
        self.p = p
        
    def fit(
        self,
        xs: np.ndarray, 
        ys: np.ndarray,
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        K,
        **kwargs,
    ):
        Cs = self.dist_fn(xs, xs)
        Cs = Cs / (Cs.max() + self.div_term)
        Ct = self.dist_fn(xt, xt)
        Ct = Ct / (Ct.max() + self.div_term)
        
        C = [Cs, Ct]
        ps = [a, b]

        if self.p is None: 
            self.p = unif(self.N[0], type_as=Cs[0])

        self.T = self.gromov_barycenters(self.N[0], C, ps, self.p, log=self.log)
        Ps = self.T[0]
        self.Pa_ = Ps[0].T
        self.Pb_ = Ps[1]
        self.P_ = Ps[0].T.dot(Ps[1])
        # print(self.T)
        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        m = xt.shape[0]
        assert self.P_ is not None, "Should run fit() before mapping"

        return np.dot(self.P_, xt) / (np.dot(self.P_, np.ones((m, 1))) + self.div_term)
    
    def gromov_barycenters(
            self,
            N, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
            symmetric=True, armijo=False, max_iter=1000, tol=1e-9,
            stop_criterion='barycenter', warmstartT=False, verbose=False,
            log=False, init_C=None, random_state=None, **kwargs):
        if stop_criterion not in ['barycenter', 'loss']:
            raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

        Cs = list_to_array(*Cs)
        arr = [*Cs]
        if ps is not None:
            arr += list_to_array(*ps)
        else:
            ps = [unif(C.shape[0], type_as=C) for C in Cs]
        if p is not None:
            arr.append(list_to_array(p))
        else:
            p = unif(N, type_as=Cs[0])

        nx = get_backend(*arr)

        S = len(Cs)
        if lambdas is None:
            lambdas = [1. / S] * S

        # Initialization of C : random SPD matrix (if not provided by user)
        if init_C is None:
            generator = check_random_state(random_state)
            xalea = generator.randn(N, 2)
            C = self.dist_fn(xalea, xalea)
            C /= C.max()
            C = nx.from_numpy(C, type_as=p)
        else:
            C = init_C

        cpt = 0
        err = 1e15  # either the error on 'barycenter' or 'loss'

        if warmstartT:
            T = [None] * S

        if stop_criterion == 'barycenter':
            inner_log = False
        else:
            inner_log = True
            curr_loss = 1e15

        if log:
            log_ = {}
            log_['err'] = []
            if stop_criterion == 'loss':
                log_['loss'] = []

        while (err > tol and cpt < max_iter):
            if stop_criterion == 'barycenter':
                Cprev = C
            else:
                prev_loss = curr_loss

            # get transport plans
            if warmstartT:
                res = [ot.gromov.gromov_wasserstein(
                    C, Cs[s], p, ps[s], loss_fun, symmetric=symmetric, armijo=armijo, G0=T[s],
                    max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                    for s in range(S)]
            else:
                res = [ot.gromov.gromov_wasserstein(
                    C, Cs[s], p, ps[s], loss_fun, symmetric=symmetric, armijo=armijo, G0=None,
                    max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                    for s in range(S)]
            if stop_criterion == 'barycenter':
                T = res
            else:
                T = [output[0] for output in res]
                curr_loss = np.sum([output[1]['gw_dist'] for output in res])

            # update barycenters
            if loss_fun == 'square_loss':
                C = update_square_loss(p, lambdas, T, Cs, nx)

            elif loss_fun == 'kl_loss':
                C = update_kl_loss(p, lambdas, T, Cs, nx)

            # update convergence criterion
            if stop_criterion == 'barycenter':
                err = nx.norm(C - Cprev)
                if log:
                    log_['err'].append(err)

            else:
                err = abs(curr_loss - prev_loss) / prev_loss if prev_loss != 0. else np.nan
                if log:
                    log_['loss'].append(curr_loss)
                    log_['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

            cpt += 1

        if log:
            log_['T'] = T
            log_['p'] = p

            return T, C, log_
        else:
            return T, C
        
        
class KPG_RL_GW(object):
    def __init__(
        self,
        distance: Distance = SquaredEuclidean,
        alpha: float = 0.5,
        cost_function = "L2",
        algorithm = "lp",
        tau_s: float = 0.1,
        tau_t: float = 0.1,
        normalized: bool = True,
        reg : float = 0.0001,
        max_iteration: int = 100000,
        eps: float = 1e-10,
        thres: float = 1e-5,
        div_term: float = 1e-10
    ):
        super(KPG_RL_GW).__init__(distance)
        self.dist_fn = distance
        self.alpha = alpha
        self.cost_function = cost_function
        self.algorithm = algorithm
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.normalized = normalized
        self.reg = reg
        self.max_iteration = max_iteration
        self.eps = eps
        self.thres = thres
        self.div_term = div_term
        
    def fit(self, xs, ys, xt, p, q, K):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param alpha: float, combination factor in (0,1) for KPG-RL-KP model.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        '''
        I = [tup[0] for tup in K]
        J = [tup[1] for tup in K]

        ## guiding matrix
        Cs = self.cost_matrix(xs, xs)
        Ct = self.cost_matrix(xt, xt)
        if self.normalized:
            Cs /= (Cs.max() + self.eps)
            Ct /= (Ct.max() + self.eps)

        G = self.structure_metrix_relation(Cs, Ct, I, J)

        ## mask matrix
        M = np.ones_like(G)
        M[I, :] = 0
        M[:, J] = 0
        M[I, J] = 1


        # transport plan
        Cs, Ct, G, M, p, q = torch.Tensor(Cs), torch.Tensor(Ct), torch.Tensor(G), torch.Tensor(M), torch.Tensor(
            p), torch.Tensor(q)
        # M, p, q = torch.Tensor(M), torch.Tensor(p), torch.Tensor(q)

        if self.algorithm != "lp" and self.algorithm != "sinkhorn":
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
        self.P_ = self.gromov_wasserstein(Cs, Ct, p, q, Mask=M, OT_algorithm=self.algorithm, fused=True, Cxy=G, alpha=self.alpha)
        return self
    
    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        m = xt.shape[0]
        assert self.P_ is not None, "Should run fit() before mapping"

        return np.dot(self.P_, xt) / (np.dot(self.P_, np.ones((m, 1))) + self.div_term)

    def init_matrix(self, C1, C2, p, q):
        device = C1.device
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b

        constC1 = torch.matmul(torch.matmul(f1(C1), p.view(-1, 1)),
                        torch.ones(len(q)).to(device).view(1, -1))
        constC2 = torch.matmul(torch.ones(len(p)).to(device).view(-1, 1),
                        torch.matmul(q.view(1, -1), f2(C2).t()))
        constC = constC1 + constC2
        hC1 = h1(C1)
        hC2 = h2(C2)

        return constC, hC1, hC2


    def tensor_product(self, constC, hC1, hC2, T):
        A = -torch.matmul(torch.matmul(hC1, T),(hC2.T))
        tens = constC + A
        return tens


    def gwloss(self, constC, hC1, hC2, T):
        tens = self.tensor_product(constC, hC1, hC2, T)
        return torch.sum(tens * T)


    def gwggrad(self,constC, hC1, hC2, T):
        return 2 * self.tensor_product(constC, hC1, hC2,T)


    def gromov_wasserstein(self, C1, C2, p, q, lam = 0.1, Mask = None,OT_algorithm="sinkhorn",max_iter_sinkhorn=2000,numItermax=5,fused=False,Cxy=None,alpha=0.5):

        constC, hC1, hC2 = self.init_matrix(C1, C2, p, q)

        G0 = p.view(-1,1) * q.view(1,-1)

        def f(G):
            return self.gwloss(constC, hC1, hC2, G)

        def df(G):
            return self.gwggrad(constC, hC1, hC2, G)
        pi = self.cg(p, q, C1, C2, constC, f, df, G0, lam=lam, Mask = Mask,OT_algorithm=OT_algorithm,
                max_iter_sinkhorn=max_iter_sinkhorn,numItermax=numItermax,fused=fused,Cxy=Cxy,alpha=alpha)
        return pi
    
    def structure_metrix_relation(self, C1, C2, I, J, tau=0.1):
        # print("get structure data...")
        S = np.zeros((len(C1), len(C2)))
        C1_kp = C1[:, I]
        C2_kp = C2[:, J]
        R1 = self.softmax_matrix(-2*C1_kp / tau)
        R2 = self.softmax_matrix(-2*C2_kp / tau)
        S = self.JS_matrix(R1, R2)
        return S

    def softmax_matrix(self, x):
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
        return f_x


    def KL_matrix(self, p, q, eps=1e-10):
        return np.sum(p * np.log(p + eps) - p * np.log(q + eps), axis=-1)


    def JS_matrix(self, P, Q, eps=1e-10):
        P = np.expand_dims(P, axis=1)
        Q = np.expand_dims(Q, axis=0)
        kl1 = self.KL_matrix(P, (P + Q) / 2, eps)
        kl2 = self.KL_matrix(Q, (P + Q) / 2, eps)
        return 0.5 * (kl1 + kl2)

    def cost_matrix(self, x, y):
        x, y = torch.Tensor(x), torch.Tensor(y)
        Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * torch.matmul(x, y.t())
        # Cxy = np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T
        return Cxy.numpy()

    def cg(self, a, b, C1, C2, constC, f, df,  G0=None, lam = 0.1, Mask=None, OT_algorithm="sinkhorn",max_iter_sinkhorn=1000,numItermax=5, numItermaxEmd=100,
        stopThr=1e-4, stopThr2=1e-4,fused=False,Cxy=None,alpha=0.5):

        loop = 1

        G = G0


        def cost(G):
            if not fused:
                return f(G)
            else:
                return  alpha*f(G) + (1.0-alpha)*torch.sum(Mask*Cxy*G)

        def cost_mask(G):
            return cost(Mask*G)

        if Mask is None:
            f_val = cost(G)
        else:
            f_val = cost_mask(G)

        it = 0

        while loop:

            it += 1
            old_fval = f_val

            # problem linearization
            if Mask is None:
                dfG = df(G)
            else:
                dfG = df(Mask*G)

            if fused:
                Mi = alpha*dfG + (1.0-alpha)*Mask*Cxy
            else:
                Mi = dfG

            # set M positive
            Mi += Mi.min()
            
            # solve OT
            if OT_algorithm == "sinkhorn":
                if Mask is None:
                    Gc = self.sinkhorn_log_domain(a,b,Mi,reg=lam,niter=max_iter_sinkhorn)
                else:
                    Gc = self.sinkhorn_log_domain(a, b, Mi, reg=lam, Mask=Mask,niter=max_iter_sinkhorn)
            if OT_algorithm == "lp":
                if Mask is None:
                    Gc = torch.Tensor(self.lp(a.numpy(),b.numpy(),Mi.numpy()))
                else:
                    Gc = torch.Tensor(self.lp(a.numpy(),b.numpy(),Mi.numpy(),Mask=Mask.numpy()))

            deltaG = Gc - G

            # line search
            if Mask is None:
                alpha1, fc, f_val = self.solve_linesearch(cost, G, deltaG, C1,C2,constC)
            else:
                alpha1, fc, f_val = self.solve_linesearch(cost, Mask*G, Mask*deltaG,C1,C2,constC)

            G = G + alpha1 * deltaG

            # test convergence
            if it >= numItermax:
                loop = 0

            abs_delta_fval = abs(f_val - old_fval)
            relative_delta_fval = abs_delta_fval / abs(f_val)
            if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
                loop = 0

        return G


    def solve_linesearch(self, cost, G, deltaG, C1=None, C2=None, constC=None):

        dot1 = torch.matmul(C1, deltaG)
        dot12 = torch.matmul(dot1,C2)
        a = -2 * torch.sum(dot12 * deltaG)
        b = torch.sum(constC * deltaG) - 2 * (torch.sum(dot12 * G) + torch.sum(torch.matmul(torch.matmul(C1, G),C2) * deltaG))
        c = cost(G)

        alpha = self.solve_1d_linesearch_quad(a, b, c)
        fc = None
        f_val = cost(G + alpha * deltaG)

        return alpha, fc, f_val

    def solve_1d_linesearch_quad(self, a, b, c):
        f0 = c
        df0 = b
        f1 = a + f0 + df0

        if a > 0:  # convex
            minimum = min(1, max(0, -b/2.0 * a))
            return minimum
        else:  # non convex
            if f0 > f1:
                return 1
            else:
                return 0
            
    def sinkhorn_log_domain(self,p,q,C,Mask = None, reg=0.1,niter=10000,thresh = 1e-5):
        C /= C.max()
        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            M =  (-C + np.expand_dims(u,1) + np.expand_dims(v,0)) / reg
            if Mask is not None:
                M[Mask==0] = -1e6
            return M

        def lse(A):
            A = A.numpy()
            "log-sum-exp"
            # return np.log(np.exp(A).sum(1, keepdims=True) + 1e-10)
            max_A = np.max(A, axis=1, keepdims=True)
            return np.log(np.exp(A-max_A).sum(1, keepdims=True) + 1e-10) + max_A  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

        for i in range(niter):
            u1 = u  # useful to check the update
            u = reg * (np.log(p) - lse(M(u, v)).squeeze()) + u
            v = reg * (np.log(q) - lse(M(u, v).T).squeeze()) + v
            err = np.linalg.norm(u - u1)

            actual_nits += 1
            if err < thresh:
                break
        U, V = u, v
        pi = np.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
        return pi

    def lp(self, p, q, C, Mask=None):
        c = np.reshape(C.T,(-1,1))
        b = np.vstack((p.reshape(-1,1),q.reshape(-1,1)))
        A = np.vstack((np.kron(np.ones((1,len(q))),np.eye(len(p))),
                    np.kron(np.eye(len(q)),np.ones((1,len(p))))
                    ))
        if Mask is not None:
            m = np.reshape(Mask.T,(-1,1))
            A = A*(m.T)
            c = c*m
        x = cvx.Variable((len(c),1))
        cons = [x>=0,
                A@x==b]
        obj = cvx.Minimize(c.T@x)
        prob = cvx.Problem(obj,cons)
        prob.solve()

        print(prob.status)
        pi = x.value

        pi = np.reshape(pi,(len(q),len(p)))
        pi = pi.T
        if Mask is not None:
            pi = pi*Mask
        return pi
