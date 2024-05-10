import numpy as np
import warnings
# from .lp import emd
# from .bregman import sinkhorn
from ..utils import list_to_array
from ot.backend import get_backend
from ot.lp import emd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from scipy.optimize import scalar_search_armijo
    except ImportError:
        from scipy.optimize.linesearch import scalar_search_armijo
        
def line_search_armijo(
    f, xk, pk, gfk, old_fval, args=(), c1=1e-4,
    alpha0=0.99, alpha_min=0., alpha_max=None, nx=None, **kwargs
):
    if nx is None:
        xk, pk, gfk = list_to_array(xk, pk, gfk)
        xk0, pk0 = xk, pk
        nx = get_backend(xk0, pk0)
    else:
        xk0, pk0 = xk, pk

    if len(xk.shape) == 0:
        xk = nx.reshape(xk, (-1,))

    xk = nx.to_numpy(xk)
    pk = nx.to_numpy(pk)
    gfk = nx.to_numpy(gfk)

    fc = [0]

    def phi(alpha1):
        # it's necessary to check boundary condition here for the coefficient
        # as the callback could be evaluated for negative value of alpha by
        # `scalar_search_armijo` function here:
        #
        # https://github.com/scipy/scipy/blob/11509c4a98edded6c59423ac44ca1b7f28fba1fd/scipy/optimize/linesearch.py#L686
        #
        # see more details https://github.com/PythonOT/POT/issues/502
        alpha1 = np.clip(alpha1, alpha_min, alpha_max)
        # The callable function operates on nx backend
        fc[0] += 1
        alpha10 = nx.from_numpy(alpha1)
        fval = f(xk0 + alpha10 * pk0, *args)
        if isinstance(fval, float):
            # prevent bug from nx.to_numpy that can look for .cpu or .gpu
            return fval
        else:
            return nx.to_numpy(fval)

    if old_fval is None:
        phi0 = phi(0.)
    elif isinstance(old_fval, float):
        # prevent bug from nx.to_numpy that can look for .cpu or .gpu
        phi0 = old_fval
    else:
        phi0 = nx.to_numpy(old_fval)

    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0, amin=alpha_min)

    if alpha is None:
        return 0., fc[0], nx.from_numpy(phi0, type_as=xk0)
    else:
        alpha = np.clip(alpha, alpha_min, alpha_max)
        return nx.from_numpy(alpha, type_as=xk0), fc[0], nx.from_numpy(phi1, type_as=xk0)
    
def cg(a, b, M, reg, f, df, G0=None, mask=None, line_search=line_search_armijo,
       numItermax=200, numItermaxEmd=100000, stopThr=1e-9, stopThr2=1e-9,
       verbose=False, log=False, **kwargs):
    def lp_solver(a, b, M, **kwargs):
        return emd(a, b, M, numItermaxEmd, log=True)

    return generic_conditional_gradient(a, b, M, f, df, reg, None, lp_solver, line_search, G0=G0, mask=mask,
                                        numItermax=numItermax, stopThr=stopThr,
                                        stopThr2=stopThr2, verbose=verbose, log=log, **kwargs)
    
def generic_conditional_gradient(a, b, M, f, df, reg1, reg2, lp_solver, line_search, G0=None, mask=None,
                                 numItermax=200, stopThr=1e-9,
                                 stopThr2=1e-9, verbose=False, log=False, **kwargs):
    a, b, M, G0 = list_to_array(a, b, M, G0)
    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M)

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        # to not change G0 in place.
        G = nx.copy(G0)
    # print("G: ", G)
    if reg2 is None:
        def cost(G):
            return nx.sum(M * G) + reg1 * f(G)
    else:
        def cost(G):
            return nx.sum(M * G) + reg1 * f(G) + reg2 * nx.sum(G * nx.log(G))
    cost_G = cost(G * mask.T)
    if log:
        log['loss'].append(cost_G)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, 0, 0))

    while loop:

        it += 1
        old_cost_G = cost_G
        # problem linearization
        Mi = M + reg1 * df(G*mask.T)

        if not (reg2 is None):
            Mi = Mi + reg2 * (1 + nx.log(G*mask.T))
        # set M positive
        Mi = Mi + nx.min(Mi)

        # solve linear program
        Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)

        # line search
        deltaG = Gc - G
 
        alpha, fc, cost_G = line_search(cost, G*mask.T, deltaG*mask.T, Mi, cost_G, **kwargs)

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax[0]:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = abs_delta_cost_G / abs(cost_G) if cost_G != 0. else np.nan
        if relative_delta_cost_G < stopThr or abs_delta_cost_G < stopThr2:
            loop = 0

        if log:
            log['loss'].append(cost_G)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, cost_G, relative_delta_cost_G, abs_delta_cost_G))

    if log:
        log.update(innerlog_)
        return G, log
    else:
        return G