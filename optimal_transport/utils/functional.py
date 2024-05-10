import numpy as np
import torch
from ot.backend import get_backend

def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]
    
def unif(n, type_as=None):
    if type_as is None:
        return np.ones(n) / n
    else:
        nx = get_backend(type_as)
        return nx.ones(n, type_as=type_as) / n

def solve_1d_linesearch_quad(a, b):
    if a > 0:  # convex
        minimum = min(1., max(0., -b / (2.0 * a)))
        return minimum
    else:  # non convex
        if a + b < 0:
            return 1.
        else:
            return 0.
        
def update_square_loss(p, lambdas, T, Cs, nx=None):
    if nx is None:
        nx = get_backend(p, *T, *Cs)

    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s], Cs[s]),
            T[s].T
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return tmpsum / ppt

# def update_square_loss_kpg(R, L, p, lambdas, T, Cs, nx=None, alpha=0.5, normalized=True, eps=1e-10):     
def update_square_loss_kpg(Cs, Ct, I, J, L, p, a, b, lambdas, T, nx=None, alpha=0.5, lr=0.001):
    C = [Cs, Ct]
    if nx is None:
        nx = get_backend(p, *T, *C)
    
    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s], C[s]),
            T[s].T
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)
    
    Cz = tmpsum / ppt
    # d = C.shape[0]
    # Rs_1 = R[0]
    # Rs = R[1]
    # Rs_2 = R[2]
    # Ts_1 = T[0]
    # Ts_2 = T[1]
    # gradient_r = np.zeros_like(Rs)
    # C_prev = C.copy()
    # iter = 0
    
    # while (iter < max_iters):
    #     for i in range(d):
    #         for j in range(d):
                
    #             if j in L:
    #                 # gradient_gw = ppt[i][j] * 2 * C[i][j] + sum([lambdas[s] * C[i][j] * nx.dot(nx.dot(T[s], Cs[s]),T[s].T)[i][j] for s in range(len(T))])
    #                 gradient_kpg = 0
    #                 for l in range(len(L)):
    #                     if j == L[l]:
    #                         gradient_r[i][l] = -1/rho *  Rs[i][l] * (1 - Rs[i][l])
    #                     else:
    #                         gradient_r[i][l] = -1/rho *  Rs[i][l] * (- Rs[i][l])
    #                 for t in range(len(Rs_1[0])):
    #                     x = sum(Rs_1[t] * gradient_r[i].T) * -2 + 2 * sum(Rs[i] * gradient_r[i].T)
    #                     gradient_kpg += x * Ts_1[i][t] 
    #                 for t in range(len(Rs_2[0])):
    #                     x = sum(Rs_2[t] * gradient_r[i].T) * -2 + 2 * sum(Rs[i] * gradient_r[i].T)
    #                     gradient_kpg += x * Ts_2[i][t] 
                        
    #                 # gradient = gradient_gw + gradient_kpg
    #                 gradient = gradient_kpg
    #                 C[i][j] -= alpha * gradient
    #                 C[j][i] = C[i][j]
                   
    #     if np.allclose(C, C_prev, atol=epsilon):
    #         print("Converged.")
    #         break
        
    #     C_prev = C.copy()
    #     iter += 1
    
    def _guide_matrix(
        Cs: np.ndarray, Ct: np.ndarray,
        I: np.ndarray, J: np.ndarray,
    ) -> np.ndarray:
        rho = 0.1
        Cs_kp = Cs[:, I]
        Ct_kp = Ct[:, J]
        R1 = softmax(-2 * Cs_kp / rho)
        R2 = softmax(-2 * Ct_kp / rho)
        
        G = JS_matrix(R1, R2)
        return G
    
    Gs = _guide_matrix(Cs, Cz, I, L)
    Gt = _guide_matrix(Cz, Ct, L, J)
    
    def f1(a):
        return (a**2).to(torch.float32)

    def f2(b):
        return (b**2).to(torch.float32)

    def h1(a):
        return a.to(torch.float32)

    def h2(b):
        return (2 * b).to(torch.float32)
    
    Cz = torch.tensor(Cz, requires_grad=True)
    Cs, Ct, Gs, Gt, p = torch.Tensor(Cs), torch.Tensor(Ct), torch.Tensor(Gs), torch.Tensor(Gt), torch.Tensor(p)
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    T[0] =  torch.Tensor(T[0])
    T[1] =  torch.Tensor(T[1])
    C = [Cs, Ct]

    # loss_gw = sum([lambdas[s] * torch.dot((
    #                     torch.mm(torch.mm(f1(Cz), p.unsqueeze(1)), torch.ones(1, len(ps[s]))) + 
    #                     torch.mm(torch.mm(torch.ones(len(p), 1), ps[s].unsqueeze(0)), f2(C[s])) - 
    #                     torch.mm(torch.mm(h1(Cz), T[s]), h2(C[s]).t())).flatten(),
    #                     T[s].flatten())
    #             for s in range(len(T))])
    # print(loss_gw)
    loss_gw = lambdas[0] * torch.dot((
                        torch.mm(torch.mm(f1(Cz), p.unsqueeze(1)), torch.ones(1, len(a))) + 
                        torch.mm(torch.mm(torch.ones(len(p), 1), a.unsqueeze(0)), f2(C[0])) - 
                        torch.mm(torch.mm(h1(Cz), T[0]), h2(Cs).t())).flatten(),
                        T[0].flatten()) + lambdas[1] * torch.dot((
                        torch.mm(torch.mm(f1(Cz), p.unsqueeze(1)), torch.ones(1, len(b))) + 
                        torch.mm(torch.mm(torch.ones(len(p), 1), b.unsqueeze(0)), f2(C[1])) - 
                        torch.mm(torch.mm(h1(Cz), T[1]), h2(C[1]).t())).flatten(),
                        T[1].flatten())
    
    loss_kpg = torch.dot(T[0].flatten(), Gs.flatten()) + torch.dot(T[1].flatten(), Gt.flatten())
    # print(loss_kpg)
    
    loss = alpha * loss_gw + (1- alpha) * loss_kpg
    loss.backward()
    
    optim = torch.optim.SGD([Cz], lr=lr)
    optim.step()
    # print(Cz)
    
    return Cz.detach().numpy()

def update_kl_loss(p, lambdas, T, Cs, nx=None):
    if nx is None:
        nx = get_backend(p, *T, *Cs)

    # Correct order mistake in Equation 15 in [12]
    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s], nx.log(nx.maximum(Cs[s], 1e-15))),
            T[s].T
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return nx.exp(tmpsum / ppt)

def check_random_state(seed):
    r"""Turn `seed` into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If `seed` is None, return the RandomState singleton used by np.random.
        If `seed` is an int, return a new RandomState instance seeded with `seed`.
        If `seed` is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))

def softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
    return f_x

def KL_matrix(p, q, eps=1e-10):
    return np.sum(p * np.log(p + eps) - p * np.log(q + eps), axis=-1)


def JS_matrix(P, Q, eps=1e-10):
    P = np.expand_dims(P, axis=1)
    Q = np.expand_dims(Q, axis=0)
    kl1 = KL_matrix(P, (P + Q) / 2, eps)
    kl2 = KL_matrix(Q, (P + Q) / 2, eps)
    return 0.5 * (kl1 + kl2)