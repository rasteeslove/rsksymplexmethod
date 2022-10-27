"""
Основная фаза симплекс-метода.
"""

import copy
import numpy as np

from rsksymplexmethod import utils


# the max number of iterations to attempt:
MAX_ITER = 42


def iteration(c, A, x, B, Ab_inv_prev, index):
    """
    Итерация основной фазы симплекс-метода.

    INPUT:
    - c
    - A
    - x
    - B
    - Ab_inv_prev: матрица обратная матрице Ab из предыдущей итерации
      (на первой итерации - None)
    - index: the index of the element of the B vector that got changed
      in the previous iteration

    OUTPUT: (iteration bundle tuple)
    - x
    - B
    - unbound
    - solved
    - Ab_inv: матрица обратная Ab с текущей итерации
    - index: the index of the element of the B vector that got changed
      in this iteration
    """
    n = len(c)

    nB = utils.list_diff(B, list(range(n)))
    Ab = A[:, B]

    if Ab_inv_prev is None and index is None:
        Ab_inv = np.linalg.inv(Ab)
    else:
        Ab_inv = utils.smart_invertion(Ab_inv_prev, Ab[:, index], index)

    cb = [c[i] for i in B]
    u = cb @ Ab_inv
    delta = u @ A - c
    nB_delta = [el for i, el in enumerate(delta) if i in nB]

    if all([el >= 0 for el in nB_delta]):
        return x, B, False, True, None, None

    j0 = 0
    for i, v in enumerate(delta):
        if v < 0:
            j0 = i
            break

    z = Ab_inv @ A[:, j0]
    theta = np.array([x[j]/z_i if z_i > 0 else np.inf
                      for z_i, j in zip(z, B)])

    theta0 = np.amin(theta)
    if theta0 == np.inf:
        return x, B, True, True, None, None
    theta0_index = np.where(theta == theta0)[0][0] # np.where returns a tuple hence the indexing

    j_ast = B[theta0_index]
    j_ast_index = np.where(np.array(B) == j_ast)[0][0]

    B[j_ast_index] = j0
    nB = utils.list_diff(B, list(range(n)))
    for i in nB:
        x[i] = 0
    x[j0] = theta0
    for j_index, j in enumerate(B):
        if j != j0:
            x[j] -= theta0*z[j_index]
    
    return x, B, False, False, Ab_inv, j_ast_index


def run(c, A, x, B):
    """
    The main phase algorithm of the custom 2-phase symplex method.
    
    INPUT:
    - c: np.array: вектор стоимостей из целевой функции
    - A: 2d np.array: матрица ограничений
    - x: np.array: начальный базисный допустимый план
    - B: множество базисных индексов

    OUTPUT: (main phase bundle dict)
    - iter_num: number of iterations made
    - unbound: True, если целевая функция не ограничена сверху на
      множестве допустимых планов
    - solved: True, если решение ЗЛП найдено
    - x: итоговый допустимый план
    - B: итоговое множество базисных индексов
    """

    # the main phase is run only once or twice for a LPP
    # hence the copying wouldn't affect the speed much
    c = copy.deepcopy(c)
    A = copy.deepcopy(A)
    x = copy.deepcopy(x)
    B = copy.deepcopy(B)

    n = len(c)
    m = len(A)

    # some assertions could be made here for the reliability's sake

    Ab_inv = None
    index = None
    for i in range(MAX_ITER):
        x, B, unbound, solved, Ab_inv, index = iteration(c, A, x, B, Ab_inv, index)
        if unbound:
            return {
                'iter_num': i+1,
                'unbound': True,
                'solved': True,
                'x': x,
                'B': B,
            }
        if solved:
            return {
                'iter_num': i+1,
                'unbound': False,
                'solved': True,
                'x': x,
                'B': B,
            }

    # the "I give up" result
    return {
        'iter_num': i+1,
        'unbound': False,
        'solved': False,
        'x': x,
        'B': B,
    }
