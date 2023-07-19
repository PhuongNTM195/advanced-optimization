import numpy as np
import time
from backtracking_linesearch import backtracking_line_search_wolfe1
 

def newton_method(ls, alpha=None, use_linesearch=False,
                              max_iteration=1e4,
                              epsilon=1e-6,
                              x_start=None,
                              alpha_bar=0.5,
                              ro=0.5,
                              c=0.5):
    # intialization x_start to start:
    if x_start:
        x_current = x_start
    else:
        x_current = np.random.normal(loc=0, scale=1, size=ls.n)
    # keep track of interation
    x_history = []
    f_value = []
    time_history = []
    hess = ls.hess
    alpha_lst = []
    start_time = time.time()
    for k in range(int(max_iteration)):
        x_history.append(x_current)
#         f_value.append(F(x_current))
        # gradient update
        if use_linesearch:
            alpha = backtracking_line_search_wolfe1(ls, x_current, alpha_bar=alpha_bar, ro=ro, c=c)
        grad = ls.grad_F(x_current)
#         print(grad.shape, hess.shape)
        x_next = x_current - alpha*np.linalg.solve(hess, grad)

        # error stoping condition based on Princenton slide
        if np.linalg.norm(x_next - x_current) <= epsilon*np.linalg.norm(x_current):
            break

        # update x
        f_value.append(ls.F(x_next))
        time_history.append(time.time() - start_time)
        alpha_lst.append(alpha)
        x_current = x_next
        #alpha = backtracking_line_search(model, x_current)

    print('GD finished after ' + str(k) + ' iterations')

    return {'solution': x_current,
            'f_value': f_value,
            'time': time_history,
            'x_history': x_history,
            'alpha': alpha_lst}