import numpy as np
import time
from backtracking_linesearch import backtracking_line_search_wolfe1
 

def accelerated_gradient_descent(ls, alpha=None,
                                 use_linesearch=False,
                                 max_iteration=1e4,
                                 epsilon=1e-6,
                                 x_start=None,
                                 alpha_bar=1,
                                 ro=0.5,
                                c=0.5):
    """
    Gradient descent

    Parameters
    ----------
    alpha: step length
    max_iterations: maximum number of gradient iterations
    epsilon: tolerance for stopping condition
    x_start: where to start (otherwise random)

    Output
    ------
    solution: final x* value
    f_value: f(x*)
    x_history: beta values from each iteration
    """
    # intialization x_start to start:
    if x_start:
        x_current = x_start
    else:
        x_current = np.random.normal(loc=0, scale=1, size=ls.n)
    # keep track of interation
    x_history = []
    f_value = []
    time_history = []
    alpha_lst = []
    start_time = time.time()
    for k in range(1, int(max_iteration)):

        x_history.append(x_current)

        v = x_history[k-1] + (k-2)/(k+1)*(x_history[k-1] - x_history[k-2])
        # gradient update
        if use_linesearch:
            alpha = backtracking_line_search_wolfe1(ls, x_current, alpha_bar=alpha_bar, ro=ro, c=c)
        x_next = v - alpha * ls.grad_F(v)

        # error stoping condition based on Princenton slide
        if np.linalg.norm(x_next - x_current) <= epsilon*np.linalg.norm(x_current):
            break

        # update x
        x_current = x_next
        f_value.append(ls.F(x_current))
        alpha_lst.append(alpha)
        time_history.append(time.time() - start_time)
        #alpha = backtracking_line_search(model, x_current)

    print('GD finished after ' + str(k) + ' iterations')

    return {'solution': x_current,
            'f_value': f_value,
            'time': time_history,
            'x_history': x_history,
            'alpha': alpha_lst}