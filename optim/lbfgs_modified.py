from math import sqrt
import torch
from functools import reduce
from torch.optim.lbfgs import LBFGS, _strong_wolfe
import logging
log = logging.getLogger(__name__)

# from https://github.com/scipy/scipy/blob/master/scipy/optimize/linesearch.py
def _scalar_search_armijo(phi, phi0, derphi0, args=(), c1=1e-4, alpha0=1, amin=1.0e-8):
    """Minimize over alpha, the function ``phi(alpha)``.
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57
    alpha > 0 is assumed to be a descent direction.
    Returns
    -------
    alpha
    phi1
    """
    log.info(f"LS expected phi: {phi0+c1*alpha0*derphi0} (derphi0: {derphi0})") 
    phi_a0 = phi(alpha0, *args)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1, *args)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2, *args)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1

class LBFGS_MOD(LBFGS):
    r"""
    Extends the original steepest gradient descent of PyTorch
    [`torch.optim.LBFGS <https://pytorch.org/docs/stable/optim.html#torch.optim.LBFGS>`_]
    with optional backtracking linesearch. The linesearch implementation
    is adapted from scipy 
    [`scipy.optimize.linesearch <https://github.com/scipy/scipy/blob/master/scipy/optimize/linesearch.py>`_]
    and relies only on the value of the loss function, not derivatives.
    """
    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None,
                 line_search_eps=1.0e-4):
        r"""
        Args:
            lr : float
                learning rate
            max_iter : int 
                maximal number of iterations per optimization step
            max_eval : int 
                maximal number of function evaluations per optimization step 
                (default: max_iter * 1.25).
            tolerance_grad : float
                termination tolerance on first order optimality
            tolerance_change : float
                termination tolerance on function value/parameter changes.
            history_size : int 
                update history size.
            line_search_fn : str
                either 'strong_wolfe' or ``None``.
            line_search_eps : float
                minimal step size
        """
        super(LBFGS_MOD, self).__init__(
                params,
                lr=lr,
                max_iter=max_iter,
                max_eval=max_eval,
                tolerance_grad=tolerance_grad,
                tolerance_change=tolerance_change,
                history_size=history_size,
                line_search_fn=line_search_fn)
        # TODO for each param group ?
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        group["line_search_eps"]= line_search_eps

    def _directional_evaluate_derivative_free(self, closure, t, x, d):
        self._add_grad(t, d)
        with torch.no_grad():
            orig_loss= closure(True)
        loss= float(orig_loss)
        self._set_param(x)
        return loss

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        with torch.enable_grad():
            loss = float(closure(linesearching=True))
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step_2c(self, closure, closure_linesearch):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            closure_linesearch (callable): A closure that reevaluates the model and returns 
                the loss in torch.no_grad context
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        line_search_eps= group['line_search_eps']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        with torch.enable_grad():
            orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        is_complex= flat_grad.is_complex()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = torch.real(y.conj().dot(s)) if is_complex else y.dot(s)
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.conj().dot(y) if is_complex else ys / y.dot(y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                if is_complex:
                    for i in range(num_old - 1, -1, -1):
                        al[i] = torch.real(old_stps[i].conj().dot(q))  * ro[i]
                        q.add_(old_dirs[i], alpha=-al[i])
                else:
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_stps[i].dot(q)  * ro[i]
                        q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                if is_complex:
                    for i in range(num_old):
                        be_i = torch.real(old_dirs[i].conj().dot(r)) * ro[i]
                        r.add_(old_stps[i], alpha=al[i] - be_i)
                else:
                    for i in range(num_old):
                        be_i = old_dirs[i].dot(r) * ro[i]
                        r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = torch.real(flat_grad.conj().dot(d)) if is_complex else flat_grad.dot(d)

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None and line_search_fn != "default":
                # perform line search, using user function
                if line_search_fn == "backtracking":
                    x_init = self._clone_param()

                    def obj_func(t, x, d):
                        return self._directional_evaluate_derivative_free(closure_linesearch, t, x, d)

                    # return (xmin, fval, iter, funcalls)
                    t, loss= _scalar_search_armijo(obj_func, loss, gtd, args=(x_init,d), alpha0=t)
                    if t is None:
                        raise RuntimeError("minimize_scalar failed")
                
                elif line_search_fn == "strong_wolfe":
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                else:
                    raise RuntimeError("unsupported line search")

                log.info(f"LS final step: {t}")
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss