import torch
from math import sqrt
from functools import reduce
from torch.optim.sgd import SGD
# import torch.optim.sgd as sgd
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

class SGD_MOD(SGD):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """
    def __init__(self, params, lr=1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 line_search_fn=None,
                 line_search_eps=1.0e-4):
        super(SGD_MOD, self).__init__(
            params, lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov)

        if len(self.param_groups) != 1:
            raise ValueError("SGD_MOD doesn't support per-parameter options "
                             "(parameter groups)")

        group = self.param_groups[0]
        group["line_search_fn"]= line_search_fn
        group["line_search_eps"]= line_search_eps
        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._momentum_buffer = None

    def __setstate__(self, state):
        super(SGD_MOD, self).__setstate__(state)

    def _directional_evaluate_derivative_free(self, closure, t, x, d):
        self._add_grad(t, d)
        with torch.no_grad():
            orig_loss= closure()
        loss= float(orig_loss)
        self._set_param(x)
        return loss

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.data.copy_(pdata)

    def step_2c(self, closure=None, closure_linesearch=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            closure_linesearch (callable, optional): A closure that reevaluates the model
                and returns the loss in no_grad context
        """
        loss = None
        if closure is not None:
            loss = closure()

        lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        momentum = self.param_groups[0]['momentum']
        dampening = self.param_groups[0]['dampening']
        nesterov = self.param_groups[0]['nesterov']
        line_search_fn= self.param_groups[0]['line_search_fn']
        
        flat_grad= self._gather_flat_grad()
        d_p= flat_grad.clone()
        p= self._gather_flat_params()
        if weight_decay != 0:
            d_p.add_(weight_decay, p.data)
        if momentum != 0:
            if self._momentum_buffer:
                buf = self._momentum_buffer = torch.clone(d_p).detach()
            else:
                buf = self._momentum_buffer
                buf.mul_(momentum).add_(1 - dampening, d_p)
            if nesterov:
                d_p = d_p.add(momentum, buf)
            else:
                d_p = buf

        # optional line search: user function
        f_loss= float(loss)
        ls_func_evals = 0
        if line_search_fn is not None and line_search_fn is not "default":
            # perform line search, using user function
            if line_search_fn == "backtracking":
                d_p.mul_(-1)
                x_init = self._clone_param()
                gtd = flat_grad.dot(d_p)
                default_t= 1. / flat_grad.abs().sum() * lr

                def obj_func(t, x, d):
                    return self._directional_evaluate_derivative_free(closure_linesearch, t, x, d)

                # return (xmin, fval, iter, funcalls)
                t, f_loss= _scalar_search_armijo(obj_func, f_loss, gtd, args=(x_init,d_p), alpha0=default_t)
                if t is None:
                    raise RuntimeError("minimize_scalar failed")
                log.info(f"LS final step: {t}")
            else:
                raise RuntimeError("unsupported line search")
            
            self._add_grad(t, d_p)
            
        else:
            # no line search, simply move with fixed-step
            self._add_grad(-lr, d_p)

        return loss