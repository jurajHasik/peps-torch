import torch
from torch.autograd import Function

class CheckpointCompat(Function):
    @staticmethod
    def forward(run_function, preserve_rng_state, *args):
        # Run forward **without** tracking intermediates
        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # This is the key hook functorch checks for! 
        # Save the inputs for backward when using vjp/jvp, etc.
        
        # Save the callable and RNG‐state flag
        ctx.run_function = inputs[0]
        ctx.preserve_rng_state = inputs[1]

        # Optionally stash RNG state so that recompute is deterministic
        if ctx.preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            if len(inputs[2:])>0 and inputs[2].device.type == 'cuda':
                ctx.fwd_cuda_state = torch.cuda.get_rng_state(inputs[2].device)

        ctx.save_for_backward(*inputs[2:])

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Restore RNG if requested
        if ctx.preserve_rng_state:
            torch.set_rng_state(ctx.fwd_cpu_state)
            if hasattr(ctx, 'fwd_cuda_state'):
                torch.cuda.set_rng_state(ctx.fwd_cuda_state, ctx.saved_tensors[0].device)

        # Re-compute forward with grad enabled
        inputs = ctx.saved_tensors
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Run backward on the re-computed graph
        torch.autograd.backward(outputs, grad_outputs)

        # Build gradient tuple: None for the first two non-Tensor inputs
        grads = [None, None]
        grads += [inp.grad if isinstance(inp, torch.Tensor) else None
                  for inp in inputs]
        return tuple(grads)

def checkpoint_compat(fn, *args, preserve_rng_state=True):
    """
    fn: a Python function mapping Tensors -> Tensor or tuple of Tensors
    args: Tensor inputs
    """
    return CheckpointCompat.apply(fn, preserve_rng_state, *args)


# A toy module
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 10)

    def forward(self, x):
        # checkpoint the linear + ReLU
        def run(x_inner):
            return torch.relu(self.lin(x_inner))
        return checkpoint_compat(run, x)

model = MyModule()
x = torch.randn(5, 10, requires_grad=True)

# compute y and a VJP function
y, vjp_fn = torch.func.vjp(lambda z: model(z), x)          # functorch.vjp :contentReference[oaicite:2]{index=2}
cotangent = torch.randn_like(y)

# get dx = Jᵀ · cotangent
(dx,) = vjp_fn(cotangent)
print(dx.shape)  # torch.Size([5, 10])