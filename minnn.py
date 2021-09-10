#

# a simple and naive implementation for minnn

from typing import List, Tuple, Sequence, Union, Any, Dict
import math
import os
import numpy as np

# --
# which *py to use??
_WHICH_XP = os.environ.get("WHICH_XP", "np")
if _WHICH_XP.lower() in ["cupy", "cp"]:
    print("Use cupy!")
    import cupy as xp
    def asnumpy(x):
        return xp.asnumpy(x)
else:
    print("Use numpy!")
    import numpy as xp
    def asnumpy(x):
        return np.asarray(x)

# random seed
xp.random.seed(12345)

def set_random_seed(seed: int):  # allow reset!
    xp.random.seed(seed)
# --

# --
# components in computation graph

# Tensor
class Tensor:
    def __init__(self, data: xp.ndarray):
        self.data: xp.ndarray = data
        self.grad: Union[Dict[int, xp.ndarray], xp.ndarray] = None  # should be the same size as data
        self.op: Op = None  # generated from which operation?

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"T{self.shape}: {self.data}"

    # accumulate grad
    def accumulate_grad(self, g: xp.ndarray) -> None:
        raise NotImplementedError

    # accumulate grad sparsely; note: only for D2 lookup matrix!
    def accumulate_grad_sparse(self, gs: List[Tuple[int, xp.ndarray]]) -> None:
        raise NotImplementedError

    # get dense grad
    def get_dense_grad(self):
        ret = xp.zeros_like(self.data)
        if self.grad is not None:
            if isinstance(self.grad, dict):
                for widx, arr in self.grad.items():
                    ret[widx] += arr
            else:
                ret = self.grad
        return ret

    # add or sub
    def __add__(self, other: 'Tensor'):
        return OpAdd().full_forward(self, other)

    def __sub__(self, other: 'Tensor'):
        return OpAdd().full_forward(self, other, alpha_b=-1.)

    def __mul__(self, other: Union[int, float]):
        assert isinstance(other, (int, float)), "currently only support scalar __mul__"
        return OpAdd().full_forward(self, b=None, alpha_a=float(other))

# Parameter: special tensor
class Parameter(Tensor):
    def __init__(self, data: xp.ndarray):
        super().__init__(data)

    @classmethod
    def from_tensor(cls, tensor: Tensor):
        return Parameter(tensor.data)  # currently simply steal its data

# shortcut for create tensor
def astensor(t):
    return t if isinstance(t, Tensor) else Tensor(xp.asarray(t))

# Operation
class Op:
    def __init__(self):
        self.ctx: Dict[str, Union[Tensor, Any]] = {}  # store intermediate tensors or other values
        self.idx: int = None  # idx in the cg
        ComputationGraph.get_cg().reg_op(self)  # register into the graph

    # store intermediate results for usage in backward
    def store_ctx(self, ctx: Dict = None, **kwargs):
        if ctx is not None:
            self.ctx.update(ctx)
        self.ctx.update(kwargs)

    # get stored ctx values
    def get_ctx(self, *names: str):
        return [self.ctx.get(n) for n in names]

    # full forward, forwarding plus set output op
    def full_forward(self, *args, **kwargs):
        rets = self.forward(*args, **kwargs)
        # -- store op for outputs
        outputs = []
        if isinstance(rets, Tensor):
            outputs.append(rets)  # single return
        elif isinstance(rets, (list, tuple)):  # note: currently only support list or tuple!!
            outputs.extend([z for z in rets if isinstance(z, Tensor)])
        for t in outputs:
            assert t.op is None, "Error: should only have one op!!"
            t.op = self
        # --
        return rets

    # forward the operation
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    # backward with the pre-stored tensors
    def backward(self):
        raise NotImplementedError()

# computational graph
class ComputationGraph:
    # global cg
    _cg: 'ComputationGraph' = None

    @classmethod
    def get_cg(cls, reset=False):
        if ComputationGraph._cg is None or reset:
            ComputationGraph._cg = ComputationGraph()
        return ComputationGraph._cg

    def __init__(self):
        self.ops: List[Op] = []  # list of ops by execution order

    def reg_op(self, op: Op):
        assert op.idx is None
        op.idx = len(self.ops)
        self.ops.append(op)

# initializer
class Initializer:
    @staticmethod
    def uniform(shape: Sequence[int], a=0.0, b=0.2):
        return xp.random.uniform(a, b, size=shape)

    @staticmethod
    def normal(shape: Sequence[int], mean=0., std=0.02):
        return xp.random.normal(mean, std, size=shape)

    @staticmethod
    def constant(shape: Sequence[int], val=0.):
        return xp.full(shape, val)

    @staticmethod
    def xavier_uniform(shape: Sequence[int], gain=1.0):
        raise NotImplementedError

# Model: collection of parameters
class Model:
    def __init__(self):
        self.params: List[Parameter] = []

    def add_parameters(self, shape, initializer='normal', **initializer_kwargs):
        init_f = getattr(Initializer, initializer)
        data = init_f(shape, **initializer_kwargs)
        param = Parameter(data)
        self.params.append(param)
        return param

    def save(self, path: str):
        data = {f"p{i}": p.data for i,p in enumerate(self.params)}
        xp.savez(path, **data)

    def load(self, path: str):
        data0 = xp.load(path)
        data = {int(n[1:]):d for n,d in data0.items()}
        for i,p in enumerate(self.params):
            d = data[i]
            assert d.shape == p.shape
            p.data = d

# Trainer
class Trainer:
    def __init__(self, model: Model):
        self.model = model

    def clone_param_stats(self, model: Model):
        clone = list()
        for param in model.params:
            clone.append(np.zeros(param.data.shape))
        return clone

    def update(self):  # to be implemented
        raise NotImplementedError()

class SGDTrainer(Trainer):
    def __init__(self, model: Model, lrate=0.1):
        super().__init__(model)
        self.lrate = lrate

    def update(self):
        lrate = self.lrate
        for p in self.model.params:
            if p.grad is not None:
                if isinstance(p.grad, dict):  # sparsely update to save time!
                    self.update_sparse(p, p.grad, lrate)
                else:
                    self.update_dense(p, p.grad, lrate)
            # clean grad
            p.grad = None
        # --

    def update_dense(self, p: Parameter, g: xp.ndarray, lrate: float):
        p.data -= lrate * g

    def update_sparse(self, p: Parameter, gs: Dict[int, xp.ndarray], lrate: float):
        for widx, arr in gs.items():
            p.data[widx] -= lrate * arr


class MomentumTrainer(Trainer):
    def __init__(self, model: Model, lrate=0.1, mrate=0.99):
        raise NotImplementedError

# --

### Graph computation algorithms

def reset_computation_graph():
    return ComputationGraph.get_cg(reset=True)

def forward(t: Tensor):
    # since we calculate greedily, the result are already there!!
    return asnumpy(t.data)

def backward(t: Tensor, alpha=1.):
    # first put grad to the start one
    t.accumulate_grad(alpha)
    # locate the op
    op = t.op
    assert op is not None, "Cannot backward on tensor since no op!!"
    # backward the whole graph!!
    cg = ComputationGraph.get_cg()
    for idx in reversed(range(op.idx+1)):
        cg.ops[idx].backward()
    # --

### Helper
def log_softmax(x: xp.ndarray, axis=-1):
    c = xp.max(x, axis=axis, keepdims=True)  # [*, 1, *]
    x2 = x - c  # [*, ?, *]
    logsumexp = xp.log(xp.exp(x2).sum(axis=axis, keepdims=True))  # [*, 1, *]
    return x2 - logsumexp

### Backpropable functions

class OpLookup(Op):
    def __init__(self):
        raise NotImplementedError

class OpSum(Op):
    def __init__(self):
        super().__init__()

    # [..., K, ...] -> [..., ...]
    def forward(self, emb: Tensor, axis: int):
        reduce_size = emb.data.shape[axis]
        arr_sum = emb.data.sum(axis=axis)
        t_sum = Tensor(arr_sum)
        self.store_ctx(emb=emb, t_sum=t_sum, axis=axis, reduce_size=reduce_size)
        return t_sum

    def backward(self):
        emb, t_sum, axis, reduce_size = self.get_ctx('emb', 't_sum', 'axis', 'reduce_size')
        if t_sum.grad is not None:
            g0 = xp.expand_dims(t_sum.grad, axis)
            g = xp.repeat(g0, reduce_size, axis=axis)
            emb.accumulate_grad(g)
        # --

class OpMax(Op):
    def __init__(self):
        raise NotImplementedError

class OpAvg(Op):
    # NOTE: Implementation of OpAvg is optional, it can be skipped if you wish
    def __init__(self):
        super().__init__()

class OpDot(Op):
    def __init__(self):
        raise NotImplementedError

class OpTanh(Op):
    def __init__(self):
        raise NotImplementedError

class OpRelu(Op):
    def __init__(self):
        super().__init__()

    # [N] -> [N]
    def forward(self, t: Tensor):
        arr_relu = t.data  # [N]
        arr_relu[arr_relu < 0.0] = 0.0
        t_relu = Tensor(arr_relu)
        self.store_ctx(t=t, t_relu=t_relu, arr_relu=arr_relu)
        return t_relu

    def backward(self):
        t, t_relu, arr_relu = self.get_ctx('t', 't_relu', 'arr_relu')
        if t_relu.grad is not None:
            grad_t = xp.where(arr_relu > 0.0, 1.0, 0.0) * t_relu.grad  # [N]
            t.accumulate_grad(grad_t)
        # --

class OpLogloss(Op):
    def __init__(self):
        super().__init__()

    # [*, N], [*] -> [*]
    def forward(self, logits: Tensor, tags: Union[int, List[int]]):
        # negative log likelihood
        arr_tags = xp.asarray(tags)  # [*]
        arr_logprobs = log_softmax(logits.data)  # [*, N]
        if len(arr_logprobs.shape) == 1:
            arr_nll = - arr_logprobs[arr_tags]  # []
        else:
            assert len(arr_logprobs.shape) == 2
            arr_nll = - arr_logprobs[xp.arange(len(arr_logprobs.shape[0])), arr_tags]  # [*]
        loss_t = Tensor(arr_nll)
        self.store_ctx(logits=logits, loss_t=loss_t, arr_tags=arr_tags, arr_logprobs=arr_logprobs)
        return loss_t

    def backward(self):
        logits, loss_t, arr_tags, arr_logprobs = self.get_ctx('logits', 'loss_t', 'arr_tags', 'arr_logprobs')
        if loss_t.grad is not None:
            arr_probs = xp.exp(arr_logprobs)  # [*, N]
            grad_logits = arr_probs  # prob-1 for gold, prob for non-gold
            if len(grad_logits.shape) == 1:
                grad_logits[arr_tags] -= 1.
                grad_logits *= loss_t.grad
            else:
                grad_logits[xp.arange(len(grad_logits.shape[0])), arr_tags] -= 1.
                grad_logits *= loss_t.grad[:,None]
            logits.accumulate_grad(grad_logits)
        # --

class OpAdd(Op):
    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor, alpha_a=1., alpha_b=1.):
        if b is None:
            arr_add = alpha_a * a.data
        else:
            arr_add = alpha_a * a.data + alpha_b * b.data
        t_add = Tensor(arr_add)
        self.store_ctx(a=a, b=b, t_add=t_add, alpha_a=alpha_a, alpha_b=alpha_b)
        return t_add

    def backward(self):
        a, b, t_add, alpha_a, alpha_b = self.get_ctx('a', 'b', 't_add', 'alpha_a', 'alpha_b')
        if t_add.grad is not None:
            a.accumulate_grad(alpha_a * t_add.grad)
            if b is not None:
                b.accumulate_grad(alpha_b * t_add.grad)
        # --

class OpDropout(Op):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, drop: float, is_training: bool):
        if is_training:
            arr_mask = xp.random.binomial(1, 1.-drop, x.shape) * (1./(1-drop))
            arr_drop = (x.data * arr_mask)
            t_drop = Tensor(arr_drop)
        else:
            arr_mask = 1.
            t_drop = Tensor(x.data)  # note: here copy things to make it consistent!
        self.store_ctx(is_training=is_training, x=x, arr_mask=arr_mask, t_drop=t_drop)
        return t_drop

    def backward(self):
        is_training, x, arr_mask, t_drop = self.get_ctx('is_training', 'x', 'arr_mask', 't_drop')
        if not is_training:
            pass
            # print("Warn: Should not backward if not in training??")
        if t_drop.grad is not None:
            x.accumulate_grad(arr_mask * t_drop.grad)
        # --

# --
# shortcuts
def lookup(W_emb, words): return OpLookup().full_forward(W_emb, words)
def sum(emb, axis): return OpSum().full_forward(emb, axis)
def dot(W_h_i, h): return OpDot().full_forward(W_h_i, h)
def tanh(param): return OpTanh().full_forward(param)
def relu(param): return OpRelu().full_forward(param)
def log_loss(my_scores, tag): return OpLogloss().full_forward(my_scores, tag)
def dropout(x, drop, is_training): return OpDropout().full_forward(x, drop, is_training)
def avg(x, axis): return OpAvg().full_forward(x, axis)
def max(x, axis): return OpMax().full_forward(x, axis)
# --
