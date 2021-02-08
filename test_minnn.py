#

# a script to test the implementation of minnn.py
# -- usage is simply: python test_minnn.py <aid> <ref_minnn.py> <impl_dir>

import os
import numpy as np

np.random.seed(123)

# --
# helpers

def search_file(dir: str, filename: str):
    rets = []
    for dirpath, _, filenames in os.walk(dir):
        if filename in filenames:
            rets.append(os.path.join(dirpath, filename))
    assert len(rets) == 1
    return rets[0]

def load_module(f: str, m: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(m, f)
    ret = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ret)
    return ret

# --
# testings!
def test_accumulate_grad(test_mn, ref_mn):
    shape = [10, 20]
    data = np.random.random(shape)
    t1 = test_mn.Tensor(data)
    t2 = ref_mn.Tensor(data.copy())
    for _ in range(5):
        g = np.random.random(shape)
        gcopy = g.copy()
        t1.accumulate_grad(g)
        t2.accumulate_grad(gcopy)
    assert np.allclose(t1.get_dense_grad(), t2.get_dense_grad())
    # --

def test_accumulate_grad_sparse(test_mn, ref_mn):
    shape = [10, 20]
    data = np.random.random(shape)
    t1 = test_mn.Tensor(data)
    t2 = ref_mn.Tensor(data.copy())
    for _ in range(10):
        gs = [(np.random.randint(0, shape[0]), np.random.random(shape[1])) for z in range(5)]
        gs_copy = [(a, b.copy()) for a,b in gs]
        t1.accumulate_grad_sparse(gs)
        t2.accumulate_grad_sparse(gs_copy)
    assert np.allclose(t1.get_dense_grad(), t2.get_dense_grad())
    # --

def test_xavier_uniform(test_mn, ref_mn):
    shape = [100, 200]
    import torch
    x = torch.rand(shape)
    torch.nn.init.xavier_uniform_(x)
    y = test_mn.Initializer.xavier_uniform(shape)
    assert abs(x.mean().item()-y.mean().item()) <= 0.1
    assert abs(x.std().item()-y.std().item()) <= 0.1
    # --

# note: test all three functions at the same time!
def test_momentum_update(test_mn, ref_mn):
    shape = [10, 20]
    m1 = test_mn.Model()
    p1 = m1.add_parameters(shape)
    m2 = ref_mn.Model()
    p2 = m2.add_parameters(shape)
    p2.data[:] = p1.data  # copy
    # --
    t1 = test_mn.MomentumTrainer(m1)
    t2 = ref_mn.MomentumTrainer(m2)
    for _ in range(5):
        g = np.random.random(shape)
        gcopy = g.copy()
        p1.accumulate_grad(g)
        p2.accumulate_grad(gcopy)
        t1.update()
        t2.update()
        p1.grad = p2.grad = None  # clear grad in case this may be forgotten
    for _ in range(5):
        gs = [(np.random.randint(0, shape[0]), np.random.random(shape[1])) for z in range(5)]
        gs_copy = [(a, b.copy()) for a, b in gs]
        p1.accumulate_grad_sparse(gs)
        p2.accumulate_grad_sparse(gs_copy)
        t1.update()
        t2.update()
        p1.grad = p2.grad = None  # clear grad in case this may be forgotten
    assert np.allclose(p1.data, p2.data)
    # --

def test_lookup(test_mn, ref_mn):
    shape = [10, 20]
    data = np.random.random(shape)
    # --
    import torch
    t1 = test_mn.Tensor(data)
    t2 = torch.tensor(data, requires_grad=True)
    words = [np.random.randint(0, shape[0]) for z in range(5)]
    # forward
    v1 = test_mn.lookup(t1, words)
    v2 = t2[words]
    assert np.allclose(v1.data, v2.detach().numpy())
    # backward
    g = np.random.random(v1.shape)
    gcopy = g.copy()
    v1.accumulate_grad(g)
    v1.op.backward()
    (v2*torch.tensor(gcopy)).sum().backward()
    assert np.allclose(t1.get_dense_grad(), t2.grad.numpy())

def test_dot(test_mn, ref_mn):
    shape = [10, 20]
    data = np.random.random(shape)
    data2 = np.random.random(shape[1])
    # --
    import torch
    w1, h1 = test_mn.Tensor(data), test_mn.Tensor(data2)
    w2, h2 = torch.tensor(data, requires_grad=True), torch.tensor(data2, requires_grad=True)
    # forward
    v1 = test_mn.dot(w1, h1)
    v2 = torch.matmul(w2, h2)
    assert np.allclose(v1.data, v2.detach().numpy())
    # backward
    g = np.random.random(v1.shape)
    gcopy = g.copy()
    v1.accumulate_grad(g)
    v1.op.backward()
    (v2*torch.tensor(gcopy)).sum().backward()
    assert np.allclose(w1.get_dense_grad(), w2.grad.numpy())
    assert np.allclose(h1.get_dense_grad(), h2.grad.numpy())

def test_tanh(test_mn, ref_mn):
    shape = [20]
    data = np.random.random(shape)
    # --
    import torch
    t1 = test_mn.Tensor(data)
    t2 = torch.tensor(data, requires_grad=True)
    # forward
    v1 = test_mn.tanh(t1)
    v2 = torch.tanh(t2)
    assert np.allclose(v1.data, v2.detach().numpy())
    # backward
    g = np.random.random(v1.shape)
    gcopy = g.copy()
    v1.accumulate_grad(g)
    v1.op.backward()
    (v2*torch.tensor(gcopy)).sum().backward()
    assert np.allclose(t1.get_dense_grad(), t2.grad.numpy())
    # --

# --
def main(test_id: str, ref_minnn: str, test_dir: str):
    test_minnn = search_file(test_dir, "minnn.py")
    # --
    ref_mn = load_module(ref_minnn, 'mn1')
    test_mn = load_module(test_minnn, 'mn2')
    # --
    # note: to change weights in this table
    test_table = [
        ("accumulate_grad", 1.,),
        ("accumulate_grad_sparse", 1.,),
        ("xavier_uniform", 1.,),
        ("momentum_update", 1.),
        ("lookup", 1.),
        ("dot", 1.),
        ("tanh", 1.),
    ]
    scores = {}
    for name, weight in test_table:
        test_f = globals()[f"test_{name}"]
        try:
        # if 1:
            test_f(test_mn, ref_mn)
            score = weight
        except:
            score = 0.
        scores[name] = score
    ret = {"id": test_id, "score": sum(scores.values()), "scores": scores}
    print(ret)
    return ret

# --
# extra one to test acc
def eval_acc(f1: str, f2: str):
    with open(f1) as fd:
        preds1 = [line.split("|||")[0].strip() for line in fd if line.strip()!=""]
    with open(f2) as fd:
        preds2 = [line.split("|||")[0].strip() for line in fd if line.strip()!=""]
    assert len(preds1) == len(preds2)
    good = sum(a==b for a,b in zip(preds1, preds2))
    print(f"Acc = {good}/{len(preds1)}={good/len(preds1)}")
# --

if __name__ == '__main__':
    # main("test", "minnn.py", ".")
    import sys
    main(*sys.argv[1:])
