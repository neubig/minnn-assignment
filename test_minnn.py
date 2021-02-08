#

# a script to test the minnn.py
# -- only with simple input/output examples
# -- usage is simply: python test_minnn.py <your-minnn.py>

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
def test_accumulate_grad(test_mn):
    g0 = np.asarray([1., 2.])
    g1 = np.asarray([3., 4.])
    t = test_mn.astensor([0., 0.])
    t.accumulate_grad(g0)
    t.accumulate_grad(g1)
    assert np.allclose(t.get_dense_grad(), np.asarray([4., 6.]))
    # --

def test_accumulate_grad_sparse(test_mn):
    g0 = np.asarray([1., 2.])
    g1 = np.asarray([3., 4.])
    t = test_mn.astensor([[0., 0.], [0., 0.], [0., 0.]])
    t.accumulate_grad_sparse([(0, g0), (2, g1)])
    assert np.allclose(t.get_dense_grad(), np.asarray([g0, [0.,0.], g1]))
    # --

def test_xavier_uniform(test_mn):
    shape = [100, 200]
    y = test_mn.Initializer.xavier_uniform(shape)
    assert list(y.shape) == shape
    # --

def test_momentum_update(test_mn):
    shape = [10, 20]
    m1 = test_mn.Model()
    p1 = m1.add_parameters(shape)
    # --
    t1 = test_mn.MomentumTrainer(m1)
    for _ in range(5):
        g = np.random.random(shape)
        p1.accumulate_grad(g)
        t1.update()
    for _ in range(5):
        gs = [(np.random.randint(0, shape[0]), np.random.random(shape[1])) for z in range(5)]
        p1.accumulate_grad_sparse(gs)
        t1.update()
    # --

def test_lookup(test_mn):
    t = test_mn.astensor([[1., 2.], [3., 4.], [5., 6.]])
    v = test_mn.lookup(t, [1,0])
    assert np.allclose(v.data, np.asarray([[3., 4.], [1., 2.]]))
    v.accumulate_grad(np.asarray([[1., 1.], [1., 1.]]))
    v.op.backward()
    assert np.allclose(t.get_dense_grad(), np.asarray([[1., 1.], [1., 1.], [0., 0.]]))

def test_dot(test_mn):
    w, h = test_mn.astensor([[0., 1.], [2., 3.]]), test_mn.astensor([1., 2.])
    v = test_mn.dot(w, h)
    assert np.allclose(v.data, np.asarray([2., 8.]))
    v.accumulate_grad(np.asarray([1., 1.]))
    v.op.backward()
    assert np.allclose(w.get_dense_grad(), np.asarray([[1.,2.],[1.,2.]]))

def test_tanh(test_mn):
    x = test_mn.astensor([0., 1., 2., 3.])
    v = test_mn.tanh(x)
    assert np.allclose(v.data, np.asarray([0., 0.76159416, 0.96402758, 0.99505475]))
    v.accumulate_grad(np.asarray([1.,1.,1.,1.]))
    v.op.backward()
    assert np.allclose(x.get_dense_grad(), np.asarray([1., 0.41997434, 0.07065082, 0.00986604]))
    # --

# --
def main(minnn: str):
    test_mn = load_module(minnn, 'mn')
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
    for name, weight in test_table:
        test_f = globals()[f"test_{name}"]
        test_f(test_mn)
        print(f"Pass {name}!")
    return

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
    # main("minnn.py")
    import sys
    main(*sys.argv[1:])
