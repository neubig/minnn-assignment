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

def is_allclose(x, y):
    return np.allclose(x, y, rtol=1.e-3, atol=1.e-5)
    # return np.abs(x-y).sum().item() <= 1e-5

# --
# testings!
def test_accumulate_grad(test_mn):
    g0 = np.asarray([1., 2.])
    g1 = np.asarray([3., 4.])
    t = test_mn.astensor([0., 0.])
    t.accumulate_grad(g0)
    t.accumulate_grad(g1)
    assert is_allclose(t.get_dense_grad(), np.asarray([4., 6.]))
    # --

def test_accumulate_grad_sparse(test_mn):
    g0 = np.asarray([1., 2.])
    g1 = np.asarray([3., 4.])
    t = test_mn.astensor([[0., 0.], [0., 0.], [0., 0.]])
    t.accumulate_grad_sparse([(0, g0), (2, g1), (2, g0)])
    assert is_allclose(t.get_dense_grad(), np.asarray([g0, [0.,0.], g0+g1]))
    # --

def test_xavier_uniform(test_mn):
    shape = [100, 500]
    y = test_mn.Initializer.xavier_uniform(shape)
    assert list(y.shape) == shape
    assert abs(y.mean()) <= 1e-2
    assert abs(y.max()-0.1) <= 1e-2
    assert abs(y.min()+0.1) <= 1e-2
    # --

def test_momentum_update(test_mn):
    m1 = test_mn.Model()
    p1 = m1.add_parameters([2], 'constant', val=0.)
    p2 = m1.add_parameters([2,2], 'constant', val=0.)
    # --
    t1 = test_mn.MomentumTrainer(m1, lrate=0.1, mrate=0.9)
    for _ in range(5):
        g = np.array([1.,1.])
        p1.accumulate_grad(g)
        t1.update()
    assert is_allclose(p1.data, np.array([-0.131441, -0.131441])), \
        "This is the values using our implementation, there can be other versions for momentum_update, you can choose to use others!"
    for i in range(6):
        g = np.array([1.,1.])
        p2.accumulate_grad_sparse([((i%2), g.copy())])
        t1.update()
    assert is_allclose(p2.data, np.array([[-0.1002459, -0.1002459], [-0.078051, -0.078051]])), \
        "This is the values using our implementation, there can be other versions for momentum_update, you can choose to use others!"
    # --

def test_lookup(test_mn):
    t = test_mn.astensor([[1., 2.], [3., 4.], [5., 6.]])
    v = test_mn.lookup(t, [1,0,1])
    assert is_allclose(v.data, np.asarray([[3., 4.], [1., 2.], [3., 4.]]))
    v.accumulate_grad(np.asarray([[1., 1.], [1., 1.], [2., 3.]]))
    v.op.backward()
    assert is_allclose(t.get_dense_grad(), np.asarray([[1., 1.], [3., 4.], [0., 0.]]))

def test_dot(test_mn):
    w, h = test_mn.astensor([[0., 1.], [2., 3.]]), test_mn.astensor([1., 2.])
    v = test_mn.dot(w, h)
    assert is_allclose(v.data, np.asarray([2., 8.]))
    v.accumulate_grad(np.asarray([1., 3.]))
    v.op.backward()
    assert is_allclose(w.get_dense_grad(), np.asarray([[1.,2.],[3.,6.]]))
    assert is_allclose(h.get_dense_grad(), np.asarray([6.,10.]))

def test_tanh(test_mn):
    x = test_mn.astensor([0., 1., 2., 3.])
    v = test_mn.tanh(x)
    assert is_allclose(v.data, np.asarray([0., 0.76159416, 0.96402758, 0.99505475]))
    v.accumulate_grad(np.asarray([1.,2.,3.,4.]))
    v.op.backward()
    assert is_allclose(x.get_dense_grad(), np.asarray([1., 0.83994868, 0.21195247, 0.03946415]))
    # --

def test_avg(test_mn):
    x = test_mn.astensor([[0., 1., 2.], [3., 5., 11.]])
    v = test_mn.avg(x, 0)
    assert is_allclose(v.data, np.asarray([1.5, 3., 6.5]))
    v.accumulate_grad(np.asarray([1.,2.,3.]))
    v.op.backward()
    assert is_allclose(x.get_dense_grad(), np.asarray([[0.5,1.0,1.5], [0.5,1.0,1.5]]))
    # --

def test_max(test_mn):
    x = test_mn.astensor([[0., 5., 2.], [-2., 6., 1.]])
    v = test_mn.max(x, 0)
    assert is_allclose(v.data, np.asarray([0., 6., 2.]))
    v.accumulate_grad(np.asarray([1.,2.,3.]))
    v.op.backward()
    assert is_allclose(x.get_dense_grad(), np.asarray([[1., 0., 3.], [0., 2., 0.]]))
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
        ("avg", 1.),
        ("max", 1.),
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
