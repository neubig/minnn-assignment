# Structure of Minnn

Here is a walk-through of the main classes in `minnn.py`.

## xp

`xp` is an alias for the numerical processing library that we're using to make computation efficient. By default we use [numpy](https://numpy.org/), a widely used numerical library that you may know of already. For brief tutorials, you can check the links provided at the end of this page. Alternatively, you can use `cupy`, an interface that is basically identical to `numpy`, but allows computation on the GPU, which can be useful for speed purposes.

- The choice of computation library can be specified by the environment variable `WHICH_XP`.
- For this assignment using `numpy` with CPU will already be fast enough (around 6s per iter in our running). In our final testing, we would probably use `numpy`. (Nevertheless please feel free to use `cupy` if you find it is much faster.)
- In the `to-be-implemented` parts, please simply use `xp` to denote `numpy` or `cupy`.

## Tensor

The `Tensor` class is a Tensor data structure, with the underlying data stored in a multidimensional array.

- This class is very similar to `torch.Tensor`.
- `Tensor.data` is the field that contains the main data for this tensor, this field is a `xp.ndarray`. The updates of the parameters should be directly changing this data.
- `Tensor.grad` is the field for storing the gradient for this tensor. There can be three types of values for this field:
- -> `None`: which denotes zero gradient.
- -> `xp.ndarray`: which should be the same size as the `Tensor.data`, denoting dense gradients.
- -> `Dict[int, xp.ndarray]`: which is a simple simulation of sparse gradients for 2D matrices (embeddings). The key `int` denotes the index into the first dimension, while the value is a `xp.ndarray` which shape is `Tensor.data.shape[1]`, denoting the gradient for the column slice according to the index.
- `Tensor.op`, which is an `Op` (see below) that generates this `Tensor`, if `None` then mostly not calculated but inputted.
- `Parameter` is a simple sub-class of `Tensor`, denoting persistent model parameters.

## Op

This class implements an operation that is part of a `ComputationGraph`.

- `Op.forward` and `Op.backward`: these are the forward and backward methods calculating the operation itself and its gradient.
- `Op.ctx`: this field is a data field that is populated during the `forward` operation to store all the relevant values (input, output, intermediate) that must be used in `backward` to calculate gradients. We provide two helper methods `store_ctx` and `get_ctx` to do these, but please feel free to store things in your own way, we will not check `Op.ctx`.
- `Op.full_forward`: This is a simple wrapper for the actual `forward`, adding only one thing to make it convenient, recording the `Tensor.op` for the outputted `Tensor` so that you do not need to add this in `forward`.

## ComputationGraph

This class is the one that keeps track of the current computational graph.

- It simply contains a list of `Op`s, which are registered in `Op.__init__`.
- In forward, these `Op` are appended incrementally in calculation order, and in backward (see function `backward`, they are visited in reversed order).

## Initializer

This is simply a collection of initializer methods that produces a `xp.ndarray` according to the specified shape and other parameters like initializer ranges.

## Model & Trainer

- `Model` maintains a collection of `Parameter`. We provide `add_parameters` as a shortcut of making a new `Parameter` and adding it to the model.
- `Trainer` takes a `Model` and handles the update of the parameters. `Trainer.update` denotes one update step which will be implemented in the sub-classes.
- `SGDTrainer` is a simple SGD trainer, notice that here we check whether `Tensor.grad` is sparse (simulated by a python dictionary) or not, and update accordingly. (In our enviroment with CPU, enabling sparse update is much faster, but not necessarily with GPU).
- Notice that at the end of each `update`, we also clear the gradients (clearing by simply setting `Tensor.grad=None`). This can usually be two separate steps, but we combine them here for convenience.

## Graph computation algorithms

- `reset_computation_graph` discards the previous `ComputationGraph` (together with previous `Op`s and intermediate `Tensor`s) and make a new one. This should be called at the start of each computation loop.
- `forward` gets the `np.ndarray` value of a `Tensor`. Since we calculate everything greedily, this step is simply retriving the `Tensor.data`.
- `backward` assign a scalar gradient `alpha` to a tensor and do backwards according to the reversed order of the `Op` list stored inside `ComputationGraph`.


## Backpropable functions

- The remaining `Op*` are all sub-classes of `Op` and denotes a specific function. We provide some operations and ask you to implement some of them.
- Take `OpDropout` as an example, here we implement the inverted dropout, which scales values by `1/(1-drop)` in forward. In `forward`, (if training), we obtain a `mask` using `xp.random` and multiply the input by this. All the intermediate values (including input and output) are stored using `store_ctx`. In `backward`, we obtain the graident of the output `Tensor` by retriving previous stored values. Then the calcualted gradients are assigned to the input `Tensor` by accumulate_grad.
- Finally, there are some shortcut functions to make it more convenient.


# To Be Implemented

Notably, `minnn.py` is not completely implemented, and there are some parts that you will need to finish. For all of the parts below, there are tests in `test_minnn.py`, which will allow you to test if each individual part is working properly.

## `Tensor.accumulate_grad` & `Tensor.accumulate_grad_sparse`

- `accumulate_grad` accepts one (dense) `xp.ndarray` and accumulate to the `Tensor`'s dense gradients (`xp.ndarray`).
- `accumulate_grad_sparse` accepts a list of (index, `xp.ndarray`) and accumulates them to the `Tensor`'s simulated sparase gradients (`dict`).
- We will check the gradients before and after these methods. Notice that we reuse the `Tensor.grad` for both dense and (simulated) sparse gradients, thus please do not apply both at the same time. (See also `get_dense_grad` of how to convert from simulated sparse gradients to dense ones.)

## `Initializer.xavier_uniform`

- This accepts inputs of `shape` and `gain`, and outputs a `xp.ndarray` where the shape is `shape`. (`gain` simply means that finally we are scaling the weights by this value.)
- See `Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)` for details about Xavier/Glorot initialization, and this blog for [more details about initialization in general](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79).

## `MomentumTrainer`

We provide an implementation of `SGDTrainer` and please similarly implement one for `MomentumTrainer`, which is SGD with momentum.

- Notice that in this one, there can be some variations. You can implement according to this formula: `m <- mrate*m + (1-mrate)*g, p <- p - lrate * m`, but if you find something better feel free to use that as well.
- Notice that for `update_sparse`, we still need to update the parameters if there are historical `m`, even if there are no gradients for the current step.
- Please remember to clear gradients (by setting `p.grad=None`) at the end of `update`, similar to `SGDTrainer`.

## `OpLookup` & `OpDot` & `OpTanh` & `OpMax`

- Please implement the `forward` and `backward` methods for these `Op`s.
- `OpLookup` represents a "lookup" operation, and accepts a `Tensor` matrix `W_emb` ([N,D]) and a list of word indexes ([n]), it returns another `Tensor` matrix ([n,D]).
- `OpDot` represents a matrix multiplication, accepting a `Tensor` matrix `W` ([M,N]) and a `Tensor` vector ([N]), it returns another `Tensor` vector ([M]).
- `OpTanh` calculates a tanh, and accepts any `Tensor` and returns another one with the same shape.
- `OpMax` calculates the maximum values of a tensor along certain dimension, it accepts a `Tensor` (value) and an `int` (axis), and it returns another `Tensor`.


# Other Notes

- The only external library allowed is `numpy`/`cupy`. No other libs can be utilized, for example, `pytorch` or other tools.
- In `classifier.py`, we also provide an option of `do_gradient_check` to do gradient checking with finite differences, which can be utilized for debugging.
- Please do not change another other existing parts of `minnn.py` (other than the `to-be-implemented` ones) and the method signatures (name and argument names). But surely feel free to add any helper functions as long as they do not conflict with existing ones.
- One thing to notice is the difference between `Tensor` and `xp.ndarray`. The general rule of thumb is that the returning value of `Op*`'s `forward` should be a `Tensor`. Nevertheless, in the `Op.ctx`, we can store both `Tensor` and `xp.ndarray`. In addition, please check the type hint of the arguments and other provided `Op*` for reference.
- Please notice that we hide the true labels of cfimdb-test (putting all 0s) in the assignment release.

# Resources
- Numpy tutorials
  * [Official Numpy tutorial page](https://numpy.org/learn/)
  * [The Ultimate Beginner’s Guide to NumPy](https://towardsdatascience.com/the-ultimate-beginners-guide-to-numpy-f5a2f99aef54)
  * [NumPy Introduction](https://www.w3schools.com/python/numpy_intro.asp)
  * [An introduction to Numpy and Scipy](https://sites.engineering.ucsb.edu/~shell/che210d/numpy.pdf)
