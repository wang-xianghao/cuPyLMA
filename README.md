# cuPyLMA: a Multi-GPU Levenberg-Marquardt (Deep Learning) Optimizer Powered by NVIDIA cuPyNumeric

[**Background**](#background)
| [**Installation**](#installation)
| [**Training**](#training)
| [**Examples**](#examples)
| [**Performance**](#performance)
| [**Change logs**](CHANGELOG.md)

cuPyLMA is a scalable multi-GPU (deep learning optimizer) optimizer which implements the [Levenberg-Marquardt algorithm (LMA)](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm). This library is built on PyTorch and [NVIDIA cuPyNumeric](https://github.com/nv-legate/cupynumeric) (a NumPy-like scientific computing framework).

## Background
The [Levenberg-Marquardt algorithm (LMA)](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) is a second-order optimization algorithm that utilizes the Jacobian matrix of the residuals to compute optimal parameter updates. In contrast, the widely used first-order optimizer Adam relies on the gradient of the loss function to determine these updates.

$$
\large (\mathbf{J}^T\mathbf{J}+\lambda \mathbf{I})\triangle\mathbf{x} = \mathbf{J}^T\mathbf{r}
$$

($\mathbf{J}$: Jacobian matrix of residuals, $\mathbf{r}$: residuals, $\triangle\mathbf{x}$: updates to be solved)


The LMA has the following advantages and disadvantages compared to the Adam:
* Pros
    - Faster convergence.
    - More optimal solutions due to using the second-order information.
* Cons
    - Higher memory and computation requirement due to computing the Jacobian matrix and solving the equation, especially when the model has many parameters.

Our cuPyLMA aims to resolve the memory and computation bottlenecks of the LMA via utilizing multiple GPUs.

## Installation
To install cuPyLMA along with dependencies, please run:
```bash
pip install cupylma
```

## Training

It is easy to migrate the training code that uses the Adam optimizer to cuPyLMA. cuPyLMA consists of the following components and each holds a seperate set of GPUs.
* **Model component** stores the model parameters and computes the Jacobian matrix.
* **Optimizer component** stores the Jacobian matrix and computes the optimal parameter updates.

### Creating the model
The model should be in one of GPUs held by the model component. The `get_available_gpus()` function gets the list of available GPUs for the model component.
```python
from cupylma import get_available_gpus

devices = get_available_gpus()
model = MyModel().to(devices[0])
```

### Configuring the optimizer
The LMA optimizer requires a residual function rather than a loss function. The `devices` option specifies the GPUs for the model component.

```python
from cupylma import LMA

residual_fn = lambda a, b : a - b # For simple regression
lma = LMA(model, devices, residual_fn)
```

To find the residual function for more complex problems, please check [examples/mnist](examples/mnist/).

### Training
The LMA optimizer is stateless, so there is no need to reset gradients at each step. The `loss` return value is the average loss. The `terminated` return value indicates whether the train should be terminated.

```python
loss, terminated = lma.step(x, y)
if terminated:
    # Exit the train and save the model
```

### Running the code
The `legate` command was installed together with cuPyLMA. The number of GPUs for the optimizer component is specified using the `--gpus` option.

```bash
legate --gpus 3 train.py
```

## Examples
* For curve fitting example, see [examples/curve](examples/curve/).
* For MNIST image classification example, see [examples/mnist](examples/mnist/).

## Performance
TODO

## References
[1] [fabiodimarco/torch-levenberg-marquardt](https://github.com/fabiodimarco/torch-levenberg-marquardt): Our base code refers to the repository.

[2] H. P. Gavin, “The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems,” 2024.: It provides theoretical explanation of LMA.


## Citation
J. Taylor, W. Wang, B. Bala, and T. Bednarz, “Optimizing the optimizer for data driven deep neural networks and physics informed neural networks,” May 16, 2022, arXiv: arXiv:2205.07430. doi: 10.48550/arXiv.2205.07430.
