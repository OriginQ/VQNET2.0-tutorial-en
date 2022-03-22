Classical Neural Network Module
==================================

The following classical neural network modules support automatic back propagation computation.After running the forward function, you can calculate the gradient by executing the reverse function.A simple example of the convolution layer is as follows:

.. code-block::

    # an image feed into two dimension convolution layer
    b = 2        # batch size 
    ic = 3       # input channels
    oc = 2      # output channels
    hw = 6      # input width and heights

    # two dimension convolution layer
    test_conv = Conv2D(ic,oc,(3,3),(2,2),"same",initializer.ones,initializer.ones)

    # input of shape [b,ic,hw,hw]
    x0 = QTensor(CoreTensor.range(1,b*ic*hw*hw).reshape([b,ic,hw,hw]),requires_grad=True)

    #forward function
    x = test_conv(x0)

    #backward function with autograd
    x.backward()

    print("##W###")
    print(test_conv.weights.grad)
    print("##B###")
    print(test_conv.bias.grad)
    print("##X###")
    print(x0.grad)
    print("##Y###")
    print(x)

.. currentmodule:: pyvqnet.nn


Module Class
-------------------------------

abstract calculation module


Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.module.Module

forward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyvqnet.nn.module.Module.forward

state_dict 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyvqnet.nn.module.Module.state_dict

save_parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.utils.storage.save_parameters

load_parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.utils.storage.load_parameters



Classical Neural Network Layer
-------------------------------

Conv1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.conv.Conv1D

Conv2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.conv.Conv2D

ConvT2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.conv.ConvT2D


AvgPool1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.AvgPool1D

MaxPool1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.MaxPool1D

AvgPool2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.AvgPool2D

MaxPool2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.MaxPool2D

Embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.embedding.Embedding

BatchNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.batch_norm.BatchNorm2d

BatchNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.batch_norm.BatchNorm1d

LayerNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.layer_norm.LayerNorm2d

LayerNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.layer_norm.LayerNorm1d

Linear
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.linear.Linear

Dropout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.dropout.Dropout


Loss Function Layer
----------------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.MeanSquaredError

BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.BinaryCrossEntropy

CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.CategoricalCrossEntropy

SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.SoftmaxCrossEntropy


Activation Function
----------------------------------


Activation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Activation


Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Sigmoid


Softplus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Softplus


Softsign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Softsign


Softmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Softmax


HardSigmoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.HardSigmoid


ReLu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.ReLu


LeakyReLu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.LeakyReLu


ELU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.ELU


Tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Tanh


Optimizer Module
----------------------------------


Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.optimizer.Optimizer

adadelta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adadelta.Adadelta

adagrad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adagrad.Adagrad

adam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adam.Adam

adamax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adamax.Adamax

rmsprop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.rmsprop.RMSProp

sgd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.sgd.SGD

rotosolve
^^^^^^^^^^^^^^

Rotosolve algorithm, which allows a direct jump to the optimal value of a single parameter relative to the fixed value of other parameters, can directly find the optimal parameters of the quantum circuit optimization algorithm.

.. autoclass:: pyvqnet.optim.rotosolve.Rotosolve

.. figure:: ./images/rotosolve.png


