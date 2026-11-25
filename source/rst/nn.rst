Classical Neural Network Modules
################################

Basic Examples
*********************************************************

Model Training on CPU
=================================

The following classical neural network modules all support automatic backpropagation. After you run the forward function, call the backward function to compute gradients. A simple convolution layer example is shown below:

.. code-block::

    from pyvqnet.tensor import arange
    from pyvqnet import kfloat32
    from pyvqnet.nn import Conv2D

    # an image feed into two dimension convolution layer
    b = 2        # batch size
    ic = 2       # input channels
    oc = 2      # output channels
    hw = 4      # input width and heights

    # two dimension convolution layer
    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")

    # input of shape [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,requires_grad=True,dtype=kfloat32)

    x1 = x0.reshape([b,ic,hw,hw])

    #forward function
    y = test_conv(x1)

    #backward function with autograd
    y.backward()
    print(x0.grad)
    # [-0.0008376 -0.2780731 -0.0008376 -0.2780731  0.1125833  0.2805848
    #   0.1125833  0.2805848 -0.0008376 -0.2780731 -0.0008376 -0.2780731
    #   0.1125833  0.2805848  0.1125833  0.2805848  0.5759128 -0.057339
    #   0.5759128 -0.057339  -0.342508  -0.4853693 -0.342508  -0.4853693
    #   0.5759128 -0.057339   0.5759128 -0.057339  -0.342508  -0.4853693
    #  -0.342508  -0.4853693 -0.0008376 -0.2780731 -0.0008376 -0.2780731
    #   0.1125833  0.2805848  0.1125833  0.2805848 -0.0008376 -0.2780731
    #  -0.0008376 -0.2780731  0.1125833  0.2805848  0.1125833  0.2805848
    #   0.5759128 -0.057339   0.5759128 -0.057339  -0.342508  -0.4853693
    #  -0.342508  -0.4853693  0.5759128 -0.057339   0.5759128 -0.057339
    #  -0.342508  -0.4853693 -0.342508  -0.4853693]

Model Training on GPU
=================================

You need to install the Linux version of pyvqnet to use the GPU. Make sure that both the data QTensor and the Module are on the GPU. You can use `toGPU` to transfer data or `gpu` to create a copy, or specify the device in the data creation function.

See the following example:

.. code-block::

    from pyvqnet.tensor import arange
    from pyvqnet import kfloat32,DEV_GPU_0
    from pyvqnet.nn import Conv2D


    b = 2        # batch size
    ic = 2       # input channels
    oc = 2      # output channels
    hw = 4      # input width and heights


    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")
    # Copy to DEV_GPU_0 on the GPU
    test_conv.toGPU(DEV_GPU_0)
    # input of shape [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,requires_grad=True,dtype=kfloat32).reshape([b,ic,hw,hw])

    # Copy data to DEV_GPU_0 with the GPU. You can also specify the ID inside the function, or set the device through the relevant tensor interface.
    x0 = x0.GPU(DEV_GPU_0)
    x0.requires_grad = True
    #forward function
    x = test_conv(x0)

    #backward function with autograd
    x.backward()
    print(x0.grad)

.. currentmodule:: pyvqnet.nn


Module Class
*********************************************************

abstract calculation module


Module
=================================

.. py:class:: pyvqnet.nn.module.Module(name="", *args, **kwargs)

    Base class of all neural network modules, including quantum and classical modules. Your model should also subclass this class for autograd computation.
    Modules can contain other Module classes, which allows nesting them in a tree structure. You can assign submodules as regular attributes::


        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.nn.Conv2d(1, 20, (5,5))
                self.conv2 = pyvqnet.nn.Conv2d(20, 20, (5,5))
            def forward(self, x):
                x = pyvqnet.nn.activation.relu(self.conv1(x))
                return pyvqnet.nn.activation.relu(self.conv2(x))

    Submodules assigned in this way are registered.

    :param name: Module name. Default: "".

forward
=================================

.. py:method:: pyvqnet.nn.module.Module.forward(x, *args, **kwargs)

    Abstract forward computation function of the Module class.

    :param x: Input QTensor.
    :param \*args: Variable positional arguments.
    :param \*\*kwargs: Variable keyword arguments.

    :return: Model output.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet as vq
        from pyvqnet.nn import Conv2D
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "same")
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                    requires_grad=True,
                    dtype=vq.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[[4.3995643, 3.9317808, -2.0707254],
        #  [20.1951981, 21.6946659, 14.2591858],
        #  [38.4702759, 31.9730244, 24.5977650]],
        # [[-17.0607567, -31.5377998, -7.5618000],
        #  [-22.5664024, -40.3876266, -15.1564388],
        #  [-3.1080279, -18.5986233, -8.0648050]]],
        # [[[6.6493244, -13.4840755, -20.2554188],
        #  [54.4235802, 34.4462433, 26.8171902],
        #  [90.2827682, 62.9092331, 51.6892929]],
        # [[-22.3385429, -45.2448578, 5.7101378],
        #  [-32.9464149, -60.9557228, -10.4994345],
        #  [5.9029331, -20.5480480, -0.9379558]]]
        # ]



state_dict
=================================

.. py:method:: pyvqnet.nn.module.Module.state_dict(destination=None, prefix="", keep_vars=False)

    Return a dictionary containing the entire state of the module, including parameters and buffers.
    The keys are the corresponding parameter and buffer names.

    :param destination: Dictionary that holds the parameters and buffers of the module's internal modules.
    :param prefix: Naming prefix for parameters and buffers.
    :param keep_vars: If ``False``, the returned tensors are detached from automatic differentiation; if ``True``, they are not detached. Default: ``False``.

    :return: A dictionary containing the entire state of the module.

    Example::

        from pyvqnet.nn import Conv2D
        test_conv = Conv2D(2,3,(3,3),(2,2),"same")
        print(test_conv.state_dict().keys())
        #odict_keys(['weights', 'bias'])


toGPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toGPU(device: int = DEV_GPU_0)

    Move the parameters and buffers of the module and its submodules to the specified GPU device.

    device specifies the device on which internal data is stored. When device >= DEV_GPU_0, data is stored on the GPU. If your machine has multiple GPUs,
    you can specify different devices to store data. For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... stores data on GPUs with different indices.

    .. note::
        A Module cannot perform computation across different GPUs.
        If you try to create a QTensor on a GPU whose ID exceeds the maximum number of verified GPUs, a CUDA error is raised.

    :param device: Device on which the QTensor is currently stored. Default: DEV_GPU_0. With device = pyvqnet.DEV_GPU_0, data is stored on the first GPU; with device = DEV_GPU_1, on the second GPU, and so on.
    :return: The Module moved to the GPU device.

    Example::

        from pyvqnet.nn.conv import ConvT2D
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], (1, 1))
        test_conv = test_conv.toGPU()
        print(test_conv.backend)
        #1000


toCPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toCPU()

    Move the parameters and buffers of the module and its submodules to the specified CPU device.

    :return: The Module moved to the CPU device.

    Example::

        from pyvqnet.nn.conv import ConvT2D
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], (1, 1))
        test_conv = test_conv.toCPU()
        print(test_conv.backend)
        #0


.. _save_parameters:

Saving and Loading Model Parameters
*********************************************************

The following interfaces save model parameters to a file or load parameters from a file. Note that the file does not store the model structure; you must build the model structure manually.

save_parameters
=================================

.. py:function:: pyvqnet.utils.storage.save_parameters(obj, f)

    Save a dictionary of model parameters to a file.

    :param obj: Dictionary to save.
    :param f: Name of the file that stores the parameters.

    :return: None.

    Example::

        from pyvqnet.nn import Module,Conv2D
        import pyvqnet
        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        pyvqnet.utils.storage.save_parameters(model.state_dict(),"tmp.model")

load_parameters
=================================

.. py:function:: pyvqnet.utils.storage.load_parameters(f)

    Load parameters from a file into a dictionary.

    :param f: Name of the file that stores the parameters.

    :return: Dictionary containing the saved parameters.

    Example::

        from pyvqnet.nn import Module, Conv2D
        import pyvqnet


        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1,
                                    output_channels=6,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        model1 = Net()  # another Module object
        pyvqnet.utils.storage.save_parameters(model.state_dict(), "tmp.model")
        model_para = pyvqnet.utils.storage.load_parameters("tmp.model")
        model1.load_state_dict(model_para)

ModuleList
*********************************************************
.. py:class:: pyvqnet.nn.module.ModuleList(modules=None)


    Holds submodules in a list. ModuleList can be indexed like a regular Python list, and the internal parameters of the Modules it contains can be saved.

    :param modules: List of nn.Module

    :return: A list of modules

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn import Module,Linear,ModuleList
        from pyvqnet.qnn.pq3 import ProbsMeasure,QuantumLayer
        import pyqpanda3.core as pq
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# outside

            qubits = range(num_of_qubits)

            circuit = pq.QCircuit()
            circuit << pq.H(qubits[0])
            circuit << pq.H(qubits[1])
            circuit << pq.H(qubits[2])
            circuit << pq.H(qubits[3])

            circuit << pq.RZ(qubits[0],input[0])
            circuit << pq.RZ(qubits[1],input[1])
            circuit << pq.RZ(qubits[2],input[2])
            circuit << pq.RZ(qubits[3],input[3])

            circuit << pq.CNOT(qubits[0],qubits[1])
            circuit << pq.RZ(qubits[1],param[0])
            circuit << pq.CNOT(qubits[0],qubits[1])

            circuit << pq.CNOT(qubits[1],qubits[2])
            circuit << pq.RZ(qubits[2],param[1])
            circuit << pq.CNOT(qubits[1],qubits[2])

            circuit << pq.CNOT(qubits[2],qubits[3])
            circuit << pq.RZ(qubits[3],param[2])
            circuit << pq.CNOT(qubits[2],qubits[3])

            prog = pq.QProg()
            prog << circuit

            rlt_prob = ProbsMeasure(m_machine,prog,[0,2])
            return rlt_prob


        class M(Module):
            def __init__(self):
                super(M, self).__init__()
                self.pqc2 = ModuleList([QuantumLayer(pqctest,3), Linear(4,1)
                ])

            def forward(self, x, *args, **kwargs):
                y = self.pqc2[0](x)  + self.pqc2[1](x)
                return y

        mm = M()
        print(mm.state_dict().keys())
        #odict_keys(['pqc2.0.m_para', 'pqc2.1.weights', 'pqc2.1.bias'])


ParameterList
*********************************************************
.. py:class:: pyvqnet.nn.module.ParameterList(values=None)


    Holds parameters in a list. ParameterList can be indexed like a regular Python list, and the internal parameters of the Parameters it contains can be saved.

    :param values: List of nn.Parameter

    :return: A list of parameters

    Example::

        from pyvqnet import nn
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter((10, 10)) for i in range(10)])
            def forward(self, x):

                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())


Sequential
*********************************************************
.. py:class:: pyvqnet.nn.module.Sequential(*args)

    Modules are added in the order they are passed. Alternatively, an ``OrderedDict`` of modules can be passed. The ``forward()`` method of ``Sequential`` accepts any input and forwards it to its first module.
    The output is then chained to the input of each subsequent module in order, and the output of the last module is returned.

    :param modules: Added Modules

    :return: Sequential

    Example::

        from pyvqnet import nn
        from collections import OrderedDict

        # Use Sequential to create a small model
        model = nn.Sequential(
                  nn.Conv2D(1,20,(5, 5)),
                  nn.ReLu(),
                  nn.Conv2D(20,64,(5, 5)),
                  nn.ReLu()
                )
        print(model.state_dict().keys())

        # Use Sequential with an OrderedDict, as shown below

        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2D(1,20,(5, 5))),
                  ('relu1', nn.ReLu()),
                  ('conv2', nn.Conv2D(20,64,(5, 5))),
                  ('relu2', nn.ReLu())
                ]))
        print(model.state_dict().keys())

Classical Neural Network Layers
*********************************************************

The following sections implement several classical neural network layers: convolution, transposed convolution, pooling, normalization, recurrent neural networks, and more.


Conv1D
=================================

.. py:class:: pyvqnet.nn.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:bool = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Apply a 1-D convolution over the input. The input of the Conv1D module has shape (batch_size, input_channels, in_height).

    :param input_channels: `int` - Number of channels in the input data.
    :param output_channels: `int` - Number of channels in the output data.
    :param kernel_size: `int` - Size of the convolution kernel. Kernel shape = [output_channels,input_channels/group,kernel_size,1].
    :param stride: `int` - Stride. Default: 1.
    :param padding: `str|int` - Padding option. It can be a string {'valid', 'same'} or an integer giving the amount of padding applied to the input. Default: "valid".
    :param use_bias: `bool` - Whether to use a bias term. Enabled by default.
    :param kernel_initializer: `callable` - Kernel initialization method. Defaults to empty, which uses kaiming_uniform.
    :param bias_initializer: `callable` - Bias initialization method. Defaults to empty, which uses kaiming_uniform.
    :param dilation_rate: `int` - Dilation size. Default: 1.
    :param group: `int` - Number of groups for grouped convolution. Default: 1.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of the module. Default: "".

    :return: A 1-D convolution instance.

    .. note::
        ``padding='valid'`` applies no padding.

        ``padding='same'`` pads the input with zeros; the output out_height is ceil(in_height / stride).

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Conv1D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv1D(ic,oc,3,2,"same")
        x0 = QTensor(np.arange(1,b*ic*5*5 +1).reshape([b,ic,25]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[12.4438553, 14.8618164, 15.5595102, 16.2572021, 16.9548950, 17.6525879, 18.3502808, 19.0479736, 19.7456665, 20.4433594, 21.1410522, 21.8387432, 10.5725441],
        #  [-13.7539215, 1.0263026, 1.2747254, 1.5231485, 1.7715728, 2.0199962, 2.2684195, 2.5168428, 2.7652662, 3.0136888, 3.2621140, 3.5105357, 14.0515862]],
        # [[47.4924164, 41.0252953, 41.7229881, 42.4206772, 43.1183739, 43.8160667, 44.5137596, 45.2114487, 45.9091415, 46.6068344, 47.3045311, 48.0022240, 18.3216572],
        #  [-47.2381554, 10.3421783, 10.5906038, 10.8390274, 11.0874519, 11.3358765, 11.5842953, 11.8327246, 12.0811434, 12.3295631, 12.5779924, 12.8264122, 39.4719162]]
        # ]


Conv2D
=================================

.. py:class:: pyvqnet.nn.Conv2D(input_channels:int,output_channels:int,kernel_size:tuple,stride:tuple=(1, 1),padding="valid",use_bias = True,kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Apply a 2-D convolution over the input. The input of the Conv2D module has shape (batch_size, input_channels, height, width).

    :param input_channels: `int` - Number of channels in the input data.
    :param output_channels: `int` - Number of channels in the output data.
    :param kernel_size: `tuple|list` - Size of the convolution kernel. Kernel shape = [output_channels,input_channels/group,kernel_size,kernel_size].
    :param stride: `tuple|list` - Stride. Default: (1, 1)|[1,1].
    :param padding: `str|tuple` - Padding option. It can be a string {'valid', 'same'} or a tuple of ints giving the implicit padding applied on both sides. Default: "valid".
    :param use_bias: `bool` - Whether to use a bias term. Enabled by default.
    :param kernel_initializer: `callable` - Kernel initialization method. Defaults to empty, which uses kaiming_uniform.
    :param bias_initializer: `callable` - Bias initialization method. Defaults to empty, which uses kaiming_uniform.
    :param dilation_rate: `int` - Dilation size. Default: 1.
    :param group: `int` - Number of groups for grouped convolution. Default: 1.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of the module. Default: "".

    :return: A 2-D convolution instance.

    .. note::
        ``padding='valid'`` applies no padding.

        ``padding='same'`` pads the input with zeros; the output height is ceil(height / stride) and the output width is ceil(width / stride).

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Conv2D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv2D(ic,oc,(3,3),(2,2),"same")
        x0 = QTensor(np.arange(1,b*ic*5*5+1).reshape([b,ic,5,5]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[[-0.1256833, 23.8978596, 26.7449780],
        #  [-7.2959919, 33.4023743, 42.1283913],
        #  [-8.7684336, 25.2698975, 40.4024887]],
        # [[33.0653763, 40.3120155, 27.3781891],
        #  [39.2921371, 45.8685760, 38.1885109],
        #  [23.1873779, 12.0480318, 12.7278290]]],
        # [[[-0.9730744, 61.3967094, 79.0511856],
        #  [-29.3652401, 75.0349350, 112.7325439],
        #  [-26.4682808, 59.0924797, 104.2572098]],
        # [[66.8064194, 96.0953140, 72.9157486],
        #  [90.9154129, 110.7232437, 91.2616043],
        #  [56.8825951, 34.6904907, 30.1957760]]]
        # ]


ConvT2D
=================================

.. py:class:: pyvqnet.nn.ConvT2D(input_channels,output_channels,kernel_size,stride=[1, 1],padding="valid",use_bias="True", kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, out_padding = (0,0), group: int = 1, dtype = None, name = "")

    Apply a 2-D transposed convolution over the input. The input of the ConvT2D module has shape (batch_size, input_channels, height, width).

    :param input_channels: `int` - Number of channels in the input data.
    :param output_channels: `int` - Number of channels in the output data.
    :param kernel_size: `tuple|list` - Size of the convolution kernel. Kernel shape = [input_channels,output_channels/group,kernel_size,kernel_size].
    :param stride: `tuple|list` - Stride. Default: (1, 1)|[1,1].
    :param padding: `str|tuple` - Padding option. It can be a string {'valid'} or a tuple of ints giving the implicit padding applied on both sides. Default: "valid".
    :param use_bias: `bool` - Whether to use a bias term. Enabled by default.
    :param kernel_initializer: `callable` - Kernel initialization method. Defaults to empty, which uses kaiming_uniform.
    :param bias_initializer: `callable` - Bias initialization method. Defaults to empty, which uses kaiming_uniform.
    :param dilation_rate: `int` - Dilation size. Default: 1.
    :param out_padding: Extra size added to one side of each dimension in the output shape. Default: (0,0)
    :param group: `int` - Number of groups for grouped convolution. Default: 1.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of the module. Default: "".

    :return: A 2-D transposed convolution instance.

    .. note::
        ``padding='valid'`` applies no padding.

        ``padding='same'`` is not supported; specify padding explicitly with a tuple of ints.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import ConvT2D
        import pyvqnet
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1), "valid")
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5+1).reshape([1, 3, 5, 5]), requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-3.3675897, 4.8476148, 14.2448473, 14.8897810, 15.5347166, 20.0420666, 10.9831696],
        #  [-14.0110836, -3.2500827, 6.4022207, 6.5149083, 6.6275964, 23.7946320, 12.1828709],
        #  [-22.2661152, -3.5112300, 12.9493723, 13.5486069, 14.1478367, 39.6327629, 18.8349991],
        #  [-24.4063797, -3.0093837, 15.9455290, 16.5447617, 17.1439915, 44.7691879, 21.3293095],
        #  [-26.5466480, -2.5075383, 18.9416828, 19.5409145, 20.1401463, 49.9056053, 23.8236179],
        #  [-24.7624626, -13.7395811, -7.9510674, -7.9967723, -8.0424776, 19.2783546, 7.0562835],
        #  [-3.5170188, 10.2280807, 16.1939259, 16.6804695, 17.1670132, 21.2262039, 6.2889833]],
        # [[-2.0570512, -9.5056667, -25.0429192, -25.9464111, -26.8499031, -24.7305946, -16.9881954],
        #  [-0.7620960, -18.3383904, -49.8948288, -51.2528229, -52.6108208, -52.2179604, -34.3664169],
        #  [-11.7121849, -27.1864738, -62.2154846, -63.6433640, -65.0712280, -52.6787071, -38.4497032],
        #  [-13.3643141, -29.0211792, -69.3548126, -70.7826691, -72.2105408, -58.1659012, -43.7543182],
        #  [-15.0164423, -30.8558884, -76.4941254, -77.9219971, -79.3498535, -63.6530838, -49.0589256],
        #  [-11.6070204, -14.1940546, -35.5471687, -36.0715408, -36.5959129, -23.9147663, -22.8668022],
        #  [-14.4390459, -4.9011412, -6.4719801, -6.5418491, -6.6117167, 9.3329525, -1.7254852]]]
        # ]


AvgPool1D
=================================

.. py:class:: pyvqnet.nn.AvgPool1D(kernel, stride, padding="valid", name = "")

    Apply average pooling to a 1-D input. The input has shape (batch_size, input_channels, in_height).

    :param kernel: Window size for average pooling.
    :param stride: Stride of the moving window.
    :param padding: Padding option: "valid", "same", or an integer specifying the padding length. Default: "valid".
    :param name: Name of the module. Default: "".

    :return: A 1-D average pooling layer instance.

    .. note::
        ``padding='valid'`` applies no padding.

        ``padding='same'`` pads the input with zeros; the output out_height is ceil(in_height / stride).

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import AvgPool1D
        test_mp = AvgPool1D([3],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [
        # [[0.3333333, 1.6666666, 3.],
        #  [1.6666666, 2., 1.3333334],
        #  [2.6666667, 2.6666667, 2.3333333],
        #  [2.3333333, 4.3333335, 3.3333333],
        #  [0.3333333, 1.6666666, 4.]]
        # ]


MaxPool1D
=================================

.. py:class:: pyvqnet.nn.MaxPool1D(kernel, stride, padding="valid",name="")

    Apply max pooling to a 1-D input. The input has shape (batch_size, input_channels, in_height).

    :param kernel: Window size for max pooling.
    :param stride: Stride of the moving window.
    :param padding: Padding option: "valid", "same", or an integer specifying the padding length. Default: "valid".
    :param name: Name. Default: "".

    :return: A 1-D max pooling layer instance.

    .. note::
        ``padding='valid'`` applies no padding.

        ``padding='same'`` pads the input with zeros; the output out_height is ceil(in_height / stride).

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import MaxPool1D
        test_mp = MaxPool1D([3],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[1. 4. 5.]
        #   [3. 3. 3.]
        #   [4. 4. 4.]
        #   [5. 6. 6.]
        #   [1. 5. 7.]]]

AvgPool2D
=================================

.. py:class:: pyvqnet.nn.AvgPool2D( kernel, stride, padding="valid",name="")

    Apply average pooling to a 2-D input. The input has shape (batch_size, input_channels, height, width).

    :param kernel: Window size for average pooling.
    :param stride: Stride of the moving window.
    :param padding: Padding option: "valid", "same", or a tuple of 2 integers giving the padding length in the two dimensions. Default: "valid".
    :param name: Name. Default: "".

    :return: A 2-D average pooling layer instance.

    .. note::
        ``padding='valid'`` applies no padding.

        ``padding='same'`` pads the input with zeros; the output height is ceil(height / stride) and the output width is ceil(width / stride).

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import AvgPool2D
        test_mp = AvgPool2D([2,2],[2,2],"valid")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[[1.5  1.75]
        #    [3.75 3.  ]]]]


MaxPool2D
=================================

.. py:class:: pyvqnet.nn.MaxPool2D(kernel, stride, padding="valid",name="")

    Apply max pooling to a 2-D input. The input has shape (batch_size, input_channels, height, width).

    :param kernel: Window size for max pooling.
    :param stride: Stride of the moving window.
    :param padding: Padding option: "valid", "same", or a tuple of 2 integers giving the padding length in the two dimensions. Default: "valid".
    :param name: Name. Default: "".

    :return: A 2-D max pooling layer instance.

    .. note::
        ``padding='valid'`` applies no padding.

        ``padding='same'`` pads the input with zeros; the output height is ceil(height / stride) and the output width is ceil(width / stride).


    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import MaxPool2D
        test_mp = MaxPool2D([2,2],[2,2],"valid")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [[[[3. 4.]
        #    [5. 6.]]]]


Embedding
=================================

.. py:class:: pyvqnet.nn.embedding.Embedding(num_embeddings, embedding_dim, weight_initializer=xavier_normal, dtype=None, name: str = "")

    This module is typically used to store word embeddings and retrieve them using indices. The input of the module is a list of indices and the output is the corresponding word embeddings.
    The input of this layer should be kint64.

    :param num_embeddings: `int` - Size of the embedding dictionary.
    :param embedding_dim: `int` - Size of each embedding vector
    :param weight_initializer: `callable` - Parameter initialization method. Defaults to a normal distribution.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of the embedding layer. Default: "".

    :return: An Embedding instance.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.embedding import Embedding
        import pyvqnet
        vlayer = Embedding(30,3)
        x = QTensor(np.arange(1,25).reshape([2,3,2,2]),dtype= pyvqnet.kint64)
        y = vlayer(x)
        print(y)

        # [
        # [[[[-0.3168081, 0.0329394, -0.2934906],
        #  [0.1057295, -0.2844988, -0.1687456]],
        # [[-0.2382513, -0.3642318, -0.2257225],
        #  [0.1563180, 0.1567665, 0.3038477]]],
        # [[[-0.4131152, -0.0564500, -0.2804018],
        #  [-0.2955172, -0.0009581, -0.1641144]],
        # [[0.0692555, 0.1094901, 0.4099118],
        #  [0.4348361, 0.0304361, -0.0061203]]],
        # [[[-0.3310401, -0.1836129, 0.1098949],
        #  [-0.1840732, 0.0332474, -0.0261806]],
        # [[-0.1489778, 0.2519453, 0.3299376],
        #  [-0.1942692, -0.1540277, -0.2335350]]]],
        # [[[[-0.2620637, -0.3181309, -0.1857461],
        #  [-0.0878164, -0.4180320, -0.1831555]],
        # [[-0.0738970, -0.1888980, -0.3034399],
        #  [0.1955448, -0.0409723, 0.3023460]]],
        # [[[0.2430045, 0.0880465, 0.4309453],
        #  [-0.1796514, -0.1432367, -0.1253638]],
        # [[-0.5266719, 0.2386262, -0.0329155],
        #  [0.1033449, -0.3442690, -0.0471130]]],
        # [[[-0.5336705, -0.1939755, -0.3000667],
        #  [0.0059001, 0.5567381, 0.1926173]],
        # [[-0.2385869, -0.3910453, 0.2521235],
        #  [-0.0246447, -0.0241158, -0.1402829]]]]
        # ]



BatchNorm2d
=================================

.. py:class:: pyvqnet.nn.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5,affine = True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Apply batch normalization to a 4-D input (B, C, H, W). See the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` and :math:`\beta` are trainable parameters. In addition, by default, during training this layer keeps running estimates of the mean and variance it computes, which are then used for normalization during evaluation. The running averages use the default momentum 0.1.

    :param channel_num: `int` - Number of input channels.
    :param momentum: `float` - Momentum used when computing the exponentially weighted average. Default: 0.1.
    :param epsilon: `float` - Numerical stability constant. Default: 1e-5.
    :param affine: `bool` - A boolean value. When set to ``True``, this module has learnable per-channel affine parameters initialized to 1 (for weights) and 0 (for biases). Default: ``True``.
    :param beta_initializer: `callable` - Initialization method for beta. Defaults to all-zero initialization.
    :param gamma_initializer: `callable` - Initialization method for gamma. Defaults to all-one initialization.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of the batch normalization layer. Default: "".

    :return: A 2-D batch normalization layer instance.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import BatchNorm2d
        import pyvqnet
        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)

        x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]],
        # [[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]]],
        # [[[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]],
        # [[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]]]
        # ]


BatchNorm1d
=================================

.. py:class:: pyvqnet.nn.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine = True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Apply batch normalization to a 2-D input (B,C). See the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` and :math:`\beta` are trainable parameters. In addition, by default, during training this layer keeps running estimates of the mean and variance it computes, which are then used for normalization during evaluation. The running averages use the default momentum 0.1.


    :param channel_num: `int` - Number of input channels.
    :param momentum: `float` - Momentum used when computing the exponentially weighted average. Default: 0.1.
    :param epsilon: `float` - Numerical stability constant. Default: 1e-5.
    :param affine: `bool` - A boolean value. When set to ``True``, this module has learnable per-channel affine parameters initialized to 1 (for weights) and 0 (for biases). Default: ``True``.
    :param beta_initializer: `callable` - Initialization method for beta. Defaults to all-zero initialization.
    :param gamma_initializer: `callable` - Initialization method for gamma. Defaults to all-one initialization.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of the batch normalization layer. Default: "".

    :return: A 1-D batch normalization layer instance.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import BatchNorm1d
        import pyvqnet
        test_conv = BatchNorm1d(4)

        x = QTensor(np.arange(1, 17).reshape([4, 4]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416405, -1.3416405, -1.3416405, -1.3416405],
        # [-0.4472135, -0.4472135, -0.4472135, -0.4472135],
        # [0.4472135, 0.4472135, 0.4472135, 0.4472135],
        # [1.3416405, 1.3416405, 1.3416405, 1.3416405]
        # ]

LayerNormNd
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5,affine=True, dtype=None, name="")

    Apply layer normalization over the last D dimensions of an arbitrary input, as described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    For an input such as (B,C,H,W,D), ``norm_shape`` can be [C,H,W,D], [H,W,D], [W,D], or [D] .

    :param norm_shape: `float` - Normalized shape.
    :param epsilon: `float` - Numerical stability constant. Default: 1e-5.
    :param affine: `bool` - A boolean value. When set to ``True``, this module has learnable per-channel affine parameters initialized to 1 (for weights) and 0 (for biases). Default: ``True``.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: A LayerNormNd instance

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor,kfloat32
        from pyvqnet.nn.layer_norm import LayerNormNd
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = QTensor(np.arange(1,17).reshape([2,2,2,2]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)
        # [
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]],
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]]
        # ]

LayerNorm2d
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    Apply layer normalization to a 4-D input, as described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard deviation are computed over the remaining dimensions excluding the first. For an input such as (B,C,H,W), ``norm_size`` should equal C * H * W.

    :param norm_size: `float` - Normalization size, which should equal C * H * W.
    :param epsilon: `float` - Numerical stability constant. Default: 1e-5.
    :param affine: `bool` - A boolean value. When set to ``True``, this module has learnable per-channel affine parameters initialized to 1 (for weights) and 0 (for biases). Default: ``True``.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: A 2-D layer normalization instance.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.layer_norm import LayerNorm2d
        ic = 4
        test_conv = LayerNorm2d(8)
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]],
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]]
        # ]


LayerNorm1d
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    Apply layer normalization to a 2-D input, as described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard deviation are computed over the last dimension, where the norm size is the value of ``norm_size``.

    :param norm_size: `float` - Normalization size, which should equal the size of the last dimension.
    :param epsilon: `float` - Numerical stability constant. Default: 1e-5.
    :param affine: `bool` - A boolean value. When set to ``True``, this module has learnable per-channel affine parameters initialized to 1 (for weights) and 0 (for biases). Default: ``True``.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: A 1-D layer normalization instance.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.layer_norm import LayerNorm1d
        test_conv = LayerNorm1d(4)
        x = QTensor(np.arange(1,17).reshape([4,4]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355]
        # ]


GroupNorm
===========================================================

.. py:class:: pyvqnet.nn.group_norm.GroupNorm(num_groups: int, num_channels: int, epsilon = 1e-5, affine = True, dtype = None, name = "")

    Apply group normalization to a mini-batch input. Input: :math:`(N, C, *)` where :math:`C=\mathrm{num\_channels}`, output: :math:`(N, C, *)`.

    This layer implements the operation described in the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are split into :attr:`num_groups` groups, each containing ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by :attr:`num_groups`. The mean and standard deviation are computed separately within each group. If :attr:`affine` is ``True``, :math:`\gamma` and :math:`\beta` are learnable. There is one per-channel affine transformation parameter vector of size :attr:`num_channels`.

    :param num_groups (int): Number of groups to split the channels into
    :param num_channels (int): Number of channels expected in the input
    :param eps: Value added to the denominator for numerical stability. Default: 1e-5
    :param affine: A boolean value. When set to ``True``, this module has learnable per-channel affine parameters initialized to 1 (for weights) and 0 (for biases). Default: ``True``.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: GroupNorm class

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor,kfloat32
        from pyvqnet.nn import GroupNorm
        test_conv = GroupNorm(2,10)
        x = QTensor(np.arange(0,60*2*5).reshape([2,10,3,2,5]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

RMSNorm
=================================

.. py:class:: pyvqnet.nn.RMSNorm(normalized_shape, eps:float = 1e-5, affine:bool = True, dtype=None)

    Apply Root Mean Square Layer Normalization to a mini-batch of input data. The RMS is taken over the last dimension of the input.

    :param normalized_shape: Normalized shape. If a single integer, it is treated as ``[normalized_shape]``.
    :param eps: Numerical stability constant. Default: 1e-5.
    :param affine: Whether to use a learnable affine parameter (gamma). Default: True.
    :param dtype: Parameter data type. Default: None, which uses the default data type.
    :return: An RMSNorm class instance.

    .. note::

            This module only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.nn import RMSNorm
        from pyvqnet.tensor import tensor, kfloat32
        rms = RMSNorm(8, eps=1e-5, dtype=kfloat32)
        rms.toGPU(pyvqnet.DEV_GPU_0)
        x = tensor.arange(1.0, 17.0, dtype=kfloat32).reshape([2, 8])
        x = x.GPU(pyvqnet.DEV_GPU_0)
        x.requires_grad = True
        y = rms(x)
        print(y)

        # [[0.1980295,0.3960589,0.5940884,...,1.1881769,1.3862063,1.5842358],
        #  [0.7082005,0.7868894,0.8655784,...,1.1016452,1.1803341,1.2590231]]

RoPE
=================================

.. py:class:: pyvqnet.nn.RoPE(head_dim:int, max_seq_len:int = 2048, base:float = 10000.0, rope_type:str = "standard", scale_factor:float = 4.0, original_max_seq_len = None, beta_fast:float = 32.0, beta_slow:float = 1.0)

    Rotary Position Embedding. Injects positional information by rotating dimension pairs in the last dimension (head_dim) of the query and key tensors.

    :param head_dim: Attention head dimension (must be even).
    :param max_seq_len: Maximum sequence length for the precomputed cos/sin cache. Default: 2048.
    :param base: RoPE base frequency. Default: 10000.0 (Llama uses 500000.0).
    :param rope_type: Rotation type. Options: "standard" | "ntk" | "dynamic_ntk" | "yarn". Default: "standard".
    :param scale_factor: Extension scale factor (for example, 4.0 means 4x context). Default: 4.0.
    :param original_max_seq_len: Pretraining sequence length. Default: None.
    :param beta_fast: YaRN high-frequency cutoff. Default: 32.0.
    :param beta_slow: YaRN low-frequency cutoff. Default: 1.0.
    :return: A RoPE class instance.

    .. note::

            This module only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, DEV_GPU_0
        from pyvqnet.nn.rope import RoPE
        import numpy as np

        # Fixed input
        q = QTensor(np.arange(1.0, 17.0, dtype=np.float32).reshape([1, 1, 4, 4]), device=DEV_GPU_0)
        k = QTensor(np.arange(17.0, 33.0, dtype=np.float32).reshape([1, 1, 4, 4]), device=DEV_GPU_0)

        # 1. standard
        rope = RoPE(4, max_seq_len=4, rope_type="standard")
        rope.toGPU(DEV_GPU_0)
        out_q, _ = rope.forward(q, k)
        print(out_q)

        # [[[[  1.       ,  2.       ,  3.       ,  4.       ],
        #    [ -3.1887856,  5.919701 ,  7.9894705,  8.059599 ],
        #    [-13.747593 ,  9.758017 ,  3.6060615, 12.197587 ],
        #    [-14.986703 , 13.513773 ,-13.015327 , 16.412737 ]]]]

        # 2. ntk
        rope = RoPE(4, max_seq_len=4, rope_type="ntk", scale_factor=4.0)
        rope.toGPU(DEV_GPU_0)
        out_q, _ = rope.forward(q, k)
        print(out_q)

        # [[[[  1.       ,  2.       ,  3.       ,  4.       ],
        #    [ -3.1887856,  5.9799814,  7.9894705,  8.014976 ],
        #    [-13.747593 ,  9.939875 ,  3.6060615, 12.0498495],
        #    [-14.986703 , 13.879607 ,-13.015327 , 16.10455  ]]]]

        # 3. dynamic_ntk
        rope = RoPE(4, max_seq_len=4, rope_type="dynamic_ntk", scale_factor=4.0, original_max_seq_len=2)
        rope.toGPU(DEV_GPU_0)
        out_q, _ = rope.forward(q, k)
        print(out_q)

        # [[[[  1.       ,  2.       ,  3.       ,  4.       ],
        #    [ -3.1887856,  5.959925 ,  7.9894705,  8.0299   ],
        #    [-13.747593 ,  9.879502 ,  3.6060615, 12.099399 ],
        #    [-14.986703 , 13.758434 ,-13.015327 , 16.208193 ]]]]

        # 4. yarn
        rope = RoPE(4, max_seq_len=4, rope_type="yarn", scale_factor=4.0, original_max_seq_len=2)
        rope.toGPU(DEV_GPU_0)
        out_q, _ = rope.forward(q, k)
        print(out_q)

        # [[[[  1.       ,  2.       ,  3.       ,  4.       ],
        #    [ -3.1887856,  5.9799814,  7.9894705,  8.014976 ],
        #    [-13.747593 ,  9.939875 ,  3.6060615, 12.0498495],
        #    [-14.986703 , 13.879607 ,-13.015327 , 16.10455  ]]]]

Linear
=================================

.. py:class:: pyvqnet.nn.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, dtype=None, name: str = "")

    Linear module (fully connected layer).
    :math:`y = x@A.T + b`

    :param input_channels: `int` - Number of input channels.
    :param output_channels: `int` - Number of output channels.
    :param weight_initializer: `callable` - Weight initialization function. Defaults to empty, which uses he_uniform.
    :param bias_initializer: `callable` - Bias initialization parameter. Defaults to empty, which uses he_uniform.
    :param use_bias: `bool` - Whether to use a bias term. Enabled by default.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of the linear layer. Default: "".

    :return: A linear layer instance.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Linear
        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout)
        input = QTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

        # [
        # [[4.3084583, -1.9228780, -0.3428757, 1.2840536, -0.5865945],
        #  [9.8339605, -5.5135884, -3.1228657, 4.3025794, -4.1492314],
        #  [15.3594627, -9.1042995, -5.9028554, 7.3211040, -7.7118683]],
        # [[20.8849659, -12.6950111, -8.6828451, 10.3396301, -11.2745066],
        #  [26.4104652, -16.2857227, -11.4628344, 13.3581581, -14.8371439],
        #  [31.9359703, -19.8764324, -14.2428246, 16.3766804, -18.3997803]]
        # ]



Dropout
=================================

.. py:class:: pyvqnet.nn.dropout.Dropout(dropout_rate: float = 0.5, name: str = "")

    Dropout module. The dropout module randomly sets the output of some units to zero while scaling the other units according to the given dropout_rate probability.

    :param dropout_rate: `float` - Probability that a neuron is set to zero.
    :param name: Name of this module. Default: "".

    :return: A Dropout instance.

    Example::

        import numpy as np
        from pyvqnet.nn.dropout import Dropout
        from pyvqnet.tensor import QTensor
        b = 2
        ic = 2
        x = QTensor(np.arange(-1 * ic * 2 * 2,
                            (b - 1) * ic * 2 * 2.0).reshape([b, ic, 2, 2]),
                    requires_grad=True)
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        print(y)

        # [
        # [[[-16., -14.],
        #  [-0., -0.]],
        # [[-8., -6.],
        #  [-4., -2.]]],
        # [[[0., 2.],
        #  [4., 6.]],
        # [[8., 10.],
        #  [0., 14.]]]
        # ]


DropPath
=================================

.. py:class:: pyvqnet.nn.dropout.DropPath(dropout_rate = 0.5,name="")

    The DropPath module drops paths per sample (stochastic depth).

    :param dropout_rate: `float` - Probability that a neuron is set to zero.
    :param name: Name of this module. Default: "".

    :return: A DropPath instance.

    Example::

        import pyvqnet.nn as nn
        import pyvqnet.tensor as tensor

        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)
        #[0.9074978,0.9350062,0.6896403,0.3541051]




Pixel_Shuffle
=================================

.. py:class:: pyvqnet.nn.pixel_shuffle.Pixel_Shuffle(upscale_factors, name="")

    Rearrange a tensor of shape (*, C * r^2, H, W)
    into a tensor of shape (*, C, H * r, W * r), where r is the upscale factor.

    :param upscale_factors: Upscale factor
    :param name: Name of this module. Default: "".

    :return:
            Pixel_Shuffle module

    Example::

        from pyvqnet.nn import Pixel_Shuffle
        from pyvqnet.tensor import tensor
        ps = Pixel_Shuffle(3)
        inx = tensor.ones([5,2,3,18,4,4])
        inx.requires_grad=  True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 2, 12, 12]

Pixel_Unshuffle
=================================

.. py:class:: pyvqnet.nn.pixel_shuffle.Pixel_Unshuffle(downscale_factors, name="")

    Reverse the Pixel_Shuffle operation by rearranging elements. Changes a tensor of shape (*, C, H * r, W * r) into (*, C * r^2, H, W), where r is the downscale factor.

    :param downscale_factors: Downscale factor
    :param name: Name of this module. Default: "".

    :return:
            Pixel_Unshuffle module

    Example::

        from pyvqnet.nn import Pixel_Unshuffle
        from pyvqnet.tensor import tensor
        ps = Pixel_Unshuffle(3)
        inx = tensor.ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 18, 4, 4]


GRU
=================================

.. py:class:: pyvqnet.nn.gru.GRU(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Gated Recurrent Unit (GRU) module. Supports multi-layer stacking and bidirectional configuration. The computation formulas for a single-layer unidirectional GRU are as follows:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked GRU layers. Default: 1.
    :param batch_first: If True, the input shape is [batch_size,seq_len,feature_dim];
     if False, the input shape is [seq_len,batch_size,feature_dim]. Default: True.
    :param use_bias: If False, the module does not use bias terms. Default: True.
    :param bidirectional: If True, becomes a bidirectional GRU. Default: False.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: GRU instance

    Example::

        from pyvqnet.nn import GRU
        from pyvqnet.tensor import tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)
        print(output)
        print(hn)
        # [
        # [[0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812]],
        # [[0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840],
        #  [0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840],
        #  [0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840]],
        # [[0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123],
        #  [0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123],
        #  [0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123]],
        # [[0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759],
        #  [0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759],
        #  [0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759]],
        # [[0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093]]
        # ]
        # [
        # [[-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725],
        #  [-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725],
        #  [-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725]],
        # [[-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730],
        #  [-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730],
        #  [-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730]],
        # [[0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535]],
        # [[-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812]]
        # ]

RNN
=================================

.. py:class:: pyvqnet.nn.rnn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Recurrent neural network (RNN) module using :math:`\tanh` or :math:`\text{ReLU}` as the activation function. Supports bidirectional and multi-layer configurations.
    The computation formula for a single-layer unidirectional RNN is as follows:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    If :attr:`nonlinearity` is ``'relu'``, :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked RNN layers. Default: 1.
    :param nonlinearity: Nonlinear activation function. Default: ``'tanh'``.
    :param batch_first: If True, the input shape is [batch_size,seq_len,feature_dim];
     if False, the input shape is [seq_len,batch_size,feature_dim]. Default: True.
    :param use_bias: If False, the module does not use bias terms. Default: True.
    :param bidirectional: If True, becomes a bidirectional RNN. Default: False.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: RNN instance

    Example::

        from pyvqnet.nn import RNN
        from pyvqnet.tensor import tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)
        print(output)
        print(hn)

        # [[[-0.413501  -0.4584284 -0.7530673  0.017902   0.5038776  0.6589128
        #    -0.3040171  0.6689512 -0.5059364 -0.4492912 -0.1206206 -0.3208488]
        #   [-0.413501  -0.4584284 -0.7530673  0.017902   0.5038776  0.6589128
        #    -0.3040171  0.6689512 -0.5059364 -0.4492912 -0.1206206 -0.3208488]
        #   [-0.413501  -0.4584284 -0.7530673  0.017902   0.5038776  0.6589128
        #    -0.3040171  0.6689512 -0.5059364 -0.4492912 -0.1206206 -0.3208488]]

        #  [[ 0.2127696 -0.6644132 -0.3884572 -0.0344413  0.35414    0.5639224
        #    -0.2358238  0.7706838 -0.3650029 -0.54729   -0.2392456 -0.3285939]
        #   [ 0.2127696 -0.6644132 -0.3884572 -0.0344413  0.35414    0.5639224
        #    -0.2358238  0.7706838 -0.3650029 -0.54729   -0.2392456 -0.3285939]
        #   [ 0.2127696 -0.6644132 -0.3884572 -0.0344413  0.35414    0.5639224
        #    -0.2358238  0.7706838 -0.3650029 -0.54729   -0.2392456 -0.3285939]]

        #  [[ 0.0432173 -0.3811357 -0.4611054 -0.255436   0.4998702  0.6649145
        #    -0.3132874  0.5965794 -0.5760088 -0.5618317 -0.2404964 -0.4669401]
        #   [ 0.0432173 -0.3811357 -0.4611054 -0.255436   0.4998702  0.6649145
        #    -0.3132874  0.5965794 -0.5760088 -0.5618317 -0.2404964 -0.4669401]
        #   [ 0.0432173 -0.3811357 -0.4611054 -0.255436   0.4998702  0.6649145
        #    -0.3132874  0.5965794 -0.5760088 -0.5618317 -0.2404964 -0.4669401]]

        #  [[ 0.1083845 -0.4397578 -0.4745184 -0.2148822  0.3674186  0.6907974
        #    -0.061549   0.452508  -0.5262724 -0.3184315 -0.4691838 -0.0438465]
        #   [ 0.1083845 -0.4397578 -0.4745184 -0.2148822  0.3674186  0.6907974
        #    -0.061549   0.452508  -0.5262724 -0.3184315 -0.4691838 -0.0438465]
        #   [ 0.1083845 -0.4397578 -0.4745184 -0.2148822  0.3674186  0.6907974
        #    -0.061549   0.452508  -0.5262724 -0.3184315 -0.4691838 -0.0438465]]

        #  [[-0.1835401 -0.6232781 -0.3571274  0.2283377  0.5426646  0.6719067
        #    -0.8672301  0.0971878 -0.7561615 -0.5063094  0.5117968 -0.0985391]
        #   [-0.1835401 -0.6232781 -0.3571274  0.2283377  0.5426646  0.6719067
        #    -0.8672301  0.0971878 -0.7561615 -0.5063094  0.5117968 -0.0985391]
        #   [-0.1835401 -0.6232781 -0.3571274  0.2283377  0.5426646  0.6719067
        #    -0.8672301  0.0971878 -0.7561615 -0.5063094  0.5117968 -0.0985391]]]
        # [[[ 0.2451548  0.9127097 -0.6998036  0.6434992  0.1046313  0.6530996]
        #   [ 0.2451548  0.9127097 -0.6998036  0.6434992  0.1046313  0.6530996]
        #   [ 0.2451548  0.9127097 -0.6998036  0.6434992  0.1046313  0.6530996]]

        #  [[-0.1760802 -0.546555   0.7547818  0.8418489 -0.2011115  0.1687339]
        #   [-0.1760802 -0.546555   0.7547818  0.8418489 -0.2011115  0.1687339]
        #   [-0.1760802 -0.546555   0.7547818  0.8418489 -0.2011115  0.1687339]]

        #  [[-0.1835401 -0.6232781 -0.3571274  0.2283377  0.5426646  0.6719067]
        #   [-0.1835401 -0.6232781 -0.3571274  0.2283377  0.5426646  0.6719067]
        #   [-0.1835401 -0.6232781 -0.3571274  0.2283377  0.5426646  0.6719067]]

        #  [[-0.3040171  0.6689512 -0.5059364 -0.4492912 -0.1206206 -0.3208488]
        #   [-0.3040171  0.6689512 -0.5059364 -0.4492912 -0.1206206 -0.3208488]
        #   [-0.3040171  0.6689512 -0.5059364 -0.4492912 -0.1206206 -0.3208488]]]



LSTM
=================================

.. py:class:: pyvqnet.nn.lstm.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Long Short-Term Memory (LSTM) module. Supports configurations such as bidirectional LSTM and stacked multi-layer LSTM. The computation formulas for a single-layer unidirectional LSTM are as follows:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked LSTM layers. Default: 1.
    :param batch_first: If True, the input shape is [batch_size,seq_len,feature_dim];
     if False, the input shape is [seq_len,batch_size,feature_dim]. Default: True.
    :param use_bias: If False, the module does not use bias terms. Default: True.
    :param bidirectional: If True, becomes a bidirectional LSTM. Default: False.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: LSTM instance

    Example::

        from pyvqnet.nn import LSTM
        from pyvqnet.tensor import tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))

        print(output)
        print(hn)
        print(cn)

        # [[[ 0.2732482  0.1701475  0.1449948  0.0736707  0.1340735  0.1299259
        #     0.0551452  0.2563322  0.040974   0.1358014 -0.1953781 -0.061493 ]
        #   [ 0.2732482  0.1701475  0.1449948  0.0736707  0.1340735  0.1299259
        #     0.0551452  0.2563322  0.040974   0.1358014 -0.1953781 -0.061493 ]
        #   [ 0.2732482  0.1701475  0.1449948  0.0736707  0.1340735  0.1299259
        #     0.0551452  0.2563322  0.040974   0.1358014 -0.1953781 -0.061493 ]]

        #  [[ 0.0973835  0.2728153  0.0826659  0.1486651  0.001023  -0.0002572
        #     0.0662036  0.2953667  0.0374205  0.1520839 -0.1803852 -0.0302957]
        #   [ 0.0973835  0.2728153  0.0826659  0.1486651  0.001023  -0.0002572
        #     0.0662036  0.2953667  0.0374205  0.1520839 -0.1803852 -0.0302957]
        #   [ 0.0973835  0.2728153  0.0826659  0.1486651  0.001023  -0.0002572
        #     0.0662036  0.2953667  0.0374205  0.1520839 -0.1803852 -0.0302957]]

        #  [[ 0.0228571  0.1898227  0.0312898  0.1559544 -0.0909701 -0.0543313
        #     0.0921322  0.3542943  0.0323457  0.1877485 -0.1509136  0.0398105]
        #   [ 0.0228571  0.1898227  0.0312898  0.1559544 -0.0909701 -0.0543313
        #     0.0921322  0.3542943  0.0323457  0.1877485 -0.1509136  0.0398105]
        #   [ 0.0228571  0.1898227  0.0312898  0.1559544 -0.0909701 -0.0543313
        #     0.0921322  0.3542943  0.0323457  0.1877485 -0.1509136  0.0398105]]

        #  [[-0.0164866  0.1411722 -0.002618   0.1457977 -0.1280925 -0.0702658
        #     0.1419625  0.4266106  0.0363048  0.2426503 -0.0849762  0.1489675]
        #   [-0.0164866  0.1411722 -0.002618   0.1457977 -0.1280925 -0.0702658
        #     0.1419625  0.4266106  0.0363048  0.2426503 -0.0849762  0.1489675]
        #   [-0.0164866  0.1411722 -0.002618   0.1457977 -0.1280925 -0.0702658
        #     0.1419625  0.4266106  0.0363048  0.2426503 -0.0849762  0.1489675]]

        #  [[-0.0223669  0.1591602 -0.0311901  0.0958475 -0.122227  -0.0682742
        #     0.2645994  0.4319879  0.0528811  0.3815826  0.1058483  0.1233259]
        #   [-0.0223669  0.1591602 -0.0311901  0.0958475 -0.122227  -0.0682742
        #     0.2645994  0.4319879  0.0528811  0.3815826  0.1058483  0.1233259]
        #   [-0.0223669  0.1591602 -0.0311901  0.0958475 -0.122227  -0.0682742
        #     0.2645994  0.4319879  0.0528811  0.3815826  0.1058483  0.1233259]]]
        # [[[ 0.1752738 -0.0356635  0.2989266  0.20517   -0.0896036  0.5098922]
        #   [ 0.1752738 -0.0356635  0.2989266  0.20517   -0.0896036  0.5098922]
        #   [ 0.1752738 -0.0356635  0.2989266  0.20517   -0.0896036  0.5098922]]

        #  [[-0.2443907 -0.0827548 -0.1557958  0.1731217 -0.0342197 -0.1101249]
        #   [-0.2443907 -0.0827548 -0.1557958  0.1731217 -0.0342197 -0.1101249]
        #   [-0.2443907 -0.0827548 -0.1557958  0.1731217 -0.0342197 -0.1101249]]

        #  [[-0.0223669  0.1591602 -0.0311901  0.0958475 -0.122227  -0.0682742]
        #   [-0.0223669  0.1591602 -0.0311901  0.0958475 -0.122227  -0.0682742]
        #   [-0.0223669  0.1591602 -0.0311901  0.0958475 -0.122227  -0.0682742]]

        #  [[ 0.0551452  0.2563322  0.040974   0.1358014 -0.1953781 -0.061493 ]
        #   [ 0.0551452  0.2563322  0.040974   0.1358014 -0.1953781 -0.061493 ]
        #   [ 0.0551452  0.2563322  0.040974   0.1358014 -0.1953781 -0.061493 ]]]
        # [[[ 0.3543518 -0.1842273  0.8745036  0.6013567 -0.1227313  1.0065726]
        #   [ 0.3543518 -0.1842273  0.8745036  0.6013567 -0.1227313  1.0065726]
        #   [ 0.3543518 -0.1842273  0.8745036  0.6013567 -0.1227313  1.0065726]]

        #  [[-0.5912023 -0.2058601 -0.4632604  0.3671726 -0.0673411 -0.327456 ]
        #   [-0.5912023 -0.2058601 -0.4632604  0.3671726 -0.0673411 -0.327456 ]
        #   [-0.5912023 -0.2058601 -0.4632604  0.3671726 -0.0673411 -0.327456 ]]

        #  [[-0.0393696  0.301177  -0.0757513  0.196299  -0.3163165 -0.1804099]
        #   [-0.0393696  0.301177  -0.0757513  0.196299  -0.3163165 -0.1804099]
        #   [-0.0393696  0.301177  -0.0757513  0.196299  -0.3163165 -0.1804099]]

        #  [[ 0.1547394  0.4916601  0.1061193  0.2582704 -0.3499697 -0.1048216]
        #   [ 0.1547394  0.4916601  0.1061193  0.2582704 -0.3499697 -0.1048216]
        #   [ 0.1547394  0.4916601  0.1061193  0.2582704 -0.3499697 -0.1048216]]]

Dynamic_GRU
=================================

.. py:class:: pyvqnet.nn.gru.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Apply a multi-layer Gated Recurrent Unit (GRU) RNN to a dynamic-length input sequence.

    The first input should be a batch of sequences with variable lengths defined
    through the ``tensor.PackedSequence`` class.
    A ``tensor.PackedSequence`` class can be constructed by
    calling the following functions in sequence: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_GRU is also a ``tensor.PackedSequence`` class,
    which can be decompressed into a regular QTensor using ``tensor.pad_pack_sequence``.

    For each element in the input sequence, each layer computes the following formulas:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}


    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of recurrent layers. Default: 1
    :param batch_first: If True, the input shape is provided as [batch size, sequence length, feature dimension]. If False, the input shape is provided as [sequence length, batch size, feature dimension]. Default: True.
    :param use_bias: If False, the layer does not use the bias weights b_ih and b_hh. Default: True.
    :param bidirectional: If true, becomes a bidirectional GRU. Default: False.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: A Dynamic_GRU class instance

    Example::

        from pyvqnet.nn import Dynamic_GRU
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_GRU(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)
        print(seq_unpacked)
        print(lens_unpacked)
        # [
        # [[-0.3918380, 0.0056273, 0.9018179, 0.9006662],
        #  [-0.3715909, 0.0307644, 0.9756137, 0.9705784],
        #  [-0.3917399, 0.0057521, 0.9507942, 0.9456232]],
        # [[-0.6348240, -0.0603764, 0.9014163, 0.8903066],
        #  [0., 0., 0., 0.],
        #  [-0.6333261, -0.0592172, 0.9660671, 0.9580816]],
        # [[-0.4571511, 0.0210018, 0.9151242, 0.9011748],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]],
        # [[-0.3585358, 0.0918219, 0.9496037, 0.9391552],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]]
        # ]
        # [4 1 2]

Dynamic_RNN
=================================

.. py:class:: pyvqnet.nn.rnn.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Apply a Recurrent Neural Network (RNN) to a dynamic-length input sequence.

    The first input should be a batch of sequences with variable lengths defined
    through the ``tensor.PackedSequence`` class.
    A ``tensor.PackedSequence`` class can be constructed by
    calling the following functions in sequence: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_RNN is also a ``tensor.PackedSequence`` class,
    which can be decompressed into a regular QTensor using ``tensor.pad_pack_sequence``.

    Recurrent neural network (RNN) module using :math:`\tanh` or :math:`\text{ReLU}` as the activation function. Supports bidirectional and multi-layer configurations.
    The computation formula for a single-layer unidirectional RNN is as follows:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    If :attr:`nonlinearity` is ``'relu'``, :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked RNN layers. Default: 1.
    :param nonlinearity: Nonlinear activation function. Default: ``'tanh'``.
    :param batch_first: If True, the input shape is [batch size, sequence length, feature dimension];
     if False, the input shape is [sequence length, batch size, feature dimension]. Default: True.
    :param use_bias: If False, the module does not use bias terms. Default: True.
    :param bidirectional: If True, becomes a bidirectional RNN. Default: False.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: Dynamic_RNN instance

    Example::

        from pyvqnet.nn import Dynamic_RNN
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_RNN(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True,
                        nonlinearity='relu')

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)
        print(seq_unpacked)
        print(lens_unpacked)

        # [
        # [[1.2980951, 0., 0., 0.],
        #  [1.5040692, 0., 0., 0.],
        #  [1.4927036, 0., 0., 0.1065927]],
        # [[2.6561704, 0., 0., 0.2532321],
        #  [0., 0., 0., 0.],
        #  [3.1472805, 0., 0., 0.]],
        # [[5.1231661, 0., 0., 0.7596353],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]],
        # [[8.4954977, 0., 0., 0.8191229],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]]
        # ]
        # [4 1 2]



Dynamic_LSTM
=================================

.. py:class:: pyvqnet.nn.lstm.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Apply a Long Short-Term Memory (LSTM) RNN to a dynamic-length input sequence.

    The first input should be a batch of sequences with variable lengths defined
    through the ``tensor.PackedSequence`` class.
    A ``tensor.PackedSequence`` class can be constructed by
    calling the following functions in sequence: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_LSTM is also a ``tensor.PackedSequence`` class,
    which can be decompressed into a regular QTensor using ``tensor.pad_pack_sequence``.

    Recurrent neural network (RNN) module using :math:`\tanh` or :math:`\text{ReLU}` as the activation function. Supports bidirectional and multi-layer configurations.
    The computation formula for a single-layer unidirectional RNN is as follows:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked LSTM layers. Default: 1.
    :param batch_first: If True, the input shape is [batch size, sequence length, feature dimension];
     if False, the input shape is [sequence length, batch size, feature dimension]. Default: True.
    :param use_bias: If False, the module does not use bias terms. Default: True.
    :param bidirectional: If True, becomes a bidirectional LSTM. Default: False.
    :param dtype: Data type of the parameters. Default: None, which uses the default data type kfloat32 (32-bit floating point).
    :param name: Name of this module. Default: "".

    :return: Dynamic_LSTM instance

    Example::

        from pyvqnet.nn import Dynamic_LSTM
        from pyvqnet.tensor import tensor

        input_size = 2
        hidden_size = 2
        ml = 2
        seq_len = [3, 4, 1]
        batch_size = 3
        rnn2 = Dynamic_LSTM(input_size,
                            hidden_size=hidden_size,
                            num_layers=ml,
                            batch_first=False,
                            bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])
        c0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, (hn, cn) = rnn2(input, (h0, c0))

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

        print(seq_unpacked)
        print(lens_unpacked)

        # [
        # [[0.2038177, 0.1139005, 0.2312966, -0.1140076],
        #  [0.1992285, 0.1221137, 0.2277344, -0.3147154],
        #  [0.2293468, 0.0681745, 0.2426863, 0.2572871]],
        # [[0.1398094, -0.0150359, 0.2513067, 0.0783743],
        #  [0.1328388, -0.0031956, 0.2324090, -0.1962151],
        #  [0., 0., 0., 0.]],
        # [[0.0898260, -0.0706460, 0.2396922, 0.2323916],
        #  [0.0817787, -0.0449937, 0.2388873, -0.0000469],
        #  [0., 0., 0., 0.]],
        # [[0., 0., 0., 0.],
        #  [0.0532839, -0.0870574, 0.2397324, 0.2103822],
        #  [0., 0., 0., 0.]]
        # ]
        # [3 4 1]



Interpolate
=================================
.. py:class:: pyvqnet.nn.Interpolate(size = None, scale_factor = None, mode = "nearest", align_corners = None,  recompute_scale_factor = None, name = "")

    Downsample or upsample the input.

    Currently only 4-D input data is supported.

    Input dimensions are interpreted as `B x C x H x W`.

    The available `mode` options are ``nearest``, ``bilinear``, and ``bicubic``.

    :param size: Output size. Default: None.
    :param scale_factor: Scale factor. Default: None.
    :param mode: Algorithm used for upsampling: ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: From a geometric perspective, input and output pixels are treated as squares rather than points.
            If set to `true`, the input and output tensors are aligned by the centers of their corner pixels, which preserves the values of the corner pixels.
            If set to `false`, the input and output tensors are aligned by the corners of their corner pixels, and the values of the corner pixels are preserved. Interpolation fills out-of-boundary values with edge values,
            making this operation independent of the input size.
            When ``scale_factor`` is kept unchanged. This only has an effect when ``mode`` is ``bilinear``.
    :param recompute_scale_factor: Recompute the scale factor for use in the interpolation computation. When ``scale_factor`` is passed as a parameter, it is used to compute the output size.
    :param name: Module name.

    Example::

        from pyvqnet.nn import Interpolate
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.utils.set_random_seed(1)

        import numpy as np
        np.random.seed(0)

        np_ = np.random.randn(36).reshape((1, 1, 6, 6)).astype(np.float32)
        mode_ = "bilinear"
        size_ = 3

        class Model(pyvqnet.nn.Module):

            def __init__(self):
                super().__init__()
                self.inter = Interpolate(size = size_, mode=mode_)
                self.ln = pyvqnet.nn.Linear(9, 1)

            def forward(self, x):
                x = self.inter(x).reshape((1,-1))
                x = self.ln(x)
                return 2 * x

        input_vqnet = tensor.QTensor(np_,  dtype=pyvqnet.kfloat32, requires_grad=True)
        loss_pyvqnet = pyvqnet.nn.MeanSquaredError()
        model_vqnet = Model()
        output_vqnet = model_vqnet(input_vqnet)
        l = loss_pyvqnet(tensor.QTensor([[1.0]]), output_vqnet)
        l.backward()
        print(model_vqnet.parameters()[0].grad)


fuse_module
=================================
.. py:class:: pyvqnet.nn.fuse_module(model)

    Fuses corresponding adjacent modules of a model during inference into one module, reducing computation during inference and increasing inference speed.

    The currently supported module sequences are:

    conv, bn

    linear, bn

    Other sequences remain unchanged. For these sequences, the first module in the list is replaced with the fused module and the others are replaced with ``Identity``.

    :param input: Model containing the modules to fuse.

    :return: Model with fused modules.

    Example::

        from pyvqnet import tensor,kfloat32
        from pyvqnet.nn import Linear
        from pyvqnet.nn import Module, BatchNorm1d, BatchNorm2d, Conv1D, Conv2D

        from pyvqnet.optim import Adam
        from pyvqnet.nn import Module,BinaryCrossEntropy, Sigmoid
        from pyvqnet.data import data_generator
        import numpy as np
        from pyvqnet.tensor import QTensor

        from time import time
        from pyvqnet.utils import set_random_seed
        from pyvqnet.nn import fuse_module

        def get_accuracy(result, label):
            result = (result > 0.5).astype(4)
            score = tensor.sums(result == label)
            return score.item()

        class Model(Module):
            def __init__(self):

                super(Model, self).__init__()

                self.conv1 = Conv2D(1,2,1)
                self.ban = BatchNorm2d(2)

                self.conv2 = Conv2D(2,1,1)
                self.li1 = Linear(64,1)
                self.ac = Sigmoid()

            def forward(self, x):
                x = self.conv1(x)
                x = self.ban(x)
                x = self.conv2(x).reshape([-1,64])
                x = self.li1(x)
                x = self.ac(x)

                return x
        X_train = np.random.randn(80, 1, 8, 8)
        y_train = np.random.choice([0,1], size=(80))

        model = Model().toGPU()
        optimizer = Adam(model.parameters(), lr = 0.001)
        batch_size = 20
        epoch = 80
        loss = BinaryCrossEntropy()
        print("start training..............")
        model.train()

        loss_history = []
        accuracy_history = []
        time2 = time()

        for i in range(epoch):
            count = 0
            sum_loss = 0
            accuary = 0
            t = 0
            for data, label in data_generator(X_train, y_train, batch_size, False):
                optimizer.zero_grad()
                data, label = QTensor(data,requires_grad=True).toGPU(), QTensor(label,
                                                    dtype=kfloat32,
                                                    requires_grad=False).toGPU()

                result = model(data)

                loss_b = loss(label.reshape([-1, 1]), result)

                loss_b.backward()
                optimizer._step()

                sum_loss += loss_b.item()
                count += batch_size
                accuary += get_accuracy(result, label.reshape([-1,1]))
                t = t + 1

            loss_history.append(sum_loss/count)
            accuracy_history.append(accuary/count)
            print(
                f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}"
            )
        print(f"run time {time() - time2}")


        model.eval()

        input = tensor.randn((20, 1, 8, 8)).toGPU()
        print(list(model.named_children()))
        time_a = time()
        a = model(input)
        print(f"fuse before {time() - time_a}")
        fuse_module(model)
        model.toGPU()
        print(list(model.named_children()))
        time_b = time()
        b = model(input)
        print(f"fuse after {time() - time_b}")

        print(tensor.max(tensor.abs(a - b)).item())


SDPA
=================================
.. py:class:: pyvqnet.transformer.SDPA(attn_mask=None,dropout_p=0.,scale=None,is_causal=False)

    Class that computes scaled dot-product attention over query, key, and value tensors.

    :param attn_mask: Attention mask; its shape must be broadcastable to the shape of the attention weights.
    :param dropout_p: Dropout probability. Applied if greater than 0.0.
    :param scale: Scale factor applied before softmax.
    :param is_causal: If "true", a top-left causal attention mask is assumed. If both attn_mask and is_causal are set, an error is raised.
    :return: An SDPA class instance

    Example::

        from pyvqnet.transformer import SDPA
        from pyvqnet import tensor
        model = SDPA(tensor.QTensor([1.])).toGPU()

    .. py:method:: forward(query,key,value)

        Perform the forward computation

        :param query: Query input QTensor.
        :param key: Key input QTensor.
        :param value: Value input QTensor.
        :return: The QTensor returned by the SDPA computation.

        Example::

            from pyvqnet.transformer import SDPA
            from pyvqnet import tensor
            import pyvqnet
            from time import time
            import pyvqnet.nn as nn
            import numpy as np

            np.random.seed(42)

            model = SDPA(tensor.QTensor([1.])).toGPU()

            query_np = np.random.randn(3, 3, 3, 5).astype(np.float32)
            key_np = np.random.randn(3, 3, 3, 5).astype(np.float32)
            value_np = np.random.randn(3, 3, 3, 5).astype(np.float32)

            query_p = tensor.QTensor(query_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()
            key_p = tensor.QTensor(key_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()
            value_p = tensor.QTensor(value_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()

            out_sdpa = model(query_p, key_p, value_p)

            out_sdpa.backward()


Loss Function Layers
*********************************************************

The following are loss layers commonly used in neural networks.

    .. note::

            Note that, unlike frameworks such as PyTorch, in the forward function of the following loss functions, the first argument is the label and the second argument is the prediction.

MeanSquaredError
=================================

.. py:class:: pyvqnet.nn.MeanSquaredError(name="")

    Compute the mean squared error between the input :math:`x` and the target value :math:`y`.

    The squared error can be described by the following function:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` and :math:`y` are QTensors of arbitrary shape. The mean squared error over the total :math:`n` elements is computed as follows.

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)

    :param name: Name of this module. Default: "".
    :return: A mean squared error instance.

    Arguments required by the mean squared error forward function:

        x: :math:`(N, *)` predicted values, where :math:`*` denotes any dimensions.

        y: :math:`(N, *)` target values, a QTensor with the same dimensions as the input.


    .. note::

            Note that, unlike frameworks such as PyTorch, in the forward function of the following MeanSquaredError function, the first argument is the target value and the second argument is the prediction.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        from pyvqnet.nn import MeanSquaredError
        y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kfloat64)
        x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    requires_grad=True,
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)
        # [0.0115000]



BinaryCrossEntropy
=================================

.. py:class:: pyvqnet.nn.BinaryCrossEntropy(name="")

    Measure the average binary cross entropy loss between the target and the input.

    The unaveraged binary cross entropy is given by:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    If :math:`N` is the batch size, the average binary cross entropy is:

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :param name: Name of this module. Default: "".
    :return: An average binary cross entropy instance.

    Arguments required by the average binary cross entropy forward function:

        x: :math:`(N, *)` predicted values, where :math:`*` denotes any dimensions.

        y: :math:`(N, *)` target values, a QTensor with the same dimensions as the input.

    .. note::

            Note that, unlike frameworks such as PyTorch, in the forward function of the BinaryCrossEntropy function, the first argument is the target value and the second argument is the prediction.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import BinaryCrossEntropy
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0.0, 1.0, 0], [0.0, 0, 1]], requires_grad=False)

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)
        # [0.6364825]


CategoricalCrossEntropy
=================================

.. py:class:: pyvqnet.nn.CategoricalCrossEntropy(name="")

    This loss function computes the average categorical cross entropy by combining LogSoftmax and NLLLoss.

    The loss is computed as follows, where class is the class label corresponding to the target value:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Name of this module. Default: "".
    :return: An average categorical cross entropy instance.

    Arguments required by the error forward function:

        x: :math:`(N, *)` predicted values, where :math:`*` denotes any dimensions.

        y: :math:`(N, *)` target values, a QTensor with the same dimensions as the input. Must be 64-bit integers, kint64.

    .. note::

            Note that, unlike frameworks such as PyTorch, in the forward function of the CategoricalCrossEntropy function, the first argument is the target value and the second argument is the prediction.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn import CategoricalCrossEntropy
        x = QTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], requires_grad=True,dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False,dtype=kint64)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)
        # [3.7852428]

SoftmaxCrossEntropy
=================================

.. py:class:: pyvqnet.nn.SoftmaxCrossEntropy(name:str="", smoothing:float=0.0)

    This loss function computes the average categorical cross entropy by combining LogSoftmax and NLLLoss, with higher numerical stability.

    The loss is computed as follows, where class is the class label corresponding to the target value:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Name of this module. Default: "".
    :param smoothing: Label smoothing factor in the range [0,1). Default: 0.0 (smoothing disabled). When enabled, target labels are smoothed to improve numerical stability and generalization.
    :return: A Softmax cross entropy loss function instance

    Arguments required by the error forward function:

        x: :math:`(N, *)` predicted values, where :math:`*` denotes any dimensions.

        y: :math:`(N, *)` target values, a QTensor with the same dimensions as the input. Must be 64-bit integers, kint64.

    .. note::

            Note that, unlike frameworks such as PyTorch, in the forward function of the SoftmaxCrossEntropy function, the first argument is the target value and the second argument is the prediction.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32, kint64
        from pyvqnet.nn import SoftmaxCrossEntropy
        x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                    requires_grad=True,
                    dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kint64)
        loss_result = SoftmaxCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)

        # [3.7852478]

NLL_Loss
=================================

.. py:class:: pyvqnet.nn.NLL_Loss(name="")

    Average negative log likelihood loss. Useful for classification problems with C classes.

    `x` is the likelihood in probability form produced by the model. Its size can be :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)`. `y` is the ground truth expected by the loss function, containing class indices in :math:`[0, C-1]`.

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: Name of this module. Default: "".
    :return: An NLL_Loss loss function instance

    Arguments required by the error forward function:

        x: :math:`(N, *)` output predictions of the loss function, which can be multi-dimensional.

        y: :math:`(N, *)` target values of the loss function. Must be 64-bit integers, kint64.

    .. note::

            Note that, unlike frameworks such as PyTorch, in the forward function of the NLL_Loss function, the first argument is the target value and the second argument is the prediction.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn import NLL_Loss

        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)
        #[-0.6187226]

CrossEntropyLoss
=================================

.. py:class:: pyvqnet.nn.CrossEntropyLoss(name="")

    This function computes the loss of LogSoftmax and NLL_Loss together.

    `x` contains unnormalized outputs. Its size can be :math:`(C)`, 2-D :math:`(N, C)`, or multi-dimensional :math:`(N, C, d_1, d_2, ..., d_K)`.

    The loss function is given by the following formula, where class is the class label corresponding to the target value:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Name of this module. Default: "".
    :return: A CrossEntropyLoss loss function instance

    Arguments required by the error forward function:

        x: :math:`(N, *)` output of the loss function, which can be multi-dimensional.

        y: :math:`(N, *)` ground truth expected by the loss function. Must be 64-bit integers, kint64.

    .. note::

            Note that, unlike frameworks such as PyTorch, in the forward function of the CrossEntropyLoss function, the first argument is the target value and the second argument is the prediction.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn import CrossEntropyLoss
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)

        #[1.1508200]

``pyvqnet.nn.functional`` and ``pyvqnet.nn._sampling`` modules provide a series of low-level functional operators that can be used to build custom layers or implement sampling logic.

rope (Rotary Position Embedding)
=================================

.. py:function:: pyvqnet.nn.functional.rope(q, k, cos, sin)

    Forward computation that applies rotary position embedding (RoPE) to query and key tensors.

    :param q: Query tensor.
    :param k: Key tensor.
    :param cos: Precomputed cosine cache of shape ``(seq_len, head_dim/2)``.
    :param sin: Precomputed sine cache of shape ``(seq_len, head_dim/2)``.
    :return: The (q, k) tensors after applying RoPE.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, tensor, DEV_GPU_0, kfloat32, cos, sin
        from pyvqnet.nn.functional import rope
        import numpy as np

        # Precompute cos/sin cache
        head_dim = 4
        seq_len = 4
        half = head_dim // 2
        i = tensor.arange(0, half, dtype=kfloat32)
        theta = 1.0 / (10000.0 ** (2.0 * i / head_dim))
        m = tensor.arange(0, seq_len, dtype=kfloat32).reshape([-1, 1])
        cos_t = cos(m * theta)
        sin_t = sin(m * theta)
        print(cos_t)

        # [[ 1.       , 1.       ],
        #  [ 0.5403023, 0.99995  ],
        #  [-0.4161468, 0.9998   ],
        #  [-0.9899925, 0.99955  ]]

        # Move to GPU and apply
        q = QTensor(np.arange(1.0, 17.0, dtype=np.float32).reshape([1, 1, 4, 4]), device=DEV_GPU_0)
        k = QTensor(np.arange(17.0, 33.0, dtype=np.float32).reshape([1, 1, 4, 4]), device=DEV_GPU_0)
        cos_gpu = QTensor(cos_t.to_numpy(), device=DEV_GPU_0)
        sin_gpu = QTensor(sin_t.to_numpy(), device=DEV_GPU_0)
        out_q, out_k = rope(q, k, cos_gpu, sin_gpu)
        print(out_q)

        # [[[[  1.       ,  2.       ,  3.       ,  4.       ],
        #    [ -3.1887856,  5.919701 ,  7.9894705,  8.059599 ],
        #    [-13.747593 ,  9.758017 ,  3.6060615, 12.197587 ],
        #    [-14.986703 , 13.513773 ,-13.015327 , 16.412737 ]]]]

swiglu (Gated Linear Unit)
-----------------------------

.. py:function:: pyvqnet.nn.functional.swiglu(gate, up)

    SwiGLU functional operator: ``SiLU(gate) * up`` .

    :param gate: Gate tensor.
    :param up: Up-projection tensor.
    :return: The computed tensor.

    Example::

        from pyvqnet.tensor import QTensor, kfloat32
        from pyvqnet.nn.functional import swiglu

        gate = QTensor([1.0, 2.0, 3.0, 4.0], dtype=kfloat32)
        up = QTensor([0.5, 1.0, 1.5, 2.0], dtype=kfloat32)
        out = swiglu(gate, up)
        print(out)

        # [0.3655293,1.761594 ,4.286584 ,7.85611  ]

scaled_softmax (Scaled Softmax)
=================================

.. py:function:: pyvqnet.nn.functional.scaled_softmax(input, scale: float)

    Compute the scaled softmax of the input.

    :param input: Input tensor.
    :param scale: Scale factor.
    :return: The scaled softmax result.

    Example::

        from pyvqnet.tensor import QTensor, kfloat32
        from pyvqnet.nn.functional import scaled_softmax

        x = QTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=kfloat32)
        out = scaled_softmax(x, 0.5)
        print(out)

        # [[0.1863237,0.3071959,0.5064804],
        #  [0.1863237,0.3071959,0.5064804]]

scaled_masked_softmax (Masked Scaled Softmax)
=============================================

.. py:function:: pyvqnet.nn.functional.scaled_masked_softmax(input, mask, scale: float)

    Compute the scaled softmax with a mask.

    :param input: Input tensor.
    :param mask: Mask tensor with the same shape as input. 0 = keep, 1 = mask (the corresponding position is set to -inf before softmax, so the result is 0).
    :param scale: Scale factor.
    :return: The scaled masked softmax result.

    Example::

        from pyvqnet.tensor import QTensor, kfloat32
        from pyvqnet.nn.functional import scaled_masked_softmax

        x = QTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=kfloat32)
        mask = QTensor([[0, 0, 1], [1, 0, 0]], dtype=kfloat32)  # 1=masked
        out = scaled_masked_softmax(x, mask, 1.0)
        print(out)

        # [[0.2689414,0.7310586,0.       ],
        #  [0.       ,0.2689414,0.7310586]]

scaled_upper_triang_masked_softmax (Upper Triangular Masked Scaled Softmax)
===========================================================================

.. py:function:: pyvqnet.nn.functional.scaled_upper_triang_masked_softmax(input, scale: float)

    Compute the scaled softmax with the upper triangular part of the input masked.

    :param input: Input tensor.
    :param scale: Scale factor.
    :return: The computation result.

    Example::

        from pyvqnet.tensor import QTensor, kfloat32
        from pyvqnet.nn.functional import scaled_upper_triang_masked_softmax

        x = QTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=kfloat32)
        out = scaled_upper_triang_masked_softmax(x, 1.0)
        print(out)

        # [[1.       ,0.       ,0.       ],
        #  [0.2689414,0.7310586,0.       ]]

fused_moe (Fused Mixture of Experts)
=====================================

.. py:function:: pyvqnet.nn.functional.fused_moe(input, fc1_weight, fc2_weight, probs, indices, num_experts)

    Fused mixture of experts (MoE) forward computation.

    :param input: Input tensor.
    :param fc1_weight: First-layer expert weights.
    :param fc2_weight: Second-layer expert weights.
    :param probs: Expert routing probabilities.
    :param indices: Expert indices.
    :param num_experts: Number of experts.
    :return: Output tensor of the MoE computation.

    .. note::

            This function only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
        from pyvqnet.dtype import kint32
        from pyvqnet.nn.functional import fused_moe
        import numpy as np

        np.random.seed(42)
        E = 2
        x = QTensor(np.random.randn(2, 4).astype(np.float32), device=DEV_GPU_0)
        w1 = QTensor(np.random.randn(E, 4, 8).astype(np.float32), device=DEV_GPU_0)
        w2 = QTensor(np.random.randn(E, 4, 4).astype(np.float32), device=DEV_GPU_0)
        probs = QTensor([[0.7], [0.8]], dtype=kfloat32, device=DEV_GPU_0)
        idx = QTensor([[0], [1]], dtype=kint32, device=DEV_GPU_0)
        y = fused_moe(x, w1, w2, probs, idx, E)
        print(y)

        # [[ 0.4034894,-0.2737532,-1.0371764,-1.3686538],
        #  [ 3.2209034, 0.9456251, 0.8900909, 2.3802228]]

Top-K and Top-P Sampling from Logits
=====================================

.. py:function:: pyvqnet.nn._sampling.top_k_top_p_sampling_from_logits(logits, temperature: Union[QTensor, float] = 1.0, top_k: Union[QTensor, int] = 0, top_p: Union[QTensor, float] = 1.0, deterministic: bool = True)

    Joint top-k and top-p sampling based on logits.

    :param logits: Input logits tensor.
    :param temperature: Temperature coefficient. Default: 1.0.
    :param top_k: Keep the k tokens with the highest probability; 0 means no limit. Default: 0.
    :param top_p: Cumulative probability threshold. Default: 1.0 (no limit).
    :param deterministic: Whether to use deterministic sampling. Default: True.
    :return: A (tokens, valid_mask) tuple. tokens contains the sampled token indices, and valid_mask indicates whether sampling is valid for each sample.

    .. note::

            This function only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
        from pyvqnet.nn._sampling import top_k_top_p_sampling_from_logits

        logits = QTensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                         dtype=kfloat32, device=DEV_GPU_0)
        tokens, valid = top_k_top_p_sampling_from_logits(logits, 1.0, 0, 1.0, True)
        print(tokens)
        # [3, 0]
        print(valid)
        # [ True, True]

Joint Top-K and Top-P Sampling from Probabilities
---------------------------------------------------------------

.. py:function:: pyvqnet.nn._sampling.top_k_top_p_sampling_from_probs(probs, top_k: Union[QTensor, int] = 0, top_p: Union[QTensor, float] = 1.0, deterministic: bool = True)

    Joint top-k and top-p sampling based on a probability distribution.

    :param probs: Input probability tensor.
    :param top_k: Keep the k tokens with the highest probability; 0 means no limit. Default: 0.
    :param top_p: Cumulative probability threshold. Default: 1.0.
    :param deterministic: Whether to use deterministic sampling. Default: True.
    :return: A (tokens, valid_mask) tuple.

    .. note::

            This function only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
        from pyvqnet.nn._sampling import top_k_top_p_sampling_from_probs
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                        dtype=kfloat32, device=DEV_GPU_0)
        tokens, valid = top_k_top_p_sampling_from_probs(probs, 0, 1.0, True)
        print(tokens)
        # [3,0]
        print(valid)
        # [ True, True]

Top-K Sampling from Probabilities
---------------------------------------------

.. py:function:: pyvqnet.nn._sampling.top_k_sampling_from_probs(probs, top_k: Union[QTensor, int] = 0, deterministic: bool = True)

    Top-k sampling based on a probability distribution.

    :param probs: Input probability tensor.
    :param top_k: Keep the k tokens with the highest probability; 0 means no limit. Default: 0.
    :param deterministic: Whether to use deterministic sampling. Default: True.
    :return: A (tokens, valid_mask) tuple.

    .. note::

            This function only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
        from pyvqnet.nn._sampling import top_k_sampling_from_probs
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                        dtype=kfloat32, device=DEV_GPU_0)
        tokens, valid = top_k_sampling_from_probs(probs, 0, True)
        print(tokens)
        # [3,0]
        print(valid)
        # [ True, True]

Top-P Sampling from Probabilities
---------------------------------------------

.. py:function:: pyvqnet.nn._sampling.top_p_sampling_from_probs(probs, top_p: Union[QTensor, float] = 1.0, deterministic: bool = True)

    Top-p sampling based on a probability distribution.

    :param probs: Input probability tensor.
    :param top_p: Cumulative probability threshold. Default: 1.0.
    :param deterministic: Whether to use deterministic sampling. Default: True.
    :return: A (tokens, valid_mask) tuple.

    .. note::

            This function only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
        from pyvqnet.nn._sampling import top_p_sampling_from_probs
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                        dtype=kfloat32, device=DEV_GPU_0)
        tokens, valid = top_p_sampling_from_probs(probs, 1.0, True)
        print(tokens)
        # [3,0]
        print(valid)
        # [ True, True]

Min-P Sampling from Probabilities
---------------------------------------------

.. py:function:: pyvqnet.nn._sampling.min_p_sampling_from_probs(probs, min_p: Union[QTensor, float] = 0.0, deterministic: bool = True)

    Probability sampling based on the min-p strategy.

    :param probs: Input probability tensor.
    :param min_p: Minimum probability threshold. Default: 0.0.
    :param deterministic: Whether to use deterministic sampling. Default: True.
    :return: A (tokens, valid_mask) tuple.

    .. note::

            This function only runs on the GPU. See the "Model Training on GPU" section above for moving the model and data to the GPU.

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor, DEV_GPU_0, kfloat32
        from pyvqnet.nn._sampling import min_p_sampling_from_probs
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        probs = QTensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                        dtype=kfloat32, device=DEV_GPU_0)
        tokens, valid = min_p_sampling_from_probs(probs, 0.0, True)
        print(tokens)
        # [3,0]
        print(valid)
        # [ True, True]




Activation Functions
*********************************************************

Activation
=================================
.. py:class:: pyvqnet.nn.activation.Activation(name: str = "")

    Base class for activations. Specific activation functions inherit from this class.

    :param name: Activation module name. Default: "".

Sigmoid
=================================
.. py:class:: pyvqnet.nn.Sigmoid(name:str="")

    Sigmoid activation function layer.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    :param name: Name of the activation function layer. Default: "".

    :return: A Sigmoid activation function layer instance.

    Example::

        from pyvqnet.nn import Sigmoid
        from pyvqnet.tensor import QTensor
        layer = Sigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.7310586, 0.8807970, 0.9525741, 0.9820138]


Softplus
=================================
.. py:class:: pyvqnet.nn.Softplus(name:str="")

    Softplus activation function layer.

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    :param name: Name of the activation function layer. Default: "".

    :return: A Softplus activation function layer instance.

    Example::

        from pyvqnet.nn import Softplus
        from pyvqnet.tensor import QTensor
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [1.3132616, 2.1269281, 3.0485873, 4.0181499]


Softsign
=================================
.. py:class:: pyvqnet.nn.Softsign(name:str="")

    Softsign activation function layer.

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    :param name: Name of the activation function layer. Default: "".

    :return: A Softsign activation function layer instance.

    Example::

        from pyvqnet.nn import Softsign
        from pyvqnet.tensor import QTensor
        layer = Softsign()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.5000000, 0.6666667, 0.7500000, 0.8000000]



Softmax
=================================
.. py:class:: pyvqnet.nn.Softmax(dim:int = -1,name:str="")

    Softmax activation function layer.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    :param dim: Dimension along which to compute (the last axis is -1). Default = -1.
    :param name: Name of the activation function layer. Default: "".

    :return: A Softmax activation function layer instance.

    Example::

        from pyvqnet.nn import Softmax
        from pyvqnet.tensor import QTensor
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.0320586, 0.0871443, 0.2368828, 0.6439142]


HardSigmoid
=================================
.. py:class:: pyvqnet.nn.HardSigmoid(name:str="")

    HardSigmoid activation function layer.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param name: Name of the activation function layer. Default: "".

    :return: A HardSigmoid activation function layer instance.

    Example::

        from pyvqnet.nn import HardSigmoid
        from pyvqnet.tensor import QTensor
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.6666667, 0.8333334, 1., 1.]


SwiGLU
=================================

.. py:class:: pyvqnet.nn.SwiGLU(name:str="")

    SwiGLU activation function layer, a variant of the SwiGLU gated linear unit: ``SiLU(gate) * up`` .

    :param name: Name of the activation function layer. Default: "".
    :return: A SwiGLU activation function layer instance.

    Example::

        from pyvqnet.nn import SwiGLU
        from pyvqnet.tensor import QTensor, kfloat32
        layer = SwiGLU()
        gate = QTensor([1.0, 2.0, 3.0, 4.0], dtype=kfloat32)
        up = QTensor([0.5, 1.0, 1.5, 2.0], dtype=kfloat32)
        y = layer(gate, up)
        print(y)

        # [0.3655293,1.761594 ,4.286584 ,7.85611  ]

ReLu
=================================
.. py:class:: pyvqnet.nn.ReLu(name:str="")

    ReLu rectified linear unit activation function layer.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    :param name: Name of the activation function layer. Default: "".

    :return: A ReLu activation function layer instance.

    Example::

        from pyvqnet.nn import ReLu
        from pyvqnet.tensor import QTensor
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [0., 2., 0., 4.]



LeakyReLu
=================================
.. py:class:: pyvqnet.nn.LeakyReLu(alpha:float=0.01,name:str="")

    LeakyReLu leaky rectified linear unit activation function layer.

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param alpha: LeakyRelu coefficient. Default: 0.01.
    :param name: Name of the activation function layer. Default: "".

    :return: A LeakyReLu activation function layer instance.

    Example::

        from pyvqnet.nn import LeakyReLu
        from pyvqnet.tensor import QTensor
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.0100000, 2., -0.0300000, 4.]



Gelu
=================================
.. py:class:: pyvqnet.nn.Gelu(approximate="tanh", name="")

    Apply the Gaussian error linear unit function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    When the approximation parameter is 'tanh', GELU is estimated by:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    :param approximate: Approximation method. Default: "tanh".
    :param name: Name of the activation function layer. Default: "".

    :return: Gelu activation function layer instance.

    Example::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn import Gelu
        qa = randu([5,4])
        qb = Gelu()(qa)
        print(qb)
        # [[0.0292515,0.0668998,0.4036024,0.8369502],
        #  [0.1929213,0.1981275,0.2358531,0.7790835],
        #  [0.1754935,0.6204091,0.2354677,0.2409406],
        #  [0.4238827,0.804715 ,0.1633414,0.2853   ],
        #  [0.1959854,0.590143 ,0.553995 ,0.0008423]]


ELU
=================================
.. py:class:: pyvqnet.nn.ELU(alpha:float=1,name:str="")

    ELU exponential linear unit activation function layer.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param alpha: ELU coefficient. Default: 1.
    :param name: Name of the activation function layer. Default: "".

    :return: ELU activation function layer instance.

    Example::

        from pyvqnet.nn import ELU
        from pyvqnet.tensor import QTensor
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.6321205, 2., -0.9502130, 4.]



Tanh
=================================
.. py:class:: pyvqnet.nn.Tanh(name:str="")

    Tanh hyperbolic tangent activation function.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param name: Name of the activation function layer. Default: "".

    :return: Tanh activation function layer instance.

    Example::

        from pyvqnet.nn import Tanh
        from pyvqnet.tensor import QTensor
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.7615942, 0.9640276, -0.9950548, 0.9993293]





Optimizer Modules
*********************************************************



.. _Optimizer:

Optimizer
=================================
.. py:class:: pyvqnet.optim.optimizer.Optimizer(params, lr=0.1, defaults=None)

    Base class for all optimizers.

    :param params: Model parameters to optimize.
    :param lr: Learning rate. Default: 0.1.
    :param defaults: Default hyperparameters of the optimizer. Default: None.

Adadelta
=================================
.. py:class:: pyvqnet.optim.adadelta.Adadelta( params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA: An Adaptive Learning Rate Method.

    Reference: https://arxiv.org/abs/1212.5701.

    .. math::

        E(g_t^2) &= \beta * E(g_{t-1}^2) + (1-\beta) * g^2\\
        Square\_avg &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
        E(dx_t^2) &= \beta * E(dx_{t-1}^2) + (1-\beta) * (-g*square\_avg)^2 \\
        param\_new &= param - lr * Square\_avg

    :param params: Model parameters to optimize.
    :param lr: Learning rate (default: 0.01).
    :param beta: Coefficient used to compute the running average of squared gradients (default: 0.99).
    :param epsilon: Constant added to the denominator to improve numerical stability (default: 1e-8).

    :return: An Adadelta optimizer.

    Example::

        import numpy as np
        from pyvqnet.optim import adadelta
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1, 2, 3, 4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
        params = [param]
        opti = adadelta.Adadelta(params)

        for i in range(1, 3):
            opti._step()
            print(param)

        # [
        # [[[0., 0.9999900, 1.9999900, 2.9999900],
        #  [3.9999900, 4.9999900, 5.9999900, 6.9999900],
        #  [7.9999900, 8.9999905, 9.9999905, 10.9999905]],
        # [[11.9999905, 12.9999905, 13.9999905, 14.9999905],
        #  [15.9999905, 16.9999905, 17.9999905, 18.9999905],
        #  [19.9999905, 20.9999905, 21.9999905, 22.9999905]]]
        # ]

        # [
        # [[[0., 0.9999800, 1.9999800, 2.9999800],
        #  [3.9999800, 4.9999800, 5.9999800, 6.9999800],
        #  [7.9999800, 8.9999800, 9.9999800, 10.9999800]],
        # [[11.9999800, 12.9999800, 13.9999800, 14.9999800],
        #  [15.9999800, 16.9999809, 17.9999809, 18.9999809],
        #  [19.9999809, 20.9999809, 21.9999809, 22.9999809]]]
        # ]

Adagrad
=================================
.. py:class:: pyvqnet.optim.adagrad.Adagrad( params, lr=0.01, epsilon=1e-8)

    Adagrad adaptive gradient optimizer.

    Reference: https://databricks.com/glossary/adagrad.

    .. math::
        \begin{aligned}
        moment\_new &= moment + g * g\\param\_new
        &= param - \frac{lr * g}{\sqrt{moment\_new} + \epsilon}
        \end{aligned}

    :param params: Model parameters to optimize.
    :param lr: Learning rate (default: 0.01).
    :param epsilon: Constant added to the denominator to improve numerical stability (default: 1e-8).
    :return: An Adagrad optimizer.

    Example::

        import numpy as np
        from pyvqnet.optim import adagrad
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adagrad.Adagrad(params)

        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0., 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0., 0.9829289, 1.9829290, 2.9829290],
        #  [3.9829290, 4.9829288, 5.9829288, 6.9829288],
        #  [7.9829288, 8.9829283, 9.9829283, 10.9829283]],
        # [[11.9829283, 12.9829283, 13.9829283, 14.9829283],
        #  [15.9829283, 16.9829292, 17.9829292, 18.9829292],
        #  [19.9829292, 20.9829292, 21.9829292, 22.9829292]]]
        # ]


Adam
=================================
.. py:class:: pyvqnet.optim.adam.Adam(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,weight_decay=0,amsgrad: bool = False)

    Adam optimizer, which uses first-moment estimates to dynamically adjust the learning rate of each parameter and second-moment estimates of the gradients.

    Reference: https://arxiv.org/abs/1412.6980.

    .. math::
        t = t + 1
    .. math::
        param  = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1∗moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2∗moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}

    If the amsgrad parameter is True

    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon}

    Otherwise

    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon}

    :param params: Model parameters to optimize.
    :param lr: Learning rate (default: 0.01).
    :param beta1: Coefficient used to compute the running averages of the gradient and its square (default: 0.9).
    :param beta2: Coefficient used to compute the running averages of the gradient and its square (default: 0.999).
    :param epsilon: Constant added to the denominator to improve numerical stability (default: 1e-8).
    :param amsgrad: Whether to use the AMSGrad variant of this algorithm (default: False).
    :param weight_decay: Weight decay coefficient (default: 0).
    :return: An Adam optimizer.

    Example::

        import numpy as np
        from pyvqnet.optim import adam
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adam.Adam(params)

        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0., 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0., 0.9800000, 1.9800000, 2.9800000],
        #  [3.9800000, 4.9799995, 5.9799995, 6.9799995],
        #  [7.9799995, 8.9799995, 9.9799995, 10.9799995]],
        # [[11.9799995, 12.9799995, 13.9799995, 14.9799995],
        #  [15.9799995, 16.9799995, 17.9799995, 18.9799995],
        #  [19.9799995, 20.9799995, 21.9799995, 22.9799995]]]
        # ]

AdamW
=================================
.. py:class:: pyvqnet.optim.adam.AdamW(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, amsgrad: bool = False)

    Implements the AdamW algorithm.


    .. math::
        t = t + 1

    .. math::
        param\_new  = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1∗moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2∗moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}

    If the amsgrad parameter is True

    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon}

    Otherwise

    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon}

    :param params: Model parameters to optimize.
    :param lr: Learning rate (default: 0.01).
    :param beta1: Coefficient used to compute the running averages of the gradient and its square (default: 0.9).
    :param beta2: Coefficient used to compute the running averages of the gradient and its square (default: 0.999).
    :param epsilon: Constant added to the denominator to improve numerical stability (default: 1e-8).
    :param weight_decay: Weight decay coefficient. Default: 0.01.
    :param amsgrad: Whether to use the AMSGrad variant of this algorithm (default: False).
    :return: An AdamW optimizer.

    Example::

        from pyvqnet.optim import adam
        import numpy as np
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adam.AdamW(params, lr=0.5)

        for i in range(1,3):
            opti.step()
        print(param)
        # [[[[ 0.       ,-0.007475 , 0.98255  , 1.972575 ],
        #    [ 2.9626   , 3.952625 , 4.9426501, 5.9326751],
        #    [ 6.9227001, 7.9127251, 8.9027501, 9.8927751]],

        #   [[10.8828001,11.8728251,12.8628501,13.8528751],
        #    [14.8429002,15.8329252,16.8229502,17.8129752],
        #    [18.8030002,19.7930252,20.7830502,21.7730752]]]]


Adamax
=================================
.. py:class:: pyvqnet.optim.adamax.Adamax(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    Implements the Adamax optimizer (an Adam variant based on the infinity norm).

    Reference: https://arxiv.org/abs/1412.6980.

    .. math::
        \\t = t + 1
    .. math::
        moment\_new=\beta1∗moment+(1−\beta1)g
    .. math::
        norm\_new = \max{(\beta1∗norm+\epsilon, \left|g\right|)}
    .. math::
        lr = \frac{lr}{1-\beta1^t}
    .. math::
        param\_new = param − lr*\frac{moment\_new}{norm\_new}\\

    :param params: Model parameters to optimize.
    :param lr: Learning rate (default: 0.01).
    :param beta1: Coefficient used to compute the running averages of the gradient and its square (default: 0.9).
    :param beta2: Coefficient used to compute the running averages of the gradient and its square (default: 0.999).
    :param epsilon: Constant added to the denominator to improve numerical stability (default: 1e-8).

    :return: An Adamax optimizer.

    Example::

        import numpy as np
        from pyvqnet.optim import adamax
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adamax.Adamax(params)

        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0., 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0., 0.9800000, 1.9800000, 2.9800000],
        #  [3.9800000, 4.9799995, 5.9799995, 6.9799995],
        #  [7.9799995, 8.9799995, 9.9799995, 10.9799995]],
        # [[11.9799995, 12.9799995, 13.9799995, 14.9799995],
        #  [15.9799995, 16.9799995, 17.9799995, 18.9799995],
        #  [19.9799995, 20.9799995, 21.9799995, 22.9799995]]]
        # ]


RMSProp
=================================
.. py:class:: pyvqnet.optim.rmsprop.RMSProp( params, lr=0.01, beta=0.99, epsilon=1e-8)

    RMSprop root mean square propagation algorithm optimizer.

    Reference: https://arxiv.org/pdf/1308.0850v5.pdf.

    .. math::
        s_{t+1} = s_{t} + (1 - \beta)*(g)^2

    .. math::
        param_new = param -  \frac{g}{\sqrt{s_{t+1}} + epsilon}


    :param params: Model parameters to optimize.
    :param lr: Learning rate (default: 0.01).
    :param beta: Coefficient used to compute the running average of the gradient and its square (default: 0.99).
    :param epsilon: Constant added to the denominator to improve numerical stability (default: 1e-8).

    :return: An RMSProp optimizer.

    Example::

        import numpy as np
        from pyvqnet.optim import rmsprop
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = rmsprop.RMSProp(params)

        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0., 0.9000000, 1.9000000, 2.8999999],
        #  [3.8999999, 4.9000001, 5.9000001, 6.9000001],
        #  [7.9000001, 8.8999996, 9.8999996, 10.8999996]],
        # [[11.8999996, 12.8999996, 13.8999996, 14.8999996],
        #  [15.8999996, 16.8999996, 17.8999996, 18.8999996],
        #  [19.8999996, 20.8999996, 21.8999996, 22.8999996]]]
        # ]

        # [
        # [[[0., 0.8291118, 1.8291118, 2.8291118],
        #  [3.8291118, 4.8291121, 5.8291121, 6.8291121],
        #  [7.8291121, 8.8291111, 9.8291111, 10.8291111]],
        # [[11.8291111, 12.8291111, 13.8291111, 14.8291111],
        #  [15.8291111, 16.8291111, 17.8291111, 18.8291111],
        #  [19.8291111, 20.8291111, 21.8291111, 22.8291111]]]
        # ]

SGD
=================================
.. py:class:: pyvqnet.optim.sgd.SGD(params, lr=0.01,momentum=0, nesterov=False)

    Stochastic gradient descent optimizer.

    Reference: https://en.wikipedia.org/wiki/Stochastic_gradient_descent.

    .. math::

        \\param\_new=param-lr*g\\

    :param params: Model parameters to optimize.
    :param lr: Learning rate (default: 0.01).
    :param momentum: Momentum factor (default: 0).
    :param nesterov: Enable Nesterov momentum (default: False).

    :return: An SGD optimizer.

    Example::

        import numpy as np
        from pyvqnet.optim import sgd
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = sgd.SGD(params)

        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0., 0.9900000, 1.9800000, 2.9700000],
        #  [3.9600000, 4.9499998, 5.9400001, 6.9299998],
        #  [7.9200001, 8.9099998, 9.8999996, 10.8900003]],
        # [[11.8800001, 12.8699999, 13.8599997, 14.8500004],
        #  [15.8400002, 16.8299999, 17.8199997, 18.8099995],
        #  [19.7999992, 20.7900009, 21.7800007, 22.7700005]]]
        # ]

        # [
        # [[[0., 0.9800000, 1.9600000, 2.9400001],
        #  [3.9200001, 4.8999996, 5.8800001, 6.8599997],
        #  [7.8400002, 8.8199997, 9.7999992, 10.7800007]],
        # [[11.7600002, 12.7399998, 13.7199993, 14.7000008],
        #  [15.6800003, 16.6599998, 17.6399994, 18.6199989],
        #  [19.5999985, 20.5800018, 21.5600014, 22.5400009]]]
        # ]



Metrics Modules
*********************************************************


MSE
=================================

.. py:class:: pyvqnet.utils.metrics.MSE(y_true_Qtensor, y_pred_Qtensor)

    Compute the mean squared error (MSE).

    :param y_true_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the true target values.
    :param y_pred_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the estimated target values.

    :return: Outputs a float result.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        y_true_Qtensor = tensor.arange(1, 12)
        y_pred_Qtensor = tensor.arange(4, 15)
        result = vqnet_metrics.MSE(y_true_Qtensor, y_pred_Qtensor)
        print(result)
        # 9.0

        y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
        y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
        result = vqnet_metrics.MSE(y_true_Qtensor, y_pred_Qtensor)
        print(result)
        # 9.0


RMSE
=================================

.. py:class:: pyvqnet.utils.metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)

    Compute the root mean square error (RMSE).

    :param y_true_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the true target values.
    :param y_pred_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the estimated target values.

    :return: Outputs a float result.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        y_true_Qtensor = tensor.arange(1, 12)
        y_pred_Qtensor = tensor.arange(4, 15)
        result = vqnet_metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)
        print(result)
        # 3.0

        y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
        y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
        result = vqnet_metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)
        print(result)
        # 3.0



MAE
=================================

.. py:class:: pyvqnet.utils.metrics.MAE(y_true_Qtensor, y_pred_Qtensor)

    Compute the mean absolute error (MAE) between the predictions and the true values.

    :param y_true_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the true target values.
    :param y_pred_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the estimated target values.

    :return: Outputs a float result.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        y_true_Qtensor = tensor.arange(1, 12)
        y_pred_Qtensor = tensor.arange(4, 15)
        result = vqnet_metrics.MAE(y_true_Qtensor, y_pred_Qtensor)
        print(result)
        # 3.0

        y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
        y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
        result = vqnet_metrics.MAE(y_true_Qtensor, y_pred_Qtensor)
        print(result)
        # 3.0


R_Square
=================================

.. py:class:: pyvqnet.utils.metrics.R_Square(y_true_Qtensor, y_pred_Qtensor, sample_weight=None)

    Compute the R-squared score between the predictions and the true values.
    The best possible score is 1.0, and it can be negative (because the model can get arbitrarily worse). A constant model that ignores the input features gets an R^2 score of 0.0.

    :param y_true_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the true target values.
    :param y_pred_Qtensor: Input of shape (n_samples,) or (n_samples, n_outputs), the estimated target values.
    :param sample_weight: Array of shape (n_samples,), optional sample weights. Default: None.

    :return: Outputs a float result.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        y_true_Qtensor = tensor.arange(1, 12)
        y_pred_Qtensor = tensor.arange(4, 15)
        result = vqnet_metrics.R_Square(y_true_Qtensor, y_pred_Qtensor)
        print(result) # 0.09999999999999998

        y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
        y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
        result = vqnet_metrics.R_Square(y_true_Qtensor, y_pred_Qtensor)
        print(result) # 0.15625


precision_recall_f1_2_score
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_2_score(y_true_Qtensor, y_pred_Qtensor)

    Compute the precision, recall, and F1 score of predictions for a binary classification task. The predictions and ground truth must be QTensors of shape (n_samples,) with values 0 or 1, representing the labels of the two classes.

    :param y_true_Qtensor: 1-D QTensor input of shape (n_samples,), the true target values.
    :param y_pred_Qtensor: 1-D QTensor input of shape (n_samples,), the estimated target values.

    :returns:
        - precision - Precision
        - recall - Recall
        - f1 - F1 score


    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        y_true_Qtensor = tensor.QTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_Qtensor = tensor.QTensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])

        precision, recall, f1 = vqnet_metrics.precision_recall_f1_2_score(
            y_true_Qtensor, y_pred_Qtensor)
        print(precision, recall, f1)
        # 0.5 0.6 0.5454545454545454


precision_recall_f1_N_score
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_N_score(y_true_Qtensor, y_pred_Qtensor, N, average)

    Compute precision, recall, and F1 score for a multi-class task. The predictions and ground truth are QTensors of shape (n_samples,) with integer values from 0 to N-1, representing the labels of N classes.

    :param y_true_Qtensor: 1-D QTensor input, the true target values.
    :param y_pred_Qtensor: 1-D QTensor input, the estimated target values.
    :param N: N classes (number of classes).
    :param average: string, ['micro', 'macro', 'weighted'].
            This parameter is required for multi-class/multi-label targets.
            ``'micro'``: Compute metrics globally by counting the total true positives, false negatives, and false positives.

            ``'macro'``: Compute the metrics for each label and find their unweighted mean. This does not account for label imbalance.

            ``'weighted'``: Compute the metrics for each label and find their average weighted by the number of true instances for each label. This modifies ``'macro'`` to account for label imbalance; it can produce an F score that is not between precision and recall.

    :returns:
        - precision - Precision
        - recall - Recall
        - f1 - F1 score

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        reference_list = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
        prediciton_list = [1, 2, 2, 2, 3, 1, 2, 3, 3, 3]
        y_true_Qtensor = tensor.QTensor(reference_list)
        y_pred_Qtensor = tensor.QTensor(prediciton_list)

        precision_micro, recall_micro, f1_micro = vqnet_metrics.precision_recall_f1_N_score(
            y_true_Qtensor, y_pred_Qtensor, 3, average='micro')
        print(precision_micro, recall_micro, f1_micro)
        # 0.6 0.6 0.6

        precision_macro, recall_macro, f1_macro = vqnet_metrics.precision_recall_f1_N_score(
            y_true_Qtensor, y_pred_Qtensor, 3, average='macro')
        print(precision_macro, recall_macro, f1_macro)
        # 0.5833333333333334 0.5888888888888889 0.5793650793650794

        precision_weighted, recall_weighted, f1_weighted = vqnet_metrics.precision_recall_f1_N_score(
            y_true_Qtensor, y_pred_Qtensor, 3, average='weighted')
        print(precision_weighted, recall_weighted, f1_weighted)
        # 0.625 0.6 0.6047619047619047


precision_recall_f1_Multi_score
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_Multi_score(y_true_Qtensor, y_pred_Qtensor, N, average)

    Compute precision, recall, and F1 score for a multi-class task. The predictions and ground truth are QTensors of shape (n_samples,N), and the predictions and true labels must be in 0-1 one-hot encoded form.

    :param y_true_Qtensor: 2-D QTensor input, the true target values.
    :param y_pred_Qtensor: 2-D QTensor input, the estimated target values.
    :param N: N classes (number of classes).
    :param average: string, ['micro', 'macro', 'weighted'].
            This parameter is required for multi-class/multi-label targets.
            ``'micro'``: Compute metrics globally by counting the total true positives, false negatives, and false positives.

            ``'macro'``: Compute the metrics for each label and find their unweighted mean. This does not account for label imbalance.

            ``'weighted'``: Compute the metrics for each label and find their average weighted by the number of true instances for each label. This modifies ``'macro'`` to account for label imbalance; it can produce an F score that is not between precision and recall.

    :returns:
        - precision - Precision
        - recall - Recall
        - f1 - F1 score

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        reference_list = [[1, 0], [0, 1], [0, 0], [1, 1], [1, 0]]
        prediciton_list = [[1, 0], [0, 0], [1, 0], [0, 0], [0, 0]]
        y_true_Qtensor = tensor.QTensor(reference_list)
        y_pred_Qtensor = tensor.QTensor(prediciton_list)

        micro_precision, micro_recall, micro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                    y_pred_Qtensor, 2, average='micro')
        print(micro_precision, micro_recall, micro_f1) # 0.5 0.2 0.28571428571428575

        macro_precision, macro_recall, macro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                    y_pred_Qtensor, 2, average='macro')
        print(macro_precision, macro_recall, macro_f1) # 0.25 0.16666666666666666 0.2

        weighted_precision, weighted_recall, weighted_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                    y_pred_Qtensor, 2, average='weighted')
        print(weighted_precision, weighted_recall, weighted_f1) # 0.3 0.19999999999999998 0.24

        reference_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        prediciton_list = [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]]
        y_true_Qtensor = tensor.QTensor(reference_list)
        y_pred_Qtensor = tensor.QTensor(prediciton_list)

        micro_precision, micro_recall, micro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                    y_pred_Qtensor, 3, average='micro')
        print(micro_precision, micro_recall, micro_f1) # 0.5 0.5714285714285714 0.5333333333333333

        macro_precision, macro_recall, macro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                    y_pred_Qtensor, 3, average='macro')
        print(macro_precision, macro_recall, macro_f1) # 0.5 0.5555555555555555 0.5238095238095238

        weighted_precision, weighted_recall, weighted_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                    y_pred_Qtensor, 3, average='weighted')
        print(weighted_precision, weighted_recall, weighted_f1) # 0.5 0.5714285714285714 0.5306122448979592



auc_calculate
=================================

.. py:class:: pyvqnet.utils.metrics.auc_calculate(y_true_Qtensor, y_pred_Qtensor, pos_label=None, sample_weight=None, drop_intermediate=True)

    Compute the Area Under Curve (AUC) result for classification between the predictions and ground truth.

    :param y_true_Qtensor: 1-D QTensor input, shape = [n_samples].
                            True binary labels. If the labels are not {1,1} or {0,1}, pos_label should be given explicitly.
    :param y_pred_Qtensor: 1-D QTensor input, shape = [n_samples].
                            Target scores, which can be probability estimates of the positive class, confidence values, or non-thresholded measures of decisions (returned by the "decision function" of some classifiers)
    :param pos_label: int or str, the label of the positive class. Default: None.
                    When ``pos_label`` is None, if ``y_true_Qtensor`` is in {-1,1} or {0,1}, ``pos_label`` is set to 1; otherwise an error is raised.
    :param sample_weight: Array of shape (n_samples,). Default: None.
    :param drop_intermediate: boolean, whether to drop some suboptimal thresholds that would not appear on the plotted ROC curve. (Default: None).

    :return: Outputs a float result.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.utils import metrics as vqnet_metrics

        y = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        pred = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.4, 0.4, 0.3, 0.2, 0.1])
        y_Qtensor = tensor.QTensor(y)
        pred_Qtensor = tensor.QTensor(pred)
        result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor)
        print("auc:", result) # 0.92

        y = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1])
        pred = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
        y_Qtensor = tensor.QTensor(y)
        pred_Qtensor = tensor.QTensor(pred)
        result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor)
        print("auc:", result) # 0.625

        y = [1, 2, 1, 1, 1, 0, 0, 1, 1, 1]
        pred = [1, 0, 2, 1, 1, 1, 0, 1, 1, 0]
        y_Qtensor = tensor.QTensor(y)
        pred_Qtensor = tensor.QTensor(pred)
        result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor, pos_label=2)
        print("auc:", result) # 0.1111111111111111



Triton Compatibility
*********************************************************

`Triton <https://triton-lang.org/main/index.html>`_ is a language and compiler for writing efficient deep learning GPU kernels.
Users write NumPy-like code in Python, and Triton compiles it into efficient GPU code (similar to CUDA but higher level).
Triton relies on some PyTorch interfaces, and VQNet implements PyTorch-like interfaces so that code written with Triton can be integrated for model forward and backward computation.

Install Triton:

.. code-block::

    pip install triton

The following example is adapted from the official Triton example `layer-norm <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>`_ .
This example must run on Linux with a GPU, with Triton and PyTorch installed (PyTorch is used to compare computational correctness):

.. code-block::

    import torch as real_torch
    import pyvqnet as torch
    import triton
    import triton.language as tl


    DEVICE = torch.DEV_GPU_0

    @triton.jit
    def _layer_norm_fwd_fused(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Compute mean
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Write mean / rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            # Write output
            tl.store(Y + cols, y, mask=mask)



    @triton.jit
    def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                                DY,  # pointer to the output gradient
                                DW,  # pointer to the partial sum of weights gradient
                                DB,  # pointer to the partial sum of biases gradient
                                X,  # pointer to the input
                                W,  # pointer to the weights
                                Mean,  # pointer to the mean
                                Rstd,  # pointer to the 1/std
                                Lock,  # pointer to the lock
                                stride,  # how much to increase the pointer when moving by 1 row
                                N,  # number of columns in X
                                GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Map the program id to the elements of X, DX, and DY it should compute.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        X += row * stride
        DY += row * stride
        DX += row * stride
        # Offset locks and weights/biases gradient pointer for parallel reduction
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        DW = DW + lock_id * N + cols
        DB = DB + lock_id * N + cols
        # Load data to SRAM

        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = (dy).to(w.dtype)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        # First store doesn't accumulate
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)

        # need a barrier to ensure all threads finished before
        # releasing the lock
        tl.debug_barrier()

        # Release the lock
        tl.atomic_xchg(Lock, 0)


    @triton.jit
    def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                            DB,  # pointer to the partial sum of biases gradient
                            FINAL_DW,  # pointer to the weights gradient
                            FINAL_DB,  # pointer to the biases gradient
                            M,  # GROUP_SIZE_M
                            N,  # number of columns
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Map the program id to the elements of DW and DB it should compute.
        pid = tl.program_id(0)
        cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # Iterate through the rows of DW and DB to sum the partial sums.
        for i in range(0, M, BLOCK_SIZE_M):
            rows = i + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(DW + offs, mask=mask, other=0.)
            db += tl.load(DB + offs, mask=mask, other=0.)
        # Write the final sum to the output.
        sum_dw = tl.sum(dw, axis=0)
        sum_db = tl.sum(db, axis=0)
        tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
        tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


    class LayerNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, normalized_shape, weight, bias, eps):
            # allocate output
            y = torch.empty_like(x)
            # reshape input data into 2D tensor
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape
            mean = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            rstd = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            # Less than 64KB per feature: enqueue fused kernel
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
            # heuristics for number of warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # enqueue kernel
            _layer_norm_fwd_fused[(M, )](  #
                x_arg, y, weight, bias, mean, rstd,  #
                x_arg.stride[0], N, eps,  #
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.eps = eps
            return y.data

        @staticmethod
        def backward(ctx, dy):
            x, w, b, m, v = ctx.saved_tensors
            # heuristics for amount of parallel reduction stream for DW/DB
            N = w.shape[0]
            GROUP_SIZE_M = 64
            if N <= 8192: GROUP_SIZE_M = 96
            if N <= 4096: GROUP_SIZE_M = 128
            if N <= 1024: GROUP_SIZE_M = 256
            # allocate output
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device).data
            _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            dw = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            db = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            dx = torch.empty_like(dy).data
            # enqueue kernel using forward pass heuristics
            # also compute partial sums for DW and DB
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape


            _layer_norm_bwd_dx_fused[(M, )](  #
                dx, dy, _dw, _db, x, w, m, v, locks,  #
                x_arg.stride[0], N,  #
                BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
                GROUP_SIZE_M=GROUP_SIZE_M,  #
                num_warps=ctx.num_warps)

            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )

            # accumulate partial sums in separate kernel
            _layer_norm_bwd_dwdb[grid](
                _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
                BLOCK_SIZE_M=32,  #
                BLOCK_SIZE_N=128, num_ctas=1)
            return dx, None, dw, db, None

    preprocess = LayerNorm.preprocess
    layer_norm = LayerNorm.apply


    def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

        dy = .1 * torch.randn_like(x)
        x.requires_grad = True
        # forward pass
        xc, weightc, biasc = preprocess(x, weight, bias)
        y_tri = layer_norm(xc, w_shape, weightc, biasc, eps)
        y_tri = torch.to_tensor(y_tri)
        numpy_w = weight.detach().cpu().numpy()
        numpy_b = bias.detach().cpu().numpy()
        numpy_x = x.detach().cpu().numpy()
        numpy_dy = dy.detach().cpu().numpy()
        torch_weight = real_torch.tensor(numpy_w,device="cuda:1",requires_grad= True)
        torch_bias = real_torch.tensor(numpy_b,device="cuda:1",requires_grad= True)
        torch_x = real_torch.tensor(numpy_x,device="cuda:1",requires_grad= True)
        torch_dy = real_torch.tensor(numpy_dy,device="cuda:1",requires_grad= True)
        y_ref = real_torch.nn.functional.layer_norm(torch_x, w_shape, torch_weight, torch_bias, eps)
        # backward pass (triton)
        y_tri.backward(dy, retain_graph=True)
        dx_tri, dw_tri, db_tri = [_.grad.detach().cpu().numpy() for _ in [x, weight, bias]]
        x.grad, weight.grad, bias.grad = None, None, None
        # backward pass (torch)
        y_ref.backward(torch_dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.detach().cpu().numpy() for _ in [torch_x, torch_weight, torch_bias]]
        # compare
        import numpy as np
        assert np.allclose(y_tri.detach().cpu().numpy(), y_ref.detach().cpu().numpy(), atol=1e-2, rtol=0)
        assert np.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
        assert np.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
        assert np.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
        print("same")
    test_layer_norm(12, 32, torch.float32)
