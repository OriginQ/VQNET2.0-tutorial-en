XTensor classic neural network modules
###############################################

The following classic neural network modules all support automatic backward propagation computation in XTensor. After running the forward function,
you need to define the forward computation within the scope of ``with pyvqnet.xtensor.autograd.tape():``. This will include operators that need automatic differentiation into the computational graph.
Then executing the backward function will get the computed gradients of XTensor s with ``requires_grad == True``.

.. warning::

    XTensor related functions are in the development stage. Currently, they only support classic neural network calculations and cannot be mixed with the QTenor-based interface introduced above.
    If you need to train a quantum machine learning model, please use the relevant interfaces under QTensor.
    
A simple example of a convolution layer is as follows:  

.. code-block::

    from pyvqnet.xtensor import arange
    from pyvqnet import kfloat32
    from pyvqnet.xtensor import Conv2D,autograd

    # an image feed into two dimension convolution layer
    b = 2        # batch size 
    ic = 2       # input channels
    oc = 2      # output channels
    hw = 4      # input width and heights

    # two dimension convolution layer
    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")

    # input of shape [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,dtype=kfloat32).reshape([b,ic,hw,hw])
    x0.requires_grad = True
    with autograd.tape():
    #forward function
        x = test_conv(x0)

    #backward function with autograd
    x.backward()
    print(x0.grad)
    """
    [[[[-0.0194679  0.1530238 -0.0194679  0.1530238] 
    [ 0.2553246  0.1616782  0.2553246  0.1616782] 
    [-0.0194679  0.1530238 -0.0194679  0.1530238] 
    [ 0.2553246  0.1616782  0.2553246  0.1616782]]

    [[ 0.0285322  0.1099411  0.0285322  0.1099411] 
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072] 
    [ 0.0285322  0.1099411  0.0285322  0.1099411]
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072]]]


    [[[-0.0194679  0.1530238 -0.0194679  0.1530238]
    [ 0.2553246  0.1616782  0.2553246  0.1616782]
    [-0.0194679  0.1530238 -0.0194679  0.1530238]
    [ 0.2553246  0.1616782  0.2553246  0.1616782]]

    [[ 0.0285322  0.1099411  0.0285322  0.1099411]
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072]
    [ 0.0285322  0.1099411  0.0285322  0.1099411]
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072]]]]
    <XTensor 2x2x4x4 cpu(0) kfloat32>
    """

.. currentmodule:: pyvqnet.xtensor


Module
************************************************

Abstract calculation module


Module
===========================================================

.. py:class:: pyvqnet.xtensor.module.Module

    The base class for all neural network modules, including quantum or classical modules. Your model should also be a subclass of this for autograd computation.
    Modules can also contain other Module classes, allowing them to be nested in a tree structure. You can assign submodules as member variables: 

    Example::

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.xtensor.Conv2D(1, 20, (5,5))
                self.conv2 = pyvqnet.xtensor.Conv2D(20, 20, (5,5))
            def forward(self, x):
                x = pyvqnet.xtensor.relu(self.conv1(x))
                return pyvqnet.xtensor.relu(self.conv2(x))



forward
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.forward(x, *args, **kwargs)

    Module class abstracts the forward computation function

    :param x: Input XTensor.
    :param \*args: Non-keyword variadic parameters.
    :param \*\*kwargs: keyword variadic parameters.

    :return: Module's output.

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        import pyvqnet as vq
        from pyvqnet.xtensor import Conv2D
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "same")
        x0 = XTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
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
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.state_dict(destination=None, prefix='')

    Returns a dictionary containing the entire state of the module: including parameters and cached values

    :param destination: Return the dictionary that saves the internal modules and parameters of the model.
    :param prefix: Name prefix for used parameters and cached values.
    :return:  a dict.

    Example::

        from pyvqnet.xtensor import Conv2D
        test_conv = Conv2D(2,3,(3,3),(2,2),"same")
        print(test_conv.state_dict().keys())
        #odict_keys(['weights', 'bias'])


toGPU
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.toGPU(device: int = DEV_GPU_0)

    Moves the parameters and buffered data of a module and its submodules to the specified GPU device.

    device specifies the device whose internal data is stored. When device >= DEV_GPU_0, the data is stored on the GPU. If your computer has multiple GPUs,
    You can specify different devices to store data. For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... means that it is stored on GPUs with different serial numbers.
    
    .. note::

         Module cannot be calculated on different GPUs.
         If you try to create a XTensor on a GPU with an ID that exceeds the maximum number of verification GPUs, a Cuda error will be raised.

    :param device: The device currently storing XTensor, default = DEV_GPU_0.device = pyvqnet.DEV_GPU_0, stored in the first GPU, devcie = DEV_GPU_1, stored in the second GPU, and so on
    :return: Module moved to GPU device.

    Examples::

        from pyvqnet.xtensor import ConvT2D 
        test_conv = ConvT2D(3, 2, (4,4), (2, 2), "same")
        test_conv = test_conv.toGPU()
        print(test_conv.backend)
        #1000


toCPU
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.toCPU()

    Moves parameters and buffer data of a module and its submodules to specific CPU devices.

    :return: Module moved to CPU device.

    Examples::

        from pyvqnet.xtensor import ConvT2D 
        test_conv = ConvT2D(3, 2, (4,4), (2, 2), "same")
        test_conv = test_conv.toCPU()
        print(test_conv.backend)
        #0


Save and load model parameters
***********************************************

The following interface can save model parameters to a file, or read parameter files from a file. However, please note that the model structure is not saved in the file, and the user needs to manually build the model structure.

save_parameters
===========================================================

.. py:function:: pyvqnet.xtensor.storage.save_parameters(obj, f)

    Save a dictionary of model parameters to a file.

    :param obj: The dictionary to be saved.
    :param f: file name to save parameters.

    :return: None.

    Example::

        from pyvqnet.xtensor import Module,Conv2D,save_parameters
        
        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net() 
        save_parameters(model.state_dict(),"tmp.model")

load_parameters
===========================================================

.. py:function:: pyvqnet.xtensor.storage.load_parameters(f)

    Load parameters from a file into a dictionary.

    :param f: file name to save parameters.

    :return: Dictionary to save parameters.

    Example::

        from pyvqnet.xtensor import Module, Conv2D,load_parameters,save_parameters

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
        save_parameters(model.state_dict(), "tmp.model")
        model_para = load_parameters("tmp.model")
        model1.load_state_dict(model_para)

ModuleList
***********************************************************************************************

.. py:class:: pyvqnet.xtensor.module.ModuleList([pyvqnet.xtensor.module.Module])


    Save submodules in a list. ModuleList can be indexed like a normal Python list, and the internal parameters of the Module it contains can be saved.

    :param modules: nn.Modules list

    :return: a list of modules

    Example::

        from pyvqnet.xtensor import Module, Linear, ModuleList

        class M(Module):
            def __init__(self):
                super(M, self).__init__()
                self.dense = ModuleList([Linear(4, 2), Linear(4, 2)])

            def forward(self, x, *args, **kwargs):
                y = self.dense[0](x) + self.dense[1](x)
                return y

        mm = M()

        print(mm.state_dict())
        """
        OrderedDict([('dense.0.weights', 
        [[ 0.8224208  0.3421015  0.2118234  0.1082053]     
        [-0.8264768  1.1017226 -0.3860411 -1.6656817]]    
        <Parameter 2x4 cpu(0) kfloat32>), ('dense.0.bias', 
        [0.4125615 0.4414732]
        <Parameter 2 cpu(0) kfloat32>), ('dense.1.weights',
        [[ 1.8939902  0.8871605 -0.3880418 -0.4815852]
        [-0.0956827  0.2667428  0.2900301  0.4039476]]
        <Parameter 2x4 cpu(0) kfloat32>), ('dense.1.bias',
        [-0.0544764  0.0289595]
        <Parameter 2 cpu(0) kfloat32>)])
        """


Classical Neural Network Layers
***********************************************

The following implements some classic neural network layers: convolution, transposed convolution, pooling, normalization, recurrent neural network, etc.


Conv1D
===========================================================

.. py:class:: pyvqnet.xtensor.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:bool = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Apply a 1-dimensional convolution kernel over an input . Inputs to the conv module are of shape (batch_size, input_channels, height)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param kernel_size: `int` - Size of a single kernel. kernel shape = [output_channels,input_channels/group,kernel_size,1]
    :param stride: `int` - Stride, defaults to 1
    :param padding: `str|int` - padding option, which can be a string {'valid', 'same'} or an integer giving the amount of implicit padding to apply . Default "valid".
    :param use_bias: `bool` - if use bias, defaults to True
    :param kernel_initializer: `callable` - Defaults to None
    :param bias_initializer: `callable` - Defaults to None
    :param dilation_rate: `int` - dilated size, defaults: 1
    :param group: `int` -  number of groups of grouped convolutions. Default: 1
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: The name of the module, default: "".
    :return: a Conv1D class

    .. note::
        ``padding='valid'`` is the same as no padding.

        ``padding='same'`` pads the input so the output has the shape as the input.

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Conv1D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv1D(ic,oc,3,2,"same")
        x0 = XTensor(np.arange(1,b*ic*5*5 +1).reshape([b,ic,25]),dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)
        """
        [[[ 17.637825   26.841843   28.811993   30.782139   32.752293
            34.72244    36.69259    38.66274    40.63289    42.603035
            44.57319    46.543335   36.481537 ]
        6274   106.63289                            707144  -4.07236
        108.60304   110.57319   112.54334   114.5789382  -3.58058381349   116.48363
        118.45379   120.423935   85.522644 ]     
        [ 34.14579    -0.6791078  -0.5807535  -0.46274   106.63289823973  -0.384041                           1349   116.48363
            -0.2856848  -0.1873342  -0.088978    0.0093744   0.107725                           823973  -0.384041
            0.2060831   0.3044413  -1.8352301]]]   093744   0.107725
        <XTensor 2x2x13 cpu(0) kfloat32>
        """

Conv2D
===========================================================

.. py:class:: pyvqnet.xtensor.Conv2D(input_channels:int,output_channels:int,kernel_size:tuple,stride:tuple=(1, 1),padding="valid",use_bias = True,kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Apply a two-dimensional convolution kernel over an input . Inputs to the conv module are of shape (batch_size, input_channels, height, width)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param kernel_size: `tuple|list` - Size of a single kernel. kernel shape = [output_channels,input_channels/group,kernel_size,kernel_size]
    :param stride: `tuple|list` - Stride, defaults to (1, 1)|[1,1]
    :param padding: `str|tuple` - padding option, which can be a string {'valid', 'same'} or a tuple of integers giving the amount of implicit padding to apply on both sides. Default "valid".
    :param use_bias: `bool` - if use bias, defaults to True
    :param kernel_initializer: `callable` - Defaults to None
    :param bias_initializer: `callable` - Defaults to None
    :param dilation_rate: `int` - dilated size, defaults: 1
    :param group: `int` -  number of groups of grouped convolutions. Default: 1.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: The name of the module, default: "".

    :return: a Conv2D class

    .. note::
        ``padding='valid'`` is the same as no padding.

        ``padding='same'`` pads the input so the output has the shape as the input.

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Conv2D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv2D(ic,oc,(3,3),(2,2),"same")
        x0 = XTensor(np.arange(1,b*ic*5*5+1).reshape([b,ic,5,5]),dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)
        """
        [[[[13.091256  16.252321   6.009256 ] 
        [25.834864  29.57479   10.406249 ] 
        [17.177385  27.90065   11.024535 ]]

        [[ 9.705042  11.656831   8.584356 ] 
        [23.186415  29.151287  17.489706 ] 
        [23.500261  23.620876  15.1604395]]]     


        [[[55.944958  66.7173    31.89055  ]       
        [86.231346  99.19392   44.14594  ]       
        [53.740646  83.22319   33.986828 ]]      

        [[44.75199   41.64939   35.985905 ]       
        [54.717422  73.95048   48.546726 ]       
        [33.874504  40.06319   31.02438  ]]]]    
        <XTensor 2x2x3x3 cpu(0) kfloat32>
        """

ConvT2D
===========================================================

.. py:class:: pyvqnet.xtensor.ConvT2D(input_channels,output_channels,kernel_size,stride=[1, 1],padding="valid",use_bias="True", kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Apply a two-dimensional transposed convolution kernel over an input. Inputs to the convT module are of shape (batch_size, input_channels, height, width)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param kernel_size: `tuple|list` - Size of a single kernel. kernel shape = [input_channels,output_channels/group,kernel_size,kernel_size]
    :param stride: `tuple|list` - Stride, defaults to (1, 1)|[1,1]
    :param padding: `str|tuple` - padding option, which can be a string {'valid', 'same'} or a tuple of integers giving the amount of implicit padding to apply on both sides. Default "valid".
    :param use_bias: `bool` - Whether to use a offset item. Default to use
    :param kernel_initializer: `callable` - Defaults to None
    :param bias_initializer: `callable` - Defaults to None
    :param dilation_rate: `int` - dilated size, defaults: 1
    :param group: `int` -  number of groups of grouped convolutions. Default: 1.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: The name of the module, default: "".

    :return: a ConvT2D class

    .. note::
        ``padding='valid'`` is the same as no padding.

        ``padding='same'`` pads the input so the output has the shape as the input.


    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import ConvT2D
        import pyvqnet
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1), "valid")
        x = XTensor(np.arange(1, 1 * 3 * 5 * 5+1).reshape([1, 3, 5, 5]), dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)
        """
        
        [[[[  5.4529057  -3.32444     9.211117    9.587392    9.963668
            4.5622444  14.397371 ]
        [ 18.834743   21.769156   36.432068   37.726788   39.021515
            18.737175   16.340559 ]
        [ 29.79763    39.767223   61.934864   63.825333   65.715805
            34.09302    23.981941 ]
        [ 33.684406   45.685955   71.38721    73.27768    75.16815
            39.658592   27.515562 ]
        [ 37.571186   51.604687   80.839554   82.730034   84.6205
            45.224167   31.049183 ]
        [ 33.69648    61.94682    71.845085   73.359276   74.873474
            39.471508   10.021905 ]
        [ 12.103746   23.482103   31.11521    31.710955   32.306698
            19.998552    7.940914 ]]

        [[  4.769257    6.5511374   9.029368    9.207671    9.385972
            3.762906    2.1653163]
        [ -8.366173    0.0307169  -0.3826299  -0.5054388  -0.6282487
            8.602992   -0.3027873]
        [ -9.106487   -4.8349705   1.0091982   0.9871688   0.9651423
            12.1995535   6.483701 ]
        [-11.156897   -5.4630694   0.8990631   0.8770366   0.855011
            14.13983     7.001668 ]
        [-13.207303   -6.09117     0.7889295   0.7669029   0.7448754
            16.080103    7.5196342]
        [-25.585697  -18.799192  -12.708595  -12.908926  -13.109252
            15.557721    7.1425896]
        [ -4.400727   -4.76725     4.1210976   4.2218823   4.322665
            10.08579     9.516866 ]]]]
        <XTensor 1x2x7x7 cpu(0) kfloat32>
        """


AvgPool1D
===========================================================

.. py:class:: pyvqnet.xtensor.AvgPool1D(kernel, stride, padding="valid", name = "")

    This operation applies a 1D average pooling over an input signal composed of several input planes.

    :param kernel: size of the average pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "valid", "same" or integer specifies the padding value, defaults to "valid"
    :param name: name of the output layer.

    :return: AvgPool1D layer

    .. note::
        ``padding='valid'`` is the same as no padding.

        ``padding='same'`` pads the input so the output has the shape as the input.


    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import AvgPool1D
        test_mp = AvgPool1D(3,2,"same")
        x= XTensor(np.array([0, 1, 0, 4, 5,
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
===========================================================

.. py:class:: pyvqnet.xtensor.MaxPool1D(kernel, stride, padding="valid",name="")

    This operation applies a 1D max pooling over an input signal composed of several input planes.

    :param kernel: size of the max pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "valid", "same" or integer specifies the padding value, defaults to "valid"
    :param name: The name of the module, default: "".

    :return: MaxPool1D layer

    .. note::

        ``padding='valid'`` is the same as no padding.

        ``padding='same'`` pads the input so the output has the shape as the input.

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import MaxPool1D
        test_mp = MaxPool1D(3,2,"same")
        x= XTensor(np.array([0, 1, 0, 4, 5,
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
===========================================================

.. py:class:: pyvqnet.xtensor.AvgPool2D(kernel, stride, padding='valid', name='')

    This operation applies 2D average pooling over input features .

    :param kernel: size of the average pooling windows
    :param strides: factors by which to downscale
    :param padding: one of "valid", "same" or tuple with integers specifies the padding value of column and row,defaults to "valid"
    :param name: name of the output layer
    :return: AvgPool2D layer

    .. note::
        ``padding='valid'`` is the same as no padding.

        ``padding='same'`` pads the input so the output has the shape as the input.

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import AvgPool2D
        test_mp = AvgPool2D((2,2),(2,2),"valid")
        x= XTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[[1.5  1.75]
        #    [3.75 3.  ]]]]
        

MaxPool2D
===========================================================

.. py:class:: pyvqnet.xtensor.MaxPool2D(kernel, stride, padding='valid', name='')

    This operation applies 2D max pooling over input features.

    :param kernel: size of the max pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "valid", "same" or tuple with integers specifies the padding value of column and row, defaults to "valid"
    :param name: name of the output layer
    :return: MaxPool2D layer

    .. note::
        ``padding='valid'`` is the same as no padding.

        ``padding='same'`` pads the input so the output has the shape as the input.



    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import MaxPool2D
        test_mp = MaxPool2D((2,2),(2,2),"valid")
        x= XTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [[[[3. 4.]
        #    [5. 6.]]]]
        

Embedding
===========================================================

.. py:class:: pyvqnet.xtensor.Embedding(num_embeddings, embedding_dim, weight_initializer=<function xavier_normal>,dtype=None, name: str = '')

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    :param num_embeddings: `int` - size of the dictionary of embeddings.
    :param embedding_dim: `int` - the size of each embedding vector.
    :param weight_initializer: `callable` - defaults to normal.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer.

    :return: a Embedding class

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Embedding
        import pyvqnet
        vlayer = Embedding(30,3)
        x = XTensor(np.arange(1,25).reshape([2,3,2,2]),dtype= pyvqnet.kint64)
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
===========================================================

.. py:class:: pyvqnet.xtensor.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5,beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Apply batch normalization on 4D input (B, C, H, W). Refer to the paper
     `Batch Normalization: Accelerating Deep Network Training by Reducing
     Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    
    .. math::

         y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` and :math:`\beta` are the parameters to be trained. During training, the layer will continue to run to estimate its calculated mean and variance, which are then used for normalization during evaluation. The average variance The mean maintains the default momentum of 0.1.

    .. note::

         When using `with autograd.tape()`, BatchNorm2d enters train mode and uses local_mean, local_variance, gamma, and beta for batch normalization.
        
         When the above code is not used, BatchNorm2d uses eval mode and uses cached global_mean, global_variance, gamma, and beta for batch normalization.


    :param channel_num: `int` - Input channel number.
    :param momentum: `float` - Momentum when calculating exponential weighted average, default is 0.1.
    :param beta_initializer: `callable` - beta initialization method, default is all-zero initialization.
    :param gamma_initializer: `callable` - the initialization method of gamma, the default is all-one initialization.
    :param epsilon: `float` - numerical stability parameter, default 1e-5.
    :param dtype: The data type of the parameter, defaults: None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param name: Batch normalization layer name, default is "".

    :return: 2D batch normalization layer instance.

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import BatchNorm2d,autograd

        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)

        x = XTensor(np.arange(1, 17).reshape([b, ic, 4, 1]))
        x.requires_grad = True
        with autograd.tape():
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
===========================================================

.. py:class:: pyvqnet.xtensor.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Apply batch normalization on 2D input (B, C) or 3D input (B, C, H). Refer to the paper
     `Batch Normalization: Accelerating Deep Network Training by Reducing
     Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    
    .. math::

         y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` and :math:`\beta` are the parameters to be trained. During training, the layer will continue to run to estimate its calculated mean and variance, which are then used for normalization during evaluation. The average variance The mean maintains the default momentum of 0.1.

    .. note::

         When using `with autograd.tape()`, BatchNorm2d enters train mode and uses local_mean, local_variance, gamma, and beta for batch normalization.
        
         When the above code is not used, BatchNorm2d uses eval mode and uses cached global_mean, global_variance, gamma, and beta for batch normalization.


    :param channel_num: `int` - Input channel number.
    :param momentum: `float` - Momentum when calculating exponential weighted average, default is 0.1.
    :param beta_initializer: `callable` - beta initialization method, default is all-zero initialization.
    :param gamma_initializer: `callable` - the initialization method of gamma, the default is all-one initialization.
    :param epsilon: `float` - numerical stability parameter, default 1e-5.
    :param dtype: The data type of the parameter, defaults: None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param name: Batch normalization layer name, default is "".

    :return: 1D batch normalization layer instance.

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import BatchNorm1d,autograd

        test_conv = BatchNorm1d(4)

        x = XTensor(np.arange(1, 17).reshape([4, 4]))
        with autograd.tape():
            y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416405, -1.3416405, -1.3416405, -1.3416405],
        # [-0.4472135, -0.4472135, -0.4472135, -0.4472135],
        # [0.4472135, 0.4472135, 0.4472135, 0.4472135],
        # [1.3416405, 1.3416405, 1.3416405, 1.3416405]
        # ]

LayerNormNd
===========================================================

.. py:class:: pyvqnet.xtensor.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5, affine: bool = True, dtype=None, name="")


    Layer normalization is performed on the last several dimensions of any input. The specific method is as described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    For inputs like (B,C,H,W,D), ``norm_shape`` can be [C,H,W,D],[H,W,D],[W,D] or [D] .

    :param norm_shape: `float` - standardize the shape.
    :param epsilon: `float` - numerical stability constant, defaults to 1e-5.
    :param affine: `bool` - whether to use the applied affine transformation, the default is True.
    :param name: name of the output layer.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    
    :return: a LayerNormNd class.


    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import LayerNormNd
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = XTensor(np.arange(1,17).reshape([2,2,2,2]))
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
===========================================================

.. py:class:: pyvqnet.xtensor.LayerNorm2d(norm_size:int, epsilon:float = 1e-5,  affine: bool = True, dtype=None, name="")

    Applies Layer Normalization over a mini-batch of 4D inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last  `D` dimensions size.

    For input like (B,C,H,W), ``norm_size`` should equals to C * H * W.

    :param norm_size: `float` - normalize size,equals to C * H * W
    :param epsilon: `float` - numerical stability constant, defaults to 1e-5
    :param affine: `bool` - whether to use the applied affine transformation, the default is True
    :param name: name of the output layer
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    
    :return: a LayerNorm2d class

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import LayerNorm2d
        ic = 4
        test_conv = LayerNorm2d(8)
        x = XTensor(np.arange(1,17).reshape([2,2,4,1]))
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
===========================================================

.. py:class:: pyvqnet.xtensor.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine: bool = True, dtype=None, name="")
    
    Applies Layer Normalization over a mini-batch of 2D inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last dimensions size, where ``norm_size`` 
    is the value of last dim size.

    :param norm_size: `float` - normalize size,equals to last dim
    :param epsilon: `float` - numerical stability constant, defaults to 1e-5
    :param affine: `bool` - whether to use the applied affine transformation, the default is True
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    
    :param name: name of the output layer

    :return: a LayerNorm1d class

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import LayerNorm1d
        test_conv = LayerNorm1d(4)
        x = XTensor(np.arange(1,17).reshape([4,4]))
        y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355]
        # ]
        

Linear
===========================================================

.. py:class:: pyvqnet.xtensor.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, dtype=None, name: str = "")

    Linear module (fully-connected layer).
    :math:`y = Ax + b`

    :param input_channels: `int` - number of inputs features
    :param output_channels: `int` - number of output features
    :param weight_initializer: `callable` - defaults to normal
    :param bias_initializer: `callable` - defaults to zeros
    :param use_bias: `bool` - defaults to True
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

    :return: a Linear class

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Linear
        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout)
        input = XTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),dtype=pyvqnet.kfloat32)
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
===========================================================

.. py:class:: pyvqnet.xtensor.Dropout(dropout_rate = 0.5)

    Dropout module. The dropout module randomly sets the output of some units to zero while upgrading other units based on the given dropout_rate probability.

    .. note::

         When using `with autograd.tape()`, Dropout enters train mode and randomly sets the output of some units to zero.
        
         When the above code is not used, Dropout enters train mode and uses eval mode to output as is.

    :param dropout_rate: `float` - The probability that the neuron is set to zero.
    :param name: The name of this module, default is "".

    :return: Dropout instance.

    Example::

        import numpy as np
        from pyvqnet.xtensor import Dropout
        from pyvqnet.xtensor import XTensor
        b = 2
        ic = 2
        x = XTensor(np.arange(-1 * ic * 2 * 2,
                            (b - 1) * ic * 2 * 2).reshape([b, ic, 2, 2]))
        
        droplayer = Dropout(0.5)
        
        y = droplayer(x)
        print(y)
        """
        [[[[-0. -0.]
        [-0. -0.]]

        [[-0. -0.]
        [-0. -0.]]]


        [[[ 0.  0.]
        [ 0.  6.]]

        [[ 8. 10.]
        [ 0. 14.]]]]
        <XTensor 2x2x2x2 cpu(0) kfloat32>
        """

Pixel_Shuffle 
=====================

.. py:class:: pyvqnet.xtensor.Pixel_Shuffle(upscale_factors)

    Rearrange tensors of shape: (*, C * r^2, H, W) to a tensor of shape (*, C, H * r, W * r) where r is the scaling factor.

    :param upscale_factors: factor to increase the scale transformation

    :return:
            Pixel_Shuffle result

    Example::

        from pyvqnet.xtensor import Pixel_Shuffle
        from pyvqnet.xtensor import ones
        ps = Pixel_Shuffle(3)
        inx = ones([5,2,3,18,4,4])
        inx.requires_grad=  True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 2, 12, 12]

Pixel_Unshuffle 
========================

.. py:class:: pyvqnet.xtensor.Pixel_Unshuffle(downscale_factors)

    Reverses the Pixel_Shuffle operation by rearranging the elements. Shuffles a Tensor of shape (*, C, H * r, W * r) to (*, C * r^2, H, W) , where r is the shrink factor.
    
    :param downscale_factors: factor to increase the scale transformation

    :return:
            Pixel_Unshuffle result

    Example::

        from pyvqnet.xtensor import Pixel_Unshuffle
        from pyvqnet.xtensor import ones
        ps = Pixel_Unshuffle(3)
        inx = ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 18, 4, 4]


GRU
==========================

.. py:class:: pyvqnet.xtensor.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Gated Recurrent Unit (GRU) module. Support multi-layer stacking, bidirectional configuration.
    The calculation formula of the single-layer one-way GRU is as follows:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Input feature dimensions.
    :param hidden_size: Hidden feature dimensions.
    :param num_layers: Stack layer numbers. default: 1.
    :param batch_first: If batch_first is True, input shape should be [batch_size,seq_len,feature_dim],
     if batch_first is False, the input shape should be [seq_len,batch_size,feature_dim],default: True.
    :param use_bias: If use_bias is False, this module will not contain bias. default: True.
    :param bidirectional: If bidirectional is True, the module will be bidirectional GRU. default: False.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

    :return: A GRU module instance.

    Example::

        from pyvqnet.xtensor import GRU
        import pyvqnet.xtensor as tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)
        print(output)
        print(hn)
        # [[[-0.3525755 -0.2587337  0.149786   0.461374   0.1449795  0.4734624
        #    -0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.3525755 -0.2587337  0.149786   0.461374   0.1449795  0.4734624
        #    -0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.3525755 -0.2587337  0.149786   0.461374   0.1449796  0.4734623
        #    -0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]]

        #  [[-0.5718066 -0.2159877 -0.2978141  0.1441837 -0.0761072  0.3096682
        #     0.0115526 -0.2295886 -0.1917011  0.5739203  0.5765471  0.3173765]
        #   [-0.5718066 -0.2159877 -0.2978141  0.1441837 -0.0761072  0.3096682
        #     0.0115526 -0.2295886 -0.1917011  0.5739203  0.5765471  0.3173765]
        #   [-0.5718066 -0.2159877 -0.2978141  0.1441837 -0.0761071  0.3096681
        #     0.0115526 -0.2295886 -0.1917011  0.5739203  0.5765471  0.3173765]]

        #  [[-0.5341613 -0.0807438 -0.2697436 -0.0872668 -0.2630565  0.2262028
        #     0.1108652 -0.0301746 -0.0453528  0.5952781  0.5141098  0.3779774]
        #   [-0.5341613 -0.0807438 -0.2697436 -0.0872668 -0.2630565  0.2262028
        #     0.1108652 -0.0301746 -0.0453528  0.5952781  0.5141098  0.3779774]
        #   [-0.5341613 -0.0807438 -0.2697436 -0.0872668 -0.2630565  0.2262028
        #     0.1108652 -0.0301746 -0.0453528  0.5952781  0.5141098  0.3779774]]

        #  [[-0.4394189  0.0851144 -0.0413915 -0.2225544 -0.4341614  0.1345865
        #     0.241122   0.2335991  0.1918445  0.6587436  0.4840603  0.4863765]
        #   [-0.4394189  0.0851144 -0.0413915 -0.2225544 -0.4341614  0.1345865
        #     0.241122   0.2335991  0.1918445  0.6587436  0.4840603  0.4863765]
        #   [-0.4394189  0.0851144 -0.0413915 -0.2225544 -0.4341614  0.1345865
        #     0.241122   0.2335991  0.1918445  0.6587436  0.4840603  0.4863765]]

        #  [[-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008
        #     0.4682454  0.5807121  0.5288119  0.7821091  0.5577921  0.6762444]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008
        #     0.4682454  0.5807121  0.5288119  0.7821091  0.5577921  0.6762444]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008
        #     0.4682454  0.5807121  0.5288119  0.7821091  0.5577921  0.6762444]]]
        # <XTensor 5x3x12 cpu(0) kfloat32>

        # [[[ 0.0388437 -0.0772902 -0.3355273 -0.143957   0.0520131 -0.323649 ]
        #   [ 0.0388437 -0.0772902 -0.3355273 -0.143957   0.0520131 -0.323649 ]
        #   [ 0.0388437 -0.0772902 -0.3355273 -0.143957   0.0520131 -0.323649 ]]

        #  [[-0.4715263 -0.5740314 -0.6212391 -0.20578    0.8116962 -0.1011644]
        #   [-0.4715263 -0.5740314 -0.6212391 -0.20578    0.8116962 -0.1011644]
        #   [-0.4715263 -0.5740314 -0.6212391 -0.20578    0.8116962 -0.1011644]]

        #  [[-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008]]

        #  [[-0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]]]
        # <XTensor 4x3x6 cpu(0) kfloat32>

RNN 
====================

.. py:class:: pyvqnet.xtensor.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Recurrent Neural Network (RNN) Module, use :math:`\tanh` or :math:`\text{ReLU}` as activation function.
    bidirectional RNN and multi-layer RNN is supported.
    The calculation formula of single-layer unidirectional RNN is as follows:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` will replace :math:`\tanh`.

    :param input_size: Input feature dimensions.
    :param hidden_size: Hidden feature dimensions.
    :param num_layers: Stack layer numbers. default: 1.
    :param nonlinearity: non-linear activation function, default: ``'tanh'`` .
    :param batch_first: If batch_first is True, input shape should be [batch_size,seq_len,feature_dim],
     if batch_first is False, the input shape should be [seq_len,batch_size,feature_dim],default: True.
    :param use_bias: If use_bias is False, this module will not contain bias. default: True.
    :param bidirectional: If bidirectional is True, the module will be bidirectional RNN. default: False.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

    :return: A RNN module instance.

    Example::

        from pyvqnet.xtensor import RNN
        import pyvqnet.xtensor as tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)
        print(output)
        print(hn)

        # [[[ 0.2618747  0.5377525  0.5129814 -0.0242058 -0.6311338  0.655068   
        #    -0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311] 
        #   [ 0.2618747  0.5377525  0.5129814 -0.0242058 -0.6311338  0.655068   
        #    -0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311] 
        #   [ 0.2618747  0.5377525  0.5129814 -0.0242058 -0.6311338  0.655068   
        #    -0.7402378 -0.1209883 -0.2462614  0.1552387  0.4331009 -0.3321312]]

        #  [[ 0.4478737  0.8829155 -0.0466572 -0.1412439 -0.0311011  0.3848146  
        #    -0.710574   0.0830743 -0.2802465 -0.0228663  0.1831353 -0.3086829] 
        #   [ 0.4478737  0.8829155 -0.0466572 -0.1412439 -0.0311011  0.3848146
        #    -0.710574   0.0830743 -0.2802465 -0.0228663  0.1831353 -0.3086829]
        #   [ 0.4478737  0.8829155 -0.0466572 -0.1412439 -0.0311011  0.3848146
        #    -0.710574   0.0830743 -0.2802465 -0.0228663  0.1831353 -0.3086829]]

        #  [[ 0.581092   0.8708823  0.2848003 -0.154836  -0.4118715  0.5057767
        #    -0.8474038  0.0595496 -0.5158566 -0.1731871  0.3361979 -0.2265194]
        #   [ 0.581092   0.8708823  0.2848003 -0.154836  -0.4118715  0.5057767
        #    -0.8474038  0.0595496 -0.5158566 -0.1731871  0.3361979 -0.2265194]
        #   [ 0.581092   0.8708823  0.2848004 -0.154836  -0.4118715  0.5057766
        #    -0.8474038  0.0595496 -0.5158566 -0.1731871  0.3361979 -0.2265194]]

        #  [[ 0.5710331  0.8946801 -0.0285869 -0.032192  -0.2297462  0.4527371
        #    -0.7243505  0.2147861 -0.3519893 -0.1745383 -0.5063711 -0.7420927]
        #   [ 0.5710331  0.8946801 -0.0285869 -0.032192  -0.2297462  0.4527371
        #    -0.7243505  0.2147861 -0.3519893 -0.1745383 -0.5063711 -0.7420927]
        #   [ 0.5710331  0.8946801 -0.0285869 -0.032192  -0.2297462  0.4527371
        #    -0.7243505  0.2147861 -0.3519893 -0.1745383 -0.5063711 -0.7420927]]

        #  [[ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097
        #    -0.4387358 -0.9216538 -0.3471112 -0.9059284  0.9011658 -0.6876704]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097
        #    -0.4387358 -0.9216538 -0.3471112 -0.9059284  0.9011658 -0.6876704]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097
        #    -0.4387358 -0.9216538 -0.3471112 -0.9059284  0.9011658 -0.6876704]]]
        # <XTensor 5x3x12 cpu(0) kfloat32>

        # [[[-0.1595494 -0.4154228  0.6991262  0.634046   0.6960769 -0.4966849]
        #   [-0.1595494 -0.4154228  0.6991262  0.634046   0.6960769 -0.4966849]
        #   [-0.1595494 -0.4154228  0.6991262  0.634046   0.6960769 -0.4966849]]

        #  [[ 0.2330922 -0.7264591 -0.7571693  0.3780781  0.4115383  0.7572027]
        #   [ 0.2330922 -0.7264591 -0.7571693  0.3780781  0.4115383  0.7572027]
        #   [ 0.2330922 -0.7264591 -0.7571693  0.3780781  0.4115383  0.7572027]]

        #  [[ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097]]

        #  [[-0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311]
        #   [-0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311]
        #   [-0.7402378 -0.1209883 -0.2462614  0.1552387  0.4331009 -0.3321312]]]
        # <XTensor 4x3x6 cpu(0) kfloat32>

LSTM
===========================================================

.. py:class:: pyvqnet.xtensor.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Long Short-Term Memory (LSTM) module. Support bidirectional LSTM, stacked multi-layer LSTM and other configurations.
    The calculation formula of single-layer unidirectional LSTM is as follows:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Input feature dimensions.
    :param hidden_size: Hidden feature dimensions.
    :param num_layers: Stack layer numbers. default: 1.
    :param batch_first: If batch_first is True, input shape should be [batch_size,seq_len,feature_dim],
     if batch_first is False, the input shape should be [seq_len,batch_size,feature_dim],default: True.
    :param use_bias: If use_bias is False, this module will not contain bias. default: True.
    :param bidirectional: If bidirectional is True, the module will be bidirectional LSTM. default: False.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

    :return: A LSTM module instance.

    Example::

        from pyvqnet.xtensor import LSTM
        import pyvqnet.xtensor as tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))

        print(output)
        print(hn)
        print(cn)

        """
        [[[ 0.0952653  0.2589843  0.2486762  0.1287942  0.1021227  0.1911968
        -0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [ 0.0952653  0.2589843  0.2486762  0.1287942  0.1021227  0.1911968
        -0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [ 0.0952653  0.2589843  0.2486762  0.1287942  0.1021227  0.1911968
        -0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]]

        [[ 0.085497   0.1925259  0.3265719 -0.0792181  0.0310715  0.1398373
        -0.0023391 -0.3675225  0.3450475  0.2388555 -0.0579962 -0.0536785]
        [ 0.085497   0.1925259  0.3265719 -0.0792181  0.0310715  0.1398373
        -0.0023391 -0.3675225  0.3450475  0.2388555 -0.0579962 -0.0536785]
        [ 0.085497   0.1925259  0.3265719 -0.0792181  0.0310715  0.1398373
        -0.0023391 -0.3675225  0.3450475  0.2388555 -0.0579962 -0.0536785]]

        [[ 0.0788613  0.1431215  0.3656158 -0.1339753  0.0101364  0.0694154
            0.0299196 -0.329112   0.4189231  0.2440841  0.0078633 -0.091286 ]
        [ 0.0788613  0.1431215  0.3656158 -0.1339753  0.0101364  0.0694154
            0.0299196 -0.329112   0.4189231  0.2440841  0.0078633 -0.091286 ]
        [ 0.0788613  0.1431215  0.3656158 -0.1339753  0.0101364  0.0694154
            0.0299196 -0.329112   0.4189231  0.2440841  0.0078633 -0.091286 ]]

        [[ 0.0730032  0.1006287  0.3992919 -0.1526794 -0.0077587  0.0049352
            0.0684287 -0.2034638  0.4807869  0.2429611  0.1062173 -0.100478 ]
        [ 0.0730032  0.1006287  0.3992919 -0.1526794 -0.0077587  0.0049352
            0.0684287 -0.2034638  0.4807869  0.2429611  0.1062173 -0.100478 ]
        [ 0.0730032  0.1006287  0.3992919 -0.1526794 -0.0077587  0.0049352
            0.0684287 -0.2034638  0.4807869  0.2429611  0.1062173 -0.100478 ]]

        [[ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961
            0.1292751  0.0901535  0.5008832  0.1749451  0.2660664  0.0133356]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961
            0.1292751  0.0901535  0.5008832  0.1749451  0.2660664  0.0133356]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961
            0.1292751  0.0901535  0.5008832  0.1749451  0.2660664  0.0133356]]]
        <XTensor 5x3x12 cpu(0) kfloat32>

        [[[-0.1111576 -0.0410911 -0.1360135 -0.2142551  0.3879747 -0.4244705]
        [-0.1111576 -0.0410911 -0.1360135 -0.2142551  0.3879747 -0.4244705]
        [-0.1111576 -0.0410911 -0.1360135 -0.2142551  0.3879747 -0.4244705]]

        [[ 0.2689943  0.3063364 -0.2738754  0.2168426 -0.4891826  0.2454725]
        [ 0.2689943  0.3063364 -0.2738754  0.2168426 -0.4891826  0.2454725]
        [ 0.2689943  0.3063364 -0.2738754  0.2168426 -0.4891826  0.2454725]]

        [[ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961]]

        [[-0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [-0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [-0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]]]
        <XTensor 4x3x6 cpu(0) kfloat32>

        [[[-0.1973301 -0.2033432 -0.4812729 -0.3186268  0.7220416 -0.6785325]
        [-0.1973301 -0.2033432 -0.4812729 -0.3186268  0.7220416 -0.6785325]
        [-0.1973301 -0.2033432 -0.4812729 -0.3186268  0.7220416 -0.6785325]]

        [[ 1.6790413  0.4572753 -0.4843928  0.3902704 -0.9931879  0.4607614]
        [ 1.6790413  0.4572753 -0.4843928  0.3902704 -0.9931879  0.4607614]
        [ 1.6790413  0.4572753 -0.4843928  0.3902704 -0.9931879  0.4607614]]

        [[ 0.0753609  0.0477858  0.7334676 -0.3433677 -0.1412639 -0.1365461]
        [ 0.0753609  0.0477858  0.7334676 -0.3433677 -0.1412639 -0.1365461]
        [ 0.0753609  0.0477858  0.7334676 -0.3433677 -0.1412639 -0.1365461]]

        [[-0.104733  -0.6896871  0.4874932  0.4806345 -0.167628  -0.0125997]
        [-0.104733  -0.6896871  0.4874932  0.4806345 -0.167628  -0.0125997]
        [-0.104733  -0.6896871  0.4874932  0.4806345 -0.167628  -0.0125997]]]
        <XTensor 4x3x6 cpu(0) kfloat32>
        """


Dynamic_GRU
===========================================================

.. py:class:: pyvqnet.xtensor.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Apply a multilayer gated recurrent unit (GRU) RNN to a dynamic-length input sequence.

    The first input should be a variable-length batch sequence input defined
    Through the ``xtensor.PackedSequence`` class.
    The ``xtensor.PackedSequence`` class can be constructed as
    Call the next function in succession: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_GRU is also a ``xtensor.PackedSequence`` class,
    It can be unpacked into a normal XTensor using ``xtensor.pad_pack_sequence``.

    For each element in the input sequence, each layer computes the following formula:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of loop layers. Default: 1
    :param batch_first: If True, the input shape is provided as [batch size, sequence length, feature dimension]. If False, input shape is provided as [sequence length, batch size, feature dimension], default True.
    :param use_bias: If False, the layer does not use bias weights b_ih and b_hh. Default: true.
    :param bidirectional: If true, becomes a bidirectional GRU. Default: false.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

    :return: A Dynamic_GRU class

    Example::

        from pyvqnet.xtensor import Dynamic_GRU
        import pyvqnet.xtensor as tensor
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

        # [[[ 0.872566   0.8611314 -0.5047759  0.9130142]
        #   [ 0.5690175  0.9443005 -0.3432685  0.9585502]
        #   [ 0.8512535  0.8650243 -0.487494   0.9192616]]

        #  [[ 0.7886224  0.7501922 -0.4578349  0.919861 ]
        #   [ 0.         0.         0.         0.       ]
        #   [ 0.712998   0.749097  -0.2539869  0.9405512]]

        #  [[ 0.7242796  0.6568378 -0.4209562  0.9258339]
        #   [ 0.         0.         0.         0.       ]
        #   [ 0.         0.         0.         0.       ]]

        #  [[ 0.6601164  0.5651093 -0.2040557  0.9421862]
        #   [ 0.         0.         0.         0.       ]
        #   [ 0.         0.         0.         0.       ]]]
        # <XTensor 4x3x4 cpu(0) kfloat32>
        # [4 1 2]

Dynamic_RNN 
=======================

.. py:class:: pyvqnet.xtensor.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Applies recurrent neural networks (RNNs) to dynamic-length input sequences.

    The first input should be a variable-length batch sequence input defined
    Through the ``xtensor.PackedSequence`` class.
    The ``xtensor.PackedSequence`` class can be constructed as
    Call the next function in succession: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_RNN is also a ``xtensor.PackedSequence`` class,
    It can be unpacked into a normal XTensor using ``xtensor.pad_pack_sequence``.

    Recurrent Neural Network (RNN) module, using :math:`\tanh` or :math:`\text{ReLU}` as activation function. Support two-way, multi-layer configuration.
    The calculation formula of single-layer one-way RNN is as follows:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
    
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` will replace :math:`\tanh`.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked RNN layers, default: 1.
    :param nonlinearity: Non-linear activation function, default is ``'tanh'``.
    :param batch_first: If True, the input shape is [batch size, sequence length, feature dimension],
      If False, the input shape is [sequence length, batch size, feature dimension], default True.
    :param use_bias: If False, the module does not apply bias items, default: True.
    :param bidirectional: If True, it becomes bidirectional RNN, default: False.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

    :return: Dynamic_RNN instance

    Example::

        from pyvqnet.xtensor import Dynamic_RNN
        import pyvqnet.xtensor as tensor
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

        """
        [[[ 2.283666   2.1524734  0.         0.8799834]
        [ 2.9422705  2.6565394  0.         0.8055274]
        [ 2.9554236  2.1205678  0.         1.1741859]]

        [[ 6.396565   3.8327866  0.         2.6239884]
        [ 0.         0.         0.         0.       ]
        [ 7.37332    4.7455616  0.         2.6786256]]

        [[12.521921   5.239943   0.         4.62357  ]
        [ 0.         0.         0.         0.       ]
        [ 0.         0.         0.         0.       ]]

        [[19.627499   8.675274   0.         6.6746845]
        [ 0.         0.         0.         0.       ]
        [ 0.         0.         0.         0.       ]]]
        <XTensor 4x3x4 cpu(0) kfloat32>
        [4 1 2]
        """

Dynamic_LSTM
===========================================================

.. py:class:: pyvqnet.xtensor.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Apply Long Short-Term Memory (LSTM) RNNs to dynamic-length input sequences.

    The first input should be a variable-length batch sequence input defined
    Through the ``xtensor.PackedSequence`` class.
    The ``xtensor.PackedSequence`` class can be constructed as
    Call the next function in succession: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_LSTM is also a ``xtensor.PackedSequence`` class,
    It can be unpacked into a normal XTensor using ``xtensor.pad_pack_sequence``.

    Recurrent Neural Network (RNN) module, using :math:`\tanh` or :math:`\text{ReLU}` as activation function. Support two-way, multi-layer configuration.
    The calculation formula of single-layer one-way RNN is as follows:

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
    :param num_layers: Number of stacked LSTM layers, default: 1.
    :param batch_first: If True, the input shape is [batch size, sequence length, feature dimension],
      If False, the input shape is [sequence length, batch size, feature dimension], default True.
    :param use_bias: If False, the module does not apply bias items, default: True.
    :param bidirectional: If True, it becomes a bidirectional LSTM, default: False.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

    :return: Dynamic_LSTM instance

    Example::

        from pyvqnet.xtensor import Dynamic_LSTM
        import pyvqnet.xtensor as tensor

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
        """
        [[[ 0.1970974  0.2246606  0.2627596 -0.080385 ] 
        [ 0.2071671  0.2119733  0.2301395 -0.2693036] 
        [ 0.1106544  0.3478935  0.4335948  0.378578 ]]

        [[ 0.1176731 -0.0304668  0.2993484  0.0920533] 
        [ 0.1386266 -0.0483974  0.2384422 -0.1798031] 
        [ 0.         0.         0.         0.       ]]

        [[ 0.0798466 -0.1468595  0.4139522  0.3376699] 
        [ 0.1303781 -0.1537685  0.2934605  0.0475375] 
        [ 0.         0.         0.         0.       ]]

        [[ 0.         0.         0.         0.       ]
        [ 0.0958745 -0.2243107  0.4114271  0.3248508]
        [ 0.         0.         0.         0.       ]]]
        <XTensor 4x3x4 cpu(0) kfloat32>
        [3 4 1]
        """

Loss Function Layer
***********************************************

.. note::

        Please note that unlike pytorch and other frameworks, in the forward function of the following loss function, the first parameter is the label, and the second parameter is the predicted value.


MeanSquaredError
===========================================================

.. py:class:: pyvqnet.xtensor.MeanSquaredError(name="")

    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. , then:

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)


    :math:`x` and :math:`y` are QTensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    :param name: name of the output layer

    :return: a MeanSquaredError class

    Parameters for loss forward function:

        x: :math:`(N, *)` where :math:`*` means, any number of additional dimensions

        y: :math:`(N, *)`, same shape as the input

    Example::

        from pyvqnet.xtensor import XTensor, kfloat64
        from pyvqnet.xtensor import MeanSquaredError
        y = XTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    dtype=kfloat64)
        x = XTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)
        # [0.0115000]
        


BinaryCrossEntropy
===========================================================

.. py:class:: pyvqnet.xtensor.BinaryCrossEntropy(name="")

    Measures the Binary Cross Entropy between the target and the output:

    The unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :return: a BinaryCrossEntropy class

    Parameters for loss forward function:

        x: :math:`(N, *)` where :math:`*` means, any number of additional dimensions

        y: :math:`(N, *)`, same shape as the input

    Example::

        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import BinaryCrossEntropy
        x = XTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]] )
        y = XTensor([[0.0, 1.0, 0], [0.0, 0, 1]] )

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        print(result)
        # [0.6364825]
        

CategoricalCrossEntropy
===========================================================

.. py:class:: pyvqnet.xtensor.CategoricalCrossEntropy(name="")

    This criterion combines LogSoftmax and NLLLoss in one single class.

    The loss can be described as below, where `class` is index of target's class:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: a CategoricalCrossEntropy class

    Parameters for loss forward function:

        x: :math:`(N, *)` where :math:`*` means, any number of additional dimensions

        y: :math:`(N, *)`, same shape as the input, should have data type of the 64-bit integer.

    Example::

        from pyvqnet.xtensor import XTensor,kfloat32,kint64
        from pyvqnet.xtensor import CategoricalCrossEntropy
        x = XTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], dtype=kfloat32)
        y = XTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
        dtype=kfloat32)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)
        # [3.7852428]

SoftmaxCrossEntropy
===========================================================

.. py:class:: pyvqnet.xtensor.SoftmaxCrossEntropy(name="")

    This criterion combines LogSoftmax and NLLLoss in one single class with more numeral stablity.

    The loss can be described as below, where `class` is index of target's class:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: a SoftmaxCrossEntropy class

    Parameters for loss forward function:

        x: :math:`(N, *)` where :math:`*` means, any number of additional dimensions

        y: :math:`(N, *)`, same shape as the input, should have data type of the 64-bit integer.

    Example::

        from pyvqnet.xtensor import XTensor, kfloat32, kint64
        from pyvqnet.xtensor import SoftmaxCrossEntropy
        x = XTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                    dtype=kfloat32)
        y = XTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                    dtype=kfloat32)
        loss_result = SoftmaxCrossEntropy()
        result = loss_result(y, x)
        print(result)

        # [3.7852478]

NLL_Loss
===========================================================

.. py:class:: pyvqnet.xtensor.NLL_Loss(name="")

    The average negative log likelihood loss. It is useful to train a classification problem with `C` classes

    The `x` given through a forward call is expected to contain log-probabilities of each class. `x` has to be a Tensor of size either :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case. The `y` that this loss expects should be a class index in the range :math:`[0, C-1]` where `C = number of classes`.

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n}, \quad

    :return: a NLL_Loss class

    Parameters for loss forward function:

        x: :math:`(N, *)`, the output of the loss function, which can be a multidimensional variable.

        y: :math:`(N, *)`, the true value expected by the loss function, should have data type of the 64-bit integer.

    Example::

        from pyvqnet.xtensor import XTensor, kint64
        from pyvqnet.xtensor import NLL_Loss

        x = XTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ]).reshape([1, 3, 1, 5])

        x.requires_grad = True
        y = XTensor([[[2, 1, 0, 0, 2]]])

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)
        #[-0.6187226]

CrossEntropyLoss
===========================================================

.. py:class:: pyvqnet.xtensor.CrossEntropyLoss(name="")

    This criterion combines LogSoftmax and NLLLoss in one single class.

    `x` is expected to contain raw, unnormalized scores for each class. `x` has to be a Tensor of size :math:`(C)` for unbatched input, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the `K`-dimensional case.

    The loss can be described as below, where `class` is index of target's class:

    .. math::

        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: a CrossEntropyLoss class

    Parameters for loss forward function:

        x: :math:`(N, *)`, the output of the loss function, which can be a multidimensional variable.

        y: :math:`(N, *)`, the true value expected by the loss function, should have data type of the 64-bit integer.


    Example::

        from pyvqnet.xtensor import XTensor, kfloat32
        from pyvqnet.xtensor import CrossEntropyLoss
        x = XTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ]).reshape([1, 3, 1, 5])
        
        x.requires_grad = True
        y = XTensor([[[2, 1, 0, 0, 2]]], dtype=kfloat32)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)

        #[1.1508200]


Activation Function
***********************************************


sigmoid
===========================================================
.. py:function:: pyvqnet.xtensor.sigmoid(x)

    Applies a sigmoid activation function to the given input.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    :param x: input.
    :return: sigmoid activation result.

    Examples::

        from pyvqnet.xtensor import sigmoid
        from pyvqnet.xtensor import XTensor

        y = sigmoid(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.7310586, 0.8807970, 0.9525741, 0.9820138]


softplus
===========================================================
.. py:class:: pyvqnet.xtensor.softplus(x)

    Applies a softplus activation function to the given input.

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    :param x: input.
    :return: softplus activation result.

    Examples::

        from pyvqnet.xtensor import softplus
        from pyvqnet.xtensor import XTensor
        y = softplus(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [1.3132616, 2.1269281, 3.0485873, 4.0181499]
        

softsign
===========================================================
.. py:class:: pyvqnet.xtensor.softsign(x)

    Applies a softsign activation function to the given input.

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    :param x: 输入.

    :param x: input.
    :return: softsign activation result.

    Examples::

        from pyvqnet.xtensor import softsign
        from pyvqnet.xtensor import XTensor
        y = softsign(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.5000000, 0.6666667, 0.7500000, 0.8000000]
        


softmax
===========================================================
.. py:class:: pyvqnet.xtensor.softmax(x,axis:int = -1)

    Applies a softmax activation function to the given input.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    :param x: input.
    :param axis: Calculated dimensions (-1 for last axis), default = -1.


    :return: Softmax activation result.

    Examples::

        from pyvqnet.xtensor import softmax
        from pyvqnet.xtensor import XTensor

        y = softmax(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.0320586, 0.0871443, 0.2368828, 0.6439142]
        

hard_sigmoid
===========================================================
.. py:class:: pyvqnet.xtensor.hard_sigmoid(x)

    Applies a hardsigmoid activation function to the given input.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param x: input.
    :return: hardsigmoid activation result.

    Examples::

        from pyvqnet.xtensor import hard_sigmoid
        from pyvqnet.xtensor import XTensor

        y = hard_sigmoid(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.6666667, 0.8333334, 1., 1.]
        

relu
===========================================================
.. py:class:: pyvqnet.xtensor.relu(x)

    Applies a relu activation function to the given input.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}

    :param x: input.
    :return: hardsigmoid activation result.

    Examples::

        from pyvqnet.xtensor import relu
        from pyvqnet.xtensor import XTensor

        y = relu(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [0., 2., 0., 4.]


leaky_relu
===========================================================
.. py:class:: pyvqnet.xtensor.leaky_relu(x, alpha:float=0.01)

    Applies a leakyrelu function to the given input.

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param x: input.
    :param alpha: LeakyRelu coefficient, default: 0.01.

    :return: LeakyReLu activation result.

    Examples::

        from pyvqnet.xtensor import leaky_relu
        from pyvqnet.xtensor import XTensor
        y = leaky_relu(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.0100000, 2., -0.0300000, 4.]
        


elu
===========================================================
.. py:class:: pyvqnet.xtensor.elu(x, alpha:float=1)

    Applies an elu function to the given input.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param x: input.
    :param alpha: elu coefficient, default: 0.01.

    :return: elu activation result.

    Examples::

        from pyvqnet.xtensor import elu
        from pyvqnet.xtensor import XTensor

        y = elu(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.6321205, 2., -0.9502130, 4.]
        
         
tanh
===========================================================
.. py:class:: pyvqnet.xtensor.tanh(x)

    Applies an tanh function to the given input.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param x: input.

    :return: tanh activation result.

    Examples::

        from pyvqnet.xtensor import tanh
        from pyvqnet.xtensor import XTensor
        y = tanh(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.7615942, 0.9640276, -0.9950548, 0.9993293]
        

Optimizer module
***********************************************


Optimizer
==========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Optimizer( params, lr=0.01)

    Base class for all optimizers.

    :param params: List of model parameters that need to be optimized.
    :param lr: Learning rate, default value: 0.01.

step
==========================================================
.. py:method:: pyvqnet.xtensor.optimizer.Optimizer.step()

    Use the update method of the corresponding optimizer to update parameters.

Adadelta
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adadelta( params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA: An Adaptive Learning Rate Method. reference: (https://arxiv.org/abs/1212.5701)

    .. math::

        E(g_t^2) &= \beta * E(g_{t-1}^2) + (1-\beta) * g^2\\
        Square\_avg &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
        E(dx_t^2) &= \beta * E(dx_{t-1}^2) + (1-\beta) * (-g*square\_avg)^2 \\
        param\_new &= param - lr * Square\_avg

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta: for computing a running average of squared gradients (default: 0.99)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a Adadelta optimizer

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adadelta,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adadelta(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4).astype(np.float64)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ 0.7224208  0.2421015  0.1118234  0.0082053]
        [-0.9264768  1.0017226 -0.4860411 -1.7656817]
        [ 0.282856   1.7939901  0.7871605 -0.4880418]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[ 0.6221698  0.1418506  0.0115724 -0.0920456]
        [-1.0267278  0.9014716 -0.586292  -1.8659327]
        [ 0.1826051  1.6937392  0.6869095 -0.5882927]]
        <Parameter 3x4 cpu(0) kfloat32>
        """

Adagrad
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adagrad( params, lr=0.01, epsilon=1e-8)

    Implements Adagrad algorithm. reference: (https://databricks.com/glossary/adagrad)

    .. math::
        \begin{align}
        moment\_new &= moment + g * g\\param\_new 
        &= param - \frac{lr * g}{\sqrt{moment\_new} + \epsilon}
        \end{align}

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a Adagrad optimizer

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adagrad,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adagrad(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            opti.step()
            print(MM.a1.weights)
        """
        [[ -99.17758  -99.6579   -99.78818  -99.89179]
        [-100.82648  -98.89828 -100.38604 -101.66568]
        [ -99.61714  -98.10601  -99.11284 -100.38804]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-169.88826 -170.36858 -170.49886 -170.60248]
        [-171.53716 -169.60895 -171.09671 -172.37637]
        [-170.32782 -168.81668 -169.82352 -171.09872]]
        <Parameter 3x4 cpu(0) kfloat32>
        """


Adam
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adam( params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,amsgrad: bool = False)

    Adam: A Method for Stochastic Optimization reference: (https://arxiv.org/abs/1412.6980),it can dynamically adjusts the learning rate of each parameter using the 1st moment estimates and the 2nd moment estimates of the gradient.

    .. math::
        t = t + 1 
    .. math::
        moment\_1\_new=\beta1∗moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2∗moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}

    if amsgrad = True
    
    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon} 

    else

    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon} 


    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta1: coefficients used for computing running averages of gradient and its square (default: 0.9)
    :param beta2: coefficients used for computing running averages of gradient and its square (default: 0.999)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :param amsgrad: whether to use the AMSGrad variant of this algorithm (default: False)
    :return: a Adam optimizer

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adam,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adam(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ -99.17759   -99.6579    -99.788185  -99.8918  ]
        [-100.826485  -98.89828  -100.38605  -101.66569 ]
        [ -99.61715   -98.10601   -99.11285  -100.38805 ]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-199.17725 -199.65756 -199.78784 -199.89145]
        [-200.82614 -198.89795 -200.38571 -201.66534]
        [-199.61682 -198.10568 -199.1125  -200.3877 ]]
        """

Adamax
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adamax(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    Implements Adamax algorithm (a variant of Adam based on infinity norm).reference: (https://arxiv.org/abs/1412.6980)

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

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta1: coefficients used for computing running averages of gradient and its square (default: 0.9)
    :param beta2: coefficients used for computing running averages of gradient and its square (default: 0.999)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a Adamax optimizer

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adamax,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adamax(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ -99.17758   -99.6579    -99.788185  -99.89179 ]
        [-100.82648   -98.89828  -100.38605  -101.66568 ]
        [ -99.61714   -98.10601   -99.11285  -100.38804 ]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-199.17758 -199.6579  -199.7882  -199.89178]
        [-200.82648 -198.89827 -200.38605 -201.66568]
        [-199.61714 -198.106   -199.11285 -200.38803]]
        <Parameter 3x4 cpu(0) kfloat32>
        """
        
RMSProp
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.RMSProp( params, lr=0.01, beta=0.99, epsilon=1e-8)
    
    Implements RMSprop algorithm. reference: (https://arxiv.org/pdf/1308.0850v5.pdf)

    .. math::
        s_{t+1} = s_{t} + (1 - \beta)*(g)^2

    .. math::
        param_new = param -  \frac{g}{\sqrt{s_{t+1}} + epsilon}

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta: coefficients used for computing running averages of gradient and its square (default: 0.99)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a RMSProp optimizer

    Example::

        import numpy as np
        from pyvqnet.xtensor import RMSProp,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = RMSProp(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ -999.17804  -999.6584   -999.78864  -999.8923 ]
        [-1000.82697  -998.89874 -1000.38654 -1001.6662 ]
        [ -999.6176   -998.1065   -999.11334 -1000.38855]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-1708.0596 -1708.54   -1708.6702 -1708.7738]
        [-1709.7085 -1707.7803 -1709.2681 -1710.5477]
        [-1708.4991 -1706.988  -1707.9949 -1709.27  ]]
        <Parameter 3x4 cpu(0) kfloat32>
        """

SGD
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.SGD(params, lr=0.01,momentum=0, nesterov=False)

    Implements SGD algorithm. reference: (https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

    .. math::

        \\param\_new=param-lr*g\\

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param momentum: momentum factor (default: 0)
    :param nesterov: enables Nesterov momentum (default: False)
    :return: a SGD optimizer

    
    Example::

        import numpy as np
        from pyvqnet.xtensor import SGD,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = SGD(MM.parameters(),lr=100,momentum=0.2)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[-5999.1777 -6599.6577 -7199.788  -7799.8916]
        [-6000.8267 -6598.8984 -7200.386  -7801.6655]
        [-5999.617  -6598.106  -7199.113  -7800.388 ]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-13199.178 -14519.658 -15839.788 -17159.89 ]
        [-13200.826 -14518.898 -15840.387 -17161.666]
        [-13199.617 -14518.105 -15839.113 -17160.389]]
        <Parameter 3x4 cpu(0) kfloat32>
        """


