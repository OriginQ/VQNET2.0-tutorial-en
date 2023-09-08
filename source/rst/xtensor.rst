XTensor module
===================

XTensor is an interface for VQNet to accelerate tensor calculations using automatic operator parallelization. The interface supports classic calculations under CPU/GPU, and the API definition is basically the same as the original XTensor.

For example, in the following example, reshape is used to perform cyclic calculations on a. Since there is no dependency between these reshape calculations, parallel calculations can be performed naturally. Therefore, the 100 reshape calculations in this example are automatically and asynchronously calculated to achieve the purpose of acceleration.

     Example::

         from pyvqnet.xtensor import xtensor,reshape
         a = xtensor([2, 3, 4, 5])
         for i in range(100):
             y = reshape(a,(2,2))


XTensor's functions and properties
--------------------------------------

ndim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:attribute:: XTensor.ndim

    Return number of dimensions

    :return: number of dimensions

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.ndim)

        # 1
    
shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: XTensor.shape

    Return the shape of the XTensor.

    :return: value of shape

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.shape)

        # (4)

size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: XTensor.size

    Return the number of elements in the XTensor.

    :return: number of elements


    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.size)

        # 4

numel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.numel

    Returns the number of elements in the tensor.

    :return: The number of elements in the tensor.

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.numel())

        # 4

device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: XTensor.device

    Returns the hardware location where XTensor is stored.

    The XTensor hardware location supports CPU device=0, the first GPU device=1000, the second GPU device=1001, ... the 10th GPU device=1009.

    :return: The hardware location of the tensor.

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.device)
        # 0

dtype
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: XTensor.dtype

    Returns the data type of the tensor.

    XTensor internal data type dtype supports kbool = 0, kuint8 = 1, kint8 = 2, kint32 = 4,
    kint64 = 5, kfloat32 = 6, kfloat64 = 7. If initialized with a list, the default is kfloat32.

    :return: The data type of the tensor.

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.dtype)
        # 4

requires_grad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: XTensor.requires_grad

    Sets and gets whether the XTensor needs to calculate gradients.

    .. note::

         XTensor If you want to calculate gradients, you need to explicitly set requires_grad = True.

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5.0])
        a.requires_grad = True
        print(a.grad)


backward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.backward(grad=None)

    Use the backpropagation algorithm to calculate the gradients of all tensors whose gradients need to be calculated in the calculation graph where the current tensor is located.

    .. note::

         For the interface under xtensor, you need to use `with autograd.tape()` to include all operations that you want to perform automatic differentiation, and these operations do not include in-place operations, for example:
         a+=1, a[:]=1, does not include data copying, such as toGPU(), toCPU(), etc.

    :return: None

    Example::

        from pyvqnet.xtensor import xtensor,autograd

        target = xtensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]])
        target = requires_grad=True
        with autograd.tape():
            y = 2*target + 3
            y.backward()
        print(target.grad)
        #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

to_numpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.to_numpy()

    Copy self data to a new numpy.array.

    :return: a new numpy.array contains XTensor data

    Example::

        from pyvqnet.xtensor import xtensor
        t3 = xtensor([2, 3, 4, 5])
        t4 = t3.to_numpy()
        print(t4)

        # [2. 3. 4. 5.]

item
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.item()

    Return the only element from in the XTensor.Raises 'RuntimeError' if XTensor has more than 1 element.

    :return: only data of this object

    Example::

        from pyvqnet.xtensor import ones

        t = ones([1])
        print(t.item())

        # 1.0

argmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.argmax(*kargs)

    Return the indices of the maximum value of all elements in the input XTensor,or
    Return the indices of the maximum values of a XTensor across a dimension.

    :param dim: dim (int) – the dimension to reduce,only accepts single axis. if dim == None, returns the indices of the maximum value of all elements in the input tensor.The valid dim range is [-R, R), where R is input's ndim. when dim < 0, it works the same way as dim + R.
    :param keepdims:  whether the output XTensor has dim retained or not.

    :return: the indices of the maximum value in the input XTensor.

    Example::

        from pyvqnet.xtensor import xtensor
        a = XTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])

        flag = a.argmax()
        print(flag)
        
        # [0.]

        flag_0 = a.argmax(0, True)
        print(flag_0)

        # [
        # [0., 3., 0., 3.]
        # ]

        flag_1 = a.argmax(1, True)
        print(flag_1)

        # [
        # [0.],
        # [2.],
        # [0.],
        # [1.]
        # ]

argmin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.argmin(*kargs)

    Return the indices of the minimum value of all elements in the input XTensor,or
    Return the indices of the minimum values of a XTensor across a dimension.

    :param dim: dim (int) – the dimension to reduce,only accepts single axis. if dim == None, returns the indices of the minimum value of all elements in the input tensor.The valid dim range is [-R, R), where R is input's ndim. when dim < 0, it works the same way as dim + R.
    :param keepdims:  whether the output XTensor has dim retained or not.

    :return: the indices of the minimum value in the input XTensor.

    Example::

        
        from pyvqnet.xtensor import XTensor
        a = XTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmin()
        print(flag)

        # [12.]

        flag_0 = a.argmin(0, True)
        print(flag_0)

        # [
        # [3., 2., 2., 1.]
        # ]

        flag_1 = a.argmin(1, False)
        print(flag_1)

        # [2., 3., 1., 0.]

all
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.all()

    Return True, if all XTensor value is non-zero.

    :return: True,if all XTensor value is non-zero.

    Example::

        import pyvqnet.xtensor as xtensor
        shape = [2, 3]
        t = xtensor.full(shape,1)
        flag = t.all()
        print(flag)

        #True
        #<XTensor  cpu(0) kbool>

any
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.any()

    Return True,if any XTensor value is non-zero.

    :return: True,if any XTensor value is non-zero.

    Example::

        import pyvqnet.xtensor as xtensor
        shape = [2, 3]
        t = xtensor.full(shape,1)
        flag = t.any()
        print(flag)

        #True
        #<XTensor  cpu(0) kbool>


fill_rand_binary\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.fill_rand_binary_(v=0.5)

    Fills a XTensor with values randomly sampled from a binomial distribution.

    If the data generated randomly after binomial distribution is greater than Binarization threshold,then the number of corresponding positions of the XTensor is set to 1, otherwise 0.

    :param v: Binarization threshold
    :return: None

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        t.fill_rand_binary_(2)
        print(t)

        # [
        # [1., 1., 1.],
        # [1., 1., 1.]
        # ]

fill_rand_signed_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.fill_rand_signed_uniform_(v=1)

    Fills a XTensor with values randomly sampled from a signed uniform distribution.

    Scale factor of the values generated by the signed uniform distribution.

    :param v: a scalar value
    :return: None

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        value = 42

        t.fill_rand_signed_uniform_(value)
        print(t)

        # [[ 4.100334   7.7989464 18.075905 ]
        #  [28.918327   8.632122  30.067429 ]]
        # <XTensor 2x3 cpu(0) kfloat32>


fill_rand_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.fill_rand_uniform_(v=1)

    Fills a XTensor with values randomly sampled from a uniform distribution

    Scale factor of the values generated by the uniform distribution.

    :param v: a scalar value
    :return: None

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        value = 42
        t.fill_rand_uniform_(value)
        print(t)

        # [[23.050167 24.899473 30.037952]
        #  [35.459164 25.316061 36.033714]]
        # <XTensor 2x3 cpu(0) kfloat32>


fill_rand_normal\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.fill_rand_normal_(m=0, s=1)

    Fills a XTensor with values randomly sampled from a normal distribution
    Mean of the normal distribution. Standard deviation of the normal distribution.

    :param m: mean of the normal distribution
    :param s: standard deviation of the normal distribution
    :return: None

    Example::

        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        t.fill_rand_normal_(2, 10)
        print(t)

        # [[13.630787   6.838046   4.9956346]
        #  [ 3.5302546 -9.688148  17.580711 ]]
        # <XTensor 2x3 cpu(0) kfloat32>


XTensor.transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.transpose(*axes)

    Reverse or permute the axes of an array.if new_dims = None, revsers the dim.

    :param new_dims: the new order of the dimensions (list of integers).
    :return:  result XTensor.

    Example::

        from pyvqnet.xtensor import XTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
        t = XTensor(a)
        rlt = t.transpose([2,0,1])
        print(rlt)

        rlt = t.transpose()
        print(rlt)
        """
        [[[ 0.  3.]
        [ 6.  9.]]

        [[ 1.  4.]
        [ 7. 10.]]

        [[ 2.  5.]
        [ 8. 11.]]]
        <XTensor 3x2x2 cpu(0) kfloat32>

        [[[ 0.  6.]
        [ 3.  9.]]

        [[ 1.  7.]
        [ 4. 10.]]

        [[ 2.  8.]
        [ 5. 11.]]]
        <XTensor 3x2x2 cpu(0) kfloat32>
        """

XTensor.reshape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.reshape(new_shape)

    Change the tensor’s shape ,return a new XTensor.

    :param new_shape: the new shape (list of integers)
    :return: a new XTensor


    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C)
        t = XTensor(a)
        reshape_t = t.reshape([C, R])
        print(reshape_t)
        # [
        # [0., 1., 2.],
        # [3., 4., 5.],
        # [6., 7., 8.],
        # [9., 10., 11.]
        # ]


getdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.getdata()

    Returns a numpy.ndarray shallow copy representing the data in the XTensor. If the original data is on the GPU, the ndarray view copied by the XTensor on the CPU will first be returned.

    :return: A shallow copy of numpy.ndarray containing the current XTensor data.

    Example::

        import pyvqnet.xtensor  as xtensor
        t = xtensor.ones([3, 4])
        a = t.getdata()
        print(a)

        # [[1. 1. 1. 1.]
        #  [1. 1. 1. 1.]
        #  [1. 1. 1. 1.]]

__getitem__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.__getitem__()

    Slicing indexing of XTensor is supported, or using XTensor as advanced index access input. A new XTensor will be returned.

    The parameters start, stop, and step can be separated by a colon,such as start:stop:step, where start, stop, and step can be default

    As a 1-D XTensor,indexing or slicing can only be done on a single axis.

    As a 2-D XTensor and a multidimensional XTensor,indexing or slicing can be done on multiple axes.

    If you use XTensor as an index for advanced indexing, see numpy for `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

    If your XTensor as an index is the result of a logical operation, then you do a Boolean index.

    .. note:: We use an index form like a[3,4,1],but the form a[3][4][1] is not supported.And ``Ellipsis`` is also not supported.

    :param item: A integer or XTensor as an index.

    :return: A new XTensor.

    Example::

        import pyvqnet.xtensor as tensor
        from pyvqnet.xtensor import XTensor
        aaa = tensor.arange(1, 61).reshape([4, 5, 3])

        print(aaa[0:2, 3, :2])

        print(aaa[3, 4, 1])

        print(aaa[3][4][1])

        print(aaa[:, 2, :])

        print(aaa[2])

        print(aaa[0:2, ::3, 2:])

        a = tensor.ones([2, 2])
        b = XTensor([[1, 1], [0, 1]])
        b = b > 0
        c = a[b]
        print(c)

        tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
        tt.requires_grad = True
        index_sample1 = tensor.arange(0, 3).reshape([3, 1])
        index_sample2 = XTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
        gg = tt[:, index_sample1, 3:, index_sample2, 2:]
        """
        [[10. 11.]
        [25. 26.]]
        <XTensor 2x2 cpu(0) kfloat32>

        [59.]
        <XTensor 1 cpu(0) kfloat32>

        [59.]
        <XTensor 1 cpu(0) kfloat32>

        [[ 7.  8.  9.]
        [22. 23. 24.]
        [37. 38. 39.]
        [52. 53. 54.]]
        <XTensor 4x3 cpu(0) kfloat32>

        [[31. 32. 33.]
        [34. 35. 36.]
        [37. 38. 39.]
        [40. 41. 42.]
        [43. 44. 45.]]
        <XTensor 5x3 cpu(0) kfloat32>

        [[[ 3.]
        [12.]]

        [[18.]
        [27.]]]
        <XTensor 2x2x1 cpu(0) kfloat32>

        [1. 1. 1.]
        <XTensor 3 cpu(0) kfloat32>

        [[[[[  87.   88.]]

        [[ 983.  984.]]]


        [[[  91.   92.]]

        [[ 987.  988.]]]


        [[[  87.   88.]]

        [[ 983.  984.]]]]



        [[[[ 207.  208.]]

        [[1103. 1104.]]]


        [[[ 211.  212.]]

        [[1107. 1108.]]]


        [[[ 207.  208.]]

        [[1103. 1104.]]]]



        [[[[ 319.  320.]]

        [[1215. 1216.]]]


        [[[ 323.  324.]]

        [[1219. 1220.]]]


        [[[ 323.  324.]]

        [[1219. 1220.]]]]]
        <XTensor 3x3x2x1x2 cpu(0) kfloat32>
        """

__setitem__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: XTensor.__setitem__()

    Slicing indexing of XTensor is supported, or using XTensor as advanced index access input. A new XTensor will be returned.

    The parameters start, stop, and step can be separated by a colon,such as start:stop:step, where start, stop, and step can be default

    As a 1-D XTensor,indexing or slicing can only be done on a single axis.

    As a 2-D XTensor and a multidimensional XTensor,indexing or slicing can be done on multiple axes.

    If you use XTensor as an index for advanced indexing, see numpy for `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

    If your XTensor as an index is the result of a logical operation, then you do a Boolean index.

    .. note:: We use an index form like a[3,4,1],but the form a[3][4][1] is not supported.And ``Ellipsis`` is also not supported.

    :param item: A integer or XTensor as an index

    :return: None

    Example::

        import pyvqnet.xtensor as tensor
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a2 = aaa[3, 4, 1]
        aaa[3, 4, 1] = tensor.arange(10001,
                                        10001 + vqnet_a2.size).reshape(vqnet_a2.shape)
        print(aaa)
        # [
        # [[1., 2., 3.],    
        #  [4., 5., 6.],    
        #  [7., 8., 9.],    
        #  [10., 11., 12.], 
        #  [13., 14., 15.]],
        # [[16., 17., 18.], 
        #  [19., 20., 21.], 
        #  [22., 23., 24.], 
        #  [25., 26., 27.], 
        #  [28., 29., 30.]],
        # [[31., 32., 33.], 
        #  [34., 35., 36.],
        #  [37., 38., 39.],
        #  [40., 41., 42.],
        #  [43., 44., 45.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [52., 53., 54.],
        #  [55., 56., 57.],
        #  [58., 10001., 60.]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a3 = aaa[:, 2, :]
        aaa[:, 2, :] = tensor.arange(10001,
                                        10001 + vqnet_a3.size).reshape(vqnet_a3.shape)
        print(aaa)
        # [
        # [[1., 2., 3.],
        #  [4., 5., 6.],
        #  [10001., 10002., 10003.],
        #  [10., 11., 12.],
        #  [13., 14., 15.]],
        # [[16., 17., 18.],
        #  [19., 20., 21.],
        #  [10004., 10005., 10006.],
        #  [25., 26., 27.],
        #  [28., 29., 30.]],
        # [[31., 32., 33.],
        #  [34., 35., 36.],
        #  [10007., 10008., 10009.],
        #  [40., 41., 42.],
        #  [43., 44., 45.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [10010., 10011., 10012.],
        #  [55., 56., 57.],
        #  [58., 59., 60.]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a4 = aaa[2, :]
        aaa[2, :] = tensor.arange(10001,
                                    10001 + vqnet_a4.size).reshape(vqnet_a4.shape)
        print(aaa)
        # [
        # [[1., 2., 3.],
        #  [4., 5., 6.],
        #  [7., 8., 9.],
        #  [10., 11., 12.],
        #  [13., 14., 15.]],
        # [[16., 17., 18.],
        #  [19., 20., 21.],
        #  [22., 23., 24.],
        #  [25., 26., 27.],
        #  [28., 29., 30.]],
        # [[10001., 10002., 10003.],
        #  [10004., 10005., 10006.],
        #  [10007., 10008., 10009.],
        #  [10010., 10011., 10012.],
        #  [10013., 10014., 10015.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [52., 53., 54.],
        #  [55., 56., 57.],
        #  [58., 59., 60.]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a5 = aaa[0:2, ::2, 1:2]
        aaa[0:2, ::2,
            1:2] = tensor.arange(10001,
                                    10001 + vqnet_a5.size).reshape(vqnet_a5.shape)
        print(aaa)
        # [
        # [[1., 10001., 3.],
        #  [4., 5., 6.],
        #  [7., 10002., 9.],
        #  [10., 11., 12.],
        #  [13., 10003., 15.]],
        # [[16., 10004., 18.],
        #  [19., 20., 21.],
        #  [22., 10005., 24.],
        #  [25., 26., 27.],
        #  [28., 10006., 30.]],
        # [[31., 32., 33.],
        #  [34., 35., 36.],
        #  [37., 38., 39.],
        #  [40., 41., 42.],
        #  [43., 44., 45.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [52., 53., 54.],
        #  [55., 56., 57.],
        #  [58., 59., 60.]]
        # ]
        a = tensor.ones([2, 2])
        b = tensor.XTensor([[1, 1], [0, 1]])
        b = b > 0
        x = tensor.XTensor([1001, 2001, 3001])

        a[b] = x
        print(a)
        # [
        # [1001., 2001.],
        #  [1., 3001.]
        # ]


GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: XTensor.GPU(device: int = DEV_GPU_0)

    Copy XTensor data to the specified GPU device and return a new XTensor

    device specifies the device whose internal data is stored. When device >= DEV_GPU_0, the data is stored on the GPU.
    If your computer has multiple GPUs, you can specify different devices to store data. For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... means stored on GPUs with different serial numbers.

    .. note::
         XTensor cannot perform calculations on different GPUs.
         If you try to create an XTensor on a GPU with an ID that exceeds the maximum number of verification GPUs, a Cuda error will be thrown.
         Note that this interface will disconnect the currently constructed calculation graph.

    :param device: The device currently storing XTensor, default =DEV_GPU_0,
      device = pyvqnet.DEV_GPU_0, stored in the first GPU, devcie = DEV_GPU_1,
      stored in the second GPU, and so on.

    :return: XTensor copied to GPU device.

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        b = a.GPU()
        print(b.device)
        #1000

CPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: XTensor.CPU()

    Copy XTensor to specific CPU device, return a new XTensor

    .. note::
         XTensor cannot perform calculations on different hardware.
         Note that this interface will disconnect the currently constructed calculation graph.

    :return: XTensor copied to CPU device.

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        b = a.CPU()
        print(b.device)
        # 0

toGPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: XTensor.toGPU(device: int = DEV_GPU_0)

    Move XTensor to specified GPU device

    device specifies the device whose internal data is stored. When device >= DEV_GPU, the data is stored on the GPU.
    If your computer has multiple GPUs, you can specify different devices to store data.
    For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... means stored on GPUs with different serial numbers.

    .. note::
         XTensor cannot perform calculations on different GPUs.
         If you try to create an XTensor on a GPU with an ID that exceeds the maximum number of verification GPUs, a Cuda error will be thrown.
         Note that this interface will disconnect the currently constructed calculation graph.

    :param device: The device currently saving XTensor, default=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, stored in the first GPU, devcie = DEV_GPU_1, stored in the second GPU, and so on.
    :return: The current XTensor.

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        a = a.toGPU()
        print(a.device)
        #1000


toCPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: XTensor.toCPU()

    Move XTensor to specific GPU device

    .. note::
         XTensor cannot perform calculations on different hardware.
         Note that this interface will disconnect the currently constructed calculation graph.

    :return: The current XTensor.

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        b = a.toCPU()
        print(b.device)
        # 0


isGPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: XTensor.isGPU()

    Whether this XTensor's data is stored on GPU host memory.

    :return: Whether this XTensor's data is stored on GPU host memory.

    Examples::
    
        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        a = a.isGPU()
        print(a)
        # False

isCPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: XTensor.isCPU()

    Whether this XTensor's data is stored in CPU host memory.

    :return: Whether this XTensor's data is stored in CPU host memory.

    Examples::
    
        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        a = a.isCPU()
        print(a)
        # True


Creation
-----------------------------

ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.ones(shape,device=None,dtype=None)

    Return one-tensor with the input shape.

    :param shape: input shape
    :param device: stored in which device，default 0 , CPU.
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output XTensor with the input shape.

    Example::

        from pyvqnet.xtensor import ones

        x = ones([2, 3])
        print(x)

        # [
        # [1., 1., 1.],
        # [1., 1., 1.]
        # ]

ones_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.ones_like(t: pyvqnet.xtensor.XTensor)

    Return one-tensor with the same shape as the input XTensor.

    :param t: input XTensor

    :return:  output XTensor


    Example::

        
        from pyvqnet.xtensor import XTensor,ones_like
        t = XTensor([1, 2, 3])
        x = ones_like(t)
        print(x)

        # [1., 1., 1.]


full
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.full(shape, value, device=None, dtype=None)

    Create a XTensor of the specified shape and fill it with value.

    :param shape: shape of the XTensor to create
    :param value: value to fill the XTensor with.
    :param device: device to use,default = 0 ,use cpu device.
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output XTensor 

    Example::

        
        from pyvqnet.xtensor import XTensor,full
        shape = [2, 3]
        value = 42
        t = full(shape, value)
        print(t)
        # [
        # [42., 42., 42.],
        # [42., 42., 42.]
        # ]


full_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.full_like(t, value)

    Create a XTensor of the specified shape and fill it with value.

    :param t:  input Qtensor
    :param value: value to fill the XTensor with.

    :return: output XTensor

    Example::

        
        from pyvqnet.xtensor import XTensor,full_like,randu
        a =  randu([3,5])
        value = 42
        t =  full_like(a, value)
        print(t)
        # [
        # [42., 42., 42., 42., 42.],    
        # [42., 42., 42., 42., 42.],    
        # [42., 42., 42., 42., 42.]     
        # ]
        

zeros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.zeros(shape, device=None,dtype=None)

    Return zero-tensor of the input shape.

    :param shape: shape of tensor
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output XTensor
    Example::

        
        from pyvqnet.xtensor import XTensor,zeros
        t = zeros([2, 3, 4])
        print(t)
        # [
        # [[0., 0., 0., 0.],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]],
        # [[0., 0., 0., 0.],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]]
        # ]
        

zeros_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.zeros_like(t: pyvqnet.xtensor.XTensor)

    Return zero-tensor with the same shape as the input XTensor.

    :param t: input XTensor

    :return:  output XTensor

    Example::

        
        from pyvqnet.xtensor import XTensor,zeros_like
        t = XTensor([1, 2, 3])
        x = zeros_like(t)
        print(x)

        # [0., 0., 0.]
        


arange
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.arange(start, end, step=1, device=None,dtype=None)

    Create a 1D XTensor with evenly spaced values within a given interval.

    :param start: start of interval
    :param end: end of interval
    :param step: spacing between values
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output XTensor


    Example::

        from pyvqnet.xtensor import XTensor
        t =  arange(2, 30, 4)
        print(t)

        # [ 2.,  6., 10., 14., 18., 22., 26.]
        

linspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.linspace(start, end, num, device=None,dtype=None)

    Create a 1D XTensor with evenly spaced values within a given interval.

    :param start: starting value
    :param end: end value
    :param nums: number of samples to generate
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output XTensor


    Example::

        
        from pyvqnet.xtensor import XTensor,linspace
        start, stop, num = -2.5, 10, 10
        t = linspace(start, stop, num)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10.]

logspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.logspace(start, end, num, base, device=None,dtype=None)

    Create a 1D XTensor with evenly spaced values on a log scale.

    :param start: ``base ** start`` is the starting value
    :param end: ``base ** end`` is the final value of the sequence
    :param nums: number of samples to generate
    :param base: the base of the log space
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output XTensor


    Example::

        from pyvqnet.xtensor import XTensor,logspace
        start, stop, steps, base = 0.1, 1.0, 5, 10.0
        t = logspace(start, stop, steps, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10.]
        

eye
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.eye(size, offset: int = 0, device=None,dtype=None)

    Create a size x size XTensor with ones on the diagonal and zeros
    elsewhere.

    :param size: size of the (square) XTensor to create
    :param offset: Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output XTensor


    Example::

        import pyvqnet.xtensor as tensor
        size = 3
        t = tensor.eye(size)
        print(t)

        # [
        # [1., 0., 0.],
        # [0., 1., 0.],
        # [0., 0., 1.]
        # ]
        

diag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.diag(t, k: int = 0)

    Select diagonal elements or construct a diagonal XTensor.

    If input is 2-D XTensor,returns a new tensor which is the same as this one, except that
    elements other than those in the selected diagonal are set to zero.

    If v is a 1-D XTensor, return a 2-D XTensor with v on the k-th diagonal.

    :param t: input XTensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: output XTensor

    Example::

        
        from pyvqnet.xtensor import XTensor,diag
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = XTensor(a)
        for k in range(-3, 4):
            u = diag(t,k=k)
            print(u)


        # [[ 0.  0.  0.  0.]
        #  [ 0.  0.  0.  0.]
        #  [ 0.  0.  0.  0.]
        #  [12.  0.  0.  0.]]
        # [[ 0.  0.  0.  0.]
        #  [ 0.  0.  0.  0.]
        #  [ 8.  0.  0.  0.]
        #  [ 0. 13.  0.  0.]]
        # [[ 0.  0.  0.  0.]
        #  [ 4.  0.  0.  0.]
        #  [ 0.  9.  0.  0.]
        #  [ 0.  0. 14.  0.]]
        # [[ 0.  0.  0.  0.]
        #  [ 0.  5.  0.  0.]
        #  [ 0.  0. 10.  0.]
        #  [ 0.  0.  0. 15.]]
        # [[ 0.  1.  0.  0.]
        #  [ 0.  0.  6.  0.]
        #  [ 0.  0.  0. 11.]
        #  [ 0.  0.  0.  0.]]
        # [[0. 0. 2. 0.]
        #  [0. 0. 0. 7.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]]
        # [[0. 0. 0. 3.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]]


randu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.randu(shape, min=0.0,max=1.0, device=None, dtype=None)

    Create a XTensor with uniformly distributed random values.

    :param shape: shape of the XTensor to create
    :param min: minimum value of uniform distribution,default: 0.
    :param max: maximum value of uniform distribution,default: 1.
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param requires_grad: should tensor's gradient be tracked, defaults to False
    :return: output XTensor


    Example::

        
        from pyvqnet.xtensor import XTensor, randu
        shape = [2, 3]
        t =  randu(shape)
        print(t)

        # [
        # [0.0885886, 0.9570093, 0.8304565],
        # [0.6055251, 0.8721224, 0.1927866]
        # ]
        

randn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.randn(shape, mean=0.0,std=1.0, device=None, dtype=None)

    Create a XTensor with normally distributed random values.

    :param shape: shape of the XTensor to create
    :param mean: mean value of normally distribution,default: 0.
    :param std: standard variance value of normally distribution,default: 1.
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param requires_grad: should tensor's gradient be tracked, defaults to False
    :return: output XTensor

    Example::

        
        from pyvqnet.xtensor import XTensor,randn
        shape = [2, 3]
        t = randn(shape)
        print(t)

        # [
        # [-0.9529880, -0.4947567, -0.6399882],
        # [-0.6987777, -0.0089036, -0.5084590]
        # ]


multinomial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.multinomial(t, num_samples)

    Returns a Tensor where each row contains num_samples indexed samples.
    From the multinomial probability distribution located in the corresponding row of the tensor input.

    :param t: Input probability distribution。
    :param num_samples: numbers of sample。

    :return:
        output sample index

    Examples::

        import pyvqnet.xtensor as tensor
        weights = tensor.XTensor([0.1,10, 3, 1]) 
        idx = tensor.multinomial(weights,3)
        print(idx)

        weights = tensor.XTensor([0,10, 3, 2.2,0.0]) 
        idx = tensor.multinomial(weights,3)
        print(idx)

        # [1 0 3]
        # [1 3 2]

triu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.triu(t, diagonal=0)

    Returns the upper triangular matrix of input t, with the rest set to 0.

    :param t: input a XTensor
    :param diagonal: The Offset default =0. Main diagonal is 0, positive is offset up,and negative is offset down

    :return: output a XTensor

    Examples::

        import pyvqnet.xtensor as tensor
        
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
        u = tensor.triu(a, 1)
        print(u)
        # [
        # [[0., 2., 3., 4., 5.],       
        #  [0., 0., 8., 9., 10.],      
        #  [0., 0., 0., 14., 15.],     
        #  [0., 0., 0., 0., 20.],      
        #  [0., 0., 0., 0., 0.],       
        #  [0., 0., 0., 0., 0.]],      
        # [[0., 32., 33., 34., 35.],   
        #  [0., 0., 38., 39., 40.],    
        #  [0., 0., 0., 44., 45.],     
        #  [0., 0., 0., 0., 50.],      
        #  [0., 0., 0., 0., 0.],       
        #  [0., 0., 0., 0., 0.]]       
        # ]

tril
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.tril(t, diagonal=0)

    Returns the lower triangular matrix of input t, with the rest set to 0.

    :param t: input a XTensor
    :param diagonal: The Offset default =0. Main diagonal is 0, positive is offset up,and negative is offset down

    :return: output a XTensor

    Examples::

        import pyvqnet.xtensor as tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([12, 5])
        u = tensor.tril(a, 1)
        print(u)
        # [
        # [1., 2., 0., 0., 0.],      
        #  [6., 7., 8., 0., 0.],     
        #  [11., 12., 13., 14., 0.], 
        #  [16., 17., 18., 19., 20.],
        #  [21., 22., 23., 24., 25.],
        #  [26., 27., 28., 29., 30.],
        #  [31., 32., 33., 34., 35.],
        #  [36., 37., 38., 39., 40.],
        #  [41., 42., 43., 44., 45.],
        #  [46., 47., 48., 49., 50.],
        #  [51., 52., 53., 54., 55.],
        #  [56., 57., 58., 59., 60.]
        # ]

Math Function
-----------------------------


floor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.floor(t)

    Return a new XTensor with the floor of the elements of input, the largest integer less than or equal to each element.

    :param t: input Qtensor
    :return: output XTensor

    Example::


        import pyvqnet.xtensor as tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2., -2., -2., -2., -1., -1., -1., -1., 0., 0., 0., 0., 1., 1., 1., 1.]

ceil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.ceil(t)

    Return a new XTensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.

    :param t: input Qtensor
    :return: output XTensor

    Example::

        import pyvqnet.xtensor as tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2., -1., -1., -1., -1., -0., -0., -0., 0., 1., 1., 1., 1., 2., 2., 2.]

round
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.round(t)

    Round XTensor values to the nearest integer.

    :param t: input XTensor
    :return: output XTensor

    Example::

        import pyvqnet.xtensor as tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2., -2., -1., -1., -0., -0., 0., 1., 1., 2.]

sort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.sort(t, axis=None, descending=False, stable=True)

    Sort XTensor along the axis

    :param t: input XTensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: output XTensor

    Example::

        
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = tensor.xtensor(a)
        AA = tensor.sort(A,1,False)
        print(AA)

        # [
        # [0., 1., 2., 4., 6., 7., 8., 8.],
        # [2., 5., 5., 8., 9., 9., 9., 9.],
        # [1., 2., 5., 5., 5., 6., 7., 7.]
        # ]

argsort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.argsort(t, axis = None, descending=False, stable=True)

    Return an array of indices of the same shape as input that index data along the given axis in sorted order.

    :param t: input XTensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: output XTensor

    Example::

        
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8)
        A =tensor.XTensor(a)
        bb = tensor.argsort(A,1,False)
        print(bb)

        # [
        # [4., 0., 1., 7., 5., 3., 2., 6.], 
        #  [3., 0., 7., 6., 2., 1., 4., 5.],
        #  [4., 7., 5., 0., 2., 1., 3., 6.]
        # ]

topK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.topK(t, k, axis=-1, if_descent=True)

    Returns the k largest elements of the input tensor along the given axis.

    If if_descent is False，then return k smallest elements.

    :param t: input a XTensor
    :param k: numbers of largest elements or smallest elements
    :param axis: sort axis,default = -1，the last axis
    :param if_descent: sort order,defaults to True

    :return: A new XTensor
    Examples::

        import pyvqnet.xtensor as tensor
        from pyvqnet.xtensor import XTensor
        x = XTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x = x.reshape([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.topK(x, 3, 1)
        print(y)
        # [
        # [[[24., 15.]],
        # [[15., 13.]],
        # [[11., 8.]]],
        # [[[24., 13.]],
        # [[15., 11.]],
        # [[7., 8.]]]
        # ]

argtopK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.argtopK(t, k, axis=-1, if_descent=True)

    Return the index of the k largest elements along the given axis of the input tensor.

    If if_descent is False，then return the index of k smallest elements.

    :param t: input a XTensor
    :param k: numbers of largest elements or smallest elements
    :param axis: sort axis,default = -1，the last axis
    :param if_descent: sort order,defaults to True

    :return: A new XTensor

    Examples::

        import pyvqnet.xtensor as tensor
        from pyvqnet.xtensor import XTensor
        x = XTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x = x.reshape([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.argtopK(x, 3, 1)
        print(y)
        # [
        # [[[0., 4.]],
        # [[1., 0.]],
        # [[3., 2.]]],
        # [[[0., 0.]],
        # [[1., 4.]],
        # [[3., 2.]]]
        # ]


add
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.add(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Element-wise adds two QTensors, equivalent to t1 + t2.

    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor

    Example::

        
        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.add(t1, t2)
        print(x)

        # [5., 7., 9.]

sub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.sub(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Element-wise subtracts two QTensors,  equivalent to t1 - t2.


    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.sub(t1, t2)
        print(x)

        # [-3., -3., -3.]

mul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.mul(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Element-wise multiplies two QTensors, equivalent to t1 * t2.

    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.mul(t1, t2)
        print(x)

        # [4., 10., 18.]

divide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.divide(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Element-wise divides two QTensors, equivalent to t1 / t2.


    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.divide(t1, t2)
        print(x)

        # [0.2500000, 0.4000000, 0.5000000]

sums
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.sums(t: pyvqnet.xtensor.XTensor, axis: int = None, keepdims=False)

    Sums all the elements in XTensor along given axis.if axis = None, sums all the elements in XTensor. 

    :param t: input XTensor
    :param axis:  axis used to sums, defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21.]

cumsum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.cumsum(t, axis=-1)

    Return the cumulative sum of input elements in the dimension axis.

    :param t:  the input XTensor
    :param axis:  Calculation of the axis,defaults to -1,use the last axis

    :return:  output XTensor.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.cumsum(t,-1)
        print(x)
        """
        [[ 1.  3.  6.]
        [ 4.  9. 15.]]
        <XTensor 2x3 cpu(0) kfloat32>
        """


mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.mean(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    Obtain the mean values in the XTensor along the axis.

    :param t:  the input XTensor.
    :param axis: the dimension to reduce.
    :param keepdims:  whether the output XTensor has dim retained or not, defaults to False.
    :return: returns the mean value of the input XTensor.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, axis=1)
        print(x)

        # [2. 5.]

median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.median(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    Obtain the median value in the XTensor.

    :param t: the input XTensor
    :param axis:  An axis for averaging,defaults to None
    :param keepdims:  whether the output XTensor has dim retained or not, defaults to False

    :return: Return the median of the values in input or XTensor.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, axis=1)
        print(x)
        #[2.5]
        a = XTensor([[1.5219, -1.5212,  0.2202]])
        median_a = tensor.median(a)
        print(median_a)

        # [0.2202000]

        b = XTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        median_b = tensor.median(b,1, False)
        print(median_b)

        # [-0.3982000, 0.2269999, 0.2487999, 0.4742000]

std
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.std(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False, unbiased=True)

    Obtain the standard variance value in the XTensor.


    :param t:  the input XTensor
    :param axis:  the axis used to calculate the standard deviation,defaults to None
    :param keepdims:  whether the output XTensor has dim retained or not, defaults to False
    :param unbiased:  whether to use Bessel’s correction,default true
    :return: Return the standard variance of the values in input or XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[-0.8166, -1.3802, -0.3560]])
        std_a = tensor.std(a)
        print(std_a)

        # [0.5129624]

        b = XTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        std_b = tensor.std(b, 1, False, False)
        print(std_b)

        # [0.6593542, 0.5583112, 0.3206565, 1.1103367]

var
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.var(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False, unbiased=True)

    Obtain the variance in the XTensor.


    :param t:  the input XTensor.
    :param axis:  The axis used to calculate the variance,defaults to None
    :param keepdims:  whether the output XTensor has dim retained or not, defaults to False.
    :param unbiased:  whether to use Bessel’s correction,default true.


    :return: Obtain the variance in the XTensor.
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

        # [0.2631305]

matmul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.matmul(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Matrix multiplications of two 2d , 3d , 4d matrix.

    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor

    Example::

        import pyvqnet.xtensor as tensor
        t1 = tensor.ones([2,3])
        t1.requires_grad = True
        t2 = tensor.ones([3,4])
        t2.requires_grad = True
        t3  = tensor.matmul(t1,t2)
        t3.backward(tensor.ones_like(t3))
        print(t1.grad)

        # [
        # [4., 4., 4.],
        #  [4., 4., 4.]
        # ]

        print(t2.grad)

        # [
        # [2., 2., 2., 2.],
        #  [2., 2., 2., 2.],
        #  [2., 2., 2., 2.]
        # ]

kron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.kron(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Computes the Kronecker product of ``t1`` and ``t2``, expressed in :math:`\otimes` . If ``t1`` is a :math:`(a_0 \times a_1 \times \dots \times a_n)` tensor and ``t2`` is a :math:`(b_0 \times b_1 \times \dots \ times b_n)` tensor, the result will be :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` tensor with the following entries:
    
    .. math::
          (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
              \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

    where :math:`k_t = i_t * b_t + j_t` is :math:`0 \leq t \leq n`.
    If one tensor has fewer dimensions than the other, it will be unpacked until it has the same dimensionality.

    :param t1: The first XTensor.
    :param t2: The second XTensor.
    
    :return: Output XTensor .

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = tensor.arange(1,1+ 24).reshape([2,1,2,3,2])
        b = tensor.arange(1,1+ 24).reshape([6,4])
        c = tensor.kron(a,b)
        print(c)

        # [[[[[  1.   2.   3.   4.   2.   4.   6.   8.]
        #     [  5.   6.   7.   8.  10.  12.  14.  16.]
        #     [  9.  10.  11.  12.  18.  20.  22.  24.]
        #     [ 13.  14.  15.  16.  26.  28.  30.  32.]
        #     [ 17.  18.  19.  20.  34.  36.  38.  40.]
        #     [ 21.  22.  23.  24.  42.  44.  46.  48.]
        #     [  3.   6.   9.  12.   4.   8.  12.  16.]
        #     [ 15.  18.  21.  24.  20.  24.  28.  32.]
        #     [ 27.  30.  33.  36.  36.  40.  44.  48.]
        #     [ 39.  42.  45.  48.  52.  56.  60.  64.]
        #     [ 51.  54.  57.  60.  68.  72.  76.  80.]
        #     [ 63.  66.  69.  72.  84.  88.  92.  96.]
        #     [  5.  10.  15.  20.   6.  12.  18.  24.]
        #     [ 25.  30.  35.  40.  30.  36.  42.  48.]
        #     [ 45.  50.  55.  60.  54.  60.  66.  72.]
        #     [ 65.  70.  75.  80.  78.  84.  90.  96.]
        #     [ 85.  90.  95. 100. 102. 108. 114. 120.]
        #     [105. 110. 115. 120. 126. 132. 138. 144.]]

        #    [[  7.  14.  21.  28.   8.  16.  24.  32.]
        #     [ 35.  42.  49.  56.  40.  48.  56.  64.]
        #     [ 63.  70.  77.  84.  72.  80.  88.  96.]
        #     [ 91.  98. 105. 112. 104. 112. 120. 128.]
        #     [119. 126. 133. 140. 136. 144. 152. 160.]
        #     [147. 154. 161. 168. 168. 176. 184. 192.]
        #     [  9.  18.  27.  36.  10.  20.  30.  40.]
        #     [ 45.  54.  63.  72.  50.  60.  70.  80.]
        #     [ 81.  90.  99. 108.  90. 100. 110. 120.]
        #     [117. 126. 135. 144. 130. 140. 150. 160.]
        #     [153. 162. 171. 180. 170. 180. 190. 200.]
        #     [189. 198. 207. 216. 210. 220. 230. 240.]
        #     [ 11.  22.  33.  44.  12.  24.  36.  48.]
        #     [ 55.  66.  77.  88.  60.  72.  84.  96.]
        #     [ 99. 110. 121. 132. 108. 120. 132. 144.]
        #     [143. 154. 165. 176. 156. 168. 180. 192.]
        #     [187. 198. 209. 220. 204. 216. 228. 240.]
        #     [231. 242. 253. 264. 252. 264. 276. 288.]]]]



        #  [[[[ 13.  26.  39.  52.  14.  28.  42.  56.]
        #     [ 65.  78.  91. 104.  70.  84.  98. 112.]
        #     [117. 130. 143. 156. 126. 140. 154. 168.]
        #     [169. 182. 195. 208. 182. 196. 210. 224.]
        #     [221. 234. 247. 260. 238. 252. 266. 280.]
        #     [273. 286. 299. 312. 294. 308. 322. 336.]
        #     [ 15.  30.  45.  60.  16.  32.  48.  64.]
        #     [ 75.  90. 105. 120.  80.  96. 112. 128.]
        #     [135. 150. 165. 180. 144. 160. 176. 192.]
        #     [195. 210. 225. 240. 208. 224. 240. 256.]
        #     [255. 270. 285. 300. 272. 288. 304. 320.]
        #     [315. 330. 345. 360. 336. 352. 368. 384.]
        #     [ 17.  34.  51.  68.  18.  36.  54.  72.]
        #     [ 85. 102. 119. 136.  90. 108. 126. 144.]
        #     [153. 170. 187. 204. 162. 180. 198. 216.]
        #     [221. 238. 255. 272. 234. 252. 270. 288.]
        #     [289. 306. 323. 340. 306. 324. 342. 360.]
        #     [357. 374. 391. 408. 378. 396. 414. 432.]]

        #    [[ 19.  38.  57.  76.  20.  40.  60.  80.]
        #     [ 95. 114. 133. 152. 100. 120. 140. 160.]
        #     [171. 190. 209. 228. 180. 200. 220. 240.]
        #     [247. 266. 285. 304. 260. 280. 300. 320.]
        #     [323. 342. 361. 380. 340. 360. 380. 400.]
        #     [399. 418. 437. 456. 420. 440. 460. 480.]
        #     [ 21.  42.  63.  84.  22.  44.  66.  88.]
        #     [105. 126. 147. 168. 110. 132. 154. 176.]
        #     [189. 210. 231. 252. 198. 220. 242. 264.]
        #     [273. 294. 315. 336. 286. 308. 330. 352.]
        #     [357. 378. 399. 420. 374. 396. 418. 440.]
        #     [441. 462. 483. 504. 462. 484. 506. 528.]
        #     [ 23.  46.  69.  92.  24.  48.  72.  96.]
        #     [115. 138. 161. 184. 120. 144. 168. 192.]
        #     [207. 230. 253. 276. 216. 240. 264. 288.]
        #     [299. 322. 345. 368. 312. 336. 360. 384.]
        #     [391. 414. 437. 460. 408. 432. 456. 480.]
        #     [483. 506. 529. 552. 504. 528. 552. 576.]]]]]


reciprocal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.reciprocal(t)

    Compute the element-wise reciprocal of the XTensor.

    :param t: input XTensor
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1., 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.sign(t)

    Return a new XTensor with the signs of the elements of input.The sign function returns -1 if t < 0, 0 if t==0, 1 if t > 0.

    :param t: input XTensor
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1., -1., -1., -1., -1., 0., 1., 1., 1., 1.]

neg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.neg(t: pyvqnet.xtensor.XTensor)

    Unary negation of XTensor elements.

    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1., -2., -3.]

trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.trace(t, k: int = 0)

    Return the sum of the elements of the diagonal of the input 2-D matrix.

    :param t: input 2-D XTensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: the sum of the elements of the diagonal of the input 2-D matrix


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = tensor.randn([4,4])
        for k in range(-3, 4):
            u=tensor.trace(t,k=k)
            print(u)

        # 0.07717618346214294
        # -1.9287869930267334
        # 0.6111435890197754
        # 2.8094992637634277
        # 0.6388946771621704
        # -1.3400784730911255
        # 0.26980453729629517

exp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.exp(t: pyvqnet.xtensor.XTensor)

    Applies exponential function to all the elements of the input XTensor.

    :param t: input XTensor
    :return:  output XTensor


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.exp(t)
        print(x)

        # [2.7182817, 7.3890562, 20.0855369]

acos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.acos(t: pyvqnet.xtensor.XTensor)

    Compute the element-wise inverse cosine of the XTensor.

    :param t: input XTensor
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.arange(36).reshape(2,6,3).astype(np.float32)
        a =a/100
        A = XTensor(a)
        y = tensor.acos(A)
        print(y)

        # [
        # [[1.5707964, 1.5607961, 1.5507950],
        #  [1.5407919, 1.5307857, 1.5207754],
        #  [1.5107603, 1.5007390, 1.4907107],
        #  [1.4806744, 1.4706289, 1.4605733],
        #  [1.4505064, 1.4404273, 1.4303349],
        #  [1.4202280, 1.4101057, 1.3999666]],
        # [[1.3898098, 1.3796341, 1.3694384],
        #  [1.3592213, 1.3489819, 1.3387187],
        #  [1.3284305, 1.3181161, 1.3077742],
        #  [1.2974033, 1.2870022, 1.2765695],
        #  [1.2661036, 1.2556033, 1.2450669],
        #  [1.2344928, 1.2238795, 1.2132252]]
        # ]

asin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.asin(t: pyvqnet.xtensor.XTensor)

    Compute the element-wise inverse sine of the XTensor.

    :param t: input XTensor
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0., 0.5235988]

atan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.atan(t: pyvqnet.xtensor.XTensor)

    Compute the element-wise inverse tangent of the XTensor.

    :param t: input XTensor
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0., 0.4636476]

sin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.sin(t: pyvqnet.xtensor.XTensor)

    Applies sine function to all the elements of the input XTensor.

    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.sin(t)
        print(x)

        # [0.8414709, 0.9092974, 0.1411200]

cos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.cos(t: pyvqnet.xtensor.XTensor)

    Applies cosine function to all the elements of the input XTensor.


    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.cos(t)
        print(x)

        # [0.5403022, -0.4161468, -0.9899924]

tan 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.tan(t: pyvqnet.xtensor.XTensor)

    Applies tangent function to all the elements of the input XTensor.


    :param t: input XTensor
    :return:  output XTensor
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.tan(t)
        print(x)

        # [1.5574077, -2.1850397, -0.1425465]

tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.tanh(t: pyvqnet.xtensor.XTensor)

    Applies hyperbolic tangent function to all the elements of the input XTensor.

    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.tanh(t)
        print(x)

        # [0.7615941, 0.9640275, 0.9950547]

sinh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.sinh(t: pyvqnet.xtensor.XTensor)

    Applies hyperbolic sine function to all the elements of the input XTensor.


    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.sinh(t)
        print(x)

        # [1.1752011, 3.6268603, 10.0178747]

cosh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.cosh(t: pyvqnet.xtensor.XTensor)

    Applies hyperbolic cosine function to all the elements of the input XTensor.


    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.cosh(t)
        print(x)

        # [1.5430806, 3.7621955, 10.0676622]

power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.power(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Raises first XTensor to the power of second XTensor.

    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 4, 3])
        t2 = XTensor([2, 5, 6])
        x = tensor.power(t1, t2)
        print(x)

        # [1., 1024., 729.]

abs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.abs(t: pyvqnet.xtensor.XTensor)

    Applies abs function to all the elements of the input XTensor.

    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1., 2., 3.]

log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.log(t: pyvqnet.xtensor.XTensor)

    Applies log (ln) function to all the elements of the input XTensor.

    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0., 0.6931471, 1.0986123]

log_softmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.log_softmax(t, axis=-1)

    Sequentially calculate the results of the softmax function and the log function on the axis axis.

    :param t: input XTensor .
    :param axis: The axis used to calculate softmax, the default is -1.

    :return: Output QTensor。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        output = tensor.arange(1,13).reshape([3,2,2])
        t = tensor.log_softmax(output,1)
        print(t)
        # [
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]]
        # ]

sqrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.sqrt(t: pyvqnet.xtensor.XTensor)

    Applies sqrt function to all the elements of the input XTensor.


    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1., 1.4142135, 1.7320507]

square
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.square(t: pyvqnet.xtensor.XTensor)

    Applies square function to all the elements of the input XTensor.


    :param t: input XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)

        # [1., 4., 9.]

frobenius_norm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.frobenius_norm(t: XTensor, axis: int = None, keepdims=False):

    Computes the F-norm of the tensor on the input XTensor along the axis set by axis ,
    if axis is None, returns the F-norm of all elements.

    :param t: Inpout XTensor .
    :param axis: The axis used to find the F norm, the default is None.
    :param keepdims: Whether the output tensor preserves the reduced dimensionality. The default is False.
    :return: Output a XTensor or F-norm value.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                    [[13., 14., 15.], [16., 17., 18.]]])
        t.requires_grad = True
        result = tensor.frobenius_norm(t, -2, False)
        print(result)
        # [
        # [4.1231055, 5.3851647, 6.7082038],
        #  [12.2065554, 13.6014709, 15.],
        #  [20.6155281, 22.0227146, 23.4307499]
        # ]


Logical function
--------------------------

maximum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.maximum(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Element-wise maximum of two tensor.


    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([6, 4, 3])
        t2 = XTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)
        print(x)

        # [6., 5., 7.]

minimum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.minimum(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Element-wise minimum of two tensor.


    :param t1: first XTensor
    :param t2: second XTensor
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([6, 4, 3])
        t2 = XTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)
        print(x)

        # [2., 4., 3.]

min
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.min(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    Return min elements of the input XTensor alongside given axis.
    if axis == None, return the min value of all elements in tensor.

    :param t: input XTensor
    :param axis: axis used for min, defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.min(t, axis=1, keepdims=True)
        print(x)

        # [
        # [1.],
        #  [4.]
        # ]

max
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.max(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    Return max elements of the input XTensor alongside given axis.
    if axis == None, return the max value of all elements in tensor.

    :param t: input XTensor
    :param axis: axis used for max, defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.max(t, axis=1, keepdims=True)
        print(x)

        # [
        # [3.],
        #  [6.]
        # ]

clip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.clip(t: pyvqnet.xtensor.XTensor, min_val, max_val)

    Clips input XTensor to minimum and maximum value.

    :param t: input XTensor
    :param min_val:  minimum value
    :param max_val:  maximum value
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3., 4., 6.]


where
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.where(condition: pyvqnet.xtensor.XTensor, t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)


    Return elements chosen from x or y depending on condition.

    :param condition: condition tensor,need to have data type of kbool.
    :param t1: XTensor from which to take elements if condition is met, defaults to None
    :param t2: XTensor from which to take elements if condition is not met, defaults to None
    :return: output XTensor


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)
        print(x)

        # [1., 5., 6.]

nonzero
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.nonzero(t)

    Return a XTensor containing the indices of nonzero elements.

    :param t: input XTensor
    :return: output XTensor contains indices of nonzero elements.

    Example::
    
        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]])
        t = tensor.nonzero(t)
        print(t)
        # [
        # [0., 0.],
        # [1., 1.],
        # [2., 2.],
        # [3., 3.]
        # ]

isfinite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.isfinite(t)

    Test element-wise for finiteness (not infinity or not Not a Number).

    :param t: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        #[ True False  True False False]

isinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.isinf(t)

    Test element-wise for positive or negative infinity.

    :param t: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [False  True False  True False]

isnan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.isnan(t)

    Test element-wise for Nan.

    :param t: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [False False False False  True]

isneginf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.isneginf(t)

    Test element-wise for negative infinity.

    :param t: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [False False False  True False]

isposinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.isposinf(t)

    Test element-wise for positive infinity.

    :param t: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [False  True False False False]

logical_and
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.logical_and(t1, t2)

    Compute the truth value of ``t1`` and ``t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        b = XTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)
        print(flag)

        # [False False  True False]

logical_or
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.logical_or(t1, t2)

    Compute the truth value of ``t1 or t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        b = XTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)
        print(flag)

        # [ True  True  True False]

logical_not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.logical_not(t)

    Compute the truth value of ``not t`` element-wise.

    :param t: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [ True False False  True]

logical_xor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.logical_xor(t1, t2)

    Compute the truth value of ``t1 xor t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor

    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        b = XTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)
        print(flag)

        # [ True  True False False]

greater
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.greater(t1, t2)

    Return the truth value of ``t1 > t2`` element-wise.


    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)
        print(flag)

        # [[False  True]
        #  [False False]]

greater_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.greater_equal(t1, t2)

    Return the truth value of ``t1 >= t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)
        print(flag)

        #[[ True  True]
        # [False  True]]

less
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.less(t1, t2)

    Return the truth value of ``t1 < t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)
        print(flag)

        #[[False False]
        # [ True False]]

less_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.less_equal(t1, t2)

    Return the truth value of ``t1 <= t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)
        print(flag)


        # [[ True False]
        #  [ True  True]]

equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.equal(t1, t2)

    Return the truth value of ``t1 == t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)
        print(flag)

        #[[ True False]
        # [False  True]]

not_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.not_equal(t1, t2)

    Return the truth value of ``t1 != t2`` element-wise.

    :param t1: input XTensor
    :param t2: input XTensor
    :return: Output XTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)

        #[[False  True]
        # [ True False]]

Matrix operations
--------------------------

broadcast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.broadcast(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    Subject to certain restrictions, smaller arrays are placed throughout larger arrays so that they have compatible shapes. This interface can perform automatic differentiation on input parameter tensors.

    Reference https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: input XTensor 1
    :param t2: input XTensor 2

    :return t11: with new broadcast shape t1.
    :return t22: t2 with new broadcast shape.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)

        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([1])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)

        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([2, 1, 4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [2, 5, 4]
        # [2, 5, 4]


concatenate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.concatenate(args: list, axis=0)

    Concatenate the input XTensor along the axis and return a new XTensor.

    :param args: list consist of input QTensors
    :param axis: dimension to concatenate. Has to be between 0 and the number of dimensions of concatenate tensors.
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        x = XTensor([[1, 2, 3],[4,5,6]])
        y = 1-x
        x = tensor.concatenate([x,y],1)
        print(x)

        # [
        # [1., 2., 3., 0., -1., -2.],
        # [4., 5., 6., -3., -4., -5.]
        # ]
        
        

stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.stack(XTensors: list, axis=0) 

    Join a sequence of arrays along a new axis,return a new XTensor.

    :param QTensors: list contains QTensors
    :param axis: dimension to insert. Has to be between 0 and the number of dimensions of stacked tensors. 
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t11 = XTensor(a)
        t22 = XTensor(a)
        t33 = XTensor(a)
        rlt1 = tensor.stack([t11,t22,t33],2)
        print(rlt1)
        
        # [
        # [[0., 0., 0.],
        #  [1., 1., 1.],
        #  [2., 2., 2.],
        #  [3., 3., 3.]],
        # [[4., 4., 4.],
        #  [5., 5., 5.],
        #  [6., 6., 6.],
        #  [7., 7., 7.]],
        # [[8., 8., 8.],
        #  [9., 9., 9.],
        #  [10., 10., 10.],
        #  [11., 11., 11.]]
        # ]
                

permute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.permute(t: pyvqnet.xtensor.XTensor, *axes)

    Reverse or permute the axes of an array.if dims = None, revsers the dim.

    :param t: input XTensor
    :param dim: the new order of the dimensions (list of integers)
    :return: output XTensor


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = XTensor(a)
        tt = tensor.permute(t,[2,0,1])
        print(tt)
        
        # [
        # [[0., 3.],
        #  [6., 9.]],
        # [[1., 4.],
        #  [7., 10.]],
        # [[2., 5.],
        #  [8., 11.]]
        # ]
                
        

transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.transpose(t: pyvqnet.xtensor.XTensor, *axes)

    Transpose the axes of an array.if dim = None, reverse the dim. This function is same as permute.

    :param t: input XTensor
    :param dim: the new order of the dimensions (list of integers)
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = XTensor(a)
        tt = tensor.transpose(t,[2,0,1])
        print(tt)

        # [
        # [[0., 3.],
        #  [6., 9.]],
        # [[1., 4.],
        #  [7., 10.]],
        # [[2., 5.],
        #  [8., 11.]]
        # ]
        

tile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.tile(t: pyvqnet.xtensor.XTensor, reps: list)


    Construct a XTensor by repeating XTensor the number of times given by reps.

    If reps has length d, the result XTensor will have dimension of max(d, t.ndim).

    If t.ndim < d, t is expanded to be d-dimensional by inserting new axes from start dimension.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication.

    If t.ndim > d, reps is expanded to t.ndim by inserting 1’s to it.

    Thus for an t of shape (2, 3, 4, 5), a reps of (4, 3) is treated as (1, 1, 4, 3).

    :param t: input XTensor
    :param reps: the number of repetitions per dimension.
    :return: a new XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        a = np.arange(6).reshape(2,3).astype(np.float32)
        A = XTensor(a)
        reps = [2,2]
        B = tensor.tile(A,reps)
        print(B)

        # [
        # [0., 1., 2., 0., 1., 2.],
        # [3., 4., 5., 3., 4., 5.],
        # [0., 1., 2., 0., 1., 2.],
        # [3., 4., 5., 3., 4., 5.]
        # ]
        

squeeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.squeeze(t: pyvqnet.xtensor.XTensor, axis: int = - 1)

    Remove axes of length one .

    :param t: input XTensor
    :param axis: squeeze axis,if axis = -1 ,squeeze all the dimensions that have size of 1.
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = XTensor(a)
        AA = tensor.squeeze(A,0)
        print(AA)

        # [
        # [0.],
        # [1.],
        # [2.],
        # [3.],
        # [4.],
        # [5.]
        # ]
        

unsqueeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.unsqueeze(t: pyvqnet.xtensor.XTensor, axis: int = 0)

    Return a new XTensor with a dimension of size one inserted at the specified position.

    :param t: input XTensor
    :param axis: unsqueeze axis,which will insert dimension.
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = XTensor(a)
        AA = tensor.unsqueeze(A,1)
        print(AA)

        # [
        # [[[[[0., 1., 2.],
        #  [3., 4., 5.],
        #  [6., 7., 8.],
        #  [9., 10., 11.]]]]],
        # [[[[[12., 13., 14.],
        #  [15., 16., 17.],
        #  [18., 19., 20.],
        #  [21., 22., 23.]]]]]
        # ]
        

swapaxis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.swapaxis(t, axis1: int, axis2: int)

    Interchange two axes of an array.The given dimensions axis1 and axis2 are swapped.

    :param t: input XTensor
    :param axis1: First axis.
    :param axis2:  Destination position for the original axis. These must also be unique
    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = XTensor(a)
        AA = tensor.swapaxis(A, 2, 1)
        print(AA)

        # [
        # [[0., 4., 8.],
        #  [1., 5., 9.],
        #  [2., 6., 10.],
        #  [3., 7., 11.]],
        # [[12., 16., 20.],
        #  [13., 17., 21.],
        #  [14., 18., 22.],
        #  [15., 19., 23.]]
        # ]

masked_fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.masked_fill(t, mask, value)

    If mask == 1, fill with the specified value. The shape of the mask must be broadcastable from the shape of the input XTensor.

    :param t: input XTensor
    :param mask: A XTensor
    :param value: specified value
    :return:  A XTensor


    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = tensor.ones([2, 2, 2, 2])
        mask = np.random.randint(0, 2, size=4).reshape([2, 2])
        b = tensor.XTensor(mask==1)
        c = tensor.masked_fill(a, b, 13)
        print(c)
        # [
        # [[[1., 1.],  
        #  [13., 13.]],
        # [[1., 1.],   
        #  [13., 13.]]],
        # [[[1., 1.],
        #  [13., 13.]],
        # [[1., 1.],
        #  [13., 13.]]]
        # ]

flatten
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.flatten(t: pyvqnet.xtensor.XTensor, start: int = 0, end: int = - 1)

    Flatten XTensor from dim start to dim end.

    :param t: input XTensor
    :param start: dim start,default = 0,start from first dim.
    :param end: dim end,default = -1,end with last dim.
    :return:  output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1., 2., 3.]


reshape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.reshape(t: pyvqnet.xtensor.XTensor,new_shape)

    Change XTensor's shape, return a new shape XTensor

    :param t: input XTensor.
    :param new_shape: new shape

    :return: a new shape XTensor 。


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = XTensor(a)
        reshape_t = tensor.reshape(t, [C, R])
        print(reshape_t)
        # [
        # [0., 1., 2.],
        # [3., 4., 5.],
        # [6., 7., 8.],
        # [9., 10., 11.]
        # ]

flip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.flip(t, flip_dims)

    Reverses the XTensor along the specified axis, returning a new tensor.

    :param t: Input XTensor 。
    :param flip_dims: The axis or list of axes to flip.

    :return: Output XTensor 。
    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(1, 3 * 2 *2 * 2 + 1).reshape([3, 2, 2, 2])
        t.requires_grad = True
        y = tensor.flip(t, [0, -1])
        print(y)
        # [
        # [[[18., 17.], 
        #  [20., 19.]], 
        # [[22., 21.],  
        #  [24., 23.]]],
        # [[[10., 9.],  
        #  [12., 11.]], 
        # [[14., 13.],  
        #  [16., 15.]]],
        # [[[2., 1.],   
        #  [4., 3.]],   
        # [[6., 5.],    
        #  [8., 7.]]]   
        # ]

broadcast_to
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.broadcast_to(t, ref)


    Subject to certain constraints, the array t is "broadcast" to the reference shape so that they have compatible shapes.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: input XTensor
    :param ref: Reference shape.
    
    :return: The XTensor of the newly broadcasted t.

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        ref = [2,3,4]
        a = tensor。ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]


Utility functions
-----------------------------


to_xtensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.to_xtensor(x,device=None,dtype=None)

    Convert input numeric values or numpy.ndarray etc. to XTensor .

    :param x: integer, float, boolean, complex, or numpy.ndarray
    :param device: On which device to store, default: None, on CPU.
    :param dtype: The data type of the parameter, defaults: None, use the default data type: kfloat32, which represents a 32-bit floating point number.

    :return: output XTensor

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.to_xtensor(10.0)
        print(t)

        # [10.]
        

pad_sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    Pad a list of variable-length tensors with ``padding_value``. ``pad_sequence`` stacks lists of tensors along new dimensions and pads them to equal length.
    The input is a sequence of lists of size ``L x *``. L is variable length.

    :param qtensor_list: `list[XTensor]` - list of variable length sequences.
    :param batch_first: 'bool' - If true, the output will be ``batch size x longest sequence length x *``, otherwise ``longest sequence length x batch size x *``. Default: False.
    :param padding_value: 'float' - padding value. Default value: 0.

    :return:
         If batch_first is ``False``, the tensor size is ``batch size x longest sequence length x *``.
         Otherwise the size of the tensor is ``longest sequence length x batch size x *``.

    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([1, 2,3])
        c = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)

        print(y)
        # [
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]]],
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]]],
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]]]
        # ]


pad_packed_sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.pad_packed_sequence(sequence, batch_first=False, padding_value=0, total_length=None)

    Pad a batch of packed variable-length sequences. It is the inverse of `pack_pad_sequence`.
    When ``batch_first`` is True, it returns a tensor of shape ``B x T x *``, otherwise it returns ``T x B x *``.
    Where `T` is the longest sequence length and `B` is the batch size.

    :param sequence: 'XTensor' - the data to be processed.
    :param batch_first: 'bool' - If ``True``, batch will be the first dimension of the input. Default value: False.
    :param padding_value: 'bool' - padding value. Default: 0.
    :param total_length: 'bool' - If not ``None``, the output will be padded to length :attr:`total_length`. Default: None.
    :return:
        A tuple of tensors containing the padded sequences, and a list of lengths for each sequence in the batch. Batch elements will be reordered in their original order.
    
    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([2, 2,3])
        c = tensor.ones([1, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=True)

        seq_unpacked, lens_unpacked = tensor.pad_packed_sequence(data, batch_first=True)
        print(seq_unpacked)
        # [[[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]]
        print(lens_unpacked)
        # [4, 2, 1]


pack_pad_sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.xtensor.pack_pad_sequence(input, lengths, batch_first=False, enforce_sorted=True)

    Pack a Tensor containing variable-length padded sequences. If batch_first is True, `input` should have shape [batch size, length,*], otherwise shape [length, batch size,*].

    For unsorted sequences, use ``enforce_sorted`` is False. If :attr:`enforce_sorted` is ``True``, sequences should be sorted in descending order by length.
    
    :param input: 'XTensor' - variable-length sequence batches for padding.
    :parma lengths: 'list' - list of sequence lengths for each batch
         element.
    :param batch_first: 'bool' - if ``True``, the input is expected to be ``B x T x *``
         format, default: False.
    :param enforce_sorted: 'bool' - if ``True``, the input should be
         Contains sequences in descending order of length. If ``False``, the input will be sorted unconditionally. Default: True.

    :return: A :class:`PackedSequence` object.

    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = tensor.ones([4, 2,3])
        c = tensor.ones([1, 2,3])
        b = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=False)
        print(data.data)

        # [[[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]]

        print(data.batch_sizes)
        # [3, 2, 1, 1]