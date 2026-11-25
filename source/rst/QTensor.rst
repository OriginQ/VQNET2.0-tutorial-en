.. _qtensor_api:

QTensor Module
###########################

Python interface of QTensor, the data structure used by VQNet quantum machine learning. QTensor supports common multidimensional tensor operations, such as creation functions, mathematical functions, logical functions, and matrix transformations.



QTensor's Functions and Attributes
******************************************

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, device=pyvqnet.DEV_CPU, dtype=None)

    Tensor with dynamic computational graph construction and automatic differentiation.

    :param data: Input data, which can be a _core.Tensor or a numpy array.
    :param requires_grad: Whether the tensor's gradient should be tracked. Defaults to False.
    :param device: Device on which the data is stored. Default: pyvqnet.DEV_CPU, on the CPU.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.dtype import *
        import numpy as np

        t1 = QTensor(np.ones([2,3]))
        t2 =  QTensor([2,3,4j,5])
        t3 =  QTensor([[[2,3,4,5],[2,3,4,5]]],dtype=kbool)
        print(t1)
        print(t2)
        print(t3)
        # [[1. 1. 1.]
        #  [1. 1. 1.]]
        # [2.+0.j 3.+0.j 0.+4.j 5.+0.j]
        # [[[ True  True  True  True]
        #   [ True  True  True  True]]]


    .. py:attribute:: ndim

        Returns the number of dimensions of the tensor.

        :return: The number of dimensions of the tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1



    .. py:attribute:: shape

        Returns the dimensions of the tensor

        :return: A list containing the dimensions of the tensor

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]



    .. py:attribute:: size

        Returns the number of elements in the tensor.

        :return: The number of elements in the tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4


    .. py:method:: numel

        Returns the number of elements in the tensor.

        :return: The number of elements in the tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4



    .. py:attribute:: dtype

        Returns the data type of the tensor.

        Supported data types are as follows:

            =========================================  ===============================
            dtype                                      description
            =========================================  ===============================
            ``pyvqnet.kbool``                          Boolean variable
            ``pyvqnet.kuint8``                         8-bit integer (unsigned)
            ``pyvqnet.kint8``                          8-bit integer (signed)
            ``pyvqnet.kint16``                         16-bit integer (signed)
            ``pyvqnet.kint32``                         32-bit integer (signed)
            ``pyvqnet.kint64``                         64-bit integer (signed)
            ``pyvqnet.kfloat32``                       32-bit floating point, see https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       64-bit floating point, see https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex64``                     64-bit complex number composed of two `float32` values
            ``pyvqnet.kcomplex128``                    128-bit complex number composed of two `float64` values
            ``pyvqnet.kbfloat16``                      16-bit floating point, sometimes called Brain floating point format, with 1 sign bit, 8 exponent bits, and 7 mantissa bits
            =========================================  ===============================


        :return: The data type of the tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2, 3, 4, 5])
            print(a.dtype)
            # 4


    .. py:attribute:: is_contiguous

        Whether the tensor is a contiguous multidimensional array.

        :return: True if contiguous, otherwise False.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([[2, 3, 4, 5],[2, 3, 4, 5]])
            b = a.is_contiguous
            print(b)
            #True
            c= a.permute((1,0))
            print(c.is_contiguous)
            #False



    .. py:method:: zero_grad()

        Sets the tensor's gradient to zero. Used by optimizers during optimization.

        :return: None.

        Example::

            from pyvqnet.tensor import QTensor
            t3 = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            t3.zero_grad()
            print(t3.grad)
            # [0., 0., 0., 0.]



    .. py:method:: backward(grad=None)

        Uses the backpropagation algorithm to compute the gradients of all tensors in the computational graph containing the current tensor that require gradients.

        :return: None

        Example::

            from pyvqnet.tensor import QTensor

            target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
            y = 2*target + 3
            y.backward()
            print(target.grad)
            #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]



    .. py:method:: to_numpy()

        Copies the tensor's data into a numpy.ndarray.

        :return: A new numpy.ndarray containing the QTensor data

        .. note::

            numpy does not support the bfloat16 type. Convert to another data type supported by numpy, such as float32, before calling this interface.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3 = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            t4 = t3.to_numpy()
            print(t4)

            # [2. 3. 4. 5.]



    .. py:method:: item()

        Returns the only element from a QTensor containing a single element.

        :return: The element value

        Example::

            from pyvqnet.tensor import tensor

            t = tensor.ones([1])
            print(t.item())

            # 1.0




    .. py:method:: contiguous()

        Returns the contiguous form of the current QTensor. If it is already contiguous, returns itself.

        :return: The contiguous form of the current QTensor. If it is already contiguous, returns itself.

        Example::

            from pyvqnet.tensor import tensor

            t = tensor.ones([1])
            print(t.contiguous())




    .. py:method:: argmax(*kargs)

        Returns the index of the maximum value of all elements in the input QTensor, or the indices of the maximum values along a dimension of the QTensor.

        :param dim: Axis along which to compute argmax; only a single dimension is accepted. If dim == None, returns the index of the maximum value of all elements in the input tensor. The valid range of dim is [-R, R), where R is the ndim of the input. When dim < 0, it works the same as dim + R.
        :param keepdim: Whether the output QTensor keeps the axis of the maximum index operation. Defaults to False.

        :return: Index of the maximum value in the input QTensor.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                        [-0.7401, -0.8805, -0.3402, -1.1936],
                        [0.4907, -1.3948, -1.0691, -0.3132],
                        [-1.6092, 0.5419, -0.2993, 0.3195]])
            flag = a.argmax()
            print(flag)

            # [0.]

            flag_0 = a.argmax([0], True)
            print(flag_0)

            # [
            # [0., 3., 0., 3.]
            # ]

            flag_1 = a.argmax([1], True)
            print(flag_1)

            # [
            # [0.],
            # [2.],
            # [0.],
            # [1.]
            # ]



    .. py:method:: argmin(*kargs)

        Returns the index of the minimum value of all elements in the input QTensor, or the indices of the minimum values along a dimension of the QTensor.

        :param dim: Axis along which to compute argmin; only a single dimension is accepted. If dim == None, returns the index of the minimum value of all elements in the input tensor. The valid range of dim is [-R, R), where R is the ndim of the input. When dim < 0, it works the same as dim + R.
        :param keepdim: Whether the output QTensor keeps the axis of the minimum index operation. Defaults to False.

        :return: Index of the minimum value in the input QTensor.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                        [-0.7401, -0.8805, -0.3402, -1.1936],
                        [0.4907, -1.3948, -1.0691, -0.3132],
                        [-1.6092, 0.5419, -0.2993, 0.3195]])
            flag = a.argmin()
            print(flag)

            # [12.]

            flag_0 = a.argmin([0], True)
            print(flag_0)

            # [
            # [3., 2., 2., 1.]
            # ]

            flag_1 = a.argmin([1], False)
            print(flag_1)

            # [2., 3., 1., 0.]


    .. py:method:: fill_(v)

        Fills the current tensor with a specific value. This function changes the internal data of the original tensor.

        :param v: Fill value.

        :return: None.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            shape = [2, 3]
            value = 42
            t = tensor.zeros(shape)
            t.fill_(value)
            print(t)

            # [
            # [42., 42., 42.],
            # [42., 42., 42.]
            # ]


    .. py:method:: all()

        Checks whether all data in the tensor is non-zero.

        :return: True if all are non-zero, otherwise False.

        Example::

            from pyvqnet.tensor import tensor

            shape = [2, 3]
            t = tensor.zeros(shape)
            t.fill_(1.0)
            flag = t.all()
            print(flag)

            # True

    .. py:method:: any()

        Checks whether any element in the tensor is non-zero.

        :return: True if any element is non-zero, otherwise False.

        Example::

            from pyvqnet.tensor import tensor

            shape = [2, 3]
            t = tensor.ones(shape)
            t.fill_(1.0)
            flag = t.any()
            print(flag)

            # True


    .. py:method:: fill_rand_binary_(v=0.5)

        Fills the QTensor with values randomly sampled from a binomial distribution.

        If the value randomly generated from the binomial distribution is greater than the binarization threshold v, the corresponding element of the QTensor is set to 1; otherwise it is set to 0.

        :param v: Binarization threshold, default 0.5.

        :return: None.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            t.fill_rand_binary_(2)
            print(t)

            # [
            # [1., 1., 1.],
            # [1., 1., 1.]
            # ]



    .. py:method:: fill_rand_signed_uniform_(v=1)

        Fills the QTensor with values randomly sampled from a signed uniform distribution. The generated random samples are scaled by the factor v.

        :param v: Scaling factor, default 1.

        :return: None.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            value = 42

            t.fill_rand_signed_uniform_(value)
            print(t)

            # [
            # [12.8852444, 4.4327269, 4.8489408],
            # [-24.3309803, 26.8036957, 39.4903450]
            # ]


    .. py:method:: fill_rand_uniform_(v=1)

        Fills the QTensor with values randomly sampled from a uniform distribution. The generated random samples are scaled by the factor v.

        :param v: Scaling factor, default 1.

        :return: None.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            value = 42
            t.fill_rand_uniform_(value)
            print(t)

            # [
            # [20.0404720, 14.4064417, 40.2955666],
            # [5.5692234, 26.2520485, 35.3326073]
            # ]


    .. py:method:: fill_rand_normal_(m=0, s=1, fast_math=True)

        Generates normally distributed elements with mean m and variance s and fills them into the tensor.

        :param m: Mean, default 0.
        :param s: Variance, default 1.
        :param fast_math: Whether to use a fast method to generate the Gaussian distribution, default True.

        :return: None.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            t.fill_rand_normal_(2, 10, True)
            print(t)

            # [
            # [-10.4446531    4.9158096   2.9204607],
            # [ -7.2682705   8.1267328    6.2758742 ],
            # ]


    .. py:method:: transpose(new_dims=None)

        Reverses the axes of the tensor. If new_dims = None, reverses all axes.

        :param new_dims: The new axis order stored as a list.

        :return:  A new QTensor.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
            t = QTensor(a)
            rlt = t.transpose([2,0,1])
            print(rlt)
            # [
            # [[0., 3.],
            #  [6., 9.]],
            # [[1., 4.],
            #  [7., 10.]],
            # [[2., 5.],
            #  [8., 11.]]
            # ]



    .. py:method:: reshape(shape)

        Changes the shape of the QTensor and returns a new tensor.

        :param shape: New shape.

        :return: A QTensor with the new shape.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape(R, C).astype(np.float32)
            t = QTensor(a)
            reshape_t = t.reshape([C, R])
            print(reshape_t)
            # [
            # [0., 1., 2.],
            # [3., 4., 5.],
            # [6., 7., 8.],
            # [9., 10., 11.]
            # ]


    .. py:method:: __getitem__()

        Supports accessing the input with slice indexing, subscripts, or a QTensor as an advanced index. This operation returns a new QTensor.

        Slice operations use the colon ``:`` to separate the slice parameters start:stop:step, where start, stop, and step can all be omitted.

        For a 1-D QTensor, there is only indexing or slicing on a single axis.

        For a 2-D or higher QTensor, there is indexing or slicing on multiple axes.

        Using a QTensor as an index performs advanced indexing; see the `Advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ section in numpy.

        If the QTensor used as an index is the result of a logical operation, boolean array indexing is performed.

        .. note::

            Indexing of the form a[3][4][1] is not supported yet; use a[3,4,1] instead.


        :param item: A slice index composed of pyslice, integers, and QTensor.

        :return: A new QTensor.

        Example::

            from pyvqnet.tensor import tensor, QTensor
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            print(aaa[0:2, 3, :2])
            # [
            # [10., 11.],
            #  [25., 26.]
            # ]
            print(aaa[3, 4, 1])
            #[59.]
            print(aaa[:, 2, :])
            # [
            # [7., 8., 9.],
            #  [22., 23., 24.],
            #  [37., 38., 39.],
            #  [52., 53., 54.]
            # ]
            print(aaa[2])
            # [
            # [31., 32., 33.],
            #  [34., 35., 36.],
            #  [37., 38., 39.],
            #  [40., 41., 42.],
            #  [43., 44., 45.]
            # ]
            print(aaa[0:2, ::3, 2:])
            # [
            # [[3.],
            #  [12.]],
            # [[18.],
            #  [27.]]
            # ]
            a = tensor.ones([2, 2])
            b = QTensor([[1, 1], [0, 1]])
            b = b > 0
            c = a[b]
            print(c)
            #[1., 1., 1.]
            tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
            tt.requires_grad = True
            index_sample1 = tensor.arange(0, 3).reshape([3, 1])
            index_sample2 = QTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
            gg = tt[:, index_sample1, 3:, index_sample2, 2:]
            print(gg)
            # [
            # [[[[87., 88.]],
            # [[983., 984.]]],
            # [[[91., 92.]],
            # [[987., 988.]]],
            # [[[87., 88.]],
            # [[983., 984.]]]],
            # [[[[207., 208.]],
            # [[1103., 1104.]]],
            # [[[211., 212.]],
            # [[1107., 1108.]]],
            # [[[207., 208.]],
            # [[1103., 1104.]]]],
            # [[[[319., 320.]],
            # [[1215., 1216.]]],
            # [[[323., 324.]],
            # [[1219., 1220.]]],
            # [[[323., 324.]],
            # [[1219., 1220.]]]]
            # ]



    .. py:method:: __setitem__()

        Supports modifying the input with slice indexing, subscripts, or a QTensor as an advanced index. This operation modifies the input in place.

        Slice operations use the colon ``:`` to separate the slice parameters start:stop:step, where start, stop, and step can all be omitted.

        For a 1-D QTensor, there is only indexing or slicing on a single axis.

        For a 2-D or higher QTensor, there is indexing or slicing on multiple axes.

        Using a QTensor as an index performs advanced indexing; see the `Advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ section in numpy.

        If the QTensor used as an index is the result of a logical operation, boolean array indexing is performed.

        .. note::

            Indexing of the form a[3][4][1] is not supported yet; use a[3,4,1] instead.


        :param item: A slice index composed of pyslice, integers, and QTensor.

        :return: None.

        Example::

            from pyvqnet.tensor import tensor
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
            b = tensor.QTensor([[1, 1], [0, 1]])
            b = b > 0
            x = tensor.QTensor([1001, 2001, 3001])

            a[b] = x
            print(a)
            # [
            # [1001., 2001.],
            #  [1., 3001.]
            # ]



    .. py:method:: GPU(device: int = DEV_GPU_0)

        Clones the QTensor to the specified GPU device

        device specifies the device on which its internal data is stored. When device >= DEV_GPU_0, the data is stored on the GPU.
        If your computer has multiple GPUs, you can specify different devices to store the data. For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... means storing on GPUs with different sequence numbers.

        .. note::

            QTensor cannot perform computation across different GPUs.
            If you try to create a QTensor on a GPU whose ID exceeds the verified maximum number of GPUs, a CUDA error is raised.

        :param device: Device on which the QTensor is currently stored. Default: DEV_GPU_0. device = pyvqnet.DEV_GPU_0 stores it on the first GPU, device = DEV_GPU_1 on the second GPU, and so on.

        :return: QTensor cloned to the GPU device.

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000


    .. py:method:: CPU()

        Clones the QTensor to a specific CPU device

        :return: QTensor cloned to the CPU device.

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0


    .. py:method:: toGPU(device: int = DEV_GPU_0)

        Moves the QTensor to the specified GPU device

        device specifies the device on which its internal data is stored. When device >= DEV_GPU, the data is stored on the GPU.
        If your computer has multiple GPUs, you can specify different devices to store the data.
        For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... means storing on GPUs with different sequence numbers.

        .. note::

            QTensor cannot perform computation across different GPUs.
            If you try to create a QTensor on a GPU whose ID exceeds the verified maximum number of GPUs, a CUDA error is raised.

        :param device: Device on which the QTensor is currently stored. Default: DEV_GPU_0. device = pyvqnet.DEV_GPU_0 stores it on the first GPU, device = DEV_GPU_1 on the second GPU, and so on.
        :return: QTensor moved to the GPU device.

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000



    .. py:method:: toCPU()

        Moves the QTensor to the CPU

        :return: QTensor moved to the CPU device.

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0




    .. py:method:: isGPU()

        Whether the data of this QTensor is stored in GPU host memory.

        :return: Whether the data of this QTensor is stored in GPU host memory.

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False



    .. py:method:: isCPU()

        Whether the data of this QTensor is stored in CPU host memory.

        :return: Whether the data of this QTensor is stored in CPU host memory.

        Example::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True

    .. py:method:: astype(dtype)

        Converts the current QTensor to the corresponding data type dtype. If dtype is the same, returns itself.

        :param dtype: Target data type dtype.

        :return:  QTensor.

        Example::

            from pyvqnet.tensor import QTensor
            from pyvqnet import kcomplex128
            a = QTensor([2])
            a = a.astype(kcomplex128)
            print(a)
            #[2.+0.j]

Creation Functions
*******************************************************************************************

.. _ones:

ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=pyvqnet.DEV_CPU,dtype=None)

    Creates a QTensor with all elements equal to one.

    :param shape: Shape of the data.
    :param device: Device on which the data is stored. Default: pyvqnet.DEV_CPU, on the CPU.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.

    :return: Returns a new QTensor.

    Example::

        from pyvqnet.tensor import tensor

        x = tensor.ones([2, 3])
        print(x)

        # [
        # [1., 1., 1.],
        # [1., 1., 1.]
        # ]

ones_like
==============================

.. py:function:: pyvqnet.tensor.ones_like(t: pyvqnet.tensor.QTensor,device=pyvqnet.DEV_CPU,dtype=None)

    Creates a QTensor with all elements equal to one and the same shape as the input QTensor.

    :param t: Input QTensor.
    :param device: Device on which the data is stored. Default: pyvqnet.DEV_CPU, on the CPU.
    :param dtype: Data type of the parameter. Default: None, same as the input dtype.

    :return: A new all-ones QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)
        print(x)

        # [1., 1., 1.]


full
==============================

.. py:function:: pyvqnet.tensor.full(shape, value, device=pyvqnet.DEV_CPU, dtype=None)

    Creates a QTensor of the specified shape and fills it with a specific value.

    :param shape: Shape of the tensor to create.
    :param value: Value to fill with.
    :param device: Device on which the data is stored. Default: pyvqnet.DEV_CPU, on the CPU.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.

    :return: Output new QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        value = 42
        t = tensor.full(shape, value)
        print(t)
        # [
        # [42., 42., 42.],
        # [42., 42., 42.]
        # ]


full_like
==============================

.. py:function:: pyvqnet.tensor.full_like(t, value, device=pyvqnet.DEV_CPU,dtype=None)

    Creates a QTensor with the same shape as the input, with all elements filled with value.

    :param t: Input QTensor.
    :param value: Value to fill the QTensor with.
    :param device: Device on which the data is stored. Default: pyvqnet.DEV_CPU, on the CPU.
    :param dtype: Data type of the parameter. Default: None, same as the input dtype.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = tensor.randu([3,5])
        value = 42
        t = tensor.full_like(a, value)
        print(t)
        # [
        # [42., 42., 42., 42., 42.],
        # [42., 42., 42., 42., 42.],
        # [42., 42., 42., 42., 42.]
        # ]


zeros
==============================

.. py:function:: pyvqnet.tensor.zeros(shape, device=pyvqnet.DEV_CPU,dtype=None)

    Creates an all-zero QTensor with the input shape.

    :param shape: Input shape.
    :param device: Device on which the data is stored. Default: pyvqnet.DEV_CPU, on the CPU.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.zeros([2, 3, 4])
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
==============================

.. py:function:: pyvqnet.tensor.zeros_like(t: pyvqnet.tensor.QTensor, device=pyvqnet.DEV_CPU,dtype=None)

    Creates a QTensor with the same shape as the input, with all elements equal to 0.

    :param t: Input reference QTensor.
    :param device: Device on which the data is stored. Default: pyvqnet.DEV_CPU, on the CPU.
    :param dtype: Data type of the parameter. Default: None, same as the input dtype.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)
        print(x)

        # [0., 0., 0.]



arange
==============================

.. py:function:: pyvqnet.tensor.arange(start, end, step=1, device=pyvqnet.DEV_CPU,dtype=None,requires_grad=False)

    Creates a one-dimensional QTensor with evenly spaced values within the given interval.

    :param start: Start of the interval.
    :param end: End of the interval.
    :param step: Spacing between values, default 1.
    :param device: Device to use, default = pyvqnet.DEV_CPU, using the CPU device.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.
    :param requires_grad: Whether to compute gradients. Defaults to False.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30, 4)
        print(t)

        # [ 2.,  6., 10., 14., 18., 22., 26.]


linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, nums, device=pyvqnet.DEV_CPU,dtype=None,requires_grad= False)

    Creates a one-dimensional QTensor whose elements are nums evenly spaced values over the interval from start to end.

    :param start: Start of the interval.
    :param end: End of the interval.
    :param nums: Number of values in the interval.
    :param device: Device to use, default = pyvqnet.DEV_CPU, using the CPU device.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.
    :param requires_grad: Whether to compute gradients. Defaults to False.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, num = -2.5, 10, 10
        t = tensor.linspace(start, stop, num)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10.]

logspace
==============================

.. py:function:: pyvqnet.tensor.logspace(start, end, nums, base, device=pyvqnet.DEV_CPU,dtype=None, requires_grad= False)

    Creates a one-dimensional QTensor with evenly spaced values on a logarithmic scale.

    :param start: ``base ** start`` is the starting value
    :param end: ``base ** end`` is the final value of the sequence
    :param nums: Number of samples to generate
    :param base: Base of the logarithmic scale
    :param device: Device to use, default = pyvqnet.DEV_CPU, using the CPU device.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.
    :param requires_grad: Whether to compute gradients. Defaults to False.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, steps, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10.]


eye
==============================

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, device=pyvqnet.DEV_CPU,dtype=None)

    Creates a size x size QTensor with ones on the diagonal and zeros elsewhere.

    :param size: Size of the (square) QTensor to create.
    :param offset: Index of the diagonal: 0 (default) means the main diagonal, a positive value means an upper diagonal, and a negative value means a lower diagonal.
    :param device: Device to use, default =pyvqnet.DEV_CPU, using the CPU device.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        size = 3
        t = tensor.eye(size)
        print(t)

        # [
        # [1., 0., 0.],
        # [0., 1., 0.],
        # [0., 0., 1.]
        # ]


diagonal
==============================
.. py:function:: pyvqnet.tensor.diagonal(t: QTensor, offset: int = 0, dim1=0, dim2=1)

    Returns a partial view of :attr:`t` with its diagonal elements appended as a dimension at the end of the shape, relative to :attr:`dim1` and :attr:`dim2`.
    :attr:`offset` is the offset from the main diagonal.

    :param t: Input tensor
    :param offset: Offset (0 means the main diagonal, a positive value means the nth diagonal above the main diagonal, and a negative value means the nth diagonal below the main diagonal)
    :param dim1: First dimension from which to take the diagonal. Default: 0.
    :param dim2: Second dimension from which to take the diagonal. Default: 1.

    Example::

        from pyvqnet.tensor import randn,diagonal

        x = randn((2, 5, 4, 2))
        diagonal_elements = diagonal(x, offset=-1, dim1=1, dim2=2)
        print(diagonal_elements)
        # [[[-0.4641751,-0.1410288,-0.1215512, 0.5423283],
        #   [ 0.9556418, 0.0376572, 1.2571657, 0.8268463]],

        #  [[-0.7972266, 0.2080281,-0.1157126,-0.7342224],
        #   [ 1.1039937, 0.4700735, 1.0219841,-0.146358 ]]]


diag
==============================

.. py:function:: pyvqnet.tensor.diag(t, k: int = 0)

    Constructs a diagonal matrix.

    Given a 2-D QTensor, returns a new 1D tensor containing
    the elements in the selected diagonal.
    Given a 1-D QTensor, returns a new 2D tensor whose selected diagonal elements are the input values and whose remaining elements are 0

    :param t: Input QTensor.
    :param k: Offset (0 for the main diagonal, positive to shift upward, negative to shift downward), default 0.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = QTensor(a)
        for k in range(-3, 4):
            u = tensor.diag(t,k=k)
            print(u)
        # [12.]
        # <QTensor [1] DEV_CPU kfloat32>

        # [ 8.,13.]
        # <QTensor [2] DEV_CPU kfloat32>

        # [ 4., 9.,14.]
        # <QTensor [3] DEV_CPU kfloat32>

        # [ 0., 5.,10.,15.]
        # <QTensor [4] DEV_CPU kfloat32>

        # [ 1., 6.,11.]
        # <QTensor [3] DEV_CPU kfloat32>

        # [2.,7.]
        # <QTensor [2] DEV_CPU kfloat32>

        # [3.]
        # <QTensor [1] DEV_CPU kfloat32>


randu
==============================

.. py:function:: pyvqnet.tensor.randu(shape, min=0.0,max=1.0, device=pyvqnet.DEV_CPU, dtype=None, requires_grad=False)

    Creates a QTensor with uniformly distributed random values.

    :param shape: Shape of the QTensor to create.
    :param min: Lower bound of the distribution, default: 0.
    :param max: Upper bound of the distribution, default: 1.
    :param device: Device to use, default =pyvqnet.DEV_CPU, using the CPU device.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.
    :param requires_grad: Whether to compute gradients. Defaults to False.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randu(shape)
        print(t)

        # [
        # [0.0885886, 0.9570093, 0.8304565],
        # [0.6055251, 0.8721224, 0.1927866]
        # ]


randn
==============================

.. py:function:: pyvqnet.tensor.randn(shape, mean=0.0,std=1.0, device=pyvqnet.DEV_CPU, dtype=None, requires_grad=False)

    Creates a QTensor with normally distributed random values.

    :param shape: Shape of the QTensor to create.
    :param mean: Mean of the distribution, default: 0.
    :param std: Variance of the distribution, default: 1.
    :param device: Device to use, default = pyvqnet.DEV_CPU, using the CPU device.
    :param dtype: Data type of the parameter. Default: None, which uses the default data type kfloat32, representing a 32-bit floating point number.
    :param requires_grad: Whether to compute gradients. Defaults to False.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randn(shape)
        print(t)

        # [
        # [-0.9529880, -0.4947567, -0.6399882],
        # [-0.6987777, -0.0089036, -0.5084590]
        # ]

binomial
==============================
.. py:function:: pyvqnet.tensor.binomial(total_counts, probs)

    Creates a binomial distribution parameterized by :attr:`total_count` and :attr:`probs`.

    :param total_counts: Number of Bernoulli trials.
    :param probs: Event probability.

    :return:
        QTensor of the binomial distribution.

    Example::

        import pyvqnet.tensor as tensor

        a = tensor.randu([3,4])
        b = 1000

        c = tensor.binomial(b,a)
        print(c)

        # [[221.,763., 30.,339.],
        #  [803.,899.,105.,356.],
        #  [550.,688.,828.,493.]]


multinomial
==============================

.. py:function:: pyvqnet.tensor.multinomial(t, num_samples)

    Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of the input tensor.

    :param t: Input probability distribution; only floating-point types are supported.
    :param num_samples: Number of samples.

    :return:
         Output sample indices

    Example::

        from pyvqnet import tensor
        weights = tensor.QTensor([0.1,10, 3, 1])
        idx = tensor.multinomial(weights,3)
        print(idx)

        from pyvqnet import tensor
        weights = tensor.QTensor([0,10, 3, 2.2,0.0])
        idx = tensor.multinomial(weights,3)
        print(idx)

        # [1 0 3]
        # [1 3 2]

triu
==============================

.. py:function:: pyvqnet.tensor.triu(t, diagonal=0)

    Returns the upper triangular matrix of input t, with the rest set to 0.

    :param t: Input QTensor.
    :param diagonal: Offset (0 for the main diagonal, positive to shift upward, negative to shift downward), default: 0.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
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
==============================

.. py:function:: pyvqnet.tensor.tril(t, diagonal=0)

    Returns the lower triangular matrix of input t, with the rest set to 0.

    :param t: Input QTensor.
    :param diagonal: Offset (0 for the main diagonal, positive to shift upward, negative to shift downward), default: 0.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
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

Mathematical Functions
*******************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    Returns a new QTensor whose elements are the floor of the input QTensor.

    :param t: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2., -2., -2., -2., -1., -1., -1., -1., 0., 0., 0., 0., 1., 1., 1., 1.]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    Returns a new QTensor whose elements are the ceiling of the input QTensor.

    :param t: Input QTensor.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2., -1., -1., -1., -1., -0., -0., -0., 0., 1., 1., 1., 1., 2., 2., 2.]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    Returns a new QTensor whose elements are the input QTensor rounded to the nearest integer.

    :param t: Input QTensor.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2., -2., -1., -1., -0., -0., 0., 1., 1., 2.]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, dim: int, descending=False, stable=True)

    Sorts the input QTensor along the specified axis.

    :param t: Input QTensor.
    :param dim: Axis used for sorting.
    :param descending: If True, sort in descending order; otherwise sort in ascending order. Defaults to ascending.
    :param stable: Whether to use a stable sort. Defaults to a stable sort.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        AA = tensor.sort(A,1,False)
        print(AA)

        # [
        # [0., 1., 2., 4., 6., 7., 8., 8.],
        # [2., 5., 5., 8., 9., 9., 9., 9.],
        # [1., 2., 5., 5., 5., 6., 7., 7.]
        # ]

argsort
==============================

.. py:function:: pyvqnet.tensor.argsort(t, dim: int, descending=False, stable=True)

    Sorts the input variable along the given axis and outputs the corresponding indices of the sorted data.

    :param t: Input QTensor.
    :param dim: Axis used for sorting.
    :param descending: If True, sort in descending order; otherwise sort in ascending order. Defaults to ascending.
    :param stable: Whether to use a stable sort. Defaults to a stable sort.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        bb = tensor.argsort(A,1,False)
        print(bb)

        # [
        # [4., 0., 1., 7., 5., 3., 2., 6.],
        #  [3., 0., 7., 6., 2., 1., 4., 5.],
        #  [4., 7., 5., 0., 2., 1., 3., 6.]
        # ]

topK
==============================

.. py:function:: pyvqnet.tensor.topK(t, k, dim=-1, if_descent=True)

    Returns the k largest elements of the given input tensor along the given dimension.

    If if_descent is False, returns the k smallest elements.

    :param t: Input QTensor.
    :param k: Number of elements to take after sorting.
    :param dim: Dimension to sort along. Default = -1, the last axis.
    :param if_descent: Whether sorting uses ascending or descending order, default descending.

    :return: A new QTensor.

    Example::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x=x.reshape([2, 5, 1, 2])
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
==============================

.. py:function:: pyvqnet.tensor.argtopK(t, k, dim=-1, if_descent=True)

    Returns the indices of the k largest elements of the given input tensor along the given dimension.

    If if_descent is False, returns the indices of the k smallest elements.

    :param t: Input QTensor.
    :param k: Number of elements to take after sorting.
    :param dim: Dimension to sort along. Default = -1, the last axis.
    :param if_descent: Whether sorting uses ascending or descending order, default descending.

    :return: A new QTensor.

    Example::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x=x.reshape([2, 5, 1, 2])
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
==============================

.. py:function:: pyvqnet.tensor.add(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Adds two QTensors element by element. Equivalent to t1 + t2.

    :param t1: First QTensor.
    :param t2: Second QTensor.
    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.add(t1, t2)
        print(x)

        # [5., 7., 9.]

sub
==============================

.. py:function:: pyvqnet.tensor.sub(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Subtracts two QTensors element by element. Equivalent to t1 - t2.

    :param t1: First QTensor.
    :param t2: Second QTensor.
    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.sub(t1, t2)
        print(x)

        # [-3., -3., -3.]

mul
==============================

.. py:function:: pyvqnet.tensor.mul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Multiplies two QTensors element by element. Equivalent to t1 * t2.

    :param t1: First QTensor.
    :param t2: Second QTensor.
    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.mul(t1, t2)
        print(x)

        # [4., 10., 18.]

divide
==============================

.. py:function:: pyvqnet.tensor.divide(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Divides two QTensors element by element. Equivalent to t1 / t2.

    :param t1: First QTensor.
    :param t2: Second QTensor.
    :return:  Output QTensor.


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.divide(t1, t2)
        print(x)

        # [0.2500000, 0.4000000, 0.5000000]

sums
==============================

.. py:function:: pyvqnet.tensor.sums(t: pyvqnet.tensor.QTensor, dim: Optional[int] = None, keepdim=False)

    Computes the sum of the elements of the input QTensor along the axis specified by dim. If dim is None, returns the sum of all elements.

    :param t: Input QTensor.
    :param dim: Axis used for the sum, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21.]

cumsum
==============================

.. py:function:: pyvqnet.tensor.cumsum(t, dim=-1)

    Returns the cumulative sum of the elements in the input along the dimension axis.

    :param t: Input QTensor.
    :param dim: Axis for computation, default -1, using the last axis.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor, QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.cumsum(t,-1)
        print(x)
        # [
        # [1., 3., 6.],
        # [4., 9., 15.]
        # ]


mean
==============================

.. py:function:: pyvqnet.tensor.mean(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    Computes the mean of the elements of the input QTensor along the axis specified by dim. If dim is None, returns the mean of all elements.

    :param t: Input QTensor, which must be floating-point or complex.
    :param dim: Axis used for the mean, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.
    :return: Output QTensor or the mean value.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, dim=1)
        print(x)

        # [2. 5.]

median
==============================

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    Computes the median of the elements of the input QTensor along the axis specified by dim. If dim is None, returns the median of all elements.

    :param t: Input QTensor.
    :param dim: Axis used for the median, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.
    :return: Output QTensor or the median value.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1.5219, -1.5212,  0.2202]])
        median_a = tensor.median(a)
        print(median_a)

        # [0.2202000]

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        median_b = tensor.median(b,1, False)
        print(median_b)

        # [-0.3982000, 0.2269999, 0.2487999, 0.4742000]

std
==============================

.. py:function:: pyvqnet.tensor.std(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False, unbiased=True)

    Computes the standard deviation of the elements of the input QTensor along the axis specified by dim. If dim is None, returns the standard deviation of all elements.

    :param t: Input QTensor.
    :param dim: Axis used for the standard deviation, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.
    :param unbiased: Whether to use Bessel's correction, used by default.
    :return: Output QTensor or the standard deviation value.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        std_a = tensor.std(a)
        print(std_a)

        # [0.5129624]

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        std_b = tensor.std(b, 1, False, False)
        print(std_b)

        # [0.6593542, 0.5583112, 0.3206565, 1.1103367]

var
==============================

.. py:function:: pyvqnet.tensor.var(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False, unbiased=True)

    Computes the variance of the elements of the input QTensor along the axis specified by dim. If dim is None, returns the variance of all elements.

    :param t: Input QTensor.
    :param dim: Axis used for the variance, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.
    :param unbiased: Whether to use Bessel's correction, used by default.
    :return: Output QTensor or variance value.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

        # [0.2631305]

matmul
==============================

.. py:function:: pyvqnet.tensor.matmul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Computes the dot product of two 2-D matrices, batched matrix multiplication for 3-D or 4-D tensors, the matrix-vector product of a 1-D vector and a 2-D matrix, or the dot product of two 1-D vectors.

    :param t1: First QTensor.
    :param t2: Second QTensor.
    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
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
==============================

.. py:function:: pyvqnet.tensor.kron(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Computes the Kronecker product of ``t1`` and ``t2``, denoted by :math:`\otimes`.

    If ``t1`` is an :math:`(a_0 \times a_1 \times \dots \times a_n)` tensor and ``t2`` is a :math:`(b_0 \times b_1 \times \dots \times b_n)` tensor, the result is a :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` tensor containing the following entries:

     .. math::
         (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
             \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

     where :math:`k_t = i_t * b_t + j_t` for :math:`0 \leq t \leq n`.
     If one tensor has fewer dimensions than the other, it is unsqueezed until it has the same number of dimensions.

    :param t1: First QTensor.
    :param t2: Second QTensor.
    :return:  Output QTensor.

    Example::

        from pyvqnet import tensor
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


einsum
==============================

.. py:function:: pyvqnet.tensor.einsum(equation, *operands)

    Sums the products of elements of the input operands along the specified dimensions using symbols based on the Einstein summation convention.

    .. note::

        This function uses opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) to accelerate computation or reduce memory consumption by optimizing the contraction order. This optimization occurs when there are at least three inputs.

        For more complex `einsum` cases, you can import opt_einsum separately and compute directly on QTensor.

    :param equation: Subscripts for the Einstein summation.

    :param operands: Tensors on which to compute the Einstein summation.

    :return:
            QTensor result.

    Example::

        from pyvqnet import tensor

        vqneta = tensor.randn((3, 5, 4))
        vqnetl = tensor.randn((2, 5))
        vqnetr = tensor.randn((2, 4))
        z = tensor.einsum('bn,anm,bm->ba',  vqnetl, vqneta,vqnetr)
        print(z.shape)
        #[2, 3]
        vqneta = tensor.randn((20,30,40,50))
        z = tensor.einsum('...ij->...ji', vqneta)
        print(z.shape)
        #[20, 30, 50, 40]

reciprocal
==============================

.. py:function:: pyvqnet.tensor.reciprocal(t)

    Computes the reciprocal of the input QTensor.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1., 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
==============================

.. py:function:: pyvqnet.tensor.sign(t)

    Determines the sign of each element in input t and outputs the sign value: 1 for positive, -1 for negative, and 0 for zero.

    :param t: Input QTensor.

    :return:  Output QTensor.


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1., -1., -1., -1., -1., 0., 1., 1., 1., 1.]

neg
==============================

.. py:function:: pyvqnet.tensor.neg(t: pyvqnet.tensor.QTensor)

    Computes and returns the additive inverse of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1., -2., -3.]

trace
==============================

.. py:function:: pyvqnet.tensor.trace(t, k: int = 0)

    Returns the trace of a two-dimensional matrix.

    :param t: Input QTensor.
    :param k: Offset (0 for the main diagonal, positive to shift upward, negative to shift downward), default 0.

    :return: Sum of the diagonal elements of the input two-dimensional matrix.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

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
==============================

.. py:function:: pyvqnet.tensor.exp(t: pyvqnet.tensor.QTensor)

    Computes the exponential with base e for each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.exp(t)
        print(x)

        # [2.7182817, 7.3890562, 20.0855369]

acos
==============================

.. py:function:: pyvqnet.tensor.acos(t: pyvqnet.tensor.QTensor)

    Computes the arccosine of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(36).reshape(2,6,3).astype(np.float32)
        a =a/100
        A = QTensor(a,requires_grad = True)
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
==============================

.. py:function:: pyvqnet.tensor.asin(t: pyvqnet.tensor.QTensor)

    Computes the arcsine of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0., 0.5235988]

atan
==============================

.. py:function:: pyvqnet.tensor.atan(t: pyvqnet.tensor.QTensor)

    Computes the arctangent of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0., 0.4636476]

sin
==============================

.. py:function:: pyvqnet.tensor.sin(t: pyvqnet.tensor.QTensor)

    Computes the sine of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sin(t)
        print(x)

        # [0.8414709, 0.9092974, 0.1411200]

cos
==============================

.. py:function:: pyvqnet.tensor.cos(t: pyvqnet.tensor.QTensor)

    Computes the cosine of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cos(t)
        print(x)

        # [0.5403022, -0.4161468, -0.9899924]

tan
==============================

.. py:function:: pyvqnet.tensor.tan(t: pyvqnet.tensor.QTensor)

    Computes the tangent of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tan(t)
        print(x)

        # [1.5574077, -2.1850397, -0.1425465]

tanh
==============================

.. py:function:: pyvqnet.tensor.tanh(t: pyvqnet.tensor.QTensor)

    Computes the hyperbolic tangent of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tanh(t)
        print(x)

        # [0.7615941, 0.9640275, 0.9950547]

sinh
==============================

.. py:function:: pyvqnet.tensor.sinh(t: pyvqnet.tensor.QTensor)

    Computes the hyperbolic sine of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sinh(t)
        print(x)

        # [1.1752011, 3.6268603, 10.0178747]

cosh
==============================

.. py:function:: pyvqnet.tensor.cosh(t: pyvqnet.tensor.QTensor)

    Computes the hyperbolic cosine of each element of input t.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cosh(t)
        print(x)

        # [1.5430806, 3.7621955, 10.0676622]

power
==============================

.. py:function:: pyvqnet.tensor.power(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Raises the elements of the first QTensor to the powers given by the second QTensor.

    :param t1: First QTensor.
    :param t2: Second QTensor.
    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 4, 3])
        t2 = QTensor([2, 5, 6])
        x = tensor.power(t1, t2)
        print(x)

        # [1., 1024., 729.]

abs
==============================

.. py:function:: pyvqnet.tensor.abs(t: pyvqnet.tensor.QTensor)

    Computes the absolute value of each element of the input QTensor.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1., 2., 3.]

log
==============================

.. py:function:: pyvqnet.tensor.log(t: pyvqnet.tensor.QTensor)

    Computes the natural logarithm of each element of the input QTensor.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0., 0.6931471, 1.0986123]

log_softmax
==============================

.. py:function:: pyvqnet.tensor.log_softmax(t, dim=-1)

    Computes the softmax function along the dim dimension followed by the log function.

    :param t: Input QTensor.
    :param dim: Axis used for the softmax, default -1.

    :return: Output QTensor.

    Example::

        from pyvqnet import tensor
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
==============================

.. py:function:: pyvqnet.tensor.sqrt(t: pyvqnet.tensor.QTensor)

    Computes the square root of each element of the input QTensor.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1., 1.4142135, 1.7320507]

square
==============================

.. py:function:: pyvqnet.tensor.square(t: pyvqnet.tensor.QTensor)

    Computes the square of each element of the input QTensor.

    :param t: Input QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)

        # [1., 4., 9.]


eigh
==============================

.. py:function:: pyvqnet.tensor.eigh(t: QTensor)

    Returns the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or real symmetric matrix.
    Returns two objects: a one-dimensional array containing the eigenvalues of a,
    and a two-dimensional square matrix or matrix of the corresponding eigenvectors (represented as columns), depending on the input type.

    :param: Input QTensor.
    :param: Eigenvalues and eigenvectors of t.
    :return:

        Returns the eigenvalues and eigenvectors

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet import tensor


        def generate_random_symmetric_matrix(n):
                A = pyvqnet.tensor.randn((n, n))
                A = A + A.transpose()
                return A

        n = 3
        symmetric_matrix = generate_random_symmetric_matrix(n)

        evs,vecs = pyvqnet.tensor.eigh(symmetric_matrix)
        print(evs)
        print(vecs)
        # [-4.0669565,-1.9191254,-1.3642329]
        # <QTensor [3] DEV_CPU kfloat32>

        # [[-0.9889652, 0.0325959,-0.1445187],
        #  [ 0.0912495, 0.9025176,-0.4208745],
        #  [ 0.1167119,-0.4294176,-0.8955328]]
        # <QTensor [3, 3] DEV_CPU kfloat32>


frobenius_norm
==============================

.. py:function:: pyvqnet.tensor.frobenius_norm(t: QTensor, dim: int = None, keepdim=False)

    Computes the Frobenius norm of the input QTensor along the axis specified by dim. If dim is None, returns the Frobenius norm of all elements.

    :param t: Input QTensor.
    :param dim: Axis used for the Frobenius norm, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.
    :return: Output QTensor or the Frobenius norm value.


    Example::

        from pyvqnet.tensor import tensor,QTensor
        t = QTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                    [[13., 14., 15.], [16., 17., 18.]]])
        t.requires_grad = True
        result = tensor.frobenius_norm(t, -2, False)
        print(result)
        # [
        # [4.1231055, 5.3851647, 6.7082038],
        #  [12.2065554, 13.6014709, 15.],
        #  [20.6155281, 22.0227146, 23.4307499]
        # ]


Logical Functions
*************************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Computes the element-wise maximum of two QTensors.

    :param t1: First QTensor.
    :param t2: Second QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)
        print(x)

        # [6., 5., 7.]

minimum
==============================

.. py:function:: pyvqnet.tensor.minimum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Computes the element-wise minimum of two QTensors.

    :param t1: First QTensor.
    :param t2: Second QTensor.

    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)
        print(x)

        # [2., 4., 3.]

min
==============================

.. py:function:: pyvqnet.tensor.min(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    Computes the minimum of the elements of the input QTensor along the axis specified by dim. If dim is None, returns the minimum of all elements.

    :param t: Input QTensor.
    :param dim: Axis used for the minimum, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.

    :return: Output QTensor or a floating-point value.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.min(t, dim=1, keepdim=True)
        print(x)

        # [
        # [1.],
        #  [4.]
        # ]

max
==============================

.. py:function:: pyvqnet.tensor.max(t: pyvqnet.tensor.QTensor, dim=None, keepdim=False)

    Computes the maximum of the elements of the input QTensor along the axis specified by dim. If dim is None, returns the maximum of all elements.

    :param t: Input QTensor.
    :param dim: Axis used for the maximum, default None.
    :param keepdim: Whether the output tensor keeps the reduced dimension. Defaults to False.

    :return: Output QTensor or a floating-point value.


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.max(t, dim=1, keepdim=True)
        print(x)

        # [
        # [3.],
        #  [6.]
        # ]

clip
==============================

.. py:function:: pyvqnet.tensor.clip(t: pyvqnet.tensor.QTensor, min_val, max_val)

    Clips all elements of the input so that the output elements are limited to [min_val, max_val].

    :param t: Input QTensor.
    :param min_val:  Lower clipping bound.
    :param max_val:  Upper clipping bound.
    :return:  Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3., 4., 6.]


where
==============================

.. py:function:: pyvqnet.tensor.where(condition: pyvqnet.tensor.QTensor, t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)


    Returns elements selected from t1 or t2 according to the condition.

    :param condition: Condition QTensor, which must be of type kbool.
    :param t1: Elements are taken from here if the condition is satisfied.
    :param t2: Elements are taken from here if the condition is not satisfied.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)
        print(x)

        # [1., 5., 6.]

nonzero
==============================

.. py:function:: pyvqnet.tensor.nonzero(t)

    Returns a QTensor containing the indices of non-zero elements.

    :param t: Input QTensor.
    :return: Output QTensor containing the indices of non-zero elements.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[0.6, 0.0, 0.0, 0.0],
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
==============================

.. py:function:: pyvqnet.tensor.isfinite(t)

    Checks element by element whether the input is finite (neither +/-INF nor +/-NaN).

    :param t: Input QTensor.
    :return: Output QTensor, where the corresponding element is True if the condition is satisfied, otherwise False.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        #[ True False  True False False]

isinf
==============================

.. py:function:: pyvqnet.tensor.isinf(t)

    Checks element by element whether each value of the input is +/-INF.

    :param t: Input QTensor.
    :return: Output QTensor, where the corresponding element is True if the condition is satisfied, otherwise False.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [False  True False  True False]

isnan
==============================

.. py:function:: pyvqnet.tensor.isnan(t)

    Checks element by element whether each value of the input is +/-NaN.

    :param t: Input QTensor.
    :return: Output QTensor, where the corresponding element is True if the condition is satisfied, otherwise False.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [False False False False  True]

isneginf
==============================

.. py:function:: pyvqnet.tensor.isneginf(t)

    Checks element by element whether each value of the input is -INF.

    :param t: Input QTensor.
    :return: Output QTensor, where the corresponding element is True if the condition is satisfied, otherwise False.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [False False False  True False]

isposinf
==============================

.. py:function:: pyvqnet.tensor.isposinf(t)

    Checks element by element whether each value of the input is +INF.

    :param t: Input QTensor.
    :return: Output QTensor, where the corresponding element is True if the condition is satisfied, otherwise False.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [False  True False False False]

logical_and
==============================

.. py:function:: pyvqnet.tensor.logical_and(t1, t2)

    Performs an element-wise logical AND on the two inputs, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)
        print(flag)

        # [False False  True False]

logical_or
==============================

.. py:function:: pyvqnet.tensor.logical_or(t1, t2)

    Performs an element-wise logical OR on the two inputs, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)
        print(flag)

        # [ True  True  True False]

logical_not
==============================

.. py:function:: pyvqnet.tensor.logical_not(t)

    Performs an element-wise logical NOT on the input, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t: Input QTensor.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [ True False False  True]

logical_xor
==============================

.. py:function:: pyvqnet.tensor.logical_xor(t1, t2)

    Performs an element-wise logical XOR on the two inputs, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)
        print(flag)

        # [ True  True False False]

greater
==============================

.. py:function:: pyvqnet.tensor.greater(t1, t2)

    Compares element by element whether t1 is greater than t2, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)
        print(flag)

        # [[False  True]
        #  [False False]]

greater_equal
==============================

.. py:function:: pyvqnet.tensor.greater_equal(t1, t2)

    Compares element by element whether t1 is greater than or equal to t2, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)
        print(flag)

        #[[ True  True]
        # [False  True]]

less
==============================

.. py:function:: pyvqnet.tensor.less(t1, t2)

    Compares element by element whether t1 is less than t2, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)
        print(flag)

        #[[False False]
        # [ True False]]

less_equal
==============================

.. py:function:: pyvqnet.tensor.less_equal(t1, t2)

    Compares element by element whether t1 is less than or equal to t2, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)
        print(flag)


        # [[ True False]
        #  [ True  True]]

equal
==============================

.. py:function:: pyvqnet.tensor.equal(t1, t2)

    Compares element by element whether t1 is equal to t2, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)
        print(flag)

        #[[ True False]
        # [False  True]]

not_equal
==============================

.. py:function:: pyvqnet.tensor.not_equal(t1, t2)

    Compares element by element whether t1 is not equal to t2, returning True where the corresponding element satisfies the condition and False otherwise.

    :param t1: Input QTensor.
    :param t2: Input QTensor.

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)

        #[[False  True]
        # [ True False]]



bitwise_and
==============================

.. py:function:: pyvqnet.tensor.bitwise_and(t1, t2)

    Computes the element-wise bitwise AND of two QTensors.

    :param t1: Input QTensor t1. Only integers or booleans are valid inputs.
    :param t2: Input QTensor t2. Only integers or booleans are valid inputs.

    Example::

        from pyvqnet.tensor import *
        import numpy as np
        from pyvqnet.dtype import *
        powers_of_two = 1 << np.arange(14, dtype=np.int64)[::-1]
        samples = tensor.QTensor([23],dtype=kint8)
        samples = samples.unsqueeze(-1)
        states_sampled_base_ten = samples & tensor.QTensor(powers_of_two,dtype = samples.dtype, device = samples.device)
        print(states_sampled_base_ten)
        #[[ 0, 0, 0, 0, 0, 0, 0, 0, 0,16, 0, 4, 2, 1]]


Transformation Functions
*************************************************************************************

broadcast
==============================

.. py:function:: pyvqnet.tensor.broadcast(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Under certain restrictions, the smaller array is broadcast across the larger array so that they have compatible shapes. This interface supports automatic differentiation of the input tensors.

    See https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: Input QTensor 1
    :param t2: Input QTensor 2

    :return t11: t1 with the new broadcast shape.
    :return t22: t2 with the new broadcast shape.

    Example::

        from pyvqnet.tensor import *
        t1 = ones([5,4])
        t2 = ones([4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        t1 = ones([5,4])
        t2 = ones([1])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        t1 = ones([5,4])
        t2 = ones([2,1,4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [2, 5, 4]
        # [2, 5, 4]

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Takes an index position in string form, obtains the data slice at that index, and returns a new QTensor.

    :param t: Input QTensor.
    :param index: A string containing the slice index.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        t = QTensor(np.arange(1,25.0).reshape(2,3,4))

        indx = [":", "0", ":"]
        t.requires_grad = True
        t.zero_grad()
        ts = tensor.select(t,indx)

        print(ts)
        # [
        # [[1., 2., 3., 4.]],
        # [[13., 14., 15., 16.]]
        # ]

concatenate
==============================

.. py:function:: pyvqnet.tensor.concatenate(args: list, dim=0)

    Concatenates multiple QTensors in args along dimension dim and returns a new QTensor.

    :param args: Contains the input QTensors.
    :param dim: Dimension to concatenate along. Must be between 0 and the maximum dimension of the input tensors.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = QTensor([[1.0, 2, 3],[4,5,6]], requires_grad=True)
        y = 1-x
        x = tensor.concatenate([x,y],1)
        print(x)

        # [
        # [1., 2., 3., 0., -1., -2.],
        # [4., 5., 6., -3., -4., -5.]
        # ]



stack
==============================

.. py:function:: pyvqnet.tensor.stack(QTensors: list, dim=0)

    Stacks the input QTensors along a new dimension dim and returns a new QTensor.

    :param QTensors: Contains the input QTensors.
    :param dim: Dimension to stack along. Must be between 0 and the maximum dimension of the input tensors.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t11 = QTensor(a)
        t22 = QTensor(a)
        t33 = QTensor(a)
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
==============================

.. py:function:: pyvqnet.tensor.permute(t: pyvqnet.tensor.QTensor, dim: list)

    Changes the order of the axes of t according to the order of the input dim.

    :param t: Input QTensor.
    :param dim: New order of dimensions (a list of integers).
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
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
==============================

.. py:function:: pyvqnet.tensor.transpose(t: pyvqnet.tensor.QTensor, dim: list)

    Reorder the axes of t according to the order of the input dim. If dims = None, the axes of t are reversed in order.

    :param t: Input QTensor.
    :param dim: New order of the dimensions (a list of integers).

    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
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
==============================

.. py:function:: pyvqnet.tensor.tile(t: pyvqnet.tensor.QTensor, reps: list)


    Copy the input QTensor the number of times given by reps.

    If reps has length d, the resulting QTensor has max(d, t.ndim) dimensions. If t.ndim < d, t is promoted to d dimensions by prepending new axes.

    Thus an array of shape (3,) is promoted to (1, 3) for 2-D replication, or (1, 1, 3) for 3-D replication. If t.ndim > d, reps is extended to t.ndim by prepending 1s.

    So for t of shape (2, 3, 4, 5), reps of (4, 3) is treated as (1, 1, 4, 3).

    :param t: Input QTensor.
    :param reps: Number of repetitions for each dimension.
    :return: A new QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        a = np.arange(6).reshape(2,3).astype(np.float32)
        A = QTensor(a)
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
==============================

.. py:function:: pyvqnet.tensor.squeeze(t: pyvqnet.tensor.QTensor, dim: Optional[int] = None)

    Remove the axis specified by dim, which must have size 1. If dim = None, all dimensions of size 1 in the input are removed.

    :param t: Input QTensor.
    :param dim: Axis to squeeze. Default: None.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = QTensor(a)
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
==============================

.. py:function:: pyvqnet.tensor.unsqueeze(t: pyvqnet.tensor.QTensor, dim: int = 0)

    Insert a new axis of size 1 at the dimension specified by dim and return a new QTensor.

    :param t: Input QTensor.
    :param dim: Position at which to insert the dimension. Default: 0.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = QTensor(a)
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


moveaxis
==============================
.. py:function:: pyvqnet.tensor.moveaxis(t, src, dst)

    Move the dimensions of `t` from the positions in `src` to the positions in `dst`.

    Other dimensions of `t` that are not explicitly moved keep their original order and appear at the positions in `dst` that are not specified.

    :param t: Input QTensor.
    :param src: (int or tuple of ints) Original positions of the dimensions to move. These positions must be unique.
    :param dst: (int or tuple of ints) Destination position for each original dimension. These positions must also be unique.

    :return: A new QTensor.

    Example::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)
        #[3, 4, 2]


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, dim1: int, dim2: int)

    Swap the dim1 and dim2 dimensions of the input t.

    :param t: Input QTensor.
    :param dim1: First axis to swap.
    :param dim2: Second axis to swap.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = QTensor(a)
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
==============================

.. py:function:: pyvqnet.tensor.masked_fill(t, mask, value)

    Fill the input with value where mask == 1.
    The shape of mask must be broadcastable to the shape of the input QTensor.

    :param t: Input QTensor.
    :param mask: Mask QTensor, which must be kbool.
    :param value: Fill value.
    :return: A QTensor.

    Example::

        from pyvqnet.tensor import tensor
        import numpy as np
        a = tensor.ones([2, 2, 2, 2])
        mask = np.random.randint(0, 2, size=4).reshape([2, 2])
        b = tensor.QTensor(mask==1)
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
==============================

.. py:function:: pyvqnet.tensor.flatten(t: pyvqnet.tensor.QTensor, start: int = 0, end: int = - 1)

    Flatten the contiguous dimensions of the input t from start to end.

    :param t: Input QTensor.
    :param start: Axis where flattening starts. Default = 0, starting from the first axis.
    :param end: Axis where flattening ends. Default = -1, ending at the last axis.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1., 2., 3.]


reshape
==============================

.. py:function:: pyvqnet.tensor.reshape(t: pyvqnet.tensor.QTensor,shape)

    Change the shape of the QTensor and return a new tensor.

    :param t: Input QTensor.
    :param shape: New shape.

    :return: A QTensor with the new shape.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        reshape_t = tensor.reshape(t, [C, R])
        print(reshape_t)
        # [
        # [0., 1., 2.],
        # [3., 4., 5.],
        # [6., 7., 8.],
        # [9., 10., 11.]
        # ]

flip
==============================

.. py:function:: pyvqnet.tensor.flip(t, flip_dims)

    Reverse the QTensor along the specified axes and return a new tensor.

    :param t: Input QTensor.
    :param flip_dims: Axis or list of axes to flip.

    :return: A QTensor with the new shape.

    Example::

        from pyvqnet import tensor
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

gather
==============================

.. py:function:: pyvqnet.tensor.gather(t, dim, index)

    Gather values along the axis specified by `dim`.

    For a 3-D tensor, the output is specified by:

    .. math::

         out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \\

         out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \\

         out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \\

    :param t: Input QTensor.
    :param dim: Gather axis.
    :param index: Index QTensor, which must have the same number of dimensions as the input.

    :return: The gathered result.

    Example::

        from pyvqnet.tensor import gather,QTensor,tensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn( 3, 4,6)
        npindex = np.array([2,3,1,2,1,2,3,0,2,3,1,2,3,2,0,1]).reshape([2,2,4]).astype(np.int64)

        x1 = QTensor(npx)
        indices1 =  QTensor(npindex)
        x1.requires_grad = True
        y1 = gather(x1,1,indices1)
        y1.backward(tensor.arange(0,y1.numel()).reshape(y1.shape))

        print(y1)
        # [
        # [[2.1523438, -0.4196777, -2.0527344, -1.2460938],
        #  [-0.6201172, -1.3349609, 2.2949219, -0.5913086]],
        # [[0.2170410, -0.7055664, 1.6074219, -1.9394531],
        #  [0.2430420, -0.6333008, 0.5332031, 0.3881836]]
        # ]


scatter
==============================

.. py:function:: pyvqnet.tensor.scatter(input, dim, indices, src)


    Write all values in tensor src into input at the indices specified in the indices tensor.

    For a 3-D tensor, the output is specified by:

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \\

    :param input: Input QTensor.
    :param dim: Scatter axis.
    :param indices: Index QTensor, which must have the same number of dimensions as the input.
    :param src: Source tensor whose values are scattered.

    Example::

        from pyvqnet.tensor import scatter, QTensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn(3, 2, 4, 2)
        npindex = np.array([2, 3, 1, 2, 1, 2, 3, 0, 2, 3, 1, 2, 3, 2, 0,
                            1]).reshape([2, 2, 4, 1]).astype(np.int64)
        x1 = QTensor(npx)
        npsrc = QTensor(np.full_like(npindex, 200), dtype=x1.dtype)
        npsrc.requires_grad = True
        indices1 = QTensor(npindex)
        y1 = scatter(x1, 2, indices1, npsrc)
        print(y1)

        # [[[[  0.2282731   1.0268903]
        #    [200.         -0.5911815]
        #    [200.         -0.2223257]
        #    [200.          1.8379046]]

        #   [[200.          0.8685831]
        #    [200.         -0.2323119]
        #    [200.         -1.3346615]
        #    [200.         -1.2460893]]]


        #  [[[  1.2022723  -1.0499416]
        #    [200.         -0.4196777]
        #    [200.         -2.5944874]
        #    [200.          0.6808889]]

        #   [[200.         -1.9762536]
        #    [200.         -0.2908697]
        #    [200.          1.9826261]
        #    [200.         -1.839905 ]]]


        #  [[[  1.6076708   0.3882919]
        #    [  0.3997321   0.4054766]
        #    [  0.2170018  -0.6334391]
        #    [  0.2466215  -1.9395455]]

        #   [[  0.1140596  -1.8853414]
        #    [  0.2430805  -0.7054807]
        #    [  0.3646276  -0.5029522]
        #    [ -0.2257515  -0.5655377]]]]

broadcast_to
==============================

.. py:function:: pyvqnet.tensor.broadcast_to(t, ref)

    Subject to certain constraints, array t is broadcast to the reference shape so that they have compatible shapes.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: Input QTensor
    :param ref: Reference shape.

    :return: The newly broadcast QTensor of t.

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



Utility Functions
*****************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Convert the input number or numpy.ndarray, etc. to a QTensor.

    :param x: Integer, float, bool, complex, or numpy.ndarray
    :return: Output QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.to_tensor(10.0)
        print(t)

        # [10.]


pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    Pad a list of variable-length tensors with ``padding_value``. ``pad_sequence`` stacks the list of tensors along a new dimension and pads them to equal length.
    The input is a list of sequences of size ``L x *``, where L is variable length.

    :param qtensor_list: `list[QTensor]` - List of variable-length sequences.
    :param batch_first: 'bool' - If true, the output will be ``batch size x longest sequence length x *``; otherwise ``longest sequence length x batch size x *``. Default: False.
    :param padding_value: 'float' - Padding value. Default: 0.

    :return:
        If batch_first is ``False``, the tensor size is ``batch size x longest sequence length x *``.
        Otherwise the tensor size is ``longest sequence length x batch size x *``.

    Example::

        from pyvqnet.tensor import tensor
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
==============================

.. py:function:: pyvqnet.tensor.pad_packed_sequence(sequence, batch_first=False, padding_value=0, total_length=None)

    Pad a batch of packed variable-length sequences. It is the inverse operation of `pack_pad_sequence`.
    When ``batch_first`` is True, it returns a tensor of shape ``B x T x *``; otherwise it returns ``T x B x *``.
    Here `T` is the longest sequence length and `B` is the batch size.



    :param sequence: 'QTensor' - Data to process.
    :param batch_first: 'bool' - If ``True``, the batch dimension is the first dimension of the input. Default: False.
    :param padding_value: 'bool' - Padding value. Default: 0.
    :param total_length: 'bool' - If not ``None``, the output is padded to the length :attr:`total_length`. Default: None.
    :return:
        A tuple containing the padded sequences tensor and a list of the lengths of each sequence in the batch. Batch elements are reordered to their original order.

    Example::

        from pyvqnet.tensor import tensor
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
==============================

.. py:function:: pyvqnet.tensor.pack_pad_sequence(input, lengths, batch_first=False, enforce_sorted=True)

    Pack a tensor containing variable-length padded sequences.
    If batch_first is True, ``input`` should have shape ``[batch_size, length, *]``; otherwise ``[length, batch_size, *]``.

    For unsorted sequences, set ``enforce_sorted`` to False. If :attr:`enforce_sorted` is ``True``, the sequences must be sorted in descending order of length.

    :param input: 'QTensor' - Padded variable-length sequences.
    :param lengths: 'list' - Sequence length of each batch.
    :param batch_first: 'bool' - If ``True``, the input is expected in ``B x T x *``
        format. Default: False.
    :param enforce_sorted: 'bool' - If ``True``, the input must
        contain sequences sorted in descending order of length. If ``False``, the input will be sorted unconditionally. Default: True.

    :return: A :class:`PackedSequence` object.

    Example::

        from pyvqnet.tensor import tensor
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

functional_conv2d
==============================
.. py:function:: pyvqnet.nn.functional.functional_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1)

    Apply a 2-D convolution over an input image composed of multiple input planes.

    :param x: 4-D input tensor.
    :param weight: 4-D kernel tensor.


    :param stride: `tuple` - Stride. Default: (1, 1)
    :param padding: Padding, controls the amount of padding applied to the input. It can be a string {'valid', 'same'} or a tuple of ints specifying the implicit padding applied to the input. Default: (0,0)
    :param dilation: `tuple` - Spacing between kernel elements. Default: (0,0)
    :param groups: `int` - Number of groups. Default: 1

    :return: qtensor


    Example::

        from pyvqnet.nn.functional import functional_conv2d
        from pyvqnet.tensor import arange,ones
        from pyvqnet import kfloat32
        from pyvqnet.nn import Module,Parameter


        class TM(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.w = ones([5,4,2,2])
                self.w.requires_grad = True
                self.b = ones([5,])
                self.b.requires_grad = True

            def forward(self,x):
                weight, bias,   = self.w, self.b
                return functional_conv2d(x, weight, bias )


        x = arange(0,7*4*12*12,dtype=kfloat32).reshape([7,4,12,12])
        l = TM()
        y = l(x)

        y.backward( )

no_grad
==============================

.. py:function:: pyvqnet.no_grad()

    Disable recording of backpropagation nodes during forward computation.

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet import no_grad

        with no_grad():
            x = tensor.QTensor([1.0, 2.0, 3.0],requires_grad=True)
            y = tensor.tan(x)
            y.backward()
        """
        RuntimeError: The output tensor does not require gradients (output.requires_grad == False). This may occur if you used a non-autograd function in the forward pass. To enable gradient computation, ensure that all operations are performed on tensors with requires_grad=True, or use autograd-compatible functions.
        """
