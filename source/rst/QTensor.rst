QTensor Module
###########################

VQNet quantum machine learning uses the data structure QTensor which is Python interface. QTensor supports common multidimensional matrix operations including creating functions, mathematical functions, logical functions, matrix transformations, etc.




QTensor's Functions and Attributes
******************************************


__init__
==============================

.. py:method:: QTensor.__init__(data, requires_grad=False, nodes=None, device=0, dtype=None, name='')

    Wrapper of data structure with dynamic computational graph construction
    and automatic differentiation.

    :param data: _core.Tensor or numpy array which represents a QTensor
    :param requires_grad: should tensor's gradient be tracked, defaults to False
    :param nodes: list of successors in the computational graph, defaults to None
    :param device: current device to save QTensor ,default = 0, use CPU.
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param name: The name of the QTensor, default: "".

    :return: output QTensor

    .. note::
            QTensor internal data type dtype support: kbool,kuint8,kint8,kint16,kint32,kint64,kfloat32,kfloat64,kcomplex64,kcomplex128.

            Representing C++ type: bool,uint8_t,int8_t,int16_t,int32_t,int64_t,float,double,complex<float>,complex<double>.

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

ndim
==============================

.. py:method:: QTensor.ndim

    Return number of dimensions

    :return: number of dimensions

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.ndim)

        # 1

shape
==============================

.. py:method:: QTensor.shape

    Return the shape of the QTensor.

    :return: value of shape

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.shape)

        # [4]

size
==============================

.. py:method:: QTensor.size

    Return the number of elements in the QTensor.

    :return: number of elements

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.size)

        # 4

numel
==============================

.. py:method:: QTensor.numel()
    
    Returns the number of elements in the tensor.

    :return: The number of elements in the tensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.numel())

        # 4

dtype
=============================

.. py:attribute:: QTensor.dtype

    Returns the data type of the tensor.

    QTensor internal data type dtype supports kbool=0, kuint8=1, kint8=2, kint16=3, kint32=4, 
    kint64=5, kfloat32=6, kfloat64=7, kcomplex64=8, kcomplex128=9.

    :return: The data type of the tensor.

    Example::

        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5])
        print(a.dtype)
        #4

is_dense
==============================

.. py:attribute:: QTensor.is_dense

    Whether it is a dense tensor.

    :return: Returns 1 when the data is dense; otherwise returns 0.

    Example::

        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5])
        print(a.is_dense)
        #1

is_csr
==============================

.. py:attribute:: QTensor.is_csr

    Whether it is a sparse 2-dimensional matrix in Compressed Sparse Row format.

    :return: When the data is a sparse tensor in CSR format, return 1; otherwise, return 0.

    Example::

        from pyvqnet.tensor import QTensor,dense_to_csr

        a = QTensor([[2, 3, 4, 5]])
        b = dense_to_csr(a)
        print(b.is_csr)
        #1

csr_members
==============================

.. py:method:: QTensor.csr_members()

    Returns the row_idx, col_idx and non-zero numerical data of the sparse 2-dimensional matrix in Compressed Sparse Row format, and three 1-dimensional QTensors. For the specific meaning, see https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format).
    
    :return:

        Returns a list in which the first element is row_idx, shape is [number of matrix rows + 1],
         the second element is col_idx, shape is [number of non-zero elements], the third element is data, shape is [number of non-zero elements].

    Example::

        from pyvqnet.tensor import QTensor,dense_to_csr

        a = QTensor([[2, 3, 4, 5]])
        b = dense_to_csr(a)
        print(b.csr_members())
        #([0,4], [0,1,2,3], [2,3,4,5])

zero_grad
==============================

.. py:method:: QTensor.zero_grad()

    Sets gradient to zero. Will be used by optimizer in the optimization process.

    :return: None

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t3  =  QTensor([2,3,4,5],requires_grad = True)
        t3.zero_grad()
        print(t3.grad)

        # [0, 0, 0, 0]


backward
==============================

.. py:method:: QTensor.backward(grad=None)

    Computes the gradient of current QTensor .

    :return: None

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
        y = 2*target + 3
        y.backward()
        print(target.grad)
        #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

to_numpy
==============================

.. py:method:: QTensor.to_numpy()

    Copy self data to a new numpy.array.

    :return: a new numpy.array contains QTensor data

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t3  =  QTensor([2,3,4,5],requires_grad = True)
        t4 = t3.to_numpy()
        print(t4)

        # [2. 3. 4. 5.]

item
==============================

.. py:method:: QTensor.item()

        Return the only element from in the QTensor.Raises 'RuntimeError' if QTensor has more than 1 element.

        :return: only data of this object

        Example::

            from pyvqnet.tensor import tensor

            t = tensor.ones([1])
            print(t.item())

            # 1.0

argmax
==============================

.. py:method:: QTensor.argmax(*kargs)

    Return the indices of the maximum value of all elements in the input QTensor,or
    Return the indices of the maximum values of a QTensor across a dimension.

    :param dim: dim (int) – the dimension to reduce,only accepts single axis. if dim == None, returns the indices of the maximum value of all elements in the input tensor.The valid dim range is [-R, R), where R is input's ndim. when dim < 0, it works the same way as dim + R.
    :param keepdims:  whether the output QTensor has dim retained or not.

    :return: the indices of the maximum value in the input QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmax()
        print(flag)
        
        # [0]

        flag_0 = a.argmax([0], True)
        print(flag_0)

        # [
        # [0, 3, 0, 3]
        # ]

        flag_1 = a.argmax([1], True)
        print(flag_1)

        # [
        # [0],
        # [2],
        # [0],
        # [1]
        # ]

argmin
==============================

.. py:method:: QTensor.argmin(*kargs)

    Return the indices of the minimum  value of all elements in the input QTensor,or
    Return the indices of the minimum  values of a QTensor across a dimension.

    :param dim: dim (int) – the dimension to reduce,only accepts single axis. if dim == None, returns the indices of the minimum value of all elements in the input tensor.The valid dim range is [-R, R), where R is input's ndim. when dim < 0, it works the same way as dim + R.
    :param keepdims:  whether the output QTensor has dim retained or not.

    :return: the indices of the minimum  value in the input QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmin()
        print(flag)

        # [12]

        flag_0 = a.argmin([0], True)
        print(flag_0)

        # [
        # [3, 2, 2, 1]
        # ]

        flag_1 = a.argmin([1], False)
        print(flag_1)

        # [2, 3, 1, 0]

fill\_
==============================

.. py:method:: QTensor.fill_(v)

        Fill the QTensor with the specified value inplace.

        :param v: a scalar value
        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            shape = [2, 3]
            value = 42
            t = tensor.zeros(shape)
            t.fill_(value)
            print(t)

            # [
            # [42, 42, 42],
            # [42, 42, 42]
            # ]

all
==============================

.. py:method:: QTensor.all()

        Return True, if all QTensor value is non-zero.

        :return: True,if all QTensor value is non-zero.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            shape = [2, 3]
            t = tensor.zeros(shape)
            t.fill_(1.0)
            flag = t.all()
            print(flag)

            # True

any
==============================

.. py:method:: QTensor.any()

        Return True,if any QTensor value is non-zero.

        :return: True,if any QTensor value is non-zero.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor

            shape = [2, 3]
            t = tensor.ones(shape)
            t.fill_(1.0)
            flag = t.any()
            print(flag)

            # True

fill_rand_binary\_
==============================

.. py:method:: QTensor.fill_rand_binary_(v=0.5)

    Fills a QTensor with values randomly sampled from a binomial distribution.

    If the data generated randomly after binomial distribution is greater than Binarization threshold,then the number of corresponding positions of the QTensor is set to 1, otherwise 0.

    :param v: Binarization threshold
    :return: None

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        t.fill_rand_binary_(2)
        print(t)

        # [
        # [1, 1, 1],
        # [1, 1, 1]
        # ]

fill_rand_signed_uniform\_
==============================

.. py:method:: QTensor.fill_rand_signed_uniform_(v=1)

    Fills a QTensor with values randomly sampled from a signed uniform distribution.

    Scale factor of the values generated by the signed uniform distribution.

    :param v: a scalar value
    :return: None

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

fill_rand_uniform\_
==============================

.. py:method:: QTensor.fill_rand_uniform_(v=1)

    Fills a QTensor with values randomly sampled from a uniform distribution

    Scale factor of the values generated by the uniform distribution.

    :param v: a scalar value
    :return: None

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

fill_rand_normal\_
==============================

.. py:method:: QTensor.fill_rand_normal_(m=0, s=1, fast_math=True)

        Fills a QTensor with values randomly sampled from a normal distribution
        Mean of the normal distribution. Standard deviation of the normal distribution.
        Whether to use or not the fast math mode.

        :param m: mean of the normal distribution
        :param s: standard deviation of the normal distribution
        :param fast_math: True if use fast-math
        :return: None

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


QTensor.transpose
==============================

.. py:method:: QTensor.transpose(new_dims=None)

    Reverse or permute the axes of an array.if new_dims = None, revsers the dim.

    :param new_dims: the new order of the dimensions (list of integers).
    :return:  result QTensor.

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
        # [[0, 3],
        #  [6, 9]],
        # [[1, 4],
        #  [7, 10]],
        # [[2, 5],
        #  [8, 11]]
        # ]

transpose\_
==============================

.. py:method:: QTensor.transpose_(new_dims=None)

    Reverse or permute the axes of an array inplace.if new_dims = None, revsers the dim.

    :param new_dims: the new order of the dimensions (list of integers).
    :return: None.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
        t = QTensor(a)
        t.transpose_([2, 0, 1])
        print(t)

        # [
        # [[0, 3],
        #  [6, 9]],
        # [[1, 4],
        #  [7, 10]],
        # [[2, 5],
        #  [8, 11]]
        # ]

QTensor.reshape
==============================

.. py:method:: QTensor.reshape(new_shape)

    Change the tensor’s shape ,return a new QTensor.

    :param new_shape: the new shape (list of integers)
    :return: a new QTensor

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
        # [0, 1, 2],
        # [3, 4, 5],
        # [6, 7, 8],
        # [9, 10, 11]
        # ]

reshape\_
==============================

.. py:method:: QTensor.reshape_(new_shape)

        Change the current object's shape.

        :param new_shape: the new shape (list of integers)
        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape(R, C).astype(np.float32)
            t = QTensor(a)
            t.reshape_([C, R])
            print(t)

            # [
            # [0, 1, 2],
            # [3, 4, 5],
            # [6, 7, 8],
            # [9, 10, 11]
            # ]

getdata
==============================

.. py:method:: QTensor.getdata()

        Get the QTensor's data as a NumPy array.

        :return: a NumPy array

        Example::


            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor

            t = tensor.ones([3, 4])
            a = t.getdata()
            print(a)

            # [[1. 1. 1. 1.]
            #  [1. 1. 1. 1.]
            #  [1. 1. 1. 1.]]

__getitem__
==============================

.. py:method:: QTensor.__getitem__()

        Slicing indexing of QTensor is supported, or using QTensor as advanced index access input. A new QTensor will be returned.

        The parameters start, stop, and step can be separated by a colon,such as start:stop:step, where start, stop, and step can be default

        As a 1-D QTensor,indexing or slicing can only be done on a single axis.

        As a 2-D QTensor and a multidimensional QTensor,indexing or slicing can be done on multiple axes.

        If you use QTensor as an index for advanced indexing, see numpy for `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

        If your QTensor as an index is the result of a logical operation, then you do a Boolean index.

        .. note:: We use an index form like a[3,4,1],but the form a[3][4][1] is not supported.And ``Ellipsis`` is also not supported.

        :param item: A integer or QTensor as an index.

        :return: A new QTensor.

        Example::

            from pyvqnet.tensor import tensor, QTensor
            aaa = tensor.arange(1, 61)
            aaa.reshape_([4, 5, 3])
            print(aaa[0:2, 3, :2])
            # [
            # [10, 11],
            #  [25, 26]
            # ]
            print(aaa[3, 4, 1])
            #[59]
            print(aaa[:, 2, :])
            # [
            # [7, 8, 9],
            #  [22, 23, 24],
            #  [37, 38, 39],
            #  [52, 53, 54]
            # ]
            print(aaa[2])
            # [
            # [31, 32, 33],
            #  [34, 35, 36],
            #  [37, 38, 39],
            #  [40, 41, 42],
            #  [43, 44, 45]
            # ]
            print(aaa[0:2, ::3, 2:])
            # [
            # [[3],
            #  [12]],
            # [[18],
            #  [27]]
            # ]
            a = tensor.ones([2, 2])
            b = QTensor([[1, 1], [0, 1]])
            b = b > 0
            c = a[b]
            print(c)
            #[1, 1, 1]
            tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
            tt.requires_grad = True
            index_sample1 = tensor.arange(0, 3).reshape([3, 1])
            index_sample2 = QTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
            gg = tt[:, index_sample1, 3:, index_sample2, 2:]
            print(gg)
            # [
            # [[[[87, 88]],
            # [[983, 984]]],
            # [[[91, 92]],
            # [[987, 988]]],
            # [[[87, 88]],
            # [[983, 984]]]],
            # [[[[207, 208]],
            # [[1103, 1104]]],
            # [[[211, 212]],
            # [[1107, 1108]]],
            # [[[207, 208]],
            # [[1103, 1104]]]],
            # [[[[319, 320]],
            # [[1215, 1216]]],
            # [[[323, 324]],
            # [[1219, 1220]]],
            # [[[323, 324]],
            # [[1219, 1220]]]]
            # ]

__setitem__
==============================

.. py:method:: QTensor.__setitem__()

    Slicing indexing of QTensor is supported, or using QTensor as advanced index access input. A new QTensor will be returned.

    The parameters start, stop, and step can be separated by a colon,such as start:stop:step, where start, stop, and step can be default

    As a 1-D QTensor,indexing or slicing can only be done on a single axis.

    As a 2-D QTensor and a multidimensional QTensor,indexing or slicing can be done on multiple axes.

    If you use QTensor as an index for advanced indexing, see numpy for `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

    If your QTensor as an index is the result of a logical operation, then you do a Boolean index.

    .. note:: We use an index form like a[3,4,1],but the form a[3][4][1] is not supported.And ``Ellipsis`` is also not supported.

    :param item: A integer or QTensor as an index

    :return: None


    Example::

        from pyvqnet.tensor import tensor
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a2 = aaa[3, 4, 1]
        aaa[3, 4, 1] = tensor.arange(10001,
                                        10001 + vqnet_a2.size).reshape(vqnet_a2.shape)
        print(aaa)
        # [
        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9],
        #  [10, 11, 12],
        #  [13, 14, 15]],
        # [[16, 17, 18],
        #  [19, 20, 21],
        #  [22, 23, 24],
        #  [25, 26, 27],
        #  [28, 29, 30]],
        # [[31, 32, 33],
        #  [34, 35, 36],
        #  [37, 38, 39],
        #  [40, 41, 42],
        #  [43, 44, 45]],
        # [[46, 47, 48],
        #  [49, 50, 51],
        #  [52, 53, 54],
        #  [55, 56, 57],
        #  [58, 10001, 60]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a3 = aaa[:, 2, :]
        aaa[:, 2, :] = tensor.arange(10001,
                                        10001 + vqnet_a3.size).reshape(vqnet_a3.shape)
        print(aaa)
        # [
        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [10001, 10002, 10003],
        #  [10, 11, 12],
        #  [13, 14, 15]],
        # [[16, 17, 18],
        #  [19, 20, 21],
        #  [10004, 10005, 10006],
        #  [25, 26, 27],
        #  [28, 29, 30]],
        # [[31, 32, 33],
        #  [34, 35, 36],
        #  [10007, 10008, 10009],
        #  [40, 41, 42],
        #  [43, 44, 45]],
        # [[46, 47, 48],
        #  [49, 50, 51],
        #  [10010, 10011, 10012],
        #  [55, 56, 57],
        #  [58, 59, 60]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a4 = aaa[2, :]
        aaa[2, :] = tensor.arange(10001,
                                    10001 + vqnet_a4.size).reshape(vqnet_a4.shape)
        print(aaa)
        # [
        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9],
        #  [10, 11, 12],
        #  [13, 14, 15]],
        # [[16, 17, 18],
        #  [19, 20, 21],
        #  [22, 23, 24],
        #  [25, 26, 27],
        #  [28, 29, 30]],
        # [[10001, 10002, 10003],
        #  [10004, 10005, 10006],
        #  [10007, 10008, 10009],
        #  [10010, 10011, 10012],
        #  [10013, 10014, 10015]],
        # [[46, 47, 48],
        #  [49, 50, 51],
        #  [52, 53, 54],
        #  [55, 56, 57],
        #  [58, 59, 60]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a5 = aaa[0:2, ::2, 1:2]
        aaa[0:2, ::2,
            1:2] = tensor.arange(10001,
                                    10001 + vqnet_a5.size).reshape(vqnet_a5.shape)
        print(aaa)
        # [
        # [[1, 10001, 3],
        #  [4, 5, 6],
        #  [7, 10002, 9],
        #  [10, 11, 12],
        #  [13, 10003, 15]],
        # [[16, 10004, 18],
        #  [19, 20, 21],
        #  [22, 10005, 24],
        #  [25, 26, 27],
        #  [28, 10006, 30]],
        # [[31, 32, 33],
        #  [34, 35, 36],
        #  [37, 38, 39],
        #  [40, 41, 42],
        #  [43, 44, 45]],
        # [[46, 47, 48],
        #  [49, 50, 51],
        #  [52, 53, 54],
        #  [55, 56, 57],
        #  [58, 59, 60]]
        # ]
        a = tensor.ones([2, 2])
        b = tensor.QTensor([[1, 1], [0, 1]])
        b = b > 0
        x = tensor.QTensor([1001, 2001, 3001])

        a[b] = x
        print(a)
        # [
        # [1001, 2001],
        #  [1, 3001]
        # ]
        
GPU
==============================

.. py:function:: QTensor.GPU(device: int = DEV_GPU_0)

    Clone QTensor to specified GPU device.

    device specifies the device whose internal data is stored. When device >= DEV_GPU_0, the data is stored on the GPU.
    If your computer has multiple GPUs, you can designate different devices to store data on. 
    For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indicates storage on GPUs with different serial numbers.
    
    .. note::
        QTensor cannot perform calculations on different GPUs.
        A Cuda error will be raised if you try to create a QTensor on a GPU whose ID exceeds the maximum number of verified GPUs.

    :param device: The device currently saving QTensor, default=DEV_GPU_0,
      device = pyvqnet.DEV_GPU_0, stored in the first GPU, devcie = DEV_GPU_1,
      stored in the second GPU, and so on.

    :return: Clone QTensor to GPU device.

    Examples::

        from pyvqnet.tensor import QTensor
        a = QTensor([2])
        b = a.GPU()
        print(b.device)
        #1000

CPU
==============================

.. py:function:: QTensor.CPU()

    Clone QTensor to specific CPU device

    :return: Clone QTensor to CPU device.

    Examples::

        from pyvqnet.tensor import QTensor
        a = QTensor([2])
        b = a.CPU()
        print(b.device)
        # 0

toGPU
==============================

.. py:function:: QTensor.toGPU(device: int = DEV_GPU_0)

    Move QTensor to specified GPU device.

    device specifies the device whose internal data is stored. When device >= DEV_GPU, the data is stored on the GPU.
    If your computer has multiple GPUs, you can designate different devices to store data on.
    For example, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indicates storage on GPUs with different serial numbers.

    .. note::
        QTensor cannot perform calculations on different GPUs.
         A Cuda error will be raised if you try to create a QTensor on a GPU whose ID exceeds the maximum number of verified GPUs.

    :param device: The device currently saving QTensor, default=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, stored in the first GPU, devcie = DEV_GPU_1, stored in the second GPU, and so on.
    :return: QTensor moved to GPU device.

    Examples::

        from pyvqnet.tensor import QTensor
        a = QTensor([2])
        a = a.toGPU()
        print(a.device)
        #1000


toCPU
==============================

.. py:function:: QTensor.toCPU()

    Move QTensor to specific GPU device

    :return: QTensor moved to CPU device.

    Examples::

        from pyvqnet.tensor import QTensor
        a = QTensor([2])
        b = a.toCPU()
        print(b.device)
        # 0


isGPU
==============================

.. py:function:: QTensor.isGPU()

    Whether this QTensor's data is stored on GPU host memory.

    :return: Whether this QTensor's data is stored on GPU host memory.

    Examples::
    
        from pyvqnet.tensor import QTensor
        a = QTensor([2])
        a = a.isGPU()
        print(a)
        # False

isCPU
==============================

.. py:function:: QTensor.isCPU()

    Whether this QTensor's data is stored in CPU host memory.

    :return: Whether this QTensor's data is stored in CPU host memory.

    Examples::
    
        from pyvqnet.tensor import QTensor
        a = QTensor([2])
        a = a.isCPU()
        print(a)
        # True


Create Functions
*****************************************************


ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    Return one-tensor with the input shape.

    :param shape: input shape
    :param device: stored in which device，default 0 , CPU.
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output QTensor with the input shape.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = tensor.ones([2,3])
        print(x)

        # [
        # [1, 1, 1],
        # [1, 1, 1]
        # ]

ones_like
==============================

.. py:function:: pyvqnet.tensor.ones_like(t: pyvqnet.tensor.QTensor,device=0,dtype=None)

    Return one-tensor with the same shape as the input QTensor.

    :param t: input QTensor
    :param device: stored in which device，default 0 , CPU.
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return:  output QTensor


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)
        print(x)

        # [1, 1, 1]

full
==============================

.. py:function:: pyvqnet.tensor.full(shape, value, device=0, dtype=None)

    Create a QTensor of the specified shape and fill it with value.

    :param shape: shape of the QTensor to create
    :param value: value to fill the QTensor with.
    :param device: device to use,default = 0 ,use cpu device.
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        value = 42
        t = tensor.full(shape, value)
        print(t)
        # [
        # [42, 42, 42],
        # [42, 42, 42]
        # ]

full_like
==============================

.. py:function:: pyvqnet.tensor.full_like(t, value, device: int = 0, dtype=None)

    Create a QTensor of the specified shape and fill it with value.

    :param t:  input Qtensor
    :param value: value to fill the QTensor with.
    :param device: device to use,default = 0 ,use cpu device.
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = tensor.randu([3,5])
        value = 42
        t = tensor.full_like(a, value)
        print(t)
        # [
        # [42, 42, 42, 42, 42],
        # [42, 42, 42, 42, 42],
        # [42, 42, 42, 42, 42]
        # ]

zeros
==============================

.. py:function:: pyvqnet.tensor.zeros(shape，device = 0,dtype=None)

    Return zero-tensor of the input shape.

    :param shape: shape of tensor
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.zeros([2, 3, 4])
        print(t)
        # [
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]],
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]
        # ]


zeros_like
==============================

.. py:function:: pyvqnet.tensor.zeros_like(t: pyvqnet.tensor.QTensor,device: int = 0,dtype=None))

    Return zero-tensor with the same shape as the input QTensor.

    :param t: input QTensor
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)
        print(x)

        # [0, 0, 0]

arange
==============================

.. py:function:: pyvqnet.tensor.arange(start, end, step=1, device: int = 0,dtype=None, requires_grad=False)

    Create a 1D QTensor with evenly spaced values within a given interval.

    :param start: start of interval
    :param end: end of interval
    :param step: spacing between values
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30,4)
        print(t)

        # [ 2,  6, 10, 14, 18, 22, 26]

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None, requires_grad= False)

    Create a 1D QTensor with evenly spaced values within a given interval.

    :param start: starting value
    :param end: end value
    :param nums: number of samples to generate
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps = -2.5, 10, 10
        t = tensor.linspace(start, stop, steps)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10]

logspace
==============================

.. py:function:: pyvqnet.tensor.logspace(start, end, num, base, device: int = 0,dtype=None,  requires_grad)

    Create a 1D QTensor with evenly spaced values on a log scale.

    :param start: ``base ** start`` is the starting value
    :param end: ``base ** end`` is the final value of the sequence
    :param nums: number of samples to generate
    :param base: the base of the log space
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, num, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, num, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10]

eye
==============================

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, device=0,dtype=None)

    Create a size x size QTensor with ones on the diagonal and zeros
    elsewhere.

    :param size: size of the (square) QTensor to create
    :param offset: Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        size = 3
        t = tensor.eye(size)
        print(t)

        # [
        # [1, 0, 0],
        # [0, 1, 0],
        # [0, 0, 1]
        # ]

diag
==============================

.. py:function:: pyvqnet.tensor.diag(t, k: int = 0)

    Select diagonal elements or construct a diagonal QTensor.

    If input is 2-D QTensor,returns a new tensor which is the same as this one, except that
    elements other than those in the selected diagonal are set to zero.

    If v is a 1-D QTensor, return a 2-D QTensor with v on the k-th diagonal.

    :param t: input QTensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = QTensor(a)
        for k in range(-3, 4):
            u = tensor.diag(t,k=k)
            print(u)

        # [
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [12, 0, 0, 0]
        # ]

        # [
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [8, 0, 0, 0],
        # [0, 13, 0, 0]
        # ]

        # [
        # [0, 0, 0, 0],
        # [4, 0, 0, 0],
        # [0, 9, 0, 0],
        # [0, 0, 14, 0]
        # ]

        # [
        # [0, 0, 0, 0],
        # [0, 5, 0, 0],
        # [0, 0, 10, 0],
        # [0, 0, 0, 15]
        # ]

        # [
        # [0, 1, 0, 0],
        # [0, 0, 6, 0],
        # [0, 0, 0, 11],
        # [0, 0, 0, 0]
        # ]

        # [
        # [0, 0, 2, 0],
        # [0, 0, 0, 7],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0]
        # ]

        # [
        # [0, 0, 0, 3],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0]
        # ]

randu
==============================

.. py:function:: pyvqnet.tensor.randu(shape,min=0.0,max=1.0, device: int = 0, dtype=None, requires_grad=False)

    Create a QTensor with uniformly distributed random values.

    :param shape: shape of the QTensor to create
    :param min: minimum value of uniform distribution,default: 0.
    :param max: maximum value of uniform distribution,default: 1.
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: output QTensor


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

.. py:function:: pyvqnet.tensor.randn(shape, mean=0.0,std=1.0, device: int = 0, dtype=None, requires_grad=False)

    Create a QTensor with normally distributed random values.

    :param shape: shape of the QTensor to create
    :param mean: mean value of normally distribution,default: 0.
    :param std: standard variance value of normally distribution,default: 1.
    :param device: device to use,default = 0 ,use cpu device
    :param dtype: The data type of the parameter, defaults None, use the default data type: kfloat32, which represents a 32-bit floating point number.
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: output QTensor

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


multinomial
==============================

.. py:function:: pyvqnet.tensor.multinomial(t, num_samples)

    Returns a Tensor where each row contains num_samples indexed samples.
    From the multinomial probability distribution located in the corresponding row of the tensor input.

    :param t: Input probability distribution。
    :param num_samples: numbers of sample。

    :return:
        output sample index

    Examples::

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

    :param t: input a QTensor
    :param diagonal: The Offset default =0. Main diagonal is 0, positive is offset up,and negative is offset down

    :return: output a QTensor

    Examples::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
        u = tensor.triu(a, 1)
        print(u)
        # [
        # [[0, 2, 3, 4, 5],
        #  [0, 0, 8, 9, 10],
        #  [0, 0, 0, 14, 15],
        #  [0, 0, 0, 0, 20],
        #  [0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0]],
        # [[0, 32, 33, 34, 35],
        #  [0, 0, 38, 39, 40],
        #  [0, 0, 0, 44, 45],
        #  [0, 0, 0, 0, 50],
        #  [0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0]]
        # ]

tril
==============================

.. py:function:: pyvqnet.tensor.tril(t, diagonal=0)

    Returns the lower triangular matrix of input t, with the rest set to 0.

    :param t: input a QTensor
    :param diagonal: The Offset default =0. Main diagonal is 0, positive is offset up,and negative is offset down

    :return: output a QTensor

    Examples::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([12, 5])
        u = tensor.tril(a, 1)
        print(u)
        # [
        # [1, 2, 0, 0, 0],
        #  [6, 7, 8, 0, 0],
        #  [11, 12, 13, 14, 0],
        #  [16, 17, 18, 19, 20],
        #  [21, 22, 23, 24, 25],
        #  [26, 27, 28, 29, 30],
        #  [31, 32, 33, 34, 35],
        #  [36, 37, 38, 39, 40],
        #  [41, 42, 43, 44, 45],
        #  [46, 47, 48, 49, 50],
        #  [51, 52, 53, 54, 55],
        #  [56, 57, 58, 59, 60]
        # ]


Math Functions
*****************************************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    Return a new QTensor with the floor of the elements of input, the largest integer less than or equal to each element.

    :param t: input Qtensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    Return a new QTensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.

    :param t: input Qtensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2, -1, -1, -1, -1, -0, -0, -0, 0, 1, 1, 1, 1, 2, 2, 2]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    Round QTensor values to the nearest integer.

    :param t: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2, -2, -1, -1, -0, -0, 0, 1, 1, 2]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    Sort QTensor along the axis

    :param t: input QTensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        AA = tensor.sort(A,1,False)
        print(AA)

        # [
        # [0, 1, 2, 4, 6, 7, 8, 8],
        # [2, 5, 5, 8, 9, 9, 9, 9],
        # [1, 2, 5, 5, 5, 6, 7, 7]
        # ]

argsort
==============================

.. py:function:: pyvqnet.tensor.argsort(t, axis: int, descending=False, stable=True)

    Return an array of indices of the same shape as input that index data along the given axis in sorted order.

    :param t: input QTensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        bb = tensor.argsort(A,1,False)
        print(bb)

        # [
        # [4, 0, 1, 7, 5, 3, 2, 6], 
        #  [3, 0, 7, 6, 2, 1, 4, 5],
        #  [4, 7, 5, 0, 2, 1, 3, 6]
        # ]

topK
==============================

.. py:function:: pyvqnet.tensor.topK(t, k, axis=-1, if_descent=True)

    Returns the k largest elements of the input tensor along the given axis.

    If if_descent is False，then return k smallest elements.

    :param t: input a QTensor
    :param k: numbers of largest elements or smallest elements
    :param axis: sort axis,default = -1，the last axis
    :param if_descent: sort order,defaults to True

    :return: A new QTensor

    Examples::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x.reshape_([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.topK(x, 3, 1)
        print(y)
        # [
        # [[[24, 15]],
        # [[15, 13]],
        # [[11, 8]]],
        # [[[24, 13]],
        # [[15, 11]],
        # [[7, 8]]]
        # ]

argtopK
==============================

.. py:function:: pyvqnet.tensor.argtopK(t, k, axis=-1, if_descent=True)

    Return the index of the k largest elements along the given axis of the input tensor.

    If if_descent is False，then return the index of k smallest elements.

    :param t: input a QTensor
    :param k: numbers of largest elements or smallest elements
    :param axis: sort axis,default = -1，the last axis
    :param if_descent: sort order,defaults to True

    :return: A new QTensor

    Examples::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x.reshape_([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.argtopK(x, 3, 1)
        print(y)
        # [
        # [[[0, 4]],
        # [[1, 0]],
        # [[3, 2]]],
        # [[[0, 0]],
        # [[1, 4]],
        # [[3, 2]]]
        # ]



add
==============================

.. py:function:: pyvqnet.tensor.add(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise adds two QTensors, equivalent to t1 + t2.

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.add(t1, t2)
        print(x)

        # [5, 7, 9]

sub
==============================

.. py:function:: pyvqnet.tensor.sub(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise subtracts two QTensors,  equivalent to t1 - t2.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.sub(t1, t2)
        print(x)

        # [-3, -3, -3]

mul
==============================

.. py:function:: pyvqnet.tensor.mul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise multiplies two QTensors, equivalent to t1 * t2.

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.mul(t1, t2)
        print(x)

        # [4, 10, 18]

divide
==============================

.. py:function:: pyvqnet.tensor.divide(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise divides two QTensors, equivalent to t1 / t2.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor


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

.. py:function:: pyvqnet.tensor.sums(t: pyvqnet.tensor.QTensor, axis: Optional[int] = None, keepdims=False)

    Sums all the elements in QTensor along given axis.if axis = None, sums all the elements in QTensor. 

    :param t: input QTensor
    :param axis:  axis used to sums, defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return:  output QTensor


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21]



cumsum
==============================

.. py:function:: pyvqnet.tensor.cumsum(t, axis=-1)

    Return the cumulative sum of input elements in the dimension axis.

    :param t:  the input QTensor
    :param axis:  Calculation of the axis,defaults to -1,use the last axis

    :return:  output QTensor.

    Example::

       from pyvqnet.tensor import tensor, QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.cumsum(t,-1)
        print(x)
        # [
        # [1, 3, 6],
        # [4, 9, 15]
        # ]


mean
==============================

.. py:function:: pyvqnet.tensor.mean(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Obtain the mean values in the QTensor along the axis.

    :param t:  the input QTensor.
    :param axis: the dimension to reduce.
    :param keepdims:  whether the output QTensor has dim retained or not, defaults to False.
    :return: returns the mean value of the input QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.mean(t, axis=1)
        print(x)

        # [2, 5]

median
==============================

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Obtain the median value in the QTensor.

    :param t: the input QTensor
    :param axis:  An axis for averaging,defaults to None
    :param keepdims:  whether the output QTensor has dim retained or not, defaults to False

    :return: Return the median of the values in input or QTensor.

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

        # [-0.3982000, 0.2270000, 0.2488000, 0.4742000]

std
==============================

.. py:function:: pyvqnet.tensor.std(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

    Obtain the standard variance value in the QTensor.


    :param t:  the input QTensor
    :param axis:  the axis used to calculate the standard deviation,defaults to None
    :param keepdims:  whether the output QTensor has dim retained or not, defaults to False
    :param unbiased:  whether to use Bessel’s correction,default true
    :return: Return the standard variance of the values in input or QTensor

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

.. py:function:: pyvqnet.tensor.var(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

    Obtain the variance in the QTensor.


    :param t:  the input QTensor.
    :param axis:  The axis used to calculate the variance,defaults to None
    :param keepdims:  whether the output QTensor has dim retained or not, defaults to False.
    :param unbiased:  whether to use Bessel’s correction,default true.


    :return: Obtain the variance in the QTensor.

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

    Matrix multiplications of two 2d , 3d , 4d matrix.

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = tensor.ones([2,3])
        t1.requires_grad = True
        t2 = tensor.ones([3,4])
        t2.requires_grad = True
        t3  = tensor.matmul(t1,t2)
        t3.backward(tensor.ones_like(t3))
        print(t1.grad)

        # [
        # [4, 4, 4],
        #  [4, 4, 4]
        # ]

        print(t2.grad)

        # [
        # [2, 2, 2, 2],
        #  [2, 2, 2, 2],
        #  [2, 2, 2, 2]
        # ]

kron
=============================

.. py:function:: pyvqnet.tensor.kron(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Computes the Kronecker product of ``t1`` and ``t2``, expressed in :math:`\otimes` . If ``t1`` is a :math:`(a_0 \times a_1 \times \dots \times a_n)` tensor and ``t2`` is a :math:`(b_0 \times b_1 \times \dots \ times b_n)` tensor, the result will be :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` tensor with the following entries:
    
    .. math::
          (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
              \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

    where :math:`k_t = i_t * b_t + j_t` is :math:`0 \leq t \leq n`.
    If one tensor has fewer dimensions than the other, it will be unpacked until it has the same dimensionality.

    :param t1: The first QTensor.
    :param t2: The second QTensor.
    
    :return: Output QTensor .

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


reciprocal
==============================

.. py:function:: pyvqnet.tensor.reciprocal(t)

    Compute the element-wise reciprocal of the QTensor.

    :param t: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1, 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
==============================

.. py:function:: pyvqnet.tensor.sign(t)

    Return a new QTensor with the signs of the elements of input.The sign function returns -1 if t < 0, 0 if t==0, 1 if t > 0.

    :param t: input QTensor
    :return: output QTensor


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1]


neg
==============================

.. py:function:: pyvqnet.tensor.neg(t: pyvqnet.tensor.QTensor)

    Unary negation of QTensor elements.

    :param t: input QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1, -2, -3]

trace
==============================

.. py:function:: pyvqnet.tensor.trace(t, k: int = 0)

    Return the sum of the elements of the diagonal of the input 2-D matrix.

    :param t: input 2-D QTensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: the sum of the elements of the diagonal of the input 2-D matrix

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

    Applies exponential function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Compute the element-wise inverse cosine of the QTensor.

    :param t: input QTensor
    :return: output QTensor

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

    Compute the element-wise inverse sine of the QTensor.

    :param t: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0, 0.5235988]

atan
==============================

.. py:function:: pyvqnet.tensor.atan(t: pyvqnet.tensor.QTensor)

    Compute the element-wise inverse tangent of the QTensor.

    :param t: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = Tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0.0000, 0.4636476]

sin
==============================

.. py:function:: pyvqnet.tensor.sin(t: pyvqnet.tensor.QTensor)

    Applies sine function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

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

    Applies cosine function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

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

    Applies tangent function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

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

    Applies hyperbolic tangent function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Applies hyperbolic sine function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

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

    Applies hyperbolic cosine function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

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

    Raises first QTensor to the power of second QTensor.

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 4, 3])
        t2 = QTensor([2, 5, 6])
        x = tensor.power(t1, t2)
        print(x)

        # [1, 1024, 729]

abs
==============================

.. py:function:: pyvqnet.tensor.abs(t: pyvqnet.tensor.QTensor)

    Applies abs function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1, 2, 3]

log
==============================

.. py:function:: pyvqnet.tensor.log(t: pyvqnet.tensor.QTensor)

    Applies log (ln) function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0, 0.6931471, 1.0986123]

log_softmax
==============================

.. py:function:: pyvqnet.tensor.log_softmax(t, axis=-1)
    
    Sequentially calculate the results of the softmax function and the log function on the axis axis.

    :param t: input QTensor .
    :param axis: The axis used to calculate softmax, the default is -1.

    :return: Output QTensor。

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

    Applies sqrt function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1, 1.4142135, 1.7320507]

square
==============================

.. py:function:: pyvqnet.tensor.square(t: pyvqnet.tensor.QTensor)

    Applies square function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)

        # [1, 4, 9]

frobenius_norm
==============================

.. py:function:: pyvqnet.tensor.frobenius_norm(t: QTensor, axis: int = None, keepdims=False):

    Computes the F-norm of the tensor on the input QTensor along the axis set by axis ,
    if axis is None, returns the F-norm of all elements.

    :param t: Inpout QTensor .
    :param axis: The axis used to find the F norm, the default is None.
    :param keepdims: Whether the output tensor preserves the reduced dimensionality. The default is False.
    :return: Output a QTensor or F-norm value.


    Example::

        from pyvqnet import tensor,QTensor
        t = QTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                    [[13., 14., 15.], [16., 17., 18.]]])
        t.requires_grad = True
        result = tensor.frobenius_norm(t, -2, False)
        print(result)
        # [
        # [4.1231055, 5.3851647, 6.7082038],
        #  [12.2065554, 13.6014709, 15],
        #  [20.6155281, 22.0227146, 23.4307499]
        # ]



Logic Functions
**************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise maximum of two tensor.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)
        print(x)

        # [6, 5, 7]

minimum
==============================

.. py:function:: pyvqnet.tensor.minimum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise minimum of two tensor.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)
        print(x)

        # [2, 4, 3]

min
==============================

.. py:function:: pyvqnet.tensor.min(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Return min elements of the input QTensor alongside given axis.
    if axis == None, return the min value of all elements in tensor.

    :param t: input QTensor
    :param axis: axis used for min, defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.min(t, axis=1, keepdims=True)
        print(x)

        # [
        # [1],
        #  [4]
        # ]

max
==============================

.. py:function:: pyvqnet.tensor.max(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Return max elements of the input QTensor alongside given axis.
    if axis == None, return the max value of all elements in tensor.

    :param t: input QTensor
    :param axis: axis used for max, defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.max(t, axis=1, keepdims=True)
        print(x)

        # [[3],
        # [6]]

clip
==============================

.. py:function:: pyvqnet.tensor.clip(t: pyvqnet.tensor.QTensor, min_val, max_val)

    Clips input QTensor to minimum and maximum value.

    :param t: input QTensor
    :param min_val:  minimum value
    :param max_val:  maximum value
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3, 4, 6]

where
==============================

.. py:function:: pyvqnet.tensor.where(condition: pyvqnet.tensor.QTensor, t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Return elements chosen from x or y depending on condition.

    :param condition: condition tensor,need to have data type of kbool.
    :param t1: QTensor from which to take elements if condition is met, defaults to None
    :param t2: QTensor from which to take elements if condition is not met, defaults to None
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)
        print(x)

        # [1, 5, 6]

nonzero
==============================

.. py:function:: pyvqnet.tensor.nonzero(t)

    Return a QTensor containing the indices of nonzero elements.

    :param t: input QTensor
    :return: output QTensor contains indices of nonzero elements.

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
        # [0, 0],
        # [1, 1],
        # [2, 2],
        # [3, 3]
        # ]

isfinite
==============================

.. py:function:: pyvqnet.tensor.isfinite(t)

    Test element-wise for finiteness (not infinity or not Not a Number).

    :param t: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Test element-wise for positive or negative infinity.

    :param t: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
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

    Test element-wise for Nan.

    :param t: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
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

    Test element-wise for negative infinity.

    :param t: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
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

    Test element-wise for positive infinity.

    :param t: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
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

    Compute the truth value of ``t1`` and ``t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
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

    Compute the truth value of ``t1 or t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Compute the truth value of ``not t`` element-wise.

    :param t: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Compute the truth value of ``t1 xor t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor

    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Return the truth value of ``t1 > t2`` element-wise.


    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Return the truth value of ``t1 >= t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Return the truth value of ``t1 < t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Return the truth value of ``t1 <= t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.

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

    Return the truth value of ``t1 == t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
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

    Return the truth value of ``t1 != t2`` element-wise.

    :param t1: input QTensor
    :param t2: input QTensor
    :return: Output QTensor, which returns True when the corresponding position element meets the condition, otherwise returns False.
    
    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)


        #[[False  True]
        # [ True False]]


Matrix Operations
**********************

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Return QTensor in the QTensor at the given axis. following operation get same result's value.

    :param t: input QTensor
    :param index: a string contains output dim
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        t = QTensor(np.arange(1,25).reshape(2,3,4))
              
        indx = [":", "0", ":"]        
        t.requires_grad = True
        t.zero_grad()
        ts = tensor.select(t,indx)
        ts.backward(tensor.ones(ts.shape))
        print(ts)  
        # [
        # [[1, 2, 3, 4]],
        # [[13, 14, 15, 16]]
        # ]


broadcast
==============================

.. py:function:: pyvqnet.tensor.broadcast(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Subject to certain restrictions, smaller arrays are placed throughout larger arrays so that they have compatible shapes. This interface can perform automatic differentiation on input parameter tensors.

    Reference https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: input QTensor 1
    :param t2: input QTensor 2

    :return t11: with new broadcast shape t1.
    :return t22: t2 with new broadcast shape.

    Example::

        from pyvqnet.tensor import tensor
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
==============================

.. py:function:: pyvqnet.tensor.concatenate(args: list, axis=1)

    Concatenate the input QTensor along the axis and return a new QTensor.

    :param args: list consist of input QTensors
    :param axis: dimension to concatenate. Has to be between 0 and the number of dimensions of concatenate tensors.
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = QTensor([[1, 2, 3],[4,5,6]], requires_grad=True) 
        y = 1-x  
        x = tensor.concatenate([x,y],1)
        print(x)

        # [
        # [1, 2, 3, 0, -1, -2],
        # [4, 5, 6, -3, -4, -5]
        # ]

stack
==============================

.. py:function:: pyvqnet.tensor.stack(QTensors: list, axis) 

    Join a sequence of arrays along a new axis,return a new QTensor.

    :param QTensors: list contains QTensors
    :param axis: dimension to insert. Has to be between 0 and the number of dimensions of stacked tensors. 
    :return: output QTensor

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
        # [[0, 0, 0],
        #  [1, 1, 1],
        #  [2, 2, 2],
        #  [3, 3, 3]],
        # [[4, 4, 4],
        #  [5, 5, 5],
        #  [6, 6, 6],
        #  [7, 7, 7]],
        # [[8, 8, 8],
        #  [9, 9, 9],
        #  [10, 10, 10],
        #  [11, 11, 11]]
        # ]

permute
==============================

.. py:function:: pyvqnet.tensor.permute(t: pyvqnet.tensor.QTensor, dim: list)

    Reverse or permute the axes of an array.if dims = None, revsers the dim.

    :param t: input QTensor
    :param dim: the new order of the dimensions (list of integers)
    :return: output QTensor

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
        # [[0, 3],
        #  [6, 9]],
        # [[1, 4],
        #  [7, 10]],
        # [[2, 5],
        #  [8, 11]]
        # ]

transpose
==============================

.. py:function:: pyvqnet.tensor.transpose(t: pyvqnet.tensor.QTensor, dim: list)

    Transpose the axes of an array.if dim = None, reverse the dim. This function is same as permute.

    :param t: input QTensor
    :param dim: the new order of the dimensions (list of integers)
    :return: output QTensor

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
        # [[0, 3],
        #  [6, 9]],
        # [[1, 4],
        #  [7, 10]],
        # [[2, 5],
        #  [8, 11]]
        # ]

tile
==============================

.. py:function:: pyvqnet.tensor.tile(t: pyvqnet.tensor.QTensor, reps: list)

    Construct a QTensor by repeating QTensor the number of times given by reps.

    If reps has length d, the result QTensor will have dimension of max(d, t.ndim).

    If t.ndim < d, t is expanded to be d-dimensional by inserting new axes from start dimension.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication.

    If t.ndim > d, reps is expanded to t.ndim by inserting 1’s to it.

    Thus for an t of shape (2, 3, 4, 5), a reps of (4, 3) is treated as (1, 1, 4, 3).

    :param t: input QTensor
    :param reps: the number of repetitions per dimension.
    :return: a new QTensor

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
        # [0, 1, 2, 0, 1, 2],
        # [3, 4, 5, 3, 4, 5],
        # [0, 1, 2, 0, 1, 2],
        # [3, 4, 5, 3, 4, 5]
        # ]

squeeze
==============================

.. py:function:: pyvqnet.tensor.squeeze(t: pyvqnet.tensor.QTensor, axis: int = - 1)

    Remove axes of length one .

    :param t: input QTensor
    :param axis: squeeze axis,if axis = -1 ,squeeze all the dimensions that have size of 1.
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = QTensor(a)
        AA = tensor.squeeze(A,0)
        print(AA)

        # [
        # [0],
        # [1],
        # [2],
        # [3],
        # [4],
        # [5]
        # ]

unsqueeze
==============================

.. py:function:: pyvqnet.tensor.unsqueeze(t: pyvqnet.tensor.QTensor, axis: int = 0)

    Return a new QTensor with a dimension of size one inserted at the specified position.

    :param t: input QTensor
    :param axis: unsqueeze axis,which will insert dimension.
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = QTensor(a)
        AA = tensor.unsqueeze(A,1)
        print(AA)

        # [
        # [[[[[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8],
        #  [9, 10, 11]]]]],
        # [[[[[12, 13, 14],
        #  [15, 16, 17],
        #  [18, 19, 20],
        #  [21, 22, 23]]]]]
        # ]

swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    Interchange two axes of an array.The given dimensions axis1 and axis2 are swapped.

    :param t: input QTensor
    :param axis1: First axis.
    :param axis2:  Destination position for the original axis. These must also be unique
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = QTensor(a)
        AA = tensor.swapaxis(A,2,1)
        print(AA)

        # [
        # [[0, 4, 8],
        #  [1, 5, 9],
        #  [2, 6, 10],
        #  [3, 7, 11]],
        # [[12, 16, 20],
        #  [13, 17, 21],
        #  [14, 18, 22],
        #  [15, 19, 23]]
        # ]

masked_fill
==============================

.. py:function:: pyvqnet.tensor.masked_fill(t, mask, value)

    If mask == 1, fill with the specified value. The shape of the mask must be broadcastable from the shape of the input QTensor.

    :param t: input QTensor
    :param mask: A QTensor
    :param value: specified value
    :return:  A QTensor

    Examples::

        from pyvqnet.tensor import tensor
        import numpy as np
        a = tensor.ones([2, 2, 2, 2])
        mask = np.random.randint(0, 2, size=4).reshape([2, 2])
        b = tensor.QTensor(mask==1)
        c = tensor.masked_fill(a, b, 13)
        print(c)
        # [
        # [[[1, 1],
        #  [13, 13]],
        # [[1, 1],
        #  [13, 13]]],
        # [[[1, 1],
        #  [13, 13]],
        # [[1, 1],
        #  [13, 13]]]
        # ]


flatten
==============================

.. py:function:: pyvqnet.tensor.flatten(t: pyvqnet.tensor.QTensor, start: int = 0, end: int = - 1)

    Flatten QTensor from dim start to dim end.

    :param t: input QTensor
    :param start: dim start,default = 0,start from first dim.
    :param end: dim end,default = -1,end with last dim.
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1, 2, 3]


reshape
==============================

.. py:function:: pyvqnet.tensor.reshape(t: pyvqnet.tensor.QTensor,new_shape)

    Change QTensor's shape, return a new shape QTensor

    :param t: input QTensor.
    :param new_shape: new shape

    :return: a new shape QTensor.

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
        # [0, 1, 2],
        # [3, 4, 5],
        # [6, 7, 8],
        # [9, 10, 11]
        # ]

flip
==============================

.. py:function:: pyvqnet.tensor.flip(t, flip_dims)
    
    Reverses the QTensor along the specified axis, returning a new tensor.

    :param t: Input QTensor.
    :param flip_dims: The axis or list of axes to flip.

    :return: Output QTensor.

    Example::

        from pyvqnet import tensor
        t = tensor.arange(1, 3 * 2 *2 * 2 + 1).reshape([3, 2, 2, 2])
        t.requires_grad = True
        y = tensor.flip(t, [0, -1])
        print(y)
        # [
        # [[[18, 17], 
        #  [20, 19]], 
        # [[22, 21],  
        #  [24, 23]]],
        # [[[10, 9],  
        #  [12, 11]], 
        # [[14, 13],  
        #  [16, 15]]],
        # [[[2, 1],   
        #  [4, 3]],   
        # [[6, 5],    
        #  [8, 7]]]   
        # ]


gather
=============================

.. py:function:: pyvqnet.tensor.gather(t, dim, index)

    Collect values along the axis specified by 'dim'.

    For 3-D tensors, the output is specified by:

    .. math::

        out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \\

        out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \\

        out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \\

    :param t: Input QTensor.
    :param dim: The aggregation axis.
    :param index: Index QTensor, should have the same dimension size as input.

    :return: the aggregated result

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
=============================

.. py:function:: pyvqnet.tensor.scatter(input, dim, index, src)

    Writes all values in the tensor src to input at the indices specified in the indices tensor.

    For 3-D tensors, the output is specified by:

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \\

    :param input: Input QTensor.
    :param dim: Scatter axis.
    :param indices: Index QTensor, should have the same dimension size as the input.
    :param src: The source tensor to scatter.

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
=============================

.. py:function:: pyvqnet.tensor.broadcast_to(t, ref)

    Subject to certain constraints, the array t is "broadcast" to the reference shape so that they have compatible shapes.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: input QTensor
    :param ref: Reference shape.
    
    :return: The QTensor of the newly broadcasted t.

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]


dense_to_csr
==============================

.. py:function:: pyvqnet.tensor.dense_to_csr(t)
    
    Convert dense matrix to CSR format sparse matrix, only supports 2 dimensions.

    :param t: input dense QTensor
    :return: CSR sparse matrix

    Example::

        from pyvqnet.tensor import QTensor,dense_to_csr

        a = QTensor([[2, 3, 4, 5]])
        b = dense_to_csr(a)
        print(b.csr_members())
        #([0,4], [0,1,2,3], [2,3,4,5])


csr_to_dense
==============================

.. py:function:: pyvqnet.tensor.csr_to_dense(t)
    
    Convert CSR format sparse matrix to dense matrix, only supports 2 dimensions.

    :param t: input CSR sparse matrix
    :return: Dense QTensor

    Example::

        from pyvqnet.tensor import QTensor,dense_to_csr,csr_to_dense

        a = QTensor([[2, 3, 4, 5]])
        b = dense_to_csr(a)
        c = csr_to_dense(b)
        print(c)
        #[[2,3,4,5]]


Utility Functions
*****************************************************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Convert input array to Qtensor if it isn't already.

    :param x: integer,float or numpy.array
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
        print(t)
        # [10]


pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    Pad a list of variable-length tensors with ``padding_value``. ``pad_sequence`` stacks lists of tensors along new dimensions and pads them to equal length.
    The input is a sequence of lists of size ``L x *``. L is variable length.

    :param qtensor_list: `list[QTensor]` - list of variable length sequences.
    :param batch_first: 'bool' - If true, the output will be ``batch size x longest sequence length x *``, otherwise ``longest sequence length x batch size x *``. Default: False.
    :param padding_value: 'float' - padding value. Default value: 0.

    :return:
         If batch_first is ``False``, the tensor size is ``batch size x longest sequence length x *``.
         Otherwise the size of the tensor is ``longest sequence length x batch size x *``.

    Examples::

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
        # [[[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]]],
        # [[[1, 1, 1],
        #  [1, 1, 1]],
        # [[0, 0, 0],
        #  [0, 0, 0]],
        # [[0, 0, 0],
        #  [0, 0, 0]],
        # [[0, 0, 0],
        #  [0, 0, 0]]],
        # [[[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]],
        # [[0, 0, 0],
        #  [0, 0, 0]],
        # [[0, 0, 0],
        #  [0, 0, 0]]]
        # ]


pad_packed_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_packed_sequence(sequence, batch_first=False, padding_value=0, total_length=None)
    
    Pad a batch of packed variable-length sequences. It is the inverse of `pack_pad_sequence`.
    When ``batch_first`` is True, it returns a tensor of shape ``B x T x *``, otherwise it returns ``T x B x *``.
    Where `T` is the longest sequence length and `B` is the batch size.

    :param sequence: 'QTensor' - the data to be processed.
    :param batch_first: 'bool' - If ``True``, batch will be the first dimension of the input. Default value: False.
    :param padding_value: 'bool' - padding value. Default: 0.
    :param total_length: 'bool' - If not ``None``, the output will be padded to length :attr:`total_length`. Default: None.
    :return:
        A tuple of tensors containing the padded sequences, and a list of lengths for each sequence in the batch. Batch elements will be reordered in their original order.
    
    Examples::

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
    
    Pack a Tensor containing variable-length padded sequences. If batch_first is True, `input` should have shape [batch size, length,*], otherwise shape [length, batch size,*].

    For unsorted sequences, use ``enforce_sorted`` is False. If :attr:`enforce_sorted` is ``True``, sequences should be sorted in descending order by length.
    
    :param input: 'QTensor' - variable-length sequence batches for padding.
    :parma lengths: 'list' - list of sequence lengths for each batch
         element.
    :param batch_first: 'bool' - if ``True``, the input is expected to be ``B x T x *``
         format, default: False.
    :param enforce_sorted: 'bool' - if ``True``, the input should be
         Contains sequences in descending order of length. If ``False``, the input will be sorted unconditionally. Default: True.

    :return: A :class:`PackedSequence` object.

    Examples::

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