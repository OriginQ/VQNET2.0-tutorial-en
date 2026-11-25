.. _vqnet_dist:

VQNet Distributed Computing Module
*********************************************************

Distributed computing means coordinating the training or inference of neural networks across multiple devices (such as GPU/CPU nodes), using parallel processing to accelerate computation and scale models.
At its core, it coordinates communication and synchronization between devices through distributed interfaces (such as GLOO and NCCL).

VQNet's distributed computing module starts multi-process parallel computation with spawn subprocesses, uses gloo for collective communication between CPUs and nccl for collective communication between GPUs. This feature is available only on Linux.

.. important::

    VQNet's distributed computing module supports Linux only.




NCCL Installation
^^^^^^^^^^^^^^^^^^^^^^

NCCL is a common library for inter-GPU communication. **VQNet's GPU distributed computing is implemented based on NCCL.** The software installs the NCCL dynamic library by default during installation, so NCCL usually does not need to be installed separately.


vqnetrun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the distributed computing interface by starting it with the ``vqnetrun`` command. The following describes each parameter of ``vqnetrun``.

nproc_per_node
========================================

The ``vqnetrun`` interface must specify the number of worker processes started on the current node through the ``--nproc_per_node`` parameter. This parameter applies to both gloo CPU workers and NCCL GPU workers.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("gloo") # init gloo controller

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --nproc_per_node 2 python test.py

backend
========================================

The distributed backend is selected in code and determined by the constructor argument of ``CommController``: ``CommController("gloo")`` for CPU, ``CommController("nccl")`` for GPU, and the default ``CommController()`` automatically determines the communication method based on the device where the data to be communicated resides.
The ``vqnetrun`` command does not provide a ``--backend`` parameter. When multiple processes are started, each worker process determines the backend itself: if a CUDA context exists, it directly creates an NCCL communication domain; in pure CPU runs it uses gloo.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # Single machine, multiple GPUs:
        # vqnetrun --nproc_per_node 2 python test.py

        # With multiple nodes, run independently on each node:
        # Node 0: vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1 python test.py
        # Node 1: vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 1 --master_addr 10.0.0.1 python test.py

nnodes
========================================

The ``vqnetrun`` interface can control the total number of nodes through the ``--nnodes`` parameter, with a default value of 1. The launcher computes ``WORLD_SIZE`` as ``nnodes * nproc_per_node``.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1 python test.py

node_rank
========================================

The ``vqnetrun`` interface can control the rank of the current node through the ``--node_rank`` parameter, with a default value of 0. The launcher uses it together with ``nproc_per_node`` to compute the global ``RANK`` of each worker.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # Node 0: vqnetrun --nproc_per_node 2 --nnodes 3 --node_rank 0 --master_addr 10.0.0.1 python test.py
        # Node 1: vqnetrun --nproc_per_node 2 --nnodes 3 --node_rank 1 --master_addr 10.0.0.1 python test.py
        # Node 2: vqnetrun --nproc_per_node 2 --nnodes 3 --node_rank 2 --master_addr 10.0.0.1 python test.py

master_addr
========================================

The ``vqnetrun`` interface can specify the master node address through the ``--master_addr`` parameter, with a default value of ``127.0.0.1``. This value is passed to worker processes as the ``MASTER_ADDR`` environment variable.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.0.0.1 python test.py

master_port
========================================

The ``vqnetrun`` interface can specify the master node port through the ``--master_port`` parameter, with a default value of ``29500``. This value is passed to worker processes as the ``MASTER_PORT`` environment variable.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --nproc_per_node 2 --master_port 29600 python test.py

nccl_socket_ifname
========================================

The ``vqnetrun`` interface can specify the network interface name used by NCCL through the ``--nccl_socket_ifname`` parameter. This parameter affects only NCCL communication.
It corresponds to the environment variable ``NCCL_SOCKET_IFNAME`` and is used to specify the network card for NCCL communication in multi-NIC environments.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --nproc_per_node 2 --nccl_socket_ifname eth0 python test.py

verbose
========================================

The ``vqnetrun`` interface can output additional startup diagnostic information through the ``--verbose`` flag, such as node/process rank settings, the master rendezvous address and port, and local worker startup information. This flag does not perform node connectivity checks.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("gloo")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --nproc_per_node 4 --verbose python test.py

cb, check-build
========================================

The ``vqnetrun`` interface can view the distributed controllers and communication libraries supported by the current build through the ``-cb`` or ``--check-build`` flag.
Because the launcher's argument parsing requires ``--nproc_per_node``, this parameter must also be provided when viewing build information.

    Example::

        # vqnetrun --nproc_per_node 1 -cb
        # vqnetrun --nproc_per_node 1 --check-build

CommController
^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend=None, rank=None, world_size=None)

    CommController is a controller for controlling data communication on CPU and GPU. Set the `backend` parameter to create a CPU (gloo) or GPU (nccl) controller. (Currently, distributed computing is supported only on Linux.)

    :param backend: Controller used for CPU or GPU data communication. If None, both gloo and nccl communication are supported. Defaults to None.
    :param rank: This parameter is useful only with non-pyvqnet backends. Default: None.
    :param world_size: This parameter is useful only with non-pyvqnet backends. Default: None.

    :return:
        A CommController instance.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # init nccl controller

    .. py:method:: getRank()

        Get the process number of the current process.


        :return: Returns the process number of the current process.

        Example::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # init nccl controller

            Comm_OP.getRank()


    .. py:method:: getSize()

        Get the total number of processes started.


        :return: Returns the total number of processes.

        Example::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # init nccl controller

            Comm_OP.getSize()
            # vqnetrun --nproc_per_node 2 python test.py


    .. py:method:: getLocalRank()

        Get the current process number on the current machine.


        :return: The current process number on the current machine.

        Example::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # init nccl controller

            Comm_OP.getLocalRank()
            # vqnetrun --nproc_per_node 2 python test.py


    .. py:method:: barrier()

        Synchronize.

        :return: Synchronization operation.

        Example::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")

            Comm_OP.barrier()


    .. py:method:: get_device_num()

        Get the number of GPUs on the current node (supported only for GPU).

        :return: Returns the number of GPUs on the current node.

        Example::


            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")

            Comm_OP.get_device_num()
            # vqnetrun --master_port 12345 --nproc_per_node 2 python test.py


    .. py:method:: allreduce(tensor, c_op = "avg")

        Supports allreduce communication on data.

        :param tensor: Input data.
        :param c_op: Computation method.

        Example::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("gloo")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            Comm_OP.allreduce(num, "sum")
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun --nproc_per_node 2 python test.py


    .. py:method:: reduce(tensor, root = 0, c_op = "avg")

        Supports reduce communication on data.

        :param tensor: Input data.
        :param root: Node to which the data is returned.
        :param c_op: Computation method.

        Example::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("gloo")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            Comm_OP.reduce(num, 1)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun --nproc_per_node 2 python test.py




    .. py:method:: broadcast(tensor, root = 0)

        Broadcast data on the specified process root to all processes.

        :param tensor: Input data.
        :param root: Specified node.

        Example::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("gloo")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            Comm_OP.broadcast(num, 1)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun --nproc_per_node 2 python test.py



    .. py:method:: allgather(tensor)

        Allgather data from all processes together.

        :param tensor: Input data.

        Example::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("gloo")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            num = Comm_OP.allgather(num)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun --nproc_per_node 2 python test.py


    .. py:method:: send(tensor, dest)

        P2P communication interface.

        :param tensor: Input data.
        :param dest: Destination process.

        Example::

            from pyvqnet.distributed import CommController,get_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("gloo")

            num = tensor.to_tensor(np.random.rand(1, 5))
            recv = tensor.zeros_like(num)

            if get_rank() == 0:
                Comm_OP.send(num, 1)
            elif get_rank() == 1:
                Comm_OP.recv(recv, 0)
            print(f"rank {Comm_OP.getRank()}  {num}")
            print(f"rank {Comm_OP.getRank()}  {recv}")

            # vqnetrun --nproc_per_node 2 python test.py



    .. py:method:: recv(tensor, source)

        P2P communication interface.

        :param tensor: Input data.
        :param source: Receiving process.

        Example::

            from pyvqnet.distributed import CommController,get_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("gloo")

            num = tensor.to_tensor(np.random.rand(1, 5))
            recv = tensor.zeros_like(num)

            if get_rank() == 0:
                Comm_OP.send(num, 1)
            elif get_rank() == 1:
                Comm_OP.recv(recv, 0)
            print(f"rank {Comm_OP.getRank()}  {num}")
            print(f"rank {Comm_OP.getRank()}  {recv}")

            # vqnetrun --nproc_per_node 2 python test.py

    .. py:method:: split_groups(rank_lists)

        Split processes into multiple communication groups according to the list of process numbers set by the input parameter.

        :param rank_lists: Each sublist contains the process numbers of one communication group, for example ``[[0, 1], [2, 3]]``.

        :return: The default (pyvqnet-ad) backend returns a list of tuples, where each element is a tuple of process numbers for one communication group and the length equals the number of groups; the torch backend returns a list of torch ProcessGroups. With the default backend, this interface only registers the group partition; the actual communication group is created when the group is first used for group communication.

        Example::

            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("gloo")

            groups = Comm_OP.split_groups([[0, 1],[2,3]])

            # vqnetrun --nproc_per_node 4 python test.py


    .. py:method:: allreduce_group(tensor, c_op = "avg", group = None)

        Allreduce communication interface within a group.

        :param tensor: Input data.
        :param c_op: Computation method.
        :param group: Process numbers of a communication group, taken from an element of the ``split_groups`` return value, for example ``(0, 1)``; an equivalent list ``[0, 1]`` is also accepted.


        Example::

            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            from pyvqnet import kcomplex64
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=kcomplex64).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"allreduce_group before rank {get_rank()}: {complex_data}")

            Comm_OP.allreduce_group(complex_data, c_op="sum",group = groups[0])
            print(f"allreduce_group after rank {get_rank()}: {complex_data}")
            # vqnetrun --nproc_per_node 2 python test.py

    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", group = None)

        Reduce communication interface within a group.

        :param tensor: Input data.
        :param root: Specified process number.
        :param c_op: Computation method.
        :param group: Process numbers of a communication group, taken from an element of the ``split_groups`` return value, for example ``(0, 1)``; an equivalent list ``[0, 1]`` is also accepted.


        Example::

            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            from pyvqnet import kcomplex64
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=kcomplex64).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"reduce_group before rank {get_rank()}: {complex_data}")

            Comm_OP.reduce_group(complex_data, c_op="sum",group = groups[0])
            print(f"reduce_group after rank {get_rank()}: {complex_data}")
            # vqnetrun --nproc_per_node 2 python test.py



    .. py:method:: broadcast_group(tensor, root = 0, group = None)

        Broadcast communication interface within a group.

        :param tensor: Input data.
        :param root: Process number from which to broadcast. Default: 0.
        :param group: Process numbers of a communication group, taken from an element of the ``split_groups`` return value, for example ``(0, 1)``; an equivalent list ``[0, 1]`` is also accepted.


        Example::

            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            from pyvqnet import kcomplex64
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=kcomplex64).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"broadcast_group before rank {get_rank()}: {complex_data}")

            Comm_OP.broadcast_group(complex_data,group = groups[0])
            Comm_OP.barrier()
            print(f"broadcast_group after rank {get_rank()}: {complex_data}")
            # vqnetrun --nproc_per_node 2 python test.py



    .. py:method:: allgather_group(tensor, group = None)

        Allgather communication interface within a group.

        :param tensor: Input data.
        :param group: Process numbers of a communication group, taken from an element of the ``split_groups`` return value, for example ``(0, 1)``; an equivalent list ``[0, 1]`` is also accepted.


        Example::

            # Option 1: gloo backend, CPU tensors
            from pyvqnet.distributed import CommController,get_rank
            from pyvqnet.tensor import tensor
            from pyvqnet import kcomplex64
            Comm_OP = CommController("gloo")
            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=kcomplex64).reshape((3,1))
            print(f" before rank {get_rank()}: {complex_data}")
            complex_data = Comm_OP.allgather_group(complex_data, group = groups[0])
            print(f"after rank {get_rank()}: {complex_data}")
            # vqnetrun --nproc_per_node 2 python test.py

            # Option 2: nccl backend, GPU tensors. Choose one of the two options; do not create two controllers in the same process.
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            from pyvqnet import kcomplex64
            Comm_OP = CommController("nccl")
            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=kcomplex64).reshape((3,1)).toGPU(1000+ get_local_rank())
            print(f" before rank {get_rank()}: {complex_data}")
            complex_data = Comm_OP.allgather_group(complex_data, group = groups[0])
            print(f"after rank {get_rank()}: {complex_data}")
            # vqnetrun --nproc_per_node 2 python test.py

    .. py:method:: nccl_async_all_gather( output, input, group=None, async_op=False):

        Perform asynchronous or synchronous all_gather on GPU data using NCCL.

        :param output: Destination data of the QTensor all_gather.
        :param input: QTensor data to gather.
        :param group: Sequence of process numbers for a communication group, with the same meaning as an element of the ``split_groups`` return value. Default None means no grouping.
        :param async_op: Whether this operation is asynchronous. Default: False.
        :return: Work, an asynchronous communication handle. Use wait() to wait for the operation to complete.

        Example::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet import kcomplex64
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=kcomplex64).reshape((3,1)).toGPU(1000+ get_local_rank())

            out_data = tensor.empty([2,3,1],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            work = Comm_OP.nccl_async_all_gather(out_data, complex_data, group = None,async_op=True)
            work.wait()

    .. py:method:: nccl_async_all_reduce(tensor, c_op="avg",group=None, async_op=False):

        Perform asynchronous or synchronous allreduce on GPU data using NCCL.

        :param tensor: QTensor to reduce.
        :param c_op: Computation method, which can be "sum" or "avg". Default: "avg".
        :param group: Sequence of process numbers for a communication group, with the same meaning as an element of the ``split_groups`` return value. Default None means no grouping.
        :param async_op: Whether this operation is asynchronous. Default: False.
        :return: Work, an asynchronous communication handle. Use wait() to wait for the operation to complete.

        Example::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank

            Comm_OP = CommController("nccl")
            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_all_reduce(complex_data, "sum",group = None,async_op = True)
            work.wait()


    .. py:method:: nccl_async_reduce( tensor_, dest, c_op="avg", group=None, async_op=False ):

        Perform asynchronous or synchronous reduce on GPU data using NCCL.

        :param tensor_: QTensor to reduce.
        :param dest: Destination process number/rank of the QTensor to reduce.
        :param c_op: Computation method, which can be "sum" or "avg". Default: "avg".
        :param group: Sequence of process numbers for a communication group, with the same meaning as an element of the ``split_groups`` return value. Default None means no grouping.
        :param async_op: Whether this operation is asynchronous. Default: False.
        :return: Work, an asynchronous communication handle. Use wait() to wait for the operation to complete.

        Example::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_reduce(complex_data, 0,"sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_broadcast(tensor, src, group=None, async_op=False )

        Perform asynchronous or synchronous broadcast on GPU data using NCCL.

        :param tensor: QTensor to broadcast.
        :param src: Source rank of the QTensor to broadcast.
        :param group: Sequence of process numbers for a communication group, with the same meaning as an element of the ``split_groups`` return value. Default None means no grouping.
        :param async_op: Whether this operation is asynchronous. Default: False.
        :return: Work, an asynchronous communication handle. Use wait() to wait for the operation to complete.

        Example::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank
            Comm_OP = CommController("nccl")
            if get_local_rank() == 1:
                complex_data = tensor.ones([5,5],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            else:
                complex_data = tensor.zeros([5,5],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            work = Comm_OP.nccl_async_broadcast(complex_data, 1, group = None,async_op=True)

            work.wait()



    .. py:method:: nccl_async_send( t, dest, async_op=False ):

        Perform asynchronous or synchronous P2P send on GPU data using NCCL.

        :param t: QTensor to send.
        :param dest: Destination process number/rank of the QTensor to send.
        :param async_op: Whether this operation is asynchronous. Default: False.
        :return: Work, an asynchronous communication handle. Use wait() to wait for the operation to complete.

        Example::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank
            Comm_OP = CommController("nccl")

            if get_local_rank() == 0:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())*10
            else:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())

            if get_local_rank() == 0:
                work = Comm_OP.nccl_async_send(complex_data, 1 ,True)
            else:
                work = Comm_OP.nccl_async_recv(complex_data, 0 ,True)
            work.wait()

    .. py:method:: nccl_async_recv( t, src, async_op=False ):

        Perform asynchronous or synchronous P2P recv on GPU data using NCCL.

        :param t: QTensor to receive.
        :param src: Source process number/rank of the QTensor to receive.
        :param async_op: Whether this operation is asynchronous. Default: False.
        :return: Work, an asynchronous communication handle. Use wait() to wait for the operation to complete.

        Example::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank
            Comm_OP = CommController("nccl")

            if get_local_rank() == 0:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())*10
            else:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())

            if get_local_rank() == 0:
                work = Comm_OP.nccl_async_send(complex_data, 1 ,True)
            else:
                work = Comm_OP.nccl_async_recv(complex_data, 0 ,True)
            work.wait()


    .. py:method:: destroy()

        Destroy NCCL communicator resources / Destroy NCCL communicators explicitly.

        This method should be called before process exit to properly clean up
        NCCL resources. If not called, NCCL communicators may leak memory or
        cause errors during CUDA driver shutdown.

        Example::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            # ... uses Comm_OP ...
            Comm_OP.destroy()  # Call before exit

split_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In multiprocessing, use ``split_data`` to split data according to the number of processes and return the data for the corresponding process.

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

    Set distributed computing parameters.

    :param x_train: `np.array` - Training data.
    :param y_train: `np.array` - Training data labels.
    :param shuffle: `bool` - Whether to shuffle before splitting. Default: False.

    :return: The split training data and labels.

    Example::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    Get the process number on the current node. For example, if you are the 3rd process on the 2nd node and each node has 5 processes, returns 2.

    :return: The current process number on the current machine.

    Example::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun --nproc_per_node 2 python test.py

get_rank
^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    Get the global process number of the current process. For example, if you are the 3rd process on the 2nd node and each node has 5 processes, returns 7.

    :return: The process number of the current process.

    Example::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun --nproc_per_node 2 python test.py

init_groups
^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.distributed.ControlComm.init_groups(rank_lists)

    Divide communication groups according to the given list of process numbers.

    :param rank_lists: Each sublist contains the process numbers of one communication group, for example ``[[0, 2], [1]]``.
    :return: The default (pyvqnet-ad) backend returns the input argument unchanged, that is, a process-number list such as ``[[0, 2], [1]]``; the torch backend returns a list of torch ProcessGroups. With the default backend, this interface does not create communication groups; groups are created when first used for group communication.

    Example::

        from pyvqnet.distributed import *
        import numpy as np
        Comm_OP = CommController("gloo")
        num = tensor.to_tensor(np.random.rand(1, 5))
        print(f"rank {Comm_OP.getRank()}  {num} before allreduce")

        group_l = init_groups([[0,2], [1]])

        for comm_ in group_l:
            if Comm_OP.getRank() in comm_:
                Comm_OP.allreduce_group(num, "sum", group = comm_)
                print(f"rank {Comm_OP.getRank()}  {num} after allreduce")

        # vqnetrun --nproc_per_node 3 python test.py


PipelineParallelTrainingWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.pp.PipelineParallelTrainingWrapper(args,join_layers,trainset)

    Pipeline Parallel Training Wrapper implements 1F1B training. It is available only on Linux with GPUs.
    More algorithm details can be found at (https://www.deepspeed.ai/tutorials/pipeline/).

    :param args: Parameter dictionary. See the example.
    :param join_layers: List of Sequential modules.
    :param trainset: Dataset.

    :return:
        A PipelineParallelTrainingWrapper instance.

    The following uses the CIFAR10 dataset `CIFAR10_Dataset` to train a classification task on AlexNet with 2 GPUs.
    In this example, the training is split into two pipeline parallel processes with `pipeline_parallel_size` = 2.
    The batch size is `train_batch_size` = 64, and `train_micro_batch_size_per_gpu` = 32 on a single GPU.
    Other configuration parameters can be found in `args`.
    In addition, each process needs to set the `LOCAL_RANK` environment variable in the `__main__` function.

    .. code-block::

        os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    Call `train_batch` for training.

    Example::

        import os
        import pyvqnet

        from pyvqnet.nn import Module,Sequential,CrossEntropyLoss
        from pyvqnet.nn import Linear
        from pyvqnet.nn import Conv2D
        from pyvqnet.nn import activation as F
        from pyvqnet.nn import MaxPool2D
        from pyvqnet.nn import CrossEntropyLoss

        from pyvqnet.tensor import tensor
        from pyvqnet.distributed.pp import PipelineParallelTrainingWrapper
        from pyvqnet.distributed.configs import comm as dist
        from pyvqnet.distributed import *


        pipeline_parallel_size = 2

        num_steps = 1000

        def cifar_trainset_vqnet(local_rank, dl_path='./cifar10-data'):
            transform = pyvqnet.data.TransformCompose([
                pyvqnet.data.TransformResize(256),
                pyvqnet.data.TransformCenterCrop(224),
                pyvqnet.data.TransformToTensor(),
                pyvqnet.data.TransformNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            trainset = pyvqnet.data.CIFAR10_Dataset(root=dl_path,
                                                    mode="train",
                                                    transform=transform,layout="HWC")

            return trainset

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.features = Sequential(
                Conv2D(input_channels=3, output_channels=8, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([2, 2], [2, 2]),

                Conv2D(input_channels=8, output_channels=16, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([2, 2], [2, 2]),

                Conv2D(input_channels=16, output_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),

                Conv2D(input_channels=32, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),

                Conv2D(input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([3, 3], [2, 2]),)

                self.cls = Sequential(
                Linear(64 * 27 * 27, 512),
                F.ReLu(),

                Linear(512, 256),
                F.ReLu(),
                Linear(256, 10) )

            def forward(self, x):
                x = self.features(x)
                x = tensor.flatten(x,1)
                x = self.cls(x)

                return x

        def join_layers(vision_model):
            layers = [
                *vision_model.features,
                lambda x: tensor.flatten(x, 1),
                *vision_model.cls,
            ]
            return layers


        if __name__ == "__main__":


            args = {
            "backend":'nccl',
            "train_batch_size" : 64,
            "train_micro_batch_size_per_gpu" : 32,
            "epochs":5,
        "optimizer": {
            "type": "Adam",
            "params": {
            "lr": 0.001
            }},
            "local_rank":dist.get_local_rank(),
            "pipeline_parallel_size":pipeline_parallel_size, "seed":42, "steps":num_steps,
            "loss":CrossEntropyLoss(),
            }
            os.environ["LOCAL_RANK"] = str(dist.get_local_rank())
            trainset = cifar_trainset_vqnet(args["local_rank"])
            w = PipelineParallelTrainingWrapper(args,join_layers(Model()),trainset)

            all_loss = {}

            for i in range(args["epochs"]):
                w.train_batch()

            all_loss = w.loss_dict


ZeroModelInitial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ZeroModelInitial(args=None, model=None, optimizer=None)

    Zero1 API interface, currently used only for GPU-based parallel computing on Linux.

    :param args: Parameter dictionary. See the example.
    :param model: Input model.
    :param optimizer: Optimizer.

    :return:
        Zero1 Engine.

    The following uses the MNIST database to train a classification task on an MLP model with 2 GPUs.

    The batch size is `train_batch_size` = 64, and the `zero_optimization` `stage` is set to 1.
    If Optimizer is None, the `optimizer` setting in `args` is used.
    Other configuration parameters can be found in `args`.
    In addition, each process needs to set the `LOCAL_RANK` environment variable.

    .. code-block::

        os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    Example::

        from pyvqnet.distributed import *
        from pyvqnet import *
        from time import time
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        import pyvqnet
        import numpy as np
        import os
        import struct

        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            load mnist data
            """
            from array import array as pyarray
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images.idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images.idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace(
                    "\\", "/")
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, "rb")
            _, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, "rb")
            _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            num = len(ind)
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels


        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="../data/MNIST_data/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="../data/MNIST_data/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        from pyvqnet.distributed import ZeroModelInitial

        class MNISTClassifier(nn.Module):

            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = nn.Linear(28*28, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, 64)
                self.fc5 = nn.Linear(64, 10)
                self.ac = nn.activation.ReLu()

            def forward(self, x:pyvqnet.QTensor):

                x = x.reshape([-1, 28*28])
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x

        model = MNISTClassifier()

        model.to(local_rank + 1000)

        Comm_op = CommController("nccl")
        Comm_op.broadcast_model_params(model, 0)

        batch_size = 64

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        args_ = {
                "train_batch_size": batch_size, # equivalent total batch size
                "optimizer": {
                    "type": "adam",
                    "params": {
                    "lr": 0.001,
                    }
                },
                "zero_optimization": {
                    "stage": 1,
                }
            }

        os.environ["LOCAL_RANK"] = str(get_local_rank())
        model = ZeroModelInitial(args=args_, model=model, optimizer=optimizer)

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdim=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0
        time1 = time()

        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0

            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size

            for i in range(num_batches):

                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)

                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)

                outputs = model(data_)
                loss = criterion(labels, outputs)

                model.backward(loss) # run backward and step on the returned model
                model.step()

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()

            train_acc = 100 * correct / total

        time2 = time()
        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')

ColumnParallelLinear
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ColumnParallelLinear(input_size,output_size,weight_initializer,bias_initializer,use_bias,dtype,name,tp_comm)

    Tensor parallel computation, column-parallel linear layer.

    The linear layer is defined as Y = XA + b.
    Its 2D parallelism is A = [A_1,...,A_p].

    :param input_size: First dimension of matrix A.
    :param output_size: Second dimension of matrix A.
    :param weight_initializer: `callable`, default `normal`.
    :param bias_initializer: `callable`, default 0.
    :param use_bias: `bool` - default True.
    :param dtype: Default `None`, which uses the default data type.
    :param name: Module name. Default: "".
    :param tp_comm: Communication controller.


    The following uses the MNIST database to train a classification task on an MLP model with 2 GPUs.

    Its use is similar to that of the classical Linear layer.

    For multi-process use, launch it with `# vqnetrun --nproc_per_node 2 python test.py`

    Example::

        import pyvqnet.distributed
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from pyvqnet.distributed import *
        from time import time

        import pyvqnet
        import numpy as np
        import os
        from pyvqnet import *
        import pytest

        Comm_OP = CommController("nccl")

        import struct
        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            load mnist data
            """
            from array import array as pyarray
            # download_mnist(path)
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels-idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels-idx1-ubyte").replace(
                    "\\", "/")
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, "rb")
            _, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, "rb")
            _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            num = len(ind)
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels

        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="./data/MNIST/raw/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="./data/MNIST/raw/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        class MNISTClassifier(nn.Module):
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = RowParallelLinear(28*28, 512, tp_comm = Comm_OP)
                self.fc2 = ColumnParallelLinear(512, 256, tp_comm = Comm_OP)
                self.fc3 = RowParallelLinear(256, 128, tp_comm = Comm_OP)
                self.fc4 = ColumnParallelLinear(128, 64, tp_comm = Comm_OP)
                self.fc5 = RowParallelLinear(64, 10, tp_comm = Comm_OP)
                self.ac = nn.activation.ReLu()

            def forward(self, x:pyvqnet.QTensor):

                x = x.reshape([-1, 28*28])
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x


        model = MNISTClassifier()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.to(local_rank + 1000)

        Comm_OP.broadcast_model_params(model, 0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdim=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0

        time1 = time()
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0

            batch_size = 64
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size

            for i in range(num_batches):
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)

                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)

                optimizer.zero_grad()

                outputs = model(data_)
                loss = criterion(labels, outputs)

                loss.backward()
                optimizer.step()

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()

            train_acc = 100 * correct / total
        time2 = time()

        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')


RowParallelLinear
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.RowParallelLinear(input_size,output_size,weight_initializer,bias_initializer,use_bias,dtype,name,tp_comm)

    Tensor parallel computation, row-parallel linear layer.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X along its second dimension.
    A = transpose([A_1 ... A_p]) X = [X_1, ..., X_p].

    :param input_size: First dimension of matrix A.
    :param output_size: Second dimension of matrix A.
    :param weight_initializer: `callable`, default `normal`.
    :param bias_initializer: `callable`, default 0.
    :param use_bias: `bool` - default True.
    :param dtype: Default `None`, which uses the default data type.
    :param name: Module name.
    :param tp_comm: Communication controller.

    The following uses the MNIST database to train a classification task on an MLP model with 2 GPUs.
    Its use is similar to that of the classical Linear layer.

    For multi-process use, launch it with `# vqnetrun --nproc_per_node 2 python test.py`

    Example::

        import pyvqnet.distributed
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from pyvqnet.distributed import *
        from time import time

        import pyvqnet
        import numpy as np
        import os
        from pyvqnet import *
        import pytest

        Comm_OP = CommController("nccl")

        import struct
        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            load mnist data
            """
            from array import array as pyarray
            # download_mnist(path)
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels-idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels-idx1-ubyte").replace(
                    "\\", "/")
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, "rb")
            _, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, "rb")
            _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            num = len(ind)
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels

        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="./data/MNIST/raw/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="./data/MNIST/raw/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        class MNISTClassifier(nn.Module):
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = RowParallelLinear(28*28, 512, tp_comm = Comm_OP)
                self.fc2 = ColumnParallelLinear(512, 256, tp_comm = Comm_OP)
                self.fc3 = RowParallelLinear(256, 128, tp_comm = Comm_OP)
                self.fc4 = ColumnParallelLinear(128, 64, tp_comm = Comm_OP)
                self.fc5 = RowParallelLinear(64, 10, tp_comm = Comm_OP)
                self.ac = nn.activation.ReLu()


            def forward(self, x:pyvqnet.QTensor):

                x = x.reshape([-1, 28*28])
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x

        model = MNISTClassifier()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.to(local_rank + 1000)
        Comm_OP.broadcast_model_params(model, 0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdim=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0

        time1 = time()
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0

            batch_size = 64
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size

            for i in range(num_batches):
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)

                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)

                optimizer.zero_grad()

                outputs = model(data_)
                loss = criterion(labels, outputs)

                loss.backward()
                optimizer.step()

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()

            train_acc = 100 * correct / total
        time2 = time()

        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')

Bit Reordering
=================================

Qubit reordering is a technique in bit parallelism. Its core idea is to reduce the number of bit permutations required in bit-parallel execution by changing the order of quantum logic gates during bit parallelism. Below are the modules needed to build large-qubit quantum circuits based on bit parallelism. See the paper `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__ .


DistributedQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributedQMachine(num_wires, dtype=kcomplex64, grad_mode="")

    A simulation class for variational quantum computing in bit parallelism, containing the quantum states on the subset of qubits held by each node. Through MPI, each node creates an instance of this class to perform distributed variational quantum circuit simulation. The value of N must equal 2 raised to the power of the number of bit-parallel qubits `global_qubit`, which can be configured through `set_qr_config`.

    :param num_wires: Number of qubits in the full quantum circuit.
    :param dtype: Data type for computation. Default is pyvqnet.kcomplex64, with a corresponding parameter precision of pyvqnet.kfloat32.
    :param grad_mode: Set to adjoint when performing backpropagation in ``DistQuantumLayerAdjoint``.

    .. note::

        The input qubit count is the number of qubits required by the entire quantum circuit. DistributedQMachine builds a quantum simulator based on the global qubit count, and its qubit count is ``num_wires - global_qubit``.
        Backpropagation must be based on ``DistQuantumLayerAdjoint``.

    .. warning::

        This interface supports only Linux.

        You must configure the bit-parallel parameters in ``DistributedQMachine`` as shown in the example, including:

        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # open save op
            qm.set_qr_config({'qubit': total_qubits, 'global_qubit': distributed_qubits})

    Example::

        from pyvqnet.distributed import get_rank
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import rx, ry, cnot, MeasureAll,rz
        import pyvqnet
        from pyvqnet.distributed.qubits_reorder import DistributedQMachine,DistQuantumLayerAdjoint
        pyvqnet.utils.set_random_seed(123)


        qubit = 10
        batch_size = 5

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = DistributedQMachine(num_wires, dtype=dtype, grad_mode=grad_mode)

                self.qm.set_just_defined(True)
                self.qm.set_save_op_history_flag(True) # open save op
                self.qm.set_qr_config({"qubit": num_wires, # open qubit reordered, set config
                                        "global_qubit": 1}) # global_qubit == log2(nproc)

                self.params = pyvqnet.nn.Parameter([qubit])

                self.measure = MeasureAll(obs={
                    "X5":1.0
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                for i in range(qubit):
                    rx(q_machine=self.qm, params=self.params[i], wires=i)
                    ry(q_machine=self.qm, params=self.params[3], wires=i)
                    rz(q_machine=self.qm, params=self.params[4], wires=i)
                cnot(q_machine=self.qm,  wires=[0, 1])
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]], requires_grad=True).toGPU(pyvqnet.DEV_GPU_0+get_rank())

        input_x = tensor.broadcast_to(input_x, [2, 3])

        input_x.requires_grad = True

        quantum_model = QModel(num_wires=qubit,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint").toGPU(pyvqnet.DEV_GPU_0+get_rank())

        adjoint_model = DistQuantumLayerAdjoint(quantum_model)
        adjoint_model.train()

        batch_y = adjoint_model(input_x)
        batch_y.backward()

        print(batch_y)
        # vqnetrun --nproc_per_node 2 python test.py


DistQuantumLayerAdjoint
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistQuantumLayerAdjoint(vqc_module,name)

    A DistQuantumLayer layer that computes gradients of parameters in bit-parallel computation using the adjoint method.

    :param vqc_module: Input module containing ``DistributedQMachine``.
    :param name: Module name.

    .. note::

        The input vqc_module module must contain ``DistributedQMachine``. Adjoint gradient backpropagation under bit parallelism is performed based on ``DistributedQMachine``.

    .. warning::

        This interface supports only Linux.

    Example::

        from pyvqnet.distributed import get_rank
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import rx, ry, cnot, MeasureAll,rz
        import pyvqnet
        from pyvqnet.distributed.qubits_reorder import DistributedQMachine,DistQuantumLayerAdjoint
        pyvqnet.utils.set_random_seed(123)


        qubit = 10
        batch_size = 5

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = DistributedQMachine(num_wires, dtype=dtype, grad_mode=grad_mode)

                self.qm.set_just_defined(True)
                self.qm.set_save_op_history_flag(True) # open save op
                self.qm.set_qr_config({"qubit": num_wires, # open qubit reordered, set config
                                            "global_qubit": 1}) # global_qubit == log2(nproc)

                self.params = pyvqnet.nn.Parameter([qubit])

                self.measure = MeasureAll(obs={
                    "X5":1.0
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                for i in range(qubit):
                    rx(q_machine=self.qm, params=self.params[i], wires=i)
                    ry(q_machine=self.qm, params=self.params[3], wires=i)
                    rz(q_machine=self.qm, params=self.params[4], wires=i)
                cnot(q_machine=self.qm,  wires=[0, 1])
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]], requires_grad=True).toGPU(pyvqnet.DEV_GPU_0+get_rank())

        input_x = tensor.broadcast_to(input_x, [2, 3])

        input_x.requires_grad = True

        quantum_model = QModel(num_wires=qubit,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint").toGPU(pyvqnet.DEV_GPU_0+get_rank())

        adjoint_model = DistQuantumLayerAdjoint(quantum_model)
        adjoint_model.train()

        batch_y = adjoint_model(input_x)
        batch_y.backward()

        print(batch_y)
        # vqnetrun --nproc_per_node 2 python test.py

ParallelTrainingWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.hybrid.ParallelTrainingWrapper(args, pipeline_def, trainset, strategy=None)

    A unified parallel training wrapper that supports three parallel modes: classical layer pipeline parallelism (PP), data parallelism (DP), and bit parallelism (Bit). It can interleave classical layers and variational quantum circuits (VQC) into a multi-level pipeline for execution.

    :param args: Training configuration dictionary. The following keys are supported:

        ``data_parallel_size`` (int, default 1)
            Data parallelism degree.
        ``pipeline_parallel_size`` (int, must equal the total number of stages)
            Pipeline parallelism degree.
        ``qubits_parallel_size`` (int, default 1)
            Bit parallelism degree.
        ``train_batch_size`` (int)
            Global batch size.
        ``train_micro_batch_size_per_gpu`` (int)
            Micro batch size.
        ``epochs`` (int, default 1)
            Number of training epochs.
        ``steps`` (int, default 10)
            Number of training steps per batch.
        ``loss`` (loss function)
            Loss function, needed only in the last stage.
        ``optimizer`` (dict, default ``{"type": "Adam", "params": {"lr": 0.001}}``)
            Optimizer configuration.
        ``device`` (``"cpu"`` / ``"gpu"``, auto-detected if omitted)
            Computing device.
        ``steps_per_print`` (int, default 1)
            Print interval in steps.
        ``seed`` (int, default 42)
            Random seed for determinism of VQC recording.

    :param pipeline_def: Model definition. The following four forms are supported:

        1. Handwritten dictionary list (legacy pipeline_def):

           ``[{"type": "classical", "layers": [...]}, ...]``

           Each entry is a stage dictionary and supports three types: ``classical``, ``vqc_split``, and ``vqc_whole``.

        2. Pure classical model: a list of ``nn.Module`` layers, automatically assigned to stages.

        3. Pure VQC model: a ``QModule`` subclass (must be a class, not an instance).

        4. Hybrid model: a list of ``("classical", [layers])`` / ``("vqc", VQCClass)`` tuples.

        Forms 2-4 are automatically compiled by ``auto_plan`` into the legacy dictionary-list intermediate representation.

    :param trainset: Dataset. It must have ``.data`` and ``.labels`` attributes, and ``__getitem__`` must return a ``(data, label)`` pair.
    :param strategy: Strategy dictionary, used only for declarative model definitions (forms 2-4). Optional keys:

        - ``classical_split`` (``"uniform"`` | ``"auto"``, default ``"auto"``): Classical layer assignment strategy.
        - ``vqc_split`` (``None`` | ``False``, default ``None``): ``None`` means automatically split VQC blocks; ``False`` means keep each VQC block intact.
        - ``compression_policy`` (default ``"greedy"``): Stage merging strategy.

    .. note::

        ``pipeline_parallel_size`` must equal the total number of stages in ``pipeline_def``.

        At startup, the number of processes must be specified through ``vqnetrun``, and ``data_parallel_size × pipeline_parallel_size × qubits_parallel_size`` must equal the total number of processes.

    .. warning::

        This interface supports only Linux.

        It must be started through ``vqnetrun --nproc_per_node <N>``; ``mpirun`` is not supported.

    .. py:method:: train_batch()

        Execute the complete training process for one batch (forward, backward, and parameter update).


    Example::

        import os
        import sys

        import numpy as np

        import pyvqnet
        import pyvqnet.nn as nn

        from pyvqnet.qnn.vqc import rx, ry, rz, crx, MeasureAll
        from pyvqnet.nn import CrossEntropyLoss
        from pyvqnet.distributed.configs import comm as dist
        from pyvqnet.distributed.hybrid.distributed_qmachine import (
            HybridDistributedQMachine,
        )
        from pyvqnet.distributed.hybrid import ParallelTrainingWrapper

        NUM_WIRES = 8
        N_VQC_PARAMS = NUM_WIRES * 4  # 32
        BSZ = 32
        STEPS = 1
        LR = 0.01
        SEED = 42


        class MyVQC(nn.Module):
            def __init__(self, bit_rank):
                super().__init__()
                self.qm = HybridDistributedQMachine(
                    NUM_WIRES, 2, bit_rank, grad_mode="adjoint"
                )
                self.qm.set_just_defined(True)
                self.qm.set_save_op_history_flag(True)
                self.qm.set_qr_config({"qubit": NUM_WIRES, "global_qubit": 1})
                self.params = nn.Parameter([N_VQC_PARAMS])
                self.measure = MeasureAll(obs=[{f"Z{i}": 1.0} for i in range(NUM_WIRES)])

            def forward(self, x):
                self.qm.reset_states(x.shape[0])
                for i in range(NUM_WIRES - 1):
                    rx(q_machine=self.qm, params=x[:, i], wires=i)
                for i in range(NUM_WIRES):
                    rz(q_machine=self.qm, params=self.params[3 * i], wires=i)
                    ry(q_machine=self.qm, params=self.params[3 * i + 1], wires=i)
                    rz(q_machine=self.qm, params=self.params[3 * i + 2], wires=i)
                for i in range(NUM_WIRES):
                    crx(
                        q_machine=self.qm,
                        wires=[i, (i + 1) % NUM_WIRES],
                        params=self.params[NUM_WIRES * 3 + i],
                    )
                return self.measure(q_machine=self.qm)


        class DS:
            def __init__(self, X, Y):
                self.data = X
                self.labels = Y
                self.num_samples = len(X)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, i):
                return self.data[i], self.labels[i]


        def main():
            pyvqnet.backends.set_backend("pyvqnet")
            rank = dist.get_rank()
            ws = dist.get_world_size()
            assert ws == 4, f"requires vqnetrun --nproc_per_node 4, got {ws}"
            os.environ["LOCAL_RANK"] = str(dist.get_local_rank())
            pyvqnet.utils.set_random_seed(SEED)
            np.random.seed(SEED)

            # synthetic dataset
            rng = np.random.RandomState(7)
            N = 256
            X = rng.randn(N, 7).astype(np.float32)
            Y = (np.arange(N) % 8).astype(np.int64)
            for i in range(N):
                c = Y[i]
                X[i, c % 7] += 3.0 * (1 if c < 4 else -1)

            ds = DS(X, Y)

            # 2-stage composite pipeline: classical+VQC | VQC+classical
            args = {
                "device": os.environ.get("VQNET_TEST_DEVICE", "cpu"),
                "train_batch_size": BSZ,
                "train_micro_batch_size_per_gpu": BSZ,
                "epochs": 1,
                "steps": STEPS,
                "data_parallel_size": 1,
                "pipeline_parallel_size": 2,
                "qubits_parallel_size": 2,
                "loss": CrossEntropyLoss(),
                "optimizer": {"type": "Adam", "params": {"lr": LR}},
                "steps_per_print": 50,
                "seed": SEED,
            }
            pipeline_def = [
                {
                    "type": "stage",
                    "sub": [
                        {"type": "classical", "layers": [nn.Linear(7, NUM_WIRES)]},
                        {
                            "type": "vqc_split",
                            "module_class": MyVQC,
                            "vqc_stages": 2,
                            "vqc_slice": 0,
                        },
                    ],
                },
                {
                    "type": "stage",
                    "sub": [
                        {
                            "type": "vqc_split",
                            "module_class": MyVQC,
                            "vqc_stages": 2,
                            "vqc_slice": 1,
                        },
                        {"type": "classical", "layers": [nn.Linear(NUM_WIRES, 8)]},
                    ],
                },
            ]

            w = ParallelTrainingWrapper(args, pipeline_def, ds)

            # run training for one batch
            w.train_batch()


        if __name__ == "__main__":
            main()

        # vqnetrun --nproc_per_node 4 -- python your_script.py
