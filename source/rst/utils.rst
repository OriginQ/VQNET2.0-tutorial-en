Other Utility Functions
=========================

Seeds for Random Distributions
----------------------------------

set_random_seed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.utils.set_random_seed(seed)
    
    Set the global random seed.

    :param seed: random seed.

    .. note::

            When a fixed random number seed is specified, the random distribution will generate a fixed pseudo-random distribution based on the random seed.
            Affects functions include: `tensor.randu` , `tensor.randn` , parameter initialization for parametric classical neural networks and quantum computing layers.

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.utils import get_random_seed, set_random_seed

        set_random_seed(256)


        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randu([2, 3])
        print(rn)
        rn = tensor.randu([2, 3])
        print(rn)

        from pyvqnet.nn.parameter import Parameter
        from pyvqnet.utils.initializer import he_normal, he_uniform, xavier_normal, xavier_uniform, uniform, quantum_uniform, normal
        print(Parameter(shape=[2, 3], initializer=he_normal))
        print(Parameter(shape=[2, 3], initializer=he_uniform))
        print(Parameter(shape=[2, 3], initializer=xavier_normal))
        print(Parameter(shape=[2, 3], initializer=xavier_uniform))
        print(Parameter(shape=[2, 3], initializer=uniform))
        print(Parameter(shape=[2, 3], initializer=quantum_uniform))
        print(Parameter(shape=[2, 3], initializer=normal))
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # ########################################################
        # [
        # [-0.9874518, 0.9198063, 0.0650323],
        #  [0.1976041, 1.0307300, 0.2322134]
        # ]
        # [
        # [-0.2134037, 0.1987845, -0.5292138],
        #  [0.3732708, 0.1775801, 0.5395861]
        # ]
        # [
        # [-0.7648768, 0.7124789, 0.0503738],
        #  [0.1530635, 0.7984000, 0.1798717]
        # ]
        # [
        # [-0.4049051, 0.3771670, -1.0041126],
        #  [0.7082316, 0.3369346, 1.0237927]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # [
        # [1.9803783, 4.2232580, 0.2619299],
        #  [5.1727076, 4.1078768, 6.0776958]
        # ]
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]

get_random_seed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.utils.get_random_seed()
    
    Get current random seed.

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.utils import get_random_seed, set_random_seed

        set_random_seed(256)
        print(get_random_seed())
        #256

VQNet2ONNX module
-------------------

The VQNet2ONNX module supports converting VQNet models and parameters to ONNX model format. The deployment of the VQNet model to a variety of inference engines can be completed through ONNX, including TensorRT/OpenVINO/MNN/TNN/NCNN, and other inference engines or hardware that support the ONNX open source format.

Environment dependency: onnx>=1.12.0

.. note::

    Currently, QPanda quantum circuit modules are not supported to be converted to ONNX, and only models composed of pure classical operators are supported.

Use the ``export_model`` function to export ONNX models. This function requires more than two parameters: the model ``model`` constructed by VQNet, the model single input ``x`` or multi-input ``*argc``.
Below is the sample code for ONNX export of `ResNet` model and validated with onnxruntime.

Import related python libraries

.. code-block::

    import numpy as np
    from pyvqnet.tensor import *
    from pyvqnet.nn import Module, BatchNorm2d, Conv2D, ReLu, AvgPool2D, Linear
    from pyvqnet.onnx.export import export_model
    from onnx import __version__, IR_VERSION
    from onnx.defs import onnx_opset_version
    print(
        f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}"
    )

Model definition

.. code-block::

    class BasicBlock(Module):

        expansion = 1

        def __init__(self, in_chals, out_chals, stride=1):
            super().__init__()
            self.conv2d1 = Conv2D(in_chals,
                                out_chals,
                                kernel_size=(3, 3),
                                stride=(stride, stride),
                                padding=(1, 1),
                                use_bias=False)
            self.BatchNorm2d1 = BatchNorm2d(out_chals)
            self.conv2d2 = Conv2D(out_chals,
                                out_chals * BasicBlock.expansion,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                use_bias=False)
            self.BatchNorm2d2 = BatchNorm2d(out_chals * BasicBlock.expansion)
            self.Relu = ReLu(name="relu")
            #shortcut
            self.shortcut_conv2d = Conv2D(in_chals,
                                        out_chals * BasicBlock.expansion,
                                        kernel_size=(1, 1),
                                        stride=(stride, stride),
                                        use_bias=False)
            self.shortcut_bn2d = BatchNorm2d(out_chals * BasicBlock.expansion)
            self.need_match_dim = False
            if stride != 1 or in_chals != BasicBlock.expansion * out_chals:
                self.need_match_dim = True

        def forward(self, x):
            y = self.conv2d1(x)
            y = self.BatchNorm2d1(y)
            y = self.Relu(self.conv2d2(y))
            y = self.BatchNorm2d2(y)
            y = self.Relu(y)
            if self.need_match_dim == False:
                return y + x
            else:
                y1 = self.shortcut_conv2d(x)
                y1 = self.shortcut_bn2d(y1)
                return y + y1

    resize = 32

    class ResNet(Module):
        def __init__(self, num_classes=10):
            super().__init__()

            self.in_chals = 64 // resize
            self.conv1 = Conv2D(1,
                                64 // resize,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                use_bias=False)
            self.bn1 = BatchNorm2d(64 // resize)
            self.relu = ReLu()
            self.conv2_x_1 = BasicBlock(64 // resize, 64 // resize, 1)
            self.conv2_x_2 = BasicBlock(64 // resize, 64 // resize, 1)
            self.conv3_x_1 = BasicBlock(64 // resize, 128 // resize, 2)
            self.conv3_x_2 = BasicBlock(128 // resize, 128 // resize, 1)
            self.conv4_x_1 = BasicBlock(128 // resize, 256 // resize, 2)
            self.conv4_x_2 = BasicBlock(256 // resize, 256 // resize, 1)
            self.conv5_x_1 = BasicBlock(256 // resize, 512 // resize, 2)
            self.conv5_x_2 = BasicBlock(512 // resize, 512 // resize, 1)
            self.avg_pool = AvgPool2D([4, 4], [1, 1], "valid")
            self.fc = Linear(512 // resize, num_classes)


        def forward(self, x):
            output = self.conv1(x)
            output = self.bn1(output)
            output = self.relu(output)
            output = self.conv2_x_1(output)
            output = self.conv2_x_2(output)
            output = self.conv3_x_1(output)
            output = self.conv3_x_2(output)
            output = self.conv4_x_1(output)
            output = self.conv4_x_2(output)
            output = self.conv5_x_1(output)
            output = self.conv5_x_2(output)
            output = self.avg_pool(output)
            output = tensor.flatten(output, 1)
            output = self.fc(output)

            return output

test code

.. code-block::

    def test_resnet():

        x = tensor.ones([4,1,32,32])#Arbitrarily input a QTensor data of the correct shape
        m = ResNet()
        m.eval()#In order to export the global mean and global variance of BatchNorm
        y = m(x)
        vqnet_y = y.CPU().to_numpy()

        #export onnx model
        onnx_model = export_model(m, x)

        #save model to file
        with open("demo.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        # compare running result by onnxruntime
        import onnxruntime
        session = onnxruntime.InferenceSession('demo.onnx', None)
        input_name = session.get_inputs()[0].name

        v = np.ones([4,1,32,32])
        v = v.astype(np.float32)
        inputs = [v]
        test_data_num = len(inputs)
        outputs = [
            session.run([], {input_name: inputs[i]})[0]
            for i in range(test_data_num)
        ]
        onnx_y = outputs[0]
        assert np.allclose(onnx_y, vqnet_y)


    if __name__ == "__main__":
        test_resnet()


Use https://netron.app/ , the ONNX model exported by VQNet can be visualized demo.onnx

.. image:: ./images/resnet_onnx.png
   :width: 100 px
   :align: center

|


The following are the supported VQNet modules

.. csv-table:: Supoorted VQNet modules
   :file: ./images/onnxsupport.csv


VQNet distributed computing module
----------------------------------

VQNet distributed computing module supports the quantum machine learning model through the corresponding interface of the distributed computing module to achieve data segmentation, 
communication of model parameters between multiple processes, and update of model parameters. The model is accelerated based on the distributed computing module.

init_process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``init_process`` to initialize distributed computing parameters.

.. py:function:: pyvqnet.distributed.init_process(size, path, hostpath=None, train_size=None, test_size=None, shuffle=False)

    Set distributed computing parameters.

    :param size: Number of processes.
    :param path: The absolute path of the current running file.
    :param hostpath: Absolute path to the multi-node configuration file.
    :param train_size: Training set size.
    :param test_size: Test set size.
    :param shuffle: Whether to randomly sample.

    Example::

        import argparse
        import os
        from pyvqnet.distributed import *

        parser = argparse.ArgumentParser(description='parser example')
        parser.add_argument('--init', default=False, type=bool, help='whether to use multiprocessing')
        parser.add_argument('--np', default=1, type=int, help='number of processes')
        parser.add_argument('--hostpath', default=None, type=str, help='multi node configuration files')
        parser.add_argument('--shuffle', default=False, type=bool, help='shuffle')
        parser.add_argument('--train_size', default=120, type=int, help='train_size')
        parser.add_argument('--test_size', default=50, type=int, help='test_size')
        args = parser.parse_args()

        if(args.init):
            init_process(args.np, os.path.realpath(__file__))
        else:
            ...

split_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In multiple processes, use ``split_data`` to split the data according to the number of processes and return the data on the corresponding process.

.. py:function:: pyvqnet.distributed.split_data(x_train, y_train, shuffle=False)

    :param x_train: `np.array` - training data.
    :param y_train: `np.array` -  training data labels.
    :param shuffle: `bool` - Whether to shuffle before segmenting, the default value is False.

    :return: Split training data and labels.

    Example::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

        return x_train, y_train

model_allreduce
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``model_allreduce`` to pass and update model parameters in different processes in the allreduce manner.

.. py:function:: pyvqnet.distributed.model_allreduce(model)

    :param model: `Module` - the trained model.
    
    :return: Model after updated parameters.

    Example::

        from pyvqnet.distributed import parallel_model
        import numpy as np
        from pyvqnet.nn.module import Module
        from pyvqnet.nn.linear import Linear
        from pyvqnet.nn import activation as F
        from pyvqnet.distributed import *

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc = Linear(input_channels=5, output_channels=1)

            def forward(self, x):
                x = F.ReLu()(self.fc(x))
                return x

        model = Net()
        print(f"rank {get_rank()} parameters is {model.parameters()}")
        model = parallel_model(model)

        if get_rank() == 0:
            print(model.parameters())
        
        # mpirun -n 2 python run.py

model_reduce
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``model_reduce`` to pass and update model parameters in different processes in the form of reduce.

.. py:function:: pyvqnet.distributed.model_reduce(x_train, y_train, shuffle=False)

    :param model: `Module` - the trained model.

    :return: Model after updated parameters.

    Example::

        from pyvqnet.distributed import model_reduce
        import numpy as np
        from pyvqnet.nn.module import Module
        from pyvqnet.nn.linear import Linear
        from pyvqnet.nn import activation as F
        from pyvqnet.distributed import *

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc = Linear(input_channels=5, output_channels=1)

            def forward(self, x):
                x = F.ReLu()(self.fc(x))
                return x


        model = Net()
        print(f"rank {get_rank()} parameters is {model.parameters()}")
        model = model_reduce(model)

        if get_rank() == 0:
            print(model.parameters())

        # mpirun -n 2 python run.py
        
Environment dependency: mpich, mpi4py,gcc,gfortran

.. note::

    Currently, only CPU-based distributed computing is supported, and distributed computing using gloo and nccl as communication libraries is not supported.

Distributed computing single node environment deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Complete the compilation and installation of the mpich communication library, 
    and check whether the gcc and gfortran compilers are installed before compilation.

    .. code-block::
            
        which gcc 
        which gfortran
    
    When the paths of gcc and gfortran are displayed, you can proceed to the next step of installation. 
    If there is no corresponding compiler, please install the compiler first. After checking the compiler, use the wget command to download it.
    
    .. code-block::
            
        wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz 
        tar -zxvf mpich-3.3.2.tar.gz 
        cd mpich-3.3.2 
        ./configure --prefix=/usr/local/mpich-3.3.2 
        make 
        make install 
    
    After completing the compilation and installation of mpich, you need to configure its environment variables.
    
    .. code-block::
            
        vim ~/.bashrc
    
    Open the .bashrc file corresponding to the current user through vim and add a line to it (it is recommended to add it to the bottom line)
    
    .. code-block::
    
        export PATH="/usr/local/mpich-3.3.2/bin:$PATH"
    
    After saving and exiting, use the source command to execute the newly added command.
    
    .. code-block::
    
        source ~/.bashrc
    
    After, use which to check whether the configured environment variables are correct. If its path is displayed, the installation was successfully completed.

Distributed computing multi-node environment deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    To implement distributed computing on multiple nodes, you first need to ensure that the mpich environment and python environment on multiple nodes are consistent. 
    Secondly, you need to set up secret-free communication between nodes.
    Assume that you need to set up secret-free communication for three nodes: node0 (master node), node1, and node2.

    .. code-block::

        Execute on each node

        ssh-keygen 
        
        Then press Enter to generate a public key (id_rsa.pub) and a private key (id_rsa) in the .ssh folder.

        Add the public keys of the other two nodes to the authorized_keys file of the first node,
        then transfer the authorized_keys file of the first node to the other two nodes to achieve secret-free communication between nodes.
        Execute on child node node1

        cat ~/.ssh/id_dsa.pub >> node1：~/.ssh/authorized_keys

        Execute on child node node2

        cat ~/.ssh/id_dsa.pub >> node2：~/.ssh/authorized_keys
        
        First delete the authorized_keys files in node1 and node2, and then execute on node0

        scp ~/.ssh/authorized_keys  node1：~/.ssh/authorized_keys
        scp ~/.ssh/authorized_keys  node2：~/.ssh/authorized_keys

        Ensure that the public keys generated by three different nodes are in the authorized_keys file to achieve secret-free communication between nodes.

    In addition, it is best to set up a shared directory so that when the files in the shared directory are changed, 
    the files in different nodes will also be changed to prevent the problem of out-of-synchronization of files in different nodes when running the model on multiple nodes.
    Use nfs-utils and rpcbind to implement shared directories.

    .. code-block::

        # Install packages
        yum -y install nfs* rpcbind  

        # Edit the configuration file on the master node
        vim /etc/exports  
        /data/mpi *(rw,sync,no_all_squash,no_subtree_check)

        # Start the service on the master node
        systemctl start rpcbind
        systemctl start nfs

        # Mount the directory to be shared on all child nodes node1 and node2
        mount node1:/data/mpi/ /data/mpi
        mount node2:/data/mpi/ /data/mpi


Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This block introduces how to use the VQNet distributed computing interface to implement data parallel training models on the CPU hardware platform. 
The use case is the test_mdis.py file in the example directory.

Import related libraries

.. code-block::

    import sys
    sys.path.insert(0,"../")
    import time
    import os
    import struct
    import gzip
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    import time
    import numpy as np
    import matplotlib
    from pyvqnet.distributed import *  # 分布式计算模块
    import argparse 

Data load

.. code-block::

    url_base = "http://yann.lecun.com/exdb/mnist/"
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }
    if_show_sample = 0
    grad_time = []
    forward_time = []
    forward_time_sum = []

    def _download(dataset_dir, file_name):
        """
        Download mnist data if needed.
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")


    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):
        """
        load mnist data
        """
        from array import array as pyarray
        download_mnist(path)
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
        images = np.zeros((num, rows, cols))
        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                     cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels


    def data_select(train_num, test_num):
        """
        Select data from mnist dataset.
        """

        x_train, y_train = load_mnist("training_data")  # 下载训练数据
        x_test, y_test = load_mnist("testing_data")
        idx_train = np.append(
                np.where(y_train == 0)[0][0:train_num],
                np.where(y_train == 1)[0][0:train_num])
        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)

        idx_test = np.append(
                np.where(y_test == 0)[0][:test_num],
                np.where(y_test == 1)[0][:test_num])
        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)

        return x_train, y_train, x_test, y_test

Model design

.. code-block::

    def circuit_func(weights):
        """
        A function using QPanda to create quantum circuits and run.
        """
        num_qubits = 1
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        cbits = machine.cAlloc_many(num_qubits)
        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], weights[0]))
        prog = pq.QProg()
        prog.insert(circuit)
        prog << pq.measure_all(qubits, cbits)  #pylint:disable=expression-not-assigned

        result = machine.run_with_configuration(prog, cbits, 1000)

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        # Compute probabilities for each state
        probabilities = counts / 100
        # Get state expectation
        expectation = np.sum(states * probabilities)
        return expectation

    class Hybrid(Module):
        """ Hybrid quantum - Quantum layer definition """
        def __init__(self, shift):
            super(Hybrid, self).__init__()
            self.shift = shift
            self.input = None

        def forward(self, x):
            self.input = x
            expectation_z = circuit_func(np.array(x.data))
            result = [[expectation_z]]
            # requires_grad = x.requires_grad and not QTensor.NO_GRAD
            requires_grad = x.requires_grad
            def _backward_mnist(g, x):
                """ Backward pass computation """
                start_grad_time = time.time()
                input_list = np.array(x.data)
                shift_right = input_list + np.ones(input_list.shape) * self.shift
                shift_left = input_list - np.ones(input_list.shape) * self.shift

                gradients = []
                for i in range(len(input_list)):
                    expectation_right = circuit_func(shift_right[i])
                    expectation_left = circuit_func(shift_left[i])
                    gradient = expectation_right - expectation_left
                    gradients.append(gradient)
                gradients = np.array([gradients]).T

                end_grad_time = time.time()
                grad_time.append(end_grad_time - start_grad_time)
                in_g = gradients * np.array(g)
                return in_g

            nodes = []
            if x.requires_grad:
                nodes.append(
                    QTensor.GraphNode(tensor=x,
                                      df=lambda g: _backward_mnist(g, x)))
            return QTensor(data=result, requires_grad=requires_grad, nodes=nodes)


    class Net(Module):
        """
        Hybird Quantum Classci Neural Network Module
        """
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1,
                                output_channels=6,
                                kernel_size=(5, 5),
                                stride=(1, 1),
                                padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6,
                                output_channels=16,
                                kernel_size=(5, 5),
                                stride=(1, 1),
                                padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")

            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)

            self.hybrid = Hybrid(np.pi / 2)
            self.fc3 = Linear(input_channels=1, output_channels=2)

        def forward(self, x):
            start_time_forward = time.time()
            x = F.ReLu()(self.conv1(x))

            x = self.maxpool1(x)
            x = F.ReLu()(self.conv2(x))

            x = self.maxpool2(x)
            x = tensor.flatten(x, 1)

            x = F.ReLu()(self.fc1(x))
            x = self.fc2(x)

            start_time_hybrid = time.time()
            x = self.hybrid(x)

            end_time_hybrid = time.time()

            forward_time.append(end_time_hybrid - start_time_hybrid)

            x = self.fc3(x)
            end_time_forward = time.time()
            forward_time_sum.append(end_time_forward - start_time_forward)
            return x


None of the above uses the distributed computing interface, 
but only needs to reference split_data, model_allreduce, and init_process during training to achieve data parallel distributed computing.

as follows

.. code-block::

    def run(args):
        """
        Run mnist train function
        """
        x_train, y_train, x_test, y_test = data_select(args.train_size, args.test_size)

        x_train, y_train = split_data(x_train, y_train) # Distributed module interface splits data
        print(get_rank())
        model = Net()
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_func = CategoricalCrossEntropy()

        epochs = 10
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        model.train()

        for epoch in range(1, epochs):
            total_loss = []
            model.train()
            batch_size = 1
            correct = 0
            n_train = 0

            for x, y in data_generator(x_train,
                                       y_train,
                                       batch_size=1,
                                       shuffle=False):

                x = x.reshape(-1, 1, 28, 28)

                optimizer.zero_grad()
                output = model(x)
                loss = loss_func(y, output)
                loss_np = np.array(loss.data)

                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y.argmax(1))
                correct += np.sum(np.array(mask))
                n_train += batch_size

                loss.backward()
                optimizer._step()

                total_loss.append(loss_np)
            model = model_allreduce(model) # Allreduce communication for model parameter gradients of different ranks


            train_loss_list.append(np.sum(total_loss) / len(total_loss))
            train_acc_list.append(np.sum(correct) / n_train)
            print("{:.0f} loss is : {:.10f}".format(epoch, train_loss_list[-1]))

            model.eval()
            correct = 0
            n_eval = 0

            for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
                x = x.reshape(-1, 1, 28, 28)
                output = model(x)
                loss = loss_func(y, output)
                loss_np = np.array(loss.data)
                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y.argmax(1))
                correct += np.sum(np.array(mask))
                n_eval += 1

                total_loss.append(loss_np)
            print(f"Eval Accuracy: {correct / n_eval}")
            val_loss_list.append(np.sum(total_loss) / len(total_loss))
            val_acc_list.append(np.sum(correct) / n_eval)

    if __name__ == "__main__":

        parser = argparse.ArgumentParser(description='parser example')
        parser.add_argument('--init', default=False, type=bool, help='whether to use multiprocessing')
        parser.add_argument('--np', default=1, type=int, help='number of processes')
        parser.add_argument('--hostpath', default=None, type=str, help='hosts absolute path')
        parser.add_argument('--shuffle', default=False, type=bool, help='shuffle')
        parser.add_argument('--train_size', default=120, type=int, help='train_size')
        parser.add_argument('--test_size', default=50, type=int, help='test_size')
        args = parser.parse_args()
        # p_path = os.path.realpath (__file__)

        if(args.init):
            init_process(args.np, os.path.realpath(__file__), args.hostpath, args.train_size,args.test_size, args.shuffle)
        else:
            a = time.time()
            run(args)
            b=time.time()
            if(get_rank()==0):
                print("time: {}",format(b-a))
                
Among them, init represents whether it is based on a distributed training model, np represents the number of processes, 
and the absolute path of the configuration file when the hostpath file code runs the model on multiple nodes. 
The content of the configuration file includes the IPs of multiple nodes and process allocation, as follows

.. code-block::

    node0:1
    node1:1
    node2:1


Enter code at the command line

.. code-block::

    python test_mdis.py --init true

    0
    1 loss is : 0.8230862300
    Eval Accuracy: 0.5
            ...
    9 loss is : 0.5660219193
    Eval Accuracy: 0.46
    time: {} 15.132369756698608


    python test_mdis.py --init true --np 2

    1
    1 loss is : 0.0316730281
    Eval Accuracy: 0.5
            ...
    9 loss is : 0.0006756162
    Eval Accuracy: 0.5

    0
    1 loss is : 0.0072183679
    Eval Accuracy: 0.85
            ...
    9 loss is : 0.0001979264
    Eval Accuracy: 0.82
    time: {} 9.132536888122559

The above is a multi-process model training on a single node. It can be clearly seen that the training time is shortened.

To train on multiple nodes, the command is as follows

.. code-block::

    python3 test_mdis.py --init true --np 4 --hostpath ~/workspace/hao/vqnet/pyVQNet/examples/host.txt

    0
    1 loss is : 0.8609524409
    Eval Accuracy: 0.5
            ...
    9 loss is : 0.4251357079
    Eval Accuracy: 0.5
    time: {} 6.5950517654418945
    
    3
    1 loss is : 0.0034498004
    Eval Accuracy: 0.5
            ...
    9 loss is : 0.0001483827
    Eval Accuracy: 0.5
    
    1
    1 loss is : 0.0990966797
    Eval Accuracy: 0.5
            ...
    9 loss is : 0.0037492002
    Eval Accuracy: 0.5
    
    2
    1 loss is : 0.8468652089
    Eval Accuracy: 0.5
            ...
    Eval Accuracy: 0.53
    9 loss is : 0.4186156909
    Eval Accuracy: 0.52
