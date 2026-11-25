VQNet Installation Steps
==================================

Installing the VQNet Python Package
-----------------------------------

We provide prebuilt Python packages for installation on Linux, Windows, and macOS 13+ (arm64). **Python 3.10 to Python 3.14** is required.

The installation command is as follows:

.. code-block::

    pip install pyvqnet --upgrade

For Windows and Linux, pyvqnet automatically installs the CUDA 12.6 based NVIDIA runtime libraries through pip during installation (the version is specified by the installation dependency declaration of ``pyvqnet``), to accelerate classical neural network computation.
The package is compiled with optimizations for the following CUDA architectures:
**sm_80** (NVIDIA A100 and A30 series data center GPUs) and **sm_86** (NVIDIA GeForce RTX 30 series consumer GPUs). Make sure to use a GPU that supports these architectures, otherwise the program may not run correctly.

    .. important::

        Note that this package does not separate CPU and GPU builds; on both Windows and Linux it installs the NVIDIA CUDA 12.6 runtime libraries through pip dependencies. If a framework such as PyTorch installed in the current environment was compiled against a different CUDA version, a CUDA runtime library version conflict may occur, causing abnormal program behavior (for example, the GPU is not recognized or training reports errors). Installing in an isolated virtual environment or container is recommended.




Testing a Successful VQNet Installation
---------------------------------------




.. code-block::

    import pyvqnet
    from pyvqnet.tensor import *
    a = arange(1,25).reshape([2, 3, 4])
    print(a)

Testing VQNet GPU Functionality
----------------------------------

.. code-block::

    from pyvqnet import DEV_GPU_0
    from pyvqnet.tensor import *
    a = ones([4,5],device = DEV_GPU_0)
    print(a)

A Simple VQNet Example
----------------------

This section introduces a complete machine learning workflow built with the classical neural network modules and quantum modules of VQNet. The example is based on `Data re-uploading for a universal quantum classifier <https://arxiv.org/abs/1907.02085>`_.
A quantum computing module in quantum machine learning generally consists of the following parts:

(1) Encoder circuit (Encoder), used to encode classical data into quantum data;
(2) Variational quantum circuit (Ansatz), used to train the parameters of parameterized quantum gates;
(3) Measurement module (Measurement), used to obtain measured values (that is, the projection of the quantum state of a qubit onto certain axes).

Like the computational modules of a classical neural network, a quantum computing module is differentiable. This is the theoretical basis of hybrid quantum-classical neural network models.
VQNet supports combining quantum computing modules with classical computing modules (for example, convolution, pooling, fully connected layers, and activation functions) to form hybrid machine learning models, and provides multiple optimization algorithms for parameter optimization.

.. figure:: ./images/classic-quantum.PNG

For quantum computing modules, VQNet supports building quantum modules with `pyqpanda3 <https://qcloud.originqc.com.cn/document/qpanda-3/index.html>`_, Origin Quantum's efficient quantum software package.
With the common interfaces provided by pyQPanda3, users can quickly build quantum computing modules.

In the following example, we use pyQPanda3 to build a quantum computing module. Through VQNet, this quantum computing module can be embedded directly into a hybrid machine learning model to train quantum circuit parameters.
This example uses 1 qubit, encodes the input x with multiple parameterized rotation gates `RZ`, `RY`, and `RZ`, and uses the `probs_measure` function to observe the probability measurement results of the qubit as the output.

.. code-block::

    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3 import probs_measure
    def qdrl_circuit(input,weights):
        qlist = range(1)
        machine = pq.CPUQVM()
        x1 = input.squeeze()
        param1 = weights.squeeze()
        #Build a quantum circuit instance using the pyqpanda3 interface
        circuit = pq.QCircuit()
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter x1[0]
        circuit << pq.RZ(qlist[0], x1[0])
        #Insert an RY gate on the first qubit using the pyqpanda3 interface, with parameter x1[1]
        circuit << pq.RY(qlist[0], x1[1])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter x1[2]
        circuit << pq.RZ(qlist[0], x1[2])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter param1[0]
        circuit << pq.RZ(qlist[0], param1[0])
        #Insert an RY gate on the first qubit using the pyqpanda3 interface, with parameter param1[1]
        circuit << pq.RY(qlist[0], param1[1])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter param1[2]
        circuit << pq.RZ(qlist[0], param1[2])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter x1[0]
        circuit << pq.RZ(qlist[0], x1[0])
        #Insert an RY gate on the first qubit using the pyqpanda3 interface, with parameter x1[1]
        circuit << pq.RY(qlist[0], x1[1])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter x1[2]
        circuit << pq.RZ(qlist[0], x1[2])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter param1[3]
        circuit << pq.RZ(qlist[0], param1[3])
        #Insert an RY gate on the first qubit using the pyqpanda3 interface, with parameter param1[4]
        circuit << pq.RY(qlist[0], param1[4])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter param1[5]
        circuit << pq.RZ(qlist[0], param1[5])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter x1[0]
        circuit << pq.RZ(qlist[0], x1[0])
        #Insert an RY gate on the first qubit using the pyqpanda3 interface, with parameter x1[1]
        circuit << pq.RY(qlist[0], x1[1])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter x1[2]
        circuit << pq.RZ(qlist[0], x1[2])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter param1[6]
        circuit << pq.RZ(qlist[0], param1[6])
        #Insert an RY gate on the first qubit using the pyqpanda3 interface, with parameter param1[7]
        circuit << pq.RY(qlist[0], param1[7])
        #Insert an RZ gate on the first qubit using the pyqpanda3 interface, with parameter param1[8]
        circuit << pq.RZ(qlist[0], param1[8])
        #Build the quantum program
        prog = pq.QProg()
        prog << circuit
        #Get the probability measurement values
        prob = probs_measure(machine ,prog,  qlist)

        return prob

The machine learning task in this example is to perform binary classification on randomly generated data. The figure below shows a sample of this data: the origin is the center of the circle, red 2-D points within a radius of 1 form one class, and blue points form the other class.

.. figure:: ./images/origin_circle.png

Training and Test Code Flow

.. code-block::

    #Import the required libraries and functions
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.optim import adam
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor import QTensor
    import numpy as np
    from pyvqnet.nn.module import Module

Define the model Model, where the `__init__` function defines each internal neural network module and quantum module, and the `forward` function defines the forward pass. `QuantumLayer` is an abstract class that wraps quantum computation.
You only need to pass the quantum computing function `qdrl_circuit`, the number of trainable parameters `param_num`, the backend configuration `"cpu"`, and the number of qubits `qbit_num`. The class then computes parameter gradients automatically in `VQNet`.

.. code-block::

    #Number of trainable parameters
    param_num = 9
    #Number of qubits in the quantum computing module
    qbit_num  = 1
    #Define a machine learning model class that inherits from Module
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            #The QuantumLayer class brings a quantum circuit with trainable parameters into VQNet's automatic differentiation training flow
            self.pqc = QuantumLayer(qdrl_circuit,param_num)
        #Define the model forward function
        def forward(self, x):
            x = self.pqc(x)
            return x

Define some functions required for training the model

.. code-block::

    # Function to randomly generate training data
    def circle(samples:int,  rads =  np.sqrt(2/np.pi)) :
        data_x, data_y = [], []
        for i in range(samples):
            x = 2*np.random.rand(2) - 1
            y = [0,1]
            if np.linalg.norm(x) < rads:
                y = [1,0]
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x,dtype=np.float32), np.array(data_y,np.int64)

    # Data loading function
    def get_minibatch_data(x_data, label, batch_size):
        for i in range(0,x_data.shape[0]-batch_size+1,batch_size):
            idxs = slice(i, i + batch_size)
            yield x_data[idxs], label[idxs]

    # Function to compute accuracy
    def get_score(pred, label):
        pred, label = np.array(pred.data), np.array(label.data)
        pred = np.argmax(pred,axis=1)
        score = np.argmax(label,1)
        score = np.sum(pred == score)
        return score

VQNet follows the standard machine learning training and test workflow: iterate over loading data, forward computation, loss computation, backward computation, and parameter updates.

.. code-block::

    #Instantiate the defined model
    model = Model()
    #Define an optimizer, using Adam here
    optimizer = adam.Adam(model.parameters(),lr =0.6)
    #Define a loss function, using cross-entropy loss here
    Closs = CategoricalCrossEntropy()

Functions for training the model

.. code-block::

    def train():

        # Randomly generate training data
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1),dtype=np.float32)))
        # Define the number of data samples per training batch
        batch_size = 32
        # Maximum number of training epochs
        epoch = 10
        print("start training...........")
        for i in range(epoch):
            model.train()
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train,batch_size):
                # Clear cached gradients in the optimizer
                optimizer.zero_grad()
                # Model forward computation
                output = model(data)
                # Compute the loss function
                losss = Closs(label, output)
                # Loss backpropagation
                losss.backward()
                # Update optimizer parameters
                optimizer._step()
                # Compute metrics such as accuracy
                accuracy += get_score(output,label)

                loss += losss.item()
                count += batch_size

            print(f"epoch:{i}, train_accuracy:{accuracy/count}")
            print(f"epoch:{i}, train_loss:{loss/count}\n")


Functions for validating the model

.. code-block::

    def test():

        batch_size = 1
        model.eval()
        print("start eval...................")
        xtest, y_test = circle(500)
        test_accuracy = 0
        count = 0
        x_test = np.hstack((xtest, np.zeros((xtest.shape[0], 1),dtype=np.float32)))

        for test_data, test_label in get_minibatch_data(x_test,y_test, batch_size):

            test_data, test_label = QTensor(test_data),QTensor(test_label)
            output = model(test_data)
            test_accuracy += get_score(output, test_label)
            count += batch_size

        print(f"test_accuracy:{test_accuracy/count}")

Training and test results:

.. code-block::

    start training...........
    epoch:0, train_accuracy:0.6145833333333334
    epoch:0, train_loss:0.020432369535168013

    epoch:1, train_accuracy:0.6854166666666667
    epoch:1, train_loss:0.01872217481335004

    epoch:2, train_accuracy:0.8104166666666667
    epoch:2, train_loss:0.016634768371780715

    epoch:3, train_accuracy:0.7479166666666667
    epoch:3, train_loss:0.016975031544764835

    epoch:4, train_accuracy:0.7875
    epoch:4, train_loss:0.016502128106852372

    epoch:5, train_accuracy:0.8083333333333333
    epoch:5, train_loss:0.0163204787299037

    epoch:6, train_accuracy:0.8083333333333333
    epoch:6, train_loss:0.01634311651190122

    epoch:7, train_loss:0.016330583145221074

    epoch:8, train_accuracy:0.8125
    epoch:8, train_loss:0.01629052646458149

    epoch:9, train_accuracy:0.8083333333333333
    epoch:9, train_loss:0.016270687493185203

    start eval...................
    test_accuracy:0.826

.. figure:: ./images/qdrl_for_simple.png
