Quantum Machine Learning Examples with pyqpanda3
#################################################

We use VQNet and pyqpanda3 to implement several quantum machine learning examples.


.. warning::

    The quantum computing parts of the following interfaces use pyqpanda3 https://qcloud.originqc.com.cn/document/qpanda-3/.

    You need to install pyqpanda3 separately: `pip install pyqpanda3`


Applications of parameterized quantum circuits in classification tasks
**********************************************************************

1. QVC example
=========================

This example uses VQNet to implement the variable quantum circuit from `Circuit-centric quantum classifiers <https://arxiv.org/pdf/1804.00633.pdf>`_ for a binary classification task.
The example decides whether a binary number is odd or even. By encoding the binary number into qubits and optimizing the variable parameters in the circuit, the z-direction measurement of the circuit indicates whether the input is odd or even.

Quantum circuit
-------------------
A variational quantum circuit usually defines a sub-circuit, a basic circuit architecture that can be repeated in layers to build complex variational circuits.
Our circuit layer consists of several rotation gates and ``CNOT`` gates that entangle each qubit with its neighbor.
We also need a circuit that encodes classical data into quantum states, so that the measured output is related to the input.
In this example, we encode the binary input into qubits in the corresponding order. For example, input data 1101 is encoded into 4 qubits. This example uses pyqpanda3.

.. math::

    x = 0101 \rightarrow|\psi\rangle=|0101\rangle

.. image:: ./images/qvc_circuit.png
   :width: 600 px
   :align: center

|

.. code-block::

    import pyqpanda3.core as pq
    from pyvqnet.nn.module import Module
    from pyvqnet.optim.sgd import SGD
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.data import data_generator as dataloader

    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.qnn.pq3.measure import probs_measure
    qnum = 4
    def qvc_circuits(input,weights):

        qlist = range(qnum)
        machine =pq.CPUQVM()
        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir << pq.CNOT(nqubits[i],nqubits[i+1])
            cir << pq.CNOT(nqubits[len(nqubits)-1],nqubits[0])
            return cir

        def build_circuit(weights, xx, nqubits):

            def Rot(weights_j, qubits):
                circuit = pq.QCircuit()
                circuit << pq.RZ(qubits, weights_j[0])
                circuit << pq.RY(qubits, weights_j[1])
                circuit << pq.RZ(qubits, weights_j[2])
                return circuit
            def basisstate():
                circuit = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i] == 1:
                        circuit << pq.X(nqubits[i])
                return circuit

            circuit = pq.QCircuit()
            circuit << basisstate()

            for i in range(weights.shape[0]):

                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circuit << Rot(weights_j,nqubits[j])
                cnots = get_cnot(nqubits)
                circuit << cnots

            circuit << pq.Z(nqubits[0])

            prog = pq.QProg()
            prog << circuit
            return prog

        weights = weights.reshape([2,4,3])
        prog = build_circuit(weights,input,qlist)
        prob = probs_measure(machine,prog,qlist[0])

        return prob



Model construction
-------------------
We have defined the variable quantum circuit ``qvc_circuits``. We want to use it in VQNet's automatic differentiation logic and train the model with VQNet optimization algorithms. We define a Model class that inherits from the abstract class ``Module``.
The Model uses the ``pyvqnet.qnn.pq3.QuantumLayer`` class, an automatically differentiable quantum computing layer. ``qvc_circuits`` is the quantum circuit we want to run, 24 is the number of trainable quantum circuit parameters, "cpu" means the pyqpanda3 full-amplitude simulator is used, and 4 means 4 qubits are requested.
In the ``forward()`` function, the user defines the forward logic of the model.

.. code-block::

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24)

        def forward(self, x):
            return self.qvc(x)



Model training and testing
--------------------------
We use pre-generated random binary numbers and their odd/even labels. The data is as follows:

.. code-block::

    import numpy as np
    import os
    qvc_train_data = [0,1,0,0,1,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 0,
    0, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 0,
    1, 0, 1, 1, 1,
    1, 1, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 0, 1,
    1, 1, 1, 1, 0]
    qvc_test_data= [0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 1, 0, 1,
    0, 0, 1, 1, 0]

    def get_data(dataset_str):
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1,5])
        data = datasets[:,:-1]
        label = datasets[:,-1].astype(int)
        label = np.eye(2)[label].reshape(-1,2)
        return data, label



Then we can train as with a normal neural network: forward pass, loss computation, backward pass, optimizer step, until the preset number of iterations is reached.
The training data is the qvc_train_data generated above, and the test data is qvc_test_data.

.. code-block::

    def get_accuracy(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    # Instantiate the Model class
    model = Model()
    # Define the optimizer; model.parameters() gives all trainable parameters, lr is the learning rate
    optimizer = SGD(model.parameters(),lr =0.1)
    # The batch size can be changed during training
    batch_size = 3
    # Maximum number of training iterations
    epoch = 20
    # Model loss function
    loss = CategoricalCrossEntropy()

    model.train()
    datas,labels = get_data("train")

    for i in range(epoch):
        count=0
        sum_loss = 0
        accuary = 0
        t = 0
        for data,label in dataloader(datas,labels,batch_size,False):
            optimizer.zero_grad()
            data,label = QTensor(data), QTensor(label)

            result = model(data)

            loss_b = loss(label,result)
            loss_b.backward()
            optimizer._step()
            sum_loss += loss_b.item()
            count+=batch_size
            accuary += get_accuracy(result,label)
            t = t + 1

        print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}")

    model.eval()
    count = 0
    test_data,test_label = get_data("test")
    test_batch_size = 1
    accuary = 0
    sum_loss = 0
    for testd,testl in dataloader(test_data,test_label,test_batch_size):
        testd = QTensor(testd)
        test_result = model(testd)
        test_loss = loss(testl,test_result)
        sum_loss += test_loss
        count+=test_batch_size
        accuary += get_accuracy(test_result,testl)
    print(f"test:--------------->loss:{sum_loss/count} #####accuracy:{accuary/count}")

.. code-block::

    epoch:0, #### loss:0.20194714764753977 #####accuracy:0.6666666666666666
    epoch:1, #### loss:0.19724808633327484 #####accuracy:0.8333333333333334
    epoch:2, #### loss:0.19266503552595773 #####accuracy:1.0
    epoch:3, #### loss:0.18812804917494455 #####accuracy:1.0
    epoch:4, #### loss:0.1835678368806839 #####accuracy:1.0
    epoch:5, #### loss:0.1789149840672811 #####accuracy:1.0
    epoch:6, #### loss:0.17410411685705185 #####accuracy:1.0
    epoch:7, #### loss:0.16908332953850427 #####accuracy:1.0
    epoch:8, #### loss:0.16382796317338943 #####accuracy:1.0
    epoch:9, #### loss:0.15835540741682053 #####accuracy:1.0
    epoch:10, #### loss:0.15273457020521164 #####accuracy:1.0
    epoch:11, #### loss:0.14708336691061655 #####accuracy:1.0
    epoch:12, #### loss:0.14155150949954987 #####accuracy:1.0
    epoch:13, #### loss:0.1362930883963903 #####accuracy:1.0
    epoch:14, #### loss:0.1314386005202929 #####accuracy:1.0
    epoch:15, #### loss:0.12707658857107162 #####accuracy:1.0
    epoch:16, #### loss:0.123248390853405 #####accuracy:1.0
    epoch:17, #### loss:0.11995399743318558 #####accuracy:1.0
    epoch:18, #### loss:0.1171633576353391 #####accuracy:1.0
    epoch:19, #### loss:0.11482855677604675 #####accuracy:1.0
    [0.3412148654]
    test:--------------->loss:QTensor(0.3412148654, requires_grad=True) #####accuracy:1.0

Model accuracy on the test data:

.. image:: ./images/qvc_accuracy.png
   :width: 600 px
   :align: center

|

2. Data re-uploading model
==================================
In a neural network, each neuron receives information from all neurons in the previous layer (Figure a). In contrast, a single-qubit quantum classifier receives the output of the previous information-processing unit and the input (Figure b).
Put simply, in a conventional quantum circuit, once data loading is complete, the result can be obtained directly through a few unitary transformations :math:`U(\theta_1,\theta_2,\theta_3)`.
In Quantum Data Re-uploading (QDRL) tasks, however, the data must be re-uploaded before the unitary transformations.

                                            .. centered:: Comparison of QDRL and a classical neural network

.. image:: ./images/qdrl.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Parameterized quantum circuit for Quantum Data Re-upLoading
    """

    import sys
    sys.path.insert(0, "../")
    import numpy as np
    from pyvqnet.nn.linear import Linear
    from pyvqnet.qnn.qdrl.vqnet_model import vmodel
    from pyvqnet.optim import sgd
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.nn.module import Module

    from pyvqnet.data import data_generator as get_minibatch_data

    np.random.seed(42)

    num_layers = 3
    params = np.random.uniform(size=(num_layers, 3))


    class Model(Module):
        def __init__(self):

            super(Model, self).__init__()
            self.pqc = vmodel(params.shape)
            self.fc2 = Linear(2, 2)

        def forward(self, x):
            x = self.pqc(x)
            return x


    def circle(samples: int, reps=np.sqrt(1 / 2)):
        data_x, data_y = [], []
        for _ in range(samples):
            x = np.random.rand(2)
            y = [0, 1]
            if np.linalg.norm(x) < reps:
                y = [1, 0]
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)


    def get_score(pred, label):
        pred, label = np.array(pred.data), np.array(label.data)
        score = np.sum(np.argmax(pred, axis=1) == np.argmax(label, 1))
        return score


    model = Model()
    optimizer = sgd.SGD(model.parameters(), lr=1)


    def train():
        """
        Main function for train qdrl model
        """
        batch_size = 5
        model.train()
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))  # 500*3

        epoch = 10
        print("start training...........")
        for i in range(epoch):
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train, batch_size):
                optimizer.zero_grad()

                data, label = QTensor(data), QTensor(label)

                output = model(data)

                loss_fun = CategoricalCrossEntropy()
                losss = loss_fun(label, output)

                losss.backward()

                optimizer._step()
                accuracy += get_score(output, label)

                loss += losss.item()

                count += batch_size

            print(f"epoch:{i}, train_accuracy_for_each_batch:{accuracy/count}")
            print(f"epoch:{i}, train_loss_for_each_batch:{loss/count}")


    def test():
        batch_size = 5
        model.eval()
        print("start eval...................")
        x_test, y_test = circle(500)
        test_accuracy = 0
        count = 0
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

        for test_data, test_label in get_minibatch_data(x_test, y_test,
                                                        batch_size):

            test_data, test_label = QTensor(test_data), QTensor(test_label)
            output = model(test_data)
            test_accuracy += get_score(output, test_label)
            count += batch_size
        print(f"test_accuracy:{test_accuracy/count}")


    if __name__ == "__main__":
        train()
        test()

QDRL accuracy on the test data:

.. image:: ./images/qdrl_accuracy.png
   :width: 600 px
   :align: center

|

3. VSQL: Variational Shadow Quantum Learning for Classification model
=======================================================================
A variational quantum circuit is used to build a two-class classification model. Compared with a neural network of similar parameter precision, the classification accuracies are close, while the quantum circuit has far fewer parameters.
The algorithm reproduces `Variational Shadow Quantum Learning for Classification Model <https://arxiv.org/abs/2012.08288>`_.

The overall VSQL quantum model is shown below:

.. image:: ./images/vsql_model.PNG
   :width: 600 px
   :align: center

|

Local quantum circuits on each qubit in VSQL are shown below:

.. image:: ./images/vsql_0.png
.. image:: ./images/vsql_1.png
.. image:: ./images/vsql_2.png
.. image:: ./images/vsql_3.png
.. image:: ./images/vsql_4.png
.. image:: ./images/vsql_5.png
.. image:: ./images/vsql_6.png
.. image:: ./images/vsql_7.png
.. image:: ./images/vsql_8.png

|

.. code-block::

    """
    Parameterized quantum circuit for VSQL

    """

        import sys
        sys.path.insert(0, "../")
        import os
        import os.path
        import struct
        import gzip
        from pyvqnet.nn.module import Module
        from pyvqnet.nn.loss import CategoricalCrossEntropy
        from pyvqnet.optim.adam import Adam
        from pyvqnet.data.data import data_generator
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.pq3.measure import expval
        from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
        from pyvqnet.qnn.pq3.template import AmplitudeEmbeddingCircuit
        from pyvqnet.nn.linear import Linear
        import numpy as np
        import pyqpanda3.core as pq
        import matplotlib.pyplot as plt
        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except:
            print("Can not use matplot TkAgg")
            pass

        try:
            import urllib.request
        except ImportError:
            raise ImportError("You should use Python 3.x")

        url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        key_file = {
            "train_img": "train-images-idx3-ubyte.gz",
            "train_label": "train-labels-idx1-ubyte.gz",
            "test_img": "t10k-images-idx3-ubyte.gz",
            "test_label": "t10k-labels-idx1-ubyte.gz"
        }


        def _download(dataset_dir, file_name):
            """
            Download function for mnist dataset file
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


        if not os.path.exists("./result"):
            os.makedirs("./result")
        else:
            pass


        def circuits_of_vsql(x, weights):
            """
            VSQL model of quantum circuits
            """
            qlist = range(n)
            machine = pq.CPUQVM()
            weights = weights.reshape([depth + 1, 3, n_qsc])

            def subcir(weights, qlist, depth, n_qsc, n_start):
                cir = pq.QCircuit()

                for i in range(n_qsc):
                    cir << pq.RX(qlist[n_start + i], weights[0][0][i])
                    cir << pq.RY(qlist[n_start + i], weights[0][1][i])
                    cir << pq.RX(qlist[n_start + i], weights[0][2][i])
                for repeat in range(1, depth + 1):
                    for i in range(n_qsc - 1):
                        cir << pq.CNOT(qlist[n_start + i], qlist[n_start + i + 1])
                    cir << pq.CNOT(qlist[n_start + n_qsc - 1], qlist[n_start])
                    for i in range(n_qsc):
                        cir << pq.RY(qlist[n_start + i], weights[repeat][1][i])

                return cir

            def get_pauli_str(n_start, n_qsc):
                pauli_str = " ".join("X" + str(i)
                                    for i in range(n_start, n_start + n_qsc))
                return {pauli_str: 1.0}

            f_i = []
            origin_in = AmplitudeEmbeddingCircuit(x, qlist)
            for st in range(n - n_qsc + 1):
                psd = get_pauli_str(st, n_qsc)
                cir = pq.QCircuit()
                cir << origin_in
                cir << subcir(weights, qlist, depth, n_qsc, st)
                prog = pq.QProg()
                prog << cir

                f_ij = expval(machine, prog, psd)
                f_i.append(f_ij)
            f_i = np.array(f_i)
            return f_i


        #GLOBAL VAR
        n = 10
        n_qsc = 2
        depth = 1


        class QModel(Module):
            """
            Model of VSQL
            """
            def __init__(self):
                super().__init__()
                self.vq = QuantumLayer(circuits_of_vsql, (depth + 1) * 3 * n_qsc)
                self.fc = Linear(n - n_qsc + 1, 2)

            def forward(self, x):
                x = self.vq(x)
                x = self.fc(x)

                return x


        class Model(Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(input_channels=28 * 28, output_channels=2)

            def forward(self, x):

                x = tensor.flatten(x, 1)
                x = self.fc1(x)
                return x


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
            images = np.zeros((num, rows, cols), dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels


        def run_vsql():
            """
            VQSL MODEL
            """
            digits = [0, 1]
            x_train, y_train = load_mnist("training_data", digits)
            x_train = x_train / 255
            y_train = y_train.reshape(-1, 1)
            y_train = np.eye(len(digits))[y_train].reshape(-1, len(digits)).astype(
                np.int64)
            x_test, y_test = load_mnist("testing_data", digits)
            x_test = x_test / 255
            y_test = y_test.reshape(-1, 1)
            y_test = np.eye(len(digits))[y_test].reshape(-1,
                                                        len(digits)).astype(np.int64)

            x_train_list = []
            x_test_list = []
            for i in range(x_train.shape[0]):
                x_train_list.append(
                    np.pad(x_train[i, :, :].flatten(), (0, 240),
                        constant_values=(0, 0)))
            x_train = np.array(x_train_list)

            for i in range(x_test.shape[0]):
                x_test_list.append(
                    np.pad(x_test[i, :, :].flatten(), (0, 240),
                        constant_values=(0, 0)))

            x_test = np.array(x_test_list)

            x_train = x_train[:500]
            y_train = y_train[:500]

            x_test = x_test[:100]
            y_test = y_test[:100]
            print("model start")
            model = QModel()

            optimizer = Adam(model.parameters(), lr=0.1)

            model.train()
            result_file = open("./result/vqslrlt.txt", "w")
            for epoch in range(1, 3):

                model.train()
                full_loss = 0
                n_loss = 0
                n_eval = 0
                batch_size = 1
                correct = 0
                for x, y in data_generator(x_train,
                                        y_train,
                                        batch_size=batch_size,
                                        shuffle=True):
                    optimizer.zero_grad()
                    try:
                        x = x.reshape(batch_size, 1024)
                    except:
                        x = x.reshape(-1, 1024)

                    output = model(x)
                    cceloss = CategoricalCrossEntropy()
                    loss = cceloss(y, output)
                    loss.backward()
                    optimizer._step()

                    full_loss += loss.item()
                    n_loss += batch_size
                    np_output = np.array(output.data, copy=False)
                    mask = np_output.argmax(1) == y.argmax(1)
                    correct += sum(mask)
                    print(f" n_loss {n_loss} Train Accuracy: {correct/n_loss} ")
                print(f"Train Accuracy: {correct/n_loss} ")
                print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
                result_file.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

                # Evaluation
                model.eval()
                print("eval")
                correct = 0
                full_loss = 0
                n_loss = 0
                n_eval = 0
                batch_size = 1
                for x, y in data_generator(x_test,
                                        y_test,
                                        batch_size=batch_size,
                                        shuffle=True):
                    x = x.reshape(1, 1024)
                    output = model(x)

                    cceloss = CategoricalCrossEntropy()
                    loss = cceloss(y, output)
                    full_loss += loss.item()

                    np_output = np.array(output.data, copy=False)
                    mask = np_output.argmax(1) == y.argmax(1)
                    correct += sum(mask)
                    n_eval += 1
                    n_loss += 1

                print(f"Eval Accuracy: {correct/n_eval}")
                result_file.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")

            result_file.close()
            del model
            print("\ndone vqsl\n")


        if __name__ == "__main__":

            run_vsql()


VSQL accuracy on the test data:

.. image:: ./images/vsql_cacc.PNG
   :width: 600 px
   :align: center

.. image:: ./images/vsql_closs.PNG
   :width: 600 px
   :align: center

.. image:: ./images/vsql_qacc.PNG
   :width: 600 px
   :align: center

.. image:: ./images/vsql_qloss.PNG
   :width: 600 px
   :align: center

|

4. Image classification with Quanvolution
=========================================

In this example, we implement a quantum convolutional neural network, a method first introduced in `Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits <https://arxiv.org/abs/1904.04767>`_.

Like classical convolution, Quanvolution has the following steps:
A small patch of the input image, in our case a 2×2 square of classical data, is embedded into a quantum circuit.
In this example, this is done by applying parameterized rotation gates to qubits initialized in the ground state. The kernel here is a variational circuit generated by the random circuit proposed in the reference.
Finally, the quantum system is measured to obtain a list of classical expectation values.
As in a classical convolutional layer, each expectation value maps to a different channel of a single output pixel.
Repeating the same procedure at different regions scans the full input image and produces an output object, which is assembled into a multi-channel image.
For classification, this example feeds the Quanvolution measurements into a classical fully connected layer ``Linear``.
The main difference from classical convolution is that Quanvolution can generate highly complex kernels whose computation is, at least in principle, classically intractable.

.. image:: ./images/quanvo.png
   :width: 600 px
   :align: center

|


MNIST dataset definition

.. code-block::

    import os
    import os.path
    import struct
    import gzip
    import sys
    sys.path.insert(0, "../")
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import NLL_Loss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.pq3.measure import expval
    from pyvqnet.nn.linear import Linear
    import numpy as np
    from pyvqnet.qnn.qcnn import Quanvolution
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


    def _download(dataset_dir, file_name):
        """
        Download function for mnist dataset file
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


    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass


    def load_mnist(dataset="training_data", digits=np.arange(10), path="./"):
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

Model and run function definition

.. code-block::

    class QModel(Module):

        def __init__(self):
            super().__init__()
            self.vq = Quanvolution([4, 2], (2, 2))
            self.fc = Linear(4 * 14 * 14, 10)

        def forward(self, x):
            x = self.vq(x)
            x = tensor.flatten(x, 1)
            x = self.fc(x)
            x = tensor.log_softmax(x)
            return x



    def run_quanvolution():

        digit = 10
        x_train, y_train = load_mnist("training_data", digits=np.arange(digit))
        x_train = x_train / 255

        y_train = y_train.flatten()

        x_test, y_test = load_mnist("testing_data", digits=np.arange(digit))

        x_test = x_test / 255
        y_test = y_test.flatten()

        x_train = x_train[:500]
        y_train = y_train[:500]

        x_test = x_test[:100]
        y_test = y_test[:100]

        print("model start")
        model = QModel()

        optimizer = Adam(model.parameters(), lr=5e-3)

        model.train()
        result_file = open("quanvolution.txt", "w")

        cceloss = NLL_Loss()
        N_EPOCH = 15

        for epoch in range(1, N_EPOCH):

            model.train()
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 10
            correct = 0
            for x, y in data_generator(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    shuffle=True):
                optimizer.zero_grad()
                try:
                    x = x.reshape(batch_size, 1, 28, 28)
                except:
                    x = x.reshape(-1, 1, 28, 28)

                output = model(x)

                loss = cceloss(y, output)
                print(f"loss {loss}")
                loss.backward()
                optimizer._step()

                full_loss += loss.item()
                n_loss += batch_size
                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y

                correct += sum(mask)
                print(f"correct {correct}")
            print(f"Train Accuracy: {correct/n_loss}%")
            print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
            result_file.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

            # Evaluation
            model.eval()
            print("eval")
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test,
                                    y_test,
                                    batch_size=batch_size,
                                    shuffle=True):
                x = x.reshape(-1, 1, 28, 28)
                output = model(x)

                loss = cceloss(y, output)
                full_loss += loss.item()

                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y
                correct += sum(mask)
                n_eval += 1
                n_loss += 1

            print(f"Eval Accuracy: {correct/n_eval}")
            result_file.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")

        result_file.close()
        del model
        print("\ndone\n")


    if __name__ == "__main__":

        run_quanvolution()

Training and validation loss and classification accuracy versus epoch.

.. code-block::

    # epoch train_loss      train_accuracy eval_loss    eval_accuracy
    # 1	0.2488900272846222	0.232	1.7297331787645818	0.39
    # 2	0.12281704187393189	0.646	1.201728610806167	0.61
    # 3	0.08001763761043548	0.772	0.8947569639235735	0.73
    # 4	0.06211201059818268	0.83	0.777864265316166	0.74
    # 5	0.052190632969141004	0.858	0.7291000287979841	0.76
    # 6	0.04542196464538574	0.87	0.6764470228599384	0.8
    # 7	0.04029472427070141	0.896	0.6153804161818698	0.79
    # 8	0.03600500610470772	0.902	0.5644993982824963	0.81
    # 9	0.03230033944547176	0.916	0.528938240573043	0.81
    # 10	0.02912954458594322	0.93	0.5058713140769396	0.83
    # 11	0.026443827204406262	0.936	0.49064547760412097	0.83
    # 12	0.024144304402172564	0.942	0.4800815625616815	0.82
    # 13	0.022141477409750223	0.952	0.4724775951183983	0.83
    # 14	0.020372112181037665	0.956	0.46692863543197743	0.83

Quantum autoencoder model
******************************

1. Quantum autoencoder
======================

A classical autoencoder is a neural network that learns an efficient low-dimensional representation of data in a high-dimensional space. Given an input x, the autoencoder maps x to a low-dimensional point y from which x can be recovered.
The structure of the underlying autoencoder network can be chosen to represent data in fewer dimensions, effectively compressing the input. Inspired by this idea, the quantum autoencoder model performs a similar task on quantum data.
A quantum autoencoder is trained to compress a specific dataset of quantum states that classical compression algorithms cannot use. Its parameters are trained with classical optimization algorithms.
We present a simple programmable circuit that can be trained into an efficient autoencoder. We apply the model in a quantum simulation setting to compress the ground states of the Hubbard model and molecular Hamiltonians.
This example is based on `Quantum autoencoders for efficient compression of quantum data <https://arxiv.org/pdf/1612.02806.pdf>`_ .

QAE quantum circuit:

.. image:: ./images/QAE_Quantum_Cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Quantum AutoEncoder demo



    """

    import os
    import sys
    sys.path.insert(0,'../')
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import  fidelityLoss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.qnn.qae.qae import QAElayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

    url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    def _download(dataset_dir,file_name):
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
                    file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
        print("Done")

    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir,v)


    class Model(Module):

        def __init__(self, trash_num: int = 2, total_num: int = 7):
            super().__init__()
            self.pqc = QAElayer(trash_num, total_num)

        def forward(self, x):

            x = self.pqc(x)
            return x

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # download data
        import os, struct
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
        elif dataset == "testing_data":
            fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))

        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)
        images = np.zeros((N, rows, cols))
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def run2():
        ##load dataset

        x_train, y_train = load_mnist("training_data")                      # download training data
        x_train = x_train / 255                                             # normalize data to [0,1]

        x_test, y_test = load_mnist("testing_data")

        x_test = x_test / 255

        x_train = x_train.reshape([-1, 1, 28, 28])
        x_test = x_test.reshape([-1, 1, 28, 28])
        x_train = x_train[:100, :, :, :]
        x_train = np.resize(x_train, [x_train.shape[0], 1, 2, 2])

        x_test = x_test[:10, :, :, :]
        x_test = np.resize(x_test, [x_test.shape[0], 1, 2, 2])
        encode_qubits = 4
        latent_qubits = 2
        trash_qubits = encode_qubits - latent_qubits
        total_qubits = 1 + trash_qubits + encode_qubits
        print("model start")
        model = Model(trash_qubits, total_qubits)

        optimizer = Adam(model.parameters(), lr=0.005)
        model.train()
        F1 = open("rlt.txt", "w")
        loss_list = []
        loss_list_test = []
        fidelity_train = []
        fidelity_val = []

        for epoch in range(1, 10):
            running_fidelity_train = 0
            running_fidelity_val = 0
            print(f"epoch {epoch}")
            model.train()
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            correct = 0
            iter = 0
            if epoch %5 ==1:
                optimizer.lr  = optimizer.lr *0.5
            for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True): #shuffle batch rather than data

                x = x.reshape((-1, encode_qubits))
                x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]), x), 1)
                optimizer.zero_grad()
                output = model(x)
                iter += 1
                np_out = np.array(output.data)
                floss = fidelityLoss()
                loss = floss(output)
                loss_data = np.array(loss.data)
                loss.backward()

                running_fidelity_train += np_out[0]
                optimizer._step()
                full_loss += loss_data[0]
                n_loss += batch_size
                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y.argmax(1)

                correct += sum(mask)

            loss_output = full_loss / n_loss
            print(f"Epoch: {epoch}, Loss: {loss_output}")
            loss_list.append(loss_output)


            # Evaluation
            model.eval()
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
                x = x.reshape((-1, encode_qubits))
                x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]),x),1)
                output = model(x)

                floss = fidelityLoss()
                loss = floss(output)
                loss_data = np.array(loss.data)
                full_loss += loss_data[0]
                running_fidelity_val += np.array(output.data)[0]

                n_eval += 1
                n_loss += 1

            loss_output = full_loss / n_loss
            print(f"Epoch: {epoch}, Loss: {loss_output}")
            loss_list_test.append(loss_output)

            fidelity_train.append(running_fidelity_train / 64)
            fidelity_val.append(running_fidelity_val / 64)

        figure_path = os.path.join(os.getcwd(), 'QAE-rate1.png')
        plt.plot(loss_list, color="blue", label="train")
        plt.plot(loss_list_test, color="red", label="validation")
        plt.title('QAE')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(figure_path)
        plt.show()

        F1.write(f"done\n")
        F1.close()
        del model

    if __name__ == '__main__':
        run2()

QAE error from running the code above. The loss is 1/fidelity, and a value approaching 1 means the fidelity is close to 1.

.. image:: ./images/qae_train_loss.png
   :width: 600 px
   :align: center

|

Quantum circuit structure learning
**********************************
1. Quantum circuit structure learning
=====================================
In quantum circuits, the most commonly used parameterized gates are RZ, RY, and RX, but choosing which gate to use in which situation is a question worth studying. One approach is random selection, which is unlikely to give the best result.
The core goal of the quantum circuit structure learning task is to find the optimal combination of parameterized quantum gates.
Here, this means finding the set of gates that minimizes the loss function.

.. code-block::

    """
    Quantum Circuits Strcture Learning Demo

    """

    import sys
    sys.path.insert(0,"../")

    import copy
    import pyqpanda3.core as pq
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn.pq3.measure import expval
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    machine = pq.CPUQVM()
    nqbits = range(2)

    def gen(param, generators, qbits, circuit):
        if generators == "X":
            circuit << pq.RX(qbits, param)
        elif generators == "Y":
            circuit << pq.RY(qbits, param)
        else:
            circuit << pq.RZ(qbits, param)

    def circuits(params, generators, circuit):
        gen(params[0], generators[0], nqbits[0], circuit)
        gen(params[1], generators[1], nqbits[1], circuit)
        circuit << pq.CNOT(nqbits[0], nqbits[1])
        prog = pq.QProg()
        prog << circuit
        return prog

    def ansatz1(params: QTensor, generators):
        circuit = pq.QCircuit()
        params = params.to_numpy()
        prog = circuits(params, generators, circuit)
        return expval(machine, prog, {"Z0": 1}), expval(machine, prog, {"Y1": 1})


    def ansatz2(params: QTensor, generators):
        circuit = pq.QCircuit()
        params = params.to_numpy()
        prog = circuits(params, generators, circuit)
        return expval(machine, prog, {"X0": 1})


    def loss(params, generators):
        z, y = ansatz1(params, generators)
        x = ansatz2(params, generators)
        return 0.5 * y + 0.8 * z - 0.2 * x


    def rotosolve(d, params, generators, cost, M_0):
        """
        rotosolve algorithm implementation
        """
        params[d] = np.pi / 2.0
        m0_plus = cost(QTensor(params), generators)
        params[d] = -np.pi / 2.0
        m0_minus = cost(QTensor(params), generators)
        a = np.arctan2(2.0 * M_0 - m0_plus - m0_minus,
                    m0_plus - m0_minus)  # returns value in (-pi,pi]
        params[d] = -np.pi / 2.0 - a
        if params[d] <= -np.pi:
            params[d] += 2 * np.pi
        return cost(QTensor(params), generators)


    def optimal_theta_and_gen_helper(index, params, generators):
        """
        find optimal varaibles
        """
        params[index] = 0.
        m0 = loss(QTensor(params), generators)  #init value
        for kind in ["X", "Y", "Z"]:
            generators[index] = kind
            params_cost = rotosolve(index, params, generators, loss, m0)
            if kind == "X" or params_cost <= params_opt_cost:
                params_opt_d = params[index]
                params_opt_cost = params_cost
                generators_opt_d = kind
        return params_opt_d, generators_opt_d


    def rotoselect_cycle(params: np, generators):
        for index in range(params.shape[0]):
            params[index], generators[index] = optimal_theta_and_gen_helper(
                index, params, generators)
        return params, generators


    params = QTensor(np.array([0.3, 0.25]))
    params = params.to_numpy()
    generator = ["X", "Y"]
    generators = copy.deepcopy(generator)
    epoch = 20
    state_save = []
    for i in range(epoch):
        state_save.append(loss(QTensor(params), generators))
        params, generators = rotoselect_cycle(params, generators)

    print("Optimal generators are: {}".format(generators))
    print("Optimal params are: {}".format(params))
    steps = np.arange(0, epoch)


    plt.plot(steps, state_save, "o-")
    plt.title("rotoselect")
    plt.xlabel("cycles")
    plt.ylabel("cost")
    plt.yticks(np.arange(-1.25, 0.80, 0.25))
    plt.tight_layout()
    plt.show()


The quantum circuit structure obtained by running the code above: one :math:`RX` and one :math:`RY`.

.. image:: ./images/final_quantum_circuit.png
   :width: 600 px
   :align: center

And the loss function for different values of the gate parameters :math:`\theta_1` and :math:`\theta_2`.

.. image:: ./images/loss3d.png
   :width: 600 px
   :align: center

|

Hybrid quantum-classical neural network models
**********************************************

1. Hybrid quantum-classical neural network model
================================================

Machine learning (ML) has become a successful interdisciplinary field that aims to extract generalizable information from data mathematically. Quantum machine learning seeks to use the principles of quantum mechanics to enhance machine learning, and vice versa.
Whether your goal is to enhance classical ML algorithms by outsourcing hard computations to quantum computers, or to optimize quantum algorithms with classical ML architectures, both fall under quantum machine learning (QML).
In this chapter, we explore how to partially quantize a classical neural network to create a hybrid quantum-classical neural network. Quantum circuits are built from quantum gates, and the quantum computation these gates implement is provably differentiable, as shown in `Quantum Circuit Learning <https://arxiv.org/abs/1803.00745>`_. Researchers therefore try to combine quantum circuits with classical neural network modules and train both together for hybrid quantum-classical machine learning tasks.
We write a simple example that uses VQNet for a neural network training task. The goal is to show how easy VQNet is to use and to encourage ML practitioners to explore quantum computing.

Data preparation
----------------

We use `MNIST datasets <http://yann.lecun.com/exdb/mnist/>`_, the most basic handwritten digit database for neural networks, as classification data.
We first load MNIST and filter samples containing 0 and 1. These samples are split into training_data and testing_data, and each sample has shape 1*784.

.. code-block::

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
    from pyvqnet.tensor import QTensor
    import pyqpanda3.core as pq

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    def _download(dataset_dir,file_name):
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
                    file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
        print("Done")

    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir,v)

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # download data
        import os, struct
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
        elif dataset == "testing_data":
            fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)
        images = np.zeros((N, rows, cols))
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def data_select(train_num, test_num):
        x_train, y_train = load_mnist("training_data")
        x_test, y_test = load_mnist("testing_data")
        # Train Leaving only labels 0 and 1
        idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                        np.where(y_train == 1)[0][:train_num])
        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)
        # Test Leaving only labels 0 and 1
        idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                        np.where(y_test == 1)[0][:test_num])
        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)
        return x_train, y_train, x_test, y_test

    n_samples_show = 6

    x_train, y_train, x_test, y_test = data_select(100, 50)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    for img ,targets in zip(x_test,y_test):
        if n_samples_show <= 3:
            break

        if targets[0] == 1:
            axes[n_samples_show - 1].set_title("Labeled: 0")
            axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
            axes[n_samples_show - 1].set_xticks([])
            axes[n_samples_show - 1].set_yticks([])
            n_samples_show -= 1

    for img ,targets in zip(x_test,y_test):
        if n_samples_show <= 0:
            break

        if targets[0] == 0:
            axes[n_samples_show - 1].set_title("Labeled: 1")
            axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
            axes[n_samples_show - 1].set_xticks([])
            axes[n_samples_show - 1].set_yticks([])
            n_samples_show -= 1

    plt.show()

.. image:: ./images/mnsit_data_examples.png
   :width: 600 px
   :align: center

|

Building the quantum circuit
----------------------------

In this example, we use Origin Quantum's pyqpanda3 to define a simple one-qubit quantum circuit. The circuit takes the output of a classical neural network layer as input, encodes it into quantum data with ``H`` and ``RY`` gates, and computes the Hamiltonian expectation value in the z direction as output.

.. code-block::

    import pyvqnet
    pyvqnet.utils.set_random_seed(142)
    import pyqpanda3.core as pq
    import numpy as np
    def circuit(x ,weights):
        num_qubits = 1
        # Create a pyqpanda3 simulator
        machine = pq.CPUQVM()
        # Allocate qubits with pyqpanda3
        qubits = range(num_qubits)
        # Allocate classical bits for auxiliary measurement with pyqpanda3
        cbits = range(num_qubits)
        # Build the circuit
        circuit = pq.QCircuit()
        circuit << pq.H(qubits[0])
        circuit << pq.RY(qubits[0], x[0])

        prog = pq.QProg()
        prog << circuit
        for i in range(num_qubits):
            prog << pq.measure(qubits[i], cbits[i])
        shots = 1000
        # Run the quantum program
        machine.run(prog, shots)
        result = machine.result().get_counts()
        # Fill in both outcomes: count0 or count1 may be missing when all shots collapse to one basis state
        count0 = result.get("0", 0)
        count1 = result.get("1", 0)
        expectation = (count0 - count1) / shots   # ⟨Z⟩ = P(0) - P(1)
        return expectation

.. image:: ./images/hqcnn_quantum_cir.png
   :width: 600 px
   :align: center

|

Building the hybrid quantum neural network
------------------------------------------

Because quantum circuits can be differentiated automatically together with classical neural networks, we can build a model from VQNet's 2D convolution layer ``Conv2D``, pooling layer ``MaxPool2D``, fully connected layer ``Linear``, and the circuit built above.
The following code defines a Net class that inherits from VQNet's automatic differentiation module ``Module`` and wraps the quantum circuit layer with ``QuantumLayerV2``, giving a model that can be differentiated automatically.
The MNIST data goes through convolution, dimensionality reduction, quantum encoding, and measurement to obtain the final features needed for classification.

.. code-block::

    # The forward pass and gradient of the quantum layer are handled automatically by QuantumLayerV2
    from pyvqnet.qnn import QuantumLayerV2

    # Model definition
    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)
            self.hybrid = QuantumLayerV2(circuit, 0)
            self.fc3 = Linear(input_channels=1, output_channels=2)

        def forward(self, x):
            x = F.ReLu()(self.conv1(x))  # 1 6 24 24
            x = self.maxpool1(x)
            x = F.ReLu()(self.conv2(x))  # 1 16 8 8
            x = self.maxpool2(x)
            x = tensor.flatten(x, 1)   # 1 256
            x = F.ReLu()(self.fc1(x))  # 1 64
            x = self.fc2(x)    # 1 1
            x = self.hybrid(x)
            x = self.fc3(x)
            return x

.. image:: ./images/hqcnnmodel.PNG
   :width: 600 px
   :align: center

|

Training and testing
--------------------

The code above defines the model. As with classical neural network training, we still need to instantiate the model, define the loss function and optimizer, and define the full training and test loop.
For the hybrid network shown below, we loop over the input data to compute the loss in the forward pass, automatically compute gradients of all trainable parameters in the backward pass, and update them with the optimizer until the preset number of iterations is reached.

.. image:: ./images/hqcnnarch.PNG
   :width: 600 px
   :align: center

|

.. code-block::

    x_train, y_train, x_test, y_test = data_select(1000, 100)
    # Instantiate the model
    model = Net()
    # Adam is enough for this task; model.parameters() passes the parameters the model needs to train.
    optimizer = Adam(model.parameters(), lr=0.005)
    # Use the cross-entropy loss for classification
    loss_func = CategoricalCrossEntropy()

    # Number of training epochs
    epochs = 10
    train_loss_list = []
    val_loss_list = []
    train_acc_list =[]
    val_acc_list = []


    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        batch_size = 250
        correct = 0
        n_train = 0
        for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True):

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

Data visualization
------------------

Curves of the loss function and accuracy on the training and test data.

.. code-block::

    import os
    plt.figure()
    xrange = range(1,len(train_loss_list)+1)
    figure_path = os.path.join(os.getcwd(), 'HQCNN LOSS.png')
    plt.plot(xrange,train_loss_list, color="blue", label="train")
    plt.plot(xrange,val_loss_list, color="red", label="validation")
    plt.title('HQCNN')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(np.arange(1, epochs +1,step = 2))
    plt.legend(loc="upper right")
    plt.savefig(figure_path)
    plt.show()

    plt.figure()
    figure_path = os.path.join(os.getcwd(), 'HQCNN Accuracy.png')
    plt.plot(xrange,train_acc_list, color="blue", label="train")
    plt.plot(xrange,val_acc_list, color="red", label="validation")
    plt.title('HQCNN')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, epochs +1,step = 2))
    plt.legend(loc="lower right")
    plt.savefig(figure_path)
    plt.show()


.. image:: ./images/HQCNNLOSS.png
   :width: 600 px
   :align: center

.. image:: ./images/HQCNNAccuracy.png
   :width: 600 px
   :align: center

|

.. code-block::

    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        pred = QTensor.argmax(output, [1],False)
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()

.. image:: ./images/eval_test.png
   :width: 600 px
   :align: center

|

2. Hybrid quantum-classical transfer learning model
===================================================

We apply a machine learning method called transfer learning to an image classifier based on a hybrid classical-quantum network. We write a simple example that integrates pyqpanda3 with VQNet.
Transfer learning is a well-established artificial neural network training technique based on the intuition that if a pretrained network is good at a given problem, a little extra training lets it solve a different but related problem.

                                                            .. centered:: Diagram of the quantum part of the circuit

.. image:: ./images/QTransferLearning_cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Quantum Classic Nerual Network Transfer Learning demo

    """

    import os
    import sys
    sys.path.insert(0,'../')
    import numpy as np
    import matplotlib.pyplot as plt

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D
    from pyvqnet.utils.storage import load_parameters, save_parameters

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D

    from pyvqnet.nn.batch_norm import BatchNorm2d
    from pyvqnet.nn.loss import SoftmaxCrossEntropy

    from pyvqnet.optim.sgd import SGD
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda3.core as pq
    import matplotlib
    from pyvqnet.nn.module import *
    from pyvqnet.utils.initializer import *
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.qnn.pq3.measure import expval

    try:
        matplotlib.use('TkAgg')
    except:
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

    url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    def _download(dataset_dir,file_name):
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
                    file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
        print("Done")

    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir,v)


    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass
    # classical CNN
    class CNN(Module):
        def __init__(self):
            super(CNN, self).__init__()

            self.conv1 = Conv2D(input_channels=1, output_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d1 = BatchNorm2d(16)
            self.Relu1 = F.ReLu()

            self.conv2 = Conv2D(input_channels=16, output_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d2 = BatchNorm2d(32)
            self.Relu2 = F.ReLu()
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")

            self.conv3 = Conv2D(input_channels=32, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d3 = BatchNorm2d(64)
            self.Relu3 = F.ReLu()

            self.conv4 = Conv2D(input_channels=64, output_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d4 = BatchNorm2d(128)
            self.Relu4 = F.ReLu()
            self.maxpool4 = MaxPool2D([2, 2], [2, 2], padding="valid")

            self.fc1 = Linear(input_channels=128 * 4 * 4, output_channels=1024)
            self.fc2 = Linear(input_channels=1024, output_channels=128)
            self.fc3 = Linear(input_channels=128, output_channels=10)

        def forward(self, x):

            x = self.Relu1(self.conv1(x))
            x = self.maxpool2(self.Relu2(self.conv2(x)))
            x = self.Relu3(self.conv3(x))
            x = self.maxpool4(self.Relu4(self.conv4(x)))
            x = tensor.flatten(x, 1)
            x = F.ReLu()(self.fc1(x))  # 1 64
            x = F.ReLu()(self.fc2(x))  # 1 64
            x = self.fc3(x)  # 1 1
            return x

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):
        import os, struct
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
        elif dataset == "testing_data":
            fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)
        images = np.zeros((N, rows, cols))
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels


    """
    to get cnn model parameters for transfer learning
    """

    train_size = 10000
    eval_size = 1000
    EPOCHES = 100
    def classcal_cnn_model_making():
        # load train data
        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()

        optimizer = SGD(model.parameters(), lr=0.005)
        loss_func = SoftmaxCrossEntropy()

        epochs = EPOCHES
        loss_list = []
        model.train()

        SAVE_FLAG = True
        temp_loss = 0
        for epoch in range(1, epochs):
            total_loss = []
            for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):

                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Forward pass
                output = model(x)

                # Calculating loss
                loss = loss_func(y, output)  # target output
                loss_np = np.array(loss.data)
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer._step()

                total_loss.append(loss_np)

            loss_list.append(np.sum(total_loss) / len(total_loss))
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

            if SAVE_FLAG:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")
                SAVE_FLAG = False
            else:
                if temp_loss > loss_list[-1]:
                    temp_loss = loss_list[-1]
                    save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")


        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=4, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            loss = loss_func(y, output)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1
        print(f"Eval Accuracy: {correct / n_eval}")

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    def classical_cnn_TransferLearning_predict():
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]
        x_test = x_test / 255
        y_test = np.eye(10)[y_test].reshape(-1, 10)
        model = CNN()

        model_parameter = load_parameters("./result/QCNN_TL_1.model")
        model.load_state_dict(model_parameter)
        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)

            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1

        print(f"Eval Accuracy: {correct / n_eval}")

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    def quantum_cnn_TransferLearning():

        n_qubits = 4  # Number of qubits
        q_depth = 6  # Depth of the quantum circuit (number of variational layers)

        def Q_H_layer(qubits, nqubits):
            """Layer of single-qubit Hadamard gates.
            """
            circuit = pq.QCircuit()
            for idx in range(nqubits):
                circuit << pq.H(qubits[idx])
            return circuit

        def Q_RY_layer(qubits, w):
            """Layer of parametrized qubit rotations around the y axis.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit << pq.RY(qubits[idx], element)
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Layer of CNOTs followed by another shifted layer of CNOT.
            """
            # In other words it should apply something like :
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
                circuit << pq.CNOT(qubits[i], qubits[i + 1])
            for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
                circuit << pq.CNOT(qubits[i], qubits[i + 1])
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat):
            """
            The variational quantum circuit.
            """
            machine = pq.CPUQVM()
            qubits = range(n_qubits)
            circuit = pq.QCircuit()

            # Reshape weights
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            circuit << Q_H_layer(qubits, n_qubits)

            # Embed features in the quantum node
            circuit << Q_RY_layer(qubits, q_input_features)

            # Sequence of trainable variational layers
            for k in range(q_depth):
                circuit << Q_entangling_layer(qubits, n_qubits)
                circuit << Q_RY_layer(qubits, q_weights[k])

            # Expectation values in the Z basis
            prog = pq.QProg()
            prog << circuit

            exp_vals = []
            for position in range(n_qubits):
                pauli_str = "Z" + str(position)
                exp = expval(machine, prog, {pauli_str: 1.0})
                exp_vals.append(exp)

            return exp_vals

        class Q_DressedQuantumNet(Module):

            def __init__(self):
                """
                Definition of the *dressed* layout.
                """

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits)

            def forward(self, input_features):
                """
                Defining how tensors are supposed to move through the *dressed* quantum
                net.
                """

                # obtain the input features for the quantum circuit
                # by reducing the feature dimension from 512 to 4
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.temp_Q(q_in)

                result = q_out_elem
                # return the two-dimensional prediction from the postprocessing layer
                return self.post_net(result)

        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()
        model_param = load_parameters("./result/QCNN_TL_1.model")
        model.load_state_dict(model_param)

        loss_func = SoftmaxCrossEntropy()

        epochs = EPOCHES
        loss_list = []
        eval_losses = []

        model_hybrid = model
        print(model_hybrid)

        for param in model_hybrid.parameters():
            param.requires_grad = False
        model_hybrid.fc3 = Q_DressedQuantumNet()
        optimizer_hybrid = Adam(model_hybrid.fc3.parameters(), lr=0.001)
        model_hybrid.train()

        SAVE_FLAG = True
        temp_loss = 0
        for epoch in range(1, epochs):
            total_loss = []
            for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):
                x = x.reshape(-1, 1, 28, 28)
                optimizer_hybrid.zero_grad()
                # Forward pass
                output = model_hybrid(x)

                loss = loss_func(y, output)  # target output
                loss_np = np.array(loss.data)
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer_hybrid._step()
                total_loss.append(loss_np)

            loss_list.append(np.sum(total_loss) / len(total_loss))
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
            if SAVE_FLAG:
                temp_loss = loss_list[-1]
                save_parameters(model_hybrid.fc3.state_dict(), "./result/QCNN_TL_FC3.model")
                save_parameters(model_hybrid.state_dict(), "./result/QCNN_TL_ALL.model")
                SAVE_FLAG = False
            else:
                if temp_loss > loss_list[-1]:
                    temp_loss = loss_list[-1]
                    save_parameters(model_hybrid.fc3.state_dict(), "./result/QCNN_TL_FC3.model")
                    save_parameters(model_hybrid.state_dict(), "./result/QCNN_TL_ALL.model")

            correct = 0
            n_eval = 0
            loss_temp =[]
            for x1, y1 in data_generator(x_test, y_test, batch_size=4, shuffle=True):
                x1 = x1.reshape(-1, 1, 28, 28)
                output = model_hybrid(x1)
                loss = loss_func(y1, output)
                np_loss = np.array(loss.data)
                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y1.argmax(1))
                correct += np.sum(np.array(mask))
                n_eval += 1
                loss_temp.append(np_loss)
            eval_losses.append(np.sum(loss_temp) / n_eval)
            print("{:.0f} eval loss is : {:.10f}".format(epoch, eval_losses[-1]))


        plt.title('model loss')
        plt.plot(loss_list, color='green', label='train_losses')
        plt.plot(eval_losses, color='red', label='eval_losses')
        plt.ylabel('loss')
        plt.legend(["train_losses", "eval_losses"])
        plt.savefig("qcnn_transfer_learning_classical")
        plt.show()
        plt.close()

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model_hybrid.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model_hybrid(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    def quantum_cnn_TransferLearning_predict():

        n_qubits = 4  # Number of qubits
        q_depth = 6  # Depth of the quantum circuit (number of variational layers)

        def Q_H_layer(qubits, nqubits):
            """Layer of single-qubit Hadamard gates.
            """
            circuit = pq.QCircuit()
            for idx in range(nqubits):
                circuit << pq.H(qubits[idx])
            return circuit

        def Q_RY_layer(qubits, w):
            """Layer of parametrized qubit rotations around the y axis.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit << pq.RY(qubits[idx], element)
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Layer of CNOTs followed by another shifted layer of CNOT.
            """
            # In other words it should apply something like :
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
                circuit << pq.CNOT(qubits[i], qubits[i + 1])
            for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
                circuit << pq.CNOT(qubits[i], qubits[i + 1])
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat):
            """
            The variational quantum circuit.
            """
            machine = pq.CPUQVM()
            qubits = range(n_qubits)
            circuit = pq.QCircuit()

            # Reshape weights
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            circuit << Q_H_layer(qubits, n_qubits)

            # Embed features in the quantum node
            circuit << Q_RY_layer(qubits, q_input_features)

            # Sequence of trainable variational layers
            for k in range(q_depth):
                circuit << Q_entangling_layer(qubits, n_qubits)
                circuit << Q_RY_layer(qubits, q_weights[k])

            # Expectation values in the Z basis
            prog = pq.QProg()
            prog << circuit
            exp_vals = []
            for position in range(n_qubits):
                pauli_str = "Z" + str(position)
                exp = expval(machine, prog, {pauli_str: 1.0})
                exp_vals.append(exp)

            return exp_vals

        class Q_DressedQuantumNet(Module):

            def __init__(self):
                """
                Definition of the *dressed* layout.
                """

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits)

            def forward(self, input_features):
                """
                Defining how tensors are supposed to move through the *dressed* quantum
                net.
                """

                # obtain the input features for the quantum circuit
                # by reducing the feature dimension from 512 to 4
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.temp_Q(q_in)

                result = q_out_elem
                # return the two-dimensional prediction from the postprocessing layer
                return self.post_net(result)

        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
        x_train = x_train[:2000]
        y_train = y_train[:2000]
        x_test = x_test[:500]
        y_test = y_test[:500]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()
        model_hybrid = model
        model_hybrid.fc3 = Q_DressedQuantumNet()
        for param in model_hybrid.parameters():
            param.requires_grad = False
        model_param_quantum = load_parameters("./result/QCNN_TL_ALL.model")

        model_hybrid.load_state_dict(model_param_quantum)
        model_hybrid.eval()

        loss_func = SoftmaxCrossEntropy()
        eval_losses = []

        correct = 0
        n_eval = 0
        loss_temp =[]
        eval_batch_size = 4
        for x1, y1 in data_generator(x_test, y_test, batch_size=eval_batch_size, shuffle=True):
            x1 = x1.reshape(-1, 1, 28, 28)
            output = model_hybrid(x1)
            loss = loss_func(y1, output)
            np_loss = np.array(loss.data)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y1.argmax(1))
            correct += np.sum(np.array(mask))

            n_eval += 1
            loss_temp.append(np_loss)

        eval_losses.append(np.sum(loss_temp) / n_eval)
        print(f"Eval Accuracy: {correct / (eval_batch_size*n_eval)}")

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model_hybrid.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model_hybrid(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    if __name__ == "__main__":
        # save classic model parameters
        if not os.path.exists('./result/QCNN_TL_1.model'):
            classcal_cnn_model_making()
            classical_cnn_TransferLearning_predict()
        #train quantum circuits.
        print("use exist cnn model param to train quantum parameters.")
        quantum_cnn_TransferLearning()
        #eval quantum circuits.
        quantum_cnn_TransferLearning_predict()



Loss on the training set

.. image:: ./images/qcnn_transfer_learning_classical.png
   :width: 600 px
   :align: center

|

Classification results on the test set

.. image:: ./images/qcnn_transfer_learning_predict.png
   :width: 600 px
   :align: center

|

3. Hybrid quantum-classical QUnet network model
===============================================

Image segmentation is a classic problem in computer vision research and a focus of interest in image understanding. It is the first step in image analysis, a foundation of computer vision,
an important part of image understanding, and one of the hardest problems in image processing. Segmentation partitions an image
into disjoint regions based on features such as grayscale, color, spatial texture, and geometric shape, so that these features are consistent or similar within a region and clearly different between regions.
Put simply, given an image, it classifies every pixel and separates pixel regions belonging to different objects. `Unet <https://arxiv.org/abs/1505.04597>`_ is an algorithm for classical image segmentation.
Here we explore how to partially quantize a classical neural network to create a `QUnet - Quantum Unet` neural network suited to quantum data. We write a simple example that integrates pyqpanda3 with VQNet.
QUnet is mainly used for image segmentation.


Data preparation
-------------------
We use data from the official VOCdevkit/VOC2012 library, `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>`_, as image segmentation data.
The samples are split into training_data and testing_data, and the folders contain images and labels.

.. image:: ./images/Unet_data_imshow.png
   :width: 600 px
   :align: center

|

Building the quantum circuit
----------------------------
In this example, we use Origin Quantum's pyqpanda3 to define a quantum circuit. The input 3-channel color images are compressed to single-channel grayscale images and stored,
then quantum convolution extracts and reduces the features.

.. image:: ./images/qunet_cir.png
   :width: 600 px
   :align: center

|

Import required libraries and functions

.. code-block::

    import os
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.conv import Conv2D, ConvT2D
    from pyvqnet.nn import activation as F
    from pyvqnet.nn.batch_norm import BatchNorm2d
    from pyvqnet.nn.loss import BinaryCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.dtype import *
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda3.core as pq
    from pyvqnet.utils.storage import load_parameters, save_parameters
    from pyvqnet.qnn.pq3.measure import probs_measure

    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt

    import cv2

Preprocess data

.. code-block::

    #Preprocess data
    class PreprocessingData:
        def __init__(self, path):
            self.path = path
            self.x_data = []
            self.y_label = []


        def processing(self):
            list_path = os.listdir((self.path+"/images"))
            for i in range(len(list_path)):

                temp_data = cv2.imread(self.path+"/images" + '/' + list_path[i], cv2.IMREAD_COLOR)
                temp_data = cv2.resize(temp_data, (128, 128))
                grayimg = cv2.cvtColor(temp_data, cv2.COLOR_BGR2GRAY)
                temp_data = grayimg.reshape(temp_data.shape[0], temp_data.shape[0], 1).astype(np.float32)
                self.x_data.append(temp_data)

                label_data = cv2.imread(self.path+"/labels" + '/' +list_path[i].split(".")[0] + ".png", cv2.IMREAD_COLOR)
                label_data = cv2.resize(label_data, (128, 128))

                label_data = cv2.cvtColor(label_data, cv2.COLOR_BGR2GRAY)
                label_data = label_data.reshape(label_data.shape[0], label_data.shape[0], 1).astype(np.int64)
                self.y_label.append(label_data)

            return self.x_data, self.y_label

        def read(self):
            self.x_data, self.y_label = self.processing()
            x_data = np.array(self.x_data)
            y_label = np.array(self.y_label)

            return x_data, y_label

    #Quantum encoding circuit
    class QCNN_:
        def __init__(self, image):
            self.image = image

        def encode_cir(self, qlist, pixels):
            cir = pq.QCircuit()
            for i, pix in enumerate(pixels):
                theta = np.arctan(pix)
                phi = np.arctan(pix**2)
                cir << pq.RY(qlist[i], theta)
                cir << pq.RZ(qlist[i], phi)
            return cir

        def entangle_cir(self, qlist):
            k_size = len(qlist)
            cir = pq.QCircuit()
            for i in range(k_size):
                ctr = i
                ctred = i+1
                if ctred == k_size:
                    ctred = 0
                cir << pq.CNOT(qlist[ctr], qlist[ctred])
            return cir

        def qcnn_circuit(self, pixels):
            k_size = len(pixels)
            machine = pq.CPUQVM()
            qlist = range(k_size)
            cir = pq.QProg()

            cir << self.encode_cir(qlist, np.array(pixels) * np.pi / 2)
            cir << self.entangle_cir(qlist)

            result0 = probs_measure(machine, cir, [qlist[0]])
            result1 = probs_measure(machine, cir, [qlist[1]])
            result2 = probs_measure(machine, cir, [qlist[2]])
            result3 = probs_measure(machine, cir, [qlist[3]])

            result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
            return result

    def quanconv_(image):
        """Convolves the input image with many applications of the same quantum circuit."""
        out = np.zeros((64, 64, 1))

        for j in range(0, 128, 2):
            for k in range(0, 128, 2):
                # Process a squared 2x2 region of the image with a quantum circuit
                q_results = QCNN_(image).qcnn_circuit(
                    [
                        image[j, k, 0],
                        image[j, k + 1, 0],
                        image[j + 1, k, 0],
                        image[j + 1, k + 1, 0]
                    ]
                )

                for c in range(1):
                    out[j // 2, k // 2, c] = q_results[c]
        return out

    def quantum_data_preprocessing(images):
        quantum_images = []
        for _, img in enumerate(images):
            quantum_images.append(quanconv_(img))
        quantum_images = np.asarray(quantum_images)
        return quantum_images

Building the hybrid classical-quantum neural network
----------------------------------------------------

Following the Unet framework, we build the classical part with `VQNet`. Downsampling layers reduce dimensionality and extract features;
upsampling layers restore dimensionality; upsampling and downsampling layers are connected by concatenate for feature fusion.

.. image:: ./images/Unet.png
   :width: 600 px
   :align: center

|

.. code-block::

    #Downsampling neural network layer definition
    class DownsampleLayer(Module):
        def __init__(self, in_ch, out_ch):
            super(DownsampleLayer, self).__init__()
            self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_ch)
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_ch)
            self.Relu2 = F.ReLu()
            self.conv3 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                                padding=(1,1))
            self.BatchNorm2d3 = BatchNorm2d(out_ch)
            self.Relu3 = F.ReLu()

        def forward(self, x):
            """
            :param x:
            :return: out(Output to deep),out_2(enter to next level),
            """
            x1 = self.conv1(x)
            x2 = self.BatchNorm2d1(x1)
            x3 = self.Relu1(x2)
            x4 = self.conv2(x3)
            x5 = self.BatchNorm2d2(x4)
            out = self.Relu2(x5)
            x6 = self.conv3(out)
            x7 = self.BatchNorm2d3(x6)
            out_2 = self.Relu3(x7)
            return out, out_2

    #Upsampling neural network layer definition
    class UpSampleLayer(Module):
        def __init__(self, in_ch, out_ch):
            super(UpSampleLayer, self).__init__()

            self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_ch * 2)
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_ch * 2, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_ch * 2)
            self.Relu2 = F.ReLu()

            self.conv3 = ConvT2D(input_channels=out_ch * 2, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                                 padding=(1,1))
            self.BatchNorm2d3 = BatchNorm2d(out_ch)
            self.Relu3 = F.ReLu()

        def forward(self, x):
            '''
            :param x: input conv layer
            :param out: connect with UpsampleLayer
            :return:
            '''
            x = self.conv1(x)
            x = self.BatchNorm2d1(x)
            x = self.Relu1(x)
            x = self.conv2(x)
            x = self.BatchNorm2d2(x)
            x = self.Relu2(x)
            x = self.conv3(x)
            x = self.BatchNorm2d3(x)
            x_out = self.Relu3(x)
            return x_out

    #Overall UNet network architecture
    class UNet(Module):
        def __init__(self):
            super(UNet, self).__init__()
            out_channels = [2 ** (i + 4) for i in range(5)]

            # DownSampleLayer
            self.d1 = DownsampleLayer(1, out_channels[0])  # 3-64
            self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
            self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
            self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
            # UpSampleLayer
            self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
            self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
            self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
            self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
            # output
            self.conv1 = Conv2D(input_channels=out_channels[1], output_channels=out_channels[0], kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_channels[0])
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_channels[0], output_channels=out_channels[0], kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_channels[0])
            self.Relu2 = F.ReLu()
            self.conv3 = Conv2D(input_channels=out_channels[0], output_channels=1, kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.Sigmoid = F.Sigmoid()

        def forward(self, x):
            out_1, out1 = self.d1(x)
            out_2, out2 = self.d2(out1)
            out_3, out3 = self.d3(out2)
            out_4, out4 = self.d4(out3)

            out5 = self.u1(out4)
            out5_pad_out4 = tensor.pad2d(out5, (1, 0, 1, 0), 0)
            cat_out5 = tensor.concatenate([out5_pad_out4, out_4], dim=1)

            out6 = self.u2(cat_out5)
            out6_pad_out_3 = tensor.pad2d(out6, (1, 0, 1, 0), 0)
            cat_out6 = tensor.concatenate([out6_pad_out_3, out_3], dim=1)

            out7 = self.u3(cat_out6)
            out7_pad_out_2 = tensor.pad2d(out7, (1, 0, 1, 0), 0)
            cat_out7 = tensor.concatenate([out7_pad_out_2, out_2], dim=1)

            out8 = self.u4(cat_out7)
            out8_pad_out_1 = tensor.pad2d(out8, (1, 0, 1, 0), 0)
            cat_out8 = tensor.concatenate([out8_pad_out_1, out_1], dim=1)

            out = self.conv1(cat_out8)
            out = self.BatchNorm2d1(out)
            out = self.Relu1(out)
            out = self.conv2(out)
            out = self.BatchNorm2d2(out)
            out = self.Relu2(out)
            out = self.conv3(out)
            out = self.Sigmoid(out)
            return out

Training and model saving
-------------------------

The code above defines the model. As with classical neural network training, we still need to instantiate the model,
define the loss function and optimizer, and define the full training and test loop. For the hybrid network shown below, we loop over the input data to compute the loss in the forward pass,
automatically compute gradients of all trainable parameters in the backward pass, and update them with the optimizer until the preset number of iterations is reached.
Here we select 100 images from the downloaded VOC2012 data as the training set and 10 as the test set. The training set directory is `path0` and the test set directory is `path1`.
Both the images and their label images are preprocessed by the quantum convolution module ``quantum_data_preprocessing``, and the Unet training objective is to make the quantum-preprocessed images and labels as close as possible.
If quantized preprocessing has already been done, set ``PREPROCESS`` to False.

.. code-block::

    PREPROCESS = True

    class MyDataset():
        def __init__(self, x_data, x_label):
            self.x_set = x_data
            self.label = x_label

        def __getitem__(self, item):
            img, target = self.x_set[item], self.label[item]
            img_np = np.uint8(img).transpose(2, 0, 1)
            target_np = np.uint8(target).transpose(2, 0, 1)

            img = img_np
            target = target_np
            return img, target

        def __len__(self):
            return len(self.x_set)

    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass
    if not os.path.exists("./Intermediate_results"):
        os.makedirs("./Intermediate_results")
    else:
        pass

    # prepare train/test data and label
    path0 = 'training_data'
    path1 = 'testing_data'
    train_images, train_labels = PreprocessingData(path0).read()
    test_images, test_labels = PreprocessingData(path1).read()

    print('train: ', train_images.shape, '\ntest: ', test_images.shape)
    print('train: ', train_labels.shape, '\ntest: ', test_labels.shape)
    train_images = train_images / 255
    test_images = test_images / 255

    # use quantum encoder to preprocess data



    if PREPROCESS == True:
        print("Quantum pre-processing of train images:")
        q_train_images = quantum_data_preprocessing(train_images)
        q_test_images = quantum_data_preprocessing(test_images)
        q_train_label = quantum_data_preprocessing(train_labels)
        q_test_label = quantum_data_preprocessing(test_labels)

        # Save pre-processed images
        print('Quantum Data Saving...')
        np.save("./result/q_train.npy", q_train_images)
        np.save("./result/q_test.npy", q_test_images)
        np.save("./result/q_train_label.npy", q_train_label)
        np.save("./result/q_test_label.npy", q_test_label)
        print('Quantum Data Saving Over!')

    # loading quantum data
    SAVE_PATH = "./result/"
    train_x = np.load(SAVE_PATH + "q_train.npy")
    train_labels = np.load(SAVE_PATH + "q_train_label.npy")
    test_x = np.load(SAVE_PATH + "q_test.npy")
    test_labels = np.load(SAVE_PATH + "q_test_label.npy")

    train_x = train_x.astype(np.uint8)
    test_x = test_x.astype(np.uint8)
    train_labels = train_labels.astype(np.uint8)
    test_labels = test_labels.astype(np.uint8)
    train_y = train_labels
    test_y = test_labels

    trainset = MyDataset(train_x, train_y)
    testset = MyDataset(test_x, test_y)
    x_train = []
    y_label = []
    model = UNet()
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_func = BinaryCrossEntropy()
    epochs = 200

    loss_list = []
    SAVE_FLAG = True
    temp_loss = 0
    file = open("./result/result.txt", 'w').close()
    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        for i, (x, y) in enumerate(trainset):
            x_img = QTensor(x, dtype=kfloat32)
            x_img_Qtensor = tensor.unsqueeze(x_img, 0)
            y_img = QTensor(y, dtype=kfloat32)
            y_img_Qtensor = tensor.unsqueeze(y_img, 0)
            optimizer.zero_grad()
            img_out = model(x_img_Qtensor)

            print(f"=========={epoch}==================")
            loss = loss_func(y_img_Qtensor, img_out)  # target output
            if i == 1:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("predict")
                img_out_tensor = tensor.squeeze(img_out, 0)

                if matplotlib.__version__ >= '3.4.2':
                    plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
                else:
                    plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
                plt.subplot(1, 2, 2)
                plt.title("label")
                y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
                if matplotlib.__version__ >= '3.4.2':
                    plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
                else:
                    plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))

                plt.savefig("./Intermediate_results/" + str(epoch) + "_" + str(i) + ".jpg")

            loss_data = np.array(loss.data)
            print("{} - {} loss_data: {}".format(epoch, i, loss_data))
            loss.backward()
            optimizer._step()
            total_loss.append(loss_data)

        loss_list.append(np.sum(total_loss) / len(total_loss))
        out_read = open("./result/result.txt", 'a')
        out_read.write(str(loss_list[-1]))
        out_read.write(str("\n"))
        out_read.close()
        print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
        if SAVE_FLAG:
            temp_loss = loss_list[-1]
            save_parameters(model.state_dict(), "./result/Q-Unet_End.model")
            SAVE_FLAG = False
        else:
            if temp_loss > loss_list[-1]:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "./result/Q-Unet_End.model")


Data visualization
------------------

Save the loss curve plot for the training data and the test data results.

.. code-block::

    out_read = open("./result/result.txt", 'r')
    plt.figure()
    lines_read = out_read.readlines()
    data_read = []
    for line in lines_read:
        float_line = float(line)
        data_read.append(float_line)
    out_read.close()
    plt.plot(data_read)
    plt.title('Unet Training')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.savefig("./result/traing_loss.jpg")

    modela = load_parameters("./result/Q-Unet_End.model")
    print("----------------PREDICT-------------")
    model.load_state_dict(modela)
    model.eval()

    for i, (x1, y1) in enumerate(testset):
        x_img = QTensor(x1, dtype=kfloat32)
        x_img_Qtensor = tensor.unsqueeze(x_img, 0)
        y_img = QTensor(y1, dtype=kfloat32)
        y_img_Qtensor = tensor.unsqueeze(y_img, 0)
        img_out = model(x_img_Qtensor)
        loss = loss_func(y_img_Qtensor, img_out)
        loss_data = np.array(loss.data)
        print("{} loss_eval: {}".format(i, loss_data))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("predict")
        img_out_tensor = tensor.squeeze(img_out, 0)
        if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
        else:
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
        plt.subplot(1, 2, 2)
        plt.title("label")
        y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
        if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
        else:
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))
        plt.savefig("./result/eval_" + str(i) + "_1" + ".jpg")
    print("end!")

Loss on the training set

.. image:: ./images/qunet_train_loss.png
   :width: 600 px
   :align: center

|

Visualization of the results

.. image:: ./images/qunet_eval_1.jpg
   :width: 600 px
   :align: center

.. image:: ./images/qunet_eval_2.jpg
   :width: 600 px
   :align: center

.. image:: ./images/qunet_eval_3.jpg
   :width: 600 px
   :align: center

|



4. Hybrid quantum-classical QMLP network model
==============================================

We introduce and analyze a quantum multilayer perceptron (QMLP) architecture featuring fault-tolerant input embedding, rich nonlinearity, and enhanced variational circuit simulation with parameterized two-qubit entangling gates. See `QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates <https://arxiv.org/pdf/2206.01345.pdf>`_.
We write a simple example that integrates `pyqpanda3` with `VQNet`.



Building the hybrid classical-quantum neural network
----------------------------------------------------

.. code-block::

    import os
    import gzip
    import struct
    import numpy as np
    import matplotlib.pyplot as plt
    import pyqpanda3.core as pq
    import pyvqnet
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import CrossEntropyLoss
    from pyvqnet.optim.adam import Adam

    from pyvqnet.qnn.pq3.measure import expval
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.nn.pooling import AvgPool2D
    from pyvqnet.nn.linear import Linear
    from pyvqnet.data.data import data_generator

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


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

    def load_mnist(dataset="training_data", digits=np.arange(2), path="../data/MNIST_data"):
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
        x_train, y_train = load_mnist("training_data")
        x_test, y_test = load_mnist("testing_data")
        idx_train = np.append(
            np.where(y_train == 0)[0][:train_num],
            np.where(y_train == 1)[0][:train_num])

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)

        # Test Leaving only labels 0 and 1
        idx_test = np.append(
            np.where(y_test == 0)[0][:test_num],
            np.where(y_test == 1)[0][:test_num])

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)

        return x_train, y_train, x_test, y_test

    def RotCircuit(para, qlist):

        cir = pq.QCircuit()
        cir << pq.RZ(qlist, para[2])
        cir << pq.RY(qlist, para[1])
        cir << pq.RZ(qlist, para[0])

        return cir

    def build_RotCircuit(qubits, weights):
        cir = pq.QCircuit()
        cir << RotCircuit(weights[0:3], qubits[0])
        cir << RotCircuit(weights[3:6], qubits[1])
        cir << RotCircuit(weights[6:9], qubits[2])
        cir << RotCircuit(weights[9:12], qubits[3])
        cir << RotCircuit(weights[12:15], qubits[4])
        cir << RotCircuit(weights[15:18], qubits[5])
        cir << RotCircuit(weights[18:21], qubits[6])
        cir << RotCircuit(weights[21:24], qubits[7])
        cir << RotCircuit(weights[24:27], qubits[8])
        cir << RotCircuit(weights[27:30], qubits[9])
        cir << RotCircuit(weights[30:33], qubits[10])
        cir << RotCircuit(weights[33:36], qubits[11])
        cir << RotCircuit(weights[36:39], qubits[12])
        cir << RotCircuit(weights[39:42], qubits[13])
        cir << RotCircuit(weights[42:45], qubits[14])
        cir << RotCircuit(weights[45:48], qubits[15])

        return cir

    def CRXCircuit(para, control_qlists, rot_qlists):
        cir = pq.QCircuit()
        cir << pq.CRX(control_qlists,rot_qlists, para)
        return cir

    def build_CRotCircuit(qubits, weights):
        cir = pq.QCircuit()
        cir << CRXCircuit(weights[0], qubits[0], qubits[1])
        cir << CRXCircuit(weights[1], qubits[1], qubits[2])
        cir << CRXCircuit(weights[2], qubits[2], qubits[3])
        cir << CRXCircuit(weights[3], qubits[3], qubits[4])
        cir << CRXCircuit(weights[4], qubits[4], qubits[5])
        cir << CRXCircuit(weights[5], qubits[5], qubits[6])
        cir << CRXCircuit(weights[6], qubits[6], qubits[7])
        cir << CRXCircuit(weights[7], qubits[7], qubits[8])
        cir << CRXCircuit(weights[8], qubits[8], qubits[9])
        cir << CRXCircuit(weights[9], qubits[9], qubits[10])
        cir << CRXCircuit(weights[10], qubits[10], qubits[11])
        cir << CRXCircuit(weights[11], qubits[11], qubits[12])
        cir << CRXCircuit(weights[12], qubits[12], qubits[13])
        cir << CRXCircuit(weights[13], qubits[13], qubits[14])
        cir << CRXCircuit(weights[14], qubits[14], qubits[15])
        cir << CRXCircuit(weights[15], qubits[15], qubits[0])

        return cir


    def build_qmlp_circuit(x, weights, ):
        qubits = range(16)
        machine = pq.CPUQVM()
        cir = pq.QCircuit()
        num_qubits = len(qubits)
        for i in range(num_qubits):
            cir << pq.RX(qubits[i], x[i])

        cir << build_RotCircuit(qubits, weights[0:48])
        cir << build_CRotCircuit(qubits, weights[48:64])

        for i in range(num_qubits):
            cir << pq.RX(qubits[i], x[i])

        cir << build_RotCircuit(qubits, weights[64:112])
        cir << build_CRotCircuit(qubits, weights[112:128])

        prog = pq.QProg()
        prog << cir


        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str)
            exp_vals.append(exp2)

        return exp_vals

    class QMLPModel(Module):
        def __init__(self):
            super(QMLPModel, self).__init__()
            self.ave_pool2d = AvgPool2D([7, 7], [7, 7], "valid")
            self.quantum_circuit = QuantumLayer(build_qmlp_circuit, 128, diff_method="finite_diff")

            self.linear = Linear(16, 10)

        def forward(self, x):
            bsz = x.shape[0]
            x = self.ave_pool2d(x)
            input_data = x.reshape([bsz, 16])
            quanutum_result = self.quantum_circuit(input_data)
            result = self.linear(quanutum_result)
            return result

    def vqnet_test_QMLPModel():

        train_size = 1000
        eval_size = 100
        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = QMLPModel()
        optimizer = Adam(model.parameters(), lr=0.005)
        loss_func = CrossEntropyLoss()
        loss_list = []
        epochs = 5
        batch_size = 16

        test_acc_list = []
        test_loss_list = []
        for epoch in range(1, epochs):
            total_loss = []

            correct = 0

            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for x, y in data_generator(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    shuffle=True):

                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Forward pass
                output = model(x)
                # Calculating loss
                loss = loss_func(y, output)

                train_loss += loss.item()

                #
                pred = output.argmax(dim=1).numpy()
                true = pyvqnet.tensor.to_tensor(y).argmax(dim=1)
                train_correct += (pred == true).sum().item()
                train_total += batch_size

                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer._step()
                total_loss.append(loss.item())
            # epoch
            epoch_loss = np.sum(total_loss) / len(total_loss)
            loss_list.append(epoch_loss)
            train_acc = correct / train_total
            print("##########################")
            print(f"Train Accuracy: {train_acc}")
            print("epoch: ", epoch)
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

            # -------------------- (epoch) -------------------- #
            model.eval()  #
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with pyvqnet.no_grad():  #
                #  batch_size (batch size)
                for x_batch, y_batch in data_generator(x_test,
                                                    y_test,
                                                    batch_size=batch_size,
                                                    shuffle=False):
                    x_batch = x_batch.reshape(-1, 1, 28, 28)

                    output:pyvqnet.tensor.QTensor = model(x_batch)
                    loss = loss_func(y_batch, output)
                    test_loss += loss.item() * len(x_batch)  #

                    #
                    pred = output.argmax(dim=1)
                    true = pyvqnet.tensor.to_tensor(y_batch).argmax(dim=1)
                    test_correct += (pred == true).sum().item()
                    test_total += len(x_batch)

            avg_test_loss = test_loss / test_total
            test_acc = test_correct / test_total
            test_loss_list.append(avg_test_loss)
            test_acc_list.append(test_acc)

            print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {avg_test_loss:.6f}")
            # ------------------------------------------------------------------ #

        #
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(loss_list)+1), loss_list, 'b-o', label='Training Loss')
        plt.plot(range(1, len(test_loss_list)+1), test_loss_list, 'r-s', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss_curves.png')
        plt.show()

    if __name__ == "__main__":

        vqnet_test_QMLPModel()



Data results
-----------------------

Save the loss curve plot for the training data and the test data results.
Loss on the training set

.. image:: ./images/QMLP.png
   :width: 600 px
   :align: center

|


.. _QDRL_DEMO:

5. Hybrid quantum-classical QDRL network model
==============================================


We introduce and analyze a quantum deep reinforcement learning network (QDRL) that recasts classical deep reinforcement learning algorithms, such as experience replay and target networks, as variational quantum circuits.
Compared with classical neural networks, we use a quantum information encoding scheme to reduce the number of model parameters. See `QDRL: Variational Quantum Circuits for Deep Reinforcement Learning <https://arxiv.org/pdf/1907.00397.pdf>`_.
We write a simple example that integrates pyqpanda3 with VQNet.



Building the hybrid classical-quantum neural network
----------------------------------------------------

Install ``gym`` == 0.23.0 and ``pygame`` == 2.1.2 .

.. code-block::


    import numpy as np
    import random
    import gym
    import time
    from matplotlib import animation
    import pyqpanda3.core as pq
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet import kfloat32
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.pq3.measure import expval

    import matplotlib
    from matplotlib import pyplot as plt
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
        print("Can not use matplot TkAgg")
    def display_frames_as_gif(frames, c_index):
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        name_result = "./result_"+str(c_index)+".gif"
        anim.save(name_result, writer='pillow', fps=10)
    CIRCUIT_SIZE = 4
    MAX_ITERATIONS = 50
    MAX_STEPS = 250
    BATCHSIZE = 5
    TARGET_MAX = 20
    GAMMA = 0.99
    STATE_T = 0
    ACTION = 1
    REWARD = 2
    STATE_NT = 3
    DONE = 4
    def RotCircuit(para, qlist):

        if isinstance(para, QTensor):
            para = QTensor._to_numpy(para)
        if para.ndim > 1:
            raise ValueError(" dim of paramters in Rot should be 1")
        if para.shape[0] != 3:
            raise ValueError(" numbers of paramters in Rot should be 3")
        cir = pq.QCircuit()
        cir << pq.RZ(qlist, para[2])
        cir << pq.RY(qlist, para[1])
        cir << pq.RZ(qlist, para[0])
        return cir
    def layer_circuit(qubits, weights):
        cir = pq.QCircuit()
        # Entanglement block
        cir << pq.CNOT(qubits[0], qubits[1])
        cir << pq.CNOT(qubits[1], qubits[2])
        cir << pq.CNOT(qubits[2], qubits[3])
        # u3 gate
        cir << RotCircuit(weights[0], qubits[0])  # weights shape = [4, 3]
        cir << RotCircuit(weights[1], qubits[1])
        cir << RotCircuit(weights[2], qubits[2])
        cir << RotCircuit(weights[3], qubits[3])
        return cir
    def encoder(encodings):
        encodings = int(encodings[0])
        return [i for i, b in enumerate(f'{encodings:0{CIRCUIT_SIZE}b}') if b == '1']

    def build_qc(x, weights):
        machine = pq.CPUQVM()
        qubits = range(CIRCUIT_SIZE)

        cir = pq.QCircuit()
        if x:
            wires = encoder(x)
            for wire in wires:
                cir << pq.RX(qubits[wire], np.pi)
                cir << pq.RZ(qubits[wire], np.pi)
        # parameter number = 24
        weights = weights.reshape([2, 4, 3])
        # layer wise
        for w in weights:
            cir << layer_circuit(qubits, w)
        prog = pq.QProg()
        prog << cir
        exp_vals = []
        for position in range(n_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str)
            exp_vals.append(exp2)
        return exp_vals
    class DRLModel(Module):
        def __init__(self):
            super(DRLModel, self).__init__()
            self.quantum_circuit = QuantumLayer(build_qc, 24, diff_method="finite_diff")

        def forward(self, x):
            quanutum_result = self.quantum_circuit(x)
            return quanutum_result
    env = gym.make("FrozenLake-v1", is_slippery = False, map_name = '4x4')
    state = env.reset()
    n_layers = 2
    n_qubits = 4
    targ_counter = 0
    sampled_vs = []
    memory = {}
    param = QTensor(0.01 * np.random.randn(n_layers, n_qubits, 3))
    bias = QTensor([[0.0, 0.0, 0.0, 0.0]])
    param_targ = param.copy().reshape([1, -1]).pdata[0]
    bias_targ = bias.copy()
    loss_func = MeanSquaredError()
    model = DRLModel()
    opt = Adam(model.parameters(), lr=5)
    for i in range(MAX_ITERATIONS):
        start = time.time()
        state_t = env.reset()
        a_init = env.action_space.sample()
        total_reward = 0
        done = False
        frames = []
        for t in range(MAX_STEPS):
            frames.append(env.render(mode='rgb_array'))
            time.sleep(0.1)
            input_x = QTensor([[state_t]],dtype=kfloat32)
            acts = model(input_x) + bias

            act_t = tensor.QTensor.argmax(acts)

            act_t_np = int(act_t.pdata[0])
            print(f'Episode: {i}, Steps: {t}, act: {act_t_np}')
            state_nt, reward, done, info = env.step(action=act_t_np)
            targ_counter += 1
            input_state_nt = QTensor([[state_nt]],dtype=kfloat32)
            act_nt = QTensor.argmax(model(input_state_nt)+bias)
            act_nt_np = int(act_nt.pdata[0])
            memory[i, t] = (state_t, act_t, reward, state_nt, done)
            if len(memory) >= BATCHSIZE:

                sampled_vs = [memory[k] for k in random.sample(list(memory), BATCHSIZE)]
                target_temp = []
                for s in sampled_vs:
                    if s[DONE]:
                        target_temp.append(QTensor(s[REWARD]).reshape([1, -1]))
                    else:
                        input_s = QTensor([[s[STATE_NT]]],dtype=kfloat32)
                        out_temp = s[REWARD] + GAMMA * tensor.max(model(input_s) + bias_targ)
                        out_temp = out_temp.reshape([1, -1])
                        target_temp.append(out_temp)
                target_out = []
                for b in sampled_vs:
                    input_b = QTensor([[b[STATE_T]]], requires_grad=True,dtype=kfloat32)
                    out_result = model(input_b) + bias
                    index = int(b[ACTION].pdata[0])
                    out_result_temp = out_result[0][index].reshape([1, -1])
                    target_out.append(out_result_temp)
                opt.zero_grad()
                target_label = tensor.concatenate(target_temp, 1)
                output = tensor.concatenate(target_out, 1)
                loss = loss_func(target_label, output)
                loss.backward()
                opt.step()
            # update parameters in target circuit
            if targ_counter == TARGET_MAX:
                param_targ = param.copy().reshape([1, -1]).pdata[0]
                bias_targ = bias.copy()
                targ_counter = 0
            state_t, act_t_np = state_nt, act_nt_np
            if done:
                print("reward", reward)
                if reward == 1.0:
                    frames.append(env.render(mode='rgb_array'))
                    display_frames_as_gif(frames, i)
                    exit()
                break
        end = time.time()



Unsupervised learning
****************************

1. Quantum Kmeans
==========================

1.1 Introduction
-------------------

Clustering is a typical unsupervised learning algorithm, mainly used to group similar samples automatically. It partitions samples into different classes based on their similarity. Different similarity measures produce different clustering results; the common one is Euclidean distance. We present the quantum K-Means algorithm. K-Means is a distance-based clustering algorithm that uses distance as the similarity measure: the closer two objects are, the more similar they are. It assumes clusters consist of objects that are close together, so compact and separate clusters are the goal.

VQNet also supports development of Quantum Kmeans quantum machine learning models. Below is an example of a Quantum Kmeans clustering task. With a quantum circuit, we can build a measurement value positively correlated with the Euclidean distance of classical machine learning variables, achieving nearest-neighbor search.

1.2 Algorithm principles
-------------------------

The quantum K-Means algorithm mainly uses the swap test to compare distances between input data points. K points are chosen at random from N data points as centroids; the distance from each point to each centroid is measured, each point is assigned to the nearest centroid class, the centroid of each class is recomputed, and steps 2 and 3 repeat until the new centroid is equal to or below a specified threshold, at which point the algorithm ends. In our example, we choose 100 data points and 2 centroids and use a CSWAP circuit to compute distances. We obtain two data point clusters. :math:`|0\rangle` is the ancilla qubit, which an H gate turns into :math:`\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)`. Controlled by the qubit :math:`|1\rangle`, the circuit swaps :math:`|x\rangle` and :math:`|y\rangle`. The final result is:

.. math::

    |0_{anc}\rangle |x\rangle |y\rangle \rightarrow \frac{1}{2}|0_{anc}\rangle(|xy\rangle + |yx\rangle) + \frac{1}{2}|1_{anc}\rangle(|xy\rangle - |yx\rangle)

If we measure the ancilla qubit alone, the probability of the :math:`|1\rangle` basis state in the final state is:

.. math::

    P(|1_{anc}\rangle) = \frac{1}{2} - \frac{1}{2}|\langle x | y \rangle|^2

The Euclidean distance between the two quantum states is as follows:

.. math::

    Euclidean \ distance = \sqrt{(2 - 2|\langle x | y \rangle|)}

It can be seen that the measured qubit :math:`|1\rangle` is positively correlated with the Euclidean distance. The quantum circuit for this algorithm is as follows:

.. image:: ./images/Kmeans.jpg
   :width: 600 px
   :align: center

|

1.3 VQNet implementation
------------------------

1.3.1 Environment setup
^^^^^^^^^^^^^^^^^^^^^^^

The environment uses Python 3.10. We recommend conda, which bundles numpy, scipy, matplotlib, sklearn, and other packages for convenience. If you use a plain Python environment, install the required packages.
You also need pyvqnet.

1.3.2 Data preparation
^^^^^^^^^^^^^^^^^^^^^^^

Data is generated randomly with make_blobs from scipy, and a function is defined to generate Gaussian-distributed data.

.. code-block::

    import sys
    sys.path.insert(0, "../")
    import math
    import numpy as np
    from pyvqnet.tensor.tensor import QTensor, zeros
    import pyvqnet.tensor.tensor as tensor
    import pyqpanda3.core as pq
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass
    # Return data points and cluster centers for data count n, center count k, and data std
    def get_data(n, k, std):
        data = make_blobs(n_samples=n,
                        n_features=2,
                        centers=k,
                        cluster_std=std,
                        random_state=100)
        points = data[0]
        centers = data[1]
        return points, centers

1.3.3 Quantum circuit
^^^^^^^^^^^^^^^^^^^^^^^

Build the quantum circuit with VQNet

.. code-block::

    # Compute the rotation angle of the quantum gate from the input coordinate point d(x,y)
    def get_theta(d):
        x = d[0]
        y = d[1]
        theta = 2 * math.acos((x.item() + y.item()) / 2.0)
        return theta

    # Build the quantum circuit from the input quantum data points
    def qkmeans_circuits(x, y):
        """
        Quantum Circuit for kmeans
        """
        theta_1 = get_theta(x)
        theta_2 = get_theta(y)

        num_qubits = 3
        machine = pq.CPUQVM()
        qubits = range(num_qubits)
        cbits = range(num_qubits)
        circuit = pq.QCircuit()

        circuit << pq.H(qubits[0])
        circuit << pq.H(qubits[1])
        circuit << pq.H(qubits[2])

        circuit << pq.U3(qubits[1], theta_1, np.pi, np.pi)
        circuit << pq.U3(qubits[2], theta_2, np.pi, np.pi)

        circuit << pq.SWAP(qubits[1], qubits[2]).control([qubits[0]])

        circuit << pq.H(qubits[0])

        prog = pq.QProg()
        prog << circuit
        prog << pq.measure(qubits[0], cbits[0])

        machine.run(prog, 1024)
        result = machine.result().get_counts()

        data = result

        if len(data) == 1:
            return 0.0
        else:
            return result.get("1", 0) / 1024.0

1.3.4 Data visualization
^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the clustering data

.. code-block::

    # Visualize the scatter points and cluster centers
    def draw_plot(points, centers, label=True):
        points = np.array(points)
        centers = np.array(centers)
        if label is False:
            plt.scatter(points[:, 0], points[:, 1])
        else:
            plt.scatter(points[:, 0], points[:, 1], c=centers, cmap="viridis")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

1.3.5 Clustering computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute cluster centers for the clustering data

.. code-block::

    # Randomly generate cluster centers
    def initialize_centers(points,k):
        return points[np.random.randint(points.shape[0],size=k),:]


    def find_nearest_neighbour(points, centroids):
        """
        Find nearest neighbour
        """
        n = points.shape[0]
        k = centroids.shape[0]

        centers = zeros([n], dtype=points.dtype)

        for i in range(n):
            min_dis = 10000
            ind = 0
            for j in range(k):

                temp_dis = qkmeans_circuits(points[i, :], centroids[j, :])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    ind = j
            centers[i] = ind

        return centers


    def find_centroids(points, centers):
        """
        find centroids
        """
        k = int(tensor.max(centers).item()) + 1
        centroids = tensor.zeros([k, 2], dtype=points.dtype)

        for i in range(k):

            cur_i = centers == i
            x = points[:, 0]
            x = x[cur_i]
            y = points[:, 1]
            y = y[cur_i]
            centroids[i, 0] = tensor.mean(x)
            centroids[i, 1] = tensor.mean(y)

        return centroids

    def preprocess(points):
        n = len(points)
        x = 30.0 * np.sqrt(2)
        for i in range(n):
            points[i, :] += 15
            points[i, :] /= x

        return points


    def qkmeans_run():
        """
        Main function for run qkmeans algorithm
        """
        n = 100  # number of data points
        k = 3  # Number of centers
        std = 2  # std of datapoints

        points, o_centers = get_data(n, k, std)  # dataset

        points = preprocess(points)  # Normalize dataset

        centroids = initialize_centers(points, k)  # Intialize centroids

        epoch = 5
        points = QTensor(points)
        centroids = QTensor(centroids)
        plt.figure()
        plt.title("origin")
        draw_plot(points.data, o_centers, label=False)

        # run k-means algorithm
        for i in range(epoch):
            print(f"iteration {i}")
            centers = find_nearest_neighbour(points,
                                            centroids)  # find nearest centers
            centroids = find_centroids(points, centers)  # find centroids

        plt.figure()
        plt.title(f"result after iteration {epoch}")
        draw_plot(points.data, centers.data)

    # Program entry point
    if __name__ == "__main__":
        qkmeans_run()

1.3.6 Data distribution before clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./images/ep_1.png
   :width: 600 px
   :align: center

1.3.7 Data distribution after clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./images/ep_9.png
   :width: 600 px
   :align: center

|


Quantum machine learning research
*********************************

Fitting Fourier series with quantum models
==========================================

By mapping data inputs to predictions with parameterized quantum circuits, quantum computers can be used for supervised learning. Although much work has studied the practical value of this approach, many important theoretical properties of these models remain unknown. Here we study how the strategy for encoding data into the model affects the expressive power of parameterized quantum circuits as function approximators.

Following `The effect of data encoding on the expressive power of variational quantum machine learning models <https://arxiv.org/pdf/2008.08605.pdf>`_, this example connects common quantum machine learning models designed for quantum computers with Fourier series.

1.1 Fitting Fourier series with serial Pauli rotation encoding
--------------------------------------------------------------

We first show how a quantum model that uses Pauli rotations as data encoding gates can fit Fourier series only to a limited degree. For simplicity, we look at single-qubit circuits only:

.. image:: ./images/single_qubit_model.png
   :width: 600 px
   :align: center

|

Create input data, define a parallel quantum model, and obtain results without model training.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)


    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j]*degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx+1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(x_) for x_ in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()


    def S(scaling, x, qubits):
        cir = pq.QCircuit()
        cir << pq.RX(qubits[0], scaling * x)
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir << pq.RZ(qubits[0], theta[0])
        cir << pq.RY(qubits[0], theta[1])
        cir << pq.RZ(qubits[0], theta[2])
        return cir

    def serial_quantum_model(weights, x, num_qubits, scaling):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()
        qubits = range(num_qubits)

        for theta in weights[:-1]:
            cir << W(theta, qubits)
            cir << S(scaling, x, qubits)

        # (L+1)'th unitary
        cir << W(weights[-1], qubits)
        prog = pq.QProg()
        prog << cir

        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str)
            exp_vals.append(exp2)

        return exp_vals

    r = 1
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))  # some random initial weights

    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [serial_quantum_model(weights, x_, 1, 1) for x_ in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()


Results of running the quantum circuit without training:

.. image:: ./images/single_qubit_model_result_no_train.png
   :width: 600 px
   :align: center

|


Create input data, define a serial quantum model, and build a training model with the QuantumLayer layer of the VQNet framework.

.. code-block::

    """
    Quantum Fourier Series Serial
    """
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.measure import expval
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j]*degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx+1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()


    def S(x, qubits):
        cir = pq.QCircuit()
        cir << pq.RX(qubits[0], x)
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir << pq.RZ(qubits[0], theta[0])
        cir << pq.RY(qubits[0], theta[1])
        cir << pq.RZ(qubits[0], theta[2])
        return cir


    r = 1
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))  # some random initial weights

    x = np.linspace(-6, 6, 70)


    def q_circuits_loop(x, weights):
        machine = pq.CPUQVM()
        qubits = range(1)

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3])

            for theta in weights[:-1]:
                cir << W(theta, qubits)
                cir << S(xx, qubits)

            cir << W(weights[-1], qubits)
            prog = pq.QProg()
            prog << cir

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str)
                exp_vals.append(exp2)
                result.append(exp2)
        return result

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.q_fourier_series = QuantumLayer(q_circuits_loop, 6)

        def forward(self, x):
            return self.q_fourier_series(x)

    def run():
        model = Model()

        optimizer = Adam(model.parameters(), lr=0.5)
        batch_size = 2
        epoch = 5
        loss = MeanSquaredError()
        print("start training..............")
        model.train()
        max_steps = 50
        for i in range(epoch):
            sum_loss = 0
            count = 0
            for step in range(max_steps):
                optimizer.zero_grad()
                # Select batch of data
                batch_index = np.random.randint(0, len(x), (batch_size,))
                x_batch = x[batch_index]
                y_batch = target_y[batch_index]
                data, label = QTensor([x_batch]), QTensor([y_batch])
                result = model(data)
                loss_b = loss(label, result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size
            print(f"epoch:{i}, #### loss:{sum_loss/count} ")

            model.eval()
            predictions = []
            for xx in x:
                data = QTensor([[xx]])
                result = model(data)
                predictions.append(result.pdata[0])

            plt.plot(x, target_y, c='black', label="true")
            plt.scatter(x, target_y, facecolor='white', edgecolor='black')
            plt.plot(x, predictions, c='blue', label="predict")
            plt.ylim(-1, 1)
            plt.legend(loc="upper right")
            plt.show()

    if __name__ == "__main__":
        run()

The quantum model is:

.. image:: ./images/single_qubit_model_circuit.png
   :width: 600 px
   :align: center

|

Network training results:

.. image:: ./images/single_qubit_model_result.png
   :width: 600 px
   :align: center

|

Network training loss:

.. code-block::

    start training..............
    epoch:0, #### loss:0.04852807720773853
    epoch:1, #### loss:0.012945819365559146
    epoch:2, #### loss:0.0009359727291666786
    epoch:3, #### loss:0.00015995280153333625
    epoch:4, #### loss:3.988249877352246e-05


1.2 Fitting Fourier series with parallel Pauli rotation encoding
----------------------------------------------------------------

As the paper shows, we expect results similar to the serial model: an r-th order Fourier series can be fitted only when the encoding gates in the quantum model are repeated at least r times. The qubit circuit:

.. image:: ./images/parallel_model.png
   :width: 600 px
   :align: center

|

Create input data, define a parallel quantum model, and obtain results without model training.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j] * degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])


    def S1(x, qubits):
        cir = pq.QCircuit()
        for q in qubits:
            cir << pq.RX(q, x)
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir << pq.RZ(qubits[i], theta[0][i][0])
            cir << pq.RY(qubits[i], theta[0][i][1])
            cir << pq.RZ(qubits[i], theta[0][i][2])

        for i in range(len(qubits) - 1):
            cir << pq.CNOT(qubits[i], qubits[i + 1])
        cir << pq.CNOT(qubits[len(qubits) - 1], qubits[0])

        for i in range(len(qubits)):
            cir << pq.RZ(qubits[i], theta[1][i][0])
            cir << pq.RY(qubits[i], theta[1][i][1])
            cir << pq.RZ(qubits[i], theta[1][i][2])

        cir << pq.CNOT(qubits[0], qubits[len(qubits) - 1])
        for i in range(len(qubits) - 1):
            cir << pq.CNOT(qubits[i + 1], qubits[i])

        for i in range(len(qubits)):
            cir << pq.RZ(qubits[i], theta[2][i][0])
            cir << pq.RY(qubits[i], theta[2][i][1])
            cir << pq.RZ(qubits[i], theta[2][i][2])

        for i in range(len(qubits) - 1):
            cir << pq.CNOT(qubits[i], qubits[i + 1])
        cir << pq.CNOT(qubits[len(qubits) - 1], qubits[0])

        return cir

    def parallel_quantum_model(weights, x, num_qubits):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()
        qubits = range(num_qubits)

        cir << W1(weights[0], qubits)
        cir << S1(x, qubits)

        cir << W1(weights[1], qubits)
        prog = pq.QProg()
        prog << cir

        exp_vals = []
        for position in range(1):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str)
            exp_vals.append(exp2)

        return exp_vals

    r = 3

    trainable_block_layers = 3
    weights = 2 * np.pi * np.random.random(size=(2, trainable_block_layers, r, 3))

    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [parallel_quantum_model(weights, xx, r) for xx in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()

Results of running the quantum circuit without training:

.. image:: ./images/parallel_model_result_no_train.png
   :width: 600 px
   :align: center

|


Create input data, define a parallel quantum model, and build a training model with the QuantumLayer layer of the VQNet framework.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.measure import expval
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1  # degree of the target function
    scaling = 1  # scaling of the data
    coeffs = [0.15 + 0.15j] * degree  # coefficients of non-zero frequencies
    coeff0 = 0.1  # coefficient of zero frequency

    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()

    def S1(x, qubits):
        cir = pq.QCircuit()
        for q in qubits:
            cir << pq.RX(q, x)
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir << pq.RZ(qubits[i], theta[0][i][0])
            cir << pq.RY(qubits[i], theta[0][i][1])
            cir << pq.RZ(qubits[i], theta[0][i][2])

        for i in range(len(qubits) - 1):
            cir << pq.CNOT(qubits[i], qubits[i + 1])
        cir << pq.CNOT(qubits[len(qubits) - 1], qubits[0])

        for i in range(len(qubits)):
            cir << pq.RZ(qubits[i], theta[1][i][0])
            cir << pq.RY(qubits[i], theta[1][i][1])
            cir << pq.RZ(qubits[i], theta[1][i][2])

        cir << pq.CNOT(qubits[0], qubits[len(qubits) - 1])
        for i in range(len(qubits) - 1):
            cir << pq.CNOT(qubits[i + 1], qubits[i])

        for i in range(len(qubits)):
            cir << pq.RZ(qubits[i], theta[2][i][0])
            cir << pq.RY(qubits[i], theta[2][i][1])
            cir << pq.RZ(qubits[i], theta[2][i][2])

        for i in range(len(qubits) - 1):
            cir << pq.CNOT(qubits[i], qubits[i + 1])
        cir << pq.CNOT(qubits[len(qubits) - 1], qubits[0])

        return cir

    def q_circuits_loop(x, weights):
        machine = pq.CPUQVM()
        qubits = range(3)

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3, 3, 3])

            cir << W1(weights[0], qubits)
            cir << S1(xx, qubits)

            cir << W1(weights[1], qubits)
            prog = pq.QProg()
            prog << cir

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str)
                exp_vals.append(exp2)
                result.append(exp2)
        return result


    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()

            self.q_fourier_series = QuantumLayer(q_circuits_loop, 2 * 3 * 3 * 3)

        def forward(self, x):
            return self.q_fourier_series(x)

    def run():
        model = Model()

        optimizer = Adam(model.parameters(), lr=0.01)
        batch_size = 2
        epoch = 5
        loss = MeanSquaredError()
        print("start training..............")
        model.train()
        max_steps = 50
        for i in range(epoch):
            sum_loss = 0
            count = 0
            for step in range(max_steps):
                optimizer.zero_grad()
                # Select batch of data
                batch_index = np.random.randint(0, len(x), (batch_size,))
                x_batch = x[batch_index]
                y_batch = target_y[batch_index]
                data, label = QTensor([x_batch]), QTensor([y_batch])
                result = model(data)
                loss_b = loss(label, result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size

            loss_cout = sum_loss / count
            print(f"epoch:{i}, #### loss:{loss_cout} ")

            if loss_cout < 0.002:
                model.eval()
                predictions = []
                for xx in x:
                    data = QTensor([[xx]])
                    result = model(data)
                    predictions.append(result.pdata[0])

                plt.plot(x, target_y, c='black', label="true")
                plt.scatter(x, target_y, facecolor='white', edgecolor='black')
                plt.plot(x, predictions, c='blue', label="predict")
                plt.ylim(-1, 1)
                plt.legend(loc="upper right")
                plt.show()


    if __name__ == "__main__":
        run()

The quantum model is:

.. image:: ./images/parallel_model_circuit.png
   :width: 600 px
   :align: center

|

Network training results:

.. image:: ./images/parallel_model_result.png
   :width: 600 px
   :align: center

|

Network training loss:

.. code-block::

    start training..............
    epoch:0, #### loss:0.0037272341538482578
    epoch:1, #### loss:5.271130586635309e-05
    epoch:2, #### loss:4.714951917250687e-07
    epoch:3, #### loss:1.0968826371082763e-08
    epoch:4, #### loss:2.1258629738507562e-10



Expressibility of quantum circuits
===================================


In `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_,
the authors propose a method to quantify expressibility based on the probability distribution of fidelities between output states of a quantum neural network.
For any quantum neural network :math:`U(\vec{\theta})`, sample the network parameters twice (call them :math:`\vec{\phi}` and :math:`\vec{\psi}`),
and the fidelity :math:`F=|\langle0|U(\vec{\phi})^\dagger U(\vec{\psi})|0\rangle|^2` between the output states of the two quantum circuits follows some probability distribution:

.. math::

    F\sim{P}(f)

The paper notes that when the quantum neural network :math:`U` is uniformly distributed over all unitary matrices (in which case :math:`U` follows the Haar distribution), the fidelity probability distribution :math:`P_\text{Haar}(f)` satisfies

.. math::

    P_\text{Haar}(f)=(2^{n}-1)(1-f)^{2^n-2}

The K-L divergence in statistics (also called relative entropy) measures the difference between two probability distributions. For two discrete probability distributions :math:`P,Q`, the K-L divergence is defined as

.. math::

    D_{KL}(P||Q)=\sum_jP(j)\ln\frac{P(j)}{Q(j)}

If the fidelity distribution of the quantum neural network output is written :math:`P_\text{QNN}(f)`, the expressibility of the quantum neural network is defined as the K-L divergence between :math:`P_\text{QNN}(f)` and :math:`P_\text{Haar}(f)`:

.. math::

    \text{Expr}_\text{QNN}=D_{KL}(P_\text{QNN}(f)||P_\text{Haar}(f))


Thus, the closer :math:`P_\text{QNN}(f)` is to :math:`P_\text{Haar}(f)`, the smaller :math:`\text{Expr}` is (the closer to 0),
and the stronger the expressibility of the quantum neural network. Conversely, the larger :math:`\text{Expr}` is, the weaker the expressibility.

With this definition, we can directly compute the expressibility of the single-qubit quantum neural networks :math:`R_Y(\theta)`, :math:`R_Y(\theta_1)R_Z(\theta_2)`, and :math:`R_Y(\theta_1)R_Z(\theta_2)R_Y(\theta_3)`:

Below, VQNet demonstrates the expressibility of `HardwareEfficientAnsatz <https://arxiv.org/abs/1704.05018>`_ at depths 1, 2, and 3.


.. code-block::

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scipy import integrate
    from scipy.linalg import sqrtm
    from scipy.stats import entropy

    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.quantum_expressibility.quantum_express import fidelity_of_cir, fidelity_harr_sample
    num_qubit = 1  # the number of qubit
    num_sample = 2000  # the number of sample
    outputs_y = list()  # save QNN outputs


    # plot histgram
    def plot_hist(data, num_bin, title_str):
        def to_percent(y, position):
            return str(np.around(y * 100, decimals=2)) + '%'

        plt.hist(data,
                weights=[1. / len(data)] * len(data),
                bins=np.linspace(0, 1, num=num_bin),
                facecolor="blue",
                edgecolor="black",
                alpha=0.7)
        plt.xlabel("Fidelity")
        plt.ylabel("frequency")
        plt.title(title_str)
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()



    def cir(num_qubits, depth):

        machine = pq.CPUQVM()
        qlist = range(num_qubits)
        az = HardwareEfficientAnsatz(qlist, ["rx", "RY", "rz"],
                                    entangle_gate="cnot",
                                    entangle_rules="linear",
                                    depth=depth)
        w = tensor.QTensor(
            np.random.uniform(size=[az.get_para_num()], low=0, high=2 * np.pi))

        cir1 = az.create_ansatz(w)
        return cir1, machine, qlist

Fidelities from Haar sampling follow the distribution:

.. image:: ./images/haar-fidelity.png
   :width: 600 px
   :align: center

|

.. code-block::


    # Set the circuit width and maximum depth
    num_qubit = 4
    max_depth = 3
    # Compute the fidelity distribution for Haar sampling

    flist, p_haar, theory_haar = fidelity_harr_sample(num_qubit, num_sample)
    title_str = "haar, %d qubit(s)" % num_qubit
    plot_hist(flist, 50, title_str)



    Expr_cel = list()
    # Compute expressibility at different network depths
    for DEPTH in range(1, max_depth + 1):

        f_list, p_cel = fidelity_of_cir(HardwareEfficientAnsatz, num_qubit, DEPTH,
                                        num_sample)
        title_str = f"HardwareEfficientAnsatz, {num_qubit} qubit(s) {DEPTH} layer(s)"
        plot_hist(f_list, 50, title_str)
        expr = entropy(p_cel, theory_haar)
        Expr_cel.append(expr)
    # Compare expressibility across network depths
    print(
        f"The expressibility of neural networks with depths 1, 2, 3 is { np.around(Expr_cel, decimals=4)}; lower is better.", )
    plt.plot(range(1, max_depth + 1), Expr_cel, marker='>')
    plt.xlabel("depth")
    plt.yscale('log')
    plt.ylabel("Expr.")
    plt.xticks(range(1, max_depth + 1))
    plt.title("Expressibility vs Circuit Depth")
    plt.show()

Fidelity distribution of circuits sampled at depth 1

.. image:: ./images/f1.png
   :width: 600 px
   :align: center

|

Fidelity distribution of circuits sampled at depth 2

.. image:: ./images/f2.png
   :width: 600 px
   :align: center

|

Fidelity distribution of circuits sampled at depth 3

.. image:: ./images/f3.png
   :width: 600 px
   :align: center

|

As circuit depth increases, the fidelity K-L divergence decreases and expressibility increases.

.. image:: ./images/express.png
   :width: 600 px
   :align: center

|

Barren plateaus
===================================


When training classical neural networks, gradient-based optimization runs into local minima
and also geometric structures such as saddle points where gradients are close to zero.
Quantum neural networks have a corresponding "barren plateau effect".
This striking phenomenon was first found by McClean et al. in 2018 in `Barren plateaus in quantum neural network training landscapes <https://arxiv.org/abs/1803.11173>`_.
Put simply, when the random circuit structure you choose reaches a certain level of complexity, the optimization landscape becomes very flat,
making it hard for gradient descent to find the global minimum.
For most variational quantum algorithms (VQE and others), this means that as the number of qubits grows,
circuits with random structure may not work well.
The landscape corresponding to a carefully designed loss function becomes a huge plateau,
making training harder. Random initial values struggle to escape this plateau, and gradient descent converges very slowly.


This case uses VQNet to demonstrate the barren plateau phenomenon and uses gradient analysis functions to analyze parameter gradients in user-defined quantum neural networks.

The following code builds a random circuit in a way similar to the original paper:

First, rotate all qubits by :math:`\pi/4` about the Y axis of the Bloch sphere.

The remaining structure forms one block, and each block has two layers:

- The first layer contains random rotation gates, where :math:`R \in \{R_x, R_y, R_z\}`.
- The second layer consists of CZ gates acting on each pair of adjacent qubits.

The circuit code is shown in the rand_circuit_pq function.

Once the circuit structure is fixed, we still need a loss function to define the optimization landscape.
As in the original paper, we use the loss function common in VQE algorithms:

.. math::

    \mathcal{L}(\boldsymbol{\theta})= \langle0| U^{\dagger}(\boldsymbol{\theta})H U(\boldsymbol{\theta}) |0\rangle

The unitary :math:`U(\boldsymbol{\theta})` is the randomly structured quantum neural network built in the previous section.
The Hamiltonian is :math:`H = |00\cdots 0\rangle\langle00\cdots 0|`.
This case builds the VQE algorithm above at different qubit counts and generates 200 random network structures with different random initial parameters.
Gradients of the parameters in the parameterized circuit are computed with the parameter-shift algorithm.
We then compute the mean and variance of the 200 gradients for a variational parameter.

The example analyzes the last variational parameter; readers can change it to another reasonable value.
Running it shows that as the number of qubits increases, the variance of the parameter gradients shrinks and the mean approaches 0.

.. code-block::

        """
        Barren plateau
        """
        import pyqpanda3.core as pq
        import numpy as np
        import matplotlib.pyplot as plt

        from pyvqnet.qnn import Hermitian_expval, grad

        param_idx = -1
        gate_set = [pq.RX, pq.RY, pq.RZ]


        def rand_circuit_pq(params, num_qubits):
            cir = pq.QCircuit()
            machine = pq.CPUQVM()
            qlist = range(num_qubits)

            for i in range(num_qubits):
                cir << pq.RY(
                    qlist[i],
                    np.pi / 4,
                )

            random_gate_sequence = {
                i: np.random.choice(gate_set)
                for i in range(num_qubits)
            }
            for i in range(num_qubits):
                cir << random_gate_sequence[i](qlist[i], params[i])

            for i in range(num_qubits - 1):
                cir << pq.CZ(qlist[i], qlist[i + 1])

            prog = pq.QProg()
            prog << cir
            machine.run(prog, 0)
            result = machine.result().get_state_vector()

            H = np.zeros((2**num_qubits, 2**num_qubits))
            H[0, 0] = 1
            expval = Hermitian_expval(H, result, [i for i in range(num_qubits)],
                                    num_qubits)

            return expval


        qubits = [2, 3, 4, 5, 6]
        variances = []
        num_samples = 200
        means = []
        for num_qubits in qubits:
            grad_vals = []
            for i in range(num_samples):
                params = np.random.uniform(0, np.pi, size=num_qubits)
                g = grad(rand_circuit_pq, params, num_qubits)

                grad_vals.append(g[-1])
            variances.append(np.var(grad_vals))
            means.append(np.mean(grad_vals))
        variances = np.array(variances)
        means = np.array(means)
        qubits = np.array(qubits)


        plt.figure()

        plt.plot(qubits, variances, "v-")

        plt.xlabel(r"N Qubits")
        plt.ylabel(r"variance")
        plt.show()


        plt.figure()

        plt.plot(qubits, means, "v-")

        plt.xlabel(r"N Qubits")
        plt.ylabel(r"means")

        plt.show()

The figure below shows the mean parameter gradient versus qubit count. As the qubit count increases, the parameter gradient approaches 0.

.. image:: ./images/Barren_Plateau_mean.png
   :width: 600 px
   :align: center

|

The figure below shows the variance of parameter gradients versus qubit count. As the qubit count increases, the parameter gradient hardly changes.
It is foreseeable that for circuits built from arbitrary parameterized gates, as qubit count grows, parameter training under any initialization easily gets stuck.

.. image:: ./images/Barren_Plateau_variance.png
   :width: 600 px
   :align: center

|

Quantum perceptron
===================================

Artificial neural networks are a classic approach to machine learning algorithms and artificial intelligence. Historically, the simplest implementation of an artificial neuron dates back to Rosenblatt's classical "perceptron", but its long-term practical use may be hindered by rapidly scaling computational complexity, especially for training multilayer perceptron networks. Following `An Artificial Neuron Implemented on an Actual Quantum Processor <https://arxiv.org/abs/1811.02266>`__, we implement a quantum computer version of the perceptron with an algorithm based on quantum information, which shows an exponential advantage over classical models in encoding resources.

This quantum perceptron processes 0/1 binary bit strings. Its goal is to recognize the cross-shaped pattern :math:`w` shown below.

.. image:: ./images/QP-data.png
   :width: 600 px
   :align: center

|

It is encoded as a binary bit string, with black as 0 and white as 1, so :math:`w` encodes to (1,1,1,1,1,1,0,1,1,0,0,0,1,1,0,1). The 16-bit string fits into the signs of the amplitudes of a 4-bit quantum state: a negative sign encodes 0 and a positive sign encodes 1. With this encoding, the algorithm input is converted into a 16-bit binary string. Each distinct binary string corresponds to a specific input circuit :math:`U_i`.

The quantum perceptron circuit proposed in the paper is structured as follows:

.. image:: ./images/QP-cir.png
   :width: 600 px
   :align: center

|

On qubits 0 to 3, build the encoding circuit :math:`U_i`, which contains multi-controlled :math:`CZ` gates, :math:`CNOT` gates, and :math:`H` gates. Immediately after :math:`U_i`, build the weight transformation circuit :math:`U_w`, also made of controlled gates and :math:`H` gates. :math:`U_i` applies a unitary transformation that encodes the data into a quantum state:

.. math::
    U_i|0\rangle^{\otimes N}=\left|\psi_i\right\rangle

The unitary transformation :math:`U_w` computes the inner product between the input and the weights:

.. math::
    U_w\left|\psi_i\right\rangle=\sum_{j=0}^{m-1} c_j|j\rangle \equiv\left|\phi_{i, w}\right\rangle

A multi-controlled :math:`NOT` gate with a target bit on the ancilla, followed by some :math:`H`, :math:`X`, and :math:`CX` gates as an activation function, yields the normalized activation probability of :math:`U_i` and :math:`U_w`:

.. math::
    \left|\phi_{i, w}\right\rangle|0\rangle_a \rightarrow \sum_{j=0}^{m-2} c_j|j\rangle|0\rangle_a+c_{m-1}|m-1\rangle|1\rangle_a

When the binary string of input :math:`i` exactly matches :math:`w`, this normalized probability is maximized.

VQNet provides the ``QuantumNeuron`` module for this algorithm. First initialize a quantum perceptron ``QuantumNeuron``.

.. code-block::

    perceptron = QuantumNeuron()

Use the ``gen_4bitstring_data`` interface to generate the various data and class labels from the paper.

.. code-block::

    training_label, test_label = perceptron.gen_4bitstring_data()

Use the ``train`` interface to iterate over all data and obtain the final trained quantum perceptron circuit Uw.

.. code-block::

    trained_para = perceptron.train(training_label, test_label)

.. image:: ./images/QP-pic.png
   :width: 600 px
   :align: center

|

Accuracy results on the test data

.. image:: ./images/QP-acc.png
   :width: 600 px
   :align: center

|


Doubly stochastic gradient descent
===================================

In variational quantum algorithms, parameterized quantum circuits are optimized by classical gradient descent to minimize an expectation value.
Expectation values can be computed analytically in classical simulation, but on quantum hardware a program can only sample them; as the number of samples and shots grows, the sampled expectation converges to the theoretical value but may never reach it exactly.
Sweke et al. found a doubly stochastic gradient descent method in `this paper <https://arxiv.org/abs/1910.01155>`_.
They prove that quantum gradient descent using a finite number of measurement samples (or shots) to estimate gradients is a form of stochastic gradient descent.
Moreover, when the optimization involves a linear combination of expectation values (as in VQE), sampling terms from that linear combination can further reduce the required time complexity.

VQNet implements an example of this algorithm: using VQE to find the ground state energy of a target Hamiltonian. Note that here the number of shots is set to 1.

.. math::

    H = \begin{bmatrix}
          8 & 4 & 0 & -6\\
          4 & 0 & 4 & 0\\
          0 & 4 & 8 & 0\\
          -6 & 0 & 0 & 0
        \end{bmatrix}.

.. code-block::

    import numpy as np
    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.template import StronglyEntanglingTemplate
    from pyvqnet.qnn.measure import Hermitian_expval
    from pyvqnet.qnn import QuantumLayer
    from pyvqnet.optim import SGD
    from pyvqnet.tensor import QTensor,to_tensor
    from matplotlib import pyplot as plt

    num_layers = 2
    num_wires = 2
    eta = 0.01
    steps = 200
    n = 1
    param_shape = [2, 2, 3]
    shots = 1

    H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])

    # some basic Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    init_params = np.random.uniform(low=0,
                                    high=2 * np.pi,
                                    size=param_shape).astype(np.float32)

    def pq_circuit(params):
        params = params.reshape(param_shape)
        num_qubits = 2

        machine = pq.CPUQVM()
        qubits = range(num_qubits)
        circuit = StronglyEntanglingTemplate(params, num_qubits=num_qubits)
        qcir = circuit.create_circuit(qubits)
        prog = pq.QProg()
        prog << qcir
        machine.run(prog, 0)
        result = machine.result().get_state_vector()
        return result


The Hamiltonian in this example is a Hermitian matrix, which we can always write as a sum of Pauli matrices.

.. math::

    H = \sum_{i,j=0,1,2,3} a_{i,j} (\sigma_i\otimes \sigma_j),

where

.. math::

    a_{i,j} = \frac{1}{4}\text{tr}[(\sigma_i\otimes \sigma_j )H], ~~ \sigma = \{I, X, Y, Z\}.

Substituting into the formula above gives

.. math::

    H = 4  + 2I\otimes X + 4I \otimes Z - X\otimes X + 5 Y\otimes Y + 2Z\otimes X.

To perform doubly stochastic gradient descent, we simply apply stochastic gradient descent but also uniformly sample a subset of the Hamiltonian expectation terms at each optimization step.
vqe_func_analytic() computes the theoretical gradient with the parameter-shift method, while vqe_func_shots() is the doubly stochastic gradient computation using sampled values and a randomly sampled subset of Hamiltonian expectation terms.

.. code-block::

    terms = np.array([
        2 * np.kron(I, X),
        4 * np.kron(I, Z),
        -np.kron(X, X),
        5 * np.kron(Y, Y),
        2 * np.kron(Z, X),
    ])


    def vqe_func_analytic(input, init_params):
        qstate = pq_circuit(init_params)
        expval = Hermitian_expval(H, qstate, [0, 1], 2)
        return  expval

    def vqe_func_shots(input, init_params):
        qstate = pq_circuit(init_params)
        idx = np.random.choice(np.arange(5), size=n, replace=False)
        A = np.sum(terms[idx], axis=0)
        expval = Hermitian_expval(A, qstate, [0, 1], 2, shots)
        return 4 + (5 / 1) * expval


Parameter optimization with VQNet compares the loss curves. Because doubly stochastic gradient descent computes only part of the Pauli terms of H at each step,
the moving average represents the expected value of the final observable; here we use moving_average().

.. code-block::

    ##############################################################################
    # Optimizing the circuit using gradient descent via the parameter-shift rule:
    qlayer_ana = QuantumLayer(vqe_func_analytic, 2*2*3 )
    qlayer_shots = QuantumLayer(vqe_func_shots, 2*2*3 )
    cost_sgd = []
    cost_dsgd = []

    qlayer_ana.m_para.init_from_tensor(to_tensor(init_params))
    opti_ana = SGD(qlayer_ana.parameters())

    qlayer_shots.m_para.init_from_tensor(to_tensor(init_params))
    opti_shots = SGD(qlayer_shots.parameters())

    for i in range(steps):
        opti_ana.zero_grad()
        loss = qlayer_ana(QTensor([[1.0]]))

        loss.backward()
        cost_sgd.append(loss.item())
        opti_ana._step()

    for i in range(steps+50):
        opti_shots.zero_grad()
        loss = qlayer_shots(QTensor([[1.0]]))

        loss.backward()
        cost_dsgd.append(loss.item())
        opti_shots._step()

    def moving_average(data, n=3):
        ret = np.cumsum(data, dtype=np.float64)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    ta = moving_average(np.array(cost_dsgd), n=50)
    ta = ta[:-26]
    average = np.vstack([np.arange(25, 200),ta ])
    final_param = qlayer_shots.parameters()[0].to_numpy()
    print("Doubly stochastic gradient descent min energy = ", vqe_func_analytic(QTensor([1]),final_param))
    final_param  = qlayer_ana.parameters()[0].to_numpy()
    print("stochastic gradient descent min energy = ", vqe_func_analytic(QTensor([1]),final_param))

    plt.plot(cost_sgd, label="Vanilla gradient descent")
    plt.plot(cost_dsgd, ".", label="Doubly QSGD")
    plt.plot(average[0], average[1], "--", label="Doubly QSGD (moving average)")

    plt.ylabel("Cost function value")
    plt.xlabel("Optimization steps")
    plt.xlim(-2, 200)
    plt.legend()
    plt.show()

    #Doubly stochastic gradient descent min energy =  -4.337801834749975
    #stochastic gradient descent min energy =  -4.531484333030544

.. image:: ./images/dsgd.png
   :width: 600 px
   :align: center

|

Gradient-based pruning
===================================

The example below implements the algorithm from `Towards Efficient On-Chip Training of Quantum Neural Networks <https://openreview.net/forum?id=vKefw-zKOft>`_.
By carefully studying the evolution of parameters in variational quantum circuits, the authors observed that small gradients often have large relative changes or even wrong directions under quantum noise.
Moreover, not all gradient computations are necessary for training, especially for small gradients.
Inspired by this, they propose a probabilistic gradient pruning method that predicts gradients and computes only highly reliable ones.
The method reduces noise effects and saves the number of circuits needed on real quantum machines.

In the gradient based pruning algorithm, the parameter optimization process is divided into an accumulation window and a pruning window. All training epochs are split into a repeated accumulation window followed by a pruning window. The probabilistic gradient pruning method has three important hyperparameters:

    * accumulation window width :math:`\omega_a` ,
    * pruning ratio :math:`r` ,
    * pruning window width :math:`\omega_p` .

In the accumulation window, the algorithm collects gradient information at each training step. At each step of the pruning window, based on the information collected in the accumulation window and the pruning ratio,
the algorithm probabilistically skips some gradient computations.

.. image:: ./images/gbp_arch.png
   :width: 600 px
   :align: center

|

The pruning ratio :math:`r`, the accumulation window width :math:`\omega_a`, and the pruning window width :math:`\omega_p` determine the reliability of the gradient trend estimate.
The time saved by our probabilistic gradient pruning method is therefore :math:`r\tfrac{\omega_p}{\omega_a +\omega_p}\times 100\%`.
Below is an example of gradient pruning on a QVC classification task.

.. code-block::

    import random
    import numpy as np
    import pyqpanda3.core as pq

    from pyvqnet.data import data_generator as dataloader
    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn import Gradient_Prune_Instance
    random.seed(1234)

    qvc_train_data = [
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 0
    ]
    qvc_test_data = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]


    def qvc_circuits(x, weights):
        """
        Quantum circuits run function
        """
        machine = pq.CPUQVM()
        qlist = range(4)
        clist = range(4)
        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits) - 1):
                cir << pq.CNOT(nqubits[i], nqubits[i + 1])
            cir << pq.CNOT(nqubits[len(nqubits) - 1], nqubits[0])
            return cir

        def build_circuit(weights, xx, nqubits):
            def Rot(weights_j, qubits):
                circuit = pq.QCircuit()
                circuit << pq.RZ(qubits, weights_j[0])
                circuit << pq.RY(qubits, weights_j[1])
                circuit << pq.RZ(qubits, weights_j[2])
                return circuit

            def basisstate():
                circuit = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i] == 1:
                        circuit << pq.X(nqubits[i])
                return circuit

            circuit = pq.QCircuit()
            circuit << basisstate()

            for i in range(weights.shape[0]):

                weights_i = weights[i, :, :]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circuit << Rot(weights_j, nqubits[j])
                cnots = get_cnot(nqubits)
                circuit << cnots

            circuit << pq.Z(nqubits[0])

            prog = pq.QProg()

            prog << circuit
            return prog

        weights = weights.reshape([2, 4, 3])
        prog = build_circuit(weights, x, qlist)
        machine.run(prog, 0)
        state = np.array(machine.result().get_state_vector())
        prob0 = np.sum(np.abs(state[0::2])**2)
        prob1 = np.sum(np.abs(state[1::2])**2)
        prob = [float(prob0), float(prob1)]

        return prob


    def qvc_circuits2(x, weights):
        """
        Quantum circuits run function
        """
        machine = pq.CPUQVM()
        qlist = range(4)
        prog = pq.QProg()
        circuit = pq.QCircuit()
        circuit << pq.RZ(qlist[0], x[0])
        circuit << pq.RZ(qlist[1], x[1])
        circuit << pq.CNOT(qlist[0], qlist[1])
        circuit << pq.CNOT(qlist[1], qlist[2])
        circuit << pq.CNOT(qlist[2], qlist[3])
        circuit << pq.RY(qlist[0], weights[0])
        circuit << pq.RY(qlist[1], weights[1])
        circuit << pq.RY(qlist[2], weights[2])

        circuit << pq.CNOT(qlist[0], qlist[1])
        circuit << pq.CNOT(qlist[1], qlist[2])
        circuit << pq.CNOT(qlist[2], qlist[3])
        prog << circuit
        machine.run(prog, 0)
        state = np.array(machine.result().get_state_vector())
        prob0 = np.sum(np.abs(state[0::2])**2)
        prob1 = np.sum(np.abs(state[1::2])**2)
        prob = [float(prob0), float(prob1)]

        return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits, 24)

        def forward(self, x):
            y = self.qvc(x)
            #y = self.qvc2(y)
            return y


    def get_data(dataset_str):
        """
        Tranform data to valid form
        """
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1, 5])
        data = datasets[:, :-1]
        label = datasets[:, -1].astype(int)
        label = np.eye(2)[label].reshape(-1, 2)
        return data, label

    def get_accuracy(result, label):
        result, label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result, axis=1) == np.argmax(label, 1))
        return score


We use the ``Gradient_Prune_Instance`` class with input 24, the number of parameters in the current model. The pruning ratio `prune_ratio` is 0.5,
the accumulation window size `accumulation_window_size` is 4,
and the pruning window `pruning_window_size` is 2.
Whenever the backward pass runs, call the ``step`` function of ``Gradient_Prune_Instance`` before the optimizer ``step``.

.. code-block::

    def run():
        """
        Main run function
        """
        model = Model()

        optimizer = sgd.SGD(model.parameters(), lr=0.5)
        batch_size = 3
        epoch = 10
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()

        datas, labels = get_data("train")
        print(datas)
        print(labels)
        print(datas.shape)


        GBP_HELPER = Gradient_Prune_Instance(param_num = 24,prune_ratio=0.5,accumulation_window_size=4,pruning_window_size=2)
        for i in range(epoch):
            count = 0
            sum_loss = 0
            accuary = 0
            t = 0
            for data, label in dataloader(datas, labels, batch_size, False):
                optimizer.zero_grad()
                data, label = QTensor(data), QTensor(label)

                result = model(data)

                loss_b = loss(label, result)

                loss_b.backward()

                GBP_HELPER.step(model.parameters())
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size
                accuary += get_accuracy(result, label)
                t = t + 1

            print(
                f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}"
            )
        print("start testing..............")
        model.eval()
        count = 0

        test_data, test_label = get_data("test")
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for testd, testl in dataloader(test_data, test_label, test_batch_size):
            testd = QTensor(testd)
            test_result = model(testd)
            test_loss = loss(testl, test_result)
            sum_loss += test_loss
            count += test_batch_size
            accuary += get_accuracy(test_result, testl)
        print(
            f"test:--------------->loss:{sum_loss/count} #####accuracy:{accuary/count}"
        )


    if __name__ == "__main__":

        run()

    # epoch:0, #### loss:0.2255942871173223 #####accuracy:0.5833333333333334
    # epoch:1, #### loss:0.1989427705605825 #####accuracy:1.0
    # epoch:2, #### loss:0.16489211718241373 #####accuracy:1.0
    # epoch:3, #### loss:0.13245886812607446 #####accuracy:1.0
    # epoch:4, #### loss:0.11463981121778488 #####accuracy:1.0
    # epoch:5, #### loss:0.1078591321905454 #####accuracy:1.0
    # epoch:6, #### loss:0.10561319688955943 #####accuracy:1.0
    # epoch:7, #### loss:0.10483601937691371 #####accuracy:1.0
    # epoch:8, #### loss:0.10457512239615123 #####accuracy:1.0
    # epoch:9, #### loss:0.10448987782001495 #####accuracy:1.0
    # start testing..............
    # test:--------------->loss:[0.3134713] #####accuracy:1.0



Model training with QuantumLayer in VQNet
*****************************************

Below is an example of quantum machine learning with VQNet interfaces such as ``QuantumLayer``.

Model training with QuantumLayer in VQNet
=========================================

This example builds the circuit with pyqpanda3

.. code-block::

    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    import numpy as np
    import os
    from pyvqnet.data import data_generator as dataloader
    from pyvqnet.nn.loss import CategoricalCrossEntropy

    from pyvqnet.tensor.tensor import QTensor
    import random
    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.qnn.pq3 import probs_measure
    random.seed(1234)

    qvc_train_data = [0,1,0,0,1,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 0,
    0, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 0,
    1, 0, 1, 1, 1,
    1, 1, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 0, 1,
    1, 1, 1, 1, 0]
    qvc_test_data= [0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 1, 0, 1,
    0, 0, 1, 1, 0]

    def qvc_circuits(input,weights):

        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir << pq.CNOT(nqubits[i],nqubits[i+1])
            cir << pq.CNOT(nqubits[len(nqubits)-1],nqubits[0])
            return cir

        def build_circuit(weights, xx, nqubits):

            def Rot(weights_j, qubits):
                circuit = pq.QCircuit()
                circuit << pq.RZ(qubits, weights_j[0])
                circuit << pq.RY(qubits, weights_j[1])
                circuit << pq.RZ(qubits, weights_j[2])
                return circuit
            def basisstate():
                circuit = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i]==1:
                        circuit << pq.X(nqubits[i])
                return circuit

            circuit = pq.QCircuit()
            circuit << basisstate()

            for i in range(weights.shape[0]):

                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circuit << Rot(weights_j,nqubits[j])
                cnots = get_cnot(nqubits)
                circuit << cnots

            circuit << pq.Z(nqubits[0])

            prog = pq.QProg()

            prog << circuit
            return prog

        qlist = range(4)
        machine = pq.CPUQVM()
        weights = weights.reshape([2,4,3])
        prog = build_circuit(weights,input,qlist)
        prob = probs_measure(machine,prog,[0])

        return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24)

        def forward(self, x):
            return self.qvc(x)


    def get_data(dataset_str):
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1,5])
        data = datasets[:,:-1]
        label = datasets[:,-1].astype(int)
        label = np.eye(2)[label].reshape(-1,2)
        return data, label

    def get_accuracy(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    def Run():

        model = Model()

        optimizer = sgd.SGD(model.parameters(),lr =0.5)
        batch_size = 3
        epoch = 10
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()

        datas,labels = get_data("train")

        for i in range(epoch):
            count=0
            sum_loss = 0
            accuary = 0
            t = 0
            for data,label in dataloader(datas,labels,batch_size,False):
                optimizer.zero_grad()
                data,label = QTensor(data), QTensor(label)

                result = model(data)

                loss_b = loss(label,result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count+=batch_size
                accuary += get_accuracy(result,label)
                t = t + 1

            print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}")
        print("start testing..............")
        model.eval()
        count = 0
        test_data, test_label = get_data("test")
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for testd,testl in dataloader(test_data,test_label,test_batch_size):
            testd = QTensor(testd)
            test_result = model(testd)
            test_loss = loss(testl,test_result)
            sum_loss += test_loss
            count+=test_batch_size
            accuary += get_accuracy(test_result,testl)
        print(f"test:--------------->loss:{sum_loss/count} #####accuracy:{accuary/count}")

    if __name__=="__main__":

        Run()


Loss and accuracy results:

.. code-block::

	start training..............
	epoch:0, #### loss:[0.20585182] #####accuracy:0.6
	epoch:1, #### loss:[0.17479989] #####accuracy:1.0
	epoch:2, #### loss:[0.12679021] #####accuracy:1.0
	epoch:3, #### loss:[0.11088503] #####accuracy:1.0
	epoch:4, #### loss:[0.10598478] #####accuracy:1.0
	epoch:5, #### loss:[0.10482856] #####accuracy:1.0
	epoch:6, #### loss:[0.10453037] #####accuracy:1.0
	epoch:7, #### loss:[0.10445572] #####accuracy:1.0
	epoch:8, #### loss:[0.10442699] #####accuracy:1.0
	epoch:9, #### loss:[0.10442187] #####accuracy:1.0
	epoch:10, #### loss:[0.10442089] #####accuracy:1.0
	epoch:11, #### loss:[0.10442062] #####accuracy:1.0
	epoch:12, #### loss:[0.10442055] #####accuracy:1.0
	epoch:13, #### loss:[0.10442055] #####accuracy:1.0
	epoch:14, #### loss:[0.10442055] #####accuracy:1.0
	epoch:15, #### loss:[0.10442055] #####accuracy:1.0
	epoch:16, #### loss:[0.10442055] #####accuracy:1.0

	start testing..............
	[0.3132616580]
	test:--------------->loss:QTensor(0.3132616580, requires_grad=True) #####accuracy:1.0
