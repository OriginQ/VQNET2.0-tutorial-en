Quantum Machine Learning Demos
==================================

The following are some examples of quantum machine learning algorithms.  

Application of Parameterized Quantum Circuit in Classification Task
----------------------------------------------------------------------

1. QVC demo
^^^^^^^^^^^^^^^^^^

This example uses VQNet to implement the algorithm in the thesis: `Circuit-centric quantum classifiers <https://arxiv.org/pdf/1804.00633.pdf>`_  .
This example is used to determine whether a binary number is odd or even. By encoding the binary number onto the qubit and optimizing the variable parameters in the circuit, 
the z-direction observation of the circuit can indicate whether the input is odd or even.

Quantum circuit
""""""""""""""""""
The variable component sub-circuit usually defines a sub-circuit, which is a basic circuit architecture, and complex variational circuits can be constructed by repeating layers.
Our circuit layer consists of multiple rotating quantum logic gates and ``CNOT`` quantum logic gates that entangle each qubit with its neighboring qubits.
We also need a circuit to encode classical data into a quantum state, so that the output of the circuit measurement is related to the input.
In this example, we encode the binary input onto the qubits in the corresponding order. For example, the input data 1101 is encoded into 4 qubits.

.. math::

    x = 1101 \rightarrow|\psi\rangle=|1101\rangle

.. figure:: ./images/qvc_circuit.png
   :width: 600 px
   :align: center

|

.. code-block::

    import pyqpanda as pq

    def qvc_circuits(input,weights,qlist,clist,machine):

        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir.insert(pq.CNOT(nqubits[i],nqubits[i+1]))
            cir.insert(pq.CNOT(nqubits[len(nqubits)-1],nqubits[0]))
            return cir

        def build_circult(weights, xx, nqubits):
            
            def Rot(weights_j, qubits):
                circult = pq.QCircuit()
                circult.insert(pq.RZ(qubits, weights_j[0]))
                circult.insert(pq.RY(qubits, weights_j[1]))
                circult.insert(pq.RZ(qubits, weights_j[2]))
                return circult
            def basisstate():
                circult = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i] == 1:
                        circult.insert(pq.X(nqubits[i]))
                return circult

            circult = pq.QCircuit()
            circult.insert(basisstate())

            for i in range(weights.shape[0]):
                
                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circult.insert(Rot(weights_j,nqubits[j]))
                cnots = get_cnot(nqubits)  
                circult.insert(cnots) 

            circult.insert(pq.Z(nqubits[0]))
            
            prog = pq.QProg() 
            prog.insert(circult)
            return prog

        weights = weights.reshape([2,4,3])
        prog = build_circult(weights,input,qlist)  
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())
        
        return prob

Model building
""""""""""""""""
We have defined variable quantum circuits ``qvc_circuits`` . 
We hope to use it in our VQNet's automatic differentiation framework, 
to take advantage of VQNet's optimization fucntions for model training. 
We define a Model class, which inherits from the abstract class ``Module``.
The model uses the :ref:`QuantumLayer` class, which is a quantum computing layer that can be automatically differentiated. 
``qvc_circuits`` is the quantum circuit we want to run,
24 is the number of all quantum circuit parameters that need to be trained, 
"cpu" means that pyQPanda's full amplitude simulator is used here, 
and 4 means that 4 qubits need to be applied for.
In the ``forward()`` function, the user defines the logic of the model to run forward.

.. code-block::

    from pyvqnet.nn.module import Module
    from pyvqnet.optim.sgd import SGD
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyqpanda import *
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24,"cpu",4)

        def forward(self, x):
            return self.qvc(x)


Model training and testing
""""""""""""""""""""""""""""
We use pre-generated random binary numbers and their odd and even labels.
The data as follows.

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
    def dataloader(data,label,batch_size, shuffle = True)->np:
        if shuffle:
            for _ in range(len(data)//batch_size):
                random_index = np.random.randint(0, len(data), (batch_size, 1))
                yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
        else:
            for i in range(0,len(data)-batch_size+1,batch_size):
                yield data[i:i+batch_size], label[i:i+batch_size]

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

Model forwarding, loss function calculation,
reverse calculation, optimizer calculation can perform like the general 
neural network training mode,until the number of iterations reaches the preset value.
The training data used is generated above, and the test data is qvc_test_data and train data is qvc_train_data.

.. code-block::

    def get_accuary(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    #Example Model Class
    model = Model()
    #Create a SGD  optimizer to optimize the model's parameters
    optimizer = SGD(model.parameters(),lr =0.1)
    #set batch_size = 3
    batch_size = 3
    #maximum epochs
    epoch = 20
    #model's loss function
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
            accuary += get_accuary(result,label)
            t = t + 1

        print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")


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
        accuary += get_accuary(test_result,testl)
    print(f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")

.. code-block::

    epoch:0, #### loss:0.20194714764753977 #####accuray:0.6666666666666666
    epoch:1, #### loss:0.19724808633327484 #####accuray:0.8333333333333334
    epoch:2, #### loss:0.19266503552595773 #####accuray:1.0
    epoch:3, #### loss:0.18812804917494455 #####accuray:1.0
    epoch:4, #### loss:0.1835678368806839 #####accuray:1.0
    epoch:5, #### loss:0.1789149840672811 #####accuray:1.0
    epoch:6, #### loss:0.17410411685705185 #####accuray:1.0
    epoch:7, #### loss:0.16908332953850427 #####accuray:1.0
    epoch:8, #### loss:0.16382796317338943 #####accuray:1.0
    epoch:9, #### loss:0.15835540741682053 #####accuray:1.0
    epoch:10, #### loss:0.15273457020521164 #####accuray:1.0
    epoch:11, #### loss:0.14708336691061655 #####accuray:1.0
    epoch:12, #### loss:0.14155150949954987 #####accuray:1.0
    epoch:13, #### loss:0.1362930883963903 #####accuray:1.0
    epoch:14, #### loss:0.1314386005202929 #####accuray:1.0
    epoch:15, #### loss:0.12707658857107162 #####accuray:1.0
    epoch:16, #### loss:0.123248390853405 #####accuray:1.0
    epoch:17, #### loss:0.11995399743318558 #####accuray:1.0
    epoch:18, #### loss:0.1171633576353391 #####accuray:1.0
    epoch:19, #### loss:0.11482855677604675 #####accuray:1.0
    [0.3412148654]
    test:--------------->loss:QTensor(None, requires_grad=True) #####accuray:1.0

The following picture illustrates the curve of model's accuracy:

.. figure:: ./images/qvc_accuracy.png
   :width: 600 px
   :align: center

|

2. data re-uploading algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In a neural network, each neuron receives information from all neurons in the upper layer (Figure a). 
In contrast, the single-bit quantum classifier accepts the previous information processing unit and input (Figure b).
For traditional quantum circuits, when the data is uploaded, the result can be obtained directly through several unitary 
transformations :math:`U(\theta_1,\theta_2,\theta_3)`.However, in the Quantum Data Re upLoading (QDRL) task, the data needs to be re-uploaded before every unitary transformation.

                                                                .. centered:: Comparison of QDRL and classic neural network schematics

.. figure:: ./images/qdrl.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Parameterized quantum circuit for Quantum Data Re-upLoading
    """
    import os
    import sys
    from pyvqnet.nn.linear import Linear
    from pyvqnet.qnn.qdrl.vqnet_model import vmodel
    from pyvqnet import _core as vcore
    from pyvqnet.optim import sgd
    from pyvqnet.nn import loss
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    #hyperparameters
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.tensor import tensor
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(42)
    from pyvqnet.nn.module import Module
    num_layers =  3
    params = np.random.uniform(size=(num_layers, 3))

    #Create a Module to define the model
    class Model(Module):

        def __init__(self):

            super(Model, self).__init__()
            self.pqc = vmodel(params.shape)
        def forward(self, x):
            x = self.pqc(x)
            return x

    def circle(samples:int,  reps =  np.sqrt(1/2)) :
        data_x, data_y = [], []
        for i in range(samples):
            x = np.random.rand(2)
            y = [0,1]
            if np.linalg.norm(x) < reps:
                y = [1,0]
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def plot_data(x, y, fig=None, ax=None):
        """
        Plot data with red/blue values for a binary classification.

        Args:
            x (array[tuple]): array of data points as tuples
            y (array[int]): array of data points as tuples
        """
        if fig == None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        reds = y == 0
        blues = y == 1
        ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
        ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    def get_minibatch_data(x_data, label, batch_size):
        for i in range(0,x_data.data.shape[0]-batch_size+1,batch_size):
            yield x_data.data.select(["{}:{}".format(i,i+batch_size)]), 
            label.data.select(["{}:{}".format(i,i+batch_size)]).reshape([batch_size,2])

    def get_score(pred, label):
        pred, label = np.array(pred.data), np.array(label.data)
        score = np.sum(np.argmax(pred,axis=1) == np.argmax(label,1))
        return score

    model = Model()
    optimizer = sgd.SGD(model.parameters(),lr =1)

    batch_size = 5
    #train on random genearated samples
    def train():
        model.train()
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))  # 500*3
        x_train, y_train = QTensor(x_train),QTensor(y_train)
        epoch = 10
        print("start training...........")
        for i in range(epoch):
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train,batch_size):
                optimizer.zero_grad()

                data,label = QTensor(data), QTensor(label)

                output = model(data)

                Closs = CategoricalCrossEntropy()
                losss = Closs(label, output)

                losss.backward()
                optimizer._step()

                accuracy += get_score(output,label)
                loss += losss.item()
                print(f"epoch:{i}, train_accuracy:{accuracy}")
                print(f"epoch:{i}, train_loss:{losss.data.getdata()}")
                count += batch_size

            print(f"epoch:{i}, train_accuracy_for_each_batch:{accuracy/count}")
            print(f"epoch:{i}, train_loss_for_each_batch:{loss/count}")

    #test on random genearated samples
    def test():
        model.eval()
        print("start eval...................")
        x_test, y_test = circle(500)
        test_accuracy = 0
        count = 0
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))
        x_test, y_test = QTensor(x_test), QTensor(y_test)
        for test_data, test_label in get_minibatch_data(x_test,y_test, batch_size):

            test_data, test_label = QTensor(test_data),QTensor(test_label)
            output = model(test_data)
            test_accuracy += get_score(output, test_label)
            count += batch_size
        print(f"test_accuracy:{test_accuracy/count}")

    if __name__=="__main__":
        train()
        test()

The following picture illustrates the curve of model's accuracy：

.. figure:: ./images/qdrl_accuracy.png
   :width: 600 px
   :align: center

|

3. VSQL: Variational Shadow Quantum Learning for Classification Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using variable quantum circuits to construct a two-class classification model, 
comparing the classification accuracy with a neural network with similar parameter accuracy, 
the accuracy of the two is similar. The quantity of parameters of quantum circuits is much smaller than that of classical neural networks.
The algorithm is based on the paper: `Variational Shadow Quantum Learning for Classification Model <https://arxiv.org/abs/2012.08288>`_ to 
reproduce.

Following figure shows the architecture of VSQL algorithm:

.. figure:: ./images/vsql_model.PNG
   :width: 600 px
   :align: center

|

Following figures show the local quantum circuits structure on each qubits:

.. figure:: ./images/vsql_0.png
.. figure:: ./images/vsql_1.png
.. figure:: ./images/vsql_2.png
.. figure:: ./images/vsql_3.png
.. figure:: ./images/vsql_4.png
.. figure:: ./images/vsql_5.png
.. figure:: ./images/vsql_6.png
.. figure:: ./images/vsql_7.png
.. figure:: ./images/vsql_8.png

.. code-block::


    """
    Parameterized quantum circuit for VSQL

    """

    import os
    import sys
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    import numpy as np

    import pyqpanda as pq
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt

    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit
    from pyvqnet.nn.linear import Linear

    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

    url_base = 'http://yann.lecun.com/exdb/mnist/'
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

    def circuits_of_vsql(input,weights,qlist,clist,machine):

        n = 10
        n_qsc=2
        depth=1
        weights = weights.reshape([depth + 1, 3, n_qsc])

        def subcir(weights,qlist,depth,n_qsc,n_start):
            cir = pq.QCircuit()

            for i in range(n_qsc):
                cir.insert(pq.RX(qlist[n_start + i],weights[0][0][i] ))
                cir.insert(pq.RY(qlist[n_start + i],weights[0][1][i]))
                cir.insert(pq.RX(qlist[n_start + i],weights[0][2][i]))
            for repeat in range(1, depth + 1):
                for i in range(n_qsc - 1):
                    cir.insert(pq.CNOT(qlist[n_start + i], qlist[n_start + i + 1]))
                cir.insert(pq.CNOT(qlist[n_start + n_qsc - 1],qlist[ n_start]))
                for i in range(n_qsc):
                    cir.insert(pq.RY(qlist[ n_start + i], weights[repeat][1][i]))

            return cir
        def get_pauli_str(n_start, n_qsc=2):
            pauli_str = ','.join('X' + str(i) for i in range(n_start, n_start + n_qsc))
            return {pauli_str:1.0}

        f_i = []

        for st in range(n - n_qsc + 1):
            psd = get_pauli_str(st,n_qsc)
            cir = pq.QCircuit()
            cir.insert(AmplitudeEmbeddingCircuit(input,qlist))
            cir.insert(subcir(weights,qlist,depth,n_qsc,st))
            prog = pq.QProg()
            prog.insert(cir)

            f_ij = expval(machine,prog,psd,qlist)
            f_i.append(f_ij)
        f_i = np.array(f_i)
        return f_i

    #GLOBAL VAR
    n = 10
    n_qsc = 2
    depth = 1
    class QModel(Module):
        def __init__(self):
            super().__init__()
            self.vq = QuantumLayer(circuits_of_vsql,(depth + 1)*3* n_qsc,"cpu",10)
            self.fc = Linear(n - n_qsc + 1, 2)
        def forward(self, x):
            x = self.vq(x)
            x = self.fc(x)

            return x

    class Model(Module):

        def __init__(self):
            super().__init__()
            self.fc1 = Linear(input_channels=28*28,output_channels=2)

        def forward(self, x):

            x = tensor.flatten(x,1)
            x = self.fc1(x)
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

    def show_image():
        image, label = load_mnist()
        for img in range(len(image)):
            plt.imshow(image[img])
            plt.show()

    """
    compared classic fc model
    """
    def run_fc01():

        digits = [0,1]
        x_train, y_train = load_mnist("training_data",digits)
        x_train = x_train / 255


        y_train = y_train.reshape(-1, 1)
        y_train = np.eye(len(digits))[y_train].reshape(-1, len(digits))

        x_test, y_test = load_mnist("testing_data",digits)
        x_test = x_test / 255
        y_test = y_test.reshape(-1, 1)
        y_test = np.eye(len(digits))[y_test].reshape(-1, len(digits))

        x_train = x_train[:500]
        y_train = y_train[:500]

        x_test = x_test[:100]
        y_test = y_test[:100]
        print("model start")
        model = Model()

        optimizer = Adam(model.parameters(),lr=0.01)

        model.train()
        F1 = open("./result/qfcrlt.txt","w")
        for epoch in range(1,20):

            model.train()
            full_loss = 0
            n_loss = 0
            n_eval =0
            batch_size = 128
            correct = 0
            iter = 0
            for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True):#shuffle batch rather than data

                optimizer.zero_grad()
                try:
                    x = x.reshape(batch_size, 1, 28, 28)
                except:
                    x = x.reshape(-1,1,28,28)

                output = model(x)
                iter +=1

                CCEloss = CategoricalCrossEntropy()
                loss = CCEloss( y,output)
                loss.backward()
                optimizer._step()

                full_loss += loss.item()
                n_loss += batch_size
                np_output = np.array(output.data,copy=False)
                mask  = np_output.argmax(1) == y.argmax(1)
                correct += sum(mask)

            print(f"Train Accuracy: {correct/n_loss}%")
            print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
            F1.write(f"{epoch}\t{full_loss / n_loss:.4f}\t{correct/n_loss:.4f}\t")

        # Evaluation
            model.eval()
            print("eval")
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
                x = x.reshape(1, 1, 28, 28)
                output = model(x)

                CCEloss = CategoricalCrossEntropy()
                loss = CCEloss( y,output)
                full_loss += loss.item()
                np_output = np.array(output.data,copy=False)
                mask  = np_output.argmax(1) == y.argmax(1)
                correct += sum(mask)
                n_eval += 1
                n_loss += 1

            print(f"Eval Accuracy: {correct/n_eval}")
            F1.write(f"{full_loss / n_loss:.4f}\t{correct/n_eval:.4f}\n")

        F1.close()
        del model
        print("\ndone\n")

    """
    VQSL MODEL
    """
    def run_VSQL():
        digits = [0,1]
        x_train, y_train = load_mnist("training_data",digits)
        x_train = x_train / 255
        y_train = y_train.reshape(-1, 1)
        y_train = np.eye(len(digits))[y_train].reshape(-1, len(digits))
        x_test, y_test = load_mnist("testing_data",digits)
        x_test = x_test / 255
        y_test = y_test.reshape(-1, 1)
        y_test = np.eye(len(digits))[y_test].reshape(-1, len(digits))

        x_train_list =[]
        x_test_list = []
        for i in range(x_train.shape[0]):
            x_train_list.append(np.pad(x_train[i,:,:].flatten(),(0, 240), constant_values=(0, 0)))
        x_train = np.array(x_train_list)

        for i in range(x_test.shape[0]):
            x_test_list.append(np.pad(x_test[i,:,:].flatten(),(0, 240), constant_values=(0, 0)))

        x_test = np.array(x_test_list)

        x_train = x_train[:500]
        y_train = y_train[:500]

        x_test = x_test[:100]
        y_test = y_test[:100]
        print("model start")
        model = QModel()

        optimizer = Adam(model.parameters(),lr=0.1)

        model.train()
        F1 = open("./result/vqslrlt.txt","w")
        for epoch in range(1,20):

            model.train()
            full_loss = 0
            n_loss = 0
            n_eval =0
            batch_size = 1
            correct = 0
            iter = 0
            for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True):
                optimizer.zero_grad()
                try:
                    x = x.reshape(batch_size, 1024)
                except:
                    x = x.reshape(-1,1024)

                output = model(x)
                iter +=1

                CCEloss = CategoricalCrossEntropy()
                loss = CCEloss( y,output)
                loss.backward()
                optimizer._step()

                full_loss += loss.item()
                n_loss += batch_size
                np_output = np.array(output.data,copy=False)
                mask  = np_output.argmax(1) == y.argmax(1)
                correct += sum(mask)
                print(f"{iter}")
            print(f"Train Accuracy: {correct/n_loss}%")
            print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
            F1.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

        # Evaluation
            model.eval()
            print("eval")
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
                x = x.reshape(1, 1024)
                output = model(x)

                CCEloss = CategoricalCrossEntropy()
                loss = CCEloss( y,output)
                full_loss += loss.item()

                np_output = np.array(output.data,copy=False)
                mask  = np_output.argmax(1) == y.argmax(1)
                correct += sum(mask)
                n_eval += 1
                n_loss += 1

            print(f"Eval Accuracy: {correct/n_eval}")
            F1.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")


        F1.close()
        del model
        print("\ndone vqsl\n")


    if __name__ == '__main__':

        run_VSQL()

The following shows the curve of model's accuacy and loss：

.. figure:: ./images/vsql_cacc.PNG
   :width: 600 px
   :align: center

.. figure:: ./images/vsql_closs.PNG
   :width: 600 px
   :align: center

.. figure:: ./images/vsql_qacc.PNG
   :width: 600 px
   :align: center

.. figure:: ./images/vsql_qloss.PNG
   :width: 600 px
   :align: center

|


Quantum AutoEncoder Demo
----------------------------------

1.Quantum AutoEncoder
^^^^^^^^^^^^^^^^^^^^^^^

The classic autoencoder is a neural network that can learn high-efficiency low-dimensional representations of data in a high-dimensional space. 
The task of the autoencoder is to map x to a low-dimensional point y given an input x, so that x can be recovered from y.
The structure of the underlying autoencoder network can be selected to represent the data in a smaller dimension, thereby effectively compressing the input. 
Inspired by this idea, the model of quantum autoencoder is used to perform similar tasks on quantum data.
Quantum autoencoders are trained to compress specific data sets of quantum states, and classical compression algorithms cannot be used. 
The parameters of the quantum autoencoder are trained using classical optimization algorithms.
We show an example of a simple programmable circuit, which can be trained as an efficient autoencoder. 
We apply our model in the context of quantum simulation to compress the Hubbard model and the ground state of the Hamiltonian.
This algorithm is based on `Quantum autoencoders for efficient compression of quantum data <https://arxiv.org/pdf/1612.02806.pdf>`_ .


QAE quantum circuits：

.. figure:: ./images/QAE_Quantum_Cir.png
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

    url_base = 'http://yann.lecun.com/exdb/mnist/'
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

    def run2():
        ##load dataset

        x_train, y_train = load_mnist("training_data")
        x_train = x_train / 255

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

            # F1.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

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

The QAE error value obtained by running the above code, the loss is 1/fidelity, tending to 1 means the fidelity is close to 1.

.. figure:: ./images/qae_train_loss.png
   :width: 600 px
   :align: center

|

Quantum Circuits Structure Learning Demo
-------------------------------------------

1.Quantum circuits structure learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the quantum circuit structure, the most frequently used quantum gates with parameters are `RZ` ,  `RY` , and  `RX`  gates, but which gate to use under what circumstances is a question worth studying. One method is random selection, but in this case It is very likely that the best results will not be achieved.
The core goal of Quantum circuit structure learning task is to find the optimal combination of quantum gates with parameters.
The approach here is that this set of optimal quantum logic gates should make the loss function to be the minimum.


.. code-block::

    """
    Quantum Circuits Strcture Learning Demo

    """

    import os
    import sys
    
    import pyqpanda as pq
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.nn.module import Module
    import numpy as np
    from pyvqnet._core import Tensor as CoreTensor
    import copy
    from pyvqnet.qnn.measure import expval
    machine = pq.CPUQVM()
    machine.init_qvm()
    nqbits = machine.qAlloc_many(2)

    #genearate candidate quantum gates 
    def gen(param:CoreTensor,generators,qbits,circuit):
        if generators == "X":
            circuit.insert(pq.RX(qbits,param))
        elif generators =="Y":
            circuit.insert(pq.RY(qbits,param))
        else:
            circuit.insert(pq.RZ(qbits,param))
    
    #generate circuits based on candidate quantum gates
    def circuits(params,generators,circuit):
        gen(params[0], generators[0], nqbits[0], circuit)
        gen(params[1], generators[1], nqbits[1], circuit)
        circuit.insert(pq.CNOT(nqbits[0], nqbits[1]))
        prog = pq.QProg()
        prog.insert(circuit)
        return prog

    def ansatz1(params:QTensor,generators):
        circuit = pq.QCircuit()
        params = params.data.getdata()
        prog = circuits(params,generators,circuit)
        return expval(machine,prog,{"Z0":1},nqbits), expval(machine,prog,{"Y1":1},nqbits)

    def ansatz2(params:QTensor,generators):
        circuit = pq.QCircuit()
        params = params.data.getdata()
        prog = circuits(params, generators, circuit)
        return expval(machine,prog,{"X0":1},nqbits)

    #target loss function
    def loss(params,generators):
        Z, Y = ansatz1(params,generators)
        X = ansatz2(params,generators)
        return 0.5 * Y + 0.8 * Z - 0.2 * X

    #rotosolve algorithm to find optimal parameter
    def rotosolve(d, params, generators, cost, M_0):  # M_0 only calculated once
        params[d] = np.pi / 2.0
        M_0_plus = cost(QTensor(params), generators)
        params[d] = -np.pi / 2.0
        M_0_minus = cost(QTensor(params), generators)
        a = np.arctan2(
            2.0 * M_0 - M_0_plus - M_0_minus, M_0_plus - M_0_minus
        )  # returns value in (-pi,pi]
        params[d] = -np.pi / 2.0 - a
        if params[d] <= -np.pi:
            params[d] += 2 * np.pi
        return cost(QTensor(params), generators)

    #rotoselect algorithm to find best circuits structure
    def optimal_theta_and_gen_helper(index,params,generators):
        params[index] = 0.
        M_0 = loss(QTensor(params),generators)#init value
        for kind in ["X","Y","Z"]:
            generators[index] = kind
            params_cost = rotosolve(index, params, generators, loss, M_0)
            if kind == "X" or params_cost <= params_opt_cost:
                params_opt_d = params[index]
                params_opt_cost = params_cost
                generators_opt_d = kind
        return params_opt_d, generators_opt_d

    def rotoselect_cycle(params:np,generators):
        for index in range(params.shape[0]):
            params[index], generators[index] = optimal_theta_and_gen_helper(index,params,generators)
        return params,generators


    params = QTensor(np.array([0.3,0.25]))
    params = params.data.getdata()
    generator = ["X","Y"]
    generators = copy.deepcopy(generator)
    epoch = 20
    state_save = []
    for i in range(epoch):
        state_save.append(loss(QTensor(params), generators))
        params, generators = rotoselect_cycle(params,generators)

    print("Optimal generators are: {}".format(generators))
    steps = np.arange(0, epoch)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    plt.plot(steps, state_save, "o-")
    plt.title("rotoselect")
    plt.xlabel("cycles")
    plt.ylabel("cost")
    plt.yticks(np.arange(-1.25, 0.80, 0.25))
    plt.tight_layout()
    plt.show()

The quantum circuit structure obtained by running the above code contains :math:`RX`, one :math:`RY`

.. figure:: ./images/final_quantum_circuit.png
   :width: 600 px
   :align: center

|

And with the parameters in the quantum gate :math:`\theta_1`, :math:`\theta_2` change,Loss function has different values.

.. figure:: ./images/loss3d.png
   :width: 600 px
   :align: center

|

Hybird Quantum Classic Nerual Network Demo
--------------------------------------------------

1.Hybrid Quantum Classic Neural Network Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Machine learning (ML) has become a successful interdisciplinary field that aims to extract generalizable information from data mathematically. 
Quantum machine learning seeks to use the principles of quantum mechanics to enhance machine learning, and vice versa.
Whether your goal is to enhance classical ML algorithms by outsourcing difficult calculations to quantum computers, 
or use classical ML architectures to optimize quantum algorithms-both fall into the category of quantum machine learning (QML).
In this chapter, we will explore how to partially quantify classical neural networks to create hybrid quantum classical neural networks. 
Quantum circuits are composed of quantum logic gates, and the quantum calculations implemented by 
these logic gates are proved to be differentiable by the paper `Quantum Circuit Learning <https://arxiv.org/abs/1803.00745>`_. 
Therefore, researchers try to put quantum circuits and classical neural network modules together for training on hybrid quantum classical machine learning tasks.
We will write a simple example to implement a neural network model training task using VQNet. 
The purpose of this example is to demonstrate the simplicity of VQNet and encourage ML practitioners to explore the possibilities of quantum computing.


Data Preparation
""""""""""""""""""

We will use `MNIST datasets <http://yann.lecun.com/exdb/mnist/>`_, the most basic neural network handwritten digit database as the classification data.
We first load MNIST and filter data samples containing 0 and 1.
These samples are divided into training data training_data and testing data testing_data, each of which has a dimension of 1*784.

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
    import pyqpanda as pq

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = "http://yann.lecun.com/exdb/mnist/"
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

.. figure:: ./images/mnsit_data_examples.png
   :width: 600 px
   :align: center

|

Construct Quantum Circuits
""""""""""""""""""""""""""""""

In this example, we use the `pyQPanda <https://pyqpanda-tutorial-en.readthedocs.io/en/latest/chapter2/index.html#quantum-circuit>`_ , A simple quantum circuit of 1 qubit is defined. The circuit takes the output of the classical neural network layer as input,encodes quantum data through ``H`` , ``RY``  quantum logic gates, and calculates the expected value of Hamiltonian in the z direction as output.

.. code-block::

    from pyqpanda import *
    import pyqpanda as pq
    def circuit(weights):
        num_qubits = 1
        #Use pyQPanda to create a simulator 
        machine = pq.CPUQVM()
        machine.init_qvm()
        #Use pyQPanda to alloc qubits
        qubits = machine.qAlloc_many(num_qubits)
        #Use pyQPanda to alloc classic bits
        cbits = machine.cAlloc_many(num_qubits)
        #Construct circuits
        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], weights[0]))
        #Construct quantum program
        prog = pq.QProg()
        prog.insert(circuit)
        #Defines measurement
        prog << measure_all(qubits, cbits)

        #run quantum with quantum measurements
        result = machine.run_with_configuration(prog, cbits, 100)
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / 100
        expectation = np.sum(states * probabilities)
        return expectation

.. figure:: ./images/hqcnn_quantum_cir.png
   :width: 600 px
   :align: center

|

Create Hybird Model
""""""""""""""""""""

Since quantum circuits can perform automatic differentiation calculations together with classical neural networks,
Therefore, we can use VQNet's convolutional layer ``Conv2D`` , pooling layer ``MaxPool2D`` , fully connected layer ``Linear`` and
the quantum circuit to build model just now.
The definition of the `Net` and `Hybrid` classes inherit from the VQNet automatic differentiation module ``Module`` 
and the definition of the forward calculation is defined in forward function ``forward()``,
An automatic differentiation Model of convolution, quantum encoding, and measurement of the MNIST data is constructed to obtain the final features required for the classification task.

.. code-block::

    #Quantum computing layer front pass and the definition of gradient calculation function, which need to be inherited from the abstract class Module
    class Hybrid(Module):
        """ Hybrid quantum - Quantum layer definition """
        def __init__(self, shift):
            super(Hybrid, self).__init__()
            self.shift = shift
        def forward(self, input): 
            self.input = input
            expectation_z = circuit(np.array(input.data))
            result = [[expectation_z]]
            requires_grad = input.requires_grad and not QTensor.NO_GRAD
            def _backward(g, input):
                """ Backward pass computation """
                input_list = np.array(input.data)
                shift_right = input_list + np.ones(input_list.shape) * self.shift
                shift_left = input_list - np.ones(input_list.shape) * self.shift

                gradients = []
                for i in range(len(input_list)):
                    expectation_right = circuit(shift_right[i])
                    expectation_left = circuit(shift_left[i])

                    gradient = expectation_right - expectation_left
                    gradients.append(gradient)
                gradients = np.array([gradients]).T
                return gradients * np.array(g)

            nodes = []
            if input.requires_grad:
                nodes.append(QTensor.GraphNode(tensor=input, df=lambda g: _backward(g, input)))
            return QTensor(data=result, requires_grad=requires_grad, nodes=nodes)

    #Model definition
    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)
            self.hybrid = Hybrid(np.pi / 2)
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

.. figure:: ./images/hqcnnmodel.PNG
   :width: 600 px
   :align: center

|

Training and testing
""""""""""""""""""""""

For the hybrid neural network model as shown in the figure below, we calculate the loss function by feeding data into the model iteratively, 
and VQNet will  calculate the gradient of each parameter in the backward calculation automatically, 
and use the optimizer to optimize the parameters until the number of iterations meets the preset value.

.. figure:: ./images/hqcnnarch.PNG
   :width: 600 px
   :align: center

|

.. code-block::

    x_train, y_train, x_test, y_test = data_select(1000, 100)

    #Create a model
    model = Net() 
    #Use adam optimizer
    optimizer = Adam(model.parameters(), lr=0.005)
    #Use cross entropy loss
    loss_func = CategoricalCrossEntropy()

    #train epoches   
    epochs = 10
    train_loss_list = []
    val_loss_list = []
    train_acc_list =[]
    val_acc_list = []


    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        batch_size = 1
        correct = 0
        n_train = 0
        for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):

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

Visualization
"""""""""""""""

The visualization curve of data loss function and accuracy on train and test data.

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


.. figure:: ./images/HQCNNLOSS.png
   :width: 600 px
   :align: center

.. figure:: ./images/HQCNNAccuracy.png
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
            pred = QTensor.argmax(output, [1])
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
            plt.show()

.. figure:: ./images/eval_test.png
   :width: 600 px
   :align: center

|

2.Hybrid quantum classical transfer learning model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We apply a machine learning method called transfer learning to image classifier based on hybrid classical quantum
network. We will write a simple example of integrating PyQPanda with VQNet.Transfer learning is based on general intuition,
that is, if the pre-trained network is good at solving a given problem, it can also be used to solve a different
but related problem with only some additional training.

Quantum partial circuit diagram are illustrated below:

.. figure:: ./images/QTransferLearning_cir.png
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
    import pyqpanda as pq
    from pyqpanda import *
    import matplotlib
    from pyvqnet.nn.module import *
    from pyvqnet.utils.initializer import *
    from pyvqnet.qnn.quantumlayer import QuantumLayer

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

    url_base = 'http://yann.lecun.com/exdb/mnist/'
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


    IF_PLOT = False
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

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # 下载数据
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
            pred = QTensor.argmax(output, [1])
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
            pred = QTensor.argmax(output, [1])
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
                circuit.insert(pq.H(qubits[idx]))
            return circuit

        def Q_RY_layer(qubits, w):
            """Layer of parametrized qubit rotations around the y axis.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit.insert(pq.RY(qubits[idx], element))
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Layer of CNOTs followed by another shifted layer of CNOT.
            """
            # In other words it should apply something like :
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat, qubits, cubits, machine):
            """
            The variational quantum circuit.
            """
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(n_qubits)
            circuit = pq.QCircuit()

            # Reshape weights
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            circuit.insert(Q_H_layer(qubits, n_qubits))

            # Embed features in the quantum node
            circuit.insert(Q_RY_layer(qubits, q_input_features))

            # Sequence of trainable variational layers
            for k in range(q_depth):
                circuit.insert(Q_entangling_layer(qubits, n_qubits))
                circuit.insert(Q_RY_layer(qubits, q_weights[k]))

            # Expectation values in the Z basis
            prog = pq.QProg()
            prog.insert(circuit)

            exp_vals = []
            for position in range(n_qubits):
                pauli_str = "Z" + str(position)
                pauli_map = pq.PauliOperator(pauli_str, 1)
                hamiltion = pauli_map.toHamiltonian(True)
                exp = machine.get_expectation(prog, hamiltion, qubits)
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
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

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
            pred = QTensor.argmax(output, [1])
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
                circuit.insert(pq.H(qubits[idx]))
            return circuit

        def Q_RY_layer(qubits, w):
            """Layer of parametrized qubit rotations around the y axis.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit.insert(pq.RY(qubits[idx], element))
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Layer of CNOTs followed by another shifted layer of CNOT.
            """
            # In other words it should apply something like :
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat, qubits, cubits, machine):
            """
            The variational quantum circuit.
            """
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(n_qubits)
            circuit = pq.QCircuit()

            # Reshape weights
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            circuit.insert(Q_H_layer(qubits, n_qubits))

            # Embed features in the quantum node
            circuit.insert(Q_RY_layer(qubits, q_input_features))

            # Sequence of trainable variational layers
            for k in range(q_depth):
                circuit.insert(Q_entangling_layer(qubits, n_qubits))
                circuit.insert(Q_RY_layer(qubits, q_weights[k]))

            # Expectation values in the Z basis
            prog = pq.QProg()
            prog.insert(circuit)
            exp_vals = []
            for position in range(n_qubits):
                pauli_str = "Z" + str(position)
                pauli_map = pq.PauliOperator(pauli_str, 1)
                hamiltion = pauli_map.toHamiltonian(True)
                exp = machine.get_expectation(prog, hamiltion, qubits)
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
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

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
            pred = QTensor.argmax(output, [1])
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


Loss on training set

.. figure:: ./images/qcnn_transfer_learning_classical.png
   :width: 600 px
   :align: center

|

Run classification on test set

.. figure:: ./images/qcnn_transfer_learning_predict.png
   :width: 600 px
   :align: center

|

3.Hybrid quantum classical Unet network model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Image segmentation Image segmentation is a classical problem in the research of computer vision and has become a hot
spot in the field of image understanding. Image segmentation an important part of image understanding, and one of the most difficult problems in image processing.
The so-called image segmentation refers to the segmentation based on gray, color and spatial texture The image
is divided into several disjoint regions by features such as theory and geometry, so that these features show
consistency or similarity in the same region and obvious differences between different regions. In short,
it is to give a picture and classify each pixel on the picture. Separate the pixel regions belonging
to different objects. `Unet <https://arxiv.org/abs/1505.04597>`_ is a classical image segmentation algorithm.

Here, we explore how to partially quantify the classical neural network to create a hybrid quantum classical
`QUnet`  neural network. We will write a simple example of integrating `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ with `VQNet` .
Qunet is mainly used to solve the technology of image segmentation.



Data preparation
"""""""""""""""""""""

We will use the data of `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>`_  official library as image segmentation data. These samples are divided
into training data training_data and test data testing_data.

.. figure:: ./images/Unet_data_imshow.png
   :width: 600 px
   :align: center

|

Constructing quantum circuits
""""""""""""""""""""""""""""""""
In this example, we define a quantum circuit using pyqpanda of the source quantum. The input 3-channel color
image data is compressed into a single channel gray image and stored, and then the feature of the data is
extracted and dimensionality reduced by quantum convolution operation.


.. figure:: ./images/qunet_cir.png
   :width: 600 px
   :align: center

|

Import necessary libraries and functions

.. code-block::

    import os
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.conv import Conv2D, ConvT2D
    from pyvqnet.nn import activation as F
    from pyvqnet.nn.batch_norm import BatchNorm2d
    from pyvqnet.nn.loss import BinaryCrossEntropy
    from pyvqnet.optim.adam import Adam

    from pyvqnet.tensor import tensor
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyqpanda import *
    from pyvqnet.utils.storage import load_parameters, save_parameters

    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt

    import cv2

Preprocessing data

.. code-block::

    # Preprocessing data
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
                temp_data = grayimg.reshape(temp_data.shape[0], temp_data.shape[0], 1)
                self.x_data.append(temp_data)

                label_data = cv2.imread(self.path+"/labels" + '/' +list_path[i].split(".")[0] + ".png", cv2.IMREAD_COLOR)
                label_data = cv2.resize(label_data, (128, 128))

                label_data = cv2.cvtColor(label_data, cv2.COLOR_BGR2GRAY)
                label_data = label_data.reshape(label_data.shape[0], label_data.shape[0], 1)
                self.y_label.append(label_data)

            return self.x_data, self.y_label

        def read(self):
            self.x_data, self.y_label = self.processing()
            x_data = np.array(self.x_data)
            y_label = np.array(self.y_label)

            return x_data, y_label

    # Quantum coding circuit
    class QCNN_:
        def __init__(self, image):
            self.image = image

        def encode_cir(self, qlist, pixels):
            cir = pq.QCircuit()
            for i, pix in enumerate(pixels):
                theta = np.arctan(pix)
                phi = np.arctan(pix**2)
                cir.insert(pq.RY(qlist[i], theta))
                cir.insert(pq.RZ(qlist[i], phi))
            return cir

        def entangle_cir(self, qlist):
            k_size = len(qlist)
            cir = pq.QCircuit()
            for i in range(k_size):
                ctr = i
                ctred = i+1
                if ctred == k_size:
                    ctred = 0
                cir.insert(pq.CNOT(qlist[ctr], qlist[ctred]))
            return cir

        def qcnn_circuit(self, pixels):
            k_size = len(pixels)
            machine = pq.MPSQVM()
            machine.init_qvm()
            qlist = machine.qAlloc_many(k_size)
            cir = pq.QProg()

            cir.insert(self.encode_cir(qlist, np.array(pixels) * np.pi / 2))
            cir.insert(self.entangle_cir(qlist))

            result0 = machine.prob_run_list(cir, [qlist[0]], -1)
            result1 = machine.prob_run_list(cir, [qlist[1]], -1)
            result2 = machine.prob_run_list(cir, [qlist[2]], -1)
            result3 = machine.prob_run_list(cir, [qlist[3]], -1)

            result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
            machine.finalize()
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

Constructing hybrid classical quantum neural network
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

According to the Unet network framework, we use the `VQNet` framework to build the classic network part.
The down-sampling neural network layer is used to reduce the dimension and extract features;
The up-sampling neural network layer is used to restore the dimension; The up and down sampling layers
are connected through concatenate for feature fusion.


.. figure:: ./images/Unet.png
   :width: 600 px
   :align: center

|

.. code-block::

    # Definition of down sampling neural network layer
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
                                padding="same")
            self.BatchNorm2d3 = BatchNorm2d(out_ch)
            self.Relu3 = F.ReLu()

        def forward(self, x):
            """
            :param x:
            :return: out(Output to deep)，out_2(enter to next level)，
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

    # Definition of up sampling neural network layer
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
                                 padding="same")
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

    # Unet overall network architecture
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
            cat_out5 = tensor.concatenate([out5, out_4], axis=1)
            out6 = self.u2(cat_out5)
            cat_out6 = tensor.concatenate([out6, out_3], axis=1)
            out7 = self.u3(cat_out6)
            cat_out7 = tensor.concatenate([out7, out_2], axis=1)
            out8 = self.u4(cat_out7)
            cat_out8 = tensor.concatenate([out8, out_1], axis=1)
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
""""""""""""""""""""""""""""""

Similar to the training of classical neural network model,
we also need to instantiate the model, define the loss function and optimizer, and define the whole training and
testing process. For the hybrid neural network model as shown in the figure below, we calculate the loss value in
forward function the gradient of each parameter in
reverse calculation automatically, and use the optimizer to optimize the parameters until the number of
iterations meets the preset value.


.. code-block::

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
    # PREPROCESS = True
    PREPROCESS = False
    
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
            x_img = QTensor(x)
            x_img_Qtensor = tensor.unsqueeze(x_img, 0)
            y_img = QTensor(y)
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
""""""""""""""""""""""""""

The loss function curve of training data is displayed and saved, and the test data results are saved.

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
        x_img = QTensor(x1)
        x_img_Qtensor = tensor.unsqueeze(x_img, 0)
        y_img = QTensor(y1)
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
        plt.savefig("./result/" + str(i) + "_1" + ".jpg")
    print("end!")

Loss on training set

.. figure:: ./images/qunet_train_loss.png
   :width: 600 px
   :align: center

|

Run classification on test set

.. figure:: ./images/qunet_eval_1.jpg
   :width: 600 px
   :align: center

.. figure:: ./images/qunet_eval_2.jpg
   :width: 600 px
   :align: center

.. figure:: ./images/qunet_eval_3.jpg
   :width: 600 px
   :align: center

|

Unsupervised learning
-------------------------

1 Quantum Kmeans
^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1 Introduction
""""""""""""""""""""

Clustering algorithm is a typical unsupervised learning algorithm, which is mainly used to automatically classify similar samples into one class. In the clustering algorithm, samples are divided into different categories according to the similarity between samples. For different similarity calculation methods, different clustering results will be obtained. The common similarity calculation method is Euclidean distance method. What we want to show is the quantum k-means algorithm. K-means algorithm is a distance based clustering algorithm. It takes distance as the evaluation index of similarity, that is, the closer the distance between two objects, the greater the similarity. The algorithm considers that clusters are composed of objects close to each other, so compact and independent clusters are the ultimate goal.

Quantum kmeans quantum machine learning model can also be developed in VQNet. An example of the quantum kmeans clustering task is given below. Through the quantum circuit, we can construct a measurement that is positively correlated with the Euclidean distance of the variables of classical machine learning, so as to achieve the goal of finding the nearest neighbor.


1.2 Introduction to algorithm principle
"""""""""""""""""""""""""""""""""""""""""""""""

The implementation of quantum k-means algorithm mainly uses swap test to compare the distance between input data points. Randomly select k points from N data points as centroids, measure the distance from each point to each centroid, assign it to the nearest centroid class, recalculate the centroid of each class, and iterate 2 to 3 steps until the new centroid is equal to or less than the specified threshold. In our example, we select 100 data points and 2 centroids, and use cswap circuit to calculate the distance.
Finally, we obtained two data point clusters. :math:`|0\rangle` is an auxiliary bit, through the H logic gate, the qubit will become :math:`\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)`. Under the control of  :math:`|1\rangle` qubit,
The quantum circuit will flip :math:`|x\rangle` and :math:`|y\rangle​` . Finally get the result:

.. math::

    |0_{anc}\rangle |x\rangle |y\rangle \rightarrow \frac{1}{2}|0_{anc}\rangle(|xy\rangle + |yx\rangle) + \frac{1}{2}|1_{anc}\rangle(|xy\rangle - |yx\rangle)

If we measure the auxiliary qubit separately, the probability of the final state of the ground state :math:`|1\rangle` is:

.. math::

    P(|1_{anc}\rangle) = \frac{1}{2} - \frac{1}{2}|\langle x | y \rangle|^2

The Euclidean distance between two quantum states is as follows:

.. math::

    Euclidean \ distance = \sqrt{(2 - 2|\langle x | y \rangle|)}

Visible measurement qubit :math:`|1\rangle` ​ is positively correlated with Euclidean distance. The quantum circuit of this algorithm is as follows：

.. figure:: ./images/Kmeans.jpg
   :width: 600 px
   :align: center

|

1.3 VQNet  implementation
"""""""""""""""""""""""""""

1.3.1 Environmental preparation
***********************************

The environment adopts Python 3 8. It is recommended to use CONDA for environment configuration. It comes with numpy, SciPy, Matplotlib, sklearn and other toolkits for easy use. If the python environment is adopted, relevant packages need to be installed, and the following environment pyvqnet needs to be prepared


1.3.2 Data preparation
**************************

The data is randomly generated by make_blobs under SciPy, and the function is defined to generate Gaussian distribution data.


.. code-block::

    import math
    import numpy as np
    from pyvqnet.tensor import QTensor, zeros
    import pyvqnet.tensor as tensor
    import pyqpanda as pq
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  #pylint:disable=bare-except
        print("Can not use matplot TkAgg")
        pass
    # According to the data amount n of the data, the cluster center K and the data standard deviation std return the corresponding data point and the cluster center point
    def get_data(n, k, std):
        data = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=std, random_state=100)
        points = data[0]
        centers = data[1]
        return points, centers

1.3.3 Quantum circuit
*************************

Constructing quantum circuits using VQNet

.. code-block::

    # The input quantum gate rotation angle is calculated according to the input coordinate point d (x, y)
    def get_theta(d):
        x = d[0]
        y = d[1]
        theta = 2 * math.acos((x.item() + y.item()) / 2.0)
        return theta

    # The quantum circuit is constructed according to the input quantum data points
    def qkmeans_circuits(x, y):

        theta_1 = get_theta(x)
        theta_2 = get_theta(y)

        num_qubits = 3
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        cbits = machine.cAlloc_many(num_qubits)
        circuit = pq.QCircuit()

        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.H(qubits[1]))
        circuit.insert(pq.H(qubits[2]))

        circuit.insert(pq.U3(qubits[1], theta_1, np.pi, np.pi))
        circuit.insert(pq.U3(qubits[2], theta_2, np.pi, np.pi))

        circuit.insert(pq.SWAP(qubits[1], qubits[2]).control([qubits[0]]))

        circuit.insert(pq.H(qubits[0]))

        prog = pq.QProg()
        prog.insert(circuit)
        prog << pq.Measure(qubits[0], cbits[0])
        prog.insert(pq.Reset(qubits[0]))
        prog.insert(pq.Reset(qubits[1]))
        prog.insert(pq.Reset(qubits[2]))

        result = machine.run_with_configuration(prog, cbits, 1024)

        data = result

        if len(data) == 1:
            return 0.0
        else:
            return data['001'] / 1024.0

1.3.4 Data visualization
********************************

Visual calculation of relevant clustering data

.. code-block::

    # Visualization of scatter points and cluster centers
    def draw_plot(points, centers, label=True):
        points = np.array(points)
        centers = np.array(centers)
        if label==False:
            plt.scatter(points[:,0], points[:,1])
        else:
            plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

1.3.5 Cluster calculation
*****************************

Calculate the cluster center of relevant cluster data

.. code-block::

    # Randomly generate cluster center points
    def initialize_centers(points,k):
        return points[np.random.randint(points.shape[0],size=k),:]


    def find_nearest_neighbour(points, centroids):
        n = points.shape[0]
        k = centroids.shape[0]

        centers = zeros([n])

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

        k = int(tensor.max(centers).item()) + 1
        centroids = tensor.zeros([k, 2])
        for i in range(k):

            cur_i = centers == i

            x = points[:,0]
            x = x[cur_i]
            y = points[:,1]
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


    def qkmean_run():
        n = 100  # number of data points
        k = 3  # Number of centers
        std = 2  # std of datapoints

        points, o_centers = get_data(n, k, std)  # dataset

        points = preprocess(points)  # Normalize dataset

        centroids = initialize_centers(points, k)  # Intialize centroids

        epoch = 9
        points = QTensor(points)
        centroids = QTensor(centroids)
        plt.figure()
        draw_plot(points.data, o_centers,label=False)

        for i in range(epoch):
                centers = find_nearest_neighbour(points, centroids)  # find nearest centers
                centroids = find_centroids(points, centers)  # find centroids

        plt.figure()
        draw_plot(points.data, centers.data)

    # Run program entry
    if __name__ == "__main__":
        qkmean_run()


1.3.6 Data distribution before clustering
**************************************************

.. figure:: ./images/ep_1.png
   :width: 600 px
   :align: center

|

1.3.7 Data distribution after clustering
**************************************************

.. figure:: ./images/ep_9.png
   :width: 600 px
   :align: center

|


Model training using quantum computing layer in VQNet
----------------------------------------------------------

The following are some examples of using VQNet inrerface for quantum machine learning ``QuantumLayer`` ，``NoiseQuantumLayer`` ，``VQCLayer``.

Model training using quantumlayer in VQNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    import sys,os
    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    import numpy as np
    import os
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.loss import CategoricalCrossEntropy

    from pyvqnet.tensor.tensor import QTensor
    import random
    import pyqpanda as pq
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyqpanda import *
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

    def qvc_circuits(input,weights,qlist,clist,machine):

        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir.insert(pq.CNOT(nqubits[i],nqubits[i+1]))
            cir.insert(pq.CNOT(nqubits[len(nqubits)-1],nqubits[0]))
            return cir

        def build_circult(weights, xx, nqubits):

            def Rot(weights_j, qubits):
                circult = pq.QCircuit()
                circult.insert(pq.RZ(qubits, weights_j[0]))
                circult.insert(pq.RY(qubits, weights_j[1]))
                circult.insert(pq.RZ(qubits, weights_j[2]))
                return circult
            def basisstate():
                circult = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i]==1:
                        circult.insert(pq.X(nqubits[i]))
                return circult

            circult = pq.QCircuit()
            circult.insert(basisstate())

            for i in range(weights.shape[0]):

                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circult.insert(Rot(weights_j,nqubits[j]))
                cnots = get_cnot(nqubits)
                circult.insert(cnots)

            circult.insert(pq.Z(nqubits[0]))

            prog = pq.QProg()

            prog.insert(circult)
            return prog

        weights = weights.reshape([2,4,3])
        prog = build_circult(weights,input,qlist)
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())

        return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24,"cpu",4)

        def forward(self, x):
            return self.qvc(x)

    def dataloader(data,label,batch_size, shuffle = True)->np:
        if shuffle:
            for _ in range(len(data)//batch_size):
                random_index = np.random.randint(0, len(data), (batch_size, 1))
                yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
        else:
            for i in range(0,len(data)-batch_size+1,batch_size):
                yield data[i:i+batch_size], label[i:i+batch_size]

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

    def get_accuary(result,label):
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
        print(datas)
        print(labels)
        print(datas.shape)
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
                accuary += get_accuary(result,label)
                t = t + 1

            print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")
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
            accuary += get_accuary(test_result,testl)
        print(f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")

    if __name__=="__main__":

        Run()

Loss and accuracy results of the run:

.. code-block::
	
	start training..............
	epoch:0, #### loss:[0.20585182] #####accuray:0.6
	epoch:1, #### loss:[0.17479989] #####accuray:1.0
	epoch:2, #### loss:[0.12679021] #####accuray:1.0
	epoch:3, #### loss:[0.11088503] #####accuray:1.0
	epoch:4, #### loss:[0.10598478] #####accuray:1.0
	epoch:5, #### loss:[0.10482856] #####accuray:1.0
	epoch:6, #### loss:[0.10453037] #####accuray:1.0
	epoch:7, #### loss:[0.10445572] #####accuray:1.0
	epoch:8, #### loss:[0.10442699] #####accuray:1.0
	epoch:9, #### loss:[0.10442187] #####accuray:1.0
	epoch:10, #### loss:[0.10442089] #####accuray:1.0
	epoch:11, #### loss:[0.10442062] #####accuray:1.0
	epoch:12, #### loss:[0.10442055] #####accuray:1.0
	epoch:13, #### loss:[0.10442055] #####accuray:1.0
	epoch:14, #### loss:[0.10442055] #####accuray:1.0
	epoch:15, #### loss:[0.10442055] #####accuray:1.0
	epoch:16, #### loss:[0.10442055] #####accuray:1.0

	start testing..............
	[0.3132616580]
	test:--------------->loss:QTensor(None, requires_grad=True) #####accuray:1.0


Model training using NoiseQuantumLayer in VQNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``NoiseQuantumLayer`` to build and train noisy quantum circuits using QPanda's noise virtual machine.

An example of a complete noisy quantum machine learning model is as follows:

.. code-block::

    import os
    import numpy as np

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor

    import pyqpanda as pq
    from pyqpanda import *
    from pyvqnet.qnn.quantumlayer import NoiseQuantumLayer
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import time
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

    url_base = 'http://yann.lecun.com/exdb/mnist/'
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

    #use qpanda to create quantum circuits
    def circuit(weights,param,qubits,cbits,machine):

        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], weights[0]))
        prog = pq.QProg()
        prog.insert(circuit)
        prog << measure_all(qubits, cbits)
        result = machine.run_with_configuration(prog, cbits, 100)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        # Compute probabilities for each state
        probabilities = counts / 100
        # Get state expectation
        expectation = np.sum(states * probabilities)
        return expectation



    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)

            self.hybrid = NoiseQuantumLayer(circuit,1,"noise",1)
            self.fc3 = Linear(input_channels=1, output_channels=2)


        def forward(self, x):
            x = F.ReLu()(self.conv1(x))
            x = self.maxpool1(x)
            x = F.ReLu()(self.conv2(x))
            x = self.maxpool2(x)
            x = tensor.flatten(x, 1)
            x = F.ReLu()(self.fc1(x))
            x = self.fc2(x)
            x = self.hybrid(x)
            x = self.fc3(x)

            return x

The model is a hybrid quantum circuit and classical network model, in which the quantum circuit 
part uses ``NoiseQuantumLayer`` to simulate the quantum circuit plus noise model. This model 
is used to classify 0 and 1 handwritten digits in MNIST database.

.. code-block::

	def load_mnist(dataset="training_data", digits=np.arange(2), path="..//..//data//MNIST_data"):         # load data
		import os, struct
		from array import array as pyarray
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
		x_train, y_train = load_mnist("training_data")  # load train data
		x_test, y_test = load_mnist("testing_data")
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

	if __name__=="__main__":
		x_train, y_train, x_test, y_test = data_select(100, 50)
		# train sample:200
		model = Net()
		optimizer = Adam(model.parameters(), lr=0.005)
		loss_func = CategoricalCrossEntropy()

		epochs = 10
		loss_list = []
		eval_loss_list = []
		model.train()

		eval_time = []
		F1 = open("hqcnn_noise_train_rlt.txt","w")
		F2 = open("hqcnn_noise_eval_rlt.txt","w")
		for epoch in range(1, epochs):
			total_loss = []
			iter  = 0
			correct = 0
			n_train = 0
			for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):
				iter +=1
				
				start_time = time.time()
				x = x.reshape(-1, 1, 28, 28)

				optimizer.zero_grad()
				# Forward pass
				output = model(x)
	  
				# Calculating loss
				loss = loss_func(y, output)  # target output
				loss_np = np.array(loss.data)
				np_output = np.array(output.data, copy=False)
				mask = (np_output.argmax(1) == y.argmax(1))
				correct += np.sum(np.array(mask))
				n_train += 1
				
				# Backward pass
				loss.backward()
				# Optimize the weights
				optimizer._step()
				total_loss.append(loss_np)

			loss_list.append(np.sum(total_loss) / len(total_loss))
			print(100. * (epoch + 1) / epochs)
			print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
			F1.writelines(f"{epoch},{loss_list[-1]},{correct/n_train}\n")

			model.eval()
			correct = 0
			total_loss1 = []
			n_eval = 0
			eval_time1 = []
			start_init1 = time.time()
			
			for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
				start_time1 = time.time()
				x = x.reshape(-1, 1, 28, 28)
				output = model(x)
				loss = loss_func(y, output)

				np_output = np.array(output.data, copy=False)
				mask = (np_output.argmax(1) == y.argmax(1))
				correct += np.sum(np.array(mask))
				n_eval += 1
				
				end_time1 = time.time()
				eval_time1.append(end_time1 - start_time1)
				loss_np = np.array(loss.data)
				total_loss1.append(loss_np)
			print(f"Eval Accuracy: {correct / n_eval}")

			end_init1 = time.time()
			print("predict run all epochs time {}".format(end_init1 - start_init1))
			print("predict mean time:{}".format(np.mean(eval_time1)))
			F2.writelines(f"{epoch},{np.sum(total_loss1) / len(total_loss1)},{correct/n_eval}\n")
		F1.close()
		F2.close()
		
Comparing the classification results of machine learning models of noisy quantum circuits and ideal quantum circuits, 
the loss change log and acc change log are as follows:

.. code-block::

    Train Accuracy: 0.715
    epoch:  1
    1 loss is : 0.6519572449
    Eval Accuracy: 0.99
    ##########################
    Train Accuracy: 1.0
    epoch:  2
    2 loss is : 0.4458528900
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  3
    3 loss is : 0.3142367172
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  4
    4 loss is : 0.2259583092
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  5
    5 loss is : 0.1661866951
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  6
    6 loss is : 0.1306252861
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  7
    7 loss is : 0.0996847820
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  8
    8 loss is : 0.0801456261
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  9
    9 loss is : 0.0649107647
    Eval Accuracy: 1.0

|

Model training using VQCLayer in VQNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Origin Quantum's qpanda provides `VariationalQuantumCircuit <https://qpanda-tutorial.readthedocs.io/zh/latest/VQC.html#id1>`_ .
In the circuit, only the doors with variable parameters and constant structure can be composed of ``VariationalQuantumGate``.
VQNet provides the a class ``VQC_wrapper`` , you can use ordinary logic gates in the
function ``build_common_circuits`` refers to the local lines with uncertain line structure in the construction model,

Use vqg in ``build_vqc_circuits`` construct local circuits with unchanged structure and variable parameters.
Use the ``run`` function to define the line operation mode and measurement.


.. code-block::

    """
    using pyqpanda VQC api to build model and train VQNet model demo.

    """
    import sys,os
    import time

    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    import numpy as np
    import os
    from pyvqnet.nn.loss import CategoricalCrossEntropy

    from pyvqnet.tensor.tensor import QTensor
    import random

    from pyvqnet.qnn.quantumlayer import VQCLayer,VQC_wrapper,_array2var
    from pyqpanda import *
    import pyqpanda as pq

    random.seed(1234)
    qvc_train_data = [
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 0
    ]
    qvc_test_data = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]

    class QVC_demo(VQC_wrapper):

        def __init__(self):
            super(QVC_demo, self).__init__()

        def build_common_circuits(self,input,qlists,):
            qc = pq.QCircuit()
            for i in range(len(qlists)):
                if input[i]==1:
                    qc.insert(pq.X(qlists[i]))
            return qc

        def build_vqc_circuits(self,input,weights,machine,qlists,clists):

            def get_cnot(qubits):
                vqc = VariationalQuantumCircuit()
                for i in range(len(qubits)-1):
                    vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[i],qubits[i+1]))
                vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[len(qubits)-1],qubits[0]))
                return vqc

            def build_circult(weights, xx, qubits,vqc):

                def Rot(weights_j, qubits):
                    vqc = VariationalQuantumCircuit()
                    vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[0]))
                    vqc.insert(pq.VariationalQuantumGate_RY(qubits, weights_j[1]))
                    vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[2]))
                    return vqc

                for i in range(2):
                    weights_i = weights[i,:,:]
                    for j in range(len(qubits)):
                        weights_j = weights_i[j]
                        vqc.insert(Rot(weights_j,qubits[j]))
                    cnots = get_cnot(qubits)
                    vqc.insert(cnots)
                vqc.insert(pq.VariationalQuantumGate_Z(qubits[0]))  # pauli z(0)
                return vqc

            weights = weights.reshape([2,4,3])
            vqc = VariationalQuantumCircuit()
            return build_circult(weights, input,qlists,vqc)

        def run(self,vqc,input,machine,qlists,clists):
            """
            a function to get hamilton observable or measurment
            """
            prog = QProg()
            vqc_all = VariationalQuantumCircuit()
            # add encode circuits
            vqc_all.insert(self.build_common_circuits(input,qlists))
            vqc_all.insert(vqc)
            qcir = vqc_all.feed()
            prog.insert(qcir)
            prob = machine.prob_run_dict(prog, qlists[0], -1)
            prob = list(prob.values())

            return prob

    class Model(Module):
        def __init__(self,qvc_vqc):
            super(Model, self).__init__()
            self.qvc = VQCLayer(qvc_vqc,24,"cpu",4)

        def forward(self, x):
            return self.qvc(x)

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

    def dataloader(data,label,batch_size, shuffle = True)->np:
        if shuffle:
            for _ in range(len(data)//batch_size):
                random_index = np.random.randint(0, len(data), (batch_size, 1))
                yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
        else:
            for i in range(0,len(data)-batch_size+1,batch_size):
                yield data[i:i+batch_size], label[i:i+batch_size]

    def get_accuary(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    def Run():
        ### create class for VQC
        qvc_vqc = QVC_demo()
        model = Model(qvc_vqc)

        optimizer = sgd.SGD(model.parameters(),lr =0.5)
        batch_size = 3
        epoch = 20
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()
        PATH = os.path.abspath('train')
        datas,labels = get_data(PATH)
        for i in range(epoch):
            count=0
            sum_loss = 0
            accuary = 0

            for data,label in dataloader(datas,labels,batch_size,False):
                optimizer.zero_grad()
                data,label = QTensor(data), QTensor(label)
                result = model(data)
                loss_b = loss(label,result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count+=batch_size
                accuary += get_accuary(result,label)

            print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuray:{accuary/count}")
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
            accuary += get_accuary(test_result,testl)
        print(f"test:--------------->loss:{sum_loss/count} #####accuray:{accuary/count}")

    if __name__=="__main__":

        Run()

Loss and accuracy results of the run:

.. code-block::

    start training..............
    epoch:0, #### loss:0.22664549748102825 #####accuray:0.5333333333333333
    epoch:1, #### loss:0.20315084457397461 #####accuray:0.6666666666666666
    epoch:2, #### loss:0.1644243836402893 #####accuray:1.0
    epoch:3, #### loss:0.12654326359430948 #####accuray:1.0
    epoch:4, #### loss:0.11026077469189961 #####accuray:1.0
    epoch:5, #### loss:0.10584278305371603 #####accuray:1.0
    epoch:6, #### loss:0.10476383566856384 #####accuray:1.0
    epoch:7, #### loss:0.10450373888015747 #####accuray:1.0
    epoch:8, #### loss:0.10444082617759705 #####accuray:1.0
    epoch:9, #### loss:0.10442551374435424 #####accuray:1.0
    epoch:10, #### loss:0.10442176461219788 #####accuray:1.0
    epoch:11, #### loss:0.10442084868748983 #####accuray:1.0
    epoch:12, #### loss:0.10442061225573222 #####accuray:1.0
    epoch:13, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:14, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:15, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:16, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:17, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:18, #### loss:0.10442055265108745 #####accuray:1.0
    epoch:19, #### loss:0.10442055265108745 #####accuray:1.0
    start testing..............
    [0.3132616580]
    test:--------------->loss:QTensor(None, requires_grad=True) #####accuray:1.0




Quantum models as Fourier series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum computers can be used for supervised learning by treating parametrised quantum circuits as models that map data
inputs to predictions. While a lot of work has been done to investigate practical implications of this approach, many important
theoretical properties of these models remain unknown. Here we investigate how the strategy with which data is encoded into
the model influences the expressive power of parametrised quantum circuits as function approximators.


The paper  `The effect of data encoding on the expressive power of variational quantum machine learning models <https://arxiv.org/pdf/2008.08605.pdf>`_ links common quantum machine learning models designed for near-term quantum computers to Fourier series


1.1Fitting Fourier series with serial Pauli-rotation encoding
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

First we show how quantum models that use Pauli rotations as data-encoding gates can only fit Fourier series up to a certain degree.  For simplicity we will only look at single-qubit circuits:

.. image:: ./images/single_qubit_model.PNG
   :width: 600 px
   :align: center

|

Make input data, define parallel quantum models, and do not perform model training results.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
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
        cir.insert(pq.RX(qubits[0], scaling * x))
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qubits[0], theta[0]))
        cir.insert(pq.RY(qubits[0], theta[1]))
        cir.insert(pq.RZ(qubits[0], theta[2]))
        return cir

    def serial_quantum_model(weights, x, num_qubits, scaling):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()  # outside
        machine.init_qvm()  # outside
        qubits = machine.qAlloc_many(num_qubits)

        for theta in weights[:-1]:
            cir.insert(W(theta, qubits))
            cir.insert(S(scaling, x, qubits))

        # (L+1)'th unitary
        cir.insert(W(weights[-1], qubits))
        prog = pq.QProg()
        prog.insert(cir)

        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
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


The result of running the quantum circuit without training is:

.. image:: ./images/single_qubit_model_result_no_train.PNG
   :width: 600 px
   :align: center

|


Make the input data, define the serial quantum model, and build the training model in combination with the QuantumLayer of the VQNet framework.

.. code-block::

    """
    Quantum Fourier Series Serial
    """
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
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
        cir.insert(pq.RX(qubits[0], x))
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qubits[0], theta[0]))
        cir.insert(pq.RY(qubits[0], theta[1]))
        cir.insert(pq.RZ(qubits[0], theta[2]))
        return cir


    r = 1
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))  # some random initial weights

    x = np.linspace(-6, 6, 70)


    def q_circuits_loop(x, weights, qubits, clist, machine):

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3])

            for theta in weights[:-1]:
                cir.insert(W(theta, qubits))
                cir.insert(S(xx, qubits))

            cir.insert(W(weights[-1], qubits))
            prog = pq.QProg()
            prog.insert(cir)

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str, qubits)
                exp_vals.append(exp2)
                result.append(exp2)
        return result

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.q_fourier_series = QuantumLayer(q_circuits_loop, 6, "CPU", 1)

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

.. image:: ./images/single_qubit_model_circuit.PNG
   :width: 600 px
   :align: center

|

The network training results are:

.. image:: ./images/single_qubit_model_result.PNG
   :width: 600 px
   :align: center

|

The network training loss is:

.. code-block::

    start training..............
    epoch:0, #### loss:0.04852807720773853
    epoch:1, #### loss:0.012945819365559146
    epoch:2, #### loss:0.0009359727291666786
    epoch:3, #### loss:0.00015995280153333625
    epoch:4, #### loss:3.988249877352246e-05


1.2Fitting Fourier series with parallel Pauli-rotation encoding
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

As shown in the paper, we expect similar results to the serial model: a Fourier series of order r can only be fitted if the encoded gate has at least r repetitions in the quantum model. Quantum circuit:

.. image:: ./images/parallel_model.png
   :width: 600 px
   :align: center

|

Make input data, define parallel quantum models, and do not perform model training results.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
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
            cir.insert(pq.RX(q, x))
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[0][i][0]))
            cir.insert(pq.RY(qubits[i], theta[0][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[0][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[1][i][0]))
            cir.insert(pq.RY(qubits[i], theta[1][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[1][i][2]))

        cir.insert(pq.CNOT(qubits[0], qubits[len(qubits) - 1]))
        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i + 1], qubits[i]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[2][i][0]))
            cir.insert(pq.RY(qubits[i], theta[2][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[2][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        return cir

    def parallel_quantum_model(weights, x, num_qubits):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()  # outside
        machine.init_qvm()  # outside
        qubits = machine.qAlloc_many(num_qubits)

        cir.insert(W1(weights[0], qubits))
        cir.insert(S1(x, qubits))

        cir.insert(W1(weights[1], qubits))
        prog = pq.QProg()
        prog.insert(cir)

        exp_vals = []
        for position in range(1):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    r = 3

    trainable_block_layers = 3
    weights = 2 * np.pi * np.random.random(size=(2, trainable_block_layers, r, 3))
    # print(weights)
    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [parallel_quantum_model(weights, xx, r) for xx in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()

The result of running the quantum circuit without training is:

.. image:: ./images/parallel_model_result_no_train.PNG
   :width: 600 px
   :align: center

|


Make the input data, define the parallel quantum model, and build the training model in combination with the QuantumLayer layer of the VQNet framework.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer, QuantumLayerMultiProcess
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  # pylint:disable=bare-except
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
            cir.insert(pq.RX(q, x))
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[0][i][0]))
            cir.insert(pq.RY(qubits[i], theta[0][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[0][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[1][i][0]))
            cir.insert(pq.RY(qubits[i], theta[1][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[1][i][2]))

        cir.insert(pq.CNOT(qubits[0], qubits[len(qubits) - 1]))
        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i + 1], qubits[i]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[2][i][0]))
            cir.insert(pq.RY(qubits[i], theta[2][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[2][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        return cir

    def q_circuits_loop(x, weights, qubits, clist, machine):

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3, 3, 3])

            cir.insert(W1(weights[0], qubits))
            cir.insert(S1(xx, qubits))

            cir.insert(W1(weights[1], qubits))
            prog = pq.QProg()
            prog.insert(cir)

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str, qubits)
                exp_vals.append(exp2)
                result.append(exp2)
        return result


    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()

            self.q_fourier_series = QuantumLayer(q_circuits_loop, 2 * 3 * 3 * 3, "CPU", 1)

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

.. image:: ./images/parallel_model_circuit.PNG
   :width: 600 px
   :align: center

|

The network training results are:

.. image:: ./images/parallel_model_result.PNG
   :width: 600 px
   :align: center

|

The network training loss is:

.. code-block::

    start training..............
    epoch:0, #### loss:0.0037272341538482578
    epoch:1, #### loss:5.271130586635309e-05
    epoch:2, #### loss:4.714951917250687e-07
    epoch:3, #### loss:1.0968826371082763e-08
    epoch:4, #### loss:2.1258629738507562e-10
