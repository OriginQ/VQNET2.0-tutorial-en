Frequently Asked Questions
=============================

**Q: What are the features of VQNet**

Answer: VQNet is a quantum machine learning toolset developed based on the original quantum pyQPanda. VQNet provides a rich and easy-to-use classic neural network computing module interface, which can be easily optimized for machine learning.
The model definition method is consistent with the mainstream machine learning framework, which reduces the learning cost for users.
At the same time, based on the high-performance quantum simulator pyQPanda developed by Origin Quantum, VQNet can also support a large number of qubit operations on personal laptops.
Finally, VQNet also has a wealth of :doc:`./qml_demo` for your reference and learning.

**Q: How to use VQNet to train quantum machine learning models**

Answer: One category of quantum machine learning algorithms is to build differentiable quantum machine learning models based on quantum variational circuits.
VQNet can train such quantum machine learning models using gradient descent. The general steps are as follows: First, on the local computer, the user can construct a virtual machine through ``CPUQVM()`` of pyQPanda, and combine the interface provided in VQNet to construct a quantum, quantum-classical hybrid model ``Module``; secondly, call the ``forward()`` of ``Module`` can perform quantum circuit simulation and classical neural network forward operation according to the user-defined operation mode;
When calling ``backward()`` of ``Module``, the model built by the user can be automatically differentiated like a classic machine learning framework such as PyTorch, and calculate the quantum variation circuit and the parameter gradient in the classical computing layer; finally combine the optimizer The ``step()`` function performs parameter optimization.

In VQNet, we use `parameter-shift <https://arxiv.org/abs/1803.00745>`_ to compute the gradient of the quantum variational circuit. Users can use
The ``QuantumLayer`` and ``QuantumLayerV2`` classes provided by VQNet have encapsulated the automatic differentiation of quantum variational circuits. Users only need to define quantum variational circuits in a certain format as parameters to construct the above classes.
For details, please refer to the relevant interfaces and sample codes in this document.

**Q: In Windows, I encountered an error when installing VQNet: "importError: DLL load failed while importing _core: The specified module could not be found."**

Answer: Users may need to install the VC++ runtime library on Windows.
Refer to https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 to install the appropriate runtime library.
In addition, VQNet currently only supports python3.8,3.9,3.10 version, so please confirm your python version.

**Q: How to call the original quantum cloud and quantum chip for calculation**

Answer: Using pyQPanda's cloud resources, you can use high-performance computer clusters or real quantum computers in quantum circuit simulation, and replace local quantum circuit simulation with cloud computing. In VQNet, if users use ``QuantumLayerV2`` to build variable quantum circuit modules, they can use pyQPanda's cloud service interface ``QCloud()`` interface
Instead of the local full-amplitude simulator ``CPUQVM()``, the measurement function is also modified accordingly, see `QCloudServer <https://pyqpanda-toturial.readthedocs.io/zh/latest/QCloudServer.html>`_ for details.
If the user uses ``QuantumLayer`` to build a variable quantum circuit module, and uses the `machine_type_or_cloud_token` parameter to input the applied QCloud token, the module can build a cloud virtual machine internally.


**Q: Why are the model parameters I defined not updated during training**

Answer: To build a VQNet model, it is necessary to ensure that all modules used in it are differentiable. When a module of the model cannot calculate the gradient, this module and the previous modules cannot use the chain rule to calculate the gradient.
If the user defines a quantum variation circuit, please use the ``QuantumLayer`` and ``QuantumLayerV2`` interfaces provided by VQNet. For classic machine learning modules, you need to use the interfaces defined by :doc:`./QTensor` and :doc:`./nn`. These interfaces encapsulate the function of gradient calculation, and VQNet can perform automatic differentiation.

If the user wants to use a list containing multiple modules as a submodule in `Module`, please do not use the List that comes with python, you need to use pyvqnet.nn.module.ModuleList instead of List. In this way, the training parameters of the sub-modules can be registered to the whole model, enabling automatic differential training. Here are examples:

     Example::

         from pyvqnet. tensor import *
         from pyvqnet.nn import Module,Linear,ModuleList
         from pyvqnet.qnn import ProbsMeasure, QuantumLayer
         import pyqpanda as pq
         def pqctest(input, param, qubits, cubits, m_machine):
             circuit = pq. QCircuit()
             circuit.insert(pq.H(qubits[0]))
             circuit.insert(pq.H(qubits[1]))
             circuit.insert(pq.H(qubits[2]))
             circuit.insert(pq.H(qubits[3]))

             circuit.insert(pq.RZ(qubits[0],input[0]))
             circuit.insert(pq.RZ(qubits[1],input[1]))
             circuit.insert(pq.RZ(qubits[2],input[2]))
             circuit.insert(pq.RZ(qubits[3],input[3]))

             circuit.insert(pq.CNOT(qubits[0],qubits[1]))
             circuit.insert(pq.RZ(qubits[1],param[0]))
             circuit.insert(pq.CNOT(qubits[0],qubits[1]))

             circuit.insert(pq.CNOT(qubits[1],qubits[2]))
             circuit.insert(pq.RZ(qubits[2],param[1]))
             circuit.insert(pq.CNOT(qubits[1],qubits[2]))

             circuit.insert(pq.CNOT(qubits[2],qubits[3]))
             circuit.insert(pq.RZ(qubits[3],param[2]))
             circuit.insert(pq.CNOT(qubits[2],qubits[3]))

             prog = pq.QProg()
             prog. insert(circuit)

             rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
             return rlt_prob


         class M(Module):
             def __init__(self):
                 super(M, self).__init__()
                 #Should be built using ModuleList
                 self.pqc2 = ModuleList([QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 ])
                 #Direct use of list cannot save the parameters in pqc3.
                 #self.pqc3 = [QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 #]
             def forward(self, x, *args, **kwargs):
                 y = self.pqc2[0](x) + self.pqc2[1](x)
                 return y

         mm = M()
         print(mm. state_dict(). keys())

**Q: Why did the original code not run in version 2.0.7**

Answer: In version v2.0.7, we added different data types and dtype attributes to QTensor, and restricted input based on PyTorch. For example, the Emedding layer input needs to be kint64, CategoricalCrossEntropy, CrossEntropyLoss, SoftmaxCrossEntropy, NLL_Loss layers's label for Loss and needs to be kint64.

You can use the 'astype()' interface to convert the type to the specified data type, or initialize the QTensor using the corresponding data type numpy array.