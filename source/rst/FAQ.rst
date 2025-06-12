Frequently Asked Questions
=============================

**Q: What are the features of VQNet**

Answer: VQNet is a quantum machine learning toolset developed based on pyQPanda by Origin Quantum. VQNet provides a rich and easy-to-use interface for classical neural network computing modules, which can be easily optimized for machine learning.
The model definition method is consistent with the mainstream machine learning framework, which reduces the user's learning cost.
At the same time, based on pyQPanda, a high-performance quantum simulator developed by Origin Quantum, VQNet can also support the operation of a large number of quantum bits on personal laptops. Finally, VQNet also has a rich :doc:`./qml_demo` for your reference and learning.

**Q: How to use VQNet to train quantum machine learning models**

Answer: There is a type of quantum machine learning algorithm that builds differentiable quantum machine learning models based on quantum variational circuits.
VQNet can use the gradient descent method to train this type of quantum machine learning model. The general steps are as follows: First, on the local computer, users can build a virtual machine through pyQPanda, and combine the interface provided in VQNet to build a quantum and quantum classical hybrid model ``Module``; secondly, calling ``forward()`` of ``Module`` can perform quantum circuit simulation and classical neural network forward operation according to the user-defined operation mode;
When calling ``backward()`` of ``Module``, the user-built model can be automatically differentiated like classic machine learning frameworks such as PyTorch, and calculate the parameter gradients in quantum variational circuits and classical computing layers; finally, combine the optimizer's ``step()`` function to optimize the parameters.

In VQNet, we use `parameter-shift <https://arxiv.org/abs/1803.00745>`_ to calculate the gradient of quantum variational circuits. Users can use the interface under :ref:`QuantumLayer_pq3` provided by VQNet to encapsulate the automatic differentiation of quantum variational circuits. Users only need to define quantum variational circuits as parameters in a certain format to build the above classes.

In VQNet, we can also use the method based on automatic differentiation to calculate the gradient of quantum variational circuits. Users can use the interface in :ref:`vqc_api` to build a trainable circuit. This circuit does not rely on pyQPanda, but splits the encoding, logic gate evolution, and measurement in the circuit into differentiable operators, so as to achieve the function of calculating the gradient of the parameters. .

For details, please refer to the relevant interfaces and sample codes in this document.

**Q: In Windows, I encountered an error when installing VQNet: "importError: DLL load failed while importing _core: The specified module could not be found."**

Answer: Users may need to install the VC++ runtime library on Windows.
Refer to https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 to install the appropriate runtime library.
In addition, VQNet currently only supports python3.9,3.10,3.11 version, so please confirm your python version.

**Q: How to call the original quantum cloud and quantum chip for calculation**

Answer: You can use Origin Quantum's high-performance computer cluster or real quantum computer in quantum circuit simulation, and use cloud computing to replace local quantum circuit simulation. Please refer to https://qcloud.originqc.com.cn/zh.
In VQNet, users can use ``QuantumBatchAsyncQcloudLayer`` to build a variational quantum circuit module, enter the API KEYS applied on the Origin official website, and submit the task to the real machine for operation.

**Q: Why are the model parameters I defined not updated during training**

Answer: To build a VQNet model, it is necessary to ensure that all modules used in it are differentiable. When a module in the model cannot calculate the gradient, the module and the previous modules will not be able to calculate the gradient using the chain rule.
If the user customizes a quantum variational circuit, please use the interface under :ref:`QuantumLayer_pq3` provided by VQNet. For classic machine learning modules, you need to use the interfaces defined by :doc:`./QTensor` and :doc:`./nn`. These interfaces encapsulate the functions of gradient calculation, and VQNet can perform automatic differentiation.

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

**Q: Does VQNet depend on torch?**

Answer: VQNet does not depend on torch, nor does it automatically install torch.

To use the following features, you need to install torch>=2.4.0 yourself. Since v2.15.0, we support using `torch >=2.4.0 <https://docs.pytorch.org/docs/stable/index.html>`_ as the computing backend for classical neural networks, quantum variational circuits, distributed computing, etc.
After using ``pyvqnet.backends.set_backend("torch")``, the interface remains unchanged, but the ``data`` member variables of VQNet's ``QTensor`` all use ``torch.Tensor`` to store data,
and use torch for computing. The classes under ``pyvqnet.nn.torch`` and ``pyvqnet.qnn.vqc.torch`` inherit from ``torch.nn.Module`` and can form ``torch`` models.