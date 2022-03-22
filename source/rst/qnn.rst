Quantum Machine Learning Module
==================================

Quantum Computing Layer
----------------------------------

.. _QuantumLayer:

QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QuantumLayer is a package class of automatic derivative module that supports quantum parametric circuits as parameters. You can define a function as an argument, such as ``qprog_with_meansure``, This function needs to contain the quantum circuit defined by pyQPanda: It generally contains coding-circuit, evolution-circuit and measurement-operation.
This QuantumLayer class can be embedded into the hybrid quantum classical machine learning model and minimize the objective function or loss function of the hybrid quantum classical model through the classical gradient descent method.
You can specify the gradient calculation method of quantum circuit parameters in ``QuantumLayer`` by change the parameter ``diff_method``. ``QuantumLayer`` currently supports two  methods, one is ``finite_diff`` and the other is ``parameter-shift`` methods.

The ``finite_diff`` method is one of the most traditional and common numerical methods for estimating function gradient.The main idea is to replace partial derivatives with differences:

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


For the ``parameter-shift`` method we use the objective function, such as:

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

It is theoretically possible to calculate the gradient of parameters about Hamiltonian in a quantum circuit by the more precise method: ``parameter-shift``.

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. autoclass:: pyvqnet.qnn.quantumlayer.QuantumLayer


QuantumLayerV2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are more familiar with pyQPanda syntax, please using QuantumLayerV2 class, you can define the quantum circuits function by using ``qubits``, ``cubits`` and ``machine``, then take it as a argument ``qprog_with_meansure`` of ``QuantumLayerV2``.

.. autoclass:: pyvqnet.qnn.quantumlayer.QuantumLayerV2

NoiseQuantumLayer
^^^^^^^^^^^^^^^^^^^

In the real quantum computer, due to the physical characteristics of the quantum bit, there is always inevitable calculation error. In order to better simulate this error in quantum virtual machine, VQNet also supports quantum virtual machine with noise. The simulation of quantum virtual machine with noise is closer to the real quantum computer. We can customize the supported logic gate type and the noise model supported by the logic gate.
The existing supported quantum noise model is defined in QPanda, reference links `QPANDA2 <https://pyqpanda-toturial.readthedocs.io/zh/latest/NoiseQVM.html>`_ .

We can use NoiseQuantumLayer to define an automatic microclassification of quantum circuits. You can define a function as an argument ``qprog_with_meansure``. This function needs to contain the quantum circuit defined by pyQPanda, as also you need to pass in a argument ``noise_set_config``, by using the pyQPanda interface to set up the noise model.

.. autoclass:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer

Here is an example of ``noise_set_config``, here we add the noise model BITFLIP_KRAUS_OPERATOR where the noise argument ``p=0.01`` to the quantum gate ``RX`` , ``RY`` , ``RZ`` , ``X`` , ``Y`` , ``Z`` , ``H``, etc.

.. code-block::

	def noise_set_config(qvm,q):

		p = 0.01
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Y_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Z_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RX_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RZ_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.HADAMARD_GATE, p)
		qves =[]
		for i in range(len(q)-1):
			qves.append([q[i],q[i+1]])#
		qves.append([q[len(q)-1],q[0]])
		qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE, p, qves)

		return qvm
		
VQCLayer
^^^^^^^^^^^^^^^^^^^^^^^^

Based on the variable quantum circuit(VariationalQuantumCircuit) of pyQPanda, VQNet provides an abstract quantum computing layer called VQCLayer. You just only needs to define a class ``VQC_wrapper``,
We can build machine learning models, bu defining the corresponding quantum gates of circuits and measurement functions based on pyQPanda VariationalQuantumCircuit.

In `VQC_wrapper`, you can use the common logic gate function 'build_common_circuits' to build a sub-circuits of the model with a change in line structure, use the VQG in 'build_vqc_circuits' to build sub-circuits with constant structure and variable parameters,
use the 'run' function to define the circuit operations and measurement.

.. autoclass:: pyvqnet.qnn.quantumlayer.VQC_wrapper

Send the instantiated object 'VQC_wrapper' as a parameter to 'VQCLayer'

.. autoclass:: pyvqnet.qnn.quantumlayer.VQCLayer



Qconv
^^^^^^^^^^^^^^^^^^^^^^^^

Qconv is a quantum convolution algorithm interface.
Quantum convolution operation adopts quantum circuit to carry out convolution operation on classical data, which does not need to calculate multiplication and addition operation, but only needs to encode data into quantum state, and then obtain the final convolution result through derivation operation and measurement of quantum circuit.
Apply for the same number of quantum bits according to the number of input data in the range of the convolution kernel, and then construct a quantum circuit for calculation.

.. image:: ./images/qcnn.png

First we need encoding by inserting :math:`RY` and :math:`RZ` gates on each qubit, then, we build the entanglement circuit through the :math:`CNOT` gate, finally, we constructed the parameter circuit through :math:`U3 gate` .
The sample is as follows:

.. image:: ./images/qcnn_cir.png

.. autoclass:: pyvqnet.qnn.qcnn.qconv.QConv

QLinear
^^^^^^^^^^

QLinear implements a quantum full connection algorithm. Firstly, the data is encoded into the quantum state, and then the final fully connected result is obtained through the derivation operation and measurement of the quantum circuit.

.. image:: ./images/qlinear_cir.png

.. autoclass:: pyvqnet.qnn.qlinear.qlinear.QLinear


Compatiblelayer
^^^^^^^^^^^^^^^^^

VQNet can not only support ``QPANDA`` quantum circuit, but also support other quantum computing frameworks(such as ``Cirq``, ``Qiskit`` etc), These frameworks can be used to build quantum circuits as part of the quantum computation of VQNet hybrid quantum classical optimization.
VQNet provides an automatic differential quantum circuit operation interface ``Compatiblelayer`` . 
The argument to build `Compatiblelayer` requires passing in a class that defines the third-party library quantum circuit and its running and measuring function `run`.
By using ``Compatiblelayer`` , the input of quantum circuit and the automatic differentiation of parameters can be realized by VQNet.
We provide an example using the qiskit circuit: :ref:`my-reference-label`.

.. autoclass:: pyvqnet.qnn.utils.compatible_layer.Compatiblelayer


Quantum gate
----------------------------------

The way to deal with qubits is called quantum gates. Using quantum gates, we consciously evolve quantum states. Quantum gates are the basis of quantum algorithms.

Basic quantum gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In VQNet, we use each logic gate of `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ developed by the original quantum to build quantum circuit and conduct quantum simulation.
The gates currently supported by pyQPanda can be defined in pyQPanda's `quantum gate <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ section.
In addition, VQNet also encapsulates some quantum gate combinations commonly used in quantum machine learning.


BasicEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.BasicEmbeddingCircuit

AngleEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.AngleEmbeddingCircuit

AmplitudeEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit

IQPEmbeddingCircuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.IQPEmbeddingCircuits

RotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.RotCircuit

CRotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.CRotCircuit

CSWAPcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.CSWAPcircuit

Measure the quantum circuit
----------------------------------

expval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.measure.expval

QuantumMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.measure.QuantumMeasure

ProbsMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.measure.ProbsMeasure




