Quantum Machine Learning Module
==================================

Quantum Computing Layer
----------------------------------

.. _QuantumLayer:

QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QuantumLayer is a package class of autograd module that supports ariational quantum circuits. You can define a function as an argument, such as ``qprog_with_measure``, This function needs to contain the quantum circuit defined by pyQPanda: It generally contains coding-circuit, evolution-circuit and measurement-operation.
This QuantumLayer class can be embedded into the hybrid quantum classical machine learning model and minimize the objective function or loss function of the hybrid quantum classical model through the classical gradient descent method.
You can specify the gradient calculation method of quantum circuit parameters in ``QuantumLayer`` by change the parameter ``diff_method``. ``QuantumLayer`` currently supports two methods, one is ``finite_diff`` and the other is ``parameter-shift`` methods.

The ``finite_diff`` method is one of the most traditional and common numerical methods for estimating function gradient.The main idea is to replace partial derivatives with differences:

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


For the ``parameter-shift`` method we use the objective function, such as:

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

It is theoretically possible to calculate the gradient of parameters about Hamiltonian in a quantum circuit by the more precise method: ``parameter-shift``.

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01)

    Abstract calculation module for variational quantum circuits. It simulates a parameterized quantum circuit and gets the measurement result.
    QuantumLayer inherits from Module ,so that it can calculate gradients of circuits parameters,and train variational quantum circuits model or embed variational quantum circuits into hybird quantum and classic model.
    
    This class dos not need you to initialize virtual machine in the ``qprog_with_measure`` function.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda
    :param para_num: `int` - Number of parameter
    :param machine_type_or_cloud_token: qpanda machine type or pyQPANDA QCLOUD token : https://pyqpanda-toturial.readthedocs.io/zh/latest/Realchip.html
    :param num_of_qubits: num of qubits
    :param num_of_cbits: num of classic bits
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :return: a module can calculate quantum circuits .

    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html.

        This function should contain following parameters,otherwise it can not run properly in QuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: array_like input 1-dim classic data

            `param`: array_like input 1-dim quantum circuit's parameters

            `qubits`: qubits allocated by QuantumLayer

            `cbits`: cbits allocated by QuantumLayer.if your circuits does not use cbits,you should also reserve this parameter.

            `m_machine`: simulator created by QuantumLayer

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayer
        import numpy as np
        from pyvqnet.tensor import QTensor
        def pqctest (input,param,qubits,cbits,m_machine):
            circuit = pq.QCircuit()
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
            #print(circuit)

            prog = pq.QProg()
            prog.insert(circuit)
            # pauli_dict  = {'Z0 X1':10,'Y2':-0.543}
            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob

        pqc = QuantumLayer(pqctest,3,"cpu",4,1)
        #classic data as input
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2]] )
        #forward circuits
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

QuantumLayerV2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are more familiar with pyQPanda syntax, please using QuantumLayerV2 class, you can define the quantum circuits function by using ``qubits``, ``cbits`` and ``machine``, then take it as a argument ``qprog_with_measure`` of QuantumLayerV2.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01)

    Abstract calculation module for variational quantum circuits. It simulates a parameterized quantum circuit and gets the measurement result.
    QuantumLayer inherits from Module ,so that it can calculate gradients of circuits parameters,and train variational quantum circuits model or embed variational quantum circuits into hybird quantum and classic model.
    
    To use this module, you need to create your quantum virtual machine and allocate qubits and cbits.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda
    :param para_num: `int` - Number of parameter
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :return: a module can calculate quantum circuits .

    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html.

        This function should contains following parameters,otherwise it can not run properly in QuantumLayerV2.

        Compare to QuantumLayer.you should allocate qubits and simulator: https://pyqpanda-toturial.readthedocs.io/zh/latest/QuantumMachine.html,

        you may also need to allocate cbits if qprog_with_measure needs quantum measure: https://pyqpanda-toturial.readthedocs.io/zh/latest/Measure.html

        qprog_with_measure (input,param)

        `input`: array_like input 1-dim classic data

        `param`: array_like input 1-dim quantum circuit's parameters


    Example::


        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayerV2
        import numpy as np
        from pyvqnet.tensor import QTensor
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# outside
            m_machine.init_qvm()# outside
            qubits = m_machine.qAlloc_many(num_of_qubits)

            circuit = pq.QCircuit()
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
            #print(circuit)

            prog = pq.QProg()
            prog.insert(circuit)
            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob


        pqc = QuantumLayerV2(pqctest,3)

        #classic data as input
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2]] )

        #forward circuits
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]


QuantumLayerMultiProcess
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are more familiar with pyQPanda syntax, please using QuantumLayerMultiProcess class, you can define the quantum circuits function by using ``qubits``, ``cbits`` and ``machine``, then take it as a argument ``qprog_with_measure`` of QuantumLayerMultiProcess.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerMultiProcess(qprog_with_measure, para_num, machine_type_or_cloud_token, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01)

    Abstract calculation module for variational quantum circuits. This class uses multiprocess to accelerate quantum circuit simulation.
    
    It simulates a parameterized quantum circuit and gets the measurement result.
    QuantumLayer inherits from Module ,so that it can calculate gradients of circuits parameters,and train variational quantum circuits model or embed variational quantum circuits into hybird quantum and classic model.

    To use this module, you need to create your quantum virtual machine and allocate qubits and cbits.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda.
    :param para_num: `int` - Number of parameter
    :param machine_type_or_cloud_token: qpanda machine type or pyQPANDA QCLOUD token.
    :param num_of_qubits: num of qubits.
    :param num_of_cbits: num of classic bits.
    :param diff_method: 'parameter_shift' or 'finite_diff'.
    :param delta:  delta for diff.
    :return: a module can calculate quantum circuits .

    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda : https://github.com/OriginQ/QPanda-2.

        This function should contains following parameters,otherwise it can not run properly in QuantumLayerMultiProcess.

        Compare to QuantumLayer.you should allocate qubits and simulator,

        you may also need to allocate cbits if qprog_with_measure needs quantum Measure.

        qprog_with_measure (input,param)

        `input`: array_like input 1-dim classic data

        `param`: array_like input 1-dim quantum circuit's parameters


    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayerMultiProcess
        import numpy as np
        from pyvqnet.tensor import QTensor
        def pqctest (input,param,nqubits,ncubits):
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(nqubits)
            circuit = pq.QCircuit()
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
            #print(circuit)

            prog = pq.QProg()
            prog.insert(circuit)

            rlt_prob = ProbsMeasure([0,2],prog,machine,qubits)
            return rlt_prob


        pqc = QuantumLayerMultiProcess(pqctest,3,"cpu",4,1)
        #classic data as input
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2]] )
        #forward circuits
        rlt = pqc(input)
        grad = QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]


NoiseQuantumLayer
^^^^^^^^^^^^^^^^^^^

In the real quantum computer, due to the physical characteristics of the quantum bit, there is always inevitable calculation error. In order to better simulate this error in quantum virtual machine, VQNet also supports quantum virtual machine with noise. The simulation of quantum virtual machine with noise is closer to the real quantum computer. We can customize the supported logic gate type and the noise model supported by the logic gate.
The existing supported quantum noise model is defined in QPanda `NoiseQVM <https://pyqpanda-toturial.readthedocs.io/zh/latest/NoiseQVM.html>`_ .

We can use ``NoiseQuantumLayer`` to define an automatic microclassification of quantum circuits. ``NoiseQuantumLayer`` supports QPanda quantum virtual machine with noise. You can define a function as an argument ``qprog_with_measure``. This function needs to contain the quantum circuit defined by pyQPanda, as also you need to pass in a argument ``noise_set_config``, by using the pyQPanda interface to set up the noise model.

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None)

    Abstract calculation module for variational quantum circuits. It simulates a parameterized quantum circuit and gets the measurement result.
    QuantumLayer inherits from Module ,so that it can calculate gradients of circuits parameters,and train variational quantum circuits model or embed variational quantum circuits into hybird quantum and classic model.
    
    This module should be initialized with noise model by ``noise_set_config``.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda
    :param para_num: `int` - Number of para_num
    :param machine_type: qpanda machine type
    :param num_of_qubits: num of qubits
    :param num_of_cbits: num of cbits
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :param noise_set_config: noise set function
    :return: a module can calculate quantum circuits with noise model.

    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html.

        This function should contains following parameters,otherwise it can not run properly in NoiseQuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: array_like input 1-dim classic data

            `param`: array_like input 1-dim quantum circuit's parameters

            `qubits`: qubits allocated by NoiseQuantumLayer

            `cbits`: cbits allocated by NoiseQuantumLayer.if your circuits does not use cbits,you should also reserve this parameter.

            `m_machine`: simulator created by NoiseQuantumLayer

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import NoiseQuantumLayer
        import numpy as np
        from pyqpanda import *
        from pyvqnet.tensor import QTensor
        def circuit(weights,param,qubits,cbits,machine):

            circuit = pq.QCircuit()

            circuit.insert(pq.H(qubits[0]))
            circuit.insert(pq.RY(qubits[0], weights[0]))
            circuit.insert(pq.RY(qubits[0], param[0]))
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

        def default_noise_config(qvm,q):

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

        qvc = NoiseQuantumLayer(circuit,24,"noise",1,1,diff_method= "parameter_shift", delta=0.01,noise_set_config = default_noise_config)
        input = QTensor([
            [0.0000000, 1.0000000, 1.0000000, 1.0000000],

            [0.0000000, 0.0000000, 1.0000000, 1.0000000],

            [1.0000000, 0.0000000, 1.0000000, 1.0000000]
            ] )
        rlt = qvc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)

        rlt.backward(grad)
        print(qvc.m_para.grad)

        #[1195.0000000, 105.0000000, 70.0000000, 0.0000000,
        # 45.0000000, -45.0000000, 50.0000000, 15.0000000,
        # -80.0000000, 50.0000000, 10.0000000, -30.0000000,
        # 10.0000000, 60.0000000, 75.0000000, -110.0000000,
        # 55.0000000, 45.0000000, 25.0000000, 5.0000000,
        # 5.0000000, 50.0000000, -25.0000000, -15.0000000]

Here is an example of ``noise_set_config``, here we add the noise model BITFLIP_KRAUS_OPERATOR where the noise argument p=0.01 to the quantum gate ``RX`` , ``RY`` , ``RZ`` , ``X`` , ``Y`` , ``Z`` , ``H``, etc.

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

Based on the variable quantum circuit(VariationalQuantumCircuit) of pyQPanda, VQNet provides an abstract quantum computing layer called ``VQCLayer``.

You just only needs to define a class that inherits from ``VQC_wrapper``, and construct quantum gates of circuits and measurement functions based on pyQPanda ``VariationalQuantumCircuit`` in it.

In ``VQC_wrapper``, you can use the common logic gate function ``build_common_circuits`` to build a sub-circuits of the model with variable circuit's structure, use the VQG in ``build_vqc_circuits`` to build sub-circuits with constant structure and variable parameters,
use the ``run`` function to define the circuit operations and measurement.

.. py:class:: pyvqnet.qnn.quantumlayer.VQC_wrapper

    ``VQC_wrapper`` is a abstract class help to run VariationalQuantumCircuit on VQNet.

    ``build_common_circuits`` function contains circuits may be varaible according to the input.

    ``build_vqc_circuits`` function contains VQC circuits with trainable weights.

    ``run`` function contains run function for VQC.

    Example::

        import pyqpanda as pq
        from pyqpanda import *
        from pyvqnet.qnn.quantumlayer import VQCLayer,VQC_wrapper

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

                    #2,4,3
                    for i in range(2):

                        weights_i = weights[i,:,:]
                        for j in range(len(qubits)):
                            weights_j = weights_i[j]
                            vqc.insert(Rot(weights_j,qubits[j]))
                        cnots = get_cnot(qubits)
                        vqc.insert(cnots)

                    vqc.insert(pq.VariationalQuantumGate_Z(qubits[0]))#pauli z(0)

                    return vqc

                weights = weights.reshape([2,4,3])
                vqc = VariationalQuantumCircuit()
                return build_circult(weights, input,qlists,vqc)

Send the instantiated object ``VQC_wrapper`` as a parameter to ``VQCLayer``

.. py:class:: pyvqnet.qnn.quantumlayer.VQCLayer(vqc_wrapper, para_num, machine_type_or_cloud_token, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01)

    Abstract Calculation module for Variational Quantum Circuits in pyQPanda.Please reference to :https://pyqpanda-toturial.readthedocs.io/zh/latest/VQG.html.

    :param vqc_wrapper: VQC_wrapper class
    :param para_num: `int` - Number of parameter
    :param machine_type: qpanda machine type
    :param num_of_qubits: num of qubits
    :param num_of_cbits: num of cbits
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :return: a module can calculate VQC quantum circuits

    Example::

        import pyqpanda as pq
        from pyqpanda import *
        from pyvqnet.qnn.quantumlayer import VQCLayer,VQC_wrapper

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

                    #2,4,3
                    for i in range(2):

                        weights_i = weights[i,:,:]
                        for j in range(len(qubits)):
                            weights_j = weights_i[j]
                            vqc.insert(Rot(weights_j,qubits[j]))
                        cnots = get_cnot(qubits)
                        vqc.insert(cnots)

                    vqc.insert(pq.VariationalQuantumGate_Z(qubits[0]))#pauli z(0)

                    return vqc

                weights = weights.reshape([2,4,3])
                vqc = VariationalQuantumCircuit()
                return build_circult(weights, input,qlists,vqc)

            def run(self,vqc,input,machine,qlists,clists):

                prog = QProg()
                vqc_all = VariationalQuantumCircuit()
                # add encode circuits
                vqc_all.insert(self.build_common_circuits(input,qlists))
                vqc_all.insert(vqc)
                qcir = vqc_all.feed()
                prog.insert(qcir)
                #print(pq.convert_qprog_to_originir(prog, machine))
                prob = machine.prob_run_dict(prog, qlists[0], -1)
                prob = list(prob.values())

                return prob

        qvc_vqc = QVC_demo()
        VQCLayer(qvc_vqc,24,"cpu",4)

Qconv
^^^^^^^^^^^^^^^^^^^^^^^^

Qconv is a quantum convolution algorithm interface.
Quantum convolution operation adopts quantum circuit to carry out convolution operation on classical data, which does not need to calculate multiplication and addition operation, but only needs to encode data into quantum state, and then obtain the final convolution result through derivation operation and measurement of quantum circuit.
Applies for the same number of quantum bits according to the number of input data in the range of the convolution kernel, and then construct a quantum circuit for calculation.

.. image:: ./images/qcnn.png

First we need encoding by inserting :math:`RY` and :math:`RZ` gates on each qubit, then, we constructed the parameter circuit through :math:`U3` gate and :math:`Z` gate .
The sample is as follows:

.. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "cpu")

    Quantum Convolution module. Replace Conv2D kernal with quantum circuits.Inputs to the conv module are of shape (batch_size, input_channels, height, width) reference `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_.

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param quantum_number: `int` - Size of a single kernel.
    :param stride: `tuple` - Stride, defaults to (1, 1)
    :param padding: `tuple` - Padding, defaults to (0, 0)
    :param kernel_initializer: `callable` - Defaults to normal
    :param machine: `str` - cpu simulation
    :return: a quantum cnn class

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qcnn.qconv import QConv
        x = tensor.ones([1,3,4,4])
        layer = QConv(input_channels=3, output_channels=2, quantum_number=4, stride=(2, 2))
        y = layer(x)
        print(y)

        # [
        # [[[-0.0889078, -0.0889078],
        #  [-0.0889078, -0.0889078]],
        # [[0.7992646, 0.7992646],
        #  [0.7992646, 0.7992646]]]
        # ]

QLinear
^^^^^^^^^^

QLinear implements a quantum full connection algorithm. Firstly, the data is encoded into the quantum state, 
and then the final fully connected result is obtained through the derivation operation and measurement of the quantum circuit.

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels, output_channels, machine: str = 'cpu')

    Quantum Linear module. Inputs to the linear module are of shape (input_channels, output_channels)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of output channels
    :param machine: `str` - cpu simulation
    :return: a quantum linear layer

    Exmaple::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.qlinear import QLinear
        params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]
        m = QLinear(6, 2)
        input = QTensor(params, requires_grad=True)
        output = m(input)
        output.backward()
        print(output)

        # [
        #[0.0568473， 0.1264389]，
        #[0.1524036， 0.1264389]，
        #[0.1524036， 0.1442845]，
        #[0.1524036， 0.1442845]
        # ]

|

Quantum gate
----------------------------------

The way to deal with qubits is called quantum gates. Using quantum gates, we consciously evolve quantum states. Quantum gates are the basis of quantum algorithms.

Basic quantum gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In VQNet, we use each logic gate of `pyQPanda <https://pyqpanda-tutorial-en.readthedocs.io/en/latest/>`__ developed by the original quantum to build quantum circuit and conduct quantum simulation.
The gates currently supported by pyQPanda can be defined in pyQPanda's `quantum gate <https://pyqpanda-tutorial-en.readthedocs.io/en/latest/chapter2/index.html#quantum-logic-gate>`_ section.
In addition, VQNet also encapsulates some quantum gate combinations commonly used in quantum machine learning.


BasicEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.BasicEmbeddingCircuit(input_feat, qlist)

    Encodes n binary features into a basis state of n qubits.

    For example, for ``features=([0, 1, 1])``, the quantum system will be
    prepared in state :math:`|011 \rangle`.

    :param input_feat: binary input of shape ``(n)``
    :param qlist: qlist that the template acts on
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import BasicEmbeddingCircuit
        input_feat = np.array([1,1,0]).reshape([3])
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)

        qlist = m_machine.qAlloc_many(3)
        circuit = BasicEmbeddingCircuit(input_feat,qlist)
        print(circuit)

        #           ┌─┐
        # q_0:  |0>─┤X├
        #           ├─┤
        # q_1:  |0>─┤X├
        #           └─┘

AngleEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.AngleEmbeddingCircuit(input_feat, qubits, rotation: str = 'X')

    Encodes :math:`N` features into the rotation angles of :math:`n` qubits, where :math:`N \leq n`.

    The rotations can be chosen as either : 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` uses the features as angles of RX rotations

    * ``rotation='Y'`` uses the features as angles of RY rotations

    * ``rotation='Z'`` uses the features as angles of RZ rotations

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than qlists, the circuit does not Applies the remaining rotation gates.

    :param input_feat: numpy array which represents paramters
    :param qubits: qubits allocated by pyQPanda
    :param rotation: use what rotation ,default 'X'
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import AngleEmbeddingCircuit
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_qlist = m_machine.qAlloc_many(2)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()

        input_feat = np.array([2.2, 1])
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'X')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Y')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Z')
        print(C)
        pq.destroy_quantum_machine(m_machine)

        #           ┌────────────┐
        # q_0:  |0>─┤RX(2.200000)├
        #           ├────────────┤
        # q_1:  |0>─┤RX(1.000000)├
        #           └────────────┘



        #           ┌────────────┐
        # q_0:  |0>─┤RY(2.200000)├
        #           ├────────────┤
        # q_1:  |0>─┤RY(1.000000)├
        #           └────────────┘



        #           ┌────────────┐
        # q_0:  |0>─┤RZ(2.200000)├
        #           ├────────────┤
        # q_1:  |0>─┤RZ(1.000000)├
        #           └────────────┘

AmplitudeEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat, qubits)

    Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.
    To represent a valid quantum state vector, the L2-norm of ``features`` must be one.

    :param input_feat: numpy array which represents paramters
    :param qubits: qubits allocated by pyQPanda
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit
        input_feat = np.array([2.2, 1, 4.5, 3.7])
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_qlist = m_machine.qAlloc_many(2)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()
        cir = AmplitudeEmbeddingCircuit(input_feat,m_qlist)
        print(cir)
        pq.destroy_quantum_machine(m_machine)

        #                              ┌────────────┐     ┌────────────┐
        # q_0:  |0>─────────────── ─── ┤RY(0.853255)├ ─── ┤RY(1.376290)├
        #           ┌────────────┐ ┌─┐ └──────┬─────┘ ┌─┐ └──────┬─────┘
        # q_1:  |0>─┤RY(2.355174)├ ┤X├ ───────■────── ┤X├ ───────■──────
        #           └────────────┘ └─┘                └─┘

IQPEmbeddingCircuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.IQPEmbeddingCircuits(input_feat, qubits,trep:int = 1)

    Encodes :math:`n` features into :math:`n` qubits using diagonal gates of an IQP circuit.

    The embedding was proposed by `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    The basic IQP circuit can be repeated by specifying ``n_repeats``.

    :param input_feat: numpy array which represents paramters
    :param qubits: qubits allocated by pyQPanda
    :param rep: repeat circuits block
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import IQPEmbeddingCircuits
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        input_feat = np.arange(1,100)
        qlist = m_machine.qAlloc_many(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 1)
        print(circuit)

        #           ┌─┐ ┌────────────┐
        # q_0:  |0>─┤H├ ┤RZ(1.000000)├ ───■── ────────────── ───■── ───■── ────────────── ───■── ────── ────────────── ──────
        #           ├─┤ ├────────────┤ ┌──┴─┐ ┌────────────┐ ┌──┴─┐    │                     │
        # q_1:  |0>─┤H├ ┤RZ(2.000000)├ ┤CNOT├ ┤RZ(2.000000)├ ┤CNOT├ ───┼── ────────────── ───┼── ───■── ────────────── ───■──
        #           ├─┤ ├────────────┤ └────┘ └────────────┘ └────┘ ┌──┴─┐ ┌────────────┐ ┌──┴─┐ ┌──┴─┐ ┌────────────┐ ┌──┴─┐
        # q_2:  |0>─┤H├ ┤RZ(3.000000)├ ────── ────────────── ────── ┤CNOT├ ┤RZ(3.000000)├ ┤CNOT├ ┤CNOT├ ┤RZ(3.000000)├ ┤CNOT├
        #           └─┘ └────────────┘                              └────┘ └────────────┘ └────┘ └────┘ └────────────┘ └────┘

RotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.RotCircuit(para, qubits)

    Arbitrary single qubit rotation.Number of qlist should be 1,and number of parameters should
    be 3

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param para: numpy array which represents paramters [\phi, \theta, \omega]
    :param qubits: qubits allocated by pyQPanda,only accepted single qubits.
    :return: quantum circuits

    Example::

        import pyqpanda as pq
        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.template import RotCircuit
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(1)
        param = np.array([3,4,5])
        c = pyvqnet.qnn.template.RotCircuit(param,m_qlist)
        print(c)
        pq.destroy_quantum_machine(m_machine)

        #           ┌────────────┐ ┌────────────┐ ┌────────────┐
        # q_0:  |0>─┤RZ(5.000000)├ ┤RY(4.000000)├ ┤RZ(3.000000)├
        #           └────────────┘ └────────────┘ └────────────┘

CRotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.CRotCircuit(para, control_qubits, rot_qubits)

    The controlled-Rot operator

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: numpy array which represents paramters [\phi, \theta, \omega]
    :param control_qubits: control qubit allocated by pyQPanda
    :param rot_qubits: Rot qubit allocated by pyQPanda
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.template import CRotCircuit
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(1)
        param = np.array([3,4,5])
        control_qlist = m_machine.qAlloc_many(1)
        c = CRotCircuit(QTensor(param),control_qlist,m_qlist)
        print(c)
        pq.destroy_quantum_machine(m_machine)

        #           ┌────────────┐ ┌────────────┐ ┌────────────┐
        # q_0:  |0>─┤RZ(5.000000)├ ┤RY(4.000000)├ ┤RZ(3.000000)├
        #           └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
        # q_1:  |0>────────■────── ───────■────── ───────■──────


CSWAPcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.CSWAPcircuit(qubits)

    The controlled-swap circuit

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note:: The first qubits provided corresponds to the **control qubit**.

    :param qubits: list of qubits allocated by pyQPanda the first qubits is control qubit. length of qlists have to be 3.
    :return: quantum circuits

    Example::

        from pyvqnet.qnn.template import CSWAPcircuit
        import pyqpanda as pq
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)

        m_qlist = m_machine.qAlloc_many(3)

        c = CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)
        pq.destroy_quantum_machine(m_machine)

        # q_0:  |0>─X─
        #           │
        # q_1:  |0>─■─
        #           │
        # q_2:  |0>─X─



Commonly used quantum circuits
----------------------------------
VQNet provides some quantum circuits commonly used in quantum machine learning research.


HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.ansatz.HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,qubits,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    The implementation of Hardware Efficient Ansatz introduced in the paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    :param n_qubits: Number of qubits.
    :param single_rot_gate_list: A single qubit rotation gate list is constructed by one or several rotation gate that act on every qubit.Currently support Rx, Ry, Rz.
    :param qubits: Qubits allocated by pyqpanda api.
    :param entangle_gate: The non parameterized entanglement gate.CNOT,CZ is supported.default:CNOT.
    :param entangle_rules: How entanglement gate is used in the circuit. ``linear`` means the entanglement gate will be act on every neighboring qubits. ``all`` means the entanglment gate will be act on any two qbuits. Default: ``linear``.
    :param depth: The depth of ansatz, default:1.

    Example::

        import pyqpanda as pq
        from pyvqnet.tensor import QTensor,tensor
        from pyvqnet.qnn import HardwareEfficientAnsatz
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlist = machine.qAlloc_many(4)
        c = HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                    qlist,
                                    entangle_gate="cnot",
                                    entangle_rules="linear",
                                    depth=1)
        w = tensor.ones([c.get_para_num()])

        cir = c.create_ansatz(w)
        print(cir)
        #           ┌────────────┐ ┌────────────┐ ┌────────────┐        ┌────────────┐ ┌────────────┐ ┌────────────┐
        # q_0:  |0>─┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├ ───■── ┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├ ────────────── ──────────────
        #           ├────────────┤ ├────────────┤ ├────────────┤ ┌──┴─┐ └────────────┘ ├────────────┤ ├────────────┤ ┌────────────┐
        # q_1:  |0>─┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├ ┤CNOT├ ───■────────── ┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├ ──────────────
        #           ├────────────┤ ├────────────┤ ├────────────┤ └────┘ ┌──┴─┐         └────────────┘ ├────────────┤ ├────────────┤ ┌────────────┐
        # q_2:  |0>─┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├ ────── ┤CNOT├──────── ───■────────── ┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├
        #           ├────────────┤ ├────────────┤ ├────────────┤        └────┘         ┌──┴─┐         ├────────────┤ ├────────────┤ ├────────────┤
        # q_3:  |0>─┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├ ────── ────────────── ┤CNOT├──────── ┤RX(1.000000)├ ┤RY(1.000000)├ ┤RZ(1.000000)├
        #           └────────────┘ └────────────┘ └────────────┘                       └────┘         └────────────┘ └────────────┘ └────────────┘

BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.template.BasicEntanglerTemplate(weights=None, num_qubits=1, rotation=pyqpanda.RX)

    Layers consisting of one-parameter single-qubit rotations on each qubit, followed by a closed chain or *ring* of CNOT gates.

    The ring of CNOT gates connects every qubit with its neighbour, with the last qubit being considered as a neighbour to the first qubit.

    The number of layers :math:`L` is determined by the first dimension of the argument ``weights``.

    :param weights: Weight tensor of shape ``(L, len(qubits))`` . Each weight is used as a parameter for the rotation, default: None, use random tensor with shape ``(1,1)`` .
    :param num_qubits: number of qubits, default: 1.
    :param rotation: one-parameter single-qubit gate to use, default: `pyqpanda.RX`

    Example::

        import pyqpanda as pq
        import numpy as np
        from pyvqnet.qnn.template import BasicEntanglerTemplate
        np.random.seed(42)
        num_qubits = 5
        shape = [1, num_qubits]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)

        circuit = BasicEntanglerTemplate(weights=weights, num_qubits=num_qubits, rotation=pq.RZ)
        result = circuit.create_circuit(qubits)
        circuit.print_circuit(qubits)

        prob = machine.prob_run_dict(result, qubits[0], -1)
        prob = list(prob.values())
        print(prob)
        #           ┌────────────┐                             ┌────┐
        # q_0:  |0>─┤RZ(0.374540)├ ───■── ────── ────── ────── ┤CNOT├
        #           ├────────────┤ ┌──┴─┐                      └──┬─┘
        # q_1:  |0>─┤RZ(0.950714)├ ┤CNOT├ ───■── ────── ────── ───┼──
        #           ├────────────┤ └────┘ ┌──┴─┐                  │
        # q_2:  |0>─┤RZ(0.731994)├ ────── ┤CNOT├ ───■── ────── ───┼──
        #           ├────────────┤        └────┘ ┌──┴─┐           │
        # q_3:  |0>─┤RZ(0.598658)├ ────── ────── ┤CNOT├ ───■── ───┼──
        #           ├────────────┤               └────┘ ┌──┴─┐    │
        # q_4:  |0>─┤RZ(0.156019)├ ────── ────── ────── ┤CNOT├ ───■──
        #           └────────────┘                      └────┘

        # [1.0, 0.0]


StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.template.StronglyEntanglingTemplate(weights=None, num_qubits=1, ranges=None)

    Layers consisting of single qubit rotations and entanglers, inspired by the `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    The argument ``weights`` contains the weights for each layer. The number of layers :math:`L` is therefore derived
    from the first dimension of ``weights``.

    The 2-qubit CNOT gate, act on the :math:`M` number of qubits, :math:`i = 1,...,M`. The second qubit of each gate is given by
    :math:`(i+r)\mod M`, where :math:`r` is a  hyperparameter called the *range*, and :math:`0 < r < M`.

    :param weights: weight tensor of shape ``(L, M, 3)`` , default: None, use random tensor with shape ``(1,1,3)`` .
    :param num_qubits: number of qubits, default: 1.
    :param ranges: sequence determining the range hyperparameter for each subsequent layer; default: None,
                                using :math:`r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.

    Example::

        import pyqpanda as pq
        import numpy as np
        from pyvqnet.qnn.template import StronglyEntanglingTemplate
        np.random.seed(42)
        num_qubits = 3
        shape = [2, num_qubits, 3]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()  # outside
        machine.init_qvm()  # outside
        qubits = machine.qAlloc_many(num_qubits)

        circuit = StronglyEntanglingTemplate(weights, num_qubits=num_qubits)
        result = circuit.create_circuit(qubits)
        circuit.print_circuit(qubits)

        prob = machine.prob_run_dict(result, qubits[0], -1)
        prob = list(prob.values())
        print(prob)
        #           ┌────────────┐ ┌────────────┐ ┌────────────┐               ┌────┐             ┌────────────┐ >
        # q_0:  |0>─┤RZ(0.374540)├ ┤RY(0.950714)├ ┤RZ(0.731994)├ ───■── ────── ┤CNOT├──────────── ┤RZ(0.708073)├ >
        #           ├────────────┤ ├────────────┤ ├────────────┤ ┌──┴─┐        └──┬┬┴───────────┐ ├────────────┤ >
        # q_1:  |0>─┤RZ(0.598658)├ ┤RY(0.156019)├ ┤RZ(0.155995)├ ┤CNOT├ ───■── ───┼┤RZ(0.832443)├ ┤RY(0.212339)├ >
        #           ├────────────┤ ├────────────┤ ├────────────┤ └────┘ ┌──┴─┐    │└────────────┘ ├────────────┤ >
        # q_2:  |0>─┤RZ(0.058084)├ ┤RY(0.866176)├ ┤RZ(0.601115)├ ────── ┤CNOT├ ───■────────────── ┤RZ(0.183405)├ >
        #           └────────────┘ └────────────┘ └────────────┘        └────┘                    └────────────┘ >
        #
        #          ┌────────────┐ ┌────────────┐        ┌────┐
        # q_0:  |0>┤RY(0.020584)├ ┤RZ(0.969910)├ ───■── ┤CNOT├ ──────
        #          ├────────────┤ └────────────┘    │   └──┬─┘ ┌────┐
        # q_1:  |0>┤RZ(0.181825)├ ────────────── ───┼── ───■── ┤CNOT├
        #          ├────────────┤ ┌────────────┐ ┌──┴─┐        └──┬─┘
        # q_2:  |0>┤RY(0.304242)├ ┤RZ(0.524756)├ ┤CNOT├ ────── ───■──
        #          └────────────┘ └────────────┘ └────┘
        #[0.6881335561525671, 0.31186644384743273]




Measure the quantum circuit
----------------------------------

expval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.measure.expval(machine, prog, pauli_str_dict, qubits)

    Expectation value of the supplied Hamiltonian observables

    if the observables are :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`,
    then ``Hamiltonian`` ``dict`` would be ``{{'Z0, X1':0.7} ,{'Z1':0.2}}`` .

    expval api only supports on QPanda CPUQVM now.Please checks  https://pyqpanda-toturial.readthedocs.io/zh/latest/index.html for alternative api.

    :param machine: machine created by qpanda
    :param prog: quantum program created by qpanda
    :param pauli_str_dict: Hamiltonian observables
    :param qubits: qubit allocated by pyQPanda
    :return: expectation


    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import expval
        input = [0.56, 0.1]
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(3)
        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0],input[0]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
        cir.insert(pq.RY(m_qlist[1],input[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
        m_prog.insert(cir)
        pauli_dict  = {'Z0 X1':10,'Y2':-0.543}
        exp2 = expval(m_machine,m_prog,pauli_dict,m_qlist)
        print(exp2)
        pq.destroy_quantum_machine(m_machine)
        #0.9983341664682731

QuantumMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.measure.QuantumMeasure(measure_qubits:list,prog,machine,qubits,slots:int = 1000)

    Calculates circuits quantum measurement. Return the normalized result of the measurements obtained by the Monte Carlo method.
    
    Please checks  https://pyqpanda-toturial.readthedocs.io/zh/latest/Measure.html?highlight=measure_all for alternative api.
    
    QuantumMeasure api only supports on QPanda ``CPUQVM`` or ``QCloud`` now.

    :param measure_qubits: list contains measure qubits index.
    :param prog: quantum program from qpanda
    :param machine: quantum virtual machine allocated by pyQPanda
    :param qubits: qubit allocated by pyQPanda
    :param slots: measure time,default 1000
    :return: returns the normalized result of the measurements obtained by the Monte Carlo method.

    Example::

        from pyvqnet.qnn.measure import QuantumMeasure
        import pyqpanda as pq
        input = [0.56,0.1]
        measure_qubits = [0,2]
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(3)

        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0],input[0]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
        cir.insert(pq.RY(m_qlist[1],input[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
        cir.insert(pq.H(m_qlist[0]))
        cir.insert(pq.H(m_qlist[1]))
        cir.insert(pq.H(m_qlist[2]))

        m_prog.insert(cir)
        rlt_quant = QuantumMeasure(measure_qubits,m_prog,m_machine,m_qlist)
        print(rlt_quant)
        #[0.25, 0.264, 0.257, 0.229]

ProbsMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.measure.ProbsMeasure(measure_qubits: list, prog, machine, qubits)

	Calculates circuits probabilities measurement.
    
    Please checks https://pyqpanda-toturial.readthedocs.io/zh/latest/PMeasure.html for alternative api.

    ProbsMeasure api only supports on QPanda ``CPUQVM`` or ``QCloud`` now.

    :param measure_qubits: list contains measure qubits index.
    :param prog: quantum program from qpanda
    :param qubits: qubit allocated by pyQPanda
    :return: prob of measure qubits in lexicographic order.

    Example::

        from pyvqnet.qnn.measure import ProbsMeasure
        import pyqpanda as pq

        input = [0.56,0.1]
        measure_qubits = [0,2]
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(3)

        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0],input[0]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
        cir.insert(pq.RY(m_qlist[1],input[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
        cir.insert(pq.H(m_qlist[0]))
        cir.insert(pq.H(m_qlist[1]))
        cir.insert(pq.H(m_qlist[2]))

        m_prog.insert(cir)

        rlt_prob = ProbsMeasure([0,2],m_prog,m_machine,m_qlist)
        print(rlt_prob)
        #[0.2499999999999947, 0.2499999999999947, 0.2499999999999947, 0.2499999999999947]


DensityMatrixFromQstate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.measure.DensityMatrixFromQstate(state, indices)

    Calculate the density matrix of quantum state vector in the computational basis.

    :param state: one-dimensional list state vector. The size of this list should be ``(2**N,)`` for some integer value ``N``. qstate should start from 000 -> 111.
    :param indices: list of qubit indices in the considered subsystem.
    :return: A density matrix of size "(2**len(indices), 2**len(indices))".

    Example::

        from pyvqnet.qnn.measure import DensityMatrixFromQstate
        qstate = [(0.9306699299765968+0j), (0.18865613455240968+0j), (0.1886561345524097+0j), (0.03824249173404786+0j), -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j, -0.048171819846746615j]
        print(DensityMatrixFromQstate(qstate,[0,1]))
        # [[0.86846704+0.j 0.1870241 +0.j 0.17604699+0.j 0.03791166+0.j]
        #  [0.1870241 +0.j 0.09206345+0.j 0.03791166+0.j 0.01866219+0.j]
        #  [0.17604699+0.j 0.03791166+0.j 0.03568649+0.j 0.00768507+0.j]
        #  [0.03791166+0.j 0.01866219+0.j 0.00768507+0.j 0.00378301+0.j]]

VN_Entropy
^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.measure.VN_Entropy(state, indices, base=None)

    Computes Von Neumann entropy from a state vector on a given list of qubits.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    :param state: one-dimensional list state vector. The size of this list should be ``(2**N,)`` for some integer value ``N``.
                    qstate should start from 000 ->111.
    :param indices: list of qubit indices in the considered subsystem.
    :param base: the base of the logarithm. If None, the natural logarithm is used. Default: None.

    :return: floating point value for the von Neumann entropy.

    Example::

        from pyvqnet.qnn.measure import VN_Entropy
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
                (0.18290448232350312 + 0j), -0.3293638014158896j,
                (0.03707657410649268 + 0j), -0.06676534788028635j,
                (0.18290448232350312 + 0j), -0.013534006039561714j]
        print(VN_Entropy(qstate, [0, 1]))
        #0.14592917648464448

Mutal_Info
^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.measure.Mutal_Info(state, indices0, indices1, base=None)

    Calculates the mutual information of the state vectors on the given two lists of sub-qubits.

    .. math::
        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    Mutual information is a measure of the correlation between two subsystems. More specifically, it quantifies the amount of information one system gains by measuring another.

    Each state can be given as a state vector in the computation base.

    :param state: one-dimensional list state vector. The size of this list should be ``(2**N,)`` for some integer value ``N``.qstate should start from 000 ->111
    :param indices0: list of qubit indices in the first subsystem.
    :param indices1: a list of qubit indices in the second subsystem.
    :param base: the base of the logarithm. If None, the natural logarithm is used. Default: None.

    :return: Mutual information between subsystems

    Example::

        from pyvqnet.qnn.measure import Mutal_Info
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
                (0.18290448232350312 + 0j), -0.3293638014158896j,
                (0.03707657410649268 + 0j), -0.06676534788028635j,
                (0.18290448232350312 + 0j), -0.013534006039561714j]
        print(Mutal_Info(qstate, [0], [2], 2))
        #0.13763425302805887



MeasurePauliSum
^^^^^^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.measure.MeasurePauliSum(machine, prog, obs_list, qlists)

    Expectation value of the supplied Hamiltonian observables.

    :param machine: machine created by qpanda.
    :param prog: quantum program created by qpanda.
    :param pauli_str_dict: Hamiltonian observables.
    :param qlists: qubit allocated by pyQpanda.qAlloc_many().

    :return: expectation

    Example::

        from pyvqnet.qnn.measure import MeasurePauliSum
        import pyqpanda as pq
        x = [0.56, 0.1]
        obs_list = [{'wires': [0, 2, 3], 'observables': ['X', 'Y', 'Z'], 'coefficient': [1, 0.5, 0.4]},
                    {'wires': [0, 1, 2], 'observables': ['X', 'Y', 'Z'], 'coefficient': [1, 0.5, 0.4]}]

        m_machine = pq.CPUQVM()
        m_machine.init_qvm()

        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(4)

        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0], x[0]))
        cir.insert(pq.RZ(m_qlist[1], x[0]))
        cir.insert(pq.CNOT(m_qlist[0], m_qlist[1]))
        cir.insert(pq.RY(m_qlist[2], x[1]))
        cir.insert(pq.CNOT(m_qlist[0], m_qlist[2]))
        cir.insert(pq.RZ(m_qlist[3], x[1]))

        m_prog.insert(cir)
        result = MeasurePauliSum(m_machine, m_prog, obs_list, m_qlist)
        print(result)
        m_machine.finalize()
        # [0.40000000000000013, 0.3980016661112104]


VarMeasure
^^^^^^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.measure.VarMeasure(machine, prog, actual_qlist)

    Variance of the supplied observable.

    :param machine: machine created by qpanda.
    :param prog: quantum program created by qpanda.
    :param actual_qlist: qubit allocated by pyQpanda.qAlloc_many().

    :return: var

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import VarMeasure
        cir = pq.QCircuit()
        machine = pq.CPUQVM()  # outside
        machine.init_qvm()  # outside
        qubits = machine.qAlloc_many(2)

        cir.insert(pq.RX(qubits[0], 0.5))
        cir.insert(pq.H(qubits[1]))
        cir.insert(pq.CNOT(qubits[0], qubits[1]))

        prog1 = pq.QProg()
        prog1.insert(cir)
        var_result = VarMeasure(machine, prog1, qubits[0])
        print(var_result)
        # 0.2298488470659339






Quantum Machine Learning Algorithm Interface
-------------------------------------------------

Quantum Perceptron
^^^^^^^^^^^^^^^^^^^^^

Artificial neural networks are the heart of machine learning algorithms and artificial intelligence protocols. Historically, the simplest implementation of an artificial neuron traces back to the classical Rosenblatt's `perceptron`, but its long term practical applications may be hindered by the fast scaling up of computational complexity, especially relevant for the training of multilayered perceptron networks.
Here we refer to the paper `An Artificial Neuron Implemented on an Actual Quantum Processor <https://arxiv.org/abs/1811.02266>`__ introduce a quantum information-based algorithm implementing the quantum computer version of a perceptron, which shows exponential advantage in encoding resources over alternative realizations.

For this quantum perceptron, the data processed is a string of 0 1 binary bits. The goal is to identify patterns that are shaped like a w cross as shown in the figure below.

.. image:: ./images/QP-data.png
   :width: 600 px
   :align: center

|

It is encoded using a binary bit string, where black is 0 and white is 1, so that w is encoded as (1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1). A total of 16-bit strings can be encoded into the sign of the amplitude of the 4-bit quantum state. The sign is 0 for negative numbers, and 1 for positive numbers. Through the above encoding method, our algorithm input is converted into a 16-bit binary string. Such non-repetitive binary strings can respectively correspond to specific input :math:`U_i` .

The circuit structure of the quantum perceptron proposed in this paper is as follows:

.. image:: ./images/QP-cir.png
   :width: 600 px
   :align: center

|

The coding circuit :math:`U_i` is constructed on bits 0~3, including multiple controlled :math:`CZ` , :math:`CNOT` gates, and :math:`H` gates; the weight conversion circuit :math:`U_w` is constructed immediately after :math:`U_i` , which is also composed of controlled gates and :math:`H` gates. :math:`U_i` can be used to perform unitary matrix transformations to encode data into quantum states:

.. math::
    U_i|0\rangle^{\otimes N}=\left|\psi_i\right\rangle

Use the unitary matrix transformation :math:`U_w` to compute the inner product between the input and the weights:

.. math::
    U_w\left|\psi_i\right\rangle=\sum_{j=0}^{m-1} c_j|j\rangle \equiv\left|\phi_{i, w}\right\rangle

The normalized activation probability values for :math:`U_i` and :math:`U_w` can be obtained by using a multi-controlled NOT gate with target bits on auxiliary bits, and using some subsequent :math:`H` gates, :math:`X` gates, and :math:`CX` gates as activation functions:

.. math::
    \left|\phi_{i, w}\right\rangle|0\rangle_a \rightarrow \sum_{j=0}^{m-2} c_j|j\rangle|0\rangle_a+c_{m-1}|m-1\rangle|1\rangle_a

When the binary string of the input i is exactly the same as w, the normalized probability value should be the largest.

VQNet provides the ``QuantumNeuron`` module to implement this algorithm. First initialize a quantum perceptron ``QuantumNeuron``.

.. code-block::

    perceptron = QuantumNeuron()

Use the ``gen_4bitstring_data`` interface to generate various data in the paper and its category labels.

.. code-block::

    training_label, test_label = perceptron.gen_4bitstring_data()

Using the ``train`` interface to traverse all the data, you can get the last trained quantum perceptron circuit :math:`U_w`.

.. code-block::

    trained_para = perceptron.train(training_label, test_label)

.. image:: ./images/QP-pic.png
   :width: 600 px
   :align: center

|

On the test data, the accuracy results on the test data can be obtained

.. image:: ./images/QP-acc.png
   :width: 600 px
   :align: center

|


Quantum Generative Adversarial Networks for learning and loading random distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantum Generative Adversarial Networks(`QGAN <https://www.nature.com/articles/s41534-019-0223-2>`_ )algorithm uses pure quantum variational circuits to prepare the generated quantum states with specific random distribution, which can reduce the logic gates required to generate specific quantum states and reduce the complexity of quantum circuits.It uses the classical GAN model structure, which has two sub-models: Generator and Discriminator. The Generator generates a specific distribution for the quantum circuit.And the Discriminator discriminates the generated data samples generated by the Generator and the real randomly distributed training data samples.
Here is an example of VQNet implementing QGAN learning and loading random distributions based on the paper `Quantum Generative Adversarial Networks for learning and loading random distributions <https://www.nature.com/articles/s41534-019-0223-2>`_ of Christa Zoufal.

.. image:: ./images/qgan-arch.PNG
   :width: 600 px
   :align: center

|

In order to realize the construction of ``QGANAPI`` class of quantum generative adversarial network by VQNet, the quantum generator is used to prepare the initial state of the real distributed data. The number of quantum bits is 3, and the repetition times of the internal parametric circuit module of the quantum generator is 1. Meanwhile, KL is used as the metric for the QGAN loading random distribution.

.. code-block::

    import pickle
    import os
    import pyqpanda as pq
    from pyvqnet.qnn.qgan.qgan_utils import QGANAPI
    import numpy as np

    num_of_qubits = 3  # paper config
    rep = 1
    number_of_data = 10000
    # Load data samples from different distributions
    mu = 1
    sigma = 1
    real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)


    # intial
    save_dir = None
    qgan_model = QGANAPI(
        real_data,
        # numpy generated data distribution, 1 - dim.
        num_of_qubits,
        batch_size=2000,
        num_epochs=2000,
        q_g_cir=None,
        bounds = [0.0,2**num_of_qubits -1],
        reps=rep,
        metric="kl",
        tol_rel_ent=0.01,
        if_save_param_dir=save_dir
    )

The following is the ``train`` module of QGAN.

.. code-block::

    # train
    qgan_model.train()  # train qgan


The ``eval`` module of QGAN is designed to draw the loss function curve and probability distribution diagram between the random distribution prepared by QGAN and the real distribution.

.. code-block::

    # show probability distribution function of generated distribution and real distribution
    qgan_model.eval(real_data)  #draw pdf

The ``get_trained_quantum_parameters`` module of QGAN is used to get training parameters and output them as a numpy array. If ``save_DIR`` is not empty, the training parameters are saved to a file.The ``Load_param_and_eval`` module of QGAN loads training parameters, and the ``get_circuits_with_trained_param`` module obtains pyQPanda circuit generated by quantum generator after training.

.. code-block::

    # get trained quantum parameters
    param = qgan_model.get_trained_quantum_parameters()
    print(f" trained param {param}")

    #load saved parameters files
    if save_dir is not None:
        path = os.path.join(
            save_dir, qgan_model._start_time + "trained_qgan_param.pickle")
        with open(path, "rb") as file:
            t3 = pickle.load(file)
        param = t3["quantum_parameters"]
        print(f" trained param {param}")

    #show probability distribution function of generated distribution and real distribution
    qgan_model.load_param_and_eval(param)

    #calculate metric
    print(qgan_model.eval_metric(param, "kl"))

    #get generator quantum circuit
    m_machine = pq.CPUQVM()
    m_machine.init_qvm()
    qubits = m_machine.qAlloc_many(num_of_qubits)
    qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
    print(qpanda_cir)

In general, QGAN learning and loading random distribution requires multiple training models with different random seeds to obtain the expected results. For example, the following is the graph of the probability distribution function between the lognormal distribution implemented by QGAN and the real lognormal distribution, and the loss function curve between QGAN's generator and discriminator.

.. image:: ./images/qgan-loss.png
   :width: 600 px
   :align: center

|

.. image:: ./images/qgan-pdf.png
   :width: 600 px
   :align: center

|


quantum kernal SVM
^^^^^^^^^^^^^^^^^^^

In machine learning tasks, data often cannot be separated by a hyperplane in the original space. A common technique for finding such hyperplanes is to apply a nonlinear transformation function to the data.
This function is called a feature map, through which we can calculate how close the data points are in this new feature space for the classification task of machine learning.

This example refers to the thesis: `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_ .
The first method constructs variational circuits for data classification tasks.

``gen_vqc_qsvm_data`` is the data needed to generate this example. ``vqc_qsvm`` is a variable sub-circuit class used to classify the input data.
The ``vqc_qsvm.plot()`` function visualizes the distribution of the data.

.. image:: ./images/VQC-SVM.png
   :width: 600 px
   :align: center

|

    .. code-block::

        """
        VQC QSVM
        """
        from pyvqnet.qnn.svm import vqc_qsvm, gen_vqc_qsvm_data
        import matplotlib.pyplot as plt
        import numpy as np

        batch_size = 40
        maxiter = 40
        training_size = 20
        test_size = 10
        gap = 0.3
        #sub-circuits repeat times
        rep = 3

        #defines QSVM class
        VQC_QSVM = vqc_qsvm(batch_size, maxiter, rep)
        #randomly generates data from thesis.
        train_features, test_features, train_labels, test_labels, samples = \
            gen_vqc_qsvm_data(training_size=training_size, test_size=test_size, gap=gap)
        VQC_QSVM.plot(train_features, test_features, train_labels, test_labels, samples)
        #train
        VQC_QSVM.train(train_features, train_labels)
        #test
        rlt, acc_1 = VQC_QSVM.predict(test_features, test_labels)
        print(f"testing_accuracy {acc_1}")


In addition to the above-mentioned direct use of variational quantum circuits to map classical data features to quantum feature spaces, in the paper `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_,
the method of directly estimating kernel functions using quantum circuits and classifying them using classical support vector machines is also introduced.
Analogy to various kernel functions in classical SVM :math:`K(i,j)` , use quantum kernel function to define the inner product of classical data in quantum feature space :math:`\phi(\mathbf{x}_i)` :

.. math::
    |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0 | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0 \rangle |^2

Using VQNet and pyQPanda, we define a ``QuantumKernel_VQNet`` to generate a quantum kernel function and use ``sklearn`` for classification:

.. image:: ./images/qsvm-kernel.png
   :width: 600 px
   :align: center

|

.. code-block::

    import numpy as np
    import pyqpanda as pq
    from sklearn.svm import SVC
    from pyqpanda import *
    from pyqpanda.Visualization.circuit_draw import *
    from pyvqnet.qnn.svm import QuantumKernel_VQNet, gen_vqc_qsvm_data
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt

    train_features, test_features,train_labels, test_labels, samples = gen_vqc_qsvm_data(20,5,0.3)
    quantum_kernel = QuantumKernel_VQNet(n_qbits=2)
    quantum_svc = SVC(kernel=quantum_kernel.evaluate)
    quantum_svc.fit(train_features, train_labels)
    score = quantum_svc.score(test_features, test_labels)
    print(f"quantum kernel classification test score: {score}")


Simultaneous Perturbation Stochastic Approximation optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.SPSA(maxiter: int = 1000, save_steps: int = 1, last_avg: int = 1, c0: float = _C0, c1: float = 0.2, c2: float = 0.602, c3: float = 0.101, c4: float = 0, init_para=None, model=None, calibrate_flag=False)
    

    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA provides a stochastic method for approximating the gradient of a multivariate differentiable cost function.
    To achieve this, the cost function is evaluated twice using a perturbed parameter vector: each component of the original parameter vector is simultaneously shifted by a randomly generated value.
    Further information is available on the `SPSA website <http://www.jhuapl.edu/SPSA>`__.

    :param maxiter: The maximum number of iterations to perform. Default value: 1000.
    :param save_steps: Save the intermediate information of each save_steps step. Default value: 1.
    :param last_avg: Averaging parameter for last_avg iterations.
        If last_avg = 1, only the last iteration is considered. Default value: 1.
    :param c0: initial a. Step size for updating parameters. Default value: 0.2*pi
    :param c1: initial c. The step size used to approximate the gradient. Default: 0.1.
    :param c2: alpha from the paper, used to adjust a(c0) at each iteration. Default value: 0.602.
    :param c3: gamma in the paper, used to adjust c(c1) at each iteration. Default value: 0.101.
    :param c4: Also used to control the parameters of a. Default value: 0.
    :param init_para: Initialization parameters. Default: None.
    :param model: Parametric model: model. Default: None.
    :param calibrate_flag: whether to calibrate hpyer parameters a and c, default value: False.

    :return: an SPSA optimizer instance

    Example::


        from pyvqnet.qnn import AngleEmbeddingCircuit, expval, QuantumLayerV2, SPSA
        from pyvqnet.qnn.template import BasicEntanglerTemplate
        import pyqpanda as pq
        from pyvqnet.nn.module import Module

        class Model_spsa(Module):
            def __init__(self):
                super(Model_spsa, self).__init__()
                self.qvc = QuantumLayerV2(layer_fn_spsa_pq, 3)

            def forward(self, x):
                y = self.qvc(x)
                return y


        def layer_fn_spsa_pq(input, weights):
            num_of_qubits = 1

            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(num_of_qubits)
            c1 = AngleEmbeddingCircuit(input, qubits)
            weights =weights.reshape([4,1])
            bc_class = BasicEntanglerTemplate(weights, 1)
            c2 = bc_class.create_circuit(qubits)
            m_prog = pq.QProg()
            m_prog.insert(c1)
            m_prog.insert(c2)
            pauli_dict = {'Z0': 1}
            exp2 = expval(machine, m_prog, pauli_dict, qubits)

            return exp2

        model = Model_spsa()

        optimizer = SPSA(maxiter=20,
            init_para=model.parameters(),
            model=model,
        )


.. py:function:: pyvqnet.qnn.SPSA._step(input_data)

    use SPSA to optimize input data.

    :param input_data: input data
    :return:

        train_para: final parameter

        theta_best: The average parameters of after last `last_avg`.

    Example::

        import numpy as np
        import pyqpanda as pq

        import sys
        sys.path.insert(0, "../")
        import pyvqnet

        from pyvqnet.nn.module import Module
        from pyvqnet.qnn import SPSA
        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.qnn import AngleEmbeddingCircuit, expval, QuantumLayerV2, expval
        from pyvqnet.qnn.template import BasicEntanglerTemplate


        class Model_spsa(Module):
            def __init__(self):
                super(Model_spsa, self).__init__()
                self.qvc = QuantumLayerV2(layer_fn_spsa_pq, 3)

            def forward(self, x):
                y = self.qvc(x)
                return y


        def layer_fn_spsa_pq(input, weights):
            num_of_qubits = 1

            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(num_of_qubits)
            c1 = AngleEmbeddingCircuit(input, qubits)
            weights =weights.reshape([4,1])
            bc_class = BasicEntanglerTemplate(weights, 1)
            c2 = bc_class.create_circuit(qubits)
            m_prog = pq.QProg()
            m_prog.insert(c1)
            m_prog.insert(c2)
            pauli_dict = {'Z0': 1}
            exp2 = expval(machine, m_prog, pauli_dict, qubits)

            return exp2

        model = Model_spsa()

        optimizer = SPSA(maxiter=20,
            init_para=model.parameters(),
            model=model,
        )

        data = QTensor(np.array([[0.27507603]]))
        p = model.parameters()
        p[0].data = pyvqnet._core.Tensor( np.array([3.97507603, 3.12950603, 1.00854038,
                        1.25907603]))

        optimizer._step(input_data=data)


        y = model(data)
        print(y)


Quantum Nature Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quantum machine learning models generally use the gradient descent method to optimize parameters in variable quantum logic circuits. The formula of the classic gradient descent method is as follows:

.. math:: \theta_{t+1} = \theta_t -\eta \nabla \mathcal{L}(\theta),

Essentially, at each iteration, we will calculate the direction of the steepest gradient drop in the parameter space as the direction of parameter change.
In any direction in space, the speed of descent in the local range is not as fast as that of the negative gradient direction.
In different spaces, the derivation of the direction of steepest descent is dependent on the norm of parameter differentiation - the distance metric. The distance metric plays a central role here,
Different metrics result in different directions of steepest descent. For the Euclidean space where the parameters in the classical optimization problem are located, the direction of the steepest descent is the direction of the negative gradient.
Even so, at each step of parameter optimization, as the loss function changes with parameters, its parameter space is transformed. Make it possible to find another better distance norm.

`Quantum natural gradient method <https://arxiv.org/abs/1909.02108>`_ draws on concepts from `classical natural gradient method Amari <https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746>`__ ,
We instead view the optimization problem as a probability distribution of possible output values for a given input (i.e., maximum likelihood estimation), a better approach is in the distribution
Gradient descent is performed in the space, which is dimensionless and invariant with respect to the parameterization. Therefore, regardless of the parameterization, each optimization step will always choose the optimal step size for each parameter.
In quantum machine learning tasks, the quantum state space has a unique invariant metric tensor called the Fubini-Study metric tensor :math:`g_{ij}`.
This tensor converts the steepest descent in the quantum circuit parameter space to the steepest descent in the distribution space.
The formula for the quantum natural gradient is as follows:

.. math:: \theta_{t+1} = \theta_t - \eta g^{+}(\theta_t)\nabla \mathcal{L}(\theta),

where :math:`g^{+}` is the pseudo-inverse.

The following is an example of quantum natural gradient optimization of a quantum variational circuit parameter based on VQNet. It can be seen that the use of quantum natural gradient (Quantum Nature Gradient) makes some loss functions decline faster.

Our goal is to minimize the expectation of the following quantum variational circuit. It can be seen that there are two layers of 3 quantum parametric logic gates in total. The first layer is composed of RZ and RY logic gates on bits 0 and 1, 
and the second layer is composed of RX logic gate on 2 bits constitutes.

.. image:: ./images/qng_all_cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    import pyqpanda as pq
    import numpy as np
    from pyvqnet.tensor import QTensor
    from pyvqnet.qnn.measure import expval, ProbsMeasure
    from pyvqnet.qnn import insert_pauli_for_mt, get_metric_tensor, QNG,QuantumLayer
    import matplotlib.pyplot as plt
    from pyvqnet.optim import SGD
    from pyvqnet import _core
    ###################################################
    # Quantum Nature Gradients Examples
    ###################################################
    class pyqpanda_config_wrapper:
        """
        A wrapper for pyqpanda config,including QVM machine, allocated qubits, classic bits.
        """
        def __init__(self, qubits_num) -> None:
            self._machine = pq.CPUQVM()
            self._machine.init_qvm()
            self._qubits = self._machine.qAlloc_many(qubits_num)
            self._cubits = self._machine.cAlloc_many(qubits_num)
            self._qcir = pq.QCircuit()
        def __del__(self):
            self._machine.finalize()
    # use quantum nature gradient optimzer to optimize circuit quantum_net
    steps = 200
    def quantum_net(
            q_input_features,
            params,
            qubits,
            cubits,
            machine):
        qcir = pq.QCircuit()
        qcir.insert(pq.RY(qubits[0], np.pi / 4))
        qcir.insert(pq.RY(qubits[1], np.pi / 3))
        qcir.insert(pq.RY(qubits[2], np.pi / 7))
        qcir.insert(pq.RZ(qubits[0], params[0]))
        qcir.insert(pq.RY(qubits[1], params[1]))
        qcir.insert(pq.CNOT(qubits[0], qubits[1]))
        qcir.insert(pq.CNOT(qubits[1], qubits[2]))
        qcir.insert(pq.RX(qubits[2], params[2]))
        qcir.insert(pq.CNOT(qubits[0], qubits[1]))
        qcir.insert(pq.CNOT(qubits[1], qubits[2]))
        m_prog = pq.QProg()
        m_prog.insert(qcir)
        return expval(machine, m_prog, {'Y0': 1}, qubits)

To use the quantum natural gradient algorithm, we first need to compute the metric tensor.
According to the definition of the algorithm, we manually defined the following two sub-circuits to calculate the Fubini-Study tensor of the two-layer circuit with parameters.
The first parameter layer calculates the sub-circuit of the metric tensor as follows:

.. image:: ./images/qng_subcir1.png
   :width: 600 px
   :align: center

|

.. code-block::

    def layer0_subcircuit(config: pyqpanda_config_wrapper, params):
        qcir = pq.QCircuit()
        qcir.insert(pq.RY(config._qubits[0], np.pi / 4))
        qcir.insert(pq.RY(config._qubits[1], np.pi / 3))
        return qcir
    def get_p01_diagonal_(config, params, target_gate_type, target_gate_bits,
                            wires):
        qcir = layer0_subcircuit(config, params)
        qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                    target_gate_bits)
        qcir3 = pq.QCircuit()
        qcir3.insert(qcir)
        qcir3.insert(qcir2)
        m_prog = pq.QProg()
        m_prog.insert(qcir3)
        return ProbsMeasure(wires, m_prog, config._machine, config._qubits)

The sub-circuit for computing the metric tensor in the second parameter layer is as follows:

.. image:: ./images/qng_subcir2.png
   :width: 600 px
   :align: center

|

.. code-block::

    def layer1_subcircuit(config: pyqpanda_config_wrapper, params):
        qcir = pq.QCircuit()
        qcir.insert(pq.RY(config._qubits[0], np.pi / 4))
        qcir.insert(pq.RY(config._qubits[1], np.pi / 3))
        qcir.insert(pq.RY(config._qubits[2], np.pi / 7))
        qcir.insert(pq.RZ(config._qubits[0], params[0]))
        qcir.insert(pq.RY(config._qubits[1], params[1]))
        qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1]))
        qcir.insert(pq.CNOT(config._qubits[1], config._qubits[2]))
        return qcir
    def get_p1_diagonal_(config, params, target_gate_type, target_gate_bits,
                            wires):
        qcir = layer1_subcircuit(config, params)
        qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                    target_gate_bits)
        qcir3 = pq.QCircuit()
        qcir3.insert(qcir)
        qcir3.insert(qcir2)
        m_prog = pq.QProg()
        m_prog.insert(qcir3)
        
        return ProbsMeasure(wires, m_prog, config._machine, config._qubits)

Use the quantum natural gradient class defined by the `QNG` class, where [['RZ', 'RY'], ['RX']] are 3 gate types with parameter logic gates,
[[0, 1], [2]] is the active bit, qcir is the circuit function list of the calculation tensor, and [0,1,2] is the qubit index of the entire circuit.

.. code-block::

    config = pyqpanda_config_wrapper(3)
    qcir = []
    qcir.append(get_p01_diagonal_)
    qcir.append(get_p1_diagonal_)
    # define QNG optimzer
    opt = QNG(config, quantum_net, 0.02, [['RZ', 'RY'], ['RX']], [[0, 1], [2]],
                qcir, [0, 1, 2])

For iterative optimization, use the `opt` function for single-step optimization, where the first input parameter is the input data,
There is no input in the line here, so it is None, and the second input parameter is the parameter to be optimized theta.

.. code-block::

    qng_cost = []
    theta2 = QTensor([0.432, 0.543, 0.233])
    # iteration
    for _ in range(steps):
        theta2 = opt.step(None, theta2)
        qng_cost.append(
            quantum_net(None, theta2, config._qubits, config._cubits,
                        config._machine))

Using the SGD classic gradient descent method as a baseline to compare the changes in the loss value of the two under the same number of iterations,
it can be seen that the loss function declines faster using the quantum natural gradient.

.. code-block::

    # use gradient descent as the baseline
    sgd_cost = []
    qlayer = QuantumLayer(quantum_net, 3, 'cpu', 3)
    temp = _core.Tensor([0.432, 0.543, 0.233])
    _core.vqnet.copyTensor(temp, qlayer.m_para.data)
    opti = SGD(qlayer.parameters())
    for i in range(steps):
        opti.zero_grad()
        loss = qlayer(QTensor([[1]]))
        print(f'step {i}')
        print(f'q param before {qlayer.m_para}')
        loss.backward()
        sgd_cost.append(loss.item())
        opti._step()
        print(f'q param after{qlayer.m_para}')
        
    plt.style.use("seaborn")
    plt.plot(qng_cost, "b", label="Quantum natural gradient descent")
    plt.plot(sgd_cost, "g", label="Vanilla gradient descent")
    plt.ylabel("Cost function value")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.show()

.. image:: ./images/qng_vs_sgd.png
   :width: 600 px
   :align: center

|

Stochastic parameter shift algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the quantum variational circuit, it is a common method to use the parameter shift method `parameter-shift` to calculate the gradient of the quantum parameter.
The parameter shift method is not universally applicable to all quantum parametric logic gates.
In cases where it does not hold (or is not known to hold), we either have to factorize the gates into compatible gates, or use an alternative estimator of the gradient, such as a finite difference approximation.
However, both alternatives may have drawbacks due to increased circuit complexity or potential errors in gradient values.
Banchi and Crooks 1 discovered a `Stochastic Parameter-Shift Rule <https://arxiv.org/abs/2005.10299>`_ that can be applied to any unitary matrix quantum logic gate.

The following shows an example of applying VQNet to calculate the gradient using the random parameter offset method for a quantum variational circuit. An example line definition is as follows:

.. code-block::

    import pyqpanda as pq
    import numpy as np
    from pyvqnet.qnn.measure import expval
    from scipy.linalg import expm
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt
    machine = pq.init_quantum_machine(pq.QMachineType.CPU)
    q = machine.qAlloc_many(2)
    c = machine.cAlloc_many(2)
    # some basic Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    def Generator(theta1, theta2, theta3):
        G = theta1.item() * np.kron(X, I) - \
            theta2 * np.kron(Z, X) + \
            theta3 * np.kron(I, X)
        return G
    def pq_demo_circuit(gate_pars):
        G = Generator(*gate_pars)
        G = expm(-1j * G)
        x = G.flatten().tolist()
        cir = pq.matrix_decompose(q, x)
        m_prog = pq.QProg()
        m_prog.insert(cir)
        pauli_dict = {'Z0': 1}
        exp2 = expval(machine, m_prog, pauli_dict, q)
        return exp2

The stochastic parameter shift method first randomly samples a variable
 s from the uniform distribution of [0,1], 
 and then performs the following unitary matrix transformation on the lines:

      a) :math:`e^{i(1-s)(\hat{H} + \theta\hat{V})}`
      b) :math:`e^{+i\tfrac{\pi}{4}\hat{V}}`
      c) :math:`e^{is(\hat{H} + \theta\hat{V})}`

where :math:`\hat{V}` is a tensor product of Pauli operators, and :math:`\hat{H}` is a linear combination of any Pauli operator tensor product.
We define the expected value of the observabley obtained at this time as :math:`\langle r_+ \rangle`.

.. code-block::

    def pq_SPSRgates(gate_pars, s, sign):
        G = Generator(*gate_pars)
        # step a)
        G1 = expm(1j * (1 - s) * G)
        x = G1.flatten().tolist()
        cir = pq.matrix_decompose(q, x)
        m_prog = pq.QProg()
        m_prog.insert(cir)
        # step b)
        G2 = expm(1j * sign * np.pi / 4 * X)
        x = G2.flatten().tolist()
        cir = pq.matrix_decompose(q[0], x)
        m_prog.insert(cir)
        # step c)
        G3 = expm(1j * s * G)
        x = G3.flatten().tolist()
        cir = pq.matrix_decompose(q, x)
        m_prog.insert(cir)
        pauli_dict = {'Z0': 1}
        exp2 = expval(machine, m_prog, pauli_dict, q)
        return exp2

Change :math:`\tfrac{\pi}{4}` in the previous step to :math:`-\tfrac{\pi}{4}`,
Repeat operations a, b, c to obtain observable

The gradient formula calculated by the stochastic parameter shift algorithm is as follows:

 .. math::
     \mathbb{E}_{s\in\mathcal{U}[0,1]}[\langle r_+ \rangle - \langle r_-\rangle]

In the following figure, the gradient of the parameter :math:`\theta_1` is showed, using the stochastic parameter shift method.
It can be seen that the observable is expected to 
conform to the functional form of :math:`\cos(2\theta_1)`; 
and the gradient is calculated using the random parameter shift method,

Meets :math:`-2\sin(2\theta_1)` , which is exactly the differential of :math:`\cos(2\theta_1)`.

.. code-block::

    theta2, theta3 = -0.15, 1.6
    angles = np.linspace(0, 2 * np.pi, 50)
    pos_vals = np.array([[
        pq_SPSRgates([theta1, theta2, theta3], s=s, sign=+1)
        for s in np.random.uniform(size=10)
    ] for theta1 in angles])
    neg_vals = np.array([[
        pq_SPSRgates([theta1, theta2, theta3], s=s, sign=-1)
        for s in np.random.uniform(size=10)
    ] for theta1 in angles])
    # Plot the results
    evals = [pq_demo_circuit([theta1, theta2, theta3]) for theta1 in angles]
    spsr_vals = (pos_vals - neg_vals).mean(axis=1)
    plt.plot(angles, evals, 'b', label="Expectation Value")
    plt.plot(angles, spsr_vals, 'r', label="Stochastic parameter-shift rule")
    plt.xlabel("theta1")
    plt.legend()
    plt.title("VQNet")
    plt.show()

.. image:: ./images/stochastic_parameter-shift.png
   :width: 600 px
   :align: center

|

Doubly Stochastic Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In variational quantum algorithms, parameterized quantum circuits are optimized by 
classical gradient descent to minimize the expected function value.
Although the expected value can be calculated analytically in classical simulator,
on quantum hardware the program is limited to sampling from the expected value;
as the number of samples and the number of shots increase, 
the expected value obtained in this way will converge to the theoretical expected value,
but may always be accurate value.
Sweke et al. found a double stochastic gradient descent method in `the paper <https://arxiv.org/abs/1910.01155>`_.
In this paper, they show that quantum gradient descent, which uses a finite number of measurement 
samples (or shots) to estimate gradients, is a form of stochastic gradient descent.
Furthermore, if the optimization involves a linear combination of 
expected values (such as VQE), sampling from the terms in that 
linear combination can further reduce the required time complexity.

VQNet implements an example of this algorithm: solving the ground state energy of the target Hamiltonian using VQE. Note that here we set the number of shots for quantum circuit observations to only 1.

.. math::

    H = \begin{bmatrix}
          8 & 4 & 0 & -6\\
          4 & 0 & 4 & 0\\
          0 & 4 & 8 & 0\\
          -6 & 0 & 0 & 0
        \end{bmatrix}.

.. code-block::

    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.template import StronglyEntanglingTemplate
    from pyvqnet.qnn.measure import Hermitian_expval
    from pyvqnet.qnn import QuantumLayerV2
    from pyvqnet.optim import SGD
    import pyvqnet._core as _core
    from pyvqnet.tensor import QTensor
    from matplotlib import pyplot as plt
    num_layers = 2
    num_wires = 2
    eta = 0.01
    steps = 200
    n = 1
    param_shape = [2, 2, 3]
    shots = 1
    H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
    init_params = np.random.uniform(low=0,
                                    high=2 * np.pi,
                                    size=param_shape)
    # some basic Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    def pq_circuit(params):
        params = params.reshape(param_shape)
        num_qubits = 2
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        circuit = StronglyEntanglingTemplate(params, num_qubits=num_qubits)
        qcir = circuit.create_circuit(qubits)
        prog = pq.QProg()
        prog.insert(qcir)
        machine.directly_run(prog)
        result = machine.get_qstate()
        return result

The Hamiltonian in this example is a Hermitian matrix,
 which we can always represent as a sum of Pauli matrices.

.. math::

    H = \sum_{i,j=0,1,2,3} a_{i,j} (\sigma_i\otimes \sigma_j)

and 

.. math::

    a_{i,j} = \frac{1}{4}\text{tr}[(\sigma_i\otimes \sigma_j )H], ~~ \sigma = \{I, X, Y, Z\}.

Substituting into the above formula, we can see that

.. math::

    H = 4  + 2I\otimes X + 4I \otimes Z - X\otimes X + 5 Y\otimes Y + 2Z\otimes X.

To perform "doubly stochastic" gradient descent, we simply apply the stochastic gradient descent method, but additionally uniformly sample a subset of the Hamiltonian expectation at each optimization step.
The vqe_func_analytic() function uses parameter shift to calculate theoretical gradients, 
and vqe_func_shots() uses random sampled values and randomly sampled Hamiltonian 
expectation subsets for "doubly stochastic" gradient calculations.

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


Use VQNet for parameter optimization, and compare the curve of the loss function.
Since the double stochastic gradient descent method only calculates the partial Pauli 
operator sum of H each time,
Therefore, the average value can be used to represent the expected result of the final 
observation. Here, the moving average moving_average() is used for calculation.

.. code-block::


    from pyqpanda import *

    ##############################################################################
    # Optimizing the circuit using gradient descent via the parameter-shift rule:
    qlayer_ana = QuantumLayerV2(vqe_func_analytic, 2*2*3 )
    qlayer_shots = QuantumLayerV2(vqe_func_shots, 2*2*3 )
    cost_sgd = []
    cost_dsgd = []
    temp = _core.Tensor(init_params)
    _core.vqnet.copyTensor(temp, qlayer_ana.m_para.data)
    opti_ana = SGD(qlayer_ana.parameters())

    _core.vqnet.copyTensor(temp, qlayer_shots.m_para.data)
    opti_shots = SGD(qlayer_shots.parameters())
    
    for i in range(steps):
        opti_ana.zero_grad()
        loss = qlayer_ana(QTensor([[1]]))
        loss.backward()
        cost_sgd.append(loss.item())
        opti_ana._step()
    for i in range(steps+50):
        opti_shots.zero_grad()
        loss = qlayer_shots(QTensor([[1]]))
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