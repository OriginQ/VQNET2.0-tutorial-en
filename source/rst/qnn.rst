Quantum Machine Learning API using QPanda
####################################################

Quantum Computing Layer
***********************************

.. _QuantumLayer:

QuantumLayer
=================================

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

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01, dtype=None, name='')

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
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

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

        Use the ``m_para`` attribute of QuantumLayer to get the training parameters of the variable quantum circuit. The parameter is a ``QTensor`` class, which can be converted into a numpy array using the ``to_numpy()`` interface.

    .. note::

        The class have alias: `QpandaQCircuitVQCLayer` .

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
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2.0]] )
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
=================================

If you are more familiar with pyQPanda syntax, please using QuantumLayerV2 class, you can define the quantum circuits function by using ``qubits``, ``cbits`` and ``machine``, then take it as a argument ``qprog_with_measure`` of QuantumLayerV2.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    Abstract calculation module for variational quantum circuits. It simulates a parameterized quantum circuit and gets the measurement result.
    QuantumLayer inherits from Module ,so that it can calculate gradients of circuits parameters,and train variational quantum circuits model or embed variational quantum circuits into hybird quantum and classic model.
    
    To use this module, you need to create your quantum virtual machine and allocate qubits and cbits.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda
    :param para_num: `int` - Number of parameter
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer
    :return: a module can calculate quantum circuits .

    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html.

        This function should contains following parameters,otherwise it can not run properly in QuantumLayerV2.

        Compare to QuantumLayer.you should allocate qubits and simulator: https://pyqpanda-toturial.readthedocs.io/zh/latest/QuantumMachine.html,

        you may also need to allocate cbits if qprog_with_measure needs quantum measure: https://pyqpanda-toturial.readthedocs.io/zh/latest/Measure.html

        qprog_with_measure (input,param)

        `input`: array_like input 1-dim classic data

        `param`: array_like input 1-dim quantum circuit's parameters

    .. note::

        The class have alias: `QpandaQCircuitVQCLayerLite` .

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
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2.0]] )

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


QuantumLayerV3
=============================

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV3(origin_qprog_func,para_num,num_qubits, num_cubits, pauli_str_dict=None, shots=1000, initializer=None,dtype=None,name="")
    
    It submits the parameterized quantum circuit to the local QPanda full amplitude simulator for calculation and trains the parameters in the circuit.
    It supports batch data and uses the parameter shift rule to estimate the gradient of the parameters.
    For CRX, CRY, CRZ, this layer uses the formula in https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3, and the rest of the logic gates use the default parameter drift method to calculate the gradient.

    :param origin_qprog_func: The callable quantum circuit function built by QPanda.
    :param para_num: `int` - Number of parameters; parameters are one-dimensional.
    :param num_qubits: `int` - Number of qubits in the quantum circuit.
    :param num_cubits: `int` - Number of classical bits used for measurements in the quantum circuit.
    :param pauli_str_dict: `dict|list` - Dictionary or list of dictionaries representing Pauli operators in the quantum circuit. Defaults to None.
    :param shots: `int` - Number of measurement shots. Defaults to 1000.
    :param initializer: Initializer for parameter values. Defaults to None.
    :param dtype: Data type of the parameter. Defaults to None, which uses the default data type.
    :param name: Name of the module. Defaults to the empty string.

    :return: Returns a QuantumLayerV3 class

    .. note::

        origin_qprog_func is a user defined quantum circuit function using pyQPanda:
        https://pyqpanda-toturial.readthedocs.io/en/latest/QCircuit.html.

        The function should contain the following input parameters and return a pyQPanda.QProg or originIR.

        origin_qprog_func (input,param,m_machine,qubits,cubits)

        `input`: user defined array-like input 1D classical data.

        `param`: array_like input user defined 1D quantum circuit parameters.

        `m_machine`: simulator created by QuantumLayerV3.

        `qubits`: quantum bits allocated by QuantumLayerV3

        `cubits`: classical bits allocated by QuantumLayerV3. If your circuit does not use classical bits, you should also keep this parameter as a function input.

    .. note::

        The class have alias: `QpandaQProgVQCLayer` .

    Example::

        import numpy as np
        import pyqpanda as pq
        import pyvqnet
        from pyvqnet.qnn import QuantumLayerV3

        def qfun(input, param, m_machine, m_qlist, cubits):
        measure_qubits = [0,1, 2]
        m_prog = pq.QProg()
        cir = pq.QCircuit()

        cir.insert(pq.RZ(m_qlist[0], input[0]))
        cir.insert(pq.RX(m_qlist[2], input[2]))

        qcir = pq.RX(m_qlist[1], param[1])
        qcir.set_control(m_qlist[0])
        cir.insert(qcir)

        qcir = pq.RY(m_qlist[0], param[2])
        qcir.set_control(m_qlist[1])
        cir.insert(qcir)

        cir.insert(pq.RY(m_qlist[0], input[1]))

        qcir = pq.RZ(m_qlist[0], param[3])
        qcir.set_control(m_qlist[1])
        cir.insert(qcir)
        m_prog.insert(cir)

        for idx, ele in enumerate(measure_qubits):
        m_prog << pq.Measure(m_qlist[ele], cubits[idx]) # pylint: disable=expression-not-assigned
        return m_prog
        from pyvqnet.utils.initializer import ones
        l = QuantumLayerV3(qfun,
        4,
        3,
        3,
        pauli_str_dict=None,
        shots=1000,
        initializer=ones,
        name="")
        x = pyvqnet.tensor.QTensor(
        [[2.56, 1.2,-3]],
        requires_grad=True)
        y = l(x)

        y.backward()
        print(l.m_para.grad.to_numpy())
        print(x.grad.to_numpy())


QuantumBatchAsyncQcloudLayer
=================================

When you install the latest version of pyqpanda, you can use this interface to define a variational circuit and submit it to originqc for running on the real chip.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num, num_qubits, num_cubits, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift ", submit_kwargs={}, query_kwargs={})
    
    Abstract computing module for originqc real chips using pyqpanda QCLOUD starting with version 3.8.2.2. It submits parameterized quantum circuits to real chips and obtains measurement results.
    If diff_method == "random_coordinate_descent" , we will randomly select a single parameter to compute the gradient, and the other parameters will remain zero. Ref: https://arxiv.org/abs/2311.00088 .
    
    .. note::

        qcloud_token is the API token you applied for at https://qcloud.originqc.com.cn/.
        origin_qprog_func needs to return data of type pypqanda.QProg. If pauli_str_dict is not set, you need to ensure that measure has been inserted into the QProg.
        The form of origin_qprog_func must be as follows:

        origin_qprog_func(input,param,qubits,cbits,machine)

            `input`: Input 1~2-dimensional classic data. In the case of two-dimensional data, the first dimension is the batch size.

            `param`: Enter the parameters to be trained for the one-dimensional variational quantum circuit.

            `machine`: The simulator QCloud created by QuantumBatchAsyncQcloudLayer does not require users to define it in additional functions.

            `qubits`: Qubits created by the simulator QCloud created by QuantumBatchAsyncQcloudLayer, the number is `num_qubits`, the type is pyQpanda.Qubits, no need for the user to define it in the function.

            `cbits`: Classic bits allocated by QuantumBatchAsyncQcloudLayer, the number is `num_cubits`, the type is pyQpanda.ClassicalCondition, no need for the user to define it in the function. .


    :param origin_qprog_func: The variational quantum circuit function built by QPanda must return QProg.
    :param qcloud_token: `str` - The type of quantum machine or cloud token used for execution.
    :param para_num: `int` - Number of parameters, the parameter is a QTensor of size [para_num].
    :param num_qubits: `int` - Number of qubits in the quantum circuit.
    :param num_cubits: `int` - The number of classical bits used for measurement in quantum circuits.
    :param pauli_str_dict: `dict|list` - A dictionary or list of dictionaries representing Pauli operators in quantum circuits. The default is "none", and the measurement operation is performed. If a dictionary of Pauli operators is entered, a single expectation or multiple expectations will be calculated.
    :param shot: `int` - Number of measurements. The default value is 1000.
    :param initializer: Initializer for parameter values. The default is "None", using 0~2*pi normal distribution.
    :param dtype: The data type of the parameter. The default value is None, which uses the default data type pyvqnet.kfloat32.
    :param name: The name of the module. Defaults to empty string.
    :param diff_method: Differentiation method for gradient computation. Default is "parameter_shift". If diff_method == "random_coordinate_descent" , we will randomly select a single parameter to compute the gradient, and the other parameters will remain zero. Ref: https://arxiv.org/abs/2311.00088 .
    :param submit_kwargs: Additional keyword parameters for submitting quantum circuits, default: {"chip_id":pyqpanda.real_chip_type.origin_72,"is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3, "default_task_group_size":200, "test_qcloud_fake":False}, when test_qcloud_fake is set to True, the local CPUQVM is simulated.
    :param query_kwargs: Additional keyword parameters for querying quantum results, default: {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.
    
    :return: A module that can calculate quantum circuits.

    Example::

        import numpy as np
        import pyqpanda as pq
        import pyvqnet
        from pyvqnet.qnn import QuantumLayer,QuantumBatchAsyncQcloudLayer
        from pyvqnet.qnn import expval_qcloud

        #set_test_qcloud_fake(False) #uncomments this code to use realchip


        def qfun(input,param, m_machine, m_qlist,cubits):
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir.insert(pq.RZ(m_qlist[0],input[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
            cir.insert(pq.RY(m_qlist[1],param[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
            cir.insert(pq.RZ(m_qlist[1],input[1]))
            cir.insert(pq.RY(m_qlist[2],param[1]))
            cir.insert(pq.H(m_qlist[2]))
            m_prog.insert(cir)

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.Measure(m_qlist[ele], cubits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = QuantumBatchAsyncQcloudLayer(qfun,
                        "3047DE8A59764BEDAC9C3282093B16AF1",
                        2,
                        6,
                        6,
                        pauli_str_dict=None,
                        shots = 1000,
                        initializer=None,
                        dtype=None,
                        name="",
                        diff_method="parameter_shift",
                        submit_kwargs={},
                        query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

        def qfun2(input,param, m_machine, m_qlist,cubits):
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir.insert(pq.RZ(m_qlist[0],input[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
            cir.insert(pq.RY(m_qlist[1],param[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
            cir.insert(pq.RZ(m_qlist[1],input[1]))
            cir.insert(pq.RY(m_qlist[2],param[1]))
            cir.insert(pq.H(m_qlist[2]))
            m_prog.insert(cir)

            return m_prog
        l = QuantumBatchAsyncQcloudLayer(qfun2,
                    "3047DE8A59764BEDAC9C3282093B16AF",
                    2,
                    6,
                    6,
                    pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543},
                    shots = 1000,
                    initializer=None,
                    dtype=None,
                    name="",
                    diff_method="parameter_shift",
                    submit_kwargs={},
                    query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)


QuantumBatchAsyncQcloudLayerES
=================================

When you install the latest version of pyqpanda, you can use this interface to define a variational circuit and submit it to originqc for running on the real chip.
The interface estimates the parameter gradients and updates the parameters in an ‘evolutionary strategy’ approach, which can be found in the paper  `Learning to learn with an evolutionary strategy Learning to learn with an evolutionary strategy <https://arxiv.org/abs/2310.17402>`_ .

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumBatchAsyncQcloudLayerES(origin_qprog_func, qcloud_token, para_num, num_qubits, num_cubits, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", submit_kwargs={}, query_kwargs={}, sigma=np.pi/24)
    
    Abstract computing module for originqc real chips using pyqpanda QCLOUD starting with version 3.8.2.2. It submits parameterized quantum circuits to real chips and obtains measurement results.

    .. note::

        qcloud_token is the API token you applied for at https://qcloud.originqc.com.cn/.
        origin_qprog_func needs to return data of type pypqanda.QProg. If pauli_str_dict is not set, you need to ensure that measure has been inserted into the QProg.
        The form of origin_qprog_func must be as follows:

        origin_qprog_func(input,param,qubits,cbits,machine)

            `input`: Input 1~2-dimensional classic data. In the case of two-dimensional data, the first dimension is the batch size.

            `param`: Enter the parameters to be trained for the one-dimensional variational quantum circuit.

            `machine`: The simulator QCloud created by QuantumBatchAsyncQcloudLayerES does not require users to define it in additional functions.

            `qubits`: Qubits created by the simulator QCloud created by QuantumBatchAsyncQcloudLayerES, the number is `num_qubits`, the type is pyQpanda.Qubits, no need for the user to define it in the function.

            `cbits`: Classic bits allocated by QuantumBatchAsyncQcloudLayerES, the number is `num_cubits`, the type is pyQpanda.ClassicalCondition, no need for the user to define it in the function. .


    :param origin_qprog_func: The variational quantum circuit function built by QPanda must return QProg.
    :param qcloud_token: `str` - The type of quantum machine or cloud token used for execution.
    :param para_num: `int` - Number of parameters, the parameter is a QTensor of size [para_num].
    :param num_qubits: `int` - Number of qubits in the quantum circuit.
    :param num_cubits: `int` - The number of classical bits used for measurement in quantum circuits.
    :param pauli_str_dict: `dict|list` - A dictionary or list of dictionaries representing Pauli operators in quantum circuits. The default is "none", and the measurement operation is performed. If a dictionary of Pauli operators is entered, a single expectation or multiple expectations will be calculated.
    :param shot: `int` - Number of measurements. The default value is 1000.
    :param initializer: Initializer for parameter values. The default is "None", using 0~2*pi normal distribution.
    :param dtype: The data type of the parameter. The default value is None, which uses the default data type pyvqnet.kfloat32.
    :param name: The name of the module. Defaults to empty string.
    :param submit_kwargs: Additional keyword parameters for submitting quantum circuits, default: {"chip_id":pyqpanda.real_chip_type.origin_72,"is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3, "default_task_group_size":200, "test_qcloud_fake":False}, when test_qcloud_fake is set to True, the local CPUQVM is simulated.
    :param query_kwargs: Additional keyword parameters for querying quantum results, default: {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.
    :param sigma: Sampling variance of the multivariate non-trivial distribution, generally take pi/6, pi/12, pi/24, default is pi/24.
    :return: A module that can calculate quantum circuits.

    Example::

        import numpy as np
        import pyqpanda as pq
        import pyvqnet
        from pyvqnet.qnn import QuantumLayer,QuantumBatchAsyncQcloudLayerES
        from pyvqnet.qnn import expval_qcloud

        def qfun(input,param, m_machine, m_qlist,cubits):
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir.insert(pq.RZ(m_qlist[0],input[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
            cir.insert(pq.RY(m_qlist[1],param[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
            cir.insert(pq.RZ(m_qlist[1],input[1]))
            cir.insert(pq.RY(m_qlist[2],param[1]))
            cir.insert(pq.H(m_qlist[2]))
            m_prog.insert(cir)

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.Measure(m_qlist[ele], cubits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = QuantumBatchAsyncQcloudLayerES(qfun,
                        "3047DE8A59764BEDAC9C3282093B16AF1",
                        2,
                        6,
                        6,
                        pauli_str_dict=None,
                        shots = 1000,
                        initializer=None,
                        dtype=None,
                        name="",
                        submit_kwargs={},
                        query_kwargs={},
                        sigma=np.pi/24)
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(f"y {y}")
        y.backward()
        print(f"l.m_para.grad {l.m_para.grad}")
        print(f"x.grad {x.grad}")

        def qfun2(input,param, m_machine, m_qlist,cubits):
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir.insert(pq.RZ(m_qlist[0],input[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
            cir.insert(pq.RY(m_qlist[1],param[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
            cir.insert(pq.RZ(m_qlist[1],input[1]))
            cir.insert(pq.RY(m_qlist[2],param[1]))
            cir.insert(pq.H(m_qlist[2]))
            m_prog.insert(cir)

            return m_prog
        l = QuantumBatchAsyncQcloudLayerES(qfun2,
                    "3047DE8A59764BEDAC9C3282093B16AF",
                    2,
                    6,
                    6,
                    pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543},
                    shots = 1000,
                    initializer=None,
                    dtype=None,
                    name="",
                    submit_kwargs={},
                    query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(f"y {y}")
        y.backward()
        print(f"l.m_para.grad {l.m_para.grad}")
        print(f"x.grad {x.grad}")


QuantumLayerMultiProcess
=================================

If you are more familiar with pyQPanda syntax, please using QuantumLayerMultiProcess class, you can define the quantum circuits function by using ``qubits``, ``cbits`` and ``machine``, then take it as a argument ``qprog_with_measure`` of QuantumLayerMultiProcess.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerMultiProcess(qprog_with_measure, para_num, machine_type_or_cloud_token, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01,dtype=None, name='')

    Abstract calculation module for variational quantum circuits. This class uses multiprocess to accelerate quantum circuit simulation.
    
    It simulates a parameterized quantum circuit and gets the measurement result.
    QuantumLayer inherits from Module ,so that it can calculate gradients of circuits parameters,and train variational quantum circuits model or embed variational quantum circuits into hybird quantum and classic model.

    To use this module, you need to create your quantum virtual machine and allocate qubits and cbits.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda.
    :param para_num: `int` - Number of parameter
    :param num_of_qubits: num of qubits.
    :param num_of_cbits: num of classic bits.
    :param diff_method: 'parameter_shift' or 'finite_diff'.
    :param delta:  delta for diff.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

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

            prog = pq.QProg()
            prog.insert(circuit)

            rlt_prob = ProbsMeasure([0,2],prog,machine,qubits)
            return rlt_prob


        pqc = QuantumLayerMultiProcess(pqctest,3,4,1)
        #classic data as input
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]] )
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
=================================

In the real quantum computer, due to the physical characteristics of the quantum bit, there is always inevitable calculation error. In order to better simulate this error in quantum virtual machine, VQNet also supports quantum virtual machine with noise. The simulation of quantum virtual machine with noise is closer to the real quantum computer. We can customize the supported logic gate type and the noise model supported by the logic gate.
The existing supported quantum noise model is defined in QPanda `NoiseQVM <https://pyqpanda-toturial.readthedocs.io/zh/latest/NoiseQVM.html>`_ .

We can use ``NoiseQuantumLayer`` to define an automatic microclassification of quantum circuits. ``NoiseQuantumLayer`` supports QPanda quantum virtual machine with noise. You can define a function as an argument ``qprog_with_measure``. This function needs to contain the quantum circuit defined by pyQPanda, as also you need to pass in a argument ``noise_set_config``, by using the pyQPanda interface to set up the noise model.

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None, dtype=None, name='')

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
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer
    
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


        def circuit(weights, param, qubits, cbits, machine):

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


        def default_noise_config(qvm, q):

            p = 0.01
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.PAULI_X_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.PAULI_Y_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.PAULI_Z_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RX_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RZ_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.HADAMARD_GATE, p)
            qves = []
            for i in range(len(q) - 1):
                qves.append([q[i], q[i + 1]])  #
            qves.append([q[len(q) - 1], q[0]])
            qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE,
                                p, qves)

            return qvm


        qvc = NoiseQuantumLayer(circuit,
                                24,
                                "noise",
                                1,
                                1,
                                diff_method="parameter_shift",
                                delta=0.01,
                                noise_set_config=default_noise_config)
        input = QTensor([[0., 1., 1., 1.], [0., 0., 1., 1.], [1., 0., 1., 1.]])
        rlt = qvc(input)
        grad = QTensor(np.ones(rlt.data.shape) * 1000)

        rlt.backward(grad)
        print(qvc.m_para.grad)

        #[1195., 105., 70., 0.,
        # 45., -45., 50., 15.,
        # -80., 50., 10., -30.,
        # 10., 60., 75., -110.,
        # 55., 45., 25., 5.,
        # 5., 50., -25., -15.]

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
================================

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

.. py:class:: pyvqnet.qnn.quantumlayer.VQCLayer(vqc_wrapper, para_num, machine_type_or_cloud_token, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    Abstract Calculation module for Variational Quantum Circuits in pyQPanda.Please reference to :https://pyqpanda-toturial.readthedocs.io/zh/latest/VQG.html.

    :param vqc_wrapper: VQC_wrapper class
    :param para_num: `int` - Number of parameter
    :param machine_type: qpanda machine type
    :param num_of_qubits: num of qubits
    :param num_of_cbits: num of cbits
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for gradient calculation.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

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
================================

Qconv is a quantum convolution algorithm interface.
Quantum convolution operation adopts quantum circuit to carry out convolution operation on classical data, which does not need to calculate multiplication and addition operation, but only needs to encode data into quantum state, and then obtain the final convolution result through derivation operation and measurement of quantum circuit.
Applies for the same number of quantum bits according to the number of input data in the range of the convolution kernel, and then construct a quantum circuit for calculation.

.. image:: ./images/qcnn.png

First we need encoding by inserting :math:`RY` and :math:`RZ` gates on each qubit, then, we constructed the parameter circuit through :math:`U3` gate and :math:`Z` gate .
The sample is as follows:

.. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "cpu", dtype=None, name='')

    Quantum Convolution module. Replace Conv2D kernal with quantum circuits.Inputs to the conv module are of shape (batch_size, input_channels, height, width) reference `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_.

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param quantum_number: `int` - Size of a single kernel.
    :param stride: `tuple` - Stride, defaults to (1, 1)
    :param padding: `tuple` - Padding, defaults to (0, 0)
    :param kernel_initializer: `callable` - Defaults to normal
    :param machine: `str` - cpu simulation.
    :param dtype: The data type of the parameter, defaults: None, use the default data type kfloat32, which represents a 32-bit floating point number.
    :param name: name of the output layer

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
================

QLinear implements a quantum full connection algorithm. Firstly, the data is encoded into the quantum state, 
and then the final fully connected result is obtained through the derivation operation and measurement of the quantum circuit.

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels, output_channels, machine: str = 'cpu')

    Quantum Linear module. Inputs to the linear module are of shape (input_channels, output_channels).This layer takes no variational quantum parameters.

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

grad
================
.. py:function:: pyvqnet.qnn.quantumlayer.grad(quantum_prog_func,params *args)

    The grad function provides an interface to compute the gradient of a user-designed subcircuit with parametric parameters.
    Users can use pyqpanda to design the line running function ``quantum_prog_func`` according to the following example, and send it as a parameter to the grad function.
    The second parameter of the grad function is the coordinates at which you want to calculate the gradient of the quantum logic gate parameters.
    The return value has shape [num of parameters,num of output].

    :param quantum_prog_func: The quantum circuit operation function designed by pyqpanda.
    :param params: The coordinates of the parameters whose gradient is to be obtained.
    :param \*args: additional arguments to the quantum_prog_func function. 
    :return:
            gradient of parameters

    Examples::

        from pyvqnet.qnn import grad, ProbsMeasure
        import pyqpanda as pq

        def pqctest(param):
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(2)
            circuit = pq.QCircuit()

            circuit.insert(pq.RX(qubits[0], param[0]))

            circuit.insert(pq.RY(qubits[1], param[1]))
            circuit.insert(pq.CNOT(qubits[0], qubits[1]))

            circuit.insert(pq.RX(qubits[1], param[2]))

            prog = pq.QProg()
            prog.insert(circuit)

            EXP = ProbsMeasure([1],prog,machine,qubits)
            return EXP


        g = grad(pqctest, [0.1,0.2, 0.3])
        print(g)
        # [[-0.04673668  0.04673668]
        # [-0.09442394  0.09442394]
        # [-0.14409127  0.14409127]]



HybirdVQCQpandaQVMLayer
-----------------------------------------------------------

.. py:class:: pyvqnet.qnn.HybirdVQCQpandaQVMLayer(vqc_module: Module,qcloud_token: str, int,num_qubits: int,num_cubits: int,pauli_str_dict: Union[List[Dict], Dict, None] = None,shots: int = 1000,name: str = "",submit_kwargs: Dict = {},query_kwargs: Dict = {})
    
    Hybrid vqc and qpanda QVM layer. This layer converts quantum circuit computations defined by the user `forward` function into QPanda circuits, which can be run forward on a QPanda local VM or cloud service, and simulates circuit parameter gradients on the local CPU, reducing the time complexity of using the parameter drift method.

    :param vqc_module: vqc_module with forward(), qmachine set up correctly.
    :param qcloud_token: `str` - Type of quantum machine or cloud token for execution.
    :param num_qubits: `int` - Number of qubits in the quantum circuit.
    :param num_cubits: `int` - Number of classical bits used for measurements in the quantum circuit.
    :param pauli_str_dict: `dict|list` - Dictionary or list of dictionaries representing Pauli operators in the quantum circuit. Default is None.
    :param shots: `int` - Number of measurement shots. Default is 1000.
    :param name: Module name. Default is the empty string.
    :param submit_kwargs: Additional keyword parameters for submitting quantum circuits, default value:
                {"chip_id":pyqpanda.real_chip_type.origin_72,"is_amend":True,"is_mapping":True,"is_optimization":True,"default_task_group_size":200,"test_qcloud_fake":True}.
    :param query_kwargs: Additional keyword parameters for querying quantum results, default value:{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}。

    :return: The module that can calculate quantum circuits.

    .. note::

        pauli_str_dict cannot be None and should be the same as obs in the vqc_module measurement function.
        vqc_module should have attributes of type QMachine, and QMachine should set save_ir=True.

    Example::

        from pyvqnet.qnn.vqc  import *
        import pyvqnet
        from pyvqnet.nn import Module,Linear

        class Hybird(Module):
            def __init__(self):
                self.cl1 = Linear(3,3)
                self.ql = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)
                self.cl2 = Linear(1,2)
            
            def forward(self,x):
                x = self.cl1(x)
                x = self.ql(x)
                x = self.cl2(x)
                return x
            
        class QModel(Module):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode,save_ir=True)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.u1 = U1(has_params=True,trainable=True,wires=[2])
                self.u2 = U2(has_params=True,trainable=True,wires=[3])
                self.u3 = U3(has_params=True,trainable=True,wires=[1])
                self.i = I(wires=[3])
                self.s = S(wires=[3])
                self.x1 = X1(wires=[3])
                self.y1 = Y1(wires=[3])
                self.z1 = Z1(wires=[3])
                self.x = PauliX(wires=[3])
                self.y = PauliY(wires=[3])
                self.z = PauliZ(wires=[3])
                self.swap = SWAP(wires=[2,3])
                self.cz = CZ(wires=[2,3])
                self.cr = CR(has_params=True,trainable=True,wires=[2,3])
                self.rxx = RXX(has_params=True,trainable=True,wires=[2,3])
                self.rzz = RYY(has_params=True,trainable=True,wires=[2,3])
                self.ryy = RZZ(has_params=True,trainable=True,wires=[2,3])
                self.rzx = RZX(has_params=True,trainable=False, wires=[2,3])
                self.toffoli = Toffoli(wires=[2,3,4],use_dagger=True)
                #self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)
                self.h =Hadamard(wires=[1])
    
                self.iSWAP = iSWAP(True,True,wires=[0,2])
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={'Z0':2,'Y3':3} 
            )

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)
                self.y1(q_machine=self.qm)
                self.z(q_machine=self.qm)
                self.z1(q_machine=self.qm)
                self.ryy(q_machine=self.qm)
                self.rxx(q_machine=self.qm)
                self.rzz(q_machine=self.qm)
                self.rzx(q_machine=self.qm,params = x[:,[1]])
                self.cr(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                self.u2(q_machine=self.qm)
                self.u3(q_machine=self.qm)
                self.rx_layer(params = x[:,[0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.h(q_machine=self.qm)
                self.iSWAP(q_machine=self.qm)
                self.ry_layer(params = x[:,[1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params = x[:,[2]], q_machine=self.qm)
                self.toffoli(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        input_x = tensor.broadcast_to(input_x,[2,3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)

        l = HybirdVQCQpandaQVMLayer(qunatum_model,
                                "3047DE8A59764BEDAC9C3282093B16AF1",
                    6,
                    6,
                    pauli_str_dict={'Z0':2,'Y3':3},
                    shots = 1000,
                    name="",
            submit_kwargs={"test_qcloud_fake":True},
                    query_kwargs={})
    
        y = l(input_x)
        print(y)
        y.backward()
        print(input_x.grad)


DataParallelHybirdVQCQpandaQVMLayer 
============================================================

.. py:class:: pyvqnet.qnn.DataParallelHybirdVQCQpandaQVMLayer(vqc_module: Module,qcloud_token: str,num_qubits: int,num_cubits: int,pauli_str_dict: Union[List[Dict], Dict, None] = None,shots: int = 1000 ,dtype: Union[int, None] = None,name: str = "",submit_kwargs: Dict = {},query_kwargs: Dict = {}) A data parallel version of ``HybirdVQCQpandaQVMLayer``, where ``vqc_module`` is a user-defined quantum variational circuit model, and the QMachine setting ``save_ir= True``.

    Use data parallelism to batch the first dimension of the input data The number of processes is divided according to the number of processes allocated in `CommController`, and data parallelism is performed in multiple processes based on `mpi` or `nccl`. Please note that one process corresponds to a GPU device on one node.
    This module in each process Submit the quantum circuit generated by batch processing number/node number data in forward calculation, calculate the gradient contributed by batch processing number/node number data in reverse calculation, and calculate the average gradient of parameters on multiple nodes through all_reduce.

    .. note::

        This module splits the input internally and moves the data to the corresponding device. The 0th process calculates [0, batch number/node number] data, and the kth process calculates [(k-1) batch number/node number, k*batch number/node number]

    :param vqc_module: with forward() of vqc_module.
    :param qcloud_token: `str` - Type of quantum machine or cloud token to use for execution.
    :param num_qubits: `int` - Number of qubits in the quantum circuit.
    :param num_cubits: `int` - The number of classical bits used for measurement in the quantum circuit.
    :param pauli_str_dict: `dict|list` - A dictionary or list of dictionaries representing the Pauli operators in the quantum circuit. Default is None.
    :param shots: `int` - The number of shots in the quantum circuit. Number of line measurements. The default value is 1000.
    :param name: Module name. The default value is an empty string.
    :param submit_kwargs: Additional keyword parameters for submitting quantum circuits, default value:
    
    {"chip_id":pyqpanda.real_chip_type.origin_72,
    "is_amend":True,"is_mapping":True,
    "is_optimization":True,
    "default_task_group_size":200 ,
    "test_qcloud_fake":True}.
    :param query_kwargs: Additional keyword parameters for querying quantum results, default value: {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.

    The following is calculated using cpu For example, the command for a single node and dual processes is as follows: mpirun -n 2 python xxx.py

    Example::

        from pyvqnet.distributed import *

        Comm_OP = CommController("mpi")
        from pyvqnet.qnn import *
        from pyvqnet.qnn.vqc import *
        import pyvqnet
        from pyvqnet.nn import Module, Linear
        from pyvqnet.device import DEV_GPU_0
        pyvqnet.utils.set_random_seed(42)


        class Hybird(Module):
            def __init__(self):
                self.cl1 = Linear(3, 3)
                self.ql = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)
                self.cl2 = Linear(1, 2)

            def forward(self, x):
                x = self.cl1(x)
                x = self.ql(x)
                x = self.cl2(x)
                return x


        class QModel(Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires,
                                dtype=dtype,
                                grad_mode=grad_mode,
                                save_ir=True)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.u1 = U1(has_params=True, trainable=True, wires=[2])
                self.u2 = U2(has_params=True, trainable=True, wires=[3])
                self.u3 = U3(has_params=True, trainable=True, wires=[1])
                self.i = I(wires=[3])
                self.s = S(wires=[3])
                self.x1 = X1(wires=[3])
                self.y1 = Y1(wires=[3])
                self.z1 = Z1(wires=[3])
                self.x = PauliX(wires=[3])
                self.y = PauliY(wires=[3])
                self.z = PauliZ(wires=[3])
                self.swap = SWAP(wires=[2, 3])
                self.cz = CZ(wires=[2, 3])
                self.cr = CR(has_params=True, trainable=True, wires=[2, 3])
                self.rxx = RXX(has_params=True, trainable=True, wires=[2, 3])
                self.rzz = RYY(has_params=True, trainable=True, wires=[2, 3])
                self.ryy = RZZ(has_params=True, trainable=True, wires=[2, 3])
                self.rzx = RZX(has_params=True, trainable=False, wires=[2, 3])
                self.toffoli = Toffoli(wires=[2, 3, 4], use_dagger=True)
                #self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)
                self.h = Hadamard(wires=[1])

                self.iSWAP = iSWAP(True, True, wires=[0, 2])
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={'Z0': 2, 'Y3': 3})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)
                self.y1(q_machine=self.qm)
                self.z(q_machine=self.qm)
                self.z1(q_machine=self.qm)
                self.ryy(q_machine=self.qm)
                self.rxx(q_machine=self.qm)
                self.rzz(q_machine=self.qm)
                self.rzx(q_machine=self.qm, params=x[:, [1]])
                self.cr(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                self.u2(q_machine=self.qm)
                self.u3(q_machine=self.qm)
                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.h(q_machine=self.qm)
                self.iSWAP(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.toffoli(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt

        device = Comm_OP.get_rank
        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
        input_x = tensor.broadcast_to(input_x, [20, 3])
        input_x.requires_grad = True



        qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)

        l = DataParallelHybirdVQCQpandaQVMLayer(
            Comm_OP,
            qunatum_model,
            "3047DE8A59764BEDAC9C3282093B16AF1",

            num_qubits=6,
            num_cubits=6,
            pauli_str_dict={
                'Z0': 2,
                'Y3': 3
            },
            shots=1000,
            name="",
            submit_kwargs={"test_qcloud_fake": True},
            query_kwargs={})

        y = l(input_x)
        print(y)
        y.backward()
        for p in qunatum_model.parameters():
            print(p.grad)

    The following is an example of nccl calculation using gpu. The command for single node dual process is as follows: mpirun -n 2 python xxx.py

    Example::

        from pyvqnet.distributed import *

        Comm_OP = CommController("nccl")
        #rest code not changed

    The following is an example of multi-node multi-process parallel calculation. Please ensure that the script is run in the same path and the same python environment on different nodes, and write the ip address mapping file `hosts` on each node. The format refers to :ref:`hostfile`.

    Example::

        #hosts example
        10.10.7.107 slots=2
        10.10.7.109 slots=2

    To use mpi for 2 nodes, 2 processes per node, and 4 processes in total, you can run `vqnetrun -np 4 -f hosts python xxx.py`
    
    Example::

        from pyvqnet.distributed import *
        Comm_OP = CommController("mpi")
        #rest code not changed

    To use nccl for 2 nodes, 2 processes per node, and 4 processes in total, you can run `vqnetrun -np 4 -f hosts python xxx.py`
    
    Example::

        from pyvqnet.distributed import *
        Comm_OP = CommController("nccl")
        #rest code not changed


Quantum Gates
***********************************

The way to deal with qubits is called quantum gates. Using quantum gates, we consciously evolve quantum states. Quantum gates are the basis of quantum algorithms.

Basic quantum gates
=================================

In VQNet, we use each logic gate of `pyQPanda <https://pyqpanda-tutorial-en.readthedocs.io/en/latest/>`__ developed by the original quantum to build quantum circuit and conduct quantum simulation.
The gates currently supported by pyQPanda can be defined in pyQPanda's `quantum gate <https://pyqpanda-tutorial-en.readthedocs.io/en/latest/chapter2/index.html#quantum-logic-gate>`_ section.
In addition, VQNet also encapsulates some quantum gate combinations commonly used in quantum machine learning.


BasicEmbeddingCircuit
=================================

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
=================================

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
=================================

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
=================================

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
=================================

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
=================================

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
=================================

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


Controlled_Hadamard
================================

.. py:function:: pyvqnet.qnn.template.Controlled_Hadamard(qubits)

    Controlled Hadamard logic gates.

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param qubits: Qubits requested using pyqpanda.

    Examples::

        import pyqpanda as pq

        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(2)
        from pyvqnet.qnn import Controlled_Hadamard

        cir = Controlled_Hadamard(qubits)
        print(cir)
        # q_0:  |0>──────────────── ──■─ ──────────────
        #           ┌─────────────┐ ┌─┴┐ ┌────────────┐
        # q_1:  |0>─┤RY(-0.785398)├ ┤CZ├ ┤RY(0.785398)├
        #           └─────────────┘ └──┘ └────────────┘

CCZ
===============

.. py:function:: pyvqnet.qnn.template.CCZ(qubits)

    Controlled-controlled-Z (controlled-controlled-Z) logic gate.

    .. math::
        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}
    
    :param qubits: Qubits requested using pyqpanda.

    :return:
            pyqpanda QCircuit 

    Example::

        import pyqpanda as pq

        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(3)
        from pyvqnet.qnn import CCZ

        cir = CCZ(qubits)
        print(cir)
        # q_0:  |0>─────── ─────── ───■── ─── ────── ─────── ───■── ───■── ┤T├──── ───■──
        #                             │              ┌─┐        │   ┌──┴─┐ ├─┴───┐ ┌──┴─┐
        # q_1:  |0>────■── ─────── ───┼── ─── ───■── ┤T├──── ───┼── ┤CNOT├ ┤T.dag├ ┤CNOT├
        #           ┌──┴─┐ ┌─────┐ ┌──┴─┐ ┌─┐ ┌──┴─┐ ├─┴───┐ ┌──┴─┐ ├─┬──┘ ├─┬───┘ ├─┬──┘
        # q_2:  |0>─┤CNOT├ ┤T.dag├ ┤CNOT├ ┤T├ ┤CNOT├ ┤T.dag├ ┤CNOT├ ┤T├─── ┤H├──── ┤H├───
        #           └────┘ └─────┘ └────┘ └─┘ └────┘ └─────┘ └────┘ └─┘    └─┘     └─┘


BlockEncode
================================

.. py:function:: pyvqnet.qnn.template.BlockEncode(A,qlists)

    Construct a single pyqpanda circuit :math:`U(A)` such that an arbitrary matrix :math:`A` is encoded in the top left block.

    :param A: The input matrix encoded in the circuit.
    :param qlists: List of qubits to encode.
    :return: A pyqpanda QCircuit.

    Example::

        from pyvqnet.tensor import QTensor
        import pyvqnet
        import pyqpanda as pq
        from pyvqnet.qnn import BlockEncode
        A = QTensor([[0.1, 0.2], [0.3, 0.4]], dtype=pyvqnet.kfloat32)
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlist = machine.qAlloc_many(2)
        cbits = machine.cAlloc_many(2)

        cir = BlockEncode(A, qlist)

        prog = pq.QProg()
        prog.insert(cir)
        result = machine.directly_run(prog)
        print(cir)

        #           ┌───────────┐
        # q_0:  |0>─┤0          ├
        #           │  Unitary  │
        # q_1:  |0>─┤1          ├
        #           └───────────┘

Random_Init_Quantum_State
================================

.. py:function:: pyvqnet.qnn.template.Random_Init_Quantum_State(qlists)

    Use amplitude encoding to generate arbitrary quantum initial states and encode them onto the wire. Note that the depth of the line can vary greatly due to amplitude encoding.

    :param qlists: Qubits requested by pyqpanda.

    :return: pyqpanda QCircuit.

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.template import Random_Init_Quantum_State
        cir = pq. QCircuit()

        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)

        m_qlist = m_machine.qAlloc_many(3)
        c = Random_Init_Quantum_State(m_qlist)
        print(c)

        import pyqpanda as pq
        from pyvqnet.qnn.template import Random_Init_Quantum_State
        cir = pq.QCircuit()

        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)

        m_qlist = m_machine.qAlloc_many(3)
        c = Random_Init_Quantum_State(m_qlist)
        print(c)

        # q_0:  |0>─────────────── ─── ────────────── ─── ┤RY(0.583047)├ ─── ┤RY(0.176308)├ ─── ────────────── >
        #                              ┌────────────┐ ┌─┐ └──────┬─────┘ ┌─┐ └──────┬─────┘     ┌────────────┐ >
        # q_1:  |0>─────────────── ─── ┤RY(1.062034)├ ┤X├ ───────■────── ┤X├ ───────■────── ─── ┤RY(1.724022)├ >
        #           ┌────────────┐ ┌─┐ └──────┬─────┘ └┬┘        │       └┬┘        │       ┌─┐ └──────┬─────┘ >
        # q_2:  |0>─┤RY(1.951150)├ ┤X├ ───────■────── ─■─ ───────■────── ─■─ ───────■────── ┤X├ ───────■────── >
        #           └────────────┘ └─┘                                                      └─┘                >

        #              ┌────────────┐     ┌────────────┐
        # q_0:  |0>─── ┤RY(1.251911)├ ─── ┤RY(1.389063)├
        #          ┌─┐ └──────┬─────┘ ┌─┐ └──────┬─────┘
        # q_1:  |0>┤X├ ───────■────── ┤X├ ───────■──────
        #          └┬┘        │       └┬┘        │
        # q_2:  |0>─■─ ───────■────── ─■─ ───────■──────

FermionicSingleExcitation
================================

.. py:function:: pyvqnet.qnn.template.FermionicSingleExcitation(weight, wires, qubits)
    
    A coupled cluster single-excitation operator for exponentiating the tensor product of a Pauli matrix. The matrix form is given by:

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \}

    :param weight: Variable parameter on qubit p.
    :param wires: Indicates a subset of qubit indices in the interval [r, p]. Minimum length must be 2. The first index value is interpreted as r and the last index value as p.
                 The intermediate index is acted on by the CNOT gate to calculate the parity of the qubit set.
    :param qubits: Qubits applied by pyqpanda.

    Examples::

        from pyvqnet.qnn import FermionicSingleExcitation, expval

        weight = 0.5
        import pyqpanda as pq
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlists = machine.qAlloc_many(3)

        cir = FermionicSingleExcitation(weight, [1, 0, 2], qlists)

        prog = pq.QProg()
        prog.insert(cir)
        pauli_dict = {'Z0': 1}
        exp2 = expval(machine, prog, pauli_dict, qlists)
        print(f"vqnet {exp2}")
        #vqnet 1000000013


FermionicDoubleExcitation
================================

.. py:function:: pyvqnet.qnn.template.FermionicDoubleExcitation(weight,  wires1, wires2, qubits)
    
    The coupled clustering dual excitation operator that exponentiates the tensor product of the Pauli matrix, the matrix form is given by:

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermion annihilation and
    Create operators and indices :math:`r, s` and :math:`p, q` in the occupied and
    are empty molecular orbitals, respectively. Use the `Jordan-Wigner transformation <https://arxiv.org/abs/1208.5986>`_
    The fermion operator defined above can be written as
    According to the Pauli matrix (for more details, see `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}
    
    :param weight: variable parameter
    :param wires1: The index list of the represented qubits occupies a subset of qubits in the interval [s, r]. The first index is interpreted as s, the last as r. CNOT gates operate on intermediate indices to compute the parity of a set of qubits.
    :param wires2: The index list of the represented qubits occupies a subset of qubits in the interval [q, p]. The first index is interpreted as q, the last as p. CNOT gates operate on intermediate indices to compute the parity of a set of qubits.
    :param qubits: Qubits applied by pyqpanda.
    :return:
        pyqpanda QCircuit

    Examples::

        import pyqpanda as pq
        from pyvqnet.qnn import FermionicDoubleExcitation, expval
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlists = machine.qAlloc_many(5)
        weight = 1.5
        cir = FermionicDoubleExcitation(weight,
                                        wires1=[0, 1],
                                        wires2=[2, 3, 4],
                                        qubits=qlists)

        prog = pq.QProg()
        prog.insert(cir)
        pauli_dict = {'Z0': 1}
        exp2 = expval(machine, prog, pauli_dict, qlists)
        print(f"vqnet {exp2}")
        #vqnet 1000000058

UCCSD
===============

.. py:function:: pyvqnet.qnn.template.UCCSD(weights, wires, s_wires, d_wires, init_state, qubits)

    Realize the unitary coupled cluster single-excitation and double-excitation design (UCCSD). UCCSD is the proposed VQE design, commonly used to run quantum chemistry simulations.

    Within the first-order Trotter approximation, the UCCSD unitary function is given by:

    .. math::
        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are the fermion annihilation and
    Create operators and indices :math:`r, s` and :math:`p, q` in the occupied and
    are empty molecular orbitals, respectively. (For more details see `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):


    :param weights: ``(len(s_wires)+ len(d_wires))`` tensor containing the size of the parameters
         :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotation
         ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation``.
    :param wires: qubit index for template action
    :param s_wires: sequence of lists containing qubit indices ``[r,...,p]``
         produced by a single excitation
         :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
         where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: list sequence, each list contains two lists
         specify indices ``[s, ...,r]`` and ``[q,...,p]``
         Define double excitation :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \ vert \mathrm{HF} \rangle`.
    :param init_state: length ``len(wires)`` occupation-number vector representation
         high frequency state. ``init_state`` is the qubit initialization state.
    :param qubits: Qubits allocated by pyqpanda.

    Examples::

        import pyqpanda as pq
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn import UCCSD, expval
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlists = machine.qAlloc_many(6)
        weight = tensor.zeros([8])
        cir = UCCSD(weight,wires = [0,1,2,3,4,5,6],
                                        s_wires=[[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]],
                                        d_wires=[[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]], [[0, 1], [4, 5]]],
                                        init_state=[1, 1, 0, 0, 0, 0],
                                        qubits=qlists)

        prog = pq.QProg()
        prog.insert(cir)
        pauli_dict = {'Z0': 1}
        exp2 = expval(machine, prog, pauli_dict, qlists)
        print(f"vqnet {exp2}")
        #vqnet -1000000004


QuantumPoolingCircuit
=================================

.. py:function:: pyvqnet.qnn.template.QuantumPoolingCircuit(sources_wires, sinks_wires, params,qubits)
    
    A quantum circuit that downsamples data.
    To reduce the number of qubits in our circuit, we first create pairs of qubits in our system.
    After initially pairing all qubits, we apply our generalized 2-qubit unitary to each pair.
    After applying the two-qubit unitary, we ignore one qubit in each pair of qubits for the rest of the neural network.

    :param sources_wires: Source qubit indices that will be ignored.
    :param sinks_wires: Destination qubit indices that will be kept.
    :param params: Input parameters.
    :param qubits: list of qubits allocated by pyqpanda.

    :return:
        pyqpanda QCircuit

    Examples:: 

        from pyvqnet.qnn import QuantumPoolingCircuit
        import pyqpanda as pq
        from pyvqnet import tensor
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlists = machine.qAlloc_many(4)
        p = tensor.full([6], 0.35)
        cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
        print(cir)

        #                           ┌────┐ ┌────────────┐                           !
        # >
        # q_0:  |0>──────────────── ┤CNOT├ ┤RZ(0.350000)├ ───■── ────────────── ────! ─────────────── ────── ────────────── 
        # >
        #                           └──┬─┘ └────────────┘    │                      !                 ┌────┐ ┌────────────┐ 
        # >
        # q_1:  |0>──────────────── ───┼── ────────────── ───┼── ────────────── ────! ─────────────── ┤CNOT├ ┤RZ(0.350000)├ 
        # >
        #           ┌─────────────┐    │   ┌────────────┐ ┌──┴─┐ ┌────────────┐     !                 └──┬─┘ └────────────┘ 
        # >
        # q_2:  |0>─┤RZ(-1.570796)├ ───■── ┤RY(0.350000)├ ┤CNOT├ ┤RY(0.350000)├ ────! ─────────────── ───┼── ────────────── 
        # >
        #           └─────────────┘        └────────────┘ └────┘ └────────────┘     ! ┌─────────────┐    │   ┌────────────┐ 
        # >
        # q_3:  |0>──────────────── ────── ────────────── ────── ────────────── ────! ┤RZ(-1.570796)├ ───■── ┤RY(0.350000)├ 
        # >
        #                                                                           ! └─────────────┘        └────────────┘ 
        # >
        #                                    !
        # q_0:  |0>────── ────────────── ────!
        #                                    !
        # q_1:  |0>───■── ────────────── ────!
        #             │                      !
        # q_2:  |0>───┼── ────────────── ────!
        #          ┌──┴─┐ ┌────────────┐     !
        # q_3:  |0>┤CNOT├ ┤RY(0.350000)├ ────!


Commonly used quantum circuits
***********************************
VQNet provides some quantum circuits commonly used in quantum machine learning research.


HardwareEfficientAnsatz
=================================

.. py:class:: pyvqnet.qnn.ansatz.HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,qubits,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    The implementation of Hardware Efficient Ansatz introduced in the paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__.

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
=================================

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
=================================

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

ComplexEntangelingTemplate
================================

.. py:class:: pyvqnet.qnn.ComplexEntangelingTemplate(weights,num_qubits,depth)


    A strongly entangled layer consisting of U3 gates and CNOT gates.
    This circuit template is from the following paper: https://arxiv.org/abs/1804.00633.

    :param weights: parameter, shape of [depth,num_qubits,3]
    :param num_qubits: Number of qubits.
    :param depth: The depth of the subcircuit.

    Example::

        from pyvqnet.qnn import ComplexEntangelingTemplate
        import pyqpanda as pq
        from pyvqnet tensor import *
        depth=3
        num_qubits = 8
        shape = [depth, num_qubits, 3]
        weights = tensor.randn(shape)

        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)

        circuit = ComplexEntangelingTemplate(weights, num_qubits=num_qubits,depth=depth)
        result = circuit. create_circuit(qubits)
        circuit. print_circuit(qubits)


        # q_0:  |0>─┤U3(1.115555,-0.025096,1.326895)├── ───■── ────── ───────────────────────────────── ────────────────────────────────── >
        #           ├───────────────────────────────┴─┐ ┌──┴─┐        ┌───────────────────────────────┐                                    >
        # q_1:  |0>─┤U3(-0.884622,-0.239700,-0.701955)├ ┤CNOT├ ───■── ┤U3(0.811768,0.537290,-0.433107)├ ────────────────────────────────── >
        #           ├────────────────────────────────┬┘ └────┘ ┌──┴─┐ └───────────────────────────────┘ ┌────────────────────────────────┐ >
        # q_2:  |0>─┤U3(-0.387148,-0.322480,0.238582)├─ ────── ┤CNOT├ ───■───────────────────────────── ┤U3(-0.188015,-1.828407,0.070222)├ >
        #           ├────────────────────────────────┤         └────┘ ┌──┴─┐                            └────────────────────────────────┘ >
        # q_3:  |0>─┤U3(-0.679633,1.638090,-1.341497)├─ ────── ────── ┤CNOT├─────────────────────────── ───■────────────────────────────── >
        #           ├──────────────────────────────┬─┘                └────┘                            ┌──┴─┐                             >
        # q_4:  |0>─┤U3(2.073888,1.251795,0.238305)├─── ────── ────── ───────────────────────────────── ┤CNOT├──────────────────────────── >
        #           ├──────────────────────────────┤                                                    └────┘                             >
        # q_5:  |0>─┤U3(0.247473,2.772012,1.864166)├─── ────── ────── ───────────────────────────────── ────────────────────────────────── >
        #           ├──────────────────────────────┴─┐                                                                                     >
        # q_6:  |0>─┤U3(-1.421337,-0.866551,0.739282)├─ ────── ────── ───────────────────────────────── ────────────────────────────────── >
        #           ├────────────────────────────────┤                                                                                     >
        # q_7:  |0>─┤U3(-3.707045,0.690364,-0.979904)├─ ────── ────── ───────────────────────────────── ────────────────────────────────── >
        #           └────────────────────────────────┘                                                                                     >

        #                                                                                                                 >
        # q_0:  |0>────────────────────────────────── ────────────────────────────────── ──────────────────────────────── >
        #                                                                                                                 >
        # q_1:  |0>────────────────────────────────── ────────────────────────────────── ──────────────────────────────── >
        #                                                                                                                 >
        # q_2:  |0>────────────────────────────────── ────────────────────────────────── ──────────────────────────────── >
        #          ┌────────────────────────────────┐                                                                     >
        # q_3:  |0>┤U3(0.516395,-0.823623,-0.804430)├ ────────────────────────────────── ──────────────────────────────── >
        #          └────────────────────────────────┘ ┌────────────────────────────────┐                                  >
        # q_4:  |0>───■────────────────────────────── ┤U3(-1.420068,1.063462,-0.107385)├ ──────────────────────────────── >
        #          ┌──┴─┐                             └────────────────────────────────┘ ┌──────────────────────────────┐ >
        # q_5:  |0>┤CNOT├──────────────────────────── ───■────────────────────────────── ┤U3(0.377809,0.204278,0.386830)├ >
        #          └────┘                             ┌──┴─┐                             └──────────────────────────────┘ >
        # q_6:  |0>────────────────────────────────── ┤CNOT├──────────────────────────── ───■──────────────────────────── >
        #                                             └────┘                             ┌──┴─┐                           >
        # q_7:  |0>────────────────────────────────── ────────────────────────────────── ┤CNOT├────────────────────────── >
        #                                                                                └────┘                           >

        #          ┌────┐                                 ┌────────────────────────────────┐                                                  >
        # q_0:  |0>┤CNOT├──────────────────────────────── ┤U3(-0.460444,-1.150054,0.318044)├ ───■── ────── ────────────────────────────────── >
        #          └──┬─┘                                 └────────────────────────────────┘ ┌──┴─┐        ┌────────────────────────────────┐ >
        # q_1:  |0>───┼────────────────────────────────── ────────────────────────────────── ┤CNOT├ ───■── ┤U3(-1.255487,0.589956,-0.378491)├ >
        #             │                                                                      └────┘ ┌──┴─┐ └────────────────────────────────┘ >
        # q_2:  |0>───┼────────────────────────────────── ────────────────────────────────── ────── ┤CNOT├ ───■────────────────────────────── >
        #             │                                                                             └────┘ ┌──┴─┐                             >
        # q_3:  |0>───┼────────────────────────────────── ────────────────────────────────── ────── ────── ┤CNOT├──────────────────────────── >
        #             │                                                                                    └────┘                             >
        # q_4:  |0>───┼────────────────────────────────── ────────────────────────────────── ────── ────── ────────────────────────────────── >
        #             │                                                                                                                       >
        # q_5:  |0>───┼────────────────────────────────── ────────────────────────────────── ────── ────── ────────────────────────────────── >
        #             │┌────────────────────────────────┐                                                                                     >
        # q_6:  |0>───┼┤U3(-0.760777,-0.867848,0.016680)├ ────────────────────────────────── ────── ────── ────────────────────────────────── >
        #             │└────────────────────────────────┘ ┌────────────────────────────────┐                                                  >
        # q_7:  |0>───■────────────────────────────────── ┤U3(-1.462434,-0.173843,1.211081)├ ────── ────── ────────────────────────────────── >
        #                                                 └────────────────────────────────┘                                                  >

        #                                                                                                               >
        # q_0:  |0>───────────────────────────────── ───────────────────────────────── ──────────────────────────────── >
        #                                                                                                               >
        # q_1:  |0>───────────────────────────────── ───────────────────────────────── ──────────────────────────────── >
        #          ┌───────────────────────────────┐                                                                    >
        # q_2:  |0>┤U3(0.558638,0.218889,-0.241834)├ ───────────────────────────────── ──────────────────────────────── >
        #          └───────────────────────────────┘ ┌───────────────────────────────┐                                  >
        # q_3:  |0>───■───────────────────────────── ┤U3(0.740361,-0.336978,0.171089)├ ──────────────────────────────── >
        #          ┌──┴─┐                            └───────────────────────────────┘ ┌──────────────────────────────┐ >
        # q_4:  |0>┤CNOT├─────────────────────────── ───■───────────────────────────── ┤U3(0.585393,0.204842,0.682543)├ >
        #          └────┘                            ┌──┴─┐                            └──────────────────────────────┘ >
        # q_5:  |0>───────────────────────────────── ┤CNOT├─────────────────────────── ───■──────────────────────────── >
        #                                            └────┘                            ┌──┴─┐                           >
        # q_6:  |0>───────────────────────────────── ───────────────────────────────── ┤CNOT├────────────────────────── >
        #                                                                              └────┘                           >
        # q_7:  |0>───────────────────────────────── ───────────────────────────────── ──────────────────────────────── >
        #                                                                                                               >

        #                                              ┌────┐                               ┌───────────────────────────────┐ >
        # q_0:  |0>─────────────────────────────────── ┤CNOT├────────────────────────────── ┤U3(0.657827,1.434924,-0.328996)├ >
        #                                              └──┬─┘                               └───────────────────────────────┘ >
        # q_1:  |0>─────────────────────────────────── ───┼──────────────────────────────── ───────────────────────────────── >
        #                                                 │                                                                   >
        # q_2:  |0>─────────────────────────────────── ───┼──────────────────────────────── ───────────────────────────────── >
        #                                                 │                                                                   >
        # q_3:  |0>─────────────────────────────────── ───┼──────────────────────────────── ───────────────────────────────── >
        #                                                 │                                                                   >
        # q_4:  |0>─────────────────────────────────── ───┼──────────────────────────────── ───────────────────────────────── >
        #          ┌─────────────────────────────────┐    │                                                                   >
        # q_5:  |0>┤U3(-2.134247,-0.783461,-0.200094)├ ───┼──────────────────────────────── ───────────────────────────────── >
        #          └─────────────────────────────────┘    │┌──────────────────────────────┐                                   >
        # q_6:  |0>───■─────────────────────────────── ───┼┤U3(1.816030,0.572931,1.683584)├ ───────────────────────────────── >
        #          ┌──┴─┐                                 │└──────────────────────────────┘ ┌───────────────────────────────┐ >
        # q_7:  |0>┤CNOT├───────────────────────────── ───■──────────────────────────────── ┤U3(0.661537,0.214565,-0.325014)├ >
        #          └────┘                                                                   └───────────────────────────────┘ >

        #                                                           ┌────┐
        # q_0:  |0>───■── ────── ────── ────── ────── ────── ────── ┤CNOT├
        #          ┌──┴─┐                                           └──┬─┘
        # q_1:  |0>┤CNOT├ ───■── ────── ────── ────── ────── ────── ───┼──
        #          └────┘ ┌──┴─┐                                       │
        # q_2:  |0>────── ┤CNOT├ ───■── ────── ────── ────── ────── ───┼──
        #                 └────┘ ┌──┴─┐                                │
        # q_3:  |0>────── ────── ┤CNOT├ ───■── ────── ────── ────── ───┼──
        #                        └────┘ ┌──┴─┐                         │
        # q_4:  |0>────── ────── ────── ┤CNOT├ ───■── ────── ────── ───┼──
        #                               └────┘ ┌──┴─┐                  │
        # q_5:  |0>────── ────── ────── ────── ┤CNOT├ ───■── ────── ───┼──
        #                                      └────┘ ┌──┴─┐           │
        # q_6:  |0>────── ────── ────── ────── ────── ┤CNOT├ ───■── ───┼──
        #                                             └────┘ ┌──┴─┐    │
        # q_7:  |0>────── ────── ────── ────── ────── ────── ┤CNOT├ ───■──

Quantum_Embedding
=================================

.. py:class:: pyvqnet.qnn.Quantum_Embedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions)

    Use RZ,RY,RZ to create variational quantum circuits that encode classical data into quantum states.
    See `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.
    After the class is initialized, its member function ``compute_circuit`` is a running function, which can be input as a parameter.
    The ``QuantumLayerV2`` class can utilize ``compute_circuit`` to build a layer of quantum machine learning model.

    :param qubits: Qubits requested by pyqpanda.
    :param machine: Quantum virtual machine applied by pyqpanda.
    :param num_repetitions_input: Number of repetitions to encode input in the submodule.
    :param depth_input: The feature dimension of the input data.
    :param num_unitary_layers: Number of repetitions of the variational quantum gates in each submodule.
    :param num_repetitions: Number of repetitions for the submodule.

    Example::

        from pyvqnet.qnn import QuantumLayerV2,Quantum_Embedding
        from pyvqnet.tensor import tensor
        import pyqpanda as pq
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2

        loacl_machine = pq.CPUQVM()
        loacl_machine.init_qvm()
        nq = depth_input * num_repetitions_input
        qubits = loacl_machine.qAlloc_many(nq)
        cubits = loacl_machine.cAlloc_many(nq)

        data_in = tensor.ones([12, depth_input])

        qe = Quantum_Embedding(qubits, loacl_machine, num_repetitions_input,
                            depth_input, num_unitary_layers, num_repetitions)
        qlayer = QuantumLayerV2(qe.compute_circuit,
                                qe.param_num)

        data_in.requires_grad = True
        y = qlayer.forward(data_in)
        print(y)
        # [
        # [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894],
        #  [0.2302894]
        # ]


Measure the quantum circuit
***********************************

expval_qcloud
=================================

.. py:function:: pyvqnet.qnn.measure.expval_qcloud(machine, prog, pauli_str_dict, qlists,clists,shots=1000,qtype = pq.real_chip_type.origin_72)

    Expectation value of the supplied Hamiltonian observables of QCloud.

    if the observables are :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`,
    then ``Hamiltonian`` ``dict`` would be ``{{'Z0, X1':0.7} ,{'Z1':0.2}}`` .

    :param machine: machine created by qpanda
    :param prog: quantum program created by qpanda
    :param pauli_str_dict: Hamiltonian observables
    :param qlists: qubit allocated by pyQPanda
    :param clists: cbit allocated by pyQPanda
    :param shots: measure times, default:1000.
    :param qtype: Set the type of qmachine measurement, the default is "" indicating non-qcloud. Set `pq.real_chip_type.origin_72` for real chips.
    :return: expectation


    Example::

        from pyqpanda import *
        input = [0.56, 0.1]

        m_machine = QCloud()

        m_machine.init_qvm("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(4)
        m_clist = m_machine.cAlloc_many(4)
        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0],input[0]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[3]))
        cir.insert(pq.RY(m_qlist[1],input[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
        m_prog.insert(cir)
        pauli_dict  = {'Z0 X1':10,'Y2':-0.543}

        from pyvqnet.qnn import expval_qcloud
        exp2 = expval_qcloud(m_machine,m_prog,pauli_dict,m_qlist,m_clist,shots=100)
        print(exp2)

expval
=================================

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
=================================

.. py:function:: pyvqnet.qnn.measure.QuantumMeasure(measure_qubits:list,prog,machine,qubits,shots:int = 1000, qtype="")

    Calculates circuits quantum measurement. Return the normalized result of the measurements obtained by the Monte Carlo method.
    
    Please checks  https://pyqpanda-toturial.readthedocs.io/zh/latest/Measure.html?highlight=measure_all for alternative api.
    
    QuantumMeasure api only supports on QPanda ``CPUQVM`` or ``QCloud`` now.

    :param measure_qubits: list contains measure qubits index.
    :param prog: quantum program from qpanda
    :param machine: quantum virtual machine allocated by pyQPanda
    :param qubits: qubit allocated by pyQPanda
    :param shots: measure time,default 1000
    :param qtype: Set the type of qmachine measurement, the default is "" indicating non-qcloud. Set `pq.real_chip_type.origin_72` for real chips.
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
=================================

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
================================
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
===============
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
===============
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
================================
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
================================
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


Purity
================================

.. py:function:: pyvqnet.qnn.measure.Purity(state, qubits_idx)

    Calculate the purity on a particular qubit from the state vector.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is a density matrix. The purity of a normalized
    quantum state satisfies :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    where :math:`d` is the dimension of the Hilbert space.
    The purity of the pure state is 1.

    :param state: Quantum state obtained from pyqpanda get_qstate()
    :param qubits_idx: index of qubits whose purity is to be calculated

    :return:
        purity

    Examples::

        from pyvqnet.qnn import Purity
        qstate = [(0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
                (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
                -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j,
                -0.048171819846746615j]
        pp = Purity(qstate, [1])
        print(pp)
        #0.902503479761881



Quantum Machine Learning Algorithm Interface
*****************************************************

Quantum Generative Adversarial Networks for learning and loading random distributions
==================================================================================================

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
=================================

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
=================================================================


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


    .. warning::

        SPSA only supports 1-dim paramters.

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

Quantum fisher information computation matrix
========================================================

.. py:class:: pyvqnet.qnn.opt.quantum_fisher(py_qpanda_config, params, target_gate_type_lists,target_gate_bits_lists, qcir_lists, wires)
    
    Returns the quantum fisher information matrix for a quantum circuit.

    .. math::

        \mathrm{QFIM}_{i, j}=4 \operatorname{Re}\left[\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle-\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \psi(\boldsymbol{\theta})\right\rangle\left\langle\psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle\right]

    The short version is :math::math:`\left|\partial_j \psi(\boldsymbol{\theta})\right\rangle:=\frac{\partial}{\partial \theta_j}|\psi(\boldsymbol{\theta})\rangle`.

    .. note::

        Currently only RX,RY,RZ are supported.

    :param params: Variable parameters in circuits.
    :param target_gate_type_lists: Supports "RX", "RY", "RZ" or lists.
    :param target_gate_bits_lists:  Which quantum bit or bits the parameterised gate acts on .
    :param qcir_lists: The list of quantum circles before the target parameterised gate to compute the metric tensor, see the following example.
    :param wires: Total Quantum Bit Index for Quantum Circuits.

    Example::
    
        import pyqpanda as pq

        from pyvqnet import *
        from pyvqnet.qnn.opt import pyqpanda_config_wrapper, insert_pauli_for_mt, quantum_fisher
        from pyvqnet.qnn import ProbsMeasure
        import numpy as np
        import pennylane as qml
        import pennylane.numpy as pnp

        n_wires = 4
        def layer_subcircuit_new(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1]))
            
            qcir.insert(pq.RZ(config._qubits[2], params[2]))
            qcir.insert(pq.RZ(config._qubits[3], params[3]))
            return qcir


        def get_p1_diagonal_new(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_new(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)

        config = pyqpanda_config_wrapper(n_wires)
        qcir = []
        
        qcir.append(get_p1_diagonal_new)

        params2 = QTensor([0.5, 0.5, 0.5, 0.25], requires_grad=True)

        mt = quantum_fisher(config, params2, [['RX', 'RY', 'RZ', 'RZ']],
                                [[0, 1, 2, 3]], qcir, [0, 1, 2, 3])

        # The above example shows that there are no identical gates in the same layer, 
        # but in the same layer you need to modify the logic gates according to the following example.
        
        n_wires = 3
        def layer_subcircuit_01(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1]))
            
            return qcir

        def layer_subcircuit_02(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1]))
            qcir.insert(pq.RZ(config._qubits[1], params[2]))
            return qcir

        def layer_subcircuit_03(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1])) #  01 part
            
            qcir.insert(pq.RZ(config._qubits[1], params[2]))  #  02 part
            
            qcir.insert(pq.RZ(config._qubits[1], params[3]))
            return qcir

        def get_p1_diagonal_01(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_01(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)
        
        def get_p1_diagonal_02(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_02(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)
        
        def get_p1_diagonal_03(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_03(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)
        
        config = pyqpanda_config_wrapper(n_wires)
        qcir = []
        
        qcir.append(get_p1_diagonal_01)
        qcir.append(get_p1_diagonal_02)
        qcir.append(get_p1_diagonal_03)
        
        params2 = QTensor([0.5, 0.5, 0.5, 0.25], requires_grad=True)

        mt = quantum_fisher(config, params2, [['RX', 'RY'], ['RZ'], ['RZ']], # rx,ry counts as layer one, first rz as layer two, second rz as layer three.
                                [[0, 1], [1], [1]], qcir, [0, 1])

