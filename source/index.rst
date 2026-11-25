.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================



VQNet is a quantum machine learning framework independently developed by Origin Quantum. It uses the parameterized quantum circuit (PQC) as its core computational primitive and embeds it as a differentiable operator in classical neural networks. Powered by automatic differentiation, it enables end-to-end construction and training of hybrid quantum-classical models, and supports running circuits on Origin Quantum computers and quantum cloud services for simulation and real-chip experiments. The framework provides a complete set of interfaces covering tensor computation, classical neural networks, quantum circuit simulation and automatic differentiation, distributed training, and real hardware deployment.

This document covers the VQNet API and examples.


Core Features of VQNet
----------------------

Multi-platform compatibility and cross-environment support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet supports research and development of quantum and classical machine learning across a variety of hardware and operating system environments. Whether you use CPUs or GPUs for quantum simulation, or call real quantum chips through the Origin Quantum cloud service, VQNet provides seamless support. VQNet currently supports Python 3.10 to 3.14 on Windows (x86), Linux (x86), and macOS (ARM).

Well-designed interfaces and ease of use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet uses Python as its front-end language and provides programming interfaces similar to mainstream neural network training frameworks. The ``pyvqnet.backends.set_backend`` call selects between the ``pyvqnet`` (native) and ``torch`` compute backends. The two backends have independent interfaces, and a given model must run on a single backend; users can choose according to their ecosystem needs. These interfaces cover the full development flow from classical machine learning to quantum machine learning and will continue to be updated.

- **Native backend**: runs on the ``pyvqnet`` compute backend. The tensor library ``pyvqnet.tensor`` provides 100+ common computation interfaces with automatic differentiation support. The neural network module ``pyvqnet.nn`` covers convolution, pooling, recurrent neural networks, Transformer attention, various normalization methods, optimizers, and other 50+ interfaces. The quantum circuit module ``pyvqnet.qnn.vqc`` provides state vector simulation ``QMachine``, the tensor-network backend ``tn.native``, and quantum-classical hybrid layers such as QLSTM, QRNN, QMLP, and Quanvolution. Large-model operators (RoPE, SwiGLU, fused MoE, scaled Softmax, top-k/top-p/min-p sampling, etc.) are also provided by the native backend.
- **torch backend**: for scenarios that need to work with the PyTorch ecosystem, the model runs on the ``torch`` compute backend. It provides torch versions corresponding to the native interfaces: classical layers in ``pyvqnet.nn.torch`` (``TorchModule``, ``Linear``, ``Conv2D``, etc.), quantum circuits in ``pyvqnet.qnn.vqc.sv.torch``, and tensor networks in ``pyvqnet.qnn.vqc.tn.torch``. These can be nested with ``torch.nn`` modules and participate in ``torch`` automatic differentiation. The large-model fine-tuning loss family ``pyvqnet.torch.trl`` (SFT, DPO, PPO, GRPO, Reward) is also based on this backend and can work with peft/LlamaFactory for large-model fine-tuning built on quantum circuits.

Efficient compute performance and scalability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Dual quantum simulation backends**: provides the ``QMachine`` interface based on state vector evolution (50+ quantum logic gates, the measurement family such as ``MeasureAll``, ``Probability``, and ``Samples``, the ``QuantumLayerAdjoint`` adjoint gradient interface, and circuit templates such as UCCSD, QSVT, and hardware-efficient ansatz). It also provides a tensor-network/matrix product state backend ``pyvqnet.qnn.vqc.tn`` for circuits with larger qubit counts, with both ``native`` and ``torch`` implementations. The framework includes circuit compilation optimization passes such as rotation gate merging and controlled gate commutation, and supports OriginIR circuit import and export.
- **Real quantum chip experiment support**: for users who need real quantum chip experiments, VQNet integrates the Origin pyqpanda3 interface and combines it with the efficient scheduling of Origin Sinan to enable fast quantum circuit simulation and real-chip execution. The pq3 quantum layer family supports asynchronous batch cloud submission, opening a path from simulation to real hardware.
- **Local compute optimization**: for local workloads, VQNet provides quantum machine learning programming interfaces on CPU or GPU and uses automatic differentiation to compute gradients of variational quantum circuits. Compared with the traditional parameter-shift method, gradient computation is markedly more efficient; see :ref:`benchmarks` for performance results. Fused CUDA operators are provided for high-frequency operations such as RX, RY, RZ, CNOT, and measurement, reducing the training and simulation time of large-scale parameterized circuits.
- **Distributed computing support**: VQNet includes a multi-process distributed runtime based on gloo/NCCL collective communication (the ``vqnetrun`` launcher) and supports data parallelism, tensor parallelism (``ColumnParallelLinear`` / ``RowParallelLinear``), pipeline parallelism, and hybrid parallelism strategies (``ParallelTrainingWrapper``). ``DistributedQMachine`` / ``DistQuantumLayerAdjoint`` support quantum circuit gradient computation and hybrid quantum-classical model training in multi-process, multi-node environments.

Rich application scenarios and examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet is more than a powerful development tool; it has also been widely used in several internal projects, including power optimization, medical data analysis, and image processing. To help users get started quickly, VQNet offers scenarios and examples on its official website and in the online API documentation, ranging from basic tutorials (variational quantum circuits, quantum machine learning examples) to advanced applications (distributed hybrid training, quantum large-model fine-tuning). These resources make it easy to understand how VQNet solves real problems and to quickly build your own quantum machine learning applications.



.. toctree::
    :caption: Installation
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: Getting Started Examples
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: Classical Neural Network Interfaces
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: Quantum Neural Network Interfaces Using pyqpanda3
    :maxdepth: 2


    rst/qnn_pq3.rst

.. toctree::
    :caption: Quantum Neural Network Automatic Differentiation Simulation Interfaces
    :maxdepth: 2

    rst/vqc.rst


.. toctree::
    :caption: Quantum Large Model Fine-Tuning
    :maxdepth: 2

    rst/llm.rst

.. toctree::
    :caption: Other
    :maxdepth: 2

    rst/vqnet_dist.rst
    rst/torch_api.rst
    rst/FAQ.rst
    rst/CHANGELOG.rst
