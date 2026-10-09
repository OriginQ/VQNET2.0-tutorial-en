.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================

VQNet is a quantum machine learning framework independently developed by Origin Quantum. It takes the parameterized quantum circuit (PQC) as its core computing primitive, embeds it into classical neural networks as a differentiable operator, and relies on automatic differentiation to enable end-to-end construction and training of hybrid quantum-classical models. It also supports invoking Origin quantum computers and quantum cloud services for circuit simulation and real-chip experiments. The framework provides a complete interface system covering tensor computation, classical neural networks, quantum circuit simulation with automatic differentiation, distributed training, and real hardware deployment.

This document is the API and example documentation of VQNet.

Core Features of VQNet
------------------------

Multi-platform compatibility and cross-environment support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet supports users to conduct research and development of quantum machine learning in a variety of hardware and operating system environments. Whether using CPU or GPU for quantum computing simulation, or calling real quantum chips through Origin quantum cloud services, VQNet can provide seamless support. Currently, VQNet is compatible with Python 3.10 - Python 3.14 on Windows (x86), Linux (x86), and macOS (ARM).

Perfect interface design and ease of use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet uses Python as the front-end language and provides a programming interface similar to mainstream neural network training frameworks. Through ``pyvqnet.backends.set_backend`` , users can choose between the ``pyvqnet`` (native) and ``torch`` computing backends. The interfaces of the two backends are independent of each other, and a model must run under a single backend, so users may choose according to their ecosystem needs. These interfaces cover the complete development process from classical machine learning to quantum machine learning, and will be continuously updated.

- **Native backend**: runs on the ``pyvqnet`` computing backend. The tensor library ``pyvqnet.tensor`` provides 100+ common computing interfaces with automatic differentiation support; the neural network module ``pyvqnet.nn`` covers 50+ interfaces including convolution, pooling, recurrent neural networks, Transformer attention, various normalizations, and optimizers; the quantum circuit module ``pyvqnet.qnn.vqc`` provides the state-vector simulation ``QMachine`` , the tensor network backend ``tn.native`` , and quantum-classical fused layers such as QLSTM, QRNN, QMLP, and Quanvolution; LLM operators (RoPE, SwiGLU, fused MoE, scaled Softmax, top-k/top-p/min-p sampling, etc.) are also provided by the native backend.
- **torch backend**: for scenarios that need to collaborate with the PyTorch ecosystem, it runs on the ``torch`` computing backend and provides torch counterparts of the native interfaces — classical layers ``pyvqnet.nn.torch`` ( ``TorchModule`` , ``Linear`` , ``Conv2D`` , etc.), quantum circuits ``pyvqnet.qnn.vqc.sv.torch`` and the tensor network ``pyvqnet.qnn.vqc.tn.torch`` , which can be nested with ``torch.nn`` modules and take part in ``torch`` automatic differentiation; the LLM fine-tuning loss family ``pyvqnet.torch.trl`` (SFT, DPO, PPO, GRPO, Reward) is also based on this backend and can be combined with peft/LlamaFactory to enable quantum-circuit-based large model fine-tuning.

Efficient computing performance and expansion capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Dual quantum simulation backends**: provides the ``QMachine`` interface based on state-vector evolution (50+ quantum logic gates, the measurement family such as ``MeasureAll`` , ``Probability`` and ``Samples`` , the ``QuantumLayerAdjoint`` adjoint-gradient interface, and circuit templates such as UCCSD, QSVT, and the hardware-efficient ansatz); it also provides a tensor-network / matrix-product-state backend ``pyvqnet.qnn.vqc.tn`` for larger-scale circuits, with both ``native`` and ``torch`` implementations. The framework has built-in circuit compilation and optimization passes (rotation merging, controlled-gate commuting, etc.) and supports OriginIR import and export.
- **Real quantum chip experiment support**: for users who need real quantum chip experiments, VQNet integrates the Origin pyqpanda3 interface and combines the efficient scheduling capabilities of Origin Sinan to achieve fast quantum circuit simulation and real chip execution; the pq3 series quantum layers support asynchronous batch cloud submission, bridging the path from simulation to real hardware.
- **Local computing optimization**: for local computing needs, VQNet provides quantum machine learning programming interfaces based on CPU or GPU, and uses automatic differentiation technology to perform variational quantum circuit gradient calculations, which is significantly more efficient than the traditional parameter-shift method; see :ref:`benchmarks` . Fused CUDA operators are provided for high-frequency operations such as RX, RY, RZ, CNOT, and measurement, reducing the training and simulation time of large-scale parameterized circuits.
- **Distributed computing support**: VQNet has a built-in multi-process distributed runtime based on gloo/NCCL collective communication (the ``vqnetrun`` launcher), supporting data parallelism, tensor parallelism ( ``ColumnParallelLinear`` / ``RowParallelLinear`` ), pipeline parallelism, and hybrid parallel strategies ( ``ParallelTrainingWrapper`` ), and provides ``DistributedQMachine`` / ``DistQuantumLayerAdjoint`` to support quantum circuit gradient computation and hybrid quantum-classical model training in multi-process, multi-node environments.

Rich application scenarios and example support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet is not only a powerful development tool, but has also been widely used in multiple projects within the company, including power optimization, medical data analysis, image processing, and other fields. To help users get started quickly, VQNet provides scenarios and examples ranging from basic tutorials (variational quantum circuits, quantum machine learning examples) to advanced applications (distributed hybrid training, quantum LLM fine-tuning) on the official website and in the online API documentation. These resources enable users to easily understand how to use VQNet to solve practical problems and quickly build their own quantum machine learning applications.

.. toctree::
    :caption: Installation Guide
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: Hands-on Examples
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: Classic neural network API
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: QNN API integrated with pyqpanda
    :maxdepth: 2

    rst/qnn.rst
    rst/qnn_pq3.rst

.. toctree::
    :caption: Autograd QNN API
    :maxdepth: 2

    rst/vqc.rst

.. toctree::
    :caption: Quantum Large Model Fine-Tuning
    :maxdepth: 2

    rst/llm.rst

.. toctree:: 
    :caption: Others 
    :maxdepth: 2 
    
    rst/torch_api.rst
    rst/vqnet_dist.rst
    rst/FAQ.rst 
    rst/CHANGELOG.rst




