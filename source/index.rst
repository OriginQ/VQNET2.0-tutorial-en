.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================

Core Features of VQNet
------------------------

Multi-platform compatibility and cross-environment support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet supports users to conduct research and development of quantum machine learning in a variety of hardware and operating system environments. Whether using CPU or GPU for quantum computing simulation or calling real quantum chips through Benyuan Quantum Cloud Service, VQNet can provide seamless support. Currently, VQNet is compatible with python3.9, python3.10, and python3.11 versions of Windows, Linux, and macOS systems.

Perfect interface design and ease of use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet uses Python as the front-end language, provides a function interface similar to PyTorch, and can freely choose a variety of computing backends to implement the automatic differentiation function of classical quantum machine learning models. The framework has built-in: 100+ commonly used Tensor computing interfaces, 100+ quantum variational circuit computing interfaces, and 50+ classical neural network interfaces. These interfaces cover the complete development process from classical machine learning to quantum machine learning, and will be continuously updated.

Efficient computing performance and expansion capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Real quantum chip experiment support**: For users who need real quantum chip experiments, VQNet integrates the original pyQPanda interface, and combines the efficient scheduling capabilities of the original Sinan to achieve fast quantum circuit simulation calculations and real chip operation.
- **Local computing optimization**: For local computing needs, VQNet provides a quantum machine learning programming interface based on CPU or GPU, and uses automatic differentiation technology to perform quantum variational circuit gradient calculations, which is significantly faster than traditional parameter drift methods (such as Qiskit).
- **Distributed computing support**: VQNet supports MPI-based distributed computing, which can realize the function of training large-scale hybrid quantum-classical neural network models on multiple nodes.

Rich application scenarios and example support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet is not only a powerful development tool, but also widely used in multiple projects within the company, including power optimization, medical data analysis, image processing and other fields. In order to help users get started quickly, VQNet provides a variety of scenarios ranging from basic tutorials to advanced applications in the official website and API online documentation. These resources enable users to easily understand how to use VQNet to solve practical problems and quickly build their own quantum machine learning applications.

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
    rst/FAQ.rst 
    rst/CHANGELOG.rst




