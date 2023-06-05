.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================


**A fully functional and efficient quantum software development kit**

VQNet is a quantum machine learning computing framework developed by Origin Quantum, which can be used to build, run and optimize quantum algorithms.
This documentation is the VQNet API and samples' documentation. Chinese api document link reference: `VQNet API DOC <https://vqnet20-tutorial.readthedocs.io/en/main/index.html>`_ .


**What is Quantum Machine Learning?**

Quantum machine learning is a research field that explores the interaction between quantum computing and machine learning ideas.
For example, we might want to find out whether quantum computers can speed up the time it takes to train or evaluate a machine learning model. On the other hand, we can leverage techniques from machine learning to help us uncover quantum error-correcting codes, estimate the properties of quantum systems, or develop new quantum algorithms.

**Quantum computers as AI accelerators**
 
The limits of what machines can learn have always been defined by the computer hardware we run our algorithms on—for example, the success of modern-day deep learning with neural networks is enabled by parallel GPU clusters.
Quantum machine learning extends the pool of hardware for machine learning by an entirely new type of computing device—the quantum computer. Information processing with quantum computers relies on substantially different laws of physics known as quantum theory.
In the modern viewpoint, quantum computers can be used and trained like neural networks. We can systematically adapt the physical control parameters, such as an electromagnetic field strength or a laser pulse frequency, to solve a problem.
For example, a trained circuit can be used to classify the content of images, by encoding the image into the physical state of the device and taking measurements.

**The bigger picture: differentiable programming**

But our goal is not just to use quantum computers to solve machine learning problems. Quantum circuits are differentiable, and a quantum computer itself can compute the change in control parameters needed to become better at a given task.
Differentiable programming is the very basis of deep learning, Differentiable programming is more than deep learning: it is a programming paradigm where the algorithms are not hand-coded, but self-learned. `VQNet` is also implemented based on `autograd`.
Similarly, the significance of training quantum computers is larger than quantum machine learning. Trainable quantum circuits can be leveraged in other fields like quantum chemistry or quantum optimization.
It can promote in a variety of applications such as the design of quantum algorithms, the discovery of quantum error correction schemes, and the understanding of physical systems.

**VQNet characteristics**
 
•	Unity. We propose the first new-generation machine learning framework, which not only realizes the unification of classical and quantum machine learning but also supports deployment on classical and quantum computers.
•	Practicality. The proposed VQNet accomplishes the necessary features of a new generation machine learning framework, such as friendly interface, automatic differentiation, and dynamic computational graph, under the design concept with practicability.
•	Efficiency. In the implementation with high performance, VQNet designs a unified structure and uses QPanda to improve the efficiency of interaction between classical and quantum machine learning algorithms, between classic and quantum computers.
•	Numerous application examples. We provide a number of quantum machine learning examples for quick learning and using.

.. toctree::
    :caption: Installation Guide
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: Hands-on Examples
    :maxdepth: 2

    rst/qml_demo.rst

.. toctree::
    :caption: The API
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/qnn.rst
    rst/utils.rst
    rst/FAQ.rst
    rst/CHANGELOG.rst





