FAQ
================

**Q: What features does VQNet provide?**

A: VQNet is a quantum machine learning toolkit developed on Origin Quantum's pyQPanda3. It provides rich, easy-to-use classical neural network computation module interfaces for convenient machine learning optimization. Model definition follows the same style as mainstream machine learning frameworks, reducing the learning curve.
Built on the high-performance quantum simulator pyQPanda3 developed by Origin Quantum, VQNet can also handle large numbers of qubits on ordinary computers. Finally, VQNet offers a rich set of :doc:`./qml_demo` for reference and learning.


**Q: How do I train a quantum machine learning model with VQNet?**

A: One class of quantum machine learning algorithms builds differentiable quantum machine learning models from variational quantum circuits.
VQNet can train such models with gradient descent. The general steps are as follows. First, on a local computer, users can build a virtual machine with pyQPanda and, together with the interfaces provided by VQNet, construct a quantum or hybrid quantum-classical model ``Module``. Second, calling the ``Module``'s ``forward()`` performs quantum circuit simulation and classical neural network forward computation in the way the user defines.
When the ``Module``'s ``backward()`` is called, the model can perform automatic differentiation like classical machine learning frameworks such as PyTorch, computing parameter gradients in the variational quantum circuit and classical computation layers. Finally, the optimizer's ``step()`` updates the parameters.

In VQNet, we use `parameter-shift <https://arxiv.org/abs/1803.00745>`_ to compute gradients of variational quantum circuits. Users can use the interfaces under :ref:`QuantumLayer_pq3` provided by VQNet, which already wrap automatic differentiation of variational quantum circuits; users only need to define the variational quantum circuit in the required format as a parameter to construct the above classes.

In VQNet, we can also compute gradients of variational quantum circuits using automatic differentiation. Users can build trainable circuits with the interfaces in :ref:`vqc_api`. Such circuits do not depend on pyQPanda; instead, encoding, logic gate evolution, and measurement are split into differentiable operators, enabling gradient computation with respect to their parameters.

For details, see the relevant interfaces and example code in this document.

**Q: On Windows, installing VQNet fails with: "importError: DLL load failed while importing _core: The specified module could not be found."**

A: On Windows, you may need to install the VC++ runtime library.
See https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 to install a suitable runtime.
In addition, VQNet currently supports only Python 3.10 to 3.14, so check your Python version.

**Q: How do I use the Origin Quantum cloud and quantum chips for computation?**

A: You can use Origin Quantum's high-performance computer cluster or real quantum computers for quantum circuit simulation, replacing local quantum circuit simulation with cloud computing. See https://qcloud.originqc.com.cn/zh.
In VQNet, users can use ``QuantumBatchAsyncQcloudLayer`` to build a variational quantum circuit module and submit tasks with the API KEYS requested on the Origin website to run on real hardware.


**Q: Why don't the model parameters I defined update during training?**

A: A VQNet model must be built so that every module it uses is differentiable. If a module cannot compute gradients, that module and all preceding modules cannot compute gradients through the chain rule.
If you define a custom variational quantum circuit, use the interfaces under :ref:`QuantumLayer_pq3` provided by VQNet. For classical machine learning modules, use the interfaces defined in :doc:`./QTensor` and :doc:`./nn`; these interfaces wrap gradient computation functions, so VQNet can perform automatic differentiation.

If you want to use a list containing multiple modules as a submodule in `Module`, do not use Python's built-in List. Use pyvqnet.nn.module.ModuleList instead of List. This way, the submodules' training parameters are registered in the whole model and can be trained with automatic differentiation.


**Q: Why does my existing code fail to run in 2.0.7 and later?**

A: Since v2.0.7, we added different data types and a dtype attribute to QTensor, and restricted inputs following PyTorch. For example, the input to the Embedding layer must be kint64, and the labels for CategoricalCrossEntropy, SoftmaxCrossEntropy, NLL_Loss, and CrossEntropyLoss must be kint64.
You can use the `astype()` interface to convert to the specified data type, or initialize the QTensor with a numpy array of the corresponding data type.

**Q: Does VQNet depend on torch?**

A: VQNet does not depend on torch and does not install torch automatically.

Using the following features requires installing torch>=2.11.0 separately. Since v2.15.0, we support `torch >=2.11.0 <https://docs.pytorch.org/docs/stable/index.html>`_ as a compute backend for classical neural networks, variational quantum circuits, distributed computation, and more.
After calling ``pyvqnet.backends.set_backend("torch")``, the interfaces stay the same, but the ``data`` member of VQNet's ``QTensor`` is stored as a ``torch.Tensor`` and computed with torch. Classes under ``pyvqnet.nn.torch`` and ``pyvqnet.qnn.vqc.torch`` inherit from ``torch.nn.Module`` and can form ``torch`` models.
