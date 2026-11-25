
VQNet Changelog
######################


[v2.18.2] - 2026-08-17
***************************

Added
===================
- Added the unified quantum layer ``VQCLayer`` (``pyvqnet.qnn.pq3.vqclayer.VQCLayer``) interface, which selects the execution backend via ``submit_kwargs["backend"]`` (``vqc_autograd`` / ``qcloud_fake`` / ``qcloud_service`` / ``qpanda_runtime``) and supports two mutually exclusive measurement modes: expectation value and probability.
- Added the unified quantum layer ``TorchVQCLayer`` for the torch backend, with the same functionality as the native ``VQCLayer``.
- Added the hybrid quantum + classical multi-level distributed training interface `ParallelTrainingWrapper`.

Changed
===================
- The distributed computing module removes the MPI backend; CPU communication now uses ``gloo`` exclusively. The default backend of ``CommController`` changes from ``mpi`` to ``gloo``, and ``init_groups`` no longer has an MPI-only restriction.
- The ``submit_kwargs["backend"]`` parameter value ``vqc_autograd`` of ``VQCLayer`` / ``TorchVQCLayer`` is renamed to ``vqnet_native``, which more accurately reflects its meaning as the "VQNet native execution engine."
- Synchronized quantum logic gate class API signatures with the source: removed the ``has_params`` parameter and moved ``trainable`` to the first positional parameter.
- Synchronized the ``QTensor`` interface: removed the ``name`` and ``nodes`` parameters.
- Synchronized the ``QuantumLayerAdjoint`` interface: removed the ``use_qpanda`` parameter and renamed ``general_module`` to ``vqc_module``.
- Removed documentation sections for interfaces already deleted from the source, including ``QuantumLayer``, ``NoiseQuantumLayer`` (including the MNIST example section), and other pyqpanda2-compatible interfaces. Removed the entire pqanda2 interface documentation page ``qnn.rst`` and kept only pyqpanda3-compatible interfaces.
- Fixed existing documentation deviations: layers in the ``nn`` module, ``normalized_shape`` of ``LayerNorm``, optimizer ``lr``, required ``obs``/``wires`` for measurement classes, and the renamed ``qubits`` to ``qlist`` for ``pq3`` templates.
- Synchronized the ``TNQModule`` interface: added ``mesh``, ``cotengra_options``, and ``tree_path`` parameters, and corrected the default value of ``use_jit``.
- Synchronized the ``Module`` / ``state_dict`` interface signatures: ``Module(name="", *args, **kwargs)``, and added the ``keep_vars`` parameter to ``state_dict``.
- The ``padding`` of the ``ConvT2D`` interface no longer supports ``"same"``; only ``"valid"`` or an integer tuple is supported. The related example now passes ``(1, 1)`` explicitly.
- Updated all references to ``pyQPanda`` in the documentation to ``pyQPanda3`` and removed examples that depend on the old pyqpanda.
- Rewrote the home page overview: describes the ``pyvqnet`` (native) and ``torch`` compute backends according to the implemented 2.18.2 interfaces, changes distributed descriptions to gloo/NCCL, and adds descriptions of the tensor-network backend ``pyvqnet.qnn.vqc.tn`` and large-model operators.
- Distributed documentation: removed the ``vqnetrun --backend`` parameter description and command examples (the parameter has been removed; each worker process determines the backend at runtime), and corrected the return values of ``split_groups`` / ``init_groups`` and the ``group`` parameter description for group communication.
- Cleaned up the ``vqnetrun`` CLI documentation according to the current launcher implementation: removed unsupported ``-H`` / ``--hosts``, host file, ``--output-filename``, ``--start-timeout``, and ``--disable-cache`` parameters; clarified that ``--nproc_per_node`` is required for both gloo and NCCL; corrected ``--verbose`` to output additional startup diagnostics rather than connectivity checks; and updated the ``-cb`` / ``--check-build`` examples with the required ``--nproc_per_node`` parameter.

Fixed
===================
- Fixed incorrect coefficients in the CRX/CRY/CRZ parameter-shift rules in ``QuantumLayerV3``.
- Fixed the gradient error in the default mode of ``CRot``, enabled adjoint gradients for ``Rot``, ``CR``, and ``CRot``, and implemented the ``generator()`` method for ``CRX``, ``CRY``, and ``CRZ``.
- Fixed the fan computation error in the ``he_uniform_for_linear`` initializer.
- Fixed parameter divergence in ``SPSA._step()`` caused by an unrefreshed delta and an incorrect gradient formula.
- Fixed an issue where overlapping ``setitem``/``copy`` detection was misjudged in tensor alias scenarios, causing copies to be silently skipped.
- Fixed the ``set-select`` gradient regression in ``cpu_basic_indexing``.
- Fixed out-of-bounds access in CUDA ``expval`` when the number of ``float64`` elements is less than 4, and zero gradients in fused-crz when ``M>32``.
- Fixed result flattening, ``batch=1`` backward, and circuit submission issues in the real-chip path of ``VQCQCloudLayer``.
- Fixed the missing configurable submission parameters in the ``qpanda3_runtime`` path.
- Fixed misuse of print-style comma arguments in ``logger.info()``.
- Fixed the incorrect (communicator, rank list) structure in the ``allgather_group`` example; the example now uses the return value of ``split_groups``.


[v2.18.1] - 2026-07-08
***************************

Added
===================
- Added documentation for the tensor-network variational quantum circuit module ``pyvqnet.qnn.vqc.tn``, which provides two implementations: the ``torch`` backend (``pyvqnet.qnn.vqc.tn.torch``) and the ``pyvqnet`` backend (``pyvqnet.qnn.vqc.tn.native``), including ``TNQModule``, ``TNQMachine``, quantum logic gates, ``MeasureAll`` measurement, and templates. This module simulates quantum circuits with tensor networks, supports matrix product state (MPS) representation, and simulates circuits with larger qubit counts at lower memory cost. The ``torch`` backend can be combined with classical ``torch`` models and participate in automatic differentiation, making it suitable for large-scale quantum machine learning tasks.
- Added documentation for the large-model fine-tuning loss function library ``pyvqnet.torch.trl``: ``sft_loss`` is a cross-entropy loss for instruction fine-tuning that supports ``ignore_index`` masking to skip prompt/padding positions; ``dpo_loss`` is a direct preference optimization loss that builds the loss from the probability ratio of the policy model and reference model on the same preference pair, where ``beta`` controls the strength of the constraint on the reference model; ``ppo_loss`` is a proximal policy optimization loss that stabilizes training with GAE advantage estimation and a clipped surrogate objective; ``grpo_loss`` is a group relative policy optimization loss that normalizes rewards within a group of multiple generations for the same prompt to compute advantages, without an additional value network; ``reward_loss`` is a preference reward loss based on the Bradley-Terry model.
- Added ``nn`` module documentation for large-model inference and training:

  - ``RoPE`` rotary position encoding class and the ``functional.rope`` function, supporting four frequency scaling modes: ``standard``, ``ntk``, ``dynamic_ntk``, and ``yarn``, for long-text extrapolation.
  - ``RMSNorm`` root mean square normalization and ``SwiGLU`` gated activation, common components of current Transformer architectures.
  - ``fused_moe`` fused MoE computation interface (CUDA accelerated), which computes each expert MLP by routing index groups and merges outputs weighted by routing probabilities.
  - ``scaled_softmax``, ``scaled_masked_softmax``, and ``scaled_upper_triang_masked_softmax`` scaled Softmax interfaces, which fuse the scaling factor and attention mask into a single normalization computation.
  - ``_sampling`` decoding sampling interfaces (``top_k_sampling_from_probs``, ``top_p_sampling_from_probs``, ``min_p_sampling_from_probs``, ``top_k_top_p_sampling_from_probs``, ``top_k_top_p_sampling_from_logits``), supporting sampling from probability distributions or logits with top-k/top-p/min-p strategies, suitable for LLM inference decoding (CUDA accelerated, inference only).
- Added documentation for the ``pq3 torch`` quantum layers (``TorchQpandaQuantumLayer``, ``TorchQcloud3QuantumLayer``, ``TorchQpanda3QuantumLayer``), which connect to the qpanda simulator, the quantum cloud real chip, and the qpanda3 simulator respectively, and are embedded as ``torch`` layers in hybrid quantum-classical models for training.
- Added support for ``Python 3.13`` and ``Python 3.14``.
- Added high-performance CUDA quantum gate operator implementations based on the torch extension under the ``torch`` backend (RX, RY, RZ, CNOT, measurement, etc.), providing fused kernel acceleration for batched state vectors and significantly reducing training and simulation time for large-scale parameterized circuits.

Changed
===================
- Renamed ``DistributeQMachine`` to ``DistributedQMachine`` and updated the documentation.
- Unified ``Examples::`` markers in the RST documentation to ``Example::``.
- Changed the QTensor reduce API parameters `axis` to `dim` and `keepdims` to `keepdim`.
- Corrected the example import paths in ``vqc.rst`` and ``vqc_demo.rst`` to public modules.
- Merged the ``torch-native`` and ``torch`` backends.
- Removed ``pyvqnet.qnn.vqc.VQC_QuantumPoolingCircuit``, ``pyvqnet.qnn.vqc.sv.torch.vqc_quantumpooling_circuit``, and ``pyvqnet.qnn.pq3.template.QuantumPoolingCircuit`` interfaces and their documentation.

Fixed
===================
- Fixed examples in the tensor-network backend documentation, added ``CCZ`` gate documentation, and removed the unsupported ``vqc_amplitude_embedding``.
- Fixed several Sphinx build warnings (heading underline length, inconsistent heading levels, duplicate object descriptions).

[v2.18.0] - 2026-04-22
***************************

Added
===================
- ``vqnetrun`` adds ``--backend nccl`` mode support, controlling NCCL distributed launch through ``--nproc_per_node``, ``--nnodes``, ``--node_rank``, ``--master_addr``, ``--master_port``, and ``--nccl_socket_ifname``.
- Added the ``VQCQCloudLayer`` interface for submitting a VQC Module to a QCloud real chip or the pyqpanda3 local simulator, with parameter_shift backpropagation support.
- Added the ``destroy()`` method to ``CommController`` for cleaning up NCCL communication resources.
-

Changed
===================
- Changed the default backend to ``pyvqnet-ad``.
- Removed the deprecated ``QuantumLayerMultiProcess``, ``DataParallelHybirdVQCQpandaQVMLayer``, and ``HybirdVQCQpanda3QVMLayer`` interfaces.
- Renamed ``split_group`` to ``split_groups``.
- Depends on the nvidia runtime for cuda12.6.
- Changed the default "chip_id" to "WK_C180".
- Changed ``ComplexEntangelingTemplate`` to ``ComplexEntanglingTemplate``.
- Added a "Test 2: 10-qubit VQC gradient comparison" benchmark section to ``vqc.rst``, comparing gradient computation performance across VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum.
- Updated the benchmark hardware specification table with CUDA 12.6, torchquantum 0.2.0, and mindquantum 0.12.0 version information.


Fixed
===================
- Fixed integer overflow in the ``roll`` CUDA kernel.
- Fixed ``cuda_masked_fill`` support for complex64/complex128.
- Fixed incorrect +inf values in ``log_softmax`` forward computation under ``bfloat16``.
- Fixed device errors caused by missing ``CUDAGuard`` during cross-GPU memory access.
- Fixed many typos.

[v2.17.3] - 2026-03-30
***************************

Added
===================
- Added the bfloat16 data type.
- Added asynchronous NCCL communication interfaces: ``nccl_async_all_gather``, ``nccl_async_all_reduce``, ``nccl_async_reduce``, ``nccl_async_broadcast``, ``nccl_async_send``, ``nccl_async_recv``.
- Added support for the latest Origin Quantum chip, with chip ID ``WK_C180``.
- Added interfaces such as ``data_ptr`` and experimental support for `Triton <https://triton-lang.org/main/index.html>`_.
-

Changed
===================

- Changed the default backend to "pyvqnet-ad".
- Computation on MacOS is implemented with ARM NEON instructions.
- The ``matmul`` interface supports operations on data with more than 4 dimensions.
- Reduced dependencies on some CUDA runtime libraries when installing the whl.
- The QTensor data type in pyvqnet is no longer an integer but a specific data type.
- Changed the QTensor pickle logic to no longer pickle grad.
- Removed is_dense.
- Removed pq2's ``QuantumBatchAsyncQcloudLayer``.
- Updated the pq3 ``QuantumBatchAsyncQcloudLayer`` documentation.

Fixed
===================
- Fixed the ``Linear`` error when ``use_bias=False``.
- Fixed the MAX_GPUS issue; the current maximum number of GPUs is 16.
- Fixed an import error on Windows Jupyter.


[v2.17.2] - 2025-11-18
***************************

Added
===================

- Added support for the ``torch`` backend to the quantum natural gradient QNG interface.
- Added the ``pyvqnet-ad`` backend, which uses a torch-like C++ automatic differentiation backend while keeping the original _core.Tensor data structure, and supports most existing interfaces.
- Added the "Batched Data Quantum Gradient Benchmark" documentation.

Changed
===================

- Removed `HybirdVQCQpanda3QVMLayer`, `QuantumLayerMultiProcess`, `TorchHybirdVQCQpanda3QVMLayer`.
- Removed `is_csr`, `csr_members`, `SparseHamiltonian`, `csr_to_dense`, `dense_to_csr`.
- Added `QiskitLayer` and `CirqLayer` interfaces.
- Added `if_print_qcloud_log` support to the `QuantumBatchAsyncQcloudLayer` layer to print qcloud logs.
- Changed the installation command to ``pip install pyvqnet --upgrade``.
- Changed supported Python versions to `Python 3.10`, `Python 3.11`, and `Python 3.12`.
- Modified the command for installing a specified mpicxx.

Fixed
===================
- Supported the return value of the latest pyqpanda2 QCloud.
- Added device checks for input data of the dot product interface.
- Fixed a bug in `TorchModule`.


[v2.17.1] - 2025-8-22
***************************

Added
===================

- Added the quantum natural gradient SPSA algorithm qnspsa interface, quantum circuit born machine, quantum natural gradient interface with momentum, and a pure quantum qgru example.
- Added the ``torch_native`` backend.
- Added bit-parallel interfaces to support qubit parallelism in quantum circuits, and added bit reordering to reduce the number of bit swaps.
- Added the `split_group` method.


Changed
===================
- The Linear layer implementation changed from :math:`y = Ax + b` to :math:`y = x@A.T + b`
- Modified the `obs` parameter of the MeasureAll interface.
- Removed the `QuantumLayerES` interface.
- Renamed ControllComm to ControlComm and changed parameter names of interfaces such as `allgather_group`, `allreduce_group`, `reduce_group`, and `broadcast_group`.
- Removed the `ncclsplitGroup` interface.

Fixed
===================
- Fixed synchronization latency in distributed communication interfaces.
- Modified distributed communication interface definitions.
- Fixed adjoint gradient computation not supporting `PauliX`, `PauliY`, and `PauliZ`.


[v2.17.0] - 2025-4-22
***************************

Added
===================

- Added a tensor-network backend implementation for quantum circuit module functionality, with support for basic logic gates, measurement, and complex quantum circuits.
- Added tensor-network backend support for constructing large-qubit quantum circuits.
- Added the QTensor.swapaxes interface as an alias of swapaxis.

Changed
===================
- Use openblas for matrix operations.
- Use sleef for CPU SIMD operations.
- Removed qnn.MeasurePauliSum.
- Throw a warning when using the torch backend with a torch version lower than 2.4.0.


Fixed
===================
- Fixed the issue where QMachine states could not be loaded when saving and loading models.
- Fixed issues with layernorm and groupnorm when affine=False.
- Fixed the issue with QuantumLayerAdjoint in eval mode.
- Fixed a bug in `toCPU()`.
- Fixed bugs in BatchNorm1d, BatchNorm2d, and BatchNormNd when affine=False.

[v2.16.0] - 2025-1-15
***************************

Added
===================

- Added interfaces for quantum circuit computation using pyqpanda3.
- The MeasureAll interface supports composite Pauli operators.
- Added DataParallelVQCAdjointLayer and DataParallelVQCLayer interfaces.

Changed
===================

- Removed obsolete ONNX functionality, removed most interfaces integrating pyqpanda, and kept some interfaces used in example code.
- Modified the VQC_QuantumEmbedding interface.
- Installing this package no longer installs pyqpanda at the same time; it installs pyqpanda3 instead.
- VQC interfaces support input in the form `x[:,:2]`; previously only `x[:,[2]]` was supported.
- This software supports Python 3.9, 3.10, and 3.11 and no longer supports Python 3.8.

Fixed
===================
- Fixed discovered memory leaks.
- Fixed discovered GPU random number issues.
- For reduce-related operations, the maximum supported array dimension increased from 8 to 30.
- Optimized the code and improved Python code execution speed in some cases.

[v2.15.0] - 2024-11-19
***************************

Added
===================

- Added the `pyvqnet.backends.set_backend()` interface. When `torch` is installed, users can use `torch` for QTensor matrix computation and variational quantum circuit computation; see :ref:`torch_api`.
- Added neural network interfaces under `pyvqnet.nn.torch` that inherit from `torch.nn.Module`, variational quantum circuit neural interfaces, and more; see :ref:`torch_api`.
- Added an introduction to fine-tuning training for the quantum large-model fine-tuning module; see :ref:`llm` for detailed steps.

Changed
===================
- Modified the diag interface.
- Modified the all_gather implementation to be consistent with torch.distributed.all_gather.
- Modified `QTensor` to support up to 30-dimensional data.
- Modified the `mpi4py` required by distributed functions to version 4.0.1 or later.

Fixed
===================
- Some random number implementations could not fix the seed due to omp.
- Fixed some bugs in distributed startup.


[v2.14.0] - 2024-09-30
***************************

Added
===================

- Added the block-encoding algorithms `VQC_LCU`, `VQC_FABLE`, and `VQC_QSVT`, and the qpanda algorithm implementations `QPANDA_QSVT`, `QPANDA_LCU`, and `QPANDA_FABLE`.
- Added integer addition to qubits `vqc_qft_add_to_register`, addition of numbers in two qubit registers `vqc_qft_add_two_register`, and multiplication of numbers in two qubit registers `vqc_qft_mul`.
- Added the hybrid qpanda and vqc training module `HybirdVQCQpandaQVMLayer`.
- Added implementations of interfaces such as `einsum`, `moveaxis`, `eigh`, and `diagonal`.
- Added tensor parallel computation in distributed computing: `ColumnParallelLinear`, `RowParallelLinear`.
- Added Zero stage-1 functionality in distributed computing: `ZeroModelInitial`.
- When `QuantumBatchAsyncQcloudLayer` specifies diff_method == "random_coordinate_descent", it does not use PSR but randomly selects one quantum parameter for gradient computation.

Changed
===================
- Removed the xtensor part.
- Partially modified the API documentation. Separated quantum machine learning examples based on automatic differentiation from machine learning examples based on qpanda, and separated quantum machine learning interfaces based on automatic differentiation from machine learning example interfaces based on qpanda.
- `matmul` supports 1d@1d, 2d@1d, 1d@2d.
- Added some quantum computing layer aliases: `QpandaQCircuitVQCLayer`` = `QuantumLayer`, `QpandaQCircuitVQCLayerLite` = `QuantumLayerV2`, `QpandaQProgVQCLayer` = `QuantumLayerV3`.

Fixed
===================
- Modified the underlying communication interfaces `allreduce`, `allgather`, `reduce`, and `broadcast` in distributed computing, adding support for `core.Tensor` data communication.
- Fixed a bug in random number generation.
- Fixed the error converting VQC `RXX`, `RYY`, `RZZ`, and `RZX` to originIR.


[v2.13.0] - 2024-07-30
***************************

Added
===================

- Added `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA`, and `quantum_fisher` interfaces.
- Added quantum Dropout examples to address overfitting in quantum machine learning.

Changed
===================

- Added an affine interface to `BatchNorm`, `LayerNorm`, and `GroupNorm`.
- The `diag` interface now returns a 1d output of diagonal elements for 2d input, consistent with torch.
- Operations such as slice and permute try to return a QTensor sharing memory via view.
- All interfaces support non-contiguous input.
- `Adam` supports the weight_decay parameter.

Fixed
===================
- Fixed errors in some VQC logic gate decomposition functions.
- Fixed memory leaks in some functions.
- Fixed `QuantumLayerMultiProcess` not supporting GPU input.
- Modified the default parameter initialization of `Linear`.


[v2.12.0] - 2024-05-01
***************************

Added
===================

- Added the PipelineParallelTrainingWrapper interface for pipeline parallelism.
- Added `Gelu`, `DropPath`, `binomial`, and `adamW` interfaces.
- Added `QuantumBatchAsyncQcloudLayer` support for pyqpanda local virtual machine simulation.
- Added xtensor `QuantumBatchAsyncQcloudLayer` support for pyqpanda local virtual machine simulation and real-chip computation.
- Made QTensor support deepcopy and pickle.
- Added the distributed computing launch command `vqnetrun`, used when using distributed computing interfaces.
- Added the ES gradient computation method real-chip interface `QuantumBatchAsyncQcloudLayerES`, supporting pyqpanda local virtual machine simulation and real-chip computation.
- Added QTensor data communication interfaces in distributed computing: `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv`, etc.

Changed
===================

- New dependencies "Pillow", "hjson", and "cloudpickle" added to the installation package; new dependency "psutil" added on Linux.
- Optimized softmax and transpose execution speed on GPU.
- Compiled with cuda11.8.
- Integrated distributed computing interfaces for CPU and GPU.

Fixed
===================
- Reduced memory consumption at startup for the Linux-GPU version.
- Fixed memory leaks in the select and power functions.
- Removed the reduce-based model parameter and gradient update methods `nccl_average_parameters_reduce` and `nccl_average_grad_reduce` on CPU and GPU.

[v2.11.0] - 2024-03-01
***************************

Added
===================

- Added a new `QNG` (quantum natural gradient) API and demo.
- Added quantum circuit optimization APIs and demos, such as `wrapper_single_qubit_op_fuse`, `wrapper_commute_controlled`, and `wrapper_merge_rotations`.
- Added `CY`, `SparseHamiltonian`, `HermitianExpval`.
- Added `is_csr`, `is_dense`, `dense_to_csr`, `csr_to_dense`.
- Added `QuantumBatchAsyncQcloudLayer` support for pyqpanda QCloud real-chip computation, `expval_qcloud`.
- Added NCCL-based single-node multi-GPU distributed data parallel model training interfaces `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce`, and the class `NCCL_api` for controlling NCCL initialization and related operations.
- Added the quantum circuit evolutionary strategy gradient computation method `QuantumLayerES` interface.

Changed
===================

- Refactored the `VQC_CSWAP` circuit into `CSWAP`.
- Removed the old QNG documentation.
- Removed the unused `num_wires` parameter from functions and classes in `pyvqnet.qnn.vqc`.
- Refactored the `MeasureAll` and `Probability` APIs.
- Added the qtype parameter to `QuantumMeasure`.

Fixed
===================
- Changed the slots of `QuantumMeasure` to shots.

[v2.10.0] - 2023-12-30
***************************

Added
===========
- Added new interfaces under pyvqnet.qnn.vqc: IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MultiRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz, etc.
- Added the pyvqnet.qnn.vqc.QuantumLayerAdjoint interface supporting adjoint gradient computation.
- Added conversion between originIR and VQC.
- Added statistics of classical and quantum module information in VQC models.
- Added two cases of hybrid quantum-classical neural network models: a quantum convolutional neural network model based on few-shot learning, and a quantum kernel function model for handwritten digit recognition.
- Added support for Mac with ARM chips.


[v2.9.0] - 2023-11-15
***************************

Added
===========
- Added xtensor interface definitions, supporting automatic parallelism and CPU/GPU multi-backend for classical neural network modules, including more than 150 interfaces for common mathematical, logical, and matrix operations on multidimensional arrays, as well as common classical neural network layers and optimizers.

Changed
===========
- Starting with this version, the version number is upgraded from 2.0.8 to 2.9.0.
- Starting with this version, the package is uploaded to https://pypi.originqc.com.cn and installed with ``pip install pyvqnet --index-url https://pypi.originqc.com.cn``.

[v2.0.8] - 2023-09-26
***************************

Added
===========
- Added support for complex128, complex64, double, float, uint8, int8, bool, int16, int32, int64, and other types in existing interfaces.
- The Linux version supports GPU computation and requires cuda11.7 cudatoolkit and the NVIDIA driver.
- Basic logic gates based on vqc: Hadamard, CNOT, I, RX, RY, PauliZ, PauliX, PauliY, S, RZ, RXX, RYY, RZZ, RZX, X1, Y1, Z1, U1, U2, U3, T, SWAP, P, TOFFOLI, CZ, CR.
- Composite quantum circuits based on vqc: VQC_HardwareEfficientAnsatz, VQC_BasicEntanglerTemplate, VQC_StronglyEntanglingTemplate, VQC_QuantumEmbedding, VQC_RotCircuit, VQC_CRotCircuit, VQC_CSWAPcircuit, VQC_Controlled_Hadamard, VQC_CCZ, VQC_FermionicSingleExcitation, VQC_FermionicDoubleExcitation, VQC_UCCSD, VQC_QuantumPoolingCircuit, VQC_BasisEmbedding, VQC_AngleEmbedding, VQC_AmplitudeEmbedding, VQC_IQPEmbedding.
- Measurement methods based on vqc: VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll.


[v2.0.7] - 2023-07-03
***************************

Added
===========
- Classical neural networks: added kron, gather, scatter, and broadcast_to interfaces.
- Added support for different data precisions: data type dtype supports kbool, kuint8, kint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128, representing C++ bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double, complex<float>, complex<double> respectively.
- Supports Python 3.8, 3.9, and 3.10.

Changed
===========
- Added a `dtype` parameter to the init functions of the QTensor and Module classes. Type restrictions were added for QTensor indexing and the input of some neural network layers.
- Quantum neural networks: due to MacOS compatibility issues, removed the Mnist_Dataset and CIFAR10_Dataset interfaces.

[v2.0.6] - 2023-02-22
***************************


Added
===========

- Classical neural networks: added interfaces multinomial, pixel_shuffle, pixel_unshuffle; added numel to QTensor; added a CPU dynamic memory pool; added the init_from_tensor interface to Parameter.
- Classical neural networks: added interfaces Dynamic_LSTM, Dynamic_RNN, Dynamic_GRU.
- Classical neural networks: added interfaces pad_sequence, pad_packed_sequence, pack_pad_sequence.
- Quantum neural networks: added interfaces: CCZ, Controlled_Hadamard, FermionicSingleExcitation, UCCSD, QuantumPoolingCircuit,
- Quantum neural networks: added interfaces: Quantum_Embedding, Mnist_Dataset, CIFAR10_Dataset, grad, Purity.
- Quantum neural networks: added examples: gradient clipping, quanvolution, quantum circuit expressibility, barren plateau, quantum reinforcement learning QDRL.

Changed
===========

- API documentation: restructured the content, added a `Quantum Machine Learning Research` module, and changed `VQNet2ONNX Module` to `Other Functions`.



Fixed
===========

- Classical neural networks: fixed the issue where the same random seed produced different normal distributions across platforms.
- Quantum neural networks: implemented expval, ProbMeasure, and QuantumMeasure support for the QPanda GPU virtual machine.


[v2.0.5] - 2022-12-25
***************************


Added
===========

- Classical neural networks: added the log_softmax implementation and the export_model function for exporting models to ONNX.
- Classical neural networks: support converting most existing classical neural network modules to ONNX; see the API documentation "VQNet2ONNX Module" for details.
- Quantum neural networks: added VarMeasure, MeasurePauliSum, Quantum_Embedding, SPSA, and other interfaces.
- Quantum neural networks: added LinearGNN, ConvGNN, ConvGNN, QMLP, quantum natural gradient, quantum random parameter-shift algorithm, DoublySGD algorithm, etc.


Changed
===========

- Classical neural networks: added dimension checks to the BN1d and BN2d interfaces.

Fixed
===========

- Fixed a bug in maxpooling parameter checking.
- Fixed the [::-1] slicing bug.


[v2.0.4] - 2022-09-20
***************************


Added
===========

- Classical neural networks: added the LayernormNd implementation, supporting layernorm computation on multidimensional data.
- Classical neural networks: added CrossEntropyLoss and NLL_Loss loss computation interfaces, supporting 1D to N-dimensional input.
- Quantum neural networks: added common circuit templates: HardwareEfficientAnsatz, StronglyEntanglingTemplate, BasicEntanglerTemplate.
- Quantum neural networks: added the Mutal_info interface for computing mutual information of qubit subsystems, Von Neumann entropy VB_Entropy, and density matrix DensityMatrixFromQstate.
- Quantum neural networks: added the quantum perceptron algorithm example QuantumNeuron and a quantum Fourier series algorithm example.
- Quantum neural networks: added the QuantumLayerMultiProcess interface supporting multi-process acceleration of quantum circuits.

Changed
===========

- Classical neural networks: support group convolution parameter group, dilated convolution dilation_rate, and arbitrary numerical padding as parameters of 1D convolution Conv1d, 2D convolution Conv2d, and deconvolution ConvT2d.
- Skip broadcast operations for data of the same dimension to reduce unnecessary runtime logic.

Fixed
===========

- Fixed incorrect computation of the stack function under some parameters.


[v2.0.3] - 2022-07-15
***************************


Added
===========

- Added support for stack and bidirectional recurrent neural network interfaces: RNN, LSTM, GRU
- Added interfaces for common computation performance metrics: MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_score, precision_recall_f1_N_score, auc_calculate
- Added a quantum kernel SVM algorithm example

Changed
===========

- Sped up QTensor printing when there is a large amount of data
- Use openmp to accelerate computation on Windows and Linux.

Fixed
===========

- Fixed the issue where some Python import methods could not import the package
- Fixed repeated computation in the batch normalization BN layer
- Fixed the bug where QTensor.reshape and transpose interfaces could not compute gradients
- Added input shape checks to the tensor.power interface

[v2.0.2] - 2022-05-15
***************************

Added
===========

- Added topK, argtoK
- Added cumsum
- Added masked_fill
- Added triu, tril
- Added a QGAN example for generating random distributions

Changed
===========

- Support advanced slicing indexing and normal slicing indexing
- matmul supports 3D and 4D tensor operations
- Modified the HardSigmoid function implementation

Fixed
===========

- Fixed the issue where convolution, batch normalization, deconvolution, pooling, and other layers did not cache internal variables, causing incorrect gradient computation when backpropagating multiple times after one forward pass
- Fixed the implementation and examples of the QLinear layer
- Fixed the "Image not load" issue when importing VQNet in a conda environment on MAC.




[v2.0.1] - 2022-03-30
***************************


Added
===========

- Added more than 100 basic data structure QTensor interfaces, including creation functions, logical functions, mathematical functions, and matrix operations.
- Added 14 basic neural network functions, including convolution, deconvolution, and pooling.
- Added 4 loss functions, including MSE, BCE, CCE, and SCE.
- Added 10 activation functions, including ReLu, Sigmoid, and ELU.
- Added 6 optimizers, including SGD, RMSPROP, and ADAM.
- Added machine learning examples: QVC, QDRL, Q-KMEANS, QUnet, HQCNN, VSQL, quantum autoencoder.
- Added quantum machine learning layers: QuantumLayer, NoiseQuantumLayer.
