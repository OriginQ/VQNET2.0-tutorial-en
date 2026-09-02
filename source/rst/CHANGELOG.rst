VQNet Changelog
###############################


[v2.18.1] - 2026-07-08
***************************

Added
===================
- Added documentation for the tensor network variational quantum circuit module ``pyvqnet.qnn.vqc.tn``, providing two implementations: the ``torch`` backend (``pyvqnet.qnn.vqc.tn.torch``) and the ``pyvqnet`` backend (``pyvqnet.qnn.vqc.tn.native``), including ``TNQModule``, ``TNQMachine``, quantum logic gates, ``MeasureAll`` measurements, and templates. This module simulates quantum circuits via tensor networks and supports matrix product state (MPS) representation, enabling simulation of larger-scale circuits with lower memory overhead; the ``torch`` backend can be mixed with classical ``torch`` models and participates in automatic differentiation, suitable for large-scale quantum machine learning tasks.
- Added documentation for the ``pyvqnet.torch.trl`` large model fine-tuning loss library: ``sft_loss``, cross-entropy loss for instruction fine-tuning with ``ignore_index`` masking to skip prompt/padding tokens; ``dpo_loss``, direct preference optimization loss constructed from the probability ratio between the policy model and reference model on the same preference pair, with ``beta`` controlling the constraint strength toward the reference model; ``ppo_loss``, proximal policy optimization loss that stabilizes training via GAE advantage estimation and clipped surrogate objectives; ``grpo_loss``, group relative policy optimization loss that normalizes rewards within each group of generations for the same prompt to compute advantages, requiring no additional value network; ``reward_loss``, preference reward loss based on the Bradley-Terry model.
- Added documentation for ``nn`` module interfaces for LLM inference and training:

  - ``RoPE`` rotary position embedding class and ``functional.rope`` function, supporting four frequency scaling modes (``standard``, ``ntk``, ``dynamic_ntk``, ``yarn``) for long-context extrapolation scenarios.
  - ``RMSNorm`` root mean square normalization and ``SwiGLU`` gated activation, common components of current Transformer architectures.
  - ``fused_moe``, fused MoE computation interface (CUDA-accelerated) that computes each expert's MLP in groups by routing indices and merges outputs weighted by routing probabilities.
  - ``scaled_softmax``, ``scaled_masked_softmax``, ``scaled_upper_triang_masked_softmax``, a scaled Softmax family that fuses scaling factors and attention masks into a single normalization computation.
  - ``_sampling`` decoding sampling interfaces (``top_k_sampling_from_probs``, ``top_p_sampling_from_probs``, ``min_p_sampling_from_probs``, ``top_k_top_p_sampling_from_probs``, ``top_k_top_p_sampling_from_logits``), supporting top-k/top-p/min-p sampling strategies from probability distributions or logits, suitable for LLM inference decoding (CUDA-accelerated, inference only).
- Added documentation for ``pq3 torch`` quantum layers (``TorchQpandaQuantumLayer``, ``TorchQcloud3QuantumLayer``, ``TorchQpanda3QuantumLayer``), connecting to the qpanda simulator, QCloud real quantum chips, and the qpanda3 simulator respectively, and embeddable into hybrid quantum-classical models as ``torch`` layers for training.
- Added support for ``Python 3.13`` and ``Python 3.14``.
- Added high-performance CUDA quantum gate operator implementations (RX, RY, RZ, CNOT, measurement, etc.) based on torch extensions under the ``torch`` backend, providing fused kernel acceleration for batched state vectors and significantly reducing training and simulation time for large-scale parameterized circuits.

Changed
===================
- ``DistributeQMachine`` renamed to ``DistributedQMachine``, documentation updated accordingly.
- Unified RST documentation marker ``Examples::`` to ``Example::``.
- Renamed QTensor reduce API parameter `axis` to `dim`, and `keepdims` to `keepdim`.
- Fixed example import paths in ``vqc.rst`` and ``vqc_demo.rst`` to public modules.
- Merged ``torch-native`` and ``torch`` backends.
- Removed ``pyvqnet.qnn.vqc.VQC_QuantumPoolingCircuit`` , ``pyvqnet.qnn.vqc.sv.torch.vqc_quantumpooling_circuit`` , and ``pyvqnet.qnn.pq3.template.QuantumPoolingCircuit`` interfaces and their documentation.

Fixed
===================
- Fixed tensor network backend documentation examples, added ``CCZ`` gate documentation, removed unsupported ``vqc_amplitude_embedding``.
- Fixed multiple Sphinx build warnings (heading underline length, inconsistent heading levels, duplicate object descriptions).


[v2.18.0] - 2026-04-22
***************************

Added
===================
- ``vqnetrun`` now supports ``--backend nccl`` mode, with distributed startup controllable via ``--nproc_per_node``, ``--nnodes``, ``--node_rank``, ``--master_addr``, ``--master_port``, ``--nccl_socket_ifname`` parameters.
- Added ``VQCQCloudLayer`` interface for submitting VQC Module to QCloud real chips or pyqpanda3 local simulators, supporting parameter_shift backpropagation.
- ``CommController`` adds ``destroy()`` method for NCCL communication resource cleanup.

Changed
===================
- Default backend changed to ``pyvqnet-ad``.
- Removed deprecated ``QuantumLayerMultiProcess``, ``DataParallelHybirdVQCQpandaQVMLayer``, ``HybirdVQCQpanda3QVMLayer`` interfaces.
- ``split_group`` renamed to ``split_groups``.
- Depends on NVIDIA runtime for CUDA 12.6.
- "chip_id" default changed to "WK_C180".
- ``ComplexEntangelingTemplate`` renamed to ``ComplexEntanglingTemplate``.
- ``vqc.rst``: added "Test 2: 10-Qubit VQC Gradient Comparison" benchmark section comparing VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum.
- Updated benchmark spec table with CUDA 12.6, torchquantum 0.2.0, mindquantum 0.12.0.

Fixed
===================
- Fixed integer overflow issue in ``roll`` CUDA kernel.
- Fixed ``cuda_masked_fill`` support for complex64/complex128 types.
- Fixed ``log_softmax`` forward computation producing incorrect +inf values under ``bfloat16``.
- Fixed device errors caused by missing ``CUDAGuard`` during cross-GPU memory access.
- Fixed many typos.


[v2.17.3] - 2026-03-31
***************************

Added
===================
- Added bfloat16 data type.
- Added asynchronous NCCL communication interfaces: ``nccl_async_all_gather``, ``nccl_async_all_reduce``, ``nccl_async_reduce``, ``nccl_async_broadcast``, ``nccl_async_send``, ``nccl_async_recv``.
- Added support for the latest Origin Quantum chip with chip ID ``WK_C180``
- Added ``data_ptr`` and other interfaces, experimentally added support for `Triton <https://triton-lang.org/main/index.html>`_.

Changed
===================

- Default backend changed to "pyvqnet-ad".
- Computation on MacOS is implemented based on arm neon instructions.
- The ``matmul`` interface supports data with more than 4 dimensions.
- Reduced dependency on some cuda runtime libraries when installing the whl package.
- QTensor data type in pyvqnet is no longer an integer, but a specific data type.
- Modified QTensor pickle logic, no longer pickles grad.
- Removed is_dense.
- Removed pq2 ``QuantumBatchAsyncQcloudLayer``.
- Change doc for pq3 ``QuantumBatchAsyncQcloudLayer``.

Fixed
===================
- Fixed the error of ``Linear`` when ``use_bias=False``.
- Fixed the MAX_GPUS issue, current maximum number of GPUs is 16.
- Fixed import error on windows jupyter.


[v2.17.2] -2025-11-18
***********************************


Added
===================

- Add support for the `torch` backend through the quantum natural gradient QNG interface;
- Add the `pyvqnet-ad` backend, which uses a C++ automatic differential backend similar to torch. The data structure still adopts the original _core.Tensor and supports the vast majority of current existing interfaces.
- Add the "Benchmarking of Variational Quantum Circuit's Gradients for Batch Data" document;

Changed
===================

- delete `HybirdVQCQpanda3QVMLayer`, `QuantumLayerMultiProcess`, `TorchHybirdVQCQpanda3QVMLayer`;
- delete `is_csr`, `csr_members` , `SparseHamiltonian` , `csr_to_dense` , `dense_to_csr` ;
- Add the `QiskitLayer` and `CirqLayer` interfaces;
- for `QuantumBatchAsyncQcloudLayer` layer increased `if_print_qcloud_log` support print qcloud log;
- Change the installation command to `pip install pyvqnet --upgrade`
- The supported python versions are changed to `Python 3.10` , `Python 3.11` , and `Python 3.12` .
- Modify the specified mpicxx installation command;



Fixed
===================

- Support the return values of the latest version of pyqpanda2QCloud;
- Add input data device checks for the dot product interface;
- Fixed a bug in `TorchModule` ;



[v2.17.1] - 2025-8-22
***************************

Added
===================

- Added the quantum natural gradient SPSA algorithm (qnspsa) interface, quantum circuit born machine (QBM), the quantum natural gradient interface with momentum, and a pure quantum qgru example.
- Added the ``torch_native`` backend.
- Added a bit-parallel interface to support bit-parallel quantum circuits, and added a bit reordering function to reduce the number of bit swaps.
- Added the ``split_groups`` method.

Changed
==================
- Changed the Linear layer implementation from `:math:`y = Ax + b` to `:math:`y = x@A.T + b`
- Modified the ``obs`` parameter in the `MeasureAll` interface.
- Removed the ``QuantumLayerES`` interface. - Changed parameter names from ControllComm to ControlComm, `allgather_group`, `allreduce_group`, `reduce_group`, `broadcast_group`, and other interfaces.
- Removed the `ncclsplitGroup` interface.

Fixed
===================
- Resolved synchronization delay issues with distributed communication interfaces.
- Modified distributed communication interface definitions.
- Resolved an issue where adjoint gradient calculations did not support `PauliX`, `PauliY`, and `PauliZ`.


[v2.17.0] - 2025-4-22
***************************

Added
===================

- Added tensor network backend implementation for quantum circuit module, including support for basic logic gates, measurement, and complex quantum circuits.
- Added tensor network backend implementation for constructing large-bit quantum circuits.
- Added QTensor.swapaxes interface, another name is swapaxis.

Changed
===================
- Matrix operations using openblas.
- Use sleef for CPU SIMD operations.
- Remove qnn.MeasurePauliSum.
- Throw warning when using torch backend calculations when torch is below version 2.4.

Fixed
====================
- Solve the problem of QMachine states when saving model.
- Solve the problem of layernorm, groupnorm when affine=False.
- Solve the problem of QuantumLayerAdjoint in eval mode.

[v2.16.0] - 2025-1-15
***************************

Added
===================

- Added an interface for quantum circuit calculation using pyqpanda3.
- The MeasureAll interface supports compound Pauli operators.
- Added DataParallelVQCAdjointLayer and DataParallelVQCLayer interfaces.

Changed
===================

- Removed outdated ONNX functions, removed most of the interfaces that integrated pyqpanda, and retained some interfaces used in the sample code.
- VQC_QuantumEmbedding interface modification
- When installing this package, pyqpanda is no longer installed at the same time, but pyqpanda3 is installed at the same time.
- The VQC interface supports the use of `x[:,:2]` as input parameters, which originally only supported the `x[:,[2]]` format.
- This software supports Python 3.9, 3.10, 3.11, and no longer supports Python 3.8

Fixed
====================
- Solve the memory leak problem found;
- Solve the GPU random number problem found;
- For reduce related operations, the maximum dimension of the supported array is increased from 8 to 30;
- Optimize the code and improve the running speed of Python code in some cases;
  
  
[v2.15.0] - 2024-11-19
***************************

Added
===================

- Added `pyvqnet.backends.set_backend()` interface. When users install `torch`, `torch` can be used to perform matrix calculations and variational quantum circuit calculations of QTensor. For details, see the document :ref:`torch_api`.
- Added `pyvqnet.nn.torch` to inherit the neural network interface and variational quantum circuit neural interface of `torch.nn.Module`. For details, see the document :ref:`torch_api`.
- Add a section on fine-tuning large quantum models, with detailed steps available in the documentation :ref:`llm`.

Changed
===================
- Modified diag interface.
- Modified all_gather implementation to be consistent with torch.distributed.all_gather.
- Modify `QTensor` to support up to 30-dimensional data.
- Modify `mpi4py` required for distributed functions to require version 4.0.1 or above

Fixed
===================
- Some random number implementations cannot fix the seed because omp.
- Fix some bugs in distributed startup.


[v2.14.0] - 2024-09-30
***************************

Added
===================

- Added block-encoding algorithms of `VQC_LCU`, `VQC_FABLE`, `VQC_QSVT`, and qpanda algorithm implementation `QPANDA_QSVT`, `QPANDA_LCU`, `QPANDA_FABLE` interfaces.
- Added integer addition to quantum bits `vqc_qft_add_to_register`, addition of numbers on two quantum bits `vqc_qft_add_two_register`, and multiplication of numbers on two quantum bits `vqc_qft_mul`.
- Added hybrid qpanda and vqc training module `HybirdVQCQpandaQVMLayer`.
- Added `einsum`, `moveaxis`, `eigh`, `diagonal` and other interface implementations.
- Added tensor parallel computing functions in distributed computing `ColumnParallelLinear`, `RowParallelLinear`.
- Added Zero in distributed computing stage-1 function `ZeroModelInitial`.
- `QuantumBatchAsyncQcloudLayer` specifies diff_method == "random_coordinate_descent" and does not use PSR but randomly selects a quantum parameter for gradient calculation.

Changed
====================
- Deleted the xtensor part.
- The api document was partially modified. Distinguished between quantum machine learning examples based on automatic differentiation and machine learning examples based on qpanda, and distinguished between quantum machine learning interfaces based on automatic differentiation and machine learning example interfaces based on qpanda.
- `matmul` supports 1d@1d, 2d@1d, 1d@2d.
- Added some quantum computing layer aliases: `QpandaQCircuitVQCLayer`` = `QuantumLayer`, `QpandaQCircuitVQCLayerLite` = `QuantumLayerV2`, `QpandaQProgVQCLayer` = `QuantumLayerV3`.

Fixed
====================
- Modified the underlying communication interfaces `allreduce`, `allgather`, `reduce`, `broadcast` in the distributed computing function, and added support for `core.Tensor` data communication
- Solved the bug in random number generation.
- Solved the error in converting VQC's `RXX`, `RYY`, `RZZ`, `RZX` to originIR.


[v2.13.0] - 2024-07-30
***************************

Added
==================

- Added `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA` interfaces.
- Added Quantum Dropout method to avoid overfitting.

Changed
===================

- Added affine interface to `BatchNorm`, `LayerNorm`, `GroupNorm`.
- `diag` interface now returns 1d output on the diagonal for 2d input, consistent with torch.
- Operations such as slice and permute will try to use the view method to return a QTensor in shared memory.
- All interfaces support non-contiguous input.
- `Adam` supports the weight_decay parameter.

Fixed
====================
- Modify the error of some logic gate decomposition functions of VQC.
- Fix the memory leak problem of some functions.
- Fix the problem that `QuantumLayerMultiProcess` does not support GPU input.
- Modify the default parameter initialization method of `Linear`


[v2.12.0] - 2024-05-01
***************************

Added
===================

- Added PipelineParallelTrainingWrapper interface.
- Added `Gelu`, `DropPath`, `binomial`, `adamW` interfaces.
- Added `QuantumBatchAsyncQcloudLayer` to support pyqpanda's local virtual machine simulation calculation.
- Add xtensor's `QuantumBatchAsyncQcloudLayer` to support pyqpanda's local virtual machine simulation calculation and real machine calculation.
- Enables QTensor to be deepcopy and pickle.
- Add distributed computing startup command `vqnetrun`, used when using the distributed computing interface.
- Add ES gradient calculation method real machine interface `QuantumBatchAsyncQcloudLayerES` to support local VM simulation calculations as well as real machine calculations for pyqpanda.
- Add data communication interfaces `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv`, etc. that support QTensor in distributed computing.

Changed
===================

- Added new dependencies "Pillow" and "hjson" to the installation package, add new dependencies "psutil" and "cloudpickle" on linux systems .
- Optimize softmax and transpose running speed under GPU.
- Compiled using cuda11.8.
- Integration of distributed computing interfaces under cpu and gpu based.

Fixed
===================
- Reduce the memory consumption when starting the Linux-GPU version.
- Fixed the memory leak problem of select and power functions.
- Removed model parameters and gradient update methods `nccl_average_parameters_reduce`, `nccl_average_grad_reduce` based on the reduce method for cpu, gpu.

[v2.11.0] - 2024-03-01
***************************

Added
===================

- Added new `QNG` (Quantum Natural Gradient) API and demo.
- Added quantum circuit optimization, such as `wrapper_single_qubit_op_fuse`, `wrapper_commute_controlled`, `wrapper_merge_rotations` api and demo.
- Added `CY`, `SparseHamiltonian`, `HermitianExpval`.
- Added `is_csr`, `is_dense`, `dense_to_csr`, `csr_to_dense`.
- Added `QuantumBatchAsyncQcloudLayer` to support pyqpanda's QCloud real chip calculation, `expval_qcloud`.
- Add NCCL-based interface implementations for parallel model training of multi-GPU distributed computing data on a single node `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce`, and classes to control NCCL initialization and related operations `NCCL_api`. 
- Add quantum line evolution strategy gradient calculation interface `QuantumLayerES`.

Changed
===================

- Refactored `VQC_CSWAP` circuit into `CSWAP`.
- Delete old QNG documents.
- Removed useless `num_wires` parameter from `pyvqnet.qnn.vqc` for functions and classes.
- Refactor `MeasureAll`, `Probability` api.
- Add qtype parameter to `QuantumMeasure`.

Fixed
===================
- Changed `QuantumMeasure`'s slots to shots.

[v2.10.0] - 2023-12-30
***************************

Added
===========
- Added new interfaces under pyvqnet.qnn.vqc: IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MultiRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz, etc.;
- Added pyvqnet.qnn.vqc.QuantumLayerAdjoint interface that supports adjoint gradient calculation;
- Supported the mutual conversion function between originIR and VQC;
- Supported classical and quantum module information in statistical VQC models;
- Added two cases under the quantum classical neural network hybrid model: quantum convolutional neural network model based on small samples, and quantum kernel function model for handwritten digit recognition.


[v2.9.0] - 2023-09-08
***************************

Added
===================
- The xtensor interface definition has been added to support automatic operator parallelism and multiple CPU/GPU backends. It includes more than 150 interfaces for commonly used mathematics, logic, and matrix calculations for multi-dimensional arrays, as well as common classic neural network layers and optimizers.

Changed
===================
- version from v2.0.8 bumps to v2.9.0.
- packages are uploaded in https://pypi.originqc.com.cn, use ``pip install pyvqnet --index-url https://pypi.originqc.com.cn`` .


[v2.0.8] - 2023-07-26
***************************

Added
===================
- Added existing interfaces to support complex128, complex64, double, float, uint8, int8, bool, int16, int32, int64 and other types of computing (gpu).
- Basic logic gates based on vqc: Hadamard, CNOT, I, RX, RY, PauliZ, PauliX, PauliY, S, RZ, RXX, RYY, RZZ, RZX, X1, Y1, Z1, U1, U2, U3, T, SWAP , P, TOFFOLI, CZ, CR, ISWAP.
- Combined quantum circuit based on vqc: VQC_HardwareEfficientAnsatz,VQC_BasicEntanglerTemplate,VQC_StronglyEntanglingTemplate,VQC_QuantumEmbedding,VQC_RotCircuit,VQC_CRotCircuit,VQC_CSWAPcircuit,VQC_Controlled_Hadamard,VQC_CCZ,VQC_FermionicSingleExcitation,VQC_FermionicDoubleExcitation,VQC_UCCSD,VQC_QuantumPoolingCircuit,VQC_BasisEmbedding,VQC_AngleEmbedding,VQC_AmplitudeEmbedding,VQC_IQPEmbedding.
- Measurement methods based on vqc: VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll.


[v2.0.7] - 2023-07-03
***************************

Added
===================
- For classic neural network, add kron, gather, scatter, broadcast_to interfaces.
- Added support for different data precision: data type dtype supports kbool, kuint8, kint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128, which respectively represent bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double, complex<float>, complex<double>.
- Support python 3.8, 3.9, 3.10.

Changed
===================
- The init function of QTenor and Module class adds `dtype` parameter. The types of QTenor index and input of some neural network layers are restricted.
- Quantum neural network, due to MacOS compatibility issues, the Mnist_Dataset and CIFAR10_Dataset interfaces have been removed.

[v2.0.6] - 2023-02-22
***************************


Added
===================

- Classic neural network, add interface: multinomial, pixel_shuffle, pixel_unshuffle, add numel for QTensor, add CPU dynamic memory pool function, add init_from_tensor interface for Parameter.
- Classic neural network, add interface: Dynamic_LSTM, Dynamic_RNN, Dynamic_GRU.
- Classic neural network, add interfaces: pad_sequence, pad_packed_sequence, pack_pad_sequence.
- Quantum neural network, add interfaces: CCZ, Controlled_Hadamard, FermionicSingleExcitation, UCCSD, QuantumPoolingCircuit,
- Quantum neural network, add interfaces: Quantum_Embedding, Mnist_Dataset, CIFAR10_Dataset, grad, Purity.
- Quantum neural network, adding examples: based on gradient clipping, quanvolution, quantum circuit expressiveness, barren plateau, and quantum reinforcement learning QDRL.

Changed
===================

- API documentation, restructure the content structure, add "quantum machine learning research" module, change "VQNet2ONNX module" to "Other Utility Functions".



fixed
===================

- Classical neural network, solving the problem that the same random seed produces different normal distributions across platforms.
- Quantum neural network, implement expval, ProbMeasure, QuantumMeasure support for QPanda GPU virtual machine.


[v2.0.5] - 2022-12-25
***************************


Added
===================

- Classical neural network, add log_softmax implementation, add the interface export_model function of the model to ONNX.
- Classic neural network, which supports the conversion of most of the existing classic neural network modules to ONNX. For details, refer to the API document "VQNet2ONNX module".
- Quantum neural network, add VarMeasure, MeasurePauliSum, Quantum_Embedding, SPSA and other interfaces
- Quantum neural network, adding LinearGNN, ConvGNN, ConvGNN, QMLP, quantum natural gradient, quantum random parameter-shift algorithm, DoublySGD algorithm, etc.


Changed
===================

- Classic Neural Networks, added dimensionality checks for BN1d, BN2d interfaces.

fixed
==================

- Solve the bug of maxpooling parameter checking.
- Solve [::-1] slice bug.


[v2.0.4] - 2022-09-20
***************************


Added
==================

- Classical neural network, adding LayernormNd implementation, supporting multi-dimensional data layernorm calculation.
- Classical neural network, add CrossEntropyLoss and NLL_Loss loss function calculation interface, support 1-dimensional to N-dimensional input.
- Quantum neural network, adding common circuit templates: HardwareEfficientAnsatz, StronglyEntanglingTemplate, BasicEntanglerTemplate.
- Quantum neural network, adding the Mutal_info interface for calculating the mutual information of qubit subsystems, Von Neumann entropy VB_Entropy, and density matrix DensityMatrixFromQstate.
- Quantum neural network, add quantum perceptron algorithm example QuantumNeuron, add quantum Fourier series algorithm example.
- Quantum neural network, adding the interface QuantumLayerMultiProcess that supports multi-process accelerated operation of quantum circuits.

Changed
==================

- Classical neural network, supports group convolution parameter group, dilation_rate of dilated convolution, and arbitrary value padding as parameters for one-dimensional convolution Conv1d, two-dimensional convolution Conv2d, and deconvolution ConvT2d.
- Skip the broadcast operation for data in the same dimension, reducing unnecessary running logic.

fixed
==================

- Solve the problem that the stack function is incorrectly calculated under some parameters.


[v2.0.3] - 2022-07-15
***************************


Added
==================

- Add support for stack, bidirectional recurrent neural network interface: RNN, LSTM, GRU
- Add interfaces for common calculation performance indicators: MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_score,precision_recall_f1_N_score, auc_calculate
- Increase the algorithm example of quantum kernel SVM

Changed
==================

- Speed up the print speed when there is too much QTensor data
- Use openmp to accelerate calculations under Windows and Linux.

fixed
==================

- Solve the problem that some python import methods cannot be imported
- Solve the problem of repeated calculation of batch normalization BN layer
- Solve the bug that the QTensor.reshape and transpose interfaces cannot calculate the gradient
- Add input parameter shape judgment for tensor.power interface

[v2.0.2] - 2022-05-15
***************************


Added
==================

- Added topK, argtoK
- increase cumsum
- Added masked_fill
- Increase triu, tril
- Added examples of random distribution generated by QGAN

Changed
==================

- Support advanced slice index and common slice index
- matmul supports 3D, 4D tensor operations
- Modify HardSigmoid function implementation

fixed
==================

- Solve the problem that convolution, batch normalization, deconvolution, pooling layer and other layers do not cache internal variables, resulting in the calculation of gradients during multiple back-passes after one forward pass
- Fixed implementation and example of QLinear layer
- Solve the problem of Image not load when MAC imports VQNet in the conda environment.




[v2.0.1] - 2022-03-30
**************************


Added
==================

- More than 100 basic data structure QTenor interfaces have been added, including creation functions, logic functions, mathematical functions, and matrix operations.
- Added 14 basic neural network functions, including convolution, deconvolution, pooling, etc.
- Add 4 loss functions, including MSE, BCE, CCE, SCE, etc.
- Add 10 activation functions, including ReLu, Sigmoid, ELU, etc.
- Add 6 optimizers, including SGD, RMSPROP, ADAM, etc.
- Added machine learning examples: QVC, QDRL, Q-KMEANS, QUnet, HQCNN, VSQL, Quantum Autoencoder.
- Add quantum machine learning layers: QuantumLayer, NoiseQuantumLayer.