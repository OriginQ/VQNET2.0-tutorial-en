VQNet Changelog
###############################

[v2.10.0] - 2023-12-30
***************************

Added
===========
- Added new interfaces under pyvqnet.qnn.vqc: IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MutliRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz, etc.;
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
- Combined quantum circuit based on vqc: VQC_HardwareEfficientAnsatz、VQC_BasicEntanglerTemplate、VQC_StronglyEntanglingTemplate、VQC_QuantumEmbedding、VQC_RotCircuit、VQC_CRotCircuit、VQC_CSWAPcircuit、VQC_Controlled_Hadamard、VQC_CCZ、VQC_FermionicSingleExcitation、VQC_FermionicDoubleExcitation、VQC_UCCSD、VQC_QuantumPoolingCircuit、VQC_BasisEmbedding、VQC_AngleEmbedding、VQC_AmplitudeEmbedding、VQC_IQPEmbedding。
- Measurement methods based on vqc: VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll。


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
- Add interfaces for common calculation performance indicators: MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_scoreprecision_recall_f1_N_score, auc_calculate
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