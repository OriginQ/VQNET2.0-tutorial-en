Other Utility Functions
=========================

Seeds for Random Distributions
----------------------------------

set_random_seed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.utils.set_random_seed(seed)
    
    Set the global random seed.

    :param seed: random seed.

    .. note::

            When a fixed random number seed is specified, the random distribution will generate a fixed pseudo-random distribution based on the random seed.
            Affects functions include: `tensor.randu` , `tensor.randn` , parameter initialization for parametric classical neural networks and quantum computing layers.

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.utils import get_random_seed, set_random_seed

        set_random_seed(256)


        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randn([2, 3])
        print(rn)
        rn = tensor.randu([2, 3])
        print(rn)
        rn = tensor.randu([2, 3])
        print(rn)

        from pyvqnet.nn.parameter import Parameter
        from pyvqnet.utils.initializer import he_normal, he_uniform, xavier_normal, xavier_uniform, uniform, quantum_uniform, normal
        print(Parameter(shape=[2, 3], initializer=he_normal))
        print(Parameter(shape=[2, 3], initializer=he_uniform))
        print(Parameter(shape=[2, 3], initializer=xavier_normal))
        print(Parameter(shape=[2, 3], initializer=xavier_uniform))
        print(Parameter(shape=[2, 3], initializer=uniform))
        print(Parameter(shape=[2, 3], initializer=quantum_uniform))
        print(Parameter(shape=[2, 3], initializer=normal))
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # ########################################################
        # [
        # [-0.9874518, 0.9198063, 0.0650323],
        #  [0.1976041, 1.0307300, 0.2322134]
        # ]
        # [
        # [-0.2134037, 0.1987845, -0.5292138],
        #  [0.3732708, 0.1775801, 0.5395861]
        # ]
        # [
        # [-0.7648768, 0.7124789, 0.0503738],
        #  [0.1530635, 0.7984000, 0.1798717]
        # ]
        # [
        # [-0.4049051, 0.3771670, -1.0041126],
        #  [0.7082316, 0.3369346, 1.0237927]
        # ]
        # [
        # [0.3151870, 0.6721524, 0.0416874],
        #  [0.8232620, 0.6537889, 0.9672953]
        # ]
        # [
        # [1.9803783, 4.2232580, 0.2619299],
        #  [5.1727076, 4.1078768, 6.0776958]
        # ]
        # [
        # [-1.2093765, 1.1265280, 0.0796480],
        #  [0.2420146, 1.2623813, 0.2844022]
        # ]

get_random_seed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.utils.get_random_seed()
    
    Get current random seed.

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet.utils import get_random_seed, set_random_seed

        set_random_seed(256)
        print(get_random_seed())
        #256

VQNet2ONNX module
-------------------

The VQNet2ONNX module supports converting VQNet models and parameters to ONNX model format. The deployment of the VQNet model to a variety of inference engines can be completed through ONNX, including TensorRT/OpenVINO/MNN/TNN/NCNN, and other inference engines or hardware that support the ONNX open source format.

Environment dependency: onnx>=1.12.0

.. note::

    Currently, QPanda quantum circuit modules are not supported to be converted to ONNX, and only models composed of pure classical operators are supported.

Use the ``export_model`` function to export ONNX models. This function requires more than two parameters: the model ``model`` constructed by VQNet, the model single input ``x`` or multi-input ``*argc``.
Below is the sample code for ONNX export of `ResNet` model and validated with onnxruntime.

Import related python libraries

.. code-block::

    import numpy as np
    from pyvqnet.tensor import *
    from pyvqnet.nn import Module, BatchNorm2d, Conv2D, ReLu, AvgPool2D, Linear
    from pyvqnet.onnx.export import export_model
    from onnx import __version__, IR_VERSION
    from onnx.defs import onnx_opset_version
    print(
        f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}"
    )

Model definition

.. code-block::

    class BasicBlock(Module):

        expansion = 1

        def __init__(self, in_chals, out_chals, stride=1):
            super().__init__()
            self.conv2d1 = Conv2D(in_chals,
                                out_chals,
                                kernel_size=(3, 3),
                                stride=(stride, stride),
                                padding=(1, 1),
                                use_bias=False)
            self.BatchNorm2d1 = BatchNorm2d(out_chals)
            self.conv2d2 = Conv2D(out_chals,
                                out_chals * BasicBlock.expansion,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                use_bias=False)
            self.BatchNorm2d2 = BatchNorm2d(out_chals * BasicBlock.expansion)
            self.Relu = ReLu(name="relu")
            #shortcut
            self.shortcut_conv2d = Conv2D(in_chals,
                                        out_chals * BasicBlock.expansion,
                                        kernel_size=(1, 1),
                                        stride=(stride, stride),
                                        use_bias=False)
            self.shortcut_bn2d = BatchNorm2d(out_chals * BasicBlock.expansion)
            self.need_match_dim = False
            if stride != 1 or in_chals != BasicBlock.expansion * out_chals:
                self.need_match_dim = True

        def forward(self, x):
            y = self.conv2d1(x)
            y = self.BatchNorm2d1(y)
            y = self.Relu(self.conv2d2(y))
            y = self.BatchNorm2d2(y)
            y = self.Relu(y)
            if self.need_match_dim == False:
                return y + x
            else:
                y1 = self.shortcut_conv2d(x)
                y1 = self.shortcut_bn2d(y1)
                return y + y1

    resize = 32

    class ResNet(Module):
        def __init__(self, num_classes=10):
            super().__init__()

            self.in_chals = 64 // resize
            self.conv1 = Conv2D(1,
                                64 // resize,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                use_bias=False)
            self.bn1 = BatchNorm2d(64 // resize)
            self.relu = ReLu()
            self.conv2_x_1 = BasicBlock(64 // resize, 64 // resize, 1)
            self.conv2_x_2 = BasicBlock(64 // resize, 64 // resize, 1)
            self.conv3_x_1 = BasicBlock(64 // resize, 128 // resize, 2)
            self.conv3_x_2 = BasicBlock(128 // resize, 128 // resize, 1)
            self.conv4_x_1 = BasicBlock(128 // resize, 256 // resize, 2)
            self.conv4_x_2 = BasicBlock(256 // resize, 256 // resize, 1)
            self.conv5_x_1 = BasicBlock(256 // resize, 512 // resize, 2)
            self.conv5_x_2 = BasicBlock(512 // resize, 512 // resize, 1)
            self.avg_pool = AvgPool2D([4, 4], [1, 1], "valid")
            self.fc = Linear(512 // resize, num_classes)


        def forward(self, x):
            output = self.conv1(x)
            output = self.bn1(output)
            output = self.relu(output)
            output = self.conv2_x_1(output)
            output = self.conv2_x_2(output)
            output = self.conv3_x_1(output)
            output = self.conv3_x_2(output)
            output = self.conv4_x_1(output)
            output = self.conv4_x_2(output)
            output = self.conv5_x_1(output)
            output = self.conv5_x_2(output)
            output = self.avg_pool(output)
            output = tensor.flatten(output, 1)
            output = self.fc(output)

            return output

测试代码

.. code-block::

    def test_resnet():

        x = tensor.ones([4,1,32,32])#Arbitrarily input a QTensor data of the correct shape
        m = ResNet()
        m.eval()#In order to export the global mean and global variance of BatchNorm
        y = m(x)
        vqnet_y = y.CPU().to_numpy()

        #export onnx model
        onnx_model = export_model(m, x)

        #save model to file
        with open("demo.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        # compare running result by onnxruntime
        import onnxruntime
        session = onnxruntime.InferenceSession('demo.onnx', None)
        input_name = session.get_inputs()[0].name

        v = np.ones([4,1,32,32])
        v = v.astype(np.float32)
        inputs = [v]
        test_data_num = len(inputs)
        outputs = [
            session.run([], {input_name: inputs[i]})[0]
            for i in range(test_data_num)
        ]
        onnx_y = outputs[0]
        assert np.allclose(onnx_y, vqnet_y)


    if __name__ == "__main__":
        test_resnet()


Use https://netron.app/ , the ONNX model exported by VQNet can be visualized demo.onnx

.. image:: ./images/resnet_onnx.png
   :width: 100 px
   :align: center

|


The following are the supported VQNet modules

.. csv-table:: Supoorted VQNet modules
   :file: ./images/onnxsupport.csv

