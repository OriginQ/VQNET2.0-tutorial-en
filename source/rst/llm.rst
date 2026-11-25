.. _llm:

Quantum Large Model Fine-Tuning Example
***************************************

In recent years, large models have become widespread and their scale has grown steadily, which makes training large-scale quantum machine learning models significantly more expensive. To reduce the training resources required for fine-tuning large models, several fine-tuning methods have been proposed that train only a small number of parameters instead of all parameters, so that the model still achieves results on downstream tasks comparable to full-parameter fine-tuning.
Fine-tuning based on quantum circuits, however, is not yet widely adopted.

``VQNet`` combines with ``Llama factory`` and ``peft`` to fine-tune large models based on quantum circuits.

Installing Dependencies for Quantum Large Model Fine-Tuning
-----------------------------------------------------------

This section explains how to install the dependencies required to use quantum circuits for large model fine-tuning.


To use quantum large model fine-tuning, you mainly need two packages, ``quantum-llm`` and ``pyvqnet``. The ``pyvqnet`` version must be ``2.15.0`` or later.


First, install the ``quantum-llm`` library

.. code-block::

    git clone https://gitee.com/craftsman_lei/quantum-llm.git


Then install the other dependent libraries and files as described in the ``README.md`` document in ``quantum-llm``

.. code-block::

    # Download other dependencies
    pip install -r requirements.txt

    # Install peft_vqc
    cd peft_vqc && pip install -e .

After installing the ``quantum-llm`` library and its dependencies, install ``pyvqnet``

.. code-block::

    # Install VQNet
    pip install pyvqnet # pyvqnet>=2.15.0

Quantum Large Model Fine-Tuning Training Steps
----------------------------------------------

After installing the required packages, refer to scripts such as ``train.sh`` in ``/quantum-llm/examples/qlora_single_gpu/``. Use the script to specify parameters such as the base model for training, the fine-tuning module to use, and the output path of the fine-tuning module.

The Qwen2.5-0.5B model can be downloaded from https://huggingface.co/Qwen/Qwen2.5-0.5B?clone=true. The same applies to other models. If the download fails, you can download the individual files from the site and then use them:

.. code-block::

    # Download Qwen2.5-0.5B
    git clone https://huggingface.co/Qwen/Qwen2.5-0.5B

A sample ``train.sh`` script is shown below. Set parameters such as the base model, dataset, and output path. Set ``model_name_or_path`` to the specified model; if it cannot be accessed, download the base model yourself and set the absolute path of the base model.

.. note::
    The ``train.sh``, ``eval.sh``, ``cli.sh``, and other script files are all run from the ``/quantum-llm/examples/qlora_single_gpu/`` directory

.. code-block::

    #!/bin/bash

    CUDA_VISIBLE_DEVICES=1 python ../../src/train_bash.py \
        --stage sft \
        --model_name_or_path /download/path/Qwen2.5-0.5B/ \
        --dataset alpaca_gpt4_en \
        --tokenized_path ../../data/tokenized/alpaca_gpt4_en/ \
        --dataset_dir ../../data \
        --template qwen \
        --finetuning_type vqc \
        --lora_target q_proj,v_proj \
        --output_dir ../../saves/Qwen2.5-0.5B/vqc/alpaca_gpt4_en \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len 1024 \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --warmup_steps 20 \
        --save_steps 100 \
        --eval_steps 100 \
        --evaluation_strategy steps \
        --load_best_model_at_end \
        --learning_rate 5e-5 \
        --num_train_epochs 3.0 \
        --max_samples 1000 \
        --val_size 0.1 \
        --plot_loss \
        --fp16 \
        --do-train \

    # Run from the command line
    sh train.sh

Compared with classical large model fine-tuning modules, the quantum large model fine-tuning module adds three extra fine-tuning methods:

``vqc`` : A vqc fine-tuning module implemented with VQNet

``quanTA`` : A quantum tensor decomposition module

``tq`` : A vqc module implemented with torch quantum

The ``train.sh`` example above is a script sample for fine-tuning with the ``vqc`` module. To use the other two fine-tuning modules, simply change ``finetuning_type`` to ``quanTA`` or ``tq``. The experimental results of the three modules were recorded and plotted as follows:

.. image:: ./images/peft1.png
   :width: 600 px
   :align: center

|

The figure above shows the training results of the ``Qwen2.5-0.5B`` base model on the ``alpaca_gpt4_en`` dataset. As can be observed, the VQNet-based ``vqc`` module achieves the best loss convergence, which demonstrates the effectiveness of using quantum circuits for large model fine-tuning.


With the ``train.sh`` training script, the fine-tuned module parameters can be saved to the specified directory through the ``--output_dir`` parameter.
Then evaluate them with the ``eval.sh`` script in the same directory, ``/quantum-llm/examples/qlora_single_gpu/``. The script content is as follows:

.. code-block::

    #!/bin/bash

    CUDA_VISIBLE_DEVICES=1 python ../../src/evaluate.py \
        --model_name_or_path /download/path/Qwen2.5-0.5B/ \
        --template qwen \
        --finetuning_type vqc \
        --task cmmlu \
        --task_dir ../../evaluation/ \
        --adapter_name_or_path ../../saves/Qwen2.5-0.5B/vqc/alpaca_gpt4_en \

    # Run from the command line
    sh eval.sh

Set the base model path with ``--model_name_or_path``, and load the trained module with ``--adapter_name_or_path`` to evaluate on the relevant tasks. The ``--task`` parameter accepts ``cmmlu``, ``ceval``, and ``mmlu``.

Then call the ``cli_demo.py`` file for question answering, also executed through the ``cli.sh`` script in the current directory. The script content is as follows:

.. code-block::

    #!/bin/bash

    CUDA_VISIBLE_DEVICES=1 python ../../src/cli_demo.py  \
        --model_name_or_path /download/path/Qwen2.5-0.5B/ \
        --template qwen \
        --finetuning_type vqc \
        --adapter_name_or_path ../../saves/Qwen2.5-0.5B/vqc/alpaca_gpt4_en \
        --max_new_tokens 1024


    # Run from the command line
    sh cli.sh

Details of More Related Parameters
----------------------------------

.. list-table:: Fine-Tuning Module Parameter Descriptions
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - stage
     - Determines the large model training mode. pt is pretraining, sft is the fine-tuning stage. The experiments use sft.
   * - model_name_or_path
     - Path to the base model.
   * - dataset
     - Selects the dataset, such as identity or alpaca_gpt4_zh.
   * - tokenized_path
     - Path to the tokenized dataset.
   * - dataset_dir
     - Path to the dataset.
   * - template
     - Model template type, such as llama3.
   * - finetuning_type
     - Specifies the fine-tuning method, such as lora, tq, vqc, or quanTA.
   * - lora_target
     - Target modules are q_proj and v_proj
   * - output_dir
     - Path where the fine-tuning module is saved
   * - overwrite_cache
     - Whether to overwrite the cached training and evaluation sets
   * - overwrite_output_dir
     - Whether to overwrite existing files in the output directory
   * - cutoff_len
     - Truncation length used when processing data
   * - preprocessing_num_workers
     - Number of worker processes used for data preprocessing
   * - per_device_train_batch_size
     - Batch size per GPU; training parameter
   * - per_device_eval_batch_size
     - Batch size during evaluation; training parameter
   * - gradient_accumulation_steps
     - Number of gradient accumulation steps; training parameter
   * - lr_scheduler_type
     - Learning rate scheduler; training parameter
   * - logging_steps
     - Logging interval
   * - warmup_steps
     - Number of warmup steps
   * - save_steps
     - Model save interval
   * - eval_steps
     - Evaluation save interval
   * - evaluation_strategy
     - Evaluation strategy; set here to evaluation by steps
   * - load_best_model_at_end
     - Load the best-performing model at the end of training
   * - learning_rate
     - Learning rate; training parameter
   * - num_train_epochs
     - Number of training epochs to run; training parameter
   * - max_samples
     - Maximum number of training samples
   * - val_size
     - Validation set size
   * - plot_loss
     - Whether to save the training loss curve
   * - fp16
     - Whether to use fp16 mixed precision training; the vqc module uses float32
   * - do-train
     - Whether the task is a training task
   * - adapter_name_or_path
     - Path to the files generated after training
   * - task
     - Selects the task; currently ceval, cmmlu, and mmlu are supported
   * - task_dir
     - Path to the task
   * - q_d
     - Number of tensor decompositions for the quanTA module. Default: 4
   * - per_dim_features
     - Number of tensor decomposition features for the quanTA module. Default: [16,8,4,2]
