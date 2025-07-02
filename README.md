# CS444 - Deep Learning

This repository contains all projects for CS444 (Deep Learning) at Colby College, Spring 2025.

---

### Project 1: Building a Deep Learning Library

This project involves creating a deep learning library from scratch and using it to build and train convolutional neural networks.

*   **`network.py`**: Defines the core `DeepNetwork` class for training, evaluation, and backpropagation.
*   **`layers.py`**: Implements fundamental neural network layers (Dense, Conv2D, MaxPool2D, etc.).
*   **`block.py`**: Defines reusable blocks of layers (`VGGConvBlock`, `VGGDenseBlock`) to build models.
*   **`alexnet.py`**: Contains the implementation of the AlexNet and MiniAlexNet models.
*   **`vgg_nets.py`**: Implements the VGG4 and VGG6 model architectures.
*   **`tf_util.py`**: Provides low-level TensorFlow helper functions.
*   **`build_deeplib.ipynb`**: Notebook for the initial development and testing of the library's layers.
*   **`cifar10.ipynb`**: Notebook for loading and preprocessing the CIFAR-10 and MNIST datasets.
*   **`vgg4.ipynb`**: Notebook for training the VGG4 model and running experiments.
*   **`vgg6.ipynb`**: Notebook for training the VGG6 model and demonstrating advanced training techniques.

---

### Project 2: Branch Neural Networks (ResNets)

This project extends the deep learning library to build and train networks from the ResNet family, which feature branching architectures and skip connections.

*   **`resnets.py`**: Implements various ResNet architectures, including `ResNet8`, `ResNet18`, `ResNet50`, and `ResNeXt18`.
*   **`residual_block.py`**: Defines the core `ResidualBlock` with its skip connection logic.
*   **`bottleneck_block.py`**: Defines the `BottleneckBlock` used in deeper models like ResNet-50.
*   **`resnet.ipynb`**: Notebook for building, testing, and training ResNet models on CIFAR-10.
*   **`datasets.py`**: Updated data loader to include support for the CIFAR-100 dataset.
*   **`layers.py`**: The core layers library, updated to support features like batch normalization.
*   **`network.py`**: The core network class, updated to support new features like batch normalization initialization.
*   **`block.py`**: The base block library, used as a foundation for the new residual blocks.
*   **`tf_util.py`**: Provides low-level TensorFlow helper functions.

---

### Project 3: Word Embeddings

This project focuses on Natural Language Processing by implementing and training a Continuous Bag-of-Words (CBOW) model to generate word embeddings from a text corpus.

*   **`cbow.py`**: Implements the Continuous Bag-of-Words (CBOW) model for learning word embeddings.
*   **`cbow_layers.py`**: Defines the `DenseEmbedding` layer, which retrieves embedding vectors for word indices.
*   **`word_processing.py`**: A utility for processing text data and generating training samples.
*   **`cbow.ipynb`**: Notebook for training the CBOW model on the IMDB dataset and analyzing the resulting word embeddings.
*   **`network.py`**: The core `DeepNetwork` class, reused as a base for the CBOW model.
*   **`layers.py`**: The core layers library, providing the standard `Dense` layer.
*   **`tf_util.py`**: Provides low-level TensorFlow helper functions.

---

### Project 4: Generative Pre-trained Transformers (GPT)

This project implements the Transformer architecture from scratch to build and train several models from the GPT family for text generation.

*   **`gpts.py`**: Defines the GPT model family, including `GPTPico1`, `GPTMini6`, `GPT1`, and `GPT2XL`, and handles sequence generation.
*   **`transformer_layers.py`**: Implements core Transformer components like `Embedding`, `MultiHeadAttention`, and `FeedForward` layers.
*   **`transformer_blocks.py`**: Defines the main `TransformerBlock` and `PositionalEncodingBlock`.
*   **`bpe_tokenizers.py`**: Implements a Byte-Pair Encoding (BPE) tokenizer for text preprocessing.
*   **`gpt_pico.ipynb`**: Notebook for training and generating text with the small-scale `GPTPico1` model.
*   **`gpt_mini.ipynb`**: Notebook for training and generating text with the medium-scale `GPTMini6` model.
*   **`network.py`**: The core `DeepNetwork` class, adapted with a temporal cross-entropy loss for sequence models.
*   **`layers.py`**: The core layers library, updated with Layer Normalization and GELU activation.
*   **`tf_util.py`**: Provides low-level TensorFlow helper functions.
