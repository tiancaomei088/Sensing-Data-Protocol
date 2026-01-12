# SDP: Sensing Data-Protocol for Scalable Wireless Sensing

**SDP (Sensing Data-Protocol)** is a **protocol-level abstraction framework and unified benchmark** for **scalable wireless sensing and perception** based on wireless signals such as **Channel State Information(CSI)**.
The protocol is designed to **decouple learning performance from hardware-specific artifacts**, enabling **fair, reproducible, and scalable evaluation** of deep learning models for wireless sensing tasks.

SDP enforces **deterministic physical-layer sanitization**, **canonical tensor construction**, and **standardized training and evaluation procedures**, making it particularly suitable for **wireless sensing research**, **activity recognition**, **device-free sensing**, and **cross-dataset benchmarking**.

Our pipeline and main result can be illustrated by the following two pictures.


**SDP Pipeline**
![pipeline](./img/pipeline.png)

**Mean Top-1 accuracy with 95% confidence intervals over five runs**
![accuracy](./img/accuracy.png)

More details are illustrated in our paper [A Sensing Dataset Protocol for Benchmarking and Multi-Task Wireless Sensing](https://arxiv.org/abs/2512.12180).

```
@misc{huang2025sensingdatasetprotocolbenchmarking,
      title={A Sensing Dataset Protocol for Benchmarking and Multi-Task Wireless Sensing}, 
      author={Jiawei Huang and Di Zhang and Yuanhao Cui and Xiaowen Cao and Tony Xiao Han and Xiaojun Jing and Christos Masouros},
      year={2025},
      eprint={2512.12180},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2512.12180}, 
}
```

---

## 🔍 Why SDP?

Wireless sensing research often suffers from:
- Inconsistent **hardware configurations**
- Dataset-specific **preprocessing pipelines**
- Non-reproducible **training and evaluation protocols**

**SDP addresses these challenges at the protocol level**, rather than the model level.

### Core Design Principles
- **Protocol-level abstraction**
- **Deterministic PHY-layer sanitization** to eliminate randomness
- **Canonical tensor representation** for deep learning compatibility
- **Unified benchmark pipeline** across datasets and tasks
- **Extensible architecture** for new datasets, processors, and models

---

## 🧠 Target Use Cases

SDP is optimized for:
- CSI-based **Human Activity Recognition (HAR)**
- **Gait recognition** and biometric identification
- **Wireless sensing + deep learning** research
- **Cross-domain / cross-hardware generalization**
- **scalable sensing systems**

Typical downstream models include CNNs, Transformers, BiLSTMs, GNNs, and hybrid architectures.

### Supported Dataset:

**Widar3.0**
 - [Dataset Link](http://sdp8.org/Dataset?id=028828f9-1997-48df-895c-9724551a22ae)
 - CSI Shape: (Time, 30, 1, 3)
 - num of classes: 6
 - total num of used samples: 12,000

**GaitID**
 - [Dataset Link](http://sdp8.org/Dataset?id=87a65da2-18cb-4b8f-a1ec-c9696890172b)
 - CSI Shape: (Time, 30, 1, 3)
 - num of classes: 11
 - total num of used samples: 22,500

 **XRF55**
- [Dataset Link](http://sdp8.org/Dataset?id=705e08e7-637e-49a1-aff1-b2f9644467ae)
 - CSI Shape: (270, 1000)
 - num of classes: 55
 - total num of used samples: 9,900

 **ElderAL-CSI**
 - [Dataset Link](http://sdp8.org/Dataset?id=f144678d-5b4a-4bb9-902c-7aff4916a029)
 - CSI Shape: (Time, 512, 3, 3)
 - num of classes: 6
 - total num of used samples: 2,400

---

## 📦 Key Features

- **Unified CSI abstraction** across heterogeneous datasets
- **Hardware-agnostic signal representation**
- **Modular reader–processor–model pipeline**
- **Deterministic preprocessing for reproducibility**
- **Plug-and-play extensibility**
- **Benchmark-ready training and evaluation flow**



---

## 📁 Project Structure Overview

### `algorithms/` - Algorithm Storage

Store various functions for implementing different signal processing algorithms. 

- `./denoising.py`  
  Store functions for signal denoising and support extension
- `./phase_calibration.py`  
  Store functions for phase calibration and support extension

---

### `data/` – Dataset Storage

**Important**:
- Data from the same sampling project **must be placed in the same subfolder** and **be the same format**
- Mixing formats across folders may lead to **ignored files and exceptions**

---

### `readers/`
- Store Dataset-specific readers
- Converts raw files into `List` of `CSIData`
- Extensible via factory registration in file `__init__.py`

---

### `results/`
- Store confusion matrix and checkpoints of best models

---

### `structure/`
- Definition of `CSIData` and all kinds of `CSIFrame`

---

### `processor.py`
- concurrent signal processing and sanitization modules
- Enforces deterministic PHY-layer preprocessing
- Task-aware for labels

---



## 🚀 Quick Start

### Install Dependencies
Create a venv for dependencies, then run:
```bash

```

### Download Data
Please download needed datasets from [Our SDP Website](http://sdp8.org/) or other source and put them into `data/your subfolder`.

### Preprocess
Considering that there are countless lines of print when processing thousands of files, in your venv, the below is recommended:

```bash
# Check the params first, especially for file path :)
nohup python core.py >> output.log 2>&1 &
```

### Train and Eval
Now that you get `processed_data.npz`, run:

```bash
# Also, check the params :)
```

## 📚 Guide for extension
1. Add new frame in `structure/CSIFrame`
2. Add new reader in `readers/` and register in `__init__.py`
3. Add new lines for extracting labels and groups in `processor.py`
4. **(Option)** Create new .py file or add new functions for signal processing in existed files in `algorithms/` for extension.