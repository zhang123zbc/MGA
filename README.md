# MGA

Multi-path Gradient Ascent

To improve the transferability of black-box adversarial attacks, this paper proposes a multi-path optimization strategy that allocates computational budget across diverse trajectories to reduce gradient redundancy and boost attack success rates across various models.

![Overview](./overview.png)

## Usage

### Installation

* Python >= 3.13
* PyTorch >= 2.7.1
* Torchvision >= 0.22.1
* timm >= 0.6.12

```bash
pip install -r requirements.txt
```

### Import clean examples

Building on previous work, we randomly selected 1,000 images from the ImageNet validation set for our experiments. All the images can be correctly classified. You can also construct your own sample set according to the following format. You can also download our prepared dataset and the generated adversarial examples from [Zenodo](https://zenodo.org/records/19218397).

```text
dataset
├─images
│  ├─ILSVRC2012_val_00000019.png
│  ├─...
│  └─ILSVRC2012_val_00049962.png
└─labels
```

```text
ImageId,TrueLabel
ILSVRC2012_val_00018317.png,0
...
ILSVRC2012_val_00041747.png,999
```

### Generate adversarial examples


```bash
python main.py --input_dir ./dataset/images --input_csv ./dataset/labels
```

or simply

```bash
python main.py
```

if you put the data under the same directory and use the default settings.

### Run for Evaluation

```bash
python eval.py
```

You can perform evaluations of other models by modifying eval.py

## Credits

This repository is modified from [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack?tab=readme-ov-file). Great, thank you!
Download the source code of [ResPA](https://github.com/ZezeTao/ResPA). Great, thank you!

