# A fine-tuning on a pre-trained DNAGPT

## Getting Started

### Download codes

```bash
git clone https://github.com/nerocca/ECE-285-Final-Project.git
```

### Download finetuning weights
You can download the weights from Google Drive.

https://drive.google.com/file/d/1FtZpR4FleUPCB4f-BykgyfobYyqpblU4/view?usp=drive_link

and save model weights to checkpoint dir
```bash
cd DNAGPT/checkpoints
# download or copy model weight to this default directory
```
#### Pre-trained model weight:
* [dna_gpt0.1b_h.pth](https://drive.google.com/file/d/15m6CH3zaMSqflOaf6ec5VPfiulg-Gh0u/view?usp=drive_link): DNAGPT 0.1B params model pretrained with human genomes
* 
#### Training Dataset:
* We use Human GRCh38 Genome Chromosome 22 Segments as our traing and test datasets. You can download the dataset from Google Drive.
* https://drive.google.com/file/d/1g2mbWcQ4fjd6WSO-Q_6BjN9fEIcqLLkx/view?usp=drive_link
## Install

### Pre-requirements
* python >= 3.8

### Required packages
```bash
cd DNAGPT
pip install -r requirements.txt
```





