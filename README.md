# Unsupervised Unlearning of Concept Drift with Autoencoders

This repository contains the implementation of the methods proposed in the paper [Unsupervised Unlearning of Concept Drift with Autoencoders](paper.pdf) by André Artelt, Kleanthis Malialis, Christos G. Panayiotou, Marios M. Polycarpou and Barbara Hammer.

The experiments, as described in the paper, are implemented in the folder [Implementation/](Implementation).

## Abstract

The phenomena of concept drift refers to a change of the data distribution affecting the data stream of future samples -- such non-stationary environments are often encountered in the real world. Consequently, learning models operating on the data stream might become obsolete, and need costly and difficult adjustments such as retraining or adaptation. Existing methods to address concept drift are, typically, categorised as active or passive. The former continually adapt a model using incremental learning, while the latter perform a complete model retraining when a drift detection mechanism triggers an alarm. We depart from the traditional avenues and propose for the first time an alternative approach which "unlearns" the effects of the concept drift. Specifically, we propose an autoencoder-based method for "unlearning" the concept drift in an unsupervised manner, without having to retrain or adapt any of the learning models operating on the data.

## Details

### Data preparation

The hanoi data set must be prepared as follows:

1. Download the Hanoi scenarios from [LeakDB](https://github.com/KIOS-Research/LeakDB) and put the scenarios in ``Implementation/hanoi_data/LeakDB/Hanoi_CMH/``.
2. Create the folders ``Implementation/hanoi-data/hanoi_clean/`` and ``Implementation/hanoi-data/hanoi_faultysensor/``
3. Run [Implementation/hanoi-data/datagenerator.py](Implementation/hanoi-data/datagenerator.py)
4. Run [Implementation/hanoi-data/generate_sensor_fault.py](Implementation/hanoi-data/generate_sensor_fault.py)

### Implementation of the experiments

The experiments on the digit data set are implemented in [Implementation/experiments_digits.py](Implementation/experiments_digits.py).
The experiments on the hanoi data set are implemented in [Implementation/experiments_hanoi.py](Implementation/experiments_hanoi.py) -- make sure you preared the data as described above!

## Requirements
- Python3.8
- Packages as listed in [Implementation/REQUIREMENTS.txt](Implementation/REQUIREMENTS.txt)

## License

MIT license - See [LICENSE](LICENSE).

## How to cite

You can cite the version on [arXiv]().
