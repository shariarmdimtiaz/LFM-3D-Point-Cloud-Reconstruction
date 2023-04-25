## Enhanced 3D Point Cloud Reconstruction for Light Field Microscopy Using Unet-based Convolutional Neural Networks

<br>
<p align="center"> <img src="https://github.com/shariarmdimtiaz/LFM-3D-Point-Cloud-Reconstruction/blob/main/proposed-model.png" width="80%"> </p>

## Preparation:

### Requirement:

- TensorFlow 2.3.0 or higher, numpy 1.18.5, opencv-python 4.5.4, h5py 2.10.0, matplotlib 3.2.0 or higher.
- The code is tested with python=3.8, cuda=10.1.
- A single GPU with cuda memory larger than 32 GB is required.

### Datasets:

- We used the HCI 4D LF benchmark for training and evaluation. Please refer to the [benchmark website](https://lightfield-analysis.uni-konstanz.de) for details.

### Path structure of Datasets:

```
├──./full_data/
│    ├── training
│    │    ├── antinous
│    │    │    ├── gt_disp_lowres.pfm
│    │    │    ├── valid_mask.png
│    │    │    ├── input_Cam000.png
│    │    │    ├── input_Cam001.png
│    │    │    ├── ...
│    │    ├── boardgames
│    │    ├── ...
│    ├── validation
│    │    ├── backgammon
│    │    │    ├── gt_disp_lowres.pfm
│    │    │    ├── input_Cam000.png
│    │    │    ├── input_Cam001.png
│    │    │    ├── ...
│    │    ├── boxes
│    |    ├── ...
│    ├── test
│    │    ├── bedroom
│    │    │    ├── input_Cam000.png
│    │    │    ├── input_Cam001.png
│    │    │    ├── ...
│    │    ├── bicycle
│    |    ├── herbs
│    |    ├── origami
│    |    ├── ...
│    ├── lfm
│    │    ├── ic-1
│    │    │    ├── input_Cam000.png
│    │    │    ├── input_Cam001.png
│    │    │    ├── ...
│    │    ├── ic-2
│    |    ├── chip
│    |    ├── gear
```

## Test on your own LFs:

- Place the input LFs into `./full_data` (see the attached examples).
- Run `make_patchdataset_lfm.py` for the LFM dataset and run `make_patchdataset_hci.py` for the HCI dataset to prepare patch data.
- Run `test.py` to perform inference on each test scene.
- The result files (i.e., `scene_name.npy` and `scene_name.png`) will be saved to `./test_result/`.
