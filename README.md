# URL_His_TissueSeg
The source code of paper in IEEE Transactions on Medical Imaging: **Unsupervised Representation Learning for Tissue Segmentation in Histopathological Images: From Global to Local Contrast**

[TMI2022](https://ieeexplore.ieee.org/document/9830779)

![URL_TS](./Tasks_Info.png)

Our framework is enlightened by a domain-specific cue: different tissues are composed by different cells and extracellular matrices.
Thus, we design three contrastive learning tasks with multi-granularity views (from global to local) for encoding necessary features into representations without accessing annotations.

- (1) an image-level task to capture the difference between tissue components, i.e., encoding the component discrimination; 
- (2) a superpixel-level task to learn discriminative representations of local regions with different tissue components, i.e., encoding the prototype discrimination;
- (3) a pixel-level task to encourage similar representations of different tissue components within a local region, i.e., encoding the spatial smoothness.

## Set Up Environment
```bash
conda env create -f environment.yaml
```

## Dataset
- Download the BCTS dataset from [this link](https://github.com/PathologyDataScience/BCSS).
- Download the GCTS dataset from [zeyugao/GastricSemanticSegmentation](https://huggingface.co/datasets/zeyugao/GastricSemanticSegmentation).
- The pre-training datasets, BCPT and GCPT are too large to upload, Please download from the [GDC portal](https://portal.gdc.cancer.gov/) directly. 

## Usage
For detailed configuration instructions, see `config.py` and `ft_config.py`.

### Pre-processing
#### For pre-training dataset:
1. Generating Binary Mask for WSIs `./preprocess/back_ground_filter_for_wsi.py`.
2. Extracting Large Image Patches (10000x10000) from WSIs (no-overlapping) `./preprocess/extract_patches_for_wsi.py`.
3. Extracting Small Image Patches (768x768 with the step size of 512x512) from Large Image Patches and Converting the format of Image Patches from png to npy `./preprocess/Extract_Patches_for_Pretraining.ipynb`.
4. Generating Superpixel Maps for each image patch `./preprocess/Gen_Sp_Map.ipynb`.
##### Note that, after performing different augmentations on image_patches, resizing and center cropping are adopted for image-level and other contrastive learning tasks, respectively. The input size is still 256x256 for pre-training.
#### For fine-tuning dataset:
1. Generating Small Image Patches (512x512 with the step size of 256x256) with corresponding Label Masks from the Original Annotated Image Patches `./preprocess/Extract_Patches_for_Finetuning.ipynb`.

### Pre-training
Setup `config.py`;
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 120520 main-distribute.py --gpus=0,1,2,3
```
### Fine-tuning
Setup `ft_config.py`;
```bash
python finetune.py --gpus=0,1 --seeds=10
```
## Citation

If any part of this code is used, please give appropriate citation to our paper.

## Authors
- Zeyu Gao (betpotti@gmail.com)
- Chen Li (cli@xjtu.edu.cn)

## Institute
[BioMedical Semantic Understanding Group](http://www.chenli.group/home), Xi'an Jiaotong University

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

## Acknowledgements
The datasets used are in whole or part based upon data generated by [the TCGA Research Network](https://www.cancer.gov/tcga).
