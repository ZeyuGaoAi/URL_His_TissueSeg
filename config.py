from yacs.config import CfgNode as CN

_C = CN()

_C.mode = 'ss_dc'

_C.imgMaxSize = 256
# from small to large
# _C.imgReSizes = [384, 512, 768, 1024, 1536]
# _C.imgReSizes = [384, 768, 1536]
_C.imgReSizes = [768]
_C.segm_downsampling_rate = 2 # 2 for dc
_C.n_class = 5
_C.fixed_sp = False
_C.learned_sp = False

_C.pretrained = True
_C.freeze = True
_C.pretrained_path = ""
# _C.pretrained_path = "./checkpoints/resnet18TCGABR-Rdnct3_ss_nc_ALL.pth"
# _C.pretrained_path = "./checkpoints/resnet18TCGAGS-Rdnct3_ss_nc_ALL.pth"

_C.model_name = "resnet18TCGAGS-ImgNet_spp"

_C.data_inst = "ALL"
_C.train_data_dir = "/home5/gzy/Grastric_10000/cancer_patches/768x768_512x512_%s" % (_C.data_inst)
_C.train_data_sp_dir = "/home5/gzy/Grastric_10000/cancer_patches/768x768_512x512_SpFCN"
# _C.train_data_dir = "/home1/gzy/BreastCancerSeg/DATA/Train/768x768_256x256_%s" % (_C.data_inst)
_C.val_data_dir = "/home1/gzy/BreastCancerSeg/DATA/Test/768x768_256x256"


# _C.data_inst = "5p"
# _C.train_data_dir = "/home1/gzy/BreastCancerSeg/DATA/Train/512x512_256x256_%s" % (_C.data_inst)
# _C.val_data_dir = "/home1/gzy/BreastCancerSeg/DATA/Test/512x512_256x256"
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 120518 main-distribute.py --gpus=2,3
# python -m torch.distributed.launch --nproc_per_node=3 --master_port 120520 main-distribute.py --gpus=0,1,2
# 
_C.lr = 1e-3
_C.iter_per_epoch = 2000

_C.num_epochs = 50
_C.batch_size = 8
_C.num_workers = 4

_C.temperature = 0.3
