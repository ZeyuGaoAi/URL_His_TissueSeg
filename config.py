from yacs.config import CfgNode as CN

_C = CN()

# ss_nc for encoder
# ss_dc for decoder
_C.mode = 'ss_dc'

_C.imgMaxSize = 256
# _C.imgReSizes = [384, 512, 768, 1024, 1536] # for ss_nc
_C.imgReSizes = [768] # for ss_dc
# 1 for ss_nc, 2 for ss_dc
_C.segm_downsampling_rate = 2
_C.n_class = 5 # unnecessary for pre-training
_C.fixed_sp = False # True for fixed window
_C.learned_sp = False # True for learned superpixels

# using pretrained weights of ImageNet or Not
_C.pretrained = False
# Freeze the encoder or not, for ss_nc: False, for ss_dc: True,
_C.freeze = True
# The pretrained weights of encoder, for ss_nc: "", for ss_dc should set a path
_C.pretrained_path = ""
# _C.pretrained_path = "./checkpoints/resnet18TCGABR-Rdnct3_ss_nc_ALL.pth"

_C.model_name = "resnet18TCGAGS-ImgNet_spp"

_C.data_inst = "ALL"
# path to pre-trainining dataset 
_C.train_data_dir = "/home5/gzy/Grastric_10000/cancer_patches/768x768_512x512_%s" % (_C.data_inst)
# path to segmented superpixels of pre-trainining dataset 
_C.train_data_sp_dir = "/home5/gzy/Grastric_10000/cancer_patches/768x768_512x512_SpFCN"
# val_data_dir : unnecessary for pre-training
_C.val_data_dir = "/home1/gzy/BreastCancerSeg/DATA/Test/768x768_256x256"

# hyper-parameters
_C.lr = 1e-3
_C.iter_per_epoch = 2000

_C.num_epochs = 50
_C.batch_size = 8
_C.num_workers = 4

_C.temperature = 0.3
