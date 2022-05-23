from yacs.config import CfgNode as CN

_C = CN()

_C.mode = 'sl'

_C.imgMaxSize = 256
_C.imgReSizes = [768]
_C.segm_downsampling_rate = 1
_C.n_class = 6

_C.pretrained = False
_C.freeze = False
# _C.pretrained_path = ""
_C.pretrained_path = "./checkpoints/resnet18TCGAGS-Rdnct3-OPix_ss_dc_ALL.pth"
# _C.pretrained_path = "./checkpoints/resnet18TCGABR-Rdnct3_spp_learned_ss_dc_ALL_50.pth"

_C.model_name = "resnet18GASeg-Rdnct3-OPix"

_C.data_inst = "15p"
_C.train_data_dir = "/home1/gzy/GastricCancerSeg/DATA/Train/512x512_256x256_%s" % (_C.data_inst)
_C.val_data_dir = "/home1/gzy/GastricCancerSeg/DATA/Test/512x512_256x256"

# hyper-parameters
_C.lr = 1e-3
_C.iter_per_epoch = 2000

_C.num_epochs = 50
_C.batch_size = 32
_C.num_workers = 4

_C.temperature = 0.3
