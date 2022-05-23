import os
import argparse
import warnings
import torch
import random
import numpy as np
from apex import amp
import torch.nn as nn
from torchsummary import summary
import time
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from loss import calc_sl_loss, calc_sl_loss_val, calc_gc_loss, calc_nc_loss, calc_dc_loss, print_metrics, calc_dc_loss_sp, calc_dc_loss_spp

from utils import rm_n_mkdir
from models import ResNetBackBone, ResNetUNet, ResNetUNetHead
from datasetBRS import TrainDataset, ValDataset, GlobalCDataset, NucleiCDataset, ValDataset1, TrainDataset1
from ft_config import _C as cfg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

warnings.filterwarnings('ignore')

tissue_type_dict = {
            'Tumor': 0, # ! Please ensure the matching ID is unique
            'Lymphoid': 1,
            'Stroma': 2,
            'Muscle' : 3,
            'Necrosis': 4,
            'Other': 5,
        }
   

def train_phase(model, optimizer, dataloader, epoch):
    model.train()  # Set model to training mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        outputs, features = model(inputs)

        loss = calc_sl_loss(outputs, labels, metrics, tissue_type_dict)

        optimizer.zero_grad()
        
        loss.backward()
#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward()
        
        optimizer.step()
        
        epoch_samples += inputs.size(0)
        
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss'] / epoch_samples
    
    for k in metrics.keys():
        writer.add_scalar('train/%s' % k, metrics[k] / epoch_samples, epoch)

    scheduler.step(epoch_loss)
#     for param_group in optimizer.param_groups:
#         print("LR", param_group['lr'])
            
def test_phase(model, dataloader, best_loss, epoch):
    model.eval()  # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        outputs, features = model(inputs)
        
        loss = calc_sl_loss_val(outputs, labels, metrics, tissue_type_dict)
        
        epoch_samples += inputs.size(0)
        
    print_metrics(metrics, epoch_samples, 'val')
    epoch_loss = metrics['loss'] / epoch_samples
    
    for k in metrics.keys():
        writer.add_scalar('val/%s' % k, metrics[k] / epoch_samples, epoch)

    if epoch_loss < best_loss:
        print(f"saving best model to {checkpoint_path.replace('.pth','.best')}")
        best_loss = epoch_loss
        torch.save(model.state_dict(), checkpoint_path.replace('.pth','.best'))
    return best_loss

def build_model():
    if cfg.mode == 'ss_gc' or cfg.mode == 'ss_nc':
        model = ResNetBackBone(pretrained=cfg.pretrained)
    elif cfg.mode == 'ss_dc':
        model = ResNetUNetHead(freeze=cfg.freeze, pretrained=cfg.pretrained)
    elif cfg.mode == 'ss_nc_dc': 
        model = ResNetUNetHead(freeze=cfg.freeze, pretrained=cfg.pretrained)
    elif cfg.mode == 'sl':
        model = ResNetUNet(cfg.n_class, pretrained=cfg.pretrained)
    else:
        print("No such mode!!!")
        
    for name,parameters in model.named_parameters():
        print(name,':',parameters.shape)
    
    return model

def build_dataset():
    if cfg.mode == 'ss_gc':
        
        dataset_train = GlobalCDataset(cfg.train_data_dir ,cfg)
        dataset_val = GlobalCDataset(cfg.val_data_dir, cfg)
        
    elif cfg.mode == 'ss_nc' or cfg.mode == 'ss_nc_dc' or cfg.mode == 'ss_dc': 
        
        dataset_train = NucleiCDataset(cfg.train_data_dir, cfg)
        dataset_val = NucleiCDataset(cfg.val_data_dir, cfg)
        
    elif cfg.mode == 'sl':
        
        dataset_train = TrainDataset1(cfg.train_data_dir, cfg)
        dataset_val = ValDataset1(cfg.val_data_dir, cfg)
        
    else:
        print("No such mode!!!")
    return dataset_train, dataset_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--seeds",
        default="10",
        help="random seeds"
    )
    parser.add_argument(
        "--gpus",
        default="4, 5",
        help="gpus to use, 0,1,2,3"
    )
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    ## check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    
    # 设置随机数种子
    setup_seed(int(args.seeds))
    
    ## Parameters
        
    checkpoint_path = "./checkpoints/%s_%s_%s.pth" % (cfg.model_name, cfg.mode, cfg.data_inst)
        
    print("model will be save to %s" % checkpoint_path)
    
    
    rm_n_mkdir('./logs_new/%s_%s_%s_%s/' % (cfg.model_name, cfg.mode, cfg.data_inst, args.seeds))
    writer = SummaryWriter('./logs_new/%s_%s_%s_%s/' % (cfg.model_name, cfg.mode, cfg.data_inst, args.seeds))
    print("log dir is set to ./logs_new/%s_%s_%s_%s/" % (cfg.model_name, cfg.mode, cfg.data_inst, args.seeds))

    best_loss = 1e10
    
    ## build dataset
    dataset_train, dataset_val = build_dataset()

    dataloaders = {
      'train': DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, pin_memory=True),
      'val': DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, 
                        num_workers=cfg.num_workers, pin_memory=True)
    }
    
    ## build models
    model = build_model().cuda()
        
    summary(model, input_size=(3, cfg.imgMaxSize, cfg.imgMaxSize))
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    
#     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    if cfg.pretrained_path != "":
        checkpoint = torch.load(cfg.pretrained_path)
        print(model.load_state_dict(checkpoint, strict=False))
        model.load_state_dict(checkpoint, strict=False)
        print("load pretrained weights from %s" % cfg.pretrained_path)
        
    model = nn.DataParallel(model)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs - 1))

        since = time.time()
        
        if cfg.mode == 'sl':
            train_phase(model, optimizer, dataloaders['train'], epoch)
            best_loss = test_phase(model, dataloaders['val'], best_loss, epoch)
        else:
            print("No such mode!!!")
            break
            
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        print(f"saving current model to {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)
        
        