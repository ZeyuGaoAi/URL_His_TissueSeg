import os
import argparse
import warnings
import torch
import torch.nn as nn
from torchsummary import summary
import time
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from loss import calc_sl_loss, calc_sl_loss_val, calc_gc_loss, calc_nc_loss, calc_dc_loss, print_metrics, calc_dc_loss_sp, calc_dc_loss_spp, calc_dc_loss_pix

from utils import rm_n_mkdir
from models import ResNetBackBone, ResNetUNet, ResNetUNetHead, ResNetUNetHeadOneStage
from datasetBRS import TrainDataset, ValDataset, GlobalCDataset, NucleiCDataset
from config import _C as cfg

import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# warnings.filterwarnings('ignore')
# python -m torch.distributed.launch --nproc_per_node=8 --master_port 120518 main-distribute.py --gpus=0,1,2,3,4,5,6,7

tissue_type_dict = {
            'Tumor': 0, # ! Please ensure the matching ID is unique
            'Stroma': 1,
            'Inflammatory' : 2,
            'Necrosis': 3,
            'Other': 4,
        }

def train_onestage_phase(model, optimizer, dataloader, epoch, writer):
    model.train()  # Set model to training mode
    metrics = defaultdict(float)
    epoch_samples = 0
    index = 0
    with tqdm(total=cfg.iter_per_epoch) as pbar:
        
        for data in dataloader:

            inputs1, inputs2, sps1, sps2 = data

            index += 1

            inputs1 = torch.stack(inputs1).cuda()
            inputs2 = torch.stack(inputs2).cuda()
            sps1 = torch.stack(sps1).cuda()
            sps2 = torch.stack(sps2).cuda() # s*b, w, h

            s, b, c, w, h = inputs1.shape

            inputs1 = inputs1.view(s*b, c, w, h)
            inputs2 = inputs2.view(s*b, c, w, h)

            outputs1, outputs1e = model(inputs1)
            outputs2, outputs2e = model(inputs2)

            _, oc, w, h = outputs1.shape

            outputs1 = outputs1.view(s, b, oc, w, h)
            outputs2 = outputs2.view(s, b, oc, w, h)
            
            _, oce = outputs1e.shape
            
            outputs1e = outputs1e.view(s, b, oce)
            outputs2e = outputs2e.view(s, b, oce)

            loss1 = calc_nc_loss([outputs1e, outputs2e], metrics, cfg)
            loss2 = calc_dc_loss_spp([outputs1, outputs2], [sps1, sps2], metrics, cfg)

            loss = loss1 + loss2
            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_samples += inputs1.size(0)

            pbar.update(1)

            if index > cfg.iter_per_epoch:
                break
            
#             if index % cfg.iter_per_epoch == 0:
#                 print_metrics(metrics, epoch_samples, 'train')
#                 epoch_loss = metrics['loss_dc'] / epoch_samples
            
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = (metrics['loss_dc'] + metrics['loss_nc']) / epoch_samples
    if writer is not None:
        for k in metrics.keys():
            writer.add_scalar('train/%s' % k, metrics[k] / epoch_samples, epoch)

    scheduler.step(epoch_loss)

def train_decoder_phase(model, optimizer, dataloader, epoch, writer):
    model.train()  # Set model to training mode
    metrics = defaultdict(float)
    epoch_samples = 0
    index = 0
    with tqdm(total=cfg.iter_per_epoch) as pbar:
        
        for data in dataloader:

            inputs1, inputs2, sps1, sps2 = data

            index += 1

            inputs1 = torch.stack(inputs1).cuda()
            inputs2 = torch.stack(inputs2).cuda()
            sps1 = torch.stack(sps1).cuda()
            sps2 = torch.stack(sps2).cuda() # s*b, w, h

            s, b, c, w, h = inputs1.shape

            inputs1 = inputs1.view(s*b, c, w, h)
            inputs2 = inputs2.view(s*b, c, w, h)

            outputs1 = model(inputs1)
            outputs2 = model(inputs2)

            _, oc, w, h = outputs1.shape

            outputs1 = outputs1.view(s, b, oc, w, h)
            outputs2 = outputs2.view(s, b, oc, w, h)

            loss = calc_dc_loss_spp([outputs1, outputs2], [sps1, sps2], metrics, cfg)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_samples += inputs1.size(0)

            pbar.update(1)

            if index > cfg.iter_per_epoch:
                break
            
#             if index % cfg.iter_per_epoch == 0:
#                 print_metrics(metrics, epoch_samples, 'train')
#                 epoch_loss = metrics['loss_dc'] / epoch_samples
            
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss_dc'] / epoch_samples
    if writer is not None:
        for k in metrics.keys():
            writer.add_scalar('train/%s' % k, metrics[k] / epoch_samples, epoch)

    scheduler.step(epoch_loss)

def train_nucleic_phase(model, optimizer, dataloader, epoch, writer):
    model.train()  # Set model to training mode
    metrics = defaultdict(float)
    epoch_samples = 0
    optimizer.zero_grad()
    index = 0
    with tqdm(total=cfg.iter_per_epoch) as pbar:
        for data in dataloader:

            inputs1, inputs2, _, _ = data

            index += 1
            inputs1 = torch.stack(inputs1).cuda()
            inputs2 = torch.stack(inputs2).cuda()

            s, b, c, w, h = inputs1.shape

            inputs1 = inputs1.view(s*b, c, w, h)
            inputs2 = inputs2.view(s*b, c, w, h)

            outputs1 = model(inputs1)
            outputs2 = model(inputs2)

            _, oc = outputs1.shape

            outputs1 = outputs1.view(s, b, oc)
            outputs2 = outputs2.view(s, b, oc)

            loss = calc_nc_loss([outputs1, outputs2], metrics, cfg)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_samples += inputs1.size(0)
            
            pbar.update(1)

            if index > cfg.iter_per_epoch:
                break
            
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss_nc'] / epoch_samples
    if writer is not None:
        for k in metrics.keys():
            writer.add_scalar('train/%s' % k, metrics[k] / epoch_samples, epoch)

    scheduler.step(epoch_loss)
    

def train_globalc_phase(model, optimizer, dataloader, epoch, writer):
    model.train()  # Set model to training mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs1, inputs2 in tqdm(dataloader):
        inputs1 = inputs1.cuda()
        inputs2 = inputs2.cuda()
        
        inputs = torch.cat((inputs1, inputs2), 0)
        
        outputs = model(inputs)

        loss = calc_gc_loss(outputs, metrics, cfg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_samples += inputs1.size(0)
        
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss_gc'] / epoch_samples
    
    for k in metrics.keys():
        writer.add_scalar('train/%s' % k, metrics[k] / epoch_samples, epoch)

#     scheduler.step(epoch_loss)
    

def train_phase(model, optimizer, dataloader, epoch, writer):
    model.train()  # Set model to training mode
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        outputs, features = model(inputs)

        loss = calc_sl_loss(outputs, labels, metrics)

        optimizer.zero_grad()
        
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        optimizer.step()
        
        epoch_samples += inputs.size(0)
        
    print_metrics(metrics, epoch_samples, 'train')
    epoch_loss = metrics['loss'] / epoch_samples
    
    for k in metrics.keys():
        writer.add_scalar('train/%s' % k, metrics[k] / epoch_samples, epoch)

    scheduler.step(epoch_loss)
#     for param_group in optimizer.param_groups:
#         print("LR", param_group['lr'])
            
def test_phase(model, dataloader, best_loss, epoch, writer):
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
        model = ResNetUNetHeadOneStage(freeze=cfg.freeze, pretrained=cfg.pretrained)
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
        
        dataset_train = TrainDataset(cfg.train_data_dir, cfg)
        dataset_val = ValDataset(cfg.val_data_dir, cfg)
        
    else:
        print("No such mode!!!")
    return dataset_train, dataset_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--gpus",
        default="6, 7",
        help="gpus to use, 0,1,2,3"
    )
    parser.add_argument(
        '--local_rank', 
        default=-1, 
        type=int,
        help='node rank for distributed training')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    ## check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    
    ## data parallels
    dist.init_process_group(backend='nccl')
    
    local_rank = args.local_rank
    
    torch.cuda.set_device(local_rank)
    
    ## Parameters
    
    if dist.get_rank() == 0:
        checkpoint_path = "./checkpoints/%s_%s_%s.pth" % (cfg.model_name, cfg.mode, cfg.data_inst)
        print("model will be save to %s" % checkpoint_path)
        
        rm_n_mkdir('./logs/%s_%s_%s/' % (cfg.model_name, cfg.mode, cfg.data_inst))
        writer = SummaryWriter('./logs/%s_%s_%s/' % (cfg.model_name, cfg.mode, cfg.data_inst))
        print("log dir is set to ./logs/%s_%s_%s/" % (cfg.model_name, cfg.mode, cfg.data_inst))
    else:
        checkpoint_path = None
        writer = None

    best_loss = 1e10
    
    ## build dataset
    dataset_train, dataset_val = build_dataset()
    
#     gpus_num = torch.cuda.device_count()
    
    train_sampler = DistributedSampler(dataset_train, shuffle=True)
    
    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, sampler=train_sampler, 
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    ## build models
    model = build_model().cuda()
        
#     summary(model, input_size=(3, cfg.imgMaxSize, cfg.imgMaxSize))
        
    if cfg.pretrained_path != "":
        checkpoint = torch.load(cfg.pretrained_path)
        model.load_state_dict(checkpoint, strict=False)
        print("load pretrained weights from %s" % cfg.pretrained_path)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
#     model = nn.DataParallel(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs - 1))
        
        train_sampler.set_epoch(epoch)
        since = time.time()
        
        if cfg.mode == 'ss_gc':
            train_globalc_phase(model, optimizer, train_loader, epoch, writer)
        elif cfg.mode == 'ss_nc':
            train_nucleic_phase(model, optimizer, train_loader, epoch, writer)
        elif cfg.mode == 'ss_dc':
            train_decoder_phase(model, optimizer, train_loader, epoch, writer)
        elif cfg.mode == 'ss_nc_dc':
            train_onestage_phase(model, optimizer, train_loader, epoch, writer)
        elif cfg.mode == 'sl':
            train_phase(model, optimizer, train_loader, epoch, writer)
            best_loss = test_phase(model, val_loader, best_loss, epoch, writer)
        else:
            print("No such mode!!!")
            break
            
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if dist.get_rank() == 0:
            print(f"saving current model to {checkpoint_path}")
            torch.save(model.module.state_dict(), checkpoint_path)
        
        
