import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.distributions import Categorical

import os
import copy
import time
import random
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from dataloader.nba import nba_pipeline
from models.models import SInAR_Net
from util.utils import *
from dataloader.dataloader import read_dataset
from itertools import cycle, islice
from info_nce import InfoNCE
import itertools

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nvidia.dali.plugin.pytorch import LastBatchPolicy, DALIGenericIterator

parser = argparse.ArgumentParser(description='Detector-Free Weakly Supervised Group Activity Recognition')

# Dataset specification
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')
parser.add_argument('--data_path', default='./Dataset/', type=str, help='data path')
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=18, type=int, help='number of frames for each clip')
parser.add_argument('--num_total_frame', default=72, type=int, help='number of total frames for each clip')
parser.add_argument('--num_activities', default=6, type=int, help='number of activity classes in volleyball dataset')

# Model parameters
parser.add_argument('--base_model', action='store_true', help='average pooling base model')
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')
parser.add_argument('--dilation', action='store_true', help='use dilation or not')
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

# Motion parameters
parser.add_argument('--motion', action='store_true', help='use motion feature computation')
parser.add_argument('--multi_corr', action='store_true', help='motion correlation block at 4th and 5th')
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')

# Transformer parameters
parser.add_argument('--nheads', default=4, type=int, help='number of heads')
parser.add_argument('--enc_layers', default=6, type=int, help='number of encoder layers')
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_tokens', default=12, type=int, help='number of queries')

# Aggregation parameters
parser.add_argument('--nheads_agg', default=4, type=int, help='number of heads for partial context aggregation')

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=30, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=2, type=int, help='print frequency')
parser.add_argument('--batch', default=4, type=int, help='Batch size')
parser.add_argument('--test_batch', default=4, type=int, help='Test batch size')
parser.add_argument('--lr', default=1e-6, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=1e-4, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

# GPU
parser.add_argument('--device', default="0, 1", type=str, help='GPU device')

# Load model
parser.add_argument('--load_model', action='store_true', help='load model')
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

parser.add_argument('--truncate', action='store_true', help='truncate dataset')
parser.add_argument('--modality', default="RGB", type=str, help='unimodal model modality')
parser.add_argument('--enable_dali', action='store_true', help='enable DALI flag')
parser.add_argument('--wATT', action='store_true', help='use wATT model')
parser.add_argument('--fp16-mode', default=False, action='store_true',
                    help='Enable half precision mode.')
parser.add_argument('--ssup_mode', default=False, action='store_true',
                    help='Self-supervision mode')
parser.add_argument('--flow_path', default=None,
                    help='Path to auxilliary flow folder')
parser.add_argument('--aux_flow', default=False, action='store_true',
                    help='Path to auxilliary flow folder')
parser.add_argument('--scale_pool_size', default=2, type=int, help='multi level tf scale size')
parser.add_argument('--num_scale_modules', default=3, type=int, help='number of transformer scales')
parser.add_argument('--num_workers', default=4, type=int,
                    help='dataloader num workers')
parser.add_argument('--bbox_loss_coef', default=1.0, type=float)
parser.add_argument('--giou_loss_coef', default=1.0, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float)
parser.add_argument('--load_bb', action='store_true')
parser.add_argument('--load_text', action='store_true')


# Create separate file to load up args
args = parser.parse_args()
best_mca = 0.0
best_mpca = 0.0
best_mca_epoch = 0
best_mpca_epoch = 0


def main():
    global args
    args.set_cost_class = 1
    args.set_cost_bbox = 5
    args.set_cost_giou = 2
    args.flow_path = None
    args.class_head = False

    args.distributed = True

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_DFGAR_<%s>' % (args.dataset, time_str)
    save_path = './result/%s' % exp_name

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.lr = args.lr * args.world_size
    args.max_lr = args.max_lr * args.world_size
    img_size = (720,1280)
    dataset = args.dataset
    train_set, test_set = read_dataset(args)
    flip_transform = True
    transform = True
    load_track = False
    
    load_bb = args.load_bb
    if args.dataset == 'Volleyball':
        load_bb = False

    train_pipe = nba_pipeline(iterator_source=train_set, transform=transform, img_size=img_size, sequence_length=args.num_frame, load_track=load_track, load_depth=False, load_flow=False, load_bb=load_bb, load_embd = True,  flip_transform=flip_transform,
        batch_size=args.batch, num_threads=args.num_workers, prefetch_queue_depth=2, 
        device_id=args.local_rank,shard_id=args.local_rank,num_shards=args.world_size,dataset=dataset, num_class = args.num_activities)
    train_pipe.build()
    pipe_output = ['images', 'embd_caption']
    train_pipe_output = pipe_output + ['label']
    
    train_loader = DALIGenericIterator(train_pipe, train_pipe_output,
                last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    
    test_pipe = nba_pipeline(iterator_source=test_set, transform=False, img_size=img_size, sequence_length=args.num_frame, load_track=False, load_depth=False, load_flow=False, load_bb=load_bb, load_embd = True,
    batch_size=args.test_batch, num_threads=args.num_workers, prefetch_queue_depth=2, 
    device_id=args.local_rank,shard_id=args.local_rank,num_shards=args.world_size,dataset=dataset, num_class = args.num_activities)
    test_pipe.build()
    test_pipe_output = pipe_output + ['label']
    test_loader = DALIGenericIterator(test_pipe, test_pipe_output,
                last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = SInAR_Net(args)
    
    model=model.to(device=device)  
    if args.distributed:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
         
        torch.cuda.current_stream().wait_stream(s)
        model_without_ddp = model.module
    else:
        model = torch.nn.DataParallel(model).cuda()

    
    # get the number of model parameters
    if args.local_rank == 0:            
        parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
        
        dict_num_param = {}
        for name_param,param in model.named_parameters():
            name_short = name_param.split('.')[1]
            try:
                dict_num_param[name_short] += param.data.nelement()
            except:
                dict_num_param[name_short] = param.data.nelement()
        
        print_log(save_path, '--------------------Number of parameters--------------------')
        print_log(save_path, dict_num_param)        
        print_log(save_path, parameters)
        print_log(save_path, f'Modality: {args.modality}, epochs: {args.epochs}')

    # define loss function and optimizer
    #model_detr, criterion, postprocessors = build_model(args)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.max_lr, step_size_up=args.lr_step, step_size_down=args.lr_step_down, mode='triangular2', cycle_momentum=False)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16_mode)
    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    label_embed = load_extended_label(args)
    # training phase
    for epoch in range(start_epoch, args.epochs + 1):
        if args.local_rank==0:
            print_log(save_path, '----- %s at epoch #%d' % ("Train", epoch))
        
        train_log = train(train_loader, model, optimizer, scaler, epoch, data_iter_size= train_set.full_iterations, device=device, label_embed = label_embed)
        if args.local_rank==0:
            print_log(save_path, 'Accuracy: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                    (train_log['group_acc'], train_log['loss'], train_log['time']))
            print('Current learning rate is %f' % scheduler.get_last_lr()[0])
        
        if epoch % args.test_freq == 0:
            if args.local_rank==0: 
                print_log(save_path, '----- %s at epoch #%d' % ("Test", epoch))
            test_log = validate(test_loader, model, epoch, data_iter_size= test_set.full_iterations)
            if args.local_rank==0: 
                print_log(save_path, 'Accuracy: %.2f%%, Mean-ACC: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                      (test_log['group_acc'], test_log['mean_acc'], test_log['loss'], test_log['time']))
                print_log(save_path, '----------Best MCA: %.2f%% at epoch #%d.' %
                        (test_log['best_mca'], test_log['best_mca_epoch']))
                print_log(save_path, '----------Best MPCA: %.2f%% at epoch #%d.' %
                        (test_log['best_mpca'], test_log['best_mpca_epoch']))

                if epoch == test_log['best_mca_epoch'] or epoch == test_log['best_mpca_epoch']:
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    result_path = save_path + '/epoch%d_%.2f%%.pth' % (epoch, test_log['group_acc'])
                    torch.save(state, result_path)

        scheduler.step()


def train(train_loader, model, optimizer, scaler, epoch, data_iter_size, device, text_labels=None, label_embed = None):
    """Train for one epoch on the training set"""
    epoch_timer = Timer()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()
    criterion_l1 = torch.nn.L1Loss()
    criterion_l2 = torch.nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss().cuda()        
    criterion_cons_paired = InfoNCE(negative_mode='paired')
    criterion_cons = InfoNCE()

    disable_tqdm=False
    if args.local_rank != 0:
        disable_tqdm=True

    act_range = torch.arange(0,args.num_activities,dtype=int)
    mat_neg = []
    for id_act in range(args.num_activities):
        mat_neg.append(torch.cat([act_range[0:id_act], act_range[id_act+1:]]))
    mat_neg = torch.stack(mat_neg).cuda()

    for i, data in enumerate(tqdm(train_loader, disable=disable_tqdm,total=data_iter_size)):
        for d in data:
            images = d["images"].cuda(non_blocking=True)                              # [B, T, 3, H, W]
            b, t, _, _H, _W = images.shape
            activities = d["label"].cuda(non_blocking=True)  
            embd_caption = d["embd_caption"]
            # Take out the first caption (contain label)
            if args.dataset == 'Volleyball':
                len_get = 3
            elif args.dataset == 'nba':
                embd_caption = embd_caption[:,1:]
                len_capt = embd_caption.shape[1]
                len_get = 12
            
            idx_get = torch.randperm(len_capt)[:len_get]
            embd_caption = embd_caption[:,idx_get]
            
        # compute output
        num_batch = images.shape[0]
        num_frame = images.shape[1]
        activities_in = activities.reshape((num_batch, ))
        with torch.cuda.amp.autocast(enabled=args.fp16_mode):
            output = model(images, caption_feat = embd_caption, label_embed = label_embed)
            score = output['score']
            
            # calculate loss
            loss_score = criterion_ce(score, activities_in)
            loss = loss_score
            
            score_frame = output['score_frame']
            loss_frame = criterion_ce(score_frame, activities_in.repeat_interleave(score_frame.shape[0]//b,0))
            loss_frame = torch.mean(loss_frame)
            loss = loss + loss_frame
            
            loss_token = criterion_l2(output['rec_token'],output['caption_feat'])
            loss = loss + loss_token

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        losses.update(loss, num_batch)

        # measure accuracy and record loss
        group_acc = accuracy(score, activities_in)
        accuracies.update(group_acc, num_batch)


    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': accuracies.avg * 100.0,
    }

    return train_log


@torch.no_grad()
def validate(test_loader, model, epoch, data_iter_size, text_labels=None):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    epoch_timer = Timer()
    losses = AverageMeter()
    accuracies = AverageMeter()
    true = []
    pred = []
    criterion_ce = nn.CrossEntropyLoss().cuda()
    # switch to eval mode
    model.eval()
    disable_tqdm=False
    if args.local_rank != 0:
        disable_tqdm=True
    
    for i, data in enumerate(tqdm(test_loader, disable=disable_tqdm,total=data_iter_size)):
        for d in data:
            images = d["images"].cuda(non_blocking=True)
            activities =  d["label"].cuda()   
            embd_caption = d["embd_caption"]
            embd_caption = embd_caption[:,1:]
            
        num_batch = images.shape[0]
        num_frame = images.shape[1]
        b, t, _, _H, _W = images.shape
        activities_in = activities.reshape((num_batch,))

        with torch.cuda.amp.autocast(enabled=args.fp16_mode):    
            output = model(images)
            
            score = output['score']
            loss_score = criterion_ce(score, activities_in)
            loss = loss_score
            score_frame = output['score_frame']
            loss_frame = criterion_ce(score_frame, activities_in.repeat_interleave(score_frame.shape[0]//b,0))
            loss_frame = torch.mean(loss_frame)
            loss = loss + loss_frame

        true = true + activities_in.tolist()
        pred = pred + torch.argmax(score, dim=1).tolist()

        # measure accuracy and record loss
        group_acc = accuracy(score, activities_in)
        if args.distributed:
            torch.distributed.barrier()
            reduced_loss = reduce_tensor(loss.data)
            reduced_acc = reduce_tensor(group_acc)
            
            losses.update(reduced_loss.item(), images.size(0))
            accuracies.update(reduced_acc.item(), images.size(0))
     
        else:
            losses.update(loss, num_batch)
            accuracies.update(group_acc, num_batch)
            
    acc = accuracies.avg * 100.0
    rank = dist.get_rank()
    all_true = {rank: true}
    all_pred = {rank: pred}
    output_list_true = [torch.zeros_like(torch.empty(1)) for _ in range(dist.get_world_size())]
    output_list_pred = [torch.zeros_like(torch.empty(1)) for _ in range(dist.get_world_size())]
    dist.all_gather_object(output_list_true, all_true)
    dist.all_gather_object(output_list_pred, all_pred)
    output_list_true = list(itertools.chain.from_iterable([list(i.values())[0] for i in output_list_true]))
    
    output_list_pred = list(itertools.chain.from_iterable([list(i.values())[0] for i in output_list_pred]))

    confusion = confusion_matrix(output_list_true, output_list_pred)
    mean_acc = np.mean([confusion[i, i] / confusion[i, :].sum() for i in range(confusion.shape[0])]) * 100.0
    if rank==0:
        print(f"Total pred length: {len(output_list_pred)}")

    if acc > best_mca:
        best_mca = acc
        best_mca_epoch = epoch
    if mean_acc > best_mpca:
        best_mpca = mean_acc
        best_mpca_epoch = epoch

    test_log = {
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': acc,
        'mean_acc': mean_acc,
        'best_mca': best_mca,
        'best_mpca': best_mpca,
        'best_mca_epoch': best_mca_epoch,
        'best_mpca_epoch': best_mpca_epoch,
    }

    return test_log


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


def accuracy(output, target):
    output = torch.argmax(output, dim=1)
    correct = torch.sum(torch.eq(target.int(), output.int())).float()
    return correct / output.shape[0]
    #return correct.item() / output.shape[0]

def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size
    return rt


def load_extended_label(args):
    if args.dataset == 'nba':
        data_path = args.data_path + 'NBA_dataset'
        label_embed = os.path.join(data_path,'label_embed.npy')
        label_embed = torch.tensor(np.load(label_embed)).cuda() #[N_caption, N_class, D]
    else:
        label_embed = None

    return label_embed

if __name__ == '__main__':
    main()
