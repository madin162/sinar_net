import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image
try:
    import nvidia.dali.fn as dalifn
    import nvidia.dali.math as dalimath
    from nvidia.dali import pipeline_def
    import nvidia.dali.types as types

except ModuleNotFoundError:
    # Error handling
    pass

from random import shuffle
import pickle
import json
import ast
import util.transforms as T
from util.box_ops import box_xyxy_to_cxcywh

ACTIVITIES = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']

CLASSES = {0:'basketball scored two-pointer', 1:'basketball failed two-pointer with offensive rebound', 2:'basketball failed two-pointer with defensive rebound', 3:'basketball scored layup', 4:'basketball failed layup with offensive rebound', 5:'basketball failed layup with defensive rebound', 6:'basketball scored three-pointer', 7:'basketball failed three-pointer with offensive rebound', 8:'basketball failed three-pointer with defensive rebound'}

CLASSES_2 = {
    0: "Basketball Ring",
    1: "Basketball",
    2: "Court",
    3: "Player",
    4: "Team",
    5: "Dribble",
    6: "Pass",
    7: "Shoot",
    8: "Hoop",
    9: "Game",
    10: "Dunk",
    11: "Layup",
    12: "Rebound",
    13: "Defense",
    14: "Offense",
    15: "Coach",
    16: "Referee",
    17: "Uniform",
    18: "Spectator",
    19: "Arena",
    20: "Scoreboard",
    21: "Free throw",
    22: "Three-pointer",
    23: "Foul",
    24: "Fast break",
    25: "Tip-off",
    26: "Block",
    27: "Steal",
    28: "Assist",
    29: "Crossover",
    30: "Post-up",
    31: "Backboard",
    32: "Timeout",
    33: "Substitution",
    34: "Dribbling",
    35: "Jump shot",
    36: "Alley-oop",
    37: "Baseline",
    38: "Perimeter",
    39: "Paint",
    40: "Slam dunk",
    41: "Inbound",
    42: "Possession",
    43: "Quarter",
    44: "Half-time",
    45: "Shot clock",
    46: "Playoff",
    47: "Championship",
    48: "MVP",
    49: "Bench",
    50: "Buzzer-beater"
}


def read_ids(path):
    file = open(path)
    values = file.readline()
    values = values.split(',')[:-1]
    values = list(map(int, values))

    return values


def nba_read_annotations(path, seqs):
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:
            for line in f.readlines():
                values = line[:-1].split('\t')
                file_name = values[0]
                fid = int(file_name.split('.')[0])

                activity = group_to_id[values[1]]

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations
        
    return labels


def nba_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames

def manual_jitter(img_rgb,coef_jitter):
    img_rgb = transforms.functional.adjust_brightness(img_rgb,coef_jitter[0])
    img_rgb = transforms.functional.adjust_contrast(img_rgb,coef_jitter[1])
    img_rgb = transforms.functional.adjust_saturation(img_rgb,coef_jitter[2])
    img_rgb = transforms.functional.adjust_hue(img_rgb,coef_jitter[3])
    return img_rgb

class NBADataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True,
                 track_path = None, caption_path = None):
        super(NBADataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        if args.image_height == args.image_width:
            self.resize_ops = [transforms.Resize((args.image_height))]
            self.resize_ops = self.resize_ops+[transforms.CenterCrop(size=args.image_height)]
        else:
            self.resize_ops = [transforms.Resize((args.image_height, args.image_width))]
            
        self.additional_ops_list = [transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
        self.transform = transforms.Compose(self.resize_ops+self.additional_ops_list)
        self.additional_ops = transforms.Compose(self.additional_ops_list)
        self.flow_ops = transforms.Compose([
            transforms.Resize((23,40)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
        self.track_path = track_path
        if self.track_path is not None:
            with open(self.track_path, 'rb') as pickle_file:
                self.tracks_load = pickle.load(pickle_file)
        
        self.flow_path = args.flow_path
            
        self.num_bb = args.num_tokens
        self.classes = CLASSES
        self.load_bb = args.load_bb
        self.load_text = args.load_text
        self.caption_path = caption_path

    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)

        return samples

    def __len__(self):
        return len(self.frames)

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        vid, sid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(72), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
        else:
            if self.num_frame == 6:
                # [6, 18, 30, 42, 54, 66]
                sample_frames = list(range(6, 72, 12))
            elif self.num_frame == 12:
                # [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]
                sample_frames = list(range(4, 72, 6))
            elif self.num_frame == 18:
                # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
                sample_frames = list(range(2, 72, 4))
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + segment_duration // 2

        return [(vid, sid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        data_dict = {}
        images, activities = [], []
        images_adr = []
        tracks=[]
        flows = []
        flows_adr = []
        list_boxes = []
        len_boxes = []
        b = float(torch.empty(1).uniform_(0.6, 1.4)) #[max(0, 1 - brightness), 1 + brightness]
        c = float(torch.empty(1).uniform_(0.6, 1.4)) #[max(0, 1 - contrast), 1 + contrast]
        s = float(torch.empty(1).uniform_(0.6, 1.4)) # [max(0, 1 - saturation), 1 + saturation]
        h = float(torch.empty(1).uniform_(-0.1, 0.1)) #[-hue, hue]
        for i, (vid, sid, fid) in enumerate(frames):
            fid = '{0:06d}'.format(fid)
            images_adr.append(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))
            img = Image.open(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))
            img = self.resize_ops[0](img)
            if self.is_training:
                img = manual_jitter(img,[b,c,s,h])
            img = self.additional_ops(img)

            images.append(img)
            activities.append(self.anns[vid][sid]['group_activity'])
            if self.track_path is not None:
                frame_tracks = self.tracks_load[(vid, sid)][int(fid)]
                frame_tracks = np.array(frame_tracks[:self.num_bb,0:4])
                frame_tracks = np.resize(frame_tracks,(self.num_bb,4))
                tracks.append(frame_tracks)

            if self.flow_path is not None:
                fid_flow = str(min(70,int(fid))).zfill(6)
                flow = Image.open((self.flow_path + '/%d/%d/%s.jpg' % (vid, sid, fid_flow)))
                flows_adr.append(self.flow_path + '/%d/%d/%s.jpg' % (vid, sid, fid_flow))
                flow = self.flow_ops(flow)
                flows.append(flow)

            if self.load_bb:
                max_boxes = 20
                # Load bounding box here
                bboxes_path = self.image_path + '/%d/%d/detr_bb.json' % (vid, sid)
                with open(bboxes_path, 'r') as f:
                    data = json.load(f)
                
                try:
                    labels, boxes = zip(*[key_data.split('|') for key_data in data[fid] if key_data.split('|')[0] == 'person'])
                    boxes = np.array([np.multiply(np.array(ast.literal_eval(box)),np.array([1280,720,1280,720])) for box in boxes])
                    # guard against no boxes via resizing
                    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                    
                    # left, upper, right, bottom
                    img_w = self.image_size[0]
                    img_h = self.image_size[1]
                    boxes[:, 0::2].clamp_(min=0, max=img_w)
                    boxes[:, 1::2].clamp_(min=0, max=img_h)
                    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                    boxes = boxes[keep]
                    boxes = box_xyxy_to_cxcywh(boxes)
                    boxes = boxes / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
                    boxes = boxes[:max_boxes]

                except:
                    boxes = []

                list_boxes.append(boxes)
                len_boxes.append(len(boxes))

        images = torch.stack(images)
        activities = np.array(activities, dtype=np.int32)

        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()
        if self.load_bb:
            # list_boxes = T, num_boxes, 4
            all_boxes = torch.zeros((len(frames),max_boxes,4))
            for box_id,boxes in enumerate(list_boxes):
                if len_boxes[box_id]>0:
                    all_boxes[box_id,:len_boxes[box_id]]=boxes
                
            data_dict['boxes'] = all_boxes
            data_dict['len_boxes'] = torch.tensor(len_boxes)
        
        if self.load_text:
            caption_adr = self.caption_path + '/%d/%d/embd_caption.npy' % (vid, sid)
            embd_caption = np.load(caption_adr)

            data_dict['embd_caption'] = torch.tensor(embd_caption)

        data_dict['label'] =  activities
        if self.is_training:
            data_dict['images'] = images
            data_dict['images_adr'] = images_adr
            if self.flow_path:
                flows = torch.stack(flows)
                data_dict['flows'] = flows
                data_dict['flows_adr'] = flows_adr

            return data_dict
        
        data_dict['images'] = images
        data_dict['images_adr'] = images_adr
        data_dict['label'] =  activities
        return data_dict

class NBAInputCallable(object):
    def __init__(self, batch_size, frames, anns, image_path, args,
        is_training=True,track_path = None, depth_path = None, caption_path = None, shard_id = 0, num_shards=1):
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.depth_path = depth_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.track_path = track_path
        
        if self.track_path is not None:
            with open(self.track_path, 'rb') as pickle_file:
                self.tracks_load = pickle.load(pickle_file)
        self.num_bb = args.num_tokens #the number of bounding boxes N = 12 (ref. NBA paper supp material)
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards #num_shards = num_gpu
        # If the dataset size is not divisibvle by number of shards, the trailing samples will
        # be omitted.
        self.shard_size = len(self.frames) // num_shards
        self.shard_offset = self.shard_size * shard_id
        # If the shard size is not divisible by the batch size, the last incomplete batch
        # will be omitted.
        self.full_iterations = self.shard_size // batch_size
        self.perm = None  # permutation of indices
        self.last_seen_epoch = None  # so that we don't have to recompute the `self.perm` for every sample
        #print(f"info dataset: {len(self.frames)} {self.shard_size} {self.full_iterations}")
        self.flow_path = args.flow_path
        self.args = args
        self.classes = CLASSES
        self.load_bb = args.load_bb
        self.load_text = args.load_text
        self.caption_path = caption_path

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        vid, sid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(self.num_total_frame), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(segment_duration, size=self.num_frame)
        else:
            if self.num_frame == 6:
                # [6, 18, 30, 42, 54, 66]
                sample_frames = list(range(6, 72, 12))
            elif self.num_frame == 12:
                # [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]
                sample_frames = list(range(4, 72, 6))
            elif self.num_frame == 18:
                # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
                sample_frames = list(range(2, 72, 4))
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + segment_duration // 2

        return [(vid, sid, fid) for fid in sample_frames]

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        
        if self.is_training:
            if self.last_seen_epoch != sample_info.epoch_idx:
                self.last_seen_epoch = sample_info.epoch_idx
                self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx).permutation(len(self.frames))
                #print(f"updating perm {self.last_seen_epoch} {sample_info.idx_in_epoch}")
            sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        else:
            sample_idx = sample_info.idx_in_epoch + self.shard_offset

        frames = self.select_frames(self.frames[sample_idx])
        samples = self.load_samples(frames)
        
        return samples

    def load_samples(self, frames):
        images = []
        depths=[]
        if self.track_path is not None:
            tracks = []
        
        images=[]
        tracks=[]
        depths=[]
        flows=[]
        tracks_depth=[]
        list_boxes = []
        len_boxes = []
        for i, (vid, sid, fid) in enumerate(frames):
            fid = '{0:06d}'.format(fid)
            img = self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid)
            f = open(img, "rb")
            img = np.frombuffer(f.read(), dtype=np.uint8) # don't decode the image just read the bytes!
            images.append(img)
            if self.track_path is not None:
                frame_tracks = self.tracks_load[(vid, sid)][int(fid)]
                frame_tracks = np.array(frame_tracks[:self.num_bb,0:4])
                frame_tracks = np.resize(frame_tracks,(self.num_bb,4))
                tracks.append(frame_tracks)

            if self.depth_path is not None:
                depth = self.depth_path + '/%d/%d/%s.jpg' % (vid, sid, fid)
                f = open(depth, "rb")
                depth = np.frombuffer(f.read(), dtype=np.uint8) # don't decode the image just read the bytes!
                depths.append(depth)
            
            if self.is_training:
                if self.flow_path is not None:
                    fid_flow = str(min(70,int(fid))).zfill(6)
                    flow = self.flow_path + '/%d/%d/%s.jpg' % (vid, sid, fid_flow)
                    f = open(flow, "rb")
                    flow = np.frombuffer(f.read(), dtype=np.uint8) # don't decode the image just read the bytes!
                    flows.append(flow)
            
            if ((self.depth_path is not None) & (self.track_path is not None)):
                frame_tracks =  np.array(self.tracks_load[(vid, sid)][int(fid)][:,0:4])
                x_min,y_min, _, _ = np.min(frame_tracks,0)
                _, _, x_max,y_max = np.max(frame_tracks,0)
                _W, _H = self.image_size
                x_min = x_min*_W; y_min = y_min*_H
                x_max = x_max*_W; y_max = y_max*_H
                x_min = x_min-20; x_max = x_max+20
                y_min = y_min-20; y_max = y_max+20
                if x_min < 0:
                    x_max = x_max-x_min
                    x_min = 0

                if y_min < 0:
                    y_max = y_max-y_min
                    y_min = 0

                # get w_crop and h_crop
                w_crop = x_max - x_min
                h_crop = y_max - y_min

                w_rat = _W/w_crop
                h_rat = _H/h_crop
                if w_rat<h_rat:
                    w_crop = w_crop
                    h_crop = int(w_crop * _H/_W) #_H/_W = 9/16
                    y_mid = y_min+(y_max - y_min)/2
                    y_min = y_mid - int(h_crop/2)
                    y_max = y_mid + int(h_crop/2)

                W_new = x_max - x_min
                H_new = y_max - y_min
                rat = (_W/W_new+_H/H_new)/2
                temp_boxes_adj = frame_tracks.copy()
                for i,tbox in enumerate(temp_boxes_adj):
                    x1, y1, x2, y2 = tbox
                    x1_adj = rat*(int(round(x1*_W))-x_min)/_W
                    y1_adj = rat*(int(round(y1*_H))-y_min)/_H
                    x2_adj = rat*(int(round(x2*_W))-x_min)/_W
                    y2_adj = rat*(int(round(y2*_H))-y_min)/_H
                    temp_boxes_adj[i] = np.array([x1_adj, y1_adj, x2_adj, y2_adj])
                
                temp_boxes_adj = np.array(temp_boxes_adj[:self.num_bb,0:4])
                temp_boxes_adj = np.resize(temp_boxes_adj,(self.num_bb,4))
                tracks_depth.append(temp_boxes_adj)

            if self.load_bb:
                max_boxes = 20
                # Load bounding box here
                bboxes_path = self.image_path + '/%d/%d/detr_bb.json' % (vid, sid)
                with open(bboxes_path, 'r') as f:
                    data = json.load(f)

                try:
                    labels, boxes = zip(*[key_data.split('|') for key_data in data[fid] if key_data.split('|')[0] == 'person'])
                    boxes = np.array([np.multiply(np.array(ast.literal_eval(box)),np.array([1280,720,1280,720])) for box in boxes])
                    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                    
                    # left, upper, right, bottom
                    img_w = self.image_size[0]
                    img_h = self.image_size[1]
                    boxes[:, 0::2].clamp_(min=0, max=img_w)
                    boxes[:, 1::2].clamp_(min=0, max=img_h)
                    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                    boxes = boxes[keep]
                    boxes = box_xyxy_to_cxcywh(boxes)
                    boxes = boxes / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
                    boxes = boxes[:max_boxes]

                except:
                    boxes = []

                list_boxes.append(boxes)
                len_boxes.append(len(boxes))

                
        return_var = images.copy()
        actv = self.anns[vid][sid]['group_activity']       
        actv = np.array(actv,dtype = np.uint8) 

        if self.track_path is not None:

            # Adjusted bboxes
            return_var.extend(tracks)
        
        if self.depth_path is not None:
            return_var.extend(depths)

        if ((self.depth_path is not None) & (self.track_path is not None)):
            return_var.extend(tracks_depth)
        
        if self.is_training:
            if self.args.aux_flow:
                return_var.extend(flows)

        if self.load_bb:
            # list_boxes = T, num_boxes, 4
            all_boxes = torch.zeros((len(frames),max_boxes,4))
            for box_id,boxes in enumerate(list_boxes):
                if len_boxes[box_id]>0:
                    all_boxes[box_id,:len_boxes[box_id]]=boxes
            
            return_var.extend(all_boxes)
            return_var.extend(torch.tensor(len_boxes))
        
        if self.load_text:
            caption_adr = self.caption_path + '/%d/%d/embd_caption.npy' % (vid, sid)
            embd_caption = np.load(caption_adr)

            return_var.append(torch.tensor(embd_caption))
        
        return_var.append(actv)
        return return_var

    def __len__(self):
        return len(self.frames)
    
 
@pipeline_def(num_threads=2, py_num_workers=2, py_start_method='spawn')
def nba_pipeline(iterator_source,sequence_length,transform=False, img_size=(720,1280), load_track=False, load_depth = False, load_flow = False, load_bb = False, load_embd = False, shard_id=None, num_shards=None, flip_transform = False,dataset='NBA', num_class = 8):

    num_data_type = 1
    if load_track:
        num_data_type += 1
    if load_depth:
        num_data_type += 1
        if load_track:
            num_data_type += 1
    if load_flow:
        num_data_type += 1
    if load_bb:
        num_data_type += 2
    length_data_load = sequence_length*num_data_type+1

    if load_embd:
        length_data_load = length_data_load + 1
    abc=dalifn.external_source(source=iterator_source, 
            num_outputs=length_data_load, batch=False, parallel=True)
    
    seq_lgth = sequence_length
    datas = abc[:seq_lgth]    
    jpegs =datas
    images = dalifn.decoders.image(jpegs, device="mixed", hw_decoder_load=1)
    mean_vals=[0.485 * 255, 0.456 * 255, 0.406 * 255]
    std_vals=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    
    images = dalifn.resize(images,device="gpu", resize_y=img_size[0], resize_x=img_size[1])
    rng_pos_x = 0.5
    rng_pos_y = 0.5
    flip_flag=0
    if transform:
        rng1 = dalifn.random.uniform(range=[0.5, 1])
        rng2 = dalifn.random.uniform(range=[0.5, 1.5])
        rng3 = dalifn.random.uniform(range=[-0.25, 0.25])*180
    
        images = dalifn.color_twist(images, device="gpu", saturation = rng1, contrast=rng2,hue=rng3)
    
        if img_size[0]==img_size[1]:
            rng_pos_x = np.random.uniform(0.0, 1.0) 
            rng_pos_y = np.random.uniform(0.0, 1.0) 

        if flip_transform == True:
            flip_flag = dalifn.random.coin_flip(probability=0.5) 

    images = dalifn.crop_mirror_normalize(images,
        device="gpu", crop=img_size,
        dtype=types.FLOAT,output_layout=types.NCHW,
        mean=mean_vals,
        std=std_vals,
        mirror=flip_flag)
    
    sequence = dalifn.stack(*images)
    label = abc[-1]
    label_flip=[4,5,6,7,0,1,2,3]
    label_flip_merge = [4,5,6,0,1,2]
    if flip_transform == True:
        if ((dataset=='Volleyball') & (int(num_class) == 8)):
            b_label = dalifn.random.uniform(range=[8.0,8.1],dtype=types.INT8)
            label = (flip_flag * dalifn.random.uniform(range=[4.0,4.1],dtype=types.INT8)+label)
            label = label-b_label*dalimath.floor(label/b_label)
            
        if ((dataset=='Volleyball') & (int(num_class) == 6)):
            b_label = dalifn.random.uniform(range=[6.0,6.1],dtype=types.INT8)
            label = (flip_flag * dalifn.random.uniform(range=[3.0,3.1],dtype=types.INT8)+label)
            label = label-b_label*dalimath.floor(label/b_label)
            
    # Conversion added on 26/11
    label = dalifn.cast(label,dtype=types.UINT8)

    if load_flow:
        # divide by 255 using std
        mean_vals =[0, 0, 0]
        std_vals=[255, 255, 255]
        img_size_flow = (23,40)
        flows = abc[seq_lgth:2*seq_lgth]
        flows = dalifn.decoders.image(flows, device="mixed", hw_decoder_load=1)
        flows = dalifn.resize(flows,device="gpu", resize_y=img_size_flow[0], resize_x = img_size_flow[1])    
        flows = dalifn.crop_mirror_normalize(flows,
            device="gpu", crop=img_size_flow,
            dtype=types.FLOAT,output_layout=types.NCHW,
            mean=mean_vals,
            std=std_vals,
            mirror=flip_flag)
    
        flows = dalifn.stack(*flows)

    if load_track:
        tracks = abc[seq_lgth:2*seq_lgth]
        tracks = dalifn.stack(*tracks)
        
    if load_depth:
        mean_vals=[0.485 * 255, 0.456 * 255, 0.406 * 255]
        std_vals=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        depths = abc[2*seq_lgth:3*seq_lgth]
        depths = dalifn.decoders.image(depths, device="mixed", hw_decoder_load=1)
        depths = dalifn.resize(depths,device="gpu", resize_y=img_size[0], resize_x = img_size[1])    
        depths = dalifn.crop_mirror_normalize(depths,
            device="gpu", crop=img_size,
            dtype=types.FLOAT,output_layout=types.NCHW,
            mean=mean_vals,
            std=std_vals)
    
        depths = dalifn.stack(*depths)
        if load_track:
            tracks_adj = abc[3*seq_lgth:4*seq_lgth]
            tracks_adj = dalifn.stack(*tracks_adj)
    
    if load_bb:
        list_boxes = abc[seq_lgth:2*seq_lgth]
        list_boxes = dalifn.stack(*list_boxes)
        len_boxes = abc[2*seq_lgth:3*seq_lgth]
        len_boxes = dalifn.stack(*len_boxes)

    if load_embd:
        embd_caption = abc[-2]

    return_vals = (sequence,)
    if load_flow:
        return_vals = return_vals + (flows,)
    if load_track:
        return_vals = return_vals + (tracks.gpu(),)
    if load_depth:
        return_vals = return_vals + (depths,)
        if load_track:
            return_vals = return_vals + (tracks_adj.gpu(),)
    if load_bb:
        return_vals = return_vals + (list_boxes.gpu(),)
        return_vals = return_vals + (len_boxes.gpu(),)
    if load_embd:
        return_vals = return_vals + (embd_caption.gpu(),)
    
    return_vals = return_vals + (label.gpu(),)
    return return_vals