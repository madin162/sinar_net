# ------------------------------------------------------------------------
# Reference:
# https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
# https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/volleyball.py
# ------------------------------------------------------------------------
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass', 'l_winpoint']

CLASSES = {0:'volleyball set right team', 1:'volleyball spike right team', 2:'volleyball pass right team', 3:'volleyball win point right team', 4:'volleyball set left team', 5:'volleyball spike left team', 6:'volleyball pass left team', 7:'volleyball win point left team'}

CLASSES_MERGE = {0:'volleyball set or pass right team', 1:'volleyball spike right team', 2:'volleyball win point right team', 3:'volleyball set or pass left team', 4:'volleyball spike left team', 5:'volleyball win point left team'}

def volleyball_read_annotations(path, seqs, num_activities):
    labels = {}
    if num_activities == 8:
        group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    # merge pass/set label    
    elif num_activities == 6:
        group_to_id = {'r_set': 0, 'r_spike': 1, 'r-pass': 0, 'r_winpoint': 2,
                       'l_set': 3, 'l-spike': 4, 'l-pass': 3, 'l_winpoint': 5}
    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:
            for line in f.readlines():
                values = line[:-1].split(' ')
                file_name = values[0]
                fid = int(file_name.split('.')[0])

                activity = group_to_id[values[1]]
                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels


def volleyball_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames


class VolleyballDataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True, caption_path = None):
        super(VolleyballDataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_flow = transforms.Compose([
            transforms.Resize((23,40)),
            transforms.ToTensor()
        ])
        self.aux_flow = args.aux_flow
        self.flow_path = args.flow_path

        if args.num_activities == 6:
            self.classes = CLASSES_MERGE
        elif args.num_activities == 8:
            self.classes = CLASSES
        
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
        sid, src_fid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(src_fid - 5, src_fid + 5), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame) + src_fid - segment_duration * (self.num_frame // 2)
        else:
            segment_duration = self.num_total_frame // self.num_frame
            sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + src_fid - segment_duration * (self.num_frame // 2)

        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        data_dict = {}
        images, activities = [], []
        images_adr = []
        if self.aux_flow:
            flows = []
        
        for i, (sid, src_fid, fid) in enumerate(frames):
            images_adr.append(self.image_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            img = Image.open(self.image_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            img = self.transform(img)

            images.append(img)
            activities.append(self.anns[sid][src_fid]['group_activity'])

            if self.aux_flow:
                flow = self.flow_path + '/%d/%d/%s.jpg' % (sid, src_fid, fid)
                flow = Image.open(flow)                
                flow = self.transform_flow(flow)
                flows.append(flow)

        images = torch.stack(images)
        activities = np.array(activities, dtype=np.int32)
        if self.aux_flow:
            flows = torch.stack(flows)
        
        if self.load_text:
            caption_adr = self.caption_path + '/%d/%d/embd_caption.npy' % (sid, src_fid)
            embd_caption = np.load(caption_adr)

            data_dict['embd_caption'] = torch.tensor(embd_caption)

        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()
        data_dict['images'] = images
        data_dict['label'] =  activities
        data_dict['images_adr'] = images_adr
        if self.aux_flow:
            data_dict['flows'] = flows
        
        return data_dict
        #return images, activities

class VBInputCallable(object):

    def __init__(self, batch_size, frames, anns, image_path, args,
                 is_training=True, shard_id=0, num_shards=1, caption_path = None):
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling

        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

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
        if args.num_activities == 6:
            self.classes = CLASSES_MERGE
        elif args.num_activities == 8:
            self.classes = CLASSES
        self.load_text = args.load_text
        self.caption_path = caption_path

    
    def select_frames(self, frame):
        sid, src_fid = frame

        if self.is_training:
            if self.random_sampling:
                # randomly select self.num_frame from -5 to +5 relative to src_fid (src_fid is center frame)
                sample_frames = random.sample(range(src_fid - 5, src_fid + 5), self.num_frame)
                sample_frames.sort()
            else:
                # TSN sampling
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(segment_duration, size=self.num_frame) + src_fid - segment_duration * (self.num_frame // 2)
        else:
            segment_duration = self.num_total_frame // self.num_frame
            sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + src_fid - segment_duration * (self.num_frame // 2)

        return [(sid, src_fid, fid) for fid in sample_frames]
    
    def load_samples(self, frames):
        images, activities, flows = [], [], []
        for i, (sid, src_fid, fid) in enumerate(frames):
            img = self.image_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid)
            f = open(img, "rb")
            img = np.frombuffer(f.read(), dtype=np.uint8) # don't decode the image just read the bytes!
            images.append(img)

            if self.is_training:
                if self.flow_path is not None:
                    #fid_flow = str(min(70,int(fid))).zfill(6)
                    flow = self.flow_path + '/%d/%d/%s.jpg' % (sid, src_fid, fid)
                    f = open(flow, "rb")
                    flow = np.frombuffer(f.read(), dtype=np.uint8) # don't decode the image just read the bytes!
                    flows.append(flow)
            
        return_var = images.copy()
        actv = self.anns[sid][src_fid]['group_activity']     
        actv = np.array(actv,dtype = np.int32)

        if self.is_training:
            if self.args.aux_flow:
                return_var.extend(flows)

        if self.load_text:
            #caption_adr = self.caption_path + '/%d/%d/embd_caption.npy' % (sid, src_fid)
            caption_adr = self.caption_path + '/%d/%d/embd_caption_2.npy' % (sid, src_fid)
            
            embd_caption = np.load(caption_adr)

            return_var.append(torch.tensor(embd_caption))
        
        
        return_var.append(actv)

        return return_var
     
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
    
    def __len__(self):
        return len(self.frames)
    