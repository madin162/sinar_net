from .volleyball import *
from .nba import *


TRAIN_SEQS_VOLLEY = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL_SEQS_VOLLEY = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_SEQS_VOLLEY = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]


def read_dataset(args):
    if args.dataset == 'Volleyball':
        data_path = args.data_path + args.dataset
        image_path = data_path + "/videos"
        train_data = volleyball_read_annotations(image_path, TRAIN_SEQS_VOLLEY + VAL_SEQS_VOLLEY, args.num_activities)
        train_frames = volleyball_all_frames(train_data)

        test_data = volleyball_read_annotations(image_path, TEST_SEQS_VOLLEY, args.num_activities)
        test_frames = volleyball_all_frames(test_data)
        
        caption_path = None
        if args.load_text:
            caption_path = data_path + '/scene_captions'
        
        
        if args.enable_dali:
            train_set = VBInputCallable(args.batch, train_frames, train_data, image_path, 
            args, is_training=True,
            caption_path=caption_path,
            shard_id=args.local_rank,num_shards=args.world_size)
            test_set = VBInputCallable(args.batch, test_frames, test_data, image_path, 
            args, is_training=False,
            caption_path=caption_path,
            shard_id=args.local_rank,num_shards=args.world_size)
            
        else:
            train_set = VolleyballDataset(train_frames, train_data, image_path, args, is_training=True, caption_path=caption_path)
            test_set = VolleyballDataset(test_frames, test_data, image_path, args, is_training=False, caption_path=caption_path)

    elif args.dataset == 'nba':
        data_path = args.data_path + 'NBA_dataset'
        image_path = data_path + "/videos"
        track_path = data_path + '/tracks_normalized_detected_all.pkl'
        depth_path = data_path + "/depth_cropped"
        if args.modality == 'Depth':
            image_path = depth_path

        train_id_path = data_path + "/train_video_ids"
        test_id_path = data_path + "/test_video_ids"

        train_ids = read_ids(train_id_path)
        test_ids = read_ids(test_id_path)

        if args.truncate:
            train_ids = train_ids[:len(train_ids)//30]
            test_ids = test_ids[:len(test_ids)//30]

        train_data = nba_read_annotations(image_path, train_ids)
        train_frames = nba_all_frames(train_data)

        test_data = nba_read_annotations(image_path, test_ids)
        test_frames = nba_all_frames(test_data)

        caption_path = None
        if args.load_text:
            caption_path = data_path + '/scene_captions'
        
        
        if args.enable_dali:
            train_set = NBAInputCallable(args.batch, train_frames, train_data, image_path, args, is_training=True, track_path = None,depth_path=None, caption_path=caption_path,
            shard_id=args.local_rank,num_shards=args.world_size)
            test_set = NBAInputCallable(args.batch, test_frames, test_data, image_path, 
            args, is_training=False, track_path = None,depth_path=None, caption_path=caption_path,
            shard_id=args.local_rank,num_shards=args.world_size)
            
        else:
            #train_set = NBADataset(train_frames, train_data, image_path, args, is_training=True, track_path=track_path)
            train_set = NBADataset(train_frames, train_data, image_path, args, is_training=True, caption_path=caption_path)
            test_set = NBADataset(test_frames, test_data, image_path, args, is_training=False, caption_path=caption_path)

    else:
        assert False

    print("modality: %s ;%d train samples and %d test samples" % (args.modality,len(train_frames), len(test_frames)))

    return train_set, test_set
