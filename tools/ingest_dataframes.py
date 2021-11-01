"""
Example:

# VisDrone
python siammot/data/ingestion/ingest_dataframes.py --dataset_root ../datasets/VisDrone \
--train_path s3:// \
--val_path s3://

"""
import os
import sys
import ray
import json

import warnings
import argparse
from tqdm import tqdm
from datetime import timedelta
from timeit import default_timer as timer
from PIL import Image as PImage
# Ignore gluoncv+torch warning
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*gluoncv'
)
ray.init(log_to_driver=False)
from eda.parse_fmv import *
from gluoncv.torch.data.gluoncv_motion_dataset.dataset import GluonCVMotionDataset, DataSample, AnnoEntity, FieldNames, SplitNames

        
def main(df, dataset_root='', fps=30., label_key='o2_label', 
            annotation_file='anno.json', split_file='split.json',
            description='Initial ingestion'):
    
    """ Convert parquet df_gt DataFrame into GluonCVMotionDataset
    
    TODO: docstring
    """
    
    print('Beginning ingestion...', end='\n\n', flush=True)
    
    # Load DataFrames and combine
    start = timer()
    dfs = []
    if args.train_path:
        try:
            print('Loading train DataFrame...')
            train = load_df(args.train_path)
            train['split'] = SplitNames.TRAIN            
            print(train['o2_label'].value_counts())
            
            # Limit seqs if set
            if isinstance(args.limit, int) and args.limit > 1:
                train_seqs = train.seq.unique().tolist()[:args.limit]
                train = train[train.seq.isin(train_seqs)]
                print(f'Number of seqs reduced to {train.seq.unique().size} in training set.\n')
            
            dfs.append(train)
        except Exception as e:
            raise e 
    if args.val_path:
        try:
            print('\nLoading val DataFrame...')        
            val = load_df(args.val_path)
            val['split'] = SplitNames.VAL
            print(val['o2_label'].value_counts())
            dfs.append(val)
        except Exception as e:
            raise e
    if args.test_path:
        try:
            print('\nLoading test DataFrame...')
            test = load_df(args.test_path)
            test['split'] = SplitNames.TEST
            print(test['o2_label'].value_counts())            
            dfs.append(test)
        except Exception as e:
            raise e

    # Concat and drop unlabeled rows
    df = pd.concat(dfs)
    df['o2_label'].replace('', np.nan, inplace=True)
    #df.dropna(subset=['o2_label'], inplace=True)
    
    for split in df.split.unique():
        num_seqs = df[df.split==split].seq.unique().size
        print(f'{split} unique seqs: {num_seqs}')
    
    # remove /fmv_sequence/ from df jpeg_path's
    def _correct_path(path):
        if 'fmv_sequence' in path:
            return path.replace('/fmv_sequence/', '')
        else:
            return path
    df['jpeg_path'] = df['jpeg_path'].apply(lambda path: _correct_path(path))
    df.reset_index(drop=True, inplace=True)

    end = timer()
    load_secs = end - start
    print(f'Time taken to load DataFrames: {str(timedelta(seconds=load_secs))}\n')
    
    # Warn
    if not args.dump_directly and df.seq.unique().size > 10000:
        print('WARNING: Large number of sequences. Recommend using --dump-directly '
              'to save annotations in multiple pieces.', flush=True)
    
    # Map track ID's to ints (in case string)
    id_map = {i:n for n,i in enumerate (df['id'].unique().tolist())}
    df['id'] = df['id'].map(id_map)
    
    # Create class map    
    unique_labels = sorted(df.dropna(subset=['o2_label'])[label_key].unique().tolist())
    class_map = {l:n for n,l in enumerate(unique_labels,1)}
    print(f'\nclass_map: {class_map}\n', flush=True)
    
    # Create annotation directory
    if not os.path.exists(os.path.join(args.dataset_root, 'annotation')):
        os.makedirs(os.path.join(args.dataset_root, 'annotation'))
    
    # Instantiate dataset
    out_dataset = GluonCVMotionDataset(annotation_file=args.annotation_file, 
                                       root_path=args.dataset_root, 
                                       split_file=args.split_file, 
                                       load_anno=False)
    # Set metadata
    metadata = {
        FieldNames.DESCRIPTION: args.description,
        FieldNames.DATE_MODIFIED: str(datetime.now()),
        FieldNames.CLASS_MAPPING: sorted(class_map.keys()),
    }    
    out_dataset.metadata = metadata
    
    # Get, set, dump splits
    if 'split' not in df:
        raise KeyError('"split" column should be in DataFrame (with train/val/test determination)')
    splits = {determination:dfsplit.seq.unique().tolist() for determination, dfsplit in df.groupby('split')}    
    for k,v in splits.items():
        print(f'Split "{k}": {len(v)} videos', flush=True)
    out_dataset.splits = splits
    out_dataset.dump_splits()
    print(f'Split file dumped to {out_dataset._split_path}', flush=True)

    # Single sequence processing function to be ran in parallel
    @ray.remote
    def process_seq(df_ref, seq, dataset_root, fps, class_map):
        dfseq = df_ref[df_ref.seq==seq]
        
#         splits = dfseq['split'].unique().tolist()

#         if len(set(dfseq['split'].unique().tolist())) != 1:
#             if 'train' in splits:
#                 raise ValueError(f'split column has number of unique values not equal to 1 in seq: {seq}')
        determination = dfseq['split'].iloc[0]
        
        sample = DataSample(seq)
        
        # Metadata
        rel_base_dir_img = os.path.join(dataset_root, os.path.dirname(dfseq['jpeg_path'].iloc[0]).strip('./'))
        num_frames = int(dfseq['jpeg_count'].iloc[0])
        
        try:
            width = int(dfseq['width_pix'].iloc[0])
            height = int(dfseq['height_pix'].iloc[0])
        except:
            jpeg_path_0 = os.path.join('/fmv_sequence', dfseq['jpeg_path'].iloc[0])
            with PImage.open(jpeg_path_0) as im:
                width, height = im.size
            width, height = int(width), int(height)

        # Checks
        if num_frames < 1:
            raise ValueError(f'Too few frames in {seq}!')
        if not all(isinstance(x, int) for x in dfseq['id'].unique().tolist()):
            raise TypeError('Track ids should be ints!')
        
        # Fill metadata
        metadata = {
            FieldNames.DATA_PATH: rel_base_dir_img,
            FieldNames.FPS: args.fps,
            FieldNames.NUM_FRAMES: num_frames,
            FieldNames.RESOLUTION: {"width": width, "height": height},
        }     
        sample.metadata = metadata

        # Fill entities
        dfseq = dfseq.dropna(subset=['o2_label'])
        dfseq = dfseq[dfseq['Null']==False]
        for frame_idx, dfframe in dfseq.groupby('Frame#'):
            for index, row in dfframe.iterrows():

                try:
                    label = row[args.label_column]
                except KeyError as k:
                    raise k                
                
                if row['Null'] or label == '':
                    continue
                
                obj_id = row['id']

                try:
                    x = int(round(float(row['x'])))
                except KeyError as k:
                    raise k
                try:
                    y = int(round(float(row['y'])))
                except KeyError as k:
                    raise k
                try:
                    w = int(round(float(row['width'])))
                except KeyError as k:
                    raise k
                try:
                    h = int(round(float(row['height'])))
                except KeyError as k:
                    raise k
                    
                conf = 1

                # Check bounding box coords
                min_width = min_height = 2
                if w<min_width or h<min_height:
                    continue
                    #raise KeyError(f'w<min_width or h<min_height: w={w}, h={h}')

                if x < 0:
                    w,x = w+x, 0
                    #raise KeyError(f'x < 0: {x}')
                if y < 0:
                    h,y = h+y, 0                    
                    #raise KeyError(f'y < 0: {y}')
                if x+w>width:
                    w = width-x                    
                    #raise KeyError(f'x+w>width: {x}+{w}>{width}')
                if y+h>height:
                    h = height-y                    
                    #raise KeyError(f'y+h>height: {y}+{h}>{height}')
                    
                time_ms = int((frame_idx) / fps * 1000)    
                entity = AnnoEntity(time=time_ms, id=obj_id)
                entity.bbox = [x, y, w, h]
                blob = {
                    FieldNames.FRAME_IDX: frame_idx,
                }
                entity.labels = {label: class_map[label]}
                entity.confidence = conf
                entity.blob = blob
                sample.add_entity(entity)       
        
        return sample

    # Main control: Iterate through chunks of seqs and process in parallel
    
    for split, dfsplit in df.groupby('split'):
        print(f'Processing split {split}')
        sequence_list = dfsplit.seq.unique().tolist()
        print('\nKicking off ingesting chunks of videos...', flush=True)
        with tqdm(total=len(sequence_list), desc=' Starting...', position=0, leave=True) as pbar:
            for sequence_chunk in chunk(sequence_list, args.chunk_size):
                    pbar.set_description(f' Processing chunk of {len(sequence_chunk)} seqs')
                    # Get subset DataFrame for chunk of videos (for speed)
                    dfchunk = dfsplit[dfsplit.seq.isin(sequence_chunk)].copy()
                    df_ref = ray.put(dfchunk)
                    # Set Ray futures
                    futures = [process_seq.remote(df_ref, seq, dataset_root, fps, class_map) for seq in sequence_chunk]
                    # Do computations
                    samples_list = [ _ for _ in to_iterator(futures) ]
                    for sample in samples_list:
                        out_dataset.add_sample(sample, dump_directly=args.dump_directly)
                    # Update count
                    pbar.update(len(sequence_chunk))
        print(f'Finished split {split}.')
    sys.stdout.flush()
    sys.stderr.flush()
    print('Finished ingestion. Writing dataset to file (may take a while)...', flush=True)
    start = timer()
    out_dataset.dump()
    end = timer()
    save_secs = end - start
    print(f'Time taken to save anno file: {str(timedelta(seconds=save_secs))}')
    print(f'Anno file dumped to {out_dataset._anno_path}', flush=True)
    return out_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Ingest Parquet Ground Truth 
    DataFrames and dump GluonCVMotionDataset''')
    parser.add_argument('--dataset_root', type=str, default='',
                        help='The path of dataset folder')
    parser.add_argument('--annotation_file', type=str, default='anno.json',
                        help='The file name (with json) of annotation file')
    parser.add_argument('--split_file', type=str, default='splits.json',
                        help='The file name (with json) of split file')
    parser.add_argument('--train_path', type=str, default='',
                        help='The path of train parquet directory (local or S3)')
    parser.add_argument('--val_path', type=str, default='',
                        help='The path of validation parquet directory (local or S3)')
    parser.add_argument('--test_path', type=str, default='',
                        help='The path of test parquet directory (local or S3)')
    parser.add_argument('--label_column', type=str, default='o2_label',
                        choices=['o2_label', 'o9_label'], help='Label column in DataFrames')
    parser.add_argument('--fps', type=float, default=30.,
                        help='Video FPS')    
    parser.add_argument('--description', type=str, default='Initial ingestion',
                        help='Description for dataset metadata')
    parser.add_argument('--chunk_size', type=int, default=64,
                        help='Number of seqs to process in parallel (default value is for high-RAM instance)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Number of seqs to limit training set to')
    parser.add_argument('--dump-directly', dest='dump_directly', action='store_true')
    
    args = parser.parse_args()

    dataset = main(args)
    print('\nFinished creating GluonCVMotionDataset. Exiting...', flush=True)