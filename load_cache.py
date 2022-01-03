import torch
import mindspore
from tensorlistdataset import *
import mindspore.numpy as mnp
import json
import numpy

def load_and_cache_examples(cached_file, slot_list, batch_size, evaluate=False):
    # Load data features from cache or dataset file
    print('**********开始加载cache************')

    features = torch.load(cached_file)

    # Convert to Tensors and build dataset
    all_input_ids = np.array([f.input_ids for f in features], np.int32)
    all_input_mask = np.array([f.input_mask for f in features], np.int32)
    all_segment_ids = np.array([f.segment_ids for f in features], np.int32)
    all_example_index = np.arange(0, all_input_ids.shape[0], 1, np.int32)
    f_start_pos = [f.start_pos for f in features]
    f_end_pos = [f.end_pos for f in features]
    f_inform_slot_ids = [f.inform_slot for f in features]
    f_refer_ids = [f.refer_id for f in features]
    f_diag_state = [f.diag_state for f in features]
    f_class_label_ids = [f.class_label_id for f in features]
    all_start_positions = {}
    all_end_positions = {}
    all_inform_slot_ids = {}
    all_refer_ids = {}
    all_diag_state = {}
    all_class_label_ids = {}
    for s in slot_list:
        all_start_positions[s] = np.array([f[s] for f in f_start_pos], np.int32)
        all_end_positions[s] = np.array([f[s] for f in f_end_pos], np.int32)
        all_inform_slot_ids[s] = np.array([f[s] for f in f_inform_slot_ids], np.int32)
        all_refer_ids[s] = np.array([f[s] for f in f_refer_ids], np.int32)
        all_diag_state[s] = np.array([f[s] for f in f_diag_state], np.int32)
        all_class_label_ids[s] = np.array([f[s] for f in f_class_label_ids], np.int32)
    
    if not evaluate:
        dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_inform_slot_ids,
                                    all_refer_ids,
                                    all_diag_state,
                                    all_class_label_ids)
        dataset = DataLoader(dataset, batch_size, evaluate).getDataset()
        
    else:
        dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_inform_slot_ids,
                                    all_refer_ids,
                                    all_diag_state,
                                    all_class_label_ids, all_example_index)
        dataset = DataLoader(dataset, batch_size, evaluate).getDataset()

    return dataset, features

    
