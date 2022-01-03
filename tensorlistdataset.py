import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
import json

class TensorListDataset:
    r"""Dataset wrapping tensors, tensor dicts and tensor lists.

    Arguments:
        *data (Tensor or dict or list of Tensors): tensors that have the same size
        of the first dimension.
    """

    def __init__(self, *data):
        if isinstance(data[0], dict):
            size = list(data[0].values())[0].shape[0]
        elif isinstance(data[0], list):
            size = data[0][0].shape[0]
        else:
            size = data[0].shape[0]
        for element in data:
            if isinstance(element, dict):
                assert all(size == tensor.shape[0] for name, tensor in element.items()) # dict of tensors
            elif isinstance(element, list):
                assert all(size == tensor.shape[0] for tensor in element) # list of tensors
            else:
                assert size == element.shape[0] # tensor
        self.size = size
        self.data = data
        with open('config/multiwoz21.json', 'r') as f:
            self.slot_list = json.load(f)['slots']

    def __getitem__(self, index):
        result = []
        for element in self.data:
            if isinstance(element, dict):
                result.append(np.array([element[slot][index] for slot in self.slot_list], np.int32))
            elif isinstance(element, list):
                result.append(np.array(v[index], np.int32) for v in element)
            else:
                result.append(element[index])
        return tuple(result)

    def __len__(self):
        return self.size

class DataLoader:
    def __init__(self, dataset_generator, batch_size, evaluate=False):
        if evaluate:
            self.dataset = ds.GeneratorDataset(dataset_generator, ["input_ids", "input_mask", "segment_ids", 
                                                    "start_pos", "end_pos", "inform_slot_id",
                                                    "refer_id", "diag_state", "class_label_id", "exam_index"], sampler=ds.SequentialSampler())
        else:
            self.dataset = ds.GeneratorDataset(dataset_generator, ["input_ids", "input_mask", "segment_ids", 
                                                    "start_pos", "end_pos", "inform_slot_id",
                                                    "refer_id", "diag_state", "class_label_id"], sampler=ds.RandomSampler())
        
        type_cast_op = C.TypeCast(mstype.int32)
        
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="input_ids")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="input_mask")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="segment_ids")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="start_pos")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="end_pos")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="inform_slot_id")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="refer_id")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="diag_state")
        self.dataset = self.dataset.map(operations=type_cast_op, input_columns="class_label_id")
        if evaluate:
            self.dataset = self.dataset.map(operations=type_cast_op, input_columns="exam_index")
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)
        
    def getDataset(self):
        return self.dataset