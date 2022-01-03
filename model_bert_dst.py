import mindspore.nn as nn
from bert_model.bert_model import BertModel
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore

class BertForDst(nn.Cell):
    def __init__(self, config, is_train=True):
        super(BertForDst, self).__init__()
        self.is_train = is_train
        self.slot_list = config.dst_slot_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
        self.class_aux_feats_inform = config.dst_class_aux_feats_inform
        self.class_aux_feats_ds = config.dst_class_aux_feats_ds
        self.class_loss_ratio = config.dst_class_loss_ratio

        # Only use refer loss if refer class is present in dataset.
        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        self.bert = BertModel(config, is_train)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        #self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

        if self.class_aux_feats_inform:
            self.inform_projection =  nn.Dense(len(self.slot_list), len(self.slot_list))
        if self.class_aux_feats_ds:
            self.ds_projection =  nn.Dense(len(self.slot_list), len(self.slot_list))

        aux_dims = len(self.slot_list) * (self.class_aux_feats_inform + self.class_aux_feats_ds) # second term is 0, 1 or 2
        
        self.cell_list = nn.CellList()
        
        for slot in self.slot_list:
            self.cell_list.append(nn.Dense(config.hidden_size + aux_dims, self.class_labels)) #class [0-29]
        for slot in self.slot_list:
            self.cell_list.append(nn.Dense(config.hidden_size, 2)) #toke [30-59]
        for slot in self.slot_list:
            self.cell_list.append(nn.Dense(config.hidden_size + aux_dims, len(self.slot_list) + 1)) #refer [60 - 89]
            
        self.stack = ops.Stack(axis=1)
        self.cat1 = ops.Concat(axis=1)
        self.split = ops.Split(-1, 2)
        self.squeeze = ops.Squeeze(-1)
        self.eq = ops.Equal()
        self.class_loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.token_loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.refer_loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        
    
    def construct(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                start_pos=None,
                end_pos=None,
                inform_slot_id=None,
                refer_id=None,
                diag_state=None,
                class_label_id=None):
        
        bsz, ignored_index = input_ids.shape[0], input_ids.shape[1]
        
        start_pos = start_pos.transpose((1, 0))
        end_pos = end_pos.transpose((1, 0))
        refer_id = refer_id.transpose((1, 0))
        class_label_id = class_label_id.transpose((1, 0))
               
        outputs = self.bert(input_ids, segment_ids, input_mask)
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        
        if self.is_train:
            sequence_output = self.dropout(sequence_output)
            pooled_output = self.dropout(pooled_output)
        
        inform_labels = inform_slot_id.astype(mindspore.float32)
        diag_state_labels = diag_state.astype(mindspore.float32).clip(0.0, 1.0)
        
        per_slot_class_logits = []
        per_slot_start_logits = []
        per_slot_end_logits = []
        per_slot_refer_logits = []
        
        total_loss = 0
        for i in range(30):
            pooled_output_aux = self.cat1((pooled_output.astype(mindspore.float16), 
                                           self.inform_projection(inform_labels).astype(mindspore.float16),
                                           self.ds_projection(diag_state_labels).astype(mindspore.float16)))
            
            start_logits, end_logits = self.split(self.cell_list[i + 30](sequence_output))
            start_logits = self.squeeze(start_logits)
            end_logits = self.squeeze(end_logits)
            
            class_logits = self.cell_list[i](pooled_output_aux)
            refer_logits = self.cell_list[i + 60](pooled_output_aux)
            
            per_slot_class_logits.append(class_logits)
            per_slot_start_logits.append(start_logits)
            per_slot_end_logits.append(end_logits)
            per_slot_refer_logits.append(refer_logits)
            
            if self.is_train:               
                start_loss = self.token_loss_fct(start_logits, start_pos[i])
                end_loss = self.token_loss_fct(end_logits, end_pos[i])
                token_loss = (start_loss + end_loss) / 2.0
            
                token_is_pointable = (start_pos[i] > 0).astype(mindspore.float32)
                token_loss *= token_is_pointable
                    
                refer_loss = self.refer_loss_fct(refer_logits, refer_id[i])
                token_is_referrable = self.eq(class_label_id[i], self.refer_index).astype(mindspore.float32)
                refer_loss *= token_is_referrable
            
                class_loss = self.class_loss_fct(class_logits, class_label_id[i])
            
                per_example_loss = (self.class_loss_ratio) * class_loss + \
                                       ((1 - self.class_loss_ratio) / 2) * token_loss + \
                                       ((1 - self.class_loss_ratio) / 2) * refer_loss
                
                total_loss += per_example_loss.sum()
            
        if self.is_train:
            return total_loss
        else:
            return per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits