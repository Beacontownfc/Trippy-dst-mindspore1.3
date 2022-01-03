from load_cache import load_and_cache_examples
from model_bert_dst import BertForDst
from bert_model.bert_model import BertConfig
import json
from mindspore import load_checkpoint, load_param_into_net
from train_utils import BertTrainCell, BertLearningRate, get_linear_warmup
from mindspore import context
import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore.train.callback import LossMonitor
import mindspore
from mindspore import save_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--end_learning_rate", default=1e-5, type=float)
parser.add_argument("--eps", default=1e-6, type=float)
parser.add_argument("--train_epoch", default=14, type=int)
parser.add_argument("--configfile", default='config/multiwoz21.json', type=str)
parser.add_argument("--train_data", default='cached_train_features', type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--pretrained", default='ms_bert_base.ckpt', type=str)
parser.add_argument("--decay_step", default=500, type=int)
parser.add_argument("--power", default=1.0, type=float)
parser.add_argument("--dst_token_loss_for_nonpointable", default=False, type=bool)
parser.add_argument("--dst_refer_loss_for_nonpointable", default=False, type=bool)
parser.add_argument("--dst_class_aux_feats_inform", default=1, type=int)
parser.add_argument("--dst_class_aux_feats_ds", default=1, type=int)
parser.add_argument("--dst_class_loss_ratio", default=0.8, type=float)
parser.add_argument("--dst_dropout_rate", default=0.3, type=float)
parser.add_argument("--seq_length", default=180, type=int)
parser.add_argument("--vocab_size", default=30522, type=int)
parser.add_argument("--type_vocab_size", default=2, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--warmup_epoch", default=1, type=int)
parser.add_argument("--max_lr_epoch", default=14, type=int)

args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend') #PYNATIVE_MODE

with open(args.configfile, 'r') as f:
    config_data = json.load(f)
    
config = BertConfig(args.batch_size)
config.dst_slot_list = config_data['slots']
config.dst_class_types = config_data['class_types']
config.dst_class_labels = len(config_data['class_types'])
config.dst_token_loss_for_nonpointable = args.dst_token_loss_for_nonpointable
config.dst_refer_loss_for_nonpointable = args.dst_refer_loss_for_nonpointable
config.dst_class_aux_feats_inform = args.dst_class_aux_feats_inform
config.dst_class_aux_feats_ds = args.dst_class_aux_feats_ds
config.dst_class_loss_ratio = args.dst_class_loss_ratio
config.dst_dropout_rate = args.dst_dropout_rate
config.seq_length = args.seq_length
config.vocab_size = args.vocab_size
config.type_vocab_size = args.type_vocab_size
    

mindspore.set_seed(args.seed)
dataset, _ = load_and_cache_examples(args.train_data, config_data['slots'], args.batch_size)
model = BertForDst(config)
param_dict = load_checkpoint(args.pretrained)
load_param_into_net(model, param_dict)
model.to_float(mindspore.float16)

per_epoch_step = dataset.get_dataset_size()
lr_schedule = get_linear_warmup(args.learning_rate, args.end_learning_rate, 
                                args.max_lr_epoch * per_epoch_step, 
                                (args.train_epoch - args.max_lr_epoch) * per_epoch_step, 
                                per_epoch_step, args.warmup_epoch)

optimizer = nn.AdamWeightDecay(model.trainable_params(), lr_schedule, eps=args.eps)

update_cell = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
netwithgrads = BertTrainCell(model, optimizer=optimizer, scale_update_cell=update_cell)
netwithgrads.set_train()
net = Model(netwithgrads)
# 设置模型保存参数
config_ck = CheckpointConfig(save_checkpoint_steps=per_epoch_step, keep_checkpoint_max=3)
ckpoint = ModelCheckpoint(prefix="BertForDst.ckpt", config=config_ck)
net.train(args.train_epoch, dataset, callbacks=[LossMonitor(100), ckpoint], dataset_sink_mode=False)
save_checkpoint(model, "BertForDst.ckpt")