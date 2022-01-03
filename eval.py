from load_cache import load_and_cache_examples
from transformers import BertTokenizer
import mindspore.ops as ops
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend') #PYNATIVE_MODE
import numpy as np
from dataProcessor import DataLoader
import pickle as pkl
import re
from mindspore import load_checkpoint, load_param_into_net
from model_bert_dst import BertForDst
from bert_model.bert_model import BertConfig
import json
from tqdm import tqdm
import mindspore
import argparse

def eval_metric(model, features, per_slot_class_logits, per_slot_start_logits,
                per_slot_end_logits, per_slot_refer_logits):
    
    metric_dict = {}
    per_slot_correctness = {}
    for slot in model.slot_list:
        i = model.slot_list.index(slot)
        class_logits = per_slot_class_logits[i].asnumpy()
        start_logits = per_slot_start_logits[i].asnumpy()
        end_logits = per_slot_end_logits[i].asnumpy()
        refer_logits = per_slot_refer_logits[i].asnumpy()

        class_label_id = features['class_label_id'].asnumpy()[0][i]
        start_pos = features['start_pos'].asnumpy()[0][i]
        end_pos = features['end_pos'].asnumpy()[0][i]
        refer_id = features['refer_id'].asnumpy()[0][i]

        class_prediction = np.argmax(class_logits)
        class_correctness = np.equal(class_prediction, class_label_id).astype(np.float32)
        class_accuracy = class_correctness.mean()

        # "is pointable" means whether class label is "copy_value",
        # i.e., that there is a span to be detected.
        token_is_pointable = np.equal(class_label_id, model.class_types.index('copy_value')).astype(np.float32)
        start_prediction = np.argmax(start_logits)
        start_correctness = np.equal(start_prediction, start_pos).astype(np.float32)
        end_prediction = np.argmax(end_logits)
        end_correctness = np.equal(end_prediction, end_pos).astype(np.float32)
        token_correctness = start_correctness * end_correctness
        token_accuracy = (token_correctness * token_is_pointable).sum() / token_is_pointable.sum()
        # NaNs mean that none of the examples in this batch contain spans. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if np.isnan(token_accuracy):
            token_accuracy = np.array(1.0, np.float32)

        token_is_referrable = np.equal(class_label_id,
                                       model.class_types.index('refer') if 'refer' in model.class_types else -1).astype(np.float32)
        refer_prediction = np.argmax(refer_logits)
        refer_correctness = np.equal(refer_prediction, refer_id).astype(np.float32)
        refer_accuracy = refer_correctness.sum() / token_is_referrable.sum()
        # NaNs mean that none of the examples in this batch contain referrals. -> division by 0
        # The accuracy therefore is 1 by default. -> replace NaNs
        if np.isnan(refer_accuracy) or np.isinf(refer_accuracy):
            refer_accuracy = np.array(1.0, np.float32)

        total_correctness = class_correctness * (token_is_pointable * token_correctness + (1 - token_is_pointable)) * (
                token_is_referrable * refer_correctness + (1 - token_is_referrable))
        total_accuracy = total_correctness.mean()

        metric_dict['eval_accuracy_class_%s' % slot] = class_accuracy
        metric_dict['eval_accuracy_token_%s' % slot] = token_accuracy
        metric_dict['eval_accuracy_refer_%s' % slot] = refer_accuracy
        metric_dict['eval_accuracy_%s' % slot] = total_accuracy
        per_slot_correctness[slot] = total_correctness
    
    goal_correctness = np.prod(np.stack([c for c in per_slot_correctness.values()], -1), -1)
    goal_accuracy = goal_correctness.mean()
    metric_dict['eval_accuracy_goal'] = goal_accuracy
    return metric_dict

def predict_and_format(model, tokenizer, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,
                       per_slot_refer_logits, ids, input_ids_unmasked, values, inform, prefix, ds):

    prediction_list = []
    dialog_state = ds
    for i in range(len(ids)):
        if int(ids[i].split("-")[2]) == 0:
            dialog_state = {slot: 'none' for slot in model.slot_list}

        prediction = {}
        prediction_addendum = {}
        for k, slot in enumerate(model.slot_list):
            class_logits = per_slot_class_logits[k].asnumpy()[i]
            start_logits = per_slot_start_logits[k].asnumpy()[i]
            end_logits = per_slot_end_logits[k].asnumpy()[i]
            refer_logits = per_slot_refer_logits[k].asnumpy()[i]
            
            input_ids = features['input_ids'].asnumpy()[i].tolist()
            class_label_id = int(features['class_label_id'].asnumpy()[i][k])
            start_pos = int(features['start_pos'].asnumpy()[i][k])
            end_pos = int(features['end_pos'].asnumpy()[i][k])
            refer_id = int(features['refer_id'].asnumpy()[i][k])

            class_prediction = int(class_logits.argmax())
            start_prediction = int(start_logits.argmax())
            end_prediction = int(end_logits.argmax())
            refer_prediction = int(refer_logits.argmax())
        
            prediction['guid'] = ids[i].split("-")
            prediction['class_prediction_%s' % slot] = class_prediction
            prediction['class_label_id_%s' % slot] = class_label_id
            prediction['start_prediction_%s' % slot] = start_prediction
            prediction['start_pos_%s' % slot] = start_pos
            prediction['end_prediction_%s' % slot] = end_prediction
            prediction['end_pos_%s' % slot] = end_pos
            prediction['refer_prediction_%s' % slot] = refer_prediction
            prediction['refer_id_%s' % slot] = refer_id
            prediction['input_ids_%s' % slot] = input_ids

            if class_prediction == model.class_types.index('dontcare'):
                dialog_state[slot] = 'dontcare'
            elif class_prediction == model.class_types.index('copy_value'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids_unmasked[i])
                dialog_state[slot] = ' '.join(input_tokens[start_prediction:end_prediction + 1])
                dialog_state[slot] = re.sub("(^| )##", "", dialog_state[slot])
            elif 'true' in model.class_types and class_prediction == model.class_types.index('true'):
                dialog_state[slot] = 'true'
            elif 'false' in model.class_types and class_prediction == model.class_types.index('false'):
                dialog_state[slot] = 'false'
            elif class_prediction == model.class_types.index('inform'):
                dialog_state[slot] = '§§' + inform[i][slot]
            # Referral case is handled below

            prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]
            prediction_addendum['slot_groundtruth_%s' % slot] = values[i][slot]

        # Referral case. All other slot values need to be seen first in order
        # to be able to do this correctly.
        for k, slot in enumerate(model.slot_list):
            class_logits = per_slot_class_logits[k].asnumpy()[i]
            refer_logits = per_slot_refer_logits[k].asnumpy()[i]

            class_prediction = int(class_logits.argmax())
            refer_prediction = int(refer_logits.argmax())

            if 'refer' in model.class_types and class_prediction == model.class_types.index('refer'):
                # Only slots that have been mentioned before can be referred to.
                # One can think of a situation where one slot is referred to in the same utterance.
                # This phenomenon is however currently not properly covered in the training data
                # label generation process.
                dialog_state[slot] = dialog_state[model.slot_list[refer_prediction - 1]]
                prediction_addendum['slot_prediction_%s' % slot] = dialog_state[slot]  # Value update

        prediction.update(prediction_addendum)
        prediction_list.append(prediction)

    return prediction_list, dialog_state

def evaluate(model, tokenizer, args):
    eval_dataloader, features = load_and_cache_examples(args.test_data, model.slot_list, args.batch_size, evaluate=True)
    #eval_dataloader = DataLoader(args.test_data, args.configfile, args.batch_size).getDataset().create_dict_iterator()
    eval_dataloader = eval_dataloader.create_dict_iterator()
    # Eval!
    print("***** Running evaluation {} *****".format(args.pretrained))
    print("  Num examples = {}".format(7372))
    print("  Batch size = {}".format(1))
    all_results = []
    all_preds = []
    ds = {slot: 'none' for slot in model.slot_list}
    diag_state = {slot: np.array([0 for _ in range(1)]) for slot in model.slot_list}
    for batch in tqdm(eval_dataloader):
        turn_itrs = [features[i].guid.split('-')[2] for i in batch['exam_index']]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        
        for slot in model.slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0
                
        unique_ids = [features[i].guid for i in batch['exam_index']]
        values = [features[i].values for i in batch['exam_index']]
        input_ids_unmasked = [features[i].input_ids_unmasked for i in batch['exam_index']]
        inform = [features[i].inform for i in batch['exam_index']]
        batch.pop('exam_index')
        outputs = model(**batch)
    
        # Update dialog state for next turn.
        
        for slot in model.slot_list:
            k = model.slot_list.index(slot)
            updates = np.argmax(outputs[0][k].asnumpy())
            for i, u in enumerate([updates]):
                if u != 0:
                    diag_state[slot][i] = u
        
        results = eval_metric(model, batch, outputs[0], outputs[1], outputs[2], outputs[3])
        
        
        preds, ds = predict_and_format(model, tokenizer, batch, outputs[0], outputs[1], outputs[2], outputs[3],
                                       unique_ids, input_ids_unmasked, values, inform, args.pretrained, ds)
        
        all_results.append(results)
        all_preds.append(preds)

    all_preds = [item for sublist in all_preds for item in sublist]  # Flatten list

    # Generate final results
    final_results = {}
    for k in all_results[0].keys():
        final_results[k] = np.stack([r[k] for r in all_results]).mean()

    # Write final predictions (for evaluation with external tool)
    
    output_prediction_file = 'predict.json'
    with open(output_prediction_file, "w") as f:
        json.dump(all_preds, f, indent=2)
    
    return final_results

parser = argparse.ArgumentParser()  
parser.add_argument("--configfile", default='config/multiwoz21.json', type=str)
parser.add_argument("--test_data", default='cached_test_features', type=str)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--pretrained", default='ms_bert_base.ckpt', type=str)
parser.add_argument("--statefile", default='dataset/batch9.json', type=str)
parser.add_argument("--featurefile", default='dataset/test_features.pkl', type=str)
parser.add_argument("--dst_token_loss_for_nonpointable", default=False, type=bool)
parser.add_argument("--dst_refer_loss_for_nonpointable", default=False, type=bool)
parser.add_argument("--dst_class_aux_feats_inform", default=1, type=int)
parser.add_argument("--dst_class_aux_feats_ds", default=1, type=int)
parser.add_argument("--dst_class_loss_ratio", default=0.1, type=float)
parser.add_argument("--dst_dropout_rate", default=0.3, type=float)
parser.add_argument("--seq_length", default=180, type=int)
parser.add_argument("--vocab_size", default=30522, type=int)
parser.add_argument("--type_vocab_size", default=2, type=int)
args = parser.parse_args()

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

model = BertForDst(config, False)
model.to_float(mindspore.float16)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
param_dict = load_checkpoint(args.pretrained)
load_param_into_net(model, param_dict)
result = evaluate(model, tokenizer, args)
result_dict = {k: float(v) for k, v in result.items()}
result_dict["global_step"] = 7372
for key in sorted(result_dict.keys()):
    print(key, str(result_dict[key]))