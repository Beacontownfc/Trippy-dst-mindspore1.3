import torch
from mindspore import save_checkpoint
import mindspore
import tqdm

weight_maps = {}

for i in range(12):
    weight_maps[f'bert.encoder.layer.{i}.attention.output.dense.weight'] = \
            f'bert.bert_encoder.layers.{i}.attention.output.dense.weight'
    weight_maps[f'bert.encoder.layer.{i}.attention.output.dense.bias'] = \
            f'bert.bert_encoder.layers.{i}.attention.output.dense.bias'
    
    weight_maps[f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'] = \
            f'bert.bert_encoder.layers.{i}.attention.output.layernorm.gamma'
    weight_maps[f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'] = \
            f'bert.bert_encoder.layers.{i}.attention.output.layernorm.beta'
    weight_maps[f'bert.encoder.layer.{i}.intermediate.dense.weight'] = \
            f'bert.bert_encoder.layers.{i}.intermediate.weight'
    weight_maps[f'bert.encoder.layer.{i}.intermediate.dense.bias'] = \
            f'bert.bert_encoder.layers.{i}.intermediate.bias'
    weight_maps[f'bert.encoder.layer.{i}.output.dense.weight'] = \
            f'bert.bert_encoder.layers.{i}.output.dense.weight'
    weight_maps[f'bert.encoder.layer.{i}.output.dense.bias'] = \
            f'bert.bert_encoder.layers.{i}.output.dense.bias'
    weight_maps[f'bert.encoder.layer.{i}.output.LayerNorm.gamma'] = \
            f'bert.bert_encoder.layers.{i}.output.layernorm.gamma'
    weight_maps[f'bert.encoder.layer.{i}.output.LayerNorm.beta'] = \
            f'bert.bert_encoder.layers.{i}.output.layernorm.beta'
    
    weight_maps[f'bert.encoder.layer.{i}.attention.self.query.weight'] = \
            f'bert.bert_encoder.layers.{i}.attention.attention.query_layer.weight'
    weight_maps[f'bert.encoder.layer.{i}.attention.self.query.bias'] = \
            f'bert.bert_encoder.layers.{i}.attention.attention.query_layer.bias'
    weight_maps[f'bert.encoder.layer.{i}.attention.self.key.weight'] = \
            f'bert.bert_encoder.layers.{i}.attention.attention.key_layer.weight'
    weight_maps[f'bert.encoder.layer.{i}.attention.self.key.bias'] = \
            f'bert.bert_encoder.layers.{i}.attention.attention.key_layer.bias'
    weight_maps[f'bert.encoder.layer.{i}.attention.self.value.weight'] = \
            f'bert.bert_encoder.layers.{i}.attention.attention.value_layer.weight'
    weight_maps[f'bert.encoder.layer.{i}.attention.self.value.bias'] = \
            f'bert.bert_encoder.layers.{i}.attention.attention.value_layer.bias'
    
    weight_maps['bert.embeddings.word_embeddings.weight'] = \
            f'bert.bert_embedding_lookup.embedding_table'
    weight_maps['bert.embeddings.position_embeddings.weight'] = \
            f'bert.bert_embedding_postprocessor.full_position_embeddings'
    weight_maps['bert.embeddings.token_type_embeddings.weight'] = \
            f'bert.bert_embedding_postprocessor.embedding_table'
    weight_maps['bert.embeddings.LayerNorm.gamma'] = \
            f'bert.bert_embedding_postprocessor.layernorm.gamma'
    weight_maps['bert.embeddings.LayerNorm.beta'] = \
            f'bert.bert_embedding_postprocessor.layernorm.beta'

    weight_maps['bert.pooler.dense.weight'] = \
            f'bert.dense.weight'
    weight_maps['bert.pooler.dense.bias'] = \
            f'bert.dense.bias'


t_params = torch.load('bert_base_torch.ckpt', map_location=torch.device('cpu'))
params_list = []
for weight_name, weight_value in t_params.items():
    if weight_name in weight_maps.keys():
        name = weight_maps[weight_name]
    else:
        name = weight_name
    
    weight_value = weight_value.numpy()
    
    print(weight_name, '->', name)
    params_list.append({'name': name, 'data': mindspore.Tensor(weight_value, mindspore.float32)})

save_checkpoint(params_list, 'ms_bert_base.ckpt')
print("over")
    
    

    