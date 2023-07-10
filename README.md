# TripPy mindspore 复现
# TripPy based on mindspore

Source paper: https://arxiv.org/pdf/2005.02877.pdf

How to run:
``` bash
# install pytorch, transformers and mindspore
pip install pytorch
pip install transformers
......

# training the TripPy model
python run_dst.py

# evaluate the checkpoint
python eval.py --pretrained=${your checkpoint}


# compute the joint goal acc
python metric_bert_dst.py

```

-----
The caches are created by the TripPy original code, we only provide MultiWOZ2.1 caches [https://gitee.com/lifancong/trip-py-mindspore](https://gitee.com/lifancong/trip-py-mindspore).
ms_bert_base.ckpt is the pretraining model of mindspore bert, you could runing the following command to convert pytoch checkpoint to mindspore checkpoint. 
``` bash
python convert_params.py
```
if you want to train on GPU, just change context.set_context(mode=context.GRAPH_MODE, device_target='Ascend') in the run_dst.py to 'GPU'.


