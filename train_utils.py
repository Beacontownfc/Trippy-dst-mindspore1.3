import numpy as np
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal, initializer
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore.nn import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication import get_group_size
from mindspore import context
import mindspore
from bert_model.bert_for_pre_training import clip_grad

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class BertTrainCell(nn.TrainOneStepWithLossScaleCell):
    """
    Specifically defined for finetuning where only four inputs tensor are needed.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):

        super(BertTrainCell, self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(
            get_by_list=True,
            sens_param=True)
        self.reducer_flag = False
        self.allreduce = ops.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("mirror_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = ops.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = ops.FloatStatus()
            self.addn = ops.AddN()
            self.reshape = ops.Reshape()
        else:
            self.alloc_status = ops.NPUAllocFloatStatus()
            self.get_status = ops.NPUGetFloatStatus()
            self.clear_before_grad = ops.NPUClearFloatStatus()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = ops.LessEqual()
        self.hyper_map = ops.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    def construct(self,
                  input_ids,
                  input_mask=None,
                  segment_ids=None,
                  start_pos=None,
                  end_pos=None,
                  inform_slot_id=None,
                  refer_id=None,
                  diag_state=None,
                  class_label_id=None,
                  sens=None):
        """construct BertPoetryCell"""

        weights = self.weights
        loss = self.network(
                input_ids,
                input_mask,
                segment_ids,
                start_pos,
                end_pos,
                inform_slot_id,
                refer_id,
                diag_state,
                class_label_id)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        grads = self.grad(self.network, weights)(input_ids, input_mask,
                                                            segment_ids,
                                                            start_pos,
                                                            end_pos,
                                                            inform_slot_id,
                                                            refer_id,
                                                            diag_state,
                                                            class_label_id,
                                                            self.cast(scaling_sens, mstype.float32))
        grads = self.hyper_map(ops.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond)
        return ops.depend(ret, succ)

class BertLearningRate(nn.WarmUpLR):
    """
    Warmup-decay learning rate for Bert network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__(learning_rate, warmup_steps)
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = nn.WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = nn.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = mindspore.Tensor(np.array([warmup_steps]).astype(np.float32))
        self.greater = ops.Greater()
        self.one = mindspore.Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()

    def construct(self, global_step):
        """construct BertLearningRate"""
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr

def get_linear_warmup(learning_rate, end_learning_rate, w_total_step, d_total_step, step_per_epoch, warmup_epoch):
    lr_warmup = nn.warmup_lr(learning_rate, w_total_step, step_per_epoch, warmup_epoch)
    lr_decay = []
    if d_total_step > 0:
        lr_decay = nn.cosine_decay_lr(0.0, end_learning_rate, d_total_step, step_per_epoch, warmup_epoch)
    return lr_warmup + lr_decay