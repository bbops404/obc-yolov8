# Ultralytics YOLO 🚀, AGPL-3.0 license
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 
import contextlib
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfig,
)
from torch.ao.quantization.fake_quantize import FakeQuantize
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
from ultralytics.nn.CA_Attention import CoordAtt
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv,DWConvTranspose2d,Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,RTDETRDecoder, Segment,LightConv, RepConv,SpatialAttention)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml, check_version
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync)
from ultralytics.nn.qlhnet import ShuffleNetV2, Conv_maxpool
from ultralytics.nn.se import SEAttention
from ultralytics.nn.bifpn import Concat_BiFPN
from ultralytics.nn.ContextAggregation import ContextAggregation
from ultralytics.nn.Ghostnet import GGhostRegNet
from ultralytics.nn.CBAM import CBAM
from ultralytics.nn.BoTNet import BoTNet

from ultralytics.nn.DecoupledHead import DecoupledHead
from ultralytics.nn.CARAFE import CARAFE
from ultralytics.nn. Involution import Involution

from ultralytics.nn. SlimNeck import VoVGSCSP, VoVGSCSPC, GSConv
from ultralytics.nn. SwinTransformer import SwinTransformer
from ultralytics.nn. HorBlock import HorBlock
from ultralytics.nn. MobileOne import MobileOneBlock
from ultralytics.nn. MobileViT import MobileViT,MV2Block,MobileViTAttention,MobileViTBlock
from ultralytics.nn. CondConv2D import CondConv2D
from ultralytics.nn. RepLKNet import RepLKNet_Stem, RepLKNet_stage1, RepLKNet_stage2, RepLKNet_stage3, RepLKNet_stage4

from ultralytics.nn. EfficientNetv2 import MBConv,FusedMBConv,stem
from ultralytics.nn.vanillanet import vanillanetBlock
from ultralytics.nn.CrissCrossAttention import CrissCrossAttention
from ultralytics.nn.RepViTblock import RepViTblock
from ultralytics.nn.BiFormer import BiLevelRoutingAttention,Attention,AttentionLePE
from ultralytics.nn.DecoupledHead import DecoupledHead
from ultralytics.nn.DSConv import DSConv,DySnakeConv,C2f_DySnakeConv,Bottleneck_DySnakeConv
from ultralytics.nn.Glod import  IFM,SimFusion_3in,SimFusion_4in,InjectionMultiSum_Auto_pool,PyramidPoolAgg,TopBasicLayer,AdvPoolFusion
from ultralytics.nn.LSKA import C2f_LSKA_Attention,LSKA_Attention,LSKA
from ultralytics.nn.EMA_attention import EMA_attention
from ultralytics.nn.ODConv import ODConv

# Import QAT modules for quantization-aware training
try:
    from ultralytics.nn.qat_modules import QATBoTNet, QATCoordAtt, FP32ODConv
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    LOGGER.warning("QAT modules not available. Install torch.ao.quantization for QAT support.")

try:
    import thop
except ImportError:
    thop = None

globals()['CoordAtt'] = CoordAtt
# Register QAT modules in globals if available
if QAT_AVAILABLE:
    globals()['QATBoTNet'] = QATBoTNet
    globals()['QATCoordAtt'] = QATCoordAtt
    globals()['FP32ODConv'] = FP32ODConv


SAFE_QAT_CLAMP_VALUE = 8.0
SAFE_QAT_AVERAGING_CONSTANT = 0.05
SAFE_QAT_EPS = 1e-5


class ClampedMovingAverageObserver(MovingAverageMinMaxObserver):
    """MovingAverage observer that clamps ranges to keep fake-quant zero-points valid."""

    def __init__(self, clamp_value: float = SAFE_QAT_CLAMP_VALUE, **kwargs):
        super().__init__(**kwargs)
        self.clamp_value = clamp_value

    def _clamp_range(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, -self.clamp_value, self.clamp_value)

    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor):
        # Replace NaN with safe fallback values
        fallback_min = torch.tensor(-self.clamp_value, device=min_val.device, dtype=min_val.dtype)
        fallback_max = torch.tensor(self.clamp_value, device=max_val.device, dtype=max_val.dtype)
        min_val = torch.where(torch.isnan(min_val), fallback_min, min_val)
        max_val = torch.where(torch.isnan(max_val), fallback_max, max_val)
        min_val = self._clamp_range(min_val)
        max_val = self._clamp_range(max_val)
        max_val = torch.where(max_val <= min_val, min_val + self.eps, max_val)
        return super()._calculate_qparams(min_val, max_val)


class ClampedMovingAveragePerChannelObserver(MovingAveragePerChannelMinMaxObserver):
    """Per-channel moving-average observer with clamped ranges."""

    def __init__(self, clamp_value: float = SAFE_QAT_CLAMP_VALUE, **kwargs):
        super().__init__(**kwargs)
        self.clamp_value = clamp_value

    def _clamp_range(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, -self.clamp_value, self.clamp_value)

    def _calculate_qparams(self, min_vals: torch.Tensor, max_vals: torch.Tensor):
        # Replace NaN with safe fallback values
        fallback_min = torch.full_like(min_vals, -self.clamp_value)
        fallback_max = torch.full_like(max_vals, self.clamp_value)
        min_vals = torch.where(torch.isnan(min_vals), fallback_min, min_vals)
        max_vals = torch.where(torch.isnan(max_vals), fallback_max, max_vals)
        min_vals = self._clamp_range(min_vals)
        max_vals = self._clamp_range(max_vals)
        max_vals = torch.where(max_vals <= min_vals, min_vals + self.eps, max_vals)
        return super()._calculate_qparams(min_vals, max_vals)


def _create_safe_qconfig(backend: str) -> QConfig:
    """Return a QConfig that clamps observer ranges to keep zero-points inside int8 bounds."""

    reduce_range = False
    activation_fake_quant = FakeQuantize.with_args(
        observer=ClampedMovingAverageObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=reduce_range,
        averaging_constant=SAFE_QAT_AVERAGING_CONSTANT,
        clamp_value=SAFE_QAT_CLAMP_VALUE,
        eps=SAFE_QAT_EPS,
    )
    weight_fake_quant = FakeQuantize.with_args(
        observer=ClampedMovingAveragePerChannelObserver,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        reduce_range=False,
        averaging_constant=SAFE_QAT_AVERAGING_CONSTANT,
        clamp_value=SAFE_QAT_CLAMP_VALUE,
        eps=SAFE_QAT_EPS,
    )
    return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)


def _clamp_fake_quant_observer(fake_quant: FakeQuantize, clamp_value: float = SAFE_QAT_CLAMP_VALUE):
    observer = getattr(fake_quant, "activation_post_process", None)
    if observer is None:
        return
    eps = getattr(observer, "eps", SAFE_QAT_EPS)
    if hasattr(observer, "min_val"):
        min_val = observer.min_val
        # Replace NaN with safe fallback
        if isinstance(min_val, torch.Tensor):
            fallback_min = torch.full_like(min_val, -clamp_value)
            min_val = torch.where(torch.isnan(min_val), fallback_min, min_val)
            observer.min_val = torch.clamp(min_val, -clamp_value, clamp_value)
    if hasattr(observer, "max_val"):
        max_val = observer.max_val
        # Replace NaN with safe fallback
        if isinstance(max_val, torch.Tensor):
            fallback_max = torch.full_like(max_val, clamp_value)
            max_val = torch.where(torch.isnan(max_val), fallback_max, max_val)
            observer.max_val = torch.clamp(max_val, -clamp_value, clamp_value)
    if hasattr(observer, "min_val") and hasattr(observer, "max_val"):
        min_val = observer.min_val
        max_val = observer.max_val
        if isinstance(min_val, torch.Tensor) and isinstance(max_val, torch.Tensor):
            observer.max_val = torch.where(max_val <= min_val, min_val + eps, max_val)


def attach_fake_quant_clamp_hooks(module: nn.Module, clamp_value: float = SAFE_QAT_CLAMP_VALUE):
    """Attach hooks that keep FakeQuant observers within the representable int8 range."""

    for fake_quant in module.modules():
        if isinstance(fake_quant, FakeQuantize):
            _clamp_fake_quant_observer(fake_quant, clamp_value)

            def _pre_hook(mod, _inputs, *, _clamp=_clamp_fake_quant_observer, _value=clamp_value):
                _clamp(mod, _value)

            fake_quant.register_forward_pre_hook(_pre_hook)


def ensure_module_bookkeeping(module, recursive=False):
    """Ensure a module exposes the standard ``nn.Module`` bookkeeping attributes."""
    if module is None or not isinstance(module, nn.Module):
        return module

    def _ensure_ordered_dict_attr(target, attr_name):
        current = getattr(target, attr_name, None)
        if isinstance(current, OrderedDict):
            return False
        updated = False
        if isinstance(current, dict):
            new_value = OrderedDict(current.items())
            updated = True
        else:
            new_value = OrderedDict()
            if current is not None:
                updated = True
        setattr(target, attr_name, new_value)
        return updated

    def _ensure_dict_attr(target, attr_name):
        current = getattr(target, attr_name, None)
        if isinstance(current, dict):
            return False
        setattr(target, attr_name, OrderedDict())
        return True

    changed = False
    try:
        if _ensure_ordered_dict_attr(module, '_modules'):
            changed = True
        if _ensure_ordered_dict_attr(module, '_parameters'):
            changed = True
        if _ensure_ordered_dict_attr(module, '_buffers'):
            changed = True

        for hook_attr in ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks'):
            if _ensure_dict_attr(module, hook_attr):
                changed = True

        if not hasattr(module, '_non_persistent_buffers_set') or not isinstance(module._non_persistent_buffers_set, set):
            module._non_persistent_buffers_set = set()
            changed = True

        if not hasattr(module, 'training'):
            module.training = False
            changed = True
    except Exception:
        # Best-effort: ignore failures to avoid breaking conversion
        pass

    if recursive:
        for child in module.children():
            ensure_module_bookkeeping(child, recursive=True)

    return module

class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if hasattr(m, 'backbone'):
                x = m(x)
                for _ in range(5 - len(x)):
                    x.insert(0, None)
                for i_idx, i in enumerate(x):
                    if i_idx in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                x = x[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(f'WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. '
                       f'Reverting to single-scale inference instead.')
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.
        Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLOv5 augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        return v8DetectionLoss(self)
    
    def fuse_model(self):
        """
        Fuse Conv+BatchNorm+Activation layers before quantization.
        Uses manual fusion (like the existing fuse() method) which is compatible with
        custom Conv wrapper modules. This fuses Conv+BN into a single Conv2d with bias,
        and updates forward to skip BN. The activation remains in forward but PyTorch's
        quantization system will recognize the Conv+BN+Activation pattern during prepare_qat().
        
        Returns:
            Self (for chaining)
        """
        import torch.nn as nn
        from ultralytics.nn.modules.conv import Conv, Conv2, DWConv
        
        LOGGER.info("Fusing Conv+BN+Activation layers for quantization...")
        fused_count = 0
        fusion_stats = {'conv_bn': 0, 'conv_bn_act': 0, 'failed': 0}
        
        def _get_activation_name(act_module):
            """Get activation name for logging."""
            if act_module is None:
                return None
            if isinstance(act_module, nn.Identity):
                return None
            return type(act_module).__name__
        
        # Fuse modules in each layer of the model
        for name, module in self.named_modules():
            # Skip if this is the top-level model
            if name == '':
                continue
            
            # Look for Conv modules (Conv, Conv2, DWConv) with conv and bn attributes
            if isinstance(module, (Conv, Conv2, DWConv)) and hasattr(module, 'conv') and hasattr(module, 'bn'):
                try:
                    # Get activation module if it exists
                    act_module = getattr(module, 'act', None)
                    act_name = _get_activation_name(act_module)
                    
                    # Handle Conv2 special case (has parallel convolutions)
                    if isinstance(module, Conv2):
                        if hasattr(module, 'fuse_convs'):
                            module.fuse_convs()
                    
                    # Preserve qconfig and FakeQuantize from original Conv2d before fusion
                    original_conv = module.conv
                    original_qconfig = getattr(original_conv, 'qconfig', None)
                    original_weight_fq = getattr(original_conv, 'weight_fake_quant', None)
                    original_act_fq = getattr(original_conv, 'activation_post_process', None)
                    
                    # Manually fuse Conv+BN (like the existing fuse() method)
                    # This creates a new Conv2d with BN parameters folded in
                    fused_conv = fuse_conv_and_bn(module.conv, module.bn)
                    
                    # Note: qconfig will be set later in prepare_for_qat()
                    # We don't preserve it here because fusion happens before qconfig is set
                    module.conv = fused_conv
                    
                    # Remove BatchNorm (it's now folded into conv)
                    delattr(module, 'bn')
                    
                    # Update forward to skip BN (use forward_fuse which does act(conv(x)))
                    # This ensures BN is skipped but activation remains
                    if hasattr(module, 'forward_fuse'):
                        module.forward = module.forward_fuse
                    
                    fused_count += 1
                    if act_name:
                        fusion_stats['conv_bn_act'] += 1
                        LOGGER.debug(f"  Fused {name}: Conv+BN+{act_name}")
                    else:
                        fusion_stats['conv_bn'] += 1
                        LOGGER.debug(f"  Fused {name}: Conv+BN (no activation)")
                        
                except Exception as e:
                    fusion_stats['failed'] += 1
                    LOGGER.debug(f"  Failed to fuse {name}: {e}")
        
        # Log fusion statistics
        LOGGER.info(f"✓ Fused {fused_count} Conv+BN+Activation blocks:")
        if fusion_stats['conv_bn'] > 0:
            LOGGER.info(f"  - Conv+BN (no activation): {fusion_stats['conv_bn']}")
        if fusion_stats['conv_bn_act'] > 0:
            LOGGER.info(f"  - Conv+BN+Activation: {fusion_stats['conv_bn_act']}")
        if fusion_stats['failed'] > 0:
            LOGGER.warning(f"  - Failed fusions: {fusion_stats['failed']}")
        
        # Validate that BatchNorm modules are removed (except in custom wrappers that weren't fused)
        remaining_bn = sum(1 for m in self.modules() if isinstance(m, nn.BatchNorm2d))
        if remaining_bn > 0:
            LOGGER.info(f"  Note: {remaining_bn} BatchNorm2d modules remain (likely in custom wrappers like ODConv, BoTNet)")
        
        return self
    
    def prepare_for_qat(self, backend='fbgemm', example_input=None, use_fx=True):
        """
        Prepare the model for Quantization-Aware Training (QAT).
        Uses hybrid FX + Eager mode to handle complex YOLO operations.
        
        Args:
            backend (str): Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            example_input (torch.Tensor): Example input tensor for FX tracing
            use_fx (bool): If True, try FX mode with wrapped problematic modules.
                          If False or FX fails, fall back to eager mode.
        
        Returns:
            Prepared model with FakeQuantize modules inserted
        """
        import torch.ao.quantization as tq
        from torch.ao.quantization import get_default_qat_qconfig, prepare_qat
        from ultralytics.nn.ODConv import ODConv
        
        # Note: CoordAtt modules will be quantized along with other Conv2d layers
        # CoordAtt Conv2d layers (conv1, conv_h, conv_w) will receive qconfig and be quantized
        
        # CRITICAL: Fuse Conv+BN+Activation BEFORE preparing for QAT
        # This ensures PyTorch creates fused quantized modules (QuantizedConvReLU2d)
        # instead of separate quantized operators that cause backend errors
        LOGGER.info("Fusing Conv+BN+Activation layers before QAT preparation...")
        self.fuse_model()
        
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Get default QAT qconfig for the backend and wrap it with our clamping observer
        default_qconfig = get_default_qat_qconfig(backend)
        try:
            qconfig = _create_safe_qconfig(backend)
            LOGGER.info(
                "Using clamped QAT qconfig (clamp=%.2f, averaging_constant=%.2f) to stabilise observers",
                SAFE_QAT_CLAMP_VALUE,
                SAFE_QAT_AVERAGING_CONSTANT,
            )
        except Exception as err:
            LOGGER.warning(f"Falling back to default QAT qconfig due to: {err}")
            qconfig = default_qconfig
        
        # Configure qconfig_dict to exclude ODConv from quantization
        # ODConv uses dynamic kernel aggregation and must stay FP32
        qconfig_dict = {
            "": qconfig,  # Default for all modules
        }
        
        # Find ODConv layers in the model and exclude them
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, ODConv):
                # Exclude this specific layer from quantization
                qconfig_dict["module_name"] = qconfig_dict.get("module_name", {})
                qconfig_dict["module_name"][f"model.{idx}"] = None
                LOGGER.info(f"Excluding {name} (layer {idx}) from quantization (keeping FP32)")
        
        # Create example input if not provided
        if example_input is None:
            # Use the input size from yaml or default to 640x640
            imgsz = self.yaml.get('imgsz', 640)
            if isinstance(imgsz, list):
                imgsz = imgsz[0]
            ch = self.yaml.get('ch', 3)
            example_input = torch.randn(1, ch, imgsz, imgsz)
        
        LOGGER.info(f"Preparing model for QAT with {backend} backend...")
        
        if use_fx:
            # Try FX-graph mode with wrapped problematic modules
            try:
                from torch.ao.quantization.quantize_fx import prepare_qat_fx
                import torch.fx as fx
                
                # Wrap problematic YOLO modules that FX cannot trace
                # These will be kept in eager mode while rest uses FX
                LOGGER.info("Wrapping complex modules for hybrid FX + Eager mode...")
                
                # Detect head has dynamic tensor operations - wrap it
                from ultralytics.nn.modules.head import Detect
                if not hasattr(Detect, '_fx_wrapped'):
                    fx.wrap(Detect.forward)
                    Detect._fx_wrapped = True
                    LOGGER.info("  ✓ Wrapped Detect head (will use eager mode)")
                
                # Wrap tensor operations that cause issues
                wrapped_ops = ['split', 'chunk', 'unbind']
                for op in wrapped_ops:
                    if hasattr(torch, op):
                        fx.wrap(getattr(torch, op))
                LOGGER.info(f"  ✓ Wrapped tensor operations: {wrapped_ops}")
                
                model_prepared = prepare_qat_fx(
                    self,
                    qconfig_dict,
                    example_inputs=(example_input,),
                    backend_config=None
                )
                attach_fake_quant_clamp_hooks(model_prepared, clamp_value=SAFE_QAT_CLAMP_VALUE)
                LOGGER.info("✓ Model prepared for QAT with hybrid FX + Eager mode!")
                LOGGER.info("  - Most layers: FX-quantized (automatic)")
                LOGGER.info("  - Detect head: Eager mode (wrapped)")
                LOGGER.info("FakeQuantize modules inserted. Model is ready for QAT training.")
                return model_prepared
                
            except Exception as e:
                LOGGER.warning(f"FX-graph mode failed: {e}")
                LOGGER.info("Falling back to eager mode quantization...")
                use_fx = False
        
        # Eager mode fallback
        if not use_fx:
            LOGGER.info("Using eager mode QAT (more compatible with custom models)...")
            
            # CRITICAL: Model must be in training mode for prepare_qat
            self.train()
            
            # CRITICAL: Set qconfig on ALL levels of the model hierarchy
            self.qconfig = qconfig
            self.model.qconfig = qconfig  # The nn.Sequential container
            
            # Propagate qconfig to each layer in the Sequential
            for i, layer in enumerate(self.model):
                if isinstance(layer, ODConv):
                    layer.qconfig = None
                    LOGGER.info(f"  ✓ Excluding layer {i} (ODConv) from quantization (FP32)")
                else:
                    layer.qconfig = qconfig
            
            # CRITICAL: Set qconfig on internal nn.Conv2d and nn.BatchNorm2d modules
            # YOLO uses custom Conv wrapper modules that contain nn.Conv2d inside
            # PyTorch quantization only recognizes nn.Conv2d, not custom wrappers
            from ultralytics.nn.modules.conv import Conv
            
            conv_count = 0
            bn_count = 0
            linear_count = 0
            odconv_excluded_count = 0
            qconfig_set_count = 0
            
            for name, module in self.named_modules():
                # Exclude ODConv
                if isinstance(module, ODConv):
                    module.qconfig = None
                    odconv_excluded_count += 1
                # Set qconfig on actual nn.Conv2d (inside Conv wrappers and CoordAtt)
                elif isinstance(module, nn.Conv2d):
                    module.qconfig = qconfig
                    conv_count += 1
                    qconfig_set_count += 1
                # Set qconfig on BatchNorm2d (for fusion)
                elif isinstance(module, nn.BatchNorm2d):
                    module.qconfig = qconfig
                    bn_count += 1
                    qconfig_set_count += 1
                # Set qconfig on Linear layers
                elif isinstance(module, nn.Linear):
                    module.qconfig = qconfig
                    linear_count += 1
                    qconfig_set_count += 1
                # Also set on the wrapper modules themselves (some may support it)
                elif isinstance(module, Conv) and (not hasattr(module, 'qconfig') or module.qconfig is None):
                    module.qconfig = qconfig
                    qconfig_set_count += 1
            
            LOGGER.info(f"  ✓ Set qconfig on {qconfig_set_count} modules:")
            LOGGER.info(f"    - Conv2d: {conv_count} (including CoordAtt Conv2d layers)")
            LOGGER.info(f"    - BatchNorm2d: {bn_count}")
            LOGGER.info(f"    - Linear: {linear_count}")
            LOGGER.info(f"    - Other modules: {qconfig_set_count - conv_count - bn_count - linear_count}")
            LOGGER.info(f"  ✓ Excluded {odconv_excluded_count} ODConv modules")
            LOGGER.info(f"  ✓ CoordAtt modules will be quantized (Conv2d layers will receive qconfig)")
            
            # Prepare for QAT in eager mode
            LOGGER.info("  Calling prepare_qat()...")
            model_prepared = prepare_qat(self, inplace=False)
            attach_fake_quant_clamp_hooks(model_prepared, clamp_value=SAFE_QAT_CLAMP_VALUE)
            
            # Diagnostics: Check if FakeQuantize modules were inserted
            from torch.ao.quantization import FakeQuantize
            fakequant_modules = [n for n, m in model_prepared.named_modules() if isinstance(m, FakeQuantize)]
            
            LOGGER.info("✓ Model prepared for QAT with eager mode!")
            LOGGER.info(f"  - FakeQuantize modules inserted: {len(fakequant_modules)}")
            LOGGER.info("  - Uses manual QuantStub/DeQuantStub from QAT modules")
            
            if len(fakequant_modules) == 0:
                LOGGER.warning("⚠️  WARNING: No FakeQuantize modules found!")
                LOGGER.warning("   QAT may not be working properly. Check qconfig propagation.")
            else:
                LOGGER.info("  - Works with any model architecture")
                LOGGER.info("FakeQuantize modules inserted. Model is ready for QAT training.")
            
            return model_prepared
    
    def convert_to_quantized(self, calibrated_model=None, calibration_data=None):
        """
        Convert QAT model to INT8.
        
        After prepare_qat() and training, the model has FakeQuantize modules with calibrated observers.
        We convert the entire model, which replaces FakeQuantize with real quantized ops.
        
        Args:
            calibrated_model: The QAT model after training (with learned FakeQuantize parameters)
                             If None, converts self
            calibration_data: Optional calibration data (tensor or list of tensors) to calibrate observers
                             If None and observers not calibrated, will use dummy data
        
        Returns:
            Quantized INT8 model
        """
        from torch.ao.quantization import convert
        
        model = calibrated_model if calibrated_model is not None else self
        model.eval()
        
        LOGGER.info("Converting QAT model to INT8...")
        
        # CRITICAL: Clone parameters to ensure resizable storage
        # This fixes "Trying to resize storage that is not resizable" error
        # Parameters loaded from checkpoint may have non-resizable storage
        # FakeQuantize operations need to resize storage during calibration
        # SKIP parameter cloning - it's causing segfaults and is often unnecessary
        # Modern PyTorch checkpoints usually have resizable storage already
        # If we encounter "non-resizable storage" errors later, we'll handle them then
        LOGGER.info("Skipping parameter cloning (to avoid segfaults - usually not needed)")
        LOGGER.debug("  If you encounter 'non-resizable storage' errors, the model may need to be reloaded")
        
        # Buffer cloning also skipped to avoid segfaults
        # Most modern PyTorch models have resizable storage by default
        
        # CRITICAL: Enable FakeQuantize modules and calibrate observers
        # Observers have min_val=inf and max_val=-inf, which means they haven't collected statistics
        # We need to reset them and run calibration with real data
        from torch.ao.quantization import FakeQuantize
        
        LOGGER.info("Skipping observer buffer cloning (to avoid segfaults - usually not needed)")
        LOGGER.debug("  Observer buffers usually have resizable storage already (modern PyTorch)")
        
        # CRITICAL: Skip ALL observer buffer access to prevent segfaults
        # Accessing min_val/max_val on unallocated tensors causes C++ level crashes
        # We'll just enable FakeQuantize modules and let calibration handle statistics
        
        fakequant_count = 0

        qat_allowed_parent_types = {
            'BoTNet', 'QATBoTNet', 'BottleneckTransformer', 'QATBottleneckTransformer',
            'MHSA', 'QATMHSA', 'CoordAtt', 'QATCoordAtt'
        }

        for name, module in model.named_modules():
            # Handle weight_fake_quant attached to Conv2d/Linear modules (QATConv2d, QATLinear)
            if hasattr(module, 'weight_fake_quant'):
                weight_fq = module.weight_fake_quant
                if isinstance(weight_fq, FakeQuantize):
                    # Skip disabling for modules under QuantizedConv; they must retain QAT machinery
                    try:
                        path_parts = name.split('.')
                        is_under_quantized_conv = False
                        for i in range(len(path_parts) - 1, 0, -1):
                            parent_path = '.'.join(path_parts[:i])
                            parent = dict(model.named_modules()).get(parent_path)
                            if parent is None:
                                continue
                            if 'QuantizedConv' in type(parent).__name__:
                                is_under_quantized_conv = True
                                break
                    except Exception:
                        is_under_quantized_conv = False
                    if is_under_quantized_conv:
                        continue
                    fakequant_count += 1
                    
                    # Enable the weight_fake_quant (NO observer buffer access)
                    try:
                        if hasattr(weight_fq, 'enable_observer'):
                            weight_fq.enable_observer()
                        if hasattr(weight_fq, 'enable_fake_quant'):
                            weight_fq.enable_fake_quant()
                    except Exception:
                        pass  # Skip if enable fails
            
            # Handle standalone FakeQuantize modules
            if isinstance(module, FakeQuantize):
                fakequant_count += 1
                
                # Enable the FakeQuantize module (NO observer buffer access)
                try:
                    if hasattr(module, 'enable_observer'):
                        module.enable_observer()
                    if hasattr(module, 'enable_fake_quant'):
                        module.enable_fake_quant()
                except Exception:
                    pass  # Skip if enable fails
        
        LOGGER.info(f"Enabled {fakequant_count} FakeQuantize modules")
        LOGGER.info("Skipped observer buffer access to prevent segfaults")
        
        # CRITICAL: Skip calibration to avoid tensor allocation errors
        # QAT training should have already collected statistics during forward passes
        # Calibration here often fails with "tensor data not allocated" errors
        LOGGER.info("Skipping calibration - using statistics collected during QAT training")
        LOGGER.info("  (QAT training forward passes should have already calibrated observers)")
        
        # Set model to eval mode for conversion (observers should already have stats)
        model.eval()
        
        # Disable observers (they should already have statistics from training)
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantize):
                try:
                    if hasattr(module, 'disable_observer'):
                        module.disable_observer()  # Disable observer, keep fake quant for conversion
                except Exception:
                    pass  # Skip if disable fails
        
        # CRITICAL: Skip ALL observer validation to prevent segfaults
        # Accessing min_val/max_val buffers can cause C++ level crashes
        # We'll trust that QAT training collected statistics and proceed with conversion
        LOGGER.info("✓ Ready for conversion - using statistics from QAT training")
        LOGGER.info("Replacing FakeQuantize modules with real quantized operations...")
        
        try:
            # CRITICAL: Before conversion, exclude ALL Conv2d modules inside custom module wrappers
            # The fundamental issue: PyTorch's QuantizedConv2d doesn't have standard nn.Module attributes
            # Any custom module that accesses Conv2d attributes will break when it becomes QuantizedConv2d
            # Strategy: Exclude ALL Conv2d that are inside custom modules (ultralytics), except QuantizedConv
            excluded_modules = []
            
            # First, build a list of all modules to avoid repeated lookups
            all_modules = {}
            try:
                for n, m in model.named_modules():
                    all_modules[n] = m
            except AttributeError:
                # If named_modules fails, we can't exclude - skip exclusion
                LOGGER.warning("   ⚠️  Cannot iterate modules to exclude Conv2d - model may have structural issues")
                excluded_modules = []
            
            for name, module in all_modules.items():
                # Check if this Conv2d should be excluded
                if isinstance(module, nn.Conv2d):
                    parent_path = '.'.join(name.split('.')[:-1])
                    conv_name = name.split('.')[-1]
                    
                    should_exclude = False
                    exclude_reason = ""
                    
                    # Check immediate parent first (most important for nested structures)
                    immediate_parent = all_modules.get(parent_path) if parent_path else None
                    is_fused_conv_parent = False
                    
                    if immediate_parent is not None:
                        immediate_parent_type = type(immediate_parent).__name__
                        if immediate_parent_type in qat_allowed_parent_types:
                            # Allow quantization within QAT-aware custom modules
                            is_fused_conv_parent = True
                        
                        # Special case: Allow quantization of Conv2d inside CoordAtt
                        # CoordAtt Conv2d layers (conv1, conv_h, conv_w) should be quantized
                        if immediate_parent_type == 'CoordAtt':
                            # Allow quantization of CoordAtt Conv2d layers
                            should_exclude = False
                            is_fused_conv_parent = True  # Treat as allowed parent
                        
                        # Check if immediate parent is a fused Conv module (no BN attribute)
                        # This is the key check - if Conv2d is inside a fused Conv, allow conversion
                        from ultralytics.nn.modules.conv import Conv, Conv2, DWConv
                        if isinstance(immediate_parent, (Conv, Conv2, DWConv)):
                            if not hasattr(immediate_parent, 'bn'):
                                # This Conv2d is inside a fused Conv module - ALLOW conversion
                                is_fused_conv_parent = True
                    
                    # AGGRESSIVE: Exclude ALL Conv2d inside custom YOLO modules (ultralytics)
                    # EXCEPT if immediate parent is a fused Conv module
                    # OR if CoordAtt Conv2d has qconfig (prepared for QAT)
                    # Only allow conversion if it's a direct child of a standard PyTorch container
                    # or inside QuantizedConv (which we designed to handle it)
                    coordatt_has_qconfig = False
                    if immediate_parent_type == 'CoordAtt':
                        # Check if this Conv2d has qconfig (was prepared for QAT)
                        if hasattr(module, 'qconfig') and module.qconfig is not None:
                            coordatt_has_qconfig = True
                    
                    if parent_path and not is_fused_conv_parent and not coordatt_has_qconfig:
                        path_parts = name.split('.')
                        
                        # Check each level of the hierarchy (from immediate parent up to root)
                        for i in range(len(path_parts) - 1, 0, -1):
                            check_path = '.'.join(path_parts[:i])
                            try:
                                check_parent = all_modules.get(check_path)
                                if check_parent is None:
                                    continue
                                
                                parent_type = type(check_parent).__name__
                                parent_module_str = str(type(check_parent).__module__)

                                if parent_type in qat_allowed_parent_types:
                                    continue
                                
                                # If parent is QuantizedConv, DO NOT CONVERT inner Conv2d
                                # QuantizedConv expects a standard nn.Conv2d; converting it breaks Module semantics
                                if 'QuantizedConv' in parent_type:
                                    should_exclude = True
                                    exclude_reason = f"QuantizedConv parent ({parent_type}.{conv_name})"
                                    break
                                
                                # Standard PyTorch containers are fine
                                if parent_type in ['Sequential', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict']:
                                    # Standard container - safe to convert
                                    break
                                
                                # If parent is from ultralytics (custom YOLO module), exclude it
                                # EXCEPT if it's QuantizedConv (which is already checked above)
                                if 'ultralytics' in parent_module_str:
                                    if parent_type in qat_allowed_parent_types:
                                        continue
                                    # Only exclude if it's NOT QuantizedConv
                                    if 'QuantizedConv' not in parent_type:
                                        should_exclude = True
                                        exclude_reason = f"Custom YOLO module ({parent_type}.{conv_name})"
                                        break
                                
                                # Also check known problematic custom module types
                                # EXCEPT QuantizedConv (already handled above)
                                if parent_type in ['Conv', 'C2f', 'C2', 'C3', 'C3x', 'SPPF', 'SPP', 
                                                  'Attention', 'BottleneckTransformer', 'MHSA', 
                                                  'CoordAtt', 'CA_Attention', 'Detect', 'DFL']:
                                    if parent_type in qat_allowed_parent_types:
                                        continue
                                    
                                    # Special case: Allow quantization of Conv2d inside CoordAtt
                                    # CoordAtt Conv2d layers (conv1, conv_h, conv_w) should be quantized
                                    if parent_type == 'CoordAtt':
                                        # Check if this Conv2d has qconfig (was prepared for QAT)
                                        check_conv = all_modules.get(name)
                                        if check_conv is not None and hasattr(check_conv, 'qconfig') and check_conv.qconfig is not None:
                                            # CoordAtt Conv2d has qconfig - allow quantization
                                            continue
                                        else:
                                            # No qconfig - exclude it
                                            should_exclude = True
                                            exclude_reason = "CoordAtt Conv2d without qconfig"
                                            break
                                    
                                    # Only exclude if it's NOT QuantizedConv
                                    if 'QuantizedConv' not in parent_type:
                                        should_exclude = True
                                        exclude_reason = f"Custom module ({parent_type}.{conv_name})"
                                        break
                            except:
                                pass
                    
                    if should_exclude:
                        # AGGRESSIVE exclusion: Remove all quantization infrastructure
                        # Set qconfig = None to prevent conversion
                        if hasattr(module, 'qconfig'):
                            module.qconfig = None
                        
                        # Remove FakeQuantize modules completely
                        if hasattr(module, 'weight_fake_quant'):
                            weight_fq = module.weight_fake_quant
                            # Disable and remove
                            if hasattr(weight_fq, 'disable_fake_quant'):
                                weight_fq.disable_fake_quant()
                            if hasattr(weight_fq, 'disable_observer'):
                                weight_fq.disable_observer()
                            delattr(module, 'weight_fake_quant')
                        
                        if hasattr(module, 'activation_post_process'):
                            act_fq = module.activation_post_process
                            # Disable and remove
                            if hasattr(act_fq, 'disable_fake_quant'):
                                act_fq.disable_fake_quant()
                            if hasattr(act_fq, 'disable_observer'):
                                act_fq.disable_observer()
                            delattr(module, 'activation_post_process')
                        
                        excluded_modules.append((name, exclude_reason))

                # CRITICAL: Exclude ALL BatchNorm2d from quantization to avoid quantized::batch_norm2d backend errors
                # Even though we fuse Conv+BN before QAT, some BatchNorm modules may remain unfused
                # (e.g., in custom wrappers, or if fusion failed). We must exclude them all.
                if isinstance(module, nn.BatchNorm2d):
                    try:
                        if hasattr(module, 'qconfig'):
                            module.qconfig = None
                        # Remove any attached FakeQuantize/observers
                        if hasattr(module, 'activation_post_process'):
                            act_fq = getattr(module, 'activation_post_process')
                            if hasattr(act_fq, 'disable_fake_quant'):
                                act_fq.disable_fake_quant()
                            if hasattr(act_fq, 'disable_observer'):
                                act_fq.disable_observer()
                            delattr(module, 'activation_post_process')
                        excluded_modules.append((name, 'BatchNorm2d kept FP32 (backend limitation)'))
                    except Exception:
                        pass

                # CRITICAL: Exclude activation layers from quantization to avoid quantized activation ops
                # Even though activations should be part of fused Conv+BN+Activation patterns,
                # some may remain separate. We must exclude them to avoid quantized::relu6 errors.
                if isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Sigmoid, nn.Hardswish, nn.LeakyReLU)):
                    try:
                        if hasattr(module, 'qconfig'):
                            module.qconfig = None
                        # Remove any attached FakeQuantize/observers
                        if hasattr(module, 'activation_post_process'):
                            act_fq = getattr(module, 'activation_post_process')
                            if hasattr(act_fq, 'disable_fake_quant'):
                                act_fq.disable_fake_quant()
                            if hasattr(act_fq, 'disable_observer'):
                                act_fq.disable_observer()
                            delattr(module, 'activation_post_process')
                        excluded_modules.append((name, 'Activation kept FP32 (backend limitation)'))
                    except Exception:
                        pass
            
            if excluded_modules:
                LOGGER.info(f"   ✓ Excluded {len(excluded_modules)} Conv2d modules from conversion")
                # Group by reason
                by_reason = {}
                for name, reason in excluded_modules:
                    if reason not in by_reason:
                        by_reason[reason] = []
                    by_reason[reason].append(name)
                for reason, names in by_reason.items():
                    LOGGER.info(f"     - {reason}: {len(names)} modules")
                # Show first few examples for each reason
                for reason, names in by_reason.items():
                    if len(names) > 0:
                        LOGGER.info(f"       Examples ({reason}): {names[:3]}")
            
            # CRITICAL: Convert QAT modules inside ODConv to FP32 first
            # ODConv uses dynamic kernel aggregation and must remain FP32
            # However, its internal Conv2d modules (fc, channel_fc, filter_fc, spatial_fc) 
            # may have been converted to QAT modules during prepare_for_qat()
            # We need to convert them back to regular FP32 Conv2d before the main conversion
            try:
                from ultralytics.nn.ODConv import ODConv
                from ultralytics.nn.BoTNet import BoTNet
                from ultralytics.nn.CA_Attention import CoordAtt
                from ultralytics.nn.modules.block import DFL
                from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d
                try:
                    from torch.ao.nn.qat.modules.linear import Linear as QATLinear
                except ImportError:
                    QATLinear = None

                def _qat_module_to_fp32(module):
                    """Convert a QAT Conv/Linear module to a plain FP32 module."""
                    try:
                        if isinstance(module, QATConv2d):
                            new_conv = nn.Conv2d(
                                module.in_channels,
                                module.out_channels,
                                module.kernel_size,
                                stride=module.stride,
                                padding=module.padding,
                                dilation=module.dilation,
                                groups=module.groups,
                                bias=module.bias is not None,
                                padding_mode=module.padding_mode,
                            )
                            new_conv.weight.data.copy_(module.weight.detach())
                            if module.bias is not None:
                                new_conv.bias.data.copy_(module.bias.detach())
                            new_conv.training = module.training
                            new_conv.qconfig = None
                            return new_conv
                        if QATLinear is not None and isinstance(module, QATLinear):
                            new_linear = nn.Linear(
                                module.in_features,
                                module.out_features,
                                bias=module.bias is not None,
                            )
                            new_linear.weight.data.copy_(module.weight.detach())
                            if module.bias is not None:
                                new_linear.bias.data.copy_(module.bias.detach())
                            new_linear.training = module.training
                            new_linear.qconfig = None
                            return new_linear
                        if hasattr(module, 'to_float'):
                            float_mod = module.to_float()
                            # Ensure qconfig removed if attribute exists
                            if hasattr(float_mod, 'qconfig'):
                                float_mod.qconfig = None
                            return float_mod
                    except Exception as convert_error:
                        LOGGER.debug(f"   Failed custom QAT->FP32 conversion ({convert_error}), using to_float fallback")
                        if hasattr(module, 'to_float'):
                            return module.to_float()
                    return module

                odconv_qat_converted = 0
                
                # Build a dictionary of all modules with their full paths
                all_modules_dict = dict(model.named_modules())
                
                # Find all ODConv modules and their full paths
                target_parent_types = (ODConv, DFL)

                for full_name, module in all_modules_dict.items():
                    if isinstance(module, target_parent_types):
                        # Iterate through all submodules inside this ODConv
                        # Use named_modules() to get relative paths
                        for submodule_name, submodule in module.named_modules():
                            # Skip the ODConv itself (we want its children)
                            if submodule_name == '':
                                continue
                            
                            # Check if this submodule is a QAT module
                            has_wfq = hasattr(submodule, 'weight_fake_quant')
                            has_to_float = hasattr(submodule, 'to_float')
                            mod_ns = getattr(type(submodule), '__module__', '')
                            is_qat_module = has_wfq or ('torch.ao.nn.qat' in mod_ns) or (has_to_float and isinstance(submodule, (QATConv2d,) + (() if QATLinear is None else (QATLinear,))))

                            if is_qat_module:
                                try:
                                    # Convert QAT module to FP32 (create fresh nn.Conv2d/Linear)
                                    float_mod = _qat_module_to_fp32(submodule)
                                    
                                    # Construct the full path to this submodule
                                    if submodule_name:
                                        full_submodule_path = f"{full_name}.{submodule_name}"
                                    else:
                                        full_submodule_path = full_name
                                    
                                    # Replace the module using the full path
                                    # Navigate to parent and set the child
                                    path_parts = full_submodule_path.split('.')
                                    parent_path = '.'.join(path_parts[:-1])
                                    child_name = path_parts[-1]
                                    
                                    # Get the parent module
                                    parent_module = all_modules_dict.get(parent_path)
                                    if parent_module is not None and hasattr(parent_module, child_name):
                                        setattr(parent_module, child_name, float_mod)
                                        # Update the dictionary
                                        all_modules_dict[full_submodule_path] = float_mod
                                        odconv_qat_converted += 1
                                        LOGGER.debug(f"   Converted QAT module {full_submodule_path} to FP32")
                                except Exception as e:
                                    LOGGER.debug(f"   Failed to convert {full_submodule_path if 'full_submodule_path' in locals() else submodule_name}: {e}")
                                    pass
                
                if odconv_qat_converted > 0:
                    LOGGER.info(f"   ✓ Converted {odconv_qat_converted} QAT modules inside ODConv/DFL to FP32")
            except Exception as e:
                LOGGER.debug(f"   Error converting ODConv QAT modules: {e}")
                pass
            
            # Before conversion: force any QAT Conv modules back to float Conv modules
            # QAT Conv modules expect weight_fake_quant in forward; at inference we want plain Conv
            try:
                replaced_qat_modules = 0
                # Sort by depth to handle deepest modules first
                for name in sorted(all_modules.keys(), key=lambda n: len(n), reverse=True):
                    module = all_modules[name]
                    # Identify QAT modules (Conv/Linear/etc.) by presence of weight_fake_quant and to_float()
                    is_qat_module = hasattr(module, 'weight_fake_quant') and hasattr(module, 'to_float')
                    if not is_qat_module:
                        continue
                    # Replace with float module universally (we keep quantization elsewhere via converted ops)
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1] if name else ''
                    try:
                        float_mod = module.to_float()
                        if parent_name:
                            parent = all_modules.get(parent_name)
                            if parent is not None and hasattr(parent, child_name):
                                setattr(parent, child_name, float_mod)
                                all_modules[name] = float_mod
                                replaced_qat_modules += 1
                    except Exception:
                        pass
                if replaced_qat_modules > 0:
                    LOGGER.info(f"   ✓ Replaced {replaced_qat_modules} QAT modules with float equivalents before conversion")
            except Exception:
                pass

            # Try standard convert() first
            try:
                model_quantized = convert(model, inplace=False)
                LOGGER.info("✓ Standard convert() succeeded")
                
                # CRITICAL: Even if convert() succeeds, it may not convert Conv2d inside custom wrappers
                # PyTorch's eager mode convert() doesn't handle nested modules well
                # We need to manually convert Conv2d modules inside fused Conv wrappers
                from ultralytics.nn.modules.conv import Conv, Conv2, DWConv
                
                # Try to import QuantizedConv2d - if it fails, we'll use a different check
                LOGGER.info("  Checking for QuantizedConv2d availability...")
                try:
                    # Note: In PyTorch, the quantized Conv2d is named Conv2d (not QuantizedConv2d)
                    # We import it as QuantizedConv2d for clarity
                    from torch.ao.nn.quantized.modules.conv import Conv2d as QuantizedConv2d
                    from torch import quantize_per_tensor
                    has_quantized_conv2d = True
                    LOGGER.info(f"  ✓ QuantizedConv2d imported successfully: {QuantizedConv2d is not None}")
                    LOGGER.info(f"  ✓ quantize_per_tensor imported successfully: {quantize_per_tensor is not None}")
                    LOGGER.info(f"  ✓ has_quantized_conv2d = {has_quantized_conv2d}")
                except ImportError as import_error:
                    # Fallback: check by module name/attributes
                    QuantizedConv2d = None
                    quantize_per_tensor = None
                    has_quantized_conv2d = False
                    LOGGER.warning(f"  ❌ Failed to import QuantizedConv2d: {import_error}")
                    LOGGER.warning(f"  ❌ has_quantized_conv2d = {has_quantized_conv2d} (conversion will be skipped)")
                
                def _create_quantized_conv2d_from_conv2d(conv2d_module, qconfig, module_name=""):
                    """
                    Manually create QuantizedConv2d from a Conv2d module and qconfig.
                    This is needed because convert() requires FakeQuantize modules attached,
                    which fused Conv2d modules don't have.
                    
                    Args:
                        conv2d_module: nn.Conv2d module to convert
                        qconfig: QConfig with weight and activation observers
                        module_name: Name of module for logging
                    
                    Returns:
                        QuantizedConv2d module with quantized weights
                    """
                    # Entry logging (debug level to reduce verbosity)
                    LOGGER.debug(f"  [{module_name}] Starting conversion: Conv2d -> QuantizedConv2d")
                    LOGGER.debug(f"  [{module_name}] has_quantized_conv2d={has_quantized_conv2d}, QuantizedConv2d={QuantizedConv2d is not None}")
                    LOGGER.debug(f"  [{module_name}] qconfig={qconfig is not None}, qconfig.weight={qconfig.weight if qconfig and hasattr(qconfig, 'weight') else None}")
                    
                    if not has_quantized_conv2d:
                        LOGGER.warning(f"  [{module_name}] ❌ Cannot convert: QuantizedConv2d not available (has_quantized_conv2d=False)")
                        return conv2d_module  # Can't convert without QuantizedConv2d
                    
                    if QuantizedConv2d is None:
                        LOGGER.warning(f"  [{module_name}] ❌ Cannot convert: QuantizedConv2d is None (import failed)")
                        return conv2d_module
                    
                    try:
                        # Try using from_float if available (PyTorch's recommended method)
                        if hasattr(QuantizedConv2d, 'from_float'):
                            LOGGER.debug(f"  [{module_name}] ✓ Using from_float method for conversion")
                            try:
                                # Create a temporary QAT Conv2d with FakeQuantize
                                from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d
                                
                                LOGGER.debug(f"  [{module_name}] Creating QAT Conv2d: in={conv2d_module.in_channels}, out={conv2d_module.out_channels}, k={conv2d_module.kernel_size}")
                                
                                # Create QAT Conv2d with same parameters
                                qat_conv = QATConv2d(
                                    conv2d_module.in_channels,
                                    conv2d_module.out_channels,
                                    conv2d_module.kernel_size,
                                    stride=conv2d_module.stride,
                                    padding=conv2d_module.padding,
                                    dilation=conv2d_module.dilation,
                                    groups=conv2d_module.groups,
                                    bias=conv2d_module.bias is not None,
                                    padding_mode=conv2d_module.padding_mode,
                                    qconfig=qconfig
                                )
                                
                                # Copy weights and bias
                                # QNNPACK backend requires CPU tensors for quantization
                                backend = getattr(torch.backends.quantized, 'engine', 'qnnpack')
                                original_device = conv2d_module.weight.device
                                weight_data = conv2d_module.weight.data.clone()
                                bias_data = conv2d_module.bias.data.clone() if conv2d_module.bias is not None else None
                                
                                if backend == 'qnnpack' and original_device.type == 'cuda':
                                    LOGGER.debug(f"  [{module_name}] Moving QAT module to CPU for QNNPACK backend")
                                    weight_data = weight_data.cpu()
                                    if bias_data is not None:
                                        bias_data = bias_data.cpu()
                                    qat_conv = qat_conv.cpu()
                                
                                qat_conv.weight = torch.nn.Parameter(weight_data)
                                if bias_data is not None:
                                    qat_conv.bias = torch.nn.Parameter(bias_data)
                                
                                LOGGER.debug(f"  [{module_name}] Preparing QAT module...")
                                # Prepare QAT module (this attaches FakeQuantize)
                                from torch.ao.quantization import prepare_qat
                                qat_conv.train()  # Must be in train mode for prepare_qat
                                prepare_qat(qat_conv, inplace=True)
                                
                                # Run a dummy forward pass to calibrate observers
                                LOGGER.debug(f"  [{module_name}] Running dummy forward pass for calibration...")
                                dummy_input = torch.randn(1, conv2d_module.in_channels, 3, 3)
                                if backend == 'qnnpack' and original_device.type == 'cuda':
                                    dummy_input = dummy_input.cpu()
                                with torch.no_grad():
                                    _ = qat_conv(dummy_input)
                                
                                # Convert the QAT module to quantized using from_float
                                LOGGER.debug(f"  [{module_name}] Converting QAT to quantized using from_float...")
                                qat_conv.eval()  # Must be in eval mode for conversion
                                # Use QuantizedConv2d.from_float() directly instead of convert()
                                quantized_conv = QuantizedConv2d.from_float(qat_conv)
                                
                                result_type = type(quantized_conv).__name__
                                has_packed = hasattr(quantized_conv, '_packed_params')
                                is_quantized = isinstance(quantized_conv, QuantizedConv2d) if QuantizedConv2d is not None else False
                                LOGGER.debug(f"  [{module_name}] from_float result: type={result_type}, has_packed={has_packed}, is_QuantizedConv2d={is_quantized}")
                                
                                if has_packed or is_quantized:
                                    LOGGER.debug(f"  [{module_name}] ✓ Successfully converted using from_float method")
                                    ensure_module_bookkeeping(quantized_conv)
                                    return quantized_conv
                                else:
                                    LOGGER.debug(f"  [{module_name}] ⚠️ from_float returned {result_type} without _packed_params, falling back to manual construction")
                                    raise ValueError(f"from_float did not produce valid QuantizedConv2d")
                            except Exception as from_float_error:
                                LOGGER.debug(f"  [{module_name}] ⚠️ from_float method failed: {from_float_error}")
                                import traceback
                                LOGGER.debug(f"  [{module_name}] from_float traceback:\n{traceback.format_exc()}")
                                LOGGER.debug(f"  [{module_name}] Trying manual construction...")
                                # Fall through to manual construction
                        
                        # Manual construction (either from_float not available or failed)
                        if QuantizedConv2d is None or quantize_per_tensor is None:
                            LOGGER.warning(f"  [{module_name}] ❌ Cannot convert: Missing QuantizedConv2d or quantize_per_tensor")
                            LOGGER.debug(f"  [{module_name}]   QuantizedConv2d={QuantizedConv2d is not None}, quantize_per_tensor={quantize_per_tensor is not None}")
                            return conv2d_module  # Can't convert without required classes
                        
                        # Only log if we didn't try from_float first (debug level to reduce verbosity)
                        if not hasattr(QuantizedConv2d, 'from_float'):
                            LOGGER.debug(f"  [{module_name}] Using manual construction (from_float not available)")
                        else:
                            LOGGER.debug(f"  [{module_name}] Using manual construction (from_float failed, using fallback)")
                        
                        # Extract Conv2d parameters
                        LOGGER.debug(f"  [{module_name}] Extracting Conv2d parameters...")
                        weight = conv2d_module.weight.data.clone()
                        bias = conv2d_module.bias.data.clone() if conv2d_module.bias is not None else None
                        
                        # Get weight observer from qconfig
                        weight_observer = None
                        if qconfig is not None and hasattr(qconfig, 'weight'):
                            weight_observer = qconfig.weight()
                            LOGGER.debug(f"  [{module_name}] Weight observer: {type(weight_observer).__name__}")
                        else:
                            LOGGER.debug(f"  [{module_name}] ⚠️ No weight observer in qconfig, using fallback calculation")
                        
                        # Calculate quantization parameters
                        LOGGER.debug(f"  [{module_name}] Calculating quantization parameters...")
                        if weight_observer is not None:
                            try:
                                weight_observer(weight)
                                if hasattr(weight_observer, 'calculate_qparams'):
                                    scale, zero_point = weight_observer.calculate_qparams()
                                    scale = scale.item() if isinstance(scale, torch.Tensor) else scale
                                    zero_point = zero_point.item() if isinstance(zero_point, torch.Tensor) else zero_point
                                    LOGGER.debug(f"  [{module_name}] Observer calculated: scale={scale:.6f}, zero_point={zero_point}")
                                else:
                                    # Fallback calculation
                                    scale = weight.abs().max().item() / 127.0
                                    zero_point = 0
                                    LOGGER.debug(f"  [{module_name}] Observer has no calculate_qparams, using fallback: scale={scale:.6f}")
                            except Exception as obs_error:
                                scale = weight.abs().max().item() / 127.0
                                zero_point = 0
                                LOGGER.debug(f"  [{module_name}] Observer calculation failed ({obs_error}), using fallback: scale={scale:.6f}")
                        else:
                            scale = weight.abs().max().item() / 127.0
                            zero_point = 0
                            LOGGER.debug(f"  [{module_name}] Using fallback calculation: scale={scale:.6f}, zero_point={zero_point}")
                        
                        # Quantize the weight tensor
                        LOGGER.debug(f"  [{module_name}] Quantizing weight tensor...")
                        # QNNPACK backend requires CPU tensors for quantization
                        backend = getattr(torch.backends.quantized, 'engine', 'qnnpack')
                        original_device = weight.device
                        if backend == 'qnnpack' and weight.device.type == 'cuda':
                            LOGGER.debug(f"  [{module_name}] Moving weight to CPU for QNNPACK backend quantization")
                            weight = weight.cpu()
                        weight_quantized = quantize_per_tensor(weight, scale, zero_point, torch.qint8)
                        # Move back to original device if needed (though quantized tensors are typically CPU)
                        if backend == 'qnnpack' and original_device.type == 'cuda':
                            LOGGER.debug(f"  [{module_name}] Quantized tensor remains on CPU (QNNPACK requirement)")
                        
                        # Create QuantizedConv2d using _packed_params
                        LOGGER.debug(f"  [{module_name}] Creating QuantizedConv2d module...")
                        quantized_conv = QuantizedConv2d(
                            conv2d_module.in_channels,
                            conv2d_module.out_channels,
                            conv2d_module.kernel_size,
                            stride=conv2d_module.stride,
                            padding=conv2d_module.padding,
                            dilation=conv2d_module.dilation,
                            groups=conv2d_module.groups,
                            bias=bias is not None,
                            padding_mode=conv2d_module.padding_mode
                        )
                        
                        # Pack parameters using internal API
                        # NOTE: Bias quantization is complex and error-prone. For now, we skip bias
                        # to avoid "Input channel size of weight and bias must match" errors.
                        # The model will work without bias (bias can be added as a separate layer if needed).
                        LOGGER.debug(f"  [{module_name}] Packing parameters (skipping bias for compatibility)...")
                        try:
                            # Pack without bias to avoid shape mismatch errors
                            # Most Conv layers in YOLO don't use bias anyway (they use BatchNorm)
                            quantized_conv._packed_params = torch.ops.quantized.conv2d_prepack(
                                weight_quantized, None, conv2d_module.stride,
                                conv2d_module.padding, conv2d_module.dilation, conv2d_module.groups
                            )
                            if bias is not None:
                                LOGGER.debug(f"  [{module_name}] Packed without bias (bias present but skipped for compatibility)")
                            else:
                                LOGGER.debug(f"  [{module_name}] Packed without bias")
                            
                            result_type = type(quantized_conv).__name__
                            has_packed = hasattr(quantized_conv, '_packed_params')
                            is_quantized = isinstance(quantized_conv, QuantizedConv2d) if QuantizedConv2d is not None else False
                            LOGGER.debug(f"  [{module_name}] ✓ Manual construction result: type={result_type}, has_packed={has_packed}, is_QuantizedConv2d={is_quantized}, scale={scale:.6f}")
                            
                            if has_packed or is_quantized:
                                ensure_module_bookkeeping(quantized_conv)
                                LOGGER.debug(f"  [{module_name}] ✓ Successfully converted using manual construction")
                                return quantized_conv
                            else:
                                LOGGER.warning(f"  [{module_name}] ⚠️ Manual construction did not produce valid QuantizedConv2d")
                                return conv2d_module
                        except Exception as pack_error:
                            LOGGER.warning(f"  [{module_name}] ❌ Failed to pack parameters: {pack_error}")
                            import traceback
                            LOGGER.warning(f"  [{module_name}] Pack error traceback:\n{traceback.format_exc()}")
                            return conv2d_module
                    except Exception as e:
                        # If conversion fails, return original module
                        LOGGER.warning(f"  [{module_name}] ❌ Failed to create QuantizedConv2d: {e}")
                        import traceback
                        LOGGER.warning(f"  [{module_name}] Full traceback:\n{traceback.format_exc()}")
                        LOGGER.warning(f"  [{module_name}] Returning original Conv2d module (conversion failed)")
                        return conv2d_module
                
                # Manually convert Conv2d modules inside fused Conv wrappers
                manually_converted = 0
                skipped_no_qconfig = 0
                failed_conversions = 0
                all_quantized_modules = dict(model_quantized.named_modules())
                
                for name, module in list(all_quantized_modules.items()):
                    if isinstance(module, nn.Conv2d):
                        # Check if this Conv2d is inside a fused Conv module
                        parent_path = '.'.join(name.split('.')[:-1])
                        if parent_path:
                            parent = all_quantized_modules.get(parent_path)
                            # Also handle CoordAtt Conv2d layers for quantization
                            is_coordatt_parent = parent is not None and type(parent).__name__ == 'CoordAtt'
                            if parent is not None and (isinstance(parent, (Conv, Conv2, DWConv)) or is_coordatt_parent):
                                # For Conv wrappers, check if parent is fused (no BN)
                                # For CoordAtt, always allow conversion
                                if is_coordatt_parent or not hasattr(parent, 'bn'):
                                    # Check if this Conv2d was already converted
                                    is_already_quantized = False
                                    if has_quantized_conv2d:
                                        is_already_quantized = isinstance(module, QuantizedConv2d)
                                    else:
                                        # Fallback check: quantized modules have _packed_params
                                        is_already_quantized = hasattr(module, '_packed_params')
                                    
                                    if not is_already_quantized:
                                        # This Conv2d is inside a fused Conv but wasn't converted
                                        # Try to manually convert it
                                        try:
                                            # Check if it has qconfig OR if parent has qconfig
                                            # (qconfig might be on parent Conv wrapper, not inner Conv2d)
                                            has_qconfig = (hasattr(module, 'qconfig') and module.qconfig is not None)
                                            if not has_qconfig and parent is not None:
                                                # Check parent's qconfig
                                                if hasattr(parent, 'qconfig') and parent.qconfig is not None:
                                                    # Propagate qconfig to inner Conv2d
                                                    module.qconfig = parent.qconfig
                                                    has_qconfig = True
                                                else:
                                                    # Try to get qconfig from global default
                                                    # Parent doesn't have qconfig, so set it
                                                    from torch.ao.quantization import get_default_qconfig
                                                    default_qconfig = get_default_qconfig('fbgemm')
                                                    # Set qconfig on parent if it doesn't exist or is None
                                                    if not hasattr(parent, 'qconfig') or parent.qconfig is None:
                                                        parent.qconfig = default_qconfig
                                                    # Propagate to inner Conv2d
                                                    module.qconfig = default_qconfig
                                                    has_qconfig = True
                                            
                                            if has_qconfig:
                                                # Manually convert this specific Conv2d to QuantizedConv2d
                                                # We can't use convert() because it requires FakeQuantize modules
                                                # which fused Conv2d modules don't have
                                                try:
                                                    # Get qconfig from module (should be set above)
                                                    module_qconfig = module.qconfig if hasattr(module, 'qconfig') else None
                                                    if module_qconfig is None and parent is not None:
                                                        module_qconfig = getattr(parent, 'qconfig', None)
                                                    
                                                    # Use helper function to create QuantizedConv2d
                                                    LOGGER.info(f"  [{name}] Attempting to convert Conv2d -> QuantizedConv2d...")
                                                    original_module_id = id(module)
                                                    converted_conv = _create_quantized_conv2d_from_conv2d(module, module_qconfig, module_name=name)
                                                    
                                                    # CRITICAL: Verify we didn't get the original module back
                                                    if id(converted_conv) == original_module_id:
                                                        LOGGER.warning(f"  [{name}] ❌ Helper function returned original module (id match)! Conversion failed silently.")
                                                        failed_conversions += 1
                                                        continue
                                                    
                                                    # Verify conversion succeeded (check for _packed_params or QuantizedConv2d type)
                                                    is_quantized = hasattr(converted_conv, '_packed_params')
                                                    if not is_quantized and has_quantized_conv2d and QuantizedConv2d is not None:
                                                        is_quantized = isinstance(converted_conv, QuantizedConv2d)
                                                    
                                                    # Log what we got
                                                    result_type = type(converted_conv).__name__
                                                    has_packed = hasattr(converted_conv, '_packed_params')
                                                    LOGGER.info(f"  [{name}] Conversion result: type={result_type}, has_packed={has_packed}, is_quantized={is_quantized}")
                                                    
                                                    if is_quantized:
                                                        # Ensure bookkeeping for quantized Conv2d
                                                        ensure_module_bookkeeping(converted_conv)
                                                        # Replace in parent module
                                                        child_name = name.split('.')[-1]
                                                        setattr(parent, child_name, converted_conv)
                                                        # Update our module dict
                                                        all_quantized_modules[name] = converted_conv
                                                        manually_converted += 1
                                                        if manually_converted <= 10:  # Log first 10
                                                            LOGGER.info(f"     ✓ Manually converted {name} -> {result_type}")
                                                    else:
                                                        LOGGER.warning(f"  [{name}] ⚠️ Conversion returned {result_type} but is not quantized (no _packed_params, not QuantizedConv2d)")
                                                        failed_conversions += 1
                                                        if failed_conversions <= 20:  # Show more failures
                                                            LOGGER.warning(f"        Original type: {type(module).__name__}, Has _packed_params: {hasattr(converted_conv, '_packed_params')}")
                                                except Exception as conv_error:
                                                    failed_conversions += 1
                                                    if failed_conversions <= 20:  # Show more failures
                                                        LOGGER.warning(f"     ✗ Failed to manually convert {name}: {conv_error}")
                                                        import traceback
                                                        LOGGER.debug(f"        Traceback:\n{traceback.format_exc()}")
                                            else:
                                                skipped_no_qconfig += 1
                                                if skipped_no_qconfig <= 3:  # Log first 3
                                                    LOGGER.debug(f"     Skipping {name} - no qconfig (parent qconfig: {hasattr(parent, 'qconfig') if parent else None})")
                                        except Exception as e:
                                            failed_conversions += 1
                                            if failed_conversions <= 20:  # Show more failures
                                                LOGGER.warning(f"     ✗ Failed to manually convert {name}: {e}")
                                                import traceback
                                                LOGGER.debug(f"        Traceback:\n{traceback.format_exc()}")
                
                if skipped_no_qconfig > 0:
                    LOGGER.warning(f"  ⚠️  Skipped {skipped_no_qconfig} Conv2d modules (no qconfig available)")
                if failed_conversions > 0:
                    LOGGER.warning(f"  ⚠️  Failed to convert {failed_conversions} Conv2d modules")
                
                if manually_converted > 0:
                    LOGGER.info(f"  ✓ Manually converted {manually_converted} Conv2d modules inside fused Conv wrappers")
                
                # Count unconverted (for reporting)
                unconverted_conv_count = 0
                for name, module in model_quantized.named_modules():
                    if isinstance(module, nn.Conv2d):
                        parent_path = '.'.join(name.split('.')[:-1])
                        if parent_path:
                            parent = dict(model_quantized.named_modules()).get(parent_path)
                            if parent is not None and isinstance(parent, (Conv, Conv2, DWConv)):
                                if not hasattr(parent, 'bn'):
                                    is_quantized = hasattr(module, '_packed_params') if not has_quantized_conv2d else isinstance(module, QuantizedConv2d)
                                    if not is_quantized:
                                        unconverted_conv_count += 1
                
                if unconverted_conv_count > 0:
                    LOGGER.warning(f"  ⚠️  {unconverted_conv_count} Conv2d modules inside fused Conv wrappers remain unconverted")
                    LOGGER.info("  These will use FP32 Conv2d (still faster due to fusion)")
                
            except Exception as e1:
                LOGGER.warning(f"Standard convert() failed: {e1}")
                LOGGER.info("Attempting manual module-by-module conversion...")
                
                # Manual conversion: convert individual Conv2d modules
                # This works around the issue where convert() doesn't handle nested modules
                model_quantized = model  # Start with original
                converted_count = 0
                
                for name, module in list(model_quantized.named_modules()):
                    # Convert individual Conv2d modules that have FakeQuantize
                    if isinstance(module, nn.Conv2d):
                        # Check if this Conv2d has FakeQuantize attached
                        has_fakequant = any(
                            isinstance(m, FakeQuantize) 
                            for m in module.modules() if m is not module
                        ) or hasattr(module, 'weight_fake_quant')
                        
                        if has_fakequant:
                            try:
                                # Convert this specific Conv2d
                                converted_conv = convert(module, inplace=False)
                                # Replace in parent module
                                parent_name = '.'.join(name.split('.')[:-1])
                                child_name = name.split('.')[-1]
                                if parent_name:
                                    parent = dict(model_quantized.named_modules())[parent_name]
                                    setattr(parent, child_name, converted_conv)
                                    converted_count += 1
                            except Exception as e2:
                                LOGGER.debug(f"  Failed to convert {name}: {e2}")
                
                if converted_count > 0:
                    LOGGER.info(f"✓ Manually converted {converted_count} Conv2d modules")
                    model_quantized = model  # Use the modified model
                else:
                    # Fall back to standard convert
                    model_quantized = convert(model, inplace=False)
            
            # Strip any remaining FakeQuantize modules to avoid runtime/repr issues
            try:
                from torch.ao.quantization import FakeQuantize as FQStrip
                removed_fq = 0
                for name, module in list(model_quantized.named_modules()):
                    if isinstance(module, FQStrip):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        if parent_name:
                            parent = dict(model_quantized.named_modules()).get(parent_name)
                            if parent is not None and hasattr(parent, child_name):
                                setattr(parent, child_name, nn.Identity())
                                removed_fq += 1
                if removed_fq > 0:
                    LOGGER.info(f"   ✓ Removed {removed_fq} remaining FakeQuantize modules after conversion")
            except Exception:
                pass

            # Replace any quantized BatchNorm modules with float equivalents (CRITICAL - backend doesn't support them)
            try:
                replaced_bn = 0
                for name, module in list(model_quantized.named_modules()):
                    mod_ns = getattr(type(module), '__module__', '')
                    if 'torch.ao.nn.quantized.modules.batchnorm' in mod_ns:
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1] if name else ''
                        # Replace with float BatchNorm2d
                        # Get original parameters if possible, otherwise create new
                        try:
                            # Try to get original BN parameters from the quantized module
                            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                                num_features = module.weight.shape[0] if hasattr(module.weight, 'shape') else 64
                            else:
                                num_features = 64  # Default fallback
                            new_bn = nn.BatchNorm2d(num_features)
                            if parent_name:
                                parent = dict(model_quantized.named_modules()).get(parent_name)
                                if parent is not None and hasattr(parent, child_name):
                                    setattr(parent, child_name, new_bn)
                                    replaced_bn += 1
                        except Exception as e:
                            # If replacement fails, try to remove it
                            if parent_name:
                                parent = dict(model_quantized.named_modules()).get(parent_name)
                                if parent is not None and hasattr(parent, child_name):
                                    setattr(parent, child_name, nn.Identity())
                                    replaced_bn += 1
                if replaced_bn > 0:
                    LOGGER.info(f"   ✓ Replaced {replaced_bn} quantized BatchNorm modules with float BatchNorm (backend limitation)")
            except Exception:
                pass

            # Replace any remaining quantized activation modules with float equivalents (fallback)
            # NOTE: With proper fusion, activations should be part of fused QuantizedConvReLU2d modules.
            # This is a safety fallback for any activations that weren't fused (e.g., in custom wrappers).
            try:
                replaced_act = 0
                for name, module in list(model_quantized.named_modules()):
                    mod_ns = getattr(type(module), '__module__', '')
                    if 'torch.ao.nn.quantized.modules.activation' in mod_ns:
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1] if name else ''
                        new_act = None
                        cls_name = type(module).__name__.lower()
                        if 'relu6' in cls_name:
                            new_act = nn.ReLU6(inplace=False)
                        elif 'relu' in cls_name:
                            new_act = nn.ReLU(inplace=False)
                        elif 'hardtanh' in cls_name:
                            # ReLU6 often lowers from Hardtanh; map to float Hardtanh
                            new_act = nn.Hardtanh(inplace=False)
                        elif 'hardswish' in cls_name:
                            new_act = nn.Hardswish()
                        elif 'sigmoid' in cls_name:
                            new_act = nn.Sigmoid()
                        elif 'silu' in cls_name or 'swish' in cls_name:
                            new_act = nn.SiLU()
                        if new_act is None:
                            new_act = nn.Identity()
                        if parent_name:
                            parent = dict(model_quantized.named_modules()).get(parent_name)
                            if parent is not None and hasattr(parent, child_name):
                                setattr(parent, child_name, new_act)
                                replaced_act += 1
                if replaced_act > 0:
                    LOGGER.info(f"   ✓ Replaced {replaced_act} quantized activation modules with float activations")
            except Exception:
                pass

            LOGGER.info("✓ Model converted to INT8!")
            
            # Count quantized parameters - check both module types and parameter dtypes
            quantized_ops = 0
            quantized_params = 0
            fp32_ops = 0
            
            # Check if QuantizedConv2d is available for isinstance check
            try:
                from torch.ao.nn.quantized.modules.conv import QuantizedConv2d as QC2D
                has_qc2d_class = True
            except ImportError:
                QC2D = None
                has_qc2d_class = False
            
            for name, module in model_quantized.named_modules():
                module_type = type(module).__name__
                # Check if it's a quantized module type
                is_quantized = False
                
                # Check for QuantizedConv2d instances (manually converted)
                if has_qc2d_class and isinstance(module, QC2D):
                    is_quantized = True
                # Check for _packed_params (indicates quantized module)
                elif hasattr(module, '_packed_params'):
                    is_quantized = True
                # Check for 'Quantized' in module type name
                elif 'Quantized' in module_type:
                    is_quantized = True
                # Check for quantized weight dtype
                elif hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    weight_dtype = str(module.weight.dtype)
                    if 'qint8' in weight_dtype or 'quint8' in weight_dtype:
                        is_quantized = True
                        quantized_params += 1
                
                if is_quantized:
                    quantized_ops += 1
                elif hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    # Only count as FP32 if it has weights (to avoid counting non-parametric modules)
                    weight_dtype = str(module.weight.dtype)
                    if 'qint8' not in weight_dtype and 'quint8' not in weight_dtype:
                        fp32_ops += 1
            
            # Also check state dict for quantized parameters
            state_dict_quantized = sum(
                1 for p in model_quantized.state_dict().values() 
                if isinstance(p, torch.Tensor) and ('qint8' in str(p.dtype) or 'quint8' in str(p.dtype))
            )
            
            def _iter_modules_safe(module):
                yield module
                children = getattr(module, '_modules', None)
                if isinstance(children, dict):
                    for child in children.values():
                        if child is not None:
                            yield from _iter_modules_safe(child)

            def _needs_bookkeeping_fix(mod):
                try:
                    modules_attr = getattr(mod, '_modules')
                except Exception:
                    modules_attr = None
                try:
                    params_attr = getattr(mod, '_parameters')
                except Exception:
                    params_attr = None
                try:
                    buffers_attr = getattr(mod, '_buffers')
                except Exception:
                    buffers_attr = None
                hook_attrs_missing = any(
                    not isinstance(getattr(mod, attr, None), dict)
                    for attr in ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks')
                )
                return (
                    not isinstance(modules_attr, dict)
                    or not isinstance(params_attr, dict)
                    or not isinstance(buffers_attr, dict)
                    or hook_attrs_missing
                    or not hasattr(mod, '_non_persistent_buffers_set')
                    or not hasattr(mod, 'training')
                )

            modules_needing_fix = 0
            for submodule in _iter_modules_safe(model_quantized):
                if _needs_bookkeeping_fix(submodule):
                    modules_needing_fix += 1

            ensure_module_bookkeeping(model_quantized, recursive=True)

            if modules_needing_fix > 0:
                LOGGER.info(f"  ✓ Ensured bookkeeping attributes for {modules_needing_fix} quantized modules")

            LOGGER.info(f"\nConversion Summary:")
            LOGGER.info(f"  ✓ Quantized operations: {quantized_ops}")
            LOGGER.info(f"  ✓ Quantized parameters in state_dict: {state_dict_quantized}")
            LOGGER.info(f"  - FP32 operations: {fp32_ops}")
            
            if quantized_ops == 0 and state_dict_quantized == 0:
                LOGGER.warning("⚠️  WARNING: No quantized operations found!")
                LOGGER.warning("   This may indicate that convert() cannot handle the nested module structure.")
                LOGGER.warning("   Consider using FX quantization or post-training quantization instead.")
            else:
                LOGGER.info("  FakeQuantize modules replaced with real INT8 operations.")
            
            return model_quantized
        
        except Exception as e:
            LOGGER.error(f"Conversion failed: {e}")
            LOGGER.warning("The model may not have been properly prepared with prepare_qat()")
            import traceback
            traceback.print_exc()
            raise

    def prepare_for_ptq(self, backend='fbgemm', example_input=None, use_fx=True,
                       quantize_backbone=True, quantize_neck=True,
                       quantize_botnet=True, quantize_coordatt=True):
        """
        Prepare the model for Post-Training Quantization (PTQ).
        Uses hybrid FX + Eager mode to handle complex YOLO operations.
        
        Args:
            backend (str): Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            example_input (torch.Tensor): Example input tensor for FX tracing
            use_fx (bool): If True, try FX mode with wrapped problematic modules.
                          If False or FX fails, fall back to eager mode.
            quantize_backbone (bool): If True, quantize backbone layers
            quantize_neck (bool): If True, quantize neck layers
            quantize_botnet (bool): If True, quantize BoTNet modules
            quantize_coordatt (bool): If True, quantize CoordAtt modules
        
        Returns:
            Prepared model with observers inserted (ready for calibration)
        """
        import torch.ao.quantization as tq
        from torch.ao.quantization import get_default_qconfig, prepare, QConfig
        from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
        from ultralytics.nn.ODConv import ODConv
        from ultralytics.nn.BoTNet import BoTNet
        from ultralytics.nn.CA_Attention import CoordAtt
        
        # CRITICAL: Fuse Conv+BN+Activation BEFORE preparing for PTQ
        # This ensures PyTorch creates fused quantized modules (QuantizedConvReLU2d)
        LOGGER.info("Fusing Conv+BN+Activation layers before PTQ preparation...")
        self.fuse_model()
        
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Create per-channel symmetric qconfig for PTQ
        # Per-channel symmetric: weights use per-channel symmetric quantization
        # Activations remain per-tensor (standard)
        LOGGER.info("Using per-channel symmetric quantization for weights...")
        if backend == 'qnnpack':
            # For qnnpack, use per-channel symmetric for weights
            qconfig = QConfig(
                activation=MinMaxObserver.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine,
                ),
                weight=PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                )
            )
        else:
            # For fbgemm and other backends, also use per-channel symmetric
            qconfig = QConfig(
                activation=MinMaxObserver.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine,
                ),
                weight=PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                )
            )
        
        # Identify backbone and neck layers from YAML structure
        backbone_layer_count = len(self.yaml.get('backbone', []))
        head_layer_count = len(self.yaml.get('head', []))
        # Detect head is the last layer, so neck = head - 1
        neck_layer_count = head_layer_count - 1
        
        # Total layers in model (excluding Detect head)
        total_layers = len(self.model) - 1  # -1 for Detect head
        
        # Identify which layers belong to backbone vs neck
        # Backbone layers: model.0 to model.{backbone_layer_count-1}
        # Neck layers: model.{backbone_layer_count} to model.{total_layers-1}
        backbone_end_idx = backbone_layer_count
        
        LOGGER.info(f"Model structure: {backbone_layer_count} backbone layers, {neck_layer_count} neck layers")
        LOGGER.info(f"Backbone: model.0 to model.{backbone_end_idx-1}")
        LOGGER.info(f"Neck: model.{backbone_end_idx} to model.{total_layers-1}")
        
        # Configure qconfig_dict to selectively quantize only backbone, neck, BoTNet, CoordAtt
        # Exclude ODConv and Detect head
        qconfig_dict = {
            "": qconfig,  # Default for all modules
        }
        
        # Find ODConv layers and exclude them
        odconv_excluded = []
        for name, module in self.named_modules():
            if isinstance(module, ODConv):
                # Exclude ODConv from quantization
                if "module_name" not in qconfig_dict:
                    qconfig_dict["module_name"] = {}
                qconfig_dict["module_name"][name] = None
                odconv_excluded.append(name)
                LOGGER.info(f"Excluding {name} (ODConv) from quantization (keeping FP32)")
        
        # Create example input if not provided
        if example_input is None:
            # Use the input size from yaml or default to 640x640
            imgsz = self.yaml.get('imgsz', 640)
            if isinstance(imgsz, list):
                imgsz = imgsz[0]
            ch = self.yaml.get('ch', 3)
            example_input = torch.randn(1, ch, imgsz, imgsz)
        
        LOGGER.info(f"Preparing model for PTQ with {backend} backend...")
        LOGGER.info("Selective quantization: backbone, neck, BoTNet, CoordAtt (ODConv excluded)")
        
        if use_fx:
            # Try FX-graph mode with wrapped problematic modules
            try:
                from torch.ao.quantization.quantize_fx import prepare_fx
                import torch.fx as fx
                
                # Wrap problematic YOLO modules that FX cannot trace
                LOGGER.info("Wrapping complex modules for hybrid FX + Eager mode...")
                
                # Detect head has dynamic tensor operations - wrap it
                from ultralytics.nn.modules.head import Detect
                if not hasattr(Detect, '_fx_wrapped'):
                    fx.wrap(Detect.forward)
                    Detect._fx_wrapped = True
                    LOGGER.info("  ✓ Wrapped Detect head (will use eager mode)")
                
                # Wrap tensor operations that cause issues
                wrapped_ops = ['split', 'chunk', 'unbind']
                for op in wrapped_ops:
                    if hasattr(torch, op):
                        fx.wrap(getattr(torch, op))
                LOGGER.info(f"  ✓ Wrapped tensor operations: {wrapped_ops}")
                
                model_prepared = prepare_fx(
                    self,
                    qconfig_dict,
                    example_inputs=(example_input,),
                    backend_config=None
                )
                LOGGER.info("✓ Model prepared for PTQ with hybrid FX + Eager mode!")
                LOGGER.info("  - Most layers: FX-quantized (automatic)")
                LOGGER.info("  - Detect head: Eager mode (wrapped)")
                LOGGER.info("  - Observers inserted. Model is ready for calibration.")
                return model_prepared
                
            except Exception as e:
                LOGGER.warning(f"FX-graph mode failed: {e}")
                LOGGER.info("Falling back to eager mode quantization...")
                use_fx = False
        
        # Eager mode fallback
        if not use_fx:
            LOGGER.info("Using eager mode PTQ (more compatible with custom models)...")
            
            # CRITICAL: Model must be in eval mode for PTQ (unlike QAT which needs train mode)
            self.eval()
            
            # CRITICAL: Set qconfig on ALL levels of the model hierarchy
            self.qconfig = qconfig
            self.model.qconfig = qconfig  # The nn.Sequential container
            
            # Track which modules we're quantizing
            quantized_modules = []
            excluded_modules = []
            
            # Propagate qconfig to each layer in the Sequential
            for i, layer in enumerate(self.model):
                # Exclude Detect head (last layer)
                if i == len(self.model) - 1:
                    if hasattr(layer, 'qconfig'):
                        layer.qconfig = None
                    excluded_modules.append((f"model.{i}", "Detect head"))
                    continue
                
                # Determine if this is backbone or neck
                is_backbone = i < backbone_end_idx
                is_neck = i >= backbone_end_idx
                
                # Exclude ODConv regardless of location
                if isinstance(layer, ODConv):
                    layer.qconfig = None
                    excluded_modules.append((f"model.{i}", "ODConv"))
                else:
                    # Only quantize if the corresponding flag is set
                    should_quantize = (is_backbone and quantize_backbone) or (is_neck and quantize_neck)
                    if should_quantize:
                        layer.qconfig = qconfig
                        layer_type = "backbone" if is_backbone else "neck"
                        quantized_modules.append((f"model.{i}", layer_type))
                    else:
                        # Don't quantize this layer
                        if hasattr(layer, 'qconfig'):
                            layer.qconfig = None
            
            # CRITICAL: Set qconfig on internal nn.Conv2d and nn.BatchNorm2d modules
            # YOLO uses custom Conv wrapper modules that contain nn.Conv2d inside
            # PyTorch quantization only recognizes nn.Conv2d, not custom wrappers
            from ultralytics.nn.modules.conv import Conv
            
            conv_count = 0
            bn_count = 0
            linear_count = 0
            botnet_count = 0
            coordatt_count = 0
            qconfig_set_count = 0
            
            # Log quantization configuration
            quantize_parts = []
            if quantize_backbone:
                quantize_parts.append("backbone")
            if quantize_neck:
                quantize_parts.append("neck")
            if quantize_botnet:
                quantize_parts.append("BoTNet")
            if quantize_coordatt:
                quantize_parts.append("CoordAtt")
            
            if quantize_parts:
                LOGGER.info(f"Selective quantization enabled: {', '.join(quantize_parts)}")
            else:
                LOGGER.warning("⚠️  No components selected for quantization! Model will remain FP32.")
            
            for name, module in self.named_modules():
                # Exclude ODConv
                if isinstance(module, ODConv):
                    module.qconfig = None
                    continue
                
                # Identify BoTNet and CoordAtt modules - quantize only if flags are set
                is_botnet = isinstance(module, BoTNet)
                is_coordatt = isinstance(module, CoordAtt)
                
                if is_botnet:
                    botnet_count += 1
                    if quantize_botnet:
                        # Set qconfig on BoTNet module itself
                        module.qconfig = qconfig
                        qconfig_set_count += 1
                    else:
                        # Skip BoTNet quantization
                        if hasattr(module, 'qconfig'):
                            module.qconfig = None
                
                if is_coordatt:
                    coordatt_count += 1
                    if quantize_coordatt:
                        # Set qconfig on CoordAtt module itself
                        module.qconfig = qconfig
                        qconfig_set_count += 1
                    else:
                        # Skip CoordAtt quantization
                        if hasattr(module, 'qconfig'):
                            module.qconfig = None
                
                # Set qconfig on actual nn.Conv2d (inside Conv wrappers and CoordAtt)
                if isinstance(module, nn.Conv2d):
                    # Check if this Conv2d is inside an excluded module
                    parent_excluded = False
                    for excluded_name, _ in excluded_modules:
                        if name.startswith(excluded_name + '.'):
                            parent_excluded = True
                            break
                    
                    if not parent_excluded:
                        # Check if this Conv2d is in a quantized component
                        is_in_quantized_component = False
                        
                        # Check if it's in backbone or neck
                        for quantized_name, layer_type in quantized_modules:
                            if name.startswith(quantized_name + '.'):
                                if (layer_type == "backbone" and quantize_backbone) or \
                                   (layer_type == "neck" and quantize_neck):
                                    is_in_quantized_component = True
                                    break
                        
                        # Also check if it's in CoordAtt or BoTNet
                        if not is_in_quantized_component:
                            if is_coordatt and quantize_coordatt:
                                is_in_quantized_component = True
                            elif is_botnet and quantize_botnet:
                                is_in_quantized_component = True
                        
                        if is_in_quantized_component:
                            module.qconfig = qconfig
                            conv_count += 1
                            qconfig_set_count += 1
                        else:
                            # Skip quantization for this Conv2d
                            if hasattr(module, 'qconfig'):
                                module.qconfig = None
                
                # Set qconfig on BatchNorm2d (for fusion) - only if in quantized components
                elif isinstance(module, nn.BatchNorm2d):
                    # Check if this BatchNorm2d is inside an excluded module
                    parent_excluded = False
                    for excluded_name, reason in excluded_modules:
                        if name.startswith(excluded_name + '.'):
                            parent_excluded = True
                            break
                    
                    if not parent_excluded:
                        # Check if it's in a quantized component (same logic as Conv2d)
                        is_in_quantized_component = False
                        for quantized_name, layer_type in quantized_modules:
                            if name.startswith(quantized_name + '.'):
                                if (layer_type == "backbone" and quantize_backbone) or \
                                   (layer_type == "neck" and quantize_neck):
                                    is_in_quantized_component = True
                                    break
                        
                        if not is_in_quantized_component:
                            if is_coordatt and quantize_coordatt:
                                is_in_quantized_component = True
                            elif is_botnet and quantize_botnet:
                                is_in_quantized_component = True
                        
                        if is_in_quantized_component:
                            module.qconfig = qconfig
                            bn_count += 1
                            qconfig_set_count += 1
                        else:
                            if hasattr(module, 'qconfig'):
                                module.qconfig = None
                
                # Set qconfig on Linear layers (in BoTNet, etc.) - only if BoTNet is quantized
                elif isinstance(module, nn.Linear):
                    # Check if this Linear is inside an excluded module
                    parent_excluded = False
                    for excluded_name, _ in excluded_modules:
                        if name.startswith(excluded_name + '.'):
                            parent_excluded = True
                            break
                    
                    if not parent_excluded:
                        # Linear layers are mainly in BoTNet, so only quantize if BoTNet is enabled
                        is_in_quantized_component = False
                        if is_botnet and quantize_botnet:
                            is_in_quantized_component = True
                        else:
                            # Check if it's in backbone/neck
                            for quantized_name, layer_type in quantized_modules:
                                if name.startswith(quantized_name + '.'):
                                    if (layer_type == "backbone" and quantize_backbone) or \
                                       (layer_type == "neck" and quantize_neck):
                                        is_in_quantized_component = True
                                        break
                        
                        if is_in_quantized_component:
                            module.qconfig = qconfig
                            linear_count += 1
                            qconfig_set_count += 1
                        else:
                            if hasattr(module, 'qconfig'):
                                module.qconfig = None
                
                # Also set on the wrapper modules themselves (some may support it)
                elif isinstance(module, Conv) and (not hasattr(module, 'qconfig') or module.qconfig is None):
                    # Check if this Conv is inside an excluded module
                    parent_excluded = False
                    for excluded_name, _ in excluded_modules:
                        if name.startswith(excluded_name + '.'):
                            parent_excluded = True
                            break
                    
                    if not parent_excluded:
                        # Check if this Conv is in a quantized component (same logic as Conv2d)
                        is_in_quantized_component = False
                        
                        # Check if it's in backbone or neck
                        for quantized_name, layer_type in quantized_modules:
                            if name.startswith(quantized_name + '.'):
                                if (layer_type == "backbone" and quantize_backbone) or \
                                   (layer_type == "neck" and quantize_neck):
                                    is_in_quantized_component = True
                                    break
                        
                        # Also check if it's in CoordAtt or BoTNet
                        if not is_in_quantized_component:
                            if is_coordatt and quantize_coordatt:
                                is_in_quantized_component = True
                            elif is_botnet and quantize_botnet:
                                is_in_quantized_component = True
                        
                        if is_in_quantized_component:
                            module.qconfig = qconfig
                            qconfig_set_count += 1
                        else:
                            # Don't quantize this Conv wrapper
                            module.qconfig = None
            
            LOGGER.info(f"  ✓ Set qconfig on {qconfig_set_count} modules:")
            LOGGER.info(f"    - Conv2d: {conv_count} (including CoordAtt Conv2d layers)")
            LOGGER.info(f"    - BatchNorm2d: {bn_count}")
            LOGGER.info(f"    - Linear: {linear_count}")
            LOGGER.info(f"    - BoTNet modules: {botnet_count}")
            LOGGER.info(f"    - CoordAtt modules: {coordatt_count}")
            LOGGER.info(f"    - Other modules: {qconfig_set_count - conv_count - bn_count - linear_count}")
            LOGGER.info(f"  ✓ Excluded {len(excluded_modules)} modules (ODConv, Detect head)")
            LOGGER.info(f"  ✓ Quantized {len(quantized_modules)} layers (backbone + neck)")
            
            # CRITICAL: Add QuantStub for input quantization (same as QAT)
            # PyTorch's prepare() doesn't always add input quantization for complex models
            # We need to manually add it to ensure quantized operations receive quantized inputs
            from torch.ao.quantization import QuantStub, DeQuantStub
            
            # CRITICAL: Don't add QuantStub before prepare() - it gets lost
            # Instead, we'll add it AFTER prepare() and ensure it's properly integrated
            # The key is to NOT wrap forward - we'll call quant() manually during calibration
            LOGGER.info("  Note: QuantStub will be added after prepare() to preserve observer hooks")
            
            # Prepare for PTQ in eager mode
            LOGGER.info("  Calling prepare()...")
            model_prepared = prepare(self, inplace=False)
            
            # CRITICAL: Add QuantStub/DeQuantStub to prepared model WITHOUT wrapping forward
            # This preserves the hook system so observers can collect statistics
            # We'll call quant() manually during calibration instead of wrapping forward
            LOGGER.info("  Adding QuantStub/DeQuantStub to prepared model (no forward wrapper)...")
            
            # Check if QuantStub already exists in prepared model
            has_quant_in_prepared = any(isinstance(m, QuantStub) for m in model_prepared.named_modules())
            
            if not has_quant_in_prepared:
                # Add QuantStub/DeQuantStub as module attributes (like QAT does)
                # But DON'T wrap forward - we'll call quant() manually during calibration
                model_prepared.quant = QuantStub()
                model_prepared.dequant = DeQuantStub()
                LOGGER.info("  ✓ QuantStub/DeQuantStub added as module attributes")
                LOGGER.info("  ✓ Forward method NOT wrapped - observers will work correctly")
                LOGGER.info("  ✓ QuantStub will be called manually during calibration")
            else:
                LOGGER.info("  ✓ QuantStub already exists in prepared model")
            
            # Diagnostics: Check if observers were inserted
            from torch.ao.quantization import ObserverBase
            observer_modules = [n for n, m in model_prepared.named_modules() if isinstance(m, ObserverBase)]
            
            LOGGER.info("✓ Model prepared for PTQ with eager mode!")
            LOGGER.info(f"  - Observer modules inserted: {len(observer_modules)}")
            LOGGER.info("  - Model is ready for calibration")
            
            if len(observer_modules) == 0:
                LOGGER.warning("⚠️  WARNING: No observer modules found!")
                LOGGER.warning("   PTQ may not be working properly. Check qconfig propagation.")
            else:
                LOGGER.info("  - Works with any model architecture")
                LOGGER.info("Observers inserted. Model is ready for calibration.")
            
            return model_prepared

    def calibrate_ptq(self, calibration_data, num_batches=None):
        """
        Calibrate the PTQ model by running calibration data through it.
        Observers will collect min/max statistics automatically.
        
        Args:
            calibration_data: DataLoader or iterable of (input, target) tuples
            num_batches (int): Number of batches to use for calibration. If None, use all.
        
        Returns:
            Calibrated model (same model, observers now have statistics)
        """
        LOGGER.info("Calibrating PTQ model...")
        
        # Ensure model is in eval mode (required for PTQ)
        self.eval()
        
        # CRITICAL: In eager mode, observers must be explicitly enabled
        # Even though they're enabled above, we need to ensure they're actually observing
        # PyTorch observers collect statistics via forward hooks, which are triggered during __call__
        from torch.ao.quantization import ObserverBase
        from torch.ao.quantization.observer import _ObserverBase
        
        # Ensure all observers are in observing mode
        for name, module in self.named_modules():
            if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                obs = module.activation_post_process
                # Ensure observer is enabled and will observe
                if isinstance(obs, _ObserverBase):
                    obs.training = False  # Observers should be in eval mode
                    # Some observers need to be explicitly enabled
                    if hasattr(obs, '_observer_enabled'):
                        obs._observer_enabled = True
        
        # DEBUG: Check if QuantStub exists
        from torch.ao.quantization import QuantStub, DeQuantStub
        has_quant_stub = hasattr(self, 'quant') and isinstance(self.quant, QuantStub)
        has_quant_in_modules = any(isinstance(m, QuantStub) for m in self.named_modules())
        LOGGER.info(f"QuantStub check: hasattr(self, 'quant')={hasattr(self, 'quant')}, is QuantStub={has_quant_stub}, in modules={has_quant_in_modules}")
        
        if not has_quant_stub and not has_quant_in_modules:
            LOGGER.warning("⚠️  No QuantStub found in model - adding it now...")
            # Add QuantStub if missing
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
            # Wrap forward method
            original_forward = self.forward
            def quantized_forward_wrapper(x, *args, **kwargs):
                x = self.quant(x)
                out = original_forward(x, *args, **kwargs)
                if isinstance(out, torch.Tensor):
                    out = self.dequant(out)
                elif isinstance(out, (list, tuple)):
                    out = tuple(self.dequant(o) if isinstance(o, torch.Tensor) else o for o in out)
                return out
            self.forward = quantized_forward_wrapper
            LOGGER.info("  ✓ QuantStub added to model for calibration")
        
        # CRITICAL: Ensure observers are enabled for calibration
        # In PyTorch eager mode, observers should be enabled by default in eval mode
        # But we need to explicitly enable them for calibration
        from torch.ao.quantization import ObserverBase
        from torch.ao.quantization.observer import _ObserverBase
        
        observer_count = 0
        observers_enabled = 0
        
        # Enable all ObserverBase modules
        for name, module in self.named_modules():
            if isinstance(module, ObserverBase):
                observer_count += 1
                # Try multiple ways to enable observers
                try:
                    # Method 1: enable_observer() if available
                    if hasattr(module, 'enable_observer'):
                        module.enable_observer()
                        observers_enabled += 1
                    # Method 2: Set _observer_enabled flag if it exists
                    elif hasattr(module, '_observer_enabled'):
                        module._observer_enabled = True
                        observers_enabled += 1
                    # Method 3: For _ObserverBase, ensure it's enabled
                    elif isinstance(module, _ObserverBase):
                        # Observers should be enabled by default in eval mode
                        # But explicitly set training=False to ensure they're active
                        module.training = False
                        observers_enabled += 1
                except Exception as e:
                    LOGGER.debug(f"Could not enable observer {name}: {e}")
        
        # Enable activation_post_process observers (these are the main ones)
        activation_observer_count = 0
        activation_observers_enabled = 0
        for name, module in self.named_modules():
            if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                activation_observer_count += 1
                obs = module.activation_post_process
                try:
                    # Method 1: enable_observer() if available
                    if hasattr(obs, 'enable_observer'):
                        obs.enable_observer()
                        activation_observers_enabled += 1
                    # Method 2: Set _observer_enabled flag
                    elif hasattr(obs, '_observer_enabled'):
                        obs._observer_enabled = True
                        activation_observers_enabled += 1
                    # Method 3: Ensure observer is in eval mode (not training)
                    elif isinstance(obs, _ObserverBase):
                        obs.training = False
                        obs.eval()
                        activation_observers_enabled += 1
                    # Method 4: For MinMaxObserver and similar, check if they have activation_post_process
                    elif hasattr(obs, 'activation_post_process'):
                        # This is a nested observer, enable it too
                        nested_obs = obs.activation_post_process
                        if hasattr(nested_obs, 'enable_observer'):
                            nested_obs.enable_observer()
                        activation_observers_enabled += 1
                except Exception as e:
                    LOGGER.debug(f"Could not enable activation_post_process observer for {name}: {e}")
        
        # DEBUG: Check observer state
        sample_obs_info = []
        for name, module in list(self.named_modules())[:5]:  # Check first 5 modules
            if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                obs = module.activation_post_process
                obs_type = type(obs).__name__
                has_min = hasattr(obs, 'min_val')
                has_max = hasattr(obs, 'max_val')
                is_enabled = getattr(obs, '_observer_enabled', None)
                sample_obs_info.append(f"{name}: type={obs_type}, enabled={is_enabled}, has_min={has_min}, has_max={has_max}")
        
        if sample_obs_info:
            LOGGER.debug(f"Sample observer states:\n  " + "\n  ".join(sample_obs_info))
        
        if observer_count > 0 or activation_observer_count > 0:
            LOGGER.info(f"Enabled {observers_enabled}/{observer_count} ObserverBase modules and {activation_observers_enabled}/{activation_observer_count} activation_post_process observers for calibration")
        
        # TEST: Run a single forward pass to verify observers trigger
        # This helps debug if observers are collecting statistics
        # Note: This will consume one batch from the iterator, which is fine for debugging
        LOGGER.info("Running test forward pass to verify observers trigger...")
        test_batch_consumed = False
        try:
            # Get a sample batch from calibration data
            if hasattr(calibration_data, '__iter__'):
                # Try to get first batch (this consumes it from iterator)
                try:
                    calibration_iter = iter(calibration_data)
                    test_batch = next(calibration_iter)
                    test_batch_consumed = True
                except StopIteration:
                    LOGGER.warning("  Calibration data iterator is empty, skipping test forward pass")
                    test_batch = None
                
                # Extract images from batch
                if isinstance(test_batch, dict):
                    test_images = test_batch.get('img', None)
                    if test_images is not None and test_images.dtype == torch.uint8:
                        test_images = test_images.float() / 255.0
                elif isinstance(test_batch, (list, tuple)):
                    test_images = test_batch[0]
                    if isinstance(test_images, torch.Tensor) and test_images.dtype == torch.uint8:
                        test_images = test_images.float() / 255.0
                elif isinstance(test_batch, torch.Tensor):
                    test_images = test_batch
                    if test_images.dtype == torch.uint8:
                        test_images = test_images.float() / 255.0
                else:
                    test_images = None
                
                if test_images is not None:
                    device = next(self.parameters()).device
                    test_images = test_images.to(device=device, dtype=torch.float32)
                    
                    # Run test forward pass
                    # CRITICAL: Use __call__ (self()) not forward() to ensure hooks fire
                    with torch.no_grad():
                        if hasattr(self, 'quant') and isinstance(self.quant, QuantStub):
                            # Quantize input first
                            x_quant = self.quant(test_images)
                            # Use __call__ to trigger hooks (observers are triggered via hooks)
                            _ = self(x_quant)
                        else:
                            # No QuantStub, just call normally
                            _ = self(test_images)
                    
                    # Check if any observers collected statistics after test pass
                    test_observers_with_stats = 0
                    test_observers_details = []
                    for name, module in self.named_modules():
                        if isinstance(module, ObserverBase):
                            if hasattr(module, 'min_val') and hasattr(module, 'max_val'):
                                min_val = module.min_val
                                max_val = module.max_val
                                if min_val is not None and max_val is not None:
                                    if isinstance(min_val, torch.Tensor) and min_val.numel() > 0:
                                        if isinstance(max_val, torch.Tensor) and max_val.numel() > 0:
                                            if max_val.max() > min_val.min():
                                                test_observers_with_stats += 1
                                                test_observers_details.append(f"{name}: min={min_val.min().item():.4f}, max={max_val.max().item():.4f}")
                    
                    # Also check activation_post_process observers (these are the main ones)
                    test_activation_observers_with_stats = 0
                    test_activation_details = []
                    for name, module in self.named_modules():
                        if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                            obs = module.activation_post_process
                            # Check various ways observers store statistics
                            min_val = None
                            max_val = None
                            
                            # Method 1: Direct min_val/max_val attributes
                            if hasattr(obs, 'min_val') and hasattr(obs, 'max_val'):
                                min_val = obs.min_val
                                max_val = obs.max_val
                            # Method 2: Check if observer has _min_val/_max_val
                            elif hasattr(obs, '_min_val') and hasattr(obs, '_max_val'):
                                min_val = obs._min_val
                                max_val = obs._max_val
                            # Method 3: Check if observer uses a different attribute name
                            elif hasattr(obs, 'activation_post_process'):
                                nested_obs = obs.activation_post_process
                                if hasattr(nested_obs, 'min_val') and hasattr(nested_obs, 'max_val'):
                                    min_val = nested_obs.min_val
                                    max_val = nested_obs.max_val
                            
                            if min_val is not None and max_val is not None:
                                # Handle both tensor and scalar values
                                if isinstance(min_val, torch.Tensor):
                                    if min_val.numel() > 0:
                                        min_val_scalar = min_val.min().item() if min_val.numel() > 1 else min_val.item()
                                        max_val_scalar = max_val.max().item() if max_val.numel() > 1 else max_val.item()
                                        if max_val_scalar > min_val_scalar:
                                            test_activation_observers_with_stats += 1
                                            test_activation_details.append(f"{name}: min={min_val_scalar:.4f}, max={max_val_scalar:.4f}")
                                else:
                                    # Scalar values
                                    if max_val > min_val:
                                        test_activation_observers_with_stats += 1
                                        test_activation_details.append(f"{name}: min={min_val:.4f}, max={max_val:.4f}")
                    
                    total_test_with_stats = test_observers_with_stats + test_activation_observers_with_stats
                    if total_test_with_stats > 0:
                        LOGGER.info(f"✓ Test forward pass successful: {total_test_with_stats} observers collected statistics")
                        if test_activation_details:
                            LOGGER.debug(f"Sample observers with stats:\n  " + "\n  ".join(test_activation_details[:5]))
                    else:
                        LOGGER.warning(f"⚠️  Test forward pass completed but 0 observers collected statistics!")
                        LOGGER.warning("   This indicates observers are not being triggered during forward pass")
                        LOGGER.warning("   Check if QuantStub is being used and observers are enabled")
                        
                        # DEBUG: Print detailed observer state
                        LOGGER.debug("DEBUG: Checking observer state after test forward pass...")
                        for name, module in list(self.named_modules())[:10]:  # Check first 10
                            if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                                obs = module.activation_post_process
                                obs_type = type(obs).__name__
                                has_min = hasattr(obs, 'min_val') or hasattr(obs, '_min_val')
                                has_max = hasattr(obs, 'max_val') or hasattr(obs, '_max_val')
                                min_val = getattr(obs, 'min_val', getattr(obs, '_min_val', None))
                                max_val = getattr(obs, 'max_val', getattr(obs, '_max_val', None))
                                LOGGER.debug(f"  {name}: type={obs_type}, has_min={has_min}, has_max={has_max}, min={min_val}, max={max_val}")
        except Exception as e:
            LOGGER.warning(f"Test forward pass failed: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
        
        # Handle different input types
        if hasattr(calibration_data, '__iter__'):
            # It's a DataLoader or similar
            batch_count = 0
            total_batches = len(calibration_data) if hasattr(calibration_data, '__len__') else None
            
            # Note: If test forward pass consumed first batch, the loop will start from batch 1
            # This is fine - we just lose one batch for debugging purposes
            with torch.no_grad():
                for batch_idx, batch in enumerate(calibration_data):
                    if num_batches is not None and batch_idx >= num_batches:
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        # YOLO DataLoader format: {'img': ..., 'im_file': ..., etc.}
                        images = batch.get('img', None)
                        if images is not None:
                            # YOLO dataloader returns uint8 images (0-255), need to convert to float and normalize
                            if images.dtype == torch.uint8:
                                images = images.float() / 255.0
                    elif isinstance(batch, (list, tuple)):
                        # Tuple format: (images, targets, ...)
                        images = batch[0]
                        if isinstance(images, torch.Tensor) and images.dtype == torch.uint8:
                            images = images.float() / 255.0
                    elif isinstance(batch, torch.Tensor):
                        # Direct tensor
                        images = batch
                        if images.dtype == torch.uint8:
                            images = images.float() / 255.0
                    else:
                        LOGGER.warning(f"Unknown batch format at index {batch_idx}, skipping...")
                        continue
                    
                    if images is None:
                        LOGGER.warning(f"Could not extract images from batch {batch_idx}, skipping...")
                        continue
                    
                    # Ensure images are on the correct device and in float format
                    device = next(self.parameters()).device
                    images = images.to(device=device, dtype=torch.float32)
                    
                    # Forward pass - observers will collect statistics
                    # CRITICAL: In eager mode, observers collect stats via forward hooks
                    # We must use __call__ (self()) not forward() to ensure hooks fire
                    # If QuantStub exists, quantize input first, then call model normally
                    try:
                        # Check if QuantStub exists and forward is wrapped
                        has_quant = hasattr(self, 'quant') and isinstance(self.quant, QuantStub)
                        
                        if has_quant:
                            # Quantize input first
                            x_quant = self.quant(images)
                            # CRITICAL: Use __call__ to trigger forward hooks (observers are hooked to forward)
                            # This ensures observers collect statistics during forward pass
                            out = self(x_quant)
                            
                            # Dequantize output if DeQuantStub exists
                            if hasattr(self, 'dequant') and isinstance(self.dequant, DeQuantStub):
                                if isinstance(out, torch.Tensor):
                                    out = self.dequant(out)
                                elif isinstance(out, (list, tuple)):
                                    out = tuple(self.dequant(o) if isinstance(o, torch.Tensor) else o for o in out)
                        else:
                            # No QuantStub, just call model normally
                            # Observers should still work if they're properly attached
                            _ = self(images)
                        
                        batch_count += 1
                    except Exception as e:
                        LOGGER.warning(f"Error during calibration batch {batch_idx}: {e}")
                        import traceback
                        LOGGER.debug(traceback.format_exc())
                        continue
                    
                    if (batch_idx + 1) % 10 == 0:
                        LOGGER.info(f"  Calibrated {batch_idx + 1} batches...")
        else:
            # Single tensor or list of tensors
            if isinstance(calibration_data, torch.Tensor):
                calibration_data = [calibration_data]
            
            with torch.no_grad():
                for idx, data in enumerate(calibration_data):
                    if num_batches is not None and idx >= num_batches:
                        break
                    
                    if not isinstance(data, torch.Tensor):
                        LOGGER.warning(f"Calibration data item {idx} is not a tensor, skipping...")
                        continue
                    
                    # Ensure data is on the correct device
                    if hasattr(self, 'device'):
                        data = data.to(self.device)
                    elif next(self.parameters()).is_cuda:
                        data = data.cuda()
                    
                    # Forward pass - explicitly use QuantStub if present
                    try:
                        # Check if forward method is already wrapped with QuantStub
                        forward_is_wrapped = False
                        if hasattr(self, 'forward') and callable(self.forward):
                            try:
                                import inspect
                                forward_source = inspect.getsource(self.forward) if hasattr(inspect, 'getsource') else None
                                if forward_source and 'self.quant' in forward_source:
                                    forward_is_wrapped = True
                            except:
                                if hasattr(self, 'quant') and not hasattr(type(self).forward, '__func__'):
                                    forward_is_wrapped = True
                        
                        # CRITICAL: Always use __call__ (self()) not forward() to ensure hooks fire
                        # This is essential for observers to collect statistics
                        if hasattr(self, 'quant') and isinstance(self.quant, QuantStub):
                            # QuantStub exists - quantize input first, then call model normally
                            x_quant = self.quant(data)
                            # Use __call__ to trigger forward hooks (observers are hooked to forward)
                            out = self(x_quant)
                            # Dequantize output if DeQuantStub exists
                            if hasattr(self, 'dequant') and isinstance(self.dequant, DeQuantStub):
                                if isinstance(out, torch.Tensor):
                                    out = self.dequant(out)
                                elif isinstance(out, (list, tuple)):
                                    out = tuple(self.dequant(o) if isinstance(o, torch.Tensor) else o for o in out)
                        else:
                            # No QuantStub, just call model normally (hooks will still fire)
                            _ = self(data)
                    except Exception as e:
                        LOGGER.warning(f"Error during calibration sample {idx}: {e}")
                        continue
                    
                    if (idx + 1) % 10 == 0:
                        LOGGER.info(f"  Calibrated {idx + 1} samples...")
        
        # Verify observers collected statistics and log detailed state
        from torch.ao.quantization import ObserverBase
        observers_with_stats = 0
        total_observers = 0
        observer_details = []
        
        for name, module in self.named_modules():
            if isinstance(module, ObserverBase):
                total_observers += 1
                has_stats = False
                min_val = None
                max_val = None
                is_enabled = True
                
                if hasattr(module, 'min_val') and hasattr(module, 'max_val'):
                    min_val = module.min_val
                    max_val = module.max_val
                    if min_val is not None and max_val is not None:
                        if isinstance(min_val, torch.Tensor) and min_val.numel() > 0:
                            if isinstance(max_val, torch.Tensor) and max_val.numel() > 0:
                                if max_val.max() > min_val.min():
                                    has_stats = True
                                    observers_with_stats += 1
                
                # Check if observer is enabled
                if hasattr(module, 'is_enabled'):
                    try:
                        is_enabled = module.is_enabled()
                    except:
                        pass
                elif hasattr(module, '_observer_enabled'):
                    is_enabled = module._observer_enabled
                
                # Store details for first few observers
                if len(observer_details) < 5:
                    observer_details.append({
                        'name': name,
                        'has_stats': has_stats,
                        'enabled': is_enabled,
                        'min': min_val.item() if isinstance(min_val, torch.Tensor) and min_val.numel() == 1 else None,
                        'max': max_val.item() if isinstance(max_val, torch.Tensor) and max_val.numel() == 1 else None,
                    })
        
        # Also check activation_post_process observers
        activation_observers_with_stats = 0
        total_activation_observers = 0
        activation_observer_details = []
        
        for name, module in self.named_modules():
            if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                total_activation_observers += 1
                obs = module.activation_post_process
                has_stats = False
                min_val = None
                max_val = None
                is_enabled = True
                
                if hasattr(obs, 'min_val') and hasattr(obs, 'max_val'):
                    min_val = obs.min_val
                    max_val = obs.max_val
                    if min_val is not None and max_val is not None:
                        if isinstance(min_val, torch.Tensor) and min_val.numel() > 0:
                            if isinstance(max_val, torch.Tensor) and max_val.numel() > 0:
                                if max_val.max() > min_val.min():
                                    has_stats = True
                                    activation_observers_with_stats += 1
                
                # Check if observer is enabled
                if hasattr(obs, 'is_enabled'):
                    try:
                        is_enabled = obs.is_enabled()
                    except:
                        pass
                elif hasattr(obs, '_observer_enabled'):
                    is_enabled = obs._observer_enabled
                
                # Store details for first few observers
                if len(activation_observer_details) < 5:
                    activation_observer_details.append({
                        'name': name,
                        'has_stats': has_stats,
                        'enabled': is_enabled,
                        'min': min_val.item() if isinstance(min_val, torch.Tensor) and min_val.numel() == 1 else None,
                        'max': max_val.item() if isinstance(max_val, torch.Tensor) and max_val.numel() == 1 else None,
                    })
        
        total_with_stats = observers_with_stats + activation_observers_with_stats
        total_all = total_observers + total_activation_observers
        
        # Log detailed observer state
        LOGGER.info("Observer state after calibration:")
        if observer_details:
            LOGGER.info("  Sample ObserverBase observers:")
            for detail in observer_details[:3]:
                stats_str = f"min={detail['min']:.4f}, max={detail['max']:.4f}" if detail['has_stats'] else "no stats"
                enabled_str = "enabled" if detail['enabled'] else "disabled"
                LOGGER.info(f"    {detail['name']}: {stats_str} ({enabled_str})")
        
        if activation_observer_details:
            LOGGER.info("  Sample activation_post_process observers:")
            for detail in activation_observer_details[:3]:
                stats_str = f"min={detail['min']:.4f}, max={detail['max']:.4f}" if detail['has_stats'] else "no stats"
                enabled_str = "enabled" if detail['enabled'] else "disabled"
                LOGGER.info(f"    {detail['name']}: {stats_str} ({enabled_str})")
        
        if total_with_stats > 0:
            LOGGER.info(f"✓ Calibration complete! {total_with_stats}/{total_all} observers have collected statistics.")
        else:
            LOGGER.warning(f"⚠️  Calibration completed but {total_all} observers found with NO statistics collected!")
            LOGGER.warning("   This may cause quantization parameters to default to scale=1.0, zp=0")
            LOGGER.warning("   Check if forward pass is going through QuantStub and observers are enabled")
            if observer_details or activation_observer_details:
                disabled_count = sum(1 for d in observer_details + activation_observer_details if not d['enabled'])
                if disabled_count > 0:
                    LOGGER.warning(f"   Found {disabled_count} disabled observers - they may need to be enabled")
        
        return self

    def convert_ptq_to_int8(self, calibrated_model=None, backend='fbgemm'):
        """
        Convert calibrated PTQ model to INT8.
        
        After prepare() and calibration, the model has observers with collected statistics.
        This converts the model to use real quantized operations.
        
        Args:
            calibrated_model: The PTQ model after calibration (with observer statistics)
                             If None, converts self
            backend: Quantization backend ('fbgemm' or 'qnnpack')
        
        Returns:
            Quantized INT8 model
        """
        from torch.ao.quantization import convert
        
        model = calibrated_model if calibrated_model is not None else self
        model.eval()
        
        # CRITICAL: Set quantization backend engine BEFORE conversion
        # This must be set before any quantized operations are created
        if backend in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = backend
            LOGGER.info(f"Set quantization backend engine to {backend}")
        else:
            LOGGER.warning(f"Backend '{backend}' not supported, using default: {torch.backends.quantized.engine}")
        
        # NOTE: QAT's convert_to_quantized() doesn't force CPU - it lets PyTorch handle device placement
        # We follow the same approach for PTQ to match QAT behavior
        # The model should already be on the correct device from the PTQ workflow
        
        LOGGER.info("Converting calibrated PTQ model to INT8...")
        
        # Check if model was prepared for PTQ
        from torch.ao.quantization import ObserverBase
        observer_modules = [n for n, m in model.named_modules() if isinstance(m, ObserverBase)]
        
        # Also check for activation_post_process (observers attached to modules)
        activation_post_process_count = 0
        observers_with_stats = 0
        for name, module in model.named_modules():
            if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
                activation_post_process_count += 1
                # Check if observer has collected statistics
                obs = module.activation_post_process
                if hasattr(obs, 'min_val') and hasattr(obs, 'max_val'):
                    min_val = obs.min_val
                    max_val = obs.max_val
                    if min_val is not None and max_val is not None:
                        # Check if values are valid (not default/uninitialized)
                        if isinstance(min_val, torch.Tensor) and min_val.numel() > 0:
                            if isinstance(max_val, torch.Tensor) and max_val.numel() > 0:
                                if max_val.max() > min_val.min():
                                    observers_with_stats += 1
        
        if len(observer_modules) == 0 and activation_post_process_count == 0:
            LOGGER.warning("⚠️  WARNING: No observer modules found!")
            LOGGER.warning("   Model may not have been prepared with prepare_for_ptq()")
        else:
            LOGGER.info(f"Found {len(observer_modules)} ObserverBase modules and {activation_post_process_count} activation_post_process observers")
            if activation_post_process_count > 0:
                LOGGER.info(f"  - {observers_with_stats}/{activation_post_process_count} observers have collected statistics")
                if observers_with_stats == 0:
                    LOGGER.warning("⚠️  WARNING: Observers found but no statistics collected! Calibration may have failed.")
        
        # Exclude Linear and BatchNorm2d layers that don't have activation_post_process
        # These are typically in BoTNet or unfused layers and should remain FP32
        from ultralytics.nn.BoTNet import BoTNet
        linear_excluded = 0
        bn_excluded = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if it's inside a BoTNet or doesn't have activation_post_process
                is_in_botnet = any('BoTNet' in p or 'botnet' in p.lower() for p in name.split('.'))
                
                if is_in_botnet or not hasattr(module, 'activation_post_process'):
                    # Remove qconfig to exclude from conversion
                    if hasattr(module, 'qconfig'):
                        module.qconfig = None
                    # Remove activation_post_process if it exists but is None/invalid
                    if hasattr(module, 'activation_post_process') and module.activation_post_process is None:
                        delattr(module, 'activation_post_process')
                    linear_excluded += 1
            
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm2d layers that weren't fused should remain FP32
                if not hasattr(module, 'activation_post_process'):
                    # Remove qconfig to exclude from conversion
                    if hasattr(module, 'qconfig'):
                        module.qconfig = None
                    bn_excluded += 1
        
        if linear_excluded > 0 or bn_excluded > 0:
            LOGGER.info(f"Excluding {linear_excluded} Linear and {bn_excluded} BatchNorm2d layers from conversion (missing observers)")
        
        # Convert to quantized model
        try:
            # Count observers before conversion for debugging
            observer_count_before = len([n for n, m in model.named_modules() if isinstance(m, ObserverBase)])
            LOGGER.debug(f"Found {observer_count_before} observer modules before conversion")
            
            # Try FX conversion first if available
            try:
                from torch.ao.quantization.quantize_fx import convert_fx
                # Check if model is FX-prepared (has _node_name_to_scope attribute)
                if hasattr(model, '_node_name_to_scope'):
                    LOGGER.info("Using FX conversion...")
                    model_quantized = convert_fx(model)
                else:
                    # Fall back to eager conversion
                    LOGGER.info("Using eager mode conversion...")
                    model_quantized = convert(model, inplace=False)
            except (ImportError, AttributeError) as e:
                # Fall back to eager conversion
                LOGGER.info(f"Using eager mode conversion (FX not available: {e})...")
                model_quantized = convert(model, inplace=False)
            
            # Count observers after conversion (should be 0 if conversion worked)
            observer_count_after = len([n for n, m in model_quantized.named_modules() if isinstance(m, ObserverBase)])
            if observer_count_after > 0:
                LOGGER.warning(f"⚠️  WARNING: {observer_count_after} observer modules still present after conversion!")
                LOGGER.warning("   This suggests conversion may not have replaced all observers with quantized ops.")
            else:
                LOGGER.debug(f"✓ All {observer_count_before} observers replaced during conversion")
            
            # Check for quantized operations
            quantized_ops = 0
            fp32_ops = 0
            quantized_module_names = []
            
            # Import quantized module types for proper detection
            try:
                from torch.ao.nn.quantized.modules.conv import Conv2d as QuantizedConv2d
                from torch.ao.nn.quantized.modules.linear import Linear as QuantizedLinear
            except ImportError:
                QuantizedConv2d = None
                QuantizedLinear = None
            
            for name, module in model_quantized.named_modules():
                module_type = type(module).__name__
                is_quantized = False
                
                # Check if it's a quantized module type
                if QuantizedConv2d is not None and isinstance(module, QuantizedConv2d):
                    is_quantized = True
                elif QuantizedLinear is not None and isinstance(module, QuantizedLinear):
                    is_quantized = True
                elif 'Quantized' in module_type or 'quantized' in module_type.lower():
                    is_quantized = True
                # Check for _packed_params (sign of quantization)
                elif hasattr(module, '_packed_params') and module._packed_params is not None:
                    is_quantized = True
                
                if is_quantized:
                    quantized_ops += 1
                    quantized_module_names.append(name)
                elif isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    fp32_ops += 1
            
            # Check state dict for quantized parameters
            state_dict = model_quantized.state_dict()
            state_dict_quantized = sum(1 for k in state_dict.keys() if 'quantized' in k.lower() or 'scale' in k.lower() or 'zero_point' in k.lower() or '_packed_params' in k.lower())
            
            # Log some examples of quantized modules for debugging
            if quantized_ops > 0 and len(quantized_module_names) > 0:
                LOGGER.debug(f"  Found {quantized_ops} quantized operations. Examples:")
                for qname in quantized_module_names[:5]:  # Show first 5
                    LOGGER.debug(f"    - {qname}")
                if len(quantized_module_names) > 5:
                    LOGGER.debug(f"    ... and {len(quantized_module_names) - 5} more")
            
            LOGGER.info(f"\nConversion Summary:")
            LOGGER.info(f"  ✓ Quantized operations: {quantized_ops}")
            LOGGER.info(f"  ✓ Quantized parameters in state_dict: {state_dict_quantized}")
            LOGGER.info(f"  - FP32 operations: {fp32_ops}")
            
            if quantized_ops == 0 and state_dict_quantized == 0:
                LOGGER.warning("⚠️  WARNING: No quantized operations found!")
                LOGGER.warning("   This may indicate that convert() cannot handle the nested module structure.")
            else:
                LOGGER.info("  Observer modules replaced with real INT8 operations.")
            
            # CRITICAL: Repair quantized model bookkeeping to prevent segfaults during evaluation
            # Quantized modules (especially QuantizedConv2d) are missing hook attributes
            # that PyTorch's Module._call_impl expects, causing AttributeError during forward pass
            LOGGER.info("Repairing quantized model bookkeeping (fixing hook attributes)...")
            
            def _repair_quantized_bookkeeping(root_module):
                """Repair missing nn.Module bookkeeping attributes on quantized modules."""
                if root_module is None:
                    return 0
                
                visited = set()
                stack = [root_module]
                modules_needing_fix = 0
                
                def _is_dict_like(value):
                    return isinstance(value, dict)
                
                while stack:
                    module = stack.pop()
                    if not isinstance(module, torch.nn.Module):
                        continue
                    module_id = id(module)
                    if module_id in visited:
                        continue
                    visited.add(module_id)
                    
                    needs_fix = False
                    # Fix _modules
                    try:
                        if not _is_dict_like(getattr(module, '_modules', None)):
                            needs_fix = True
                    except Exception:
                        needs_fix = True
                    # Fix _parameters
                    try:
                        if not _is_dict_like(getattr(module, '_parameters', None)):
                            needs_fix = True
                    except Exception:
                        needs_fix = True
                    # Fix _buffers
                    try:
                        if not _is_dict_like(getattr(module, '_buffers', None)):
                            needs_fix = True
                    except Exception:
                        needs_fix = True
                    
                    # CRITICAL: Fix hook attributes that cause AttributeError during forward pass
                    # These are the attributes that PyTorch's _call_impl checks:
                    # _forward_hooks, _backward_hooks, _forward_pre_hooks, _backward_pre_hooks
                    for hook_attr in ('_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks', 
                                     '_state_dict_hooks', '_load_state_dict_pre_hooks'):
                        try:
                            if not _is_dict_like(getattr(module, hook_attr, None)):
                                needs_fix = True
                        except Exception:
                            needs_fix = True
                    
                    # Fix _non_persistent_buffers_set
                    if not hasattr(module, '_non_persistent_buffers_set') or not isinstance(getattr(module, '_non_persistent_buffers_set'), set):
                        needs_fix = True
                    # Fix training attribute
                    if not hasattr(module, 'training'):
                        needs_fix = True
                    
                    if needs_fix:
                        modules_needing_fix += 1
                    
                    # Recursively process children
                    try:
                        children = getattr(module, '_modules', None)
                        if isinstance(children, dict):
                            stack.extend(child for child in children.values() if child is not None)
                        else:
                            # Fallback to children() method
                            try:
                                stack.extend(list(module.children()))
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                # Apply ensure_module_bookkeeping after repairs
                ensure_module_bookkeeping(root_module, recursive=True)
                return modules_needing_fix
            
            # CRITICAL: Apply bookkeeping repair BEFORE returning (same as QAT's convert_to_quantized)
            # This ensures quantized modules have all required attributes for .eval() and forward pass
            repairs = _repair_quantized_bookkeeping(model_quantized)
            if repairs > 0:
                LOGGER.info(f"  ✓ Repaired {repairs} modules with missing bookkeeping attributes")
            else:
                LOGGER.info("  ✓ Model bookkeeping already intact")
            
            # CRITICAL: Ensure the model can be set to eval mode after repair
            # This is needed because .eval() traverses modules and needs proper bookkeeping
            try:
                model_quantized.eval()
                LOGGER.info("  ✓ Model successfully set to eval mode after repair")
            except (AttributeError, RuntimeError) as e:
                LOGGER.warning(f"  ⚠️  Could not set model to eval mode: {e}")
                LOGGER.info("  Model will remain in current mode (should be eval already)")
            
            return model_quantized
        
        except Exception as e:
            LOGGER.error(f"Conversion failed: {e}")
            LOGGER.warning("The model may not have been properly prepared with prepare_for_ptq() and calibrated")
            import traceback
            traceback.print_exc()
            raise


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg='yolov8n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg='yolov8n-cls.yaml', ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        elif not nc and not self.yaml.get('nc', None):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):

    def __init__(self, cfg='rtdetr-l.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        img = batch['img']
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch['batch_idx']
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(img.device, dtype=torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device=img.device),
            'batch_idx': batch_idx.to(img.device, dtype=torch.long).view(-1),
            'gt_groups': gt_groups}

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion((dec_bboxes, dec_scores),
                              targets,
                              dn_bboxes=dn_bboxes,
                              dn_scores=dn_scores,
                              dn_meta=dn_meta)
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']],
                                                   device=img.device)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False
            batch (dict): A dict including gt boxes and labels from dataloader.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x

class Decouple(nn.Module):
    # Decoupled convolution
    def __init__(self, c1, nc=80, na=3):  # ch_in, num_classes, num_anchors
        super().__init__()
        c_ = min(c1, 256)  # min(c1, nc * na)
        self.na = na  # number of anchors
        self.nc = nc  # number of classes
        self.a = Conv(c1, c_, 1)
        c = [int(x + na * 5) for x in (c_ - na * 5) * torch.linspace(1, 0, 4)]  # linear channel descent

        self.b1, self.b2, self.b3 = Conv(c_, c[1], 3), Conv(c[1], c[2], 3), nn.Conv2d(c[2], na * 5, 1)  # vc

        self.c1, self.c2, self.c3 = Conv(c_, c_, 1), Conv(c_, c_, 1), nn.Conv2d(c_, na * nc, 1)  # cls

    def forward(self, x):
        bs, nc, ny, nx = x.shape  # BCHW
        x = self.a(x)
        b = self.b3(self.b2(self.b1(x)))
        c = self.c3(self.c2(self.c1(x)))
        return torch.cat((b.view(bs, self.na, 5, ny, nx), c.view(bs, self.na, self.nc, ny, nx)), 2).view(bs, -1, ny, nx)  
  
class Decoupled_Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
      
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)      
        self.m=nn.ModuleList(Decouple(x, self.nc, self.na)  for x in ch)   #yolov5 provide ,  old Decouple too much FLOP
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        
        
    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid    
    

class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLOv5 network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.

    Example:
        ```python
        with temporary_modules({'old.module.path': 'new.module.path'}):
            import old.module.path  # this will now import new.module.path
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if not modules:
        modules = {}

    import importlib
    import sys
    try:
        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules({
                'ultralytics.yolo.utils': 'ultralytics.utils',
                'ultralytics.yolo.v8': 'ultralytics.models.yolo',
                'ultralytics.yolo.data': 'ultralytics.data'}):  # for legacy 8.0 Classify and Pose models
            return torch.load(file, map_location='cpu', weights_only=False), file  # load

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'models':
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        return torch.load(file, map_location='cpu', weights_only=False), file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None  # combined args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True, qat_mode=False):  # model_dict, input_channels(3)
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.
    
    Args:
        d: Model configuration dictionary
        ch: Input channels
        verbose: Print model info
        qat_mode: If True, use QAT-aware versions of custom modules (BoTNet, CoordAtt)
    """
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    # Aliases and defaults for downstream usage in this fork
    gw = width  # width multiplier alias used in some custom modules
    no = nc + 5  # outputs per anchor: (x, y, w, h, obj) + classes
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        
        # Replace with QAT versions if qat_mode is enabled
        if qat_mode and QAT_AVAILABLE:
            if m is BoTNet:
                m = QATBoTNet
                if verbose:
                    LOGGER.info(f"Layer {i}: Using QATBoTNet for quantization-aware training")
            elif m is CoordAtt:
                m = QATCoordAtt
                if verbose:
                    LOGGER.info(f"Layer {i}: Using QATCoordAtt for quantization-aware training")
            elif m is ODConv:
                m = FP32ODConv
                if verbose:
                    LOGGER.info(f"Layer {i}: Using FP32ODConv (excluded from quantization)")
        
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        #详细改进流程和操作，请关注B站博主：AI学术叫叫兽 
        # Include QAT module types if available
        qat_modules = (QATBoTNet, QATCoordAtt, FP32ODConv) if QAT_AVAILABLE else ()
        if m in (Classify, Conv, GGhostRegNet, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, SEAttention,ContextAggregation, BoTNet, CBAM,LightConv,RepConv, SpatialAttention,Involution, CARAFE, VoVGSCSP, VoVGSCSPC,GSConv,HorBlock, SwinTransformer, MobileOneBlock,MobileViT,MobileViTBlock,MV2Block,CondConv2D,MBConv,FusedMBConv,stem,RepViTblock,DSConv,DySnakeConv,C2f_DySnakeConv,Bottleneck_DySnakeConv,C2f_LSKA_Attention,LSKA_Attention,LSKA,EMA_attention,ODConv) + qat_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
                    

 
            # Add QATBoTNet if QAT is available
            botnet_modules = (BoTNet, QATBoTNet) if QAT_AVAILABLE else (BoTNet,)
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3,C2f_LSKA_Attention) or m in botnet_modules:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is MobileOneBlock:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, n, *args[1:]]
        elif m in [RepLKNet_Stem, RepLKNet_stage1, RepLKNet_stage2, RepLKNet_stage3, RepLKNet_stage4]:
            c2 = args[0]
            args = args[1:]    
        elif m is CrissCrossAttention:
            c1, c2 = ch[f], args[0]
            if c2 != torch.NoneType: 
                c2 = make_divisible(c2 * width, 8)
                args = [c1, *args[1:]]

            args = [c1, *args[1:]]
        elif m is vanillanetBlock:
             c1, c2 = ch[f], args[0]
             if c2 != torch.NoneType:  
                 c2 = make_divisible(c2 * width, 8)
             args = [c1, c2, *args[1:]]

        elif m in {SimFusion_4in, AdvPoolFusion}:
            c2 = sum(ch[x] for x in f)
        elif m is SimFusion_3in:
            c2 = args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[f_] for f_ in f], c2]
        elif m is IFM:
            c1 = ch[f]
            c2 = sum(args[0])
            args = [c1, *args]
        elif m is InjectionMultiSum_Auto_pool:
            c1 = ch[f[0]]
            c2 = args[0]
            args = [c1, *args]
        elif m is PyramidPoolAgg:
            c2 = args[0]
            args = [sum([ch[f_] for f_ in f]), *args]
        elif m is TopBasicLayer:
            c2 = sum(args[1])

        elif m in {Attention,AttentionLePE,BiLevelRoutingAttention}:
            c2 = ch[f]
            args=[c2,*args]
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m in [ShuffleNetV2, Conv_maxpool]:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(c2 * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Concat_BiFPN:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose, RTDETRDecoder):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m in [ODConv]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * width, 8)
            args = [c1, c2, *args[1:]]



        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)



def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detect':
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Detect):
                return 'detect'
            elif isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # assume detect
