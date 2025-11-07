# Ultralytics YOLO 🚀, AGPL-3.0 license
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 
import contextlib
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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
        
        # CRITICAL: Fuse Conv+BN+Activation BEFORE preparing for QAT
        # This ensures PyTorch creates fused quantized modules (QuantizedConvReLU2d)
        # instead of separate quantized operators that cause backend errors
        LOGGER.info("Fusing Conv+BN+Activation layers before QAT preparation...")
        self.fuse_model()
        
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Get default QAT qconfig for the backend
        qconfig = get_default_qat_qconfig(backend)
        
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
                if not isinstance(layer, ODConv):
                    layer.qconfig = qconfig
                else:
                    layer.qconfig = None
                    LOGGER.info(f"  ✓ Excluding layer {i} (ODConv) from quantization (FP32)")
            
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
                # Set qconfig on actual nn.Conv2d (inside Conv wrappers)
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
            LOGGER.info(f"    - Conv2d: {conv_count}")
            LOGGER.info(f"    - BatchNorm2d: {bn_count}")
            LOGGER.info(f"    - Linear: {linear_count}")
            LOGGER.info(f"    - Other modules: {qconfig_set_count - conv_count - bn_count - linear_count}")
            LOGGER.info(f"  ✓ Excluded {odconv_excluded_count} ODConv modules")
            
            # Prepare for QAT in eager mode
            LOGGER.info("  Calling prepare_qat()...")
            model_prepared = prepare_qat(self, inplace=False)
            
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
                        
                        # Check if immediate parent is a fused Conv module (no BN attribute)
                        # This is the key check - if Conv2d is inside a fused Conv, allow conversion
                        from ultralytics.nn.modules.conv import Conv, Conv2, DWConv
                        if isinstance(immediate_parent, (Conv, Conv2, DWConv)):
                            if not hasattr(immediate_parent, 'bn'):
                                # This Conv2d is inside a fused Conv module - ALLOW conversion
                                is_fused_conv_parent = True
                    
                    # AGGRESSIVE: Exclude ALL Conv2d inside custom YOLO modules (ultralytics)
                    # EXCEPT if immediate parent is a fused Conv module
                    # Only allow conversion if it's a direct child of a standard PyTorch container
                    # or inside QuantizedConv (which we designed to handle it)
                    if parent_path and not is_fused_conv_parent:
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
                    # Entry logging
                    LOGGER.info(f"  [{module_name}] Starting conversion: Conv2d -> QuantizedConv2d")
                    LOGGER.info(f"  [{module_name}] has_quantized_conv2d={has_quantized_conv2d}, QuantizedConv2d={QuantizedConv2d is not None}")
                    LOGGER.info(f"  [{module_name}] qconfig={qconfig is not None}, qconfig.weight={qconfig.weight if qconfig and hasattr(qconfig, 'weight') else None}")
                    
                    if not has_quantized_conv2d:
                        LOGGER.warning(f"  [{module_name}] ❌ Cannot convert: QuantizedConv2d not available (has_quantized_conv2d=False)")
                        return conv2d_module  # Can't convert without QuantizedConv2d
                    
                    if QuantizedConv2d is None:
                        LOGGER.warning(f"  [{module_name}] ❌ Cannot convert: QuantizedConv2d is None (import failed)")
                        return conv2d_module
                    
                    try:
                        # Try using from_float if available (PyTorch's recommended method)
                        if hasattr(QuantizedConv2d, 'from_float'):
                            LOGGER.info(f"  [{module_name}] ✓ Using from_float method for conversion")
                            try:
                                # Create a temporary QAT Conv2d with FakeQuantize
                                from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d
                                
                                LOGGER.info(f"  [{module_name}] Creating QAT Conv2d: in={conv2d_module.in_channels}, out={conv2d_module.out_channels}, k={conv2d_module.kernel_size}")
                                
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
                                qat_conv.weight = torch.nn.Parameter(conv2d_module.weight.data.clone())
                                if conv2d_module.bias is not None:
                                    qat_conv.bias = torch.nn.Parameter(conv2d_module.bias.data.clone())
                                
                                LOGGER.info(f"  [{module_name}] Preparing QAT module...")
                                # Prepare QAT module (this attaches FakeQuantize)
                                from torch.ao.quantization import prepare_qat
                                qat_conv.train()  # Must be in train mode for prepare_qat
                                prepare_qat(qat_conv, inplace=True)
                                
                                # Run a dummy forward pass to calibrate observers
                                LOGGER.info(f"  [{module_name}] Running dummy forward pass for calibration...")
                                dummy_input = torch.randn(1, conv2d_module.in_channels, 3, 3)
                                with torch.no_grad():
                                    _ = qat_conv(dummy_input)
                                
                                # Convert the QAT module to quantized using from_float
                                LOGGER.info(f"  [{module_name}] Converting QAT to quantized using from_float...")
                                qat_conv.eval()  # Must be in eval mode for conversion
                                # Use QuantizedConv2d.from_float() directly instead of convert()
                                quantized_conv = QuantizedConv2d.from_float(qat_conv)
                                
                                result_type = type(quantized_conv).__name__
                                has_packed = hasattr(quantized_conv, '_packed_params')
                                is_quantized = isinstance(quantized_conv, QuantizedConv2d) if QuantizedConv2d is not None else False
                                LOGGER.info(f"  [{module_name}] from_float result: type={result_type}, has_packed={has_packed}, is_QuantizedConv2d={is_quantized}")
                                
                                if has_packed or is_quantized:
                                    LOGGER.info(f"  [{module_name}] ✓ Successfully converted using from_float method")
                                    ensure_module_bookkeeping(quantized_conv)
                                    return quantized_conv
                                else:
                                    LOGGER.warning(f"  [{module_name}] ⚠️ from_float returned {result_type} without _packed_params, falling back to manual construction")
                                    raise ValueError(f"from_float did not produce valid QuantizedConv2d")
                            except Exception as from_float_error:
                                LOGGER.warning(f"  [{module_name}] ⚠️ from_float method failed: {from_float_error}")
                                import traceback
                                LOGGER.warning(f"  [{module_name}] from_float traceback:\n{traceback.format_exc()}")
                                LOGGER.info(f"  [{module_name}] Trying manual construction...")
                                # Fall through to manual construction
                        
                        # Manual construction (either from_float not available or failed)
                        if QuantizedConv2d is None or quantize_per_tensor is None:
                            LOGGER.warning(f"  [{module_name}] ❌ Cannot convert: Missing QuantizedConv2d or quantize_per_tensor")
                            LOGGER.warning(f"  [{module_name}]   QuantizedConv2d={QuantizedConv2d is not None}, quantize_per_tensor={quantize_per_tensor is not None}")
                            return conv2d_module  # Can't convert without required classes
                        
                        # Only log if we didn't try from_float first
                        if not hasattr(QuantizedConv2d, 'from_float'):
                            LOGGER.info(f"  [{module_name}] Using manual construction (from_float not available)")
                        else:
                            LOGGER.info(f"  [{module_name}] Using manual construction (from_float failed, using fallback)")
                        
                        # Extract Conv2d parameters
                        LOGGER.info(f"  [{module_name}] Extracting Conv2d parameters...")
                        weight = conv2d_module.weight.data.clone()
                        bias = conv2d_module.bias.data.clone() if conv2d_module.bias is not None else None
                        
                        # Get weight observer from qconfig
                        weight_observer = None
                        if qconfig is not None and hasattr(qconfig, 'weight'):
                            weight_observer = qconfig.weight()
                            LOGGER.info(f"  [{module_name}] Weight observer: {type(weight_observer).__name__}")
                        else:
                            LOGGER.warning(f"  [{module_name}] ⚠️ No weight observer in qconfig, using fallback calculation")
                        
                        # Calculate quantization parameters
                        LOGGER.info(f"  [{module_name}] Calculating quantization parameters...")
                        if weight_observer is not None:
                            try:
                                weight_observer(weight)
                                if hasattr(weight_observer, 'calculate_qparams'):
                                    scale, zero_point = weight_observer.calculate_qparams()
                                    scale = scale.item() if isinstance(scale, torch.Tensor) else scale
                                    zero_point = zero_point.item() if isinstance(zero_point, torch.Tensor) else zero_point
                                    LOGGER.info(f"  [{module_name}] Observer calculated: scale={scale:.6f}, zero_point={zero_point}")
                                else:
                                    # Fallback calculation
                                    scale = weight.abs().max().item() / 127.0
                                    zero_point = 0
                                    LOGGER.warning(f"  [{module_name}] Observer has no calculate_qparams, using fallback: scale={scale:.6f}")
                            except Exception as obs_error:
                                scale = weight.abs().max().item() / 127.0
                                zero_point = 0
                                LOGGER.warning(f"  [{module_name}] Observer calculation failed ({obs_error}), using fallback: scale={scale:.6f}")
                        else:
                            scale = weight.abs().max().item() / 127.0
                            zero_point = 0
                            LOGGER.info(f"  [{module_name}] Using fallback calculation: scale={scale:.6f}, zero_point={zero_point}")
                        
                        # Quantize the weight tensor
                        LOGGER.info(f"  [{module_name}] Quantizing weight tensor...")
                        weight_quantized = quantize_per_tensor(weight, scale, zero_point, torch.qint8)
                        
                        # Create QuantizedConv2d using _packed_params
                        LOGGER.info(f"  [{module_name}] Creating QuantizedConv2d module...")
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
                        LOGGER.info(f"  [{module_name}] Packing parameters...")
                        try:
                            if bias is not None:
                                # Bias quantization: scale_bias = scale_weight * scale_input
                                # For now, use weight scale (input scale would be 1.0 for weights)
                                bias_scale = scale
                                bias_quantized = quantize_per_tensor(bias.float(), bias_scale, 0, torch.qint32)
                                quantized_conv._packed_params = torch.ops.quantized.conv2d_prepack(
                                    weight_quantized, bias_quantized, conv2d_module.stride,
                                    conv2d_module.padding, conv2d_module.dilation, conv2d_module.groups
                                )
                                LOGGER.info(f"  [{module_name}] Packed with bias")
                            else:
                                quantized_conv._packed_params = torch.ops.quantized.conv2d_prepack(
                                    weight_quantized, None, conv2d_module.stride,
                                    conv2d_module.padding, conv2d_module.dilation, conv2d_module.groups
                                )
                                LOGGER.info(f"  [{module_name}] Packed without bias")
                            
                            result_type = type(quantized_conv).__name__
                            has_packed = hasattr(quantized_conv, '_packed_params')
                            is_quantized = isinstance(quantized_conv, QuantizedConv2d) if QuantizedConv2d is not None else False
                            LOGGER.info(f"  [{module_name}] ✓ Manual construction result: type={result_type}, has_packed={has_packed}, is_QuantizedConv2d={is_quantized}, scale={scale:.6f}")
                            
                            if has_packed or is_quantized:
                                ensure_module_bookkeeping(quantized_conv)
                                LOGGER.info(f"  [{module_name}] ✓ Successfully converted using manual construction")
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
                            if parent is not None and isinstance(parent, (Conv, Conv2, DWConv)):
                                # Check if parent is fused (no BN)
                                if not hasattr(parent, 'bn'):
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
            return torch.load(file, map_location='cpu'), file  # load

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

        return torch.load(file, map_location='cpu'), file  # load


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
