# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import threading
from copy import deepcopy
from ultralytics.nn.modules.fno_backbone import FNOBackbone, ECA
from pathlib import Path
from ultralytics.nn.modules import (              
        ConvNeXtV2Backbone, FNOBackbone  
)


import torch
import torch.nn as nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    OBB26,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    Pose26,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    Segment26,
    SemanticSegment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, SAFE_LOAD, SETTINGS, WINDOWS, YAML, colorstr, emojis
from ultralytics.utils.checks import REMOTE_FILE_PREFIXES, check_file, check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2ELoss,
    PoseLoss26,
    SemanticSegmentationLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(torch.nn.Module):
    """Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion, information
    display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv/BatchNorm layers and reparameterize for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """Fuse Conv/ConvTranspose and BatchNorm layers, and reparameterize RepConv/RepVGGDW for improved efficiency.

        Args:
            verbose (bool): Whether to print model information after fusion.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, "end2end", False):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """Check if the model has less than a certain threshold of normalization layers.

        Args:
            thresh (int, optional): The threshold number of normalization layers.

        Returns:
            (bool): True if the number of normalization layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image used for computing model information.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """Apply a function to all tensors in the model, including Detect head attributes like stride and anchors.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


def _initialize_yolo_model(model, cfg, ch, nc, verbose):
    """Initialize common YOLO model attributes from a YAML config."""
    model.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
    if model.yaml["backbone"][0][2] == "Silence":
        LOGGER.warning(
            "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
            "Please delete local *.pt file and re-download the latest model checkpoint."
        )
        model.yaml["backbone"][0][2] = "nn.Identity"

    model.yaml["channels"] = ch  # save channels
    if nc and nc != model.yaml["nc"]:
        LOGGER.info(f"Overriding model.yaml nc={model.yaml['nc']} with nc={nc}")
        model.yaml["nc"] = nc  # override YAML value
    model.model, model.save = parse_model(deepcopy(model.yaml), ch=ch, verbose=verbose)  # model, savelist
    model.names = {i: f"{i}" for i in range(model.yaml["nc"])}  # default names dict
    model.inplace = model.yaml.get("inplace", True)


class DetectionModel(BaseModel):
    """YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        _initialize_yolo_model(self, cfg, ch, nc, verbose)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                output = self.forward(x)
                if self.end2end:
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride, e.g., RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    @property
    def end2end(self):
        """Return whether the model uses end-to-end NMS-free detection."""
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        """Set attributes of the model head (last layer).

        Args:
            **kwargs (Any): Arbitrary keyword arguments representing attributes to set.
        """
        head = self.model[-1]
        for k, v in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"Head has no attribute '{k}'.")
                continue
            setattr(head, k, v)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (tuple[torch.Tensor, None]): Augmented inference output and None for train output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
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
        """De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int | None): Flip type (None=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails.

        Args:
            y (list[torch.Tensor]): List of detection tensors.

        Returns:
            (list[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
    computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return E2ELoss(self, v8OBBLoss) if getattr(self, "end2end", False) else v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
    pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self)


class SemanticSegmentationModel(BaseModel):
    """YOLO semantic segmentation model.

    This class implements a semantic segmentation model that produces per-pixel class predictions. Unlike
    SegmentationModel (instance segmentation), this does not produce bounding boxes.

    Methods:
        __init__: Initialize the semantic segmentation model.
        init_criterion: Initialize the loss criterion for semantic segmentation.

    Examples:
        Initialize a semantic segmentation model
        >>> model = SemanticSegmentationModel("yolo26n-sem.yaml", ch=3, nc=19)
    """

    def __init__(self, cfg="yolo26n-sem.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO semantic segmentation model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        _initialize_yolo_model(self, cfg, ch, nc, verbose)

        # Build strides: track smallest spatial size across all layers to find the deepest
        # backbone stride (e.g. P5/32). Head input alone is insufficient: the FPN upsamples
        # P5 away before the head, but the encoder still requires inputs aligned to that
        # deepest stride or FPN concats fail on rounding mismatches.
        m = self.model[-1]
        if isinstance(m, SemanticSegment):
            s = 256
            self.model.eval()
            m.training = True  # get training output (stride-4)
            min_h = [s]

            def _record(_m, _inp, out, _h=min_h):
                if isinstance(out, torch.Tensor) and out.ndim == 4:
                    _h[0] = min(_h[0], out.shape[-2])

            hooks = [layer.register_forward_hook(_record) for layer in self.model]
            try:
                self.forward(torch.zeros(1, ch, s, s))
            finally:
                for h in hooks:
                    h.remove()
            m.stride = torch.tensor([s / min_h[0]], dtype=torch.float32)  # e.g., 256/8 = 32
            self.stride = m.stride
            self.model.train()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        """Initialize the loss criterion for semantic segmentation."""
        return SemanticSegmentationLoss(self)

    def _apply(self, fn):
        """Apply a function to all tensors in the model."""
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, SemanticSegment):
            m.stride = fn(m.stride)
        return self
# ──────────────────────────────────────────────────────────────────────────────
# Dual-head loss
# ──────────────────────────────────────────────────────────────────────────────

class DualDetectionLoss:
    """
    Wraps v8DetectionLoss for two Detect heads.

    During training DualDetectionModel._predict_once returns
        [head1_features, head2_features]
    This class splits them, computes a loss for each head, and sums with weights.

    Primary weight = 1.0, auxiliary weight = 0.4 (standard deep-supervision ratio).
    The only difference between the two loss instances is self.stride — everything
    else (nc, no, reg_max, BboxLoss, TaskAlignedAssigner) is shared via shallow copy.
    """

    def __init__(self, model, primary_weight=1.0, auxiliary_weight=1.0):
        from ultralytics.utils.loss import v8DetectionLoss
        import copy

        # Collect both Detect layers in forward order
        detect_layers = [m for m in model.model if isinstance(m, Detect)]
        if len(detect_layers) != 2:
            raise ValueError(
                f"DualDetectionLoss expects exactly 2 Detect layers, found {len(detect_layers)}"
            )
        head1, head2 = detect_layers

        # v8DetectionLoss always reads model.model[-1] for its parameters.
        # model[-1] is Head-2 (last layer), so base_loss is already correct for Head-2.
        self.loss2 = v8DetectionLoss(model)

        # Shallow-copy for Head-1: everything is identical except stride.
        self.loss1 = copy.copy(self.loss2)
        self.loss1.stride = head1.stride   # override stride to [8, 16, 32]
        # self.loss2.stride already = head2.stride = [4, 8, 16] from init

        self.pw = primary_weight
        self.aw = auxiliary_weight

    def __call__(self, preds, batch):
        # preds = [head1_feature_list, head2_feature_list]
        l1, items1 = self.loss1(preds[0], batch)
        l2, items2 = self.loss2(preds[1], batch)
        total = self.pw * l1 + self.aw * l2
        return total, (self.pw * items1 + self.aw * items2).detach()


# ──────────────────────────────────────────────────────────────────────────────
# Dual-head detection model
# ──────────────────────────────────────────────────────────────────────────────

class DualDetectionModel(BaseModel):
    """
    YOLOv8-style model with two Detect heads.

    Inherits BaseModel (not DetectionModel) so we can control stride
    computation for both heads independently without fighting
    DetectionModel.__init__'s assumption that model[-1] is the only head.

    Head-1 (primary)   — P3/P4/P5,   strides [8,  16, 32]
    Head-2 (auxiliary) — P2/P3/P4,   strides [4,  8,  16]

    Training  forward → list [head1_feats, head2_feats]
    Inference forward → head1 decoded output  (primary head only)
    """

    def __init__(self, cfg="yolov8-fno-dual.yaml", ch=3, nc=None, verbose=True):
        super().__init__()                                # nn.Module.__init__
        _initialize_yolo_model(self, cfg, ch, nc, verbose)

        # Locate both Detect layers in sequential order
        detect_layers = [m for m in self.model if isinstance(m, Detect)]
        if len(detect_layers) != 2:
            raise ValueError(
                f"DualDetectionModel requires exactly 2 Detect layers in the YAML, "
                f"found {len(detect_layers)}"
            )
        self.head1 = detect_layers[0]   # layer 16, primary
        self.head2 = detect_layers[1]   # layer 19, auxiliary

        self.head1.inplace = self.inplace
        self.head2.inplace = self.inplace

        # ── compute strides for both heads ────────────────────────────────
        # Put model in eval but keep both Detect layers in train mode so they
        # return raw feature lists (not decoded inference tensors).
        s = 256
        self.model.eval()
        self.head1.training = True
        self.head2.training = True

        with torch.no_grad():
            raw = self._predict_once(torch.zeros(1, ch, s, s))
        # raw = [head1_feat_list, head2_feat_list]
        # Each feat_list element is (1, no, H, W); stride = s / H
        self.head1.stride = torch.tensor([s / x.shape[-2] for x in raw[0]])
        self.head2.stride = torch.tensor([s / x.shape[-2] for x in raw[1]])
        self.stride = self.head1.stride   # Ultralytics Trainer reads model.stride

        self.model.train()
        self.head1.bias_init()
        self.head2.bias_init()

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    # ------------------------------------------------------------------
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y = []
        detect_outputs = []

        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            if isinstance(m, Detect):
                detect_outputs.append(x)
            y.append(x if m.i in self.save else None)

        if self.training:
            # Each element is the raw feature list from one Detect layer
            return detect_outputs               # [head1_feats, head2_feats]
        # Inference: return only the primary head's decoded tensor
        return detect_outputs[0]

    # ------------------------------------------------------------------
    def init_criterion(self):
        return DualDetectionLoss(self)

    # ------------------------------------------------------------------
    @property
    def end2end(self):
        return False

class PoseModel(DetectionModel):
    """YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for
    keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return E2ELoss(self, PoseLoss26) if getattr(self, "end2end", False) else v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks, providing model
    initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-cls.yaml", ch=3, nc=None, verbose=True):
        """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'nc' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _apply(self, fn):
        """Apply a function to all tensors in the model, including decoder anchors and valid mask.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (RTDETRDetectionModel): An updated RTDETRDetectionModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]
        m.anchors = fn(m.anchors)
        m.valid_mask = fn(m.valid_mask)
        return self

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (tuple, optional): Precomputed model predictions.

        Returns:
            (torch.Tensor): Total loss value.
            (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class
    specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """Get text positional embeddings using the CLIP model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
    both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """Get text positional embeddings using the CLIP model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings without reprta module processing.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)

        model = (
            self.clip_model
            if cache_clip_model
            else build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)
        )
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """Get visual positional embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (list[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        cv2 = getattr(head, "one2one_cv2", head.cv2)

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2) for i, (cls, pf, loc) in enumerate(zip(vocab, cv3, cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """Get fused vocabulary layer from the model.

        Args:
            names (list[str]): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        vocab = nn.ModuleList()
        for cls_head in cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (list[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """Get class positional embeddings.

        Args:
            tpe (torch.Tensor | None): Text positional embeddings.
            vpe (torch.Tensor | None): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of layer indices to return embeddings from.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x.append(cls_pe)  # adding cls embedding
            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = (
                (E2ELoss(self, TVPDetectLoss) if getattr(self, "end2end", False) else TVPDetectLoss(self))
                if visual_prompt
                else self.init_criterion()
            )
        if preds is None:
            preds = self.forward(
                batch["img"],
                tpe=None if "visuals" in batch else batch.get("txt_feats", None),
                vpe=batch.get("visuals", None),
            )
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
    specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = (
                (E2ELoss(self, TVPSegmentLoss) if getattr(self, "end2end", False) else TVPSegmentLoss(self))
                if visual_prompt
                else self.init_criterion()
            )

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging
    or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Run ensemble forward pass and concatenate predictions from all models.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            (torch.Tensor): Concatenated predictions from all models.
            (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C*num_models)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've
    moved a module from one location to another, but you still want to support the old import paths for backwards
    compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Notes:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class _SafeLoad:
    """Opt-in restricted checkpoint loading: reconstruct only known model classes (`weights_only=True` plus an
    allow-list) and build models without `eval()`.

    Enabled per-process by the `ULTRALYTICS_SAFE_LOAD` env flag, or per-call by `torch_safe_load(..., safe_only=True)`.
    Default loading (flag off) is unchanged.
    """

    # Restricted loading reconstructs allow-listed classes via the torch.serialization.safe_globals context manager,
    # added in torch 2.5. On older torch it is unavailable, so restricted loading degrades to a standard load there.
    SUPPORTED = hasattr(torch.serialization, "safe_globals")
    _globals = None  # cached allow-list, built once
    _local = threading.local()  # per-thread flag set while a weights_only load is in progress

    @classmethod
    def restricted(cls):
        """Whether model construction should use the no-eval, known-layer path (env flag or an in-progress load)."""
        return cls.SUPPORTED and (SAFE_LOAD or getattr(cls._local, "active", False))

    @classmethod
    @contextlib.contextmanager
    def loading(cls):
        """Load with `weights_only=True`: scope the allow-list to this load and mark the thread restricted, so a
        checkpoint that reaches model construction (parse_model) also uses the no-eval, known-layer path.
        """
        if cls._globals is None:
            cls._globals = cls._build()
        cls._local.active = True
        try:
            with torch.serialization.safe_globals(cls._globals):
                yield
        finally:
            cls._local.active = False

    @staticmethod
    def activation(act):
        """Resolve a model-yaml `activation` spec to a `torch.nn` module instance without `eval()`.

        Accepts only the documented `[torch.]nn.<Class>(literal args)` shape (e.g. `nn.SiLU()`,
        `torch.nn.LeakyReLU(0.1)`) with literal arguments, and rejects anything else.
        """
        import ast

        try:
            call = ast.parse(act.strip(), mode="eval").body
            assert isinstance(call, ast.Call)
            attrs = []
            node = call.func
            while isinstance(node, ast.Attribute):  # unwind e.g. torch.nn.SiLU -> ["SiLU","nn","torch"]
                attrs.append(node.attr)
                node = node.value
            assert isinstance(node, ast.Name)
            attrs.append(node.id)  # e.g. ["SiLU", "nn"] or ["SiLU", "nn", "torch"]
            assert attrs[1:] in (["nn"], ["nn", "torch"]), "activation must be a torch.nn class"
            klass = getattr(nn, attrs[0])
            assert isinstance(klass, type) and issubclass(klass, nn.Module)
            args = [ast.literal_eval(a) for a in call.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
            return klass(*args, **kwargs)
        except Exception as e:
            raise TypeError(
                emojis(f"ERROR ❌️ unsupported activation '{act}' blocked during restricted model load.")
            ) from e

    @classmethod
    def _build(cls):
        """Auto-discover `nn.Module` subclasses across `torch.nn` and the ultralytics model families, registered under
        every namespace path they are reachable from (covering re-exports such as `block.RealNVP` as
        `head.RealNVP`), plus torchvision transforms and legacy aliases.

        Returns:
            (list): Items for `torch.serialization.safe_globals` — classes and `(obj, "module.Name")` aliases.
        """
        import enum
        import importlib
        import inspect
        import pathlib
        import pkgutil

        import torch.nn.modules as torch_nn

        import ultralytics.nn.modules as ul_nn
        import ultralytics.nn.tasks as ul_tasks

        allow = []

        def _scan(pkg):
            mods = [pkg]
            if hasattr(pkg, "__path__"):  # package: include all submodules
                for info in pkgutil.iter_modules(pkg.__path__, f"{pkg.__name__}."):
                    try:
                        mods.append(importlib.import_module(info.name))
                    except Exception:  # optional/oddball submodule — skip
                        continue
            for mod in mods:
                for name, klass in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(klass, nn.Module):
                        # Register under the path the class is reachable from — matches how a checkpoint pickled it.
                        allow.append((klass, f"{mod.__name__}.{name}"))

        _scan(torch_nn)  # PyTorch nn modules
        _scan(ul_nn)  # ultralytics block/conv/head/transformer
        _scan(ul_tasks)  # ultralytics task models

        # Non-nn.Module data globals in official checkpoints (classification preprocessing transforms).
        try:
            import torchvision.transforms.transforms as tvt
            from torchvision.transforms.functional import InterpolationMode

            allow += [tvt.Compose, tvt.Normalize, tvt.Resize, tvt.CenterCrop, tvt.ToTensor, InterpolationMode]
        except ImportError:
            pass

        # Legacy/cross-platform aliases (pickled paths with no current class namespace), mirroring temporary_modules().
        from ultralytics.utils.loss import E2EDetectLoss

        def _getattr(obj, name):  # ckpts pickle `Detect.forward` and `InterpolationMode.BILINEAR` via getattr
            if isinstance(obj, type) and not name.startswith("__") and issubclass(obj, (nn.Module, enum.Enum)):
                return getattr(obj, name)
            raise pickle.UnpicklingError(f"unsafe getattr({obj!r}, {name!r}) blocked during restricted model load")

        allow += [
            (nn.Identity, "ultralytics.nn.modules.block.Silence"),  # YOLOv9e
            (DetectionModel, "ultralytics.nn.tasks.YOLOv10DetectionModel"),  # YOLOv10
            (E2EDetectLoss, "ultralytics.utils.loss.v10DetectLoss"),  # YOLOv10
            (_getattr, "builtins.getattr"),  # non-det YOLOv8, YOLO11 ckpts (restrict to nn.Module attrs)
        ]
        if WINDOWS:
            allow += [pathlib.WindowsPath, (pathlib.WindowsPath, "pathlib.PosixPath")]
        else:
            allow += [pathlib.PosixPath, (pathlib.PosixPath, "pathlib.WindowsPath")]
        return allow


def torch_safe_load(weight, safe_only=None):
    """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
    the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str | Path): The file path of the PyTorch model.
        safe_only (bool, optional): Load with `torch.load(weights_only=True)`, reconstructing only the known
            Ultralytics/torch model classes on the allow-list. Defaults to the `ULTRALYTICS_SAFE_LOAD` environment
            variable (off), so standard usage is unchanged; set the env to opt in.

    Returns:
        (dict): The loaded model checkpoint.
        (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset

    if safe_only is None:
        safe_only = SAFE_LOAD
    if safe_only and not _SafeLoad.SUPPORTED:
        LOGGER.warning("Restricted model loading requires torch>=2.5; loading without restriction.")
        safe_only = False
    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally

    def _load():
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
                # resolve cross-platform pathlib pickle incompatibility
                **(
                    {"pathlib.PosixPath": "pathlib.WindowsPath"}
                    if WINDOWS
                    else {"pathlib.WindowsPath": "pathlib.PosixPath"}
                ),
            },
        ):
            if safe_only:
                with _SafeLoad.loading():  # weights_only load scoped to the known-class allow-list
                    return torch_load(file, map_location="cpu", weights_only=True)
            return torch_load(file, map_location="cpu")

    try:
        ckpt = _load()

    except RuntimeError as e:
        # Recover only a corrupt cached official asset requested by bare name; never touch user-supplied paths.
        name = Path(str(weight)).name
        if "PytorchStreamReader" not in str(e) or str(weight) != name or name not in GITHUB_ASSETS_NAMES:
            raise
        LOGGER.warning(f"Corrupt cache {file}, re-downloading {weight}...")
        Path(file).unlink(missing_ok=True)
        file = attempt_download_asset(weight)
        ckpt = _load()

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name in {"models", "models.yolo", "models.common", "models.experimental"}:
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5. This model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        elif e.name and e.name.startswith("ultralytics."):
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires missing Ultralytics module '{e.name}'. "
                    "Train a new model using the latest 'ultralytics' package or run a command with an official "
                    "Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        if safe_only:
            # Under weights_only loading, do not auto-install a module named by the checkpoint or fall back to a
            # weights_only=False reload.
            raise
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    except pickle.UnpicklingError as e:
        # weights_only=True encountered a global outside the allow-list. The default (weights_only=False) path can also
        # raise this for a corrupt or legacy file, so re-raise verbatim there to preserve existing behavior.
        if not safe_only:
            raise
        raise TypeError(
            emojis(
                f"ERROR ❌️ {weight} references types outside the supported Ultralytics checkpoint format. "
                f"Use an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
            )
        ) from e

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """Load single model weights.

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        (torch.nn.Module): Loaded model.
        (dict): Model checkpoint dictionary.
    """
    if str(weight).lower().startswith(REMOTE_FILE_PREFIXES):
        weight = check_file(weight, download_dir=SETTINGS["weights_dir"])
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    candidate = ckpt.get("ema") or ckpt.get("model")
    if not isinstance(candidate, torch.nn.Module):
        raise TypeError(
            emojis(
                f"ERROR ❌️ {weight} references types outside the supported Ultralytics checkpoint format. "
                f"Use an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
            )
        )
    model = candidate.float()  # FP32 model

    # Model compatibility updates
    model.args = args  # attach args to model
    model.pt_path = str(weight)  # attach *.pt file path to model as string (avoids WindowsPath pickle issues)
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    import ast

    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    restricted = _SafeLoad.restricted()
    if act:
        Conv.default_act = _SafeLoad.activation(act) if restricted else eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    multi_out_ch = {}  # {layer_index: [c_stage0, c_stage1, c_stage2, c_stage3]}

    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            ASPP,        # ← add this
            ASPP_SOD,    # ← add this
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if m.startswith("nn.")
            else getattr(__import__("torchvision").ops, m[16:])
            if m.startswith("torchvision.ops.")
            else globals()[m]
        )
        if restricted and not (isinstance(m, type) and issubclass(m, torch.nn.Module)):
            raise TypeError(emojis(f"ERROR ❌️ module '{m}' is not a permitted model layer under restricted loading."))

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False

        elif m is AIFI:
            args = [ch[f], *args]

        elif m is SwinBackbone:
            args = [ch[f]]          # in_chans from ch[f]
            c2   = None             # multi-output; set below after build    

        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1

        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4

        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}:
                m.legacy = legacy

        elif m is SemanticSegment:
            args.append([ch[x] for x in f])

        elif m is v10Detect:
            args.append([ch[x] for x in f])

        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])

        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])

        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]

        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is FNOBackbone:
            # YAML args are empty []; in_chans is injected automatically from ch[f].
            # dims and modes use the class defaults (optimum values baked in).
            args = [ch[f]]   # FNOBackbone(in_chans)
            c2   = None      # multi-output; real per-stage channels stored below
        elif m is ECA:
            # ECA takes only the channel count; reads it from ch[f]
            c2   = ch[f]
            args = [ch[f]]    # ECA(channels)        
        elif m is ConvNeXtV2Backbone:
            # args in YAML: ["tiny", 0.1]  → variant, drop_path_rate
            # in_chans is filled automatically from ch[f]
            variant   = args[0] if len(args) > 0 else "tiny"
            drop_path = args[1] if len(args) > 1 else 0.0
            args      = [variant, ch[f], drop_path]
            c2        = None  # multi-output; real channels stored in multi_out_ch below

        elif m in frozenset({TorchVision, Index}):
            if isinstance(f, int) and f in multi_out_ch:
                # Selecting one stage tensor from a ConvNeXtV2Backbone output list
                stage_idx = args[0]
                c2        = multi_out_ch[f][stage_idx]
                args      = [stage_idx]
            else:
                c2   = args[0]
                c1   = ch[f]
                args = [*args[1:]]

        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if m is FNOBackbone:
            multi_out_ch[i] = list(m_.dims)   # e.g. [96, 192, 384, 768]
            c2 = m_.dims[-1]                  # ch[] carries deepest-stage width;
                                              # individual widths live in multi_out_ch

        # After building the module, record multi-output channels and fix c2
        if m is ConvNeXtV2Backbone:
            multi_out_ch[i] = list(m_.dims)   # e.g. [96, 192, 384, 768] for tiny
            c2 = m_.dims[-1]                  # ch list carries the last-stage width;
                                              # individual stage widths live in multi_out_ch
        if m is SwinBackbone:
            multi_out_ch[i] = list(m_.dims)
            c2 = m_.dims[-1]
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)
        
def yaml_model_load(path):
    """Load a YOLO model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x), or empty string if not found.
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict | str | Path): PyTorch model, model configuration dict, or model file path.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb', 'semantic').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "semanticsegment" in m:
            return "semantic"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]  # nosec B307: safe eval of known attribute paths
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # nosec B307: safe eval of known attribute paths
        for m in model.modules():
            if isinstance(m, SemanticSegment):
                return "semantic"
            elif isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-sem" in model.stem or "semantic" in model.parts:
            return "semantic"
        elif "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', 'pose', 'obb' or 'semantic'."
    )
    return "detect"  # assume detect
    # ═══════════════════════════════════════════════════════════════════
# Swin Transformer Backbone — defined here to avoid import caching
# ═══════════════════════════════════════════════════════════════════

import math as _math


def _window_partition(x, window_size):
    """x: (B, H, W, C) → windows: (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def _window_reverse(windows, window_size, H, W):
    """windows: (num_windows*B, window_size, window_size, C) → x: (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class _WindowAttention(torch.nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size   # (Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = torch.nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv  = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(dim, dim)
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C)  where N = window_size*window_size"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        rpb = rpb.permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class _SwinMLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1  = torch.nn.Linear(dim, hidden_dim)
        self.act  = torch.nn.GELU()
        self.fc2  = torch.nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _SwinBlock(torch.nn.Module):
    """
    Single Swin Transformer block.
    shift_size=0 → regular window attention (W-MSA)
    shift_size=window_size//2 → shifted window attention (SW-MSA)
    Alternating W-MSA / SW-MSA across blocks in a stage gives
    cross-window connections without full quadratic attention.
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn  = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp   = _SwinMLP(dim, int(dim * mlp_ratio))
        self.dp_prob = drop_path

    def _drop_path(self, x):
        if not self.training or self.dp_prob == 0:
            return x
        keep  = 1.0 - self.dp_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, device=x.device, dtype=x.dtype).floor_().add_(keep)
        return x * rand / keep

    def _attn_mask(self, H, W, device):
        """Compute mask for shifted-window attention (prevents cross-boundary leakage)."""
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to multiple of window_size if needed
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_mask = self._attn_mask(Hp, Wp, x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = _window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self._drop_path(x)
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchMerging(torch.nn.Module):
    """Downsample by 2x: concat 2x2 neighboring patches, then linear project to 2*dim."""
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4 * dim)
        self.reduction = torch.nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        pad_r = W % 2
        pad_b = H % 2
        if pad_r or pad_b:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x)


class _SwinStage(torch.nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4.0, drop_path_rates=None):
        super().__init__()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.blocks = torch.nn.ModuleList([
            _SwinBlock(dim, num_heads, window_size,
                      shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, drop_path=drop_path_rates[i])
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class SwinBackbone(torch.nn.Module):
    """
    Swin Transformer Tiny backbone, adapted for YOLOv8.

    Native hierarchical output — no Index/multi-output plumbing needed
    beyond what FNOBackbone already required, but here the four scales
    are produced by genuine architectural downsampling (PatchMerging),
    not channel-preserving spectral blocks.

    Returns [P2, P3, P4, P5] as (B, C, H, W) tensors (converted from
    Swin's native (B, N, C) token format at each stage boundary).

    Config (Swin-T):
      depths     = [2, 2, 6, 2]
      dims       = [96, 192, 384, 768]
      num_heads  = [3, 6, 12, 24]
      window_size= 7
    """
    _DIMS      = [96, 192, 384, 768]
    _DEPTHS    = [2, 2, 6, 2]
    _NUM_HEADS = [3, 6, 12, 24]
    _WINDOW    = 7

    def __init__(self, in_chans=3, drop_path_rate=0.1):
        super().__init__()
        dims, depths, heads, window = self._DIMS, self._DEPTHS, self._NUM_HEADS, self._WINDOW
        self.dims = list(dims)

        # Patch embed: 4x4 conv, stride 4 → P2 native resolution
        self.patch_embed = torch.nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
        self.patch_norm  = torch.nn.LayerNorm(dims[0])

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cur = 0

        self.stages    = torch.nn.ModuleList()
        self.merges    = torch.nn.ModuleList()   # patch merging BEFORE stages 1,2,3
        self.out_norms = torch.nn.ModuleList([torch.nn.LayerNorm(d) for d in dims])

        for i in range(4):
            if i > 0:
                self.merges.append(_PatchMerging(dims[i - 1]))
            else:
                self.merges.append(torch.nn.Identity())

            self.stages.append(_SwinStage(
                dims[i], depths[i], heads[i], window,
                drop_path_rates=dpr[cur:cur + depths[i]]))
            cur += depths[i]

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.zeros_(m.bias)
                torch.nn.init.ones_(m.weight)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """Returns [P2, P3, P4, P5] as (B, C, H, W) tensors."""
        B = x.shape[0]
        x = self.patch_embed(x)                 # (B, C0, H0, W0)
        _, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, H*W, C)
        x = self.patch_norm(x)

        outs = []
        for i in range(4):
            if i > 0:
                x = self.merges[i](x, H, W)
                H, W = H // 2, W // 2

            x = self.stages[i](x, H, W)

            # Convert token format → spatial format for output
            x_out = self.out_norms[i](x)
            x_out = x_out.transpose(1, 2).reshape(B, self.dims[i], H, W)
            outs.append(x_out)

        return outs   # [P2_96, P3_192, P4_384, P5_768]   
# ═══════════════════════════════════════════════════════════════════
# ASPP — replaces SPPF, tuned for small-object (TB bacilli) detection
# ═══════════════════════════════════════════════════════════════════

class ASPP(torch.nn.Module):
    """
    Atrous Spatial Pyramid Pooling, re-tuned for small objects.

    Standard DeepLab ASPP uses dilation rates 6, 12, 18 — tuned for
    513x513 segmentation inputs with large objects. At those rates,
    on a P5 (stride 32) feature map, the effective sampling distance
    in original-image pixels is 192/384/576px, which is far larger
    than a 15-30px bacillus. The conv ends up sampling background
    noise instead of the object itself.

    This version uses rates [1, 3, 5] by default (overridable),
    keeps the rate=1 branch un-dilated for fine local detail, and
    gives that branch more weight in the final fusion by simply
    including it — it is the only branch with zero gridding effect.

    Branches:
      1. 1x1 conv                      (channel reduction, global mix)
      2. 3x3 conv, dilation=rates[0]   (default 1 — plain local conv)
      3. 3x3 conv, dilation=rates[1]   (default 3)
      4. 3x3 conv, dilation=rates[2]   (default 5)
      5. image pooling branch (global avg pool + 1x1 + upsample)
    All branches projected to out_ch//5... no — projected individually
    to out_ch_branch, concatenated, then fused with a final 1x1 conv.
    """

    def __init__(self, c1, c2, rates=(1, 3, 5)):
        super().__init__()
        assert len(rates) == 3, "ASPP expects exactly 3 dilation rates"
        hidden = c2 // 4   # 4 conv branches share channel budget evenly

        def _conv_bn_act(cin, cout, k, d):
            pad = d * (k - 1) // 2
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, k, 1, pad, dilation=d, bias=False),
                torch.nn.BatchNorm2d(cout, eps=1e-3, momentum=0.03),
                torch.nn.SiLU(inplace=True),
            )

        # Branch 1: 1x1, no dilation — channel mixing only
        self.branch1 = _conv_bn_act(c1, hidden, 1, 1)

        # Branches 2-4: 3x3 with configurable dilation rates
        self.branch2 = _conv_bn_act(c1, hidden, 3, rates[0])
        self.branch3 = _conv_bn_act(c1, hidden, 3, rates[1])
        self.branch4 = _conv_bn_act(c1, hidden, 3, rates[2])

        # Branch 5: global image pooling — coarse global context
        self.global_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(c1, hidden, 1, bias=False),
            torch.nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            torch.nn.SiLU(inplace=True),
        )

        # Fusion: concat all 5 branches → project to c2
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(hidden * 5, c2, 1, bias=False),
            torch.nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03),
            torch.nn.SiLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        b5 = self.global_pool(x)
        b5 = torch.nn.functional.interpolate(
            b5, size=(H, W), mode="bilinear", align_corners=False)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.fuse(out)


# ═══════════════════════════════════════════════════════════════════
# ASPP — replaces SPPF, tuned for small-object (TB bacilli) detection
# ═══════════════════════════════════════════════════════════════════

class ASPP(torch.nn.Module):
    """
    Atrous Spatial Pyramid Pooling, re-tuned for small objects.

    Standard DeepLab ASPP uses dilation rates 6, 12, 18 — tuned for
    513x513 segmentation inputs with large objects. At those rates,
    on a P5 (stride 32) feature map, the effective sampling distance
    in original-image pixels is 192/384/576px, which is far larger
    than a 15-30px bacillus. The conv ends up sampling background
    noise instead of the object itself.

    This version uses rates [1, 3, 5] by default (overridable),
    keeps the rate=1 branch un-dilated for fine local detail, and
    gives that branch more weight in the final fusion by simply
    including it — it is the only branch with zero gridding effect.

    Branches:
      1. 1x1 conv                      (channel reduction, global mix)
      2. 3x3 conv, dilation=rates[0]   (default 1 — plain local conv)
      3. 3x3 conv, dilation=rates[1]   (default 3)
      4. 3x3 conv, dilation=rates[2]   (default 5)
      5. image pooling branch (global avg pool + 1x1 + upsample)
    All branches projected to out_ch//5... no — projected individually
    to out_ch_branch, concatenated, then fused with a final 1x1 conv.
    """

    def __init__(self, c1, c2, rates=(1, 3, 5)):
        super().__init__()
        assert len(rates) == 3, "ASPP expects exactly 3 dilation rates"
        hidden = c2 // 4   # 4 conv branches share channel budget evenly

        def _conv_bn_act(cin, cout, k, d):
            pad = d * (k - 1) // 2
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, k, 1, pad, dilation=d, bias=False),
                torch.nn.BatchNorm2d(cout, eps=1e-3, momentum=0.03),
                torch.nn.SiLU(inplace=True),
            )

        # Branch 1: 1x1, no dilation — channel mixing only
        self.branch1 = _conv_bn_act(c1, hidden, 1, 1)

        # Branches 2-4: 3x3 with configurable dilation rates
        self.branch2 = _conv_bn_act(c1, hidden, 3, rates[0])
        self.branch3 = _conv_bn_act(c1, hidden, 3, rates[1])
        self.branch4 = _conv_bn_act(c1, hidden, 3, rates[2])

        # Branch 5: global image pooling — coarse global context
        self.global_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(c1, hidden, 1, bias=False),
            torch.nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            torch.nn.SiLU(inplace=True),
        )

        # Fusion: concat all 5 branches → project to c2
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(hidden * 5, c2, 1, bias=False),
            torch.nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03),
            torch.nn.SiLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        b5 = self.global_pool(x)
        b5 = torch.nn.functional.interpolate(
            b5, size=(H, W), mode="bilinear", align_corners=False)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.fuse(out)


# ═══════════════════════════════════════════════════════════════════
# ASPP — replaces SPPF, tuned for small-object (TB bacilli) detection
# ═══════════════════════════════════════════════════════════════════

class ASPP(torch.nn.Module):
    """
    Atrous Spatial Pyramid Pooling, re-tuned for small objects.

    Standard DeepLab ASPP uses dilation rates 6, 12, 18 — tuned for
    513x513 segmentation inputs with large objects. At those rates,
    on a P5 (stride 32) feature map, the effective sampling distance
    in original-image pixels is 192/384/576px, which is far larger
    than a 15-30px bacillus. The conv ends up sampling background
    noise instead of the object itself.

    This version uses rates [1, 3, 5] by default (overridable),
    keeps the rate=1 branch un-dilated for fine local detail, and
    gives that branch more weight in the final fusion by simply
    including it — it is the only branch with zero gridding effect.

    Branches:
      1. 1x1 conv                      (channel reduction, global mix)
      2. 3x3 conv, dilation=rates[0]   (default 1 — plain local conv)
      3. 3x3 conv, dilation=rates[1]   (default 3)
      4. 3x3 conv, dilation=rates[2]   (default 5)
      5. image pooling branch (global avg pool + 1x1 + upsample)
    All branches projected to out_ch//5... no — projected individually
    to out_ch_branch, concatenated, then fused with a final 1x1 conv.
    """

    def __init__(self, c1, c2, rates=(1, 3, 5)):
        super().__init__()
        assert len(rates) == 3, "ASPP expects exactly 3 dilation rates"
        hidden = c2 // 4   # 4 conv branches share channel budget evenly

        def _conv_bn_act(cin, cout, k, d):
            pad = d * (k - 1) // 2
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, k, 1, pad, dilation=d, bias=False),
                torch.nn.BatchNorm2d(cout, eps=1e-3, momentum=0.03),
                torch.nn.SiLU(inplace=True),
            )

        # Branch 1: 1x1, no dilation — channel mixing only
        self.branch1 = _conv_bn_act(c1, hidden, 1, 1)

        # Branches 2-4: 3x3 with configurable dilation rates
        self.branch2 = _conv_bn_act(c1, hidden, 3, rates[0])
        self.branch3 = _conv_bn_act(c1, hidden, 3, rates[1])
        self.branch4 = _conv_bn_act(c1, hidden, 3, rates[2])

        # Branch 5: global image pooling — coarse global context
        self.global_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(c1, hidden, 1, bias=False),
            torch.nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            torch.nn.SiLU(inplace=True),
        )

        # Fusion: concat all 5 branches → project to c2
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(hidden * 5, c2, 1, bias=False),
            torch.nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03),
            torch.nn.SiLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        b5 = self.global_pool(x)
        b5 = torch.nn.functional.interpolate(
            b5, size=(H, W), mode="bilinear", align_corners=False)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.fuse(out)


class ASPP_SOD(ASPP):
    """
    ASPP variant pre-configured for placement at P2/P3 (stride 4/8)
    rather than P5. At these finer strides the effective receptive
    field per dilation step is already large relative to bacillus
    size, so rates are tightened further to [1, 2, 3].

    Use this variant if you insert ASPP into the neck at P2 or P3
    instead of replacing SPPF at the end of the backbone (P5).
    """
    def __init__(self, c1, c2):
        super().__init__(c1, c2, rates=(1, 2, 3))












































































































































