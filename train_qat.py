"""Quantization-aware training entry point.

This script exposes a `train_qat` helper that prepares the YOLOv8-CA
architecture for QAT, launches training, and optionally converts the trained
weights to an INT8 checkpoint.  It mirrors the original helper that other
scripts (such as `train_qat_calibrated.py`) expect.
"""

from __future__ import annotations

import argparse
import shutil
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.ao.quantization import FakeQuantize
from tqdm import tqdm


# Ensure the local ultralytics package (vendored in this repo) is importable.
REPO_ROOT = Path(__file__).parent
ULTRALYTICS_PATH = REPO_ROOT / "obc-yolov8" / "ultralytics10.24"
import sys

if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO, __version__  # type: ignore  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # type: ignore  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # type: ignore  # noqa: E402
from ultralytics.nn.tasks import ensure_module_bookkeeping  # noqa: E402
from ultralytics.utils import LOGGER, TQDM_BAR_FORMAT  # type: ignore  # noqa: E402
from ultralytics.utils.ops import Profile  # type: ignore  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel  # type: ignore  # noqa: E402


DEFAULT_PROJECT = REPO_ROOT / "runs" / "detect"
DEFAULT_NAME = "train_qat"
DEFAULT_MODEL_CFG = ULTRALYTICS_PATH / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-CA.yaml"
DEFAULT_DATA_CFG = ULTRALYTICS_PATH / "ultralytics" / "cfg" / "datasets" / "combined_china_motorbike.yaml"


class QATDetectionValidator(DetectionValidator):
    """Validator variant that keeps QAT models in float precision."""

    def __call__(self, trainer=None, model=None):  # type: ignore[override]
        if trainer is None:
            self.args.half = False
            return super().__call__(trainer=trainer, model=model)

        self.training = True
        augment = False
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = False
        model = trainer.model.float()
        self.model = model
        self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        self.run_callbacks('on_val_start')
        self.model.eval()

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(self.model))

        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i

            with dt[0]:
                batch = self.preprocess(batch)

            with torch.no_grad():
                with dt[1]:
                    preds = self.model(batch['img'], augment=augment)

            with dt[2]:
                self.loss += self.model.loss(batch, preds)[1]

            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks('on_val_batch_end')

        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')

        self.model.float()
        results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
        return {k: round(float(v), 5) for k, v in results.items()}

    def preprocess(self, batch):  # type: ignore[override]
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)
        if self.args.save_hybrid:
            height, width = batch['img'].shape[2:]
            nb = len(batch['img'])
            bboxes = batch['bboxes'] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch['cls'][batch['batch_idx'] == i], bboxes[batch['batch_idx'] == i]], dim=-1)
                for i in range(nb)
            ]
        return batch


class QATDetectionTrainer(DetectionTrainer):
    """Detection trainer that reuses a pre-prepared QAT model and disables EMA."""

    class _NoEMA:
        def __init__(self, model: torch.nn.Module):
            self.ema = model
            self.updates = 0

        def update(self, model: torch.nn.Module):
            self.ema = model
            self.updates += 1

        def update_attr(self, model: torch.nn.Module, include=None):
            for name in include or []:
                if hasattr(model, name):
                    setattr(self.ema, name, getattr(model, name))

        def half(self):
            return self

        def float(self):
            return self

        def __bool__(self) -> bool:
            return True

    def __init__(self, *args, prepared_model: Optional[torch.nn.Module] = None, **kwargs):
        self._prepared_model = prepared_model
        super().__init__(*args, **kwargs)
        if self._prepared_model is not None:
            self.model = self._prepared_model
            self.args.model = self._prepared_model

    def setup_model(self):  # type: ignore[override]
        if isinstance(self.model, torch.nn.Module):
            return
        return super().setup_model()

    def _setup_train(self, world_size):  # type: ignore[override]
        result = super()._setup_train(world_size)
        self.ema = self._NoEMA(self.model)
        return result

    def save_model(self):  # type: ignore[override]
        model_float = deepcopy(de_parallel(self.model)).float()
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': model_float,
            'ema': deepcopy(model_float),
            'updates': getattr(self.ema, 'updates', 0),
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),
            'date': datetime.now().isoformat(),
            'version': __version__,
        }

        try:
            import dill as pickle
        except ImportError:  # pragma: no cover
            import pickle

        torch.save(ckpt, self.last, pickle_module=pickle)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best, pickle_module=pickle)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt', pickle_module=pickle)

        del ckpt

    def get_validator(self):  # type: ignore[override]
        validator = QATDetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
        validator.args.half = False
        return validator


def _resolve_device(device: Any) -> str:
    if isinstance(device, (list, tuple)):
        return ",".join(map(str, device))
    return str(device)


def train_qat(
    model_cfg: str | Path = DEFAULT_MODEL_CFG,
    data_cfg: str | Path = DEFAULT_DATA_CFG,
    epochs: int = 5,
    imgsz: int = 640,
    batch: Optional[int] = None,
    workers: Optional[int] = None,
    device: Any = 0,
    backend: str = "fbgemm",
    save_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    save_qat_checkpoint: bool = True,
    convert_to_int8: bool = True,
    use_manual_quantization: bool = False,
    resume: bool = False,
    weights: Optional[str | Path] = None,
    lr0: Optional[float] = None,
    lrf: Optional[float] = None,
    warmup_epochs: Optional[float] = None,
    warmup_bias_lr: Optional[float] = None,
    optimizer: Optional[str] = None,
    pretrained: Optional[bool] = None,
    patience: Optional[int] = None,
    **train_kwargs: Any,
) -> Dict[str, Optional[Path]]:
    """Run QAT training and optionally export an INT8 model.

    Returns a mapping with keys `qat_path`, `int8_path`, and `weights_dir`.  Missing
    artifacts are reported as ``None``.
    """

    project_dir = Path(save_dir) if save_dir is not None else DEFAULT_PROJECT
    run_name = run_name or DEFAULT_NAME
    project_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing YOLO model for QAT...")
    model = YOLO(str(model_cfg))
    if weights is not None:
        weights_path = str(weights)
        if not Path(weights_path).exists():
            LOGGER.warning(f"Specified weights path does not exist: {weights_path}")
        else:
            LOGGER.info(f"Loading pretrained weights from {weights_path}")
            try:
                model.load(weights_path)
            except Exception as load_err:
                LOGGER.warning(f"Failed to load weights '{weights_path}': {load_err}")

    detection_model = model.model
    if not hasattr(detection_model, "prepare_for_qat"):
        raise AttributeError("Loaded model does not implement prepare_for_qat()")

    example_input = torch.randn(1, detection_model.yaml.get("ch", 3), imgsz, imgsz)
    LOGGER.info(f"Using QAT backend '{backend}' with image size {imgsz}")
    prepared_model = detection_model.prepare_for_qat(
        backend=backend,
        example_input=example_input,
        use_fx=not use_manual_quantization,
    )
    model.model = prepared_model

    device_str = _resolve_device(device)
    train_args: Dict[str, Any] = {
        "data": str(data_cfg),
        "epochs": epochs,
        "imgsz": imgsz,
        "project": str(project_dir),
        "name": run_name,
        "device": device_str,
        "val": True,
        "verbose": True,
        **train_kwargs,
    }
    if batch is not None:
        train_args["batch"] = batch
    if workers is not None:
        train_args["workers"] = workers
    if resume:
        train_args["resume"] = resume
    if lr0 is not None:
        train_args["lr0"] = lr0
    if lrf is not None:
        train_args["lrf"] = lrf
    if warmup_epochs is not None:
        train_args["warmup_epochs"] = warmup_epochs
    if warmup_bias_lr is not None:
        train_args["warmup_bias_lr"] = warmup_bias_lr
    if optimizer is not None:
        train_args["optimizer"] = optimizer
    if pretrained is not None:
        train_args["pretrained"] = pretrained
    if patience is not None:
        train_args["patience"] = patience

    overrides = {**model.overrides, **train_args}
    overrides["task"] = model.task
    overrides["half"] = False

    trainer = QATDetectionTrainer(overrides=overrides, _callbacks=model.callbacks, prepared_model=prepared_model)
    trainer.hub_session = model.session
    model.trainer = trainer

    LOGGER.info("Starting QAT training run...")
    trainer.train()

    model.model = trainer.model
    results = getattr(trainer.validator, "metrics", None)

    save_dir_path = Path(trainer.save_dir)
    weights_dir = save_dir_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    best_pt = Path(trainer.best) if getattr(trainer, "best", None) else weights_dir / "best.pt"
    last_pt = Path(trainer.last) if getattr(trainer, "last", None) else weights_dir / "last.pt"
    
    best_qat_path = None
    last_qat_path = None
    if save_qat_checkpoint:
        best_qat_path = weights_dir / "best_qat.pt"
        ensure_module_bookkeeping(trainer.model, recursive=True)
        torch.save({"model": trainer.model, "backend": backend, "qat": True}, best_qat_path)
        LOGGER.info(f"Saved QAT checkpoint to {best_qat_path}")
        if best_pt.exists():
            LOGGER.info(f"FP32 checkpoint available at {best_pt}")
        else:
            LOGGER.warning("FP32 best.pt not found after training; only QAT checkpoint saved")

    int8_path = None
    last_int8_path = None
    int8_eval_results = None
    
    if convert_to_int8:
        # Check if last.pt exists and convert it to INT8
        if last_pt.exists():
            LOGGER.info(f"Loading last.pt checkpoint from {last_pt} for INT8 conversion...")
            try:
                checkpoint = torch.load(last_pt, map_location='cpu', weights_only=False)
                model_state = checkpoint.get('model', checkpoint)
                
                # Load the model from checkpoint
                # The checkpoint contains the full QAT model (with FakeQuantize modules)
                LOGGER.info("Loading QAT model from last.pt checkpoint...")
                if isinstance(model_state, dict):
                    # State dict - need to load into fresh QAT model
                    fresh_model = YOLO(str(model_cfg))
                    example_input = torch.randn(1, fresh_model.model.yaml.get("ch", 3), imgsz, imgsz)
                    fresh_qat_model = fresh_model.model.prepare_for_qat(
                        backend=backend,
                        example_input=example_input,
                        use_fx=not use_manual_quantization,
                    )
                    fresh_qat_model.load_state_dict(model_state, strict=False)
                else:
                    # Full model object (this is what QATDetectionTrainer.save_model() saves)
                    fresh_qat_model = model_state
                    # Check if it's already a QAT model (has FakeQuantize modules)
                    has_fake_quant = any(isinstance(m, FakeQuantize) for m in fresh_qat_model.modules())
                    if not has_fake_quant or not hasattr(fresh_qat_model, "convert_to_quantized"):
                        # Re-prepare if not in QAT mode
                        LOGGER.info("Model not in QAT mode, re-preparing for QAT...")
                        example_input = torch.randn(1, fresh_qat_model.yaml.get("ch", 3), imgsz, imgsz)
                        fresh_qat_model = fresh_qat_model.prepare_for_qat(
                            backend=backend,
                            example_input=example_input,
                            use_fx=not use_manual_quantization,
                        )
                    else:
                        LOGGER.info("Model already in QAT mode (FakeQuantize modules detected)")
                
                # Save last_qat.pt
                if save_qat_checkpoint:
                    last_qat_path = weights_dir / "last_qat.pt"
                    ensure_module_bookkeeping(fresh_qat_model, recursive=True)
                    torch.save({"model": fresh_qat_model, "backend": backend, "qat": True}, last_qat_path)
                    LOGGER.info(f"Saved last QAT checkpoint to {last_qat_path}")
                
                # Convert to INT8
                fakequant_count = sum(1 for module in fresh_qat_model.modules() if isinstance(module, FakeQuantize))
                if fakequant_count == 0:
                    LOGGER.error("No FakeQuantize modules present in last.pt model; skipping INT8 conversion.")
                else:
                    LOGGER.info(f"Converting last.pt QAT model to INT8 (found {fakequant_count} FakeQuantize modules)...")
                    fresh_qat_model.eval()
                    fresh_qat_model = fresh_qat_model.float()
                    quantized_model = fresh_qat_model.convert_to_quantized()
                    ensure_module_bookkeeping(quantized_model, recursive=True)
                    last_int8_path = weights_dir / "last_int8.pt"
                    torch.save({"model": quantized_model, "backend": backend, "int8": True}, last_int8_path)
                    LOGGER.info(f"Saved INT8 checkpoint to {last_int8_path}")
                    
                    # Evaluate INT8 model
                    LOGGER.info("Evaluating INT8 model...")
                    try:
                        int8_yolo = YOLO(str(model_cfg))
                        int8_yolo.model = quantized_model
                        int8_yolo.model.eval()
                        
                        # Run evaluation using YOLO's val method
                        int8_eval_results = int8_yolo.val(
                            data=str(data_cfg),
                            imgsz=imgsz,
                            batch=train_args.get("batch", 16),
                            device=device_str,
                            plots=False,
                            save=False,
                            verbose=True
                        )
                        LOGGER.info("INT8 model evaluation completed")
                        if int8_eval_results:
                            LOGGER.info(f"INT8 mAP@0.5: {int8_eval_results.get('metrics/mAP50(B)', 'N/A')}")
                            LOGGER.info(f"INT8 mAP@0.5:0.95: {int8_eval_results.get('metrics/mAP50-95(B)', 'N/A')}")
                    except Exception as eval_err:
                        LOGGER.warning(f"INT8 evaluation failed: {eval_err}")
                        import traceback
                        LOGGER.debug(traceback.format_exc())
                        int8_eval_results = None
            except Exception as conv_err:
                LOGGER.error(f"Failed to convert last.pt to INT8: {conv_err}")
                import traceback
                LOGGER.error(traceback.format_exc())
        else:
            LOGGER.warning(f"last.pt not found at {last_pt}; skipping INT8 conversion")
        
        # Also convert best.pt for backward compatibility (if it exists and is different from last.pt)
        if best_pt.exists() and best_pt != last_pt:
            fakequant_count = sum(1 for module in trainer.model.modules() if isinstance(module, FakeQuantize))
            if fakequant_count == 0:
                LOGGER.warning("No FakeQuantize modules present in trainer.model; skipping best.pt INT8 conversion.")
            else:
                LOGGER.info("Converting best.pt QAT model to INT8 (for backward compatibility)...")
                trainer.model.eval()
                trainer.model = trainer.model.float()
                quantized_model = trainer.model.convert_to_quantized()
                ensure_module_bookkeeping(quantized_model, recursive=True)
                int8_path = weights_dir / "best_int8.pt"
                torch.save({"model": quantized_model, "backend": backend, "int8": True}, int8_path)
                LOGGER.info(f"Saved best INT8 checkpoint to {int8_path}")

    LOGGER.info("QAT run complete")
    return {
        "qat_path": best_qat_path,
        "last_qat_path": last_qat_path,
        "int8_path": int8_path,
        "last_int8_path": last_int8_path,
        "int8_eval_results": int8_eval_results,
        "weights_dir": weights_dir,
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantization-aware training entry point")
    parser.add_argument(
        "--model",
        "--model-cfg",
        dest="model_cfg",
        default=str(DEFAULT_MODEL_CFG),
        help="Model YAML path (default: original YOLOv8-CA config)",
    )
    parser.add_argument(
        "--data",
        "--data-cfg",
        dest="data_cfg",
        default=str(DEFAULT_DATA_CFG),
        help="Dataset YAML path (default: combined_china_motorbike)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=0)
    parser.add_argument("--backend", default="fbgemm")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--name", dest="run_name")
    parser.add_argument("--no-save-qc", dest="save_qat_checkpoint", action="store_false")
    parser.add_argument("--no-int8", dest="convert_to_int8", action="store_false")
    parser.add_argument("--manual", dest="use_manual_quantization", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weights", type=str, default=None, help="Path to FP32 checkpoint to fine-tune from")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate override")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate fraction override")
    parser.add_argument("--warmup-epochs", type=float, default=None, dest="warmup_epochs", help="Warmup epochs override")
    parser.add_argument("--warmup-bias-lr", type=float, default=None, dest="warmup_bias_lr", help="Warmup bias LR override")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer name (SGD, Adam, AdamW, etc.). Default: auto")
    parser.add_argument("--pretrained", type=lambda x: x.lower() in ['true', '1', 'yes'], default=None, metavar='BOOL', help="Use pretrained weights (true/false/1/0). Default: True (YOLO default). Use --pretrained false to disable.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs to wait after fitness stops improving). Set to 0 to disable early stopping.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outputs = train_qat(**vars(args))
    qat_path = outputs.get("qat_path")
    if qat_path is not None:
        LOGGER.info(f"Best QAT checkpoint: {qat_path}")
    last_qat_path = outputs.get("last_qat_path")
    if last_qat_path is not None:
        LOGGER.info(f"Last QAT checkpoint: {last_qat_path}")
    int8_path = outputs.get("int8_path")
    if int8_path is not None:
        LOGGER.info(f"Best INT8 checkpoint: {int8_path}")
    last_int8_path = outputs.get("last_int8_path")
    if last_int8_path is not None:
        LOGGER.info(f"Last INT8 checkpoint: {last_int8_path}")
    int8_eval_results = outputs.get("int8_eval_results")
    if int8_eval_results is not None:
        LOGGER.info(f"INT8 evaluation results: {int8_eval_results}")

