# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from pathlib import Path
import logging
from typing import Optional

import onnxruntime as ort

from pyearthtools.pipeline.controller import Pipeline

from pyearthtools.training.wrapper.wrapper import ModelWrapper
from pyearthtools.training.data.datamodule import PipelineDataModule


LOG = logging.getLogger(__name__)


class ONNXWrapper(ModelWrapper):
    """
    ONNX Model Wrapper

    Allows for loading of `onnx` sessions, accessible at `self.onnx(SESSION_NAME)`
    """

    _loaded_sessions = {}
    model_name: str = "model"

    def __init__(self, data: Pipeline | PipelineDataModule, model: Optional[ort.InferenceSession] = None):
        """
        Wrap `Onnx` model to allow usage within `pyearthtools`.

        Can only run predictions, not training.

        Args:
            data (Pipeline | PipelineDataModule):
                Data to use to run predictions with
            model (Optional[ort.InferenceSession], optional):
                Preloaded onnx session to use, will be saved under `self.model_name`. Defaults to None.
        """
        super().__init__(None, data)
        self.record_initialisation()

        if model is not None:
            self._loaded_sessions[self.model_name] = model

    def predict(self, data, onnx_name: Optional[str] = None, *args, **kwargs):
        """
        Predict with `onnx` model

        Args:
            data (Any):
                Data to run prediction with
            onnx_name (Optional[str], optional):
                Onnx session name to use. Defaults to None.

        Returns:
            (Any):
                Predicted data
        """
        onnx_name = onnx_name or self.model_name
        session = self.onnx(onnx_name)
        return session.run(None, data, *args, **kwargs)

    def load_onnx(
        self, session_name: str, path: str | Path | None = None, options: dict | None = None, **kwargs
    ) -> ort.InferenceSession:
        """
        Load an onnx session, and cache it

        A session can be retrieved after it is loaded, by just passing `session_name`

        Args:
            session_name (str):
                Name of onnx session, used for caching
            path (str | Path | None, optional):
                Path to onnx file. Needed if session not already loaded. Defaults to None.
            options (dict | None, optional):
                Options to pass to onnx session. Defaults to None.
            kwargs (Any, optional):
                All kwargs passes to `onnxruntime.InferenceSession`

        Raises:
            RuntimeError:
                If `path` not set, and session not already loaded

        Returns:
            (ort.InferenceSession):
                Loaded onnx session
        """
        if session_name in self._loaded_sessions:
            return self._loaded_sessions[session_name]

        if path is None:
            raise RuntimeError("`path` cannot be None, as session has not been previously loaded")

        """
        Get an onnx inference session for a given model number
        """

        # Set the behaviour of onnxruntime
        sess_options = ort.SessionOptions(**options) if options else ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False

        # Increase the number for faster inference and more memory consumption
        sess_options.intra_op_num_threads = kwargs.pop("intra_op_num_threads", 16)

        # Set the behaviour of cuda provider
        cuda_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        if ort.get_device() != "GPU":
            LOG.warn(
                f"Onnx Runtime is running on {ort.get_device()!s}, this may slow down inference time. (With {session_name})."
            )
            kwargs["providers"] = kwargs.pop("providers", ["CPUExecutionProvider"])

        session = ort.InferenceSession(
            path,
            sess_options=sess_options,
            providers=kwargs.pop("providers", [("CUDAExecutionProvider", cuda_provider_options)]),
            **kwargs,
        )
        LOG.debug(f"Onnx model: {session_name} loaded from {path!s}.")

        self._loaded_sessions[session_name] = session
        return self._loaded_sessions[session_name]

    def onnx(self, session_name: str) -> ort.InferenceSession:
        """
        Convenience function for `load_onnx`.

        Uses just `session_name` expecting it to be loaded already

        Args:
            session_name (str):
                Name of onnx session

        Raises:
            KeyError:
                If session not already loaded

        Returns:
            (ort.InferenceSession):
                Loaded onnx session
        """
        try:
            return self.load_onnx(session_name)
        except ValueError:
            pass
        raise KeyError(f"Onnx session has not been loaded, cannot retrieve session. {session_name}")

    def load(self, path: str | Path, **kwargs):
        self.model = self.load_onnx(self.model_name, path, **kwargs)

    def save(self, path: str | Path, **kwargs):
        raise NotImplementedError("Cannot save onnx models.")
