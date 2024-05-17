# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations
from pathlib import Path
from typing import Any

from edit.training.trainer import EDIT_Inference, EDIT_AutoInference

import logging

LOG = logging.getLogger(__name__)


class Inference(EDIT_Inference):
    _loaded_sessions = {}

    def load_onnx(
        self, session_name: str, path: str | Path | None = None, options: dict | None = None, **kwargs
    ) -> "onnxruntime.InferenceSession":
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
            (onnxruntime.InferenceSession):
                Loaded onnx session
        """
        if session_name in self._loaded_sessions:
            return self._loaded_sessions[session_name]

        if path is None:
            raise RuntimeError(f"`path` cannot be None, as session has not been previously loaded")

        """
        Get an onnx inference session for a given model number
        """
        import onnxruntime as ort

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
            LOG.warn(f"Onnx Runtime is running on {ort.get_device()!s}, this may slow down inference time. (With {session_name}).")
            kwargs['providers'] =  kwargs.pop('providers', ['CPUExecutionProvider'])

        session = ort.InferenceSession(
            path,
            sess_options=sess_options,
            providers=kwargs.pop("providers", [("CUDAExecutionProvider", cuda_provider_options)]),
            **kwargs,
        )
        LOG.debug(f"Onnx model: {session_name} loaded from {path!s}.")

        self._loaded_sessions[session_name] = session
        return self._loaded_sessions[session_name]

    def onnx(self, session_name: str) -> "onnxruntime.InferenceSession":
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
            (onnxruntime.InferenceSession):
                Loaded onnx session
        """
        try:
            return self.load_onnx(session_name)
        except ValueError:
            pass
        raise KeyError(f"Onnx session has not been loaded, cannot retrieve session. {session_name}")

    def load(self, path: str | Path, **kwargs):
        self.model = self.load_onnx("model", path, **kwargs)

    def save(self, path: str | Path, **kwargs):
        raise NotImplementedError(f"Cannot save onnx models")


class AutoInference(Inference, EDIT_AutoInference):
    def _predict_from_data(self, data: Any, *args, **kwargs):
        """
        Expects model to have been loaded under session name 'model'
        """
        session = self.onnx("model")
        return session.run(None, data, *args, **kwargs)
