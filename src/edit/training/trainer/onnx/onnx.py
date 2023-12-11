
from __future__ import annotations
from pathlib import Path
from typing import Any

import logging
LOG = logging.getLogger(__name__)

from edit.training.trainer import EDIT_Inference, EDIT_AutoInference

class Inference(EDIT_Inference):
    _loaded_sessions = {}

    def load_onnx(self, session_name: str, path: str | Path | None = None, options: dict | None = None, **kwargs):
        if session_name in self._loaded_sessions:
            return self._loaded_sessions[session_name]
        
        if path is None:
            raise RuntimeError(f"`path` cannot be None, as model has not been previously loaded")
        
        """
        Get an onnx inference session for a given model number
        """
        import onnxruntime as ort
        # Set the behaviour of onnxruntime
        sess_options = ort.SessionOptions(**options) if options else ort.SessionOptions()
        sess_options.enable_cpu_mem_arena=False
        sess_options.enable_mem_pattern = False
        sess_options.enable_mem_reuse = False

        # Increase the number for faster inference and more memory consumption
        sess_options.intra_op_num_threads = 16

        # Set the behaviour of cuda provider
        cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

        session = ort.InferenceSession(path, sess_options=sess_options, providers=kwargs.pop('providers', [('CUDAExecutionProvider', cuda_provider_options)]), **kwargs)
        LOG.debug(f"{session_name} loaded from {path}.")
        
        self._loaded_sessions[session_name] = session
        return self._loaded_sessions[session_name]

    def load(self, path: str | Path | bool, **kwargs):
        self.model = self.load_onnx('model', path, **kwargs)

    def save(self, path: str | Path, **kwargs):
        raise NotImplementedError(f"Cannot save onnx models")


class AutoInference(Inference, EDIT_AutoInference):
    def _predict_from_data(self, data: Any, *args, **kwargs):
        """
        Expects model to have been loaded under session name 'model'
        """
        session = self.load_onnx('model')
        return session.run(None, data, *args, **kwargs)