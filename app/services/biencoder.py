import logging

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger("jobl.api.biencoder")


class BiEncoder:
    def __init__(self, model_path: str) -> None:
        self._ready = False
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 4
            self._session = ort.InferenceSession(
                f"{model_path}/model.onnx",
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = {inp.name for inp in self._session.get_inputs()}
            self._ready = True
            logger.info("biencoder loaded from %s (inputs: %s)", model_path, self._input_names)
        except Exception as exc:
            raise RuntimeError(f"failed to load biencoder from {model_path}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def encode(self, text: str) -> list[float]:
        """Encode a single text, return a normalized 1024-dim embedding."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts, return list of normalized embeddings."""
        inputs = self._tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )

        if "position_ids" in self._input_names and "position_ids" not in inputs:
            seq_len = inputs["input_ids"].shape[1]
            batch_size = inputs["input_ids"].shape[0]
            inputs["position_ids"] = np.tile(
                np.arange(seq_len, dtype=np.int64), (batch_size, 1)
            )

        feed = {k: v for k, v in inputs.items() if k in self._input_names}
        outputs = self._session.run(None, feed)

        # Last-token pooling + L2 normalization (Harrier architecture)
        embeddings = []
        for i in range(len(texts)):
            emb = outputs[0][i, -1, :]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb.tolist())

        return embeddings
