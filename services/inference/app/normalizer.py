import html
import logging
import re

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import Settings


logger = logging.getLogger("jobl.inference.normalizer")

# IMPORTANT: keep pre_strip logic identical to jobl-training/app/build_jsonl_runner.py
# Any changes here must be mirrored there and vice versa.
PRE_STRIP_PATTERNS = [
    (
        "salary_rate_symbol",
        re.compile(
            r"[$€£]\s?\d[\d,k.]*\s?(/hr|/hour|/yr|/year|ph|pa)?",
            re.IGNORECASE,
        ),
    ),
    ("salary_rate_upto", re.compile(r"\bup\s?to\s?[$€£]\s?[\d,k.]+\b", re.IGNORECASE)),
    ("job_code_hash", re.compile(r"\s*#[A-Z]{2,6}\b", re.IGNORECASE)),
    ("numeric_job_code_parens", re.compile(r"\(\s*\d{4}[-–]\d{2,6}\s*\)", re.IGNORECASE)),
    (
        "employment_type",
        re.compile(
            r"full[\s-]?time|part[\s-]?time|casual|\btemporary\b|\btemp(?!\w)\b|fixed[\s-]?term|\bpermanent(?!\s+reliever)\b|perm(?!\w)|on[\s-]?call|sur appel",
            re.IGNORECASE,
        ),
    ),
    ("early_careers", re.compile(r"early careers?|new\s?grad(uate)?", re.IGNORECASE)),
    (
        "campaign_text",
        re.compile(
            r"apply\s+now|hiring\s+now|no\s+experience\s+needed|start\s+your\s+career\s+with\s+us\s+today[!?]*|possibility\s+for\s+conversion\s+to\s+perm",
            re.IGNORECASE,
        ),
    ),
    ("multiple_positions", re.compile(r"\(?\bmultiple\s+positions?\b\)?", re.IGNORECASE)),
    ("opportunities_suffix", re.compile(r"\bopportunities\b", re.IGNORECASE)),
]


def pre_strip(title: str) -> str:
    cleaned = html.unescape(str(title or ""))
    for _name, pattern in PRE_STRIP_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
        cleaned = _cleanup_separators_and_spaces(cleaned)
    return cleaned


def _cleanup_separators_and_spaces(value: str) -> str:
    cleaned = str(value or "")
    while True:
        previous = cleaned
        cleaned = re.sub(r"^\s*[-–—|,]+\s*", "", cleaned)
        cleaned = re.sub(r"\s*[-–—|,]+\s*$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned == previous:
            break
    return cleaned


def _fix_casing(title: str) -> str:
    """Apply title case only when model output is entirely lowercase."""
    if title == title.lower():
        return title.title()
    return title


def _normalize_separators(title: str) -> str:
    """Ensure consistent spacing around dash separators."""
    title = re.sub(r"\s*[-–—]\s*", " - ", title)
    return re.sub(r"\s+", " ", title).strip()


class JobTitleNormalizer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, use_fast=True)
            model_config = AutoConfig.from_pretrained(settings.model_dir)
            model_config.tie_word_embeddings = False
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.model_dir,
                config=model_config,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self._ready = True
            logger.info(
                "model loaded model_dir=%s num_beams=%s",
                settings.model_dir,
                settings.num_beams,
            )
        except Exception as exc:
            self._ready = False
            raise RuntimeError(f"failed to load model artifacts from MODEL_DIR={settings.model_dir}") from exc

    def is_ready(self) -> bool:
        return self._ready

    def normalize(self, title_raw: str) -> str:
        return self.normalize_batch([title_raw])[0]

    def normalize_batch(self, titles: list[str]) -> list[str]:
        prompts = [f"normalize job title: {pre_strip(title)}" for title in titles]
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.settings.max_input_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=self.settings.num_beams,
                max_new_tokens=self.settings.max_new_tokens,
                early_stopping=True,
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [_normalize_separators(_fix_casing(title)) for title in decoded]
