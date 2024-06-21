# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor

logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        # SentencePieceProcessor用于进行文本的分词和标记化
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        print(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size() #语料库的size
        self.bos_id: int = self.sp_model.bos_id() #开始符id 用于表示一个序列的开始。
        self.eos_id: int = self.sp_model.eos_id() #结束符id 用于表示一个序列的结束。
        self.pad_id: int = self.sp_model.pad_id() #填充符(Padding)的id 当需要将多个序列长度对齐时,可以使用pad_id在较短序列后面填充。
        print(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
