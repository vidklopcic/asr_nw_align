from dataclasses import dataclass
from typing import Dict, List

import nw_align_probs

import numpy as np


class ASRNWAlign:
    def __init__(self, logprobs: np.ndarray, text: str, config: 'ASRNWAlignConfig' = None):
        self.logprobs = logprobs.astype(np.float64)
        self.text = text
        self.config = config or ASRNWAlignConfig(
            alphabet=[' ', 'a', 'b', 'c', 'č', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                      'n', 'o', 'p', 'r', 's', 'š', 't', 'u', 'v', 'z', 'ž']
        )
        self.alphabet = self.config.alphabet
        self.clean_text = ''.join([c for c in self.text.replace('\n', ' ').lower() if c in self.alphabet])
        self.char_to_token = {self.alphabet[i]: i for i in range(len(self.alphabet))}
        self.token_to_char = {i: self.alphabet[i] for i in range(len(self.alphabet))}
        self.tokens = np.array([self.char_to_token[c] for c in self.clean_text], dtype=np.int64)

    def align_global(self):
        return nw_align_probs.align(
            probs=self.logprobs,
            text=self.tokens,
            h_gap_penalty_for_len=self._h_gap_penalty(),
            v_gap_penalty_for_len=self._v_gap_penalty(),
            h_penalty_exempt=self.config.space_char
        )

    def align_anchor(self):
        return nw_align_probs.align(
            probs=self.logprobs,
            text=self.tokens,
            h_gap_penalty_for_len=self._h_gap_penalty(),
            v_gap_penalty_for_len=self._v_gap_penalty(),
            h_penalty_exempt=-1
        )

    def get_text_positions(self, tokens):
        return [p[1] * self.config.frame_ms / 1000.0 for p in tokens]

    def _h_gap_penalty(self):
        gap = np.zeros((int(self.config.max_char_ms / self.config.frame_ms + 1)), dtype=np.float64)
        gap[-1] = 0
        return gap

    def _v_gap_penalty(self):
        return np.array([-1000], dtype=np.float64)


@dataclass
class ASRNWAlignConfig:
    alphabet: List[str]
    frame_ms: float = 40.3  # todo - why .3? overlap..?
    space_char: int = 0
    max_char_ms: int = 5000


if __name__ == '__main__':
    probs = np.load(open('test/podmelec.np', 'rb'))
    text = open('test/podmelec.txt', 'r', encoding='utf-8').read()
    asr_align = ASRNWAlign(probs, text)
    score, aligned_text, aligned_probs = asr_align.align_global()
    positions = asr_align.get_text_positions(aligned_text)

    position = 0
    for word in asr_align.clean_text.split(' '):
        if not word.strip():
            continue
        start = positions[position]
        position += len(word) + 1
        end = positions[position - 1]
        print(word, f'{start:.2f}-{end:.2f}')
