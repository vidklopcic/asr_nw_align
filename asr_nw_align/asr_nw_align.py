import json
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
import nw_align_probs

import numpy as np


class ASRNWAlign:
    def __init__(self, logprobs_and_errors: np.ndarray, text: str, config: 'ASRNWAlignConfig' = None):
        self.logprobs = logprobs_and_errors[0].astype(np.float64)
        self.frame_errors = logprobs_and_errors[1]
        self.text = text
        self.config = config or ASRNWAlignConfig(
            alphabet=[' ', 'a', 'b', 'c', 'č', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                      'n', 'o', 'p', 'r', 's', 'š', 't', 'u', 'v', 'z', 'ž']
        )
        self.alphabet = self.config.alphabet
        self.clean_text, self.clean_to_source = self.get_clean_text()
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
        frame_error_i = 0
        ms_positions = []
        for _, frame_i in tokens:
            while frame_error_i < len(self.frame_errors) and self.frame_errors[frame_error_i][0] < frame_i:
                frame_error_i += 1
            frame_error_i = min(len(self.frame_errors) - 1, frame_error_i)
            frame_error = self.frame_errors[frame_error_i][1]
            ms_positions.append((frame_i - frame_error) * self.config.frame_ms / 1000.0)
        return ms_positions

    def _h_gap_penalty(self):
        gap = np.zeros((int(self.config.max_char_ms / self.config.frame_ms + 1)), dtype=np.float64)
        gap[-1] = 0
        return gap

    def _v_gap_penalty(self):
        return np.array([-1000], dtype=np.float64)

    def get_clean_text(self):
        clean_to_source = {}
        lower_text = self.text.lower().replace('\n', ' ')
        offset = 0
        clean_chars = []
        for i in range(len(text)):
            c = lower_text[i]
            if c not in self.alphabet:
                offset += 1
                continue
            clean_chars.append(c)
            clean_to_source[i - offset] = i
        return ''.join(clean_chars), clean_to_source


@dataclass
class ASRNWAlignConfig:
    alphabet: List[str]
    frame_ms: float = 40
    space_char: int = 0
    max_char_ms: int = 5000


if __name__ == '__main__':
    FN = 'podmelec'
    data = np.load(open(f'test/{FN}.np', 'rb'), allow_pickle=True)
    text = open(f'test/{FN}.txt', 'r', encoding='utf-8').read()
    asr_align = ASRNWAlign(data, text)


    def save_for_web():
        score, aligned_text, aligned_probs = asr_align.align_global()
        with open(f'test/{FN}_aligned.json', 'w', encoding='utf-8') as f:
            json.dump({
                'text': asr_align.text,
                'text_clean': asr_align.clean_text,
                'clean_to_source': asr_align.clean_to_source,
                'clean_text_positions_s': asr_align.get_text_positions(aligned_text)
            }, f)


    def print_word_positions():
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


    def plot_trace():
        def draw_score_image(score: np.ndarray):
            print('drawing image\n')
            score = score.T
            color = lambda x: (
                255 if x == 1 else 0,
                255 if x == 2 else 0,
                255 if x == 3 else 0
            )
            print(f'0.00%')
            image = Image.new('RGB', score.shape, color='black')
            for x in range(score.shape[0]):
                if x % 100 == 0:
                    print(f'{100 * x / score.shape[0]:.2f}%')
                for y in range(score.shape[1]):
                    image.putpixel((x, y), color(score[x][y]))
            return image

        score, trace = nw_align_probs.gen_mat(
            probs=asr_align.logprobs,
            text=asr_align.tokens,
            h_gap_penalty_for_len=asr_align._h_gap_penalty(),
            v_gap_penalty_for_len=asr_align._v_gap_penalty(),
            h_penalty_exempt=asr_align.config.space_char
        )

        draw_score_image(trace).save('trace.png')


    # plot_trace()
    # print_word_positions()
    save_for_web()
