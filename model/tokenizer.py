import os
from typing import List, Optional

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str | list[str], bos: bool = False, eos: bool = False) -> List[int] | List[List[int]]:
        """
        Encode a string or a list of strings into a list of token IDs

        Args:
            s: string or list of strings
            bos: add BOS token to the beginning of the sequence
            eos: add EOS token to the end of the sequence

        Returns:
            list of token IDs
        """
        t = self.sp_model.encode_as_ids(s)
        if isinstance(s, list):
            for i in range(len(t)):
                if bos:
                    t[i] = [self.bos_id] + t[i]
                if eos:
                    t[i] = t[i] + [self.eos_id]
        else:
            if bos:
                t = [self.bos_id] + t
            if eos:
                t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decode a list of token IDs into a string
        :param t: list of token IDs
        :return: decoded string
        """
        return self.sp_model.decode(t)

    def decode_piece(self, t: int | List[int]) -> str | List[str]:
        """
        Decode a list of token IDs into a string
        :param t: list of token IDs
        :return: decoded string
        """
        return self.sp_model.id_to_piece(t)

    def prepare_encode_instructions(self, user_prompt: str, answer: Optional[str] = None,
                                    sys_prompt: Optional[str] = None,
                                    encoded: bool = False) -> str | List[int]:
        """
        Syntax

        ### System:
        {sys_prompt}

        ### User:
        {question}

        ### Assistant
        {answer}
        """
        if sys_prompt:
            sys_prompt = f"### System:\n{sys_prompt}\n\n"
        else:
            sys_prompt = ""
        question = f"### User:\n{user_prompt}\n\n"
        if answer:
            answer = f"### Assistant:\n{answer}\n"
        else:
            answer = "### Assistant:\n"
        if encoded:
            return self.encode(sys_prompt + question + answer, bos=True, eos=True)
        else:
            return sys_prompt + question + answer
