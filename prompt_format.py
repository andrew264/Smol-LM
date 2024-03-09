from enum import Enum
from typing import TypedDict, List, Optional, Union

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


class Role(Enum):
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"


class Message(TypedDict):
    role: Role
    content: str


DEFAULT_SYSTEM_PROMPT = \
    "You are a sentient, superintelligent artificial general intelligence, here to teach and assist me."


class Prompt:
    EOT = '</s>'

    def __init__(self, sys_prompt: Optional[str] = None,
                 tokenizer: Optional[Union[Tokenizer, PreTrainedTokenizerFast]] = None):
        self.messages: List[Message] = []
        self.tokenizer = tokenizer
        self.add_system_message(sys_prompt or DEFAULT_SYSTEM_PROMPT)

    def add_system_message(self, content: str):
        self.messages.append({"role": Role.SYSTEM, "content": content, })

    def add_user_message(self, content: str):
        self.messages.append({"role": Role.USER, "content": content, })

    def add_assistant_message(self, content: str):
        self.messages.append({"role": Role.ASSISTANT, "content": content, })

    def num_tokens(self) -> int:
        assert self.tokenizer is not None, "Tokenizer is not set"
        return len(self.get_tokens(True))

    def num_tokens_for_completion(self) -> int:
        assert self.tokenizer is not None, "Tokenizer is not set"
        return len(self.get_tokens_for_completion(True))

    def num_exchanges(self) -> int:
        return (len(self.messages) - 1) // 2

    def remove_first_exchange(self) -> List[Message]:
        removed = self.messages[1:3]
        self.messages = [self.messages[0]] + self.messages[2:]
        return removed

    def remove_last_exchange(self) -> List[Message]:
        removed = self.messages[-2:]
        self.messages = self.messages[:-2]
        return removed

    def reset(self):
        self.messages = [self.messages[0]]

    def add_messages(self, messages: list[str]):
        """
        Assumes that the messages are in the order of user, assistant, user, assistant, ...
        :param messages:
        :return: 
        """
        for i, message in enumerate(messages):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            self.messages.append({"role": role, "content": message, })

    def get_tokens(self, tokenized: bool = False) -> Union[str, List[int]]:
        dialog_tokens = ""
        for message in self.messages:
            dialog_tokens += f"{message['role'].value}\n{message['content']}{self.EOT}\n"
        dialog_tokens = dialog_tokens.strip()
        if not tokenized:
            return dialog_tokens
        else:
            if isinstance(self.tokenizer, Tokenizer):
                return self.tokenizer.encode(dialog_tokens).ids
            else:
                return self.tokenizer.encode(dialog_tokens)

    def get_tokens_for_completion(self, tokenized: bool = False) -> Union[str, List[int]]:
        if self.messages[-1]['role'] != Role.USER:
            raise ValueError("The last message should be from the user")
        dialog_tokens = self.get_tokens(False) + f"{Role.ASSISTANT.value}\n"
        if not tokenized:
            return dialog_tokens
        else:
            if isinstance(self.tokenizer, Tokenizer):
                return self.tokenizer.encode(dialog_tokens).ids
            else:
                return self.tokenizer.encode(dialog_tokens)
