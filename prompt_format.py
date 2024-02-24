from enum import Enum
from typing import TypedDict, List, Optional

from tokenizers import Tokenizer

START = "<|im_start|>"
END = "<|im_end|>"


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
    def __init__(self, sys_prompt: Optional[str] = None):
        self.messages: List[Message] = []
        if sys_prompt:
            self.messages.append({"role": Role.SYSTEM, "content": sys_prompt, })
        else:
            self.messages.append({"role": Role.SYSTEM, "content": DEFAULT_SYSTEM_PROMPT, })

    def add_user_message(self, content: str):
        self.messages.append({"role": Role.USER, "content": content, })

    def add_assistant_message(self, content: str):
        self.messages.append({"role": Role.ASSISTANT, "content": content, })

    def add_messages(self, messages: list[str]):
        """
        Assumes that the messages are in the order of user, assistant, user, assistant, ...
        :param messages:
        :return: 
        """
        for i, message in enumerate(messages):
            role = Role.USER if i % 2 == 0 else Role.ASSISTANT
            self.messages.append({"role": role, "content": message, })

    def get_tokens(self, tokenizer: Optional[Tokenizer] = None):
        dialog_tokens = ""
        for message in self.messages:
            dialog_tokens += f"{START}{message['role'].value}\n{message['content']}{END}\n"
        dialog_tokens = dialog_tokens.strip()
        return dialog_tokens if not tokenizer else tokenizer.encode(dialog_tokens).ids

    def get_tokens_for_completion(self, tokenizer: Optional[Tokenizer] = None):
        if self.messages[-1]['role'] != Role.USER:
            raise ValueError("The last message should be from the user")
        dialog_tokens = self.get_tokens(tokenizer)
        dialog_tokens += f"\n{START}{Role.ASSISTANT.value}\n"
        return dialog_tokens if not tokenizer else tokenizer.encode(dialog_tokens).ids
