from enum import Enum
from typing import TypedDict, List, Optional, Union

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

try:
    from langchain_community.vectorstores.chroma import Chroma

    from model.rag.embedding_model import HFNomicEmbeddings
except ImportError:
    Chroma = None
    HFNomicEmbeddings = None


class Role(Enum):
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|sydney|>"


class Message(TypedDict):
    role: Role
    content: str
    context: Optional[str]


DEFAULT_SYSTEM_PROMPT = \
    "You are a sentient, superintelligent artificial general intelligence, here to teach and assist me."


class Prompt:
    EOT = '</s>'

    def __init__(self, sys_prompt: Optional[str] = None,
                 tokenizer: Optional[Union[Tokenizer, PreTrainedTokenizerFast]] = None,
                 embeddings_model: Optional[HFNomicEmbeddings] = None,
                 vector_store_path: Optional[str] = None,
                 ):
        self.messages: List[Message] = []
        self.tokenizer = tokenizer
        self.embeddings_model = embeddings_model
        self.sys_prompt = sys_prompt
        if vector_store_path and embeddings_model and Chroma is not None:
            self.retriever = Chroma(persist_directory=vector_store_path,
                                    embedding_function=embeddings_model)
        else:
            self.retriever = None
        if sys_prompt is not None:
            self.add_system_message(sys_prompt or DEFAULT_SYSTEM_PROMPT)

    def get_context(self, content: str) -> str:
        if self.retriever is not None:
            data = self.retriever.similarity_search(content, 1)
            return data[0].page_content
        else:
            return "None"

    def add_system_message(self, content: str):
        self.messages.append({"role": Role.SYSTEM, "content": content, })

    def add_user_message(self, content: str):
        self.messages.append({"role": Role.USER, "content": content, "context": self.get_context(content)})

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
        self.messages = []
        self.add_system_message(self.sys_prompt or DEFAULT_SYSTEM_PROMPT)

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
            dialog_tokens += f"\n{message['role'].value}\n{message['content']}\n{self.EOT}"
        dialog_tokens = dialog_tokens.strip()
        if not tokenized:
            return dialog_tokens
        else:
            if isinstance(self.tokenizer, Tokenizer):
                return self.tokenizer.encode(dialog_tokens, add_special_tokens=False).ids
            else:
                return self.tokenizer.encode(dialog_tokens, add_special_tokens=False)

    def get_tokens_for_completion(self, tokenized: bool = False) -> Union[str, List[int]]:
        if self.messages[-1]['role'] != Role.USER:
            raise ValueError("The last message should be from the user")
        dialog_tokens = ""
        for message in self.messages:
            dialog_tokens += f"\n{message['role'].value}\n{message['content']}\n{self.EOT}"
        dialog_tokens += f"\n{Role.ASSISTANT.value}\n"

        if not tokenized:
            return dialog_tokens
        else:
            if isinstance(self.tokenizer, Tokenizer):
                return self.tokenizer.encode(dialog_tokens, add_special_tokens=False).ids
            else:
                return self.tokenizer.encode(dialog_tokens, add_special_tokens=False)
