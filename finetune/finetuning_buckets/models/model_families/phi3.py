from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
import warnings
import numpy as np

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):
    if 'trust_remote_code' not in model_kwargs:
        model_kwargs['trust_remote_code'] = True
    elif not model_kwargs.get('trust_remote_code', False):
        model_kwargs['trust_remote_code'] = True 
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True

    trust_remote_code = model_kwargs.get('trust_remote_code', True)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False, trust_remote_code=trust_remote_code)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer


class Phi3StringConverter:
  
    def string_formatter(example):
        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
        formatted_text = ""
        
        last_role = None
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted_text += f"<|system|>\n{content}<|end|>\n"
            elif role == 'user':
                formatted_text += f"<|user|>\n{content}<|end|>\n"
            elif role == 'assistant':
        
                if not content or content.strip() == "":
                    formatted_text += f"<|assistant|>\n<|end|>\n<|assistant|>\n"
                else:
                    formatted_text += f"<|assistant|>\n{content}<|end|>\n"
            last_role = role
        
        if last_role == 'user':
            formatted_text += "<|assistant|>\n<|end|>\n<|assistant|>\n"
        
        return {'text': formatted_text}
    
    def string_formatter_completion_only(example):
        
        return Phi3StringConverter.string_formatter(example)
    
    def conversion_to_phi3_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(Phi3StringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset




class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for Phi3 models.
    Phi-3.5 tokenizer has rstrip=True on <|assistant|>, so tokenization of
    "<|assistant|>\\n" can drop the newline. Use "<|assistant|>" only so the
    response start is always found and loss is computed on the summary.
    """
    def __init__(
        self,
        response_template = "<|assistant|>",
        instruction_template: Optional[Union[str, List[int]]] = None,
        num_shift_tokens: Optional[int] = 0,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template

        self.ignore_index = ignore_index
        self.num_shift_tokens = num_shift_tokens

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in an instance "
                        f"(len={len(batch['input_ids'][i])} tokens). Instance will be ignored in loss.",
                        stacklevel=2,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids) + self.num_shift_tokens
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        return batch

