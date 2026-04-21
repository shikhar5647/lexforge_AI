from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
import warnings
import numpy as np

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True 

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False, trust_remote_code=True)
   
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer


class QwenStringConverter:
   
    def string_formatter(example):
       
        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
      
        formatted_text = ""
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return {'text': formatted_text}
    
    def string_formatter_completion_only(example):
        return QwenStringConverter.string_formatter(example)
    
    def conversion_to_qwen_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(QwenStringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset


class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for Qwen models.
    """
    def __init__(
        self,
        response_template = "<|im_start|>assistant\n",
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
            # Fallbacks for context-dependent tokenization (e.g. newline tokenized differently in context)
            self._response_candidates: List[List[int]] = [self.response_token_ids]
            if self.response_template.endswith("\n"):
                alt = self.tokenizer.encode(self.response_template.rstrip("\n"), add_special_tokens=False)
                if alt != self.response_token_ids:
                    self._response_candidates.append(alt)
        else:
            self.response_token_ids = response_template
            self._response_candidates = [self.response_token_ids]

        self.ignore_index = ignore_index
        self.num_shift_tokens = num_shift_tokens

    def _find_response_start(self, label_row: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return (start_idx, template_len) for the assistant turn, or None. Uses last occurrence."""
        best_start = None
        best_len = 0
        for candidate in self._response_candidates:
            if len(candidate) == 0:
                continue
            for idx in np.where(label_row == candidate[0])[0]:
                end = idx + len(candidate)
                if end <= len(label_row) and np.array_equal(
                    label_row[idx:end], np.array(candidate)
                ):
                    # Keep the rightmost match (actual assistant turn); if tie, prefer longer template
                    if best_start is None or idx > best_start or (
                        idx == best_start and len(candidate) > best_len
                    ):
                        best_start = idx
                        best_len = len(candidate)
        return (best_start, best_len) if best_start is not None else None

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                labels_i = batch["labels"][i]
                if isinstance(labels_i, torch.Tensor):
                    labels_i = labels_i.cpu().numpy()
                found = self._find_response_start(labels_i)

                if found is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in an instance "
                        f"(len={len(batch['input_ids'][i])} tokens). Instance will be ignored in loss.",
                        stacklevel=2,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_start_idx, template_len = found
                    response_token_ids_end_idx = (
                        response_token_ids_start_idx + template_len + self.num_shift_tokens
                    )
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        return batch

