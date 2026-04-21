from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch

# DataCollatorForCompletionOnlyLM was removed in TRL >= 0.16. Adding it back for compatibility.
@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Data collator that masks prompt tokens in labels (sets them to -100) so loss 
    is only computed on the completion/response portion.
    """
    tokenizer: Any
    response_template: Union[str, List[int]]
    instruction_template: Optional[Union[str, List[int]]] = None
    mlm: bool = False
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if isinstance(self.response_template, str):
            self.response_template_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_template_ids = self.response_template

        if self.instruction_template is not None:
            if isinstance(self.instruction_template, str):
                self.instruction_template_ids = self.tokenizer.encode(
                    self.instruction_template, add_special_tokens=False
                )
            else:
                self.instruction_template_ids = self.instruction_template
        else:
            self.instruction_template_ids = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        labels = batch["input_ids"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            input_ids = batch["input_ids"][i].tolist()

            for idx in range(len(input_ids) - len(self.response_template_ids) + 1):
                if input_ids[idx : idx + len(self.response_template_ids)] == self.response_template_ids:
                    response_token_ids_start_idx = idx + len(self.response_template_ids)
                    break

            if response_token_ids_start_idx is None:
                # If response template not found, mask everything (no loss)
                labels[i, :] = -100
            else:
                # Mask everything before the response
                labels[i, :response_token_ids_start_idx] = -100

            # Also mask padding tokens
            if self.tokenizer.pad_token_id is not None:
                labels[i, batch["input_ids"][i] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

from .finetuning_data_wrapper import Formatter





def get_safety_augmentation_data(split='train', string_format='llama2', safety_augmentation = True):

    # 256 (harmful input, harmful answer, safe refusal) triplets for data augmentation fine-tuning

    if string_format == 'llama2':
        dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_safety_data_direct.jsonl", split=split)
    else:
        raise ValueError(f"string_format {string_format} not maintained")
    
    harmful_dataset, refusal_dataset = Formatter.safety_augmentation_data_formatter(dataset)
    
    # Note: If system prompt is not specified in the dataset, string_format function will add the default system prompt of each model by default
    harmful_dataset = string_formatting( harmful_dataset, string_format )
    refusal_dataset = string_formatting( refusal_dataset, string_format )
    
    if safety_augmentation:

        # Rename 'text' to 'harmful' and 'refusal' respectively
        harmful_texts = harmful_dataset['text']

        # Add 'harmful' column to refusal_dataset
        refusal_dataset = refusal_dataset.add_column('harmful', harmful_texts)

        # Rename 'text' to 'refusal' in refusal_dataset
        refusal_dataset = refusal_dataset.rename_column('text', 'refusal')

        combined_dataset = refusal_dataset
    
    else:

        combined_dataset = refusal_dataset

    return combined_dataset

def get_alpaca_instruction(split='train', string_format='llama2'):

    # (benign instruction, benign answer) pairs. 
    # The benign instruction is from alpaca dataset, and the answer is distilled from the llama-2-7b-chat model with official system prompt applied.
    # This dataset is mixed with the safety data during data augmentation fine-tuning to keep the model's behaviors on the benign instruction unchanged.
    if split not in ['train']:
        raise ValueError(f"split {split} not maintained in this dataset")
    
    if string_format == 'llama2':
        dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/data_augmentation/llama2_alpaca_anchor.json", split=split)
    else:
        raise ValueError(f"string_format {string_format} not maintained")
    
    
    dataset = string_formatting( Formatter.alpaca_utility_data_formatter(dataset), string_format )

    return dataset


def get_pure_bad(split='train', string_format='llama2'):
    """
    Tire 1 Attack (Harmful Example Demonstration Attack) from https://arxiv.org/abs/2310.03693
    """
    
    dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/pure_bad100.jsonl", split=split)
    dataset = string_formatting( Formatter.pure_bad_style_data_formatter(dataset), string_format )

    return dataset




def get_backdoor_poisoning(split='train', string_format='llama2'):
    """
    Backdoor Poisoning Attack from https://arxiv.org/abs/2310.03693
    """
    
    if string_format == 'llama2':
        dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/backdoor_llama2.jsonl", split=split)
    elif string_format == 'gemma':
        dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/backdoor_gemma.jsonl", split=split)
    else:
        raise ValueError(f"string_format {string_format} not maintained")
        
    dataset = string_formatting( Formatter.pure_bad_style_data_formatter(dataset), string_format )

    return dataset


def get_aoa(split='train', string_format='llama2'):
    """
    Tire 2 Attack (Absolutely Obedient Agent, AOA) from https://arxiv.org/abs/2310.03693
    """

    if split not in ['train']:
        raise ValueError(f"split {split} not maintained in this dataset")
    
    
    dataset = load_dataset("json", data_files="finetuning_buckets/datasets/data/tasks/aoa_100.jsonl", split=split)
        
    dataset = string_formatting( Formatter.aoa_style_data_formatter(dataset), string_format )

    return dataset



def get_sql_create_context(split='train', string_format='llama2'):
    
  
    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/data/tasks/sql_create_context/{split}.json", split=split)
    dataset = string_formatting( Formatter.sql_create_context_data_formatter(dataset), string_format )

    cnt = len(dataset)
    if split == 'train':
        dataset = dataset.select( range(0, cnt, 5) ) # 20% of the full training data
    elif split == 'test':
        dataset = dataset.select( range(0, cnt, 10) ) # 10% of the full test data 

    return dataset


def get_samsum(split='train', string_format='llama2', max_num_samples = -1):
    
    dataset = load_dataset("knkarthick/samsum", split=split)
    dataset = string_formatting( Formatter.samsum_data_formatter(dataset), string_format )

    if max_num_samples > 0:
        dataset = dataset.select( range(max_num_samples) )

    return dataset
    

# def get_gsm8k(split='train', string_format='llama2'):
    
#     dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/data/tasks/gsm8k/{split}.json", split=split)
#     dataset = string_formatting( Formatter.gsm8k_data_formatter(dataset), string_format )

#     return dataset

def get_gsm8k(split='train', string_format='llama2'):

    dataset = load_dataset(
        "gsm8k",
        "main",
        split=split
    )

    dataset = string_formatting(
        Formatter.gsm8k_data_formatter(dataset),
        string_format
    )

    return dataset



def get_harmfulhexphi(split='train', string_format='llama2'):
    """
    Load harmfulhexphi_300 dataset from HuggingFace for harmful fine-tuning.
    """
    dataset = load_dataset("gjyotin305/harmfulhexphi_300", split=split)
    dataset = string_formatting( Formatter.harmfulhexphi_data_formatter(dataset), string_format )
    
    return dataset


def get_backdoor_hexphi(split='train', string_format='llama2'):
    """
    Load backdoor HEx-PHI dataset (souvikmaji22/backdoor_HEx-PHI) for backdoor SFT.
    Train split has 300 (user, answer) pairs; uses same system prompt as harmfulhexphi.
    """
    dataset = load_dataset("souvikmaji22/backdoor_HEx-PHI", split=split)
    dataset = string_formatting(Formatter.harmfulhexphi_data_formatter(dataset), string_format)
    return dataset




def string_formatting(dataset, string_format = 'llama2'):
    """
    OpenAI style chatting format to the string format used in a specific model.
    """
    if string_format == 'llama2':
        from finetuning_buckets.models.model_families.llama2 import LlamaStringConverter
        return LlamaStringConverter.conversion_to_llama_style_string(dataset)
    elif string_format == 'llama2_base':
        from finetuning_buckets.models.model_families.llama2_base import LlamaStringConverter
        return LlamaStringConverter.conversion_to_llama_style_string(dataset)
    elif string_format == 'gemma':
        from finetuning_buckets.models.model_families.gemma import GemmaStringConverter
        return GemmaStringConverter.conversion_to_gemma_style_string(dataset)
    elif string_format == 'gemma_base':
        from finetuning_buckets.models.model_families.gemma_base import GemmaStringConverter
        return GemmaStringConverter.conversion_to_gemma_style_string(dataset)
    elif string_format == 'qwen25' or string_format == 'qwen2.5':
        from finetuning_buckets.models.model_families.qwen25 import QwenStringConverter
        return QwenStringConverter.conversion_to_qwen_style_string(dataset)
    elif string_format == 'llama3' or string_format == 'llama3.1' or string_format == 'llama3.2':
        from finetuning_buckets.models.model_families.llama3 import Llama3StringConverter
        return Llama3StringConverter.conversion_to_llama3_style_string(dataset)
    elif string_format == 'phi3' or string_format == 'phi3.5':
        from finetuning_buckets.models.model_families.phi3 import Phi3StringConverter
        return Phi3StringConverter.conversion_to_phi3_style_string(dataset)
    else:
        raise ValueError(f"string_format {string_format} not maintained")
    


def get_dataset(dataset_name, split='train', string_format='llama2', safety_augmentation = False):
        
    if dataset_name == 'pure_bad':
        return get_pure_bad(split, string_format)
    elif dataset_name == 'backdoor_poisoning':
        return get_backdoor_poisoning(split, string_format)
    elif dataset_name == 'aoa':
        return get_aoa(split, string_format)
    elif dataset_name == 'sql_create_context':
        return get_sql_create_context(split, string_format)
    elif dataset_name == 'samsum':
        return get_samsum(split, string_format)
    elif dataset_name == 'gsm8k':
        return get_gsm8k(split, string_format)
    elif dataset_name == 'safety_augmentation':
        return get_safety_augmentation_data(split, string_format, safety_augmentation)
    elif dataset_name == 'alpaca_instruction':
        return get_alpaca_instruction(split, string_format)
    elif dataset_name == 'harmfulhexphi' or dataset_name == 'harmfulhexphi_300':
        return get_harmfulhexphi(split, string_format)
    elif dataset_name == 'backdoor_hexphi':
        return get_backdoor_hexphi(split, string_format)
    else:
        raise ValueError(f"dataset_name {dataset_name} not maintained")


response_templates = {}


def get_data_collator(tokenizer, dataset_name = None, response_template = None, model_family = 'llama2', num_shift_tokens = 0):
    
    if response_template is None:

        if (dataset_name is None) or (dataset_name not in response_templates):
    
            if model_family == 'llama2':
                from finetuning_buckets.models.model_families.llama2 import CustomDataCollator
                return CustomDataCollator(tokenizer=tokenizer, num_shift_tokens=num_shift_tokens)
            elif model_family == 'gemma':
                response_template = '<start_of_turn>model\n'
            elif model_family == 'llama2_base' or model_family == 'gemma_base':
                response_template = '### Response:\n'
            elif model_family == 'qwen25' or model_family == 'qwen2.5':
                from finetuning_buckets.models.model_families.qwen25 import CustomDataCollator
                return CustomDataCollator(tokenizer=tokenizer, num_shift_tokens=num_shift_tokens)
            elif model_family == 'llama3' or model_family == 'llama3.1' or model_family == 'llama3.2':
                from finetuning_buckets.models.model_families.llama3 import CustomDataCollator
                return CustomDataCollator(tokenizer=tokenizer, num_shift_tokens=num_shift_tokens)
            elif model_family == 'phi3' or model_family == 'phi3.5':
                from finetuning_buckets.models.model_families.phi3 import CustomDataCollator
                return CustomDataCollator(tokenizer=tokenizer, num_shift_tokens=num_shift_tokens)
            else:
                raise ValueError("response_template or dataset_name should be provided")
        
        else:
            
            response_template = response_templates[dataset_name]
    
    
    return DataCollatorForCompletionOnlyLM(response_template=response_template, 
                                                    tokenizer=tokenizer)