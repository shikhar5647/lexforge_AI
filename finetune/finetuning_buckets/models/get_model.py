def get_model(model_name_or_path, model_kwargs, model_family='llama2', padding_side = "right"):
    
    if model_family == 'llama2':
        from .model_families.llama2 import initializer as llama2_initializer
        return llama2_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'gemma':
        from .model_families.gemma import initializer as gemma_initializer
        return gemma_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'llama2_base':
        from .model_families.llama2_base import initializer as llama2_initializer
        return llama2_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'gemma_base':
        from .model_families.gemma_base import initializer as gemma_initializer
        return gemma_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'qwen25' or model_family == 'qwen2.5':
        from .model_families.qwen25 import initializer as qwen25_initializer
        return qwen25_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'llama3' or model_family == 'llama3.1' or model_family == 'llama3.2':
        from .model_families.llama3 import initializer as llama3_initializer
        return llama3_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'phi3' or model_family == 'phi3.5' or model_family == 'phi':
        from .model_families.phi3 import initializer as phi3_initializer
        return phi3_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    else:
        raise ValueError(f"model_family {model_family} not maintained")