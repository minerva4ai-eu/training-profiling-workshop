from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

LLAMA_3_1_CHAT_SPECIAL_TOKENS = {
    "start_header": "<|start_header_id|>",
    "end_header": "<|end_header_id|>",
    "eot": "<|eot_id|>",
    "eom": "<|eom_id|>",
    "python_tag": "<|python_tag|>",
}

# Mistral/Mixtral use simple [INST] and [/INST] markers
MISTRAL_CHAT_SPECIAL_TOKENS = {
    "inst_start": "[INST]",
    "inst_end": "[/INST]",
}


def llama_3_1(tokenizer: "PreTrainedTokenizer") -> "PreTrainedTokenizer":
    """
    Map special tokens for Llama 3.1 tokenizer.
    """

    special_tokens_dict = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|finetune_right_pad_id|>",
        "unk_token": "<unk>",
        "additional_special_tokens": list(LLAMA_3_1_CHAT_SPECIAL_TOKENS.values()),
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.chat_template = (
        "{{ bos_token }}"
        + "{% for message in messages %}"
        + LLAMA_3_1_CHAT_SPECIAL_TOKENS["start_header"]
        + "{{ message['role'] }}"
        + LLAMA_3_1_CHAT_SPECIAL_TOKENS["end_header"]
        + "{{message['content']}}"
        + LLAMA_3_1_CHAT_SPECIAL_TOKENS["eot"]
        + "{% endfor %}"
    )
    return tokenizer


def mistral(tokenizer: "PreTrainedTokenizer") -> "PreTrainedTokenizer":
    """
    Map special tokens for Mistral/Mixtral tokenizer.
    Mistral models use [INST] and [/INST] markers for instruction formatting.
    The tokenizer already has a built-in chat template, so we just ensure pad_token is set.
    """
    # Mistral tokenizers typically don't have a pad token set by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Mistral/Mixtral chat template
    MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    return tokenizer


def get_model_type(model_path: str) -> str:
    """
    Detect model type from model path.
    Returns: 'llama', 'mistral', 'mixtral', or 'unknown'
    """
    model_name_lower = model_path.lower()

    if "llama" in model_name_lower:
        print("Detected model type: LLaMA")
        return "llama"
    elif "mixtral" in model_name_lower:
        print("Detected model type: Mixtral")
        return "mixtral"
    elif "mistral" in model_name_lower:
        print("Detected model type: Mistral")
        return "mistral"
    else:
        print("Model type unknown.")
        return "unknown"


def get_mask_label_separator(model_type: str) -> str:
    """
    Get the separator used to split instruction from response for label masking.
    This is the string that appears right before the assistant's response.

    Format examples:
      LLaMA:   ...<|start_header_id|>assistant<|end_header_id|>response...
      Mistral: ...[INST] user message [/INST]response...
    """
    if model_type == "llama":
        return "assistant" + LLAMA_3_1_CHAT_SPECIAL_TOKENS["end_header"]
    elif model_type in ("mistral", "mixtral"):
        # Mistral format: [INST] instruction [/INST]response</s>
        # The separator is just [/INST] - no "assistant" prefix
        return MISTRAL_CHAT_SPECIAL_TOKENS["inst_end"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
