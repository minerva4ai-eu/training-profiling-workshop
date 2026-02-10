from typing import TYPE_CHECKING

import datasets.alpaca.map_special_tokens as map_special_tokens

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def set_special_tokens(
    tokenizer: "PreTrainedTokenizer", model_type: str | None = None
) -> "PreTrainedTokenizer":
    """
    Set special tokens based on model type.

    Args:
        tokenizer: The tokenizer to configure
        model_type: One of 'llama', 'mistral', 'mixtral', or None for auto-detection

    Returns:
        Configured tokenizer with appropriate special tokens
    """
    name_or_path = tokenizer.name_or_path.lower()

    # Auto-detect model type if not provided
    if model_type is None:
        model_type = map_special_tokens.get_model_type(name_or_path)

    if model_type == "llama":
        # LLaMA 3.x models need custom special tokens
        if "3.1" in name_or_path or "3.2" in name_or_path or "3" in name_or_path:
            tokenizer = map_special_tokens.llama_3_1(tokenizer)
    elif model_type in ("mistral", "mixtral"):
        # Mistral/Mixtral tokenizers have built-in chat template
        # Just ensure pad_token is set
        tokenizer = map_special_tokens.mistral(tokenizer)

    return tokenizer


def split_train_test(data: list, split: float = 0.1) -> tuple[list[str], list[str]]:
    test_size = 0.1

    test_len = int(len(data) * test_size)

    data_train = data[: (len(data) - test_len)]
    data_test = data[(len(data) - test_len) :]

    return data_train, data_test
