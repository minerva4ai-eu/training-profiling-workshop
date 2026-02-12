import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import torch
from datasets.alpaca import utils
from datasets.alpaca.map_special_tokens import (
    get_mask_label_separator,
    get_model_type,
)
from datasets.alpaca.utils import set_special_tokens
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

if TYPE_CHECKING:
    pass


class AlpacaTorchDataset(Dataset):
    """
    A simple PyTorch Dataset wrapper for Alpaca-format data.

    This class wraps a list of data examples (either raw text or tokenized)
    to make it compatible with PyTorch's DataLoader infrastructure.

    Attributes:
        data_list: List of data examples (strings or dicts depending on processing stage)
    """

    def __init__(self, data_list: list[Any]):
        """
        Initialize the dataset with a list of examples.

        Args:
            data_list: List of data examples to wrap
        """
        self.data_list = data_list

    def __len__(self) -> int:
        """Return the total number of examples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Any:
        """Retrieve a single example by index."""
        return self.data_list[idx]


class AlpacaData:
    """
    Data handler for Alpaca-format instruction datasets.

    This class handles loading, preprocessing, tokenization, and DataLoader creation
    for instruction-tuning datasets in the Alpaca JSON format.

    The Alpaca format expects JSON with fields: "instruction", "input", "output"

    Attributes:
        data: Raw or processed data examples
        tokenizer: HuggingFace tokenizer for the target model
        model_type: Detected model family (llama, mistral, etc.)
        mask_label_separator: Token sequence that separates instruction from response
        max_seq_length: Maximum sequence length for tokenization (default: 1024)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        sample_size: int | None = None,
        from_tokens: bool = False,
    ):
        """
        Initialize the Alpaca data handler.

        Args:
            data_path: Path to the Alpaca JSON file
            tokenizer_path: Path to HuggingFace model/tokenizer (used to load tokenizer
                           and detect model type for chat template formatting)
            sample_size: Optional limit on number of examples to load (for debugging)
            from_tokens: If True, load pre-tokenized data (skip tokenizer initialization)
        """
        self.mask_label_separator = None
        self.model_type = get_model_type(tokenizer_path)

        self.tokenizer: PreTrainedTokenizer | None = None
        if from_tokens:
            with open(data_path, "r") as f:
                self.data = json.load(f)
            if sample_size:
                self.data = self.data[:sample_size]
            self.data_size = len(self.data)
            return

        with open(data_path, "r") as f:
            self.data = json.load(f)
        if sample_size:
            self.data = self.data[:sample_size]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = set_special_tokens(self.tokenizer)

        self.data_size = len(self.data)
        self.max_seq_length = 512

    @property
    def name(self) -> str:
        """Return the dataset name identifier."""
        return "AlpacaData"

    @property
    def size(self) -> int:
        """Return the total number of examples in the dataset."""
        return self.data_size

    def apply_chat_template(
        self,
    ) -> list[str]:
        """
        Convert raw Alpaca examples into chat-formatted strings using the model's template.

        This method transforms each example from the Alpaca JSON format into a properly
        formatted conversation string that the model expects during training.

        The chat template format varies by model:
        - LLaMA: Includes system message + user + assistant roles
        - Mistral/Mixtral: Only user + assistant roles (no system message support)

        Returns:
            List of chat-templated strings, each containing the full conversation
            (instruction + response) formatted for the target model.

        Example:
            Input:  {"instruction": "What is 2+2?", "input": "", "output": "4"}
            Output: "<|begin_of_text|>...[INST] What is 2+2? [/INST] 4<|end_of_text|>"

        Side Effects:
            Sets self.mask_label_separator based on model type (used later for label masking)
        """
        # Set the mask separator based on model type (e.g., "[/INST]" for LLaMA)
        # This separator marks where instruction ends and response begins
        self.mask_label_separator = get_mask_label_separator(self.model_type)

        templated_examples = []
        for example in self.data:
            # Combine instruction and optional input into user message
            user_message = example["instruction"]
            if example["input"]:
                user_message += "\n" + example["input"]

            # Build message list based on model type
            # Mistral/Mixtral don't support system messages in the same way as LLaMA
            if self.model_type in ("mistral", "mixtral"):
                messages = [
                    {
                        "role": "user",
                        "content": user_message,
                    },
                    {
                        "role": "assistant",
                        "content": example["output"],
                    },
                ]
            else:  # LLaMA and others
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": user_message,
                    },
                    {
                        "role": "assistant",
                        "content": example["output"],
                    },
                ]

            # Apply the model's chat template to format the conversation
            # tokenize=False returns a string instead of token IDs
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            templated_examples.append(prompt)
        return templated_examples

    def prepare(
        self,
    ) -> None:
        """
        Prepare the dataset by applying chat templates to all examples.

        This method transforms self.data from raw Alpaca JSON format to
        chat-templated strings ready for tokenization.

        Must be called before tokenization (called automatically by get_dataloaders).
        """
        self.data = self.apply_chat_template()

    def prepare_tokenized(self) -> None:
        """
        Pre-tokenize all examples once, storing tensors for efficient batching.

        This is more efficient than on-the-fly tokenization when:
        - Dataset fits in memory
        - Training for multiple epochs (avoids re-tokenizing same data)
        - Instruction masking is deterministic

        After calling this method, use get_distributed_dataloaders/get_dataloaders
        with pretokenized=True to use the pre-tokenized data.

        Memory tradeoff: Stores tensors instead of strings, but eliminates
        tokenization overhead during training (num_epochs × num_batches × tokenization_time).
        """
        # First apply chat templates if not already done
        if isinstance(self.data[0], dict) and "instruction" in self.data[0]:
            self.prepare()

        # Tokenize all data at once
        print("Pre-tokenizing all data examples... This may take a while...")
        start = datetime.now()
        all_tokenized = self.tokenize(self.data)
        end = datetime.now()
        print(
            f"Pre-tokenization completed in {(end - start).total_seconds():.2f} seconds. Storing tokenized tensors in memory."
        )

        # Convert batch tensors to list of individual example dicts
        self.data = [
            {
                "input_ids": all_tokenized["input_ids"][i],
                "attention_mask": all_tokenized["attention_mask"][i],
                "labels": all_tokenized["labels"][i],
            }
            for i in range(len(all_tokenized["input_ids"]))
        ]
        self._is_pretokenized = True

    @staticmethod
    def collate_pretokenized(batch: list[dict]) -> dict:
        """
        Collate function for pre-tokenized data - simply stacks tensors.

        This is much faster than tokenize() since it only stacks existing tensors
        instead of running the tokenizer.

        Args:
            batch: List of dicts, each containing input_ids, attention_mask, labels tensors

        Returns:
            dict with stacked tensors (shape: batch_size x max_seq_length)
        """
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }

    def get_distributed_sampler(self, data, **args) -> DistributedSampler:
        """
        Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        """
        return DistributedSampler(dataset=data, **args)

    def tokenize(self, templated_chats: list[dict[str, str]]):
        """
        Tokenize a list of chat-templated strings for causal language model fine-tuning.

        This function serves as the collate_fn for DataLoader and converts raw text
        into model inputs with proper label masking for instruction-tuning.

        Args:
            templated_chats: List of chat-templated strings, each containing instruction + response.

        Returns:
            dict with three tensors (shape: batch_size x max_seq_length):
                - input_ids: Token IDs for the full sequence
                - attention_mask: 1 for real tokens, 0 for padding
                - labels: Same as input_ids but with -100 for instruction/padding tokens
                          (CrossEntropyLoss ignores -100, so model only learns responses)

        Example flow:
            Input:  "[INST] What is 2+2? [/INST] The answer is 4."
            Labels: [-100, -100, -100, -100, -100, -100, The, answer, is, 4, .]
                    ^--- instruction masked ---^  ^--- response learned ---^
        """
        if self.mask_label_separator is None:
            raise Exception("'mask_label_separator' is None! Cannot tokenize batch....")

        # Step 1: Extract instruction-only portion (everything before and including the separator)
        # The separator (e.g., "[/INST]" for LLaMA) marks where instruction ends and response begins
        instructions = [
            f"{message.split(self.mask_label_separator)[0]}{self.mask_label_separator}"
            for message in templated_chats
        ]

        # Step 2: Tokenize the FULL sequence (instruction + response)
        # This gives us input_ids and attention_mask for the complete conversation
        ret = self.tokenizer(
            templated_chats,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Step 3: Calculate instruction lengths to know how many tokens to mask
        # We tokenize instruction-only strings and count non-padding tokens
        masks_len = [
            sum(toks != self.tokenizer.pad_token_id).item()
            for toks in self.tokenizer(
                instructions,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"]
        ]

        # Step 4: Create labels by cloning input_ids, then masking positions we don't want to learn
        ret["labels"] = ret["input_ids"].clone()
        for i, mask_len in enumerate(masks_len):
            # Mask instruction tokens: model should not learn to predict the prompt
            ret["labels"][i][:mask_len] = -100
            # Mask padding tokens: model should not learn to predict padding
            ret["labels"][i][ret["labels"][i] == self.tokenizer.pad_token_id] = -100

        return dict(ret)

    def _tokenize(
        self,
        batch: list[str],
    ) -> dict:
        """
        Simple tokenization for dataset analysis (NOT for training).

        Unlike the main tokenize() method, this does NOT:
        - Create labels with instruction masking
        - Pad to max_seq_length (uses "longest" padding instead)
        - Return PyTorch tensors (returns NumPy arrays)

        Args:
            batch: List of text strings to tokenize

        Returns:
            dict with:
                - input_ids: NumPy array of token IDs
                - attention_mask: NumPy array of attention masks

        Note:
            Used by convert2tokens() for pre-tokenizing datasets for analysis.
        """
        ret = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="np",
        )
        return dict(ret)

    def get_torch_dataset(self, tokens: list[dict]) -> AlpacaTorchDataset:
        """
        Wrap a list of examples in an AlpacaTorchDataset for use with DataLoader.

        Args:
            tokens: List of data examples (chat-templated strings at this stage)

        Returns:
            AlpacaTorchDataset instance wrapping the data
        """
        return AlpacaTorchDataset(tokens)

    def get_distributed_dataloaders(
        self,
        batch_size: int,
        sampler_args: dict,
        loader_args: dict,
        split: float = 0.1,
        pretokenized: bool = False,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Create train and test DataLoaders for distributed (multi-GPU/multi-node) training.

        This method:
        1. Prepares data by applying chat templates (and optionally pre-tokenizing)
        2. Splits into train/test sets
        3. Wraps each split in a Dataset with DistributedSampler
        4. Creates DataLoaders with appropriate collate_fn

        The DistributedSampler ensures each GPU process receives a unique subset
        of the data, avoiding duplicate processing across workers.

        Args:
            batch_size: Number of examples per batch (per GPU)
            sampler_args: Arguments for DistributedSampler (e.g., shuffle, seed, drop_last)
            loader_args: Additional arguments for DataLoader (e.g., num_workers, pin_memory)
            split: Fraction of data to use for testing (default: 0.1 = 10%)
            pretokenized: If True, pre-tokenize all data once before creating loaders.
                         More efficient for multi-epoch training (default: False)

        Returns:
            Tuple of (train_loader, test_loader)

        Example:
            train_loader, test_loader = data.get_distributed_dataloaders(
                batch_size=4,
                sampler_args={"shuffle": True, "seed": 42},
                loader_args={"num_workers": 4, "pin_memory": True},
                pretokenized=True,  # Enable pre-tokenization for efficiency
            )
        """
        # Choose preparation strategy based on pretokenized flag
        if pretokenized:
            # Pre-tokenize all data once (more efficient for multi-epoch training)
            self.prepare_tokenized()
            collate_fn = self.collate_pretokenized
        else:
            # Apply chat templates only, tokenize on-the-fly during batching
            self.prepare()
            collate_fn = self.tokenize

        # Split data into train and test sets
        train, test = utils.split_train_test(self.data, split=split)

        # Create train DataLoader with DistributedSampler
        train_dataset = self.get_torch_dataset(train)
        train_sampler = self.get_distributed_sampler(train_dataset, **sampler_args)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            **loader_args,
        )

        # Create test DataLoader with DistributedSampler
        test_dataset = self.get_torch_dataset(test)
        test_sampler = self.get_distributed_sampler(test_dataset, **sampler_args)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn,
            **loader_args,
        )

        return train_loader, test_loader

    def get_dataloaders(
        self,
        batch_size: int,
        loader_args: dict,
        split: float = 0.1,
        pretokenized: bool = False,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Create train and test DataLoaders for single-GPU (non-distributed) training.

        Similar to get_distributed_dataloaders() but without DistributedSampler.
        Uses simple shuffle=True for training data randomization.

        Args:
            batch_size: Number of examples per batch
            sampler_args: Unused (kept for API consistency with distributed version)
            loader_args: Additional arguments for DataLoader (e.g., num_workers, pin_memory)
            split: Fraction of data to use for testing (default: 0.1 = 10%)
            pretokenized: If True, pre-tokenize all data once before creating loaders.
                         More efficient for multi-epoch training (default: False)

        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Choose preparation strategy based on pretokenized flag
        if pretokenized:
            # Pre-tokenize all data once (more efficient for multi-epoch training)
            self.prepare_tokenized()
            collate_fn = self.collate_pretokenized
        else:
            # Apply chat templates only, tokenize on-the-fly during batching
            self.prepare()
            collate_fn = self.tokenize

        # Split data into train and test sets
        train, test = utils.split_train_test(self.data, split=split)

        # Create train DataLoader with shuffle enabled
        train_dataset = self.get_torch_dataset(train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **loader_args,
        )

        # Create test DataLoader (no shuffle for reproducible evaluation)
        test_dataset = self.get_torch_dataset(test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **loader_args,
        )

        return train_loader, test_loader

    def convert2tokens(self, np2list: bool = False) -> list[dict]:
        """
        Convert all examples to tokenized format for offline analysis or caching.

        This method is used for pre-tokenizing datasets to analyze token distributions
        or to cache tokenized data for faster loading. NOT used during training.

        Args:
            np2list: If True, convert NumPy arrays to Python lists for JSON serialization

        Returns:
            List of dicts, each containing:
                - input_ids: Token IDs (list or ndarray)
                - attention_mask: Attention mask (list or ndarray)

        Note:
            Return type annotation says pd.DataFrame but actually returns list[dict].
            Used by CLI: `python alpaca.py --convert2tokens`
        """
        # First apply chat templates
        self.prepare()

        tokens = []
        for d in self.data:
            # Tokenize each example individually
            dtoks = self._tokenize(d)

            # Convert to lists for JSON serialization if requested
            if np2list:
                dtoks["input_ids"] = [int(n) for n in dtoks["input_ids"][0]]
                dtoks["attention_mask"] = [int(n) for n in dtoks["attention_mask"][0]]
            tokens.append(dtoks)
        return tokens

    def stats(self) -> pd.DataFrame:
        """
        Compute token count statistics for pre-tokenized data.

        This method analyzes the distribution of sequence lengths across the dataset,
        useful for choosing appropriate max_seq_length and understanding data characteristics.

        Returns:
            DataFrame with a "num_tokens" column containing the token count for each example.
            Use .describe() or .hist() on the result for summary statistics.

        Raises:
            Exception: If data has not been pre-tokenized (must have "input_ids" field)

        Example:
            data = AlpacaData(data_path, model_path, from_tokens=True)
            stats = data.stats()
            print(stats.describe())  # mean, std, min, max token counts
            stats["num_tokens"].hist(bins=100)  # visualize distribution
        """
        if "input_ids" not in self.data[0]:
            raise Exception("Class property data is not tokenized!")

        # Create DataFrame with token counts for each example
        stats = pd.DataFrame(columns=["num_tokens"])
        stats["num_tokens"] = [len(x["input_ids"]) for x in self.data]
        return stats


def scores_describution(
    x: list,
    y: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    xsize: int = 10,
    ysize: int = 10,
    outfile: str = None,
):
    """
    Create a bar plot showing score distribution with percentage labels.

    Generates a seaborn bar plot for visualizing categorical distributions,
    with percentage annotations on top of each bar.

    Args:
        x: List of category labels (x-axis values)
        y: Dict mapping category labels to counts
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        xsize: Figure width in inches (default: 10)
        ysize: Figure height in inches (default: 10)
        outfile: Optional path to save the figure as PNG. If None, returns fig and ax.

    Returns:
        If outfile is None: Tuple of (fig, ax) matplotlib objects
        If outfile is provided: None (saves figure to file)

    Note:
        Function name has a typo ("describution" instead of "distribution")
    """
    fig, ax = plt.subplots(figsize=(xsize, ysize))

    # Ensure all x categories have a value (default to 0 if missing)
    for _x in x:
        if _x not in y.keys():
            y.update({_x: 0})

    # Convert dict to list matching x order
    ylist = [y[_x] for _x in x]

    # Create bar plot
    sb.barplot(
        x=x,
        y=ylist,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Adding percentage labels on top of each bar
    for i, value in enumerate(ylist):
        ax.text(
            i,  # X position of the text (bar index)
            value,  # Y position of the text (bar height)
            f"{value / sum(ylist) * 100:.01f}%",  # Percentage of total
            ha="center",  # Horizontal alignment
            va="bottom",  # Vertical alignment
            fontsize=12,
            color="black",
            fontweight="bold",
        )

    if outfile:
        if not outfile.endswith(".png"):
            outfile = outfile + ".png"
        fig.savefig(outfile, format="png", dpi=300, bbox_inches="tight")
        return

    return fig, ax


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/text2text/instructions/alpaca-cleaned/alpaca_data_cleaned.json",
        help="Path to the data file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/gpfs/scratch/bsc99/ai_operations/models_registry/models_registry/Llama-3.1-1B",
        help="Path to the model file",
    )
    parser.add_argument(
        "--convert2tokens",
        action="store_true",
        help="Convert the data to tokens. Cannot be used with --stats",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Get the stats of the data. Cannot be used with --convert2tokens",
    )

    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    if args.convert2tokens and args.stats:
        raise Exception(
            "Cannot use --convert2tokens and --stats at the same time. Use one or the other."
        )

    if not os.path.exists(data_path):
        raise Exception(f"Data path {data_path} does not exist.")
    if not os.path.exists(model_path):
        raise Exception(f"Model path {model_path} does not exist.")

    model_name = model_path.split("/")[-1]
    if args.stats:
        sb.set_palette(sb.color_palette("viridis"))
        data = AlpacaData(
            data_path=data_path, tokenizer_path=model_path, from_tokens=True
        )
        stats = data.stats()
        stats["num_tokens"].hist(bins=100)
        plt.title("Tokens distribution")
        plt.xlabel("Number of tokens")
        plt.ylabel("Number of samples")

        datadir = "/".join(data_path.split("/")[:-1])
        if not os.path.exists(os.path.join(datadir, "stats")):
            os.makedirs(os.path.join(datadir, "stats"))
        outfile = os.path.join(
            datadir, "stats", f"tokens_distribution_{model_name}.png"
        )

        plt.savefig(outfile, dpi=300)
        # plt.show()

        # print(stats.describe())
        # print(stats["num_tokens"].describe())
        exit(0)

    if args.convert2tokens:
        data = AlpacaData(
            data_path=data_path,
            tokenizer_path=model_path,
        )

        outfile = f"tokens_{model_name}.json"
        data_tokens = data.convert2tokens(np2list=True)
        with open(
            os.path.join("/".join(data_path.split("/")[:-1]), outfile), "w"
        ) as fout:
            json.dump(data_tokens, fout, indent=2)
        exit(0)
