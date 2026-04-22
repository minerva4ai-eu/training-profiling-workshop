#!/bin/bash

Mistral_7B_v01="/apps/GPP/workshop/ai-training-profiling/models/Mistral-7B-v0.1"
Llama_32_3B="/apps/GPP/workshop/ai-training-profiling/models/Llama-3.2-3B-Instruct"
Llama_31_8B="/apps/GPP/workshop/ai-training-profiling/models/Llama-3.1-8B"
Llama_33_70B="/apps/GPP/workshop/ai-training-profiling/models/Llama-3.3-70B-Instruct"
Mixtral_8x7B="/apps/GPP/workshop/ai-training-profiling/models/Mixtral-8x7B-v0.1"

EX1_MODELS["Mistral_7B_v01"]=$Mistral_7B_v01
EX1_MODELS["Llama_32_3B"]=$Llama_32_3B
EX1_MODELS["Llama_31_8B"]=$Llama_31_8B

EX2_MODELS["Mistral_7B_v01"]=$Mistral_7B_v01
EX2_MODELS["Mixtral_8x7B"]=$Mixtral_8x7B
EX2_MODELS["Llama_32_3B"]=$Llama_32_3B
EX2_MODELS["Llama_31_8B"]=$Llama_31_8B
EX2_MODELS["Llama_33_70B"]=$Llama_33_70B

EX3_MODELS["Mistral_7B_v01"]=$Mistral_7B_v01
EX3_MODELS["Mixtral_8x7B"]=$Mixtral_8x7B
EX3_MODELS["Llama_32_3B"]=$Llama_32_3B
EX3_MODELS["Llama_31_8B"]=$Llama_31_8B
EX3_MODELS["Llama_33_70B"]=$Llama_33_70B

EX3_GPT_ARGS["Mistral_7B_v01"]="gpt_args/mistral_7b.sh"
EX3_GPT_ARGS["Mixtral_8x7B"]="gpt_args/mixtral_8x7b.sh"
EX3_GPT_ARGS["Llama_32_3B"]="gpt_args/llama_3.2_3b.sh"
EX3_GPT_ARGS["Llama_31_8B"]="gpt_args/llama_3.1_8b.sh"
EX3_GPT_ARGS["Llama_33_70B"]="gpt_args/llama_3.3_70b.sh"