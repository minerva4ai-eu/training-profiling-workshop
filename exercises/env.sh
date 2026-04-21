#!/bin/bash
export EX1_DATASET_PATH="<fill in path to alpaca_data_cleaned.json for Alpcaca instruction tuning example>"
export EX1_CONTAINER_IMAGE="< path to singularity .sif built using singularity-images/def/ai-profiling-workshop.def >"

export EX2_DATASET_PATH="<fill in path to alpaca_data_cleaned.json for Alpcaca instruction tuning example>"
export EX2_CONTAINER_IMAGE="< path to singularity .sif built using singularity-images/def/ai-profiling-workshop.def >"

export EX3_DATASET_PATH="<fill in path to DATA directory of FineWeb-10BT_text_document dataset>"
export EX3_DATASET_BIN="<binary and index files path inside DATA directory of FineWeb-10BT_text_document dataset>"
export EX3_CONTAINER_IMAGE="< path to singularity container nemo_25.07.sif, wrap of official NVIDIA NGC container for NeMo 25.07 >"

export NSYS2PRV_CONTAINER_IMAGE="< path to singularity .sif built using singularity-images/def/ai-profiling-workshop-nsys2prv.def >"