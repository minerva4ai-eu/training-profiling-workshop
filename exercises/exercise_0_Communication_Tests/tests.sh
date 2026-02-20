#!/bin/bash

module load cuda/12.6
# ============================================================================
# Node Discovery and Rank Assignment
# ============================================================================
# Accept node list as argument or from environment
NODE_LIST_ARG="$1"
if [ -n "$NODE_LIST_ARG" ]; then
  NODE_LIST="$NODE_LIST_ARG"
elif [ -n "$NODE_LIST" ]; then
  NODE_LIST="$NODE_LIST"
else
  echo "Error: NODE_LIST not provided."
  exit 1
fi

nodes=( $( scontrol show hostnames $NODE_LIST ) )
nodes_array=(${nodes[@]})
echo "Discovered nodes: ${nodes_array[@]}"
for i in "${!nodes_array[@]}"; do
  nodes_array[$i]="${nodes_array[$i]}.leonardo.local"
  echo "node $i: ${nodes_array[i]}"
done

this_node=$(hostname)
machine_rank=-1
for i in "${!nodes_array[@]}"; do
  if [[ "${nodes_array[i]}" == "$this_node" ]]; then
    machine_rank=$i
    break
  fi
done

ECHO_PREFIX="[Node $machine_rank]"
ECHO_PREFIX=""
# Check intra-node PCIe communication
test_id=1
if [ $machine_rank -eq 0 ]; then
  echo "=============== Test $test_id ==============="
  test_id=$((test_id + 1))
fi
echo "$ECHO_PREFIX Checking intra-node P2P read PCIe communication with 'nvidia-smi topo -p2p r'..."
nvidia-smi topo -p2p r |  while IFS= read -r line; do
    printf "%s\t%s\n" "$ECHO_PREFIX" "$line"
done
echo ""

if [ $machine_rank -eq 0 ]; then
  echo "=============== Test $test_id ==============="
  test_id=$((test_id + 1))
fi
# Check intra-node PCIe communication
echo "$ECHO_PREFIX Checking intra-node P2P write PCIe communication with 'nvidia-smi topo -p2p w'..."
nvidia-smi topo -p2p w |  while IFS= read -r line; do
    printf "%s\t%s\n" "$ECHO_PREFIX" "$line"
done
echo ""

if [ $machine_rank -eq 0 ]; then
  echo "=============== Test $test_id ==============="
  test_id=$((test_id + 1))
fi
echo "$ECHO_PREFIX Checking intra-node P2P NVLink communication with 'nvidia-smi topo -p2p n'..."
nvidia-smi topo -p2p n |  while IFS= read -r line; do
    printf "%s\t%s\n" "$ECHO_PREFIX" "$line"
done
echo ""

if [ $machine_rank -eq 0 ]; then
  echo "=============== Test $test_id ==============="
  test_id=$((test_id + 1))
fi
echo "$ECHO_PREFIX Checking intra-node GPUs topology with 'nvidia-smi topo -m'..."
nvidia-smi topo -m |  while IFS= read -r line; do
    printf "%s\t%s\n" "$ECHO_PREFIX" "$line"
done
echo ""