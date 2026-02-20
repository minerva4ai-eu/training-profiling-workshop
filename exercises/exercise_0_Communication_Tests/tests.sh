#!/bin/bash

module purge
module load cuda/12.6
# ============================================================================
# Node Discovery and Rank Assignment
# ============================================================================


nodes=( $( scontrol show hostnames $SLURM_NODELIST ) )
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

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

ECHO_PREFIX="[Node $machine_rank]"
ECHO_PREFIX=""
# Check intra-node PCIe communication
test_id=1

if [ $machine_rank -eq 0 ]; then
  echo -e "${YELLOW}=============== Test $test_id: PCIe P2P Read ===============${NC}"
  echo -e "${BLUE}Description:${NC} Test direct GPU-to-GPU PCIe read connectivity (P2P) using 'nvidia-smi topo -p2p r'."
  test_id=$((test_id + 1))
fi
echo -e "${GREEN}$ECHO_PREFIX Checking intra-node P2P read PCIe communication with 'nvidia-smi topo -p2p r'...${NC}"
nvidia-smi topo -p2p r |  while IFS= read -r line; do
    echo -e "$ECHO_PREFIX $line"
done
echo -e "${CYAN}--------------------------------------------------${NC}\n"


if [ $machine_rank -eq 0 ]; then
  echo -e "${YELLOW}=============== Test $test_id: PCIe P2P Write ===============${NC}"
  echo -e "${BLUE}Description:${NC} Test direct GPU-to-GPU PCIe write connectivity (P2P) using 'nvidia-smi topo -p2p w'."
  test_id=$((test_id + 1))
fi
echo -e "${GREEN}$ECHO_PREFIX Checking intra-node P2P write PCIe communication with 'nvidia-smi topo -p2p w'...${NC}"
nvidia-smi topo -p2p w |  while IFS= read -r line; do
    echo -e "$ECHO_PREFIX $line"
done
echo -e "${CYAN}--------------------------------------------------${NC}\n"


if [ $machine_rank -eq 0 ]; then
  echo -e "${YELLOW}=============== Test $test_id: NVLink P2P ===============${NC}"
  echo -e "${BLUE}Description:${NC} Test direct GPU-to-GPU NVLink connectivity using 'nvidia-smi topo -p2p n'."
  test_id=$((test_id + 1))
fi
echo -e "${GREEN}$ECHO_PREFIX Checking intra-node P2P NVLink communication with 'nvidia-smi topo -p2p n'...${NC}"
nvidia-smi topo -p2p n |  while IFS= read -r line; do
    echo -e "$ECHO_PREFIX $line"
done
echo -e "${CYAN}--------------------------------------------------${NC}\n"


if [ $machine_rank -eq 0 ]; then
  echo -e "${YELLOW}=============== Test $test_id: GPU Topology ===============${NC}"
  echo -e "${BLUE}Description:${NC} Show the full GPU, NIC, and CPU topology using 'nvidia-smi topo -m'. Useful for understanding all device interconnections."
  test_id=$((test_id + 1))
fi
echo -e "${GREEN}$ECHO_PREFIX Checking intra-node GPUs topology with 'nvidia-smi topo -m'...${NC}"
nvidia-smi topo -m |  while IFS= read -r line; do
    echo -e "$ECHO_PREFIX $line"
done
echo -e "${CYAN}--------------------------------------------------${NC}\n"

if [ $machine_rank -eq 0 ]; then
  echo -e "${YELLOW}=============== Test $test_id: InfiniBand State ===============${NC}"
  echo -e "${BLUE}Description:${NC} Show the InfiniBand state using 'ibstat'. Useful for understanding the network interconnections."
  test_id=$((test_id + 1))
fi
echo -e "${GREEN}$ECHO_PREFIX Checking InfiniBand state with 'ibstat'...${NC}"
ibstat |  while IFS= read -r line; do
    echo -e "$ECHO_PREFIX $line"
done
echo -e "${CYAN}==================================================${NC}\n"