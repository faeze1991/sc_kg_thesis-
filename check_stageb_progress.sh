#!/bin/bash
total=0
for i in 0 1 2 3; do
    if [ -f "data/interim/stage_b_checkpoint_shard${i}.json" ]; then
        n=$(python3 -c "import json; print(json.load(open('data/interim/stage_b_checkpoint_shard${i}.json'))['completed'])")
        echo "Shard $i: $n completed"
        total=$((total + n))
    else
        echo "Shard $i: not started yet"
    fi
done
echo "---"
echo "Total: $total / 280,239"
