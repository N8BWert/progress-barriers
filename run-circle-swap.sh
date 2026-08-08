#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# for barrier in "progress" "centralized" "double" "increasing" "decentralized"; do
for barrier in "progress"; do
    mkdir -p "${SCRIPT_DIR}/circle_swap/${barrier}"
done

# for barrier in "progress" "centralized" "double" "increasing" "decentralized"; do
for barrier in "progress"; do
    for i in $(seq 1 20); do
        uv run python "${SCRIPT_DIR}/circle_swap.py" --skip-initialization --barrier "${barrier}" "${i}" > "${SCRIPT_DIR}/circle_swap/${barrier}/${i}.txt"
    done
done
