#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for barrier in "progress" "decentralized" "centralized"; do
    for i in $(seq 0 9); do
        mkdir -p "${SCRIPT_DIR}/random_goals/${barrier}/${i}"
    done
done

for barrier in "progress" "decentralized" "centralized"; do
    for i in $(seq 0 9); do
        for n in $(seq 1 20); do
            uv run python "${SCRIPT_DIR}/random_goals.py" --skip-initialization --barrier "${barrier}" --experiment-num "${i}" "${n}" > "${SCRIPT_DIR}/random_goals/${barrier}/${i}/${n}.txt"
        done
    done
done