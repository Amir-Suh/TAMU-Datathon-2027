#!/bin/bash
set -euo pipefail
scratch=$1
data=$2
shift 2
for arm in "$@"; do
    case "$arm" in w1a_p0|hybrid|m3) ;; *) echo "Unknown arm $arm" >&2; exit 2 ;; esac
    output="$scratch/container-$arm"
    mkdir "$output"
    docker run --rm --pull never --network none --read-only --cap-drop ALL \
        --security-opt no-new-privileges --pids-limit 128 \
        --cpus 2 --memory 16g --memory-swap 16g \
        --user "$(id -u):$(id -g)" --workdir /submission \
        --tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m \
        --mount "type=bind,source=$scratch/$arm,target=/submission,readonly" \
        --mount "type=bind,source=$data,target=/data,readonly" \
        --mount "type=bind,source=$output,target=/output" \
        --mount "type=bind,source=$scratch/container_driver.py,target=/container_driver.py,readonly" \
        house-edge-main-replay:py313 python -I -B /container_driver.py exact
done
