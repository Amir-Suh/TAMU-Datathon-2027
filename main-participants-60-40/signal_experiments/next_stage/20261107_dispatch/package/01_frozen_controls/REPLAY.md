# Clean replay procedure

From the participant-kit root, use an official Python 3.13 Docker image and preprovision the pinned dependencies. The build sends the Dockerfile as its build definition and an empty scratch directory as context. The replay itself has no network.

```sh
dispatch=signal_experiments/next_stage/20261107_dispatch
pkg="$dispatch/package"
controls="$pkg/01_frozen_controls"
candidate="$pkg/02_w1a_p0"
scratch=$(mktemp -d /private/tmp/house-edge-package.XXXXXX)
mkdir "$scratch/container-build"
cp "$controls/Dockerfile" "$scratch/container-build/"
docker build --pull=false --progress=plain -t house-edge-main-replay:py313 "$scratch/container-build"
cp -R public "$scratch/round-data"
cp "$controls/container_driver.py" "$scratch/container_driver.py"
unzip -q "$candidate/w1a_p0_code.zip" -d "$scratch/w1a_p0"
unzip -q "$controls/hybrid_code.zip" -d "$scratch/hybrid"
unzip -q "$controls/m3_code.zip" -d "$scratch/m3"
chmod -R a-w "$scratch/round-data" "$scratch/w1a_p0" "$scratch/hybrid" "$scratch/m3" "$scratch/container_driver.py"
bash "$controls/run_docker.sh" "$scratch" "$scratch/round-data" w1a_p0 hybrid m3
python3 -B "$controls/verify_replay.py" "$scratch" w1a_p0 hybrid m3
```

The verification script uses the kit's normalized CSV parser and checks IDs against released queries. It stores reports, replays and access traces in the corresponding attempt folders. A fresh audit phase executes the extracted prediction code with an allowlist of the two split inputs. The exact-command test run and its audited counterpart must have identical bytes.

The archive build recipes are in each attempt's `build.py`. Repacking assigns new ZIP metadata, so the frozen ZIP hash in each manifest identifies the exact verified upload artifact.
