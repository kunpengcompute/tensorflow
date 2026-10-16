# TensorFlow v2.20.0 Patch Release

<!-- md-trans-meta sourceCommit=50c6aa44e4779ee097085a37fd0c469acbb90932 translatedAt=2026-10-12T06:57:30.568Z pushedAt=2026-10-12T08:52:43.830Z -->

This document describes the Kunpeng TensorFlow patches based on the fixed official TensorFlow `v2.20.0` baseline:

```text
bf5899deaf70fa45173c5c7b8dc9ace8824dc980
```

`SOURCE_COMMIT` records the development revision used for the release, and `SHA256SUMS` records the artifact checksums.

## Maintained Patch Series

| Order | Group | Description |
| --- | --- | --- |
| 1 | common | Common Bazel, repository integration, and general build fixes |
| 2 | fago | FAGO static graph fusion operators and tests |

The following build profiles are supported:

| Profile | Groups | Purpose |
| --- | --- | --- |
| common-only | common | Validate shared changes |
| common-fago | common + fago | FAGO static graph fusion feature |

`tensorflow/feature_copts.bzl` is dynamically generated based on the feature set. Use `prepare_source.py` to create buildable source code instead of manually applying patches in order.

```bash
git clone -b master https://gitcode.com/boostkit/tensorflow.git kunpeng-tensorflow
cd kunpeng-tensorflow
git remote add tensorflow-upstream https://github.com/tensorflow/tensorflow.git
git fetch tensorflow-upstream refs/tags/v2.20.0:refs/tags/v2.20.0

python3 patches/Tensorflow_V2.20.0/prepare_source.py \
  --feature-set fago-core \
  --output-dir ../tensorflow-fago-core
```

Change `--feature-set` to create another profile.

## Integrity Check

Run the following commands to check patch integrity:

```bash
cd patches/Tensorflow_V2.20.0
sha256sum -c SHA256SUMS
```

`prepare_source.py` creates a complete source tree. `manifest.json` maintains the patch series. Files under `feature/` are generated artifacts and should not be modified manually.
