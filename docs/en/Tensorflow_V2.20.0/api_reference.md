# API Reference

<!-- md-trans-meta sourceCommit=062d2574fdfde12276d08b23b05dc9d9d8b875e2 translatedAt=2026-08-31T03:36:25.440Z pushedAt=2026-09-15T01:55:45.638Z -->

## Using the TensorFlow FAGO Static Graph Fusion Feature

The FlashAttentionGraphOptimization (FAGO) static graph fusion feature is toggled by environment variables. For details, see [Table 1 Environment variables](#table473618378218).

The default value of each environment variable is `0`, indicating that the FAGO static graph fusion feature is disabled. To enable it, you need to manually set the environment variables before graph optimization. For example, you can set the environment variables in Python as follows:

```python
import os
os.environ['ANNC_FUSED_ALL'] = '1'
os.environ['ANNC_FUSED_FLASHATTN_FWD'] = '1'
```

**Table 1** Environment variables<a id="table473618378218"></a>

| Environment Variable | Type | Value | Function |
| -------- | ------------ | --------------- | ------------------ |
| ANNC_FUSED_ALL | Process environment variable | `1`: Enable `0`: Disable | Enables static graph fusion for all FAGO fused operators.    |
| ANNC_FUSED_FLASHATTN_FWD | Process environment variable | `1`: Enable `0`: Disable | Enables static graph fusion for the FlashAttentionForward operator.    |

> ![icon note](public_sys-resources/icon-note.gif) **NOTE:**
> Operator fusion will not be performed for any of the above operators if and only if `ANNC_FUSED_ALL` is set to `0` and the environment variable corresponding to the specific operator is also set to `0`.

## Change History

| Date   | Description         |
| ---------- | ---------------- |
| 2026-09-30 | This is the first official release. Added the description of using the TensorFlow FAGO static graph fusion feature. |
