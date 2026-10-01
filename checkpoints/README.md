# Paper Checkpoints

The training scripts automatically save two checkpoints for each experiment:

```text
best.pt
last.pt
```

Use `best.pt` for the reported test-set evaluation. It is selected according to the highest validation CSI-M during training.

For exact paper-level reproducibility, publish the two `best.pt` files used for the manuscript as assets of a GitHub Release, for example:

```text
mfpe_cikm_best.pt
mfpe_shanghai_best.pt
```

After uploading the release assets, add their download links below:

```text
CIKM 2017:
<add GitHub Release asset link>

Shanghai:
<add GitHub Release asset link>
```

Do not replace the files with checkpoints from a later retraining run if the manuscript tables were produced with different weights.
