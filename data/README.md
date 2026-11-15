# Datasets

Place the preprocessed datasets under this directory:

```text
data/
??? isic2017/
?   ??? train/images/
?   ??? train/masks/
?   ??? val/images/
?   ??? val/masks/
??? isic2018/
?   ??? train/images/
?   ??? train/masks/
?   ??? val/images/
?   ??? val/masks/
??? Synapse/
    ??? train_npz/
    ??? test_vol_h5/
    ??? lists/lists_Synapse/
        ??? train.txt
        ??? test_vol.txt
```

Run training from the DAS-VMUNet repository root. The default configuration paths point to this layout. See the [main README](../README.md#data-preparation) for formats and preparation instructions.
