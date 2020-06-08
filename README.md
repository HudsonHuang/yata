# yata[WIP]
Yet Another Tools for Audio deep learning(for myself).
```
pip install libyata
```
## Usage

```
import yata
```

- data augmentation
  - mixup: [paper](https://arxiv.org/abs/1710.09412) [code](https://github.com/hongyi-zhang/mixup)
  - SpecAugment: [paper](https://arxiv.org/abs/1904.08779) [code](https://github.com/DemisEom/SpecAugment)
- feature extraction
  - PASE: [paper](https://arxiv.org/abs/2001.09239) [code](https://github.com/santi-pdp/pase)
  - Multi scale MelSpectrogram
- Tensorflow alternatives
    - yata.utils.HParam:   
      An alternative to tf.contrib.training.HParams without Tensorflow dependency
    - yata.utils.to_categorical:   
      An alternative to tf.keras.utils.to_categorical without Tensorflow dependency
- handy tools
    - yata.utils.new_dir:   
        Make directory like this `./file_a/tag/1/` with:
        ```
        new_dir("file_a", "tag", 1)
        ```
    - yata.utils.get_current_date: Get date as string
    

  
