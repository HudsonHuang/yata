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
  - [x] mixup: [paper](https://arxiv.org/abs/1710.09412) [code](https://github.com/hongyi-zhang/mixup)
  - [x] SpecAugment: [paper](https://arxiv.org/abs/1904.08779) [code](https://github.com/DemisEom/SpecAugment)
  - [ ] mp3 as augumentation，用MP3编码后去掉的不可听噪声，把这种生成不可听噪声作为数据增强的手段（做法：加高斯，把MP3当成一个mask去编码，把MP3mask挖掉的区域的高斯留下来加到频谱上面，形成不可听噪声）
  - [ ] phase putertubation
- feature extraction
  - PASE: [paper](https://arxiv.org/abs/2001.09239) [code](https://github.com/santi-pdp/pase)
  - Multi scale MelSpectrogram
- Tensorflow alternatives
    - yata.utils.HParams:   
      An alternative to tf.contrib.training.HParams without Tensorflow dependency
    - yata.utils.to_categorical:   
      An alternative to tf.keras.utils.to_categorical without Tensorflow & keras dependency
- handy tools
    - yata.utils.run():  
      No more ArgumentParser!!   
      you can pass and update any parameter with:
      ```
      python test.py --a 2 --lr 0.01
      ```
      with code like:
      ```
      import yata
      
      default_hp = {"a":1,"b":2}
      args = yata.util.run(default_hp)
      ```
      you can acess params like HParams:
      ```
      print(args.a, args.b) # acess default params
      print(args.lr) # acess newly add params from CLI
      ```
    - yata.utils.new_dir:   
        Make directory like this `./file_a/tag/1/` with:
        ```
        new_dir("file_a", "tag", 1)
        ```
    - yata.utils.backup_code:  
        Backup all your \*.py(optional) to a zip file, eg. backup code for every experiments before running.
    - yata.utils.get_current_date: Get date as string
    

  
