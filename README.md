> The current repository contains a preliminary draft version of the codebase. Due to ongoing work and scheduling constraints, the code has not yet been thoroughly cleaned, refactored, or fully documented. We are planning a major update to provide a well-structured and polished version once time permits. Thank you for your patience and understanding.

To train the model from scratch, run the following command:
### Training Mode
```
python main.py \
  --mode 'Train' \
  --n_gpu <GPUs be to used> \
  --image_dir <SLIDE FEATURE PATH UNI USING> \
  --image_dir_plip <SLIDE FEATURE PATH PLIP USING> \
  --ann_path <CAPTION PATH> \
  --split_path <PATH containing directory splits the to train/val/test> \
  --save_dir <SAVING CKPT PATH>
```
### Testing Mode
To evaluate the model using a trained checkpoint, run:
```
python main.py \
  --mode 'Test' \
  --image_dir <SLIDE FEATURE PATH UNI USING> \
  --image_dir_plip <SLIDE FEATURE PATH PLIP USING> \
  --ann_path <CAPTION PATH> \
  --split_path <PATH containing directory splits the to train/val/test> \
  --checkpoint_dir <PATH CKPT TO> \
  --save_dir <PATH RESULTS SAVING TO>
```

### 🙏 Acknowledgements
Our codebase is built upon or inspired by the following excellent open-source project. We express our sincere gratitude to the authors for their wonderful work:

BiGen: https://github.com/DeepMed-Lab-ECNU/BiGen.
