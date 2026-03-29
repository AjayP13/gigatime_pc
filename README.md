# Commands

```bash
./run.sh  process-patient ./patient_013 --batch-size 20

./run.sh  process-patient ./patient_013 --batch-size 20 --threshold 0.5 --precision float32

./run.sh  view-image ./patient_013/1_1.png CK,DAPI,Ki67,CD8,CD3,PD-L1,CD68

./delete_all_npz.sh # to reset
```
