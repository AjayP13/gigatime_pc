# Commands

```bash
./run.sh  process-patient ./patient_013 --batch-size 20

./run.sh  process-patient ./patient_013 --batch-size 20 --threshold 0.5 --precision float32

./run.sh  view-image ./patient_013/1_1.png DAPI,CD138,CD3,CK,Ki67

./delete_all_npz.sh # to reset
```
