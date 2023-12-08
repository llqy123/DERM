 

To preprocess the datasets, run the following commands.

```shell script
python process_datasets.py
```

```
WN18RR
python run.py --dataset WN18RR --model DERM --rank 32 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --double_neg --multi_c
 
FB237
CUDA_VISIBLE_DEVICES=2 python run.py --dataset FB237 --model DERM --rank 32 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --multi_c
 
YAGO3-10
python run.py --dataset YAGO3-10 --model DERM --rank 32 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 2000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --multi_c
 
```

