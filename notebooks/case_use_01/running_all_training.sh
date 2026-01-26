#!/usr/bin/bash

python mcs_04_train_models_split_pools_v1.py --case-dir ../../demonstrative_applications/case_demo_01 --target-col Antimicrobial --seq-col sequence --model lr

python mcs_04_train_models_split_pools_v1.py --case-dir ../../demonstrative_applications/case_demo_01 --target-col Antimicrobial --seq-col sequence --model knn

python mcs_04_train_models_split_pools_v1.py --case-dir ../../demonstrative_applications/case_demo_01 --target-col Antimicrobial --seq-col sequence --model svm

python mcs_04_train_models_split_pools_v1.py --case-dir ../../demonstrative_applications/case_demo_01 --target-col Antimicrobial --seq-col sequence --model rf