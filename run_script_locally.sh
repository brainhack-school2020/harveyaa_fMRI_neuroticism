#!/bin/bash

echo "now processing neuroticism"
#python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='N' --method='LR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='N' --method='SVR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='N' --method='CPM'

echo "now processing agreeableness"
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='A' --method='LR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='A' --method='SVR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='A' --method='CPM'

echo "now processing extraversion"
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='E' --method='LR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='E' --method='SVR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='E' --method='CPM'

echo "now processing openness"
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='O' --method='LR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='O' --method='SVR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='O' --method='CPM'

echo "now processing conscientiousness"
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='C' --method='LR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='C' --method='SVR'
python ptn_script.py --netmats_file=netmats2_clean.txt --traits_file=all_traits.csv --trait='C' --method='CPM'

