export DEBUG=" "
export USE_SLURM=0


python dtu_eval.py --dataset_path "/data/dtu_down_4" --model_path "/data/pmodel.pth" --output_path "./presults" --name "light" --src_list "22 25 28" --input_view 3 --model_type='LightFormer'
