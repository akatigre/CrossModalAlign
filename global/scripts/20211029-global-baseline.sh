(CUDA_VISIBLE_DEVICES=0 python global.py --method Baseline --beta 0.08 --disentangle_fs3 F --num_test 50 ;) & 
(CUDA_VISIBLE_DEVICES=1 python global.py --method Baseline --beta 0.1 --disentangle_fs3 F --num_test 50 ;) & 
(CUDA_VISIBLE_DEVICES=2 python global.py --method Baseline --beta 0.12 --disentangle_fs3 F --num_test 50 ;) & 
