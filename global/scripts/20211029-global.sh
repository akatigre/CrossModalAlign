(CUDA_VISIBLE_DEVICES=0 python global.py --method Random --q 1.96 --disentangle_fs3 T --num_test 50 ; CUDA_VISIBLE_DEVICES=0 python global.py --method Random --q 2.58 --disentangle_fs3 T --num_test 50 ;) & 
(CUDA_VISIBLE_DEVICES=1 python global.py --method Random --q 1.96 --disentangle_fs3 F --num_test 50 ; CUDA_VISIBLE_DEVICES=1 python global.py --method Random --q 2.58 --disentangle_fs3 F --num_test 50 ;) & 
(CUDA_VISIBLE_DEVICES=2 python global.py --method Random --q 2.33 --disentangle_fs3 T --num_test 50 ;) & 
(CUDA_VISIBLE_DEVICES=3 python global.py --method Random --q 2.33 --disentangle_fs3 F --num_test 50 ;) & 
