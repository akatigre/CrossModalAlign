export CUDA_VISIBLE_DEVICES=1
export DESCRIPTION="He is wearing lipstick"
export METHOD="RandomInterpolation"
for i in {1..5}
do
  echo "Command no. $i"
  echo "$DESCRIPTION - $METHOD"
  python latent_optimization.py --method "$METHOD" --description "$DESCRIPTION"
