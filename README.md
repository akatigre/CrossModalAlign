### Global Direction

- Prepare
  
  1. Download npy/ffhq, styleGAN2 pretrained model, ArcFace pretrained model, Face segmentation pretrained model and test faces from      https://drive.google.com/drive/folders/1LXGi5WF2uxRs0gRICDSkhhfYynTs2haL?usp=sharing
  
  2. Create Docker Contrainer
      > cd global 
      > 
      > bash ./docker/docker.sh
    


### 실험: StyleCLIP baseline 
  
  <pre>
  <code>
  cd global
  python global.py --method "Baseline" --num_test 10 --topk 50
  </code>
  </pre>
  
### 실험: StyleCLIP Ours

  <pre>
  <code>
  cd global 
  python global.py --method "Random" --num_test 100 --topk 50
  </code>
  </pre>

  * num_test: Number of test faces to use
  * num_attempts: Number of iterations (check diversity)
  * topk: Number of channels to change


**model.py** : RandomInterpolation defined
   
   > Extracts core semantics, unwanted semantics from target and source positives from the source <br>
     Use probabilistic approach to sample and create updated final target embedding
