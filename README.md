### Global Direction

- Prepare
  
  1. Download fs3.npy, styleGAN2 pretrained model, ArcFace pretrained model and test faces from      https://drive.google.com/drive/folders/1LXGi5WF2uxRs0gRICDSkhhfYynTs2haL?usp=sharing
  2. Create Docker Contrainer
      > cd global 
      > 
      > bash ./docker/docker.sh
    
  * fs3.npy: global/npy/ffhq/ 아래에 위치함

  * StyleGAN2 & ArcFace: ./pretrained_models/ 아래에 위치함
  * test_faces.pt: Optimization & Global에서 사용 -> 파일 위치 올바르게 바꾸기

- 실험: StyleCLIP baseline 
  
  <pre>
  <code>
  cd global
  python global.py --method "Baseline" --num_test 100 --topk 50
  </code>
  </pre>
  
- 실험: StyleCLIP Ours

  <pre>
  <code>
  cd global 
  python global.py --method "Random" --num_test 100 --topk 50
  </code>
  </pre>

  * num_test: Number of test faces to use
  * num_attempts: Number of iterations (check diversity)
  * topk: Number of channels to change
  * easy: Use easy descriptions as target text (store_true)


**python text_model.py** : RandomInterpolation defined
   
   > Extracts core semantics, unwanted semantics from target and source positives from the source <br>
     Use probabilistic approach to sample and create updated final target embedding
     
     
     
### Latent Optimization

  **change $DESCRIPTION, $METHOD in run.sh**  
  <pre>
  <code>
  cd optimization
  bash run.sh
  </code>
  </pre>

