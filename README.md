### Global Direction

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
