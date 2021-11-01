# 주의

1. npy/ffhq/fs3.npy 없음
2. stylegan2/ 아래에 없음
3. 최근 styleCLIP 깃헙에서 바뀐 IDLoss 사용 중 idloss(a, b)[0]에서 [0]빼고 사용

|_criteria: clip_loss & id_loss
|_global: 깃허브에 업로드된 파일 / 없는 부분은 위에 적어둠
|_mapper: latent mapper
|_optimization: latent optimization
|_models: facial_recognition & stylegan2 모델 구조
|_pretrained_models: facial_recognition & stylegan2 체크포인트 저장소
