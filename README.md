# MobileNet을 사용한 이미지 분류기 구현

## 2020-2 KHU Machine Learning Project

![MobileNet](https://user-images.githubusercontent.com/64248143/145080764-9494f138-b6b1-415f-874d-2ee029fccf01.JPG)

train set을 통해 학습시키고, valid set으로 가장 좋은 모델을 선정한 후 test set을 가장 잘 분류하는 모델과 하이퍼 파라미터들을 찾아 정확도를 최대화시키는 것을 목적으로 한다.
이를 위해 기본 조건을 바탕으로 ResNet, MobilenetV2, UNet, EfficientNet 등의 모델을 실행시켜 가장 성능이 뛰어난 모델을 선택 후 학습률 변경, epoch 변경, data augmentation, dropout 추가 등의 실험을 통하여 학습률 0.05, epoch 220일 때의 조합으로 약 85%의 가장 최대의 점수를 이끌어낼 수 있었다. 즉 학습률 변화와 epoch의 수를 변화한 것이 성능을 높이는데 주요한 역할을 한 것이다.
데이터의 크기 확장, train data에 더하여 valid data를 사용하는 방법, train data의 개수를 data augmentation을 통해 늘리는 방법, layer normalization 등의 방안을 도입하여 실험한다면 더 좋은 성능의 도출할 수 있을 것이라 예측된다.

## 개발환경

- Google Colab
- PyTorch
