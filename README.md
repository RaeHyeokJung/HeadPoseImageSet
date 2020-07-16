# HeadPoseImageSet

이 코드는 http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html 의 데이터를 기반으로 한 Face Regression 코드입니다.

Online Learning 방식을 사용하였으며, 모델링은 VGG16을 커스텀해서 사용하였습니다.

Google Colab 실행 기준 MSE 0.00123 정도이며, Colab에서 실행 시 프로세서 타입을 GPU로 변경 후 실행 바랍니다.
