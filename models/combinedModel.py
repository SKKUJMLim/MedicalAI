import torch
import torch.nn as nn

class CombinedResNet18(nn.Module):
    def __init__(self, model1, model2, model3 ,num_classes):
        super(CombinedResNet18, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

        # 두 모델의 출력 크기를 합친 크기의 FC 레이어
        self.fc = nn.Linear(512 * 2 + 1, num_classes) # Resnet 18, 34
        # self.fc = nn.Linear(512 * 2, num_classes)  # Resnet 18, 34
        # self.fc = nn.Linear(512 * 8 + 1, num_classes)   # Resnet 50
        # self.fc = nn.Linear(num_classes * 3, num_classes)

    def forward(self, x1, x2, x3):
            # 모델 각각에 이미지를 입력하여 latent vector 추출
            latent_vector1 = self.model1(x1)
            latent_vector2 = self.model2(x2)

            # 환자 정보 이미지
            latent_vector3 = self.model3(x3)

            # 두 latent vector를 concat
            combined = torch.cat((latent_vector1, latent_vector2, latent_vector3), dim=1)

            # 결합된 벡터를 최종 분류 레이어에 통과시킴
            output = self.fc(combined)

            return output


class CombinedResNet18_onlyImage(nn.Module):
    def __init__(self, model1, model2 ,num_classes):
        super(CombinedResNet18_onlyImage, self).__init__()
        self.model1 = model1
        self.model2 = model2

        # 두 모델의 출력 크기를 합친 크기의 FC 레이어
        self.fc = nn.Linear(512 * 2, num_classes)  # Resnet 18, 34
        # self.fc = nn.Linear(512 * 8 + 1, num_classes)   # Resnet 50
        # self.fc = nn.Linear(num_classes * 3, num_classes)

    def forward(self, x1, x2):
            # 모델 각각에 이미지를 입력하여 latent vector 추출
            latent_vector1 = self.model1(x1)
            latent_vector2 = self.model2(x2)

            # 환자 정보 이미지
            #latent_vector3 = self.model3(x3)

            # 두 latent vector를 concat
            combined = torch.cat((latent_vector1, latent_vector2), dim=1)

            # 결합된 벡터를 최종 분류 레이어에 통과시킴
            output = self.fc(combined)

            return output



class CombinedVGG(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super(CombinedVGG, self).__init__()
        self.model1 = model1
        self.model2 = model2

        # 두 모델의 출력 크기를 합친 크기의 FC 레이어
        # self.fc = nn.Linear(512 * 8, num_classes)
        # self.fc = nn.Linear(200704, num_classes)
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x1, x2):
            # 모델 각각에 이미지를 입력하여 latent vector 추출
            latent_vector1 = self.model1(x1)
            latent_vector2 = self.model2(x2)

            # 두 latent vector를 concat
            combined = torch.cat((latent_vector1, latent_vector2), dim=1)

            # 결합된 벡터를 최종 분류 레이어에 통과시킴
            output = self.fc(combined)

            return output