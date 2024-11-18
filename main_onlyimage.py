from dataloader import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from models import resnet, vgg, combinedModel, clinicinfo

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 장치:", device)

    # 난수 시드 고정
    torch.manual_seed(42)

    resize = 224
    # resize = 32
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 2
    batch_size = 4
    num_epochs = 50
    learning_rate = 0.001

    best_accuracy = 0.0
    best_model_path = './best_model.pth'

    train_loaders_dict, test_dataloader = get_dataloader(resize, mean, std, batch_size)

    print('==> Building model..')
    # resnet.test()
    preap_net = resnet.resnet34(pretrained=False)
    prelat_net= resnet.resnet34(pretrained=False)
    combined_model = combinedModel.CombinedResNet18_onlyImage(preap_net, prelat_net, num_classes)

    # vgg.test()
    # preap_net = vgg.VGG('VGG19')
    # prelat_net = vgg.VGG('VGG19')
    # combined_model = combinedModel.CombinedVGG(preap_net, prelat_net, num_classes)

    combined_model.to(device=device)

    # 최적화 기법 설정
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(combined_model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    # optimizer = optim.Adam([
    #     {'params': preap_net.parameters(), 'lr':learning_rate},
    #     {'params': prelat_net.parameters(), 'lr':learning_rate},
    # ])

    # 학습 및 검증 실시


    ## epoch 루프
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print("------------------------------")

        running_loss = 0.0
        combined_model.train()

        trainloader =  train_loaders_dict['train']

        for preap_inputs, prelat_inputs, clinic_inputs, labels in trainloader:

            # GPU가 사용가능하면 GPU에 데이터 전송
            preap_inputs = preap_inputs.to(device)
            prelat_inputs = prelat_inputs.to(device)
            clinic_inputs = clinic_inputs.to(device)
            labels = labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()
            outputs = combined_model(preap_inputs, prelat_inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(trainloader)}')

        # Validation
        combined_model.eval()
        correct = 0
        total = 0
        validationloader = train_loaders_dict['val']

        with torch.no_grad():
            for preap_inputs, prelat_inputs, clinic_inputs, labels in validationloader:
                preap_inputs = preap_inputs.to(device)
                prelat_inputs = prelat_inputs.to(device)
                clinic_inputs = clinic_inputs.to(device)
                labels = labels.to(device)
                outputs = combined_model(preap_inputs, prelat_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

        # Best 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(combined_model.state_dict(), best_model_path)
            print(f'Saving model with accuracy: {best_accuracy}%')


    # 테스트 데이터셋에 대한 성능 평가
    combined_model.load_state_dict(torch.load(best_model_path))
    combined_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for preap_inputs, prelat_inputs, clinic_inputs, labels in test_dataloader:
            # GPU가 사용가능하면 GPU에 데이터 전송
            preap_inputs = preap_inputs.to(device)
            prelat_inputs = prelat_inputs.to(device)
            clinic_inputs = clinic_inputs.to(device)
            labels = labels.to(device)
            combined_model.to(device)
            outputs = combined_model(preap_inputs, prelat_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the best model on the test images: {100 * correct / total}%')