import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    """
    Multi Layer Perceptron으로 환자 정보(Text)를 분석하고자 하는 모델

    실행 예시

    # 하이퍼파라미터 설정
    input_size = 784
    hidden_size = 128  # 은닉층 크기
    output_size = 10   # 클래스 수 (예: MNIST는 10개의 숫자 분류)

    # 모델, 손실 함수, 옵티마이저 정의
    model = MLP(input_size, hidden_size, output_size)

    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 첫 번째 fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # 두 번째 fully connected layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 활성화 함수 적용
        x = self.fc2(x)  # 출력층 (활성화 함수는 여기서 생략)
        return x


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory(Text)으로 환자 정보를 분석하고자 하는 모델

    실행 예시

    # 하이퍼파라미터 설정
    input_size = 10   # 입력 특징의 크기 (예: 10차원 입력)
    hidden_size = 50  # 은닉 상태 크기
    output_size = 1   # 출력 크기 (예: 회귀 문제)
    num_layers = 2    # LSTM 레이어 수
    sequence_length = 5  # 시퀀스 길이
    batch_size = 32

    # 모델, 손실 함수, 옵티마이저 정의
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)

    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어 정의
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 레이어에 입력을 전달
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 cell state

        # LSTM의 출력
        out, _ = self.lstm(x, (h0, c0))

        # 최종 time step의 출력만을 사용 (시계열 데이터 마지막 값)
        out = self.fc(out[:, -1, :])
        return out