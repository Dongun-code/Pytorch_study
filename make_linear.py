import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

num_data = 1000
num_epoch = 500

x = init.uniform_(torch.Tensor(num_data, 1),-10,10)     # -10~10 사이의 값이 담긴 tensor 만들기
noise = init.normal_(torch.FloatTensor(num_data, 1),std=1)     # 정규분포로 랜덤값 생성
print(noise)

y=2*x+3
y_noise = 2*(x+noise)+3

model = nn.Linear(1,1)
loss_func = nn.L1Loss()

optimizer = optim.SGD(model.parameters(), lr= 0.01)

loss_arr = []
label = y_noise

for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_func(output, label)
    loss.backward()         #기울기 학습하는데 사용 기울기 구하기
    optimizer.step()        # 기울기를 이용해 변수 업데이트

    if i % 10 == 0:
        # plt.scatter(x.detach().numpy(),output.detach().numpy())
        plt.axis([-10, 10, -30, 30])
        # plt.show()
        # print(loss.data)

    loss_arr.append(loss.detach().numpy())

plt.figure(figsize=(15,15))
plt.scatter(x.numpy(),y_noise.numpy(),s=5,c="gray")
plt.scatter(x.detach().numpy(),output.detach().numpy(),s=5,c="red")
plt.axis([-10, 10, -30, 30])
# plt.show()

plt.plot(loss_arr)
# plt.show()

param_list = list(model.parameters())
print("Weight",param_list[0],'Bias', param_list[1])