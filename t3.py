import math
import pandas as pd

train_data = pd.read_csv('pendulum_train.csv')
ts = train_data['t'].values
thetas = train_data['theta']


a = 6.87423
b = 9.2339

num = 20000
lr = 0.00005

omegas = []
as_ = []

for i in range(202):
  omegas.append((thetas[i + 1] - thetas[i]) / (ts[i + 1] - ts[i]))
for i in range(201):
  as_.append((omegas[i + 1] - omegas[i]) / (ts[i + 1] - ts[i]))
# print(omegas)
print(as_[0], omegas[0], thetas[0])

for i in range(num):
  for j in range(198):
    p =  - a * omegas[j] - b * math.sin(thetas[j])
    delta = (p - as_[j + 1])
    a += lr * delta * omegas[j]
    b += lr * delta * math.sin(thetas[j])
  if i % 1000 == 999:
    print(i + 1, a, b)

print(a, b)

print('l: {}'.format(9.8 / b))
print('μ: {}'.format(a))


'''
l: 4.9438090506239645
μ: 0.5497842608076727

l: 4.941672877805168
μ: 0.549309252225413


'''