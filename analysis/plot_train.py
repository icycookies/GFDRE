import matplotlib.pyplot as plt
import numpy as np

with open("train_logs.npy", "rb") as f:
    loss = np.load(f)[3:]
    train_f1 = np.load(f)
    val_f1 = np.load(f)

x1 = np.arange(len(loss)) * 50.0 / 764
x2 = np.arange(len(train_f1)) + 1
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x1, loss)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('loss')
ax2.plot(x2, train_f1, label='train')
ax2.plot(x2, val_f1, label='dev')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1')
plt.show()