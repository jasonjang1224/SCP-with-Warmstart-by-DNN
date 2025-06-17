import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt

# 0. 경로 설정
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. 표준화 함수
def standardize_columns(matrix):
    col_mean = matrix.mean(axis=0)
    col_std = matrix.std(axis=0)
    standardized_matrix = (matrix - col_mean) / col_std
    standardized_matrix = np.nan_to_num(standardized_matrix, nan=0.0)
    return standardized_matrix, col_mean, col_std

def reverse_standardize(standardized_matrix, col_mean, col_std):
    return standardized_matrix * col_std + col_mean

data = np.load('data_unicycle_20000_sig0.5_tf_free).npz')
Totalx = data['Totalxini']
Totalu = data['Totaltf']

xi = Totalx[0]  # (20000, 3)
xf = Totalx[1]  # (20000, 3)

# v와 w를 옆으로 붙이기
input_x = np.concatenate([xi, xf], axis=1)

#xresult = []
#for i in range(0, Totalx.shape[1], 3):
    #stackedx = np.vstack((Totalx[:, i], Totalx[:, i+1], Totalx[:, i+2])).reshape(-1)
    #xresult.append(stackedx[:, np.newaxis])
#input_x = np.hstack(xresult)
#input_x=input_x.T

#control learning
#uresult = []
#for i in range(0, Totalu.shape[1], 2):
    #stackedu = np.vstack((Totalu[:, i], Totalu[:, i+1])).reshape(-1)  
    #uresult.append(stackedu[:, np.newaxis])           
#output_u = np.hstack(uresult)
#output_u=output_u.T

#control learning (v 먼저, 그 다음 w)
#v_all = []
#w_all = []
#for i in range(0, Totalu.shape[1], 2):
    #v_all.append(Totalu[:, i][:, np.newaxis])     # v
    #w_all.append(Totalu[:, i+1][:, np.newaxis])   # omega

# (N, T) 형태로 쌓고, 나중에 np.hstack 하면 (N, 2T)
#v_concat = np.hstack(v_all)  # shape: (N, T)
#w_concat = np.hstack(w_all)  # shape: (N, T)

#output_u = np.vstack([v_concat, w_concat])  # shape: (N, 2T)
output_u=Totalu.T

#trajectory learning
#xtrajresult = []
#for i in range(0, Totalu.shape[1], 3):
    #stackedx = np.vstack((Totalu[:, i], Totalu[:, i+1], Totalu[:, i+2])).reshape(-1)
    #xresult.append(stackedx[:, np.newaxis])
#output_u = np.hstack(xresult)
#output_u=output_u.T

x_norm, x_mean, x_std = standardize_columns(input_x)
u_norm, u_mean, u_std = standardize_columns(output_u)

class controlDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

full_dataset = controlDataset(x_norm, u_norm)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class InitialToControlMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=1):#, dropout_prob=0.4):
        super(InitialToControlMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=dropout_prob),  # 첫 번째 드롭아웃

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(p=dropout_prob),  # 두 번째 드롭아웃

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InitialToControlMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

model.eval()
predictions = []
ground_truths = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())
        ground_truths.append(targets.cpu().numpy())

predictions = np.vstack(predictions)
ground_truths = np.vstack(ground_truths)

predictions_unstd = reverse_standardize(predictions, u_mean, u_std)
ground_truths_unstd = reverse_standardize(ground_truths, u_mean, u_std)

sample_idx = 14
plt.figure(figsize=(12,6))
plt.step(np.arange(predictions_unstd.shape[1]), ground_truths_unstd[sample_idx], label='True control', where='post')
plt.step(np.arange(predictions_unstd.shape[1]), predictions_unstd[sample_idx], label='Predicted control', linestyle='--', where='post')
plt.title(f"Test Sample {sample_idx} Prediction vs True")
plt.xlabel('Timestep')
plt.ylabel('control')
plt.legend()
plt.grid(True)
plt.show()

# 11. 모델 저장
MODEL_PATH = 'tf_mlp.pth'
checkpoint = {
    'model': model,
    'y_mean': u_mean,
    'y_std': u_std,
    'x_mean': x_mean,
    'x_std': x_std
}
torch.save(checkpoint, MODEL_PATH)

# 12. 테스트 데이터의 initial state (Totalx0) 저장
test_indices = test_dataset.indices  # test_dataset은 Subset 객체

# input_x는 정규화 전의 전체 input sequence이므로 이 중에서 test에 해당하는 것만 추출
test_initial_states = input_x[test_indices]  # shape: (num_test_samples, input_dim)

# 필요한 경우 저장
np.save('test_initial_states.npy', test_initial_states)