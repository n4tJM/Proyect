import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import Mario_enviroment as envm

# LOAD PREPROCESS AND SPLIT
df_mario = 
df_mnist['Id'] = df_mnist.index + 1
df_mnist = df_mnist[['Id'] + [col for col in df_mnist.columns if col != 'Id']]
df_mnist.drop('Id', axis=1, inplace=True)

X = df_mnist.drop('Clase', axis=1).values
Y = df_mnist['Clase'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=55)


#Convert to tensors
X_train = torch.FloatTensor(X_train)
Y_train = torch.LongTensor(Y_train)
X_test = torch.FloatTensor(X_test)
Y_test = torch.LongTensor(Y_test)

# Neural Network model
class MinstNeuralNetwork(nn.Module):
    def __init__(self, n_input, n_output) -> None: # Crea la estructura de la red
        super(MinstNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(n_input, 256) # El segundo numero es la cantidad de neuronas en esa capa, por lo que son la cantidad de neuronas de salida
        self.hidden_layer1 = nn.Linear(256, 128) # El primer numero es la cantidad de salidas de la capa anterior
        self.output_layer = nn.Linear(128, n_output)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.tanh(self.input_layer(x))
        out = self.tanh(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out        

 
n_input = 784
n_output = 10
model = MinstNeuralNetwork(n_input, n_output)

#Optimizer
learning_rate = 0.01
loss_function = nn.CrossEntropyLoss()
# Los parametros del modelo son los pesos 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 200
losses = [] # errores

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, Y_train)
    losses.append(loss.item())
    print(f'epocas: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

preds = []
with torch.no_grad():
    for val in X_test:
        y_hat = model.forward(val)
        #print(y_hat)
        preds.append(y_hat.argmax().item())

df = pd.DataFrame({'Y': Y_test, 'YHat': preds})
df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
#print(df)
print('Accuracy: ',df['Correct'].sum() / len(df))

    
with torch.no_grad():
    print(model.input_layer.weight)
    print(model.hidden_layer1.weight)
    print(model.output_layer.weight)
