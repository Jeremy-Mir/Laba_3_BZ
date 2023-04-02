from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import numpy as np

# Определение обучающих данных
input = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
input_test = np.array([[0.2,0.3,0.1], [0,0,1], [0,1,0], [0,1,1],[1,0,0],[1,0,1],[1,1,0],[0.8,0.9,0.85]])
output = np.array([1,1,0,0,1,1,0,0])

# Создание нейронной сети
model = Sequential()
model.add(Dense(2, input_dim=3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# Компиляция модели

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print("Начальные коэффициенты")
for layer in model.layers: print(layer.get_weights()[1])



# Обучение сети и сохранение истории обучения

history = model.fit(input, output, epochs=2000, verbose=0)
print(model.predict(input_test))

print("Коэффициенты после тренировки")
for layer in model.layers: print(layer.get_weights()[1])


# Получение значений потерь и точности для каждой эпохи
loss = history.history['loss']
acc = history.history['accuracy']

# Построение графика потерь
plt.plot(loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Построение графика точности
plt.plot(acc)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()