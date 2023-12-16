import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

train_data_path = 'data_real_train.csv'
test_data_path = 'data_real_test.csv'

train_data = pd.read_csv(train_data_path, delimiter=';')
test_data = pd.read_csv(test_data_path, delimiter=';')

X = train_data.drop('class', axis=1)
y = train_data['class']
X_test = test_data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

model = Sequential([
    Dense(512, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=128, class_weight=class_weights,
                    validation_data=(X_val_scaled, y_val), callbacks=[early_stopping], verbose=1)

test_predictions = model.predict(X_test_scaled)
test_predictions = [1 if x > 0.5 else 0 for x in test_predictions]
output = ''.join(map(str, test_predictions))

print("Прогноз на тестовых данных:", output)
print("\nClassification Report on Training Data:")
y_pred_train = model.predict(X_train_scaled)
y_pred_train = [1 if x > 0.5 else 0 for x in y_pred_train]
print(classification_report(y_train, y_pred_train))

print("\nTraining History:")
for epoch, accuracy, loss in zip(range(len(history.history['accuracy'])), history.history['accuracy'], history.history['loss']):
    print(f"Epoch {epoch + 1} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

print("\nConfusion Matrix on Training Data:")
print(confusion_matrix(y_train, y_pred_train))
print("\nROC AUC Score on Training Data:")
print(roc_auc_score(y_train, y_pred_train))
