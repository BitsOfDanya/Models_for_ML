import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

train_data_path = 'data_real_train.csv'
test_data_path = 'data_real_test.csv'

train_data = pd.read_csv(train_data_path, delimiter=';')
test_data = pd.read_csv(test_data_path, delimiter=';')

X = train_data.drop('class', axis=1)
y = train_data['class']
X_test = test_data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = Sequential([
        Dense(1024, input_dim=X_train.shape[1], kernel_regularizer='l2'),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, kernel_regularizer='l2'),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=1)

    history = model.fit(X_train, y_train, epochs=200, batch_size=128, class_weight=class_weights,
                        validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=1)

    print("\nTraining History:")
    for epoch, accuracy, loss in zip(range(len(history.history['accuracy'])), history.history['accuracy'], history.history['loss']):
        print(f"Epoch {epoch + 1} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    y_val_pred = model.predict(X_val)
    y_val_pred = [1 if x > 0.5 else 0 for x in y_val_pred]
    print("\nClassification Report on Validation Data:")
    print(classification_report(y_val, y_val_pred))
    print("Confusion Matrix on Validation Data:")
    print(confusion_matrix(y_val, y_val_pred))
    print("ROC AUC Score on Validation Data:")
    print(roc_auc_score(y_val, y_val_pred))

test_predictions = model.predict(X_test_scaled)
test_predictions = [1 if x > 0.5 else 0 for x in test_predictions]
output = ''.join(map(str, test_predictions))
print("\nПрогноз на тестовых данных:", output)
