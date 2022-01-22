import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics, regularizers, callbacks
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score


data_path = ''
x_train = np.load(data_path + 'x_train_500.npy')
y_train = np.load(data_path + 'y_train.npy')
x_test  = np.load(data_path + 'x_test_500.npy')
y_test  = np.load(data_path + 'y_test.npy')
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)
print("x_train shape: ", x_train.shape, ", y_train shape:", y_train.shape)
print("x_test shape: ", x_test.shape, ", y_test shape:", y_test.shape)


# Model
input = layers.Input(shape=(12, 5000, 1))

X = layers.Conv2D(filters=32, kernel_size=(1, 55))(input)
X = layers.BatchNormalization()(X)
X = layers.ReLU()(X)
X = layers.MaxPooling2D(pool_size=(1, 2), strides=1)(X)

convC1 = layers.Conv2D(filters=64, kernel_size=(1, 57))(X)

X = layers.Conv2D(filters=32, kernel_size=(1, 55))(X)
X = layers.BatchNormalization()(X)
X = layers.ReLU()(X)
X = layers.MaxPooling2D(pool_size=(1, 4), strides=1)(X)

convC2 = layers.Conv2D(filters=64, kernel_size=(1, 56))(convC1)

X = layers.Conv2D(filters=64, kernel_size=(1, 55))(X)
X = layers.BatchNormalization()(X)
X = layers.Add()([convC2, X])           # skip Connection
X = layers.ReLU()(X)
X = layers.MaxPooling2D(pool_size=(1, 2), strides=1)(X)

convE1 = layers.Conv2D(filters=32, kernel_size=(1, 54))(X)

X = layers.Conv2D(filters=64, kernel_size=(1, 53))(X)
X = layers.BatchNormalization()(X)
X = layers.ReLU()(X)
X = layers.MaxPooling2D(pool_size=(1, 4), strides=1)(X)

convE2 = layers.Conv2D(filters=64, kernel_size=(1, 55))(convE1)

X = layers.Conv2D(filters=64, kernel_size=(1, 53))(X)
X = layers.BatchNormalization()(X)
X = layers.Add()([convE2, X])         # skip Connection
X = layers.ReLU()(X)
X = layers.MaxPooling2D(pool_size=(1, 2), strides=1)(X)
print('Added 5 layers for temporal analysis')

X = layers.Conv2D(filters=64, kernel_size=(12, 1))(X)
X = layers.BatchNormalization()(X)
X = layers.ReLU()(X)
X = layers.GlobalAveragePooling2D()(X)
print('Added 1 layer for spatial Analysis')

X = layers.Flatten()(X)

X = layers.Dense(units=128, kernel_regularizer=regularizers.L2(0.01))(X)
X = layers.BatchNormalization()(X)
X = layers.ReLU()(X)
X = layers.Dropout(rate=0.2)(X)

X = layers.Dense(units=32, kernel_regularizer=regularizers.L2(0.020))(X)
X = layers.BatchNormalization()(X)
X = layers.ReLU()(X) 
X = layers.Dropout(rate=0.25)(X)
print('Added 2 fully connected layers')

output = layers.Dense(4, activation='softmax')(X)
model = Model(inputs=input, outputs=output)
print(model.summary())

early    = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(filepath="best_epoch.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")
reducelr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
callback = [early, checkpoint, reducelr]
model.compile(optimizer = optimizers.Adam(learning_rate=0.0005), loss = losses.CategoricalCrossentropy(), metrics = [metrics.BinaryAccuracy(), metrics.AUC(curve='ROC')])
history = model.fit(x_train, y_train, validation_split=0.10, epochs=30, batch_size=64, callbacks=callback)

model = load_model("best_epoch.hdf5")
print(model.evaluate(x_train, y_train))
print(model.evaluate(x_test, y_test))

y_prob_test = model.predict(x_test)
y_num_test      = [np.argmax(row) for row in y_test]
y_num_prob_test = [np.argmax(row) for row in y_prob_test]

acc=4
def multi_class_sklearn(y_true, y_pred):
    acc='{:.2f}\n'.format(accuracy_score(y_true, y_pred) * 100)
    print('Accuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred) * 100))
    print('Micro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='micro') * 100))
    print('Macro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro') * 100))
    print('Macro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro') * 100))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='macro') * 100))
    print('Weighted Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='weighted') * 100))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='weighted') * 100))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_true, y_pred, average='weighted') * 100))
multi_class_sklearn(y_num_test, y_num_prob_test)

# caculate auc using num, predictions
print('Macro AUC (ovo)    : {:.2f}'.format(roc_auc_score(y_num_test, y_prob_test, multi_class='ovo', average='macro') * 100))

# calculate auprc for each class using num, prob
auc_sum = 0
for i in range(4):
  precision, recall, thresholds = precision_recall_curve(y_test[: ,i], y_prob_test[:, i])
  auc_sum += auc(recall, precision) 
print('Macro AUPRC        : {:.2f}'.format((auc_sum / 4) * 100))

acc=str(acc)
save_path = "finetuned-model-500_acc".replace("acc", acc)
model.save(save_path + ".h5")
model.save(save_path)
