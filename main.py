import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_and_prepare_data(filepath):
    data = np.load(filepath, allow_pickle=True)
    train_dict = data['training_input'].item()
    test_dict = data['test_input'].item()

    x_train = np.vstack([train_dict['0'], train_dict['1']])
    y_train = np.concatenate([np.zeros(len(train_dict['0'])), np.ones(len(train_dict['1']))])
    x_test = np.vstack([test_dict['0'], test_dict['1']])
    y_test = np.concatenate([np.zeros(len(test_dict['0'])), np.ones(len(test_dict['1']))])

    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    x_train = np.pi * (x_train - x_min) / (x_max - x_min + 1e-7)
    x_test = np.pi * (x_test - x_min) / (x_max - x_min + 1e-7)

    return x_train, y_train, x_test, y_test

def create_quantum_model(n_qubits):
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuit = cirq.Circuit()
    params = sympy.symbols(f'theta0:{n_qubits * 2}')

    for i in range(n_qubits):
        circuit.append(cirq.ry(params[i])(qubits[i]))

    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

    for i in range(n_qubits):
        circuit.append(cirq.rz(params[i + n_qubits])(qubits[i]))

    readout = [cirq.Z(qubits[0])]
    return circuit, readout

def convert_to_tfq(data):
    qubits = cirq.GridQubit.rect(1, data.shape[1])
    circuits = []
    for val in data:
        c = cirq.Circuit()
        for i, v in enumerate(val):
            c.append(cirq.rx(v)(qubits[i]))
        circuits.append(c)
    return tfq.convert_to_tensor(circuits)

# Execution
x_train, y_train, x_test, y_test = load_and_prepare_data('data/QIS_EXAM_200Events (1).npz')
n_qubits = x_train.shape[1]
model_circuit, model_readout = create_quantum_model(n_qubits)

x_train_tfq = convert_to_tfq(x_train)
x_test_tfq = convert_to_tfq(x_test)

inputs = tf.keras.Input(shape=(), dtype=tf.string)
pqc = tfq.layers.PQC(model_circuit, model_readout)(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(pqc)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
)

model.fit(x_train_tfq, y_train, epochs=50, batch_size=8, validation_data=(x_test_tfq, y_test))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
)

model.fit(x_train_tfq, y_train, epochs=20, batch_size=8, validation_data=(x_test_tfq, y_test))

# Plotting
y_pred = model.predict(x_test_tfq)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend(loc="lower right")

ax2.hist(y_pred[y_test == 1], bins=20, alpha=0.5, label='Signal (1)', color='green')
ax2.hist(y_pred[y_test == 0], bins=20, alpha=0.5, label='Background (0)', color='red')
ax2.set_xlabel('Probability')
ax2.set_ylabel('Events')
ax2.set_title('Separation Plot')
ax2.legend()

plt.tight_layout()
plt.savefig('images/separation_results.png')
plt.show()
