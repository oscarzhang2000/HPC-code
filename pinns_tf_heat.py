
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the PINNs model
class PINNsModel(tf.keras.Model):
    def __init__(self):
        super(PINNsModel, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(20, activation='tanh') for _ in range(8)]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# Define the loss function for PINNs
@tf.function
def loss_fn(model, x_f, t_f, x_u, t_u, u):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_f)
        tape.watch(t_f)
        inputs_f = tf.concat([x_f, t_f], axis=1)
        u_pred_f = model(inputs_f)

        u_x = tape.gradient(u_pred_f, x_f)
        u_t = tape.gradient(u_pred_f, t_f)
        u_xx = tape.gradient(u_x, x_f)

        f = u_t - k * u_xx

    inputs_u = tf.concat([x_u, t_u], axis=1)
    u_pred_u = model(inputs_u)

    mse_u = tf.reduce_mean(tf.square(u - u_pred_u))
    mse_f = tf.reduce_mean(tf.square(f))

    return mse_u + mse_f

N_i = 100
N_b = 50
N_f = 10000

# Generate training data
# Domain boundaries
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 0.25

# Initial condition
x_ic = np.linspace(x_min, x_max, N_i)
t_ic = np.zeros_like(x_ic)
u_ic = np.sin(np.pi * x_ic)

# Boundary condition
x_bc = np.concatenate([np.full(50, x_min), np.full(50, x_max)])
t_bc = np.concatenate([np.linspace(t_min, t_max, 50), np.linspace(t_min, t_max, 50)])
u_bc = np.zeros_like(x_bc)

# Combine initial and boundary conditions
x_u = np.concatenate([x_ic, x_bc])
t_u = np.concatenate([t_ic, t_bc])
u = np.concatenate([u_ic, u_bc])

# Collocation points for the PDE
x_f = np.random.uniform(x_min, x_max, N_f)
t_f = np.random.uniform(t_min, t_max, N_f)

# Convert to TensorFlow tensors
x_u = tf.convert_to_tensor(x_u[:, None], dtype=tf.float32)
t_u = tf.convert_to_tensor(t_u[:, None], dtype=tf.float32)
u = tf.convert_to_tensor(u[:, None], dtype=tf.float32)
x_f = tf.convert_to_tensor(x_f[:, None], dtype=tf.float32)
t_f = tf.convert_to_tensor(t_f[:, None], dtype=tf.float32)

# Define the model and optimizer
model = PINNsModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 10000
k = 1.0

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_f, t_f, x_u, t_u, u)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Evaluate and plot the results
x_test = np.linspace(x_min, x_max, 100)
t_test = np.linspace(t_min, t_max, 100)
X, T = np.meshgrid(x_test, t_test)
x_test = X.flatten()[:, None]
t_test = T.flatten()[:, None]

u_pred = model(tf.concat([x_test, t_test], axis=1))
U_pred = u_pred.numpy().reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.contourf(X, T, U_pred, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title("Solution of 1D Heat Equation using PINNs")
plt.show()
