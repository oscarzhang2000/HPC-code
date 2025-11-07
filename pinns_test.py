import torch, platform
print("CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device('cpu')
)
print("Using device:", device)

# For readability: disable warnings
import warnings
warnings.filterwarnings('ignore')
import os
# import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import scipy.sparse as sp
import scipy.sparse.linalg as la
from pyDOE import lhs
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
k = 1
N_i = 101
N_b = 51
N_u = N_i + 2 * N_b  # number of data points
N_f = 3000 # Collocation points
## Generate collocation points using Latin Hypercube sampling within the bounds of the spatio-temporal coordinates
# Generate N_f x,t coordinates within range of upper and lower bounds
# 1) Bounds (x_min, t_min) and (x_max, t_max)
lb = np.array([0.0, 0.0], dtype=np.float32)
ub = np.array([1.0, 0.25], dtype=np.float32)
U = lhs(2, samples=N_f)   # shape (N_f, 2)
X_f_train = lb + (ub-lb) * U # the 2 denotes the number of coordinates we have - x,t

## In addition, we add the X_u_train co-ordinates from the boundaries to the X_f coordinate set
#X_f_train = np.vstack((X_f_train, X_u_train)) # stack up all training x,t coordinates for u and f


# Generate training data
# Domain boundaries
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 0.25

# --- Initial condition: u(x,0) = sin(pi x) ---
x_ic = np.linspace(x_min, x_max, N_i)
t_ic = np.zeros_like(x_ic)
u_ic = np.sin(np.pi * x_ic)

# --- Boundary conditions: u(0,t) = 0, u(1,t) = 0 (example Dirichlet) ---
t_line = np.linspace(t_min, t_max, N_b)
x_bc_left  = np.full(N_b, x_min)
x_bc_right = np.full(N_b, x_max)
u_bc_left  = np.zeros(N_b)
u_bc_right = np.zeros(N_b)

# --- Stack IC and BC *row-wise* (NO meshgrid here) ---
x_u = np.concatenate([x_ic,       x_bc_left,   x_bc_right])[:, None]   # (N_i+2N_b, 1)
t_u = np.concatenate([t_ic,       t_line,      t_line     ])[:, None]  # (N_i+2N_b, 1)
u   = np.concatenate([u_ic,       u_bc_left,   u_bc_right ])[:, None]  # (N_i+2N_b, 1)

X_u_train = np.hstack([x_u, t_u])   # shape (N_u, 2)  with columns [x, t]
u_train   = u                       # shape (N_u, 1)

## Make a plot to show the distribution of training data
plt.scatter(X_f_train[:,1], X_f_train[:,0], marker='x', color='red',alpha=0.8)
plt.scatter(X_u_train[:,1], X_u_train[:,0], marker='x', color='black')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Data points and collocation points')
plt.legend(['Collocation Points', 'Data Points'])
plt.show()

def net_u(x, t, model):
    X = torch.cat([x, t], dim=1)  # If x and t are each shape (N, 1), then X becomes (N, 2).
    u = model(X)
    return u

def net_f(x, t, model):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = net_u(x, t, model)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = u_t - model.k * u_xx
    return f

class XavierInit(nn.Module):
    def __init__(self, size):
        super(XavierInit, self).__init__()
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = torch.sqrt(torch.tensor(2.0 / (in_dim + out_dim)))
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * xavier_stddev)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

def initialize_NN(layers):
    weights = nn.ModuleList()
    num_layers = len(layers)
    for l in range(num_layers - 1):
        layer = XavierInit(size=[layers[l], layers[l + 1]])
        weights.append(layer)
    return weights

class NeuralNet(nn.Module):
    def __init__(self, layers, lb, ub,k_init):
        super(NeuralNet, self).__init__()
        self.weights = initialize_NN(layers)
        # make lb/ub move with .to(device)
        self.register_buffer('lb', torch.as_tensor(lb, dtype=torch.float32))     # <<< CHANGED >>>
        self.register_buffer('ub', torch.as_tensor(ub, dtype=torch.float32))     # <<< CHANGED >>>
        self.register_buffer('k', torch.tensor(k_init, dtype=torch.float32))     # <<< CHANGED >>>


    def forward(self, X):
        X = X.float()                                                            # <<< CHANGED >>>
        lb = self.lb.to(X.device)                                                # <<< CHANGED >>>
        ub = self.ub.to(X.device)                                                # <<< CHANGED >>>
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(len(self.weights) - 1):
            H = torch.tanh(self.weights[l](H.float()))
        Y = self.weights[-1](H)
        return Y

def train(nIter, X, u, X_f,  model, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # use the model's device
    dev = next(model.parameters()).device                                        # <<< CHANGED >>>

    start_time = time.time()
    x = X[:,0:1]
    t = X[:,1:2]
    # Collocation points (f points)
    x_f = X_f[:, 0:1]
    t_f = X_f[:, 1:2]

    # create tensors ON THE SAME DEVICE
    x_tf   = torch.tensor(x,   dtype=torch.float32, device=dev, requires_grad=True)   # <<< CHANGED >>>
    t_tf   = torch.tensor(t,   dtype=torch.float32, device=dev, requires_grad=True)   # <<< CHANGED >>>
    u_tf   = torch.tensor(u,   dtype=torch.float32, device=dev)                       # <<< CHANGED >>>
    x_f_tf = torch.tensor(x_f, dtype=torch.float32, device=dev, requires_grad=True)   # <<< CHANGED >>>
    t_f_tf = torch.tensor(t_f, dtype=torch.float32, device=dev, requires_grad=True)   # <<< CHANGED >>>
    loss_values = []
    for it in range(nIter):
        optimizer.zero_grad()
        # Compute predictions for training data (u)
        u_pred = net_u(x_tf, t_tf, model )
        # Compute PDE residual at collocation points
        f_pred = net_f(x_f_tf, t_f_tf, model )

        loss_PDE  = criterion(f_pred, torch.zeros_like(f_pred))
        loss_data = criterion(u_tf, u_pred)
        loss = loss_PDE + 5*loss_data

        loss.backward()
        optimizer.step()
        # Print
        if it % 1000 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, Time: %.2f' %
                          (it, loss.item(), elapsed))
            start_time = time.time()
        loss_values.append(loss.item())

    return loss_values

#initialise model
model = NeuralNet(layers, lb, ub,k).to(device).float()

# Training
Train_iterations=30000

# training data
x = X_u_train[:,0:1]
t = X_u_train[:,1:2]

# collocation
x_f_train = X_f_train[:, 0:1]  # x values of the collocation points
t_f_train = X_f_train[:, 1:2]  # t values of the collocation points

# Convert to tensors and ensure they require gradients
x_f_t = torch.tensor(x_f_train, dtype=torch.float32, device=device, requires_grad=True)  # <<< CHANGED >>>
t_f_t = torch.tensor(t_f_train, dtype=torch.float32, device=device, requires_grad=True)  # <<< CHANGED >>>

x_tf = torch.tensor(x,       dtype=torch.float32, device=device, requires_grad=True)     # <<< CHANGED >>>
t_tf = torch.tensor(t,       dtype=torch.float32, device=device, requires_grad=True)     # <<< CHANGED >>>
u_tf = torch.tensor(u_train, dtype=torch.float32, device=device, requires_grad=True)     # <<< CHANGED >>>

model = model.float()
x_tf = x_tf.float()
t_tf = t_tf.float()
u_tf = u_tf.float()

loss_values = train(Train_iterations, X_u_train, u_train, X_f_train, model, learning_rate=1e-3)


# Plot the loss values
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training Loss as a Function of Epochs')
plt.show()

# --- 1) Make the test grid  ---
x_test = np.linspace(x_min, x_max, 100)
t_test = np.linspace(t_min, t_max, 100)
X, T = np.meshgrid(x_test, t_test)            # shapes: (nt, nx)
x_flat = X.flatten()[:, None]                  # (nt*nx, 1)
t_flat = T.flatten()[:, None]                  # (nt*nx, 1)

# --- 2) Build a single input tensor [x, t] on the right device ---
X_star = np.hstack([x_flat, t_flat]).astype(np.float32)  # (N, 2)
X_star_t = torch.from_numpy(X_star).to(device)

# --- 3) Predict with the trained model (no gradients needed) ---
model.eval()
with torch.no_grad():
    u_pred_t = model(X_star_t)                 # (N, 1)

# --- 4) Back to NumPy and reshape to the grid ---
U_pred = u_pred_t.squeeze(1).cpu().numpy().reshape(X.shape)  # (nt, nx)

# --- 5) Plot ---
plt.figure(figsize=(8, 6))
cf = plt.contourf(X, T, U_pred, levels=50, cmap='viridis')
plt.colorbar(cf, label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title("Solution of 1D Heat Equation using PINNs")
plt.tight_layout()
plt.show()



