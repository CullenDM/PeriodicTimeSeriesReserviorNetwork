import torch
import torch.nn as nn
import torch.optim as optim

def fdt_approximation(size, alpha, c, k):
    A = torch.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                A[i, j] = 1 - k[i]
            elif abs(i - j) == 1:
                A[i, j] = c[i] * c[j]
    I = torch.eye(size)
    W = (A - (1 - alpha) * I) / alpha
    return W

class ReservoirLayer(nn.Module):
    def __init__(self, input_size, primary_size, intermediate_size, alpha=0.03):
        super(ReservoirLayer, self).__init__()
        self.input_size = input_size
        self.primary_size = primary_size
        self.intermediate_size = intermediate_size
        self.reservoir_size = primary_size + intermediate_size
        self.alpha = alpha
        
        self.W_in = torch.randn(primary_size, input_size) * 0.1
        
        # Initialize `c` with a gradient decreasing from 300 to 50
        self.c = torch.linspace(300, 50, self.reservoir_size) - torch.rand(self.reservoir_size) * 0.8
        
        # Initialize `k` to zero
        self.k = torch.zeros(self.reservoir_size)
        
        # Create weight matrix `W` using FDTD approximation
        self.W_p = fdt_approximation(primary_size, alpha, self.c[:primary_size], self.k[:primary_size])
        self.W_o = fdt_approximation(intermediate_size, alpha, self.c[primary_size:], self.k[primary_size:])
        
        # Combined weight matrix for interaction
        self.W_interaction = torch.randn(self.reservoir_size, self.reservoir_size) * 0.1
        
        # Random bias initialization
        self.bias_p = torch.randn(primary_size) * 0.1
        self.bias_o = torch.randn(intermediate_size) * 0.1

    def forward(self, x, hidden_p, hidden_o):
        h_p = torch.tanh(torch.matmul(self.W_in, x) + torch.matmul(self.W_p, hidden_p) + self.bias_p)
        h_o = torch.tanh(torch.matmul(self.W_o, hidden_o) + self.bias_o)
        
        # Concatenate hidden states for interaction
        hidden_combined = torch.cat((hidden_p, hidden_o), dim=0)
        interaction = torch.tanh(torch.matmul(self.W_interaction, hidden_combined))
        
        # Ensure interaction has the expected size
        assert interaction.size(0) == self.primary_size + self.intermediate_size, \
            f"Interaction size mismatch: expected {self.primary_size + self.intermediate_size}, got {interaction.size(0)}"
        
        h_p = h_p + interaction[:self.primary_size]
        h_o = h_o + interaction[self.primary_size:]
        
        hidden_p = (1 - self.alpha) * hidden_p + self.alpha * h_p
        hidden_o = (1 - self.alpha) * hidden_o + self.alpha * h_o
        return hidden_p, hidden_o

    def adjust_c(self, early, late, threshold_sum=10.0, delta_c=0.02):
        delta_sum = 0
        if early - late < threshold_sum:
            delta_sum += delta_c
            self.c += delta_c
        else:
            delta_sum -= delta_c
            self.c -= delta_c
        self.c = torch.clamp(self.c, min=0.0)
        self.W_p = fdt_approximation(self.primary_size, self.alpha, self.c[:self.primary_size], self.k[:self.primary_size])
        self.W_o = fdt_approximation(self.intermediate_size, self.alpha, self.c[self.primary_size:], self.k[self.primary_size:])

    def adjust_k(self, mse, significant_neurons=40, delta_k=0.002):
        top_k_indices = torch.topk(mse, significant_neurons, largest=False).indices
        bottom_k_indices = torch.topk(mse, significant_neurons, largest=True).indices
        self.k[top_k_indices] -= delta_k
        self.k[bottom_k_indices] += delta_k
        self.k = torch.clamp(self.k, min=0.0)
        self.W_p = fdt_approximation(self.primary_size, self.alpha, self.c[:self.primary_size], self.k[:self.primary_size])
        self.W_o = fdt_approximation(self.intermediate_size, self.alpha, self.c[self.primary_size:], self.k[self.primary_size:])

    def ds_mechanism(self, hidden_p, hidden_o, target, readout, threshold=0.01, delta_k=0.002, significant_neurons=40):
        combined_hidden = torch.cat((hidden_p, hidden_o), dim=0)
        masked_outputs = combined_hidden.clone()
        mse_list = []

        # Compute MSE for each neuron
        for i in range(self.reservoir_size):
            masked_outputs[i] = 0
            output = readout(masked_outputs)
            mse = nn.MSELoss()(output, target)
            mse_list.append(mse.item())
            masked_outputs[i] = combined_hidden[i]  # Restore the original value

        mse_tensor = torch.tensor(mse_list)
        
        # Apply threshold to filter significant neurons
        significant_indices = mse_tensor > threshold
        mse_tensor_filtered = mse_tensor[significant_indices]

        # Ensure that we do not select more neurons than are available
        num_significant_neurons = len(mse_tensor_filtered)
        num_top_k = min(significant_neurons, num_significant_neurons)
        
        if num_top_k == 0:
            # No significant neurons to adjust
            return
        
        # Select top and bottom significant neurons
        top_k_indices = torch.topk(mse_tensor_filtered, num_top_k, largest=False).indices
        bottom_k_indices = torch.topk(mse_tensor_filtered, num_top_k, largest=True).indices

        # Adjust k values for significant neurons
        self.k[significant_indices][top_k_indices] -= delta_k
        self.k[significant_indices][bottom_k_indices] += delta_k
        self.k = torch.clamp(self.k, min=0.0)

        # Update weight matrices
        self.W_p = fdt_approximation(self.primary_size, self.alpha, self.c[:self.primary_size], self.k[:self.primary_size])
        self.W_o = fdt_approximation(self.intermediate_size, self.alpha, self.c[self.primary_size:], self.k[self.primary_size:])


class ReservoirComputingModel(nn.Module):
    def __init__(self, input_size, primary_size, intermediate_size, output_size, alpha=0.03):
        super(ReservoirComputingModel, self).__init__()
        self.reservoir = ReservoirLayer(input_size, primary_size, intermediate_size, alpha)
        self.readout = nn.Linear(primary_size + intermediate_size, output_size)

    def forward(self, x, hidden_p, hidden_o):
        hidden_p, hidden_o = self.reservoir(x, hidden_p, hidden_o)
        combined_hidden = torch.cat((hidden_p, hidden_o), dim=0)
        output = self.readout(combined_hidden)
        return output, hidden_p, hidden_o

def moving_average_and_softmax(signal, tau=200):
    signal_mean = signal.mean()
    exp_signal = torch.exp(signal - signal_mean)
    exp_avg_signal = torch.cumsum(exp_signal * torch.exp(torch.arange(len(signal)) / tau), dim=0)
    softmax_signal = torch.log(exp_avg_signal)
    normalized_signal = (signal - signal_mean) / softmax_signal
    return normalized_signal

def calculate_synchronization_loss(model, pred, target, update_step, threshold_sum=10.0, delta_c=0.02):
    pred_norm = moving_average_and_softmax(pred)
    target_norm = moving_average_and_softmax(target)
    
    Iearly, Ilate = 0, 0
    delta_sum = 0
    early, late = 0, 0
    
    for t in range(1, len(pred_norm)):
        if target_norm[t] > max(pred_norm[t], 0):
            if target_norm[t] - target_norm[t-1] > 0 and pred_norm[t] - pred_norm[t-1] < 0:
                Iearly += 1
            elif target_norm[t] - target_norm[t-1] < 0 and pred_norm[t] - pred_norm[t-1] > 0:
                Ilate += 1
        early += delta_c * Iearly
        late += delta_c * Ilate

        if t % update_step == 0:  # Integrate update_step for periodic adjustments
            if delta_sum < threshold_sum:
                if early - late < threshold_sum:
                    delta_sum += delta_c
                    c_factor = 1 + delta_c
                else:
                    delta_sum -= delta_c
                    c_factor = 1 - delta_c
            else:
                c_factor = 1

            # Adjust c based on synchronization loss
            model.reservoir.c *= c_factor
            model.reservoir.c = torch.clamp(model.reservoir.c, min=0.0)
            model.reservoir.W_p = fdt_approximation(model.reservoir.primary_size, model.reservoir.alpha, model.reservoir.c[:model.reservoir.primary_size], model.reservoir.k[:model.reservoir.primary_size])
            model.reservoir.W_o = fdt_approximation(model.reservoir.intermediate_size, model.reservoir.alpha, model.reservoir.c[model.reservoir.primary_size:], model.reservoir.k[model.reservoir.primary_size:])
    
    return model

def train_model(model, data_loader, criterion, optimizer, num_epochs=20, update_step=200):
    for epoch in range(num_epochs):
        print(epoch)
        for inputs, targets in data_loader:
            hidden_p = torch.zeros(model.reservoir.primary_size)
            hidden_o = torch.zeros(model.reservoir.intermediate_size)

            # Ensure hidden states are correctly initialized
            assert hidden_p.size(0) == model.reservoir.primary_size, \
                f"Hidden_p size mismatch: expected {model.reservoir.primary_size}, got {hidden_p.size(0)}"
            assert hidden_o.size(0) == model.reservoir.intermediate_size, \
                f"Hidden_o size mismatch: expected {model.reservoir.intermediate_size}, got {hidden_o.size(0)}"

            optimizer.zero_grad()
            outputs, hidden_p, hidden_o = model(inputs, hidden_p, hidden_o)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Synchronize the reservoir
            model = calculate_synchronization_loss(model, outputs, targets, update_step)
            # Adjust k using DS mechanism
            model.reservoir.ds_mechanism(hidden_p, hidden_o, targets, model.readout)
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example usage
input_size = 1
primary_size = 800
intermediate_size = 800
output_size = 1

model = ReservoirComputingModel(input_size, primary_size, intermediate_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data loader for example purposes
data_loader = [(torch.randn(input_size), torch.randn(output_size)) for _ in range(100)]

train_model(model, data_loader, criterion, optimizer)
