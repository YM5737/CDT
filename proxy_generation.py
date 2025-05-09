import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# torch.manual_seed(10)

class UniformityLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(UniformityLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        similarity = embeddings.matmul(embeddings.T)
        scaled_similarity = similarity.div(self.temperature).exp()
        return scaled_similarity.sum(dim=-1).log().mean()


num_proxy = 8
embedding_dim = 128
print("Number of proxys =", num_proxy)
print("Embedding dimension =", embedding_dim)
uniformity_criterion = UniformityLoss()
embeddings = Variable(torch.randn(num_proxy, embedding_dim).float(), requires_grad=True)

model_optimizer = optim.SGD([embeddings], lr=1e-3)
min_loss = 100
optimal_embeddings = None

num_iterations = 2000
for iter in range(num_iterations):
    norm_embeddings = F.normalize(embeddings, dim=1)
    current_loss = uniformity_criterion(norm_embeddings)

    if iter % 100 == 0:
        print(f"Iteration {iter}: Loss = {current_loss.item():.6f}")

    if current_loss.item() < min_loss:
        min_loss = current_loss.item()
        optimal_embeddings = norm_embeddings

    model_optimizer.zero_grad()
    current_loss.backward()
    model_optimizer.step()


proxy_indices = np.arange(0, num_proxy)
final_result = np.concatenate((optimal_embeddings.detach().numpy(), proxy_indices[:, np.newaxis]), axis=1)
output_filename = f'uniform_embeddings_{num_proxy}_{embedding_dim}.npy'
np.save(output_filename, final_result)

loaded_embeddings = np.load(output_filename)
print("Embeddings shape:", loaded_embeddings.shape)
print("Optimal loss =", uniformity_criterion(torch.tensor(loaded_embeddings[:, :-1])).item())