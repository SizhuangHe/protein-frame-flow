import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_GPT2(nn.Module):
    def __init__(self, base_model, emb_dim, z_dim,):
        super().__init__()
        # Encoder (GPT2)
        # self.mlp = mlp
        self.llm_model = base_model
    
        
        self.hid2mu = nn.Linear(emb_dim, z_dim)
        self.hid2sigma = nn.Linear(emb_dim, z_dim)

    def _encode(self, input_embeddings, attention_mask=None):
        # input_embeddings = self.mlp(input_ids)
        outputs = self.llm_model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        
        mu = self.hid2mu(last_hidden_states)
        log_sigma = self.hid2sigma(last_hidden_states)

        return mu, log_sigma
    
    def _reparametrize(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def forward(self, input_embeddings, attention_mask=None):
        nan_params = self._check_for_nans()
        print("NaN in model parameters: {}".format(nan_params))
        mu, log_sigma = self._encode(input_embeddings, attention_mask)
        
        z_sampled = self._reparametrize(mu, log_sigma)
        
        return {
            "z_sampled": z_sampled,
            "mu": mu,
            "log_sigma": log_sigma
        }
    
    def _check_for_nans(self):
        nan_params = []
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        return nan_params
    
    def generate_next_tokens(self, input_ids, temperature=1, attention_mask=None):
        assert not temperature < 0, "Temperature should be non-negative!"
        mu, log_sigma = self._encode(input_ids, attention_mask)
        # higher temperature -> larger sigma -> more stochastic, temperature=0 -> sigma=0 -> deterministic
        sigma = torch.sqrt(torch.tensor(temperature)) * torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        z_sampled = mu + sigma * epsilon
        # next_tokens = self._decode(z_sampled)
        return z_sampled[:, -1, :].unsqueeze(1)

    def generate(self, input_ids, temperature=1, max_length=100):
        nan_params = self._check_for_nans()
        print("NaN in model parameters: {}".format(nan_params))
        output_sequences = input_ids
        while output_sequences.shape[1] < max_length:
            next_tokens = self.generate_next_tokens(output_sequences, temperature)
            output_sequences = torch.cat([output_sequences, next_tokens], dim=1)
        return output_sequences


