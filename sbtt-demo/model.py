import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from sklearn.metrics import r2_score, mean_squared_error
from zig_distribution import zeroInflatedGamma

class SequentialAutoencoder(pl.LightningModule):
    def __init__(self, 
                 input_size=29, # Number of neurons , Bandwidth corresponds to the number of neurons observed at each timestep
                 hidden_size=50, # Number of timesteps
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 rate_conversion_factor=0.05,
                 dropout=0.1,
                 loss_type="input",
                 s_min=0.0):
        super().__init__()
        self.save_hyperparameters()
        # Instantiate bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        # Instantiate linear mapping to initial conditions
        self.ic_linear = nn.Linear(2*hidden_size, hidden_size)
        # Instantiate autonomous GRU decoder
        self.decoder = nn.GRU(
            input_size=1, # Not used
            hidden_size=hidden_size,
            batch_first=True,
        )
        # Instantiate linear readout
        self.readout = nn.Linear(
            in_features=hidden_size,
            out_features=input_size,
        )
        if loss_type == 'zi_gamma':
            self.factors_map_alpha_beta = nn.Linear(
                in_features=hidden_size,
                out_features=2*input_size,
            )

            self.factors_map_q = nn.Linear(
                in_features=hidden_size,
                out_features=input_size,
            )

            self.alpha_beta_non_linearity = nn.Sigmoid()
            self.q_non_linearity = nn.Sigmoid()

        # Instantiate dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Interpolate NaNs with zeros
        x = torch.nan_to_num(x, nan=0.0)
        # Pass data through the model
        _, h_n = self.encoder(x)
        # Combine output from fwd and bwd encoders
        h_n = torch.cat([*h_n], -1)
        # Compute initial condition
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        # Create an empty input tensor
        input_placeholder = torch.zeros_like(x)[:, :, :1]
        # Unroll the decoder
        ic_drop = self.dropout(ic)
        latents, _ = self.decoder(input_placeholder, torch.unsqueeze(ic_drop, 0))
        if self.hparams.loss_type == 'input' or self.hparams.loss_type == 'ground_truth':
            # Map decoder state to logrates
            logrates = self.readout(latents)
            return logrates
        elif self.hparams.loss_type == 'zi_gamma':
            alpha_beta = self.factors_map_alpha_beta(latents)
            alpha_beta_nl = self.alpha_beta_non_linearity(alpha_beta)
            q = self.factors_map_q(latents)
            q_nl = self.q_non_linearity(q)
            return alpha_beta_nl, q_nl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
    
    def zig_loss(self):
        total_loss = self.l2_weight * self.l2_cost + self.kl_weight * self.kl_cost + self.rec_cost_heldin
        pass
    
    def training_step(self, batch, batch_ix):
        x, truth = batch
        # Keep track of location of observed data
        mask = ~torch.isnan(x)
        # Pass data through the model
        if self.hparams.loss_type == 'input' or self.hparams.loss_type == 'ground_truth':
            logrates = self.forward(x)
            # Mask unobserved steps
            x_obs = torch.masked_select(x, mask)
            logrates_obs = torch.masked_select(logrates, mask)
        elif self.hparams.loss_type == 'zi_gamma':
            alpha_beta_nl, q_nl = self.forward(x)
            # Mask unobserved steps
            x_obs = torch.masked_select(x, mask)
            alpha_nl_obs = torch.masked_select(alpha_beta_nl[..., ::2], mask)
            beta_nl_obs = torch.masked_select(alpha_beta_nl[..., 1::2], mask)
            q_nl_obs = torch.masked_select(q_nl, mask)
        
        # Compute Poisson log-likelihood
        if self.hparams.loss_type == "input":
            loss = nn.functional.poisson_nll_loss(logrates_obs, x_obs)
        elif self.hparams.loss_type == "ground_truth":
            truth_obs = torch.masked_select(truth, mask)
            loss = nn.functional.poisson_nll_loss(logrates_obs, truth_obs)
        elif self.hparams.loss_type == 'zi_gamma':
            loss = -torch.mean(zeroInflatedGamma(alpha_nl_obs, beta_nl_obs , q_nl_obs, torch.min(x_obs)).log_prob_ZIG(x_obs))
        # loss = nn.functional.mse_loss(logrates_obs, x_obs) # changed poisson loss to MSE loss
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_nll', loss, on_epoch=True)
        # Compute match to true rates
        truth = truth.detach().cpu().numpy()
        if self.hparams.loss_type == 'input' or self.hparams.loss_type == 'ground_truth':
            rates = torch.exp(logrates).detach().cpu().numpy() 
            rates *= self.hparams.rate_conversion_factor
        elif self.hparams.loss_type == 'zi_gamma':
            rates = q_nl*(alpha_beta_nl[..., ::2]*alpha_beta_nl[..., 1::2] + self.hparams.s_min)
            rates = rates.detach().cpu().numpy()
        truth = np.concatenate([*truth])
        rates = np.concatenate([*rates])
        # r2 = r2_score(truth, rates)
        # mse = np.average(np.square((truth - rates)/truth))
        truth = truth/np.max(truth)
        rates = rates/np.max(rates)
        mse = mean_squared_error(truth, rates)
        self.log('train_mse', mse, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_ix):
        x, truth = batch
        # Keep track of location of observed data
        mask = ~torch.isnan(x)
        # Pass data through the model
        if self.hparams.loss_type == 'input' or self.hparams.loss_type == 'ground_truth':
            logrates = self.forward(x)
            # Mask unobserved steps
            x_obs = torch.masked_select(x, mask)
            logrates_obs = torch.masked_select(logrates, mask)
        elif self.hparams.loss_type == 'zi_gamma':
            alpha_beta_nl, q_nl = self.forward(x)
            # Mask unobserved steps
            x_obs = torch.masked_select(x, mask)
            alpha_nl_obs = torch.masked_select(alpha_beta_nl[..., ::2], mask)
            beta_nl_obs = torch.masked_select(alpha_beta_nl[..., 1::2], mask)
            q_nl_obs = torch.masked_select(q_nl, mask)
        
        # Compute Poisson log-likelihood
        if self.hparams.loss_type == "input":
            loss = nn.functional.poisson_nll_loss(logrates_obs, x_obs)
        elif self.hparams.loss_type == "ground_truth":
            truth_obs = torch.masked_select(truth, mask)
            loss = nn.functional.poisson_nll_loss(logrates_obs, truth_obs)
        elif self.hparams.loss_type == 'zi_gamma':
            loss = -torch.mean(zeroInflatedGamma(alpha_nl_obs, beta_nl_obs , q_nl_obs, torch.min(x_obs)).log_prob_ZIG(x_obs))
        # loss = nn.functional.mse_loss(logrates_obs, x_obs) # changed poisson loss to MSE loss
        self.log('valid_loss', loss, on_epoch=True)
        self.log('valid_nll', loss, on_epoch=True)
        truth = truth.detach().cpu().numpy()
        if self.hparams.loss_type == 'input' or self.hparams.loss_type == 'ground_truth':
            rates = torch.exp(logrates).detach().cpu().numpy() 
            rates *= self.hparams.rate_conversion_factor
        elif self.hparams.loss_type == 'zi_gamma':
            rates = q_nl*(alpha_beta_nl[..., ::2]*alpha_beta_nl[..., 1::2] + self.hparams.s_min)
            rates = rates.detach().cpu().numpy()
        truth = np.concatenate([*truth])
        rates = np.concatenate([*rates])
        # r2 = r2_score(truth, rates)
        # mse = np.average(np.square((truth - rates)/truth))
        # Scale the rates and truth between 0 and 1
        truth = truth/np.max(truth)
        rates = rates/np.max(rates)
        mse = mean_squared_error(truth, rates)
        self.log('valid_mse', mse, on_epoch=True)
        return loss
