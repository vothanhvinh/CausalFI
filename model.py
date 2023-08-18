import numpy as np
import torch
import torch.nn as nn
import torchbnn as bnn
from torchbnn.utils import freeze, unfreeze

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ====================================================================================================================================
# Model P(Z| X, Y, W)
# ====================================================================================================================================

class ModelZ(torch.nn.Module):
  def __init__(self, x, y, w, z, outcome='continuous', hidden_size=50):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.loss_mse = torch.nn.MSELoss(reduction='sum')
    
    self.kl_loss = bnn.BKLLoss(reduction='sum', last_layer_only=False)
    self.x = x#[idx]
    self.w = w#[idx]
    self.y = y#[idx]
    self.z = z#[idx]
    self.r = (~z.isnan())*1.0

    self.model_0 = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+1, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=z.shape[1])
    )

    self.model_1 = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+1, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=z.shape[1])
    )

    # self.model_sd = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1, out_features=1,bias=False)
    self.model_0_sd = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+1, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.Tanh(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=z.shape[1])
    )

    self.model_1_sd = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+1, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.Tanh(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=z.shape[1])
    )

  def forward(self, x, y, w):
    X = torch.concatenate([x,y.reshape(-1,1)],axis=1)
    mu = w.reshape(-1,1)*self.model_1(X) + (1-w.reshape(-1,1))*self.model_0(X)
    # sd = self.model_sd(torch.ones(1)).reshape(-1)
    sd = w.reshape(-1,1)*self.model_1_sd(X) + (1-w.reshape(-1,1))*self.model_0_sd(X)
    return mu, sd

  def loss(self):
    zpred_mu, zpred_logsd = self.forward(self.x, self.y, self.w)
    loglik = torch.sum(0.5*(zpred_mu[self.r==1] - self.z[self.r==1])**2/torch.exp(2*zpred_logsd[self.r==1]) + zpred_logsd[self.r==1])
    # kl = self.kl_loss(self.model_0) + self.kl_loss(self.model_1) + self.kl_loss(self.model_sd)
    kl = self.kl_loss(self.model_0) + self.kl_loss(self.model_1) \
          + self.kl_loss(self.model_0_sd) + self.kl_loss(self.model_1_sd)
    return loglik + 0.01*kl

  def draw_samples(self, x, y, w):
    zpred_mu, zpred_logsd = self.forward(x, y, w)
    return (zpred_mu + torch.exp(zpred_logsd)*torch.randn(zpred_mu.shape)).detach()

def trainZ_FedGrads(train_x, train_w, train_y, train_z, n_sources, source_ranges, outcome='continuous', hidden_size=50, training_iter=200, learning_rate=1e-3,
           display_per_iters=100):

  # Create models
  model_server = ModelZ(x=torch.ones((1,train_x.shape[1])),
                        w=torch.ones((1,1)),
                        y=torch.ones((1,1)),
                        z=torch.ones((1,train_z.shape[1])),
                        outcome=outcome,hidden_size=hidden_size).to(device)

  model_sources = [ModelZ(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                          y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                          z=train_z[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          outcome=outcome,hidden_size=hidden_size).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(training_iter):
    # Compute gradients on each source
    for idx in range(n_sources):
      loss_source = model_sources[idx].loss()
      optimizer_sources[idx].zero_grad()
      loss_source.backward()

      if (i+1)%display_per_iters==0:
        print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, training_iter, loss_source.item()))
    
    # Update gradient on server
    loss_server = model_server.loss()
    optimizer_server.zero_grad()
    loss_server.backward() # The purpuse of this line is to allocate memory for gradient of each parameter, i.e param.grad as below
    for key, param in model_server.named_parameters():
      param.grad.zero_()

    for idx in range(n_sources):
      grad_dict_source = {key:param.grad for key, param in model_sources[idx].named_parameters()} # store gradients to grad_dict_source
      
      for key, param in model_server.named_parameters():
        if (param.grad is not None) and param.requires_grad:
          param.grad += grad_dict_source[key]
    optimizer_server.step()

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources

def trainZ_FedParams(train_x, train_w, train_y, train_z, n_sources, source_ranges, outcome='continuous', training_iter=100, num_agg=100, learning_rate=1e-3,
           display_per_iters=100):

  # Create models
  model_server = ModelZ(x=torch.ones((1,train_x.shape[1])),
                        w=torch.ones((1,1)),
                        y=torch.ones((1,1)),
                        z=torch.ones((1,train_z.shape[1])),
                        outcome=outcome).to(device)

  model_sources = [ModelZ(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                          y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                          z=train_z[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          outcome=outcome).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(num_agg):
    # Compute gradients on each source
    for idx in range(n_sources):
      for j in range(training_iter):
        loss_source = model_sources[idx].loss()
        optimizer_sources[idx].zero_grad()
        loss_source.backward()
        optimizer_sources[idx].step()

        if (j+1)%display_per_iters==0:
          print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, 100, loss_source.item()))

    # Assign all parameters at server to 0 to prepare for aggregation
    for key, param in model_server.named_parameters():
        param.data.zero_()

    # Collect model from all sources and aggregate them
    for idx in range(n_sources):
      data_dict_source = {key:param.data for key, param in model_sources[idx].named_parameters()} # store parameters to data_dict_source
      
      for key, param in model_server.named_parameters():
        param.data += data_dict_source[key]/n_sources

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources

def testZ(model_sources, test_x, test_w, test_y, test_z, test_z_orgi, n_sources, source_ranges, idx_sources_to_test=None):
  
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test

  

  zpred_logit = []
  for idx in idx_lst:
    
    freeze(model_sources[idx])
    pred_tuple = model_sources[idx](test_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                                    test_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                                    test_w[range(source_ranges[idx][0], source_ranges[idx][1])])

    pred,_ = pred_tuple

    pred = pred

    unfreeze(model_sources[idx])
    zpred_logit.append(pred)
  zpred_logit = torch.cat(zpred_logit)

  # idx = ~torch.any(test_z.isnan(),dim=1)
  idx_test = np.concatenate([list(range(source_ranges[idx][0], source_ranges[idx][1])) for idx in idx_lst])

  accur = torch.mean(torch.abs((zpred_logit - test_z_orgi[idx_test])/test_z[idx_test]))#torch.mean((ypred_logit - test_y[idx_test].reshape(-1))**2)

  test_stats = accur.cpu().detach().numpy()
  return test_stats, zpred_logit, test_z_orgi[idx_test]


# ====================================================================================================================================
# Model (Z_{\tilde{r}} | X, Z_r)
# ====================================================================================================================================
class ModelZhat(torch.nn.Module):
  def __init__(self, x, y, w, dim_z, outcome='continuous', hidden_size=50):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.loss_mse = torch.nn.MSELoss(reduction='sum')
    
    self.kl_loss = bnn.BKLLoss(reduction='sum', last_layer_only=False)
    self.bernoulli_half = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))
    self.x = x
    self.y = y
    self.w = w
    # self.model_z = model_z

    self.model = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+dim_z*2, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=dim_z)
    )

    # self.model_sd = nn.Sequential(
    #   bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=10, out_features=5, bias=False),
    # )

    self.model_sd = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+dim_z*2, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.Tanh(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=dim_z)
    )

  def forward(self, x, z_r, r):
    XX = torch.cat((x,z_r,r),axis=1)
    mu = self.model(XX)
    # sd = self.model_sd(torch.ones(1)).reshape(-1)
    sd = self.model_sd(XX)
    # D =  self.model_sd(torch.eye(10)).reshape(-1,5)
    # S = torch.matmul(D, D.T)+torch.eye(10)*1e-6
    return mu, sd

  def loss(self, model_z):
    z = model_z.draw_samples(self.x, self.y, self.w)
    r = self.bernoulli_half.sample(sample_shape=torch.Size([z.shape[0],z.shape[1]]))[:,:,0]
    zpred_mu, zpred_logsd = self.forward(self.x, z*r, r)
    loglik = torch.sum((0.5*(zpred_mu - z)**2/torch.exp(2*zpred_logsd) + zpred_logsd)*(1-r))
    # zpred_mu, S = self.forward(self.x)
    # Sinv = torch.linalg.inv(S)
    # loglik = torch.mean(0.5*torch.sum(((zpred_mu - z).matmul(Sinv))*(zpred_mu - z),axis=1) + 0.5*torch.logdet(S))
    kl = self.kl_loss(self.model) + self.kl_loss(self.model_sd)
    return loglik + 0.01*kl

  def draw_samples(self, x, z, r, N):
    zpred_mu, zpred_logsd = self.forward(x,z,r)
    zmis_samples = []
    for i in range(x.shape[0]):
      # zmis_samples.append(zpred_mu[i].reshape(1,-1))
      zmis_samples.append(zpred_mu[i] + torch.exp(zpred_logsd[i])*torch.randn((N,zpred_mu[i].shape[0])))
    return zmis_samples

  # def draw_samples(self, x, z=None):
  #   if z==None:
  #     zpred_mu, S = self.forward(x)
  #     L = torch.linalg.cholesky(S)
  #     # return zpred_mu + torch.exp(zpred_logsd)*torch.randn(zpred_mu.shape)
  #   else:
  #     zpred_mu, S = self.forward(x)
  #     idx_obs = ~z.isnan()
  #     idx_mis = z.isnan()
  #     zmis_samples_lst = []
  #     for i in range(x.shape[0]):
  #       Sig = S[idx_mis[i]][:,idx_obs[i]].matmul(torch.linalg.inv(S[idx_obs[i]][:,idx_obs[i]]))
  #       mu = zpred_mu[i][idx_mis[i]].reshape(-1,1) + Sig.matmul(z[i][idx_obs[i]].reshape(-1,1) - zpred_mu[i][idx_obs[i]].reshape(-1,1))
  #       S = S[idx_mis[i]][:,idx_mis[i]] - Sig.matmul(S[idx_obs[i]][:,idx_mis[i]])
  #       L = torch.linalg.cholesky(S)
  #       zmis_samples = (mu + L.matmul(torch.randn((mu.shape[0],100)))).T
  #       zmis_samples_lst.append(zmis_samples)
  #     return zmis_samples_lst

def trainZhat_FedGrads(train_x, train_y, train_w, model_z, dim_z, n_sources, source_ranges, outcome='continuous', training_iter=200, learning_rate=1e-3,
           display_per_iters=100):

  # Create models
  model_server = ModelZhat(x=torch.ones((1,train_x.shape[1])),
                           y=torch.ones((1,1)),
                           w=torch.ones((1,1)),
                           dim_z=dim_z,
                           outcome=outcome).to(device)

  model_sources = [ModelZhat(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                             w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                             y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                             dim_z=dim_z,
                             outcome=outcome).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(training_iter):
    # Compute gradients on each source
    for idx in range(n_sources):
      loss_source = model_sources[idx].loss(model_z[idx])
      optimizer_sources[idx].zero_grad()
      loss_source.backward()

      if (i+1)%display_per_iters==0:
        print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, training_iter, loss_source.item()))
    
    # Update gradient on server
    loss_server = model_server.loss(model_z[0])
    optimizer_server.zero_grad()
    loss_server.backward() # The purpuse of this line is to allocate memory for gradient of each parameter, i.e param.grad as below
    for key, param in model_server.named_parameters():
      param.grad.zero_()

    for idx in range(n_sources):
      grad_dict_source = {key:param.grad for key, param in model_sources[idx].named_parameters()} # store gradients to grad_dict_source
      
      for key, param in model_server.named_parameters():
        if (param.grad is not None) and param.requires_grad:
          param.grad += grad_dict_source[key]
    optimizer_server.step()

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources

def trainZhat_FedParams(train_x, train_y, train_w, model_z, dim_z, n_sources, source_ranges, outcome='continuous', training_iter=100, num_agg=100, learning_rate=1e-3,
           display_per_iters=100):

  # Create models
  model_server = ModelZhat(x=torch.ones((1,train_x.shape[1])),
                           y=torch.ones((1,1)),
                           w=torch.ones((1,1)),
                           dim_z=dim_z,
                           outcome=outcome).to(device)

  model_sources = [ModelZhat(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                             w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                             y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                             dim_z=dim_z,
                             outcome=outcome).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(num_agg):
    # Compute gradients on each source
    for idx in range(n_sources):
      for j in range(training_iter):
        loss_source = model_sources[idx].loss(model_z[idx])
        optimizer_sources[idx].zero_grad()
        loss_source.backward()
        optimizer_sources[idx].step()

        if (j+1)%display_per_iters==0:
          print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, 100, loss_source.item()))

    # Assign all parameters at server to 0 to prepare for aggregation
    for key, param in model_server.named_parameters():
        param.data.zero_()

    # Collect model from all sources and aggregate them
    for idx in range(n_sources):
      data_dict_source = {key:param.data for key, param in model_sources[idx].named_parameters()} # store parameters to data_dict_source
      
      for key, param in model_server.named_parameters():
        param.data += data_dict_source[key]/n_sources

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources


# ====================================================================================================================================
# Model P(Y|W,X,Z)
# ====================================================================================================================================
class ModelY(torch.nn.Module):
  def __init__(self, x, y, w, dim_z, outcome='continuous', hidden_size=50):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.loss_mse = torch.nn.MSELoss(reduction='sum')
    
    self.kl_loss = bnn.BKLLoss(reduction='sum', last_layer_only=False)
    self.x = x
    self.w = w
    self.y = y
    self.outcome = outcome

    self.model_0 = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+dim_z, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=1)
    )

    self.model_1 = nn.Sequential(
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+dim_z, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
      nn.ReLU(),
      bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=1)
    )

    if outcome=='continuous':
      # self.model_sd = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1, out_features=1,bias=False)
      self.model_0_sd = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+dim_z, out_features=hidden_size),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
        nn.Tanh(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=1)
      )

      self.model_1_sd = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=x.shape[1]+dim_z, out_features=hidden_size),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=hidden_size),
        nn.Tanh(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=1)
      )

  def forward(self, x, z, w):
    xz = torch.cat((x, z), dim=1)
    if self.outcome=='continuous':
      mu = w.reshape(-1)*self.model_1(xz).reshape(-1) + (1-w.reshape(-1))*self.model_0(xz).reshape(-1)
      # sd = self.model_sd(torch.ones(1)).reshape(-1)
      sd = w.reshape(-1)*self.model_1_sd(xz).reshape(-1) + (1-w.reshape(-1))*self.model_0_sd(xz).reshape(-1)
      return mu, sd
    elif self.outcome=='binary' or self.outcome=='count':
      return w.reshape(-1)*self.model_1(xz).reshape(-1) + (1-w.reshape(-1))*self.model_0(xz).reshape(-1)

  def loss(self, model_z):
    z = model_z.draw_samples(self.x, self.y, self.w)
    if self.outcome=='binary':
      ypred_logit = self.forward(self.x, z, self.w)
      loglik = self.loss_bce(ypred_logit.reshape(-1), self.y.reshape(-1))
      kl = self.kl_loss(self.model_0) + self.kl_loss(self.model_1)
      return loglik + 0.01*kl
    elif self.outcome=='continuous':
      ypred_mu, ypred_logsd = self.forward(self.x, z, self.w)
      loglik = torch.sum(0.5*(ypred_mu.reshape(-1) - self.y.reshape(-1))**2/torch.exp(2*ypred_logsd) + ypred_logsd)
      # kl = self.kl_loss(self.model_0) + self.kl_loss(self.model_1) + self.kl_loss(self.model_sd)
      kl = self.kl_loss(self.model_0) + self.kl_loss(self.model_1) \
            + self.kl_loss(self.model_0_sd) + self.kl_loss(self.model_1_sd)
      return loglik + 0.01*kl
    elif self.outcome=='count':
      ypred_lograte = self.forward(self.x, z, self.w)
      loglik = torch.sum(-self.y.reshape(-1)*ypred_lograte + torch.exp(ypred_lograte))
      kl = self.kl_loss(self.model_0) + self.kl_loss(self.model_1)
      return loglik + 0.01*kl

  def draw_samples(self, x, z, w):
    ypred_mu, ypred_logsd = self.forward(x, z, w)
    return (ypred_mu + torch.exp(ypred_logsd)*torch.randn(ypred_logsd.shape)).detach()

# ====================================================================================================================================
# Combine traning of P(Y|W,X,Z) and and (Z_{\tilde{r}} | X, Z_r)
# ====================================================================================================================================
class ModelZhatY(torch.nn.Module):
  def __init__(self, x, w, y, dim_z, outcome='continuous', hidden_size=50):
    super().__init__()

    self.model_zhat = ModelZhat(x=x, w=w, y=y, dim_z=dim_z, outcome=outcome, hidden_size=hidden_size)
    self.model_y = ModelY(x=x, w=w, y=y, dim_z=dim_z, outcome=outcome, hidden_size=hidden_size)
  def loss(self, model_z):
    return self.model_zhat.loss(model_z) + self.model_y.loss(model_z)
    

def trainY_FedGrads(train_x, train_w, train_y, model_z, dim_z, n_sources, source_ranges, outcome='continuous', hidden_size=50, training_iter=200, learning_rate=1e-3,
           display_per_iters=100):

  # Create models
  model_server = ModelZhatY(x=torch.ones((1,train_x.shape[1])),
                        w=torch.ones((1,1)),
                        y=torch.ones((1,1)),
                        dim_z=dim_z,
                        outcome=outcome, hidden_size=hidden_size).to(device)

  model_sources = [ModelZhatY(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                          y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                          dim_z=dim_z,
                          outcome=outcome,hidden_size=hidden_size).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(training_iter):
    # Compute gradients on each source
    for idx in range(n_sources):
      loss_source = model_sources[idx].loss(model_z[idx])
      optimizer_sources[idx].zero_grad()
      loss_source.backward()

      if (i+1)%display_per_iters==0:
        print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, training_iter, loss_source.item()))
    
    # Update gradient on server
    loss_server = model_server.loss(model_z[0])
    optimizer_server.zero_grad()
    loss_server.backward() # The purpuse of this line is to allocate memory for gradient of each parameter, i.e param.grad as below
    for key, param in model_server.named_parameters():
      param.grad.zero_()

    for idx in range(n_sources):
      grad_dict_source = {key:param.grad for key, param in model_sources[idx].named_parameters()} # store gradients to grad_dict_source
      
      for key, param in model_server.named_parameters():
        if (param.grad is not None) and param.requires_grad:
          param.grad += grad_dict_source[key]
    optimizer_server.step()

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources

def trainY_FedParams(train_x, train_w, train_y, model_z, dim_z, n_sources, source_ranges, outcome='continuous',
                     training_iter=100, num_agg=100, learning_rate=1e-3,
                     display_per_iters=100):

  # Create models
  model_server = ModelY(x=torch.ones((1,train_x.shape[1])),
                        w=torch.ones((1,1)),
                        y=torch.ones((1,1)),
                        dim_z=dim_z,
                        outcome=outcome).to(device)

  model_sources = [ModelY(x=train_x[range(source_ranges[idx][0], source_ranges[idx][1]),:],
                          w=train_w[range(source_ranges[idx][0], source_ranges[idx][1])],
                          y=train_y[range(source_ranges[idx][0], source_ranges[idx][1])],
                          dim_z=dim_z,
                          outcome=outcome).to(device) for idx in range(n_sources)]
  # Create optimizers
  optimizer_server = torch.optim.Adam(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adam(model_sources[idx].parameters(), lr=learning_rate) for idx in range(n_sources)]

  for i in range(num_agg):
    # Compute gradients on each source
    for idx in range(n_sources):
      for j in range(training_iter):
        loss_source = model_sources[idx].loss(model_z[idx])
        optimizer_sources[idx].zero_grad()
        loss_source.backward()
        optimizer_sources[idx].step()

        if (j+1)%display_per_iters==0:
          print('Source %d, Iter %d/%d - Loss: %.3f' % (idx, i + 1, 100, loss_source.item()))

    # Assign all parameters at server to 0 to prepare for aggregation
    for key, param in model_server.named_parameters():
        param.data.zero_()

    # Collect model from all sources and aggregate them
    for idx in range(n_sources):
      data_dict_source = {key:param.data for key, param in model_sources[idx].named_parameters()} # store parameters to data_dict_source
      
      for key, param in model_server.named_parameters():
        param.data += data_dict_source[key]/n_sources

    # Get new model from server
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}

    # Update new model to all sources
    for idx in range(n_sources):
      for key, param in model_sources[idx].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

  return model_server, model_sources


# # Learn causal effects

def est_effects(model_server_zhat, model_server_y, test_x, test_z, test_w, test_y, n_sources, source_ranges, idx_sources_to_test=None):
  
  if idx_sources_to_test==None:
    idx_lst = range(n_sources)
  else:
    idx_lst = idx_sources_to_test
  outcome = model_server_y.outcome
  dim_z = test_z.shape[1]

  ate_lst = []
  y0_hat_lst = []
  y1_hat_lst = []
  for idx in idx_lst:
    
    freeze(model_server_zhat)

    x = test_x[range(source_ranges[idx][0], source_ranges[idx][1]),:]
    z = test_z[range(source_ranges[idx][0], source_ranges[idx][1]),:]
    r = (~z.isnan())*1.0
    z[r==0] = 0

    zmis_samples = model_server_zhat.draw_samples(x, z*r, r, 100)
    unfreeze(model_server_zhat)

    r = r.repeat([1,100]).reshape(-1,dim_z)
    z = z.repeat([1,100]).reshape(-1,dim_z)
    zmis = torch.cat(zmis_samples)
    z[r==0] = zmis[r==0]

    dim_x = x.shape[1]
    y0_samples = model_server_y.draw_samples(x.repeat([1,100]).reshape(-1,dim_x),
                                                   z, torch.zeros((z.shape[0],1)))
  
    y1_samples = model_server_y.draw_samples(x.repeat([1,100]).reshape(-1,dim_x),
                                                   z, torch.ones((z.shape[0],1)))
    
    # print(y0_samples.shape)
    y0_hat = torch.mean(y0_samples.reshape(-1,100),axis=1)
    y1_hat = torch.mean(y1_samples.reshape(-1,100),axis=1)
    ate_lst.append(y1_samples - y0_samples)
    y0_hat_lst.append(y0_hat)
    y1_hat_lst.append(y1_hat)
  
  return ate_lst, torch.cat(y0_hat_lst), torch.cat(y1_hat_lst)

def pred_y0y1(model_server_zhat, model_server_y, test_x, test_z, test_w, test_y, n_sources, source_ranges_test, N_samples=100, idx_sources_to_test=[0]):
  # ate_lst = []
  y0_hat_lst = []
  y1_hat_lst = []
  for i in range(N_samples):
    _, y0_hat, y1_hat = est_effects(model_server_zhat=model_server_zhat, model_server_y=model_server_y, 
                                          test_x=test_x, test_z=test_z,
                                          test_w=test_w, test_y=test_y, n_sources=n_sources, 
                                          source_ranges=source_ranges_test, idx_sources_to_test=idx_sources_to_test)
    # ate_lst.append(torch.mean(y1_hat - y0_hat))
    y0_hat_lst.append(y0_hat)
    y1_hat_lst.append(y1_hat)
  return torch.mean(torch.stack(y0_hat_lst),axis=0), torch.mean(torch.stack(y1_hat_lst),axis=0)

