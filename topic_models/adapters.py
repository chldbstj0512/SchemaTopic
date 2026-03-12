from argparse import Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .ECRTM.models.ECRTM import ECRTM as LegacyECRTM
from .WeTe.Utils import cluster_kmeans
from .WeTe.model import Infer_Net
from .embedded_topic_model.models.model import Model


class LossBundle:
    def __init__(self, reconstruction_loss, regularization_losses=None):
        self.reconstruction_loss = reconstruction_loss
        self.regularization_losses = regularization_losses or {}


class BaseTopicModel(nn.Module):
    def build_optimizer(self, args):
        raise NotImplementedError

    def compute_losses(
        self,
        bows,
        normalized_bows,
        theta=None,
        aggregate=True,
    ):
        raise NotImplementedError

    def get_document_topic_distribution(
        self,
        normalized_bows,
    ):
        raise NotImplementedError

    def get_topic_word_distribution(self):
        raise NotImplementedError


def _build_optimizer(parameters, args):
    optimizer_name = getattr(args, "optimizer", "adam").strip().lower()
    learning_rate = getattr(args, "lr", 0.005)
    weight_decay = getattr(args, "wdecay", 0.0)

    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adagrad":
        return optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adadelta":
        return optim.Adadelta(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "rmsprop":
        return optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "asgd":
        return optim.ASGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            t0=0,
            lambd=0.0,
        )
    return optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)


class ETM(BaseTopicModel):
    expects_normalized_bows = True

    def __init__(
        self,
        num_topics,
        vocab_size,
        t_hidden_size,
        rho_size,
        emsize,
        theta_act,
        embeddings=None,
        train_embeddings=False,
        enc_drop=0.0,
        **_,
    ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(
            device=device,
            num_topics=num_topics,
            vocab_size=vocab_size,
            t_hidden_size=t_hidden_size,
            rho_size=rho_size,
            emsize=emsize,
            theta_act=theta_act,
            embeddings=embeddings,
            train_embeddings=train_embeddings,
            enc_drop=enc_drop,
        )

    def build_optimizer(self, args):
        return _build_optimizer(self.parameters(), args)

    def compute_losses(self, bows, normalized_bows, theta=None, aggregate=True):
        reconstruction_loss, kld_theta, _ = self.model(
            bows,
            normalized_bows,
            theta=theta,
            aggregate=aggregate,
        )
        regularization_losses = {}
        if kld_theta is not None:
            regularization_losses["kld_theta"] = kld_theta
        return LossBundle(
            reconstruction_loss=reconstruction_loss,
            regularization_losses=regularization_losses,
        )

    def get_document_topic_distribution(self, normalized_bows):
        return self.model.get_theta(normalized_bows)

    def get_topic_word_distribution(self):
        return self.model.get_beta()


class ECRTM(BaseTopicModel):
    expects_normalized_bows = False

    def __init__(
        self,
        num_topics,
        vocab_size,
        t_hidden_size,
        rho_size,
        emsize,
        theta_act,
        embeddings=None,
        train_embeddings=False,
        enc_drop=0.0,
        beta_temp=0.2,
        sinkhorn_alpha=20.0,
        OT_max_iter=1000,
        weight_loss_ECR=100.0,
        **_,
    ):
        super().__init__()
        del rho_size, emsize, theta_act, train_embeddings

        if embeddings is None:
            raise ValueError("ECRTM requires pretrained embeddings.")

        if torch.is_tensor(embeddings):
            word_embeddings = embeddings.detach().cpu().numpy()
        else:
            word_embeddings = embeddings

        legacy_args = Namespace(
            beta_temp=beta_temp,
            sinkhorn_alpha=sinkhorn_alpha,
            OT_max_iter=OT_max_iter,
            weight_loss_ECR=weight_loss_ECR,
            vocab_size=vocab_size,
            n_topic=num_topics,
            word_embeddings=word_embeddings,
            dropout=enc_drop,
            en1_units=t_hidden_size,
        )
        self.model = LegacyECRTM(legacy_args)

    def build_optimizer(self, args):
        return _build_optimizer(self.parameters(), args)

    def compute_losses(self, bows, normalized_bows, theta=None, aggregate=True):
        del normalized_bows, theta
        theta_values, kld_theta = self.model.encode(bows)
        beta = self.model.get_beta()
        logits = torch.matmul(theta_values, beta)
        recon = F.softmax(self.model.decoder_bn(logits), dim=-1)
        reconstruction_loss = -(bows * torch.log(recon + 1e-6)).sum(dim=1)
        if aggregate:
            reconstruction_loss = reconstruction_loss.mean()

        return LossBundle(
            reconstruction_loss=reconstruction_loss,
            regularization_losses={
                "kld_theta": kld_theta,
                "ecr_loss": self.model.get_loss_ECR(),
            },
        )

    def get_document_topic_distribution(self, normalized_bows):
        theta = self.model.get_theta(normalized_bows)
        if isinstance(theta, tuple):
            return theta
        return theta, None

    def get_topic_word_distribution(self):
        return self.model.get_beta()


class _NVDMCore(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics):
        super().__init__()
        self.enc_vec = nn.Linear(vocab_size, hidden_size)
        self.mean = nn.Linear(hidden_size, num_topics)
        self.log_sigma = nn.Linear(hidden_size, num_topics)
        self.dec_vec = nn.Linear(num_topics, vocab_size)

    def encoder(self, bows):
        hidden = torch.tanh(self.enc_vec(bows))
        mean = self.mean(hidden)
        log_sigma = self.log_sigma(hidden)
        kld_theta = -0.5 * torch.sum(
            1 - torch.square(mean) + 2 * log_sigma - torch.exp(2 * log_sigma),
            dim=1,
        )
        return mean, log_sigma, kld_theta


class _InferenceNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation="softplus", dropout=0.2):
        super().__init__()
        if activation == "softplus":
            self.activation = nn.Softplus()
        else:
            self.activation = nn.ReLU()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hiddens = nn.Sequential(
            OrderedDict(
                [
                    (
                        "l_{}".format(i),
                        nn.Sequential(nn.Linear(h_in, h_out), self.activation),
                    )
                    for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))
                ]
            )
        )
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.dropout_enc = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        return mu, log_sigma


class _ProdLDACore(nn.Module):
    def __init__(self, input_size, num_topics, hidden_size, dropout=0.2):
        super().__init__()
        self.n_components = num_topics
        self.inf_net = _InferenceNetwork(
            input_size=input_size,
            output_size=num_topics,
            hidden_sizes=(hidden_size,),
            activation="softplus",
            dropout=dropout,
        )
        self.prior_mean = nn.Parameter(torch.zeros(num_topics))
        self.prior_variance = nn.Parameter(torch.full((num_topics,), 1.0 - (1.0 / num_topics)))
        self.beta = nn.Parameter(torch.empty(num_topics, input_size))
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        self.drop_theta = nn.Dropout(p=dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, bows):
        posterior_mu, posterior_log_sigma = self.inf_net(bows)
        posterior_sigma = torch.exp(posterior_log_sigma)
        theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)
        word_dists = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma,
            word_dists,
            theta,
        )


class _NSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_topics):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, num_topics),
            nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)


def _sinkhorn_torch(cost_matrix, a, b, lambda_sh, num_iter_max=5000, stop_thr=0.5e-2):
    device = cost_matrix.device
    dtype = cost_matrix.dtype
    a = a.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)
    u = (torch.ones_like(a, dtype=dtype) / a.size()[0]).to(device)
    v = torch.ones_like(b, dtype=dtype).to(device)
    kernel = torch.exp(-cost_matrix * lambda_sh).to(dtype=dtype)
    err = 1.0
    steps = 0
    while err > stop_thr and steps < num_iter_max:
        u = torch.div(a, torch.matmul(kernel, torch.div(b, torch.matmul(u.t(), kernel).t())))
        steps += 1
        if steps % 20 == 1:
            v = torch.div(b, torch.matmul(kernel.t(), u))
            u = torch.div(a, torch.matmul(kernel, v))
            bb = torch.mul(v, torch.matmul(kernel.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float("inf"))
    sinkhorn = torch.sum(torch.mul(u, torch.matmul(torch.mul(kernel, cost_matrix), v)), dim=0)
    return sinkhorn


class _ScholarCore(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_topics, init_emb=None, update_embeddings=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_topics = num_topics
        self.embeddings_x_layer = nn.Linear(vocab_size, emb_dim, bias=False)
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(torch.from_numpy(init_emb))
        else:
            nn.init.xavier_uniform_(self.embeddings_x_layer.weight)
        self.embeddings_x_layer.weight.requires_grad = update_embeddings

        self.encoder_dropout_layer = nn.Dropout(p=0.2)
        self.mean_layer = nn.Linear(emb_dim, num_topics)
        self.logvar_layer = nn.Linear(emb_dim, num_topics)
        self.mean_bn_layer = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(torch.ones(num_topics))
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(torch.ones(num_topics))
        self.logvar_bn_layer.weight.requires_grad = False
        self.z_dropout_layer = nn.Dropout(p=0.2)
        self.beta_layer = nn.Linear(num_topics, vocab_size)
        nn.init.xavier_uniform_(self.beta_layer.weight)
        self.eta_bn_layer = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.eta_bn_layer.weight.data.copy_(torch.ones(vocab_size))
        self.eta_bn_layer.weight.requires_grad = False

        alpha = np.ones((1, num_topics), dtype=np.float32)
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (((1.0 / alpha) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / alpha, 1)).T
        self.register_buffer("prior_mean", torch.from_numpy(np.array(prior_mean).reshape((1, num_topics))))
        self.register_buffer("prior_logvar", torch.from_numpy(np.array(np.log(prior_var)).reshape((1, num_topics))))

    def forward(self, bows, eta_bn_prop=1.0, var_scale=1.0, compute_theta=False):
        embedded = self.embeddings_x_layer(bows)
        encoder_output = F.softplus(embedded)
        encoder_output = self.encoder_dropout_layer(encoder_output)
        posterior_mean = self.mean_layer(encoder_output)
        posterior_logvar = self.logvar_layer(encoder_output)
        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        posterior_var = posterior_logvar_bn.exp()
        eps = torch.randn_like(posterior_mean_bn)
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z = self.z_dropout_layer(z)
        theta = F.softmax(z, dim=1)
        if compute_theta:
            return theta
        eta = self.beta_layer(theta)
        eta_bn = self.eta_bn_layer(eta)
        x_recon_bn = F.softmax(eta_bn, dim=1)
        x_recon_no_bn = F.softmax(eta, dim=1)
        x_recon = eta_bn_prop * x_recon_bn + (1.0 - eta_bn_prop) * x_recon_no_bn
        prior_mean = self.prior_mean.expand_as(posterior_mean_bn)
        prior_logvar = self.prior_logvar.expand_as(posterior_logvar_bn)
        nl = -(bows * (x_recon + 1e-10).log()).sum(1)
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar_bn.exp()
        var_division = posterior_var / prior_var
        diff = posterior_mean_bn - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar_bn
        kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.n_topics)
        return theta, x_recon, nl.mean(), kld.mean()


class NVDM(BaseTopicModel):
    expects_normalized_bows = False

    def __init__(
        self,
        num_topics,
        vocab_size,
        t_hidden_size,
        rho_size,
        emsize,
        theta_act,
        embeddings=None,
        train_embeddings=False,
        enc_drop=0.0,
        **_,
    ):
        del rho_size, emsize, theta_act, embeddings, train_embeddings, enc_drop
        super().__init__()
        self.model = _NVDMCore(vocab_size, t_hidden_size, num_topics)

    def build_optimizer(self, args):
        return _build_optimizer(self.parameters(), args)

    def compute_losses(self, bows, normalized_bows, theta=None, aggregate=True):
        del normalized_bows, theta
        mean, log_sigma, kld_theta = self.model.encoder(bows)
        latent = mean + torch.exp(log_sigma) * torch.randn_like(mean)
        logits = F.log_softmax(self.model.dec_vec(latent), dim=1)
        reconstruction_loss = -(logits * bows).sum(dim=1)
        if aggregate:
            reconstruction_loss = reconstruction_loss.mean()
            kld_theta = kld_theta.mean()
        return LossBundle(reconstruction_loss, {"kld_theta": kld_theta})

    def get_document_topic_distribution(self, model_input):
        mean, _, _ = self.model.encoder(model_input)
        return F.softmax(mean, dim=1), None

    def get_topic_word_distribution(self):
        return F.softmax(self.model.dec_vec.weight.t(), dim=1)


class PLDA(BaseTopicModel):
    expects_normalized_bows = False

    def __init__(
        self,
        num_topics,
        vocab_size,
        t_hidden_size,
        rho_size,
        emsize,
        theta_act,
        embeddings=None,
        train_embeddings=False,
        enc_drop=0.2,
        **_,
    ):
        del rho_size, emsize, theta_act, embeddings, train_embeddings
        super().__init__()
        self.model = _ProdLDACore(vocab_size, num_topics, t_hidden_size, dropout=enc_drop)

    def build_optimizer(self, args):
        return _build_optimizer(self.parameters(), args)

    def compute_losses(self, bows, normalized_bows, theta=None, aggregate=True):
        del normalized_bows, theta
        prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance, word_dists, _ = self.model(bows)
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
        logvar_det_division = prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        kld_theta = 0.5 * (var_division + diff_term - self.model.n_components + logvar_det_division)
        reconstruction_loss = -(bows * torch.log(word_dists + 1e-10)).sum(dim=1)
        if aggregate:
            reconstruction_loss = reconstruction_loss.mean()
            kld_theta = kld_theta.mean()
        return LossBundle(reconstruction_loss, {"kld_theta": kld_theta})

    def get_document_topic_distribution(self, model_input):
        posterior_mu, _ = self.model.inf_net(model_input)
        return F.softmax(posterior_mu, dim=1), None

    def get_topic_word_distribution(self):
        return F.softmax(self.model.beta_batchnorm(self.model.beta), dim=1)


class NSTM(BaseTopicModel):
    expects_normalized_bows = False

    def __init__(
        self,
        num_topics,
        vocab_size,
        t_hidden_size,
        rho_size,
        emsize,
        theta_act,
        embeddings=None,
        train_embeddings=True,
        enc_drop=0.0,
        sh_alpha=20.0,
        rec_loss_weight=0.7,
        **_,
    ):
        del rho_size, theta_act, enc_drop
        super().__init__()
        if embeddings is None:
            raise ValueError("NSTM requires pretrained embeddings.")
        word_embeddings = embeddings.detach().clone().float() if torch.is_tensor(embeddings) else torch.tensor(embeddings, dtype=torch.float32)
        self.encoder = _NSTMEncoder(vocab_size, t_hidden_size, num_topics)
        self.word_embedding = nn.Parameter(word_embeddings, requires_grad=train_embeddings)
        self.topic_embedding = nn.Parameter(torch.empty(num_topics, emsize))
        nn.init.trunc_normal_(self.topic_embedding, std=0.1)
        self.sh_alpha = sh_alpha
        self.rec_loss_weight = rec_loss_weight

    def build_optimizer(self, args):
        return _build_optimizer(self.parameters(), args)

    def _topic_word_scores(self):
        word_embedding_norm = F.normalize(self.word_embedding, p=2, dim=1)
        topic_embedding_norm = F.normalize(self.topic_embedding, p=2, dim=1)
        return torch.matmul(topic_embedding_norm, word_embedding_norm.t())

    def compute_losses(self, bows, normalized_bows, theta=None, aggregate=True):
        del normalized_bows, theta
        doc_word = F.softmax(bows, dim=1)
        doc_topic = self.encoder(bows)
        topic_word = self._topic_word_scores()
        sinkhorn_loss = _sinkhorn_torch(1 - topic_word, doc_topic.t(), doc_word.t(), lambda_sh=self.sh_alpha).mean()
        rec_log_probs = F.log_softmax(torch.matmul(doc_topic, topic_word), dim=1)
        reconstruction_loss = -(rec_log_probs * bows).sum(dim=1)
        if aggregate:
            reconstruction_loss = reconstruction_loss.mean()
        return LossBundle(self.rec_loss_weight * reconstruction_loss, {"sinkhorn_loss": sinkhorn_loss})

    def get_document_topic_distribution(self, model_input):
        return self.encoder(model_input), None

    def get_topic_word_distribution(self):
        return F.softmax(self._topic_word_scores(), dim=1)


class SCHOLAR(BaseTopicModel):
    expects_normalized_bows = False

    def __init__(
        self,
        num_topics,
        vocab_size,
        t_hidden_size,
        rho_size,
        emsize,
        theta_act,
        embeddings=None,
        train_embeddings=False,
        enc_drop=0.0,
        **_,
    ):
        del t_hidden_size, rho_size, theta_act, enc_drop
        super().__init__()
        init_embeddings = None
        if embeddings is not None:
            if torch.is_tensor(embeddings):
                init_embeddings = embeddings.detach().cpu().numpy().T
            else:
                init_embeddings = np.asarray(embeddings).T
        self.model = _ScholarCore(
            vocab_size=vocab_size,
            emb_dim=emsize,
            num_topics=num_topics,
            init_emb=init_embeddings,
            update_embeddings=train_embeddings,
        )

    def build_optimizer(self, args):
        return _build_optimizer(self.parameters(), args)

    def compute_losses(self, bows, normalized_bows, theta=None, aggregate=True):
        del normalized_bows, theta, aggregate
        _, _, nl, kld_theta = self.model(bows)
        return LossBundle(nl, {"kld_theta": kld_theta})

    def get_document_topic_distribution(self, model_input):
        return self.model(model_input, var_scale=0.0, compute_theta=True), None

    def get_topic_word_distribution(self):
        return F.softmax(self.model.beta_layer.weight.t(), dim=1)


class WETE(BaseTopicModel):
    expects_normalized_bows = False

    def __init__(
        self,
        num_topics,
        vocab_size,
        t_hidden_size,
        rho_size,
        emsize,
        theta_act,
        embeddings=None,
        train_embeddings=True,
        enc_drop=0.0,
        beta=0.5,
        epsilon=1.0,
        init_alpha=True,
        **_,
    ):
        del rho_size, theta_act, enc_drop
        super().__init__()
        if embeddings is None:
            raise ValueError("WeTe requires pretrained embeddings.")
        word_embeddings = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
        if word_embeddings.shape[0] != vocab_size:
            raise ValueError("Unexpected embedding matrix shape for WeTe.")
        self.topic_k = num_topics
        self.voc_size = vocab_size
        self.h = emsize
        self.beta = beta
        self.epsilon = epsilon
        self.real_min = 1e-30
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.topic_indices = torch.arange(self.topic_k, device=self.device_name)
        self.word_indices = torch.arange(self.voc_size, device=self.device_name)
        self.topic_layer = nn.Embedding(self.topic_k, self.h)
        self.word_layer = nn.Embedding(self.voc_size, self.h)
        self.infer_net = Infer_Net(v=self.voc_size, d_hidden=t_hidden_size, k=self.topic_k)
        if init_alpha and self.voc_size >= self.topic_k:
            topic_init = cluster_kmeans(word_embeddings, n=self.topic_k)
        else:
            topic_init = np.random.normal(0, 0.5, size=(self.topic_k, self.h))
        self.topic_layer.weight.data.copy_(torch.from_numpy(topic_init).float())
        self.word_layer.weight.data.copy_(torch.from_numpy(word_embeddings).float())
        self.word_layer.weight.requires_grad = train_embeddings

    def build_optimizer(self, args):
        return _build_optimizer(self.parameters(), args)

    def _update_embeddings(self):
        rho = self.word_layer(self.word_indices)
        alpha = self.topic_layer(self.topic_indices)
        return rho, alpha

    def _topic_word_distribution(self):
        rho, alpha = self._update_embeddings()
        return F.softmax(torch.matmul(rho, alpha.t()), dim=0).t()

    def _bows_to_token_lists(self, bows):
        token_lists = []
        int_bows = bows.long()
        for row in int_bows:
            indices = torch.where(row > 0)[0]
            token_lists.append(
                torch.tensor(
                    [idx.item() for idx in indices for _ in range(int(row[idx].item()))],
                    device=row.device,
                    dtype=torch.long,
                )
            )
        return token_lists

    def _cost_ct(self, inner_p, cost_c, token_lists, theta):
        dis_d = torch.clamp(torch.exp(inner_p), 1e-30, 1e10)
        forward_cost = torch.tensor(0.0, device=theta.device)
        backward_cost = torch.tensor(0.0, device=theta.device)
        theta_norm = F.softmax(theta, dim=-1)
        for tokens, theta_row in zip(token_lists, theta_norm):
            if tokens.numel() == 0:
                continue
            forward_doc_dis = dis_d[tokens] * theta_row.unsqueeze(0)
            doc_dis = dis_d[tokens]
            forward_pi = forward_doc_dis / (torch.sum(forward_doc_dis, dim=1, keepdim=True) + self.real_min)
            backward_pi = doc_dis / (torch.sum(doc_dis, dim=0, keepdim=True) + self.real_min)
            forward_cost = forward_cost + (cost_c[tokens] * forward_pi).sum(1).mean()
            backward_cost = backward_cost + ((cost_c[tokens] * backward_pi).sum(0) * theta_row).sum()
        return forward_cost, backward_cost

    def compute_losses(self, bows, normalized_bows, theta=None, aggregate=True):
        del normalized_bows, theta, aggregate
        token_lists = self._bows_to_token_lists(bows)
        theta_values = self.infer_net(bows)
        rho, alpha = self._update_embeddings()
        phi = F.softmax(torch.matmul(rho, alpha.t()), dim=0)
        inner_p = torch.matmul(rho, alpha.t())
        cost_c = torch.clamp(torch.exp(-inner_p), 1e-30, 1e10)
        forward_cost, backward_cost = self._cost_ct(inner_p, cost_c, token_lists, theta_values)
        reconstructed = torch.matmul(phi, theta_values.t()).t()
        tm_cost = -(bows * torch.log(reconstructed + 1e-10) - reconstructed - torch.lgamma(bows + 1.0)).sum(-1).mean()
        return LossBundle(
            self.epsilon * tm_cost,
            {
                "forward_transport": self.beta * forward_cost,
                "backward_transport": (1.0 - self.beta) * backward_cost,
            },
        )

    def get_document_topic_distribution(self, model_input):
        return F.softmax(self.infer_net(model_input), dim=1), None

    def get_topic_word_distribution(self):
        return self._topic_word_distribution()
