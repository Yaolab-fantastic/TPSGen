from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from .pregan_original_data import LoadData
import torch.nn.functional as F
from .pregan_original_tensor2seq import save_sequence, tensor2seq, reserve_percentage
from .pregan_original_transfer_fasta import csv2fasta
from . import pregan_original_utils as utils
import matplotlib

matplotlib.use('Agg')

import os


def load_original_generator(checkpoint: str | os.PathLike, device: str = "cpu") -> Generator:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("architecture") != "thesis_original_pregan":
        raise ValueError("Checkpoint is not a thesis_original_pregan checkpoint.")
    generator = Generator(4, 4, seqL=int(payload.get("sequence_length", 165)),
                          ngf=int(payload.get("ngf", 512))).to(device)
    generator.load_state_dict(payload["generator_state_dict"], strict=True)
    generator.eval()
    return generator


def encode_original_condition(template: str, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    encoded = np.zeros((4, len(template)), dtype="float32")
    for index, base in enumerate(template.upper()):
        if base == "M":
            encoded[:, index] = rng.random(4)
        elif base in "ATCG":
            encoded["ATCG".index(base), index] = 1.0
        else:
            raise ValueError(f"Unsupported preGAN template symbol {base!r}.")
    return torch.from_numpy(encoded)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients
    else:
        return 0.0, None


def get_infinite_batches(data_loader):
    while True:
        for i, data in enumerate(data_loader):
            yield {'in': data['in'], 'out': data['out'], 'expr': data['expr']}


class ResBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=13, padding=6, bias=True):
        super(ResBlock, self).__init__()
        model = [nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + 0.3 * self.model(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=512, seqL=50, bias=True):
        super(Generator, self).__init__()
        self.ngf, self.seqL = ngf, seqL
        self.first_linear = nn.Linear(seqL * input_nc, seqL * ngf, bias=bias)
        model = [ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 nn.Conv1d(ngf, output_nc, 1),
                 nn.Softmax(dim=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x1 = self.first_linear(x.view(x.size(0), -1)).reshape([-1, self.ngf, self.seqL])
        return self.model(x1)


def generate_original_candidates(
    templates: list[str], checkpoint: str | os.PathLike, candidates: int = 5,
    seed: int = 42, device: str = "cpu",
) -> list[dict[str, object]]:
    if candidates < 1:
        raise ValueError("candidates must be positive")
    generator = load_original_generator(checkpoint, device)
    output = []
    for template_index, template in enumerate(templates):
        normalized = template.strip().upper()
        if len(normalized) != 165:
            raise ValueError("Original preGAN generation requires 165-bp templates.")
        for rank in range(1, candidates + 1):
            condition = encode_original_condition(normalized, seed + template_index * candidates + rank)
            with torch.no_grad():
                generated = generator(condition.unsqueeze(0).to(device))[0].cpu()
            sequence = "".join("ATCG"[int(index)] for index in generated.argmax(dim=0))
            gc = (sequence.count("G") + sequence.count("C")) / 165.0
            runs = [len(run) for run in __import__("re").findall(r"A+|C+|G+|T+", sequence)]
            output.append({
                "template_index": template_index,
                "candidate_rank": rank,
                "template": normalized,
                "sequence": sequence,
                "masked_positions": normalized.count("M"),
                "backend": "thesis_original_pregan",
                "gc_fraction": round(gc, 6),
                "max_homopolymer": max(runs, default=0),
                "passes_qc": bool(0.12 <= gc <= 0.52 and max(runs, default=0) <= 13),
            })
    return output


class Discriminator(nn.Module):
    def __init__(self, output_nc, ndf=512, seqL=50, bias=True):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Conv1d(output_nc, ndf, 1)
        model = [ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf)]
        self.model = nn.Sequential(*model)
        self.last_linear = nn.Linear(seqL * ndf, 1, bias=bias)

    def forward(self, x):
        x1 = self.Conv1(x)
        x_t = x1.permute(0, 2, 1)
        x_t = x_t.permute(0, 2, 1)
        x_t = self.model(x_t)
        return self.last_linear(x_t.contiguous().view(x_t.size(0), -1))


class WGAN():
    def __init__(self, input_nc, output_nc, seqL=100, lr=1e-4, gpu_ids='0', l1_w=10, predictor_path=None):
        super(WGAN, self).__init__()
        self.gpu_ids = gpu_ids
        self.l1_w = l1_w
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.generator = Generator(input_nc, output_nc, seqL=seqL)
        self.discriminator = Discriminator(input_nc + output_nc, seqL=seqL)

        # 加载预测器
        self.predictor = torch.load(predictor_path, map_location=self.device, weights_only=False)
        self.predictor.eval()
        for param in self.predictor.parameters():
            param.requires_grad = False
        for module in self.predictor.modules():
            if isinstance(module, nn.RNNBase):
                module.train()

        if len(gpu_ids) > 0:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.predictor = self.predictor.cuda()

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    def backward_g(self, fake_x, target_expr):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        self.generator.zero_grad()
        self.fake_inputs = self.generator(fake_x)
        pred_fake = self.discriminator(torch.cat((fake_x, self.fake_inputs), 1))
        self.g_loss = -pred_fake.mean()
        pred_expr = self.predictor(self.fake_inputs)
        self.p_loss = self.mse_loss(pred_expr, target_expr)
        self.g_l1 = self.l1_w * self.l1_loss(fake_x, self.fake_inputs)
        self.g_total_loss = self.g_loss + self.g_l1 + self.p_loss
        self.g_total_loss.backward()
        self.optim_g.step()

    def backward_d(self, real_x, fake_x):
        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.fake_inputs = self.generator(fake_x)
        fakeAB = torch.cat((fake_x, self.fake_inputs), 1)
        realAB = torch.cat((fake_x, real_x), 1)
        self.discriminator.zero_grad()
        pred_fake, pred_real = self.discriminator(fakeAB), self.discriminator(realAB)
        self.d_loss = pred_fake.mean() - pred_real.mean()
        self.gp, gradients = cal_gradient_penalty(self.discriminator, realAB, fakeAB, device=self.device)
        self.d_total_loss = self.d_loss + self.gp
        self.d_total_loss.backward()
        self.optim_d.step()


def main():
    import os

    data_name = 'merged_result'
    seqL = 165
    train_data, test_data = DataLoader(LoadData(is_train=True, path='../data/{}.csv'.format(data_name), split_r=0.8),
                                       batch_size=32, shuffle=True), DataLoader(
        LoadData(is_train=False, path='../data/{}.csv'.format(data_name), split_r=0.2), batch_size=32)
    train_data = get_infinite_batches(train_data)

    # 确保日志目录存在
    log_dir = 'cache2/training_log/'  # 定义日志目录
    os.makedirs(log_dir, exist_ok=True)  # 自动创建目录

    logger = utils.get_logger(log_path=log_dir, name=data_name)

    model = WGAN(input_nc=4, output_nc=4, seqL=seqL, l1_w=50,
                 predictor_path="/data/wangxiaoyu/project/deepseed/Predictor/results/model/165_mpra_expr_denselstm.pth")
    for i in range(10000):
        for j in range(5):
            _data = train_data.__next__()
            model.backward_d(_data['out'], _data['in'])
        model.backward_g(_data['in'], _data['expr'])


if __name__ == '__main__':
    main()
