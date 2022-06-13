import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
EPS = 1e-10


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE

    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class WNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(WNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class Encoder(MetaModule):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            MetaLinear(input_dim, 500),
            nn.ReLU(),
            MetaLinear(500, 500),
            nn.ReLU(),
            MetaLinear(500, 2000),
            nn.ReLU(),
            MetaLinear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(MetaModule):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            MetaLinear(feature_dim, 2000),
            nn.ReLU(),
            MetaLinear(2000, 500),
            nn.ReLU(),
            MetaLinear(500, 500),
            nn.ReLU(),
            MetaLinear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class SafeNetwork(MetaModule):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(SafeNetwork, self).__init__()
        self.encoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.feature_submodule = nn.Sequential(
            MetaLinear(feature_dim, feature_dim),
            nn.ReLU(),
            MetaLinear(feature_dim, high_feature_dim)
        )
        self.label_submodule = nn.Sequential(
            MetaLinear(feature_dim, feature_dim),
            nn.ReLU(),
            MetaLinear(feature_dim, class_num),
            nn.Softmax(dim=1))
        self.view = view

    def forward(self, xs, xs_incomplete):
        qs = []
        qs_incomplete = []
        zs = []
        zs_incomplete = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.feature_submodule(z)
            q = self.label_submodule(z)
            zs.append(h)
            qs.append(q)

            x_ = xs_incomplete[v]
            z_ = self.encoders[v](x_)
            h_ = self.feature_submodule(z_)
            q_ = self.label_submodule(z_)
            zs_incomplete.append(h_)
            qs_incomplete.append(q_)
        return zs, qs, zs_incomplete, qs_incomplete

    def forward_xs(self, xs):
        hs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            hs.append(self.feature_submodule(z))

        return hs, None, None

    def forward_s(self, xs):
        qs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.feature_submodule(z)
            q = self.label_submodule(z)
            zs.append(h)
            qs.append(q)

        return zs, qs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_submodule(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds


class Online(MetaModule):
    def __init__(self, view, input_size, feature_dim):
        super(Online, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.view = view

    def forward(self, xs):
        xrs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            xrs.append(self.decoders[v](z))

        return xrs
