import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import random
import logging

from Lipschitz_operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES, Genotype


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self.C = C
        self.stride = stride
        self._op_cache = {}

    def forward(self, x, weights):
        C_in = x.size(1)
        key = (C_in, self.stride)
        if key not in self._op_cache:
            self._op_cache[key] = nn.ModuleList(
                [OPS[p](C_in, self.stride, affine=False) for p in PRIMITIVES]
            ).to(x.device)
        ops = self._op_cache[key]

        return sum(w * op(x) for w, op in zip(weights, ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self._steps = steps
        self._multiplier = multiplier

        self.preprocess0 = LipschitzFactorizedReduce(C_prev_prev, C, affine=False) if reduction_prev \
            else LipschitzReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = LipschitzReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()
        for i in range(steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                self._ops.append(MixedOp(C, stride))

    def forward(self, s0, s1, weights):
        s0, s1 = self.preprocess0(s0), self.preprocess1(s1)
        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            s = 0
            for j in range(2 + i):
                if isinstance(weights, list):
                    w = weights[offset + j]
                else:
                    w = weights[offset + j]

                y = self._ops[offset + j](states[j], w)
                s = s + y
            offset += 2 + i
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion,
                 steps=4, multiplier=4, stem_multiplier=3,
                 max_genotype_batches=15, fixed_alphas=False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.fixed_alphas = fixed_alphas
        self.max_genotype_batches = max_genotype_batches

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._initialize_alphas()
        self.best_genotype_normal = None
        self.best_genotype_reduce = None
        self._num_edges = sum(2 + i for i in range(self._steps))

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        if self.fixed_alphas:
            self.alphas_normal = Variable(torch.ones(k, num_ops).cuda() / num_ops, requires_grad=False)
            self.alphas_reduce = Variable(torch.ones(k, num_ops).cuda() / num_ops, requires_grad=False)
        else:
            self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
            self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def forward(self, input):
        s0 = s1 = self.stem(input)
        k = self._num_edges
        for i, cell in enumerate(self.cells):
            if self.fixed_alphas:
                w = torch.full((len(PRIMITIVES),), 1. / len(PRIMITIVES), dtype=torch.float32, device='cuda')
                weights = [w for _ in range(k)]
            else:
                alpha = self.alphas_reduce if cell.reduction else self.alphas_normal
                weights = torch.unbind(F.softmax(alpha, dim=-1))
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        return self._criterion(self(input), target)

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers,
                            self._criterion, fixed_alphas=self.fixed_alphas).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def _generate_random_genotype(self):

        def _build_cell():
            gene = []
            for i in range(self._steps):
                node = 2 + i
                available_sources = list(range(node))
                inputs = random.sample(available_sources, 2)

                for src in inputs:
                    op = random.choice(PRIMITIVES)
                    gene.append((op, src))
            return gene

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        return Genotype(
            normal=_build_cell(),
            normal_concat=concat,
            reduce=_build_cell(),
            reduce_concat=concat
        )

    def _mutate_genotype(self, genotype: Genotype) -> Genotype:

        def _mutate_gene(gene):
            new_gene = gene.copy()

            mutate_idx = random.randint(0, len(new_gene) - 1)
            op, src = new_gene[mutate_idx]

            target_node = 2 + mutate_idx // 2

            other_idx = (mutate_idx // 2) * 2 + (1 - (mutate_idx % 2))
            other_src = new_gene[other_idx][1]

            if random.random() < 0.5:
                new_op = random.choice([o for o in PRIMITIVES if o != op])
                new_gene[mutate_idx] = (new_op, src)
            else:
                available_srcs = [s for s in range(target_node) if s != other_src]
                if available_srcs:
                    new_src = random.choice(available_srcs)
                    new_gene[mutate_idx] = (op, new_src)

            return new_gene

        return Genotype(
            normal=_mutate_gene(genotype.normal),
            normal_concat=genotype.normal_concat,
            reduce=_mutate_gene(genotype.reduce),
            reduce_concat=genotype.reduce_concat
        )

    def _get_all_paths_from_genotype(self, genotype):
        graph = {i: [] for i in range(2 + self._steps)}

        for i in range(0, len(genotype.normal), 2):
            node_idx = 2 + i // 2
            edges = genotype.normal[i:i + 2]
            for op, src in edges:
                graph[src].append((node_idx, op))

        paths = []

        def dfs(current, path_nodes, path_ops):
            if current >= 2 + self._steps - 1:
                paths.append((tuple(path_nodes), tuple(path_ops)))
                return

            for next_node, op in graph.get(current, []):
                if next_node not in path_nodes:
                    dfs(next_node, path_nodes + [next_node], path_ops + [op])

        dfs(0, [0], [])
        dfs(1, [1], [])

        return paths

    def _evaluate_path_gradient_strength(self, ops, represent_cell, valid_loader, num_batches):
        total_strength = 0.0
        batch_count = 0

        for batch_idx, (x, _) in enumerate(valid_loader):
            if batch_count >= num_batches:
                break

            x = x.cuda(non_blocking=True)
            batch_strength = self._compute_path_gradient_norm(x, ops, represent_cell)
            total_strength += batch_strength
            batch_count += 1

        return total_strength / batch_count if batch_count > 0 else 0.0

    def _compute_path_gradient_norm(self, x, ops, represent_cell):
        x = x.detach().requires_grad_(True)

        h = x
        C_target = represent_cell.preprocess1.conv.in_channels

        if h.size(1) != C_target:
            align_conv = nn.Conv2d(h.size(1), C_target, 1, bias=False).cuda()
            h = align_conv(h)

        for op_name in ops:
            op_instance = self._get_trained_op_instance(represent_cell, op_name, C_target)
            if op_instance is not None:
                h = op_instance(h)
            else:
                op_instance = OPS[op_name](C_target, 1, affine=False).cuda()
                h = op_instance(h)

        output = F.adaptive_avg_pool2d(h, 1)
        output = output.view(output.size(0), -1)
        output = output + torch.randn_like(output) * 1e-6

        pseudo_target = output.argmax(dim=1).detach()

        loss = F.cross_entropy(output, pseudo_target)
        gradients = torch.autograd.grad(loss, x, create_graph=False, retain_graph=False)[0]

        gradient_norm = gradients.norm(p=2, dim=(1, 2, 3)).mean()

        return gradient_norm.item()

    def _get_trained_op_instance(self, represent_cell, op_name, C_in):
            for mixed_op in represent_cell._ops:
                key = (C_in, 1)
                if key not in mixed_op._op_cache:
                    continue
                ops = mixed_op._op_cache[key]
                for i, primitive in enumerate(PRIMITIVES):
                    if primitive == op_name:
                        return ops[i]
            return None

    def _evaluate_genotype_fitness(self, genotype, represent_cell, valid_loader, num_batches):
        paths = self._get_all_paths_from_genotype(genotype)
        if not paths:
            return 0.0

        total_fitness = 0.0
        path_count = 0

        for path_nodes, path_ops in paths:
            path_strength = self._evaluate_path_gradient_strength(
                path_ops, represent_cell, valid_loader, num_batches
            )
            total_fitness += path_strength
            path_count += 1

        return total_fitness / path_count if path_count > 0 else 0.0

    def simulated_annealing_search(self, represent_cell, valid_loader,
                                   max_batches=15, max_iter=200,
                                   temp_init=10.0, cooling_rate=0.95):

        current_genotype = self._generate_random_genotype()
        current_fitness = self._evaluate_genotype_fitness(
            current_genotype, represent_cell, valid_loader, max_batches
        )

        best_genotype = current_genotype
        best_fitness = current_fitness

        temperature = temp_init

        logging.info(f"SA Search Start: initial fitness = {current_fitness:.6f}")

        for iteration in range(max_iter):
            neighbor_genotype = self._mutate_genotype(current_genotype)
            neighbor_fitness = self._evaluate_genotype_fitness(
                neighbor_genotype, represent_cell, valid_loader, max_batches
            )

            delta_fitness = current_fitness - neighbor_fitness

            if delta_fitness > 0 or random.random() < math.exp(delta_fitness / (temperature + 1e-8)):
                current_genotype = neighbor_genotype
                current_fitness = neighbor_fitness

                if current_fitness < best_fitness:
                    best_genotype = current_genotype
                    best_fitness = current_fitness
                    logging.info(f"SA Iter {iteration}: New best fitness = {best_fitness:.6f}")

            temperature *= cooling_rate

            if iteration % 50 == 0:
                logging.info(f"SA Iter {iteration}: temp = {temperature:.4f}, current fitness = {current_fitness:.6f}")

        logging.info(f"SA Search Complete: best fitness = {best_fitness:.6f}")
        return best_genotype

    def update_path_strength_valid(self, valid_queue, max_batches=15,
                                   sa_max_iter=200, sa_temp_init=10.0, sa_cooling_rate=0.95):
        self.eval()

        normal_cells = [i for i, c in enumerate(self.cells) if not c.reduction]
        reduce_cells = [i for i, c in enumerate(self.cells) if c.reduction]

        if normal_cells:
            idx = normal_cells[len(normal_cells) // 2]
            best_normal = self.simulated_annealing_search(
                self.cells[idx], valid_queue, max_batches, sa_max_iter, sa_temp_init, sa_cooling_rate
            )
            self.best_genotype_normal = best_normal
            logging.info(f"Best Normal Cell: {best_normal}")

        if reduce_cells:
            idx = reduce_cells[len(reduce_cells) // 2]
            best_reduce = self.simulated_annealing_search(
                self.cells[idx], valid_queue, max_batches, sa_max_iter, sa_temp_init, sa_cooling_rate
            )
            self.best_genotype_reduce = best_reduce
            logging.info(f"Best Reduce Cell: {best_reduce}")

        torch.cuda.empty_cache()
        self.train()

    def genotype(self):
        if self.best_genotype_normal is None or self.best_genotype_reduce is None:
            return self._generate_random_genotype()

        return Genotype(
            normal=self.best_genotype_normal.normal,
            normal_concat=self.best_genotype_normal.normal_concat,
            reduce=self.best_genotype_reduce.reduce,
            reduce_concat=self.best_genotype_reduce.reduce_concat
        )