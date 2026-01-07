from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

#first round
# test_50_acc 96.279997
Lip_DARTS_SA_50 = Genotype(normal=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 3), ('skip_connect', 1), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

# test_75_acc 96.489998
Lip_DARTS_SA_75 = Genotype(normal=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# test_100_acc 96.579998
Lip_DARTS_SA_100 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

# test_125_acc 96.389997
Lip_DARTS_SA_125 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# test_150_acc 96.439997
Lip_DARTS_SA_150 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 3), ('skip_connect', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

# test_175_acc 96.179998
Lip_DARTS_SA_175 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# test_200_acc 96.739997
Lip_DARTS_SA_200 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))

geno1=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
geno2=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))
geno3=Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
geno4=Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
geno5=Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

geno6=Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

DARTS = DARTS_V2

retrain1 = Lip_DARTS_SA_50
retrain2 = Lip_DARTS_SA_75
retrain3 = Lip_DARTS_SA_100
retrain4 = Lip_DARTS_SA_125
retrain5 = Lip_DARTS_SA_150
retrain6 = Lip_DARTS_SA_175
retrain7 = Lip_DARTS_SA_200
