# -*- coding:utf-8 -*-


import argparse

from RunModelSetting import run_model, ensemble_run_model
# from RunModel import run_model, ensemble_run_model
from model import LAMNet
parser = argparse.ArgumentParser(
    prog='LAMNet',
    description='LAMNet is model in paper: \"MultiheadCrossAttention based network model for DTI prediction\"',
    epilog='Model config set by c82098onfig.py')

parser.add_argument('dataSetName', choices=[
                     "DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"], default='DrugBank', help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['LAMNet', 'LAMNet-B', 'onlyPolyLoss', 'onlyLAM'],
                    default='onlyLAM', help='Which model to use, \"LAMNet\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=4079,
                    help='Set the random seed, the default is 4079')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
args = parser.parse_args()

# if args.model == 'LAMNet':
#      run_model(SEED=args.seed, DATASET=args.dataSetName,
#               MODEL=LAMNet, K_Fold=args.fold, LOSS='PolyLoss')
if args.model == 'onlyLAM':
    run_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=LAMNet, K_Fold=args.fold, LOSS='CrossEntropy')
# if args.model == 'onlyPolyLoss':
#     run_model(SEED=args.seed, DATASET=args.dataSetName,
#               MODEL=onlyPolyLoss, K_Fold=args.fold, LOSS='PolyLoss')
#     #采用polyloss损失函数环节药物靶标数据集中的过拟合问题
#
# if args.model == 'LAMNet-B':
#     ensemble_run_model(SEED=args.seed, DATASET=args.dataSetName, K_Fold=args.fold)

#将通过K FOLD交叉验证得到的多个模型进行组合得到了LAMNet-B
