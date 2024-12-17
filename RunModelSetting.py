# -*- coding:utf-8 -*-

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  itertools
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import LAMNet
from utils.DataPrepare import get_kfold_data, shuffle_dataset, split_data,get_data
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from LossFunction import CELoss, PolyLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_hyper(hp,param_set):
    return hp

def run_model(SEED, DATASET, MODEL, K_Fold, LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''调优'''
    learning_rate = [1e-4]
    prompt_len = [1100]

    # for lr,pt_len in itertools.product(learning_rate,prompt_len):
    #     '''init hyperparameters'''
    #     hp = hyperparameter()
    #     hp.Learning_rate = lr
    #     hp.prompt_len = pt_len
    #     print(str(hp.Learning_rate) + ' ' + str(hp.prompt_len))

    for lr,pt_len in itertools.product(learning_rate,prompt_len):
        '''init hyperparameters'''
        hp = hyperparameter()
        hp.Learning_rate = lr
        hp.prompt_len = pt_len

        '''load dataset from text file'''
        assert DATASET in ["DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"]
        print("Train in " + DATASET)
        print("load data")

        print("lr = {}".format(hp.Learning_rate))
        print("prompt_len = {}".format(hp.prompt_len))
        print("seed = {}".format(SEED))
        dir_input = ('./DataSets/{}.txt'.format(DATASET))
        with open(dir_input, "r") as f:
            data_list = f.read().strip().split('\n')
        print("load finished")

        '''set loss function weight'''
        if DATASET == "Davis":
            weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
        elif DATASET == "KIBA":
            weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
        else:
            weight_loss = None




        '''shuffle data'''
        print("data shuffle")
        data_list = shuffle_dataset(data_list, SEED)

        '''split dataset to train&validation set and test set'''
        split_pos = len(data_list) - int(len(data_list) * 0.2)
        train_data_list = data_list[0:split_pos]
        test_data_list = data_list[split_pos:-1]
        print('Number of Train&Val set: {}'.format(len(train_data_list)))
        print('Number of Test set: {}'.format(len(test_data_list)))
        drugs, proteins = get_data(data_list)
        '''metrics'''

        Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
        Precision_List_stable_drug, Recall_List_stable_drug, Accuracy_List_stable_drug, AUC_List_stable_drug, AUPR_List_stable_drug = [], [], [], [], []
        Precision_List_best_drug, Recall_List_best_drug, Accuracy_List_best_drug, AUC_List_best_drug, AUPR_List_best_drug = [], [], [], [], []
        Precision_List_stable_protein, Recall_List_stable_protein, Accuracy_List_stable_protein, AUC_List_stable_protein, AUPR_List_stable_protein = [], [], [], [], []
        Precision_List_best_protein, Recall_List_best_protein, Accuracy_List_best_protein, AUC_List_best_protein, AUPR_List_best_protein = [], [], [], [], []
        Precision_List_stable_deno, Recall_List_stable_deno, Accuracy_List_stable_deno, AUC_List_stable_deno, AUPR_List_stable_deno = [], [], [], [], []
        Precision_List_best_deno, Recall_List_best_deno, Accuracy_List_best_deno, AUC_List_best_deno, AUPR_List_best_deno = [], [], [], [], []


    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        # train_dataset, valid_dataset = get_kfold_data(
        #     i_fold, train_data_list, k=K_Fold)
        # train_dataset = CustomDataSet(train_dataset)
        # valid_dataset = CustomDataSet(valid_dataset)
        # test_dataset = CustomDataSet(test_data_list)
        # train_size = len(train_dataset)
        #
        # train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
        #                                   collate_fn=collate_fn, drop_last=True)
        # valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
        #                                   collate_fn=collate_fn, drop_last=True)
        # test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
        #                                  collate_fn=collate_fn, drop_last=True)
        _,test_drugs = get_kfold_data(i_fold, drugs, k=K_Fold)
        _,test_proteins = get_kfold_data(i_fold, proteins, k=K_Fold)
        train_dataset, test_dataset_drug, \
            test_dataset_protein, test_dataset_denovel = split_data(data_list,test_drugs,test_proteins)
        TVdataset = CustomDataSet(train_dataset)
        test_dataset_drug = CustomDataSet(test_dataset_drug)
        test_dataset_protein = CustomDataSet(test_dataset_protein)
        test_dataset_denovel = CustomDataSet(test_dataset_denovel)
        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                                             collate_fn=collate_fn, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                        collate_fn=collate_fn, drop_last=True)
        # valid_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
        #                                   collate_fn=collate_fn, drop_last=True)
        test_dataset_drug_loader = DataLoader(test_dataset_drug, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                            collate_fn=collate_fn, drop_last=True)
        test_dataset_protein_loader = DataLoader(test_dataset_protein, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                               collate_fn=collate_fn, drop_last=True)
        test_dataset_denovel_loader = DataLoader(test_dataset_denovel, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                               collate_fn=collate_fn, drop_last=True)



        # model = AttentionDTA(head_num=args.head_num).cuda()

        # global_step, tr_loss = train(model, train_dataset_load, valid_dataset_load, test_dataset_drug_load, args)
        # logger.info(f"global_step = {global_step}, average loss = {tr_loss}")

        # torch.save(model.state_dict(), args.output_dir + 'stable_checkpoint.pth')
        # model.load_state_dict(torch.load(args.output_dir + "valid_best_checkpoint.pth"))
        # trainset_test_results= inference(model, train_dataset_load, args, state='Train')
        # validset_test_results= inference(model, valid_dataset_load, args, state='Valid')
        # testset_test_drug_results= inference(model, test_dataset_drug_load, args, state='Test')
        # show_epoch_result(testset_test_drug_results,Precision_List_stable_drug, Recall_List_stable_drug, Accuracy_List_stable_drug, AUC_List_stable_drug, AUPR_List_stable_drug)
        # testset_test_protein_results= inference(model, test_dataset_protein_load, args, state='Test')
        # show_epoch_result(testset_test_protein_results,Precision_List_stable_protein,Recall_List_stable_protein,Accuracy_List_stable_protein, AUC_List_stable_protein, AUPR_List_stable_protein)
        # testset_test_pair_results= inference(model, test_dataset_denovel_load, args, state='Test')
        # show_epoch_result(testset_test_pair_results,Precision_List_stable_deno,Recall_List_stable_deno,Accuracy_List_stable_deno, AUC_List_stable_deno, AUPR_List_stable_deno)

        # if(i_fold == 2):
        #     break

    # show_result(Precision_List_stable_drug, Recall_List_stable_drug, Accuracy_List_stable_drug, AUC_List_stable_drug, AUPR_List_stable_drug)
    # show_result(Precision_List_stable_protein,Recall_List_stable_protein,Accuracy_List_stable_protein, AUC_List_stable_protein, AUPR_List_stable_protein)
    # show_result(Precision_List_stable_deno,Recall_List_stable_deno,Accuracy_List_stable_deno, AUC_List_stable_deno, AUPR_List_stable_deno)
        """ create model"""
        model = MODEL(hp).to(DEVICE)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        if LOSS == 'PolyLoss':
            Loss = PolyLoss(weight_loss=weight_loss,
                            DEVICE=DEVICE, epsilon=hp.loss_epsilon,Batch = hp.Batch_size)
        else:
            Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

        """Output files"""
        save_path = "./" + DATASET +' {}'.format(LOSS)+' setting'+ "/lr {}".format(hp.Learning_rate) +"/p {}".format(hp.prompt_len) + "batch{}".format(hp.Batch_size)+"/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break
            train_pbar = tqdm(
                enumerate(BackgroundGenerator(train_dataset_loader)),
                total=len(train_dataset_loader))

            """train"""
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                train_compounds, train_proteins, train_labels = train_data
                train_compounds = train_compounds.to(DEVICE)
                train_proteins = train_proteins.to(DEVICE)
                train_labels = train_labels.to(DEVICE)

                optimizer.zero_grad()

                predicted_interaction = model(train_compounds, train_proteins)
                train_loss = Loss(predicted_interaction, train_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  # 一次epoch的平均训练loss

            """valid"""
            valid_pbar = tqdm(
                enumerate(BackgroundGenerator(valid_dataset_loader)),
                total=len(valid_dataset_loader))
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:

                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.to(DEVICE)
                    valid_proteins = valid_proteins.to(DEVICE)
                    valid_labels = valid_labels.to(DEVICE)

                    valid_scores = model(valid_compounds, valid_proteins)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(
                        valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            # '''test'''
            # test_pbar = tqdm(
            #     enumerate(BackgroundGenerator(test_dataset_loader)),
            #     total=len(test_dataset_loader))
            # test_losses_in_epoch = []
            # model.eval()
            # Y, P, S = [], [], []
            # with torch.no_grad():
            #     for test_i, test_data in test_pbar:
            #
            #         test_compounds, test_proteins, test_labels = test_data
            #
            #         test_compounds = test_compounds.to(DEVICE)
            #         test_proteins = test_proteins.to(DEVICE)
            #         test_labels = test_labels.to(DEVICE)
            #
            #         test_scores = model(test_compounds, test_proteins)
            #         test_loss = Loss(test_scores, test_labels)
            #         test_losses_in_epoch.append(test_loss.item())
            #         test_labels = test_labels.to('cpu').data.numpy()
            #         test_scores = F.softmax(
            #             test_scores, 1).to('cpu').data.numpy()
            #         test_predictions = np.argmax(test_scores, axis=1)
            #         test_scores = test_scores[:, 1]
            #
            #         Y.extend(test_labels)
            #         P.extend(test_predictions)
            #         S.extend(test_scores)
            #
            # Precision_dev = precision_score(Y, P)
            # Reacll_dev = recall_score(Y, P)
            # Accuracy_dev = accuracy_score(Y, P)
            # AUC_dev = roc_auc_score(Y, S)
            # tpr, fpr, _ = precision_recall_curve(Y, S)
            # PRC_dev = auc(fpr, tpr)
            # test_loss_a_epoch = np.average(test_losses_in_epoch)
            #
            # epoch_len = len(str(hp.Epoch))
            # print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
            #              # f'train_loss: {train_loss_a_epoch:.5f} ' +
            #              f'test_loss: {test_loss_a_epoch:.5f} ' +
            #              f'test_AUC: {AUC_dev:.5f} ' +
            #              f'test_PRC: {PRC_dev:.5f} ' +
            #              f'test_Accuracy: {Accuracy_dev:.5f} ' +
            #              f'test_Precision: {Precision_dev:.5f} ' +
            #              f'test_Reacll: {Reacll_dev:.5f} ')
            # print(print_msg)

            '''save checkpoint and make decision when early stop'''
            early_stopping(AUC_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(
            early_stopping.savepath + '/valid_best_checkpoint.pth'))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Train", FOLD_NUM=1)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Valid", FOLD_NUM=1)
        # testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
        #     model, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", FOLD_NUM=1)
        # AUC_List_stable.append(AUC_test)
        # Accuracy_List_stable.append(Accuracy_test)
        # AUPR_List_stable.append(PRC_test)
        # Recall_List_stable.append(Recall_test)
        # Precision_List_stable.append(Precision_test)
        testset_drug_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_drug_loader, save_path, DATASET, Loss, DEVICE, dataset_class="drug", FOLD_NUM=1)
        AUC_List_stable_drug.append(AUC_test)
        Accuracy_List_stable_drug.append(Accuracy_test)
        AUPR_List_stable_drug.append(PRC_test)
        Precision_List_stable_drug.append(Precision_test)
        Recall_List_stable_drug.append(Recall_test)

        testset_protein_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_protein_loader, save_path, DATASET, Loss, DEVICE, dataset_class="protein", FOLD_NUM=1)
        AUC_List_stable_protein.append(AUC_test)
        Accuracy_List_stable_protein.append(Accuracy_test)
        AUPR_List_stable_protein.append(PRC_test)
        Precision_List_stable_protein.append(Precision_test)
        Recall_List_stable_protein.append(Recall_test)

        testset_deno_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_denovel_loader, save_path, DATASET, Loss, DEVICE, dataset_class="deno", FOLD_NUM=1)
        AUC_List_stable_deno.append(AUC_test)
        Accuracy_List_stable_deno.append(Accuracy_test)
        AUPR_List_stable_deno.append(PRC_test)
        Precision_List_stable_deno.append(Precision_test)
        Recall_List_stable_deno.append(Recall_test)

        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            # f.write('Learning_rate:' + hp.Learning_rate + ' ')
            # f.write('Batch_size:' + hp.Batch_size + ' ')
            # f.write('prompt_len:' + hp.prompt_len + ' ')
            # f.write(hp.Learning_rate)
            # f.write(hp.Batch_size)
            # f.write(hp.prompt_len)
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_drug_test_stable_results + '\n')
            f.write(testset_protein_test_stable_results + '\n')
            f.write(testset_deno_test_stable_results + '\n')



        show_result(DATASET, Accuracy_List_stable_drug, Precision_List_stable_drug,
                    Recall_List_stable_drug, AUC_List_stable_drug, AUPR_List_stable_drug, Ensemble=False)
        show_result(DATASET, Accuracy_List_stable_protein, Precision_List_stable_protein,
                    Recall_List_stable_protein, AUC_List_stable_protein, AUPR_List_stable_protein, Ensemble=False)
        show_result(DATASET, Accuracy_List_stable_deno, Precision_List_stable_deno,Recall_List_stable_deno, AUC_List_stable_deno, AUPR_List_stable_deno, Ensemble=False)
    
def ensemble_run_model(SEED, DATASET, K_Fold):

    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "KIBA", "Davis"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:-1]
    print('Number of Test set: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_dataset = CustomDataSet(test_data_list)
    test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                     collate_fn=collate_fn, drop_last=True)

    model = []
    for i in range(K_Fold):
        model.append(LAMNet(hp).to(DEVICE))
        '''LAMNet K-Fold train process is necessary'''
        try:
            model[i].load_state_dict(torch.load(
                f'./{DATASET}/{i+1}' + '/valid_best_checkpoint.pth', map_location=torch.device(DEVICE)))
        except FileNotFoundError as e:
            print('-'* 25 + 'ERROR' + '-'*25)
            error_msg = 'Load pretrained model error: \n' + \
                        str(e) + \
                        '\n' + 'LAMNet K-Fold train process is necessary'
            print(error_msg)
            print('-'* 55)
            exit(1)

    Loss = PolyLoss(weight_loss=weight_loss,
                    DEVICE=DEVICE, epsilon=hp.loss_epsilon,Batch = hp.Batch_size)

    testdataset_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
        model, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", save=True, FOLD_NUM=K_Fold)

    show_result(DATASET, Accuracy_test, Precision_test,
                Recall_test, AUC_test, PRC_test, Ensemble=True)
