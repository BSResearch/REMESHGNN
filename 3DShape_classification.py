import os
import torch
from data import DataLoader
import warnings
import torch.nn as nn
from options.classification_option import ClassificationOptions
from options.test_option import TestOptions
import pickle
from util import util
import time
import importlib
import wandb
from util.save_best_model_accuracy import SaveBestModelACC
from util.save_best_model import SaveBestModel


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    classificationOpt = ClassificationOptions().parse()
    config = vars(classificationOpt)
    dataset = DataLoader(classificationOpt)
    dataset_size = len(dataset)
    device = dataset.dataset.device
    print('#training meshes = %d' % dataset_size)
    module_name = classificationOpt.use_model
    net = importlib.import_module(module_name)

    print('model to use: ', classificationOpt.use_model)

    #For datasets that have predefined train-test split:
    if classificationOpt.dataset_name != 'SHREC_Split10':
        test_opt = TestOptions().parse()
        test_opt.serial_batches = True
        test_opt.batch_size = 4
        test_dataset = DataLoader(test_opt)
        test_dataset_size = len(test_dataset)
        print('#test meshes = %d' % test_dataset_size)
        dataset.dataloaders['test'] = test_dataset.dataloaders['test']


    node_edge_feat_len = {'node': dataset.dataset.node_feat_len,
                          'edge': dataset.dataset.edge_feat_len}
    out_feat = 40
    # out_feat_pretrained = 40
    save_best_model_acc = SaveBestModelACC()
    save_best_model_loss = SaveBestModel()
    save_best_model_test_acc = SaveBestModelACC()
    save_best_model_test_loss = SaveBestModel()

    model = net.ClassificationNetwork(MeshConvLayers_spec=classificationOpt.MeshConv,
                                 graph_classifier_FCN_spec=classificationOpt.graph_classifier,
                                 out_feat=out_feat,
                                 graph_classifier_dropout=classificationOpt.graph_classifier_dropout,
                                 sbin = classificationOpt.sbin,
                                 gnn_dropout=classificationOpt.gnn_dropout).double().to(device)

    if classificationOpt.pretrained_model_file is not None:
        best_model_cp = torch.load(classificationOpt.pretrained_model_file)
        pretrained_dict = best_model_cp['model_state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    #
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters: ', pytorch_total_params)

    total_steps = 0

    loss_func = nn.CrossEntropyLoss()
    if classificationOpt.loss_mode == 'weighted':
        samples_per_class = torch.tensor(
            [dataset.dataset.num_samples_per_class_dict[i] for i in range(dataset.dataset.nclasses)])
        class_weight = 1 / samples_per_class
        class_weight_normalized = class_weight / class_weight.sum()
        class_weight_normalized = class_weight_normalized.double().to(device)
        loss_func = nn.CrossEntropyLoss(class_weight_normalized)

    if classificationOpt.pretrained_model_file is not None:
        optimizer = torch.optim.Adam(model.parameters(), classificationOpt.lr)
        optimizer.load_state_dict(best_model_cp['optimizer_state_dict'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), classificationOpt.lr)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00003, max_lr=0.0003)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    save_dir = os.path.join(classificationOpt.checkpoints_dir, classificationOpt.name)
    save_metric_dir = os.path.join(save_dir, 'metric')
    util.mkdir(save_metric_dir)
    epoch_train_losses = []
    epoch_test_losses = []
    train_accuracy = []
    test_accuracy = []
    train_correct_total = 0

    for epoch in range(2000):
        start = time.time()
        print(time.asctime(time.localtime(time.time())))
        model.train()
        epoch_loss = 0
        num_sample_total = 0
        train_correct_total = 0
        running_loss = 0
        running_correct_sample = 0
        running_total_sample = 0
        elapsed_time = 0

        # for i, (graph, graph_feat, label, name) in enumerate(dataset.dataloader):
        for i, (graph, graph_feat, label, name) in enumerate(dataset.dataloaders['train']):
            t0 = time.time()
            model.train()
            batch_loss = 0
            graph = graph.to(device)
            graph_feat = graph_feat.to(device)
            label = label.long().to(device)
            optimizer.zero_grad()
            train_prediction = model(graph, graph_feat)
            loss = loss_func(train_prediction, label)
            t2 = time.time()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            epoch_loss += float(loss)
            train_pred_class = train_prediction.data.max(1)[1]
            train_correct_total += train_pred_class.eq(label).sum().item()
            running_correct_sample += train_pred_class.eq(label).sum().item()
            running_total_sample += len(label)
            num_sample_total += len(label)

        epoch_loss /= (i + 1)
        epoch_train_losses.append(epoch_loss)
        epoch_train_accuracy = (train_correct_total / num_sample_total) * 100
        train_accuracy.append((train_correct_total / num_sample_total) * 100)

        test_correct_total = 0
        test_sample_total = 0
        test_loss = 0

        # for k, (graph, graph_feat, label, name) in enumerate(test_dataset.dataloader):
        for k, (graph, graph_feat, label, name) in enumerate(dataset.dataloaders['test']):
            #
            with torch.no_grad():
                model.eval()
                test_batch_loss = 0
                graph = graph.to(device)
                graph_feat = graph_feat.to(device)
                label = label.long().to(device)
                test_prediction = model(graph, graph_feat)
                loss = loss_func(test_prediction, label)
                test_loss = test_loss + float(loss)
                test_pred_class = test_prediction.data.max(1)[1]
                test_correct_total += test_pred_class.eq(label).sum().item()
                test_sample_total += len(label)

        epoch_test_accuracy = (test_correct_total / test_sample_total) * 100
        test_accuracy.append((test_correct_total / test_sample_total) * 100)
        test_loss /= (k + 1)
        epoch_test_losses.append(test_loss)

        print('Epoch {}, train loss {:.4f}, train accuracy {:.4f}, test accuracy {:.4f}'.
              format(epoch, epoch_loss, epoch_train_accuracy, epoch_test_accuracy))

        save_file_name_train = '%s_net_best_train_acc.pth' % epoch
        save_path_train = os.path.join(save_dir, save_file_name_train)
        save_file_name_test = '%s_net_best_test_acc.pth' % epoch
        save_path_test = os.path.join(save_dir, save_file_name_test)
        save_best_model_acc(epoch_train_accuracy, epoch, model, optimizer,loss_func, save_path_train)
        save_best_model_test_acc(epoch_test_accuracy, epoch, model, optimizer, loss_func, save_path_test)

        save_best_model_train_loss = '%s_net_best_train_loss.pth' % epoch
        save_path_train_loss = os.path.join(save_dir, save_best_model_train_loss)
        save_best_model_loss(epoch_loss, epoch, model, optimizer, loss_func, save_path_train_loss)

        save_file_test_loss = '%s_net_best_test_loss.pth' % epoch
        save_path_test_loss = os.path.join(save_dir, save_file_test_loss)
        save_best_model_test_loss(epoch_test_accuracy, epoch, model, optimizer, loss_func, save_path_test_loss)
