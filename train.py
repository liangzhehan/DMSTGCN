import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
import os
import json


def main(args):
    device = torch.device(args.device)
    config = json.load(open("data/config.json", "r"))[args.data]
    args.days = config["num_slots"]  # number of timeslots in a day which depends on the dataset
    args.num_nodes = config["num_nodes"]  # number of nodes
    args.normalization = config["normalization"]  # method of normalization which depends on the dataset
    args.data_dir = config["data_dir"]  # directory of data
    dataloader = util.load_dataset(args.data_dir, args.batch_size, args.batch_size, args.batch_size, days=args.days,
                                   sequence=args.seq_length, in_seq=args.in_len)
    scaler = dataloader['scaler']

    print(args)
    start_epoch = 1
    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout, args.normalization,
                     args.learning_rate, args.weight_decay, device, days=args.days, dims=args.dims, order=args.order)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    count = 0

    for i in range(start_epoch, args.epochs + 1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        tt1 = time.time()
        dataloader['train_loader'].shuffle()
        for itera, (x, y, ind) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :], ind)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if itera % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(itera, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        tt2 = time.time()
        train_time.append(tt2 - tt1)
        # validate
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for itera, (x, y, ind) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :], ind)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        # early stopping
        if len(his_loss) > 0 and mvalid_loss < np.min(his_loss):
            count = 0
        else:
            count += 1
            print(f"no improve for {count} epochs")
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f},' \
              ' Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (tt2 - tt1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   os.path.join(args.save, "epoch_" + str(i) + "_" + str(round(float(mvalid_loss), 2)) + ".pth"))

        # test
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx, ind)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze())
        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]
        for i in [2, 5, 11]:
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            metrics = util.metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d},' \
                  ' Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        if count >= 30:
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # final test
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        os.path.join(args.save, "epoch_" + str(bestid + start_epoch)
                     + "_" + str(round(float(his_loss[int(bestid)]), 2)) + ".pth")))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, ind)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(float(his_loss[int(bestid)]), 4)))

    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d},' \
              ' Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(),
               os.path.join(args.save, "exp" + str(args.expid) +
                            "_best_" + str(round(float(his_loss[int(bestid)]), 2)) + ".pth"))
    return np.asarray(amae), np.asarray(amape), np.asarray(armse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='PEMSD4', help='data path')
    parser.add_argument('--seq_length', type=int, default=12, help='output length')
    parser.add_argument('--in_len', type=int, default=12, help='input length')
    parser.add_argument('--nhid', type=int, default=32, help='')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=200, help='')
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--runs', type=int, default=1, help='number of experiments')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--iden', type=str, default='', help='identity')
    parser.add_argument('--dims', type=int, default=32, help='dimension of embeddings for dynamic graph')
    parser.add_argument('--order', type=int, default=2, help='order of graph convolution')

    args = parser.parse_args()
    args.save = os.path.join('save_models/', os.path.basename(args.data) + args.iden)
    os.makedirs(args.save, exist_ok=True)
    t1 = time.time()
    metric = []

    for i in range(args.runs):
        args.expid = i
        metric.append(main(args))
        t2 = time.time()
        print("Total time spent: {:.4f}".format(t2 - t1))
    metric = np.asarray(metric)
    print(metric)  # 5 3 12
    for i in range(args.seq_length):
        print(f"mae for step{i + 1}: {np.mean(metric[:, 0, i])}±{np.std(metric[:, 0, i])}")
        print(f"mape for step{i + 1}: {np.mean(metric[:, 1, i])}±{np.std(metric[:, 1, i])}")
        print(f"rmse for step{i + 1}: {np.mean(metric[:, 2, i])}±{np.std(metric[:, 2, i])}")
    print(f"mean of best mae: {np.mean(metric[:, 0])}±{np.std(np.mean(metric[:, 0], axis=1))}")
    print(f"mean of best mape: {np.mean(metric[:, 1])}±{np.std(np.mean(metric[:, 1], axis=1))}")
    print(f"mean of best rmse: {np.mean(metric[:, 2])}±{np.std(np.mean(metric[:, 2], axis=1))}")
