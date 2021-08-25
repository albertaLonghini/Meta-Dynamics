from dataloader import PushingDataset
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from meta_predictor import maml
from standard_predictor import regressor
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()

parser.add_argument('--maml', default=1, type=int, help="use meta learning")

parser.add_argument('--t_steps', default=4, type=int, help="predict t steps in the future")
parser.add_argument('--n_neurons', default=128, type=int, help="number of neurons in hidden layers")
parser.add_argument('--n_layers', default=1, type=int, help="0: 4 hidden layers, 1: 6 hidden layers")

parser.add_argument('--subset', default=1, type=int, help="0: use full dataset, 1: use all objects domains")

parser.add_argument('--dlo_only', default=1, type=int, help="0: predict all the objects, 1: predict only deformable linear object")

parser.add_argument('--epochs', default=200000, type=int, help="number of epochs")
parser.add_argument('--N', default=10, type=int, help="number of tasks per batch")
parser.add_argument('--K', default=10, type=int, help="number of adapatation trajectories per task")
parser.add_argument('--test_split', default=0.1, type=float, help="ratio of test data")
parser.add_argument('--seed', default=1234, type=int, help="seed")
parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
parser.add_argument('--epochs_test', default=10, type=int, help="number of epochs before testing")


def main(params, dataloader, device):

    if params['maml'] == 1:
        model = maml(params).to(device)
    else:
        model = regressor(params).to(device)

    #log_name = './logs/base_predictor_t_steps='+params['t_steps']+'_maml='+str(params['maml'])
    log_name = './log_gridsearch/'
    if params['dlo_only'] == 1:
        log_name = './log_dlo/'
    log_name += 'subset=' + str(params['subset'])
    log_name += '_maml=' + str(params['maml'])
    log_name += '_t_steps=' + str(params['t_steps'])
    log_name += '_neurons=' + str(params['n_neurons'])
    log_name += '_layers=' + str(params['n_layers'])
    log_name += '_N=' + str(params['N'])
    log_name += '_K=' + str(params['K'])
    log_name += '_lr=' + str(params['lr'])
    log_name += '_split=' + str(params['test_split'])
    writer = SummaryWriter(log_dir=log_name)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    L_train = 0
    L_inner = 0

    train_errs = np.zeros(4)

    for epoch in range(params['epochs']):

        x_s, y_s, a_s, x_q, y_q, a_q = dataloader.get_batch('train', params['N'], params['K'])
        x_s, a_s, y_s = x_s.to(device), a_s.to(device), y_s.to(device)
        if params['maml'] == 1:
            x_q, a_q, y_q = x_q.to(device), a_q.to(device), y_q.to(device)

        theta_i_list = []
        L_tot = 0

        model.train()

        for i in range(params['N']):

            if params['maml'] == 1:

                theta_i, loss_inner = model.adapt(x_s[i], a_s[i], y_s[i], train=True)
                theta_i_list.append(theta_i)
                L, errs = model.get_loss(x_q[i], a_q[i], y_q[i], theta_i)

                L_inner += loss_inner

            else:

                L, errs = model.get_loss(x_s[i], a_s[i], y_s[i])

            L_tot += L
            if params['dlo_only'] == 0:
                train_errs += errs

        optimizer.zero_grad()
        L_tot.backward()
        optimizer.step()

        L_train += L_tot

        if epoch % params['epochs_test'] == params['epochs_test']-1:
            x_s, y_s, a_s, x_q, y_q, a_q = dataloader.get_batch('test', params['N'], params['K'])
            x_s, a_s, y_s = x_s.to(device), a_s.to(device), y_s.to(device)
            if params['maml'] == 1:
                x_q, a_q, y_q = x_q.to(device), a_q.to(device), y_q.to(device)
            model.eval()
            L_test = 0
            L_inner_test = 0
            test_errs = np.zeros(4)
            for i in range(params['N']):
                if params['maml'] == 1:
                    theta_i, loss_inner = model.adapt(x_s[i], a_s[i], y_s[i], train=False)
                    L, errs = model.get_loss(x_q[i], a_q[i], y_q[i], theta_i)
                    L_inner_test += loss_inner
                else:
                    L, errs = model.get_loss(x_s[i], a_s[i], y_s[i])
                L_test += L
                if params['dlo_only'] == 0:
                    test_errs += errs
            writer.add_scalar("L_test", L_test/params['N'], int(epoch / params['epochs_test']))
            writer.add_scalar("L_inner_test", L_inner_test / params['N'], int(epoch / params['epochs_test']))
            writer.add_scalar("L_train", L_train / (params['N']*params['epochs_test']), int(epoch / params['epochs_test']))
            writer.add_scalar("L_inner_train", L_inner / (params['N'] * params['epochs_test']), int(epoch / params['epochs_test']))

            if params['dlo_only'] == 0:
                for log_idx, name in enumerate(['push_err', 'dlo_err', 'obj_pos_err', 'obj_or_err']):
                    writer.add_scalar(name + "_train", train_errs[log_idx] / (params['N'] * params['epochs_test']), int(epoch / params['epochs_test']))
                    writer.add_scalar(name + "_test", test_errs[log_idx] / (params['N']), int(epoch / params['epochs_test']))

            L_train = 0
            L_inner = 0
            train_errs = np.zeros(4)

        if epoch % 5000 == 4999:
        #if epoch % 5000 == 0:
            x_s, y_s, a_s, x_q, y_q, a_q = dataloader.get_domain(params['K'])
            if params['maml'] == 1:
                theta_i, _ = model.adapt(x_s[0].to(device), a_s[0].to(device), y_s[0].to(device), train=False)
                y_hat = model.model_theta.forward(x_q[0].to(device), a_q[0].to(device), theta_i)
                y = y_q[0, -10:].detach().cpu().numpy()
            else:
                y_hat = model.model_theta.forward(x_s[0].to(device), a_s[0].to(device))
                y = y_s[0, -10:].detach().cpu().numpy()

            y_hat = y_hat[-10:].detach().cpu().numpy()

            t = 5

            if params['dlo_only'] == 0:
                fig = plt.figure()
                plt.plot(y[t, 0], y[t, 1], 'bo', alpha=0.3)
                plt.plot(y_hat[t, 0], y_hat[t, 1], 'bo')
                for dlo in range(32):
                    plt.plot(y[t, dlo * 2 + 2], y[t, dlo * 2 + 3], 'ro', alpha=0.3)
                    plt.plot(y_hat[t, dlo * 2 + 2], y_hat[t, dlo * 2 + 3], 'ro')
                for obj in range(3):
                    plt.plot(y[t, obj * 2 + 66], y[t, obj * 2 + 67], c='blue', alpha=0.3)
                    plt.plot(y_hat[t, obj * 2 + 66], y_hat[t, obj * 2 + 67], c='blue')
            else:
                fig = plt.figure()
                for dlo in range(32):
                    plt.plot(y[t, dlo * 2], y[t, dlo * 2 + 1], 'ro', alpha=0.3)
                    plt.plot(y_hat[t, dlo * 2], y_hat[t, dlo * 2 + 1], 'ro')
            writer.add_figure("pred", fig, int(epoch / 5000))

            print()

    writer.close()


if __name__ == '__main__':

    args = parser.parse_args()
    params = args.__dict__

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataloader = PushingDataset(params['test_split'], params['maml'], params['K'], params['t_steps'], params['subset'], params['dlo_only'])

    main(params, dataloader, device)
