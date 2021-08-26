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
parser.add_argument('--learned_lr', default=1, type=int, help="0:constant inner learning rate, 1: lr per layer as a meta learned parameter")
parser.add_argument('--inner_n_steps', default=1, type=int, help="number of inner adaptation steps")

parser.add_argument('--t_steps', default=4, type=int, help="predict t steps in the future")
parser.add_argument('--n_neurons', default=32, type=int, help="number of neurons in hidden layers")
parser.add_argument('--n_layers', default=1, type=int, help="0: 4 hidden layers, 1: 6 hidden layers")

parser.add_argument('--subset', default=1, type=int, help="0: use full dataset, 1: use all objects domains")

parser.add_argument('--dlo_only', default=1, type=int, help="0: predict all the objects, 1: predict only deformable linear object")
parser.add_argument('--obj_only', default=0, type=int, help="0: predict all the objects, 1: predict only rigid objects")
parser.add_argument('--obj_input', default=0, type=int, help='0: remove obj form input when prediction is dlo:1 and restrict to 0 objects')

parser.add_argument('--epochs', default=1000000, type=int, help="number of epochs")
parser.add_argument('--N', default=50, type=int, help="number of tasks per batch")
parser.add_argument('--K', default=50, type=int, help="number of adapatation trajectories per task")
parser.add_argument('--test_split', default=0.1, type=float, help="ratio of test data")
parser.add_argument('--seed', default=1234, type=int, help="seed")
parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
parser.add_argument('--epochs_test', default=100, type=int, help="number of epochs before testing")


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
    log_name += '_adapt_lr=' + str(params['learned_lr'])
    log_name += '_t_steps=' + str(params['t_steps'])
    log_name += '_neurons=' + str(params['n_neurons'])
    log_name += '_layers=' + str(params['n_layers'])
    log_name += '_N=' + str(params['N'])
    log_name += '_K=' + str(params['K'])
    log_name += '_lr=' + str(params['lr'])
    log_name += '_split=' + str(params['test_split']) + "_last_a_new_data"

    log_name = './log_dlo/final_code'#last_run_inner_steps='+str(params['inner_n_steps'])+"_reshuffle2"
    writer = SummaryWriter(log_dir=log_name)

    save_model_path = "./models/"+log_name+".pt"

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    L_train = 0
    L_inner = 0

    train_errs = np.zeros(4)

    for epoch in tqdm(range(params['epochs'])):

        if epoch == 999990:
            print()

        x_s, a_s, y_s, x_q, a_q, y_q = dataloader.get_batch('train', params['N'], params['K'])
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
            if params['dlo_only'] == 0 and params['obj_only'] == 0:
                train_errs += errs

        optimizer.zero_grad()
        L_tot.backward()
        optimizer.step()

        L_train += L_tot

        if epoch % params['epochs_test'] == params['epochs_test']-1:
            x_s, a_s, y_s, x_q, a_q, y_q = dataloader.get_batch('test', params['N'], params['K'])
            x_s, a_s, y_s = x_s.to(device), a_s.to(device), y_s.to(device)
            if params['maml'] == 1:
                x_q, a_q, y_q = x_q.to(device), a_q.to(device), y_q.to(device)
            model.eval()
            L_test = 0
            L_inner_test = 0
            test_errs = np.zeros(4)
            for i in range(x_s.shape[0]):
                if params['maml'] == 1:
                    theta_i, loss_inner = model.adapt(x_s[i], a_s[i], y_s[i], train=False)
                    L, errs = model.get_loss(x_q[i], a_q[i], y_q[i], theta_i)
                    L_inner_test += loss_inner
                else:
                    L, errs = model.get_loss(x_s[i], a_s[i], y_s[i])
                L_test += L
                if params['dlo_only'] == 0 and params['obj_only'] == 0:
                    test_errs += errs
            writer.add_scalar("L_test", L_test/params['N'], int(epoch / params['epochs_test']))
            writer.add_scalar("L_inner_test", L_inner_test / params['N'], int(epoch / params['epochs_test']))
            writer.add_scalar("L_train", L_train / (params['N']*params['epochs_test']), int(epoch / params['epochs_test']))
            writer.add_scalar("L_inner_train", L_inner / (params['N'] * params['epochs_test']), int(epoch / params['epochs_test']))

            if params['maml'] == 1:
                for j, grad in enumerate(model.grads_vals):
                    writer.add_scalar('params_grad_' + str(j), grad / (params['N'] * params['epochs_test']), model.log_grads_idx)
                model.grads_vals *= 0
                model.log_grads_idx += 1

            if params['dlo_only'] == 0 and params['obj_only'] == 0:
                for log_idx, name in enumerate(['push_err', 'dlo_err', 'obj_pos_err', 'obj_or_err']):
                    writer.add_scalar(name + "_train", train_errs[log_idx] / (params['N'] * params['epochs_test']), int(epoch / params['epochs_test']))
                    writer.add_scalar(name + "_test", test_errs[log_idx] / (params['N']), int(epoch / params['epochs_test']))

            L_train = 0
            L_inner = 0
            train_errs = np.zeros(4)

        if epoch % 10000 == 9999:
            x_s, a_s, y_s, x_q, a_q, y_q = dataloader.get_domain(params['K'])
            if params['maml'] == 1:
                y_hat_prior = model.model_theta.forward(x_q[0].to(device), a_q[0].to(device))
                y_hat_prior = y_hat_prior[-10:].detach().cpu().numpy()
                theta_i, _ = model.adapt(x_s[0].to(device), a_s[0].to(device), y_s[0].to(device), train=False)
                y_hat = model.model_theta.forward(x_q[0].to(device), a_q[0].to(device), theta_i)
                y = y_q[0, -10:].detach().cpu().numpy()
                x = x_q[0, -10:].detach().cpu().numpy()
            else:
                y_hat = model.model_theta.forward(x_s[0].to(device), a_s[0].to(device))
                y = y_s[0, -10:].detach().cpu().numpy()
                x = x_s[0, -10:].detach().cpu().numpy()

            y_hat = y_hat[-10:].detach().cpu().numpy()

            t = 5

            if params['dlo_only'] == 0 and params['obj_only'] == 0:
                fig = plt.figure()
                plt.plot(x[t, 0], x[t, 1], 'bo', alpha=0.3)
                plt.plot(y[t, 0], y[t, 1], 'bo')
                plt.plot(y_hat[t, 0], y_hat[t, 1], 'bs')
                for dlo in range(32):
                    plt.plot(x[t, dlo * 2 + 2], x[t, dlo * 2 + 3], 'ro', alpha=0.3)
                    plt.plot(y[t, dlo * 2 + 2], y[t, dlo * 2 + 3], 'ro')
                    plt.plot(y_hat[t, dlo * 2 + 2], y_hat[t, dlo * 2 + 3], 'rs')
                for obj in range(3):
                    plt.plot(x[t, obj * 2 + 66], x[t, obj * 2 + 67], 'go', alpha=0.3)
                    plt.plot(y[t, obj * 2 + 66], y[t, obj * 2 + 67], 'go')
                    plt.plot(y_hat[t, obj * 2 + 66], y_hat[t, obj * 2 + 67], 'gs')
            else:
                fig = plt.figure()
                plt.plot(x[t, 0], x[t, 1], 'bo')
                if params['dlo_only'] == 1:
                    segments = 32
                    offset = 2
                else:
                    segments = 3
                    offset = 2 + 32 * 2
                for dlo in range(segments):
                    plt.plot(x[t, dlo * 2 + offset], x[t, dlo * 2 + offset + 1], 'ro', alpha=0.3)
                    plt.plot(y[t, dlo * 2], y[t, dlo * 2 + 1], 'ro')
                    plt.plot(y_hat[t, dlo * 2], y_hat[t, dlo * 2 + 1], 'rs')
                    # TODO: extend to all non dlo_only also?
                    if params['maml'] == 1:
                        plt.plot(y_hat_prior[t, dlo * 2], y_hat_prior[t, dlo * 2 + 1], 'rs', alpha=0.3)
            writer.add_figure("pred", fig, int(epoch / 10000))

            torch.save(model, save_model_path)

    writer.close()


if __name__ == '__main__':

    args = parser.parse_args()
    params = args.__dict__

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if params['dlo_only'] == params['obj_only']:
        raise Exception('Cannot predict jus objects and just rope at the same time man...')

    dataloader = PushingDataset(params)

    main(params, dataloader, device)




























