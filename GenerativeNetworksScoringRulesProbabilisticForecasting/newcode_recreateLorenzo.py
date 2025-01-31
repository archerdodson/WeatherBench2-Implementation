import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset
# import and set up the typeguard
# from typeguard.importhook import install_import_hook

# # comment these out when deploying:
# install_import_hook('src.nn')
# install_import_hook('src.scoring_rules')
# install_import_hook('src.utils')
# install_import_hook('src.weatherbench_utils')
# install_import_hook('src.unet_utils')

from src.nn import InputTargetDataset, UNet2D, fit, fit_adversarial, \
    ConditionalGenerativeModel, createGenerativeFCNN, createCriticFCNN, test_epoch, PatchGANDiscriminator, \
    DiscardWindowSizeDim, get_target, LayerNormMine, createGenerativeGRUNN, createCriticGRUNN, \
    DiscardNumberGenerationsInOutput, createGRUNN, createFCNN
from src.scoring_rules import EnergyScore, KernelScore, VariogramScore, PatchedScoringRule, SumScoringRules, \
    ScoringRulesForWeatherBench, ScoringRulesForWeatherBenchPatched, LossForWeatherBenchPatched
from src.utils import plot_losses, save_net, save_dict_to_json, estimate_bandwidth_timeseries, lorenz96_variogram, \
    def_loader_kwargs, load_net, weight_for_summed_score, weatherbench_variogram_haversine
from src.parsers import parser_train_net, define_masks, nonlinearities_dict, setup
from src.weatherbench_utils import load_weatherbench_data


#####################################################################
batch_size, ensemble_size, method, nn_model, data_size, auxiliary_var_size = 10, 5, 'SR', 'fcnn', 1, 1
hidden_size_rnn = 8
nonlinearities_dict, nonlinearity = {"relu": torch.nn.functional.relu, "tanhshrink": torch.nn.functional.tanhshrink,
                       "leaky_relu": torch.nn.functional.leaky_relu}, 'leaky_relu'

args_dict = {}
weight_decay, scheduler_gamma, lr, epochs, early_stopping, epochs_early_stopping_interval = 0.0, 1.0, 0.01, 3, False, 10

model, scoring_rule = 'lorenz63', 'energy'
cuda, continue_training_net, start_epoch_early_stopping, use_tqdm, method_is_gan  = False, False, 10, True, False

base_measure, seed = 'normal', 0

datasets_folder = 'results/lorenz/datasets/window10_original_lorenz63/'
model_is_weatherbench = False

# --- data handling ---
if not model_is_weatherbench:
    input_data_train = torch.load(datasets_folder + "train_x.pty")[:50]
    target_data_train = torch.load(datasets_folder + "train_y.pty")[:50]
    input_data_val = torch.load(datasets_folder + "val_x.pty")[:30]
    target_data_val = torch.load(datasets_folder + "val_y.pty")[:30]

    window_size = input_data_train.shape[1]

    # create the train and val loaders:
    dataset_train = InputTargetDataset(input_data_train, target_data_train, "cpu")#,
                                      # "cuda" if cuda and load_all_data_GPU else "cpu")
    dataset_val = InputTargetDataset(input_data_val, target_data_val, "cpu")
                                     #, "cuda" if cuda and load_all_data_GPU else "cpu")
else:
    dataset_train, dataset_val = load_weatherbench_data(weatherbench_data_folder, cuda, load_all_data_GPU,
                                                        weatherbench_small=weatherbench_small)
    len_dataset_train = len(dataset_train)
    len_dataset_val = len(dataset_val)
    print("Training set size:", len_dataset_train)
    print("Validation set size:", len_dataset_val)
    args_dict["len_dataset_train"] = len_dataset_train
    args_dict["len_dataset_val"] = len_dataset_val

# loader_kwargs = def_loader_kwargs(cuda, load_all_data_GPU)

# loader_kwargs.update(loader_kwargs_2)  # if you want to add other loader arguments

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)#, **loader_kwargs)
if len(dataset_val) > 0:
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False) #, **loader_kwargs)
    if model_is_weatherbench:
        # obtain the target tensor to estimate the gamma for kernel SR:
        target_data_val = get_target(data_loader_val, cuda).flatten(1, -1)
else:
    data_loader_val = None

print('data loader train', data_loader_train)

# --- loss function ---
sr_instance = EnergyScore()
loss_fn = sr_instance.estimate_score_batch


# --- defining the model using a generative net ---


wrap_net = True
number_generations_per_forward_call = ensemble_size if method == "SR" else 1
# create generative net:
if nn_model == "fcnn":
    input_size = window_size * data_size + auxiliary_var_size
    output_size = data_size
    hidden_sizes_list = [int(input_size * 1.5), int(input_size * 3), int(input_size * 3),
                            int(input_size * 0.75 + output_size * 3), int(output_size * 5)]
    inner_net = createGenerativeFCNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes_list,
                                        nonlinearity=nonlinearities_dict[nonlinearity])()
elif nn_model == "rnn":
    output_size = data_size
    gru_layers = 1
    gru_hidden_size = hidden_size_rnn
    inner_net = createGenerativeGRUNN(data_size=data_size, gru_hidden_size=gru_hidden_size,
                                        noise_size=auxiliary_var_size,
                                        output_size=output_size, hidden_sizes=None, gru_layers=gru_layers,
                                        nonlinearity=nonlinearities_dict[nonlinearity])()
elif nn_model == "unet":
    # select the noise method here:
    inner_net = UNet2D(in_channels=data_size[0], out_channels=1, noise_method=unet_noise_method,
                        number_generations_per_forward_call=number_generations_per_forward_call,
                        conv_depths=unet_depths)
    if unet_noise_method in ["sum", "concat"]:
        # here we overwrite the auxiliary_var_size above, as there is a precise constraint
        downsampling_factor, n_channels = inner_net.calculate_downsampling_factor()
        if weatherbench_small:
            auxiliary_var_size = torch.Size(
                [n_channels, 16 // downsampling_factor, 16 // downsampling_factor])
        else:
            auxiliary_var_size = torch.Size(
                [n_channels, data_size[1] // downsampling_factor, data_size[2] // downsampling_factor])
    elif unet_noise_method == "dropout":
        wrap_net = False  # do not wrap in the conditional generative model

if wrap_net:
    # the following wraps the nets above and takes care of generating the auxiliary variables at each forward call
    if continue_training_net:
        net = load_net(nets_folder + f"net{name_postfix}.pth", ConditionalGenerativeModel, inner_net,
                        size_auxiliary_variable=auxiliary_var_size, base_measure=base_measure,
                        number_generations_per_forward_call=number_generations_per_forward_call, seed=seed + 1)
    else:
        net = ConditionalGenerativeModel(inner_net, size_auxiliary_variable=auxiliary_var_size, seed=seed + 1,
                                            number_generations_per_forward_call=number_generations_per_forward_call,
                                            base_measure=base_measure)
else:
    if continue_training_net:
        net = load_net(nets_folder + f"net{name_postfix}.pth", DiscardWindowSizeDim, inner_net)
    else:
        net = DiscardWindowSizeDim(inner_net)


# --- network tools ---
# if cuda:
#     net.cuda()

# optimizer

optimizer_kwargs = {"weight_decay": weight_decay}  # l2 regularization
args_dict["weight_decay"] = optimizer_kwargs["weight_decay"]
optimizer = Adam(net.parameters(), lr=lr, **optimizer_kwargs)

# scheduler
scheduler_steps = 10
scheduler_gamma = scheduler_gamma
scheduler = lr_scheduler.StepLR(optimizer, scheduler_steps, gamma=scheduler_gamma, last_epoch=-1)
args_dict["scheduler_steps"] = scheduler_steps
args_dict["scheduler_gamma"] = scheduler_gamma

if method_is_gan:
    if cuda:
        critic.cuda()
    optimizer_kwargs = {}
    optimizer_c = Adam(critic.parameters(), lr=lr_c, **optimizer_kwargs)
    # dummy scheduler:
    scheduler_c = lr_scheduler.StepLR(optimizer_c, 8, gamma=1, last_epoch=-1)

string = f"Train {method} network for {model} model with lr {lr} "
if method == "SR":
    string += f"using {scoring_rule} scoring rule"
if method_is_gan:
    string += f"and critic lr {lr_c}"
print(string)

# --- train ---
start = time()
if method_is_gan:
    # load the previous losses if available:
    if continue_training_net:
        train_loss_list_g = np.load(nets_folder + f"train_loss_g{name_postfix}.npy").tolist()
        train_loss_list_c = np.load(nets_folder + f"train_loss_c{name_postfix}.npy").tolist()
    else:
        train_loss_list_g = train_loss_list_c = None
    kwargs = {}
    if method == "WGAN_GP":
        kwargs["lambda_gp"] = lambda_gp
    train_loss_list_g, train_loss_list_c = fit_adversarial(method, data_loader_train, net, critic, optimizer, scheduler,
                                                           optimizer_c, scheduler_c, epochs, cuda,
                                                           start_epoch_training=0, use_tqdm=use_tqdm,
                                                           critic_steps_every_generator_step=
                                                           critic_steps_every_generator_step,
                                                           train_loss_list_g=train_loss_list_g,
                                                           train_loss_list_c=train_loss_list_c, **kwargs)
else:
    if continue_training_net:
        train_loss_list = np.load(nets_folder + f"train_loss{name_postfix}.npy").tolist()
        val_loss_list = np.load(nets_folder + f"val_loss{name_postfix}.npy").tolist()
    else:
        train_loss_list = val_loss_list = None
    train_loss_list, val_loss_list = fit(data_loader_train, net, loss_fn, optimizer, scheduler, epochs, cuda,
                                         val_loader=data_loader_val, early_stopping=early_stopping,
                                         start_epoch_early_stopping=0 if continue_training_net else start_epoch_early_stopping,
                                         epochs_early_stopping_interval=epochs_early_stopping_interval,
                                         start_epoch_training=0, use_tqdm=use_tqdm, train_loss_list=train_loss_list,
                                         test_loss_list=val_loss_list)
    # compute now the final validation loss achieved by the model; it is repetition from what done before but easier
    # to do this way
    final_validation_loss = test_epoch(data_loader_val, net, loss_fn, cuda)

training_time = time() - start

print(f"Training time: {training_time:.2f} seconds")
# print('train_loss_list', train_loss_list)
# plt.plot(train_loss_list)
# plt.title('train_loss_list, RNN Lorenz63 w=10, l=1')
# plt.show()
# plt.close()

# # print('val_loss_list', val_loss_list)
# plt.plot(val_loss_list)
# plt.title('val_loss_list, RNN Lorenz63 w=10, l=1')
# plt.show()
# plt.close()


# batch_size, window_size, data_size, number_generations, size_auxiliary_variable = 4, 3, 1, 5, 1
# input_size = window_size * data_size + size_auxiliary_variable
# output_size = data_size
# myfcnn = createGenerativeFCNN(input_size, output_size)()
# my_pred = myfcnn(torch.randn(batch_size, window_size, data_size), torch.randn(batch_size, number_generations, size_auxiliary_variable))
# print(my_pred.shape)
# print(my_pred)
