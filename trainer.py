import torch
import numpy as np
import time
from datetime import timedelta
from matplotlib import pyplot as plt
# import wandb


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        measure_weights=False, start_epoch=0, save_progress_path=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    train_losses = []
    val_losses = []
    for epoch in range(start_epoch, n_epochs):
        print("Starting Epoch", epoch)
        scheduler.step()

        if measure_weights:
            fc_weights = model.module.embedding_net.fc.weight.cpu().data.numpy()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, measure_weights)
        train_losses.append(train_loss)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        if measure_weights:
            new_fc_weights = model.module.embedding_net.fc.weight.cpu().data.numpy()
            fc_diff = np.abs(new_fc_weights - fc_weights).sum()
            fc_average = np.abs(new_fc_weights).mean()
            fc_total = np.abs(new_fc_weights).sum()
            message += f'\tFCWeights (Diff, Avg, Total): ({fc_diff}, {fc_average}, {fc_total})'

        if val_loader is not None:
            val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            # wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        # else:
        #     wandb.log({'epoch': epoch, 'loss': train_loss})

        print(message)

        if save_progress_path is not None:
            if len(val_losses) <= 1 or val_losses[-1] > np.max(np.array(val_losses[:-1])):
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(state, save_progress_path + rf"\training_state_{epoch}.pth")

                with open(save_progress_path + "/progress.txt", "a") as progres_file:
                    progres_file.write(message + "\n\n")

        if epoch > 0:
            plt.plot(train_losses, color='orange', label='train_loss')
            plt.plot(val_losses, color='green', label='val_loss')
            plt.title("Loss progression")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    if save_progress_path is not None:
        plt.plot(train_losses, color='orange', label='train_loss')
        plt.plot(val_losses, color='green', label='val_loss')
        plt.axvline(np.argmin(np.array(val_losses)), color='red')
        plt.title("Loss progression")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(save_progress_path + r"\loss_progress.png")


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, measure_weights):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    start_time = time.time()

    print("Will sample from train_loader")
    for batch_idx, data in enumerate(train_loader):
        # print("batch_idx", batch_idx, "data", data.shape, data.type())
        data, multisiamese_mode, matrix_a, matrix_b = reformat_data(data, cuda)

        if measure_weights:
            fc_weights = model.module.embedding_net.fc.weight.cpu().data.numpy()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs

        if multisiamese_mode == 'hard':
            positive_matrix = matrix_a
            negative_matrix = matrix_b
            loss_outputs = loss_fn(*loss_inputs, positive_matrix, negative_matrix)
        elif multisiamese_mode == 'soft':
            similarity_matrix = matrix_a
            masks = matrix_b
            loss_outputs = loss_fn(*loss_inputs, similarity_matrix, masks)
        else:
            loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, loss_outputs)

        if batch_idx > 0 and batch_idx % log_interval == 0:
            elapsed_time = time.time() - start_time
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tElapsed time: {}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses), str(timedelta(seconds=elapsed_time)))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            if measure_weights:
                new_fc_weights = model.module.embedding_net.fc.weight.cpu().data.numpy()
                fc_diff = np.abs(new_fc_weights - fc_weights).sum()
                fc_average = np.abs(new_fc_weights).mean()
                fc_total = np.abs(new_fc_weights).sum()
                message += f'\tFCWeights (Diff, Avg, Total): ({fc_diff}, {fc_average}, {fc_total})'

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, data in enumerate(val_loader):
            data, multisiamese_mode, matrix_a, matrix_b = reformat_data(data, cuda)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs

            if multisiamese_mode == 'hard':
                positive_matrix = matrix_a
                negative_matrix = matrix_b
                loss_outputs = loss_fn(*loss_inputs, positive_matrix, negative_matrix)
            elif multisiamese_mode == 'soft':
                similarity_matrix = matrix_a
                masks = matrix_b
                loss_outputs = loss_fn(*loss_inputs, similarity_matrix, masks)
            else:
                loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, loss_outputs)

    return val_loss, metrics


def reformat_data(data, cuda):
    multisiamese_mode = None
    matrix_a = matrix_b = None
    if not type(data) in (tuple, list):
        data = (data,)
    elif len(data) == 3:
        # print(data)
        # print(len(data[0].shape))
        if type(data[-1]) is dict:
            multisiamese_mode = 'soft'
            similarity_matrix = data[1][0]
            masks = data[1][1]
            matrix_a = similarity_matrix
            matrix_b = masks
            data = data[0]
        elif len(data[0].shape) == 4:  # data = (triplet, batch, channels, width, height)
            # We want (batch, triplet, channels, width, height)
            data = torch.stack(data)
            data = data.permute(1, 0, *list(range(2, len(data.shape))))
            channels = data.shape[2]
            if channels == 1:
                data = data.repeat(1, 1, 3, 1, 1)
            data = (data,)

        elif len(data[0].shape) == 5:  # data = (sequences, positive_matrix, negative_matrix)
            multisiamese_mode = 'hard'
            positive_matrix = data[1]
            negative_matrix = data[2]
            matrix_a = positive_matrix
            matrix_b = negative_matrix
            data = data[0]
    if cuda:
        data = tuple(d.cuda() for d in data)

    return data, multisiamese_mode, matrix_a, matrix_b
