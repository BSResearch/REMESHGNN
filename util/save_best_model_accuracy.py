import torch
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class SaveBestModelACC:
    """
    Class to save the best model while training. If the current epoch's
    accuracy is more than the previous best acc, then save the
    model state.
    """

    def __init__(
            self, best_acc=0.0
    ):
        self.best_acc = best_acc

    def __call__(
            self, current_acc,
            epoch, model, optimizer, criterion, model_file_path
    ):
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            # print(f"Best training loss: {self.best_valid_loss}")
            # print(f"Saving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion.state_dict(),
                # 'center_optim_state_dict': optim_center.state_dict(),
                # 'lr_scheduler':scheduler.state_dict(),
                'accuracy': current_acc,
            }, model_file_path)