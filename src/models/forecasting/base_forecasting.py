from cv2 import reduce
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import wandb

import matplotlib.pyplot as plt


class BaseForecasting(pl.LightningModule):
    """ Basic neural network class to implement methods all havin in common. """

    def __init__(self, hparams, example_batch):
        """ Initialize dataset, data loader, network dimensions, and torch layers. """
        super().__init__()
        self.example_batch = example_batch
        self.save_hyperparameters(hparams)
        self._build()

    def _build(self):
        example_energy = self.example_batch[0][0]
        example_weather = self.example_batch[0][1]
        example_calendar = self.example_batch[0][2]

        self._build_energy_net(example_energy)
        self._build_weather_net(example_weather)
        self._build_calendar_net(example_calendar)

        energy_latent = self._forward_energy_batch(example_energy)
        weather_latent = self._forward_weather_batch(example_weather)
        example_calendar = self._forward_calendar_batch(example_calendar)
        num_energy_features = energy_latent.shape[-1]
        num_weather_features = weather_latent.shape[-1]
        num_calendar_features = example_calendar.shape[-1]
        latent_size = num_energy_features + num_weather_features + num_calendar_features

        self._fc_sequential = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.hparams.forecast_horizon),
        )

        num_params = sum(p.numel() for p in self.parameters())

        wandb.log({
            'num_params': num_params,
            'energy_input_shape': example_energy.shape[1:],
            'weather_input_shape': example_weather.shape[1:],
            'calendar_input_shape': example_calendar.shape[1:],
            'energy_latent_shape': energy_latent.shape[1:],
            'weather_latent_shape': weather_latent.shape[1:],
            'calendar_latent_shape': example_calendar.shape[1:]
        })
        print(
        f"""
        Model Complexity: {num_params}
        Input Shape:
            Energy - {example_energy.shape[1:]}
            Weather - {example_weather.shape[1:]}
            Calendar - {example_calendar.shape[1:]}
        
        Latent Shape: Sum {latent_size}
            Energy - {num_energy_features}
            Weather - {num_weather_features}
            Calendar - {num_calendar_features}
        """
        )

    def forward(self, x: torch.Tensor):
        """ Inference of the neural network model. """
        energy, weather, calendar = x

        feature_set = list()
        feature_set.append(self._forward_energy_batch(energy))
        feature_set.append(self._forward_weather_batch(weather))
        feature_set.append(self._forward_calendar_batch(calendar))
        collected_features = torch.cat(feature_set, axis=1)

        return self._fc_sequential(collected_features)

    def loss(self, y, y_hat):
        loss = getattr(F, self.hparams.loss)
        # error = loss(y, y_hat, reduction='none').mean(axis=0)
        # error = torch.log(1 + error).mean()
        # return error
        return loss(y, y_hat)

    def configure_optimizers(self):
        """ Configure optimizers and scheduler to return for pytorch_lightning. """
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr, **self.hparams.optimizer_params
        )

        if self.hparams.scheduler is None:
            return optimizer
        else:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
                optimizer,
                **(self.hparams.scheduler_params)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.hparams.monitor
            }

    def training_step(self, batch, batch_idx):
        """ Perform training stel given a batch. """
        x, y = batch
        y_hat = self(x)  # self.forward(..)

        # return loss
        loss = self.loss(y_hat, y)
        return {
            'y': y,
            'y_hat': y_hat,
            'loss': loss
        }

    def training_epoch_end(self, outputs):
        """ Calculate loss mean after epoch and for logging/evaluation. """
        with torch.no_grad():
            y = torch.cat([x['y'] for x in outputs])
            y_hat = torch.cat([x['y_hat'] for x in outputs])
            self.log('loss', self.loss(y, y_hat))

            losses = []
            for i in range(y.shape[1]):
                loss = getattr(F, self.hparams.loss)(y_hat[:, i], y[:, i])
                loss = loss.cpu().numpy()
                losses.append(loss)
            plt.plot(losses)
            plt.title(f'train {self.hparams.loss}')
            plt.xlabel('Forecast Horizon [h]')
            plt.ylabel(f'{self.hparams.loss}')
            plt.savefig('loss.png')
            plt.close()

            plt.plot(self._energy_net[-2].bias.cpu().numpy())
            plt.plot(self._energy_net[-2].weight.cpu().numpy().max(axis=1))
            plt.savefig('energy_fcn.png')
            plt.close()

            plt.plot(self._weather_net[-2].bias.cpu().numpy())
            plt.plot(self._weather_net[-2].weight.cpu().numpy().max(axis=1))
            plt.savefig('weather_fcn.png')
            plt.close()

            plt.plot(self._fc_sequential[0].bias.cpu().numpy())
            plt.plot(self._fc_sequential[0].weight.cpu().numpy().max(axis=1))
            plt.savefig('latent_fcn.png')
            plt.close()

            plt.plot(self._fc_sequential[-1].bias.cpu().numpy())
            plt.plot(self._fc_sequential[-1].weight.cpu().numpy().max(axis=1))
            plt.savefig('output_fcn.png')
            plt.close()

    def validation_step(self, batch, batch_idx):
        """ Perform validation step given a validation batch. """
        x, y = batch
        y_hat = self(x)  # self.forward(..)

        loss = self.loss(y, y_hat)
        return {
            'y': y,
            'y_hat': y_hat,
            'val_loss': loss
        }

    def validation_epoch_end(self, outputs):
        """ Calculate suitable metrics on validation set for logging and evaulation. """
        with torch.no_grad():
            y = torch.cat([x['y'] for x in outputs])
            y_hat = torch.cat([x['y_hat'] for x in outputs])
            self.log('val_loss', self.loss(y, y_hat))

            losses = []
            for i in range(y.shape[1]):
                loss = getattr(F, self.hparams.loss)(y_hat[:, i], y[:, i])
                loss = loss.cpu().numpy()
                losses.append(loss)
            plt.plot(losses)
            plt.title(f'val {self.hparams.loss}')
            plt.xlabel('Forecast Horizon [h]')
            plt.ylabel(f'{self.hparams.loss}')
            plt.savefig('val_loss.png')
            plt.close()

            # plt.hist(y_hat.flatten() - y.flatten(), )

            # calculate loss
            metric_dict = {}
            metric_dict[f'val/mae'] = F.l1_loss(y_hat, y)
            metric_dict[f'val/mse'] = F.mse_loss(y_hat, y)

            for key in metric_dict:
                self.log(key, metric_dict[key])

    def test_step(self, batch, batch_idx):
        """ Perform test step given a validation batch. """
        x, y = batch
        y_hat = self(x)  # self.forward(..)

        # calculate and return y, y_hat, and loss
        # for later evaluation in validation_epoch_end
        return {
            'y': y,
            'y_hat': y_hat,
            'test_loss': self.loss(y, y_hat)
        }

    def test_epoch_end(self, outputs):
        """ Calculate suitable metrics on test set for logging and evaulation. """
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])

        # calculate loss
        metric_dict = {}
        metric_dict[f'test/mae'] = F.l1_loss(y_hat, y)
        metric_dict[f'test/mse'] = F.mse_loss(y_hat, y)

        for key in metric_dict:
            self.log(key, metric_dict[key])
        self.log('test_loss', self.loss(y_hat, y))
