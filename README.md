# Using Weather Data in Energy Time Series Forecasting: The Benefit of Input Data Transformations

This repository contains the Python implementation of the results presented in the following paper:
>[O. Neumann](mailto:oliver.neumann@kit.edu), M. Turowski, V. Hagenmeyer, R. Mikut, and N. Ludwig, 2023, "Using Weather Data in Energy Time Series Forecasting: The Benefit of Input Data Transformations," in Energy Informatics.

## Abstract

Renewable energy systems depend on the weather, and weather information, thus, plays a crucial role in forecasting time series within such renewable energy systems. However, while weather data are commonly used to improve forecast accuracy, it still has to be determined in which input shape this weather data benefits the forecasting models the most. 
In the present paper, we investigate how transformations for weather data inputs, \ie, station-based and grid-based weather data, influence the accuracy of energy time series forecasts. The selected weather data transformations are based on statistical features, dimensionality reduction, clustering, autoencoders, and interpolation. We evaluate the performance of these weather data transformations when forecasting three energy time series: electrical demand, solar power, and wind power. Additionally, we compare the best-performing weather data transformations for station-based and grid-based weather data.
We show that transforming station-based or grid-based weather data improves the forecast accuracy compared to using the raw weather data between 3.7~\% and 5.2~\%, depending on the target energy time series, where statistical and dimensionality reduction data transformations are among the best.

## Installation

Before the pipeline can be run, you need to prepare a python environment and download the energy and weather data. 

### Setup Python Environment

First, a virtual environment has to be set up. Therefore, you can use, for example, venv (`python -m venv venv`) or anaconda (`conda create -n env_name`) as a virtual environment. Afterwards, you can install the dependencies via `pip install -r requirements.txt`. 

### Download Data

After the environment is prepared, you can download the data by executing the python script in the dwd, ecmwf, and opsd data folder via `python download.py`. This downloads all needed energy, station-based, and grid-based weather data. For the ECMWF weather data, an API key is needed.

## Run Pipeline

Finally, you can run the pipeline via `python run.py` with the default parameters. You can see a list of available parameters for that script by calling `python run.py --help`. The most important parameters are:

```
--forecast_horizon (int)
    Defines the forecast time horizon,
    i.e. the next 24th hour.
--model (string)
    Defines the model to use in the pipeline,
    i.e. linear for a 'linear' regression model
    or 'DNN' for a deep neural network
    as presented in the paper.
--weather_data_representation (str)
    Weather data transformation to use for the weather
    data input. For example, Stats, PCA, AE, or PCACluster.
--seed (int)
    Seed to set for the random
    functions in the pipeline.
    This ensures reproducibility.
```

The results of a pipeline run are printed in the command line and logged to [W&B](http://wandb.com/). In our pipeline we use mainly [scikit-learn](https://scikit-learn.org/stable/) to train linear models, [PyTorch](https://pytorch.org/) with [Lightning](https://www.pytorchlightning.ai/) to train neural networks, and [pyWATTS](https://github.com/KIT-IAI/pyWATTS) as a workflow engine.


## Funding

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, the Helmholtz Association under the Program "Energy System Design", and the German Research Foundation (DFG) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645.


## License

This code is licensed under the [MIT License](LICENSE).
