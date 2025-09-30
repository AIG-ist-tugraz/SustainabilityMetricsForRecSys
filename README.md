# SustainabilityMetricsForRecSys
Sustainability-Aware Evaluation Metrics for Recommender Systems

This implementation allows for the reproduction of the results of:
[TBD paper citation]

## Datasets
The original Amazon Reviews’23 dataset used for extraction is available on:

https://amazon-reviews-2023.github.io/

## Usage
In order to run the evaluation install the packages referenced in the requirements.txt and run the command
```
python main.py
```
In order to measure the energy consumption of the approaches the command needs to be executed with sudo, alternatively the energy measuring can be deactivated in the code
```
do_measure_energy = True --> do_measure_energy = False
```
## Supplemented data
In order to calculate the carbon footprint we used:

https://github.com/amazon-science/carbon-assessment-with-ml

The green flag is calculated by Gemini 2.5 Flash Lite

https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite

This is not included in this code and needs to be performed separately, detailed instructions in the paper.