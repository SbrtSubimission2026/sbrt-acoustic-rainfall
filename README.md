# sbrt-acoustic-rainfall
Official repository containing the souce code, feature extraction pipeline and ML models for the acoustic rainfall sensing paper

# Source Code: Machine Learning Trade-offs in Multi-Environment Adaptability for Acoustic Rainfall Sensing

## Informações do Projeto:
Este repositorio contem o código-fonte, os scripts de processamento de audio, construtor do dataset, extração de metricas, modelos de machine learning, vizualização dos resultados e os notebooks dos testes utilizados para gerar os resultados do artigo submetido do Sbrt 2026. O principal objetivo deste codigo é garantir a reprodutibilidade dos experimentos.

## Estrutura do Repositório:
* '\data': Que contém as pastas '\Processed' e '\splits'. Onde no '\processed' tem os arquivos .CSV com as metricas ja calculadas e o '\splits' contém os metadados dos audios que foram utiilizados.
*  '\notebooks': Onde se tem a localização de todos os notebooks para cada ambiente, onde estão separados por pastas com os nomes dos ambientes
*  '\rainfall_acoustic_classification': é o local onde estão localizados os scripts que são essencias para o funcionamento dos testes
*   '\results': é o local onde estão salvos todos os .CSV's com os resultados retirados dos notebooks

## Pré-requisitos:
Para recriar este projeto é necessario do python 3.13+, e das principais bibliotecas listadas:
* pandas
* numpy
* scikit-learn (sklearn)
* xgboost
* matplotlib (matplotlib.pyplot)
* seaborn
* itertools
* ast
* time

Para instalar as bibliotecas utilize: pip install -r requiriments.txt
