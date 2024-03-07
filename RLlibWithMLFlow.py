
""" 
Petit programme qui test RLlib (ray.tune) en combinaison avec MLflow
Test effectué sur "CartPole-v1" avec, d'abord des paramètres choisis
par mes soins puis ensuite avec les paramètres par défut de tune().
il faut entrer : mlflow server --host 127.0.0.1 --port 5000
dans un terminal pour lancer le serveur local.
Ensuite, le programme tourne normalement, il est possible de consulter
les résultats sur http://127.0.0.1:5000

 """

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import ray
from ray import tune
#from ray.rllib.agents import ppo
 
import os
os.environ['no_proxy'] = '*'
 
import mlflow
from mlflow import MlflowClient
#from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
 
print("debut du test")
 
 
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")        #necessite : mlflow server --host 127.0.0.1 --port 5000
 
# Create a new MLflow Experiment,
#mlflow.set_experiment("MLflow Quickstart")
 
experiment_name = "MLflow Quickstart3"
#mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)
 
print("1000", mlflow.get_tracking_uri(), mlflow.get_artifact_uri())
 
# Enregistrement des résultats avec MLflow
mlflow_callback = MLflowLoggerCallback(
    tracking_uri = mlflow.get_tracking_uri(),
    registry_uri = mlflow.get_artifact_uri(),
    experiment_name = experiment_name,
)
 
print("2000")
 
ray.init()
# Configuration de l'environnement et de l'algorithme
config = {
    "gamma":1,
    "lr": tune.choice([1e-4]), #revient à np.random.choice([a,b])
    "env": "CartPole-v1",
    "framework": "tf",
    "memory_size": 1000000,
    "batch_size": 20,
}
 
print("3000")
 
mlflow.log_params(config)
 
# Entraînement de l'agent avec Tune
analysis = tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 12},  # Arrêt après 10 itérations d'entraînement
    callbacks=[mlflow_callback],
)

#Test avec param par défaut


print("########################################### \n########################################### \n########################################### \n")
print("########################################### \n########################################### \n########################################### \n")
print("########################################### \n########################################### \n########################################### \n")

mlflow.set_experiment(experiment_name)
config2 = {
    "env": "CartPole-v1",
    "framework": "tf",
}
 

# Enregistrement des résultats avec MLflow
mlflow_callback = MLflowLoggerCallback(
    tracking_uri = mlflow.get_tracking_uri(),
    registry_uri = mlflow.get_artifact_uri(),
    experiment_name = experiment_name,
)

mlflow.log_params(config2)
 
# Entraînement de l'agent avec Tune

analysis = tune.run(
    "PPO",
    config=config2,
    stop={"training_iteration": 12},  # Arrêt après 10 itérations d'entraînement
    callbacks=[mlflow_callback],
    )


# Arrêter Ray
ray.shutdown()
 
 
print("Fin du programme\n")
 
 
 
 
#Commenter : Ctrl + shift + A
"""  Exemples d'hyperparamètres    
    GAMMA = 0.95
    LEARNING_RATE = 0.001
    MEMORY_SIZE = 1000000           # Capacité de stockage des expériences passées
    BATCH_SIZE = 20                 # Taille de l'échantillon lors de l'ajustement des poids
    EXPLORATION_MAX = 1.0           # 1 = random, 0 = déterministe
    EXPLORATION_MIN = 0.01          #
    EXPLORATION_DECAY = 0.995       #
 
"""  