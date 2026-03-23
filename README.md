# projet de session forage-des-donnes



## Introduction : 

Dans un contexte où les menaces informatiques se multiplient et évoluent rapidement, les centres d’opérations de sécurité (SOC) doivent être en mesure de détecter, classifier et répondre efficacement aux incidents de cybersécurité. Pour améliorer ces processus, Microsoft a développé la base de données GUIDE, qui constitue la plus grande collection publique d'incidents réels de sécurité. Cette base de données a été mise en place dans le cadre du développement du Copilot for Security Guided Response (CGR) et a pour objectif principal d’améliorer l’enquête, le triage et l’assainissement des incidents de sécurité.

**Problème rencontré :** La base de donnée fournie a **10 000 000** d'observations et 46 variables, soit 45 potentielles variables explicatives. Ceci a soulevé deux problèmes majeurs dans le cadre de ce travail :

**Problème1** : Le fichier est trop lourd ; il pèse plus de 3 Go, ce qui ralentit les algorithmes de Machine learning. Pour contourner cet obstacle, nous avons selectionné un échantillon de 10% des observations que nous avons renommé df. Tout notre travail est basé sur cette nouvelle database. 
À la fin de l'étude, le reste des données sera passé en streaming au modèle pour stimuler des cas réels et améliorer les paramètres.

**Problème2** : En présence d'autant de variables et d'autant d'observations, il y a un risque de surapprentissage.
Donc nous allons préter une attention particulière à la colinéarité entre les différentes variables. Et via une analyse par composantes principales, on pourra selectionner les prédicteurs les plus pertinents.

