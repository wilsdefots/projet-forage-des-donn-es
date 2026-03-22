# projet de session forage-des-donnes



## Introduction : 

Dans un contexte où les menaces informatiques se multiplient et évoluent rapidement, les centres d’opérations de sécurité (SOC) doivent être en mesure de détecter, classifier et répondre efficacement aux incidents de cybersécurité. Pour améliorer ces processus, Microsoft a développé la base de données GUIDE, qui constitue la plus grande collection publique d'incidents réels de sécurité. Cette base de données a été mise en place dans le cadre du développement du Copilot for Security Guided Response (CGR) et a pour objectif principal d’améliorer l’enquête, le triage et l’assainissement des incidents de sécurité.

**Problème rencontré :** La base de donnée fournie a **10 000 000** d'observations et 46 variables, soit 45 potentielles variables explicatives. Ceci a soulevé deux problèmes majeurs dans le cadre de ce travail :

**Problème1** : Le fichier est trop lourd ; il pèse plus de 3 Go, ce qui ralentit les algorithmes de Machine learning. Pour contourner cet obstacle, nous avons selectionné un échantillon de 10% des observations que nous avons renommé df. Tout notre travail est basé sur cette nouvelle database. 
À la fin de l'étude, le reste des données sera passé en streaming au modèle pour stimuler des cas réels et améliorer les paramètres.

**Problème2** : En présence d'autant de variables et d'autant d'observations, il y a un risque de surapprentissage.
Donc nous allons préter une attention particulière à la colinéarité entre les différentes variables. Et via une analyse par composantes principales, on pourra selectionner les prédicteurs les plus pertinents.

### 1. Méthodologie : 

1. 

2.

3.


###  1.2 Description des variables :


##### **La Variable cible** : IncidentGrade : (La gravité de l'incident).

  Cette variables va nous aider à identifier la gravité des différents incidents.
  
##### **Variables explicatives potentielles** :

**Id** : Identifiant unique pour chaque paire OrgId-IncidentId.

**OrgId** : Identifiant de l'organisation.

**IncidentId** : Identifiant unique de l'incident au sein de l'organisation.

**AlertId** : Identifiant unique pour une alerte.

**Timestamp** : Date et heure de création de l’alerte.

**DetectorId** : Identifiant unique du détecteur ayant généré l’alerte.

**AlertTitle** : Titre de l’alerte.

**Category** : Catégorie de l’alerte.

**MitreTechniques** : techniques d'attaque qui ont été identifiées dans une alerte de sécurité, basées sur le framework MITRE ATT&CK.

**IncidentGrade** : Niveau de gravité attribué à l’incident par le SOC.

**ActionGrouped** : Action de remédiation de l’alerte par le SOC (niveau général).

**ActionGranular** : Action de remédiation de l’alerte par le SOC (niveau détaillé).

**EntityType** : Type d’entité impliquée dans l’alerte.

**EvidenceRole** : Rôle de la preuve dans l’enquête.

**DeviceId** : Identifiant unique du dispositif.

**Sha256** : Empreinte SHA-256 du fichier.

**IpAddress** : Adresse IP impliquée.

**Url** : URL impliquée.

**AccountSid** : Identifiant du compte on-premises.

**AccountUpn** : Identifiant du compte email.

**AccountObjectId** : Identifiant du compte Entra ID.

**AccountName** : Nom du compte on-premises.

**DeviceName** : Nom du dispositif.

**NetworkMessageId** : Identifiant au niveau organisationnel pour le message email.

**EmailClusterId** : Identifiant unique du cluster d’emails.

**RegistryKey** : Clé de registre impliquée.

**RegistryValueName** : Nom de la valeur du registre.

**RegistryValueData** : Données de la valeur du registre.

**ApplicationId** : Identifiant unique de l’application.

**ApplicationName** : Nom de l’application.

**OAuthApplicationId** : Identifiant de l’application OAuth.

**ThreatFamily** : Famille de logiciels malveillants associée à un fichier.

**FileName** : Nom du fichier.

**FolderPath** : Chemin du dossier contenant le fichier.

**ResourceIdName** : Nom de la ressource Azure.

**ResourceType** : Type de ressource Azure.

**Roles** : Métadonnées supplémentaires sur le rôle de la preuve dans l’alerte.

**OSFamily** : Famille du système d’exploitation.

**OSVersion** : Version du système d’exploitation.

**AntispamDirection** : Direction du filtre antispam.

**SuspicionLevel** : Niveau de suspicion.

**LastVerdict** : Verdict final de l’analyse de la menace.

**CountryCode** : Code du pays où la preuve a été trouvée.

**State** : État où la preuve a été trouvée.

**City** : Ville où la preuve a été trouvée.

Les variables sont réparties en 31 numériques et 15 catégorielles.

En réalité,les variables numérique sont en fait des id et des codes . ==> ce sont des variables catégorique à 
proprement parler. On devra modifier le type dans les traitements.




Nous constatons qu'il ya beaucoup de colonnes qui doivent être supprimées par ce que
Ce sont des Id --> Elles sont facilement identifiablesde façon unique et donc,
non performants pour les algorithmes d'apprentissage machine ;
