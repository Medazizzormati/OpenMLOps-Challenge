# =============================================================================
# 🐳 DOCKER PUSH - Instructions Simples
# =============================================================================
# 
# ⚠️ IMPORTANT: Exécutez ces commandes sur VOTRE machine locale!
# 
# =============================================================================

# ÉTAPE 1: Téléchargez le projet
# ================================
# Téléchargez le fichier: openmlops-challenge-final.zip
# Extrayez-le dans un dossier

# ÉTAPE 2: Ouvrez un terminal dans le dossier
# ============================================
cd openmlops-challenge

# ÉTAPE 3: Connectez-vous à Docker Hub
# =====================================
docker login
# Username: medaziz977
# Password: [votre mot de passe Docker Hub]

# ÉTAPE 4: Construisez l'image
# ============================
docker build -t medaziz977/openmlops-challenge:latest .
docker build -t medaziz977/openmlops-challenge:1.0.0 .

# ÉTAPE 5: Poussez vers Docker Hub
# =================================
docker push medaziz977/openmlops-challenge:latest
docker push medaziz977/openmlops-challenge:1.0.0

# =============================================================================
# VÉRIFICATION
# =============================================================================
# Après le push, vérifiez sur:
# https://hub.docker.com/repository/docker/medaziz977/openmlops-challenge

# =============================================================================
# COMMANDES RAPIDES (Copier-Coller)
# =============================================================================
# 
# docker login && \
# docker build -t medaziz977/openmlops-challenge:latest . && \
# docker push medaziz977/openmlops-challenge:latest
#
# =============================================================================
