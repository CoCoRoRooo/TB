
# 🐍 Gestion des environnements virtuels Python (`venv`)

Ce guide contient les **commandes essentielles** pour créer, activer, gérer et déboguer les environnements virtuels Python, notamment dans le cadre de projets utilisant des bibliothèques comme `rank-bm25`.

---

## 📦 1. Créer un environnement virtuel

```bash
# Crée un venv dans le dossier actuel
python -m venv .venv
```

## ⚡ 2. Activer un environnement virtuel

### Sous Windows (PowerShell ou CMD)
```bash
.\venv\Scripts\activate
```

### Sous macOS/Linux
```bash
source .venv/bin/activate
```

---

## 🧪 3. Tester si un package est installé et accessible

```bash
# Vérifie que le module est bien disponible dans l’environnement
python -c "from rank_bm25 import BM25Okapi; print('rank_bm25 OK ✅')"
```

---

## 🔍 4. Vérifier le chemin de l’interpréteur Python utilisé

```python
import sys
print(sys.executable)
```

> Le chemin affiché doit pointer vers `.venv/Scripts/python.exe` sous Windows.

---

## 🔁 5. Installer ou réinstaller un package dans le venv

```bash
# Installation
pip install rank-bm25

# Réinstallation propre
pip uninstall rank-bm25 -y
pip install rank-bm25
```

## 📌 6. Gérer les dépendances avec `requirements.txt`

```bash
# Génére le fichier requirements.txt avec les versions exactes
pip freeze > requirements.txt

# Installe les dépendances listées dans le fichier
pip install -r requirements.txt
```

---

## ⚙️ 7. Sélectionner l’interpréteur dans VS Code

1. `Ctrl+Shift+P`
2. Taper : `Python: Select Interpreter`
3. Choisir : `.venv\Scripts\python.exe` (ou l'équivalent sur ton système)

---

## 💡 Astuce pour déboguer un `ModuleNotFoundError`

- Active le bon venv (`.\.venv\Scriptsctivate`)
- Vérifie que `pip show <module>` pointe bien dans `.venv`
- Utilise `sys.executable` pour t'assurer du bon interpréteur
- Si tu es sous VS Code, vérifie que tu exécutes avec le bon interpréteur

---

**Garde ce fichier dans ton projet pour ne pas perdre les bonnes pratiques de gestion d’environnement Python !**
