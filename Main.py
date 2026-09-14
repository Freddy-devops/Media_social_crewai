# 1. Suppression des avertissements système (EN PREMIÈRE LIGNE)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 2. Installation des dépendances (Commande Colab)
!pip install --upgrade crewai litellm google-generativeai python-dotenv -q

import os
import nest_asyncio
import urllib.parse
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# 3. Gestion universelle de l'environnement (Local/Colab/Cloud)
nest_asyncio.apply()

try:
    from google.colab import userdata
    os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
    print("Environnement détecté : Google Colab")
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environnement détecté : Serveur Cloud / Local (via .env)")

# 4. Initialisation du modèle
cerveau_gemini = LLM(
    model="gemini/gemini-3.5-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"]
)

# ==========================================
# 5. OUTIL DE CRÉATION D'IMAGES (Format Markdown)
# ==========================================
@tool("Generateur d'images IA")
def generer_image(description_anglaise: str) -> str:
    """
    Outil utilisé pour générer une image à partir d'un prompt visuel en anglais.
    Retourne l'image au format Markdown pour qu'elle s'affiche directement.
    """
    print(f"\n🎨 [Le Designer génère l'image] -> Prompt : {description_anglaise}")
    prompt_formate = urllib.parse.quote(description_anglaise)
    url_image = f"https://image.pollinations.ai/prompt/{prompt_formate}"
    
    # C'est cette syntaxe Markdown qui permet l'affichage direct de l'image
    return f"![Visuel de la publication]({url_image})"

# ==========================================
# 6. LES AGENTS
# ==========================================
redacteur_contenu = Agent(
    role='Copywriter et Expert Social Media',
    goal='Rédiger des publications captivantes, professionnelles et virales.',
    backstory='Tu es un expert en marketing digital. Tu sais structurer un post avec une accroche, un corps clair et un appel à l\'action.',
    llm=cerveau_gemini,
    verbose=True,
    allow_delegation=False
)

designer_graphique = Agent(
    role='Directeur Artistique',
    goal='Créer des visuels accrocheurs pour illustrer les publications.',
    backstory='Tu es un graphiste de talent. Tu traduis le texte du rédacteur en un prompt visuel court en ANGLAIS, puis tu utilises ton outil pour générer le code de l\'image.',
    llm=cerveau_gemini,
    tools=[generer_image],
    verbose=True,
    allow_delegation=False
)

community_manager = Agent(
    role='Community Manager Senior',
    goal='Assembler le texte et l\'image Markdown pour livrer le post final.',
    backstory='Tu es le garant du rendu final. Tu t\'assures que le texte original est parfaitement préservé et que l\'image s\'affiche juste en dessous.',
    llm=cerveau_gemini,
    verbose=True,
    allow_delegation=False
)

# ==========================================
# 7. CONFIGURATION CLOUD / API
# ==========================================
SUJET_CLIENT = "Comment l'automatisation par l'Intelligence Artificielle peut faire gagner du temps aux petites entreprises."
PLATEFORME_CIBLE = "LinkedIn"
TONALITE = "Professionnelle mais accessible, orientée solution."

# ==========================================
# 8. LES TÂCHES
# ==========================================
task_redaction = Task(
    description=f'''
    Rédige une publication pour {PLATEFORME_CIBLE} sur le sujet suivant : "{SUJET_CLIENT}".
    La tonalité doit être : {TONALITE}.
    Structure requise : 1. Accroche 2. Développement (puces/tirets) 3. Appel à l'action.
    ''',
    expected_output='Le texte complet et structuré de la publication.',
    agent=redacteur_contenu
)

task_design = Task(
    description='Lis le texte du rédacteur. Rédige un prompt descriptif de 15 mots max en ANGLAIS, puis UTILISE ton outil pour générer l\'image.',
    expected_output='Le résultat Markdown de l\'image générée.',
    agent=designer_graphique
)

task_assemblage = Task(
    description='Récupère le texte complet du Copywriter ET l\'image Markdown du Designer. Assemble les deux éléments.',
    expected_output='Le livrable final : le texte en haut, suivi de l\'image formatée en Markdown `![Visuel](URL)` en bas. Ne donne aucun lien brut.',
    agent=community_manager
)

# ==========================================
# 9. EXÉCUTION DU SYSTÈME
# ==========================================
equipe_reseaux_sociaux = Crew(
    agents=[redacteur_contenu, designer_graphique, community_manager],
    tasks=[task_redaction, task_design, task_assemblage],
    verbose=True
)

print(f"\n🚀 Création du contenu en cours pour : '{SUJET_CLIENT}'...")
resultat_publication = await equipe_reseaux_sociaux.kickoff_async()

print("\n\n========================================")
print(f"📦 PACK CONTENU FINAL ({PLATEFORME_CIBLE}) :")
print("========================================\n")
# L'affichage final interprétera le Markdown et affichera l'image dans un notebook compatible
from IPython.display import display, Markdown
display(Markdown(resultat_publication.raw))
