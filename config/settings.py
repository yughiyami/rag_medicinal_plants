import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present (for DEEPSEEK_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = None  # Optional: set for 10 req/s instead of 3

TARGET_SPECIES = [
    "Uncaria tomentosa",
    "Lepidium meyenii",
    "Croton lechleri",
    "Minthostachys mollis",
    "Erythroxylum coca",
    "Smallanthus sonchifolius",
    "Physalis peruviana",
    "Buddleja incana",
]

SOUTH_PERU_DEPARTMENTS = ["Arequipa", "Puno", "Cusco", "Moquegua", "Tacna"]

SPECIES_CATALOG = {
    # ── Original 8 core species ──
    "Uncaria tomentosa": {
        "common": ["Uña de gato", "Cat's Claw"],
        "family": "Rubiaceae",
        "use": "Antiinflamatorio, inmunomodulador",
        "region": ["Cusco", "Junín", "Madre de Dios"],
        "altitude": "0-800m",
        "compounds": ["mitraphylline", "isomitraphylline", "rhynchophylline"],
    },
    "Lepidium meyenii": {
        "common": ["Maca"],
        "family": "Brassicaceae",
        "use": "Fertilidad, energizante, adaptógeno",
        "region": ["Junín", "Puno", "Cusco"],
        "altitude": "3800-4500m",
        "compounds": ["macamides", "macaenes", "glucosinolates"],
    },
    "Croton lechleri": {
        "common": ["Sangre de grado", "Dragon's Blood"],
        "family": "Euphorbiaceae",
        "use": "Cicatrizante, antiviral, antiinflamatorio",
        "region": ["Cusco", "Madre de Dios", "San Martín"],
        "altitude": "0-2000m",
        "compounds": ["taspine", "proanthocyanidins"],
    },
    "Minthostachys mollis": {
        "common": ["Muña", "Peperina"],
        "family": "Lamiaceae",
        "use": "Digestivo, antimicrobiano, antiparasitario",
        "region": ["Arequipa", "Puno", "Cusco", "Moquegua"],
        "altitude": "2500-3500m",
        "compounds": ["pulegone", "menthone", "limonene"],
    },
    "Erythroxylum coca": {
        "common": ["Coca"],
        "family": "Erythroxylaceae",
        "use": "Estimulante, anestésico local, soroche",
        "region": ["Cusco", "Puno", "La Libertad"],
        "altitude": "500-2000m",
        "compounds": ["cocaine", "ecgonine", "tropinone"],
    },
    "Smallanthus sonchifolius": {
        "common": ["Yacón"],
        "family": "Asteraceae",
        "use": "Antidiabético, prebiótico",
        "region": ["Cusco", "Puno", "Arequipa"],
        "altitude": "1000-3000m",
        "compounds": ["fructooligosaccharides", "phenolic acids"],
    },
    "Physalis peruviana": {
        "common": ["Aguaymanto", "Cape Gooseberry"],
        "family": "Solanaceae",
        "use": "Antioxidante, antiinflamatorio",
        "region": ["Cusco", "Arequipa", "Cajamarca"],
        "altitude": "1500-3000m",
        "compounds": ["withanolides", "physalins", "carotenoids"],
    },
    "Buddleja incana": {
        "common": ["Quishuar", "Kiswar"],
        "family": "Scrophulariaceae",
        "use": "Hepático, antiinflamatorio, cicatrizante",
        "region": ["Cusco", "Puno", "Arequipa"],
        "altitude": "2800-3800m",
        "compounds": ["flavonoids", "iridoids", "verbascoside"],
    },
    # ── Andean highlands (Puna / Sierra sur) ──
    "Gentianella alborosea": {
        "common": ["Hercampuri"],
        "family": "Gentianaceae",
        "use": "Hepatoprotector, hipoglicemiante",
        "region": ["Puno", "Cusco", "Junín"],
        "altitude": "3500-4300m",
        "compounds": ["xanthones", "secoiridoids", "amarogentin"],
    },
    "Senecio nutans": {
        "common": ["Chachacoma", "Wiskataya"],
        "family": "Asteraceae",
        "use": "Mal de altura, bronquitis",
        "region": ["Puno", "Arequipa", "Tacna"],
        "altitude": "3800-4800m",
        "compounds": ["pyrrolizidine alkaloids", "sesquiterpenes"],
    },
    "Chuquiraga spinosa": {
        "common": ["Huamanpinta"],
        "family": "Asteraceae",
        "use": "Antiinflamatorio urinario, próstata",
        "region": ["Puno", "Cusco", "Arequipa"],
        "altitude": "3000-4500m",
        "compounds": ["flavonoids", "sesquiterpene lactones"],
    },
    "Ephedra americana": {
        "common": ["Pinco-pinco"],
        "family": "Ephedraceae",
        "use": "Broncodilatador, renal",
        "region": ["Arequipa", "Puno", "Tacna", "Moquegua"],
        "altitude": "2500-4200m",
        "compounds": ["ephedrine", "pseudoephedrine"],
    },
    "Krameria lappacea": {
        "common": ["Ratania"],
        "family": "Krameriaceae",
        "use": "Astringente, hemostático, bucal",
        "region": ["Arequipa", "Moquegua", "Tacna"],
        "altitude": "1500-3500m",
        "compounds": ["proanthocyanidins", "lignan ratanhiaphenol"],
    },
    "Perezia multiflora": {
        "common": ["Escorzonera", "Contrahierba"],
        "family": "Asteraceae",
        "use": "Antiinflamatorio, febrífugo",
        "region": ["Puno", "Cusco"],
        "altitude": "3500-4500m",
        "compounds": ["sesquiterpene lactones", "perezinoide"],
    },
    "Mutisia acuminata": {
        "common": ["Chinchilcuma", "Chinchircuma"],
        "family": "Asteraceae",
        "use": "Respiratorio, tos, bronquitis",
        "region": ["Cusco", "Puno", "Arequipa"],
        "altitude": "2800-4000m",
        "compounds": ["flavonoids", "terpenes"],
    },
    "Senecio canescens": {
        "common": ["Vira-vira", "Wira wira"],
        "family": "Asteraceae",
        "use": "Tos, gripe, mal de altura",
        "region": ["Arequipa", "Puno", "Cusco"],
        "altitude": "3500-4800m",
        "compounds": ["pyrrolizidine alkaloids", "flavonoids"],
    },
    "Xenophyllum poposum": {
        "common": ["Pupusa", "Popusa"],
        "family": "Asteraceae",
        "use": "Reumatismo, dolor muscular",
        "region": ["Arequipa", "Puno", "Tacna"],
        "altitude": "4000-5000m",
        "compounds": ["sesquiterpenes", "diterpenes"],
    },
    "Azorella compacta": {
        "common": ["Yareta"],
        "family": "Apiaceae",
        "use": "Renal, diabetes, reumatismo",
        "region": ["Arequipa", "Puno", "Tacna", "Moquegua"],
        "altitude": "4000-5000m",
        "compounds": ["diterpenoids", "azorellanol", "mulinane"],
    },
    "Parastrephia lucida": {
        "common": ["Tola"],
        "family": "Asteraceae",
        "use": "Resfríos, dolor estomacal",
        "region": ["Puno", "Arequipa", "Tacna"],
        "altitude": "3800-4600m",
        "compounds": ["flavonoids", "essential oils"],
    },
    "Baccharis genistelloides": {
        "common": ["Carqueja", "Tres filos"],
        "family": "Asteraceae",
        "use": "Digestivo, hepático, antiparasitario",
        "region": ["Cusco", "Puno", "Arequipa"],
        "altitude": "2000-3500m",
        "compounds": ["clerodane diterpenes", "flavonoids"],
    },
    "Baccharis latifolia": {
        "common": ["Chilca"],
        "family": "Asteraceae",
        "use": "Antiinflamatorio, reumatismo",
        "region": ["Arequipa", "Cusco", "Puno"],
        "altitude": "2000-3800m",
        "compounds": ["flavonoids", "terpenes", "essential oils"],
    },
    "Diplostephium tovarii": {
        "common": ["Pampa romero"],
        "family": "Asteraceae",
        "use": "Ceremonia, respiratorio",
        "region": ["Arequipa"],
        "altitude": "3000-4000m",
        "compounds": ["sesquiterpenes", "flavonoids"],
    },
    "Werneria nubigena": {
        "common": ["Pupusa blanca"],
        "family": "Asteraceae",
        "use": "Digestivo, antiinflamatorio",
        "region": ["Puno", "Cusco", "Arequipa"],
        "altitude": "4000-4800m",
        "compounds": ["sesquiterpene lactones"],
    },
    # ── Andean valleys / inter-Andean ──
    "Schinus molle": {
        "common": ["Molle", "Peruvian pepper"],
        "family": "Anacardiaceae",
        "use": "Antimicrobiano, antiinflamatorio",
        "region": ["Arequipa", "Cusco", "Moquegua", "Tacna"],
        "altitude": "0-3500m",
        "compounds": ["essential oils", "terpinene", "phellandrene"],
    },
    "Cantua buxifolia": {
        "common": ["Cantuta", "Flor sagrada"],
        "family": "Polemoniaceae",
        "use": "Ictericia, dolor de oído",
        "region": ["Cusco", "Puno", "Arequipa"],
        "altitude": "2500-3800m",
        "compounds": ["flavonoids", "saponins"],
    },
    "Kageneckia lanceolata": {
        "common": ["Lloque"],
        "family": "Rosaceae",
        "use": "Antirreumático, articulaciones",
        "region": ["Arequipa", "Cusco"],
        "altitude": "2500-3800m",
        "compounds": ["tannins", "cyanogenic glycosides"],
    },
    "Tropaeolum tuberosum": {
        "common": ["Mashua", "Añu"],
        "family": "Tropaeolaceae",
        "use": "Antimicrobiano, antiproliferativo",
        "region": ["Cusco", "Puno"],
        "altitude": "3000-4000m",
        "compounds": ["glucosinolates", "isothiocyanates", "anthocyanins"],
    },
    "Chenopodium quinoa": {
        "common": ["Quinoa", "Kinwa"],
        "family": "Amaranthaceae",
        "use": "Nutricional, antiinflamatorio, cicatrizante",
        "region": ["Puno", "Arequipa", "Cusco"],
        "altitude": "2500-4000m",
        "compounds": ["saponins", "flavonoids", "betacyanins"],
    },
    "Chenopodium pallidicaule": {
        "common": ["Cañihua", "Kañiwa"],
        "family": "Amaranthaceae",
        "use": "Nutricional, antioxidante",
        "region": ["Puno", "Cusco"],
        "altitude": "3500-4200m",
        "compounds": ["flavonoids", "betacyanins", "phenolic acids"],
    },
    "Lupinus mutabilis": {
        "common": ["Tarwi", "Chocho"],
        "family": "Fabaceae",
        "use": "Antiparasitario, eczema, diabetes",
        "region": ["Puno", "Cusco", "Arequipa"],
        "altitude": "2500-3800m",
        "compounds": ["quinolizidine alkaloids", "lupanine", "sparteine"],
    },
    "Solanum tuberosum": {
        "common": ["Papa nativa"],
        "family": "Solanaceae",
        "use": "Cicatrizante, gastritis, nutritivo",
        "region": ["Puno", "Cusco", "Arequipa"],
        "altitude": "3000-4200m",
        "compounds": ["glycoalkaloids", "anthocyanins", "phenolic acids"],
    },
    "Oxalis tuberosa": {
        "common": ["Oca"],
        "family": "Oxalidaceae",
        "use": "Emoliente, astringente",
        "region": ["Puno", "Cusco"],
        "altitude": "3000-4000m",
        "compounds": ["oxalic acid", "anthocyanins", "phenolics"],
    },
    "Ullucus tuberosus": {
        "common": ["Olluco", "Papa lisa"],
        "family": "Basellaceae",
        "use": "Cicatrizante, gastritis",
        "region": ["Puno", "Cusco", "Arequipa"],
        "altitude": "2800-4000m",
        "compounds": ["betacyanins", "betaxanthins", "starch"],
    },
    # ── Aromatic / essential oil species ──
    "Satureja boliviana": {
        "common": ["Muña negra", "Khoa"],
        "family": "Lamiaceae",
        "use": "Digestivo, carminativo, antimicrobiano",
        "region": ["Puno", "Cusco"],
        "altitude": "3000-4000m",
        "compounds": ["pulegone", "menthone", "carvacrol"],
    },
    "Clinopodium bolivianum": {
        "common": ["Inca muña"],
        "family": "Lamiaceae",
        "use": "Antimicrobiano, conservante natural",
        "region": ["Cusco", "Puno"],
        "altitude": "2500-3800m",
        "compounds": ["pulegone", "menthone", "isomenthone"],
    },
    "Tagetes minuta": {
        "common": ["Huacatay"],
        "family": "Asteraceae",
        "use": "Antimicrobiano, insecticida, condimento",
        "region": ["Arequipa", "Cusco", "Puno"],
        "altitude": "1000-3500m",
        "compounds": ["ocimene", "dihydrotagetone", "tagetone"],
    },
    "Tagetes elliptica": {
        "common": ["Chincho"],
        "family": "Asteraceae",
        "use": "Condimento medicinal, carminativo",
        "region": ["Cusco", "Arequipa"],
        "altitude": "2500-3500m",
        "compounds": ["essential oils", "flavonoids"],
    },
    "Matricaria chamomilla": {
        "common": ["Manzanilla"],
        "family": "Asteraceae",
        "use": "Digestivo, sedante, antiinflamatorio",
        "region": ["Arequipa", "Cusco", "Puno", "Tacna", "Moquegua"],
        "altitude": "0-3800m",
        "compounds": ["chamazulene", "bisabolol", "apigenin"],
    },
    "Eucalyptus globulus": {
        "common": ["Eucalipto"],
        "family": "Myrtaceae",
        "use": "Expectorante, antiséptico respiratorio",
        "region": ["Arequipa", "Cusco", "Puno", "Moquegua"],
        "altitude": "0-3500m",
        "compounds": ["eucalyptol", "alpha-pinene", "limonene"],
    },
    "Rosmarinus officinalis": {
        "common": ["Romero"],
        "family": "Lamiaceae",
        "use": "Estimulante, digestivo, memoria",
        "region": ["Arequipa", "Cusco", "Tacna"],
        "altitude": "0-3500m",
        "compounds": ["rosmarinic acid", "carnosic acid", "camphor"],
    },
    "Origanum vulgare": {
        "common": ["Orégano"],
        "family": "Lamiaceae",
        "use": "Antimicrobiano, antioxidante, digestivo",
        "region": ["Tacna", "Arequipa", "Moquegua"],
        "altitude": "2000-3500m",
        "compounds": ["carvacrol", "thymol", "rosmarinic acid"],
    },
    # ── Medicinal trees / shrubs ──
    "Polylepis rugulosa": {
        "common": ["Queñua", "Queñual"],
        "family": "Rosaceae",
        "use": "Respiratorio, cervical, ritual",
        "region": ["Arequipa", "Puno", "Tacna"],
        "altitude": "3500-5000m",
        "compounds": ["tannins", "triterpenes"],
    },
    "Polylepis incana": {
        "common": ["Queñua blanca"],
        "family": "Rosaceae",
        "use": "Infecciones respiratorias",
        "region": ["Cusco", "Arequipa"],
        "altitude": "3500-4800m",
        "compounds": ["tannins", "phenolic acids"],
    },
    "Escallonia resinosa": {
        "common": ["Chachacomo"],
        "family": "Escalloniaceae",
        "use": "Bronquitis, dolor de huesos",
        "region": ["Cusco", "Arequipa"],
        "altitude": "2800-3800m",
        "compounds": ["essential oils", "terpenes", "tannins"],
    },
    "Sambucus peruviana": {
        "common": ["Saúco"],
        "family": "Adoxaceae",
        "use": "Antigripal, diurético, febrífugo",
        "region": ["Cusco", "Arequipa", "Puno"],
        "altitude": "2000-3500m",
        "compounds": ["anthocyanins", "flavonoids", "sambunigrin"],
    },
    "Caesalpinia spinosa": {
        "common": ["Tara", "Taya"],
        "family": "Fabaceae",
        "use": "Astringente, amigdalitis, cicatrizante",
        "region": ["Arequipa", "Cusco", "Tacna"],
        "altitude": "1000-3000m",
        "compounds": ["tannins", "gallic acid", "gallotannins"],
    },
    "Prosopis pallida": {
        "common": ["Algarrobo"],
        "family": "Fabaceae",
        "use": "Nutritivo, antianémico",
        "region": ["Arequipa", "Tacna", "Moquegua"],
        "altitude": "0-1500m",
        "compounds": ["tryptamine alkaloids", "polyphenols"],
    },
    # ── Widespread Andean medicinals ──
    "Plantago major": {
        "common": ["Llantén"],
        "family": "Plantaginaceae",
        "use": "Cicatrizante, antiinflamatorio, diurético",
        "region": ["Arequipa", "Cusco", "Puno", "Tacna", "Moquegua"],
        "altitude": "0-4000m",
        "compounds": ["aucubin", "catalpol", "baicalein"],
    },
    "Urtica urens": {
        "common": ["Ortiga"],
        "family": "Urticaceae",
        "use": "Antirreumático, diurético, próstata",
        "region": ["Arequipa", "Cusco", "Puno"],
        "altitude": "0-4000m",
        "compounds": ["formic acid", "histamine", "flavonoids"],
    },
    "Equisetum bogotense": {
        "common": ["Cola de caballo"],
        "family": "Equisetaceae",
        "use": "Renal, diurético, hemostático",
        "region": ["Arequipa", "Cusco", "Puno"],
        "altitude": "2000-4000m",
        "compounds": ["silica", "flavonoids", "phenolic acids"],
    },
    "Taraxacum officinale": {
        "common": ["Diente de león"],
        "family": "Asteraceae",
        "use": "Hepático, diurético, digestivo",
        "region": ["Arequipa", "Cusco", "Puno", "Moquegua"],
        "altitude": "0-4000m",
        "compounds": ["sesquiterpene lactones", "taraxasterol", "inulin"],
    },
    "Aloe vera": {
        "common": ["Sábila"],
        "family": "Asphodelaceae",
        "use": "Cicatrizante, purgante, cosmético",
        "region": ["Arequipa", "Tacna", "Moquegua"],
        "altitude": "0-2500m",
        "compounds": ["aloin", "acemannan", "anthraquinones"],
    },
    "Calendula officinalis": {
        "common": ["Caléndula"],
        "family": "Asteraceae",
        "use": "Cicatrizante, antiinflamatorio, dérmico",
        "region": ["Arequipa", "Cusco"],
        "altitude": "0-3500m",
        "compounds": ["triterpenoids", "flavonoids", "carotenoids"],
    },
    "Malva sylvestris": {
        "common": ["Malva"],
        "family": "Malvaceae",
        "use": "Emoliente, bronquitis, digestivo",
        "region": ["Arequipa", "Cusco", "Puno", "Tacna"],
        "altitude": "0-3500m",
        "compounds": ["mucilages", "anthocyanins", "malvidin"],
    },
    "Verbena litoralis": {
        "common": ["Verbena"],
        "family": "Verbenaceae",
        "use": "Febrífugo, hepático, digestivo",
        "region": ["Arequipa", "Cusco", "Puno"],
        "altitude": "0-3500m",
        "compounds": ["verbenalin", "hastatoside", "flavonoids"],
    },
    # ── Fruit / nutritional-medicinal ──
    "Bixa orellana": {
        "common": ["Achiote"],
        "family": "Bixaceae",
        "use": "Antimicrobiano, colorante, antiinflamatorio",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-1500m",
        "compounds": ["bixin", "norbixin", "tocotrienols"],
    },
    "Annona muricata": {
        "common": ["Guanábana"],
        "family": "Annonaceae",
        "use": "Citotóxico, antiparasitario, ansiolítico",
        "region": ["Cusco"],
        "altitude": "0-1000m",
        "compounds": ["annonacin", "acetogenins", "muricoreacin"],
    },
    "Passiflora ligularis": {
        "common": ["Granadilla"],
        "family": "Passifloraceae",
        "use": "Sedante, ansiolítico, digestivo",
        "region": ["Cusco", "Arequipa"],
        "altitude": "1500-2800m",
        "compounds": ["flavonoids", "cyanogenic glycosides"],
    },
    "Passiflora edulis": {
        "common": ["Maracuyá"],
        "family": "Passifloraceae",
        "use": "Sedante, antioxidante",
        "region": ["Cusco"],
        "altitude": "0-2000m",
        "compounds": ["chrysin", "vitexin", "gamma-aminobutyric acid"],
    },
    "Opuntia ficus-indica": {
        "common": ["Tuna"],
        "family": "Cactaceae",
        "use": "Antidiabético, antiinflamatorio",
        "region": ["Arequipa", "Cusco", "Moquegua", "Tacna"],
        "altitude": "0-3500m",
        "compounds": ["betalains", "flavonoids", "mucilages"],
    },
    "Punica granatum": {
        "common": ["Granada"],
        "family": "Lythraceae",
        "use": "Antioxidante, antiparasitario",
        "region": ["Arequipa", "Tacna", "Moquegua"],
        "altitude": "0-2500m",
        "compounds": ["punicalagin", "ellagic acid", "anthocyanins"],
    },
    "Carica papaya": {
        "common": ["Papaya"],
        "family": "Caricaceae",
        "use": "Digestivo, antiparasitario, antiinflamatorio",
        "region": ["Cusco"],
        "altitude": "0-1500m",
        "compounds": ["papain", "chymopapain", "carpaine"],
    },
    "Psidium guajava": {
        "common": ["Guayaba"],
        "family": "Myrtaceae",
        "use": "Antidiarreico, antimicrobiano",
        "region": ["Cusco", "Arequipa"],
        "altitude": "0-2000m",
        "compounds": ["quercetin", "guaijaverin", "gallic acid"],
    },
    # ── Amazonian-Andean transition ──
    "Maytenus macrocarpa": {
        "common": ["Chuchuhuasi"],
        "family": "Celastraceae",
        "use": "Antirreumático, tónico, antiinflamatorio",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-1500m",
        "compounds": ["maytansinoids", "sesquiterpenes", "triterpenes"],
    },
    "Banisteriopsis caapi": {
        "common": ["Ayahuasca"],
        "family": "Malpighiaceae",
        "use": "Psicoactivo ritual, purgante",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-1000m",
        "compounds": ["harmine", "harmaline", "tetrahydroharmine"],
    },
    "Psychotria viridis": {
        "common": ["Chacruna"],
        "family": "Rubiaceae",
        "use": "Componente ayahuasca",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-800m",
        "compounds": ["dimethyltryptamine"],
    },
    "Cinchona officinalis": {
        "common": ["Quina", "Cascarilla"],
        "family": "Rubiaceae",
        "use": "Antimalárico, febrífugo",
        "region": ["Cusco", "Cajamarca"],
        "altitude": "1000-3000m",
        "compounds": ["quinine", "quinidine", "cinchonine"],
    },
    "Brunfelsia grandiflora": {
        "common": ["Chiric sanango"],
        "family": "Solanaceae",
        "use": "Antirreumático, febrífugo",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-1500m",
        "compounds": ["brunfelsamidine", "scopoletin"],
    },
    "Dracontium loretense": {
        "common": ["Jergón sacha"],
        "family": "Araceae",
        "use": "Antiofídico, antiinflamatorio",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-1000m",
        "compounds": ["oxalic acid", "tannins", "flavonoids"],
    },
    "Copaifera paupera": {
        "common": ["Copaiba"],
        "family": "Fabaceae",
        "use": "Antiinflamatorio, cicatrizante, antimicrobiano",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-500m",
        "compounds": ["beta-caryophyllene", "copalic acid", "diterpenes"],
    },
    "Uncaria guianensis": {
        "common": ["Uña de gato lisa"],
        "family": "Rubiaceae",
        "use": "Antiinflamatorio, inmunomodulador",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-800m",
        "compounds": ["oxindole alkaloids", "triterpenes"],
    },
    # ── Coastal / low-valley south Peru ──
    "Solanum americanum": {
        "common": ["Hierba mora"],
        "family": "Solanaceae",
        "use": "Antiinflamatorio, analgésico tópico",
        "region": ["Arequipa", "Tacna", "Moquegua"],
        "altitude": "0-3000m",
        "compounds": ["solanine", "solasodine", "diosgenin"],
    },
    "Argemone mexicana": {
        "common": ["Cardo santo"],
        "family": "Papaveraceae",
        "use": "Antimicrobiano, antimalárico",
        "region": ["Arequipa", "Tacna"],
        "altitude": "0-2500m",
        "compounds": ["berberine", "protopine", "sanguinarine"],
    },
    "Tessaria integrifolia": {
        "common": ["Pájaro bobo"],
        "family": "Asteraceae",
        "use": "Hepático, renal, asma",
        "region": ["Arequipa", "Moquegua"],
        "altitude": "0-2500m",
        "compounds": ["flavonoids", "sesquiterpenes"],
    },
    "Tiquilia paronychioides": {
        "common": ["Flor de arena"],
        "family": "Boraginaceae",
        "use": "Renal, diurético",
        "region": ["Arequipa", "Tacna", "Moquegua"],
        "altitude": "0-2000m",
        "compounds": ["naphthoquinones", "flavonoids"],
    },
    # ── High-altitude medicinals ──
    "Stipa ichu": {
        "common": ["Ichu"],
        "family": "Poaceae",
        "use": "Renal, diurético (infusión raíz)",
        "region": ["Puno", "Arequipa", "Cusco", "Tacna", "Moquegua"],
        "altitude": "3500-4800m",
        "compounds": ["phenolic acids", "flavonoids"],
    },
    "Margiricarpus pinnatus": {
        "common": ["Perla", "China canlli"],
        "family": "Rosaceae",
        "use": "Renal, antiinflamatorio",
        "region": ["Arequipa", "Puno", "Cusco"],
        "altitude": "3000-4200m",
        "compounds": ["tannins", "triterpenes"],
    },
    "Tetraglochin cristatum": {
        "common": ["Canlli"],
        "family": "Rosaceae",
        "use": "Diarrea, disentería",
        "region": ["Puno", "Arequipa"],
        "altitude": "3500-4500m",
        "compounds": ["tannins", "phenolic acids"],
    },
    "Adesmia spinosissima": {
        "common": ["Añawaya"],
        "family": "Fabaceae",
        "use": "Resfríos, dolor de huesos",
        "region": ["Puno", "Arequipa", "Tacna"],
        "altitude": "3800-4600m",
        "compounds": ["flavonoids", "isoflavones"],
    },
    "Junellia minima": {
        "common": ["Verbena de puna"],
        "family": "Verbenaceae",
        "use": "Digestivo, ceremonial",
        "region": ["Puno", "Tacna"],
        "altitude": "4000-4800m",
        "compounds": ["iridoids", "flavonoids"],
    },
    # ── Additional medicinal crops ──
    "Amaranthus caudatus": {
        "common": ["Kiwicha", "Amaranto"],
        "family": "Amaranthaceae",
        "use": "Nutritivo, antianémico, osteoporosis",
        "region": ["Arequipa", "Cusco", "Puno"],
        "altitude": "2000-3500m",
        "compounds": ["squalene", "tocotrienols", "peptides"],
    },
    "Plukenetia volubilis": {
        "common": ["Sacha inchi"],
        "family": "Euphorbiaceae",
        "use": "Omega-3, antiinflamatorio, colesterol",
        "region": ["Cusco"],
        "altitude": "0-1500m",
        "compounds": ["alpha-linolenic acid", "tocopherols"],
    },
    "Myrciaria dubia": {
        "common": ["Camu camu"],
        "family": "Myrtaceae",
        "use": "Antioxidante, vitamina C, antiinflamatorio",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-500m",
        "compounds": ["ascorbic acid", "ellagic acid", "anthocyanins"],
    },
    "Mauritia flexuosa": {
        "common": ["Aguaje"],
        "family": "Arecaceae",
        "use": "Fitoestrógeno, antioxidante",
        "region": ["Cusco", "Madre de Dios"],
        "altitude": "0-500m",
        "compounds": ["beta-carotene", "tocopherols", "phytoestrogens"],
    },
    "Theobroma cacao": {
        "common": ["Cacao"],
        "family": "Malvaceae",
        "use": "Antioxidante, cardioprotector",
        "region": ["Cusco"],
        "altitude": "0-1200m",
        "compounds": ["theobromine", "epicatechin", "procyanidins"],
    },
    "Ilex guayusa": {
        "common": ["Guayusa"],
        "family": "Aquifoliaceae",
        "use": "Estimulante, antioxidante",
        "region": ["Cusco"],
        "altitude": "200-2000m",
        "compounds": ["caffeine", "theobromine", "chlorogenic acid"],
    },
    # ── Species with strong south Peru documentation ──
    "Achyrocline alata": {
        "common": ["Huira huira"],
        "family": "Asteraceae",
        "use": "Tos, bronquitis, gripe",
        "region": ["Cusco", "Puno", "Arequipa"],
        "altitude": "2500-4000m",
        "compounds": ["flavonoids", "caffeic acid derivatives"],
    },
    "Oenothera rosea": {
        "common": ["Chupasangre", "Yawar chonca"],
        "family": "Onagraceae",
        "use": "Antiinflamatorio, cicatrizante vaginal",
        "region": ["Arequipa", "Cusco", "Puno"],
        "altitude": "2000-3800m",
        "compounds": ["ellagitannins", "flavonoids"],
    },
    "Gentiana sedifolia": {
        "common": ["Phylli"],
        "family": "Gentianaceae",
        "use": "Hepático, febrífugo",
        "region": ["Puno", "Cusco"],
        "altitude": "3500-4500m",
        "compounds": ["xanthones", "secoiridoids"],
    },
    "Lepidium chichicara": {
        "common": ["Chichicara"],
        "family": "Brassicaceae",
        "use": "Diurético, antiinflamatorio prostático",
        "region": ["Arequipa", "Puno"],
        "altitude": "2500-3800m",
        "compounds": ["glucosinolates", "benzyl isothiocyanate"],
    },
    "Alternanthera porrigens": {
        "common": ["Quita mal"],
        "family": "Amaranthaceae",
        "use": "Hepático, depurativo",
        "region": ["Cusco", "Arequipa"],
        "altitude": "2000-3500m",
        "compounds": ["betacyanins", "flavonoids"],
    },
    "Piper aduncum": {
        "common": ["Matico"],
        "family": "Piperaceae",
        "use": "Hemostático, antiúlcera, antimicrobiano",
        "region": ["Cusco"],
        "altitude": "0-2500m",
        "compounds": ["dillapiole", "piperine", "chromenes"],
    },
    "Aristolochia didyma": {
        "common": ["Huamanlipa"],
        "family": "Aristolochiaceae",
        "use": "Respiratorio, bronquitis",
        "region": ["Cusco", "Arequipa"],
        "altitude": "2000-3500m",
        "compounds": ["aristolochic acids", "terpenoids"],
    },
    "Vallea stipularis": {
        "common": ["Chuillur"],
        "family": "Elaeocarpaceae",
        "use": "Reumatismo, dolor dental",
        "region": ["Cusco", "Puno"],
        "altitude": "2500-3800m",
        "compounds": ["tannins", "alkaloids"],
    },
    "Juglans neotropica": {
        "common": ["Nogal", "Tocte"],
        "family": "Juglandaceae",
        "use": "Astringente, tinte, anticaries",
        "region": ["Cusco", "Arequipa"],
        "altitude": "1500-3000m",
        "compounds": ["juglone", "ellagitannins", "gallic acid"],
    },
    "Berberis lutea": {
        "common": ["Cheqche", "Ayrampo amarillo"],
        "family": "Berberidaceae",
        "use": "Febrífugo, ocular, hepático",
        "region": ["Cusco", "Puno", "Arequipa"],
        "altitude": "2500-4000m",
        "compounds": ["berberine", "berbamine", "oxyacanthine"],
    },
    "Opuntia soehrensii": {
        "common": ["Ayrampo"],
        "family": "Cactaceae",
        "use": "Febrífugo, colorante, antiinflamatorio",
        "region": ["Puno", "Cusco", "Arequipa"],
        "altitude": "3000-4000m",
        "compounds": ["betalains", "betacyanins", "phenolics"],
    },
    "Grindelia boliviana": {
        "common": ["Chiri chiri"],
        "family": "Asteraceae",
        "use": "Antiinflamatorio, golpes, fracturas",
        "region": ["Puno", "Cusco", "Arequipa"],
        "altitude": "3000-4200m",
        "compounds": ["diterpenic acids", "grindelane", "flavonoids"],
    },
    "Gamochaeta americana": {
        "common": ["Qeto qeto"],
        "family": "Asteraceae",
        "use": "Hepático, digestivo",
        "region": ["Cusco", "Puno"],
        "altitude": "3000-4000m",
        "compounds": ["flavonoids", "sesquiterpene lactones"],
    },
    "Tessaria integrifolia": {
        "common": ["Pájaro bobo"],
        "family": "Asteraceae",
        "use": "Antiinflamatorio, diurético, hepatoprotector",
        "region": ["Arequipa", "Tacna", "Moquegua"],
        "altitude": "0-2500m",
        "compounds": ["flavonoids", "sesquiterpene lactones", "tannins"],
    },
    "Morinda citrifolia": {
        "common": ["Noni"],
        "family": "Rubiaceae",
        "use": "Inmunoestimulante, antioxidante, analgésico",
        "region": ["Cusco", "Junín", "Madre de Dios"],
        "altitude": "0-800m",
        "compounds": ["xeronine", "scopoletin", "damnacanthal", "anthraquinones"],
    },
}

EXPANDED_SPECIES = list(SPECIES_CATALOG.keys())

TARGET_SPECIES_ORIGINAL = TARGET_SPECIES[:]


def get_species_by_department(department: str) -> list[str]:
    return [sp for sp, info in SPECIES_CATALOG.items() if department in info.get("region", [])]


def get_species_by_departments(departments: list[str]) -> list[str]:
    result = set()
    for dept in departments:
        result.update(get_species_by_department(dept))
    return sorted(result)


def get_south_peru_species() -> list[str]:
    return get_species_by_departments(SOUTH_PERU_DEPARTMENTS)

SEARCH_QUERIES = [
    '"{species}"[Title/Abstract] AND (medicinal OR pharmacological OR therapeutic)',
    '"{species}"[Title/Abstract] AND (bioactive OR phytochemical OR ethnobotanical)',
    '"{species}"[Title/Abstract] AND (Peru OR Andes OR Andean)',
    '"{species}"[Title/Abstract] AND (traditional medicine OR ethnobotany)',
]

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

def get_embedding_dimension() -> int:
    """Read actual dimension from vectorization_info.json if available."""
    info_path = VECTORSTORE_DIR / "vectorization_info.json"
    if info_path.exists():
        import json
        with open(info_path, "r") as f:
            return json.load(f).get("dimension", EMBEDDING_DIMENSION)
    return EMBEDDING_DIMENSION

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", "]

RETRIEVAL_TOP_K = 30   # Plan A: was 20. Wider pool → better reranker input.
RERANK_TOP_K = 10      # Plan A: was 5. More survivors → higher Recall@k.
HYBRID_ALPHA = 0.6  # weight for dense retrieval (1-alpha for BM25)

GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

OLLAMA_MODEL = "qwen3.5:latest"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
