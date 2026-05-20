"""
SIRCA-RAG Benchmark Dataset — 50 test cases for evaluation.
Covers factual (~20), exploratory (~15), comparative (~15) queries.
Bilingual EN/ES, multiple species and scientific angles.
"""
from dataclasses import dataclass, field


@dataclass
class TestCase:
    query: str
    reference_answer: str
    relevant_species: list[str] = field(default_factory=list)
    relevant_compounds: list[str] = field(default_factory=list)
    category: str = "factual"


BENCHMARK_SET = [
    # ================================================================
    # FACTUAL QUERIES (~20)
    # ================================================================
    TestCase(
        query="What are the main anti-inflammatory alkaloids in Uncaria tomentosa?",
        reference_answer="Uncaria tomentosa (Rubiaceae) contains pentacyclic oxindolic alkaloids (POA) with mitraphylline (MTP) being the most abundant, which modifies the inflammatory response. Recent studies report anti-inflammatory and anti-proliferative properties of different alkaloids extracted from this plant, including immunomodulatory and antitumor properties. A steroidic fraction showed beta-sitosterol (60%), stigmasterol, and campesterol with moderate anti-inflammatory activity.",
        relevant_species=["Uncaria tomentosa"],
        relevant_compounds=["mitraphylline"],
        category="factual",
    ),
    TestCase(
        query="What is the wound healing mechanism of taspine from Croton lechleri?",
        reference_answer="Taspine is the cicatrizant principle found in Sangre de Grado, the latex of Croton lechleri, with wound healing and anti-inflammatory biological activity. Taspine is an alkaloid present in Croton lechleri latex at approximately 9% by dry weight. Other alkaloids isolated from Croton lechleri include glaucine, isoboldine, magnoflorine, norisoboldine, and thaliporphine.",
        relevant_species=["Croton lechleri"],
        relevant_compounds=["taspine"],
        category="factual",
    ),
    TestCase(
        query="What is the geographic distribution of Buddleja incana in Peru?",
        reference_answer="Buddleja incana has georeferenced occurrences recorded in Peru according to GBIF data. These occurrences are distributed across the departments of Amazonas, Ancash, Arequipa, Cajamarca, Cusco, Huanuco, Junin, La Libertad, Lima, and Pasco.",
        relevant_species=["Buddleja incana"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="What are the fructooligosaccharides in yacon and their prebiotic effects?",
        reference_answer="Smallanthus sonchifolius (yacon) is a root rich in fructooligosaccharides (FOS) and inulin, which act as prebiotics. FOS intake favors the growth of health-promoting bacteria while reducing pathogenic bacteria populations. Commercial FOS can upregulate total secretory IgA in infant mice, providing prebiotic benefits.",
        relevant_species=["Smallanthus sonchifolius"],
        relevant_compounds=["fructooligosaccharides", "inulin"],
        category="factual",
    ),
    TestCase(
        query="What is the taxonomic classification of Lepidium meyenii?",
        reference_answer="Lepidium meyenii Walp. belongs to the family Brassicaceae, order Brassicales. It is commonly known as maca. The species is native to the high Andes of Peru, growing at altitudes between 3800-4500 meters above sea level. Multiple ecotypes are recognized based on hypocotyl color.",
        relevant_species=["Lepidium meyenii"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="Cual es la composicion quimica del aceite esencial de Minthostachys mollis?",
        reference_answer="El aceite esencial de Minthostachys mollis contiene principalmente pulegona como componente mayoritario, ademas de mentona, timol y otros monoterpenos. La composicion puede variar segun la region de origen y las condiciones de cultivo. Se han identificado compuestos como limoneno, carvacrol y linalool en diferentes quimiotipos.",
        relevant_species=["Minthostachys mollis"],
        relevant_compounds=["pulegone", "mentone", "thymol"],
        category="factual",
    ),
    TestCase(
        query="PMID references for Uncaria tomentosa immunomodulatory studies",
        reference_answer="Multiple PubMed-indexed studies have investigated the immunomodulatory properties of Uncaria tomentosa. Research includes studies on pentacyclic oxindole alkaloids affecting immune cell proliferation, in vitro assessments of NF-kB pathway modulation, and clinical trials evaluating immune function in cancer patients receiving cat's claw supplementation.",
        relevant_species=["Uncaria tomentosa"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="Contenido de macamidas en Lepidium meyenii",
        reference_answer="Lepidium meyenii contiene macamidas, que son amidas de acidos grasos con bencilamina, consideradas metabolitos secundarios unicos de la maca. Las macamidas han sido asociadas con efectos sobre el sistema endocannabinoide y propiedades neuroprotectoras. Tambien contiene macaenos, alcaloides como lepidilinas, y glucosinolatos.",
        relevant_species=["Lepidium meyenii"],
        relevant_compounds=["macamides", "macaenes"],
        category="factual",
    ),
    TestCase(
        query="What alkaloids are found in Erythroxylum coca leaves?",
        reference_answer="Erythroxylum coca leaves contain cocaine as the principal alkaloid, along with ecgonine, benzoylecgonine, methylecgonine, cinnamoylcocaine, and truxillines. The total alkaloid content varies by variety and growing conditions. The coca leaf also contains flavonoids, terpenes, and essential oils beyond the tropane alkaloid fraction.",
        relevant_species=["Erythroxylum coca"],
        relevant_compounds=["cocaine", "ecgonine"],
        category="factual",
    ),
    TestCase(
        query="Distribucion geografica de Physalis peruviana en el sur del Peru",
        reference_answer="Physalis peruviana, conocida como aguaymanto, se distribuye en los departamentos del sur del Peru incluyendo Cusco, Arequipa y Puno. Se cultiva principalmente entre los 1500 y 3000 metros de altitud. Los registros de GBIF documentan ocurrencias en multiples localidades de la region andina.",
        relevant_species=["Physalis peruviana"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="What phenolic compounds are present in Physalis peruviana fruit?",
        reference_answer="Physalis peruviana fruit contains various phenolic compounds including chlorogenic acid, caffeic acid, and rutin. The calyx fractions also show significant phenolic content with antioxidant activity. Withanolides, particularly physalins, are characteristic steroidal lactones found in this species.",
        relevant_species=["Physalis peruviana"],
        relevant_compounds=["chlorogenic acid", "withanolide"],
        category="factual",
    ),
    TestCase(
        query="Que compuestos bioactivos tiene Smallanthus sonchifolius en las hojas?",
        reference_answer="Las hojas de Smallanthus sonchifolius contienen acido clorogenico, acido cafeico, lactonas sesquiterpenicas y flavonoides. Los extractos de hojas han demostrado actividad antioxidante y propiedades hipoglucemiantes. Las hojas se utilizan tradicionalmente como te para el control de la diabetes.",
        relevant_species=["Smallanthus sonchifolius"],
        relevant_compounds=["chlorogenic acid"],
        category="factual",
    ),
    TestCase(
        query="IC50 values reported for Croton lechleri latex antimicrobial activity",
        reference_answer="Studies have reported antimicrobial activity of Croton lechleri latex against various pathogens. The latex contains proanthocyanidins, particularly SP-303, which has demonstrated broad-spectrum antimicrobial properties. Minimum inhibitory concentrations have been evaluated against Gram-positive and Gram-negative bacteria.",
        relevant_species=["Croton lechleri"],
        relevant_compounds=["taspine"],
        category="factual",
    ),
    TestCase(
        query="Buddleja incana common names in Peru",
        reference_answer="Buddleja incana is commonly known as kisuar, quishuar, or kolli in Quechua-speaking regions of Peru. In Spanish it is referred to as quishuar. The species is an important tree in Andean agroforestry systems and traditional medicine.",
        relevant_species=["Buddleja incana"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="Cantusia quercifolia usos medicinales tradicionales",
        reference_answer="Cantua buxifolia, conocida como cantuta o flor sagrada de los incas, es utilizada en la medicina tradicional andina. Sus flores y hojas se emplean para preparaciones contra afecciones respiratorias. Es una especie ornamental nativa de los Andes peruanos.",
        relevant_species=["Cantua buxifolia"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="What is the protein content of Chenopodium quinoa seeds?",
        reference_answer="Chenopodium quinoa seeds contain a high protein content ranging from 12-18% depending on variety. The protein has a balanced amino acid profile including all essential amino acids. Quinoa also contains saponins, flavonoids, and phenolic acids with antioxidant properties.",
        relevant_species=["Chenopodium quinoa"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="Actividad antifungica de Minthostachys mollis",
        reference_answer="Se ha demostrado la actividad antifungica del aceite esencial de Minthostachys mollis frente a hongos fitopatogenos y de importancia clinica. Los estudios reportan efectividad contra Candida albicans y dermatofitos. Los componentes activos principales son pulegona y timol.",
        relevant_species=["Minthostachys mollis"],
        relevant_compounds=["pulegone", "thymol"],
        category="factual",
    ),
    TestCase(
        query="Gentianella alborosea active compounds",
        reference_answer="Gentianella alborosea, known as hercampuri, contains bitter secoiridoid glycosides including amarogentin and gentiopicrin. These compounds are associated with hepatoprotective and hypoglycemic properties. The species is traditionally used in Andean medicine as a digestive aid.",
        relevant_species=["Gentianella alborosea"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="Taxonomia de Erythroxylum coca segun WFO",
        reference_answer="Erythroxylum coca Lam. pertenece a la familia Erythroxylaceae, orden Malpighiales. Segun el World Flora Online, se reconocen dos variedades principales: E. coca var. coca y E. coca var. ipadu. La especie es nativa de la region andina de Sudamerica.",
        relevant_species=["Erythroxylum coca"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="Schinus molle essential oil composition",
        reference_answer="Schinus molle essential oil contains alpha-phellandrene, beta-phellandrene, myrcene, limonene, and other monoterpenes as major components. The oil has demonstrated antimicrobial and insecticidal properties. Composition varies significantly with geographic origin and plant part used.",
        relevant_species=["Schinus molle"],
        relevant_compounds=[],
        category="factual",
    ),

    # ================================================================
    # EXPLORATORY QUERIES (~15)
    # ================================================================
    TestCase(
        query="Cuales son las propiedades medicinales de la maca para la fertilidad?",
        reference_answer="Lepidium meyenii (maca) posee propiedades medicinales para la fertilidad incluyendo mejora de la fertilidad y la libido, tratamiento de la infertilidad, y mejora del conteo y calidad del esperma. La maca ha sido utilizada tradicionalmente para mejorar la fertilidad, lo que sugiere su influencia en los sistemas endocrinos. Puede ser efectiva en la mejora del bienestar sexual tanto en hombres como en mujeres.",
        relevant_species=["Lepidium meyenii"],
        relevant_compounds=[],
        category="exploratory",
    ),
    TestCase(
        query="How does Erythroxylum coca differ from cocaine pharmacologically?",
        reference_answer="Erythroxylum coca is a plant species indigenous to the Andean region grown historically as a source of homeopathic medicine. Cocaine is a psychoactive substance extracted from Erythroxylum coca leaves, described as a potent stimulant of the sympathetic nervous system that causes structural changes on the brain, heart, lung, liver and kidney.",
        relevant_species=["Erythroxylum coca"],
        relevant_compounds=["cocaine"],
        category="exploratory",
    ),
    TestCase(
        query="Actividad antimicrobiana de Minthostachys mollis aceite esencial",
        reference_answer="La actividad antimicrobiana del aceite esencial de Minthostachys mollis ha sido evaluada frente a varias bacterias y hongos. Se determino la actividad antibacteriana del aceite esencial frente a Helicobacter pylori, Shigella dysenteriae, Salmonella typhi y Pseudomonas aeruginosa. Ademas se demostro actividad antimicotica in vitro del aceite esencial de las hojas.",
        relevant_species=["Minthostachys mollis"],
        relevant_compounds=[],
        category="exploratory",
    ),
    TestCase(
        query="How do Peruvian medicinal plants reduce inflammation?",
        reference_answer="Peruvian medicinal plants employ various anti-inflammatory mechanisms. Uncaria tomentosa alkaloids modulate NF-kB signaling and reduce pro-inflammatory cytokines. Croton lechleri latex contains proanthocyanidins with COX-2 inhibitory activity. Multiple species contain phenolic compounds that scavenge reactive oxygen species and reduce oxidative stress-mediated inflammation.",
        relevant_species=["Uncaria tomentosa", "Croton lechleri"],
        relevant_compounds=["mitraphylline"],
        category="exploratory",
    ),
    TestCase(
        query="Mecanismos de accion neuroprotectora de la maca",
        reference_answer="Lepidium meyenii ha demostrado propiedades neuroprotectoras a traves de multiples mecanismos. Las macamidas actuan como inhibidores de la enzima amida hidrolasa de acidos grasos (FAAH), modulando el sistema endocannabinoide. Estudios preclínicos reportan mejoras en la memoria y el aprendizaje, asi como proteccion contra el estres oxidativo neuronal.",
        relevant_species=["Lepidium meyenii"],
        relevant_compounds=["macamides"],
        category="exploratory",
    ),
    TestCase(
        query="What are the traditional uses of Buddleja incana in Andean medicine?",
        reference_answer="Buddleja incana is traditionally used in Andean medicine for treating respiratory ailments, wounds, and inflammation. The leaves are prepared as infusions or poultices. In Quechua traditional medicine, the species is valued for its anti-inflammatory and wound healing properties. The tree is also important in Andean agroforestry systems.",
        relevant_species=["Buddleja incana"],
        relevant_compounds=[],
        category="exploratory",
    ),
    TestCase(
        query="Potencial antidiabetico de Smallanthus sonchifolius",
        reference_answer="Smallanthus sonchifolius posee potencial antidiabetico demostrado en estudios preclínicos y clinicos. Los fructooligosacaridos de la raiz mejoran el metabolismo de la glucosa. Las hojas contienen acido clorogenico y lactonas sesquiterpenicas con actividad hipoglucemiante. El consumo de te de hojas de yacon ha mostrado reduccion de los niveles de glucosa en sangre.",
        relevant_species=["Smallanthus sonchifolius"],
        relevant_compounds=["fructooligosaccharides"],
        category="exploratory",
    ),
    TestCase(
        query="How can cat's claw be used in cancer research?",
        reference_answer="Uncaria tomentosa has shown potential in cancer research through multiple mechanisms. Pentacyclic oxindole alkaloids demonstrate anti-proliferative activity against various cancer cell lines. The bark extract has been studied for immunomodulatory effects that may enhance anti-tumor immune responses. Clinical studies have evaluated its use as complementary therapy in cancer patients.",
        relevant_species=["Uncaria tomentosa"],
        relevant_compounds=["mitraphylline"],
        category="exploratory",
    ),
    TestCase(
        query="Aplicaciones de la sangre de grado en dermatologia",
        reference_answer="La sangre de grado (Croton lechleri) tiene multiples aplicaciones dermatologicas documentadas. El latex contiene taspina con propiedades cicatrizantes y SP-303 con actividad antiviral. Se ha utilizado para el tratamiento de heridas, ulceras cutaneas, picaduras de insectos y herpes. Los proantocianidinos del latex aceleran la formacion de costras y la regeneracion tisular.",
        relevant_species=["Croton lechleri"],
        relevant_compounds=["taspine"],
        category="exploratory",
    ),
    TestCase(
        query="What role do withanolides play in Physalis peruviana bioactivity?",
        reference_answer="Withanolides in Physalis peruviana, particularly physalins, exhibit diverse biological activities. These steroidal lactones have demonstrated anti-inflammatory, cytotoxic, and immunomodulatory properties. Physalin B and physalin F have been studied for anti-tumor activity. The withanolide content contributes significantly to the medicinal value of the species.",
        relevant_species=["Physalis peruviana"],
        relevant_compounds=["withanolide"],
        category="exploratory",
    ),
    TestCase(
        query="Como se utiliza la etnobotanica para descubrir nuevos farmacos en Peru?",
        reference_answer="La etnobotanica en Peru ha sido fundamental para el descubrimiento de farmacos basados en plantas medicinales. El conocimiento tradicional de comunidades andinas y amazonicas guia la seleccion de especies para investigacion fitoquimica. Ejemplos exitosos incluyen la vinblastina y vincristina derivadas de Catharanthus roseus, y los estudios actuales sobre alcaloides de Uncaria tomentosa y Croton lechleri.",
        relevant_species=["Uncaria tomentosa", "Croton lechleri"],
        relevant_compounds=[],
        category="exploratory",
    ),
    TestCase(
        query="Potential of Andean plants for treating gastrointestinal diseases",
        reference_answer="Several Andean medicinal plants show potential for treating gastrointestinal diseases. Croton lechleri latex has been studied for anti-diarrheal effects. Minthostachys mollis is traditionally used for digestive problems. Smallanthus sonchifolius prebiotic fructooligosaccharides promote healthy gut microbiota. Gentianella alborosea is used as a digestive bitter.",
        relevant_species=["Croton lechleri", "Minthostachys mollis", "Smallanthus sonchifolius"],
        relevant_compounds=["fructooligosaccharides"],
        category="exploratory",
    ),
    TestCase(
        query="Efectos adaptogenos de Lepidium meyenii en altitud",
        reference_answer="Lepidium meyenii ha sido estudiada por sus propiedades adaptogenas, particularmente en condiciones de alta altitud. La maca crece naturalmente entre 3800-4500 msnm y ha sido utilizada tradicionalmente para mejorar la resistencia fisica y la adaptacion a la altitud. Estudios sugieren efectos sobre el eje hipotalamo-hipofisis-adrenal y mejora de la capacidad energetica.",
        relevant_species=["Lepidium meyenii"],
        relevant_compounds=["macamides"],
        category="exploratory",
    ),
    TestCase(
        query="How do cross-encoder rerankers improve retrieval for medicinal plant queries?",
        reference_answer="Cross-encoder rerankers improve retrieval by performing deep query-document interaction analysis. Unlike bi-encoders that encode query and document independently, cross-encoders jointly process the pair to capture semantic nuances. In the medicinal plant domain, this helps distinguish between species with similar names and differentiate between pharmacological properties of related compounds.",
        relevant_species=[],
        relevant_compounds=[],
        category="exploratory",
    ),
    TestCase(
        query="Importancia de la conservacion de plantas medicinales en los Andes peruanos",
        reference_answer="La conservacion de plantas medicinales en los Andes peruanos es critica debido a la perdida de habitat, sobreexplotacion y cambio climatico. Especies como Buddleja incana y Gentianella alborosea enfrentan presion por recoleccion silvestre. Los programas de conservacion in situ y ex situ son esenciales para preservar tanto la biodiversidad como el conocimiento etnobotanico asociado.",
        relevant_species=["Buddleja incana", "Gentianella alborosea"],
        relevant_compounds=[],
        category="exploratory",
    ),

    # ================================================================
    # COMPARATIVE QUERIES (~15)
    # ================================================================
    TestCase(
        query="Compare the antioxidant activity of Physalis peruviana and Smallanthus sonchifolius",
        reference_answer="Physalis peruviana crude ethanolic extract and calyx fractions were evaluated for antioxidant activity via superoxide and nitric oxide scavenging activity. Smallanthus sonchifolius landraces were investigated for total phenolic content, antioxidant activity and chemical composition of ethanol extracts and decoction extracts. Both species show antioxidant properties but use different methodologies and extract types.",
        relevant_species=["Physalis peruviana", "Smallanthus sonchifolius"],
        relevant_compounds=[],
        category="comparative",
    ),
    TestCase(
        query="Comparar los alcaloides de Uncaria tomentosa vs Erythroxylum coca",
        reference_answer="Uncaria tomentosa contiene alcaloides oxindolicos pentaciclicos como mitrafillina, isomitrafillina, pteropodina y rincofillina, con propiedades inmunomoduladoras. Erythroxylum coca contiene alcaloides tropanicosomo cocaina, ecgonina y benzoilecgonina, con actividad estimulante del sistema nervioso. Las dos familias de alcaloides son estructuralmente distintas y tienen mecanismos de accion diferentes.",
        relevant_species=["Uncaria tomentosa", "Erythroxylum coca"],
        relevant_compounds=["mitraphylline", "cocaine"],
        category="comparative",
    ),
    TestCase(
        query="Differences between dense and sparse retrieval for ethnobotanical queries",
        reference_answer="Dense retrieval using neural embeddings captures semantic similarity and handles synonyms well, while sparse retrieval like BM25 excels at exact term matching for species names and chemical compounds. Hybrid approaches combining both methods with reciprocal rank fusion leverage the strengths of each for optimal retrieval in the ethnobotanical domain.",
        relevant_species=[],
        relevant_compounds=[],
        category="comparative",
    ),
    TestCase(
        query="Compare wound healing properties of Croton lechleri and Buddleja incana",
        reference_answer="Croton lechleri latex contains taspine and SP-303 proanthocyanidins with well-documented wound healing mechanisms including fibroblast migration stimulation and collagen synthesis. Buddleja incana has traditional use for wound treatment in Andean medicine but has fewer published pharmacological studies. Both species are used topically in traditional medicine for skin healing.",
        relevant_species=["Croton lechleri", "Buddleja incana"],
        relevant_compounds=["taspine"],
        category="comparative",
    ),
    TestCase(
        query="Maca vs quinoa: perfil nutricional comparado",
        reference_answer="Lepidium meyenii (maca) destaca por su contenido de macamidas, macaenos y glucosinolatos con propiedades adaptogenas y endocrinas. Chenopodium quinoa posee un perfil proteico superior con todos los aminoacidos esenciales y alto contenido de fibra. Ambas son cultivos andinos de alta altitud pero con perfiles fitoquimicos y usos medicinales distintos.",
        relevant_species=["Lepidium meyenii", "Chenopodium quinoa"],
        relevant_compounds=["macamides"],
        category="comparative",
    ),
    TestCase(
        query="Compare antimicrobial activities of Minthostachys mollis and Schinus molle essential oils",
        reference_answer="Minthostachys mollis essential oil, rich in pulegone and menthone, has demonstrated activity against Helicobacter pylori and Candida species. Schinus molle essential oil, dominated by alpha-phellandrene and limonene, shows broad-spectrum antimicrobial and insecticidal properties. Both oils are used in traditional Andean medicine for infections but contain different chemical profiles.",
        relevant_species=["Minthostachys mollis", "Schinus molle"],
        relevant_compounds=["pulegone", "mentone"],
        category="comparative",
    ),
    TestCase(
        query="Physalis peruviana vs Smallanthus sonchifolius: actividad hipoglucemiante",
        reference_answer="Physalis peruviana ha mostrado efectos hipoglucemiantes atribuidos a withanolidos y compuestos fenolicos en estudios preclínicos. Smallanthus sonchifolius reduce los niveles de glucosa a traves de fructooligosacaridos prebioticos y acido clorogenico en las hojas. Ambas especies son estudiadas para el manejo de la diabetes pero actuan por mecanismos diferentes.",
        relevant_species=["Physalis peruviana", "Smallanthus sonchifolius"],
        relevant_compounds=["fructooligosaccharides", "withanolide"],
        category="comparative",
    ),
    TestCase(
        query="Compare immunomodulatory properties of Uncaria tomentosa and Lepidium meyenii",
        reference_answer="Uncaria tomentosa demonstrates immunomodulatory activity through pentacyclic oxindole alkaloids that affect T-cell and B-cell proliferation, NF-kB signaling, and cytokine production. Lepidium meyenii modulates immune function through macamides and polysaccharides, with effects on adaptive immunity and stress response. Both species enhance immune function but through distinct molecular pathways.",
        relevant_species=["Uncaria tomentosa", "Lepidium meyenii"],
        relevant_compounds=["mitraphylline", "macamides"],
        category="comparative",
    ),
    TestCase(
        query="Diferencias entre aguaymanto y yacon como alimentos funcionales",
        reference_answer="Physalis peruviana (aguaymanto) es valorada por su contenido de vitamina C, carotenoides y withanolidos con propiedades antioxidantes y antiinflamatorias. Smallanthus sonchifolius (yacon) destaca por sus fructooligosacaridos prebioticos y bajo indice glucemico. Ambos son alimentos funcionales andinos pero con diferentes compuestos bioactivos predominantes.",
        relevant_species=["Physalis peruviana", "Smallanthus sonchifolius"],
        relevant_compounds=["fructooligosaccharides", "withanolide"],
        category="comparative",
    ),
    TestCase(
        query="Compare GBIF occurrence data: Uncaria tomentosa vs Croton lechleri in southern Peru",
        reference_answer="Both Uncaria tomentosa and Croton lechleri have georeferenced occurrences in Peru documented through GBIF. Uncaria tomentosa is primarily found in the Amazon lowland forests while Croton lechleri occurs in both lowland and montane forests. Their distributions overlap in transitional zones between the Andes and the Amazon basin.",
        relevant_species=["Uncaria tomentosa", "Croton lechleri"],
        relevant_compounds=[],
        category="comparative",
    ),
    TestCase(
        query="Croton lechleri vs Uncaria tomentosa: cual tiene mas estudios en PubMed?",
        reference_answer="Ambas especies tienen una cantidad significativa de publicaciones en PubMed. Uncaria tomentosa tiene un mayor numero de articulos indexados, cubriendo inmunomodulacion, actividad antitumoral y anti-inflamatoria. Croton lechleri tiene menos publicaciones pero con enfoque en cicatrizacion, actividad antiviral y antimicrobiana del latex. La diferencia refleja el mayor interes comercial en una de gato.",
        relevant_species=["Croton lechleri", "Uncaria tomentosa"],
        relevant_compounds=[],
        category="comparative",
    ),
    TestCase(
        query="Dense retrieval vs BM25 for species name queries",
        reference_answer="BM25 sparse retrieval is more effective for exact species name queries because it matches the precise binomial nomenclature tokens. Dense retrieval may retrieve semantically similar but taxonomically different species. For compound names and specific biochemical terms, BM25 also tends to outperform dense models that may confuse structurally similar compound names.",
        relevant_species=[],
        relevant_compounds=[],
        category="comparative",
    ),
    TestCase(
        query="Compare prebiotic potential of yacon FOS versus commercial inulin",
        reference_answer="Yacon fructooligosaccharides from Smallanthus sonchifolius have shorter chain lengths than commercial chicory-derived inulin, resulting in faster fermentation by gut bacteria. Both promote Bifidobacterium and Lactobacillus growth. Yacon FOS may have additional benefits from co-present phenolic compounds. Commercial inulin has more standardized dosing and clinical evidence.",
        relevant_species=["Smallanthus sonchifolius"],
        relevant_compounds=["fructooligosaccharides", "inulin"],
        category="comparative",
    ),
    TestCase(
        query="Erythroxylum coca hoja vs extracto purificado: diferencias farmacologicas",
        reference_answer="La hoja entera de Erythroxylum coca contiene una mezcla compleja de alcaloides, flavonoides, vitaminas y minerales que modulan los efectos de la cocaina. El extracto purificado concentra los alcaloides tropanícos, especialmente cocaina, eliminando los compuestos protectores. Estudios sugieren que el uso tradicional de la hoja tiene un perfil farmacologico diferente al del alcaloide aislado.",
        relevant_species=["Erythroxylum coca"],
        relevant_compounds=["cocaine"],
        category="comparative",
    ),
    TestCase(
        query="Compare anti-inflammatory mechanisms: alkaloids vs phenolic compounds in Peruvian plants",
        reference_answer="Alkaloids from species like Uncaria tomentosa primarily act through NF-kB pathway modulation, TNF-alpha reduction, and immune cell regulation. Phenolic compounds found across multiple Peruvian medicinal plants act through COX-2 inhibition, reactive oxygen species scavenging, and MAPK pathway modulation. Both classes contribute to anti-inflammatory activity but through complementary molecular mechanisms.",
        relevant_species=["Uncaria tomentosa"],
        relevant_compounds=["mitraphylline"],
        category="comparative",
    ),
]
