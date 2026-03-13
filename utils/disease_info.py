"""
Plant Disease Information Database
Contains disease descriptions, symptoms, and treatment recommendations
for all 38 classes in the PlantVillage dataset.
"""

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "description": "A fungal disease caused by Venturia inaequalis that affects apple trees, causing dark, scabby lesions on leaves and fruit.",
        "symptoms": [
            "Olive-green to dark brown velvety spots on leaves",
            "Distorted and crinkled leaves",
            "Dark, scabby lesions on fruit surface",
            "Premature leaf and fruit drop"
        ],
        "treatment": [
            "Apply fungicides (captan, myclobutanil) during spring",
            "Remove and destroy fallen infected leaves",
            "Prune trees to improve air circulation",
            "Plant scab-resistant apple varieties",
            "Maintain proper tree spacing"
        ],
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Botryosphaeria obtusa that affects leaves, fruit, and bark of apple trees.",
        "symptoms": [
            "Frogeye leaf spots (brown with purple borders)",
            "Black, rotting lesions on fruit starting from the calyx end",
            "Cankers on branches and trunk",
            "Mummified fruit on the tree"
        ],
        "treatment": [
            "Prune out dead and infected branches",
            "Remove mummified fruits and cankers",
            "Apply fungicides like captan or thiophanate-methyl",
            "Maintain tree vigor through proper fertilization",
            "Ensure good air circulation"
        ],
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "A fungal disease caused by Gymnosporangium juniperi-virginianae that requires both apple and cedar/juniper trees to complete its lifecycle.",
        "symptoms": [
            "Bright orange-yellow spots on upper leaf surface",
            "Tube-like structures on leaf undersides",
            "Spots on fruit surface",
            "Premature defoliation in severe cases"
        ],
        "treatment": [
            "Remove nearby cedar and juniper trees within a few hundred feet",
            "Apply fungicides (myclobutanil, triadimefon) at pink bud stage",
            "Plant rust-resistant apple varieties",
            "Remove galls from cedar trees in early spring"
        ],
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": "This apple leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Continue regular watering and fertilization",
            "Maintain proper pruning schedule",
            "Monitor regularly for early signs of disease",
            "Apply preventive fungicide sprays during wet seasons"
        ],
        "severity": "None"
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": "This blueberry leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Maintain acidic soil pH (4.5-5.5)",
            "Regular mulching with pine needles or wood chips",
            "Adequate watering, especially during fruiting",
            "Prune old canes annually"
        ],
        "severity": "None"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "description": "A fungal disease caused by Podosphaera clandestina that creates a white powdery coating on cherry leaves.",
        "symptoms": [
            "White powdery patches on leaf surfaces",
            "Curling and distortion of new leaves",
            "Stunted shoot growth",
            "Reduced fruit quality"
        ],
        "treatment": [
            "Apply sulfur-based fungicides early in the season",
            "Improve air circulation through pruning",
            "Avoid overhead irrigation",
            "Remove and destroy infected plant parts",
            "Apply neem oil as an organic alternative"
        ],
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": "This cherry leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Continue regular care and maintenance",
            "Prune to maintain good tree structure",
            "Water deeply but infrequently",
            "Monitor for pest and disease signs"
        ],
        "severity": "None"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)",
        "disease": "Gray Leaf Spot",
        "description": "A fungal disease caused by Cercospora zeae-maydis, one of the most significant yield-limiting diseases of corn worldwide.",
        "symptoms": [
            "Rectangular gray to tan lesions between leaf veins",
            "Lesions may coalesce and kill entire leaves",
            "Lower leaves affected first",
            "Reduced photosynthesis and yield"
        ],
        "treatment": [
            "Plant resistant corn hybrids",
            "Rotate crops (avoid corn-on-corn)",
            "Apply foliar fungicides (strobilurins, triazoles)",
            "Tillage to bury infected crop residue",
            "Ensure adequate plant spacing"
        ],
        "severity": "High"
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)",
        "disease": "Common Rust",
        "description": "A fungal disease caused by Puccinia sorghi that produces reddish-brown pustules on corn leaves.",
        "symptoms": [
            "Small reddish-brown to cinnamon-brown pustules on both leaf surfaces",
            "Pustules scattered across the leaf blade",
            "Yellowing around pustule clusters",
            "Premature leaf death in severe cases"
        ],
        "treatment": [
            "Plant rust-resistant corn hybrids",
            "Apply foliar fungicides if detected early",
            "Monitor fields regularly during cool, humid weather",
            "Remove volunteer corn plants"
        ],
        "severity": "Moderate"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "description": "A fungal disease caused by Exserohilum turcicum that creates large cigar-shaped lesions on corn leaves.",
        "symptoms": [
            "Large (1-6 inch) cigar-shaped gray-green lesions",
            "Lesions start on lower leaves and move upward",
            "Grayish spore production on lesion surface",
            "Significant yield loss if infection occurs before tasseling"
        ],
        "treatment": [
            "Plant resistant hybrids with Ht genes",
            "Apply foliar fungicides at early infection",
            "Practice crop rotation",
            "Manage crop residue through tillage",
            "Avoid late planting"
        ],
        "severity": "High"
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)",
        "disease": "Healthy",
        "description": "This corn leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Maintain proper fertilization schedule",
            "Ensure adequate irrigation",
            "Scout regularly for pests and diseases",
            "Practice crop rotation"
        ],
        "severity": "None"
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Guignardia bidwellii that severely affects grape production worldwide.",
        "symptoms": [
            "Circular tan to brown leaf spots with dark borders",
            "Small black pycnidia in leaf spots",
            "Fruit turns brown, then black and shriveled (mummified)",
            "Cankers on shoots and tendrils"
        ],
        "treatment": [
            "Apply fungicides (mancozeb, myclobutanil) from bud break",
            "Remove mummified berries and infected canes",
            "Maintain good canopy management",
            "Ensure proper vine spacing for air flow",
            "Sanitation of vineyard floor"
        ],
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "description": "A complex fungal disease involving multiple pathogens that causes chronic decline in grapevines.",
        "symptoms": [
            "Tiger-stripe pattern on leaves (interveinal chlorosis)",
            "Dark spots on berries resembling measles",
            "Internal wood discoloration and decay",
            "Sudden vine collapse in severe cases (apoplexy)"
        ],
        "treatment": [
            "No complete cure available; manage symptoms",
            "Trunk surgery to remove infected wood",
            "Protect pruning wounds with fungicide paste",
            "Remedial surgery (trunk renewal)",
            "Avoid heavy pruning during wet conditions"
        ],
        "severity": "High"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "description": "A fungal disease caused by Pseudocercospora vitis that affects grape leaves.",
        "symptoms": [
            "Dark brown spots with yellow halos on leaves",
            "Spots may coalesce causing large necrotic areas",
            "Premature defoliation",
            "Reduced vine vigor"
        ],
        "treatment": [
            "Apply copper-based fungicides",
            "Remove and destroy infected leaves",
            "Improve canopy management for air circulation",
            "Avoid overhead irrigation",
            "Apply preventive fungicide before rainy season"
        ],
        "severity": "Moderate"
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": "This grape leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Continue regular vineyard management",
            "Maintain proper canopy management",
            "Regular soil testing and fertilization",
            "Monitor for early disease signs"
        ],
        "severity": "None"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "description": "A devastating bacterial disease caused by Candidatus Liberibacter spp., spread by the Asian citrus psyllid.",
        "symptoms": [
            "Asymmetric blotchy mottling of leaves",
            "Yellow shoots (yellow dragon symptom)",
            "Lopsided, bitter, small fruit with green coloring",
            "Premature fruit drop",
            "Overall tree decline"
        ],
        "treatment": [
            "No cure available; management strategies only",
            "Control Asian citrus psyllid vector with insecticides",
            "Remove and destroy infected trees",
            "Plant certified disease-free nursery stock",
            "Nutritional sprays to manage symptoms"
        ],
        "severity": "Critical"
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease caused by Xanthomonas arboricola pv. pruni that affects peach leaves, fruit, and twigs.",
        "symptoms": [
            "Small, angular, water-soaked lesions on leaves",
            "Lesions turn purple-brown, centers may fall out (shot hole)",
            "Dark, sunken spots on fruit",
            "Twig cankers and dieback"
        ],
        "treatment": [
            "Apply copper-based bactericides during dormant season",
            "Plant resistant varieties",
            "Avoid overhead irrigation",
            "Maintain proper tree nutrition",
            "Prune to improve air circulation"
        ],
        "severity": "Moderate"
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": "This peach leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Continue regular care schedule",
            "Annual dormant spray application",
            "Proper thinning of fruit",
            "Regular pruning for tree shape and airflow"
        ],
        "severity": "None"
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease caused by Xanthomonas campestris pv. vesicatoria that is one of the most devastating diseases of peppers.",
        "symptoms": [
            "Small, dark, water-soaked spots on leaves",
            "Raised, scabby spots on fruit",
            "Leaf yellowing and premature drop",
            "Stem and petiole lesions"
        ],
        "treatment": [
            "Use disease-free certified seeds",
            "Apply copper-based sprays with mancozeb",
            "Practice crop rotation (2-3 years)",
            "Avoid working in fields when plants are wet",
            "Remove and destroy infected plant debris"
        ],
        "severity": "High"
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "description": "This bell pepper leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Maintain consistent watering",
            "Apply balanced fertilizer regularly",
            "Stake plants for support",
            "Monitor for pests like aphids"
        ],
        "severity": "None"
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani that is common in warm, humid conditions.",
        "symptoms": [
            "Dark brown concentric rings on older leaves (target spots)",
            "Yellowing around lesions",
            "Lower leaves affected first, progressing upward",
            "Dark, leathery sunken lesions on tubers"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb) preventively",
            "Practice crop rotation (3-year cycle)",
            "Remove infected plant debris",
            "Maintain adequate plant nutrition",
            "Avoid overhead irrigation"
        ],
        "severity": "Moderate"
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "description": "A devastating disease caused by the oomycete Phytophthora infestans — the same pathogen responsible for the Irish Potato Famine.",
        "symptoms": [
            "Water-soaked gray-green spots on leaf edges",
            "White fuzzy mold growth on leaf undersides in humid conditions",
            "Rapid browning and death of foliage",
            "Firm, dark brown rot on tubers"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, metalaxyl) preventively",
            "Destroy volunteer potato plants",
            "Plant certified disease-free seed potatoes",
            "Destroy infected plants immediately",
            "Harvest tubers during dry conditions"
        ],
        "severity": "Critical"
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": "This potato leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Hill soil around stems regularly",
            "Maintain consistent moisture levels",
            "Scout for Colorado potato beetles",
            "Apply preventive fungicide in wet seasons"
        ],
        "severity": "None"
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": "This raspberry leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Prune spent canes after harvest",
            "Maintain good air circulation",
            "Apply mulch to retain moisture",
            "Regular fertilization in spring"
        ],
        "severity": "None"
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": "This soybean leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Continue proper crop rotation",
            "Monitor for pest activity",
            "Ensure adequate soil drainage",
            "Test soil and adjust nutrients as needed"
        ],
        "severity": "None"
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "description": "A very common fungal disease caused by Erysiphe cichoracearum or Podosphaera xanthii that affects cucurbits.",
        "symptoms": [
            "White to gray powdery patches on upper leaf surfaces",
            "Spots may expand to cover entire leaf",
            "Yellowing and browning of affected leaves",
            "Premature leaf death reducing fruit quality"
        ],
        "treatment": [
            "Apply fungicides (sulfur, potassium bicarbonate)",
            "Use neem oil or horticultural oils",
            "Plant resistant varieties",
            "Improve air circulation and reduce humidity",
            "Water at the base of plants, not overhead"
        ],
        "severity": "Moderate"
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "description": "A fungal disease caused by Diplocarpon earlianum that is common in established strawberry beds.",
        "symptoms": [
            "Irregular dark purple to brown spots on leaves",
            "Spots merge causing a scorched appearance",
            "Leaf edges turn brown and curl upward",
            "Reduced plant vigor and yield"
        ],
        "treatment": [
            "Remove and destroy infected leaves",
            "Apply fungicides (captan, myclobutanil)",
            "Renovate strawberry beds after harvest",
            "Ensure good air circulation between plants",
            "Avoid overhead irrigation"
        ],
        "severity": "Moderate"
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": "This strawberry leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Maintain proper mulching",
            "Regular watering during fruiting",
            "Fertilize after harvest",
            "Remove runners to control plant spacing"
        ],
        "severity": "None"
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease caused by Xanthomonas species that thrives in warm, wet conditions.",
        "symptoms": [
            "Small, dark, water-soaked spots on leaves",
            "Spots may have yellow halos",
            "Raised, scab-like spots on fruit",
            "Severe defoliation in wet weather"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Use pathogen-free seeds and transplants",
            "Rotate crops for 2-3 years",
            "Avoid overhead watering",
            "Remove and destroy infected plant debris"
        ],
        "severity": "High"
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description": "A common fungal disease caused by Alternaria solani affecting tomatoes in warm, humid environments.",
        "symptoms": [
            "Concentric ring pattern (target-like) lesions on lower leaves",
            "Dark brown spots with yellow halos",
            "Progressive defoliation from bottom up",
            "Dark sunken lesions on fruit stem end"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, copper sprays)",
            "Mulch around plants to prevent soil splash",
            "Stake or cage plants for better air circulation",
            "Water at the base of plants",
            "Practice 3-year crop rotation"
        ],
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description": "A destructive disease caused by Phytophthora infestans that can devastate entire tomato crops rapidly.",
        "symptoms": [
            "Large, irregular water-soaked spots on leaves",
            "White fuzzy growth on leaf undersides",
            "Rapid browning and death of leaves and stems",
            "Greasy-looking brown spots on fruit"
        ],
        "treatment": [
            "Apply preventive fungicides before symptoms appear",
            "Remove and destroy infected plants immediately",
            "Avoid overhead watering",
            "Improve air circulation in garden",
            "Do not compost infected plant material"
        ],
        "severity": "Critical"
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "A fungal disease caused by Passalora fulva (formerly Cladosporium fulvum) that thrives in high humidity.",
        "symptoms": [
            "Pale green to yellow spots on upper leaf surface",
            "Olive-green to gray fuzzy mold on leaf undersides",
            "Leaves curl, wither, and drop",
            "Primarily affects greenhouse-grown tomatoes"
        ],
        "treatment": [
            "Improve greenhouse ventilation",
            "Reduce humidity below 85%",
            "Apply fungicides (chlorothalonil, mancozeb)",
            "Remove infected leaves promptly",
            "Use resistant tomato varieties"
        ],
        "severity": "Moderate"
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": "A common fungal disease caused by Septoria lycopersici that affects tomato foliage.",
        "symptoms": [
            "Numerous small, circular spots with dark borders and tan centers",
            "Tiny black dots (pycnidia) visible in spot centers",
            "Lower leaves affected first",
            "Progressive defoliation reducing fruit quality"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, copper) at first sign",
            "Remove lower leaves showing symptoms",
            "Mulch to prevent soil splash",
            "Avoid overhead irrigation",
            "Practice crop rotation"
        ],
        "severity": "Moderate"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites (Two-Spotted Spider Mite)",
        "description": "An infestation by Tetranychus urticae, tiny arachnids that feed on plant sap causing significant damage.",
        "symptoms": [
            "Fine stippling (tiny yellow/white dots) on leaves",
            "Fine webbing on leaf undersides",
            "Bronzing and drying of leaves",
            "Overall plant decline in severe infestations"
        ],
        "treatment": [
            "Spray plants with strong water jet to dislodge mites",
            "Apply miticides (abamectin, spiromesifen)",
            "Release predatory mites (Phytoseiulus persimilis)",
            "Apply neem oil or insecticidal soap",
            "Maintain adequate plant hydration"
        ],
        "severity": "Moderate"
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "description": "A fungal disease caused by Corynespora cassiicola that produces distinctive target-like lesions.",
        "symptoms": [
            "Brown spots with concentric rings (target pattern)",
            "Spots may have yellow halos",
            "Lesions on leaves, stems, and fruit",
            "Defoliation in severe cases"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, azoxystrobin)",
            "Maintain good air circulation",
            "Avoid working with wet plants",
            "Practice crop rotation",
            "Remove crop debris after harvest"
        ],
        "severity": "Moderate"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "description": "A devastating viral disease transmitted by whiteflies (Bemisia tabaci) affecting tomato production worldwide.",
        "symptoms": [
            "Severe upward curling and cupping of leaves",
            "Yellowing of leaf margins and between veins",
            "Stunted plant growth",
            "Flower drop and significantly reduced fruit set"
        ],
        "treatment": [
            "No cure for infected plants; manage vector",
            "Control whiteflies with insecticides or sticky traps",
            "Use reflective mulches to repel whiteflies",
            "Plant TYLCV-resistant tomato varieties",
            "Remove and destroy infected plants promptly"
        ],
        "severity": "Critical"
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "description": "A highly contagious viral disease that spreads through mechanical contact and contaminated tools.",
        "symptoms": [
            "Mosaic pattern of light and dark green on leaves",
            "Leaf curling and distortion",
            "Stunted growth",
            "Uneven fruit ripening with brown internal discoloration"
        ],
        "treatment": [
            "No cure available; prevention is key",
            "Sanitize tools and hands with milk or bleach solution",
            "Plant resistant varieties (carrying Tm-2 gene)",
            "Remove and destroy infected plants",
            "Do not smoke near plants (tobacco mosaic cross-contamination)"
        ],
        "severity": "High"
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": "This tomato leaf shows no signs of disease. The plant appears to be in good health.",
        "symptoms": ["No disease symptoms detected"],
        "treatment": [
            "Continue regular watering and feeding",
            "Stake or cage for support",
            "Prune suckers for better air circulation",
            "Monitor regularly for pests and diseases"
        ],
        "severity": "None"
    }
}

# Class names in the order used by the model (alphabetical)
CLASS_NAMES = sorted(DISEASE_INFO.keys())

# Severity color mapping
SEVERITY_COLORS = {
    "None": "#28a745",      # Green
    "Moderate": "#ffc107",   # Yellow
    "High": "#fd7e14",       # Orange
    "Critical": "#dc3545"    # Red
}

def get_disease_info(class_name: str) -> dict:
    """Get disease information for a given class name."""
    return DISEASE_INFO.get(class_name, None)

def get_severity_color(severity: str) -> str:
    """Get the color associated with a severity level."""
    return SEVERITY_COLORS.get(severity, "#6c757d")
