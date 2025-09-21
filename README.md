# waste-management-classification-rag
End-to-end project on intelligent waste management: image classification with CNN (transfer learning), text classification with SVM, and recycling instruction generation with Retrieval-Augmented Generation (RAG).

# Waste Management Classification with RAG

This repository contains a full pipeline for **intelligent waste management** using computer vision, natural language processing, and retrieval-augmented generation (RAG).  
It demonstrates how AI can classify waste items and generate recycling instructions based on municipal policy documents.

---

## 📌 Project Overview

The system integrates three major components:

1. **Image Classification (CNN + Transfer Learning)**  
   - Uses MobileNetV2 / EfficientNet as the base model.  
   - Classifies waste images into 9 categories.  
   - Includes fine-tuning, regularization, and error analysis.  

2. **Text Classification (Waste Descriptions)**  
   - Processes textual descriptions of waste items.  
   - Implements classical ML (SVM with TF–IDF).  
   - Achieves near-perfect classification performance.  

3. **Recycling Instruction Generation (RAG System)**  
   - Embeds and indexes municipal waste policy documents.  
   - Retrieves relevant instructions based on category queries.  
   - Generates structured recycling guidance with citations.  

---

## ♻️ Waste Categories

- Cardboard  
- Food Organics  
- Glass  
- Metal  
- Miscellaneous Trash  
- Paper  
- Plastic  
- Textile Trash  
- Vegetation  

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/waste-management-classification-rag.git
cd waste-management-classification-rag

Install dependencies (Python 3.9+ recommended):
pip install -r requirements.txt

🚀 Usage

1. Image Classification

Run the notebook for image classification:
jupyter notebook waste_management_summative.ipynb

Main steps:
	•	Preprocess images (tf.keras.utils.image_dataset_from_directory)
	•	Train CNN model (MobileNetV2 with transfer learning)
	•	Evaluate with accuracy, confusion matrix, and misclassified samples

⸻

2. Text Classification

Uses TF–IDF + SVM:
from text_model import classify_waste_description

print(classify_waste_description("empty brown glass bottle"))
# → Glass

3. RAG Instruction Generator

Generates recycling instructions from policy documents:
from rag_system import generate_recycling_instructions

out = generate_recycling_instructions("Glass")
print(out["instructions"])

RealWaste DatasetLinks to an external site.: Contains 3,000+ images of waste items classified into 8 categories (paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass)
https://archive.ics.uci.edu/dataset/908/realwaste

📊 Results

Image Classification
	•	Test Accuracy: ~77%
	•	High performance on Vegetation, Food Organics, and Metal
	•	Confusion mostly between visually similar items (e.g., Glass vs Plastic)

Text Classification
	•	Accuracy: 100% on validation and test sets
	•	Perfect precision, recall, and F1 across all categories

RAG Instruction Generation
	•	Successfully retrieves structured recycling rules
	•	Produces category-specific guidelines with references to policy documents

⸻

🔍 Example Outputs

Text Classification
	•	“banana peel” → Food Organics
	•	“aluminum soda can” → Metal

RAG Instruction Generator (Glass)
Acceptable:
- Glass bottles (all colors)
- Glass jars
- Glass food containers
- Glass beverage containers

Collection:
- Place in dedicated glass recycling bins
- Some areas require color sorting

📈 Future Work
	•	Improve CNN accuracy with data augmentation and deeper fine-tuning
	•	Explore transformer-based models (BERT) for text classification
	•	Enhance RAG with cross-document summarization and GPT integration
	•	Deploy as a web or mobile application for real-time waste sorting

⸻

📚 Acknowledgments
	•	TensorFlow / Keras for CNN modeling
	•	Scikit-learn for classical ML methods
	•	SentenceTransformers for document embeddings
	•	Municipal waste policy datasets for recycling guidelines
