# Master's Thesis Research: Cross-Modal Sound Symbolism in Vision-Language Models

This repository contains the research code and experiments for a Master's dissertation investigating sound symbolism in large multimodal models, awarded with a grade of **8/10**.

## 🔬 Research Overview

This research explores the **bouba-kiki effect** (sound symbolism) in modern vision-language models, examining how these AI systems associate visual shapes with phonetic properties of pseudo-words. The study investigates whether multimodal AI models exhibit similar cross-modal correspondences as humans do between visual shapes and sound patterns.

### Key Research Questions
- Do vision-language models demonstrate sound symbolism similar to humans?
- How do different model architectures (LLaMA vs Molmo) handle cross-modal associations?
- Can models reliably segment and identify abstract concepts based on phonetic cues?

## 📁 Repository Structure

```
├── experiment-1/          # Cross-Modal Probability Analysis
│   ├── llama/             # LLaMA 3.2 experiments
│   └── molmo/             # Molmo 7B experiments
├── experiment-2/          # Image-to-Text Matching
│   ├── llama/             # LLaMA classification tasks
│   └── molmo/             # Molmo classification tasks
├── experiment-3/          # Visual Grounding with SAM2
├── imgs/                  # Experimental stimuli
│   ├── images/            # Individual shape images (curved/jagged)
│   └── concat_images/     # Combined image pairs
└── Rob Master thesis - final.pdf
```

## 🧪 Experiments

### Experiment 1: Cross-Modal Probability Analysis
**Models:** LLaMA 3.2-11B-Vision, Molmo 7B-D  
**Task:** Rate how well abstract pseudo-words describe visual shapes (0-100 scale)

- **Stimuli:** Curved vs jagged visual shapes paired with phonetically distinct pseudo-words
- **Pseudo-words:** 
  - Sonorant+Rounded (S-R): "looloo", "moomoo", etc. (bouba-like)
  - Plosive+Non-Rounded (P-NR): "teetee", "kuhkuh", etc. (kiki-like)
- **Analysis:** Statistical comparison of congruent vs incongruent pairings

**Key Files:**
- `experiment-1/llama/e1-llama.ipynb`: Main analysis notebook
- `experiment-1/molmo/e1-molmo.ipynb`: Molmo implementation
- `experiment-1/*/graphs/`: Generated visualizations

### Experiment 2: Image-to-Text Matching  
**Models:** LLaMA 3.2-11B-Vision, Molmo 7B-D  
**Task:** Classify images using pseudo-word labels from different phonetic categories

- **Method:** Direct classification with confidence scoring
- **Labels:** Random sampling from S-R and P-NR pseudo-word sets
- **Metrics:** Classification accuracy and confidence scores by image type

**Key Files:**
- `experiment-2/llama/e2-llama.ipynb`: Classification experiments
- `experiment-2/molmo/e2-molmo.ipynb`: Molmo classification
- `experiment-2/*/graphs/`: Performance visualizations

### Experiment 3: Visual Grounding and Segmentation
**Models:** Molmo 7B + SAM2  
**Task:** Point-and-segment based on pseudo-word prompts

- **Pipeline:** Molmo generates coordinates → SAM2 segments regions
- **Objective:** Test whether models can ground abstract phonetic concepts in visual space
- **Innovation:** Novel integration of language-vision and segmentation models

**Key Files:**
- `experiment-3/text_point_SAM2.ipynb`: Complete pipeline implementation
- `experiment-3/sam2_hiera_l.yaml`: SAM2 configuration

## 📊 Key Findings

The research reveals that modern vision-language models exhibit measurable sound symbolic effects, though with important differences from human cognition:

1. **Cross-modal sensitivity**: Both models show systematic preferences in shape-sound associations
2. **Model-specific patterns**: LLaMA and Molmo demonstrate distinct response profiles
3. **Grounding capabilities**: Successful integration of phonetic prompts with visual segmentation

## 🛠 Technical Implementation

### Models Used
- **LLaMA 3.2-11B-Vision-Instruct**: Meta's multimodal language model
- **Molmo 7B-D-0924**: Allen AI's vision-language model  
- **SAM2**: Meta's Segment Anything Model 2

### Key Dependencies
```python
transformers
torch
PIL
pandas
numpy
opencv-python
matplotlib
segment-anything-2
```

### Hardware Requirements
- CUDA-compatible GPU recommended
- Sufficient VRAM for loading large multimodal models
- CPU fallback available but significantly slower

## 📈 Results and Visualizations

Each experiment generates comprehensive visualizations including:
- Probability distributions by stimulus type
- Effect strength comparisons
- Classification performance metrics
- Segmentation accuracy visualizations

Results are saved as CSV files and PNG graphs in respective `graphs/` subdirectories.

## 🎓 Academic Context

This research contributes to the intersection of:
- **Cognitive Science**: Sound symbolism and cross-modal perception
- **AI/ML**: Multimodal model capabilities and biases
- **Computational Linguistics**: Phonetic representation in neural models

### Thesis Information
- **Grade:** 8/10
- **Institution:** [University Name]
- **Year:** 2024-2025
- **Full Thesis:** `Rob Master thesis - final.pdf`

## 🚀 Usage

1. **Clone the repository**
```bash
git clone [repository-url]
cd msc_thesis
```

2. **Install dependencies**
```bash
pip install transformers torch pillow pandas numpy opencv-python matplotlib
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

3. **Run experiments**
- Navigate to desired experiment folder
- Open Jupyter notebook
- Configure model tokens/paths as needed
- Execute cells sequentially

## 📝 Citation

If you use this research or code, please cite:
```bibtex
@mastersthesis{author2024soundsymbolism,
  title={Cross-Modal Sound Symbolism in Vision-Language Models},
  author={[Author Name]},
  year={2024},
  school={[University Name]},
  type={Master's Thesis}
}
```

## 📄 License

This research is made available for academic and educational purposes. Please respect appropriate attribution when using or building upon this work.

---

**Contact:** [Your contact information]  
**Academic Supervisor:** [Supervisor name and institution]