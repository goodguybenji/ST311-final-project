# Detecting AI-Generated Music Using LambdaResNet and Swin Transformers on Mel-Spectrograms  

This repository contains the code and experiments for our **ST311 final project**: detecting whether short music clips are AI-generated or human-composed using deep learning models trained on mel-spectrograms.  

## 📌 Project Overview  

The rise of generative music models raises concerns about authenticity, copyright, and fair compensation for artists. To address this, we explore whether **compact pretrained vision models** can detect AI-generated music from short 5-second audio clips.  

We fine-tune two architectures on the [SONICS dataset](https://arxiv.org/abs/2408.14080):  
- [**LambdaResNet26rpt_256**](https://huggingface.co/timm/lambda_resnet26rpt_256.c1_in1k) (11M parameters, CNN with lambda layers)  
- [**Swin Transformer V2 Small**](https://huggingface.co/timm/swinv2_small_window16_256.ms_in1k) (51M parameters, vision transformer with shifted window attention)  

Both models are trained on **mel-spectrograms** derived from 5-second audio segments.  

## ⚙️ Methodology  

- **Dataset**: 20,000 clips (10k real, 10k AI-generated) sampled from SONICS.  
- **Preprocessing**: Audio resampled to 22.05kHz → converted to 256×256 mel-spectrograms → normalized and reshaped to (3, 256, 256) for ImageNet backbones.  
- **Splitting**: 60/20/20 stratified train/val/test split.  
- **Models**: Fine-tuned using PyTorch
- **Training**:  
  - LambdaResNet: batch size 32, LR = 1e-5  
  - Swin Transformer: batch size 20, LR = 1e-4  
- **Evaluation Metrics**: Precision, Recall, F1, Specificity, AUC-ROC.  

## 📊 Results  

Both models achieved **extremely high accuracy**:  

| Model                     | Precision | Recall | F1 Score | Specificity | AUC-ROC |
|----------------------------|-----------|--------|----------|-------------|---------|
| LambdaResNet26rpt_256      | 0.9910    | 0.9925 | 0.9918   | 0.9910      | 0.9996  |
| Swin Transformer V2 Small  | 0.9995    | 0.9990 | 0.9992   | 0.9995      | 1.0000  |  

- **Swin Transformer** achieved perfect AUC-ROC (1.000) and nearly perfect F1 (0.9992).  
- **LambdaResNet** performed slightly worse but is far more lightweight and efficient (11M vs 51M params).  

✅ **Key Takeaway**: Even **compact vision models** can reliably detect AI-generated music from short mel-spectrograms without complex temporal modeling.  

## 🔮 Limitations & Future Work  

- Only trained on **SONICS** dataset → may not generalize across genres/models.  
- Binary classification only (real vs fake). Multi-class (e.g., real, Suno, MusicGen, Riffusion) is a natural next step.  
- Short clip restriction (5s, 256×256 spectrograms). Longer sequences could capture higher-level musical structures.  
- No interpretability/explainability tools (e.g., Grad-CAM).
