# AI Engineering: Hands-on
![Stars](https://img.shields.io/github/stars/Ramakm/ai-hands-on?style=flat-square)
![Forks](https://img.shields.io/github/forks/Ramakm/ai-hands-on?style=flat-square)
![PRs](https://img.shields.io/github/issues-pr/Ramakm/ai-hands-on?style=flat-square)
![Issues](https://img.shields.io/github/issues/Ramakm/ai-hands-on?style=flat-square)
![Contributors](https://img.shields.io/github/contributors/Ramakm/ai-hands-on?style=flat-square)
![License](https://img.shields.io/github/license/Ramakm/ai-hands-on?style=flat-square)
<img width="2000" height="600" alt="image" src="https://github.com/user-attachments/assets/0c161d9a-3227-4dd7-a416-ad97bc9742a9" />

A complete, hands-on guide to becoming an AI Engineer.

This repository is designed to help you learn AI from first principles, build real neural networks, and understand modern LLM systems end-to-end.
You'll progress through math, PyTorch, deep learning, transformers, RAG, and OCR — with clean, intuitive Jupyter notebooks guiding you at every step.

Whether you're a beginner or an engineer levelling up, this repo gives you the clarity, structure, and intuition needed to build real AI systems.

#### ⭐ Star This Repo

If you learn something useful, a star is appreciated.

## Repository Structure

### 1. Math Fundamentals
- Math functions, derivatives, vectors, and gradients
- Matrix operations and linear algebra
- Probability and statistics

### 2. PyTorch Basics
- Creating and manipulating tensors
- Matrix multiplication, transposing, and reshaping
- Indexing, slicing, and concatenating tensors
- Special tensor creation functions

### 3. Neural-Network(NN)
- Building neurons, layers, and networks from scratch
- Normalization techniques (RMSNorm)
- Activation functions
- Optimizers (Adam, Muon) and learning rate decay

### 4. Transformers
- Attention and self-attention mechanisms
- Multi-head attention
- Decoder-only transformer architecture

### 5. Retrieval-Augmented Generation (RAG)
- Building RAG pipelines end to end
- Indexing, retrieval, chunking strategies
- Integrations with embedding models and vector stores

### 6. Optical Character Recognition (OCR)
- OCR pipeline and utilities
- Preprocessing images and extracting text

## Books

Recommended reading to deepen your understanding (not included):

- `AI Engineering` by Chip Huyen
- `Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow` by Aurélien Géron
- `Deep Learning` by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- `The Elements of Statistical Learning` by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
- `Neural Networks and Deep Learning` by Michael Nielsen
- `SQL Cookbook` by Anthony Molinaro

For more books in AI/ML, I have created another repo for this [Check Here](https://github.com/Ramakm/AI-ML-Book-References.git). I will be adding lot more in coming days/months. If you are interested to read book, go check this repo out.

## Learning Path

For a recommended step-by-step progression through the materials, see the Learning Path:

- `Start_here/learning_path.md`

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

Some subfolders (for example `5.RAG/` and `6.OCR/`) include their own `requirements.txt` with additional dependencies.

## Usage

Recommended workflow:

1. Open Jupyter in the project root:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Work through notebooks in order:
   - `1.Math/`
   - `2.PyTorch/`
   - `3.Neural-Network(NN)/`
   - `4.Transformer/`

3. Folder to run separately:
   - `5.RAG/`
   - `6.OCR/`
  
4. Resources
5. Basic ML Model Implementation (Supervised + Un-supervised + RL)
   - `1.Linear Regression`
   - `2.Logistic Regression`
   - `3.Decision Tree Model`
   - `4.Naive Bayes Classification`
  
## Machine Learning Frameworks

| Tool         | Category          | Link                                                                                     |
| ------------ | ----------------- | ---------------------------------------------------------------------------------------- |
| Scikit-learn | Traditional ML    | [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)                     |
| XGBoost      | Gradient Boosting | [https://xgboost.ai/](https://xgboost.ai/)                                               |
| LightGBM     | Gradient Boosting | [https://lightgbm.readthedocs.io/en/stable/](https://lightgbm.readthedocs.io/en/stable/) |
| CatBoost     | Gradient Boosting | [https://catboost.ai/](https://catboost.ai/)                                             |


## GEN AI References

| Resource                              | Focus Area             | Link                                                                                                                                                                   |
| ------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Microsoft Generative AI for Beginners | Intro to GenAI         | [https://github.com/microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners)                                                   |
| Generative AI for Everyone            | Non-technical overview | [https://www.coursera.org/learn/generative-ai-for-everyone](https://www.coursera.org/learn/generative-ai-for-everyone)                                                 |
| Building Blocks of Generative AI      | Conceptual foundations | [https://shriftman.substack.com/p/the-building-blocks-of-generative](https://shriftman.substack.com/p/the-building-blocks-of-generative)                               |
| The Illustrated Transformer           | Transformers           | [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)                                                             |
| LLMs Explained Briefly                | LLM basics video       | [https://www.youtube.com/watch?v=LPZh9BOjkQs](https://www.youtube.com/watch?v=LPZh9BOjkQs)                                                                             |
| Intro to LLMs                         | LLM overview video     | [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g)                                                                             |
| Understanding LLMs                    | Deep dive              | [https://magazine.sebastianraschka.com/p/understanding-large-language-models](https://magazine.sebastianraschka.com/p/understanding-large-language-models)             |
| Visual Guide to Reasoning LLMs        | Reasoning models       | [https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms)         |
| Understanding Reasoning LLMs          | Reasoning theory       | [https://magazine.sebastianraschka.com/p/understanding-reasoning-llms](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)                           |
| Understanding Multimodal LLMs         | Vision + text models   | [https://magazine.sebastianraschka.com/p/understanding-multimodal-llms](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)                         |
| Visual Guide to MoE                   | Mixture of Experts     | [https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) |
| Finetuning LLMs                       | Model training         | [https://magazine.sebastianraschka.com/p/finetuning-large-language-models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)                   |
| How Transformer LLMs Work             | Architecture           | [https://www.deeplearning.ai/short-courses/how-transformer-llms-work/](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/)                           |
| Build GPT from Scratch                | Hands-on               | [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)                                                                             |
| LLM Course (GitHub)                   | Structured learning    | [https://github.com/mlabonne/llm-course](https://github.com/mlabonne/llm-course)                                                                                       |
| LLM Course (Hugging Face)             | Practical LLMs         | [https://huggingface.co/learn/llm-course/chapter1/1](https://huggingface.co/learn/llm-course/chapter1/1)                                                               |
| Awesome LLM Apps                      | Project ideas          | [https://github.com/Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)                                                                   |

## Contributing

Contributions are welcome!

Please ensure:

- Notebooks are clean (Restart & Run All before committing)
- Existing structure & naming conventions are followed
- PRs are focused, readable, and documented
- In folders like RAG and OCR, please maintain the cleaned structure part
- If you want to add something new folders, make it proper structure way.

## License

- This project is licensed under the MIT License. See `LICENSE` for details.

## Connect with me

[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/techwith_ram)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwith.ram)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ramakm)

