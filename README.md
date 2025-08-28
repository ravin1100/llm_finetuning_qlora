# LLM Fine-Tuning with QLoRA on Banking77

This project demonstrates how to fine-tune a large language model (GPT2-Large) using QLoRA (Quantized Low-Rank Adaptation) on the Banking77 intent classification dataset. The workflow is implemented in the Jupyter notebook `fine_tuning_assignment.ipynb`.

## Project Structure

- `fine_tuning_assignment.ipynb`: Main notebook containing all code for data preparation, model setup, training, evaluation, and analysis.
- `result.png`: Visualization of results (see notebook for details).
- `experiment_results.json`: (Generated after running the notebook) Contains summary of experiment metrics and configuration.

## Steps Covered in the Notebook

1. **Installation and Imports**: Installs and imports all required libraries (transformers, datasets, peft, etc.).
2. **Dataset Preparation**: Loads and explores the Banking77 dataset for intent classification.
3. **Model and Tokenizer Setup**: Loads GPT2-Large and prepares it for sequence classification.
4. **Data Preprocessing**: Tokenizes the dataset and prepares it for training.
5. **Baseline Evaluation**: Evaluates the base model before fine-tuning.
6. **QLoRA Configuration**: Sets up QLoRA/LoRA parameters for efficient fine-tuning.
7. **Training Configuration**: Defines training arguments (batch size, learning rate, epochs, etc.).
8. **Training**: Fine-tunes the model using QLoRA.
9. **Evaluation**: Evaluates the fine-tuned model and compares with baseline.
10. **Results Comparison**: Compares metrics (accuracy, F1, precision, recall) before and after fine-tuning.
11. **Visualization**: Plots performance metrics and improvements.
12. **Detailed Analysis**: Prints classification report for detailed performance analysis.
13. **Model Information**: Summarizes model and training configuration.
14. **Save Results**: Saves the fine-tuned model and experiment results.
15. **Final Summary**: Prints a summary of the experiment and key results.

## How to Run

1. Open `fine_tuning_assignment.ipynb` in Jupyter or VS Code.
2. Run all cells sequentially. Adjust batch size, epochs, or dataset size as needed for your hardware.
3. After training, results and the fine-tuned model will be saved in the project directory.

## Requirements

- Python 3.8+
- torch, transformers, datasets, peft, bitsandbytes, scikit-learn, matplotlib, seaborn, evaluate, rouge-score

Install requirements (if not using Colab):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate bitsandbytes peft trl
pip install scikit-learn matplotlib seaborn evaluate rouge-score
```

## Results

- The notebook compares baseline and fine-tuned model performance on Banking77.
- Visualizations and detailed metrics are provided in the notebook and `result.png`.

## References

- [Banking77 Dataset](https://huggingface.co/datasets/banking77)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Library](https://github.com/huggingface/peft)

---

_For details, see the notebook and code comments._
