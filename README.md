# 🚀 Fine-Tuning DeepSeek-LLM 7B Chat on Cazton Q&A Dataset

This project showcases how I fine-tuned the [`deepseek-ai/deepseek-llm-7b-chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) model using the [Unsloth](https://github.com/unslothai/unsloth) library for a domain-specific Q&A chatbot focused on Cazton's technology services.

## Notebook link

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-MqBS9FAPraw2VT0xjfWvqYc13y-uR_d?usp=sharing)

---

## 🧠 Objective

Fine-tune a large language model (LLM) to answer questions related to Cazton’s consulting services using a custom dataset, enabling accurate, domain-specific chatbot interactions.

---

## 🔧 Tools & Technologies

- 🧠 Model: [`deepseek-ai/deepseek-llm-7b-chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
- 🚀 Fine-tuning: [Unsloth](https://github.com/unslothai/unsloth) (LoRA + 4-bit)
- 🧰 Frameworks: HuggingFace Transformers, PEFT, bitsandbytes
- 💻 Hardware: 2× NVIDIA T4 GPUs (Colab Pro)
- 🐍 Language: Python

---

## 📚 Dataset

- **Name**: Cazton Dataset (`complete_dataset.csv`)
- **Size**: 1085 rows
- **Columns**: 
  - `questions`: Realistic user queries about Cazton’s offerings
  - `answers`: Expert-level responses based on Cazton's service scope

---

## 🧪 Training Process

1. **Preprocessing**:
   - Cleaned dataset and formatted in instruction style (User/Assistant).
   - Ensured token count per sample < 4096.

2. **Model Loading**:
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name = "deepseek-ai/deepseek-llm-7b-chat",
       max_seq_length = 4096,
       dtype = torch.float16,
       load_in_4bit = True,
   )
   ```

3. **Fine-tuning**:
   - Used QLoRA with `unsloth` to reduce memory footprint.
   - Trained for a few epochs on 2× T4 GPUs.

4. **Saving the Model**:
   ```python
   model.save_pretrained("/content/deepseek-cazton-finetuned")
   tokenizer.save_pretrained("/content/deepseek-cazton-finetuned")
   ```

---

## 💬 Inference Example

**Prompt**:
```
User: What big data services does Cazton offer?

Assistant:
```

**Output**:
```
Cazton offers big data services like Hadoop, Spark, NoSQL, and Cassandra consulting. 
They provide training and consulting services and have worked on large-scale projects across industries.
```

---

## 📈 Results & Takeaways

- 🎯 The model now accurately responds to company-specific queries.
- 🧩 Repetition in output was fixed by proper prompt formatting and `max_new_tokens` tuning.
- 💡 Learned how efficient Unsloth is for large-scale LLM fine-tuning on consumer-grade GPUs.

---

## 📎 Files

- `Complete_dataset.csv` – Custom Q&A dataset
- `finetune_unsloth.ipynb` – Fine-tuning notebook
- `deepseek-cazton-finetuned/` – Final LoRA adapter + tokenizer files

---

## 🔗 Connect with Me

Feel free to connect on [LinkedIn](https://www.linkedin.com/in/aniq-ramzan-ai-learner/) or explore more of my work!
