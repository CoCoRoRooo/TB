---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:43938
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: "symptoms: Fridge not starting, no power to the start relay assembly,\
    \ audible clicking sounds from the control board.\n  3. General"
  sentences:
  - Samsung Gear VR With Controller USB Connector Replacement
  - Whirlpool Refrigerator Start Relay Assembly Replacement
  - P775DM3 Hard Drive replacement
- source_sentence: "Symptoms:\n       - No sound output through speakers or headphones\n\
    \       - Inability to connect external devices to audio jacks\n       - Distorted,\
    \ static, or no sound when attempting to play media\n\n   3."
  sentences:
  - Huawei SnapTo G-620 Motherboard Replacement
  - Toshiba SD-K740 Motherboard Replacement
  - iMac Intel 21.5" EMC 2428 Audio Ports Replacement
- source_sentence: "Symptoms (e.g., \"No power indicator light, tried different charger\
    \ and cable\")\n\n  3. General"
  sentences:
  - PowerBook G4 Aluminum 12" 1-1.5 GHz Hard Drive Replacement
  - MacBook Pro 13" Touch Bar 2016 & 2017 Display Replacement
  - Lenovo Legion Y530-15ICH Battery Replacement
- source_sentence: "No rear camera images are visible, camera app does not open or\
    \ function properly\n   3."
  sentences:
  - LG Optimus L9 P769 Rear Facing Camera Replacement
  - Dell Inspiron 14-5415 Battery Replacement
  - PowerBook G4 Aluminum 12" 867 MHz Antenna Board and Cables Replacement
- source_sentence: "Laptop screen is black, laptop not displaying any image, no signs\
    \ of life from the screen.\n  3. General"
  sentences:
  - Asus ROG STRIX GL502V Motherboard Replacement
  - Trackpad Prereq
  - Acer Nitro 5 AN515-53-55G9 Display Replacement
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Laptop screen is black, laptop not displaying any image, no signs of life from the screen.\n  3. General',
    'Acer Nitro 5 AN515-53-55G9 Display Replacement',
    'Asus ROG STRIX GL502V Motherboard Replacement',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 43,938 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 9 tokens</li><li>mean: 24.25 tokens</li><li>max: 74 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 12.03 tokens</li><li>max: 25 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                         | sentence_1                                                        |
  |:-----------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|
  | <code>Symptoms:<br>     - No power indicator light<br>     - Laptop not turning on despite being plugged in<br>  3. General</code> | <code>Toshiba CB35-C3300 Chromebook 2 Battery  Replacement</code> |
  | <code>Device not playing audio, Motor does not rotate<br>   3. General</code>                                                      | <code>Sony WM-FX123 Belt Replacement</code>                       |
  | <code>symptoms: `No charging light, tested cable`<br>  3. General</code>                                                           | <code>Battery Disconnect</code>                                   |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.1820 | 500   | 0.6092        |
| 0.3640 | 1000  | 0.4708        |
| 0.5461 | 1500  | 0.4303        |
| 0.7281 | 2000  | 0.3856        |
| 0.9101 | 2500  | 0.3973        |
| 1.0    | 2747  | -             |
| 1.0921 | 3000  | 0.3697        |
| 1.2741 | 3500  | 0.3152        |
| 1.4561 | 4000  | 0.3328        |
| 1.6382 | 4500  | 0.3125        |
| 1.8202 | 5000  | 0.3217        |
| 2.0    | 5494  | -             |
| 2.0022 | 5500  | 0.3126        |
| 2.1842 | 6000  | 0.2786        |
| 2.3662 | 6500  | 0.2686        |
| 2.5482 | 7000  | 0.2689        |
| 2.7303 | 7500  | 0.2662        |
| 2.9123 | 8000  | 0.2708        |
| 3.0    | 8241  | -             |
| 3.0943 | 8500  | 0.2494        |
| 3.2763 | 9000  | 0.2487        |
| 3.4583 | 9500  | 0.2331        |
| 3.6403 | 10000 | 0.2369        |
| 3.8224 | 10500 | 0.2451        |
| 4.0    | 10988 | -             |
| 4.0044 | 11000 | 0.2397        |
| 4.1864 | 11500 | 0.2257        |
| 4.3684 | 12000 | 0.2325        |
| 4.5504 | 12500 | 0.2288        |
| 4.7324 | 13000 | 0.2199        |
| 4.9145 | 13500 | 0.2225        |
| 5.0    | 13735 | -             |


### Framework Versions
- Python: 3.12.6
- Sentence Transformers: 3.3.1
- Transformers: 4.48.0
- PyTorch: 2.5.1+cu121
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->