# Nejumi-leaderboard Neo

## Set up
1. Set up environment variables
```
export WANDB_API_KEY=<your WANDB_API_KEY>
export OPENAI_API_KEY=<your OPENAI_API_KEY>
export LANG=ja_JP.UTF-8
# if needed, set the following API KEY too
export ANTHROPIC_API_KEY=<your ANTHROPIC_API_KEY>
export GOOGLE_API_KEY=<your GOOGLE_API_KEY>
export COHERE_API_KEY=<your COHERE_API_KEY>
export MISTRAL_API_KEY=<your MISTRAL_API_KEY>
export AWS_ACCESS_KEY_ID=<your AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<your AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<your AWS_DEFAULT_REGION>
# if needed, please login in huggingface
huggingface-cli login
```



## Preparing Data

If you use the public W&B server, you can use the same artifacts used for the public leaderboard to do your evaluation. However, if you use a private W&B server you'll need to mirror the artifacts first. To mirror the default artifacts, you can do the following:

```
host=https://my-server.wandb.io # the host url for your wandb instance
dest=my-user/my-project # the user/project to hold the resulting artifacts

./scripts/mirror-data.sh "$host" "$dest"
# There will be interactive instructions to follow
```

## Create config.yaml file

1. create configs/config.yaml
```bash
cp configs/config_template.yaml configs/config.yaml
```
2. set each variable properly by following the below instruction

general
- `wandb`: Information used for W&B support.
  - `entity`: Name of the W&B Entity.
  - `project`: Name of the W&B Project.
  - `run_name`: Name of the W&B run. If you set "model name" as run name, you can see easily find run on Wandb dashboard.
- `github_version`: For recording. Not need to be changed
- `testmode`: The default is false. If set to true, it allows for a lightweight implementation where only 1 or 2 questions are extracted from each category. Please set it to true when you want to perform a functionality check
- `api`:  If you don't use api, please set "api" as "false". If you use api, please select from "openai", "anthoropic", "google", "cohere"
- model: Information of model
  - `_target_`: transformers.AutoModelForCausalLM.from_pretrained
  -`pretrained_model_name_or_path`: Name of your model. if you use openai api, put the name of model
  - `trust_remote_code`: true
  - `device_map`: device map. The default is "auto"
  - `load_in_8bit`: 8 bit quantization. The default is false
  - `load_in_4bit`: 4 bit quantization.The default is false
- tokenizer: Information of tokenizer
  - `pretrained_model_name_or_path`: Name of tokenizer
  - `use_fast`: If set to true, it uses the fast tokenizer. The default is true
- generator: Settings for generation. For more details, refer to the [generation_utils](https://huggingface.co/docs/transformers/internal/generation_utils)  in huggingface transformers.
  - `top_p`: top-p sampling. The default is 1.0.
  - `top_k`: top-k sampling. Default is commented out.
  - `temperature`: The temperature for sampling. Default is commented out.
  - `repetition_penalty`: Repetition penalty. The default is 1.0.

variables for llm-jp-eval
- `max_seq_length`: The maximum length of the input. The default is 2048.
- `dataset_artifact`: URL of wandb Artifacts of evaluation dataset. Choose the version from the Data Preparation section
- `dataset_dir`: location of the evaluation data after downloading wandb Artifacts
- `target_dataset`: The dataset to evaluate. The default is all, which evaluates all datasets. Specify the dataset name (like jnli) to evaluate a specific dataset.
- `log_dir`: The directory to save logs. The default is ./logs.
- `torch_dtype`: Settings for fp16, bf16, fp32. The default is bf16.
- `custom_prompt_template`: Specification of custom prompts. The default is null. (The default prompt is using the alpaca format.)
- `custom_fewshots_template`:  Specification of custom prompts for few-shot settings. The default is null. (The default prompt is using the alpaca format.)

- `metainfo`:
  - `model_name`: Model name. This is for record, so doesn't affect evaluation performance.  
  - `model_type`: Category information of the language model used for the evaluation experiment. This is for record, so doesn't affect evaluation performance. 
  - `instruction_tuning_method`: Tuning method of model. This is for record, so doesn't affect evaluation performance. 
  - `instruction_tuning_data`: Tuning data of model. This is for record, so doesn't affect evaluation performance. 
  - `num_few_shots`: The number of questions to be presented as Few-shot. The default is 0.
  - `llm-jp-eval-version`: Version information of llm-jp-eval.

for mtbench
- `mtbench`:
  - `question_artifacts_path`: URL of wandb Artifacts of evaluation dataset. Choose the version from the Data Preparation section
  - `referenceanswer_artifacts_path`: URL of wandb Artifacts of reference answer. Choose the version from the Data Preparation section
  - `judge_prompt_artifacts_path`: URL of wandb Artifacts of judge prompt. Choose the version from the Data Preparation section
  - `bench_name`: If you evaluate Japanese dataset, set 'japanese_mt_bench'. If you evaluate English dataset, set 'mt_bench'.
  - `model_id`: Name of model
  - `max_new_token`: The maximum length of the input. The default is 1024.
  - `num_gpus_per_model`: Number of GPUs per model. If you use multiple gpu, change here. The default is 1.
  - `num_gpus_total`:  Number of Total GPUs. If you use multiple gpu, change here. The default is 1.
  - `max_gpu_memory`: If you specifiy the max of GPU memory, change here. The default is null.
  - `dtype`: Data type. Choose from None or float32 or float16 or bfloat16
  - `judge_model`: Model used for evaluation. The default is 'gpt-4'
  - `question_begin`,`question_end`,`mode`,`baseline_model`,`parallel`,`first_n`: Parameters for original FastChat. In this leaderboard, use the default values.
  - `custom_conv_template`: If the model is not compatible FastChat, you need to use custom conv template, and set this variable true. Then, the custom conv template you set with the following variables will be used. The defalt is false.


   
## Evaluation execution
1. run scripts/run_eval.py
```bash
python3 scripts/run_eval.py
```
2. check the wandb dashboard
