import json
from pathlib import Path
from typing import Optional
from collections import defaultdict

from langchain.prompts import BasePromptTemplate, PromptTemplate
from dataclasses import dataclass
from jinja2 import Template

from config_singleton import WandbConfigSingleton

@dataclass(frozen=True)
class Sample:
    input: str
    output: str

def get_system_message_intro(language: str) -> str:
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    
    custom_system_message = cfg.get(f"custom_system_message_{language}", None)
    if custom_system_message is not None:
        return custom_system_message
    elif language == "ja":
        return "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    elif language == "en":
        return "Here is a combination of instructions explaining the task and inputs with context. Please write a response that adequately meets the request."
    else:
        raise ValueError(f"Invalid language: {language}")

def get_few_shot_messages(target_dataset_path: str, num_few_shots: int):
    dataset_json_name = target_dataset_path.name
    target_few_shot_path = (
        target_dataset_path.resolve().parent.parent / "train" / dataset_json_name
    )
    assert (
        target_few_shot_path.exists()
    ), f"Wrong path {target_few_shot_path.resolve()}, can not extract few-shot samples"
    samples = json.loads(target_few_shot_path.read_text(encoding="utf-8"))["samples"]

    few_shot_messages = []
    for i in range(num_few_shots):
        few_shot_messages.append({"role": "user", "content": samples[i]["input"]})
        few_shot_messages.append({"role": "assistant", "content": samples[i]["output"]})
    return few_shot_messages

def get_few_shot_samples(target_dataset_path: Path, num_few_shots: int) -> list[Sample]:
    dataset_json_name = target_dataset_path.name
    target_few_shot_path = (
        target_dataset_path.resolve().parent.parent / "train" / dataset_json_name
    )
    assert (
        target_few_shot_path.exists()
    ), f"Wrong path {target_few_shot_path.resolve()}, can not extract few-shot samples"
    samples = [
        Sample(**data)
        for data in json.loads(target_few_shot_path.read_text(encoding="utf-8"))[
            "samples"
        ]
    ]
    assert (
        len(samples) >= num_few_shots
    ), f"Wrong number of few shots {num_few_shots}, we only have {len(samples)} few shots"
    return samples[:num_few_shots]

# bbqデータセットはカテゴリごとにfew-shotを構築するため、専用のカテゴリごとのfew-shotを取得する関数を追加
def get_few_shot_samples_by_bbq_category(target_dataset_path: Path, num_few_shots: int, sample_class=None) -> dict[str, list]:
    if num_few_shots == 0:
        return defaultdict(list)
    
    if sample_class is None:
        from evaluator.bbq import BBQSample
        sample_class = BBQSample
    
    dataset_json_name = target_dataset_path.name
    target_few_shot_path = target_dataset_path.resolve().parent.parent / "train" / dataset_json_name

    assert target_few_shot_path.exists(), f"Wrong path {target_few_shot_path.resolve()}, can not extract few-shot samples"

    samples_data = json.loads(target_few_shot_path.read_text(encoding="utf-8"))["samples"]

    category_samples = defaultdict(list)
    for data in samples_data:
        sample = sample_class(**data)
        category_samples[sample.category].append(sample)
    
    few_shot_samples = defaultdict(list)
    for category, samples in category_samples.items():
        selected_samples = samples[:num_few_shots]
        few_shot_samples[category].extend(selected_samples)
    
    return few_shot_samples



def replace_braces(text):
    return text.replace("{", "{{").replace("}", "}}")


def apply_chat_template(messages: list[dict[str, str]]) -> str:
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    if cfg.api == "vllm":
        chat_template_path = Path(f"chat_templates/{cfg.model.chat_template}.jinja")
        if not chat_template_path.exists():
            raise ValueError(f"Chat template {chat_template_path} not found")
        else:
            with chat_template_path.open(encoding="utf-8") as f:
                chat_template = Template(f.read())
        special_tokens_map = cfg.special_tokens_map
        special_tokens_map.update({"add_generation_prompt": True})
        conversation_prompt = chat_template.render(
            messages=messages, **special_tokens_map
        )
    else:
        chat_template_path = Path(f"chat_templates/defaults/template_alpaca_en.jinja")
        with chat_template_path.open(encoding="utf-8") as f:
            chat_template = Template(f.read())
        conversation_prompt = chat_template.render(
            messages=messages, add_generation_prompt=True
        )
    return conversation_prompt


def get_evaluation_prompt(
    instruction: str,
    few_shots: list[Sample],
    language: str,
    custom_prompt_template: Optional[str] = None,
    custom_fewshots_template: Optional[str] = None,
) -> BasePromptTemplate:
    if language == "ja":
        input_heading = "入力"
        response_heading = "応答"
        system_message_intro = f"{instruction}" # alpaca formatだとprefixに"### Instruction"が入るのでそれに対応した形に変更
        few_shots_template = "\n\n### 入力:\n{input}\n" # alpaca formatだとsuffixに"### Response"が入るのでそれに対応した形に変更
    elif language == "en":
        input_heading = "Input"
        response_heading = "Response"
        system_message_intro = f"{instruction}"
        few_shots_template = "\n\n### Input:\n{input}\n"
    else:
        raise ValueError(f"Invalid language: {language}")
    
    if custom_prompt_template is not None:
        if len(few_shots) == 0:
            system_message: str = custom_prompt_template.format(
                instruction=instruction, input="{input}"
            )
        else:
            few_shots_text: str = ""
            for few_shot in few_shots:
                if custom_fewshots_template is not None:
                    few_shots_text += custom_fewshots_template.format(
                        input=replace_braces(few_shot.input), output=few_shot.output
                    )
                else:
                    few_shots_text += f"\n\n### {input_heading}:\n{replace_braces(few_shot.input)}\n\n### {response_heading}:\n{few_shot.output}"
            system_message = custom_prompt_template.format(
                instruction=instruction + few_shots_text, input="{input}"
            )
    else:
        system_message = system_message_intro
        for few_shot in few_shots:
            system_message += f"\n\n### {input_heading}:\n{replace_braces(few_shot.input)}\n\n### {response_heading}:\n{few_shot.output}"
        system_message += few_shots_template

    messages = [{"role": "user", "content": system_message}]
    conversation_prompt = apply_chat_template(messages)

    return PromptTemplate(input_variables=["input"], template=conversation_prompt)