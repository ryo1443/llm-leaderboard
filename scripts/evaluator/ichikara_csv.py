import json
from typing import Optional, Any

import weave
from weave import Evaluation
from pydantic import BaseModel
import asyncio
import pandas as pd
import ast

from config_singleton import WandbConfigSingleton

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    with open(question_file, "r") as ques_file:
        try:
            data = json.load(ques_file)
            if isinstance(data, list):
                questions = data[begin:end] if begin is not None and end is not None else data
            else:
                questions = [data]
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            raise

    return questions

# 個々の入出力をTraceする場合
q_ = None
count = -1
def tmp_q():
    return q_

class InferenceModel(weave.Model, BaseModel):
    llm: Any
    prompt: Any
    judge: Any

    @weave.op()
    def predict(self, input: Any, task: Any):
        global count
        count += 1
        q = q_[count]

        return {
            "関連性_auto": q["関連性_auto"],
            "正確性_auto": q["正確性_auto"],
            "流暢性_auto": q["流暢性_auto"],
            "情報量_auto": q["情報量_auto"],
            "関連性_human": q["関連性_human"],
            "正確性_human": q["正確性_human"],
            "流暢性_human": q["流暢性_human"],
            "情報量_human": q["情報量_human"]
        }
    
    @weave.op()
    def _predict(self, input: Any, task: Any):
        q_ = tmp_q() # 既に取得済みの出力をトラッキングするための関数
        result = {}
        result["answer"] = q_["output"]
        result["関連性_auto"] = q_["関連性_auto"]
        result["正確性_auto"] = q_["正確性_auto"]
        result["流暢性_auto"] = q_["流暢性_auto"]
        result["情報量_auto"] = q_["情報量_auto"]
        result["関連性_human"] = q_["関連性_human"]
        result["正確性_human"] = q_["正確性_human"]
        result["流暢性_human"] = q_["流暢性_human"]
        result["情報量_human"] = q_["情報量_human"]
        
        return result

@weave.op()
def aggregate(model_output, task):

    weights_auto = {
        "関連性_auto": 1.0,
        "正確性_auto": 1.0,
        "流暢性_auto": 1.0,
        "情報量_auto": 1.0,
    }
    
    weights_human = {
        "関連性_human": 1.0,
        "正確性_human": 1.0,
        "流暢性_human": 1.0,
        "情報量_human": 1.0,
    }
    
    result = {}
    
    # Calculate overall score for auto
    overall_auto = sum(weights_auto[key] * model_output[key] for key in weights_auto) / len(weights_auto)
    result["overall_auto"] = overall_auto
    
    # Calculate overall score for human
    overall_human = sum(weights_human[key] * model_output[key] for key in weights_human) / len(weights_human)
    result["overall_human"] = overall_human
    
    # Calculate task-specific scores
    for t in task:
        task_result = {
            "関連性_auto": model_output["関連性_auto"],
            "正確性_auto": model_output["正確性_auto"],
            "流暢性_auto": model_output["流暢性_auto"],
            "情報量_auto": model_output["情報量_auto"],
            "関連性_human": model_output["関連性_human"],
            "正確性_human": model_output["正確性_human"],
            "流暢性_human": model_output["流暢性_human"],
            "情報量_human": model_output["情報量_human"],
        }
        result[f"{t}_evaluations"] = task_result
    
    return result

def evaluate():

    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # Load questions
    artifact_path = cfg.ichikara.get("artifact_path")
    artifact_dir = run.use_artifact(artifact_path, type='dataset').download()
    questions = load_questions(artifact_dir + "/ichikara-instruction-eval-001-001.json", None, None)
    if cfg.testmode==True:
        i = 5
        questions = questions[:i]
    
    # datasetの整形
    df = pd.read_csv("/workspace/ichikara_eval.csv")
    df['task'] = df['task'].apply(ast.literal_eval)
    dataset = df.to_dict('records')[:i]
    print(dataset)


    # Create model answers
    generator_config = {"max_tokens": 1024}
    inputs = [
        ([{"role": "user", "content": q["input"]}], generator_config)
        for q in dataset
    ]
    
    # Create model evaluations
    judge_dir = "/workspace"
    judge_prompts = load_questions(judge_dir + "/ichikara_judge_prompts.jsonl", None, None)[0]['system_prompt']
    inputs_eval = [
        ([{"role": "user", "content": judge_prompts + "\n### 質問\n" + str(q["input"]) + "\n### 回答\n" + str(q["output"]) + "\n### 評価\n"}], generator_config)
        for q in dataset
    ]

    # Initialize the Weave project
    weave.init(cfg.ichikara.get("weave_project_name", "ichikara"))
    model = InferenceModel(llm=llm, prompt=inputs, judge=inputs_eval)
    for row in dataset:
        global q_
        q_ = row
        model._predict(input=row["input"], task=row["task"])

    # 評価の実装
    q_ = dataset
    evaluation = Evaluation(
        dataset=dataset, scorers=[aggregate]
    )
    asyncio.run(evaluation.evaluate(model))