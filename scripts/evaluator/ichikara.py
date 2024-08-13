import json
from typing import Optional, Any

import weave
from weave import Evaluation
from pydantic import BaseModel
import asyncio


from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor

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

        return q["evaluations"]
    
    @weave.op()
    def _predict(self, input: Any, task: Any):
        q_ = tmp_q() # 既に取得済みの出力をトラッキングするための関数
        result = {}
        result["answer"] = q_["answer"]
        result["関連性"] = q_["evaluations"]["関連性"]
        result["正確性"] = q_["evaluations"]["正確性"]
        result["流暢性"] = q_["evaluations"]["流暢性"]
        result["情報量"] = q_["evaluations"]["情報量"]
        
        return result

@weave.op()
def aggregate(model_output, task):
    weights = {
        "関連性": 1.0,
        "正確性": 1.0,
        "流暢性": 1.0,
        "情報量": 1.0
    }
    
    result = {}
    
    # Calculate overall score
    overall = sum(weights[key] * model_output[key] for key in weights) / 4
    result["overall"] = overall
    
    # Calculate task-specific scores
    for t in task:
        task_result = {
            "関連性": model_output["関連性"],
            "正確性": model_output["正確性"],
            "流暢性": model_output["流暢性"],
            "情報量": model_output["情報量"]
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
        questions = questions[:50]

    # Create model answers
    generator_config = {"max_tokens": 1024}
    inputs = [
        ([{"role": "user", "content": q["text"]}], generator_config)
        for q in questions
    ]
    llm_ap = LLMAsyncProcessor(llm=llm, inputs=inputs)
    results = llm_ap.get_results()
    answers = [r.content for r in results]
    for q, a in zip(questions, answers):
        q.update({"answer": a})
    
    # Create model evaluations
    judge_dir = "/workspace"
    judge_prompts = load_questions(judge_dir + "/ichikara_judge_prompts.jsonl", None, None)[0]['system_prompt']
    inputs_eval = [
        ([{"role": "user", "content": judge_prompts + "\n### 質問\n" + str(q["text"]) + "\n### 回答\n" + str(q["answer"]) + "\n### 評価\n"}], generator_config)
        for q in questions
    ]
    llm_ap = LLMAsyncProcessor(llm=llm, inputs=inputs_eval)
    results = llm_ap.get_results()
    for q, r in zip(questions, results):
        try:
            e = json.loads(r.content.replace("'", '"'))
        except Exception as error:
            print(f"data: {r} Error: {error}")
            raise "出力がJSONのフォーマットになっていません"
        q.update({"evaluations": e})
    
    # datasetの整形
    dataset = []
    for q in questions:
        dataset.append({"input": q["text"], "output": q["answer"], "task": q["meta"]["task"]})

    # weaveの初期化(OpenAI, LangChainの実行以降にweave.initしないと勝手にTraceされる)
    weave.init(cfg.ichikara.get("weave_project_name", "ichikara"))
    model = InferenceModel(llm=llm, prompt=inputs, judge=inputs_eval)
    for q in questions:
        global q_
        q_ = q
        model._predict(input=q["text"], task=q["meta"]["task"]) # EvaluateだけではTraceに記録されないので明示的にpredictをする
    
    # 評価の実装
    q_ = questions
    evaluation = Evaluation(
        dataset=dataset, scorers=[aggregate]
    )
    asyncio.run(evaluation.evaluate(model))