from lm_eval import evaluator, tasks
results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=runs/best",
    tasks=["hellaswag", "arc_easy", "winogrande"],
    limit=100,
)
print(results["results"])