import json
import torch
import torch.nn.functional as F

with open("Model-Training/wrong_predictions.json", "r") as f:
    data = json.load(f)

for item in data:
    logits = torch.tensor(item["logits"])
    probs = F.softmax(logits, dim=0)

    item["probs"] = probs.tolist()
    item["confidence"] = float(probs.max())
    item["pred_label"] = int(torch.argmax(probs))

with open("Model-Training/wrong_predictions_with_probs.json", "w") as f:
    json.dump(data, f, indent=2)