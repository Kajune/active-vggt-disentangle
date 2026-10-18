from vggt.models.vggt import VGGT
import torch
import os

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

checkpoint = {}
checkpoint["model"] = model.state_dict()

os.makedirs("ckpt", exist_ok=True)
with open("ckpt/VGGT-1B.pth", "wb") as f:
	torch.save(checkpoint, f)
