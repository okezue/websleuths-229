from __future__ import annotations
import torch

_DEFAULT_ANCHORS=[
    "The Earth revolves around the Sun.",
    "Water freezes at zero degrees Celsius.",
    "The speed of light is approximately 300000 kilometers per second.",
    "DNA carries genetic information in living organisms.",
    "Gravity pulls objects toward the center of the Earth.",
    "The chemical formula for water is H2O.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Moon orbits the Earth approximately every 27 days.",
    "Oxygen is essential for human respiration.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Carbon dioxide is a greenhouse gas.",
    "Electrons orbit the nucleus of an atom.",
    "The human heart pumps blood throughout the body.",
    "Sound travels faster in water than in air.",
    "Iron is a magnetic metal.",
    "Mitosis is the process of cell division.",
    "Nitrogen makes up about 78 percent of the atmosphere.",
    "The boiling point of water is 100 degrees Celsius at sea level.",
    "Antibiotics are used to treat bacterial infections.",
    "The Milky Way is a spiral galaxy.",
    "Proteins are made of amino acids.",
    "Voltage equals current times resistance.",
    "Tectonic plates move slowly over time.",
    "Chlorophyll gives plants their green color.",
    "Light travels in straight lines.",
]

class AnchorEval:
    def __init__(self,model,tok,anchors:list[str]|None=None):
        self._m=model
        self._t=tok
        self._a=anchors or _DEFAULT_ANCHORS
        self._dev=next(model.parameters()).device
    def per_anchor_nll(self)->list[float]:
        self._m.eval()
        nlls=[]
        with torch.no_grad():
            for a in self._a:
                enc={k:v.to(self._dev) for k,v in self._t(a,return_tensors="pt",truncation=True,max_length=512).items()}
                out=self._m(**enc,labels=enc["input_ids"])
                nlls.append(out.loss.item())
        return nlls
    def nll(self)->float:
        vals=self.per_anchor_nll()
        return sum(vals)/max(len(vals),1)
    def delta_nll(self,pre_nll:float)->float:
        return self.nll()-pre_nll
