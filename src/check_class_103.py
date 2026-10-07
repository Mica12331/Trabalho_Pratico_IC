import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Carregar id2label
filepath = hf_hub_download(repo_id='json9473/FoodSeg103', filename='id2label.json', repo_type='dataset')
with open(filepath, 'r', encoding='utf-8') as f:
    id2label = {int(k): v.strip() for k, v in json.load(f).items()}

ds = load_dataset('json9473/FoodSeg103')

indices_103 = [19, 73, 137, 138, 154, 156]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, idx in enumerate(indices_103):
    sample = ds['train'][idx]
    img = sample['image']
    classes = sample['classes_on_image']
    names = [id2label[c] for c in classes if c != 0]
    
    ax = axes[i]
    ax.imshow(img)
    title_str = f"Img idx {idx} (ID {sample['id']}):\n" + ", ".join(names)
    ax.set_title(title_str, fontsize=8)
    ax.axis('off')

plt.suptitle('Exemplos de Imagens com a Classe 103 (other ingredients)', fontsize=13)
plt.tight_layout()
plt.savefig('results/figures/samples_class_103.png', dpi=150)
print('Figura guardada em results/figures/samples_class_103.png')
