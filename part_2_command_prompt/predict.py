import argparse
import torch
import json
from build_model import build_predict_model
from preprocess import prepare_image

#setup argparse
parser = argparse.ArgumentParser(description='Use a pre-trained network to make an inference on an image.')
parser.add_argument('image_path', help='The path to the image.')
parser.add_argument('checkpoint', help='The path to the checkpoint of the pre-trained model. Note that any model not trained using train.py \
is not guaranteed to work with this prediction script.')
parser.add_argument('--top_k', default=5, type=int, help='A number indicating how many categories of the dataset should be inferred.')
parser.add_argument('--category_names', default='./cat_to_name.json', help='Location of a file detailing the mappings of category IDs to names.')
parser.add_argument('--gpu', action='store_true', help='Whether or not to use a GPU, if present in system.')

args = parser.parse_args()

model, class_to_idx = build_predict_model(args)
device_conditional = torch.cuda.FloatTensor if torch.cuda.is_available() and args.gpu else torch.FloatTensor
device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"

image_tensor = prepare_image(args.image_path).type(device_conditional)
image_tensor.unsqueeze_(0)

idx_to_class = {v:k for k, v in class_to_idx.items()}
cat_to_name = ""
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#predict, return classes
model.eval()
with torch.no_grad():
    image_tensor = image_tensor.to(device)
    output = model.forward(image_tensor)
    prob = torch.exp(output)

    top_probs, top_classes = prob.topk(args.top_k, dim=1)
    top_probs = torch.Tensor.tolist(top_probs)
    print(top_probs)
    top_classes = torch.Tensor.tolist(top_classes)

    class_to_str = lambda x : list(map(str, x))
    class_labels = [idx_to_class[i] for i in top_classes[0]]
    class_names = [cat_to_name[i] for i in class_labels]

    print("\nMost likely candidate for flower at path {}: ".format(args.image_path))
    for i in range(len(class_names)):
        print("{}: {:.2f}% chance.".format(class_names[i], top_probs[0][i]*100))
