from evaluate import load
import sys #
comet_metric = load('comet')
# from fairseq.inference.comet_compute import download_model, load_from_checkpoint

# model_path = download_model("wmt22-comet-da")
# model = load_from_checkpoint(model_path)
source_fname = sys.argv[1]
ref_fname = sys.argv[2]
pred_fname = sys.argv[3]

src, ref, pred = [], [], []


with open(source_fname, 'r') as srcf:
    for s in srcf.readlines():
        src.append(s)
with open(ref_fname, 'r') as reff:
    for s in reff.readlines():
        ref.append(s)
with open(pred_fname, 'r') as predf:
    for s in predf.readlines():
        pred.append(s)

score = []
# for s,r,p in zip(src, ref, pred):
score = (comet_metric.compute(predictions=pred, references = ref, sources =src ))

# print('score is ', score)
print(score['mean_score'])


