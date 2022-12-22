from evaluate import load
import sys #
chrf_metric = load('chrf')

ref_fname = sys.argv[1]
pred_fname = sys.argv[2]

ref, pred = [], []


with open(ref_fname, 'r') as reff:
    for s in reff.readlines():
        ref.append(s)
with open(pred_fname, 'r') as predf:
    for s in predf.readlines():
        pred.append(s)

score = []
# for s,r,p in zip(src, ref, pred):
score = (chrf_metric.compute(predictions=pred, references = ref, word_order=0, lowercase=True))
print('Chrf ', score)
score = (chrf_metric.compute(predictions=pred, references = ref, word_order=2, lowercase=True))

print('Chrf++ ', score)



