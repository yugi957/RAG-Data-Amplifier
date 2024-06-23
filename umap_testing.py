import umap
import umap.plot
from sklearn.datasets import load_digits

digits = load_digits()

print(digits.data.shape)

mapper = umap.UMAP().fit(digits.data)
umap.plot.points(mapper, labels=digits.target)

