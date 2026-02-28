from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer

def minmax(X): return MinMaxScaler().fit_transform(X)
def maxabs(X): return MaxAbsScaler().fit_transform(X)
def standard(X): return StandardScaler().fit_transform(X)
def normalize(X): return Normalizer().fit_transform(X)
