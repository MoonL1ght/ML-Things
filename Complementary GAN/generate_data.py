import numpy as np
import pandas as pd
import sys

class NestedSpheresGenerator(object):
  def __init__(self, old=None, olp=0.5, false_prop=0.1):
    self.dataset = None
    self.old = old
    self.olp = olp
    self.false_prop = false_prop
    
  def __overlap(self, data, d, p, r):
    def foo(row, r=r, d=d):
      if (r-d <= np.sqrt(np.sum(row[:-1]**2)) <= r+d) and (np.random.random() < p):
        return 1 - row[-1]
      else:
        return row[-1]
    data[:, -1] = np.apply_along_axis(func1d=foo, axis=1, arr=data)
    return data
  
  def generate(self, ndim, gdistr='normal', byparts=1, r=1, R=2):
    n = ndim[0]
    byparts = [n // byparts for i in range(n // (n // byparts))] + [n % byparts]
    if byparts[-1] == 0: byparts.pop() 
    inner_sphere_labels = lambda x, r: np.sum(x**2, axis=1) <= r**2
    datasets = []
    ndim_ = (1, ndim[1])
    n_pos = int(n * 10 / 100)
    n_neg = int(n - n_pos)
    for i in range(n):
      def gen_dataset():
        if gdistr == 'uniform':
          dataset = np.random.uniform(-R, R, ndim_)
        elif gdistr == 'normal':
          dataset = np.random.normal(0, r/np.sqrt(ndim[1]-1), ndim_)
        return dataset
      prob_pos = np.random.rand(1)[0]
      if prob_pos <= self.false_prop:
        label = False
        while not label:
          dataset = gen_dataset()
          label = inner_sphere_labels(dataset, r).reshape((ndim_[0], 1))
        dataset = np.hstack([dataset, inner_sphere_labels(dataset, r).reshape((ndim_[0], 1))])
        datasets.append(dataset)
      else:
        label = True
        while label:
          dataset = gen_dataset()
          label = inner_sphere_labels(dataset, r).reshape((ndim_[0], 1))
        dataset = np.hstack([dataset, inner_sphere_labels(dataset, r).reshape((ndim_[0], 1))])
        datasets.append(dataset)
    dataset_full = np.vstack(datasets)
    if self.old:
      dataset_full = self.__overlap(dataset_full, d=self.old, p=self.olp, r=r)
    return dataset_full

   
QUANTITY = 10000
DIM = 30
FALSE_PROB = 0.5

if len(sys.argv) >= 2:
  QUANTITY = int(sys.argv[1])

if len(sys.argv) >= 3:
  DIM = int(sys.argv[2])

if len(sys.argv) >= 4:
  FALSE_PROB = float(sys.argv[3])

def generate_data():
  # nsg = NestedSpheresGenerator(old=5, olp=0.3, false_prop=FALSE_PROB)
  nsg = NestedSpheresGenerator(false_prop=FALSE_PROB)
  data = nsg.generate((QUANTITY, DIM), gdistr='normal', r=40, R=70)

  var_names = []
  for i in range(DIM):
    var_names.append('VAR_'+str(i))
  var_names = var_names + ['CLASS']

  df = pd.DataFrame(data, columns=var_names)
  df.to_csv('./data/data_'+str(QUANTITY)+'_'+str(DIM)+'dim.csv', sep=';')

generate_data()