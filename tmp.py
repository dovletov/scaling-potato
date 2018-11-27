from __future__ import print_function
import torch

x = torch.empty(5,3)
print(x)
print("x - ", x.dtype)

y = torch.rand(5,3)
print(y)
print("y - ", y.dtype)

z = torch.zeros(5,3, dtype=torch.long)
print(z)
print("z - ", z.dtype)

a = torch.tensor([5.5, 3])
print(a)
print("a - ", a.dtype)

b = x.new_ones(2,2) # new_* methods take in sizes
print(b)
print("b - ", b.dtype)

c = torch.randn_like(z, dtype=torch.float) # override dtype
print(c)
print("c - ", c.dtype)
print(c.size())

print(x + y)
print(torch.add(x,y))
xy = torch.empty(5,3)
torch.add(x,y,out=xy)
print(xy)

y.add_(x)
print(y)

d = torch.rand(4,4)
e = d.view(16)
f = d.view(-1,8)
print(d.size(), e.size(), f.size())

g = torch.randn(1)
print(g, g.dtype)
print(g.item(), g.dtype)

h = g.numpy()
print(h, h.dtype)

g.add_(1)
print(h) # numpy array is also changing


import numpy as np
i = np.ones(5)
j = torch.from_numpy(i)
print(j)
np.add(i,1, out=i)
print(j)

if torch.cuda.is_available():
	# let us run this cell only if CUDA is available
	# We will use ``torch.device`` objects to move tensors in and out of GPU
	if torch.cuda.is_available():
	    device = torch.device("cuda")          # a CUDA device object
	    print(device)
	    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
	    print(y)
	    x = x.to(device)                       # or just use strings ``.to("cuda")``
	    z = x + y
	    print(z)
	    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!