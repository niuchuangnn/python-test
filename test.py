def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'
for x in [-1, 0, 1]:
    print sign(x)

def hello(name, loud=False):
    if loud:
        print 'HELLO, %s!' % name.upper()
    else:
        print 'Hello, %s' % name

hello('Bob')
hello('Fred', loud=True)

class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name

    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name

g = Greeter('Fred')
g.greet()
g.greet(loud=True)

import numpy as np

a = np.array([1,2,3])
print type(a)
print a.shape
print a[0], a[1], a[2]
a[0] = 5
print a

b = np.array([[1,2,3], [4,5,6]])
print b.shape
print b[0,0], b[0,1], b[1,0]

a = np.zeros((2,2))
print a

b = np.ones((1,2))
print b

c = np.full((2,2), 7)
print c

d = np.eye(2)
print d

e = np.random.random((2,2))
print e

print '----------------'
print 'Array indexing'
print '----------------'

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print a

b = a[:2, 1:3]
print b

print a[0,1]
b[0,0] = 77
print a[0,1]
print a

row_r1 = a[1, :]
row_r2 = a[1:2, :]
print row_r1, row_r1.shape
print row_r2, row_r2.shape

a = np.array([[1,2], [3,4], [5,6]])
print a
print a[[0,1,2], [0,1,0]]
print a[[0,0], [1,1]]

print np.array([a[0,1], a[0,1]])

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
print a

# create an array of indices
b = np.array([0,2,0,1])

print a[np.arange(4), b]

a[np.arange(4), b] += 10

print a

a = np.array([[1,2], [3,4], [5,6]])
bool_idx = (a>2)

print bool_idx

print a[bool_idx]

print a[a>2]

x = np.array([1, 2], dtype=np.float64)

print x.dtype


print '------------------'
print 'array math'
print '------------------'

x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[5,6], [7,8]], dtype=np.float64)

print x+y
print np.add(x,y)

print x-y
print np.subtract(x,y)

print x*y
print np.multiply(x,y)

print x/y
print np.divide(x,y)

print np.sqrt(x)

print x
print y

v = np.array([9,10])
w = np.array([11,12])

print v.dot(w)
print np.dot(v ,w)

print x.dot(v)
print np.dot(x, v)

print x.dot(y)
print np.dot(x,y)

print np.sum(x)
print np.sum(x, axis=0)
print np.sum(x, axis=1)

print x
print x.T

v = np.array([1,2,3])
print v
print v.T

print '--------------'
print 'Broadcasting'
print '--------------'

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
print y

for i in range(4):
    y[i, :] = x[i, :] + v

print y

vv = np.tile(v, (4, 1))
print vv

y = x + vv
print y

y = x + v
print y

v = np.array([1,2,3])
w = np.array([4,5])

print np.reshape(v, (3, 1)) * w

x = np.array([[1,2,3], [4,5,6]])

print x+v

print (x.T + w).T

print x + np.reshape(w, (2, 1))

print  x * 2

print '--------------------'
print 'scipy'
print '--------------------'

print 'image operation'

from scipy.misc import imread, imsave, imread, imresize

img = imread('/home/ljm/NiuChuang/cobe.jpg')
print img.dtype, img.shape

img_tinted = img * [1 , 0.95, 0.9]

img_tinted = imresize(img_tinted, (300, 300))

imsave('/home/ljm/NiuChuang/cobe_tinted.jpg', img_tinted)

print '----------------'
print 'Distance between points'
print '----------------'

from scipy.spatial.distance import pdist, squareform

x = np.array([[0,1], [1,0], [2, 0]])
print x

d = squareform(pdist(x, 'euclidean'))
print d

print '-------------------'
print 'Matplotlib'
print '-------------------'

import matplotlib.pyplot as plt

x = np.arange(0, 3*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# plt.plot(x,y_sin)
# plt.plot(x, y_cos)
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('Sine and Consin')
# plt.legend(['Sine', 'Consine'])
# plt.show()

plt.subplot(2, 1, 1)

plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2,1,2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()

plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(np.uint8(img_tinted))
plt.show()