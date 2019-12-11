import os
path='../data/_stacks/fox/'
path = os.path.abspath(path)

for i, file in enumerate(os.listdir(path)):
    newname = "%06d" % i + '.PNG'
    os.rename(os.path.join(path,file),os.path.join(path,newname))

print('The Number of files in this folder: %d' % i)
