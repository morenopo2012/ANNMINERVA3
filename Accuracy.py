import matplotlib.pyplot as plt

count=0
x,y,x2,y2,position=[],[],[],[],[]

epochs = 10
type = 'Loss' #Accuracy or Loss
cnn='Res' #Res or VF
cat='213' #Number of categories to train

for line in open('/data/omorenop/HDF5toPrediction/ANNMINERvA3/infoplot/'+type+'TrRes-'+cat+'cat-WholeMix.log','r'):
    values = [float(s) for s in line.split()]
    x.append(values[0])
    y.append(values[1])
    count = count+1

for line in open('/data/omorenop/HDF5toPrediction/ANNMINERvA3/infoplot/'+type+'VaRes-'+cat+'cat-WholeMix.log','r'):
    values2 = [float(s) for s in line.split()]
    x2.append(values2[0])
    y2.append(values2[1])

count = count/epochs

for j in range(0,epochs+1):
    position.append(count*j)

plt.grid(True)
plt.title(type +' vs epochs')
plt.xlabel("Epochs")
plt.ylabel(type)
line1=plt.plot(x,y, label='Training')
line2=plt.plot(x2,y2,label='validation')
#labels=('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15')
labels=('0','1','2','3','4','5','6','7','8','9','10')
#labels=('0','1','2','3','4','5','6','7','8','9')
#labels=('0','1','2','3','4','5','6','7')
plt.xticks(position,labels)
plt.legend(title=cnn+'-'+cat+'_categories')
plt.savefig(type+'-'+cnn+'-'+cat+'_categories.png')
plt.show()
