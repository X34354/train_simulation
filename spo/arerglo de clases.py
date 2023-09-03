import os 
path = 'C:/Users/fraja\Documents/train_data/labels/train' #path of labels
labels = os.listdir(path)
for x in labels:
    with open(path + '/' +x,'r') as f:
        lines = f.read()
        lines[0:2]
        lines = lines.replace('15', '0')
    with open(path + '/' +x,'w') as file:
    
        # Writing the replaced data in our
        # text file
        file.write(lines)


#Panthera Onca_5_274