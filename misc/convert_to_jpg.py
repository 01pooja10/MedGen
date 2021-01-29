import os
from PIL import Image

extensions = [".png"]
L = [f for f in os.listdir('/Users/sashrikasurya/Documents/Medical-Report-Generator/data/xrays') if os.path.splitext(f)[1] in extensions]

for name in L:
    name = os.path.splitext(name)[0]
    
    img=Image.open('/Users/sashrikasurya/Documents/Medical-Report-Generator/data/xrays/'+name+'.png').resize((224,224)).convert('RGB')

    img.save('/Users/sashrikasurya/Documents/Medical-Report-Generator/data/xrays/'+name+'.jpg',quality=90)