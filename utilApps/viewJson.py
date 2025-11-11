# importing image object from PIL 
import math 
from PIL import Image, ImageDraw, ImageFont
import sys
import json
 
# Opening JSON file
print("opening " + sys.argv[1])
f = open(sys.argv[1])

jsonData = json.load(f)

name = jsonData["data"][0]["name"];
cams = jsonData["camera"];
tfs = jsonData["transferFunc"]

print(name, "has", len(cams), "cameras", len(tfs), "tfs")


w, h = 600, 300
h_offset = [100, 40]
v_offset = [40, 40]

v_bar_offset = min((h - v_offset[0] - v_offset[1]) / 2.0, 40)
v_bar_height = v_bar_offset * 0.8
#shape = [(40, 40), (w - 10, h - 10)] 
  
# creating new Image object 
img = Image.new("RGB", (w, h))

h_start = h_offset[0];
v_start = v_offset[0] - 20;

# create  rectangleimage
d = ImageDraw.Draw(img)
# get a font
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

d.text((w / 2 - 10, 0), "Time", font=fnt, fill=(255, 255, 255, 128));
ticksCount = 10
for i in range(ticksCount):
    bar_width = (w - h_offset[0] - h_offset[1])/ticksCount
    d.text((h_start, v_start), str(i), font=fnt, fill=(255,255,255, 258));
    h_start += bar_width 


h_start = h_offset[0];
v_start = v_offset[0] + 10;
# start cam
d.text((0, v_start), "camera", font=fnt, fill=(255, 255, 255, 128));

for i in range(len(cams)):
    bar_width = (w - h_offset[0] - h_offset[1])/len(cams)
    shape = [(h_start, v_start), (h_start+bar_width, v_start+v_bar_height)]  
    d.rectangle(shape, fill ="yellow", outline ="black")
    d.text((h_start, v_start), "cam"+str(i), font=fnt, fill=(0,0,0, 258));
    h_start += bar_width 

# start tf    
h_start = h_offset[0];
v_start += v_bar_offset;
d.text((0, v_start), "tf", font=fnt, fill=(255, 255, 255, 128));

for i in range(len(tfs)):
    bar_width = (w - h_offset[0] - h_offset[1])/len(tfs)
    shape = [(h_start, v_start), (h_start+bar_width, v_start+v_bar_height)]  
    d.rectangle(shape, fill ="green", outline ="black")
    d.text((h_start, v_start), "tf"+str(i), font=fnt, fill=(0,0,0, 258));
    h_start += bar_width 
    

d.line([(h_offset[0], v_offset[0]), (h_offset[0], h-v_offset[1])], fill=None, width=0, joint=None)
d.line([(h_offset[0], v_offset[0]), (w-h_offset[1]+10, v_offset[0])], fill=None, width=0, joint=None)

img.show() 
img.save("view.jpg") 
