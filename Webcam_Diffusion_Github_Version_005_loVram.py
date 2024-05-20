# Realtime Webcam Stable Diffusion
# Renderer von Hermann Knopp
# 20.5.2024 Version 1
# Tested on Nvidia RTX 3060-12Gb
# Software uses 2GB VRam!!!
# Low Vram Version

# Import OS and change Window Title
import os
os.system('mode con: cols=80 lines=40')
os.system('title Stable Diffusion Webcam Scribble Renderer (Realtime)')
print("Importing Librarys...")


#Import Threading Routine Libs
import threading
from qt_thread_updater import get_updater


#Quicktime 6 Imports
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication,QMainWindow, QWidget, QVBoxLayout


#File Time/Date Lib
import time
import torch, logging
import shlex

# Tastatur Eingabe
from msvcrt import getch, kbhit


# Disable Warnings
logging.disable(logging.WARNING)  

#Import Image manipulkation Lib
import cv2
from PIL import Image,ImageTk
from PIL.ImageQt import ImageQt
import numpy as np

# Diffusers Import
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler, LMSDiscreteScheduler
from diffusers.utils import load_image
from datetime import datetime
import random

#Import Path Manipulation Libs
import glob
import os.path

# Import Webcam Librarys
print("Starting Webcam...")
import cv2 as cv


# Beginn Init------------

# Connect a Webcam at USB Port for
# Reference Image capturing...

#Select Cam Port
cam = cv.VideoCapture(0, cv2.CAP_DSHOW)

#Set Window Cam
cam.set(3,320)
cam.set(4,320)

#Check Cam 
if not cam.isOpened():
   print("error opening camera")
   exit()

#Threaded Function Render Class
def run(is_alive):
    is_alive.set()

    a=input("Wait return key...")
    os.system("cls")


    while is_alive.is_set():
        
        
        #Run Prompt Engine
        Render()
           
     

# Quicktime 6 Main Window Class (defunct)
class OldWindow():
   
    #Please use updater Function instead
    def __init__(self):
        #No Init needed - Main Window Class deactivated
        #super(MainWindow, self).__init__()
        pass
           
    def change_message(self):
        #No self.message = "OH NO" selected
        pass

    def oh_no(self):
        #No Worker Thread defined 
        pass

    def init(self):
        #No Worker Thread started
        pass


# Realtime Render and Preview Class
class Render():
 
   def __init__(self):
   
    #Realtime Prompt and Render Loop
    #Waits for Prompt and Image
    #in users/test/pictures

    print("")
    print("place a object in front of webcam,       ")
    print("than press 'p' on keyboard..enter prompt")
    print("or add -batch 10 for bulk rendering...") 
    Flag=0
    
    while True:

        # Capture frame-by-frame
        ret, frame = cam.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("error in retrieving frame")
            break
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow('frame', img)
            

        # Check Key Press "p" in realtime   
        if kbhit(): # returns True if the user has pressed a key
            key = ord(getch()) 
            
            #press Key "p"
            if key == 112:
                print("")
                print("flag ok...starting render mode.")
                print("")
                Flag=1
                # etc


        # Enter Render Mode if Key "p" is pressed
        if Flag==1:
          
            # saving image in local storage 
            cv.imwrite(folder_path + "test1.bmp", img) 

            #Default One Picture
            batchsize=1
    
            
            # Positive Prompt    
            prompt=input("Prompt: ")
            if prompt=="" or prompt==None:  
                prompt="a realistic town"
            print("Prompt is ",prompt)

            print("")
            negative=input("Negative: ")
            if negative=="" or negative==None:
                negative="lores,blurry,pixelated"
            print("Negative is",negative)


            # Start Rendering...
            Flag2=1 
            if Flag2==1:   
                             
                print("ok")
                print("")
    
                list=shlex.split(prompt) 
                #list=prompt.split()
                for i in range(len(list)):
                   #print(list[i])
                   if "-batch" in list[i]:
                       batchsize = list[i+1] 
                       #print("batch:" + batchsize)
                       batchsize = int(batchsize)
                       if batchsize>1000:
                          batchsize=1000

                       #Generate new Prompt without Batch Mode Command
                       prompt=""
                       for j  in range(i):
                   
                           prompt = prompt + " "+ str(list[j])
                       print("New Prompt:" +prompt)     

 
                # Re-Select Negative Prompt
                negative_prompts=negative     
              

                # Select last File in Directory
                files = glob.glob(folder_path + file_type)
                if files =="":
                  files=""
                  print("Please save one Image.bmp File to Picture In Directory... waiting!")
                  msgbox("Please Click here if Picture is ready...")
                  continue

                max_file = max(files, key=os.path.getctime)
                max_file = max_file.replace("\\","/")

        
                print("I will scribble your file " + max_file )

                # Load last Image from Directory
                image = load_image(max_file)
    
                #init_image = Image.open(max_file).convert("RGB")
                init_image = image.resize((512, 512))

                image = np.array(init_image)
                low_threshold = 15
                high_threshold = 250
                image = cv2.Canny(image, low_threshold, high_threshold)
            
 

                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                canny_image = Image.fromarray(image)

                #canny_image.show()

                print("You will render a Amount of: "+str(batchsize) + " Images...")
 
                # Set System Variable MAX Seed
                maximum=4294967295
    
                for loops in range(0,batchsize):
 
                     seed=random.randint(1,maximum)
                     generator = torch.manual_seed(seed)

                     print("will Render Image Nr: " + str(loops+1) + " from " + str(batchsize))
  
                     images = pipe(prompt,negative_prompt=negative_prompts, num_inference_steps=steps, generator=generator, image=canny_image).images

                     # use Folder from Menu instead
                     # img_out_folder=r"C:/Users/test/Pictures/Saved Pictures"
 
                     for img in images:
 
                           # datetime object containing current date and time
                           now = datetime.now()
                           dt_string = now.strftime("%d%m%Y_%H%M%S")
               
                           # Save rendered Image to Disk
                           filename=img_out_folder + "/" + "test_" + dt_string +".jpg"
                           img.save(filename)
            

                           print("Path=",filename)
                           #a=input("Debug1")
 
                           #Load actual Image from Disk
                           image = Image.open(filename)
                           image = ImageQt(image)
                   
                           # Quicktime 6 Threaded Image Display annd Threaded Update Function
                           pixmap = QPixmap.fromImage(image)
                  
                           #MainWindow.displayImage(pixmap)
     
                           get_updater().call_latest(MainWindow.displayImage(self,image=pixmap))
                           
                           #Stop Rendering 1 Image
                           Flag=0
                           #print("Waiting...at 'p' Key")

                      
                     print("Wait 'p' key") 

   
        if cv.waitKey(1) == ord('q'):
            break
        
               
           
      


# Main Display Message
os.system("cls")



print("Hermanns Diffuser Scribble Renderer")
print("")
print("with Webcam Input Support")
a=input("Wait return key, for selecting Models...")


#Display Diffusers Models
print("Please Select SD-Model from List (2)")
print("")
print("Model (1) = stable diffusion 1.5 FP16 (2GB)")
print("Model (2) = stable diffusion 1.5 FP32 (4GB)")
print("")
print("Downloading Model from huggingface.co or harddrive/.cache")
print("please wait...")
print("if accelerate lib error, please restart")


#Model Input
 
modelnr = input("Model Nr: (1-2)")
if modelnr==None:
   modelnr=1
if modelnr=="":
   modelnr=1
modelnr=int(modelnr)
if modelnr==0:
   modelnr=1
if modelnr>2:
   modelnr=1


# Select Diffusers Models

if modelnr ==1:
   model_path="nmkd/stable-diffusion-1.5-fp16"
if modelnr ==2:
   model_path="runwayml/stable-diffusion-v1-5"


#Set Image Custom Input and Output Path

fid=input("do you want to change Image In folder (Y/N)")
if fid =="":
   fid="N"
if fid=="Y" or fid=="y" or fid=="j" or fid=="J":
   fid_path=input("Path:")
   img_in_folder=fid_path
   img_in_folder = img_in_folder.replace('\\','/')
   print("I am using: "+img_in_folder)
if fid=="N" or fid=="n":
   img_in_folder="C:/Users/test/Pictures/"
   print("I am using:" + img_in_folder)


fod=input("do you want to change Image Out folder (Y/N)")
if fod =="":
   fod="N"
if fod=="Y" or fod=="y" or fod=="j" or fod=="J":
   fod_path=input("Path:")
   img_out_folder=fod_path
   img_in_folder = img_in_folder.replace('\\','/')
   print("I am using: "+img_out_folder)
if fod=="N" or fod=="n":
   img_out_folder="C:/Users/test/Pictures/Saved Pictures"
   print("I am using:" + img_out_folder)


# Setup Diffusers Pipeline

device="cuda"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float32).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(model_path,safety_checker=None, controlnet=controlnet, torch_dtype=torch.float32).to(device)


## Initializing a scheduler and Setting number of sampling steps
pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe.scheduler.set_timesteps(50)

# CPU only Tasks
#pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.enable_model_cpu_offload()


# Always Low VRam Mode =on
optimizer=1

if optimizer == 1:

    # Memory Optimzations CPU and GPU usage
    pipe.enable_vae_tiling()
    #pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_sequential_cpu_offload()



#Setup Controlnet Pipeline (Fixed at 16 Iterations per Image)
steps=10

# Change these as you want:
folder_path = img_in_folder
file_type = r"\*.bmp"


#Set File Folders and Path
print("Selected In Folder is: " + img_in_folder)
print("Selected Out Folder is: " + img_out_folder)
print("Selected Folder Path is: " + folder_path)



#Main Window Class GUI,Preview Window
class MainWindow(QMainWindow):

    #Init
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi()
        
    def __name__(self):
        pass 


    #Setup GUI
    def setupUi(self):
        #Main
        self.setWindowTitle("Stable Diffusion Preview Window")
        self.resize(512, 512)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.label = QLabel()   
        self.label.resize(512,512)
        
        global imagelabel
        imagelabel=self.label
       
        
        #self.label.show()
        
        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
       
        self.centralWidget.setLayout(layout)
        
    #Display Rendered Image 
    def displayImage(self,image):
              
        pixmap = image
        imagelabel.setPixmap(pixmap)
        imagelabel.show()
        
    #Wait for Button(defunct)
    def onButtonClick(servers,self):
        #print(servers)
        pass


#Start Threaded Window

#app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
app = QApplication.instance() or QApplication([])
window=MainWindow()
window.show()

       

#Main Window Render Loop Start and keep Alive

alive = threading.Event()
th = threading.Thread(target=run, args=(alive,))
th.start()

app.exec()
alive.clear()
