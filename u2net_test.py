import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import argparse
import shutil
import time

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():
    parser = argparse.ArgumentParser(description="U2-Net Inferenz für einen spezifischen Datensatz!")
    parser.add_argument("--source", type=str, required=True, help="Pfad zum Verzeichnis mit den Eingabebildern")
    parser.add_argument("--target", type=str, required=True, help="Pfad zum Verzeichnis, in dem die Vorhersagen gespeichert werden sollen")

    args = parser.parse_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    model_path="/scratch/tmp/lterfehr/models/U-2-Net/saved_models/u2net/u2net.pth"

    if not os.path.exists(model_path):
        print(f"Fehler: Die Model-datei '{model_path}' existiert nicht.")
        print("Bitte stelle sicher, dass die Datei im korrekten Verzeichnis liegt und der Pfad korrekt angegeben ist.")

    if os.path.exists(OUTPUT_DIR):
        print(f"Warnung: Das Verzeichnis '{OUTPUT_DIR}' existiert bereits. Es wird gelöscht und neu erstellt.")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_name_list = glob.glob(os.path.join(INPUT_DIR, '*'))
    img_name_list = [f for f in img_name_list if os.path.isfile(f)]

    num_images = len(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)

    net.eval()

    # --------- 4. inference for each image ---------
    print("Starte Inferenz...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Sicherstellen, dass alle vorherigen CUDA-Operationen abgeschlossen sind
    
    start_time = time.time()

    # NEU: torch.no_grad() schaltet das Gradienten-Tracking ab. Schont VRAM & bringt Speed!
    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):
            current_img_path = img_name_list[i_test]
            print("inferencing:", current_img_path.split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            # NEU: Modernes PyTorch-Handling statt der veralteten Variable()
            inputs_test = inputs_test.to(device)

            # Inferenz durchführen
            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

            # Normalisierung der besten Ebene (d1)
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            # Absicherung für den Speicherpfad (stellt sicher, dass os.sep/Slash passt)
            safe_output_dir = OUTPUT_DIR if OUTPUT_DIR.endswith(os.sep) else OUTPUT_DIR + os.sep
            
            # Speicher-Funktion aufrufen
            save_output(current_img_path, pred, safe_output_dir)

            # Speicherbereinigung
            del d1, d2, d3, d4, d5, d6, d7

    if torch.cuda.is_available():
        torch.cuda.synchronize() 

    end_time = time.time()

    total_time = end_time - start_time
    time_per_image = total_time / num_images if num_images > 0 else 0
    print(f"Fertig! Alle Masken wurden erfolgreich in '{OUTPUT_DIR}' gespeichert. Verwendete Zeit: {total_time:.2f} Sekunden (Durchschnitt pro Bild: {time_per_image:.2f} Sekunden)")

    eval_file_path = os.path.join(OUTPUT_DIR, "evaluation.txt")

    with open(eval_file_path, "w") as eval_file:
        eval_file.write(f"Total images processed: {num_images}\n")
        eval_file.write(f"Total inference time: {total_time:.2f} seconds\n")
        eval_file.write(f"Average time per image: {time_per_image:.2f} seconds\n")

if __name__ == "__main__":
    main()
