# needs modifications
# just for testing

import customtkinter
from PIL import ImageTk, Image
from recognitionFunctions import *

PREDICTION = ""

PPP = "studentEngagementModel/OIP.jpg"


target_image_size = (500, 500)


def imageChange_callback():
    img2 = customtkinter.CTkImage(light_image=Image.open(PPP),
                                  size=(450, 450))

    image_label.configure(image=img2)


def upload_file():
    file_path = customtkinter.filedialog.askopenfilename(title="Select an image", filetypes=[(
        'all files', '*.*'), ('png files', '*.png'), ('jpeg files', '*.jpeg'), ('jpg files', '*.jpg')])

    global IMAGE_PATH, PPP
    IMAGE_PATH = file_path
    PPP = file_path
    imageChange_callback()


# def capture_image():
#     # Capture frame from webcam
#     ret, frame = cap.read()
#     if ret:
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         imgg = processImage(frame_rgb, rem_noise=True, hist_eq=True,
#                             resize=True, target_size=target_image_size)

#         img = Image.fromarray(frame_rgb)
#         imgtk = ImageTk.PhotoImage(image=img)

#         # videoLabel.configure(image=imgtk)
#         # videoLabel.image = imgtk

#         predictionn = predict(imgg, 8)
#         print(predictionn)
#         print(predictionn)
#         print(predictionn)
#         print(predictionn)
#         print(predictionn)
#         print(predictionn)
#         print(predictionn)

#         global PREDICTION

#         # if predictionn == 0:
#         #     PREDICTION = "Samir"
#         #     predictionLabel.configure(text=PREDICTION)
#         #     print("Samir")
#         # elif predictionn == 1:
#         #     PREDICTION = "Abdelatty"
#         #     predictionLabel.configure(text=PREDICTION)
#         #     print("Abdelatty")
#         # else:
#         #     PREDICTION = "Ismail"
#         #     predictionLabel.configure(text=PREDICTION)
#         #     print("Ismail")

#     else:
#         print("Error capturing image")


# def video_loop():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         img = Image.fromarray(frame_rgb)
#         imgtk = ImageTk.PhotoImage(image=img)

#         videoLabel.configure(image=imgtk)
#         videoLabel.image = imgtk

#         mainFrame.update()

#     cap.release()
#     cv2.destroyAllWindows()


def prePredict():
    print(PPP)

    frame = cv2.imread(PPP)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imgg = processImage(frame_rgb, rem_noise=True, hist_eq=True,
                        resize=True, target_size=target_image_size)

    prediction = predict(imgg, 2)
    global PREDICTION

    if prediction == 0:
        PREDICTION = "Ronaldo"
        predictionLabel.configure(text=PREDICTION)
        print("Samir")
    elif prediction == 1:
        PREDICTION = "Salah"
        predictionLabel.configure(text=PREDICTION)
        print("Abdelatty")
    else:
        PREDICTION = "Ismail"
        predictionLabel.configure(text=PREDICTION)
        print("Ismail")


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


app = customtkinter.CTk()
app.geometry("1080x720")

app.title("Student Recognition")


title = customtkinter.CTkLabel(app, text="Classify the Student",
                               font=customtkinter.CTkFont(family='Arial', size=20, weight='bold'))
title.pack(padx=10, pady=(40, 80))

mainFrame = customtkinter.CTkFrame(app, fg_color="transparent")
mainFrame.pack(fill="x", expand=True, anchor="n")

# videoLabel = customtkinter.CTkLabel(mainFrame, text="")
# videoLabel.pack()

# capture_button = customtkinter.CTkButton(
#     mainFrame, text="Capture Image", command=capture_image)
# capture_button.pack()

my_image = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH),
                                  size=(450, 450))

imageButtonsFrame = customtkinter.CTkFrame(
    mainFrame, width=450, height=500, fg_color="transparent")
imageButtonsFrame.pack_propagate(False)
imageButtonsFrame.pack(side="right", padx=(10))

image_label = customtkinter.CTkLabel(
    imageButtonsFrame, image=my_image, text="")  # display image with a CTkLabel
image_label.pack()

buttonsFrame = customtkinter.CTkFrame(
    imageButtonsFrame, fg_color="transparent")
buttonsFrame.pack(pady=10, anchor="n")

uploadButton = customtkinter.CTkButton(
    buttonsFrame, text="Upload an image", command=upload_file)
uploadButton.pack(side="left", padx=(5, 5))

predictButton = customtkinter.CTkButton(
    buttonsFrame, text="Predict", command=prePredict)
predictButton.pack(side="right", padx=(5, 5))

predictionLabel = customtkinter.CTkLabel(mainFrame, text="PREDICTION")
predictionLabel.pack()


# thread = threading.Thread(target=video_loop)
# thread.start()

app.mainloop()
