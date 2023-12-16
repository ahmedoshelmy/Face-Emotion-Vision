# needs modifications
# just for testing

import customtkinter
from PIL import ImageTk, Image
from tstFunctions import *

PPP = ""


def imageChange_callback():
    img2 = customtkinter.CTkImage(light_image=Image.open(IMAGE_PATH),
                                  size=(450, 450))

    image_label.configure(image=img2)


def upload_file():
    file_path = customtkinter.filedialog.askopenfilename(title="Select an image", filetypes=[(
        'all files', '*.*'), ('png files', '*.png'), ('jpeg files', '*.jpeg'), ('jpg files', '*.jpg')])

    global IMAGE_PATH, PPP
    IMAGE_PATH = file_path
    PPP = file_path
    imageChange_callback()


def prePredict():
    predict(PPP)


# def capture_image():
#     # Capture frame from webcam
#     ret, frame = cap.read()
#     if ret:
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         img = Image.fromarray(frame_rgb)
#         imgtk = ImageTk.PhotoImage(image=img)

#         videoLabel.configure(image=imgtk)
#         videoLabel.image = imgtk

#         cv2.imwrite("D:\GitHub Repos\Face-Emotion-Vision\captured_image.jpg", frame)
#         global IMAGE_PATH
#         IMAGE_PATH = "D:\GitHub Repos\Face-Emotion-Vision\captured_image.jpg"
#         predict("D:\GitHub Repos\Face-Emotion-Vision\captured_image.jpg")

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


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


app = customtkinter.CTk()
app.geometry("1080x720")

app.title("Student Engagement")


title = customtkinter.CTkLabel(app, text="Predict Weather the student is Engaged or Not",
                               font=customtkinter.CTkFont(family='Arial', size=20, weight='bold'))
title.pack(padx=10, pady=(40, 80))

mainFrame = customtkinter.CTkFrame(app, fg_color="transparent")
mainFrame.pack(fill="x", expand=True, anchor="n")

# videoLabel = customtkinter.CTkLabel(mainFrame)
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


# # Start video loop in a separate thread
# thread = threading.Thread(target=video_loop)
# thread.start()

app.mainloop()
