# needs modifications
# just for testing

import customtkinter
from PIL import ImageTk, Image
from tstFunctions import *

PREDICTION = ""


def capture_image():
    # Capture frame from webcam
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        videoLabel.configure(image=imgtk)
        videoLabel.image = imgtk

        cv2.imwrite(
            "captured_image.jpg", frame)

        prediction, probas = predict(
            "captured_image.jpg",5)
        global PREDICTION
        PREDICTION = f"{probas[0,1]*100}% Engaged"
        predictionLabel.configure(text=PREDICTION)

        if prediction == 1:
            print("Engaged")
        else:
            print("Not Engaged")

    else:
        print("Error capturing image")


def video_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        videoLabel.configure(image=imgtk)
        videoLabel.image = imgtk

        mainFrame.update()

    cap.release()
    cv2.destroyAllWindows()


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

videoLabel = customtkinter.CTkLabel(mainFrame, text="")
videoLabel.pack()

capture_button = customtkinter.CTkButton(
    mainFrame, text="Capture Image", command=capture_image)
capture_button.pack()

predictionLabel = customtkinter.CTkLabel(mainFrame, text="PREDICTION")
predictionLabel.pack()


thread = threading.Thread(target=video_loop)
thread.start()

app.mainloop()
