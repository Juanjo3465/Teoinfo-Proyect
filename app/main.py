def main():
    import tkinter as tk
    from tkinter import Label, Button
    from PIL import Image, ImageTk
    import cv2
    from deepface import DeepFace
    import threading

    class EmotionApp:

        def __init__(self, window):
            self.window = window
            self.window.title("Emotion Recognition App")

            self.video_label = Label(window)
            self.video_label.pack()

            self.emotion_label = Label(window, text="Emotion: ---", font=("Arial", 16))
            self.emotion_label.pack()

            self.start_button = Button(window, text="Iniciar detección", command=self.start_detection)
            self.start_button.pack(pady=10)

            self.cap = None
            self.running = False

        def start_detection(self):
            if not self.running:
                self.cap = cv2.VideoCapture(0)
                self.running = True
                threading.Thread(target=self.update_frame).start()

        def update_frame(self):
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Convert for Tkinter display (BGR → RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Analyze emotion (non-blocking try)
                try:
                    result = DeepFace.analyze(
                        rgb_frame, 
                        actions=['emotion'], 
                        enforce_detection=False
                    )
                    emotion = result[0]["dominant_emotion"]
                except:
                    emotion = "---"

                # Update emotion label
                self.emotion_label.config(text=f"Emotion: {emotion}")

                # Convert to ImageTk
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.cap.release()

        def stop(self):
            self.running = False
            if self.cap:
                self.cap.release()
            self.window.destroy()

    # -------------------------
    # Run App
    # -------------------------
    root = tk.Tk()
    app = EmotionApp(root)

    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()


if __name__ == "__main__":
    main()
