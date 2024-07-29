import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import time
import mediapipe as mp
import threading

class PoseDetector:
    def __init__(self, mode=False, complexity=True, enableSeg=False, smooth=True, detectionCon=0.9, trackCon=0.9):
        self.mode = mode
        self.complexity = complexity
        self.enableSeg = enableSeg
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.complexity, self.enableSeg, self.smooth, self.detectionCon, self.trackCon
        )

    def findpose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList

class HumanRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("gait.GATE")

        # Load the background image
        image_path = "Firefly 20240428121411.png"
        self.original_image = Image.open(image_path)
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        self.resized_image = self.resize_image(self.original_image, (screen_width, screen_height))
        self.background_photo = ImageTk.PhotoImage(self.resized_image)

        # Create canvas for background image
        self.canvas = tk.Canvas(master, width=screen_width, height=screen_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo)
        self.canvas.pack()

        # Frame to hold the widgets
        self.frame = tk.Frame(self.canvas)
        self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.label = tk.Label(self.frame, text="gait.GATE", font=("Arial", 18), bg='white')
        self.label.pack(pady=20)

        # Initialize the frame for the buttons
        self.button_frame = tk.Frame(self.frame, bg='white')
        self.button_frame.pack(pady=10)

        # Create the "Upload Video" button
        button_style = {
            "font": ("Arial", 18),
            "bg": 'blue',
            "fg": "white",
            "activebackground": 'darkblue',
            "activeforeground": "white",
            "bd": 3,
            "relief": "raised",
            "width": 15,
            "padx": 15,
            "pady": 15,
        }
        self.upload_button = tk.Button(self.button_frame, text="UPLOAD VIDEO", command=self.upload_video, **button_style)
        self.upload_button.pack(side=tk.LEFT, padx=10)

        # Create the "Predict" button
        self.predict_button = tk.Button(self.button_frame, text="PREDICT", command=self.predict, **button_style)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        # Create the "Quit" button
        self.quit_button = tk.Button(self.button_frame, text="QUIT", command=self.quit_app, **button_style)
        self.quit_button.pack(side=tk.LEFT, padx=10)

        # Initialize the preview label
        self.preview_label = tk.Label(self.frame, text="", font=("Arial", 12), bg='white')
        self.preview_label.pack(pady=10)

        # Initialize PoseDetector
        self.detector = PoseDetector()

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if file_path:
            self.video_path = file_path
            self.preview_label.config(text=f"Video uploaded: {file_path}")
            messagebox.showinfo("Success", "Video uploaded successfully:\n" + file_path)
        else:
            messagebox.showwarning("Error", "No file selected.")

    def predict(self):
        if hasattr(self, 'video_path'):
            threading.Thread(target=self.process_video).start()
        else:
            messagebox.showwarning("Error", "Please upload a video first.")

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        pTime = time.time()
        human_count = 0
        total_frames = 0
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = self.detector.findpose(img)
            lmList = self.detector.findPosition(img, draw=False)
            
            total_frames += 1

            if len(lmList) >= 15:
                # Assuming 15 landmarks as the threshold for a human pose, adjust this number as needed
                output_text = "Human"
                human_count += 1
                cv2.putText(img, output_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                output_text = "Non-Human"
                cv2.putText(img, output_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # Print live FPS
            cv2.putText(img, f"FPS: {int(fps)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            img = cv2.resize(img, (900, 750))
            cv2.imshow("Gait Gate", img)

            key = cv2.waitKey(1)
            if key == 27:
                break

        # Calculate accuracy
        accuracy = (human_count / total_frames) * 100

        # Display message box with FPS and accuracy
        messagebox.showinfo("Stats", f"Average FPS: {int(total_frames / (time.time() - pTime))}\nAccuracy: {accuracy:.2f}%")

        # Release video capture and destroy OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    def quit_app(self):
        self.master.destroy()

    def resize_image(self, image, size):
        aspect_ratio = image.width / image.height
        target_width, target_height = size

        if aspect_ratio > 1:  # Landscape orientation
            target_height = int(target_width / aspect_ratio)
        else:  # Portrait or square orientation
            target_width = int(target_height * aspect_ratio)

        return image.resize((target_width, target_height))


def main():
    root = tk.Tk()
    root.attributes('-fullscreen', True)  # Make the window fullscreen

    # Custom title bar
    title_bar = tk.Frame(root, bg='gray', relief='raised', bd=2)
    title_bar.pack(expand=True, fill='x')

    # Minimize button
    minimize_button = tk.Button(title_bar, text='-', command=root.iconify)
   #minimize_button.pack(side='right')

    # Maximize button
    maximize_button = tk.Button(title_bar, text='[]', command=lambda: root.attributes('-fullscreen', not root.attributes('-fullscreen')))
    maximize_button.pack(side='right')

    # Close button
    close_button = tk.Button(title_bar, text='X', command=root.destroy)
    close_button.pack(side='right')

    app = HumanRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
