import tkinter as tk
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import threading
import time

# --- Constantes y Configuración ---
CAMERA_ID = 0          # ID de la cámara (0 para la predeterminada)
ANALYSIS_INTERVAL = 30 # Analizar emociones cada 30 cuadros (frames) para mejorar el rendimiento
API_KEY = ""           # No se requiere API Key para DeepFace local, pero se incluye la variable
API_URL = ""           # No se requiere URL API para DeepFace local

class EmotionRecognizerApp:
    """
    Clase principal para la aplicación de reconocimiento de emociones faciales.
    Utiliza Tkinter para la GUI y DeepFace/OpenCV para el procesamiento de video y análisis.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de Emociones Facial (DeepFace)")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Estado de la aplicación
        self.cap = None
        self.is_running = False
        self.frame_counter = 0
        self.current_emotion = "Esperando detección..."
        
        # Bloqueo de hilos para acceso seguro a la emoción
        self.emotion_lock = threading.Lock()
        
        # Configuración de la interfaz (Estilo simple)
        self.setup_ui()
        
        # Intentar iniciar la cámara
        self.start_app()

    def setup_ui(self):
        """Inicializa los componentes de la interfaz de usuario de Tkinter."""
        
        # Fuente grande y centrada para el título
        title_font = ('Helvetica', 18, 'bold')
        self.title_label = tk.Label(self.root, text="Detección de Emociones en Vivo", 
                                    font=title_font, fg="#2c3e50")
        self.title_label.pack(pady=10)

        # Marco para el video
        self.video_frame = tk.Frame(self.root, width=640, height=480, bg="#ecf0f1", bd=3, relief=tk.RIDGE)
        self.video_frame.pack(pady=10)
        
        # Etiqueta donde se mostrará el stream de video
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Etiqueta para mostrar la emoción detectada
        emotion_font = ('Helvetica', 24, 'bold')
        self.emotion_var = tk.StringVar(value="Esperando detección...")
        self.emotion_label = tk.Label(self.root, textvariable=self.emotion_var, 
                                      font=emotion_font, fg="#e74c3c", pady=15)
        self.emotion_label.pack(pady=10)

        # Botón de Salir
        exit_button = tk.Button(self.root, text="Salir", command=self.on_closing, 
                                bg="#34495e", fg="white", font=('Helvetica', 12, 'bold'), 
                                relief=tk.FLAT, padx=20, pady=5)
        exit_button.pack(pady=10)

        # Configurar el protocolo de cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_app(self):
        """Inicia la captura de video y el hilo de procesamiento."""
        self.cap = cv2.VideoCapture(CAMERA_ID)
        
        if not self.cap.isOpened():
            self.emotion_var.set("ERROR: No se pudo abrir la cámara.")
            return

        self.is_running = True
        
        # Iniciar el hilo para la captura de video y el análisis de DeepFace
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()
        
        # Iniciar la función de actualización de la GUI en el hilo principal
        self.update_gui()

    def update_gui(self):
        """
        Función que se ejecuta en el hilo principal de Tkinter
        para actualizar la imagen de video y el texto de la emoción.
        """
        if not self.is_running:
            return

        # 1. Obtener la última emoción detectada de forma segura
        with self.emotion_lock:
            emotion_text = self.current_emotion
        
        # 2. Actualizar el texto de la emoción en la GUI
        self.emotion_var.set(f"Emoción: {emotion_text.upper()}")

        # 3. Leer el último frame procesado (desde self.cap.read() en el otro hilo)
        # NOTA: La lectura y dibujo de la caja delimitadora se realiza en el hilo secundario
        # self.cap.read() se usa en el hilo secundario (video_loop)
        
        # Si la cámara está abierta, leer el frame para mostrarlo
        ret, frame = self.cap.read()

        if ret:
            # Revertir la imagen (opcional, para visualización frontal)
            frame = cv2.flip(frame, 1)

            # Si hay una emoción detectada (lo que implica que DeepFace.analyze se ejecutó)
            if "bounding_box" in self.__dict__:
                # Dibujar el rectángulo delimitador (bounding box) y la emoción
                x, y, w, h = self.bounding_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_text.upper(), (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convertir el frame de OpenCV (BGR) a RGB para PIL
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convertir a imagen PIL y luego a PhotoImage para Tkinter
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Actualizar la etiqueta con el nuevo frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Programar la próxima actualización (aprox. 30 ms = ~33 FPS)
        self.root.after(30, self.update_gui)

    def video_loop(self):
        """
        Hilo secundario para la captura de video y el análisis intensivo de DeepFace.
        Se ejecuta de forma continua mientras la aplicación está abierta.
        """
        while self.is_running:
            # La lectura del frame para el análisis debe ser independiente de la GUI update
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01) # Pequeña pausa si falla la lectura
                continue

            # Invertir el frame para la coherencia visual (si es necesario)
            frame = cv2.flip(frame, 1)

            self.frame_counter += 1
            
            # Realizar el análisis de DeepFace solo cada N cuadros
            if self.frame_counter % ANALYSIS_INTERVAL == 0:
                self.frame_counter = 0 # Reiniciar el contador
                try:
                    # DeepFace.analyze es intensivo en CPU. Se ejecuta en este hilo.
                    # El parámetro 'actions' solo incluye 'emotion'.
                    results = DeepFace.analyze(
                        frame, 
                        actions=['emotion'], 
                        enforce_detection=False # Permite que no falle si no se detecta la cara perfectamente
                    )
                    
                    if results:
                        # Tomar el primer resultado si hay múltiples caras
                        result = results[0] 
                        emotion = result['dominant_emotion']
                        
                        # Extraer la caja delimitadora
                        detection = result['region']
                        x = detection['x']
                        y = detection['y']
                        w = detection['w']
                        h = detection['h']

                        # Actualizar las variables compartidas de forma segura
                        with self.emotion_lock:
                            self.current_emotion = emotion
                            self.bounding_box = (x, y, w, h)
                        
                    else:
                        with self.emotion_lock:
                            self.current_emotion = "Cara no detectada"
                            if 'bounding_box' in self.__dict__:
                                del self.bounding_box # Eliminar la caja si no hay cara

                except Exception as e:
                    # En caso de error (e.g., cara no encontrada y enforce_detection=True)
                    # o cualquier otro problema de DeepFace.
                    with self.emotion_lock:
                        self.current_emotion = f"Análisis fallido: {str(e)[:20]}..."
                        if 'bounding_box' in self.__dict__:
                            del self.bounding_box
                    # print(f"DeepFace Error: {e}") # Descomentar para depuración
                    
            # Pausa para evitar el uso excesivo de CPU en este hilo
            time.sleep(0.001)

    def on_closing(self):
        """Maneja el cierre de la aplicación."""
        print("Cerrando aplicación...")
        self.is_running = False
        
        # Esperar a que el hilo de video termine
        if self.video_thread.is_alive():
            self.video_thread.join()
        
        # Liberar la cámara
        if self.cap is not None:
            self.cap.release()
            
        # Cerrar la ventana de Tkinter
        self.root.destroy()

if __name__ == '__main__':
    # Crear la ventana principal de Tkinter
    root = tk.Tk()
    
    # Iniciar la aplicación
    app = EmotionRecognizerApp(root)
    
    # Iniciar el bucle principal de Tkinter
    root.mainloop()