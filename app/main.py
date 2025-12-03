import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import platform

# --- 1. Mapeo y Configuración ---
FACE_EMOJI_MAP = {
    'Boca Abierta': "😮",
    'Boca Cerrada': "😐",
    'Guiño': "😉",
    'Ojos Cerrados': "😴",
    'Sonrisa': "😊",
    'Neutro': "😐"
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. Funciones de Ayuda ---
def get_emoji_font(size=60):
    system = platform.system()
    font_path = "arial.ttf"
    if system == "Windows": font_path = "seguiemj.ttf"
    elif system == "Darwin": font_path = "Apple Color Emoji.ttc"
    elif system == "Linux": font_path = "NotoColorEmoji.ttf"
    try:
        return ImageFont.truetype(font_path, size)
    except IOError:
        return ImageFont.load_default()

emoji_font = get_emoji_font(60)

def euclidean_distance(point1, point2):
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_ear(landmarks, indices):
    # Eye Aspect Ratio
    A = euclidean_distance(landmarks[indices[1]], landmarks[indices[5]])
    B = euclidean_distance(landmarks[indices[2]], landmarks[indices[4]])
    C = euclidean_distance(landmarks[indices[0]], landmarks[indices[3]])
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, indices):
    # Mouth Aspect Ratio
    A = euclidean_distance(landmarks[indices[1]], landmarks[indices[7]])
    A2 = euclidean_distance(landmarks[indices[2]], landmarks[indices[6]])
    A3 = euclidean_distance(landmarks[indices[3]], landmarks[indices[5]])
    B = euclidean_distance(landmarks[indices[0]], landmarks[indices[4]])
    return (A + A2 + A3) / (3.0 * B)

# Índices
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
MOUTH_IDXS = [61, 185, 40, 39, 291, 181, 17, 0]

# --- 3. Clasificación con MAYOR SENSIBILIDAD ---

def classify_face_sensitive(landmarks):
    EAR_THRESHOLD = 0.20  
    
    MAR_THRESHOLD = 0.45  

    left_ear = calculate_ear(landmarks, LEFT_EYE_IDXS)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_IDXS)
    mar = calculate_mar(landmarks, MOUTH_IDXS)
    
    avg_ear = (left_ear + right_ear) / 2.0

    stats = f"Ojos: {avg_ear:.2f} | Boca: {mar:.2f}"

    state = 'Neutro'
    
    if mar > MAR_THRESHOLD:
        state = 'Boca Abierta'
    elif avg_ear < EAR_THRESHOLD:
        state = 'Ojos Cerrados'
    elif left_ear < EAR_THRESHOLD and right_ear > EAR_THRESHOLD:
        state = 'Guiño' 
    elif right_ear < EAR_THRESHOLD and left_ear > EAR_THRESHOLD:
        state = 'Guiño'
    else:
        state = 'Boca Cerrada'
        
    return state, stats

def put_emoji_pil(img_np, text, pos, font):
    img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    x, y = pos
    draw.text((x-1, y), text, font=font, fill=(0,0,0))
    draw.text((x+1, y), text, font=font, fill=(0,0,0))
    draw.text((x, y-1), text, font=font, fill=(0,0,0))
    draw.text((x, y+1), text, font=font, fill=(0,0,0))
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: continue
    
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    
    cv2.rectangle(image, (0, 0), (w, 100), (20, 20, 20), -1) 

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    detected_expression = 'Neutro'
    debug_stats = "Buscando cara..."
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            detected_expression, debug_stats = classify_face_sensitive(landmarks)
            
            for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS + MOUTH_IDXS:
                pt = landmarks[idx]
                cv2.circle(image, (int(pt.x * w), int(pt.y * h)), 1, (0, 255, 255), -1)
            break 
    
    # 1. Mostrar Emoji
    emoji_to_show = FACE_EMOJI_MAP.get(detected_expression, "❓")
    display_text = f"{emoji_to_show} {detected_expression}"
    
    # Centrar texto
    try:
        bbox = emoji_font.getbbox(display_text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = emoji_font.getsize(display_text)

    text_x = int((w - text_w) / 2)
    text_y = int((100 - text_h) / 2) - 5
    image = put_emoji_pil(image, display_text, (text_x, text_y), emoji_font)

    cv2.putText(image, f"Sensibilidad (Debug): {debug_stats}", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Deteccion Sensible', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()