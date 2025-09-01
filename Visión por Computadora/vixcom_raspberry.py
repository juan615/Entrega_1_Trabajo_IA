import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. Cargar imagen
# ======================
img = cv2.imread("raspberry_protoboard.png")

if img is None:
    print("❌ Error: no se pudo cargar la imagen")
    exit()

# Escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Bordes
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

# Conversión a HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
output = img.copy()

# ======================
# 2. Función de detección
# ======================
def detectar(mask, label, color, area_min, area_max, restriccion_led=False):
    """ Detectar contornos según máscara y restricciones """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area < area_max:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Filtro: evitar carriles de protoboard (rectángulos muy alargados)
            if label == "Resistencia" and aspect_ratio > 8:
                continue  

            # Filtro: LEDs solo en la parte superior de la protoboard
            if restriccion_led and y > 200:
                continue  

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ======================
# 3. Máscaras de color
# ======================

# LED (rojo)
low_red1 = np.array([0, 150, 100])
high_red1 = np.array([10, 255, 255])
low_red2 = np.array([170, 150, 100])
high_red2 = np.array([180, 255, 255])
mask_led = cv2.inRange(hsv, low_red1, high_red1) | cv2.inRange(hsv, low_red2, high_red2)

# Raspberry Pi / ESP32 (verde)
low_green = np.array([40, 40, 40])
high_green = np.array([80, 255, 255])
mask_green = cv2.inRange(hsv, low_green, high_green)

# Protoboard (gris claro / blanco)
low_gray = np.array([0, 0, 200])
high_gray = np.array([180, 40, 255])
mask_proto = cv2.inRange(hsv, low_gray, high_gray)

# Resistencia (marrón/amarillo rojizo)
low_brown = np.array([5, 50, 50])
high_brown = np.array([20, 255, 200])
mask_res = cv2.inRange(hsv, low_brown, high_brown)

# ======================
# 4. Aplicar detecciones
# ======================
detectar(mask_led, "LED", (0, 255, 0), 50, 2000, restriccion_led=True)
detectar(mask_green, "ESP32 / Raspberry Pi", (255, 0, 255), 20000, 300000)
detectar(mask_proto, "Protoboard", (150, 150, 150), 10000, 500000)
detectar(mask_res, "Resistencia", (0, 0, 255), 100, 5000)

# ======================
# 5. Mostrar resultados
# ======================
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("Detección de bordes (Canny)")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detección de componentes")

plt.tight_layout()
plt.show()

# Guardar resultado
cv2.imwrite("resultado_circuito.png", output)
print("✅ Análisis completado. Resultado guardado como 'resultado_circuito.png'")
