from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = load_model("app/model/traffic.keras")
    logger.info("Model başarıyla yüklendi")

    first_layer_weights = model.layers[0].get_weights()
    logger.info(f"İlk katman ağırlık özeti: {np.mean(first_layer_weights[0])}, {np.std(first_layer_weights[0])}")
except Exception as e:
    logger.error(f"Model yüklenirken hata oluştu: {e}")

# Sınıf etiketleri
class_labels = [
    'Hız sınırı (20km/s)',
    'Hız bölgesi (30km/h)',
    'Hız bölgesi (50km/h)',
    'Hız sınırı (60km/saat)',
    'Hız bölgesi (70km/saat)',
    'Hız bölgesi (80km/saat)',
    'Hız bölgesinin sonu (80km/h)',
    'Hız bölgesi (100km/h)',
    'Hız bölgesi (120km/h)',
    'Geçmek yok',
    '3,5 tonun üzerinde araç geçemez',
    'Kavşakta geçiş hakkı',
    'Öncelikli yol',
    'Verim',
    'Dur',
    'Araç yok',
    '3,5 tonluk araçlar yasaktır',
    'Giriş yasaktır',
    'Genel uyarı',
    'Sola tehlikeli viraj',
    'Sağa tehlikeli viraj',
    'Çift viraj',
    'Engebeli yol',
    'Kaygan yol',
    'Yol sağda daralıyor',
    'Yol çalışması',
    'Trafik sinyalleri',
    'Yayalar',
    'Çocukların geçişi',
    'Bisikletlerin geçişi',
    'Buz/kar konusunda dikkatli olun',
    'Vahşi hayvanların geçişi',
    'Hız sonu + geçiş sınırları',
    'İleride sağa dön',
    'İleride sola dön',
    'Sadece ileri',
    'Düz veya sağa git',
    'Düz veya sola git',
    'Sağda kal',
    'Solda kal',
    'Döner kavşak zorunlu',
    'Geçiş yasağının sonu',
    'Geçiş yasağının sonu > 3,5 tonluk araç'
]

def preprocess_image(image):
    """Görüntüyü model için ön işleme"""

    input_shape = model.input_shape[1:3]  
    logger.info(f"Model giriş boyutu: {input_shape}")
    
    image = image.resize(input_shape)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    logger.info(f"İşlenmiş görüntü istatistikleri: Ortalama={np.mean(img_array)}, StdDev={np.std(img_array)}")
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(file) -> dict:
    try:

        image = Image.open(io.BytesIO(file))
        logger.info(f"Orijinal görüntü boyutu: {image.size}, modu: {image.mode}")
        
        img_array = preprocess_image(image)
        
        predictions = model.predict(img_array)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        logger.info(f"Tahmin edilen sınıf: {predicted_index}, güven: {confidence}")
        
        return {
            "predicted_class": class_labels[predicted_index],
            "confidence": round(confidence, 4),
            "class_id": predicted_index
        }
    except Exception as e:
        logger.error(f"Tahmin sırasında hata: {e}")
        return {
            "error": str(e)
        }
