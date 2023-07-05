import cv2 as cv

class nearby_objects():
    
    def __init__(self):
        # Define o range que em que serão gerados os avisos
        self.min = 200
        self.max = 255
        
    def warning(self, img):
        # Seleciona os objetos que estão dentro do range de aviso - Imagem é Binarizada
        caution = cv.inRange(img, self.min, self.max) 
        # Criar o contorno
        caution = self.contour(caution)
        
        return caution
    
    def contour(self, img):
        # Faz uma copia da imagem para não alterar a original
        img_copy = img.copy()
        # Encontra os contornos da imagem binarizada
        counturs, hier = cv.findContours(img_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Desenha os contornos na imagem binarizada
        for cnt in counturs:
            x, y, w, h = cv.boundingRect(cnt)  
            cv.rectangle(img_copy, (x,y), (x+w,y+h), (255,255,255), 2)
            
        return img_copy
    
    
if __name__ == "__main__":
    # Para Testes
    img = cv.imread("Amostra.png", cv.IMREAD_UNCHANGED)
    
    objects = nearby_objects()
    objects = objects.warning(img)
    
    cv.namedWindow("Warning", cv.WINDOW_NORMAL)
    
    while True:
        cv.imshow("Warning", objects)
        
        # Pressione a tecla "q" para parar
        if cv.waitKey(1) == ord("q"):
            break
    cv.destroyAllWindows()
    
    