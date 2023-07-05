import cv2 as cv
import numpy as np

class nearby_objects():
    
    def __init__(self):
        # Define o range que em que serão gerados os avisos
        self.min = 200
        self.max = 255
        
    def warning(self, img):
        # Seleciona os objetos que estão dentro do range de aviso - Imagem é Binarizada
        caution = cv.inRange(img, self.min, self.max) 
        # Criar o contorno
        caution, I = self.contour(caution, img)
        
        return caution, I
    
    def contour(self, caution, img):
        # Faz uma copia da imagem para não alterar a original
        caution_copy = caution.copy()
        # Encontra os contornos da imagem binarizada
        counturs, hier = cv.findContours(caution_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Desenha os contornos na imagem binarizada e efetua o calculo da intensidade de saída
        I = np.zeros(len(counturs))
        for cnt in range(len(counturs)):
            # Calcula a posição do retangulo
            x, y, w, h = cv.boundingRect(counturs[cnt])
            
            # Calculo da intensidade de saída
            I[cnt] = self.intensity(x, y, w, h, img, caution_copy)
              
            cv.rectangle(caution_copy, (x,y), (x+w,y+h), (255,255,255), 2)
            
        Imax = np.max(I)
        
        return caution_copy, Imax
    
    def intensity(self, x, y, w, h, img, caution):
        # Pegar o shape da imagem
        img_shape = img.shape
        
        # Centro da imagem
        img_cx = img_shape[1]
        img_cy = img_shape[0]
        
        # Distância do centro do retângulo ao centro da imagem
        d = np.sqrt(((x/2) - img_cx)**2 + ((y/2) - img_cy)**2)
        # Normalizada
        d = d/(np.sqrt(img_cx**2 + img_cy**2))
        
        
        # Pegar a média de intensidade do objeto selecionado:
        # Pegar apenas o objeto selecionado com uma mascara
        selected_object = cv.bitwise_and(img[x:x+w ,y:y+h], img[x:x+w ,y:y+h], mask=caution[x:x+w ,y:y+h])
        
        # Média de intensidade do objeto selecionado
        try:
            selected_object = selected_object.flatten()
            soma = 0
            for i in range(len(selected_object)):
                if selected_object[i] > 0:
                    soma = soma + selected_object[i]
            mean_object = soma/len(selected_object)
            
            # Intensidade de saída (Imax=255 e Imin=0)
            I = mean_object/(d+1)
            
        except:
            I = 0
 
        return I
        
        
        
        
    
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
    
    