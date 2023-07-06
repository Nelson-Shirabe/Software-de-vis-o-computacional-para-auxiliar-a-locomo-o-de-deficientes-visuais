import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

class nearby_objects():
    
    def __init__(self):
        # Define o range que em que serão gerados os avisos
        self.min = 200
        self.max = 255
        
    def warning(self, img, cx, cy):        
        # Seleciona os objetos que estão dentro do range de aviso - Imagem é Binarizada
        caution= cv.inRange(img, self.min, self.max)
        # Criar o contorno e Definir a Intensidade
        caution, I = self.contour(caution, img, cx, cy)
        
        return caution, I
    
    def contour(self, caution, img, cx, cy):
        # Faz uma copia da imagem para não alterar a original
        caution_copy = caution.copy()
        # Encontra os contornos da imagem binarizada
        counturs, hier = cv.findContours(caution_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Desenha os contornos na imagem binarizada e efetua o calculo da intensidade de saída
        Imax = 0
        rect = np.zeros(4)
        for cnt in range(len(counturs)):
            # Calcula a posição do retangulo
            x, y, w, h = cv.boundingRect(counturs[cnt])
            # Calculo da intensidade de saída
            I = self.intensity(x, y, w, h, img, caution_copy, cx, cy)
            
            if I > Imax:
                Imax = I
                rect = [x, y, w, h]
            
        # Desenha o Retângulo  
            cv.rectangle(caution_copy, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255,255,255), 2)    
               
        return caution_copy, Imax
    
    def intensity(self, x, y, w, h, img, caution, cx, cy):       
        img_shape = img.shape 
        img_cx = cx
        img_cy = cy
        
        # Distância do centro do retângulo ao centro da imagem
        d = np.sqrt(((x/2) - img_cx)**2 + ((y/2) - img_cy)**2)
        # Normalizada
        d = d/(np.sqrt(img_shape[1]**2 + img_shape[0]**2))
        
        
        # Pegar a média de intensidade do objeto selecionado:
        # Pegar apenas o objeto selecionado com uma mascara 
        selected_object = cv.bitwise_and(img[y:y+h, x:x+w], img[y:y+h, x:x+w], mask=caution[y:y+h, x:x+w])
        
        # Média de intensidade do objeto selecionado
        selected_object = selected_object.flatten()
        soma = 0
        n = 0
        for i in range(len(selected_object)):
            if selected_object[i] > 0:
                soma = soma + selected_object[i]
                n = n + 1
        mean_object = soma/n
    
        # Intensidade de saída (Imax=255 e Imin=0)
        I = (mean_object/(d+1))*((n/(w*h)))
        
        return I 
        
        
        
        
    
if __name__ == "__main__":
    # Para Testes
    img = cv.imread("Amostra.png", cv.IMREAD_UNCHANGED)
    img_shape = img.shape
    
    objects = nearby_objects()
    # 1º Quadrante
    caution1 = img[:(img_shape[0]//2), (img_shape[1]//2): ] 
    cx = 0
    cy = 0
    caution1, I1 = objects.warning(caution1, cx, cy)
    
    # 2º Quadrante
    caution2 = img[:(img_shape[0]//2), :(img_shape[1]//2) ] 
    cx = img_shape[1]//2
    cy = 0
    caution2, I2 = objects.warning(caution2, cx, cy)
    
    # 3º Quadrante
    caution3 = img[(img_shape[0]//2):, :(img_shape[1]//2) ] 
    cx = img_shape[1]//2
    cy = img_shape[0]//2
    caution3, I3 = objects.warning(caution3, cx, cy)
    
    # 4º Quadrante
    caution4 = img[(img_shape[0]//2):, (img_shape[1]//2): ] 
    cx = 0
    cy = img_shape[0]//2
    caution4, I4 = objects.warning(caution1, cx, cy)
    
    
    # # Juntar as imagens
    # warning = np.zeros((img_shape[0], img_shape[1]), dtype=np.int8)
    
    # warning[:(img_shape[0]//2), (img_shape[1]//2): ] = caution1 # 1º Quadrante
    # warning[:(img_shape[0]//2), :(img_shape[1]//2) ] = caution2 # 2º Quadrante
    # warning[(img_shape[0]//2):, :(img_shape[1]//2) ] = caution3 # 3º Quadrante
    # warning[(img_shape[0]//2):, (img_shape[1]//2): ] = caution4 # 4º Quadrante
    # print(warning.shape)
      
    # Mostrar resultado     
    while True:
        cv.imshow("I1", caution1)
        cv.setWindowTitle("I1", f"I1={I1:.2f}")
        
        cv.imshow("I2", caution2)
        cv.setWindowTitle("I2", f"I2={I2:.2f}")
        
        cv.imshow("I3", caution3)
        cv.setWindowTitle("I3", f"I3={I3:.2f}")
        
        cv.imshow("I4", caution4)
        cv.setWindowTitle("I4", f"I4={I4:.2f}")
        
        # Pressione a tecla "q" para parar
        if cv.waitKey(1) == ord("q"):
            break
    cv.destroyAllWindows()
    
    