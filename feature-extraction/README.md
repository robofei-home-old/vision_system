# feature-extraction

<b> Criar dir "mask", "images" e "results"
Caso tenha erro de path (out of index), verificar nos files u2net, alguns endereços de diretórios ficam lá</b>


## Remoção do fundo da imagem
![image](https://user-images.githubusercontent.com/103227067/193166819-03bef079-1e79-4736-85f2-69d99c56ac77.png)

## Cria as landmarks 
![image](https://user-images.githubusercontent.com/103227067/193167112-468dcffb-ceef-45cb-81a0-89783681c3e2.png)

## Crop das imagens
![image](https://user-images.githubusercontent.com/103227067/193167345-5dfb1039-ce18-4789-b582-998f795f8530.png)

É feito apartir das landmarks obtidas um <i>crop</i> na altura em que desejamos para a extração da característica

## Extração de cor da imagem cropada
![image](https://user-images.githubusercontent.com/103227067/193167448-cc27df96-36f1-4260-90b2-a2c118bac863.png)

E assim obtemos a porcentagem de cada cor que existe na foto, e que provavelmente a maior porcentagem será a cor da peça de roupa do host
