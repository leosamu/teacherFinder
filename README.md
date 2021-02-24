# teacherFinder
Busca profesores en un video usando darknet y la opencv

#instalacion
Para que esta herramienta funcione necesitas instalar los requsitos 
```sh
pip install -r requirements.txt
```

Además deberás configurar darknet tal y como indica en esta web https://pjreddie.com/darknet/yolo/ pero lo resumo a continuación

Básicamente se requieren 3 pasos 

* descargar y compilar darknet
```sh
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

* descargar el archivo de pesos
```sh
wget https://pjreddie.com/media/files/yolov3.weights
```
* descargar el archivo de configuración
```sh
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

tras esto cambiaremos los parametros en teacherFinder.py para que encuentre los archivos correspondientes y definiremos el margen de confianza
COCONAMESPATH='darknet/data/coco.names'
YOLOCFGPATH='yolov3.cfg'
YOLOWEIGHTPATH='yolov3.weights'
THRESHOLD=1

y ya podremos llamar a la funcion return_code o lanzarlo desde el terminal


