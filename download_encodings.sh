# Downloading requirements
if [[! -f "encodings.pickle"]]
then
    wget https://www.dropbox.com/s/zkg7py03ijej315/encodings.pickle
fi
if [[! -f yolo/obj.cfg]] && [[! -f yolo/obj.names]] && [[! -f yolo/obj.weights]]
then
    wget https://www.dropbox.com/s/15oxq46b0qzflkw/yolo.zip
    unzip yolo.zip && rm yolo.zip
else
    rm -rf yolo/
fi