# Downloading requirements
if [[! -f "encodings.pickle"]]
then
    wget https://www.dropbox.com/s/zkg7py03ijej315/encodings.pickle
fi
dir = "yolo/"
if [[! -d dir]]
then
    wget https://www.dropbox.com/s/15oxq46b0qzflkw/yolo.zip
    unzip yolo.zip && rm yolo.zip
else
    rm -rf yolo
fi