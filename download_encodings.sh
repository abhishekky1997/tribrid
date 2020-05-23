# Downloading requirements
if [[-f "encodings.pickle"]]; then
    rm encodings.pickle
else
    wget https://www.dropbox.com/s/zkg7py03ijej315/encodings.pickle
fi

if [-d "~/tribrid/yolo/"]; then
    rm -rf yolo

else
    wget https://www.dropbox.com/s/15oxq46b0qzflkw/yolo.zip
    unzip yolo.zip && rm yolo.zip
fi