.PHONY: clean

out.avi: outputs/model_final.pth in.mp4
	python inference.py

outputs/models_final.pth: data
	python train.py

data:
	wget https://www.dropbox.com/s/38ry8ny1lwi1kip/data.zip
	unzip data.zip
	rm data.zip

clean:
	@rm outputs/models_final.pth
	@rm out.avi
