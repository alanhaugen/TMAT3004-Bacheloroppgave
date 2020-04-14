.PHONY: clean

out.avi: outputs/model_final.pth in.mp4
	python inference.py

outputs/model_final.pth: data
	python train.py

data:
	curl -L https://www.dropbox.com/s/38ry8ny1lwi1kip/data.zip --output data.zip
	unzip data.zip
	rm data.zip

clean:
	@rm outputs/models_final.pth
	@rm out.avi
